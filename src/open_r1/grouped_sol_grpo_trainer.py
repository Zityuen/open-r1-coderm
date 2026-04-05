import logging
import random
import warnings
from contextlib import nullcontext
from typing import Any, Optional, Union

import numpy as np
import torch
from accelerate.utils import broadcast_object_list, gather, gather_object
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from open_r1.trl.data_utils import is_conversational, maybe_apply_chat_template
from open_r1.trl.extras.profiling import profiling_context, profiling_decorator
from open_r1.trl.import_utils import is_vllm_available
from open_r1.trl.models import unwrap_model_for_generation
from open_r1.trl.trainer.grpo_trainer import GRPOTrainer, nanstd
from open_r1.trl.trainer.utils import pad
from open_r1.unittest_credit import build_per_token_unittest_advantages, pooled_normalize_per_test_values

if is_vllm_available():
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


logger = logging.getLogger(__name__)


class GroupedSolGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer subclass for grouped-solution training.

    Instead of one (prompt, solution) pair per dataset row, each row contains a question
    and all of its solutions. At generation time, m solutions are sampled without replacement,
    m prompts are constructed, and n completions are generated per prompt (m * n = num_generations).
    Rewards are computed via cross-solution unit test execution.

    Optional ``grpo_unittest_credit_assignment`` (with ``unittest_reward_per_test_expr`` in script
    config) enables per-test-method token credit and pooled normalization over all semantic tests
    in the ``num_generations`` group.
    """

    def __init__(
        self,
        *args,
        num_sampled_solutions: int = 4,
        num_completions_per_solution: int = 4,
        system_prompt_template: Optional[str] = None,
        user_prompt_template: str = "",
        grpo_unittest_credit_assignment: bool = False,
        unittest_reward_per_test_expr: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.m = num_sampled_solutions
        self.n = num_completions_per_solution
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.grpo_unittest_credit_assignment = grpo_unittest_credit_assignment
        self.unittest_reward_per_test_expr = unittest_reward_per_test_expr
        self._unittest_reward_idx = next(
            (i for i, n in enumerate(self.reward_func_names) if n == "unittest_reward_aggregation"),
            None,
        )

        if self.m * self.n != self.num_generations:
            raise ValueError(
                f"num_sampled_solutions ({self.m}) * num_completions_per_solution ({self.n}) "
                f"must equal num_generations ({self.num_generations}), got {self.m * self.n}"
            )

        if self.grpo_unittest_credit_assignment:
            if not self.unittest_reward_per_test_expr:
                logger.warning(
                    "grpo_unittest_credit_assignment is True but unittest_reward_per_test_expr is unset; "
                    "per-test credit payloads may be empty and unittest advantages fall back to the scalar GRPO path."
                )
            if self.use_liger_loss:
                logger.warning(
                    "unittest credit assignment uses per-token advantages; disabling use_liger_loss for this trainer."
                )
                self.use_liger_loss = False

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt_question", "solutions"]

    def _build_prompt(self, question: str, sol_func: str) -> list[dict]:
        prompt = []
        if self.system_prompt_template:
            prompt.append({"role": "system", "content": self.system_prompt_template})
        prompt.append({
            "role": "user",
            "content": self.user_prompt_template.format(
                question=question, code_solution=sol_func,
            ),
        })
        return prompt

    def _sample_solutions(self, solutions: list[dict]) -> list[dict]:
        """
        Sample m solutions without replacement. If fewer than m are available,
        use all available and pad by re-sampling with replacement.
        """
        if len(solutions) >= self.m:
            indices = random.sample(range(len(solutions)), self.m)
        else:
            indices = list(range(len(solutions)))
            while len(indices) < self.m:
                indices.append(random.randint(0, len(solutions) - 1))
        return [solutions[i] for i in indices]

    @profiling_decorator
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # ------------------------------------------------------------------
        # Phase 1: Sample solutions and construct prompts
        # ------------------------------------------------------------------
        num_groups = len(inputs) // self.num_generations

        transformed_inputs: list[dict] = []
        for g in range(num_groups):
            example = inputs[g * self.num_generations]
            question = example["prompt_question"]
            solutions = example["solutions"]

            sampled_sols = self._sample_solutions(solutions)

            for sol in sampled_sols:
                prompt = self._build_prompt(question, sol["solve_func"])
                for _ in range(self.n):
                    transformed_inputs.append({
                        "prompt": list(prompt),
                        "sampled_solutions": sampled_sols,
                    })

        inputs = transformed_inputs

        # ------------------------------------------------------------------
        # Phase 2: Standard tokenisation and generation
        # ------------------------------------------------------------------
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False,
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        if self.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Within each group of num_generations items, there are m unique prompts
                    # each repeated n times. Take every n-th to get unique prompts.
                    ordered_set_of_prompts = all_prompts_text[:: self.n]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.n,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)

                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(
                        backend="outlines", regex=self.guided_decoding_regex,
                    )
                else:
                    guided_decoding = None
                sampling_params = SamplingParams(
                    n=1,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )

                if self.vllm_tensor_parallel_size > 1:
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(
                        gathered_prompts, prompts_text, group=self.tp_group,
                    )
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(
                        all_prompts_text, sampling_params=sampling_params, use_tqdm=False,
                    )

                completion_ids = [
                    output.token_ids for outputs in all_outputs for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(
                        local_rank_in_group * orig_size,
                        (local_rank_in_group + 1) * orig_size,
                    )
                    completion_ids = completion_ids[tp_slice]

            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask,
                        generation_config=self.generation_config,
                    )

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # ------------------------------------------------------------------
        # Phase 3: EOS masking, per-token logps (identical to base)
        # ------------------------------------------------------------------
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]

        completion_lengths = completion_mask.sum(1)

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask,
                    logits_to_keep, batch_size,
                )
            else:
                old_per_token_logps = None

        # ------------------------------------------------------------------
        # Phase 4: Decode completions
        # ------------------------------------------------------------------
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True,
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # ------------------------------------------------------------------
        # Phase 5: Compute rewards
        # ------------------------------------------------------------------
        num_rf = len(self.reward_funcs)
        rewards_per_func = torch.zeros(len(prompts), num_rf, device=device)
        reward_diagnostics_local: dict[str, list[str]] = {
            name: [""] * len(prompts) for name in self.reward_func_names
        }

        keys = [
            key for key in inputs[0]
            if key not in ["prompt", "completion", "completion_ids"]
        ]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    if is_conversational(inputs[0]):
                        from open_r1.trl.data_utils import apply_chat_template
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True,
                        padding_side="right", add_special_tokens=False,
                    )
                    reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions,
                        completion_ids=completion_ids_list, **reward_kwargs,
                    )
                    diagnostics = getattr(reward_func, "_last_diagnostics", None)
                    if isinstance(diagnostics, list) and len(diagnostics) == len(prompts):
                        reward_diagnostics_local[reward_func_name] = [str(d or "") for d in diagnostics]
                    output_reward_func = [
                        r if r is not None else torch.nan for r in output_reward_func
                    ]
                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device,
                    )

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            warnings.warn(
                f"All reward functions returned NaN for row {nan_row_idx}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # ------------------------------------------------------------------
        # Phase 6: Gather, normalise, compute advantages
        # ------------------------------------------------------------------
        rewards_per_func = gather(rewards_per_func)

        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        G = self.num_generations
        mean_grouped = rewards.view(-1, G).mean(dim=1)
        std_grouped = rewards.view(-1, G).std(dim=1)
        is_std_zero = torch.isclose(std_grouped, torch.zeros_like(std_grouped))

        # expand the mean and std of the grouped rewards to the number of generations
        mean_grouped_rewards_exp = mean_grouped.repeat_interleave(G, dim=0)
        std_grouped_rewards_exp = std_grouped.repeat_interleave(G, dim=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        if self.grpo_unittest_credit_assignment and self._unittest_reward_idx is not None:
            ut_idx = self._unittest_reward_idx
            w_ut = self.reward_weights[ut_idx].to(device=device, dtype=rewards.dtype)
            rut = torch.nan_to_num(rewards_per_func[:, ut_idx], nan=0.0)
            rewards_ex_ut = rewards - w_ut * rut

            mean_ex = rewards_ex_ut.view(-1, G).mean(dim=1).repeat_interleave(G, dim=0)
            std_ex = rewards_ex_ut.view(-1, G).std(dim=1).repeat_interleave(G, dim=0)
            adv_other = rewards_ex_ut - mean_ex
            if self.scale_rewards:
                adv_other = adv_other / (std_ex + 1e-4)

            adv_full = rewards - mean_grouped_rewards_exp
            if self.scale_rewards:
                adv_full = adv_full / (std_grouped_rewards_exp + 1e-4)

            ut_fn = self.reward_funcs[ut_idx]
            local_payloads = getattr(ut_fn, "_last_credit_payload", None)
            if local_payloads is None or len(local_payloads) != len(prompts):
                local_payloads = [None] * len(prompts)
            credit_all = gather_object(local_payloads)

            raw_by_row: list[Optional[list[float]]] = []
            for p in credit_all:
                if p and p.get("per_test_rewards"):
                    raw_by_row.append([float(x) for x in p["per_test_rewards"]])
                else:
                    raw_by_row.append(None)
            # normalization over number of semantic test functions
            # e.g., num_generation = 4, semantic test functions = [2,3,4,5] respectively
            # then we need to normalize over 2 + 3 + 4 + 5 = 14 tests
            normed_rows = pooled_normalize_per_test_values(
                raw_by_row, G, self.scale_rewards,
            )

            ut_name = self.reward_func_names[ut_idx]
            n_global = len(credit_all)
            flat_raw: list[float] = []
            flat_centered: list[float] = []
            for r in raw_by_row:
                if r:
                    flat_raw.extend(r)
            for r in normed_rows:
                if r:
                    flat_centered.extend(r)

            frac_payload = (
                sum(1 for p in credit_all if p and p.get("per_test_rewards")) / n_global
                if n_global
                else float("nan")
            )
            credit_applied_n = 0
            test_counts_when_applied: list[int] = []
            for gidx in range(n_global):
                pl = credit_all[gidx]
                centered = normed_rows[gidx]
                if (
                    pl
                    and centered
                    and len(centered) > 0
                    and pl.get("canonical_test_ids")
                    and pl.get("test_code")
                ):
                    credit_applied_n += 1
                    test_counts_when_applied.append(len(centered))
            frac_applied = credit_applied_n / n_global if n_global else float("nan")

            pr = self._metrics[mode]
            # Scalar ``rewards/unittest_reward_aggregation/mean`` (below) is still the expression aggregate
            # (e.g. float(np.mean(...))) — same as non-credit. These metrics reflect per-test pooling.
            if flat_raw:
                pr[f"rewards/{ut_name}/mean_over_semantic_tests_raw"].append(float(np.mean(flat_raw)))
                pr[f"rewards/{ut_name}/std_over_semantic_tests_raw"].append(float(np.std(flat_raw)))
            else:
                pr[f"rewards/{ut_name}/mean_over_semantic_tests_raw"].append(float("nan"))
                pr[f"rewards/{ut_name}/std_over_semantic_tests_raw"].append(float("nan"))
            if flat_centered:
                pr[f"rewards/{ut_name}/mean_over_semantic_tests_centered"].append(
                    float(np.mean(flat_centered))
                )
                pr[f"rewards/{ut_name}/mean_abs_semantic_tests_centered"].append(
                    float(np.mean(np.abs(flat_centered)))
                )
            else:
                pr[f"rewards/{ut_name}/mean_over_semantic_tests_centered"].append(float("nan"))
                pr[f"rewards/{ut_name}/mean_abs_semantic_tests_centered"].append(float("nan"))
            pr["unittest_credit/frac_rows_with_payload"].append(frac_payload)
            pr["unittest_credit/frac_rows_credit_applied"].append(frac_applied)
            pr["unittest_credit/mean_num_tests_when_applied"].append(
                float(np.mean(test_counts_when_applied)) if test_counts_when_applied else float("nan")
            )

            T = completion_ids.size(1)
            advantages = torch.zeros(len(prompts), T, device=device, dtype=torch.float32)
            adv_other_local = adv_other[process_slice]
            adv_full_local = adv_full[process_slice]

            local_token_coverage: list[float] = []
            for i in range(len(prompts)):
                gidx = self.accelerator.process_index * len(prompts) + i
                mask_row = completion_mask[i].float()
                eff_len = int(mask_row.sum().item())
                pl = credit_all[gidx] if 0 <= gidx < len(credit_all) else None
                centered = normed_rows[gidx] if 0 <= gidx < len(normed_rows) else None

                if (
                    pl
                    and centered
                    and len(centered) > 0
                    and pl.get("canonical_test_ids")
                    and pl.get("test_code")
                ):
                    tok = build_per_token_unittest_advantages(
                        self.processing_class,
                        completions_text[i],
                        pl["test_code"],
                        pl["canonical_test_ids"],
                        centered,
                        eff_len,
                    )
                    ut_tok = torch.zeros(T, device=device, dtype=torch.float32)
                    L = min(int(tok.shape[0]), T)
                    ut_tok[:L] = torch.as_tensor(tok[:L], device=device, dtype=torch.float32)
                    ut_tok = ut_tok * mask_row
                    advantages[i] = adv_other_local[i] + w_ut * ut_tok
                    n_ut = (ut_tok.abs() > 1e-8).sum().item()
                    local_token_coverage.append(n_ut / max(eff_len, 1))
                else:
                    advantages[i] = adv_full_local[i] * mask_row

            cov_local = float(np.mean(local_token_coverage)) if local_token_coverage else float("nan")
            cov_t = torch.tensor([cov_local], device=device, dtype=torch.float32)
            pr["unittest_credit/mean_token_coverage_when_applied"].append(
                self.accelerator.gather(cov_t).nanmean().item()
            )

            # Per-completion mean of pooled-centered per-test values (for tables / debugging).
            self._unittest_credit_row_centered_means = [
                float(np.mean(normed_rows[g])) if normed_rows[g] else float("nan")
                for g in range(len(normed_rows))
            ]

            all_process_advantages = adv_full.clone()
        else:
            self._unittest_credit_row_centered_means = None
            advantages_global = rewards - mean_grouped_rewards_exp
            if self.scale_rewards:
                advantages_global = advantages_global / (std_grouped_rewards_exp + 1e-4)
            all_process_advantages = advantages_global.clone()
            advantages = advantages_global[process_slice]

        # ------------------------------------------------------------------
        # Phase 7: Logging (same as base)
        # ------------------------------------------------------------------
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )

        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_lengths.float().max().item()
        )

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards_exp.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards_exp.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        gathered_prompts_text = gather_object(prompts_text)
        gathered_completions_text = gather_object(completions_text)
        self._textual_logs["prompt"].extend(gathered_prompts_text)
        self._textual_logs["completion"].extend(gathered_completions_text)
        num_logged = len(gathered_prompts_text)
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            gathered_diagnostics = gather_object(reward_diagnostics_local[name])
            self._textual_logs["reward_diagnostics"][name].extend(gathered_diagnostics)
        if self.grpo_unittest_credit_assignment and self._unittest_reward_idx is not None:
            means = getattr(self, "_unittest_credit_row_centered_means", None)
            if means is not None:
                u = self.reward_func_names[self._unittest_reward_idx]
                self._textual_logs["rewards"][f"{u}_per_completion_mean_centered"].extend(means)
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }
