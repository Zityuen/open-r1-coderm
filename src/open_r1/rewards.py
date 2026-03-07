# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import ast
import asyncio
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from functools import partial, update_wrapper
from typing import Callable, Dict, Literal, Optional

import numpy as np

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils.code_providers import get_provider
from .utils.competitive_programming import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
)
from .utils.competitive_programming import patch_code as cf_patch_code
from .utils.competitive_programming import score_submission as cf_score_submission
from .utils.competitive_programming import score_subtask


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://huggingface.co/papers/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    """Initialize or get the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using a specified execution client.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    Args:
        completions: List of model completions to evaluate
        test_batch_size: Evaluate these many test cases in parallel, then check if any of them failed (0 score):
                       if so stop evaluating; otherwise continue with the next batch of test cases.
        provider_type: The execution provider to use (default: "piston"). Supported values: "piston", "morph"
        **kwargs: Additional arguments passed from the dataset
    """
    # Get the appropriate client based on provider_type
    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # for info on setting up piston workers, see slurm/piston/README.md
        execution_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from {provider_type} worker: {e}")
            return SubtaskResult()

    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                score_subtask(
                    execution_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def cf_code_reward(
    completions,
    test_batch_size: int = 1,
    patch_code: bool = False,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    **kwargs,
) -> list[float]:
    """Reward function that evaluates Codeforces problems using Piston+our CF package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/codeforces (verifiable-prompts subset)

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    languages = kwargs["language"] if "language" in kwargs else [None] * len(completions)
    code_snippets = [
        # note: grading is automatically skipped if a problem has no tests
        cf_patch_code(extract_code(completion[-1]["content"], language), language)
        if patch_code
        else extract_code(completion[-1]["content"], language)
        for completion, language in zip(completions, languages)
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return None

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                cf_score_submission(
                    piston_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                    scoring_mode=scoring_mode,
                    submission_language=problem_data.get("language", None),
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return results


def extract_code(completion: str, language: str | None = "python") -> str:
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs,
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]

    template = evaluation_script_template

    scripts = [
        template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("All verification_info must have the same language", verification_info)

    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    return execution_provider.execute_scripts(scripts, ["python"] * len(scripts))

def unittest_reward(completions, solution, is_correct, **kwargs) -> list[float]:
    """
    Reward function that executes the generated unit tests (completion) against the
    provided solution code. Returns the number of passed tests.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        # Extract code from completion (the tests)
        test_code = extract_code(content)
        # Fallback: if extract_code returns empty but content looks like code, use it
        if not test_code and ("def " in content or "class " in content):
             test_code = content

        # Extract code from solution (the solve function)
        sol_code = extract_code(sol)
        if not sol_code:
            # Assume raw code if no markdown found in solution
            sol_code = sol

        if not test_code.strip():
            rewards.append(0.0)
            continue

        script_content = f"""
import unittest
import sys
from typing import *
# Solution Code
{sol_code}
# Test Code
{test_code}
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)
    passed = result.testsRun - len(result.errors) - len(result.failures)
    print(f"METRICS:{{passed}}:{{result.testsRun}}")
"""
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name

            # Execute the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=5
            )

            output = result.stdout
            match = re.search(r"METRICS:(\d+):(\d+)", output)
            if match:
                passed = int(match.group(1))
                total = int(match.group(2))
                reward = float(passed) / float(total) if total > 0 else 0.0
                if not is_correct:
                    reward = 1 - reward
                rewards.append(reward)
            else:
                rewards.append(0.0)

        except Exception:
            rewards.append(0.0)
        finally:
            if script_path and os.path.exists(script_path):
                os.remove(script_path)

    return rewards


def _execute_unittest_suite(
    test_code: str,
    sol_code: str,
    timeout: int = 5,
) -> dict[str, int]:
    """
    Execute the full unittest suite once for a single solution and return a per-test pass/fail map.

    The returned dict maps each test id (e.g. '__main__.Test.test1') to:
      1 if the test passed,
      0 if the test failed or errored.
    """
    if not test_code.strip() or not sol_code.strip():
        return {}

    script_content = f"""
import unittest
import sys
from typing import *

# Solution Code
{sol_code}

# Test Code
{test_code}

def _flatten_suite(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten_suite(item)
        else:
            yield item

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    all_tests = list(_flatten_suite(suite))
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)
    fail_ids = {{case.id() for case, _ in result.failures}}
    error_ids = {{case.id() for case, _ in result.errors}}

    print("METRICS_START")
    for t in all_tests:
        tid = t.id()
        status = 0 if (tid in fail_ids or tid in error_ids) else 1
        print(f"{{tid}}:{{status}}")
    print("METRICS_END")
"""
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        metrics: dict[str, int] = {}
        in_block = False
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped == "METRICS_START":
                in_block = True
                continue
            if stripped == "METRICS_END":
                break
            if not in_block:
                continue
            # Line format: "<test_id>:<0-or-1>"
            try:
                test_id, status_str = stripped.rsplit(":", 1)
                metrics[test_id.strip()] = int(status_str)
            except Exception:
                continue

        return metrics
    except Exception:
        return {}
    finally:
        if script_path and os.path.exists(script_path):
            os.remove(script_path)


def _run_unittest_with_per_test_metrics(
    test_code: str,
    solutions: list[dict],
    timeout: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """
    Execute the unittest suite against every solution and return the B_ik binary matrix.

    Args:
        test_code: The generated unittest code to evaluate.
        solutions: List of dicts with keys ``"solve_func"`` (str) and ``"is_correct"`` (bool).
        timeout: Timeout in seconds for each individual execution.

    Returns:
        A tuple ``(B_ik, test_ids)`` where:

        - ``B_ik`` is a numpy ``int`` array of shape ``(len(solutions), K)``.
          ``B_ik[i, k] == 1`` iff solution *i* passes test function *k*.
        - ``test_ids`` is the canonical ordering of the *K* test-function identifiers
          (e.g. ``'__main__.TestFoo.test_bar'``).
    """
    if not test_code.strip():
        return np.empty((0, 0), dtype=int), []

    # Run the suite once per solution and collect raw per-test dicts
    per_solution_metrics: list[dict[str, int]] = []
    canonical_test_ids: list[str] = []

    for sol in solutions:
        sol_code = sol["solve_func"]
        metrics = _execute_unittest_suite(test_code, sol_code, timeout=timeout)
        per_solution_metrics.append(metrics)
        # Fix the canonical ordering from the first non-empty result
        if not canonical_test_ids and metrics:
            canonical_test_ids = list(metrics.keys())

    if not canonical_test_ids:
        return np.empty((len(solutions), 0), dtype=int), []

    # Assemble the B_ik matrix: rows = solutions, columns = test functions
    B_ik = np.array(
        [[metrics.get(tid, 0) for tid in canonical_test_ids] for metrics in per_solution_metrics],
        dtype=int,
    )

    return B_ik, canonical_test_ids


def _default_eq10_aggregation(
    B: np.ndarray,
    correct_mask: np.ndarray,
    lambda_1: float = 0.1,
    lambda_2: float = 0.1,
    lambda_t: float = 0.5,
) -> float:
    """
    Default aggregation: Equation 10 formulation (numpy vectorised).

    Each semantic test function f_k is evaluated against all solutions.
    The reward combines validity (R^1) and discrimination (R^-).

    Formulas:
      R^1_{f_k} = prod(B_ik, i in correct) + (lambda_1 / M+) * sum(B_ik, i in correct)
      R^-_{f_k} = prod(B_ik, i in correct) * (1 - prod(B_ik, i in wrong))
                  - (lambda_2 / M-) * sum(B_ik, i in wrong)
      R_{f_k}   = lambda_t * R^1 + (1 - lambda_t) * R^-
      R         = mean(R_{f_k}) over all k

    Args:
        B: ``np.ndarray`` of shape ``(num_solutions, K)`` with dtype int.
        correct_mask: ``np.ndarray`` of shape ``(num_solutions,)`` with dtype bool.
        lambda_1: Soft validity coefficient.
        lambda_2: Soft discrimination penalty coefficient.
        lambda_t: Weight between validity and discrimination.

    Returns:
        Float reward in roughly ``[-lambda_2, 1 + lambda_1]``.
    """
    B = np.asarray(B, dtype=float)
    correct_mask = np.asarray(correct_mask, dtype=bool)

    K = B.shape[1] if B.ndim == 2 and B.shape[0] > 0 else 0
    if K == 0:
        return 0.0

    B_correct = B[correct_mask]    # (M_plus, K)
    B_wrong = B[~correct_mask]     # (M_minus, K)
    M_plus = B_correct.shape[0]
    M_minus = B_wrong.shape[0]

    # Vectorised across all K test functions
    prod_correct = np.prod(B_correct, axis=0)  # (K,)  — empty → ones
    sum_correct = np.sum(B_correct, axis=0)    # (K,)  — empty → zeros
    prod_wrong = np.prod(B_wrong, axis=0)      # (K,)  — empty → ones
    sum_wrong = np.sum(B_wrong, axis=0)        # (K,)  — empty → zeros

    R1 = prod_correct + lambda_1 / max(M_plus, 1) * sum_correct
    R_minus = prod_correct * (1.0 - prod_wrong) - lambda_2 / max(M_minus, 1) * sum_wrong
    R_fk = lambda_t * R1 + (1.0 - lambda_t) * R_minus

    # ── Print B_ik matrix and per-f_k reward breakdown ──
    col_labels = [f"s+{i}" for i in range(M_plus)] + [f"s-{i}" for i in range(M_minus)]
    col_header = "        " + "  ".join(f"{c:>4}" for c in col_labels)
    print("B_ik matrix (rows=f_k, cols=solutions [correct | wrong]):")
    print(col_header)
    # Display with correct solutions first, then wrong
    B_display = np.hstack([B_correct.T, B_wrong.T]) if M_minus > 0 else B_correct.T  # (K, M)
    for k_idx in range(K):
        vals = "  ".join(f"{int(v):>4}" for v in B_display[k_idx])
        print(f"  f_{k_idx + 1:>2}:  {vals}")
    print("Per-f_k rewards:")
    for k_idx in range(K):
        print(f"  f_{k_idx + 1:>2}: R^1={R1[k_idx]:.4f}, R^-={R_minus[k_idx]:.4f}, R_fk={R_fk[k_idx]:.4f}")
    final_reward = float(np.mean(R_fk))
    print(f"Final reward = {final_reward:.4f} (K={K})")

    return final_reward


def cross_solution_unittest_reward(
    completion_text: str,
    solutions: list[dict],
    reward_aggregation_expr: Optional[str] = None,
    lambda_1: float = 0.1,
    lambda_2: float = 0.1,
    lambda_t: float = 0.5,
    timeout: int = 5,
) -> float:
    """
    Compute reward for generated unit tests by cross-executing them against
    multiple solutions.

    The function first builds the binary matrix **B_ik** via
    :func:`_run_unittest_with_per_test_metrics` (``B[i][k] == 1`` iff solution
    *i* passes test function *k*), then aggregates the matrix into a scalar
    reward.

    Aggregation can be customised through ``reward_aggregation_expr``:

    * **None** (default) — uses the built-in Equation 10 formulation
      (see :func:`_default_eq10_aggregation`).
    * **A Python eval-able string** — the expression is ``eval()``'d with the
      following variables in scope:

        - ``B``  (``np.ndarray``): int array of shape ``(num_solutions, K)``.
        - ``correct_mask`` (``np.ndarray``): bool array of shape ``(num_solutions,)``.
        - ``M_plus`` (``int``): number of correct solutions.
        - ``M_minus`` (``int``): number of wrong solutions.
        - ``K`` (``int``): number of test functions.
        - ``np``: the ``numpy`` module.
        - ``math``: the standard-library ``math`` module.

      The expression must evaluate to a ``float``.

    Args:
        completion_text: Model completion containing unittest code with test methods.
        solutions: List of dicts with keys ``"solve_func"`` (str) and ``"is_correct"`` (bool).
        reward_aggregation_expr: Python eval-able aggregation expression (see above).
        lambda_1: Soft validity coefficient (default 0.1, used by default Eq. 10).
        lambda_2: Soft discrimination penalty coefficient (default 0.1).
        lambda_t: Weight between validity and discrimination (default 0.5).
        timeout: Timeout in seconds for each test execution.

    Returns:
        Float reward.
    """
    test_code = extract_code(completion_text)
    if not test_code and ("def " in completion_text or "class " in completion_text):
        test_code = completion_text

    if not test_code.strip():
        return 0.0

    correct_mask = np.array([s["is_correct"] for s in solutions], dtype=bool)
    M_plus = int(correct_mask.sum())
    M_minus = len(correct_mask) - M_plus

    if M_plus + M_minus == 0:
        return 0.0

    # Build the B_ik matrix via the unified interface
    B, test_ids = _run_unittest_with_per_test_metrics(test_code, solutions, timeout=timeout)

    if B.size == 0 or not test_ids:
        return 0.0

    K = len(test_ids)

    # Aggregate B_ik into a scalar reward
    if reward_aggregation_expr is None:
        # Default: Equation 10
        return _default_eq10_aggregation(
            B, correct_mask,
            lambda_1=lambda_1, lambda_2=lambda_2, lambda_t=lambda_t,
        )
    else:
        # Custom aggregation via eval.
        # NOTE: all variables must live in the *globals* dict because Python 3
        # comprehensions / generator expressions create implicit function scopes
        # that only close over globals — not eval()'s locals dict.
        eval_globals = {
            "__builtins__": {},
            "B": B,
            "correct_mask": correct_mask,
            "M_plus": M_plus,
            "M_minus": M_minus,
            "K": K,
            "np": np,
            "math": math,
            "sum": sum,
            "len": len,
            "min": min,
            "max": max,
            "float": float,
            "int": int,
            "range": range,
            "zip": zip,
            "enumerate": enumerate,
            "all": all,
            "any": any,
            "abs": abs,
        }
        try:
            reward = eval(reward_aggregation_expr, eval_globals)
            return float(reward)
        except Exception as e:
            print(f"[cross_solution_unittest_reward] aggregation eval failed: {e}")
            return 0.0


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """

    def code_format_reward(completions, **kwargs):
        # if there is a language field, use it instead of the default language. This way we can have mixed language training.
        languages = kwargs["language"] if "language" in kwargs else [language] * len(completions)

        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.match(
                rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{sample_language}.*?```.*?\n</answer>$",
                content,
                re.DOTALL | re.MULTILINE,
            )
            for content, sample_language in zip(completion_contents, languages)
        ]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum length of the completion
        soft_punish_cache: Minimum length of the completion. If set to 0, no minimum length is applied.
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "cf_code": update_wrapper(
            partial(
                cf_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                scoring_mode=script_args.code_eval_scoring_mode,
            ),
            cf_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
        "unittest": unittest_reward,
        # "cross_solution_unittest": cross_solution_unittest_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
