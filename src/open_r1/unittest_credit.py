"""
AST + tokenizer helpers for assigning per-test unittest rewards to token spans.
"""

from __future__ import annotations

import ast
from typing import Any, Optional

import numpy as np


def _line_start_offsets(text: str) -> list[int]:
    offsets = [0]
    pos = 0
    for ch in text:
        pos += 1
        if ch == "\n":
            offsets.append(pos)
    return offsets


def _ast_node_char_span(node: ast.AST, line_starts: list[int]) -> tuple[int, int]:
    """Convert 1-based ast line/col to absolute char offsets in source."""
    start_line = getattr(node, "lineno", 1) - 1
    start_col = getattr(node, "col_offset", 0) or 0
    end_line = getattr(node, "end_lineno", getattr(node, "lineno", 1)) - 1
    end_col = getattr(node, "end_col_offset", 0) or 0
    start = line_starts[start_line] + start_col
    end = line_starts[end_line] + end_col
    return start, end


def _class_inherits_testcase(class_node: ast.ClassDef) -> bool:
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id == "TestCase":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "TestCase":
            return True
    name = class_node.name
    return name.startswith("Test") or name.endswith("Test")


def _ordered_unittest_ids_and_spans_in_source(
    test_code: str, module_qualname: str = "__main__"
) -> list[tuple[str, int, int]]:
    """
    Return (test_id, char_start, char_end) for each unittest method, in unittest discovery order:
    TestCase subclasses in source order; within each class, test methods sorted by name
    (matching TestLoader.getTestCaseNames).
    """
    if not test_code.strip():
        return []
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []

    line_starts = _line_start_offsets(test_code)
    out: list[tuple[str, int, int]] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not _class_inherits_testcase(node):
            continue
        candidates: list[tuple[str, ast.FunctionDef]] = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name.startswith("test"):
                candidates.append((item.name, item))
        for name, fn in sorted(candidates, key=lambda x: x[0]):
            tid = f"{module_qualname}.{node.name}.{name}"
            start, end = _ast_node_char_span(fn, line_starts)
            end = max(end, start)
            out.append((tid, start, end))
    return out


def _find_test_code_slice_in_body(completion_body: str, test_code: str) -> Optional[tuple[int, int]]:
    """Return [start, end) char offsets of test_code inside completion_body."""
    if not test_code:
        return None
    idx = completion_body.find(test_code)
    if idx >= 0:
        return idx, idx + len(test_code)
    stripped = test_code.strip()
    if not stripped:
        return None
    idx = completion_body.find(stripped)
    if idx >= 0:
        return idx, idx + len(stripped)
    return None


def _token_indices_overlapping_span(
    tokenizer: Any,
    completion_body: str,
    char_start: int,
    char_end: int,
    max_tokens: int,
) -> list[int]:
    """Map a [char_start, char_end) span in completion_body to completion token indices."""
    if char_start >= char_end or char_end <= 0:
        return []
    char_start = max(0, char_start)
    char_end = min(len(completion_body), char_end)

    is_fast = getattr(tokenizer, "is_fast", False)
    if not is_fast:
        return []

    enc = tokenizer(
        completion_body,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    offsets = enc["offset_mapping"] if "offset_mapping" in enc else enc.get("offset_mapping")
    if offsets is None:
        return []
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], (list, tuple)) and len(offsets[0]) == 2:
        off_list = offsets
    elif offsets and isinstance(offsets[0], (list, tuple)):
        off_list = offsets[0]
    else:
        return []

    out: list[int] = []
    for ti, pair in enumerate(off_list):
        if ti >= max_tokens:
            break
        ts, te = int(pair[0]), int(pair[1])
        if te <= char_start or ts >= char_end:
            continue
        if ts < te:
            out.append(ti)
    return out


def build_per_token_unittest_advantages(
    tokenizer: Any,
    completion_body: str,
    test_code: str,
    canonical_test_ids: list[str],
    per_test_advantages: list[float],
    completion_length: int,
) -> np.ndarray:
    """
    Build shape (completion_length,) with per_test_advantages[k] on tokens overlapping
    canonical_test_ids[k]. Non-test tokens stay 0. Later k overwrites overlaps.
    """
    adv = np.zeros((completion_length,), dtype=np.float32)
    if not canonical_test_ids or not per_test_advantages:
        return adv

    slc = _find_test_code_slice_in_body(completion_body, test_code)
    if slc is None:
        return adv
    code_origin, _ = slc

    id_to_span: dict[str, tuple[int, int]] = {}
    for tid, rel_start, rel_end in _ordered_unittest_ids_and_spans_in_source(test_code):
        abs_start = code_origin + rel_start
        abs_end = code_origin + rel_end
        id_to_span[tid] = (abs_start, abs_end)

    for k, tid in enumerate(canonical_test_ids):
        if k >= len(per_test_advantages):
            break
        span = id_to_span.get(tid)
        if span is None:
            continue
        token_ixs = _token_indices_overlapping_span(
            tokenizer, completion_body, span[0], span[1], completion_length
        )
        val = float(per_test_advantages[k])
        for ti in token_ixs:
            if 0 <= ti < completion_length:
                adv[ti] = val
    return adv


def pooled_normalize_per_test_values(
    values_by_row: list[Optional[list[float]]],
    group_size: int,
    scale_rewards: bool,
    eps: float = 1e-4,
) -> list[Optional[list[float]]]:
    """
    For each contiguous group of `group_size` rows, pool all per-test floats and
    return centered (/ scaled) copies per row. Rows with None are left None.
    """
    if group_size <= 0:
        return values_by_row
    n = len(values_by_row)
    out: list[Optional[list[float]]] = [None] * n
    for g in range(n // group_size):
        lo = g * group_size
        hi = lo + group_size
        pool: list[float] = []
        for i in range(lo, hi):
            v = values_by_row[i]
            if v is not None:
                pool.extend(float(x) for x in v)
        if not pool:
            continue
        mean = float(np.mean(pool))
        std = float(np.std(pool))
        for i in range(lo, hi):
            v = values_by_row[i]
            if v is None:
                continue
            centered = [float(x) - mean for x in v]
            if scale_rewards and std > eps:
                centered = [c / (std + eps) for c in centered]
            out[i] = centered
    return out
