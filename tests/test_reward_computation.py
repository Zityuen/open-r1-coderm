"""
Self-contained unit tests for the refactored reward computation pipeline.

Tested functions (copied here to avoid importing the full open_r1 stack
which requires transformers, torch, etc.):

    extract_code
    _execute_unittest_suite
    _run_unittest_with_per_test_metrics   →  returns B_ik numpy matrix
    _default_eq10_aggregation             →  Equation 10 reward (numpy)
    cross_solution_unittest_reward        →  end-to-end with eval aggregation

The tests also verify a DEFAULT_EQ10_EXPR config string that encodes the
same Equation 10 formula as a numpy-based Python eval()-able expression,
suitable for YAML configs.
"""

import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Copies of the functions under test (stdlib + numpy, no heavy deps)
# ═══════════════════════════════════════════════════════════════════════

def extract_code(completion: str, language: str | None = "python") -> str:
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def _execute_unittest_suite(
    test_code: str,
    sol_code: str,
    timeout: int = 5,
) -> dict[str, int]:
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
    if not test_code.strip():
        return np.empty((0, 0), dtype=int), []

    per_solution_metrics: list[dict[str, int]] = []
    canonical_test_ids: list[str] = []

    for sol in solutions:
        sol_code = sol["solve_func"]
        metrics = _execute_unittest_suite(test_code, sol_code, timeout=timeout)
        per_solution_metrics.append(metrics)
        if not canonical_test_ids and metrics:
            canonical_test_ids = list(metrics.keys())

    if not canonical_test_ids:
        return np.empty((len(solutions), 0), dtype=int), []

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
    prod_correct = np.prod(B_correct, axis=0)  # (K,)
    sum_correct = np.sum(B_correct, axis=0)    # (K,)
    prod_wrong = np.prod(B_wrong, axis=0)      # (K,) — empty → ones
    sum_wrong = np.sum(B_wrong, axis=0)        # (K,) — empty → zeros

    R1 = prod_correct + lambda_1 / max(M_plus, 1) * sum_correct
    R_minus = prod_correct * (1.0 - prod_wrong) - lambda_2 / max(M_minus, 1) * sum_wrong
    R_fk = lambda_t * R1 + (1.0 - lambda_t) * R_minus

    return float(np.mean(R_fk))


def _eval_aggregation(
    expr: str,
    B: np.ndarray,
    correct_mask: np.ndarray,
) -> float:
    """Replicate the eval path from cross_solution_unittest_reward."""
    B = np.asarray(B, dtype=float)
    correct_mask = np.asarray(correct_mask, dtype=bool)
    M_plus = int(correct_mask.sum())
    M_minus = len(correct_mask) - M_plus
    K = B.shape[1] if B.ndim == 2 and B.shape[0] > 0 else 0
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
    return float(eval(expr, eval_globals))


def cross_solution_unittest_reward(
    completion_text: str,
    solutions: list[dict],
    reward_aggregation_expr: Optional[str] = None,
    lambda_1: float = 0.1,
    lambda_2: float = 0.1,
    lambda_t: float = 0.5,
    timeout: int = 5,
) -> float:
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

    B, test_ids = _run_unittest_with_per_test_metrics(test_code, solutions, timeout=timeout)

    if B.size == 0 or not test_ids:
        return 0.0

    K = len(test_ids)

    if reward_aggregation_expr is None:
        return _default_eq10_aggregation(
            B, correct_mask,
            lambda_1=lambda_1, lambda_2=lambda_2, lambda_t=lambda_t,
        )
    else:
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


# ═══════════════════════════════════════════════════════════════════════
# Config-compatible eval expression for Equation 10 (default aggregation)
# ═══════════════════════════════════════════════════════════════════════

# This is the exact Equation 10 formula expressed as a single numpy-based
# Python eval()-able string.  It can be placed directly in a YAML config:
#
#   unittest_reward_aggregation: >
#     float(np.mean( ... ))
#
# Parameters baked-in: lambda_1 = 0.1, lambda_2 = 0.1, lambda_t = 0.5
DEFAULT_EQ10_EXPR = (
    "float(np.mean("
    "  0.5 * (np.prod(B[correct_mask], axis=0) + 0.1 / max(M_plus, 1) * np.sum(B[correct_mask], axis=0))"
    "  + 0.5 * (np.prod(B[correct_mask], axis=0) * (1.0 - np.prod(B[~correct_mask], axis=0))"
    "  - 0.1 / max(M_minus, 1) * np.sum(B[~correct_mask], axis=0))"
    "))"
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

CORRECT_SOLUTION = "def add(a, b):\n    return a + b\n"
WRONG_SOLUTION_OFF_BY_ONE = "def add(a, b):\n    return a + b + 1\n"
WRONG_SOLUTION_SUBTRACT = "def add(a, b):\n    return a - b\n"

TEST_CODE = (
    "import unittest\n"
    "\n"
    "class TestAdd(unittest.TestCase):\n"
    "    def test_add_positive(self):\n"
    "        self.assertEqual(add(2, 3), 5)\n"
    "\n"
    "    def test_add_zero(self):\n"
    "        self.assertEqual(add(0, 0), 0)\n"
    "\n"
    "    def test_add_negative(self):\n"
    "        self.assertEqual(add(-1, -2), -3)\n"
)

COMPLETION_TEXT = f"```python\n{TEST_CODE}```"

SOLUTIONS = [
    {"solve_func": CORRECT_SOLUTION, "is_correct": True},
    {"solve_func": WRONG_SOLUTION_OFF_BY_ONE, "is_correct": False},
    {"solve_func": WRONG_SOLUTION_SUBTRACT, "is_correct": False},
]


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════

class TestExecuteUnittestSuite(unittest.TestCase):
    """Test _execute_unittest_suite (single-solution subprocess runner)."""

    def test_correct_solution_all_pass(self):
        metrics = _execute_unittest_suite(TEST_CODE, CORRECT_SOLUTION)
        self.assertEqual(len(metrics), 3, "Should discover 3 test methods")
        self.assertTrue(all(v == 1 for v in metrics.values()),
                        f"All tests should pass: {metrics}")

    def test_wrong_solution_all_fail(self):
        metrics = _execute_unittest_suite(TEST_CODE, WRONG_SOLUTION_OFF_BY_ONE)
        self.assertEqual(len(metrics), 3)
        self.assertTrue(all(v == 0 for v in metrics.values()),
                        f"All tests should fail: {metrics}")

    def test_partial_pass(self):
        metrics = _execute_unittest_suite(TEST_CODE, WRONG_SOLUTION_SUBTRACT)
        self.assertEqual(len(metrics), 3)
        # subtract: add(0,0)=0-0=0 passes; add(2,3)=2-3=-1 fails; add(-1,-2)=-1-(-2)=1 fails
        self.assertEqual(sum(metrics.values()), 1,
                         f"Exactly 1 test should pass: {metrics}")

    def test_empty_inputs(self):
        self.assertEqual(_execute_unittest_suite("", CORRECT_SOLUTION), {})
        self.assertEqual(_execute_unittest_suite(TEST_CODE, ""), {})
        self.assertEqual(_execute_unittest_suite("", ""), {})


class TestRunUnittestWithPerTestMetrics(unittest.TestCase):
    """Test _run_unittest_with_per_test_metrics (B_ik numpy matrix builder)."""

    def test_b_ik_shape(self):
        B, test_ids = _run_unittest_with_per_test_metrics(TEST_CODE, SOLUTIONS)
        self.assertIsInstance(B, np.ndarray)
        self.assertEqual(B.shape, (3, 3), "3 solutions × 3 tests")
        self.assertEqual(len(test_ids), 3)

    def test_b_ik_row_sums(self):
        B, _ = _run_unittest_with_per_test_metrics(TEST_CODE, SOLUTIONS)
        self.assertEqual(B[0].sum(), 3, "Correct solution passes all 3 tests")
        self.assertEqual(B[1].sum(), 0, "Off-by-one solution fails all tests")
        self.assertEqual(B[2].sum(), 1, "Subtract solution passes exactly 1 test")

    def test_b_ik_values_are_binary(self):
        B, _ = _run_unittest_with_per_test_metrics(TEST_CODE, SOLUTIONS)
        self.assertTrue(np.all((B == 0) | (B == 1)), "All values should be 0 or 1")

    def test_empty_test_code(self):
        B, ids = _run_unittest_with_per_test_metrics("   ", SOLUTIONS)
        self.assertEqual(B.size, 0)
        self.assertEqual(ids, [])

    def test_single_solution(self):
        sols = [{"solve_func": CORRECT_SOLUTION, "is_correct": True}]
        B, test_ids = _run_unittest_with_per_test_metrics(TEST_CODE, sols)
        self.assertEqual(B.shape[0], 1)
        self.assertEqual(B[0].sum(), 3)


class TestDefaultEq10Aggregation(unittest.TestCase):
    """Test _default_eq10_aggregation with hand-computed expected values."""

    def test_perfect_discrimination(self):
        """Correct passes all, wrong fails all → high reward."""
        B = np.array([[1, 1],   # correct: passes both
                      [0, 0]])  # wrong: fails both
        cm = np.array([True, False])
        result = _default_eq10_aggregation(B, cm, 0.1, 0.1, 0.5)
        self.assertAlmostEqual(result, 1.05, places=9)

    def test_no_discrimination(self):
        """Both solutions pass everything → low discrimination reward."""
        B = np.array([[1, 1],
                      [1, 1]])
        cm = np.array([True, False])
        result = _default_eq10_aggregation(B, cm, 0.1, 0.1, 0.5)
        self.assertAlmostEqual(result, 0.5, places=9)

    def test_all_fail(self):
        """All tests fail on all solutions → zero reward."""
        B = np.array([[0, 0],
                      [0, 0]])
        cm = np.array([True, False])
        result = _default_eq10_aggregation(B, cm, 0.1, 0.1, 0.5)
        self.assertAlmostEqual(result, 0.0, places=9)

    def test_1_correct_2_wrong_3_tests(self):
        """Hand-computed scenario matching the add-problem fixture."""
        B = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [0, 1, 0]])
        cm = np.array([True, False, False])
        expected = (1.05 + 1.025 + 1.05) / 3.0
        result = _default_eq10_aggregation(B, cm, 0.1, 0.1, 0.5)
        self.assertAlmostEqual(result, expected, places=9)

    def test_2_correct_1_wrong_2_tests(self):
        """Multiple correct solutions."""
        B = np.array([[1, 1],   # correct 0: passes both
                      [1, 0],   # correct 1: passes test 0 only
                      [0, 1]])  # wrong 0: passes test 1 only
        cm = np.array([True, True, False])
        expected = (1.05 + (-0.025)) / 2.0
        result = _default_eq10_aggregation(B, cm, 0.1, 0.1, 0.5)
        self.assertAlmostEqual(result, expected, places=9)

    def test_different_lambdas(self):
        """Vary lambda_t to weight validity vs. discrimination."""
        B = np.array([[1, 1],
                      [0, 0]])
        cm = np.array([True, False])
        result_validity = _default_eq10_aggregation(B, cm, 0.1, 0.1, 1.0)
        self.assertAlmostEqual(result_validity, 1.1, places=9)
        result_disc = _default_eq10_aggregation(B, cm, 0.1, 0.1, 0.0)
        self.assertAlmostEqual(result_disc, 1.0, places=9)

    def test_empty_matrix(self):
        self.assertAlmostEqual(
            _default_eq10_aggregation(np.empty((0, 0), dtype=int), np.array([], dtype=bool)),
            0.0,
        )

    def test_only_correct_solutions(self):
        """No wrong solutions → discrimination term has no effect."""
        B = np.array([[1, 0], [1, 1]])
        cm = np.array([True, True])
        # k=0: prod_c=1, sum_c=2 → R1=1+0.1/2*2=1.1
        #   prod_w=1 (empty), sum_w=0 → R-=1*(1-1)-0=0
        #   R_fk=0.5*1.1+0.5*0=0.55
        # k=1: prod_c=0, sum_c=1 → R1=0+0.1/2*1=0.05
        #   R-=0*(1-1)-0=0
        #   R_fk=0.5*0.05+0.5*0=0.025
        # Final=(0.55+0.025)/2=0.2875
        result = _default_eq10_aggregation(B, cm, 0.1, 0.1, 0.5)
        self.assertAlmostEqual(result, 0.2875, places=9)


class TestEvalAggregation(unittest.TestCase):
    """Test eval()-based custom aggregation expressions (numpy)."""

    def test_simple_pass_rate(self):
        B = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [0, 1, 0]])
        cm = np.array([True, False, False])
        expr = "float(np.sum(B) / B.size)"
        result = _eval_aggregation(expr, B, cm)
        # (3+0+1) / 9 = 4/9
        self.assertAlmostEqual(result, 4.0 / 9.0, places=9)

    def test_correct_only_pass_rate(self):
        B = np.array([[1, 1, 0],
                      [0, 0, 0]])
        cm = np.array([True, False])
        expr = "float(np.mean(B[correct_mask])) if M_plus > 0 else 0.0"
        result = _eval_aggregation(expr, B, cm)
        # correct sol passes 2/3
        self.assertAlmostEqual(result, 2.0 / 3.0, places=9)

    def test_discrimination_fraction(self):
        """Fraction of tests that pass all correct AND fail ≥1 wrong."""
        B = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [0, 1, 0]])
        cm = np.array([True, False, False])
        # All-correct pass each test AND at least one wrong fails each test
        expr = (
            "float(np.mean("
            "  np.all(B[correct_mask], axis=0) & ~np.all(B[~correct_mask], axis=0)"
            "))"
        )
        result = _eval_aggregation(expr, B, cm)
        # all 3 tests discriminate → 1.0
        self.assertAlmostEqual(result, 1.0, places=9)

    def test_numpy_prod_available(self):
        """Verify np.prod works inside eval."""
        B = np.array([[1, 1], [1, 0]])
        cm = np.array([True, False])
        expr = "float(np.prod(B[0]) + np.prod(B[1]))"
        result = _eval_aggregation(expr, B, cm)
        # prod([1,1])=1, prod([1,0])=0 → 1.0
        self.assertAlmostEqual(result, 1.0, places=9)

    def test_bad_expression_raises(self):
        B = np.array([[1]])
        cm = np.array([True])
        with self.assertRaises(Exception):
            _eval_aggregation("UNDEFINED_VAR", B, cm)


class TestDefaultEq10Expr(unittest.TestCase):
    """Verify that DEFAULT_EQ10_EXPR (the numpy config string) matches
    _default_eq10_aggregation for several B_ik matrices."""

    def _compare(self, B, cm, msg=""):
        B = np.asarray(B, dtype=float)
        cm = np.asarray(cm, dtype=bool)
        expected = _default_eq10_aggregation(B, cm, 0.1, 0.1, 0.5)
        result = _eval_aggregation(DEFAULT_EQ10_EXPR, B, cm)
        self.assertAlmostEqual(result, expected, places=9, msg=msg)

    def test_perfect_discrimination(self):
        self._compare(
            [[1, 1], [0, 0]],
            [True, False],
            "perfect discrimination",
        )

    def test_no_discrimination(self):
        self._compare(
            [[1, 1], [1, 1]],
            [True, False],
            "no discrimination",
        )

    def test_all_fail(self):
        self._compare(
            [[0, 0], [0, 0]],
            [True, False],
            "all fail",
        )

    def test_1_correct_2_wrong_3_tests(self):
        self._compare(
            [[1, 1, 1], [0, 0, 0], [0, 1, 0]],
            [True, False, False],
            "1 correct 2 wrong 3 tests",
        )

    def test_2_correct_1_wrong_2_tests(self):
        self._compare(
            [[1, 1], [1, 0], [0, 1]],
            [True, True, False],
            "2 correct 1 wrong 2 tests",
        )

    def test_only_correct_solutions(self):
        self._compare(
            [[1, 0], [1, 1]],
            [True, True],
            "only correct solutions",
        )

    def test_single_test_single_solution(self):
        self._compare(
            [[1]],
            [True],
            "single test single solution",
        )

    def test_large_matrix(self):
        """3 correct, 3 wrong, 5 tests — randomish pattern."""
        B = [
            [1, 1, 0, 1, 1],  # correct
            [1, 0, 1, 1, 0],  # correct
            [0, 1, 1, 0, 1],  # correct
            [1, 0, 0, 0, 1],  # wrong
            [0, 1, 0, 1, 0],  # wrong
            [0, 0, 1, 0, 0],  # wrong
        ]
        cm = [True, True, True, False, False, False]
        self._compare(B, cm, "large matrix")


class TestCrossSolutionUnittestReward(unittest.TestCase):
    """End-to-end tests for cross_solution_unittest_reward."""

    def test_default_aggregation(self):
        reward = cross_solution_unittest_reward(
            COMPLETION_TEXT, SOLUTIONS,
            reward_aggregation_expr=None,
        )
        # Tests discriminate well → reward should be > 0.5
        self.assertGreater(reward, 0.5)
        self.assertLess(reward, 1.2)

    def test_custom_aggregation_pass_rate(self):
        expr = "float(np.sum(B) / B.size)"
        reward = cross_solution_unittest_reward(
            COMPLETION_TEXT, SOLUTIONS,
            reward_aggregation_expr=expr,
        )
        # (3+0+1)/(3*3) = 4/9 ≈ 0.444
        self.assertAlmostEqual(reward, 4.0 / 9.0, places=4)

    def test_config_eq10_matches_default(self):
        """DEFAULT_EQ10_EXPR config string should match the default function path."""
        r_default = cross_solution_unittest_reward(
            COMPLETION_TEXT, SOLUTIONS,
            reward_aggregation_expr=None,
        )
        r_expr = cross_solution_unittest_reward(
            COMPLETION_TEXT, SOLUTIONS,
            reward_aggregation_expr=DEFAULT_EQ10_EXPR,
        )
        self.assertAlmostEqual(r_default, r_expr, places=9,
                               msg="Config eq10 expr should match built-in default")

    def test_empty_completion(self):
        self.assertEqual(cross_solution_unittest_reward("", SOLUTIONS), 0.0)

    def test_empty_solutions(self):
        self.assertEqual(cross_solution_unittest_reward(COMPLETION_TEXT, []), 0.0)

    def test_bad_aggregation_returns_zero(self):
        reward = cross_solution_unittest_reward(
            COMPLETION_TEXT, SOLUTIONS,
            reward_aggregation_expr="INVALID!!!",
        )
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
