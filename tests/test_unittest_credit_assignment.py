"""Unit tests for ``open_r1.unittest_credit`` (each public and private helper)."""

import ast
import unittest

import numpy as np

import open_r1.unittest_credit as uc


class _CharTokenizer:
    """Minimal fast-tokenizer stand-in: one token per character."""

    is_fast = True

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, truncation=False):
        pairs = [(i, i + 1) for i in range(len(text))]
        # Flat ``offset_mapping`` (one sequence), not batch-padded outer list.
        return {"input_ids": list(range(len(text))), "offset_mapping": pairs}


class _SlowTokenizer:
    is_fast = False

    def __call__(self, *args, **kwargs):
        raise AssertionError("should not be called when is_fast is False")


class TestLineStartOffsets(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(uc._line_start_offsets(""), [0])

    def test_single_line(self):
        self.assertEqual(uc._line_start_offsets("abc"), [0])

    def test_multiple_lines(self):
        text = "a\nbb\nccc"
        starts = uc._line_start_offsets(text)
        self.assertEqual(starts[0], 0)
        self.assertEqual(text[starts[1]], "b")
        self.assertEqual(text[starts[2]], "c")


class TestAstNodeCharSpan(unittest.TestCase):
    def test_function_def_span_covers_full_node(self):
        src = "def foo():\n    return 1\n"
        tree = ast.parse(src)
        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef)
        line_starts = uc._line_start_offsets(src)
        start, end = uc._ast_node_char_span(fn, line_starts)
        self.assertEqual(src[start:end], src.rstrip("\n"))


class TestClassInheritsTestCase(unittest.TestCase):
    def test_explicit_unittest_testcase(self):
        tree = ast.parse("class C(unittest.TestCase):\n    pass\n")
        cls = tree.body[0]
        assert isinstance(cls, ast.ClassDef)
        self.assertTrue(uc._class_inherits_testcase(cls))

    def test_name_heuristic_starts_with_test(self):
        tree = ast.parse("class TestStuff:\n    pass\n")
        cls = tree.body[0]
        assert isinstance(cls, ast.ClassDef)
        self.assertTrue(uc._class_inherits_testcase(cls))

    def test_plain_class_false(self):
        tree = ast.parse("class Helper:\n    pass\n")
        cls = tree.body[0]
        assert isinstance(cls, ast.ClassDef)
        self.assertFalse(uc._class_inherits_testcase(cls))


class TestOrderedUnittestIdsAndSpansInSource(unittest.TestCase):
    def test_empty_and_whitespace(self):
        self.assertEqual(uc._ordered_unittest_ids_and_spans_in_source(""), [])
        self.assertEqual(uc._ordered_unittest_ids_and_spans_in_source("   \n"), [])

    def test_syntax_error_returns_empty(self):
        self.assertEqual(uc._ordered_unittest_ids_and_spans_in_source("def oops"), [])

    def test_sorted_test_method_names(self):
        code = """
import unittest

class TestFoo(unittest.TestCase):
    def test_b(self):
        self.assertTrue(True)

    def test_a(self):
        self.assertTrue(True)
"""
        rows = uc._ordered_unittest_ids_and_spans_in_source(code)
        names = [r[0].split(".")[-1] for r in rows]
        self.assertEqual(names, ["test_a", "test_b"])

    def test_module_qualname_prefix(self):
        code = "import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n        pass\n"
        rows = uc._ordered_unittest_ids_and_spans_in_source(code, module_qualname="__main__")
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0][0].startswith("__main__.T.test_x"))

    def test_spans_non_decreasing_and_within_source(self):
        code = "import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n        pass\n"
        rows = uc._ordered_unittest_ids_and_spans_in_source(code)
        self.assertEqual(len(rows), 1)
        _, s, e = rows[0]
        self.assertLessEqual(0, s <= e <= len(code))


class TestFindTestCodeSliceInBody(unittest.TestCase):
    def test_exact_substring(self):
        body = "before\nCODE\nafter"
        self.assertEqual(uc._find_test_code_slice_in_body(body, "CODE"), (7, 11))

    def test_stripped_fallback(self):
        body = "x\n  abc  \ny"
        self.assertEqual(uc._find_test_code_slice_in_body(body, "  abc  \n"), (2, 10))

    def test_not_found(self):
        self.assertIsNone(uc._find_test_code_slice_in_body("hello", "missing"))

    def test_empty_test_code(self):
        self.assertIsNone(uc._find_test_code_slice_in_body("x", ""))


class TestTokenIndicesOverlappingSpan(unittest.TestCase):
    def test_non_fast_tokenizer_returns_empty(self):
        tok = _SlowTokenizer()
        self.assertEqual(
            uc._token_indices_overlapping_span(tok, "abcdef", 1, 4, max_tokens=100),
            [],
        )

    def test_invalid_span_empty(self):
        tok = _CharTokenizer()
        self.assertEqual(uc._token_indices_overlapping_span(tok, "abc", 2, 2, 10), [])
        self.assertEqual(uc._token_indices_overlapping_span(tok, "abc", 5, 10, 10), [])

    def test_overlap_selects_token_indices(self):
        tok = _CharTokenizer()
        # chars 1,2,3 -> token indices 1,2,3
        self.assertEqual(uc._token_indices_overlapping_span(tok, "abcdef", 1, 4, 100), [1, 2, 3])

    def test_clamps_to_body_length(self):
        tok = _CharTokenizer()
        self.assertEqual(uc._token_indices_overlapping_span(tok, "ab", 0, 100, 100), [0, 1])

    def test_respects_max_tokens(self):
        tok = _CharTokenizer()
        self.assertEqual(uc._token_indices_overlapping_span(tok, "abcd", 0, 4, max_tokens=2), [0, 1])


class TestBuildPerTokenUnittestAdvantages(unittest.TestCase):
    def test_empty_ids_or_advantages(self):
        z = uc.build_per_token_unittest_advantages(
            _CharTokenizer(), "x", "y", [], [1.0], 5,
        )
        self.assertEqual(z.shape, (5,))
        self.assertEqual(z.sum(), 0.0)

    def test_no_test_code_in_body_returns_zeros(self):
        adv = uc.build_per_token_unittest_advantages(
            _CharTokenizer(),
            "no code block here",
            "class T(unittest.TestCase):\n    def test_x(self):\n        pass\n",
            ["__main__.T.test_x"],
            [0.5],
            20,
        )
        self.assertEqual(adv.sum(), 0.0)

    def test_assigns_constant_on_overlapping_tokens(self):
        body = 'prefix\n```python\nimport unittest\n\nclass T(unittest.TestCase):\n    def test_x(self):\n        assert 1\n```\n'
        test_code = """import unittest

class T(unittest.TestCase):
    def test_x(self):
        assert 1
"""
        per_tok = uc.build_per_token_unittest_advantages(
            _CharTokenizer(),
            body,
            test_code,
            ["__main__.T.test_x"],
            [0.7],
            len(body),
        )
        self.assertGreater(per_tok.sum(), 0.0)
        self.assertTrue(np.all(per_tok[np.where(per_tok > 0)] == 0.7))

    def test_unknown_canonical_id_skipped(self):
        body = "import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n        pass\n"
        per_tok = uc.build_per_token_unittest_advantages(
            _CharTokenizer(),
            body,
            body,
            ["__main__.Wrong.test_y"],
            [1.0],
            len(body),
        )
        self.assertEqual(per_tok.sum(), 0.0)


class TestPooledNormalizePerTestValues(unittest.TestCase):
    def test_group_size_four_pools_fourteen_slots(self):
        values = [
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]
        raw = [list(map(float, v)) for v in values]
        normed = uc.pooled_normalize_per_test_values(raw, group_size=4, scale_rewards=False)
        pool = [x for row in raw for x in row]
        mean = float(np.mean(pool))
        self.assertEqual(len(normed), 4)
        for row, orig in zip(normed, raw):
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(len(row), len(orig))
            for a, b in zip(row, orig):
                self.assertAlmostEqual(a, b - mean, places=5)

    def test_group_size_non_positive_returns_input(self):
        inp: list = [[1.0], [2.0]]
        self.assertIs(uc.pooled_normalize_per_test_values(inp, 0, False), inp)
        self.assertIs(uc.pooled_normalize_per_test_values(inp, -1, False), inp)

    def test_none_rows_unchanged_in_output(self):
        out = uc.pooled_normalize_per_test_values([None, [1.0, 2.0], None, [3.0, 4.0]], 2, False)
        self.assertEqual(len(out), 4)
        self.assertIsNone(out[0])
        self.assertIsNotNone(out[1])
        self.assertIsNone(out[2])
        self.assertIsNotNone(out[3])

    def test_empty_pool_skips_group(self):
        out = uc.pooled_normalize_per_test_values([None, None, None, None], 4, False)
        self.assertTrue(all(x is None for x in out))

    def test_scale_rewards_divides_by_std(self):
        raw = [[0.0], [2.0]]
        out = uc.pooled_normalize_per_test_values(raw, group_size=2, scale_rewards=True, eps=1e-4)
        self.assertIsNotNone(out[0])
        self.assertIsNotNone(out[1])
        assert out[0] is not None and out[1] is not None
        pool_std = float(np.std([0.0, 2.0]))
        mean = 1.0
        self.assertAlmostEqual(out[0][0], (0.0 - mean) / (pool_std + 1e-4), places=5)
        self.assertAlmostEqual(out[1][0], (2.0 - mean) / (pool_std + 1e-4), places=5)

    def test_incomplete_trailing_group_not_normalized(self):
        # 5 rows, group_size 4 -> only first 4 processed; row 4 stays None in out
        raw = [[1.0], [1.0], [1.0], [1.0], [99.0]]
        out = uc.pooled_normalize_per_test_values(raw, 4, False)
        self.assertIsNotNone(out[0])
        self.assertIsNone(out[4])


if __name__ == "__main__":
    unittest.main()
