"""Tests for benchmark statement parsing.

A loader bug is uniquely dangerous here: a problem that fails to parse or elaborate is scored as
*unproved*, so it depresses every arm equally and hides inside a plausible pass rate rather than
raising an error. These tests use verbatim statements from each real benchmark file.
"""

from __future__ import annotations

import pytest

from prooflens_prover.data.benchmarks import Problem, split_statement

# Verbatim from REAL-Prover's data files — one per benchmark, covering the three shapes.
FATE_M = (
    "import Mathlib\n\n\nvariable {R : Type*} [Ring R] (e : R)\n\n"
    "-- Assume e is idempotent\n"
    "example (h : e * e = e) : ∀ x : R, (x * e - e * x * e) ^ 2 = 0 := sorry\n"
)
PROOFNET = (
    "import Mathlib\n\nopen Complex Filter Function Metric Finset\n"
    "open scoped BigOperators Topology\n\n"
    "theorem exercise_1_13b {f : ℂ → ℂ} (Ω : Set ℂ) (a b : Ω) (h : IsOpen Ω)\n"
    "  (hf : DifferentiableOn ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, (f z).im = c) :\n"
    "  f a = f b := sorry\n"
)
MINIF2F = (
    "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
    "theorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) "
    "(h₁ : v = 1 / 3 * (b * h))\n    (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := sorry\n"
)


class TestSplitStatement:
    def test_fate_m_uses_example_not_theorem(self):
        # FATE-M is the primary discriminating benchmark and uses `example` throughout. A loader
        # recognising only `theorem` would silently drop all 141 problems.
        imports, preamble, decl = split_statement(FATE_M)
        assert imports == "import Mathlib"
        assert decl.startswith("example ")
        assert decl.endswith(":= sorry")

    def test_fate_m_keeps_variable_block_in_preamble(self):
        # The `variable` block binds R, the Ring instance and e. Dropping it makes the statement
        # unelaborable for a reason that has nothing to do with the prover.
        _, preamble, _ = split_statement(FATE_M)
        assert "variable {R : Type*} [Ring R] (e : R)" in preamble

    def test_proofnet_keeps_open_lines(self):
        imports, preamble, decl = split_statement(PROOFNET)
        assert imports == "import Mathlib"
        assert "open Complex Filter Function Metric Finset" in preamble
        assert "open scoped BigOperators Topology" in preamble
        assert decl.startswith("theorem exercise_1_13b")

    def test_multiline_declaration_is_kept_whole(self):
        _, _, decl = split_statement(PROOFNET)
        assert "DifferentiableOn" in decl and "f a = f b := sorry" in decl
        assert decl.count("\n") == 2, "the 3-line declaration must not be truncated"

    def test_minif2f_multiple_imports_and_set_option(self):
        imports, preamble, decl = split_statement(MINIF2F)
        assert imports == "import Mathlib\nimport Aesop"
        assert "set_option maxHeartbeats 0" in preamble
        assert "open BigOperators Real Nat Topology Rat" in preamble
        assert decl.startswith("theorem mathd_algebra_478")

    def test_imports_never_leak_into_preamble_or_declaration(self):
        # Imports must go to the cached REPL environment; re-sending them per problem would be
        # rejected by Lean ("invalid 'import' command").
        for text in (FATE_M, PROOFNET, MINIF2F):
            _, preamble, decl = split_statement(text)
            assert "import " not in preamble
            assert "import " not in decl

    def test_missing_declaration_raises(self):
        # Loud failure beats a silently-empty declaration, which would elaborate to nothing and
        # score as an unprovable problem.
        with pytest.raises(ValueError, match="no declaration"):
            split_statement("import Mathlib\n\nopen Nat\n")

    @pytest.mark.parametrize("kw", ["theorem", "lemma", "example", "abbrev", "instance"])
    def test_recognises_each_declaration_keyword(self, kw):
        _, _, decl = split_statement(f"import Mathlib\n\n{kw} foo : True := sorry\n")
        assert decl.startswith(kw)

    def test_recognises_modifiers_and_attributes(self):
        text = "import Mathlib\n\n@[simp]\nprivate theorem foo : True := sorry\n"
        _, _, decl = split_statement(text)
        assert "theorem foo" in decl


class TestProblem:
    def test_statement_joins_preamble_and_declaration_without_imports(self):
        imports, preamble, decl = split_statement(MINIF2F)
        p = Problem(id="x", source="minif2f_test", imports=imports,
                    preamble=preamble, declaration=decl)
        assert "import" not in p.statement
        assert "set_option maxHeartbeats 0" in p.statement
        assert p.statement.rstrip().endswith(":= sorry")

    def test_statement_handles_empty_preamble(self):
        p = Problem(id="x", source="s", imports="import Mathlib", preamble="",
                    declaration="theorem t : True := sorry")
        assert p.statement == "theorem t : True := sorry"

    def test_serialisable(self):
        import json

        imports, preamble, decl = split_statement(FATE_M)
        p = Problem(id="1597", source="fate_m", imports=imports,
                    preamble=preamble, declaration=decl)
        json.dumps(p.to_dict())
