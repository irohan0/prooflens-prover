"""Tests for the proof-acceptance logic — the correctness property the whole study rests on.

If `verdict_from_repl` ever returns PROVED for something Lean would not accept, every headline
number in the dissertation is inflated and no downstream check would catch it. These tests are
hermetic: no Lean, no GPU, no network.
"""

from __future__ import annotations

import pytest

from prooflens_prover.lean.backend import (
    Outcome,
    ProofState,
    TacticPolicy,
    TacticResult,
    contains_cheat,
    verdict_from_repl,
)
from prooflens_prover.lean.leaninteract_backend import _as_sorry_statement, _last_toplevel_assign


class TestVerdict:
    def test_completed_with_no_goals_is_proved(self):
        assert verdict_from_repl("Completed", [], []) is Outcome.PROVED

    def test_empty_status_no_goals_is_proved(self):
        # The REPL omits proof_status in some versions; no goals + no errors is a closed proof.
        assert verdict_from_repl("", [], []) is Outcome.PROVED

    def test_open_goals_is_progress(self):
        assert verdict_from_repl(
            "Incomplete: open goals remain", ["⊢ 1 = 1"], []
        ) is Outcome.PROGRESS

    def test_error_severity_beats_everything(self):
        # A tactic can "apply" while Lean reports an error; such a proof does not compile.
        assert verdict_from_repl(
            "Completed", [], ["unknown identifier 'foo'"], error_severities=["error"]
        ) is Outcome.ERROR

    # -- the checks that stop a non-proof being scored as a proof -----------------------------
    @pytest.mark.parametrize(
        "status,messages",
        [
            ("Incomplete: contains sorry", []),
            ("Completed", ["declaration uses 'sorry'"]),
            ("Completed", ["uses sorryAx"]),
            ("Completed", ["admit was used"]),
        ],
    )
    def test_sorry_anywhere_is_never_proved(self, status, messages):
        assert verdict_from_repl(status, [], messages) is not Outcome.PROVED

    def test_unrecognised_status_does_not_default_to_proved(self):
        # Refuse to guess in the generous direction when the REPL says something we do not model.
        assert verdict_from_repl("Something New", [], []) is not Outcome.PROVED

    def test_incomplete_with_no_goals_is_error_not_proved(self):
        assert verdict_from_repl("Incomplete: contains sorry", [], []) is Outcome.ERROR


class TestContainsCheat:
    @pytest.mark.parametrize(
        "text",
        [
            "sorry",
            "exact sorry",
            "by\n  sorry",
            "admit",
            "sorryAx",
            # Lean's OWN warning text — quoted. Missing this form would score a sorry-tainted
            # proof as PROVED, which is the single worst failure mode available to this project.
            "declaration uses 'sorry'",
            "warning: declaration uses 'sorry'",
        ],
    )
    def test_detects_real_cheats(self, text):
        assert contains_cheat(text)

    @pytest.mark.parametrize(
        "text",
        [
            "exact Nat.sorry_free_lemma",   # identifier that merely contains the substring
            "simp [notsorry]",
            "exact sorryville",
            "rw [Foo.admitted]",
        ],
    )
    def test_no_false_positives_on_substrings(self, text):
        # Identifier-boundary matching: a legitimately-named lemma must not be rejected, or we
        # would silently lose provable theorems and understate every arm.
        assert not contains_cheat(text)

    @pytest.mark.parametrize("text", ["exact sorry'", "exact h'sorry"])
    def test_over_rejects_apostrophe_forms_by_design(self, text):
        # Documented, deliberate over-rejection. `'` is treated as a quote so Lean's own
        # `declaration uses 'sorry'` warning is caught; the cost is that identifier-with-apostrophe
        # forms also match. No such Mathlib identifier exists, and rejecting one candidate tactic
        # is far cheaper than accepting one false proof. See `_CHEAT_RE` in backend.py.
        assert contains_cheat(text)


class TestTacticPolicy:
    def test_rejects_sorry(self):
        assert TacticPolicy().reject_reason("exact sorry") is not None

    def test_allows_ordinary_tactic(self):
        assert TacticPolicy().reject_reason("simp [Nat.add_comm]") is None

    def test_rejects_empty(self):
        assert TacticPolicy().reject_reason("   ") is not None

    def test_native_decide_gated_by_flag(self):
        assert TacticPolicy().reject_reason("native_decide") is not None
        assert TacticPolicy(allow_native_decide=True).reject_reason("native_decide") is None

    def test_length_cap(self):
        assert TacticPolicy(max_tactic_chars=10).reject_reason("simp " * 100) is not None


class TestSorryStatement:
    def test_appends_body_when_absent(self):
        assert _as_sorry_statement("theorem foo : 1 = 1") == "theorem foo : 1 = 1 := by sorry"

    def test_replaces_an_existing_proof_body(self):
        # A supplied real proof must be stripped, or the "root state" would already be solved and
        # the problem would score as trivially proved.
        stripped = _as_sorry_statement("theorem foo : 1 = 1 := by rfl")
        assert stripped == "theorem foo : 1 = 1 := by sorry"

    def test_idempotent(self):
        once = _as_sorry_statement("theorem foo : 1 = 1")
        assert _as_sorry_statement(once) == once

    def test_ignores_assign_inside_brackets(self):
        # `:=` inside a binder must not be mistaken for the proof body separator.
        stmt = "theorem foo (h : Nat := 3) : 1 = 1"
        assert _as_sorry_statement(stmt) == "theorem foo (h : Nat := 3) : 1 = 1 := by sorry"

    def test_last_toplevel_assign_indexing(self):
        assert _last_toplevel_assign("a := b") == 2
        assert _last_toplevel_assign("(x := 1) : T") == -1
        assert _last_toplevel_assign("no assignment here") == -1


class TestResultTypes:
    def test_proof_state_solved_and_pp(self):
        assert ProofState(pid=0, goals=()).is_solved
        assert not ProofState(pid=0, goals=("⊢ True",)).is_solved
        assert ProofState(pid=1, goals=("a", "b")).pp == "a\n\nb"

    def test_tactic_result_ok(self):
        assert TacticResult(Outcome.PROVED).ok
        assert TacticResult(Outcome.PROGRESS).ok
        for bad in (Outcome.ERROR, Outcome.TIMEOUT, Outcome.CRASH, Outcome.REJECTED):
            assert not TacticResult(bad).ok

    def test_serialisable(self):
        r = TacticResult(Outcome.PROGRESS, state=ProofState(3, ("⊢ x",)), elapsed_s=1.23456)
        d = r.to_dict()
        assert d["outcome"] == "progress"
        assert d["state"] == {"pid": 3, "goals": ["⊢ x"]}
        assert d["elapsed_s"] == 1.2346
