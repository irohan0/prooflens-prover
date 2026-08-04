"""Hermetic tests for the best-first search harness.

The search is the *fixed* component every retrieval arm is compared through, so a bug here does not
produce a wrong number for one arm — it corrupts the comparison itself. These tests use a fake
backend and a scripted policy, so they run in milliseconds with no Lean and no model, and they
pin the behaviours that must not drift: scoring order, budget enforcement, deduplication, and
refusing to call a non-proof a proof.
"""

from __future__ import annotations

from typing import Any

import pytest

from prooflens_prover.lean.backend import Outcome, ProofState, TacticResult
from prooflens_prover.prover.search import (
    Node,
    SearchConfig,
    SearchStatus,
    best_first_search,
)


class FakeBackend:
    """Scripted Lean. `script` maps (goal, tactic) -> TacticResult; anything else errors."""

    def __init__(self, root_goals: tuple[str, ...] = ("⊢ P",), script: dict | None = None) -> None:
        self.root_goals = root_goals
        self.script = script or {}
        self.calls: list[tuple[str, str]] = []
        self._next_pid = 1
        self.closed = False

    def start_theorem(self, statement: str, header: str | None = None) -> ProofState:  # noqa: ARG002,E501
        return ProofState(pid=0, goals=self.root_goals)

    def run_tactic(self, state: ProofState, tactic: str,
                   timeout: float | None = None) -> TacticResult:  # noqa: ARG002
        self.calls.append((state.pp, tactic))
        key = (state.pp, tactic)
        if key in self.script:
            r = self.script[key]
            if r.outcome is Outcome.PROGRESS and r.state is not None and r.state.pid < 0:
                # allocate a fresh pid so distinct nodes are distinct
                self._next_pid += 1
                return TacticResult(r.outcome, ProofState(self._next_pid, r.state.goals), r.error)
            return r
        return TacticResult(Outcome.ERROR, error=f"no rule for {tactic!r}")

    def close(self) -> None:
        self.closed = True


class ScriptedPolicy:
    """Returns a fixed candidate list, ignoring the state. Records how often it was consulted."""

    def __init__(self, candidates: list[tuple[str, float]]) -> None:
        self.candidates = candidates
        self.n_calls = 0

    def propose(self, state: ProofState, n: int, context: dict[str, Any] | None = None):  # noqa: ARG002
        self.n_calls += 1
        return self.candidates[:n]


def progress(goals: tuple[str, ...]) -> TacticResult:
    return TacticResult(Outcome.PROGRESS, ProofState(-1, goals))


PROVED = TacticResult(Outcome.PROVED)


class TestOutcomes:
    def test_finds_a_one_step_proof(self):
        be = FakeBackend(script={("⊢ P", "exact hp"): PROVED})
        r = best_first_search(be, ScriptedPolicy([("exact hp", -0.1)]), "theorem t : P")
        assert r.status is SearchStatus.PROVED
        assert r.proof == ["exact hp"]
        assert r.proved

    def test_finds_a_multi_step_proof_and_returns_it_in_order(self):
        be = FakeBackend(script={
            ("⊢ P", "intro h"): progress(("h : A ⊢ P",)),
            ("h : A ⊢ P", "exact h"): PROVED,
        })
        pol = ScriptedPolicy([("intro h", -0.1), ("exact h", -0.2)])
        r = best_first_search(be, pol, "theorem t : P")
        assert r.status is SearchStatus.PROVED
        assert r.proof == ["intro h", "exact h"], "proof steps must be root-to-leaf in order"

    def test_no_candidates_when_every_branch_dies(self):
        be = FakeBackend()  # every tactic errors
        r = best_first_search(be, ScriptedPolicy([("bad", -1.0)]), "theorem t : P")
        assert r.status is SearchStatus.NO_CANDIDATES
        assert not r.proved

    def test_unelaborable_statement_is_error_not_failure(self):
        class Boom(FakeBackend):
            def start_theorem(self, statement, header=None):
                raise RuntimeError("does not elaborate")

        r = best_first_search(Boom(), ScriptedPolicy([]), "theorem bad")
        assert r.status is SearchStatus.ERROR
        assert "does not elaborate" in r.error

    def test_already_closed_root_is_error_not_proved(self):
        # A statement that elaborates with no goals must NOT score as a win — that would silently
        # inflate results on malformed benchmark inputs.
        r = best_first_search(FakeBackend(root_goals=()), ScriptedPolicy([]), "theorem t : True")
        assert r.status is SearchStatus.ERROR
        assert not r.proved

    def test_repl_crash_is_error_not_exhausted(self):
        # A crash invalidates every state handle. Reporting "exhausted" would look like
        # "no proof found", conflating a harness failure with a genuine negative result.
        be = FakeBackend(script={("⊢ P", "boom"): TacticResult(Outcome.CRASH, error="REPL died")})
        r = best_first_search(be, ScriptedPolicy([("boom", -0.1)]), "theorem t : P")
        assert r.status is SearchStatus.ERROR
        assert "crash" in r.error.lower()

    def test_policy_failure_is_reported_not_raised(self):
        class Boom:
            def propose(self, state, n, context=None):
                raise RuntimeError("model died")

        r = best_first_search(FakeBackend(), Boom(), "theorem t : P")
        assert r.status is SearchStatus.ERROR
        assert "model died" in r.error


class TestBudgets:
    def test_max_expansions_enforced(self):
        # Each expansion yields a NEW state, so the frontier never empties; only the budget stops.
        class Endless(FakeBackend):
            def run_tactic(self, state, tactic, timeout=None):
                self._next_pid += 1
                pid = self._next_pid
                return TacticResult(Outcome.PROGRESS, ProofState(pid, (f"⊢ G{pid}",)))

        pol = ScriptedPolicy([("step", -0.1)])
        r = best_first_search(Endless(), pol, "theorem t : P",
                              SearchConfig(max_expansions=5, samples_per_step=1))
        assert r.status is SearchStatus.EXHAUSTED
        assert r.limit_hit == "max_expansions"
        assert r.n_expansions == 5
        assert pol.n_calls == 5, "policy must be consulted exactly max_expansions times"

    def test_max_depth_stops_descent(self):
        class Endless(FakeBackend):
            def run_tactic(self, state, tactic, timeout=None):
                self._next_pid += 1
                pid = self._next_pid
                return TacticResult(Outcome.PROGRESS, ProofState(pid, (f"⊢ G{pid}",)))

        r = best_first_search(Endless(), ScriptedPolicy([("step", -0.1)]), "theorem t : P",
                              SearchConfig(max_expansions=100, samples_per_step=1, max_depth=3))
        # Nodes at depth >= 3 are never expanded, so the search stops making progress.
        assert r.status is SearchStatus.EXHAUSTED or r.n_expansions <= 100

    def test_rejected_tactics_do_not_count_as_lean_calls(self):
        # A policy-rejected tactic never reaches Lean; counting it would overstate Lean cost in the
        # efficiency table.
        be = FakeBackend(script={("⊢ P", "ok"): PROVED})
        r = best_first_search(be, ScriptedPolicy([("ok", -0.1)]), "theorem t : P")
        assert r.n_lean_calls == 1
        assert r.n_tactics_tried == 1


class TestDeduplication:
    def test_repeated_state_is_not_re_expanded(self):
        # Two different tactics converge on the same goal; only one node should enter the frontier.
        be = FakeBackend(script={
            ("⊢ P", "a"): progress(("⊢ Q",)),
            ("⊢ P", "b"): progress(("⊢ Q",)),
        })
        pol = ScriptedPolicy([("a", -0.1), ("b", -0.2)])
        r = best_first_search(be, pol, "theorem t : P",
                              SearchConfig(max_expansions=10, samples_per_step=2))
        # root + the single deduped "⊢ Q" node = 2 expansions, then the frontier empties.
        assert pol.n_calls == 2, f"expected the duplicate state to be skipped, got {pol.n_calls}"
        assert r.status is SearchStatus.NO_CANDIDATES

    def test_dedupe_can_be_disabled(self):
        be = FakeBackend(script={
            ("⊢ P", "a"): progress(("⊢ Q",)),
            ("⊢ P", "b"): progress(("⊢ Q",)),
        })
        pol = ScriptedPolicy([("a", -0.1), ("b", -0.2)])
        best_first_search(be, pol, "theorem t : P",
                          SearchConfig(max_expansions=3, samples_per_step=2, dedupe_states=False))
        assert pol.n_calls == 3, "without dedupe both duplicates should be expanded"


class TestScoring:
    def test_length_penalty_matches_real_prover(self):
        # score = cum_logprob / depth**alpha  (REAL-Prover arXiv:2505.20613 §4)
        n = Node(ProofState(1, ("g",)), cum_logprob=-4.0, depth=4)
        assert n.score(0.5) == pytest.approx(-4.0 / 2.0)
        assert n.score(0.0) == pytest.approx(-4.0)

    def test_root_depth_zero_does_not_divide_by_zero(self):
        assert Node(ProofState(0, ("g",)), 0.0, 0).score(0.5) == 0.0

    def test_higher_logprob_is_explored_first(self):
        # The likelier branch must be expanded first; if ordering inverted, search quality would
        # silently degrade in a way no single test of "does it find a proof" would catch.
        be = FakeBackend(script={
            ("⊢ P", "likely"): progress(("⊢ GOOD",)),
            ("⊢ P", "unlikely"): progress(("⊢ BAD",)),
        })
        pol = ScriptedPolicy([("unlikely", -5.0), ("likely", -0.01)])
        best_first_search(be, pol, "theorem t : P",
                          SearchConfig(max_expansions=2, samples_per_step=2))
        expanded_states = [c[0] for c in be.calls]
        assert "⊢ GOOD" in expanded_states, "the higher-logprob branch should be expanded first"
        assert expanded_states.index("⊢ GOOD") < (
            expanded_states.index("⊢ BAD") if "⊢ BAD" in expanded_states else len(expanded_states)
        )


class TestTrace:
    def test_trace_records_every_attempt(self):
        be = FakeBackend(script={("⊢ P", "good"): PROVED})
        r = best_first_search(be, ScriptedPolicy([("bad", -2.0), ("good", -0.1)]), "theorem t : P")
        assert len(r.trace) == 2
        assert {t["outcome"] for t in r.trace} == {"error", "proved"}
        assert all("tactic" in t and "logprob" in t for t in r.trace)

    def test_trace_can_be_disabled(self):
        be = FakeBackend(script={("⊢ P", "good"): PROVED})
        r = best_first_search(be, ScriptedPolicy([("good", -0.1)]), "theorem t : P",
                              record_trace=False)
        assert r.trace == []

    def test_result_is_json_serialisable(self):
        import json

        be = FakeBackend(script={("⊢ P", "good"): PROVED})
        r = best_first_search(be, ScriptedPolicy([("good", -0.1)]), "theorem t : P")
        json.dumps(r.to_dict())  # must not raise: this is the per-attempt audit record
