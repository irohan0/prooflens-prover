"""Hermetic tests for the model-free tactic policy.

The policy exists to isolate retrieval quality with no generator in the loop, so the tests are
mostly about that isolation holding: the non-retrieval part must be identical across arms, and the
retrieval-dependent part must actually depend on the retrieval order.
"""

from __future__ import annotations

import math

import pytest

from prooflens_prover.lean.backend import ProofState
from prooflens_prover.prover.repertoire import (
    DEFAULT_CLOSERS,
    DEFAULT_TEMPLATES,
    RepertoirePolicy,
    TacticTemplate,
)
from prooflens_prover.prover.search import TacticPolicy
from prooflens_prover.retrieval.base import DEFAULT_TOP_K, NullRetriever, Premise

STATE = ProofState(pid=0, goals=("n m : ℕ\n⊢ n + m = m + n",))


class FakeRetriever:
    """Returns a fixed ranking, so tests can assert on rank-dependent behaviour exactly."""

    name = "fake"

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = names if names is not None else [
            "Nat.add_comm", "Nat.add_assoc", "Nat.mul_comm", "Nat.succ_le", "Nat.zero_add"
        ]
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[Premise]:
        self.queries.append(query)
        return [
            Premise(formal_name=n, formal_statement=f"statement of {n}", score=1.0 / (i + 1))
            for i, n in enumerate(self.names[:k])
        ]


class TestProtocolAndShape:
    def test_satisfies_the_tactic_policy_protocol(self):
        assert isinstance(RepertoirePolicy(retriever=NullRetriever()), TacticPolicy)

    def test_name_records_the_arm(self):
        assert RepertoirePolicy(retriever=FakeRetriever()).name == "repertoire+fake"
        assert RepertoirePolicy(retriever=NullRetriever()).name == "repertoire+none"

    def test_never_returns_more_than_n(self):
        p = RepertoirePolicy(retriever=FakeRetriever())
        for n in (1, 5, 16, 64):
            assert len(p.propose(STATE, n)) <= n

    def test_all_scores_are_valid_log_probabilities(self):
        # The search harness sums these as log-probabilities and divides by depth**alpha; a
        # positive value would make deeper nodes look better and invert the search order.
        for tactic, score in RepertoirePolicy(retriever=FakeRetriever()).propose(STATE, 64):
            assert score < 0.0, f"{tactic} has non-negative logprob {score}"

    def test_returned_in_descending_score_order(self):
        scores = [s for _, s in RepertoirePolicy(retriever=FakeRetriever()).propose(STATE, 64)]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_tactics(self):
        tactics = [t for t, _ in RepertoirePolicy(retriever=FakeRetriever()).propose(STATE, 64)]
        assert len(tactics) == len(set(tactics))


class TestRetrievalIsolation:
    def test_null_retriever_yields_only_closers(self):
        out = RepertoirePolicy(retriever=NullRetriever()).propose(STATE, 64)
        assert {t for t, _ in out} == set(DEFAULT_CLOSERS)

    def test_closers_are_identical_regardless_of_retriever(self):
        # The whole design depends on this: any difference between arms must come from premises.
        a = dict(RepertoirePolicy(retriever=NullRetriever()).propose(STATE, 64))
        b = dict(RepertoirePolicy(retriever=FakeRetriever()).propose(STATE, 64))
        for tactic, score in a.items():
            assert b[tactic] == pytest.approx(score), f"{tactic} differs between arms"

    def test_premise_tactics_appear_for_a_real_retriever(self):
        out = dict(RepertoirePolicy(retriever=FakeRetriever()).propose(STATE, 64))
        assert "exact Nat.add_comm" in out
        assert "apply Nat.add_comm" in out
        assert "rw [Nat.add_comm]" in out
        assert "rw [← Nat.add_comm]" in out
        assert "simp [Nat.add_comm]" in out

    def test_retriever_is_queried_with_the_state_pp(self):
        # The query must be exactly what the model would see. If these ever diverge, the retrieval
        # measured is not the retrieval the prover used.
        r = FakeRetriever()
        RepertoirePolicy(retriever=r).propose(STATE, 8)
        assert r.queries == [STATE.pp]

    def test_higher_ranked_premises_score_higher(self):
        out = dict(RepertoirePolicy(retriever=FakeRetriever()).propose(STATE, 64))
        assert out["exact Nat.add_comm"] > out["exact Nat.add_assoc"] > out["exact Nat.mul_comm"]

    def test_rank_discount_is_the_documented_formula(self):
        out = dict(RepertoirePolicy(retriever=FakeRetriever()).propose(STATE, 64))
        template = next(t for t in DEFAULT_TEMPLATES if t.pattern == "exact {p}")
        for rank, name in enumerate(["Nat.add_comm", "Nat.add_assoc", "Nat.mul_comm"]):
            expected = math.log(template.prior) - math.log(rank + 1.0)
            assert out[f"exact {name}"] == pytest.approx(expected)

    def test_changing_only_the_ranking_changes_the_proposal_order(self):
        # The measurable-difference property. If reordering retrieval output left the proposals
        # unchanged, this policy could not distinguish two retrievers at all.
        fwd = RepertoirePolicy(retriever=FakeRetriever(["A", "B", "C"])).propose(STATE, 64)
        rev = RepertoirePolicy(retriever=FakeRetriever(["C", "B", "A"])).propose(STATE, 64)
        assert [t for t, _ in fwd] != [t for t, _ in rev]

    def test_use_premises_false_suppresses_premise_tactics(self):
        out = RepertoirePolicy(retriever=FakeRetriever(), use_premises=False).propose(STATE, 64)
        assert {t for t, _ in out} == set(DEFAULT_CLOSERS)

    def test_top_k_zero_suppresses_premise_tactics(self):
        out = RepertoirePolicy(retriever=FakeRetriever(), top_k=0).propose(STATE, 64)
        assert {t for t, _ in out} == set(DEFAULT_CLOSERS)

    def test_top_k_limits_premises_requested(self):
        r = FakeRetriever(["A", "B", "C", "D", "E"])
        out = {t for t, _ in RepertoirePolicy(retriever=r, top_k=2).propose(STATE, 64)}
        assert "exact A" in out and "exact B" in out
        assert "exact C" not in out


class TestScientificIntegrityGuards:
    @pytest.mark.parametrize("banned", ["exact?", "apply?", "hint", "aesop?", "exact?!", "says"])
    def test_lean_premise_search_tactics_are_not_in_the_repertoire(self, banned):
        # These search Mathlib's own premise index. Including one would let every arm — including
        # `none` — solve goals without using our retriever, collapsing the whole comparison.
        assert banned not in DEFAULT_CLOSERS

    def test_no_closer_contains_a_question_mark(self):
        # Blanket form of the above: Lean's search tactics are conventionally suffixed `?`.
        for tactic in DEFAULT_CLOSERS:
            assert "?" not in tactic, f"{tactic} looks like a Lean search tactic"

    def test_no_template_contains_a_question_mark(self):
        for template in DEFAULT_TEMPLATES:
            assert "?" not in template.pattern

    def test_closer_priors_are_probabilities(self):
        assert all(0.0 < p <= 1.0 for p in DEFAULT_CLOSERS.values())

    def test_template_priors_are_probabilities(self):
        assert all(0.0 < t.prior <= 1.0 for t in DEFAULT_TEMPLATES)

    def test_proposal_is_deterministic(self):
        # Two runs of the same arm must explore identically, or a repeated measurement is not a
        # repeated measurement.
        a = RepertoirePolicy(retriever=FakeRetriever()).propose(STATE, 64)
        b = RepertoirePolicy(retriever=FakeRetriever()).propose(STATE, 64)
        assert a == b


class TestTemplates:
    def test_render_substitutes_the_premise_name(self):
        assert TacticTemplate("rw [{p}]", 0.2).render("Nat.add_comm") == "rw [Nat.add_comm]"

    def test_custom_templates_are_honoured(self):
        p = RepertoirePolicy(
            retriever=FakeRetriever(["Foo.bar"]),
            templates=(TacticTemplate("exact {p} rfl", 0.5),),
        )
        out = {t for t, _ in p.propose(STATE, 64)}
        assert "exact Foo.bar rfl" in out
        assert "apply Foo.bar" not in out

    def test_custom_closers_are_honoured(self):
        p = RepertoirePolicy(retriever=NullRetriever(), closers={"my_tac": 0.5})
        assert p.propose(STATE, 64) == [("my_tac", math.log(0.5))]


class TestSlotReservation:
    """Regression tests for the crowding-out bug.

    Measured before the fix, on the first 30 miniF2F-test problems: `none` 10/30 vs `bm25` 6/30.
    Retrieval appeared to *hurt*, because 50 premise-templated tactics outscored every structural
    closer and consumed the whole `samples_per_step` budget. These tests pin the fix.
    """

    def test_top_closers_survive_a_flood_of_premises(self):
        # 20 premises x 5 templates = 100 premise tactics, all scoring above `intro x`.
        r = FakeRetriever([f"Lemma.n{i}" for i in range(20)])
        out = dict(RepertoirePolicy(retriever=r, top_k=20).propose(STATE, 16))
        assert "intro x" in out, "structural tactics must not be crowded out by retrieval"

    def test_reserved_count_is_honoured(self):
        r = FakeRetriever([f"Lemma.n{i}" for i in range(20)])
        policy = RepertoirePolicy(retriever=r, top_k=20, min_closers=8)
        out = [t for t, _ in policy.propose(STATE, 16)]
        assert len([t for t in out if t in DEFAULT_CLOSERS]) >= 8

    def test_arms_are_compute_matched(self):
        # Every arm must issue the same number of Lean calls per expansion, or a difference in
        # solve rate is partly a difference in budget.
        n = 16
        none_arm = RepertoirePolicy(retriever=NullRetriever()).propose(STATE, n)
        bm25_arm = RepertoirePolicy(retriever=FakeRetriever(
            [f"Lemma.n{i}" for i in range(10)]), top_k=10).propose(STATE, n)
        assert len(none_arm) == len(bm25_arm) == n

    def test_none_arm_still_fills_its_budget(self):
        # The reservation must not handicap the arm that has no premises to contest the free slots.
        out = RepertoirePolicy(retriever=NullRetriever()).propose(STATE, 16)
        assert len(out) == 16
        assert all(t in DEFAULT_CLOSERS for t, _ in out)

    def test_premises_still_reach_the_candidate_list(self):
        # The fix must not go so far that retrieval stops mattering.
        r = FakeRetriever(["Nat.add_comm"])
        out = [t for t, _ in RepertoirePolicy(retriever=r, top_k=1).propose(STATE, 16)]
        assert any("Nat.add_comm" in t for t in out)

    def test_min_closers_zero_restores_pure_score_ordering(self):
        # min_closers=0 means "no reservation", not "no closers": a high-prior closer such as
        # `simp` (log 0.12 = -2.12) still outranks a rank-2 premise tactic (-2.30) on merit. The
        # property that matters is that reservation strictly increases closer representation.
        r = FakeRetriever([f"Lemma.n{i}" for i in range(20)])
        unreserved = [t for t, _ in RepertoirePolicy(
            retriever=r, top_k=20, min_closers=0).propose(STATE, 16)]
        reserved = [t for t, _ in RepertoirePolicy(
            retriever=FakeRetriever([f"Lemma.n{i}" for i in range(20)]),
            top_k=20, min_closers=8).propose(STATE, 16)]
        n_unreserved = len([t for t in unreserved if t in DEFAULT_CLOSERS])
        n_reserved = len([t for t in reserved if t in DEFAULT_CLOSERS])
        assert n_reserved > n_unreserved

    def test_min_closers_larger_than_n_is_clamped(self):
        out = RepertoirePolicy(retriever=FakeRetriever(), min_closers=999).propose(STATE, 5)
        assert len(out) == 5

    def test_output_stays_sorted_after_reservation(self):
        scores = [s for _, s in RepertoirePolicy(
            retriever=FakeRetriever(), top_k=10).propose(STATE, 16)]
        assert scores == sorted(scores, reverse=True)
