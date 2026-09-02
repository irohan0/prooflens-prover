"""Fusion retriever — combining single-vector and late interaction into one arm.

Tier 1 measured a tie at 74 problems each with a **union of 88**, so the pair reaches what neither
reaches alone. These tests pin the two fusion rules and, more importantly, pin the *difference*
between them: reciprocal rank fusion rewards consensus, interleaving guarantees coverage, and with
only six prompt slots that distinction decides which premises the model ever sees.

Hermetic: no index, no encoder, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.retrieval.base import Premise, RetrievalStats  # noqa: E402
from prooflens_prover.retrieval.fusion import (  # noqa: E402
    RRF_K,
    FusionRetriever,
    _interleave,
    _reciprocal_rank_fusion,
)


class FakeRetriever:
    """Returns a fixed ranking, and records the `k` it was asked for."""

    def __init__(self, name: str, names: list[str]):
        self.name = name
        self._names = names
        self.asked_for: list[int] = []
        self.stats = RetrievalStats()

    def retrieve(self, query: str, k: int = 10) -> list[Premise]:  # noqa: ARG002
        self.asked_for.append(k)
        out = [Premise(formal_name=n, formal_statement=f"statement of {n}", score=1.0 - i / 100)
               for i, n in enumerate(self._names[:k])]
        # Recorded, as every real retriever does — the in-process dense ones and the subprocess
        # client both count their own queries. A fake that skips it would let `component_stats()`
        # pass a test while reporting zeros against a live run.
        self.stats.record(0.001, len(out))
        return out


def ranking(*names: str) -> list[Premise]:
    return [Premise(formal_name=n, formal_statement=f"stmt {n}") for n in names]


# --- reciprocal rank fusion -------------------------------------------------------------------

def test_a_premise_both_retrievers_rank_first_wins():
    fused = _reciprocal_rank_fusion([ranking("a", "b"), ranking("a", "c")], RRF_K)
    assert fused[0].formal_name == "a"


def test_consensus_outranks_a_single_retrievers_top_pick():
    # THE property that motivates `interleave` existing at all, so it is asserted rather than left
    # implicit: with K=60, a premise at rank 5 from both retrievers (2/65 = 0.0308) beats one at
    # rank 1 from a single retriever (1/61 = 0.0164). That is the standard behaviour of RRF and it
    # is the wrong prior for the mechanism this arm exists to exploit -- late interaction earns its
    # keep on premises single-vector never offers.
    a = ranking("w", "x", "y", "z", "consensus")
    b = ranking("only_b", "x", "y", "z", "consensus")
    fused = _reciprocal_rank_fusion([a, b], RRF_K)
    names = [p.formal_name for p in fused]
    assert names.index("consensus") < names.index("only_b")


def test_rrf_scores_are_the_sum_of_reciprocal_ranks():
    fused = _reciprocal_rank_fusion([ranking("a"), ranking("a")], RRF_K)
    assert fused[0].score == pytest.approx(2.0 / (RRF_K + 1))


def test_a_premise_found_by_one_retriever_still_appears():
    fused = _reciprocal_rank_fusion([ranking("a"), ranking("b")], RRF_K)
    assert {p.formal_name for p in fused} == {"a", "b"}


def test_ties_break_on_the_name_not_on_construction_order():
    # Disjoint retrievers at the same rank score identically. Without a deterministic tiebreak the
    # arm's output would depend on which sub-retriever happened to be built first, and two runs of
    # the same configuration could retrieve different premises.
    forward = _reciprocal_rank_fusion([ranking("b"), ranking("a")], RRF_K)
    backward = _reciprocal_rank_fusion([ranking("a"), ranking("b")], RRF_K)
    assert [p.formal_name for p in forward] == [p.formal_name for p in backward] == ["a", "b"]


def test_the_fused_score_replaces_the_incomparable_sub_retriever_score():
    # An SV cosine and an LI MaxSim sum are on different scales; carrying either through would put
    # a meaningless number in the record.
    sub = [Premise(formal_name="a", formal_statement="s", score=0.987)]
    fused = _reciprocal_rank_fusion([sub], RRF_K)
    assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))
    assert fused[0].formal_statement == "s"


# --- interleaving -----------------------------------------------------------------------------

def test_interleaving_gives_each_retriever_half_the_prompt_slots():
    a = ranking("a1", "a2", "a3", "a4")
    b = ranking("b1", "b2", "b3", "b4")
    names = [p.formal_name for p in _interleave([a, b])][:6]
    assert names == ["a1", "b1", "a2", "b2", "a3", "b3"]


def test_interleaving_skips_a_premise_both_retrievers_returned():
    a = ranking("shared", "a2")
    b = ranking("shared", "b2")
    assert [p.formal_name for p in _interleave([a, b])] == ["shared", "a2", "b2"]


def test_interleaving_survives_rankings_of_different_lengths():
    assert [p.formal_name for p in _interleave([ranking("a1"), ranking("b1", "b2", "b3")])] == \
        ["a1", "b1", "b2", "b3"]


def test_interleaving_nothing_is_empty_not_an_error():
    assert _interleave([]) == []
    assert _interleave([[], []]) == []


# --- the retriever ----------------------------------------------------------------------------

def test_it_is_a_drop_in_retriever_named_fusion():
    f = FusionRetriever((FakeRetriever("sv", ["a"]), FakeRetriever("li", ["b"])))
    assert f.name == "fusion"
    assert hasattr(f, "retrieve")


def test_sub_retrievers_are_asked_for_more_than_k_so_fusion_has_ranks_to_work_with():
    # At fetch_k == k the rankings would be truncated before they could disagree and fusion would
    # degenerate to whichever retriever was listed first.
    sv, li = FakeRetriever("sv", [f"s{i}" for i in range(50)]), FakeRetriever("li", ["x"])
    FusionRetriever((sv, li), fetch_k=32).retrieve("goal", k=6)
    assert sv.asked_for == [32] and li.asked_for == [32]


def test_asking_for_more_than_fetch_k_widens_the_fetch_rather_than_truncating():
    sv, li = FakeRetriever("sv", ["a"]), FakeRetriever("li", ["b"])
    FusionRetriever((sv, li), fetch_k=8).retrieve("goal", k=20)
    assert sv.asked_for == [20]


def test_retrieve_returns_at_most_k():
    sv = FakeRetriever("sv", [f"s{i}" for i in range(20)])
    li = FakeRetriever("li", [f"l{i}" for i in range(20)])
    assert len(FusionRetriever((sv, li)).retrieve("goal", k=6)) == 6


def test_one_retrieve_call_counts_as_one_query_not_two():
    # `n_queries` drives the reported ms/query in the cost table. Summing the sub-retrievers'
    # counters would report the arm as twice as busy as it is and halve its apparent latency.
    f = FusionRetriever((FakeRetriever("sv", ["a"]), FakeRetriever("li", ["b"])))
    f.retrieve("goal", k=6)
    f.retrieve("goal", k=6)
    assert f.stats.n_queries == 2


def test_config_records_what_fusion_cannot_be_reproduced_without():
    f = FusionRetriever((FakeRetriever("sv", ["a"]), FakeRetriever("li", ["b"])), mode="interleave")
    cfg = f.config()
    assert cfg["mode"] == "interleave"
    assert cfg["components"] == ["sv", "li"]
    assert cfg["rrf_k"] is None          # not used in this mode, so not recorded as if it were


# --- guards -----------------------------------------------------------------------------------

def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown fusion mode"):
        FusionRetriever((FakeRetriever("sv", ["a"]), FakeRetriever("li", ["b"])), mode="magic")


def test_fusing_fewer_than_two_retrievers_is_refused():
    # One retriever fused with nothing is that retriever, and recording it as an arm called
    # 'fusion' would put a mislabelled row in the results table.
    with pytest.raises(ValueError, match="at least two"):
        FusionRetriever((FakeRetriever("sv", ["a"]),))


def test_fusing_two_retrievers_of_the_same_kind_is_refused():
    # Rank agreement between a retriever and itself is total, so fusion would be a no-op reported
    # as a result.
    with pytest.raises(ValueError, match="same kind"):
        FusionRetriever((FakeRetriever("sv", ["a"]), FakeRetriever("sv", ["b"])))


# --- per-component timing, which the fused total cannot provide ---------------------------------

class TestComponentStats:
    """The fusion pilot's whole purpose is to measure single-vector on the CPU.

    Late interaction is timed at 936-1,071 ms/query on a GPU across 32 sweep runs. Single-vector
    has only ever been timed on a GPU (36.8-38.8 ms), and this arm puts it on the CPU so the two
    indices need not share a card with a 7B model. The fused stat times the whole call and cannot
    separate them, and the two servers contend for the same eight cores, so subtracting one run's
    mean from another's is not a substitute for timing them together.
    """

    def test_each_half_is_reported_separately(self):
        sv = FakeRetriever("sv", ["a", "b", "c", "d"])
        li = FakeRetriever("li", ["b", "e", "f", "g"])
        f = FusionRetriever(retrievers=(sv, li))
        f.retrieve("q", k=3)
        per = f.component_stats()
        assert set(per) == {"sv", "li"}
        assert per["sv"]["n_queries"] == 1
        assert per["li"]["n_queries"] == 1

    def test_the_fused_total_still_counts_one_query_not_two(self):
        # Summing the halves into the arm's own stats would report it as twice as busy at half the
        # latency, which is what the class docstring says must not happen.
        f = FusionRetriever(retrievers=(FakeRetriever("sv", ["a", "b"]),
                                FakeRetriever("li", ["b", "c"])))
        f.retrieve("q", k=3)
        f.retrieve("q", k=3)
        assert f.stats.to_dict()["n_queries"] == 2

    def test_a_component_without_counters_is_skipped_rather_than_crashing(self):
        class Bare:
            name = "bare"
            stats = None

            def retrieve(self, query, k=10):  # noqa: ARG002
                return []

        f = FusionRetriever(retrievers=(Bare(), FakeRetriever("li", ["a", "b"])))
        f.retrieve("q", k=2)
        assert set(f.component_stats()) == {"li"}

    def test_the_run_records_it_so_the_pilot_needs_no_analysis(self):
        src = (Path(__file__).resolve().parent.parent / "scripts"
               / "prove_benchmark.py").read_text(encoding="utf-8")
        assert "retrieval_components=" in src
        assert "--fusion-sv-ms" in src, "the run must name the flag its measurement feeds"
