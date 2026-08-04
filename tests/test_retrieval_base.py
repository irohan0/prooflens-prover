"""Tests for the retriever interface and premise formatting.

The premise dict shape and the prompt rendering are dictated by REAL-Prover's source, not chosen
by us. If either drifts, the frozen-prover swap keeps running and silently feeds their model empty
or malformed premises — the arm would look like a weak retriever rather than a broken integration.
"""

from __future__ import annotations

from prooflens_prover.retrieval.base import (
    DEFAULT_TOP_K,
    PROMPT_PREMISE_LIMIT,
    NullRetriever,
    Premise,
    RetrievalStats,
    Retriever,
    format_premises,
)


def mk(n: int) -> list[Premise]:
    return [Premise(formal_name=f"Thm{i}", formal_statement=f"stmt{i}") for i in range(n)]


class TestPremiseShape:
    def test_realprover_dict_keys_are_exact(self):
        # Key names and capitalisation are REAL-Prover's. "Tidying" them to snake_case would make
        # build_theorems_str emit blank fields with no error anywhere.
        d = Premise("Nat.add_comm", "∀ (n m : ℕ), n + m = m + n").to_realprover_dict()
        assert set(d) == {"Formal name", "Informal name", "Formal statement"}
        assert d["Formal name"] == "Nat.add_comm"
        assert d["Formal statement"] == "∀ (n m : ℕ), n + m = m + n"

    def test_informal_name_defaults_to_empty_string_not_none(self):
        # Their formatter interpolates it directly; None would render the string "None" into the
        # prompt and quietly change what every arm's model sees.
        assert Premise("A", "B").to_realprover_dict()["Informal name"] == ""


class TestFormatPremises:
    def test_matches_real_prover_rendering(self):
        out = format_premises(mk(2))
        assert out == (
            "ID:0\nFormal name: Thm0\nInformal name: \nFormal statement: stmt0"
            "\n\n"
            "ID:1\nFormal name: Thm1\nInformal name: \nFormal statement: stmt1"
        )

    def test_id_counter_starts_at_zero(self):
        assert format_premises(mk(1)).startswith("ID:0")

    def test_truncates_to_six_by_default(self):
        # REAL-Prover's build_theorems_str slices [:6]; their headline +12pt FATE-M gain comes
        # from six premises, so 6 is the comparability point for any top-k ablation.
        assert PROMPT_PREMISE_LIMIT == 6
        out = format_premises(mk(20))
        assert out.count("Formal name:") == 6
        assert "Thm5" in out and "Thm6" not in out

    def test_limit_is_overridable_for_the_top_k_ablation(self):
        assert format_premises(mk(20), limit=2).count("Formal name:") == 2
        assert format_premises(mk(20), limit=0) == ""

    def test_empty_premises_render_as_empty_string(self):
        assert format_premises([]) == ""


class TestNullRetriever:
    def test_returns_nothing(self):
        assert NullRetriever().retrieve("⊢ True") == []

    def test_satisfies_the_protocol(self):
        assert isinstance(NullRetriever(), Retriever)

    def test_named_for_the_results_table(self):
        assert NullRetriever().name == "none"


class TestDefaults:
    def test_top_k_matches_real_prover_num_querys(self):
        assert DEFAULT_TOP_K == 10

    def test_requested_exceeds_used(self):
        # 10 requested, 6 used. Reporting the number offered rather than used would overstate
        # retrieval depth — the exact error the predecessor project caught in its own reporting.
        assert DEFAULT_TOP_K > PROMPT_PREMISE_LIMIT


class TestRetrievalStats:
    def test_accumulates_and_averages(self):
        s = RetrievalStats()
        s.record(0.010, 10)
        s.record(0.030, 6)
        d = s.to_dict()
        assert d["n_queries"] == 2
        assert d["mean_latency_ms"] == 20.0
        assert d["mean_premises_returned"] == 8.0

    def test_no_division_by_zero_when_unused(self):
        assert RetrievalStats().to_dict()["mean_latency_ms"] == 0.0
