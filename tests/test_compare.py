"""Tests for the paired arm comparison.

The FATE-M fixture below is the **real** outcome of runs
`fate_m_none_repertoire_20260803T080314696050_aa5622e` and
`fate_m_li_repertoire_20260803T004038235074_a59423a`, transcribed from their proof lists. It is
here because the first analysis of those runs was done by eye and got the headline attribution
wrong — reporting 73% premise utilisation as evidence of retrieval's contribution when the causal
figure was different. A test that reproduces the correct answer from the raw outcomes is the thing
that would have caught it.

Hermetic: no Lean, no cluster, no network.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from prooflens_prover.eval.compare import (
    Arm,
    bootstrap_ci,
    compare,
    compare_pooled,
    format_pooled_report,
    format_report,
    is_premise_tactic,
    mcnemar_exact_p,
    oracle_union,
    permutation_p,
)

# Real FATE-M outcomes: problem id -> proof steps (empty list = not proved).
NONE_PROOFS = {
    "1962": ["tauto"], "384": ["simp"], "253": ["tauto"], "273": ["aesop"],
    "1457": ["simp"], "1569": ["simp"], "252": ["aesop"], "40": ["simp"],
    "544": ["simp"], "968": ["simp"], "2439": ["aesop"], "959": ["aesop"],
}
LI_PROOFS = {
    "1962": ["tauto"], "384": ["simp [Equiv.Perm.cycleOf]"], "253": ["tauto"], "273": ["aesop"],
    "1457": ["simp [CommGroup.center_eq_top]"], "1569": ["simp"], "252": ["aesop"],
    "40": ["apply Fintype.card_pi_const"], "544": ["exact conj_pow"], "968": ["rw [orderOf_inv]"],
    "2439": ["apply Subgroup.closure_eq"], "959": ["aesop"],
    # LI-only, all naming a retrieved premise:
    "193": ["simp [commute_iff_eq]", "rw [← inv_eq_iff_mul_eq_one]", "rw [mul_eq_one_iff_inv_eq]",
            "rw [inv_eq_iff_mul_eq_one]", "exact mul_eq_one_iff_inv_eq'"],
    "336": ["apply Ideal.span_singleton_prime", "aesop"],
    "971": ["apply Subgroup.isCyclic"],
    "380": ["aesop", "simp [Subgroup.centralizer]"],
    "4343": ["constructor", "apply Finite.equivFin"],
    "1087": ["simp [Equiv.Perm.inv_eq_iff_eq]", "tauto"],
    "826": ["simp [mul_eq_one_iff_inv_eq]", "field_simp", "rw [div_eq_iff_eq_mul]",
            "rw [div_eq_iff_eq_mul]", "aesop"],
    "241": ["rw [mul_sub]", "simp [sub_mul]"],
    "7533": ["simp [Int.mem_zmultiples_iff]"],
    "2880": ["exact Subgroup.center_le_normalizer"],
}
N_FATE_M = 141


#: Sentinel distinguishing "caller said nothing" from "caller said the field is absent". Runs made
#: before `policy_kind` existed have no such key, and they were all model-free.
_UNSET = object()


def make_run(tmp_path, name, arm, proofs, n_total=N_FATE_M, benchmark="fate_m", errors=(),
             ids=None, n_candidates=None, policy_kind=_UNSET):
    """Write a minimal run directory: manifest + attempts.jsonl.

    `ids` must be the SAME for both arms of a comparison — two runs of one benchmark attempt the
    same problems, and `compare` intersects on problem id. An earlier version of this fixture
    derived the id list from each arm's own proofs, so the problems only LI solved were simply
    absent from the `none` run and the intersection silently dropped every marginal problem.
    """
    d = tmp_path / name
    d.mkdir()
    cfg = {"benchmark": benchmark, "arm": arm}
    if n_candidates is not None:
        cfg["n_candidates"] = n_candidates
    if policy_kind is not _UNSET:
        cfg["policy_kind"] = policy_kind
    (d / "manifest.json").write_text(json.dumps({
        "run_id": name, "name": name, "seed": 0, "git_commit": "test",
        "started_utc": "2026-08-03T00:00:00+00:00",
        "config": cfg, "outcome": None,
    }))
    rows = []
    if ids is None:
        ids = list(proofs) + [f"filler{i}" for i in range(n_total - len(proofs))]
    for pid in ids:
        steps = proofs.get(pid, [])
        if pid in errors:
            status = "error"
        elif steps:
            status = "proved"
        else:
            status = "exhausted"
        rows.append({"problem_id": pid, "status": status, "proved": bool(steps),
                     "proof": steps or None})
    (d / "attempts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return d


#: The 141 FATE-M problem ids, as both runs saw them: every id either arm proved, padded out with
#: unproved fillers. Shared between the arms because a benchmark run attempts the same problems.
ALL_IDS = sorted(set(NONE_PROOFS) | set(LI_PROOFS))
ALL_IDS += [f"filler{i}" for i in range(N_FATE_M - len(ALL_IDS))]


@pytest.fixture
def fate_m(tmp_path):
    b = make_run(tmp_path, "fate_m_none", "none", NONE_PROOFS, ids=ALL_IDS)
    t = make_run(tmp_path, "fate_m_li", "li", LI_PROOFS, ids=ALL_IDS)
    return Arm.load(b), Arm.load(t)


class TestPremiseClassification:
    @pytest.mark.parametrize("t", [
        "exact conj_pow", "apply Subgroup.isCyclic", "rw [orderOf_inv]",
        "simp [Equiv.Perm.cycleOf]", "rw [← inv_eq_iff_mul_eq_one]",
    ])
    def test_template_tactics_are_premise_tactics(self, t):
        assert is_premise_tactic(t)

    @pytest.mark.parametrize("t", [
        "simp", "aesop", "tauto", "linarith", "nlinarith", "omega", "norm_num", "ring",
        "field_simp", "constructor", "intro x", "rfl", "decide", "assumption",
    ])
    def test_bare_closers_are_not(self, t):
        assert not is_premise_tactic(t)

    def test_whitespace_does_not_fool_it(self):
        assert not is_premise_tactic("  simp  ")

    def test_simp_with_a_lemma_differs_from_bare_simp(self):
        # The distinction the whole attribution rests on.
        assert not is_premise_tactic("simp")
        assert is_premise_tactic("simp [Nat.two_mul]")


class TestMcNemar:
    def test_ten_to_zero_matches_the_hand_computation(self):
        # 2 * 0.5^10 = 0.001953125 — the FATE-M result.
        assert mcnemar_exact_p(10, 0) == pytest.approx(0.001953125)

    def test_symmetric_in_its_arguments(self):
        assert mcnemar_exact_p(7, 2) == mcnemar_exact_p(2, 7)

    def test_no_discordant_pairs_is_p_one(self):
        assert mcnemar_exact_p(0, 0) == 1.0

    def test_balanced_discordance_is_not_significant(self):
        assert mcnemar_exact_p(5, 5) == 1.0

    def test_never_exceeds_one(self):
        for b in range(6):
            for c in range(6):
                assert 0.0 <= mcnemar_exact_p(b, c) <= 1.0


class TestResampling:
    def test_bootstrap_ci_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        d = np.concatenate([np.ones(10), np.zeros(131)])
        lo, hi = bootstrap_ci(d, 2000, rng)
        assert lo < d.mean() < hi

    def test_all_zero_differences_give_p_one(self):
        assert permutation_p(np.zeros(50), 1000, np.random.default_rng(0)) == 1.0

    def test_permutation_p_is_never_zero(self):
        # A resampling test cannot license p = 0; the +1/(n+1) correction enforces that.
        d = np.ones(200)
        assert permutation_p(d, 1000, np.random.default_rng(0)) > 0.0

    def test_seed_makes_it_reproducible(self):
        d = np.concatenate([np.ones(10), np.zeros(131)])
        a = permutation_p(d, 2000, np.random.default_rng(3))
        b = permutation_p(d, 2000, np.random.default_rng(3))
        assert a == b


class TestFateMResult:
    """Reproduce the real FATE-M comparison from the raw per-problem outcomes."""

    def test_counts(self, fate_m):
        b, t = fate_m
        assert b.n_proved == 12
        assert t.n_proved == 22

    def test_ten_marginal_problems(self, fate_m):
        r = compare(*fate_m, n_boot=1000, n_perm=1000)
        assert len(r["only_treatment"]) == 10
        assert r["delta_problems"] == 10

    def test_no_displacement(self, fate_m):
        r = compare(*fate_m, n_boot=1000, n_perm=1000)
        assert r["only_baseline"] == []
        assert r["baseline_is_subset"] is True

    def test_every_marginal_proof_names_a_premise(self, fate_m):
        r = compare(*fate_m, n_boot=1000, n_perm=1000)
        assert r["premise_needed_rate"] == 1.0

    def test_used_rate_overstates_needed_rate(self, fate_m):
        # The error the control caught: 16/22 proofs *use* a premise, but 6 of those problems the
        # baseline also solved with a bare closer, so using != needing.
        r = compare(*fate_m, n_boot=1000, n_perm=1000)
        assert r["premise_used_rate"] == pytest.approx(16 / 22)
        assert r["premise_needed_rate"] == 1.0
        assert len([p for p in r["only_treatment"]]) == 10

    def test_significant(self, fate_m):
        r = compare(*fate_m, n_boot=10_000, n_perm=10_000)
        assert r["p_mcnemar_exact"] == pytest.approx(0.001953125)
        assert r["ci95"][0] > 0
        assert r["significant"] is True
        assert r["borderline"] is False

    def test_report_renders(self, fate_m):
        text = format_report(compare(*fate_m, n_boot=500, n_perm=500))
        assert "no displacement: True" in text
        assert "premise NEEDED: 100%" in text


class TestGuards:
    def test_refuses_different_benchmarks(self, tmp_path):
        b = Arm.load(make_run(tmp_path, "a", "none", {}, benchmark="fate_m"))
        t = Arm.load(make_run(tmp_path, "b", "li", {}, benchmark="minif2f_test"))
        with pytest.raises(ValueError, match="different benchmarks"):
            compare(b, t)

    def test_refuses_identical_arms_at_the_same_budget(self, tmp_path):
        b = Arm.load(make_run(tmp_path, "a", "li", {}))
        t = Arm.load(make_run(tmp_path, "b", "li", {}))
        with pytest.raises(ValueError, match="nothing to compare"):
            compare(b, t)

    def test_refuses_a_run_against_itself(self, tmp_path):
        d = make_run(tmp_path, "a", "li", {}, n_candidates=1000)
        # Distinct Arm objects, same run: the budget check alone would not catch this.
        with pytest.raises(ValueError, match="same run"):
            compare(Arm.load(d), Arm.load(d))

    def test_allows_the_same_arm_at_different_budgets(self, tmp_path):
        """The H1 experiment. `li@1k` vs `li@50k` holds the encoder, index, policy, search budget
        and seed fixed and varies only candidate generation — measured 22/141 against 31/141 on
        FATE-M. An earlier guard rejected it purely because both sides were named `li`, which would
        have made the project's decisive comparison unrunnable."""
        ids = ["x", "y", "z"] + [f"f{i}" for i in range(7)]
        b = Arm.load(make_run(tmp_path, "narrow", "li", {"x": ["exact foo"]},
                              ids=ids, n_candidates=1_000))
        t = Arm.load(make_run(tmp_path, "wide", "li", {"x": ["exact foo"], "y": ["exact bar"]},
                              ids=ids, n_candidates=50_000))
        r = compare(b, t, n_boot=200, n_perm=200)
        assert r["delta_problems"] == 1
        assert r["only_treatment"] == ["y"]
        # Both sides are arm `li`, so the raw arm name cannot identify either one in a report.
        assert r["baseline"]["label"] == "li@1k"
        assert r["treatment"]["label"] == "li@50k"
        assert "li@50k vs li@1k" in format_report(r)

    def test_label_falls_back_to_the_arm_when_no_budget_was_recorded(self, tmp_path):
        b = Arm.load(make_run(tmp_path, "a", "none", {}))
        assert b.label == "none"

    def test_harness_errors_can_be_excluded(self, tmp_path):
        b = Arm.load(make_run(tmp_path, "a", "none", NONE_PROOFS, errors={"filler0"}, ids=ALL_IDS))
        t = Arm.load(make_run(tmp_path, "b", "li", LI_PROOFS, ids=ALL_IDS))
        keep = compare(b, t, n_boot=200, n_perm=200)
        drop = compare(b, t, n_boot=200, n_perm=200, exclude_harness_errors=True)
        assert keep["n_problems"] == N_FATE_M
        assert drop["n_problems"] == N_FATE_M - 1

    def test_a_baseline_only_win_is_flagged(self, tmp_path):
        # Displacement must be visible, not averaged away: this is the confound that made bm25
        # look worse than `none` on an early miniF2F run.
        ids = ["x", "y"] + [f"f{i}" for i in range(8)]
        b = Arm.load(make_run(tmp_path, "a", "none", {"x": ["simp"], "y": ["aesop"]}, ids=ids))
        t = Arm.load(make_run(tmp_path, "b", "li", {"x": ["simp"]}, ids=ids))
        r = compare(b, t, n_boot=200, n_perm=200)
        assert r["only_baseline"] == ["y"]
        assert r["baseline_is_subset"] is False
        assert "investigate" in format_report(r)


class TestOracleUnion:
    """Two equal counts can hide complete disagreement about *which* problems are solved.

    Measured: on ProofNet SV and LI each solve 20/186, and four differ in each direction — so the
    union is 24, above either arm. On FATE-M, SV 35 and LI 31 with 3 LI-only gives 38. That gap is
    the only reason a fusion arm would be worth running, and the raw counts do not show it.
    """

    def test_union_exceeds_both_arms_when_they_disagree(self, tmp_path):
        ids = ["a", "b", "c", "d"] + [f"f{i}" for i in range(6)]
        sv = Arm.load(make_run(tmp_path, "sv", "sv", {"a": ["simp"], "b": ["simp"]}, ids=ids))
        li = Arm.load(make_run(tmp_path, "li", "li", {"c": ["exact f"], "d": ["exact g"]}, ids=ids))
        u = oracle_union(sv, li)
        assert (u["n_union"], u["n_best_single"], u["gain_over_best"]) == (4, 2, 2)
        assert u["union_rate"] == 0.4

    def test_union_equals_the_better_arm_when_one_set_contains_the_other(self, tmp_path):
        # miniF2F: LI 79, SV 77, zero SV-only. The union adds nothing and must not pretend to.
        ids = ["a", "b"] + [f"f{i}" for i in range(8)]
        sv = Arm.load(make_run(tmp_path, "sv", "sv", {"a": ["simp"]}, ids=ids))
        li = Arm.load(make_run(tmp_path, "li", "li", {"a": ["simp"], "b": ["exact f"]}, ids=ids))
        u = oracle_union(sv, li)
        assert (u["n_union"], u["n_best_single"], u["gain_over_best"]) == (2, 2, 0)

    def test_reconstructs_the_measured_fate_m_union(self, tmp_path):
        """Against the real FATE-M sets: SV 35, LI 31, 3 LI-only -> union 38, +3 over SV."""
        li_only = ["2024", "241", "826"]
        sv_only = ["197", "2079", "245", "334", "823", "917", "947"]
        shared = [f"s{i}" for i in range(28)]                      # 31 - 3 = 28 both arms solved
        ids = li_only + sv_only + shared + [f"f{i}" for i in range(N_FATE_M - 38)]
        sv = Arm.load(make_run(
            tmp_path, "sv", "sv", {p: ["simp"] for p in sv_only + shared}, ids=ids))
        li = Arm.load(make_run(
            tmp_path, "li", "li", {p: ["exact f"] for p in li_only + shared}, ids=ids))
        assert (sv.n_proved, li.n_proved) == (35, 31)
        u = oracle_union(sv, li)
        assert u["n_problems"] == N_FATE_M
        assert (u["n_union"], u["n_best_single"], u["gain_over_best"]) == (38, 35, 3)

    def test_is_symmetric(self, tmp_path):
        ids = ["a", "b"] + [f"f{i}" for i in range(8)]
        sv = Arm.load(make_run(tmp_path, "sv", "sv", {"a": ["simp"]}, ids=ids))
        li = Arm.load(make_run(tmp_path, "li", "li", {"b": ["exact f"]}, ids=ids))
        assert oracle_union(sv, li) == oracle_union(li, sv)

class TestPremiseAttributionRefusesWhenItCannotMeasure:
    """`is_premise_tactic` is exact for the repertoire and vacuous for a language model.

    `RepertoirePolicy` emits either a `DEFAULT_CLOSERS` key verbatim or a premise template, so "not
    a closer" means "names a premise". An LLM writes `intro x`, `ring_nf`, `linarith` — none of them
    closers either — so the same test marks every tactic as premise-naming and the metric reads 100%
    regardless of what retrieval did. It was reported that way once, as causal evidence, from this
    file, whose own docstring exists because of the same mistake made by eye.
    """

    IDS = ["a", "b", "c"]

    def llm_pair(self, tmp_path):
        base = make_run(tmp_path, "none_vllm", "none", {"c": ["simp"]}, n_total=3,
                        ids=self.IDS, policy_kind="vllm")
        treat = make_run(tmp_path, "li_vllm", "li",
                         {"a": ["intro x", "ring_nf"], "b": ["linarith"], "c": ["simp"]},
                         n_total=3, ids=self.IDS, policy_kind="vllm")
        return Arm.load(base), Arm.load(treat)

    def test_an_llm_run_reports_no_attribution_rather_than_100_percent(self, tmp_path):
        r = compare(*self.llm_pair(tmp_path), n_boot=200, n_perm=200)
        assert r["premise_attribution_available"] is False
        assert r["premise_used_rate"] is None
        assert r["premise_needed_rate"] is None
        assert r["marginal_with_premise"] is None
        assert r["policy_kind"] == "vllm"

    def test_the_report_says_why_instead_of_printing_a_number(self, tmp_path):
        text = format_report(compare(*self.llm_pair(tmp_path), n_boot=200, n_perm=200))
        assert "NOT MEASURABLE" in text
        # The vacuous figure must not appear anywhere, in any form.
        assert "premise USED" not in text
        assert "100%" not in text

    def test_a_repertoire_run_still_reports_the_rates(self, tmp_path):
        base = make_run(tmp_path, "none_r", "none", {"c": ["simp"]}, n_total=3, ids=self.IDS)
        treat = make_run(tmp_path, "li_r", "li",
                         {"a": ["simp [Nat.two_mul]"], "c": ["simp"]}, n_total=3, ids=self.IDS)
        r = compare(Arm.load(base), Arm.load(treat), n_boot=200, n_perm=200)
        assert r["premise_attribution_available"] is True
        assert r["premise_used_rate"] is not None
        assert "premise USED" in format_report(r)

    def test_runs_predating_the_field_count_as_model_free(self, tmp_path):
        # Every run made before `policy_kind` existed used the repertoire, so its attribution stays
        # valid and must not be silently withheld — that would blank the published Track A' numbers.
        base = make_run(tmp_path, "none_old", "none", {"c": ["simp"]}, n_total=3, ids=self.IDS)
        treat = make_run(tmp_path, "li_old", "li", {"a": ["simp [Nat.two_mul]"]},
                         n_total=3, ids=self.IDS)
        assert Arm.load(treat).policy_kind == "repertoire"
        r = compare(Arm.load(base), Arm.load(treat), n_boot=200, n_perm=200)
        assert r["premise_used_rate"] is not None

    def test_premise_using_returns_none_for_an_unmeasurable_policy(self, tmp_path):
        _, treat = self.llm_pair(tmp_path)
        assert treat.premise_using() is None


class TestPooledComparison:
    """Pooling FATE-M with a second benchmark, which is how a 19-discordant-pair null gets tested.

    Exact McNemar on 19 discordant pairs cannot reach p < 0.05 for any split closer than 15-4, so
    FATE-M alone cannot answer the question however the problems fall. Problems are independent and
    the contrast is identical on each benchmark, so pooling is legitimate — provided the guards
    below hold and the per-benchmark rows travel with the total.
    """

    def pair(self, tmp_path, benchmark, none_proofs, li_proofs, ids, n_candidates=50000):
        b = make_run(tmp_path, f"{benchmark}_none", "none", none_proofs, n_total=len(ids),
                     benchmark=benchmark, ids=ids, policy_kind="vllm")
        t = make_run(tmp_path, f"{benchmark}_li", "li", li_proofs, n_total=len(ids),
                     benchmark=benchmark, ids=ids, policy_kind="vllm",
                     n_candidates=n_candidates)
        return Arm.load(b), Arm.load(t)

    def two(self, tmp_path):
        ids_a = [f"a{i}" for i in range(20)]
        ids_b = [f"b{i}" for i in range(20)]
        # FATE-M-like: li gains 4, loses 1.
        fa = self.pair(
            tmp_path, "fate_m",
            {"a0": ["simp"], "a1": ["simp"]},
            {"a0": ["simp"], "a2": ["exact foo"], "a3": ["exact bar"], "a4": ["exact baz"],
             "a5": ["exact qux"]},
            ids_a,
        )
        # ProofNet-like: li gains 3, loses 1.
        pn = self.pair(
            tmp_path, "proofnet",
            {"b0": ["simp"], "b1": ["simp"]},
            {"b0": ["simp"], "b2": ["exact foo"], "b3": ["exact bar"], "b4": ["exact baz"]},
            ids_b,
        )
        return [fa, pn]

    def test_it_sums_the_problems_and_the_deltas(self, tmp_path):
        r = compare_pooled(self.two(tmp_path), n_boot=500, n_perm=500)
        assert r["n_problems"] == 40
        assert r["delta_problems"] == 3 + 2       # (4-1) + (3-1)
        assert r["benchmarks"] == ["fate_m", "proofnet"]

    def test_problem_ids_are_namespaced_by_benchmark(self, tmp_path):
        """FATE-M ids are bare integers; an un-prefixed union would merge unrelated problems."""
        r = compare_pooled(self.two(tmp_path), n_boot=500, n_perm=500)
        assert all(":" in p for p in r["only_treatment"] + r["only_baseline"])
        assert any(p.startswith("fate_m:") for p in r["only_treatment"])
        assert any(p.startswith("proofnet:") for p in r["only_treatment"])

    def test_a_collision_across_benchmarks_is_not_merged(self, tmp_path):
        ids = ["326", "1597"]
        a = self.pair(tmp_path, "fate_m", {}, {"326": ["exact foo"]}, ids)
        b = self.pair(tmp_path, "proofnet", {}, {"326": ["exact bar"]}, ids)
        r = compare_pooled([a, b], n_boot=500, n_perm=500)
        # Both `326`s must survive as distinct units, or the pooled n and delta are both wrong.
        assert r["n_problems"] == 4
        assert sorted(r["only_treatment"]) == ["fate_m:326", "proofnet:326"]

    def test_the_per_benchmark_rows_always_travel_with_the_total(self, tmp_path):
        r = compare_pooled(self.two(tmp_path), n_boot=500, n_perm=500)
        assert [row["benchmark"] for row in r["per_benchmark"]] == ["fate_m", "proofnet"]
        assert [row["delta_problems"] for row in r["per_benchmark"]] == [3, 2]
        assert "fate_m" in format_pooled_report(r) and "proofnet" in format_pooled_report(r)

    def test_pooling_increases_power_over_either_benchmark_alone(self, tmp_path):
        """The whole point: more discordant pairs, a smaller p for the same effect direction."""
        pairs = self.two(tmp_path)
        pooled = compare_pooled(pairs, n_boot=2000, n_perm=2000)
        singles = [compare(b, t, 2000, 2000)["p_mcnemar_exact"] for b, t in pairs]
        assert pooled["p_mcnemar_exact"] < min(singles)

    def test_opposite_signs_are_flagged_as_heterogeneous(self, tmp_path):
        """A pooled average over effects pointing opposite ways describes neither benchmark."""
        ids_a = [f"a{i}" for i in range(20)]
        ids_b = [f"b{i}" for i in range(20)]
        gains = self.pair(tmp_path, "fate_m", {},
                          {"a1": ["exact foo"], "a2": ["exact bar"], "a3": ["exact baz"]}, ids_a)
        loses = self.pair(tmp_path, "proofnet",
                          {"b1": ["simp"], "b2": ["simp"], "b3": ["simp"]}, {}, ids_b)
        r = compare_pooled([gains, loses], n_boot=500, n_perm=500)
        assert r["heterogeneous"] is True
        assert "HETEROGENEOUS" in format_pooled_report(r)

    def test_a_consistent_direction_is_not_flagged(self, tmp_path):
        r = compare_pooled(self.two(tmp_path), n_boot=500, n_perm=500)
        assert r["heterogeneous"] is False
        assert "HETEROGENEOUS" not in format_pooled_report(r)

    def test_one_pair_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="at least two benchmarks"):
            compare_pooled(self.two(tmp_path)[:1], n_boot=100, n_perm=100)

    def test_the_same_benchmark_twice_is_refused(self, tmp_path):
        # Counting one benchmark's problems twice halves the p-value for free.
        pairs = self.two(tmp_path)
        with pytest.raises(ValueError, match="more than once"):
            compare_pooled([pairs[0], pairs[0]], n_boot=100, n_perm=100)

    def test_mismatched_contrasts_are_refused(self, tmp_path):
        """li-vs-none pooled with sv-vs-none answers no question."""
        ids_a = [f"a{i}" for i in range(20)]
        ids_b = [f"b{i}" for i in range(20)]
        li = self.pair(tmp_path, "fate_m", {}, {"a1": ["exact foo"]}, ids_a)
        sv_b = make_run(tmp_path, "proofnet_none2", "none", {}, n_total=20,
                        benchmark="proofnet", ids=ids_b, policy_kind="vllm")
        sv_t = make_run(tmp_path, "proofnet_sv", "sv", {"b1": ["exact foo"]}, n_total=20,
                        benchmark="proofnet", ids=ids_b, policy_kind="vllm")
        with pytest.raises(ValueError, match="not the same contrast"):
            compare_pooled([li, (Arm.load(sv_b), Arm.load(sv_t))], n_boot=100, n_perm=100)

    def test_the_same_arm_at_a_different_budget_is_a_different_contrast(self, tmp_path):
        ids_a = [f"a{i}" for i in range(20)]
        ids_b = [f"b{i}" for i in range(20)]
        at_50k = self.pair(tmp_path, "fate_m", {}, {"a1": ["exact foo"]}, ids_a, n_candidates=50000)
        at_1k = self.pair(tmp_path, "proofnet", {}, {"b1": ["exact foo"]}, ids_b, n_candidates=1000)
        with pytest.raises(ValueError, match="not the same contrast"):
            compare_pooled([at_50k, at_1k], n_boot=100, n_perm=100)

    def test_a_pair_spanning_two_benchmarks_is_refused(self, tmp_path):
        ids = [f"a{i}" for i in range(20)]
        b = make_run(tmp_path, "x_none", "none", {}, n_total=20, benchmark="fate_m", ids=ids)
        t = make_run(tmp_path, "x_li", "li", {"a1": ["exact foo"]}, n_total=20,
                     benchmark="proofnet", ids=ids)
        other = self.pair(tmp_path, "minif2f", {}, {"a2": ["exact bar"]}, ids)
        with pytest.raises(ValueError, match="spans two benchmarks"):
            compare_pooled([(Arm.load(b), Arm.load(t)), other], n_boot=100, n_perm=100)

    def test_the_pooled_test_uses_the_same_problems_as_its_rows(self, tmp_path):
        """`_paired_pids` is shared so a pooled figure cannot be computed over a different set."""
        pairs = self.two(tmp_path)
        r = compare_pooled(pairs, n_boot=500, n_perm=500)
        assert r["n_problems"] == sum(row["n_problems"] for row in r["per_benchmark"])


class TestUnionIsReportedBecauseTheDeltaCanHideEverything:
    """On FATE-M under the LLM, li and sv both prove 46 with a delta of exactly 0 — and disagree
    about 22 problems, 11 each way, so their union is 57. A report showing only the delta describes
    the most interesting property of that pair as "no difference"."""

    def equal_but_different(self, tmp_path):
        ids = [f"p{i}" for i in range(20)]
        shared = {f"p{i}": ["simp"] for i in range(4)}
        sv = dict(shared, **{"p10": ["exact a"], "p11": ["exact b"]})
        li = dict(shared, **{"p12": ["exact c"], "p13": ["exact d"]})
        b = make_run(tmp_path, "sv_r", "sv", sv, n_total=20, ids=ids, policy_kind="vllm")
        t = make_run(tmp_path, "li_r", "li", li, n_total=20, ids=ids, policy_kind="vllm",
                     n_candidates=50000)
        return Arm.load(b), Arm.load(t)

    def test_a_zero_delta_still_reports_a_union_above_both_arms(self, tmp_path):
        r = compare(*self.equal_but_different(tmp_path), n_boot=300, n_perm=300)
        assert r["delta_problems"] == 0
        assert r["baseline"]["proved"] == r["treatment"]["proved"] == 6
        assert r["n_both"] == 4
        assert r["n_union"] == 8
        assert r["union_gain_over_best"] == 2

    def test_the_union_appears_in_the_report(self, tmp_path):
        text = format_report(compare(*self.equal_but_different(tmp_path), n_boot=300, n_perm=300))
        assert "union" in text
        assert "fusion" in text

    def test_two_retrievers_disagreeing_is_not_called_a_cost(self, tmp_path):
        """"retrieval COST proofs" is right against the none control and wrong between retrievers.

        An even 11-vs-11 split is symmetric disagreement; framing it as a regression would misread
        the result in the direction of a conclusion.
        """
        text = format_report(compare(*self.equal_but_different(tmp_path), n_boot=300, n_perm=300))
        assert "COST" not in text
        assert "neither arm" in text

    def test_against_the_none_control_it_is_still_called_a_cost(self, tmp_path):
        ids = [f"p{i}" for i in range(20)]
        b = make_run(tmp_path, "none_c", "none", {"p0": ["simp"], "p1": ["simp"]},
                     n_total=20, ids=ids, policy_kind="vllm")
        t = make_run(tmp_path, "li_c", "li", {"p0": ["simp"], "p2": ["exact a"]},
                     n_total=20, ids=ids, policy_kind="vllm", n_candidates=50000)
        text = format_report(compare(Arm.load(b), Arm.load(t), n_boot=300, n_perm=300))
        assert "COST" in text

    def test_no_displacement_still_reports_true(self, tmp_path):
        ids = [f"p{i}" for i in range(20)]
        b = make_run(tmp_path, "none_s", "none", {"p0": ["simp"]}, n_total=20, ids=ids)
        t = make_run(tmp_path, "li_s", "li", {"p0": ["simp"], "p1": ["exact a"]},
                     n_total=20, ids=ids)
        text = format_report(compare(Arm.load(b), Arm.load(t), n_boot=300, n_perm=300))
        assert "no displacement: True" in text
