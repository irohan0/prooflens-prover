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
    format_report,
    is_premise_tactic,
    mcnemar_exact_p,
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


def make_run(tmp_path, name, arm, proofs, n_total=N_FATE_M, benchmark="fate_m", errors=(),
             ids=None):
    """Write a minimal run directory: manifest + attempts.jsonl.

    `ids` must be the SAME for both arms of a comparison — two runs of one benchmark attempt the
    same problems, and `compare` intersects on problem id. An earlier version of this fixture
    derived the id list from each arm's own proofs, so the problems only LI solved were simply
    absent from the `none` run and the intersection silently dropped every marginal problem.
    """
    d = tmp_path / name
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "run_id": name, "name": name, "seed": 0, "git_commit": "test",
        "started_utc": "2026-08-03T00:00:00+00:00",
        "config": {"benchmark": benchmark, "arm": arm}, "outcome": None,
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

    def test_refuses_identical_arms(self, tmp_path):
        b = Arm.load(make_run(tmp_path, "a", "li", {}))
        t = Arm.load(make_run(tmp_path, "b", "li", {}))
        with pytest.raises(ValueError, match="nothing to compare"):
            compare(b, t)

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
