"""Replication variance — Phase 2's noise floor, and the guards that keep it honest.

The script exists to test three published claims against sampling noise, so the failure mode that
matters most is *silently measuring nothing*: replicates that are secretly the same draw report zero
variance, which reads as an exceptionally stable result. `TestTripwire` is the important class here.

Hermetic: every fixture is written by this file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from replication_variance import (  # noqa: E402
    discordance,
    group_draws,
    identical_proof_fraction,
    load_draw,
    solve_rate_map,
    spread,
    union_gain,
)

REPO = Path(__file__).resolve().parent.parent

BASE_CONFIG = {
    "benchmark": "fate_m", "arm": "li", "policy_kind": "vllm", "n_candidates": 50_000,
    "search": {"samples_per_step": 16}, "top_k": 10,
}


def write_run(tmp_path, name, *, arm="li", seed=0, benchmark="fate_m", solved=None,
              attempted=None, n_candidates=50_000, config_override=None):
    """`solved` maps problem id -> proof tactic list. `attempted` defaults to solved's keys."""
    solved = solved or {}
    attempted = attempted if attempted is not None else list(solved)
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    cfg = {**BASE_CONFIG, "arm": arm, "benchmark": benchmark}
    if n_candidates is None:
        cfg.pop("n_candidates")
    else:
        cfg["n_candidates"] = n_candidates
    cfg.update(config_override or {})
    (d / "manifest.json").write_text(json.dumps({
        "run_id": name, "seed": seed, "started_utc": "2026-08-10T00:00:00+00:00",
        "config": cfg, "outcome": {"n_proved": len(solved)},
    }), encoding="utf-8")
    (d / "attempts.jsonl").write_text("\n".join(
        json.dumps({"problem_id": pid, "proved": pid in solved,
                    "proof": solved.get(pid), "status": "proved" if pid in solved else "exhausted"})
        for pid in attempted
    ), encoding="utf-8")
    return d


def run_script(tmp_path, runs, *extra):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "replication_variance.py"),
         *sum([["--run", str(r)] for r in runs], []),
         "--n-boot", "300", "--n-perm", "300",
         "--json-out", str(tmp_path / "out.json"), *extra],
        capture_output=True, text=True, cwd=REPO,
    )


class TestLoading:
    def test_problem_ids_are_namespaced_by_benchmark(self, tmp_path):
        d = write_run(tmp_path, "r", benchmark="proofnet_test", solved={"1": ["rfl"]})
        draw = load_draw(d)
        assert draw.solved == {"proofnet_test:1"}
        assert draw.attempted == {"proofnet_test:1"}

    def test_the_arm_label_carries_the_first_stage_budget(self, tmp_path):
        assert load_draw(write_run(tmp_path, "r", arm="li")).arm == "li@50k"
        assert load_draw(write_run(tmp_path, "s", arm="sv", n_candidates=None)).arm == "sv"

    def test_the_seed_comes_from_the_manifest_top_level(self, tmp_path):
        assert load_draw(write_run(tmp_path, "r", seed=7)).seed == 7

    def test_unsolved_problems_are_attempted_but_not_solved(self, tmp_path):
        d = write_run(tmp_path, "r", solved={"1": ["rfl"]}, attempted=["1", "2", "3"])
        draw = load_draw(d)
        assert len(draw.attempted) == 3
        assert draw.solved == {"fate_m:1"}


class TestReplicateGuards:
    def test_two_runs_with_the_same_seed_are_refused(self, tmp_path):
        a = write_run(tmp_path, "a", seed=0, solved={"1": ["rfl"]})
        b = write_run(tmp_path, "b", seed=0, solved={"2": ["rfl"]})
        with pytest.raises(SystemExit, match="same draw"):
            group_draws([a, b])

    def test_draws_differing_in_configuration_are_refused(self, tmp_path):
        # Two runs differing in search configuration are different experiments; averaging them
        # would report as sampling variance something that is really an effect.
        a = write_run(tmp_path, "a", seed=0, solved={"1": ["rfl"]},
                      config_override={"search": {"samples_per_step": 16}})
        b = write_run(tmp_path, "b", seed=1, solved={"1": ["rfl"]},
                      config_override={"search": {"samples_per_step": 32}})
        with pytest.raises(SystemExit, match="differ in"):
            group_draws([a, b])

    def test_a_different_first_stage_budget_is_a_different_arm_not_a_bad_replicate(self, tmp_path):
        """`n_candidates` is part of the arm label, so 1k and 50k runs never meet as replicates."""
        a = write_run(tmp_path, "a", seed=0, n_candidates=1_000, solved={"1": ["rfl"]})
        b = write_run(tmp_path, "b", seed=0, n_candidates=50_000, solved={"1": ["rfl"]})
        groups = group_draws([a, b])
        assert sorted(arm for _, arm in groups) == ["li@1k", "li@50k"]

    def test_a_resumed_run_with_a_different_problem_count_is_allowed(self, tmp_path):
        a = write_run(tmp_path, "a", seed=0, solved={"1": ["rfl"]},
                      config_override={"n_problems": 141})
        b = write_run(tmp_path, "b", seed=1, solved={"1": ["rfl"]},
                      config_override={"n_problems": 140})
        assert len(group_draws([a, b])[("fate_m", "li@50k")]) == 2

    def test_draws_are_sorted_by_seed(self, tmp_path):
        a = write_run(tmp_path, "a", seed=5, solved={"1": ["rfl"]})
        b = write_run(tmp_path, "b", seed=2, solved={"1": ["rfl"]})
        assert [d.seed for d in group_draws([a, b])[("fate_m", "li@50k")]] == [2, 5]


class TestTripwire:
    """The guard the phase depends on. Two draws that are secretly one draw would report zero
    variance for every statistic below, which looks like stability rather than absence."""

    def test_identical_proofs_are_detected(self, tmp_path):
        a = load_draw(write_run(tmp_path, "a", seed=0, solved={"1": ["rfl"], "2": ["simp"]}))
        b = load_draw(write_run(tmp_path, "b", seed=1, solved={"1": ["rfl"], "2": ["simp"]}))
        assert identical_proof_fraction(a, b) == (2, 1.0)

    def test_differing_proofs_are_detected(self, tmp_path):
        a = load_draw(write_run(tmp_path, "a", seed=0, solved={"1": ["rfl"], "2": ["simp"]}))
        b = load_draw(write_run(tmp_path, "b", seed=1, solved={"1": ["rfl"], "2": ["omega"]}))
        assert identical_proof_fraction(a, b) == (2, 0.5)

    def test_no_shared_proofs_is_not_a_division_by_zero(self, tmp_path):
        a = load_draw(write_run(tmp_path, "a", seed=0, solved={"1": ["rfl"]}))
        b = load_draw(write_run(tmp_path, "b", seed=1, solved={"2": ["rfl"]}))
        assert identical_proof_fraction(a, b) == (0, None)

    def test_the_script_refuses_a_duplicated_run(self, tmp_path):
        proofs = {f"p{i}": ["rfl"] for i in range(6)}
        runs = [
            write_run(tmp_path, "li0", arm="li", seed=0, solved=proofs),
            write_run(tmp_path, "li1", arm="li", seed=1, solved=proofs),   # same draw, new seed
            write_run(tmp_path, "sv0", arm="sv", seed=0, n_candidates=None, solved=proofs),
            write_run(tmp_path, "sv1", arm="sv", seed=1, n_candidates=None, solved=proofs),
        ]
        p = run_script(tmp_path, runs)
        assert p.returncode != 0
        assert "REFUSED" in p.stdout
        assert "same draw" in p.stdout + p.stderr


class TestFloorStatistics:
    def test_discordance_counts_each_direction(self):
        assert discordance({"a", "b"}, {"b", "c"}) == (1, 1)

    def test_identical_sets_are_not_discordant(self):
        assert discordance({"a"}, {"a"}) == (0, 0)

    def test_union_gain_is_measured_against_the_better_set(self):
        assert union_gain({"a", "b"}, {"b", "c"}) == 1        # union 3, better single 2
        assert union_gain({"a", "b"}, {"a"}) == 0             # a subset adds nothing

    def test_spread_needs_two_values_for_a_deviation(self):
        assert spread([3.0]) == (3.0, None)
        mean, sd = spread([2.0, 4.0])
        assert (mean, sd) == (3.0, pytest.approx(1.4142, abs=1e-4))


class TestSolveRates:
    def test_a_rate_is_the_fraction_of_draws_that_solved_it(self, tmp_path):
        draws = [
            load_draw(write_run(tmp_path, "a", seed=0, solved={"1": ["rfl"]},
                                attempted=["1", "2"])),
            load_draw(write_run(tmp_path, "b", seed=1, solved={"1": ["rfl"], "2": ["simp"]},
                                attempted=["1", "2"])),
        ]
        assert solve_rate_map(draws) == {"fate_m:1": 1.0, "fate_m:2": 0.5}

    def test_only_problems_every_draw_attempted_are_rated(self, tmp_path):
        draws = [
            load_draw(write_run(tmp_path, "a", seed=0, solved={}, attempted=["1", "2"])),
            load_draw(write_run(tmp_path, "b", seed=1, solved={"1": ["rfl"]}, attempted=["1"])),
        ]
        assert set(solve_rate_map(draws)) == {"fate_m:1"}


class TestEndToEnd:
    def _runs(self, tmp_path, li_solved, sv_solved):
        """One draw per seed per arm, with proofs made distinct so the tripwire passes.

        SV is emitted first because arm order fixes the baseline, so every delta below reads as
        li − sv — the direction the write-up reports.
        """
        runs = []
        pids = [f"p{i}" for i in range(20)]
        for arm, per_seed, ncand in (("sv", sv_solved, None), ("li", li_solved, 50_000)):
            for seed, solved_ids in enumerate(per_seed):
                runs.append(write_run(
                    tmp_path, f"{arm}{seed}", arm=arm, seed=seed, n_candidates=ncand,
                    solved={p: [f"exact h{p}_{arm}_{seed}"] for p in solved_ids},
                    attempted=pids,
                ))
        return runs

    def test_the_contrast_is_treatment_minus_control_in_run_order(self, tmp_path):
        runs = self._runs(tmp_path, [["p0", "p1"], ["p0", "p1"]], [["p0"], ["p0"]])
        p = run_script(tmp_path, runs)
        assert p.returncode == 0, p.stderr
        payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert payload["comparisons"][0]["contrast"] == "li@50k vs sv"
        assert payload["comparisons"][0]["mean_delta"] == pytest.approx(1.0)

    def test_one_draw_per_arm_reports_that_nothing_can_be_measured(self, tmp_path):
        runs = self._runs(tmp_path, [["p0", "p1"]], [["p0", "p2"]])
        p = run_script(tmp_path, runs)
        assert p.returncode == 1
        assert "ONLY ONE DRAW PER ARM" in p.stdout

    def test_a_stable_arm_has_a_low_floor_and_the_contrast_is_measured(self, tmp_path):
        # Both arms solve the same problems in every draw: floor 0, between-arm discordance 0.
        runs = self._runs(tmp_path, [["p0", "p1"], ["p0", "p1"]], [["p0", "p1"], ["p0", "p1"]])
        p = run_script(tmp_path, runs)
        assert p.returncode == 0, p.stderr
        payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert payload["noise_floor"]["discordance_range"] == [0, 0]
        c = payload["comparisons"][0]
        assert c["mean_delta"] == 0.0
        assert c["sd_delta"] == 0.0
        assert c["mean_discordance"] == 0.0

    def test_a_noisy_arm_puts_the_between_arm_effect_inside_the_floor(self, tmp_path):
        """The finding Phase 2 exists to be able to reach: each arm flips 2 problems against itself
        and the two arms differ by 2, so the disagreement says nothing about architecture."""
        runs = self._runs(
            tmp_path,
            [["p0", "p1"], ["p0", "p2"]],       # li flips p1 -> p2 between its own draws
            [["p0", "p3"], ["p0", "p4"]],       # sv flips p3 -> p4
        )
        p = run_script(tmp_path, runs)
        assert p.returncode == 0, p.stderr
        payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert payload["noise_floor"]["discordance_range"] == [2, 2]
        c = payload["comparisons"][0]
        assert c["discordance_above_floor"] is False
        assert "INSIDE the noise floor" in p.stdout

    def test_a_real_effect_clears_the_floor(self, tmp_path):
        # li is perfectly stable and solves 6 problems sv never does: discordance 6 > floor 0.
        li = [["p0", "p1", "p2", "p3", "p4", "p5"]] * 2
        sv = [["p0"], ["p0"]]
        p = run_script(tmp_path, self._runs(tmp_path, li, sv))
        assert p.returncode == 0, p.stderr
        payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        c = payload["comparisons"][0]
        assert c["discordance_above_floor"] is True
        assert c["mean_delta"] == pytest.approx(5.0)
        assert "ABOVE the noise floor" in p.stdout

    def test_discordance_uses_every_cross_pair_not_only_matched_seeds(self, tmp_path):
        """Seed 1 of sv and seed 1 of li are independent draws with nothing pairing them, so all
        i x j pairs are valid. Restricting to i == j would use 2 of the 4 available pairs."""
        runs = self._runs(tmp_path, [["p0", "p1"], ["p0", "p2"]], [["p0"], ["p0"]])
        p = run_script(tmp_path, runs)
        assert p.returncode == 0, p.stderr
        payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert payload["comparisons"][0]["n_cross_pairs"] == 4
        assert "over all 4 cross pairs" in p.stdout

    def test_the_rate_comparison_uses_every_problem_not_only_solved_ones(self, tmp_path):
        runs = self._runs(tmp_path, [["p0", "p1"], ["p0", "p2"]], [["p0"], ["p0"]])
        p = run_script(tmp_path, runs)
        assert p.returncode == 0, p.stderr
        payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        rc = payload["comparisons"][0]["rate_comparison"]
        assert rc["n_problems"] == 20, "the denominator is the benchmark, not the solved set"
        # li solves p1 and p2 in one draw each (0.5 apiece), sv solves neither.
        assert rc["mean_difference"] == pytest.approx((0.5 + 0.5) / 20)

    def test_benchmarks_are_pooled_and_kept_distinct(self, tmp_path):
        runs = []
        for bench in ("fate_m", "proofnet_test"):
            for arm, ncand in (("li", 50_000), ("sv", None)):
                for seed in (0, 1):
                    runs.append(write_run(
                        tmp_path, f"{bench}_{arm}{seed}", arm=arm, seed=seed, benchmark=bench,
                        n_candidates=ncand, attempted=["1", "2"],
                        solved={"1": [f"exact h_{bench}_{arm}_{seed}"]},
                    ))
        p = run_script(tmp_path, runs)
        assert p.returncode == 0, p.stderr
        payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        # 2 benchmarks x 1 solved each = 2 pooled solved problems per draw, 4 pooled attempted.
        assert payload["comparisons"][0]["per_draw"][0]["proved"] == [2, 2]
        assert payload["comparisons"][0]["rate_comparison"]["n_problems"] == 4

    def test_one_run_is_refused(self, tmp_path):
        p = run_script(tmp_path, [write_run(tmp_path, "only", solved={"1": ["rfl"]})])
        assert p.returncode != 0
        assert "at least twice" in p.stdout + p.stderr
