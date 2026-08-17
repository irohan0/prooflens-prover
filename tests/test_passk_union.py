"""pass@k over an ensemble of arms — the estimator and the three things it refuses.

This script produces the paper's headline number, so its failure modes all point the same way: each
of them makes the result look *better* than it is. A duplicated seed raises a problem's solved-seed
count without adding evidence. A config drift turns "one system measured k times" into two systems.
An unverified proof is a claim. All three are refused rather than warned about.

The estimator itself is the standard unbiased `1 - C(K-c,k)/C(K,k)`, not the union of whichever k
runs finished first — that would count the lucky ordering.

Hermetic: every fixture is written by this file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passk_union import (  # noqa: E402
    PUBLISHED_BUDGET,
    check_verified,
    coverage_curve,
    discover,
    group,
    pass_at_k,
)
from prooflens_prover.eval.draws import load_draw  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def write_run(root, name, *, arm, seed, solved, attempted=("1", "2", "3", "4"),
              benchmark="proofnet_test", verified=True, n_failed=0, lean_project="/nfs/lean",
              samples=16, extra_config=None, failed_ids=None, list_failures=True,
              incomplete=False):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "benchmark": benchmark, "arm": arm, "policy_kind": "vllm",
        "n_problems": len(attempted), "lean_project": lean_project,
        "search": {"max_expansions": 64, "samples_per_step": samples},
        **(extra_config or {}),
    }
    (d / "manifest.json").write_text(json.dumps({
        "run_id": name, "seed": seed, "config": cfg, "outcome": {"n_proved": len(solved)},
    }), encoding="utf-8")
    (d / "attempts.jsonl").write_text("\n".join(
        json.dumps({"problem_id": p, "proved": p in solved,
                    "proof": ["simp"] if p in solved else None})
        for p in attempted
    ), encoding="utf-8")
    if verified:
        rejected = sorted(solved)[:n_failed] if failed_ids is None else list(failed_ids)
        report = {
            "n_claimed": len(solved), "n_verified": len(solved) - len(rejected),
            "n_failed": len(rejected),
            "failures": [{"problem_id": p, "proof": ["let"], "errors": ["invalid binder name"]}
                         for p in rejected],
        }
        if list_failures is False:      # a report that counts failures it cannot name
            report["failures"] = []
            report["n_failed"] = n_failed
        if incomplete:                  # what an interrupted verify leaves behind
            report = {"run": str(d)}
        (d / "verification.json").write_text(json.dumps(report), encoding="utf-8")
    return d


# --- the estimator ------------------------------------------------------------------------------

def test_a_problem_no_seed_solved_is_zero_at_every_k():
    assert all(pass_at_k(8, 0, k) == 0.0 for k in range(1, 9))


def test_a_problem_every_seed_solved_is_one_at_every_k():
    assert all(pass_at_k(8, 8, k) == 1.0 for k in range(1, 9))


def test_one_seed_of_four_gives_a_quarter_at_k1_and_certainty_at_k4():
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert pass_at_k(4, 1, 4) == pytest.approx(1.0)


def test_the_estimator_is_the_probability_a_k_subset_contains_a_working_seed():
    # 2 of 4 seeds work. Drawing 2 seeds: C(2,2)/C(4,2) = 1/6 chance of missing both.
    assert pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)


def test_pass_at_k_rises_monotonically_with_k():
    curve = [pass_at_k(8, 3, k) for k in range(1, 9)]
    assert curve == sorted(curve)


def test_asking_for_more_k_than_seeds_is_refused():
    # Silently clamping would print a "pass@8" computed from four seeds.
    with pytest.raises(ValueError, match="at least 8 seeds"):
        pass_at_k(4, 1, 8)


def test_the_curve_is_not_the_union_of_the_first_k_runs():
    # THE reason the closed form is used. One problem solved by exactly one of four seeds is worth
    # 0.25 at k=1, not the 1.0 you would get by taking the union of the run that happened to work.
    assert coverage_curve({"p": 1}, n_seeds=4, n_problems=1)[0] == pytest.approx(0.25)


def test_coverage_is_a_fraction_of_all_problems_not_of_the_solved_ones():
    assert coverage_curve({"a": 4, "b": 0}, n_seeds=4, n_problems=4)[0] == pytest.approx(0.25)


# --- refusals -----------------------------------------------------------------------------------

def test_a_duplicated_arm_and_seed_is_refused(tmp_path):
    a = load_draw(write_run(tmp_path, "a", arm="li", seed=0, solved={"1"}))
    b = load_draw(write_run(tmp_path, "b", arm="li", seed=0, solved={"2"}))
    with pytest.raises(SystemExit, match="seed 0"):
        group([a, b])


def test_configs_that_differ_within_an_arm_are_refused(tmp_path):
    a = load_draw(write_run(tmp_path, "a", arm="li", seed=0, solved={"1"}, samples=16))
    b = load_draw(write_run(tmp_path, "b", arm="li", seed=1, solved={"1"}, samples=32))
    with pytest.raises(SystemExit, match="differ in"):
        group([a, b])


def test_a_node_local_lean_project_is_allowed_to_vary(tmp_path):
    # Node-local staging writes the project under /tmp/slurm.<jobid>, so two staged runs of one arm
    # never share a path. Refusing that would reject every genuine cluster replicate.
    a = load_draw(write_run(tmp_path, "a", arm="li", seed=0, solved={"1"},
                            lean_project="/tmp/slurm.1/lean"))
    b = load_draw(write_run(tmp_path, "b", arm="li", seed=1, solved={"1"},
                            lean_project="/tmp/slurm.2/lean"))
    assert set(group([a, b])["li"]) == {0, 1}


def test_arms_that_ran_different_search_budgets_are_refused(tmp_path):
    a = load_draw(write_run(tmp_path, "a", arm="li", seed=0, solved={"1"}, samples=16))
    b = load_draw(write_run(tmp_path, "b", arm="sv", seed=0, solved={"1"}, samples=32))
    with pytest.raises(SystemExit, match="not comparable"):
        group([a, b])


def test_arms_differing_only_in_arm_specific_config_are_fine(tmp_path):
    # `index` and `arm` are *expected* to differ between arms; that is what an arm is.
    a = load_draw(write_run(tmp_path, "a", arm="li", seed=0, solved={"1"},
                            extra_config={"index": "data/index/li"}))
    b = load_draw(write_run(tmp_path, "b", arm="sv", seed=0, solved={"1"},
                            extra_config={"index": "data/index/sv"}))
    assert sorted(group([a, b])) == ["li", "sv"]


def test_an_unverified_run_is_detected(tmp_path):
    d = write_run(tmp_path, "a", arm="li", seed=0, solved={"1"}, verified=False)
    assert "no verification.json" in check_verified(d)


def test_a_run_with_a_failed_proof_is_detected(tmp_path):
    d = write_run(tmp_path, "a", arm="li", seed=0, solved={"1", "2"}, failed_ids=["1"])
    # NOT refused any more. A rejected claim is discounted in eval/draws.py, so it can no longer
    # inflate anything, and discarding the run would remove a whole seed over one problem.
    assert check_verified(d) is None
    assert load_draw(d).solved == {"proofnet_test:2"}
    assert load_draw(d).discounted == {"proofnet_test:1"}


def test_a_clean_verification_passes(tmp_path):
    assert check_verified(write_run(tmp_path, "a", arm="li", seed=0, solved={"1"})) is None


# --- discovery ----------------------------------------------------------------------------------

def test_discovery_skips_runs_that_never_finalised(tmp_path):
    write_run(tmp_path, "good", arm="li", seed=0, solved={"1"})
    bad = write_run(tmp_path, "bad", arm="li", seed=1, solved={"1"})
    m = json.loads((bad / "manifest.json").read_text(encoding="utf-8"))
    del m["outcome"]
    (bad / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert [d.name for d in discover(tmp_path, "proofnet_test", "vllm")] == ["good"]


def test_discovery_can_be_narrowed_to_one_search_budget(tmp_path):
    # The published Tier 1 runs are 64 x 16 and a sweep is 64 x 32, in the same directory. Without
    # a filter `group()` refuses to average across them — correctly, but only after the sweep has
    # been paid for.
    write_run(tmp_path, "old", arm="li", seed=0, solved={"1"}, samples=16)
    write_run(tmp_path, "new", arm="li", seed=0, solved={"1"}, samples=32)
    found = discover(tmp_path, "proofnet_test", "vllm", ["search.samples_per_step=32"])
    assert [d.name for d in found] == ["new"]


def test_a_match_on_a_nested_key_that_is_absent_excludes_the_run(tmp_path):
    # A run predating a config key must not silently match a filter naming it.
    write_run(tmp_path, "old", arm="li", seed=0, solved={"1"})
    spec = ["policy_config.premise_free_fraction=0.25"]
    assert discover(tmp_path, "proofnet_test", "vllm", spec) == []


def test_several_match_specs_must_all_hold(tmp_path):
    write_run(tmp_path, "a", arm="li", seed=0, solved={"1"}, samples=32,
              extra_config={"policy_config": {"premise_free_fraction": 0.25}})
    write_run(tmp_path, "b", arm="li", seed=1, solved={"1"}, samples=32,
              extra_config={"policy_config": {"premise_free_fraction": 0.0}})
    found = discover(tmp_path, "proofnet_test", "vllm",
                     ["search.samples_per_step=32", "policy_config.premise_free_fraction=0.25"])
    assert [d.name for d in found] == ["a"]


def test_discovery_ignores_another_benchmark(tmp_path):
    write_run(tmp_path, "pn", arm="li", seed=0, solved={"1"})
    write_run(tmp_path, "fm", arm="li", seed=0, solved={"1"}, benchmark="fate_m")
    assert [d.name for d in discover(tmp_path, "proofnet_test", "vllm")] == ["pn"]


# --- end to end ---------------------------------------------------------------------------------

def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "passk_union.py"), *map(str, args)],
        capture_output=True, text=True, encoding="utf-8",
    )


def test_the_ensemble_reaches_what_neither_arm_reaches_alone(tmp_path):
    # The whole point: li solves 1 and 2, sv solves 3, at every seed. Neither arm gets past two
    # problems; the ensemble gets three.
    for seed in (0, 1):
        write_run(tmp_path, f"li{seed}", arm="li", seed=seed, solved={"1", "2"})
        write_run(tmp_path, f"sv{seed}", arm="sv", seed=seed, solved={"3"})
    out = run_cli("--benchmark", "proofnet_test", "--results-root", tmp_path,
                  "--out", tmp_path / "r.json")
    assert out.returncode == 0, out.stderr
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["n_ensemble_solved"] == 3
    assert report["curves"]["ENSEMBLE"][0] == pytest.approx(0.75)


def test_the_published_budget_is_64_passes_of_the_large_config_not_64_nodes():
    # The obvious reading of "Pass@64x64" is 64 x 64 x 64 = 262,144, and it is wrong: `large` is
    # MAX_NODES=1024, NUM_SAMPLES=64. Getting this wrong understates their budget 16x and inflates
    # every "fraction of their compute" claim by the same factor. Pinned because the number is
    # quoted in the dissertation (4.2M, and 1/4,000 against a single 1,024-generation pass).
    assert PUBLISHED_BUDGET == 4_194_304
    assert PUBLISHED_BUDGET / (64 * 16) == pytest.approx(4096)


def test_the_budget_ratio_to_the_published_configuration_is_reported(tmp_path):
    for seed in (0, 1):
        write_run(tmp_path, f"li{seed}", arm="li", seed=seed, solved={"1"})
        write_run(tmp_path, f"sv{seed}", arm="sv", seed=seed, solved={"1"})
    out = run_cli("--benchmark", "proofnet_test", "--results-root", tmp_path,
                  "--out", tmp_path / "r.json")
    assert out.returncode == 0, out.stderr
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    # 2 seeds x 2 arms x 64 x 16
    assert report["generations_per_problem"] == 2 * 2 * 64 * 16
    assert "of REAL-Prover's Pass@64x64" in out.stdout


def test_an_unverified_run_stops_the_report_by_default(tmp_path):
    write_run(tmp_path, "li0", arm="li", seed=0, solved={"1"}, verified=False)
    out = run_cli("--benchmark", "proofnet_test", "--results-root", tmp_path)
    assert out.returncode != 0
    assert "not a result" in out.stdout + out.stderr


def test_allow_unverified_warns_but_proceeds(tmp_path):
    write_run(tmp_path, "li0", arm="li", seed=0, solved={"1"}, verified=False)
    out = run_cli("--benchmark", "proofnet_test", "--results-root", tmp_path,
                  "--allow-unverified")
    assert out.returncode == 0, out.stderr
    assert "WARNING unverified" in out.stdout


def test_arms_at_different_seed_sets_are_refused(tmp_path):
    write_run(tmp_path, "li0", arm="li", seed=0, solved={"1"})
    write_run(tmp_path, "li1", arm="li", seed=1, solved={"1"})
    write_run(tmp_path, "sv0", arm="sv", seed=0, solved={"1"})
    out = run_cli("--benchmark", "proofnet_test", "--results-root", tmp_path)
    assert out.returncode != 0
    assert "different seeds" in out.stdout + out.stderr


def test_no_runs_found_is_a_clear_failure_not_an_empty_table(tmp_path):
    out = run_cli("--benchmark", "proofnet_test", "--results-root", tmp_path)
    assert out.returncode != 0
    assert "no finalised" in out.stdout + out.stderr


class TestTheSweepInvocationIsRecordedAndCorrect:
    """The exact command the headline comes from, pinned against the population on disk.

    The budget pilot ran a 60-problem subset at the *winning* config -- 64x32, premise-free 0.25 --
    at seed 0, which is also a sweep seed. So filtering on budget alone selects it alongside the
    full sweep and `group()` refuses for a duplicated (li, 0). That refusal is correct, and it is
    also the only thing between a 60-problem subset and a published rate, so the third filter is
    not optional.
    """

    @staticmethod
    def _population(root):
        def write(name, *, arm, seed, samples, pf, n_problems):
            d = root / name
            d.mkdir(parents=True)
            (d / "manifest.json").write_text(json.dumps({
                "run_id": name, "seed": seed, "config": {
                    "benchmark": "proofnet_test", "arm": arm, "policy_kind": "vllm",
                    "n_problems": n_problems, "policy_config": {"premise_free_fraction": pf},
                    "search": {"max_expansions": 64, "samples_per_step": samples}},
                "outcome": {"n_proved": 33}}), encoding="utf-8")
            (d / "attempts.jsonl").write_text("\n".join(
                json.dumps({"problem_id": f"ex_{i}", "proved": i < 33, "status": "proved"})
                for i in range(n_problems)), encoding="utf-8")
            (d / "verification.json").write_text(json.dumps({"n_failed": 0}), encoding="utf-8")

        write("pilot_64x32_pf", arm="li", seed=0, samples=32, pf=0.25, n_problems=60)
        write("pilot_64x16", arm="li", seed=0, samples=16, pf=0.0, n_problems=60)
        for arm in ("li", "sv"):
            for seed in range(8):
                write(f"sweep_{arm}_{seed}", arm=arm, seed=seed, samples=32, pf=0.25,
                      n_problems=186)

    BUDGET = ["search.samples_per_step=32", "policy_config.premise_free_fraction=0.25"]

    def test_the_budget_filters_alone_pull_in_the_pilot_subset(self, tmp_path):
        self._population(tmp_path)
        found = discover(tmp_path, "proofnet_test", "vllm", self.BUDGET)
        assert len(found) == 17
        assert any("pilot_64x32_pf" == p.name for p in found)

    def test_adding_the_problem_count_isolates_the_sweep(self, tmp_path):
        self._population(tmp_path)
        found = discover(tmp_path, "proofnet_test", "vllm", [*self.BUDGET, "n_problems=186"])
        assert len(found) == 16
        assert not any("pilot" in p.name for p in found)

    def test_the_documented_invocation_carries_all_three_matches(self):
        doc = (Path(__file__).resolve().parent.parent / "scripts" / "passk_union.py").read_text(
            encoding="utf-8")
        header = doc[:doc.index("## What is being reported")]
        for spec in ("search.samples_per_step=32", "policy_config.premise_free_fraction=0.25",
                     "n_problems=186"):
            assert spec in header, f"the recorded command omits {spec}"


class TestARejectedProofIsDiscountedNotDiscarded:
    """A claimed proof that does not re-elaborate is counted as unsolved, and the run is kept.

    Measured on ProofNet / sv / seed 6, whose recorded proof begins with a bare `let`: during search
    each tactic is applied to a proof state on its own, so `let` was accepted as one step, and
    verification joins the steps with newlines where `let` swallows the next line as its binder. The
    rejection is right -- a proof that does not elaborate is not a proof -- but 34 of that run's 35
    proofs verified and it held the joint-highest count of its arm.

    So refusing the whole run is not the stricter option, it is the more biased one: it removes a
    high seed from an eight-seed estimate over a single problem. Discounting can only lower a rate.
    """

    def test_the_rejected_problem_never_enters_solved(self, tmp_path):
        d = write_run(tmp_path, "a", arm="sv", seed=6, solved={"1", "2", "3"}, failed_ids=["2"])
        draw = load_draw(d)
        assert draw.solved == {"proofnet_test:1", "proofnet_test:3"}
        assert draw.discounted == {"proofnet_test:2"}

    def test_the_rejected_problem_is_still_counted_as_attempted(self, tmp_path):
        # It stays in the denominator. Dropping it from `attempted` as well would raise the rate.
        d = write_run(tmp_path, "a", arm="sv", seed=6, solved={"1", "2"}, failed_ids=["2"])
        assert "proofnet_test:2" in load_draw(d).attempted

    def test_its_proof_is_not_kept_so_it_cannot_be_reported_as_one(self, tmp_path):
        d = write_run(tmp_path, "a", arm="sv", seed=6, solved={"1", "2"}, failed_ids=["2"])
        assert "proofnet_test:2" not in load_draw(d).proofs

    def test_discounting_can_only_lower_the_count(self, tmp_path):
        clean = load_draw(write_run(tmp_path / "x", "a", arm="sv", seed=0, solved={"1", "2", "3"}))
        docked = load_draw(write_run(tmp_path / "y", "a", arm="sv", seed=0, solved={"1", "2", "3"},
                                     failed_ids=["3"]))
        assert len(docked.solved) < len(clean.solved)

    def test_a_run_nobody_verified_is_still_refused(self, tmp_path):
        d = write_run(tmp_path, "a", arm="li", seed=0, solved={"1"}, verified=False)
        assert "no verification.json" in check_verified(d)

    def test_an_interrupted_verify_report_is_refused(self, tmp_path):
        # It cannot say what was checked, so the discount cannot be applied and a bad proof would
        # be counted as good.
        d = write_run(tmp_path, "a", arm="li", seed=0, solved={"1"}, incomplete=True)
        assert "not a complete report" in check_verified(d)

    def test_a_report_that_counts_failures_it_cannot_name_is_refused(self, tmp_path):
        # The dangerous middle case: n_failed=1 with an empty failures list would otherwise pass the
        # gate and discount nothing, counting the bad proof as good.
        d = write_run(tmp_path, "a", arm="li", seed=0, solved={"1", "2"}, n_failed=1,
                      list_failures=False)
        why = check_verified(d)
        assert why and "cannot be identified" in why

    def test_the_discount_is_printed_rather_than_folded_in(self):
        src = (Path(__file__).resolve().parent.parent / "scripts" / "passk_union.py").read_text(
            encoding="utf-8")
        assert "DISCOUNTED" in src, (
            "a rate over a smaller solved set than the manifests report must say so, or the two "
            "numbers differ with no explanation anywhere"
        )
