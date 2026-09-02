"""The pass@k arm contrast: an arm is the union of its seeds, not a run.

`compare_arms.py` and `discordance_profile.py` are pass@1 instruments — they contrast two runs.
Under a seed sweep the unit changes, and the two questions the plan set for this phase are about the
new unit: does the architecture null survive a 16x budget increase, and does late interaction still
always die at the expansion cap once the sample budget is doubled?

What is guarded here is mostly arithmetic that would be wrong *quietly*: unioning the wrong runs,
counting a rejected proof, or reporting a rate against a denominator that is not the benchmark.

Hermetic: no GPU, no Lean, no model.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from passk_profile import (  # noqa: E402
    PUBLISHED_PASS1_UNION,
    arm_of,
    arm_unions,
    attempted_union,
    loser_profile,
    status_mix,
)


def write_run(root, name, *, arm, seed, solved, statuses=None, n_problems=186,
              benchmark="proofnet_test", samples=32, pf=0.25, rejected=(), expansions=64):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({
        "run_id": name, "seed": seed, "config": {
            "benchmark": benchmark, "arm": arm, "policy_kind": "vllm", "n_problems": n_problems,
            # Fusion runs a late-interaction half, so its manifest carries the first-stage
            # budget too and `Draw` labels it "fusion@50k". The fixture said None, which made
            # every test here kinder than the real export and let a label bug through.
            "n_candidates": 50000 if arm in ("li", "fusion") else None,
            "policy_config": {"premise_free_fraction": pf},
            "search": {"max_expansions": 64, "samples_per_step": samples}},
        "outcome": {"n_proved": len(solved)}}), encoding="utf-8")
    rows = []
    for i in range(n_problems):
        pid = str(i)
        proved = pid in solved
        status = (statuses or {}).get(pid, "proved" if proved else "exhausted")
        rows.append({"problem_id": pid, "proved": proved, "status": status,
                     "proof": ["simp"] if proved else None,
                     "n_expansions": 0 if proved else expansions})
    (d / "attempts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (d / "verification.json").write_text(json.dumps({
        "n_claimed": len(solved), "n_verified": len(solved) - len(rejected),
        "n_failed": len(rejected),
        "failures": [{"problem_id": p, "errors": ["invalid binder name"]} for p in rejected],
    }), encoding="utf-8")
    return d


@pytest.fixture
def population(tmp_path):
    """Two arms x two seeds, with a deliberate disagreement and one rejected claim."""
    write_run(tmp_path, "li_0", arm="li", seed=0, solved={"1", "2"})
    write_run(tmp_path, "li_1", arm="li", seed=1, solved={"2", "3"})
    write_run(tmp_path, "sv_0", arm="sv", seed=0, solved={"1", "4"})
    # seed 1 claims 5 and the re-check rejects it, exactly as ProofNet / sv / seed 6 did
    write_run(tmp_path, "sv_1", arm="sv", seed=1, solved={"1", "5"}, rejected=["5"])
    return tmp_path


# --- the unit of the contrast -------------------------------------------------------------------

def test_an_arm_is_the_union_of_its_seeds(population):
    u = arm_unions(sorted(population.iterdir()))
    assert u["li"] == {"proofnet_test:1", "proofnet_test:2", "proofnet_test:3"}


def test_a_rejected_claim_does_not_enter_the_union(population):
    # The whole discount chain, end to end: verification.json -> load_draw -> arm union.
    u = arm_unions(sorted(population.iterdir()))
    assert "proofnet_test:5" not in u["sv"]
    assert u["sv"] == {"proofnet_test:1", "proofnet_test:4"}


def test_the_budget_suffix_does_not_split_the_late_interaction_arm(population):
    # `Draw.arm` becomes "li@50k" when the run recorded a first-stage budget. Treating that as a
    # third arm would report two half-populated li unions and no contrast at all.
    from prooflens_prover.eval.draws import load_draw
    arms = {arm_of(load_draw(p)) for p in sorted(population.iterdir())}
    assert arms == {"li", "sv"}


def test_the_budget_suffix_is_stripped_from_every_arm_not_just_li(tmp_path):
    """`fusion@50k` is a real label, and an `arm_of` that only special-cased `li` returned it whole.

    The consequence was silent: `--arm fusion` then matched nothing, `restrict` dropped all four
    fusion runs, and the script printed a complete-looking li-vs-sv contrast with the arm under
    investigation simply absent.
    """
    from prooflens_prover.eval.draws import load_draw
    write_run(tmp_path, "f0", arm="fusion", seed=0, solved={"1"}, benchmark="fate_m",
              n_problems=141)
    assert load_draw(tmp_path / "f0").arm == "fusion@50k"
    assert arm_of(load_draw(tmp_path / "f0")) == "fusion"


def test_attempted_is_the_union_so_the_denominator_is_the_benchmark(population):
    assert len(attempted_union(sorted(population.iterdir()))) == 186


# --- the denominator guard ----------------------------------------------------------------------

def test_a_short_benchmark_is_refused_rather_than_rated(tmp_path):
    # 60 problems where 186 are expected: a --limit run reaching the table would report a rate
    # against a different denominator and compare it to a published one.
    write_run(tmp_path, "li_0", arm="li", seed=0, solved={"1"}, n_problems=60)
    write_run(tmp_path, "sv_0", arm="sv", seed=0, solved={"1"}, n_problems=60)
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "passk_profile.py"),
         "--results-root", str(tmp_path), "--match", "n_problems=60"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    # The per-benchmark n is appended automatically, so n_problems=60 matches nothing for
    # proofnet_test and the run set is empty rather than mis-rated.
    assert "no runs matched" in out.stdout


def test_the_published_baseline_is_the_single_run_union_not_a_replicate_union():
    # The export holds replicates of the published runs. Unioning everything at 16 samples gives a
    # multi-seed figure reported as the published single-seed one, inflating the baseline and
    # understating the gain: ProofNet 33 and FATE-M 59 against the real 32 and 56.
    assert PUBLISHED_PASS1_UNION == {"proofnet_test": 32, "fate_m": 56}


# --- the failure-mode profile -------------------------------------------------------------------

def test_the_loser_profile_counts_every_seed_of_the_losing_arm(population):
    dirs = sorted(population.iterdir())
    prof = loser_profile(dirs, "sv", {"proofnet_test:3"})
    # One problem, two sv seeds, neither solved it.
    assert prof["n_problems"] == 1
    assert sum(prof["statuses"].values()) == 2


def test_the_profile_reports_where_an_exhausted_search_stopped(population):
    prof = loser_profile(sorted(population.iterdir()), "sv", {"proofnet_test:3"})
    assert prof["median_expansions_when_exhausted"] == 64


def test_no_candidates_is_kept_distinct_from_exhausted(tmp_path):
    # The distinction decides where the next GPU-hour goes: silence means unreachable and is
    # bought with samples, budget exhaustion means mis-ranked and bought with expansions.
    write_run(tmp_path, "li_0", arm="li", seed=0, solved={"1"})
    write_run(tmp_path, "sv_0", arm="sv", seed=0, solved=set(),
              statuses={"1": "no_candidates"})
    prof = loser_profile(sorted(tmp_path.iterdir()), "sv", {"proofnet_test:1"})
    assert prof["statuses"] == {"no_candidates": 1}
    assert prof["median_expansions_when_exhausted"] is None


def test_the_status_mix_is_a_share_of_that_arms_attempts_only(population):
    total, mix = status_mix(sorted(population.iterdir()), "li")
    assert total == 372            # 2 li seeds x 186
    assert abs(sum(mix.values()) - 100) < 1e-9


# --- end to end ---------------------------------------------------------------------------------

def test_the_report_names_both_arms_and_the_ensemble(population):
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "passk_profile.py"),
         "--results-root", str(population), "--n-boot", "200", "--n-perm", "200"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ENSEMBLE" in out.stdout
    assert "McNemar" in out.stdout
    assert "CI and permutation agree" in out.stdout


def test_the_two_significance_tests_must_agree_before_either_is_reported():
    # A CI excluding zero beside a non-significant permutation p means one is being misread. The
    # pass@1 analysis gates on their agreement rather than picking whichever is kinder, and this
    # script has to keep that standard.
    src = (REPO / "scripts" / "passk_profile.py").read_text(encoding="utf-8")
    assert "do not report either" in src


# --- a third arm, measured at fewer seeds ---------------------------------------------------------

class TestFusionEntersAsAThirdArm:
    """Phase 4 adds `fusion`, and it will be measured at four seeds against the sweep's eight.

    Two things must hold or the comparison is meaningless: fusion must not be mistaken for a
    differently-budgeted `li` (`Draw` labels it `fusion@50k`, since it inherits its sub-retriever's
    first-stage budget), and it must not be contrasted against unions built from twice as many
    draws — that difference would be budget, not architecture.
    """

    @staticmethod
    def _mixed(root):
        for seed in range(4):
            write_run(root, f"li_{seed}", arm="li", seed=seed, solved={"1", "2"})
            write_run(root, f"sv_{seed}", arm="sv", seed=seed, solved={"1", "3"})
            write_run(root, f"fu_{seed}", arm="fusion", seed=seed, solved={"1", "2", "3"})
        # the sweep's deeper seeds, which have no fusion counterpart
        for seed in (4, 5, 6, 7):
            write_run(root, f"li_{seed}", arm="li", seed=seed, solved={"1", "2", "4"})
            write_run(root, f"sv_{seed}", arm="sv", seed=seed, solved={"1", "3"})

    def test_fusion_is_its_own_arm_not_a_late_interaction_variant(self, tmp_path):
        self._mixed(tmp_path)
        u = arm_unions(sorted(tmp_path.iterdir()))
        assert set(u) == {"li", "sv", "fusion"}

    def test_mismatched_seed_depth_is_refused_rather_than_averaged(self, tmp_path):
        self._mixed(tmp_path)
        out = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "passk_profile.py"),
             "--results-root", str(tmp_path), "--n-boot", "200", "--n-perm", "200"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert out.returncode != 0
        assert "different seeds" in out.stdout + out.stderr

    def test_restricting_the_seeds_makes_the_contrast_legitimate(self, tmp_path):
        self._mixed(tmp_path)
        out = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "passk_profile.py"),
             "--results-root", str(tmp_path), "--seeds", "0-3",
             "--n-boot", "200", "--n-perm", "200"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert out.returncode == 0, out.stdout + out.stderr
        assert "pass@4" in out.stdout
        # every pair contrasted, so fusion is measured against both incumbents
        for pair in ("fusion vs li", "fusion vs sv", "li vs sv"):
            assert pair in out.stdout, out.stdout

    def test_selecting_arms_narrows_the_contrast(self, tmp_path):
        self._mixed(tmp_path)
        out = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "passk_profile.py"),
             "--results-root", str(tmp_path), "--seeds", "0-3",
             "--arm", "fusion", "--arm", "li", "--n-boot", "200", "--n-perm", "200"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert out.returncode == 0, out.stdout + out.stderr
        assert "fusion vs li" in out.stdout
        assert "vs sv" not in out.stdout

    def test_the_ensemble_line_is_not_credited_against_the_pass1_union_for_a_single_arm(
            self, tmp_path):
        # `fusion` alone is one arm, so calling its union an "ensemble" gain over the published
        # two-arm union would be comparing a retriever against a pair of retrievers.
        self._mixed(tmp_path)
        out = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "passk_profile.py"),
             "--results-root", str(tmp_path), "--seeds", "0-3", "--arm", "fusion",
             "--n-boot", "200", "--n-perm", "200"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert "vs published pass@1 union" not in out.stdout


class TestSeedSpecParsing:
    def test_a_range_and_a_list_mean_the_same_thing(self):
        from passk_union import parse_seeds
        assert parse_seeds("0-3") == parse_seeds("0,1,2,3") == {0, 1, 2, 3}

    def test_absent_means_every_seed(self):
        from passk_union import parse_seeds
        assert parse_seeds(None) is None

    def test_an_empty_selection_is_refused_rather_than_silently_taking_everything(self):
        from passk_union import parse_seeds
        with pytest.raises(SystemExit):
            parse_seeds(",")


class TestAnArmMissingFromOneBenchmarkIsNotPooled:
    """Phase 4 runs fusion on FATE-M and not on ProofNet, and that shape breaks a pooled contrast.

    An arm measured on a subset of the benchmarks has its union drawn from a smaller problem set
    while the pooled denominator is every problem, so the benchmarks it never ran on are scored as
    failures. The sign of the resulting error is not conservative and its confidence is not low:
    with fusion at 70 against late interaction's 64 on FATE-M — a clear win — the unguarded pooled
    contrast reported **fusion -34 problems at p = 0.0000**, CI excluding zero, and the CI and the
    permutation test agreeing with each other. Maximum confidence, wrong sign.
    """

    @staticmethod
    def _split(root):
        for seed in range(4):
            write_run(root, f"pn_li_{seed}", arm="li", seed=seed,
                      solved={str(i) for i in range(40)}, n_problems=186)
            write_run(root, f"pn_sv_{seed}", arm="sv", seed=seed,
                      solved={str(i) for i in range(38)}, n_problems=186)
            for arm, k in (("li", 64), ("sv", 62), ("fusion", 70)):
                write_run(root, f"fm_{arm}_{seed}", arm=arm, seed=seed,
                          solved={str(i) for i in range(k)}, n_problems=141, benchmark="fate_m")

    def _run(self, root):
        return subprocess.run(
            [sys.executable, str(REPO / "scripts" / "passk_profile.py"),
             "--results-root", str(root), "--seeds", "0-3",
             "--n-boot", "300", "--n-perm", "300"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_the_partial_arm_is_excluded_from_the_pooled_contrast(self, tmp_path):
        self._split(tmp_path)
        out = self._run(tmp_path)
        assert out.returncode == 0, out.stdout + out.stderr
        pooled = out.stdout[out.stdout.index("POOLED  ("):]
        assert "fusion vs li" not in pooled
        assert "fusion vs sv" not in pooled
        assert "li vs sv" in pooled

    def test_the_exclusion_says_which_benchmarks_it_missed(self, tmp_path):
        # Silently dropping an arm is its own failure: a reader would look for fusion in the pooled
        # table, not find it, and have no way to tell whether it lost or was never eligible.
        self._split(tmp_path)
        out = self._run(tmp_path)
        assert "NOT POOLED" in out.stdout
        assert "'fusion'" in out.stdout
        assert "proofnet_test" in out.stdout[out.stdout.index("NOT POOLED"):]

    def test_the_partial_arm_is_still_contrasted_where_it_did_run(self, tmp_path):
        # Excluding it from the pooled section must not hide it entirely — FATE-M is where it was
        # measured and where the comparison is legitimate.
        self._split(tmp_path)
        out = self._run(tmp_path)
        fate = out.stdout[out.stdout.index("fate_m  ("):out.stdout.index("NOT POOLED")]
        assert "fusion" in fate
        assert "solved only by fusion" in fate

    def test_the_pooled_totals_exclude_it_too(self, tmp_path):
        # Not just the contrasts: an ENSEMBLE line counting a partial arm would overstate coverage
        # on the benchmarks that arm never ran.
        self._split(tmp_path)
        out = self._run(tmp_path)
        pooled = out.stdout[out.stdout.index("POOLED  ("):]
        assert "fusion" not in pooled

    def test_arms_spanning_every_benchmark_are_unaffected(self, tmp_path):
        # The guard must not fire on the normal case, or it would silence the sweep's own contrast.
        for seed in range(4):
            write_run(tmp_path, f"pn_li_{seed}", arm="li", seed=seed, solved={"1"}, n_problems=186)
            write_run(tmp_path, f"pn_sv_{seed}", arm="sv", seed=seed, solved={"2"}, n_problems=186)
            write_run(tmp_path, f"fm_li_{seed}", arm="li", seed=seed, solved={"1"},
                      n_problems=141, benchmark="fate_m")
            write_run(tmp_path, f"fm_sv_{seed}", arm="sv", seed=seed, solved={"2"},
                      n_problems=141, benchmark="fate_m")
        out = self._run(tmp_path)
        assert "NOT POOLED" not in out.stdout
        assert "li vs sv" in out.stdout[out.stdout.index("POOLED  ("):]

    def test_a_single_spanning_arm_leaves_nothing_to_pool_and_says_so(self, tmp_path):
        for seed in range(4):
            write_run(tmp_path, f"pn_li_{seed}", arm="li", seed=seed, solved={"1"}, n_problems=186)
            write_run(tmp_path, f"pn_sv_{seed}", arm="sv", seed=seed, solved={"2"}, n_problems=186)
            write_run(tmp_path, f"fm_li_{seed}", arm="li", seed=seed, solved={"1"},
                      n_problems=141, benchmark="fate_m")
            write_run(tmp_path, f"fm_fu_{seed}", arm="fusion", seed=seed, solved={"3"},
                      n_problems=141, benchmark="fate_m")
        out = self._run(tmp_path)
        assert out.returncode == 0, out.stdout + out.stderr
        assert "no pooled contrast to make" in out.stdout


class TestUnverifiedRunsAreRefused:
    """This script produces numbers for the write-up, so it must gate on verification too.

    `passk_union.py` always has; this one did not, which mattered the moment a new arm arrived. An
    unverified proof is a claim — the search says it closed the goal, and nothing has checked the
    recorded tactics elaborate from the benchmark statement. The sweep found one that did not, in
    1,377 claims.

    And the asymmetry is worse than a missing figure: `eval/draws.py` discounts rejected proofs
    *from the report file*, so a run without one is not merely unchecked, it is exempt from the
    discount its verified rivals receive and competes with its bad claims intact.
    """

    @staticmethod
    def _pair(root, *, verify_second=True):
        for seed in range(2):
            for bench, n in (("proofnet_test", 186), ("fate_m", 141)):
                write_run(root, f"{bench}_li_{seed}", arm="li", seed=seed, solved={"1"},
                          n_problems=n, benchmark=bench)
                d = write_run(root, f"{bench}_sv_{seed}", arm="sv", seed=seed, solved={"2"},
                              n_problems=n, benchmark=bench)
                if not verify_second:
                    (d / "verification.json").unlink()

    def _run(self, root, *extra):
        return subprocess.run(
            [sys.executable, str(REPO / "scripts" / "passk_profile.py"),
             "--results-root", str(root), "--n-boot", "200", "--n-perm", "200", *extra],
            capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_a_run_with_no_report_stops_the_contrast(self, tmp_path):
        self._pair(tmp_path, verify_second=False)
        out = self._run(tmp_path)
        assert out.returncode == 1
        assert "have not been verified" in out.stdout + out.stderr

    def test_the_refusal_names_the_runs_and_the_fix(self, tmp_path):
        self._pair(tmp_path, verify_second=False)
        out = self._run(tmp_path)
        said = out.stdout + out.stderr
        assert "verify_proofs.sbatch" in said
        assert "sv" in said

    def test_the_escape_hatch_warns_loudly_rather_than_going_quiet(self, tmp_path):
        # A work-in-progress look is legitimate; a work-in-progress look that reads like a result
        # is not, so the warning has to survive into the output that gets pasted around.
        self._pair(tmp_path, verify_second=False)
        out = self._run(tmp_path, "--allow-unverified")
        assert out.returncode == 0, out.stdout + out.stderr
        assert "WARNING" in out.stdout
        assert "counted anyway" in out.stdout

    def test_a_fully_verified_set_passes_silently(self, tmp_path):
        self._pair(tmp_path)
        out = self._run(tmp_path)
        assert out.returncode == 0, out.stdout + out.stderr
        assert "have not been verified" not in out.stdout
        assert "WARNING" not in out.stdout

    def test_an_incomplete_report_counts_as_unverified(self, tmp_path):
        # What an interrupted verify leaves. It cannot say what was checked, so the discount cannot
        # be applied and a rejected proof would be counted as good.
        self._pair(tmp_path)
        (tmp_path / "fate_m_sv_0" / "verification.json").write_text(
            json.dumps({"run": "x"}), encoding="utf-8")
        out = self._run(tmp_path)
        assert out.returncode == 1
        assert "not a complete report" in out.stdout + out.stderr


class TestARequestedArmMustExist:
    """`--arm` matching nothing is a refusal, not a quiet omission.

    The failure this guards is the one that actually happened: a label mismatch removed the fusion
    runs and the script still printed per-benchmark tables, a pooled contrast and an agreement
    verdict — all correct for the arms that survived, and silent about the one that did not.
    """

    def _both(self, root):
        for bench, n in (("proofnet_test", 186), ("fate_m", 141)):
            for arm in ("li", "sv"):
                for seed in range(2):
                    write_run(root, f"{bench}_{arm}_{seed}", arm=arm, seed=seed,
                              solved={"1", str(seed)}, benchmark=bench, n_problems=n)

    def _run(self, root, *extra):
        return subprocess.run(
            [sys.executable, str(REPO / "scripts" / "passk_profile.py"),
             "--results-root", str(root), "--n-boot", "200", "--n-perm", "200", *extra],
            capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_an_arm_present_nowhere_is_refused(self, tmp_path):
        self._both(tmp_path)
        out = self._run(tmp_path, "--arm", "li", "--arm", "fusion")
        assert out.returncode != 0
        assert "matched no run on any benchmark" in out.stdout + out.stderr

    def test_a_typo_is_refused_rather_than_narrowing_to_one_arm(self, tmp_path):
        self._both(tmp_path)
        out = self._run(tmp_path, "--arm", "li", "--arm", "svv")
        assert out.returncode != 0
        assert "svv" in out.stdout + out.stderr

    def test_an_arm_present_on_one_benchmark_only_is_accepted(self, tmp_path):
        """Fusion ran on FATE-M alone. That is legitimate — the pooling guard handles it."""
        self._both(tmp_path)
        for seed in range(2):
            write_run(tmp_path, f"fate_m_fusion_{seed}", arm="fusion", seed=seed,
                      solved={"1", "7"}, benchmark="fate_m", n_problems=141)
        out = self._run(tmp_path, "--arm", "fusion", "--arm", "li", "--arm", "sv")
        assert out.returncode == 0, out.stdout + out.stderr
        assert "fusion" in out.stdout
        assert "NOT POOLED" in out.stdout

    def test_no_arm_filter_never_refuses(self, tmp_path):
        self._both(tmp_path)
        assert self._run(tmp_path).returncode == 0
