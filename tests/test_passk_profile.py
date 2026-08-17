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
            "n_candidates": 50000 if arm == "li" else None,
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
