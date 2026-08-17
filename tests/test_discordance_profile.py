"""Discordance profiling — which problems each arm exclusively solves, and how the other failed.

Tier 1 reports an exact tie with the arms disagreeing about *which* theorems they close. The
discussion needs to know whether one architecture wins a recognisably different kind of problem, and
the load-bearing signal is the **loser's terminal status**: `no_candidates` means it ran out of
ideas and the winner's retrieval supplied one; `max_expansions` means it had plenty to try and spent
its whole budget on the wrong things.

These tests pin that classification, because the entire interpretation rests on it.

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

from discordance_profile import (  # noqa: E402
    baseline,
    classify_area,
    failure_mode,
    profile,
    read_run,
)

REPO = Path(__file__).resolve().parent.parent


def write_run(tmp_path, name, *, arm, rows, benchmark="fate_m", n_candidates=None):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    cfg = {"benchmark": benchmark, "arm": arm}
    if n_candidates is not None:
        cfg["n_candidates"] = n_candidates
    (d / "manifest.json").write_text(
        json.dumps({"run_id": name, "config": cfg}), encoding="utf-8")
    (d / "attempts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return d


def attempt(pid, *, proved=False, status="exhausted", limit_hit=None, proof=None,
            n_expansions=1, n_tactics_tried=1, elapsed_s=1.0):
    return {"problem_id": pid, "proved": proved, "status": status, "limit_hit": limit_hit,
            "proof": proof, "n_expansions": n_expansions,
            "n_tactics_tried": n_tactics_tried, "elapsed_s": elapsed_s}


# --- the failure-mode classification, which the whole interpretation rests on ----------------

def test_running_out_of_candidates_is_distinguished_from_running_out_of_budget():
    # These mean opposite things about retrieval: the first says the winner's retriever supplied an
    # option that did not exist, the second says it reordered options both arms already had.
    silent = failure_mode(attempt("1", status="no_candidates"))
    exhausted = failure_mode(attempt("2", status="exhausted", limit_hit="max_expansions"))
    assert "no_candidates" in silent
    assert "max_expansions" in exhausted
    assert silent != exhausted


def test_a_harness_error_is_not_counted_as_a_retrieval_failure():
    # The arm never really attempted the problem, so it is evidence about neither retriever.
    assert "never really attempted" in failure_mode(attempt("1", status="error"))


def test_a_wall_clock_exhaustion_is_reported_separately_from_the_expansion_cap():
    assert failure_mode(attempt("1", status="exhausted", limit_hit="wall_clock")) == "wall_clock"


def test_an_unrecognised_exhaustion_reason_is_reported_rather_than_silently_bucketed():
    mode = failure_mode(attempt("1", status="exhausted", limit_hit="something_new"))
    assert "something_new" in mode


# --- area classification ---------------------------------------------------------------------

@pytest.mark.parametrize("statement,expected", [
    ("theorem foo (G : Type) [Group G] (H : Subgroup G) : True", "group theory"),
    ("theorem foo (R : Type) [Ring R] (I : Ideal R) : True", "ring / field"),
    ("theorem foo (f : X → Y) (h : Continuous f) : IsClosed s", "topology"),
    ("theorem foo (p : Nat) (hp : Nat.Prime p) : True", "number theory"),
    ("theorem foo : 1 + 1 = 2", "other"),
])
def test_area_is_inferred_from_the_typeclasses_a_statement_mentions(statement, expected):
    assert classify_area(statement) == expected


def test_area_ordering_is_deterministic_when_several_patterns_match():
    # A finite group acting on a topological space is called group theory; the point is that the
    # label does not depend on dict iteration order.
    s = "theorem foo (G : Type) [Group G] [TopologicalSpace G] : IsOpen (Set.univ : Set G)"
    assert classify_area(s) == classify_area(s) == "group theory"


# --- profiling -------------------------------------------------------------------------------

def test_profile_reports_how_the_loser_failed_on_each_exclusive_win():
    winner = {"a": attempt("a", proved=True, status="proved", proof=["simp"], n_expansions=3),
              "b": attempt("b", proved=True, status="proved", proof=["ring"], n_expansions=5)}
    loser = {"a": attempt("a", status="no_candidates"),
             "b": attempt("b", status="exhausted", limit_hit="max_expansions", n_expansions=64)}
    prof = profile(winner, loser, {"a", "b"}, statements={})
    assert prof["n"] == 2
    assert sum(prof["loser_failure_modes"].values()) == 2
    assert prof["loser_median_expansions"] == 32.5


def test_profile_records_win_difficulty_so_harder_can_be_checked_not_asserted():
    winner = {p: attempt(p, proved=True, status="proved", proof=["simp", "ring"],
                         n_expansions=n, n_tactics_tried=10 * n, elapsed_s=float(n))
              for p, n in (("a", 2), ("b", 4), ("c", 6))}
    prof = profile(winner, {}, {"a", "b", "c"}, statements={})
    assert prof["winner_median_expansions"] == 4
    assert prof["winner_median_proof_steps"] == 2


def test_a_single_premise_citation_is_counted_but_a_term_construction_is_not():
    winner = {
        "a": attempt("a", proved=True, proof=["exact Sylow.normalizer_normalizer P"]),
        "b": attempt("b", proved=True, proof=["exact fun h => h.elim"]),
        "c": attempt("c", proved=True, proof=["intro x", "exact foo"]),
    }
    prof = profile(winner, {}, {"a", "b", "c"}, statements={})
    assert prof["one_step_corpus_citations"] == 1


def test_an_empty_exclusive_win_set_does_not_crash():
    prof = profile({}, {}, set(), statements={})
    assert prof["n"] == 0
    assert prof["winner_median_expansions"] is None


def test_baseline_describes_only_the_problems_that_arm_actually_proved():
    rows = {"a": attempt("a", proved=True, proof=["simp"], n_expansions=2),
            "b": attempt("b", proved=False, n_expansions=64)}
    base = baseline(rows)
    assert base["n_solved"] == 1
    assert base["median_expansions"] == 2


def test_read_run_labels_the_arm_with_its_candidate_budget(tmp_path):
    d = write_run(tmp_path, "r", arm="li", n_candidates=50000,
                  rows=[attempt("1", proved=True, proof=["simp"])])
    benchmark, arm, rows = read_run(d)
    assert (benchmark, arm) == ("fate_m", "li@50k")
    assert set(rows) == {"1"}


# --- the guards ------------------------------------------------------------------------------

def test_comparing_two_runs_of_the_same_arm_is_refused(tmp_path):
    rows = [attempt("1", proved=True, proof=["simp"])]
    a = write_run(tmp_path, "a", arm="sv", rows=rows)
    b = write_run(tmp_path, "b", arm="sv", rows=rows)
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "discordance_profile.py"),
         "--a", str(a), "--b", str(b)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "arm" in (proc.stdout + proc.stderr)


def test_comparing_across_benchmarks_is_refused(tmp_path):
    rows = [attempt("1", proved=True, proof=["simp"])]
    a = write_run(tmp_path, "a", arm="sv", rows=rows, benchmark="fate_m")
    b = write_run(tmp_path, "b", arm="li", rows=rows, benchmark="proofnet_test")
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "discordance_profile.py"),
         "--a", str(a), "--b", str(b)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "different benchmarks" in (proc.stdout + proc.stderr)


def test_the_script_never_prints_a_p_value(tmp_path):
    # The exclusive-win sets are 4-10 problems. Any p-value here would invite a reader to treat a
    # descriptive breakdown as a test, which at this n it cannot be.
    a = write_run(tmp_path, "a", arm="sv",
                  rows=[attempt("1", proved=True, status="proved", proof=["simp"]),
                        attempt("2", status="no_candidates")])
    b = write_run(tmp_path, "b", arm="li",
                  rows=[attempt("1", status="no_candidates"),
                        attempt("2", proved=True, status="proved", proof=["ring"])])
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "discordance_profile.py"),
         "--a", str(a), "--b", str(b)], capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    assert "p =" not in proc.stdout and "p-value" not in proc.stdout
    assert "too small for a significance test" in proc.stdout
