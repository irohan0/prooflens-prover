"""Repairing a run that finished its work and died before recording it.

The measured case: ProofNet / sv / seed 6 of the pass@8 sweep recorded all 186 problems in 4 h 59 m
of GPU time, then exited 1 in the reporting block after the search loop. `passk_union.discover`
skips a run with no `outcome`, so those 186 results are invisible and ProofNet is a seed short
of the eight pass@8 needs. `--resume` cannot help either: with nothing left to do it returns
before finalizing, and leaves the manifest incomplete however often it is resubmitted.

Two ways this script could do harm, and both are tested here:

* **finalizing a genuinely unfinished run** — publishing a pass rate whose denominator is however
  far the job happened to get, with nothing in the record to say so;
* **fabricating the health counters** — a zero in `retrieval.n_queries` is not a missing value, it
  is a false statement about a run that queried the retriever thousands of times, and it would make
  this run look like a broken arm rather than an unrecorded one.

Hermetic: no GPU, no Lean, no model.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from finalize_run import UNRECOVERABLE, outcome_from, read_attempts  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def write_run(tmp_path, n=186, proved=33, errors=7, finalized=False, n_problems=186,
              rows=None, seed=6):
    d = tmp_path / "proofnet_test_sv_vllm_x_6b677cc7"
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": d.name, "name": "proofnet_test_sv_vllm", "seed": seed,
        "git_commit": "6b677cc7", "started_utc": "2026-08-16T22:40:22+00:00",
        "config": {"benchmark": "proofnet_test", "arm": "sv", "policy_kind": "vllm",
                   "n_problems": n_problems,
                   "search": {"max_expansions": 64, "samples_per_step": 32}},
        "environment": {}, "outcome": {"n_proved": 99} if finalized else None,
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if rows is None:
        rows = []
        for i in range(n):
            if i < proved:
                status, ok = "proved", True
            elif i < proved + errors:
                status, ok = "error", False
            else:
                status, ok = "exhausted", False
            rows.append({"problem_id": f"ex_{i}", "arm": "sv", "proved": ok, "status": status,
                         "proof": ["aesop"] if ok else None, "elapsed_s": 12.5})
    (d / "attempts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return d


def run_cli(*args):
    # `errors="replace"`: this test reads a child's stdout, and on a console whose encoding is not
    # UTF-8 the child encodes its own output in the local codepage. Decoding strictly would make the
    # harness fail on the platform rather than on the code — which is the opposite of the point, and
    # exactly the confusion the sbatch resolves by merging stderr into stdout.
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "finalize_run.py"), *map(str, args)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


# --- the counts must be the ones prove_benchmark.py would have written --------------------------

def test_the_counts_are_recomputed_from_the_attempts_file():
    rows = [{"problem_id": str(i), "proved": i < 3, "status": "proved" if i < 3 else "exhausted"}
            for i in range(10)]
    out = outcome_from(rows, "x")
    assert (out["n_problems"], out["n_proved"], out["pass_rate"]) == (10, 3, 0.3)


def test_errors_are_counted_the_same_way_the_live_path_counts_them():
    # `n_error = sum(1 for r in rows if r.get("status") == "error")` in prove_benchmark.py.
    # These are problems never really attempted, and they stay in the denominator.
    rows = [{"problem_id": "a", "proved": False, "status": "error"},
            {"problem_id": "b", "proved": False, "status": "exhausted"},
            {"problem_id": "c", "proved": True, "status": "proved"}]
    out = outcome_from(rows, "x")
    assert out["n_error"] == 1 and out["n_problems"] == 3 and out["n_proved"] == 1


def test_the_finalized_manifest_is_visible_to_the_reporter(tmp_path):
    # The whole point: `passk_union.discover` skips a run whose manifest has no outcome.
    sys.path.insert(0, str(REPO / "scripts"))
    from passk_union import discover
    d = write_run(tmp_path)
    assert discover(tmp_path, "proofnet_test", "vllm") == []
    assert run_cli(d).returncode == 0
    assert discover(tmp_path, "proofnet_test", "vllm") == [d]


def test_the_recomputed_count_matches_what_the_run_actually_proved(tmp_path):
    d = write_run(tmp_path, n=186, proved=33, errors=7)
    assert run_cli(d).returncode == 0
    outcome = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["outcome"]
    assert outcome["n_proved"] == 33
    assert outcome["n_problems"] == 186
    assert outcome["n_error"] == 7


# --- what it refuses to invent -------------------------------------------------------------------

def test_the_health_counters_are_null_and_never_zero():
    out = outcome_from([{"problem_id": "a", "proved": True, "status": "proved"}], "x")
    for field in UNRECOVERABLE:
        assert field in out, f"{field} must be present so its absence is explicit"
        assert out[field] is None, (
            f"{field} written as {out[field]!r}. A zero in the retrieval counters asserts that a "
            f"run which queried the retriever thousands of times never queried it"
        )


def test_the_outcome_says_it_was_written_after_the_fact(tmp_path):
    d = write_run(tmp_path)
    run_cli(d)
    outcome = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["outcome"]
    assert outcome["finalized_post_hoc"]["by"] == "scripts/finalize_run.py"
    assert "reason" in outcome["finalized_post_hoc"]


def test_the_original_provenance_is_left_alone(tmp_path):
    # A repaired run is still the run that produced the results; rewriting its git sha or start time
    # to the repair would misattribute 186 problems to whatever the tree looks like now.
    d = write_run(tmp_path)
    before = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    run_cli(d)
    after = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    for key in ("run_id", "seed", "git_commit", "started_utc", "config"):
        assert after[key] == before[key]


# --- what it refuses to do -----------------------------------------------------------------------

def test_a_genuinely_unfinished_run_is_refused(tmp_path):
    # 120 of 186 recorded. Finalizing this publishes 33/120 as a ProofNet rate.
    d = write_run(tmp_path, n=120, proved=20, errors=4, n_problems=186)
    out = run_cli(d)
    assert out.returncode == 1
    assert "genuinely unfinished" in out.stdout + out.stderr
    assert "120 of 186" in out.stdout + out.stderr


def test_the_refusal_tells_you_the_seed_to_resume_with(tmp_path):
    # Resuming without the original seed appends a different draw to the same run.
    d = write_run(tmp_path, n=120, n_problems=186, seed=6)
    out = run_cli(d)
    assert "SEED=6" in out.stdout + out.stderr


def test_a_run_that_already_has_an_outcome_is_not_overwritten(tmp_path):
    d = write_run(tmp_path, finalized=True)
    out = run_cli(d)
    assert out.returncode == 1
    assert "already has an outcome" in out.stdout + out.stderr
    kept = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["outcome"]
    assert kept["n_proved"] == 99


def test_force_overwrites_only_when_asked(tmp_path):
    d = write_run(tmp_path, finalized=True, proved=33)
    assert run_cli(d, "--force").returncode == 0
    written = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["outcome"]
    assert written["n_proved"] == 33


def test_more_attempts_than_problems_is_refused(tmp_path):
    d = write_run(tmp_path, n=190, n_problems=186)
    out = run_cli(d)
    assert out.returncode == 1
    assert "shared this directory" in out.stdout + out.stderr


def test_a_duplicated_problem_is_refused(tmp_path):
    # A resume gone wrong. Counting one problem twice changes the rate in both directions at once.
    rows = [{"problem_id": "ex_1", "proved": True, "status": "proved"},
            {"problem_id": "ex_1", "proved": False, "status": "exhausted"}]
    d = write_run(tmp_path, rows=rows, n_problems=2)
    out = run_cli(d)
    assert out.returncode == 1
    assert "more than once" in out.stdout + out.stderr


def test_a_truncated_final_row_is_refused_rather_than_dropped(tmp_path):
    # Exactly what a killed process leaves, and the one problem whose outcome is unknown.
    # Dropping it would report 185 of 186 as if the benchmark were 185 long.
    d = write_run(tmp_path, n=186)
    with (d / "attempts.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"problem_id": "ex_186", "prov')
    out = run_cli(d)
    assert out.returncode == 1
    said = out.stdout + out.stderr
    assert "could not be parsed" in said
    assert "repair_attempts.py" in said, "the refusal must lead somewhere"


def test_read_attempts_ignores_blank_lines_but_not_broken_ones(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"problem_id": "a"}\n\n{"problem_id": "b"}\n', encoding="utf-8")
    assert len(read_attempts(p)) == 2


def test_a_run_with_no_attempts_file_is_refused(tmp_path):
    d = write_run(tmp_path)
    (d / "attempts.jsonl").unlink()
    out = run_cli(d)
    assert out.returncode == 1
    assert "recorded nothing" in out.stdout + out.stderr


# --- the dry run ---------------------------------------------------------------------------------

def test_dry_run_reports_without_writing(tmp_path):
    d = write_run(tmp_path)
    out = run_cli(d, "--dry-run")
    assert out.returncode == 0
    assert "33/186" in out.stdout
    assert json.loads((d / "manifest.json").read_text(encoding="utf-8"))["outcome"] is None


def test_it_points_at_verification_because_an_outcome_is_not_evidence(tmp_path):
    d = write_run(tmp_path)
    out = run_cli(d)
    assert "verify_proofs.sbatch" in out.stdout


# --- the dead end this replaces ------------------------------------------------------------------

def test_resuming_a_completed_but_unfinalized_run_points_at_the_repair():
    """`--resume` cannot fix a run that finished everything and died before finalize().

    It skips problems already in `attempts.jsonl`, finds nothing left, and returns — leaving the
    manifest with no outcome however many times it is resubmitted, and a GPU allocation spent
    discovering that. The branch must say so and name the script that does work.
    """
    src = (REPO / "scripts" / "prove_benchmark.py").read_text(encoding="utf-8")
    block = src[src.index("if not selected:"):src.index("else:\n        manifest")]
    assert "manifest.outcome is None" in block, (
        "the two cases are not distinguished: a no-op resume of a finished run and a resume of a "
        "run whose 186 results are on disk but invisible to every table"
    )
    assert "scripts/finalize_run.py" in block
    assert "SystemExit(1)" in block, "an invisible run must not exit 0 and look handled"
