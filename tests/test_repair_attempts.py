"""A torn row must not be able to discard a run, or to invent a result.

The measured failure: `attempts.jsonl` is appended with `O_APPEND`, flushed and fsynced per row —
durable against a SLURM kill, but not against NFS, where an append that large is not atomic and
`results/logs` lives. ProofNet / sv / seed 6 of the pass@8 sweep ran all 186 problems in 4 h 59 m,
printed every one, and then raised `JSONDecodeError` in its own reporting block reading the file it
had just written: line 55 was a truncated 118,673-byte record. Every proof was durable on disk and
the job exited 1.

Three separate things have to hold, and each has its own section below:

* the reader **survives** it, so a finished run is never thrown away again;
* the run **says** its denominator is short, because a missing problem reads downstream as unsolved
  and would bias the arm downward silently;
* the repair **re-attempts** the unknown problem rather than deciding it either way.

Hermetic: no GPU, no Lean, no model.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.utils.io import read_jsonl_tolerant, write_jsonl  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def write_torn(tmp_path, n=186, torn_at=55, torn_bytes=118_673, seed=6):
    """A run directory whose attempts file is torn mid-file, as seed 6's was."""
    d = tmp_path / "proofnet_test_sv_vllm_x_6b677cc7"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({
        "run_id": d.name, "seed": seed,
        "config": {"benchmark": "proofnet_test", "arm": "sv", "policy_kind": "vllm",
                   "n_problems": n},
        "outcome": None,
    }), encoding="utf-8")

    lines = []
    for i in range(n):
        if i == torn_at - 1:
            # A real oversized record, cut mid-string exactly the way an interrupted append cuts it.
            row = json.dumps({"problem_id": f"ex_{i}", "arm": "sv", "proved": False,
                              "status": "exhausted", "error": "x" * torn_bytes})
            lines.append(row[:torn_bytes])
        else:
            lines.append(json.dumps({"problem_id": f"ex_{i}", "arm": "sv", "proved": i < 35,
                                     "status": "proved" if i < 35 else "exhausted"}))
    (d / "attempts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def run_cli(script, *args):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *map(str, args)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


# --- the reader survives it ----------------------------------------------------------------------

def test_the_tolerant_reader_returns_every_good_row_and_locates_the_bad_one(tmp_path):
    d = write_torn(tmp_path, n=186, torn_at=55)
    rows, bad = read_jsonl_tolerant(d / "attempts.jsonl")
    assert len(rows) == 185
    assert [n for n, _ in bad] == [55]


def test_the_reported_length_identifies_an_oversized_row(tmp_path):
    # The size is the diagnosis: 118 KB through a non-atomic NFS append is why it tore.
    d = write_torn(tmp_path, torn_at=55, torn_bytes=118_673)
    _, bad = read_jsonl_tolerant(d / "attempts.jsonl")
    assert bad[0][1] > 100_000


def test_a_torn_row_is_not_confused_with_a_blank_line(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"problem_id": "a"}\n\n{"problem_id": "b"}\n', encoding="utf-8")
    rows, bad = read_jsonl_tolerant(p)
    assert len(rows) == 2 and bad == []


def test_a_clean_file_reports_nothing_bad(tmp_path):
    p = tmp_path / "a.jsonl"
    write_jsonl(p, [{"problem_id": str(i)} for i in range(10)])
    rows, bad = read_jsonl_tolerant(p)
    assert len(rows) == 10 and bad == []


def test_the_aggregation_no_longer_dies_on_its_own_output():
    """The single line that discarded a 4 h 59 m run.

    `rows = [json.loads(line) for line in ...]` ran *after* every proof was durable on disk, and
    after the final progress line was printed. There is no version of that trade worth keeping.
    """
    src = (REPO / "scripts" / "prove_benchmark.py").read_text(encoding="utf-8")
    tail = src[src.index("elapsed = time.perf_counter() - t_start"):]
    assert "read_jsonl_tolerant" in tail
    assert "json.loads(line)" not in tail, (
        "the totals are still computed with a strict per-line parse; one torn row throws away the "
        "whole run at the finish line"
    )


def test_resume_survives_a_torn_row_rather_than_becoming_unresumable():
    # Strict parsing here is worse than at the end: it makes the run permanently unresumable, which
    # is the exact opposite of what an fsync per attempt buys.
    src = (REPO / "scripts" / "prove_benchmark.py").read_text(encoding="utf-8")
    block = src[src.index("if args.resume is not None:"):src.index("n_done_before = len(done)")]
    assert "read_jsonl_tolerant" in block
    assert "json.loads(line)" not in block


# --- the run says its denominator is short -------------------------------------------------------

def test_the_outcome_records_the_unreadable_rows():
    # A missing problem reads downstream as unsolved, biasing the arm downward. If the only trace is
    # an n_problems that quietly differs from the benchmark size, nobody will notice.
    src = (REPO / "scripts" / "prove_benchmark.py").read_text(encoding="utf-8")
    final = src[src.index("manifest.finalize("):src.index("attempts    :")]
    assert "n_unreadable_rows=len(unreadable)" in final
    assert "unreadable_row_lines" in final


def test_the_summary_says_the_rate_is_not_over_the_whole_benchmark():
    src = (REPO / "scripts" / "prove_benchmark.py").read_text(encoding="utf-8")
    assert "CORRUPT" in src
    assert "repair_attempts.py" in src


# --- the repair re-attempts rather than deciding --------------------------------------------------

def test_the_repair_drops_only_the_torn_row(tmp_path):
    d = write_torn(tmp_path, n=186, torn_at=55)
    out = run_cli("repair_attempts.py", d)
    assert out.returncode == 0, out.stdout + out.stderr
    rows, bad = read_jsonl_tolerant(d / "attempts.jsonl")
    assert len(rows) == 185 and bad == []


def test_the_original_file_is_kept_verbatim(tmp_path):
    # A torn row is the only evidence of how it tore. This script is not the place to decide that is
    # uninteresting.
    d = write_torn(tmp_path)
    before = (d / "attempts.jsonl").read_bytes()
    run_cli("repair_attempts.py", d)
    backups = list(d.glob("attempts.jsonl.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == before


def test_the_dropped_problem_is_absent_so_a_resume_re_attempts_it(tmp_path):
    d = write_torn(tmp_path, n=186, torn_at=55)
    run_cli("repair_attempts.py", d)
    rows, _ = read_jsonl_tolerant(d / "attempts.jsonl")
    assert "ex_54" not in {r["problem_id"] for r in rows}, (
        "the torn row's problem must not survive as a recorded attempt: its outcome is unknown, "
        "and an unknown recorded either way is a fabricated result in a published rate"
    )


def test_it_reports_the_missing_count_against_the_benchmark(tmp_path):
    d = write_torn(tmp_path, n=186, torn_at=55)
    out = run_cli("repair_attempts.py", d)
    assert "185" in out.stdout and "186" in out.stdout


def test_it_previews_the_torn_row_so_the_cause_is_identifiable(tmp_path):
    d = write_torn(tmp_path, torn_at=55)
    out = run_cli("repair_attempts.py", d, "--dry-run")
    assert "ex_54" in out.stdout, "the preview must name the problem that was lost"
    # The size is the diagnosis, so it has to be on screen: a six-figure row through a non-atomic
    # NFS append is why it tore, and a record that big serves no analytical purpose either.
    assert "118,67" in out.stdout, out.stdout


def test_it_prints_the_resume_command_with_the_original_seed(tmp_path):
    d = write_torn(tmp_path, seed=6)
    out = run_cli("repair_attempts.py", d)
    assert "SEED=6" in out.stdout
    assert "RESUME=" in out.stdout


def test_a_dry_run_changes_nothing(tmp_path):
    d = write_torn(tmp_path)
    before = (d / "attempts.jsonl").read_bytes()
    out = run_cli("repair_attempts.py", d, "--dry-run")
    assert out.returncode == 0
    assert (d / "attempts.jsonl").read_bytes() == before
    assert not list(d.glob("attempts.jsonl.corrupt-*"))


def test_a_clean_run_is_left_alone(tmp_path):
    d = write_torn(tmp_path)
    run_cli("repair_attempts.py", d)
    before = (d / "attempts.jsonl").read_bytes()
    out = run_cli("repair_attempts.py", d)
    assert "Nothing corrupt" in out.stdout
    assert (d / "attempts.jsonl").read_bytes() == before
    assert len(list(d.glob("attempts.jsonl.corrupt-*"))) == 1, "a second run must not re-quarantine"


def test_finalize_still_refuses_a_torn_run_and_names_the_repair(tmp_path):
    # finalize_run.py cannot re-attempt anything, so refusing remains right — but the refusal has to
    # lead somewhere.
    d = write_torn(tmp_path)
    out = run_cli("finalize_run.py", d)
    assert out.returncode == 1
    assert "could not be parsed" in out.stdout + out.stderr
    assert "repair_attempts.py" in out.stdout + out.stderr


def test_repair_then_resume_then_finalize_is_a_complete_path(tmp_path):
    """End to end on the shape of seed 6: repair leaves a resumable run one problem short."""
    d = write_torn(tmp_path, n=186, torn_at=55)
    assert run_cli("repair_attempts.py", d).returncode == 0
    # Still short of the benchmark, so finalize must refuse and point at resume rather than publish
    # a rate over 185.
    out = run_cli("finalize_run.py", d)
    assert out.returncode == 1
    assert "185 of 186" in out.stdout + out.stderr
    assert "SEED=6" in out.stdout + out.stderr


# --- the actual root cause: str.splitlines() is not a JSONL splitter -----------------------------

#: The characters that can actually reach a JSONL file raw *and* break `str.splitlines()`.
#:
#: `splitlines()` also breaks on the C0 controls, but JSON requires escaping everything below
#: U+0020, so those never survive into the file and cannot cause this. These three sit above that
#: boundary: JSON leaves them alone, and `splitlines()` treats each as a line ending.
UNICODE_BREAKS = ("\u2028", "\u2029", "\u0085")

#: Escaped by JSON, so they cannot reach the file raw. Listed to narrow the diagnosis: if one of
#: these could get through, the bug would be in the writer. None can, so it is only ever a reader
#: that splits on more than "\n".
JSON_ESCAPED = ("\v", "\f", "\x1c", "\x1d", "\x1e")


@pytest.mark.parametrize("ch", UNICODE_BREAKS, ids=["U+2028", "U+2029", "U+0085"])
def test_splitlines_tears_a_valid_record_and_the_tolerant_reader_does_not(tmp_path, ch):
    """The real cause of the "corrupt line 55", and it was never corruption at all.

    JSON does not require these to be escaped inside a string and orjson writes them through
    verbatim, while `str.splitlines()` treats each as a line ending. Lean is a unicode language
    whose goal states and error text can contain them. So one perfectly valid 118 KB record became
    two pseudo-lines, the first an unterminated string, and the run died reading back a file that
    was, and always had been, 186 good rows.
    """
    p = tmp_path / "a.jsonl"
    write_jsonl(p, [{"problem_id": "ex_54", "error": f"goal{ch}state"}, {"problem_id": "ex_55"}])
    text = p.read_text(encoding="utf-8")
    assert len(text.splitlines()) > 2, f"{ch!r} no longer needs guarding"
    rows, bad = read_jsonl_tolerant(p)
    assert bad == [], f"{ch!r} still tears a record"
    assert [r["problem_id"] for r in rows] == ["ex_54", "ex_55"]


@pytest.mark.parametrize("ch", JSON_ESCAPED)
def test_the_c0_controls_cannot_reach_the_file_so_the_writer_is_not_at_fault(tmp_path, ch):
    p = tmp_path / "a.jsonl"
    write_jsonl(p, [{"problem_id": "a", "error": f"x{ch}y"}])
    assert ch not in p.read_text(encoding="utf-8")


def test_the_strict_reader_is_also_immune_because_analysis_must_not_be_a_coin_flip(tmp_path):
    # `read_jsonl` is what every analysis script uses. If it split differently from the writer, a
    # reported number would depend on which characters a Lean error happened to contain.
    from prooflens_prover.utils.io import read_jsonl

    p = tmp_path / "a.jsonl"
    write_jsonl(p, [{"problem_id": "a", "error": "x\u2028y"}, {"problem_id": "b"}])
    assert [r["problem_id"] for r in read_jsonl(p)] == ["a", "b"]


@pytest.mark.parametrize("path", sorted(
    list((REPO / "scripts").glob("*.py")) + list((REPO / "src").rglob("*.py")), key=str))
def test_no_source_file_reads_attempts_with_splitlines(path):
    """Nine call sites had this bug at once, including `eval/draws.py`, which `passk_union.py` --
    the headline reporter -- is built on. Every one of them would have crashed on this run."""
    joined = " ".join(path.read_text(encoding="utf-8").split())
    assert 'attempts.jsonl").read_text(encoding="utf-8").splitlines()' not in joined, (
        f"{path.name} splits a JSONL file with str.splitlines(), which breaks on U+2028/U+2029/"
        "U+0085 -- characters JSON leaves unescaped and Lean output contains. Use read_jsonl()."
    )
