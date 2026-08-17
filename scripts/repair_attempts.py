#!/usr/bin/env python3
"""Quarantine unreadable rows from a run's `attempts.jsonl` so it can be resumed.

    python scripts/repair_attempts.py results/logs/<run_id>

## The failure this repairs

`attempts.jsonl` is appended with `O_APPEND`, flushed and fsynced per row, which makes a run
durable against a SLURM kill. It does **not** make it durable against NFS: an append is only atomic
there for small writes, and `results/logs` is on NFS. A big enough record gets torn, and the file
then has a partial row in the middle of otherwise perfect data.

Measured: ProofNet / sv / seed 6 of the pass@8 sweep ran all 186 problems in 4 h 59 m, printed every
one of them, and then died in its own reporting block — `json.loads` over the file it had just
written — on a truncated 118,673-byte row at line 55. Every proof was on disk; the job exited 1.

## What this does, and what it refuses to guess

The bad rows are dropped and the original file is kept **verbatim** as
`attempts.jsonl.corrupt-<UTC>`, because a torn row is the only evidence of how it tore and this
script is not the right place to decide it is uninteresting.

The problems in those rows are then simply *absent*, and a resume re-attempts exactly them — usually
minutes of GPU time. That is the honest repair: an unreadable row is an **unknown** outcome, not a
failed one, and inventing either verdict would put a fabricated result into a published rate.

It will not touch a file it cannot improve, and it will not run twice: a second invocation finds
nothing corrupt and says so.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.utils.io import read_jsonl_tolerant, write_jsonl  # noqa: E402
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: Shown for each torn row. Enough to identify the problem and the field that grew, without dumping
#: 118 KB into a terminal.
PREVIEW = 220


def preview_line(path: Path, lineno: int, limit: int = PREVIEW) -> str:
    """The first `limit` bytes of one line, for identifying which problem was lost and why."""
    with open(path, "rb") as f:
        for n, raw in enumerate(f, 1):
            if n == lineno:
                return raw[:limit].decode("utf-8", errors="replace")
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="results/logs/<run_id> whose attempts.jsonl is torn")
    ap.add_argument("--dry-run", action="store_true", help="report only; change nothing")
    args = ap.parse_args()
    ensure_utf8_output()

    attempts = args.run / "attempts.jsonl"
    if not attempts.exists():
        raise SystemExit(f"no attempts.jsonl at {attempts}")

    rows, bad = read_jsonl_tolerant(attempts)
    manifest_path = args.run / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})
    expected = (manifest.get("config") or {}).get("n_problems")

    print(f"run       : {args.run.name}")
    print(f"readable  : {len(rows)} row(s)")
    print(f"expected  : {expected} problem(s)" if expected else "expected  : unknown")

    if not bad:
        print("\nNothing corrupt. This file needs no repair.")
        return

    print(f"corrupt   : {len(bad)} row(s)")
    for lineno, nbytes in bad:
        print(f"  line {lineno}: {nbytes:,} bytes")
        print(f"    {preview_line(attempts, lineno)}...")

    recovered = {r.get("problem_id") for r in rows}
    print(f"\nAfter repair the file holds {len(rows)} rows. The problem(s) in the torn row(s) are "
          f"absent and a resume will re-attempt them.")
    if expected:
        print(f"Missing from the benchmark: {expected - len(recovered)} problem(s).")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    backup = attempts.with_suffix(f".jsonl.corrupt-{datetime.now(UTC):%Y%m%dT%H%M%S}")
    shutil.copy2(attempts, backup)
    # Written to a sibling then moved: a crash midway through the rewrite must not be able to leave
    # this file shorter than the backup it was just copied from.
    tmp = attempts.with_suffix(".jsonl.repairing")
    write_jsonl(tmp, rows)
    tmp.replace(attempts)

    print(f"\noriginal kept : {backup.name}")
    print(f"rewrote       : {attempts}  ({len(rows)} rows)")
    seed = manifest.get("seed")
    print("\nNow resume, which will re-attempt only the missing problem(s):")
    print(f"  BENCHMARK={(manifest.get('config') or {}).get('benchmark')} "
          f"ARM={(manifest.get('config') or {}).get('arm')} SEED={seed} RESUME={args.run} \\")
    print("      ... sbatch -p gpuA -A gpu-fse-ugpgt01 -G 1 slurm/prove_benchmark_llm.sbatch")


if __name__ == "__main__":
    main()
