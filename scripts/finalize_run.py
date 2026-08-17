#!/usr/bin/env python3
"""Write the outcome block of a run that finished its work but died before recording it.

    python scripts/finalize_run.py results/logs/<run_id>

## Why this exists

`manifest.finalize()` is called once, after the search loop. A crash in between leaves a run whose
`attempts.jsonl` holds every problem and whose manifest has no `outcome` — and `outcome` is what
makes a run visible: `passk_union.discover` skips a run without one, `build_table1.discover` the
same. So a completed benchmark can be absent from every table while its results sit on disk.

Measured, and the reason this is a script rather than a note: ProofNet / sv / seed 6 of the pass@8
sweep recorded all 186 problems in 4 h 59 m of GPU time and exited 1 in the reporting block. Without
`outcome` it is one seed short of the eight the estimator needs, and `passk_union` refuses a
mismatched seed set rather than silently reporting pass@7 — correctly, but the fix is not to spend
another five GPU-hours reproducing a search that already succeeded.

`--resume` cannot repair it either: with nothing left to do it logs "nothing left to do" and returns
*before* finalizing, so the manifest stays incomplete however many times it is resubmitted.

## What it will and will not write

The count fields are a pure function of `attempts.jsonl` — `prove_benchmark.py` computes them from
that file too, deliberately, "so a resumed run reports the whole benchmark rather than only the part
that ran after the restart". Recomputing them here gives byte-identical numbers.

The **health** fields cannot be recovered: `elapsed_s`, the retrieval counters, and the policy and
generator statistics lived in the dead process's memory. They are written as `null` and the outcome
is stamped `finalized_post_hoc`, because a zero in `retrieval.n_queries` would assert that a run
which queried the retriever thousands of times never queried it at all. That costs this run its
`mean_candidates_per_expansion` health check; the per-problem records in `attempts.jsonl`
remain, and verification is unaffected: proofs are re-elaborated from the attempts file, not
from the outcome.

A run that is genuinely unfinished is **refused**: it needs `--resume`, and finalizing it would
publish a pass rate whose denominator is however far it happened to get.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.utils.io import read_jsonl_tolerant  # noqa: E402
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: Recorded as `null` rather than 0. Each lived only in the process that died; a zero here would be
#: a false statement about the run, not a missing one, and `draws.py` already tolerates a null.
UNRECOVERABLE = ("elapsed_s", "retrieval", "policy_stats", "generator_stats",
                 "n_stale_env_recoveries")


def read_attempts(path: Path) -> list[dict[str, Any]]:
    """Every recorded attempt, refusing a file with a row that genuinely cannot be parsed.

    Read through `read_jsonl_tolerant`, which iterates the file by `\\n` rather than by
    `str.splitlines()`. That distinction is not pedantry: `splitlines()` also breaks on U+2028,
    U+2029 and U+0085, JSON does not require those to be escaped, and orjson writes them through —
    so a Lean goal state containing one turns a single valid record into two pseudo-lines, the first
    of which is an unterminated string. It reported this exact file as corrupt at line 55 when the
    file was, and always had been, 186 perfectly good rows.

    A row that survives that and still will not parse is a real unknown outcome, and this script
    cannot invent it: it refuses and points at the quarantine, which lets a resume re-attempt
    exactly those problems.
    """
    rows, bad = read_jsonl_tolerant(path)
    if bad:
        lines = ", ".join(f"line {n} ({nbytes:,} bytes)" for n, nbytes in bad)
        raise SystemExit(
            f"{path}: {len(bad)} row(s) could not be parsed: {lines}. One problem's outcome is "
            f"therefore unknown, and this script cannot invent it. Quarantine them and resume, "
            f"which re-attempts exactly those problems:\n"
            f"  python scripts/repair_attempts.py {path.parent}"
        )
    return rows


def outcome_from(rows: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    """The outcome block, computed the way `prove_benchmark.py` computes it from the same file."""
    total = len(rows)
    proved = sum(1 for r in rows if r.get("proved"))
    return {
        "finished_utc": datetime.now(UTC).isoformat(),
        "n_problems": total,
        "n_proved": proved,
        "pass_rate": round(proved / max(total, 1), 4),
        "n_error": sum(1 for r in rows if r.get("status") == "error"),
        **dict.fromkeys(UNRECOVERABLE),
        # Loud, and inside the outcome rather than beside it: anything reading this run should be
        # able to see that its health counters are absent by construction, not by an arm difference.
        "finalized_post_hoc": {
            "by": "scripts/finalize_run.py",
            "at": datetime.now(UTC).isoformat(),
            "reason": reason,
            "note": "counts recomputed from attempts.jsonl; per-run health counters are "
                    "unrecoverable and recorded as null rather than zero",
        },
    }


def check_complete(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Refuse anything but a run that finished the work its manifest says it took on."""
    expected = (manifest.get("config") or {}).get("n_problems")
    ids = [r.get("problem_id") for r in rows]
    if (dupes := [p for p, c in Counter(ids).items() if c > 1]):
        raise SystemExit(
            f"{len(dupes)} problem(s) appear more than once in attempts.jsonl "
            f"(e.g. {dupes[0]!r}). A pass rate over duplicated attempts counts one problem twice; "
            f"this run needs inspecting, not finalizing."
        )
    if expected is None:
        raise SystemExit("manifest config has no n_problems, so completeness cannot be checked")
    if len(rows) < expected:
        raise SystemExit(
            f"this run is genuinely unfinished: {len(rows)} of {expected} problems recorded. "
            f"Finalizing it would publish a pass rate whose denominator is however far it got. "
            f"Resume it instead:\n"
            f"  SEED={manifest.get('seed')} RESUME=<this run dir> ... sbatch "
            f"slurm/prove_benchmark_llm.sbatch"
        )
    if len(rows) > expected:
        raise SystemExit(
            f"attempts.jsonl has {len(rows)} rows but the manifest expected {expected}. More "
            f"attempts than problems means two runs shared this directory."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="results/logs/<run_id> of a run that never finalized")
    ap.add_argument("--reason", default="process exited after the search loop, before finalize()")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an outcome that is already there (it will not be recoverable)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be written, write not")
    args = ap.parse_args()
    ensure_utf8_output()

    mf = args.run / "manifest.json"
    if not mf.exists():
        raise SystemExit(f"no manifest at {mf}")
    manifest = json.loads(mf.read_text(encoding="utf-8"))

    if manifest.get("outcome") and not args.force:
        raise SystemExit(
            f"{args.run.name} already has an outcome (n_proved="
            f"{manifest['outcome'].get('n_proved')}). Nothing to repair. Pass --force only if you "
            f"mean to overwrite a real outcome with one recomputed from attempts.jsonl."
        )

    attempts = args.run / "attempts.jsonl"
    if not attempts.exists():
        raise SystemExit(f"no attempts.jsonl at {attempts} — this run recorded nothing")
    rows = read_attempts(attempts)
    check_complete(manifest, rows)

    outcome = outcome_from(rows, args.reason)
    statuses = Counter(r.get("status") for r in rows)

    print(f"run       : {manifest.get('run_id')}")
    cfg = manifest.get("config") or {}
    print(f"config    : {cfg.get('benchmark')} / arm={cfg.get('arm')} / seed={manifest.get('seed')}"
          f" / {(cfg.get('search') or {}).get('max_expansions')}x"
          f"{(cfg.get('search') or {}).get('samples_per_step')}")
    print(f"proved    : {outcome['n_proved']}/{outcome['n_problems']} "
          f"({100 * outcome['pass_rate']:.1f}%)")
    print(f"statuses  : {dict(statuses.most_common())}")
    print(f"errors    : {outcome['n_error']}")
    print(f"null      : {', '.join(UNRECOVERABLE)}  (unrecoverable, not zero)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    manifest["outcome"] = outcome
    mf.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {mf}")
    print("Now verify it — an outcome is not evidence until the proofs re-elaborate:")
    print(f"  RUNS={args.run.name} sbatch -p multicore slurm/verify_proofs.sbatch")


if __name__ == "__main__":
    main()
