"""Verify that benchmark problems actually elaborate in Lean, and yield a usable root proof state.

**Why this check earns its place.** A problem that fails to elaborate is scored as unproved, so a
loader bug — a dropped `open`, a mangled `variable` block, a declaration cut at the wrong place —
looks exactly like "the prover could not solve it". It would depress every arm equally, hide
inside a plausible-looking pass rate, and never raise an error. This is the check that separates
"our prover failed" from "we never gave Lean a valid problem".

Run it on a sample before any real evaluation, and treat a non-trivial failure count as a loader
bug until proven otherwise.

Usage::

    python scripts/check_benchmark_elaboration.py --benchmark fate_m --limit 10 \\
        --data-root ~/data/benchmarks/REAL-Prover/Realprover/data \\
        --project-dir ~/lean/prooflens_mathlib
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from prooflens_prover.data.benchmarks import load_benchmark
from prooflens_prover.utils.logging import get_logger

log = get_logger("check_elab")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="fate_m")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    problems = load_benchmark(args.benchmark, Path(args.data_root).expanduser())
    sample = problems[: args.limit]
    print(f"\n=== elaboration check: {args.benchmark}, {len(sample)}/{len(problems)} problems ===")

    from prooflens_prover.lean.leaninteract_backend import LeanInteractBackend

    t0 = time.perf_counter()
    backend = LeanInteractBackend(
        project_dir=Path(args.project_dir).expanduser(), tactic_timeout=args.timeout
    )
    print(f"backend ready (lean {backend.lean_version}) in {time.perf_counter() - t0:.1f}s\n")

    rows: list[dict] = []
    n_ok = 0
    try:
        for i, p in enumerate(sample, 1):
            t = time.perf_counter()
            try:
                state = backend.start_theorem(p.statement, header=p.imports)
                ok = bool(state.goals) and not state.is_solved
                rows.append({
                    "id": p.id, "ok": ok, "n_goals": len(state.goals),
                    "goal_preview": state.pp[:120], "secs": round(time.perf_counter() - t, 2),
                })
                n_ok += ok
                mark = "ok  " if ok else "EMPTY"
                print(f"[{i:3d}/{len(sample)}] [{mark}] {p.id:<20} "
                      f"{time.perf_counter() - t:5.2f}s  {state.pp[:80]!r}")
            except Exception as e:  # noqa: BLE001 — collect every failure, don't stop at the first
                rows.append({"id": p.id, "ok": False,
                             "error": f"{type(e).__name__}: {e}"[:300],
                             "secs": round(time.perf_counter() - t, 2)})
                print(f"[{i:3d}/{len(sample)}] [FAIL] {p.id:<20} "
                      f"{type(e).__name__}: {str(e)[:140]}")
    finally:
        backend.close()

    print(f"\n=== {n_ok}/{len(sample)} elaborated to a usable root state "
          f"({100 * n_ok / max(len(sample), 1):.0f}%) ===")
    if n_ok < len(sample):
        print("Any failure here is a LOADER bug until proven otherwise — a problem that does not")
        print("elaborate is silently scored as unproved and will depress every arm equally.")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"benchmark": args.benchmark, "n_ok": n_ok, "n": len(sample),
                        "rows": rows}, indent=2), encoding="utf-8")
        print(f"report -> {args.json_out}")

    return 0 if n_ok == len(sample) else 1


if __name__ == "__main__":
    sys.exit(main())
