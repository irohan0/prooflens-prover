#!/usr/bin/env python
"""Independently re-check every proof a run claims to have found.

    python scripts/verify_proofs.py --run results/logs/<run_id> \
        --data-root ~/data/benchmarks/REAL-Prover/Realprover/data \
        --lean-project ~/lean/mathlib_v4160

## Why a separate script and not an assertion inside the search

The search decides PROVED from the REPL's response to the *last tactic it applied*, using state
handles it has been threading through a tree. That is a lot of bookkeeping between "Lean accepted
something" and "this theorem is proved". This script throws all of it away: it takes the
benchmark's own statement, appends the recorded tactic list, and elaborates the whole declaration
from scratch in a fresh environment.

The two paths share nothing except the cheat-token regex, so a bug in the incremental bookkeeping —
a state handle pointing at the wrong subgoal, a proof reconstructed with a missing step — is caught
here rather than becoming a headline number.

**Any `VERIFY FAIL` invalidates the run's pass rate.** Exit status is non-zero in that case so it
cannot be missed in a batch log.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Importable from a fresh clone with no install and no exported PYTHONPATH: a login-node
# `python scripts/<this>.py` must work, because that is how the analysis scripts get run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.data.benchmarks import load_benchmark
from prooflens_prover.lean.leaninteract_backend import (
    LeanInteractBackend,
    statement_with_proof,
)
from prooflens_prover.utils.io import read_jsonl
from prooflens_prover.utils.logging import ensure_utf8_output, get_logger

log = get_logger(__name__)


def main() -> None:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path, help="a results/logs/<run_id> directory")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--lean-project", type=Path, default=None)
    ap.add_argument("--lean-version", default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    benchmark = manifest["config"]["benchmark"]
    rows = list(read_jsonl(args.run / "attempts.jsonl"))
    claimed = [r for r in rows if r.get("proved")]
    log.info("%s: %d attempts, %d claimed proved", benchmark, len(rows), len(claimed))
    if not claimed:
        print("nothing claimed proved — nothing to verify")
        return

    problems = {p.id: p for p in load_benchmark(benchmark, args.data_root, check_count=False)}

    backend = LeanInteractBackend(
        project_dir=args.lean_project, lean_version=args.lean_version
    )

    n_ok = 0
    failures: list[dict] = []
    for i, row in enumerate(claimed, 1):
        pid = row["problem_id"]
        problem = problems.get(pid)
        if problem is None:
            failures.append({"problem_id": pid, "reason": "problem id not found in benchmark"})
            print(f"[{i}/{len(claimed)}] {pid:<30} VERIFY FAIL — unknown problem id", flush=True)
            continue

        proof = row.get("proof") or []
        command = statement_with_proof(problem.statement, proof)
        t0 = time.perf_counter()
        ok, errors, _ = backend.check_command(
            command, header=problem.imports, timeout=args.timeout
        )
        dt = time.perf_counter() - t0

        if ok:
            n_ok += 1
            print(f"[{i}/{len(claimed)}] {pid:<30} OK    {dt:5.1f}s  {' ; '.join(proof)}",
                  flush=True)
        else:
            failures.append({"problem_id": pid, "proof": proof, "errors": errors,
                             "command": command})
            print(f"[{i}/{len(claimed)}] {pid:<30} VERIFY FAIL  {dt:5.1f}s", flush=True)
            for e in errors[:3]:
                print(f"      {e[:300]}", flush=True)

    report = {
        "run": str(args.run),
        "benchmark": benchmark,
        "n_claimed": len(claimed),
        "n_verified": n_ok,
        "n_failed": len(failures),
        "failures": failures,
    }
    (args.run / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"=== verification: {n_ok}/{len(claimed)} independently re-checked ===")
    if failures:
        print(f"{len(failures)} FAILED — the reported pass rate for this run is not trustworthy.")
        print(f"details: {args.run / 'verification.json'}")
        sys.exit(1)
    print("all claimed proofs re-elaborate cleanly from the benchmark statement.")


if __name__ == "__main__":
    main()
