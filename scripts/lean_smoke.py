"""Tier-0 gate: prove that tactic-level Lean 4 interaction works, end to end, on this machine.

**This is the check that failed in the predecessor project** (`prooflens/results/phase_logs/
phase20.md`): LeanDojo's `Dojo` launched and read the goal, but `run_tac` died with
`DojoCrashError: Unexpected EOF`, and the whole live-prover strand was abandoned for an offline
fallback. Nothing downstream in this project is worth building until this script prints PASSED,
both locally and on CSF3.

It exercises exactly the operations best-first search depends on, and nothing else:

1. elaborate a Mathlib environment                        (`import Mathlib`)
2. open a theorem and read its root proof state           (`start_theorem`)
3. apply a tactic that makes progress, keeping the state  (`run_tactic` -> PROGRESS)
4. apply a tactic that CLOSES the goal                    (`run_tactic` -> PROVED)
5. apply a tactic that FAILS, without killing the REPL    (`run_tactic` -> ERROR)
6. reject a `sorry` tactic before it ever reaches Lean    (`run_tactic` -> REJECTED)

Steps 5 and 6 matter as much as 3 and 4. Search spends most of its time on tactics that do not
work, so a harness that crashes or mis-scores on failure is useless; and step 6 checks the
guard that stops a vacuous proof being counted as a real one.

Usage (inside WSL / on a Linux node, with the venv active)::

    python scripts/lean_smoke.py --project-dir ~/lean-practice/my_proofs
    python scripts/lean_smoke.py --lean-version v4.31.0     # builds a temp Mathlib project (slow)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from prooflens_prover.lean import Outcome
from prooflens_prover.utils.logging import get_logger

log = get_logger("lean_smoke")

# `Nat.succ_ne_zero`-style trivialities elaborate fast and their proofs are stable across Mathlib
# versions, so this gate tests the harness rather than the Lean library.
THEOREM = "theorem prooflens_smoke (a b : Nat) : a + b = b + a"
TACTIC_PROGRESS_SIMPLE = "cases a"                  # splits Nat -> 2 open goals (a live node)
TACTIC_CLOSING = "exact Nat.add_comm a b"
TACTIC_FAILING = "exact Nat.add_comm a b c d"      # arity error -> Lean reports an error
TACTIC_CHEATING = "sorry"                           # must be refused before reaching Lean


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--project-dir", default=None,
                    help="pre-built Lean project that requires Mathlib (strongly preferred)")
    ap.add_argument("--lean-version", default=None,
                    help="build a temp Mathlib project at this Lean version (slow, first run only)")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-tactic timeout (s)")
    ap.add_argument("--memory-limit-mb", type=int, default=None,
                    help="REPL VIRTUAL address-space cap; omit (default) to leave unbounded. "
                         "Values below 32768 make Lean fail to start — use SLURM --mem instead.")
    ap.add_argument("--json-out", default=None, help="write the structured result here")
    args = ap.parse_args()

    if not args.project_dir and not args.lean_version:
        print("ERROR: give --project-dir (preferred) or --lean-version", file=sys.stderr)
        return 2

    report: dict[str, object] = {"steps": []}
    t_start = time.perf_counter()

    def record(step: str, ok: bool, **extra: object) -> None:
        report["steps"].append({"step": step, "ok": ok, **extra})
        mark = "ok  " if ok else "FAIL"
        detail = " ".join(f"{k}={v!r}" for k, v in extra.items() if k != "traceback")
        print(f"  [{mark}] {step}  {detail}")

    try:
        from prooflens_prover.lean.leaninteract_backend import LeanInteractBackend
    except ImportError as e:
        print(f"\nSMOKE FAILED: cannot import the Lean backend ({e}).")
        print("  Install it:  uv pip install 'lean-interact'")
        return 2

    print("\n=== ProofLens-Prover Lean smoke test ===")
    print(f"project_dir  : {args.project_dir}")
    print(f"lean_version : {args.lean_version}")
    print("Building the backend (first `import Mathlib` takes ~30-90s and ~4GB RAM) ...\n")

    backend = None
    try:
        t0 = time.perf_counter()
        backend = LeanInteractBackend(
            project_dir=args.project_dir,
            lean_version=args.lean_version,
            tactic_timeout=args.timeout,
            memory_limit_mb=args.memory_limit_mb,
        )
        report["lean_version"] = backend.lean_version
        record("build backend", True, lean=backend.lean_version,
               secs=round(time.perf_counter() - t0, 1))

        # -- 1/2: open the theorem, read the root state ---------------------------------------
        t0 = time.perf_counter()
        root = backend.start_theorem(THEOREM)
        ok = bool(root.goals) and not root.is_solved
        record("open theorem + read root state", ok,
               goal=root.pp[:90], secs=round(time.perf_counter() - t0, 1))
        if not ok:
            raise RuntimeError("root state has no goals — the statement was already closed")

        # -- 3: a tactic that makes progress without closing ----------------------------------
        # `cases a` splits Nat into zero/succ, leaving 2 open goals: a live, non-terminal node,
        # which is what every interior node of the search tree looks like.
        r = backend.run_tactic(root, TACTIC_PROGRESS_SIMPLE)
        record("progress tactic keeps a live state", r.outcome is Outcome.PROGRESS,
               outcome=r.outcome.value, n_goals=len(r.state.goals) if r.state else 0)

        # -- 4: THE critical one — a tactic that actually closes the goal ----------------------
        t0 = time.perf_counter()
        r_close = backend.run_tactic(root, TACTIC_CLOSING)
        closed = r_close.outcome is Outcome.PROVED
        record("closing tactic -> PROVED", closed,
               outcome=r_close.outcome.value, secs=round(time.perf_counter() - t0, 2),
               error=r_close.error)

        # -- 5: a failing tactic must be an ERROR result, not a crash -------------------------
        r_fail = backend.run_tactic(root, TACTIC_FAILING)
        failed_cleanly = r_fail.outcome is Outcome.ERROR
        record("failing tactic -> ERROR (no crash)", failed_cleanly,
               outcome=r_fail.outcome.value, error=(r_fail.error or "")[:70])

        # -- 5b: the REPL must still be usable after that failure -----------------------------
        r_after = backend.run_tactic(root, TACTIC_CLOSING)
        survived = r_after.outcome is Outcome.PROVED
        record("REPL still usable after a failure", survived, outcome=r_after.outcome.value)

        # -- 6: the cheat guard ---------------------------------------------------------------
        r_cheat = backend.run_tactic(root, TACTIC_CHEATING)
        guarded = r_cheat.outcome is Outcome.REJECTED
        record("`sorry` rejected before execution", guarded,
               outcome=r_cheat.outcome.value, reason=r_cheat.error)

        all_ok = closed and failed_cleanly and survived and guarded
        report["passed"] = all_ok
        report["elapsed_s"] = round(time.perf_counter() - t_start, 1)

        print()
        if all_ok:
            print("SMOKE PASSED — tactic-level Lean interaction works end to end.")
            print("  A tactic executed against live Mathlib, closed a goal, failed cleanly,")
            print("  and the cheat guard held. This is the gate that blocked the previous project.")
        else:
            print("SMOKE FAILED — see the FAIL lines above.")
        print(f"  total {report['elapsed_s']}s, lean {report.get('lean_version')}")
        return 0 if all_ok else 1

    except Exception as e:  # noqa: BLE001 - a gate: report ANY failure with its type, never crash
        import traceback

        report["passed"] = False
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"\nSMOKE FAILED: {type(e).__name__}: {e}\n")
        traceback.print_exc()
        print("\n  Common causes: elan/Lean not on PATH; the project's .lake is not built")
        print("  (run `lake exe cache get` in it); not enough RAM for `import Mathlib` (~4GB).")
        return 1
    finally:
        if backend is not None:
            backend.close()
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
            print(f"  report -> {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
