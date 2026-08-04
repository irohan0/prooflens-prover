#!/usr/bin/env python
"""Export a committable results tree from `results/logs`.

    python scripts/export_results.py --out results/exported

`attempts.jsonl` carries the full search trace — every tactic tried, its Lean verdict and timing.
That is the audit trail and it is large (tens of MB per run), so it is gitignored. This writes a
distilled copy that keeps everything a reported number depends on (outcome, proof, expansion and
tactic counts, timings) and drops only the per-tactic trace.

Every figure in the README must be recomputable from what this produces:
`scripts/build_table1.py --results-root results/exported/logs` reproduces the table exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

#: Kept from each attempt row. `trace` is deliberately absent — it is the bulk of the file and no
#: reported number depends on it.
KEEP = (
    "problem_id", "source", "arm", "status", "proved", "proof",
    "n_expansions", "n_tactics_tried", "n_lean_calls", "elapsed_s", "limit_hit", "error",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=Path("results/logs"))
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    ap.add_argument("--out", type=Path, default=Path("results/exported"))
    ap.add_argument("--keep-traces", action="store_true",
                    help="copy attempts.jsonl verbatim — only for a run being debugged")
    args = ap.parse_args()

    out_logs = args.out / "logs"
    out_logs.mkdir(parents=True, exist_ok=True)
    n_runs = n_rows = 0

    for d in sorted(args.results_root.iterdir()):
        mf, af = d / "manifest.json", d / "attempts.jsonl"
        if not d.is_dir() or not mf.exists() or not af.exists():
            continue
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        if not manifest.get("outcome"):
            print(f"  skip (never finalised): {d.name}")
            continue

        dest = out_logs / d.name
        dest.mkdir(exist_ok=True)
        shutil.copy2(mf, dest / "manifest.json")

        rows = [json.loads(x) for x in af.read_text(encoding="utf-8").splitlines() if x.strip()]
        if args.keep_traces:
            shutil.copy2(af, dest / "attempts.jsonl")
        else:
            (dest / "attempts.jsonl").write_text(
                "\n".join(json.dumps({k: r.get(k) for k in KEEP}) for r in rows) + "\n",
                encoding="utf-8",
            )
        n_runs += 1
        n_rows += len(rows)
        cfg = manifest.get("config", {})
        print(f"  {cfg.get('benchmark','?'):<14} {cfg.get('arm','?'):<5} "
              f"{len(rows):>4} rows  {d.name}")

    if args.tables.exists():
        shutil.copytree(args.tables, args.out / "tables", dirs_exist_ok=True)

    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"\nexported {n_runs} runs, {n_rows} attempts, {size / 1e6:.1f} MB -> {args.out}")
    print("verify with:")
    print(f"  python scripts/build_table1.py --results-root {out_logs}")


if __name__ == "__main__":
    main()
