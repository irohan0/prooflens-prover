#!/usr/bin/env python
"""Assemble Table 1 (Track A') from every run in `results/logs`, with significance.

    python scripts/build_table1.py

Discovers runs rather than taking run ids on the command line: pasting run ids by hand is how a
`none` row gets compared against the wrong benchmark's `li` row, and the resulting table looks
entirely plausible. For each `(benchmark, arm)` the most recent **finalised** run wins — a run whose
manifest has no `outcome` did not complete and is ignored.

Prints a markdown table plus the per-benchmark paired analysis, and writes
`results/tables/table1.md` and `table1.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prooflens_prover.eval.compare import Arm, compare, format_report

#: Published reference numbers, for context in the table. These are NOT our results and are not
#: comparable to a model-free policy — see the caption emitted below.
PUBLISHED: dict[str, dict[str, str]] = {
    "fate_m": {
        "REAL-Prover-v1 (7B, LeanSearch-PS)": "56.7",
        "REAL-Prover-v1 (no retrieval)": "44.7",
    },
    "proofnet_test": {
        "REAL-Prover-v1 (7B, LeanSearch-PS)": "23.7",
        "REAL-Prover-v1 (no retrieval)": "22.6",
        "ReProver (<1B)": "13.8",
    },
    "minif2f_test": {"REAL-Prover-v1 (7B)": "54.1"},
}

BENCHMARK_ORDER = ["fate_m", "proofnet_test", "minif2f_test", "putnambench"]


def discover(results_root: Path) -> dict[tuple[str, str], Path]:
    """Most recent finalised run per (benchmark, arm)."""
    best: dict[tuple[str, str], tuple[str, Path]] = {}
    for d in sorted(results_root.iterdir()):
        mf = d / "manifest.json"
        if not d.is_dir() or not mf.exists() or not (d / "attempts.jsonl").exists():
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        if not m.get("outcome"):
            continue                      # never finished; its counts would be partial
        cfg = m.get("config", {})
        key = (cfg.get("benchmark", "?"), cfg.get("arm", "?"))
        started = m.get("started_utc", "")
        if key not in best or started > best[key][0]:
            best[key] = (started, d)
    return {k: v[1] for k, v in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=Path("results/logs"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = discover(args.results_root)
    if not runs:
        raise SystemExit(f"no finalised runs under {args.results_root}")

    benchmarks = sorted(
        {b for b, _ in runs},
        key=lambda b: BENCHMARK_ORDER.index(b) if b in BENCHMARK_ORDER else 99,
    )
    print("discovered runs:")
    for (b, a), d in sorted(runs.items()):
        print(f"  {b:<16} {a:<6} {d.name}")
    print()

    #: Each comparison is (baseline arm, treatment arm). `li vs none` measures whether retrieval
    #: helps at all; `li vs sv` is the architecture claim the dissertation is named after, and it
    #: is the one that needs the SV arm to exist.
    PAIRS = [("none", "li"), ("sv", "li")]

    rows: list[str] = []
    comparisons: dict[str, dict] = {}
    for b in benchmarks:
        cells = []
        for arm in ("none", "sv", "li"):
            d = runs.get((b, arm))
            if d is None:
                cells.append("—")
                continue
            a = Arm.load(d)
            n = len(a.proved)
            cells.append(f"{a.n_proved}/{n} ({100 * a.n_proved / n:.1f}%)")

        deltas = []
        for base_arm, treat_arm in PAIRS:
            if (b, base_arm) not in runs or (b, treat_arm) not in runs:
                deltas.append("—")
                continue
            r = compare(
                Arm.load(runs[(b, base_arm)]), Arm.load(runs[(b, treat_arm)]),
                args.n_boot, args.n_perm, args.seed,
            )
            comparisons[f"{b}:{treat_arm}_vs_{base_arm}"] = r
            star = "**" if r["significant"] else ""
            deltas.append(
                f"{star}{r['delta_problems']:+d}{star} (p={r['p_mcnemar_exact']:.4f})"
            )
            print(format_report(r))
            print()
        rows.append(f"| {b} | {cells[0]} | {cells[1]} | {cells[2]} | {deltas[0]} | {deltas[1]} |")

    lines = [
        "# Table 1 — retrieval architecture, model-free policy (Track A')",
        "",
        "| benchmark | none (floor) | ProofLens-SV | ProofLens-LI | Δ vs none | Δ vs SV |",
        "|---|--:|--:|--:|--:|--:|",
        *rows,
        "",
        "**Δ vs none** is problems gained over the no-retrieval control; **Δ vs SV** is the",
        "architecture comparison — same premise corpus, same search budget, same checkpoint",
        "training protocol, only multi-vector against single-vector. Bold marks a result where the",
        "bootstrap CI and the sign-flip permutation test agree.",
        "",
        "Per-comparison detail — the exact problems each arm won, the displacement check, and",
        "premise-needed rates — is in `table1.json`.",
        "",
        "## Published reference numbers (context, not comparisons)",
        "",
        "These come from systems built on 7B-class fine-tuned language models. The rows above",
        "use a **model-free** tactic policy with no language model at all, so the rates are not",
        "comparable. What Table 1 measures is the *effect of retrieval* with the generator held",
        "fixed; the comparable-to-published experiment is Tier 1 (frozen REAL-Prover-v1).",
        "",
    ]
    for b in benchmarks:
        if b in PUBLISHED:
            lines.append(f"* **{b}** — " + "; ".join(
                f"{k}: {v}" for k, v in PUBLISHED[b].items()
            ))
    text = "\n".join(lines) + "\n"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "table1.md").write_text(text, encoding="utf-8")
    (args.out_dir / "table1.json").write_text(
        json.dumps(
            {"runs": {f"{b}/{a}": d.name for (b, a), d in runs.items()},
             "comparisons": comparisons},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(text)
    print(f"written: {args.out_dir / 'table1.md'} and table1.json")


if __name__ == "__main__":
    main()
