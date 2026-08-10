#!/usr/bin/env python
"""Paired comparison of two arms, over one benchmark or pooled across several.

    python scripts/compare_arms.py \
        --baseline results/logs/fate_m_none_repertoire_... \
        --treatment results/logs/fate_m_li_repertoire_...

Writes the full result as JSON next to the treatment run (`comparison_vs_<baseline arm>.json`) so
every reported number stays traceable to the two runs it came from.

The primary analysis keeps harness errors as unproved, so the denominator matches the published
benchmark size. A sensitivity analysis excluding them is always printed too: if the two disagree,
the conclusion depends on problems that were never really attempted, and that has to be said.

## Pooling

Repeat the flags in matching order to pool benchmarks into one paired test:

    python scripts/compare_arms.py \
        --baseline results/logs/fate_m_none_vllm_...   --treatment results/logs/fate_m_li_vllm_... \
        --baseline results/logs/proofnet_none_vllm_... --treatment results/logs/proofnet_li_vllm_...

FATE-M alone gives 19 discordant pairs for the li-vs-none contrast, and exact McNemar on 19 cannot
reach p < 0.05 for any split closer than 15-4 — so a null there is partly a statement about power.
Pooling buys power without buying GPU time. Every per-benchmark row is printed alongside the pooled
figure, and a sign disagreement between them is flagged loudly, because a pooled average over
benchmarks chosen for *different* retrieval sensitivity can describe neither of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Importable from a fresh clone with no install and no exported PYTHONPATH: a login-node
# `python scripts/<this>.py` must work, because that is how the analysis scripts get run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.eval.compare import (
    Arm,
    compare,
    compare_pooled,
    format_pooled_report,
    format_report,
)
from prooflens_prover.utils.logging import ensure_utf8_output


def main() -> None:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # `append`, so one pair behaves exactly as before and several pool. Repeat in matching order.
    ap.add_argument("--baseline", required=True, type=Path, action="append",
                    help="results/logs/<run_id> (control); repeat to pool benchmarks")
    ap.add_argument("--treatment", required=True, type=Path, action="append",
                    help="results/logs/<run_id>; repeat in the same order as --baseline")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if len(args.baseline) != len(args.treatment):
        raise SystemExit(
            f"{len(args.baseline)} --baseline but {len(args.treatment)} --treatment. They are "
            "zipped positionally, so an uneven count would silently pair the wrong runs."
        )

    pairs = [(Arm.load(b), Arm.load(t)) for b, t in zip(args.baseline, args.treatment, strict=True)]

    # Per-benchmark first, always — the pooled figure is never the only thing printed.
    reports: list[dict] = []
    for base, treat in pairs:
        primary = compare(base, treat, args.n_boot, args.n_perm, args.seed)
        sensitivity = compare(
            base, treat, args.n_boot, args.n_perm, args.seed, exclude_harness_errors=True
        )
        print(format_report(primary))
        print()
        print(format_report(sensitivity))
        print()
        if primary["significant"] != sensitivity["significant"]:
            print("!! WARNING: the verdict changes when harness errors are excluded. The "
                  "conclusion depends on problems that were not actually attempted — report both.")
            print()
        reports.append({"primary": primary, "sensitivity": sensitivity})

    payload: dict = {"comparisons": reports}

    if len(pairs) > 1:
        pooled = compare_pooled(pairs, args.n_boot, args.n_perm, args.seed)
        pooled_sens = compare_pooled(
            pairs, args.n_boot, args.n_perm, args.seed, exclude_harness_errors=True
        )
        print(format_pooled_report(pooled))
        print()
        print(format_pooled_report(pooled_sens))
        payload["pooled"] = {"primary": pooled, "sensitivity": pooled_sens}

    if args.json_out is not None:
        out = args.json_out
    elif len(pairs) == 1:
        out = args.treatment[0] / f"comparison_vs_{pairs[0][0].arm}.json"
    else:
        # A pooled result belongs to no single run, so it goes where cross-run artefacts go.
        tables = Path("results/tables")
        tables.mkdir(parents=True, exist_ok=True)
        t = pairs[0][1]
        out = tables / f"pooled_{t.label.replace('@', '')}_vs_{pairs[0][0].arm}.json"

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"written: {out}")


if __name__ == "__main__":
    main()
