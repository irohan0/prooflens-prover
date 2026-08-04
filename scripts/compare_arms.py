#!/usr/bin/env python
"""Paired comparison of two arms over one benchmark, with significance and premise attribution.

    python scripts/compare_arms.py \
        --baseline results/logs/fate_m_none_repertoire_... \
        --treatment results/logs/fate_m_li_repertoire_...

Writes the full result as JSON next to the treatment run (`comparison_vs_<baseline arm>.json`) so
every reported number stays traceable to the two runs it came from.

The primary analysis keeps harness errors as unproved, so the denominator matches the published
benchmark size. A sensitivity analysis excluding them is always printed too: if the two disagree,
the conclusion depends on problems that were never really attempted, and that has to be said.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prooflens_prover.eval.compare import Arm, compare, format_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, type=Path, help="results/logs/<run_id> (control)")
    ap.add_argument("--treatment", required=True, type=Path, help="results/logs/<run_id>")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    base = Arm.load(args.baseline)
    treat = Arm.load(args.treatment)

    primary = compare(base, treat, args.n_boot, args.n_perm, args.seed)
    sensitivity = compare(
        base, treat, args.n_boot, args.n_perm, args.seed, exclude_harness_errors=True
    )

    print(format_report(primary))
    print()
    print(format_report(sensitivity))

    if primary["significant"] != sensitivity["significant"]:
        print()
        print("!! WARNING: the verdict changes when harness errors are excluded. The conclusion "
              "depends on problems that were not actually attempted — report both.")

    out = args.json_out or (args.treatment / f"comparison_vs_{base.arm}.json")
    out.write_text(
        json.dumps({"primary": primary, "sensitivity": sensitivity}, indent=2), encoding="utf-8"
    )
    print()
    print(f"written: {out}")


if __name__ == "__main__":
    main()
