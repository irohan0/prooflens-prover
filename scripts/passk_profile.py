#!/usr/bin/env python3
"""The architecture contrast and failure-mode profile at pass@k, not pass@1.

    python scripts/passk_profile.py --results-root results/exported/logs \\
        --match search.samples_per_step=32 \\
        --match policy_config.premise_free_fraction=0.25

`compare_arms.py` contrasts two *runs*; `discordance_profile.py` asks how the losing arm failed on a
single pair. Both are pass@1 instruments. Under a seed sweep the unit is different: an arm is the
**union of its k seeds**, and the questions the plan set for this phase are about that object.

1. **Does the architecture null survive a 16x budget increase?** The dissertation's central
   result is that late interaction and single-vector solve the same number of problems (74 each,
   p = 1.0000) at 64x16 on one seed. T3 predicted it might break at a wider budget, because LI's
   advantage lives in the *mean* candidate and a deeper search samples further down the ranking.
   Either answer is publishable, so the test has to be run rather than assumed.

2. **Does LI still always die at the expansion cap?** T7 found a sharp asymmetry at 16 samples:
   when LI lost it exhausted the full budget (median 64 of 64) while SV died early and silent (10
   and 33). That was the basis for "LI supplies recall, SV supplies ranking". At 32 samples the
   frontier empties far less often, so the asymmetry may have been a sample-budget artefact.

Every figure printed here is recomputed from `attempts.jsonl` and `verification.json` under
`--results-root`, so nothing in the write-up depends on a number typed by hand. Rejected proofs are
discounted by `eval/draws.py` before they reach any set here.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from passk_union import discover  # noqa: E402
from prooflens_prover.eval.compare import (  # noqa: E402
    bootstrap_ci,
    mcnemar_exact_p,
    permutation_p,
)
from prooflens_prover.eval.draws import load_draw  # noqa: E402
from prooflens_prover.utils.io import read_jsonl  # noqa: E402
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: (benchmark, problem count). The counts are asserted rather than inferred: a benchmark that loads
#: short would otherwise produce a rate against a smaller denominator and never say so.
BENCHMARKS = (("proofnet_test", 186), ("fate_m", 141))

#: Single-run, node-local-staged Tier 1 unions at 64x16 — the baseline this sweep is measured
#: against. Taken from the runs themselves, not from prose, but stated here because the export also
#: contains *replicates* of those runs: a naive union over everything at 16 samples silently reports
#: a multi-seed figure as the published single-seed one, which inflates the baseline and understates
#: the gain. See dissertation.md §3.
PUBLISHED_PASS1_UNION = {"proofnet_test": 32, "fate_m": 56}


def arm_of(draw) -> str:
    """`li` or `sv`, dropping the `@50k` budget suffix `Draw` appends for the two-stage arm."""
    return "li" if draw.arm.startswith("li") else draw.arm


def arm_unions(dirs: list[Path]) -> dict[str, set[str]]:
    """`{arm: problems solved by ANY of its seeds}` — the pass@k object."""
    out: dict[str, set[str]] = {}
    for p in dirs:
        d = load_draw(p)
        out.setdefault(arm_of(d), set()).update(d.solved)
    return out


def attempted_union(dirs: list[Path]) -> set[str]:
    out: set[str] = set()
    for p in dirs:
        out |= load_draw(p).attempted
    return out


def status_mix(dirs: list[Path], arm: str) -> tuple[int, dict[str, float]]:
    """Share of every terminal status across all seeds of one arm.

    `no_candidates` is the number to watch: it means the frontier emptied because nothing sampled
    made progress, which more expansions cannot fix and more samples can. It was 42% of ProofNet
    attempts at 16 samples and is what the sweep's doubled sample budget was bought to reduce.
    """
    c: Counter[str] = Counter()
    for p in dirs:
        if arm_of(load_draw(p)) != arm:
            continue
        c.update(r.get("status") for r in read_jsonl(p / "attempts.jsonl"))
    total = sum(c.values())
    return total, {k: 100 * v / total for k, v in c.most_common()}


def loser_profile(dirs: list[Path], loser: str, pids: set[str]) -> dict:
    """How `loser` failed, across all its seeds, on the problems only the other arm solved.

    The distinction that matters is `no_candidates` against `exhausted` at the cap: silence means
    the problem was unreachable for that retriever, budget exhaustion means it was reachable and
    mis-ranked. One is bought with samples, the other with expansions, so which it is decides where
    the next GPU-hour goes.
    """
    bare = {p.split(":", 1)[1] for p in pids}
    c: Counter[str] = Counter()
    exps: list[int] = []
    for p in dirs:
        if arm_of(load_draw(p)) != loser:
            continue
        for row in read_jsonl(p / "attempts.jsonl"):
            if str(row["problem_id"]) not in bare:
                continue
            c[row.get("status")] += 1
            if row.get("status") == "exhausted":
                exps.append(row.get("n_expansions") or 0)
    return {"n_problems": len(bare), "statuses": dict(c.most_common()),
            "median_expansions_when_exhausted": median(exps) if exps else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=Path("results/exported/logs"))
    ap.add_argument("--policy", default="vllm")
    ap.add_argument("--match", action="append", default=[],
                    help="KEY=VALUE on the manifest config; the problem count is added per "
                         "benchmark automatically")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ensure_utf8_output()

    rng = np.random.default_rng(args.seed)
    pooled: dict[str, set[str]] = {"li": set(), "sv": set()}
    pooled_attempted: set[str] = set()

    for bench, n in BENCHMARKS:
        dirs = discover(args.results_root, bench, args.policy, [*args.match, f"n_problems={n}"])
        if not dirs:
            print(f"{bench}: no runs matched — check --match")
            continue
        unions = arm_unions(dirs)
        if set(unions) != {"li", "sv"}:
            print(f"{bench}: expected li and sv, found {sorted(unions)}")
            continue
        li, sv = unions["li"], unions["sv"]
        attempted = attempted_union(dirs)
        if len(attempted) != n:
            raise SystemExit(
                f"{bench}: the runs cover {len(attempted)} problems, not {n}. A rate over a "
                f"different denominator is not comparable to a published one."
            )
        pooled["li"] |= li
        pooled["sv"] |= sv
        pooled_attempted |= attempted

        both = li | sv
        base = PUBLISHED_PASS1_UNION.get(bench)
        print(f"\n{'=' * 78}\n{bench}  ({len(dirs)} runs, {n} problems)\n{'=' * 78}")
        print(f"  li union over seeds : {len(li):3}  ({100 * len(li) / n:.1f}%)")
        print(f"  sv union over seeds : {len(sv):3}  ({100 * len(sv) / n:.1f}%)")
        print(f"  ENSEMBLE            : {len(both):3}  ({100 * len(both) / n:.1f}%)"
              + (f"   vs published pass@1 union {base}  ({len(both) - base:+d})" if base else ""))
        print(f"  li only {len(li - sv):2} | sv only {len(sv - li):2} | shared {len(li & sv):3}"
              f"   McNemar p = {mcnemar_exact_p(len(li - sv), len(sv - li)):.4f}")

        print("\n  status mix across all seeds:")
        for arm in ("li", "sv"):
            total, mix = status_mix(dirs, arm)
            pretty = "  ".join(f"{k}={v:.1f}%" for k, v in mix.items())
            print(f"    {arm} ({total} attempts): {pretty}")

        print("\n  how the losing arm failed on the other's exclusive problems:")
        for winner, loser in (("li", "sv"), ("sv", "li")):
            excl = unions[winner] - unions[loser]
            if not excl:
                continue
            prof = loser_profile(dirs, loser, excl)
            print(f"    {prof['n_problems']} solved only by {winner}: {loser} -> "
                  f"{prof['statuses']}")
            if prof["median_expansions_when_exhausted"] is not None:
                print(f"      median expansions when exhausted: "
                      f"{prof['median_expansions_when_exhausted']:.0f} of 64")

    if not pooled_attempted:
        return

    li, sv = pooled["li"], pooled["sv"]
    ids = sorted(pooled_attempted)
    d = np.array([float(p in li) - float(p in sv) for p in ids])
    lo, hi = bootstrap_ci(d, args.n_boot, rng)
    perm_p = permutation_p(d, args.n_perm, rng)

    print(f"\n{'=' * 78}\nPOOLED  ({len(ids)} problems)\n{'=' * 78}")
    print(f"  li {len(li)}  sv {len(sv)}  ensemble {len(li | sv)}  "
          f"({100 * len(li | sv) / len(ids):.1f}%)")
    print(f"  li only {len(li - sv)} | sv only {len(sv - li)}   "
          f"McNemar p = {mcnemar_exact_p(len(li - sv), len(sv - li)):.4f}")
    print(f"  effect li - sv = {d.sum():+.0f} problems ({100 * d.mean():+.2f} pts)")
    print(f"  paired bootstrap 95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}] pts")
    print(f"  sign-flip permutation p = {perm_p:.4f}")
    # The two must agree. A CI excluding zero beside a non-significant permutation p (or the
    # reverse) means one of them is being read wrongly, and the pass@1 analysis treats their
    # agreement as the gate rather than picking whichever is kinder.
    agree = (lo <= 0 <= hi) == (perm_p >= 0.05)
    print(f"  CI and permutation agree: {'yes' if agree else 'NO — do not report either'}")


if __name__ == "__main__":
    main()
