#!/usr/bin/env python3
"""What the *second architecture* is worth once its generations are paid for.

    python scripts/budget_matched.py --results-root results/exported/logs \\
        --match search.samples_per_step=32 \\
        --match policy_config.premise_free_fraction=0.25

## The comparison this exists to fix

§3.4 of the write-up reports a union of 88 against 74 for either arm alone — **+14 of 327, larger
than retrieval's entire measured effect** — and that number drove an entire phase of work, including
a fused retriever built to capture it.

It is not a like-for-like comparison. The union runs **two arms at one seed**; each single arm runs
**one arm at one seed**. The union has twice the generations, so part of the +14 is just the second
draw, which would help just as much if it came from the same retriever. Nothing separates the two in
§3.4, and the fusion result (§8.2) is what forced the question: a retriever merging both rankings
captures none of the +14, which is what one expects if the +14 is mostly budget.

This script holds **generations per problem** fixed and asks what remains:

    16,384 generations/problem  =  one arm at 8 seeds  =  two arms at 4 seeds

and contrasts `ensemble@k` against `li@2k` and `sv@2k`. On this run set the ensemble beats the
better single arm by **+2.87 of 327, CI [-2.34, +8.54], p = 0.33** — roughly a quarter of the
headline figure, and not significant. Against the weaker arm it is +4.87, p = 0.046.

## The estimator, and why a draw is a seed

Per problem, the unbiased `pass@k = 1 - C(K-c,k)/C(K,k)` (Chen et al. 2021) over the K seeds
available, where `c` counts the seeds at which that *draw definition* solved the problem. A **draw**
is one seed's worth of work:

* for a single arm, that arm at that seed — 2,048 generations;
* for the ensemble, **both** arms at that seed — 4,096 generations.

So `ensemble@k` and `single@2k` cost the same, and that is the pairing reported. Seeds are the unit
rather than (arm, seed) pairs because the deployed ensemble runs both arms at each seed; pooling
them into one bag of 16 interchangeable draws would assume the arms are samples from one
distribution, which is the very thing under test.

## Significance

Paired over problems: the difference vector is the per-problem probability difference, so it is
continuous rather than 0/±1, but the pairing is the same one `compare_arms.py` uses and it goes
through the same `bootstrap_ci` and `permutation_p`. **Both must agree** or neither is reported —
a CI excluding zero beside a non-significant permutation p means one of them is being misread.

McNemar is deliberately absent: it is defined on discordant *counts*, and there are no discordant
counts here, only expected values.
"""

from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from passk_profile import BENCHMARKS, arm_of, require_verified  # noqa: E402
from passk_union import discover  # noqa: E402
from prooflens_prover.eval.compare import bootstrap_ci, permutation_p  # noqa: E402
from prooflens_prover.eval.draws import load_draw  # noqa: E402
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: Generations per problem for one seed of one arm, at the sweep's 64 nodes x 32 samples. Only used
#: to label the budget column; nothing is computed from it.
GEN_PER_DRAW = 64 * 32


def by_seed(dirs: list[Path]) -> tuple[dict[str, dict[int, set[str]]], set[str]]:
    """`{arm: {seed: solved}}` and the union of attempted problem ids.

    Refuses a duplicated (arm, seed): the same draw counted twice raises `c` in the estimator
    without adding evidence, which inflates every pass@k built on it.
    """
    solved: dict[str, dict[int, set[str]]] = {}
    attempted: set[str] = set()
    for p in dirs:
        d = load_draw(p)
        arm = arm_of(d)
        if d.seed in solved.get(arm, {}):
            raise SystemExit(
                f"duplicate ({arm}, seed {d.seed}) at {p.name}: the same draw twice raises the "
                "solve count in 1 - C(K-c,k)/C(K,k) without adding a draw."
            )
        solved.setdefault(arm, {})[d.seed] = set(d.solved)
        attempted |= d.attempted
    return solved, attempted


def passk(draws: list[set[str]], pid: str, k: int) -> float:
    """`1 - C(K-c,k)/C(K,k)` for one problem: the chance k random draws contain a working one."""
    K = len(draws)
    if not 1 <= k <= K:
        raise ValueError(f"k={k} outside 1..{K}")
    c = sum(1 for d in draws if pid in d)
    return 1.0 - comb(K - c, k) / comb(K, k)


def curve(draws: list[set[str]], ids: list[str], k: int) -> np.ndarray:
    return np.array([passk(draws, p, k) for p in ids])


def report(label: str, a: np.ndarray, b: np.ndarray, rng, n_boot: int, n_perm: int) -> None:
    d = a - b
    lo, hi = bootstrap_ci(d, n_boot, rng)
    p = permutation_p(d, n_perm, rng)
    n = len(d)
    agree = (lo <= 0 <= hi) == (p >= 0.05)
    print(f"    {label:<24} {d.sum():+6.2f} problems   "
          f"CI [{n * lo:+6.2f}, {n * hi:+6.2f}]   permutation p = {p:.4f}   "
          f"agree: {'yes' if agree else 'NO — do not report either'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=Path("results/exported/logs"))
    ap.add_argument("--policy", default="vllm")
    ap.add_argument("--match", action="append", default=[])
    ap.add_argument("--k", type=int, default=4,
                    help="ensemble draws; each single arm is given 2k to match the generations")
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ensure_utf8_output()

    rng = np.random.default_rng(args.seed)
    pooled: dict[str, list[np.ndarray]] = {}

    for bench, n in BENCHMARKS:
        dirs = discover(args.results_root, bench, args.policy,
                        [*args.match, f"n_problems={n}"])
        # Only the two single arms take part: the ensemble is *constructed* from them, and a third
        # arm (fusion) is a different budget again.
        dirs = [d for d in dirs if arm_of(load_draw(d)) in ("li", "sv")]
        if not dirs:
            print(f"{bench}: no runs matched — check --match")
            continue
        require_verified(dirs, args.allow_unverified)
        solved, attempted = by_seed(dirs)

        if set(solved) != {"li", "sv"}:
            raise SystemExit(f"{bench}: need both li and sv, found {sorted(solved)}")
        seeds = sorted(set(solved["li"]) & set(solved["sv"]))
        if len(seeds) != len(solved["li"]) or len(seeds) != len(solved["sv"]):
            raise SystemExit(
                f"{bench}: arms ran at different seeds — li {sorted(solved['li'])}, "
                f"sv {sorted(solved['sv'])}. An ensemble draw needs BOTH arms at the same seed."
            )
        K = len(seeds)
        if 2 * args.k > K:
            raise SystemExit(
                f"{bench}: --k {args.k} needs {2 * args.k} single-arm draws to match the budget, "
                f"but only {K} seeds exist. Use --k {K // 2} or fewer."
            )
        if len(attempted) != n:
            raise SystemExit(f"{bench}: runs cover {len(attempted)} problems, not {n}.")

        ids = sorted(attempted)
        li = [solved["li"][s] for s in seeds]
        sv = [solved["sv"][s] for s in seeds]
        ens = [solved["li"][s] | solved["sv"][s] for s in seeds]

        curves = {
            f"ensemble@{args.k}": curve(ens, ids, args.k),
            f"li@{2 * args.k}": curve(li, ids, 2 * args.k),
            f"sv@{2 * args.k}": curve(sv, ids, 2 * args.k),
        }
        gens = 2 * args.k * GEN_PER_DRAW
        print(f"\n{'=' * 78}\n{bench}  ({n} problems, {K} seeds, {gens:,} generations/problem)"
              f"\n{'=' * 78}")
        for name, c in curves.items():
            print(f"  {name:<14} expected solved: {c.sum():6.2f}  ({100 * c.mean():.1f}%)")
        print("\n  budget-matched contrasts:")
        ekey = f"ensemble@{args.k}"
        for name in (f"li@{2 * args.k}", f"sv@{2 * args.k}"):
            report(f"{ekey} - {name}", curves[ekey], curves[name], rng, args.n_boot, args.n_perm)
        for name, c in curves.items():
            pooled.setdefault(name, []).append(c)

    if len(pooled) < 3 or len(next(iter(pooled.values()))) != len(BENCHMARKS):
        print("\nno pooled contrast: not every benchmark contributed all three curves")
        return

    cat = {name: np.concatenate(cs) for name, cs in pooled.items()}
    n_tot = len(next(iter(cat.values())))
    gens = 2 * args.k * GEN_PER_DRAW
    print(f"\n{'=' * 78}\nPOOLED  ({n_tot} problems, {gens:,} generations/problem)\n{'=' * 78}")
    for name, c in cat.items():
        print(f"  {name:<14} expected solved: {c.sum():6.2f}  ({100 * c.mean():.1f}%)")
    print("\n  budget-matched contrasts:")
    ekey = f"ensemble@{args.k}"
    for name in (f"li@{2 * args.k}", f"sv@{2 * args.k}"):
        report(f"{ekey} - {name}", cat[ekey], cat[name], rng, args.n_boot, args.n_perm)
    print("\n  A second retriever is worth the difference above; a second SEED of the same\n"
          "  retriever is worth the rest of what the raw union reports. See the README.")


if __name__ == "__main__":
    main()
