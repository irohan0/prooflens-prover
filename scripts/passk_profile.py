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

from passk_union import check_verified, discover, parse_seeds  # noqa: E402
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
    """The bare arm name, dropping the `@50k` candidate-budget suffix `Draw` appends.

    The suffix is not exclusive to late interaction, which an earlier version of this assumed: the
    fusion arm carries `n_candidates=50000` as well, so `Draw` labels it `fusion@50k`. Matching only
    `li` left `--arm fusion` selecting nothing and the fusion runs **silently absent** from a
    contrast that still printed. Splitting on `@` is what `passk_union.arm_matches` already does.
    """
    return draw.arm.split("@", 1)[0]


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


def require_verified(dirs: list[Path], allow: bool) -> None:
    """Refuse runs whose proofs were never independently re-elaborated.

    `passk_union.py` has always gated on this; this script did not, and it produces numbers for the
    write-up just as much. An unverified proof is a claim: the search says it closed the goal, and
    nothing has checked that the recorded tactics actually elaborate from the benchmark statement.
    The sweep found one that did not, in 1,377 claims.

    Worse here than a missing figure would be: `eval/draws.py` discounts rejected proofs *from the
    report file*, so a run without one is not merely unchecked, it is silently exempt from the
    discount that every verified run receives — it competes against them with its bad claims intact.
    """
    unverified = [(d, why) for d in dirs if (why := check_verified(d)) is not None]
    if not unverified:
        return
    lines = "\n".join(f"  {d.name}: {why}" for d, why in unverified)
    if allow:
        print(f"WARNING — {len(unverified)} unverified run(s), counted anyway:\n{lines}\n")
        return
    raise SystemExit(
        f"{len(unverified)} run(s) have not been verified:\n{lines}\n\n"
        "A contrast built on unverified proofs is a claim, not a result, and an unverified run is "
        "also exempt from the discount its verified rivals get. Run slurm/verify_proofs.sbatch "
        "over them, re-export, and try again — or pass --allow-unverified for a work-in-progress "
        "look whose numbers must not be reported."
    )


def restrict(dirs: list[Path], arms: list[str] | None, seeds: set[int] | None) -> list[Path]:
    """The run directories matching the requested arms and seeds.

    Both filters exist for the same reason: a later arm is measured at fewer seeds than the sweep,
    and comparing it against eight-seed unions would credit it with draws nobody made. Restricting
    the deeper arms is how the contrast is taken at equal k from runs that already exist.
    """
    out = []
    for d in dirs:
        draw = load_draw(d)
        if arms and not any(arm_of(draw) == a for a in arms):
            continue
        if seeds is not None and draw.seed not in seeds:
            continue
        out.append(d)
    return out


def seeds_of(dirs: list[Path]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for d in dirs:
        draw = load_draw(d)
        out.setdefault(arm_of(draw), set()).add(draw.seed)
    return out


def contrast(a_name: str, a: set[str], b_name: str, b: set[str], attempted: set[str],
             rng, n_boot: int, n_perm: int) -> None:
    """Print one paired contrast, with the agreement gate the pass@1 analysis uses."""
    ids = sorted(attempted)
    d = np.array([float(p in a) - float(p in b) for p in ids])
    lo, hi = bootstrap_ci(d, n_boot, rng)
    perm_p = permutation_p(d, n_perm, rng)
    print(f"\n  {a_name} vs {b_name}")
    print(f"    {a_name} only {len(a - b):3} | {b_name} only {len(b - a):3} | "
          f"shared {len(a & b):3}   McNemar p = {mcnemar_exact_p(len(a - b), len(b - a)):.4f}")
    print(f"    effect = {d.sum():+.0f} problems ({100 * d.mean():+.2f} pts)   "
          f"CI [{100 * lo:+.2f}, {100 * hi:+.2f}]   permutation p = {perm_p:.4f}")
    # A CI excluding zero beside a non-significant permutation p means one is being misread. The
    # pass@1 analysis gates on their agreement rather than picking whichever is kinder, and the
    # verdict is printed either way — a check whose pass is silent is a check nobody sees run.
    if (lo <= 0 <= hi) == (perm_p >= 0.05):
        print("    CI and permutation agree: yes")
    else:
        print("    CI and permutation agree: NO — do not report either")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=Path("results/exported/logs"))
    ap.add_argument("--policy", default="vllm")
    ap.add_argument("--match", action="append", default=[],
                    help="KEY=VALUE on the manifest config; the problem count is added per "
                         "benchmark automatically")
    ap.add_argument("--arm", action="append",
                    help="restrict to these arms, e.g. --arm fusion --arm li; repeatable")
    ap.add_argument("--seeds", default=None, metavar="SPEC",
                    help="restrict to these seeds, e.g. 0-3, so an arm measured at four seeds is "
                         "contrasted against ones measured at eight at equal k")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="contrast runs whose proofs were never re-elaborated. For a "
                         "work-in-progress look only — never for a reported number")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ensure_utf8_output()

    rng = np.random.default_rng(args.seed)
    want_seeds = parse_seeds(args.seeds)
    pooled: dict[str, set[str]] = {}
    pooled_attempted: set[str] = set()
    #: Which arms each benchmark actually contributed. An arm run on only some benchmarks must not
    #: be pooled against arms run on all of them — see the guard below the loop.
    arms_seen: dict[str, set[str]] = {}

    #: Every arm label seen anywhere, so a `--arm` that matches nothing can be refused. An arm
    #: legitimately absent from ONE benchmark is normal (fusion ran on FATE-M only); an arm absent
    #: from all of them is a typo or a label mismatch, and it fails quietly — the remaining arms
    #: still contrast and still print a plausible table. `fusion` really did vanish this way, tagged
    #: `fusion@50k` by `Draw` against an `arm_of` that only stripped the suffix for `li`.
    seen_anywhere: set[str] = set()

    for bench, n in BENCHMARKS:
        found = discover(args.results_root, bench, args.policy, [*args.match, f"n_problems={n}"])
        seen_anywhere |= {arm_of(load_draw(d)) for d in found}
        dirs = restrict(found, args.arm, want_seeds)
        if not dirs:
            print(f"{bench}: no runs matched — check --match / --arm / --seeds")
            continue
        require_verified(dirs, args.allow_unverified)
        unions = arm_unions(dirs)
        if len(unions) < 2:
            print(f"{bench}: only {sorted(unions)} present, nothing to contrast")
            continue

        # Every arm must have been drawn at the same seeds, or a union built from more draws is
        # compared against one built from fewer, and the difference is budget, not architecture.
        per_arm_seeds = seeds_of(dirs)
        if len({frozenset(v) for v in per_arm_seeds.values()}) != 1:
            raise SystemExit(
                f"{bench}: arms ran at different seeds "
                f"{ {a: sorted(v) for a, v in per_arm_seeds.items()} }. Restrict with --seeds so "
                "every arm contributes the same number of draws."
            )

        attempted = attempted_union(dirs)
        if len(attempted) != n:
            raise SystemExit(
                f"{bench}: the runs cover {len(attempted)} problems, not {n}. A rate over a "
                f"different denominator is not comparable to a published one."
            )
        for arm, solved in unions.items():
            pooled.setdefault(arm, set()).update(solved)
        arms_seen[bench] = set(unions)
        pooled_attempted |= attempted

        arms = sorted(unions)
        k = len(next(iter(per_arm_seeds.values())))
        ensemble = set().union(*unions.values())
        base = PUBLISHED_PASS1_UNION.get(bench)
        print(f"\n{'=' * 78}\n{bench}  ({len(dirs)} runs, {n} problems, pass@{k})\n{'=' * 78}")
        for arm in arms:
            print(f"  {arm:12} union over seeds : {len(unions[arm]):3}  "
                  f"({100 * len(unions[arm]) / n:.1f}%)")
        print(f"  {'ENSEMBLE':12} of {len(arms)} arm(s) : {len(ensemble):3}  "
              f"({100 * len(ensemble) / n:.1f}%)"
              + (f"   vs published pass@1 union {base}  ({len(ensemble) - base:+d})"
                 if base and len(arms) > 1 else ""))

        print("\n  status mix across all seeds:")
        for arm in arms:
            total, mix = status_mix(dirs, arm)
            pretty = "  ".join(f"{key}={v:.1f}%" for key, v in mix.items())
            print(f"    {arm} ({total} attempts): {pretty}")

        print("\n  how the losing arm failed on the other's exclusive problems:")
        for winner in arms:
            for loser in arms:
                if winner == loser:
                    continue
                excl = unions[winner] - unions[loser]
                if not excl:
                    continue
                prof = loser_profile(dirs, loser, excl)
                print(f"    {prof['n_problems']} solved only by {winner}: {loser} -> "
                      f"{prof['statuses']}")
                if prof["median_expansions_when_exhausted"] is not None:
                    print(f"      median expansions when exhausted: "
                          f"{prof['median_expansions_when_exhausted']:.0f} of 64")

    if args.arm and (missing := sorted(set(args.arm) - seen_anywhere)):
        raise SystemExit(
            f"--arm {', '.join(missing)} matched no run on any benchmark. Present: "
            f"{sorted(seen_anywhere)}. A contrast that silently drops a requested arm still prints "
            "a plausible table, so this is a refusal rather than a warning."
        )

    if not pooled_attempted or len(pooled) < 2:
        return

    # An arm measured on a subset of the benchmarks CANNOT be pooled against arms measured on all of
    # them. Its union is drawn from a smaller problem set while the denominator is every problem, so
    # the contrast reports the benchmarks it skipped as failures.
    #
    # Not hypothetical, and not subtle in its consequences: with fusion run on FATE-M only, where it
    # solved 70 against late interaction's 64, the pooled contrast reported fusion -34 problems at
    # p = 0.0000 with the CI and permutation test agreeing. Maximum confidence, wrong sign. The
    # per-benchmark sections above are where such an arm is legitimately compared.
    complete = set.intersection(*arms_seen.values()) if arms_seen else set()
    partial = sorted(set(pooled) - complete)
    if partial:
        print("\n" + "=" * 78)
        for arm in partial:
            ran_on = sorted(b for b, a in arms_seen.items() if arm in a)
            print(f"NOT POOLED: {arm!r} ran on {ran_on} only, not on "
                  f"{sorted(set(arms_seen) - set(ran_on))}. Pooling it would score the benchmarks "
                  f"it skipped as failures. See its per-benchmark contrast above.")
        pooled = {a: v for a, v in pooled.items() if a in complete}
    if len(pooled) < 2:
        print("Fewer than two arms span every benchmark, so there is no pooled contrast to make.")
        return

    arms = sorted(pooled)
    ids = sorted(pooled_attempted)
    print(f"\n{'=' * 78}\nPOOLED  ({len(ids)} problems)\n{'=' * 78}")
    for arm in arms:
        print(f"  {arm:12} {len(pooled[arm]):3}")
    print(f"  {'ENSEMBLE':12} {len(set().union(*pooled.values())):3}  "
          f"({100 * len(set().union(*pooled.values())) / len(ids):.1f}%)")
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            contrast(a, pooled[a], b, pooled[b], pooled_attempted, rng, args.n_boot, args.n_perm)


if __name__ == "__main__":
    main()
