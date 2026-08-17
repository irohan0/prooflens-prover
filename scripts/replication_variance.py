#!/usr/bin/env python
"""Does the architecture null survive re-sampling? And is the disagreement above sampling noise?

    python scripts/replication_variance.py \
        --run results/logs/fate_m_sv_vllm_<seed0> --run results/logs/fate_m_sv_vllm_<seed1> \
        --run results/logs/fate_m_li_vllm_<seed0> --run results/logs/fate_m_li_vllm_<seed1>

## Why this is Phase 2, ahead of more interesting work

Tier 1 ran **one** pass per arm at `temperature 1.5`. Every paired test reported treats a problem's
outcome as fixed given its arm, and for a temperature-1.5 language model it is not. Three published
claims rest on that single draw:

1. **Δ(LI, SV) = +0** over 327 problems, p = 1.0000.
2. **17 gained and 17 lost** — "equal counts, different theorems".
3. **The union reaches 89**, a fusion ceiling of +17 against retrieval's own +14.

Claim 1 is the robust one: sampling noise does not manufacture a null, it hides an effect. Claims 2
and 3 are fragile, and they fail together. If re-running **one arm against itself** also flips ~17
problems, the disagreement carries no information about architecture, the fusion ceiling is mostly
resampling headroom, and two of the write-up's more interesting statements have to be withdrawn.
Nothing in the existing data can distinguish those cases, because with one draw per arm there is no
estimate of what a re-run does on its own.

So this measures the **noise floor** — same arm, same everything, different sampling draw — and
reports every between-arm quantity against it. A between-arm effect inside the floor is not an
effect.

## What a draw is

`prove_benchmark.py` passes `--seed` to the vLLM engine (`LLM(seed=...)`) while
`SamplingParams.seed` stays `None`, so a run is reproducible at a given `--seed` and a different
`--seed` is an independent draw. The six Tier 1 runs all used the default `--seed 0`, which is why
they are one draw and not three.

That reasoning could be wrong in a way no summary statistic would reveal, so the script **checks
it**: two draws of one arm whose shared proofs are byte-identical are the same draw, and the run is
refused rather than reported as an exceptionally stable result.

Reads `attempts.jsonl` and `manifest.json`. No GPU, no model, no Lean.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.eval.compare import bootstrap_ci, permutation_p  # noqa: E402
from prooflens_prover.eval.draws import (  # noqa: E402
    Draw,
    discordance,
    identical_proof_fraction,
    load_draw,
    solve_rate_map,
    union_gain,
)
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: Config keys allowed to differ between two draws of the same arm. `n_problems` can differ when a
#: run was resumed; everything else differing means the two runs are different experiments.
#: Config keys a genuine replicate is allowed to differ in.
#:
#: `lean_project` is here because node-local staging writes the Mathlib project under
#: `/tmp/slurm.<jobid>`, so **no two cluster replicates ever share the path**. Without this
#: exemption every real replicate is rejected and the script only ever runs on the NFS runs — which
#: is precisely the configuration measured to change a published number (26 vs 28 on ProofNet). The
#: environment is a variable and is reported, not one silently required to be constant.
DRAW_VARYING = frozenset({"n_problems", "lean_project"})


def group_draws(run_dirs) -> dict[tuple[str, str], list[Draw]]:
    """`{(benchmark, arm): [draws by seed]}`, refusing anything that is not a clean replicate."""
    groups: dict[tuple[str, str], list[Draw]] = defaultdict(list)
    for d in run_dirs:
        draw = load_draw(Path(d))
        groups[(draw.benchmark, draw.arm)].append(draw)

    for (bench, arm), draws in groups.items():
        seeds = [d.seed for d in draws]
        if len(set(seeds)) != len(seeds):
            dupes = sorted({s for s in seeds if seeds.count(s) > 1})
            raise SystemExit(
                f"{bench}/{arm}: two runs share seed(s) {dupes}. Those are the same draw, so "
                "averaging them would understate the very variance this script measures."
            )
        first = draws[0]
        for other in draws[1:]:
            differing = sorted(
                k for k in set(first.config) | set(other.config)
                if k not in DRAW_VARYING and first.config.get(k) != other.config.get(k)
            )
            if differing:
                raise SystemExit(
                    f"{bench}/{arm}: seeds {first.seed} and {other.seed} differ in {differing}. "
                    "A replicate must vary only --seed; these are different experiments."
                )
        draws.sort(key=lambda d: d.seed)
    return dict(groups)


def spread(values: list[float]) -> tuple[float, float | None]:
    """`(mean, sample sd)`; sd is None with fewer than two values."""
    mean = float(statistics.fmean(values))
    return mean, (float(statistics.stdev(values)) if len(values) > 1 else None)


def fmt(x: float | None, spec: str = "+.4f") -> str:
    return "—" if x is None else format(x, spec)


def by_arm_and_seed(groups) -> tuple[dict[str, dict[int, set[str]]],
                                     dict[str, dict[int, Draw]], list[str]]:
    """Pool benchmarks: `{arm: {seed: solved}}`, `{arm: {seed: merged draw}}`, arm order.

    Pooling is what the published claims are stated over (327 problems), so the noise floor has to
    be measured on the same object.

    Arm order follows first appearance in `--run`, **not** alphabetical order, because it decides
    which arm is the baseline and therefore the sign of every reported delta. Sorting would make
    `li` the baseline purely because "l" precedes "s", and a delta whose direction depends on
    spelling is a trap rather than a convention.
    """
    solved: dict[str, dict[int, set[str]]] = defaultdict(dict)
    merged: dict[str, dict[int, Draw]] = defaultdict(dict)
    arm_order: list[str] = []
    for (bench, arm), draws in groups.items():
        if arm not in arm_order:
            arm_order.append(arm)
        for d in draws:
            if d.seed in solved[arm]:
                solved[arm][d.seed] |= d.solved
                merged[arm][d.seed].attempted |= d.attempted
                merged[arm][d.seed].solved |= d.solved
                merged[arm][d.seed].proofs.update(d.proofs)
                merged[arm][d.seed].benchmark += f"+{bench}"
            else:
                solved[arm][d.seed] = set(d.solved)
                merged[arm][d.seed] = Draw(
                    run_id=d.run_id, benchmark=bench, arm=arm, seed=d.seed, config=d.config,
                    attempted=set(d.attempted), solved=set(d.solved), proofs=dict(d.proofs),
                )
    return dict(solved), dict(merged), arm_order


def report(groups: dict[tuple[str, str], list[Draw]], args) -> int:
    solved_by_arm, merged, arms = by_arm_and_seed(groups)

    print("=== draws ===")
    print(f"  {'benchmark':<15} {'arm':<10} {'seed':>5} {'attempted':>10} {'proved':>7}")
    for (bench, arm), draws in sorted(groups.items()):
        for d in draws:
            print(f"  {bench:<15} {arm:<10} {d.seed:>5} {len(d.attempted):>10} "
                  f"{len(d.solved):>7}")

    n_draws = {arm: len(s) for arm, s in solved_by_arm.items()}
    print(f"\n  draws per arm: {n_draws}")
    if max(n_draws.values(), default=0) < 2:
        print("\n  ONLY ONE DRAW PER ARM — there is nothing to measure yet.")
        print("  Re-run each arm with a different --seed (the six Tier 1 runs all used --seed 0),")
        print("  then pass every run here. Until then, every paired test in the write-up is")
        print("  reported without an estimate of its own sampling variance.")
        return 1

    # --- tripwire: are the replicates actually different draws? ---
    print("\n=== are the replicates real? ===")
    print("  Two draws of one arm must not produce byte-identical proofs. If they do, the seed")
    print("  never reached the sampler and every variance below measures nothing.")
    tripwire = []
    for arm in arms:
        seeds = sorted(solved_by_arm[arm])
        for s1, s2 in itertools.combinations(seeds, 2):
            n_shared, frac = identical_proof_fraction(merged[arm][s1], merged[arm][s2])
            tripwire.append({"arm": arm, "seeds": [s1, s2], "n_shared_solved": n_shared,
                             "fraction_identical_proofs": frac})
            print(f"  {arm:<10} seeds {s1} vs {s2}: {n_shared:>3} both solved, "
                  f"{fmt(frac, '.1%'):>6} identical proofs")
    degenerate = [t for t in tripwire if t["fraction_identical_proofs"] == 1.0
                  and t["n_shared_solved"] > 0]
    if degenerate:
        print()
        for t in degenerate:
            print(f"  REFUSED: {t['arm']} seeds {t['seeds']} agree on every one of "
                  f"{t['n_shared_solved']} shared proofs, character for character.")
        raise SystemExit(
            "these are the same draw, not replicates — check that --seed actually differs and that "
            "prove_benchmark.py forwards it to the engine"
        )

    # --- the noise floor: one arm against itself ---
    print("\n=== noise floor: the same arm, a different draw ===")
    print(f"  {'arm':<10} {'seeds':<10} {'proved':>14} {'discordant':>11} {'union gain':>11}")
    floor_disc, floor_union, within = [], [], []
    for arm in arms:
        seeds = sorted(solved_by_arm[arm])
        for s1, s2 in itertools.combinations(seeds, 2):
            a, b = solved_by_arm[arm][s1], solved_by_arm[arm][s2]
            only_a, only_b = discordance(a, b)
            gain = union_gain(a, b)
            floor_disc.append(only_a + only_b)
            floor_union.append(gain)
            within.append({"arm": arm, "seeds": [s1, s2], "proved": [len(a), len(b)],
                           "only_first": only_a, "only_second": only_b,
                           "discordant": only_a + only_b, "union_gain": gain})
            print(f"  {arm:<10} {f'{s1} vs {s2}':<10} {f'{len(a)} vs {len(b)}':>14} "
                  f"{f'{only_a}+{only_b}={only_a + only_b}':>11} {gain:>11}")
    print(f"\n  A single arm re-run against itself flips {min(floor_disc)}-{max(floor_disc)} "
          f"problems (mean {statistics.fmean(floor_disc):.1f}), and its own")
    print(f"  draws union to {min(floor_union)}-{max(floor_union)} above the better one "
          f"(mean {statistics.fmean(floor_union):.1f}).")
    print("  Those are the numbers every between-arm figure has to beat to mean anything.")

    # --- between arms, per draw ---
    comparisons = []
    for base, treat in itertools.combinations(arms, 2):
        shared_seeds = sorted(set(solved_by_arm[base]) & set(solved_by_arm[treat]))
        if not shared_seeds:
            continue
        print(f"\n=== {treat} vs {base} ===")
        print("  One draw of each arm — what a single-run experiment would have reported:")
        print(f"  {'seed':>5} {'proved':>14} {'delta':>7} {'discordant':>11} {'union gain':>11}")
        deltas, per_draw = [], []
        for s in shared_seeds:
            a, b = solved_by_arm[base][s], solved_by_arm[treat][s]
            only_a, only_b = discordance(a, b)
            gain = union_gain(a, b)
            delta = len(b) - len(a)
            deltas.append(float(delta))
            per_draw.append({"seed": s, "proved": [len(a), len(b)], "delta": delta,
                             "only_baseline": only_a, "only_treatment": only_b,
                             "discordant": only_a + only_b, "union_gain": gain})
            print(f"  {s:>5} {f'{len(a)} vs {len(b)}':>14} {delta:>+7} "
                  f"{f'{only_a}+{only_b}={only_a + only_b}':>11} {gain:>11}")

        mean_delta, sd_delta = spread(deltas)
        print(f"\n  delta over {len(deltas)} draw(s): mean {mean_delta:+.2f}, "
              f"sd {fmt(sd_delta, '.2f')}")

        # Discordance and union gain use EVERY cross pair, not only matched seeds. Seed 1 of `sv`
        # and seed 1 of `li` are two independent draws with nothing pairing them, so restricting to
        # i == j would discard most of the available pairs for no gain. The within-arm floor is
        # built the same way, which is what makes the two comparable: both are two independent
        # draws, and only the between-arm figure additionally contains an architecture difference.
        discs, gains = [], []
        for s1 in sorted(solved_by_arm[base]):
            for s2 in sorted(solved_by_arm[treat]):
                a, b = solved_by_arm[base][s1], solved_by_arm[treat][s2]
                only_a, only_b = discordance(a, b)
                discs.append(float(only_a + only_b))
                gains.append(float(union_gain(a, b)))
        mean_disc, sd_disc = spread(discs)
        mean_gain, sd_gain = spread(gains)
        print(f"  over all {len(discs)} cross pairs: discordance {mean_disc:.1f} "
              f"(sd {fmt(sd_disc, '.1f')}), union gain {mean_gain:.1f} (sd {fmt(sd_gain, '.1f')})")

        # Claim 2: is the disagreement above what one arm does against itself?
        disc_verdict = ("ABOVE the noise floor" if mean_disc > max(floor_disc)
                        else "INSIDE the noise floor")
        print(f"  discordance {mean_disc:.1f} against a floor of {min(floor_disc)}-"
              f"{max(floor_disc)}: {disc_verdict}")
        # Claim 3: is the union gain above it?
        gain_verdict = ("ABOVE the noise floor" if mean_gain > max(floor_union)
                        else "INSIDE the noise floor")
        print(f"  union gain  {mean_gain:.1f} against a floor of {min(floor_union)}-"
              f"{max(floor_union)}: {gain_verdict}")

        # Claim 1: the powered estimand — each problem's solve *rate* across draws.
        base_rates = solve_rate_map([merged[base][s] for s in sorted(solved_by_arm[base])])
        treat_rates = solve_rate_map([merged[treat][s] for s in sorted(solved_by_arm[treat])])
        pids = sorted(set(base_rates) & set(treat_rates))
        d = np.array([treat_rates[p] - base_rates[p] for p in pids], dtype=np.float64)
        rng = np.random.default_rng(args.seed)
        lo, hi = bootstrap_ci(d, args.n_boot, rng)
        p_perm = permutation_p(d, args.n_perm, rng)
        significant = bool((lo > 0 or hi < 0) and p_perm < 0.05)
        borderline = bool((lo > 0 or hi < 0) != (p_perm < 0.05))
        print(f"\n  paired on per-problem solve RATE over {len(pids)} problems "
              f"(the estimand a single draw approximates):")
        print(f"    mean difference {float(d.mean()):+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
              f"p = {p_perm:.4f}")
        verdict = ("SIGNIFICANT" if significant
                   else "borderline" if borderline else "not significant")
        print(f"    {verdict}")

        comparisons.append({
            "contrast": f"{treat} vs {base}",
            "per_draw": per_draw,
            "mean_delta": mean_delta, "sd_delta": sd_delta,
            "n_cross_pairs": len(discs),
            "mean_discordance": mean_disc, "sd_discordance": sd_disc,
            "discordance_above_floor": mean_disc > max(floor_disc),
            "mean_union_gain": mean_gain, "sd_union_gain": sd_gain,
            "union_gain_above_floor": mean_gain > max(floor_union),
            "rate_comparison": {
                "n_problems": len(pids), "mean_difference": float(d.mean()),
                "ci95": [lo, hi], "p_permutation": p_perm,
                "significant": significant, "borderline": borderline,
            },
        })

    print("\nHow to read this. The delta's sd says whether +0 was a stable measurement or one draw")
    print("of a noisy quantity. The two floor comparisons decide whether 'equal counts, different")
    print("theorems' and the fusion ceiling are architecture effects or resampling headroom — if a")
    print("single arm re-run against itself flips as many problems as the two arms do, those two")
    print("claims describe the sampler and not the retrievers.")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({
        "draws": [{"benchmark": b, "arm": a, "seed": d.seed, "run_id": d.run_id,
                   "n_attempted": len(d.attempted), "n_proved": len(d.solved)}
                  for (b, a), ds in sorted(groups.items()) for d in ds],
        "tripwire": tripwire,
        "noise_floor": {"within_arm": within,
                        "discordance_range": [min(floor_disc), max(floor_disc)],
                        "union_gain_range": [min(floor_union), max(floor_union)]},
        "comparisons": comparisons,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.json_out}")
    return 0


def main() -> int:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", required=True,
                    help="results/logs/<run_id>; repeat for every arm and every seed. Pass the "
                         "CONTROL arm's runs first — arm order fixes the sign of every delta")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the resampling tests — not a sampling draw")
    ap.add_argument("--json-out", type=Path,
                    default=Path("results/tables/replication_variance.json"))
    args = ap.parse_args()
    if len(args.run) < 2:
        raise SystemExit("--run at least twice")
    return report(group_draws(args.run), args)


if __name__ == "__main__":
    raise SystemExit(main())
