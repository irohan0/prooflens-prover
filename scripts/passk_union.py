#!/usr/bin/env python
"""pass@k for an ensemble of retrieval arms — the number the paper turns on.

The pass@8 sweep, verified against the population actually on disk:

    python scripts/passk_union.py --benchmark proofnet_test \\
        --match search.samples_per_step=32 \\
        --match policy_config.premise_free_fraction=0.25 \\
        --match n_problems=186                       # 141 for fate_m

**All three matches are load-bearing, and `n_problems` is the one that looks redundant.** The
budget pilot included a 60-problem run at exactly 64x32 with premise-free 0.25 — the winning config,
measured on a subset to choose it — at seed 0, which is also a sweep seed. The first two filters
therefore select 17 runs and `group()` refuses them for a duplicated (li, 0): correct, but the
refusal is the only thing standing between a subset run and the headline table.

## What is being reported, and why it is not an oracle

Tier 1 measured single-vector and late interaction at 74 problems each with a **union of 88**. That
union was reported as a *ceiling*, because picking the winning retriever per problem requires
knowing the answer. **This script reports something different and legitimate:** run every arm at
every seed, accept any proof any of them found, and re-elaborate all of them. Nothing here needs to
know in advance which arm will win — it is the same kind of object as REAL-Prover's published
Pass@64, which is also a union over many attempts.

## The estimator

Naively taking the union of the k runs you happen to have overstates pass@k, because you get to
count the lucky ordering. The standard unbiased estimator (Chen et al. 2021, §2.1) averages over
every k-subset in closed form:

    pass@k = 1 - C(K - c, k) / C(K, k)

for a problem solved in `c` of `K` seeds. At c = 0 it is 0, at c = K it is 1, and in between it is
the probability that a random draw of k seeds contains at least one that worked.

**Seeds are the unit of the draw, not (arm, seed) pairs.** The deployed system runs *both* arms at
each seed, so drawing k seeds means paying for 2k attempts — and the arms are not interchangeable
samples from one distribution, which is what the estimator would assume if they were pooled. So `c`
counts seeds at which *any* arm solved the problem, and the budget column multiplies by the number
of arms.

## The budget column is not decoration

REAL-Prover's published 23.7 / 56.7 are Pass@64x64 — about 4.19M generations per problem. Every rate
here is printed next to its own generations-per-problem and the ratio to theirs, because a rate
without a budget is not comparable to anything, and this project's claim is specifically about
reaching their numbers on a fraction of their compute.

## Refusals

Three things are refused rather than warned about, because each silently inflates the headline:

* **a duplicated (arm, seed)** — the same draw counted twice, which raises `c` without adding
  evidence;
* **configs that differ in anything but the seed** — then it is not one system measured k times.
  `lean_project` is exempt: node-local staging embeds `/tmp/slurm.<jobid>` in the path;
* **a run with no `verification.json`, or an incomplete one** — an unverified proof is a claim, and
  this script exists to produce a number for publication.

A run whose report *records failures* is **not** refused, and that is a deliberate change from the
first version of this script. A claimed proof that does not re-elaborate is discounted — it never
enters `solved`, in `eval/draws.py`, for every consumer at once — and the run is otherwise kept.

Refusing the whole run reads as the stricter choice and is not. Measured on ProofNet / sv / seed 6:
34 of its 35 proofs verified, and it holds the joint-highest count of its arm. Discarding it would
remove a high seed from an eight-seed estimate, which moves the arm further from the truth than the
single claim being corrected — a bias dressed as rigour. Discounting is strictly conservative: it
can only lower a rate. The count of discounted claims is printed with every table, never folded in
silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.eval.draws import Draw, load_draw  # noqa: E402
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: Config keys allowed to differ between two runs of the same arm. `lean_project` varies because
#: node-local staging writes the project under `/tmp/slurm.<jobid>` — a difference that is the
#: *point* of staging, not a difference in the experiment.
SEED_VARYING = frozenset({"lean_project"})

#: Config keys that must match across *different* arms too. The rest are expected to differ — that
#: is what an arm is.
SHARED_ACROSS_ARMS = ("benchmark", "search", "n_problems")

#: REAL-Prover's published configuration, written as its three factors because collapsing them is
#: how it gets wrong. "Pass@64x64" is **64 passes of their `large` variant**, and `large` is
#: `MAX_NODES=1024`, `NUM_SAMPLES=64` (their `conf/config.py`, commented out) — not 64 nodes. The
#: obvious reading, 64 x 64 x 64, is 262,144 and understates their budget by 16x, which would
#: inflate every "fraction of their compute" claim this script prints by the same factor.
#: `SearchConfig.large()` in `prover/search.py` carries the same three numbers.
PUBLISHED_PASSES, PUBLISHED_NODES, PUBLISHED_SAMPLES = 64, 1024, 64
PUBLISHED_BUDGET = PUBLISHED_PASSES * PUBLISHED_NODES * PUBLISHED_SAMPLES  # 4,194,304
PUBLISHED: dict[str, dict[str, float]] = {
    "proofnet_test": {"REAL-Prover-v1 (Pass@64x64)": 23.7, "ReProver (<1B, 1 pass)": 13.8},
    "fate_m": {"REAL-Prover-v1 (Pass@64x64)": 56.7},
    "minif2f_test": {"REAL-Prover-v1": 54.1},
}


def pass_at_k(n_seeds: int, n_solved_seeds: int, k: int) -> float:
    """Unbiased pass@k for one problem: `1 - C(K-c, k)/C(K, k)`."""
    if k > n_seeds:
        raise ValueError(f"pass@{k} needs at least {k} seeds, have {n_seeds}")
    if n_seeds - n_solved_seeds < k:
        return 1.0
    return 1.0 - comb(n_seeds - n_solved_seeds, k) / comb(n_seeds, k)


def config_value(cfg: dict, dotted: str):
    """`cfg["search"]["samples_per_step"]` for `"search.samples_per_step"`, or None."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def matches(cfg: dict, specs: list[str]) -> bool:
    """True if every `key=value` holds. Compared as strings, so `32` matches `"32"`."""
    for spec in specs:
        key, _, want = spec.partition("=")
        if str(config_value(cfg, key)) != want:
            return False
    return True


def discover(results_root: Path, benchmark: str, policy: str,
             specs: list[str] | None = None) -> list[Path]:
    """Finalised run directories for one (benchmark, policy), any arm, any seed.

    `specs` narrows the set to one *system*, and is not optional in practice. Earlier runs of the
    same benchmark and policy sit in the same directory at a different search budget — the published
    Tier 1 table is 64 x 16, a sweep is 64 x 32 — and `group()` rightly refuses to average across
    them. Without a filter the refusal arrives only after the sweep has been paid for, so
    `--match search.samples_per_step=32` is how a sweep names itself.
    """
    out = []
    for d in sorted(results_root.iterdir()):
        mf = d / "manifest.json"
        if not d.is_dir() or not mf.exists():
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        cfg = m.get("config", {})
        if not m.get("outcome"):
            continue
        if cfg.get("benchmark") != benchmark or cfg.get("policy_kind") != policy:
            continue
        if specs and not matches(cfg, specs):
            continue
        out.append(d)
    return out


def arm_matches(label: str, wanted: list[str]) -> bool:
    """Whether a `Draw.arm` label is one of the requested arms.

    `Draw` appends the first-stage budget to any arm that recorded one, so late interaction appears
    as `li@50k` and the fusion arm — which inherits its sub-retriever's budget — as `fusion@50k`.
    Requiring the suffix would mean typing a number that is already pinned by `--match`, and getting
    it wrong yields an empty selection rather than an error. Both spellings are accepted; the bare
    name matches any budget, which is safe because mixing budgets is refused downstream anyway.
    """
    return any(label == w or label.startswith(f"{w}@") for w in wanted)


def parse_seeds(spec: str | None) -> set[int] | None:
    """`"0-3"` or `"0,1,2,3"` -> {0,1,2,3}. None means every seed found.

    Needed to compare arms measured at different depths: the sweep ran eight seeds per arm and a
    later arm may only have four, and `group()` rightly refuses a mismatched seed set rather than
    averaging over draws that were never made. Restricting the deeper arms is how the comparison is
    made at equal k, using runs that already exist.
    """
    if spec is None:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    if not out:
        raise SystemExit(f"--seeds {spec!r} selected no seeds")
    return out


def check_verified(run_dir: Path) -> str | None:
    """None if this run's proofs were independently re-checked at all, else why not.

    Recorded *failures* are not a reason to refuse: `eval/draws.py` discounts them, so a rejected
    claim never reaches `solved` and cannot inflate anything. What is refused is a run nobody
    checked, or a report too incomplete to say what was checked — because then the discount cannot
    be applied and a bad proof would be counted as good.
    """
    vf = run_dir / "verification.json"
    if not vf.exists():
        return "no verification.json — run scripts/verify_proofs.py on it"
    report = json.loads(vf.read_text(encoding="utf-8"))
    if "n_claimed" not in report or "n_verified" not in report:
        return (f"{vf.name} is not a complete report (no n_claimed/n_verified) — it was probably "
                "written by an interrupted verify. Re-run scripts/verify_proofs.py on it.")
    # A report listing fewer failures than it counted cannot be used to discount them.
    if report.get("n_failed", 0) != len(report.get("failures") or ()):
        return (f"{vf.name} counts {report.get('n_failed')} failure(s) but lists "
                f"{len(report.get('failures') or ())} — the rejected proofs cannot be identified, "
                "so they cannot be discounted. Re-run scripts/verify_proofs.py on it.")
    return None


def group(draws: list[Draw]) -> dict[str, dict[int, Draw]]:
    """`{arm: {seed: draw}}`, refusing a duplicated (arm, seed) and mismatched configs."""
    by_arm: dict[str, dict[int, Draw]] = {}
    for d in draws:
        seeds = by_arm.setdefault(d.arm, {})
        if d.seed in seeds:
            raise SystemExit(
                f"two runs of arm {d.arm!r} at seed {d.seed}: {seeds[d.seed].run_id} and "
                f"{d.run_id}. Counting one draw twice raises its solved-seed count without adding "
                f"evidence, which inflates every pass@k below it."
            )
        seeds[d.seed] = d

    for arm, seeds in by_arm.items():
        ref = next(iter(seeds.values()))
        for d in seeds.values():
            differing = {
                k for k in set(ref.config) | set(d.config)
                if k not in SEED_VARYING and ref.config.get(k) != d.config.get(k)
            }
            if differing:
                raise SystemExit(
                    f"arm {arm!r} runs {ref.run_id} and {d.run_id} differ in {sorted(differing)}. "
                    "pass@k assumes one system sampled k times; these are two systems."
                )

    ref_arm = next(iter(by_arm.values()))
    ref_draw = next(iter(ref_arm.values()))
    for arm, seeds in by_arm.items():
        d = next(iter(seeds.values()))
        for key in SHARED_ACROSS_ARMS:
            if ref_draw.config.get(key) != d.config.get(key):
                raise SystemExit(
                    f"arm {arm!r} ran with {key}={d.config.get(key)!r} against "
                    f"{ref_draw.config.get(key)!r} elsewhere. The arms are not comparable, so "
                    "their union is not one system's result."
                )
    return by_arm


def coverage_curve(
    solved_seed_counts: dict[str, int], n_seeds: int, n_problems: int
) -> list[float]:
    """`[pass@1, …, pass@n_seeds]` as fractions of `n_problems`."""
    return [
        sum(pass_at_k(n_seeds, solved_seed_counts.get(p, 0), k)
            for p in solved_seed_counts) / n_problems
        for k in range(1, n_seeds + 1)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--results-root", type=Path, default=Path("results/logs"))
    ap.add_argument("--policy", default="vllm")
    ap.add_argument("--run", type=Path, action="append",
                    help="explicit run directory; repeatable. Overrides discovery")
    ap.add_argument("--arm", action="append",
                    help="restrict the ensemble to these arms; repeatable. The bare name matches "
                         "any first-stage budget, so `--arm li` selects `li@50k`")
    ap.add_argument("--seeds", default=None, metavar="SPEC",
                    help="restrict to these seeds, e.g. 0-3 or 0,1,2,3. Use it to compare an arm "
                         "measured at four seeds against ones measured at eight, at equal k")
    ap.add_argument("--match", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted config key, repeatable, restricting discovery to one system. "
                         "e.g. --match search.samples_per_step=32. Needed whenever runs at another "
                         "budget share the results directory, which is the normal case")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="count runs whose proofs were never re-elaborated. For a work-in-progress "
                         "look only — never for a reported number")
    ap.add_argument("--out", type=Path, help="write the full result as JSON")
    args = ap.parse_args()
    ensure_utf8_output()

    run_dirs = args.run or discover(args.results_root, args.benchmark, args.policy, args.match)
    if not run_dirs:
        raise SystemExit(
            f"no finalised {args.policy} runs for {args.benchmark} under {args.results_root}"
            + (f" matching {args.match}" if args.match else "")
        )

    want_seeds = parse_seeds(args.seeds)
    problems: list[str] = []
    draws: list[Draw] = []
    for d in run_dirs:
        draw = load_draw(d)
        # Selection first, verification second. A run that is not being counted cannot inflate
        # anything, so demanding its verification would block a report over other runs entirely —
        # which is exactly the situation when a newly-added arm is still being checked.
        if args.arm and not arm_matches(draw.arm, args.arm):
            continue
        if want_seeds is not None and draw.seed not in want_seeds:
            continue
        if (why := check_verified(d)) is not None:
            if not args.allow_unverified:
                raise SystemExit(
                    f"{d.name}: {why}.\nA pass@k figure built on unverified proofs is a claim, not "
                    "a result. Re-run slurm/verify_proofs.sbatch, or pass --allow-unverified for a "
                    "work-in-progress look."
                )
            print(f"  WARNING unverified: {d.name} ({why})")
        draws.append(draw)
        problems = problems or sorted(draw.attempted)

    if not draws:
        raise SystemExit(
            f"every discovered run was filtered out by --arm {args.arm} / --seeds {args.seeds}. "
            "Arms present: "
            f"{sorted({load_draw(d).arm for d in run_dirs})}"
        )

    by_arm = group(draws)
    seed_sets = {arm: set(s) for arm, s in by_arm.items()}
    if len({frozenset(s) for s in seed_sets.values()}) != 1:
        raise SystemExit(
            f"arms ran at different seeds: { {a: sorted(s) for a, s in seed_sets.items()} }. "
            "An ensemble draw is one seed run through every arm; a missing arm at some seed would "
            "make the k-subset average count draws that were never actually made."
        )

    seeds = sorted(next(iter(seed_sets.values())))
    n_seeds, arms = len(seeds), sorted(by_arm)
    n_problems = len(problems)
    search = next(iter(next(iter(by_arm.values())).values())).config.get("search", {})
    per_attempt = search.get("max_expansions", 0) * search.get("samples_per_step", 0)

    print(f"=== {args.benchmark}: pass@k over {n_seeds} seeds x {len(arms)} arms "
          f"({n_problems} problems) ===")
    print(f"  arms   : {', '.join(arms)}")
    print(f"  seeds  : {seeds}")
    print(f"  budget : {search.get('max_expansions')} nodes x "
          f"{search.get('samples_per_step')} samples = {per_attempt:,} generations per attempt")

    # Printed, never folded in silently: these are claims the independent re-check rejected, so
    # every rate below is over a solved set smaller than the runs' own manifests report. A reader
    # comparing this table to `n_proved` in a manifest has to be able to see why they differ.
    discounted = {d.run_id: sorted(d.discounted)
                  for arm in arms for d in by_arm[arm].values() if d.discounted}
    if discounted:
        total = sum(len(v) for v in discounted.values())
        print(f"  !! {total} claimed proof(s) DISCOUNTED after failing re-elaboration, in "
              f"{len(discounted)} of {n_seeds * len(arms)} runs. Counted as unsolved:")
        for run_id, pids in sorted(discounted.items()):
            print(f"       {run_id}: {', '.join(pids)}")
    print()

    # Per-problem: how many seeds solved it, per arm and for the ensemble.
    per_arm_counts = {
        arm: {p: sum(1 for s in seeds if p in by_arm[arm][s].solved) for p in problems}
        for arm in arms
    }
    ensemble_counts = {
        p: sum(1 for s in seeds if any(p in by_arm[a][s].solved for a in arms)) for p in problems
    }

    header = "  " + "arm".ljust(12) + "".join(f"pass@{k}".rjust(10) for k in range(1, n_seeds + 1))
    print(header)
    rows: dict[str, list[float]] = {}
    for arm in arms:
        curve = coverage_curve(per_arm_counts[arm], n_seeds, n_problems)
        rows[arm] = curve
        print("  " + arm.ljust(12) + "".join(f"{100 * c:9.1f}%" for c in curve))
    ens = coverage_curve(ensemble_counts, n_seeds, n_problems)
    rows["ENSEMBLE"] = ens
    print("  " + "ENSEMBLE".ljust(12) + "".join(f"{100 * c:9.1f}%" for c in ens))

    solved_any = {p for p in problems if ensemble_counts[p] > 0}
    total_budget = n_seeds * len(arms) * per_attempt
    print(f"\n  ensemble at k={n_seeds}: {len(solved_any)}/{n_problems} "
          f"({100 * len(solved_any) / n_problems:.1f}%)")
    print(f"  generations per problem: {total_budget:,}  "
          f"= 1/{PUBLISHED_BUDGET / total_budget:.0f} of REAL-Prover's Pass@64x64 "
          f"({PUBLISHED_BUDGET:,})")

    if published := PUBLISHED.get(args.benchmark):
        print("\n  published reference (different budgets — see above):")
        for name, rate in published.items():
            delta = 100 * len(solved_any) / n_problems - rate
            print(f"    {name:<34} {rate:5.1f}%   ours {delta:+5.1f} pts")

    print("\n  pass@k is the unbiased 1 - C(K-c,k)/C(K,k) over seed subsets, not the union of the "
          "k runs\n  that happened to be first — that would count the lucky ordering.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "benchmark": args.benchmark,
            "arms": arms,
            "seeds": seeds,
            "n_problems": n_problems,
            "search": search,
            "generations_per_attempt": per_attempt,
            "generations_per_problem": total_budget,
            "published_budget": PUBLISHED_BUDGET,
            "curves": rows,
            "ensemble_solved": sorted(solved_any),
            "n_ensemble_solved": len(solved_any),
            "runs": [d.run_id for d in draws],
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
