#!/usr/bin/env python
"""Characterise the problems exactly one arm solved, and how the other arm failed on them.

    python scripts/discordance_profile.py \
        --a results/logs/fate_m_sv_vllm_... \
        --b results/logs/fate_m_li_vllm_... \
        --control results/logs/fate_m_none_vllm_... \
        --data-root <REAL-Prover>/data

## The question

Tier 1 reports an exact tie: each retriever proves the same number of theorems, but they disagree
about *which*. A count says nothing about whether the two arms are interchangeable. The discussion
needs the next question answered: **is one architecture winning a recognisably different kind of
problem, and why did the other one fail there?**

## What is measured, and why these quantities

For each problem exactly one arm solved, this reports the **loser's terminal status**, which is the
sharpest available statement of *why* it failed:

* `no_candidates` — the policy had nothing usable left to propose. The loser ran out of *ideas*, so
  the winner's retrieval supplied a tactic that did not otherwise exist. Call this a **rescue**.
* `max_expansions` — the loser had plenty to try and spent its whole budget trying it. It ran out
  of *budget in the wrong part of the tree*, so the winner's retrieval changed the search **order**,
  not the option set. Call this a **redirect**.
* `error` / wall clock — the loser never really attempted the problem; it is not evidence about
  retrieval either way and is reported separately rather than folded in.

That distinction matters for the paper, because the two imply different fixes. Rescues say the
retriever's *recall* is doing the work and a better candidate generator should help. Redirects say
its *ranking* is doing the work, and rank fusion should help.

It also reports how hard each win was — expansions, tactics tried, wall clock, proof length — so a
claim like "LI solves harder problems" can be checked rather than asserted, and the mathematical
area of each problem when `--data-root` is given.

## The sample sizes forbid significance testing

The exclusive-win sets are 10 and 10 on FATE-M and 4 and 4 on ProofNet. **Nothing in this output can
be significant, and no p-value is printed.** It is descriptive and hypothesis-generating: its job is
to say what to measure next at a sample size that could settle it, not to settle it here.

Reads `attempts.jsonl` only. No GPU, no model, no Lean.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.eval.compare import format_budget  # noqa: E402
from prooflens_prover.utils.io import read_jsonl  # noqa: E402
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: Mathematical area, inferred from the typeclasses and constants a *statement* mentions. Ordered:
#: the first pattern that matches wins, so the more specific areas come first. This is a coarse
#: label for grouping a dozen problems in a discussion section, not a taxonomy -- a statement about
#: a finite group acting on a topological space will be called group theory, and that is fine.
AREAS: tuple[tuple[str, str], ...] = (
    ("group theory", r"\b(Group|Subgroup|Sylow|orderOf|Perm|conj|normalizer|centralizer|"
                     r"IsCyclic|Solvable|QuotientGroup|MonoidHom|comm_group)\b"),
    ("ring / field", r"\b(Ring|Ideal|Field|Polynomial|IsDomain|IsUnit|IsNilpotent|Algebra|"
                     r"CharP|Subring|RingHom|IsPrime|IsIntegral|IsAlgebraic)\b"),
    ("linear algebra", r"\b(Module|Submodule|LinearMap|Matrix|Basis|finrank|Eigenvalue|"
                       r"VectorSpace|LinearIndependent)\b"),
    ("topology", r"\b(Topological|IsOpen|IsClosed|Continuous|Compact|Metric|Filter|"
                 r"nhds|IsConnected|Homeomorph)\b"),
    ("analysis", r"\b(Real|Complex|Tendsto|deriv|integral|Summable|HasSum|Cauchy|"
                 r"MeasureTheory|norm|abs)\b"),
    ("number theory", r"\b(Nat\.Prime|ZMod|Int\.ModEq|totient|divisors|padic|Nat\.gcd|"
                      r"Nat\.Coprime|legendreSym)\b"),
    ("order / lattice", r"\b(Lattice|PartialOrder|sSup|sInf|Galois|OrderIso|IsGLB|IsLUB)\b"),
    ("set / logic", r"\b(Set\.|Finset|Function\.Injective|Function\.Surjective|Equiv|Cardinal)\b"),
)

#: The head of a tactic, for a coarse profile of what the winning proofs are made of.
TACTIC_HEAD = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_']*)")

#: A whole proof that is a single citation of a named premise -- see `contamination_audit.py`.
ONE_STEP = re.compile(r"^(?:exact|apply)\s+([A-Za-z_][A-Za-z0-9_.'!?₀-₉¹²³]*)")
NOT_A_PREMISE = frozenset({"fun", "by", "this", "if", "let", "have", "show"})


def classify_area(statement: str) -> str:
    for name, pattern in AREAS:
        if re.search(pattern, statement):
            return name
    return "other"


def read_run(run_dir: Path) -> tuple[str, str, dict[str, dict]]:
    """`(benchmark, arm label, {problem id: attempt row})`."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest.get("config", {})
    n_candidates = cfg.get("n_candidates")
    arm = cfg.get("arm", "?")
    label = f"{arm}@{format_budget(n_candidates)}" if n_candidates else arm

    rows: dict[str, dict] = {}
    for row in read_jsonl(run_dir / "attempts.jsonl"):
        rows[str(row["problem_id"])] = row
    return cfg.get("benchmark", "?"), label, rows


def failure_mode(row: dict) -> str:
    """Why this arm did not prove this problem, in the terms the discussion needs."""
    status = row.get("status")
    if status == "no_candidates":
        return "no_candidates (nothing left to propose)"
    if status == "error":
        return "error (never really attempted)"
    if status == "exhausted":
        hit = row.get("limit_hit")
        if hit == "max_expansions":
            return "max_expansions (searched hard, wrong direction)"
        if hit == "wall_clock":
            return "wall_clock"
        return f"exhausted ({hit})"
    return str(status)


def proof_steps(row: dict) -> list[str]:
    return [s.strip() for s in (row.get("proof") or ()) if str(s).strip()]


def median(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def profile(winner_rows, loser_rows, ids, statements) -> dict:
    """Describe one arm's exclusive wins and the other arm's failures on the same problems."""
    modes = Counter(failure_mode(loser_rows[p]) for p in ids if p in loser_rows)
    heads: Counter = Counter()
    one_step = 0
    for p in ids:
        steps = proof_steps(winner_rows[p])
        for s in steps:
            m = TACTIC_HEAD.match(s)
            if m:
                heads[m.group(1)] += 1
        if len(steps) == 1:
            m = ONE_STEP.match(steps[0])
            if m and m.group(1) not in NOT_A_PREMISE:
                one_step += 1

    areas = Counter(
        classify_area(statements[p]) for p in ids if p in statements
    ) if statements else Counter()

    return {
        "n": len(ids),
        "problems": sorted(ids),
        "loser_failure_modes": dict(modes),
        "winner_median_expansions": median(winner_rows[p].get("n_expansions") for p in ids),
        "winner_median_tactics_tried": median(winner_rows[p].get("n_tactics_tried") for p in ids),
        "winner_median_elapsed_s": median(winner_rows[p].get("elapsed_s") for p in ids),
        "winner_median_proof_steps": median(len(proof_steps(winner_rows[p])) for p in ids),
        "loser_median_expansions": median(
            loser_rows[p].get("n_expansions") for p in ids if p in loser_rows),
        "one_step_corpus_citations": one_step,
        "tactic_heads": dict(heads.most_common(8)),
        "areas": dict(areas.most_common()),
    }


def baseline(rows: dict[str, dict]) -> dict:
    """The same difficulty measures over *every* problem this arm proved, as a reference point."""
    solved = [r for r in rows.values() if r.get("proved")]
    return {
        "n_solved": len(solved),
        "median_expansions": median(r.get("n_expansions") for r in solved),
        "median_tactics_tried": median(r.get("n_tactics_tried") for r in solved),
        "median_elapsed_s": median(r.get("elapsed_s") for r in solved),
        "median_proof_steps": median(len(proof_steps(r)) for r in solved),
    }


def print_profile(label, other, prof, base) -> None:
    print(f"\n--- {prof['n']} problems only {label} solved "
          f"(and what {other} did on them) ---")
    if not prof["n"]:
        print("  none")
        return
    print(f"  problems: {', '.join(prof['problems'])}")

    print(f"\n  why {other} failed there:")
    for mode, n in sorted(prof["loser_failure_modes"].items(), key=lambda kv: -kv[1]):
        print(f"    {n:>3}  {mode}")

    print(f"\n  how hard the win was ({label}, median — its own all-solved median in brackets):")
    print(f"    expansions      {prof['winner_median_expansions']}"
          f"  [{base['median_expansions']}]")
    print(f"    tactics tried   {prof['winner_median_tactics_tried']}"
          f"  [{base['median_tactics_tried']}]")
    print(f"    wall clock (s)  {prof['winner_median_elapsed_s']}"
          f"  [{base['median_elapsed_s']}]")
    print(f"    proof steps     {prof['winner_median_proof_steps']}"
          f"  [{base['median_proof_steps']}]")
    print(f"    {other} spent {prof['loser_median_expansions']} expansions failing")

    if prof["areas"]:
        print("\n  mathematical area:")
        for area, n in prof["areas"].items():
            print(f"    {n:>3}  {area}")
    print(f"\n  one-step corpus citations: {prof['one_step_corpus_citations']} of {prof['n']}")
    print(f"  tactic heads: {', '.join(f'{k}×{v}' for k, v in prof['tactic_heads'].items())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="first arm's run directory")
    ap.add_argument("--b", type=Path, required=True, help="second arm's run directory")
    ap.add_argument("--control", type=Path, help="the no-retrieval run, to flag non-exclusive wins")
    ap.add_argument("--data-root", type=Path,
                    help="REAL-Prover data dir; enables the mathematical-area breakdown")
    ap.add_argument("--out", type=Path, help="write the full result as JSON")
    args = ap.parse_args()
    ensure_utf8_output()

    bench_a, arm_a, rows_a = read_run(args.a)
    bench_b, arm_b, rows_b = read_run(args.b)
    if bench_a != bench_b:
        raise SystemExit(f"different benchmarks: {bench_a} vs {bench_b}")
    if arm_a == arm_b:
        raise SystemExit(f"both runs are arm {arm_a!r}; this compares two different arms")

    statements: dict[str, str] = {}
    if args.data_root:
        from prooflens_prover.data.benchmarks import load_benchmark
        for prob in load_benchmark(bench_a, args.data_root):
            statements[prob.id] = prob.declaration

    solved_a = {p for p, r in rows_a.items() if r.get("proved")}
    solved_b = {p for p, r in rows_b.items() if r.get("proved")}
    only_a, only_b = solved_a - solved_b, solved_b - solved_a

    print(f"=== {bench_a}: {arm_a} vs {arm_b} ===")
    print(f"  {arm_a:<8} {len(solved_a)} proved   {arm_b:<8} {len(solved_b)} proved   "
          f"union {len(solved_a | solved_b)}   discordant {len(only_a) + len(only_b)}")
    if not statements:
        print("  (no --data-root: mathematical-area breakdown unavailable)")

    if args.control:
        _, _, rows_c = read_run(args.control)
        solved_c = {p for p, r in rows_c.items() if r.get("proved")}
        print(f"  of {arm_a}'s {len(only_a)} exclusive wins, {len(only_a & solved_c)} were also "
              f"solved by the control; of {arm_b}'s {len(only_b)}, {len(only_b & solved_c)}")

    prof_a = profile(rows_a, rows_b, only_a, statements)
    prof_b = profile(rows_b, rows_a, only_b, statements)
    base_a, base_b = baseline(rows_a), baseline(rows_b)
    print_profile(arm_a, arm_b, prof_a, base_a)
    print_profile(arm_b, arm_a, prof_b, base_b)

    print("\n  NOTE: these sets are too small for a significance test and none is computed. "
          "This is descriptive.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "benchmark": bench_a,
            "arms": [arm_a, arm_b],
            "solved": {arm_a: len(solved_a), arm_b: len(solved_b)},
            "union": len(solved_a | solved_b),
            f"only_{arm_a}": prof_a,
            f"only_{arm_b}": prof_b,
            "all_solved_baseline": {arm_a: base_a, arm_b: base_b},
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
