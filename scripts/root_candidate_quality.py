#!/usr/bin/env python
"""Candidate quality at the ROOT state, where every arm sees the identical theorem.

    python scripts/root_candidate_quality.py \
        --run results/logs/fate_m_none_vllm_... \
        --run results/logs/fate_m_sv_vllm_... \
        --run results/logs/fate_m_li_vllm_...

## Why the root state specifically

`PolicyStats.mean_candidate_logprob` averages over every expansion of a run, and the arms **diverge
after their first tactic** — a later state reached by the LI arm need not exist in the SV arm's tree
at all. So those run-level averages describe different distributions of proof states and cannot be
compared, however tempting the ordering looks (measured: li -0.810, sv -0.835, none -0.887 on
FATE-M, and the same ordering on ProofNet).

At depth 0 there is no divergence yet. Every arm is prompted with the same benchmark statement and
differs *only* in the premises attached to it. Comparing candidate log-probabilities there is a
properly paired measurement — same problem, same goal, one variable — of what retrieval does to the
generator itself.

## What it is for

Tier 1 found that LI and SV prove *identical* numbers of theorems (+0 over 327 problems, 17 gained
and 17 lost, p = 1.000) while retrieval as such helps significantly (+14, p = 0.004). A null is far
more useful with a mechanism attached, and this tests the candidate mechanism: does better retrieval
produce measurably better-conditioned generations that simply fail to convert into proofs?

Reads `attempts.jsonl` only. No GPU, no model, no re-run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.eval.compare import (  # noqa: E402
    bootstrap_ci,
    format_budget,
    permutation_p,
)
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: The depth recorded for tactics proposed at the root. `best_first_search` writes the *parent's*
#: depth on each trace row, so the root's own candidates carry 0.
ROOT_DEPTH = 0


def root_candidates(run_dir: Path) -> tuple[str, dict[str, list[float]]]:
    """`(arm label, {problem id: [logprob of each root candidate]})` for one run.

    The recorded `logprob` is the per-token mean the search ranks on — see
    `vllm_policy.Generation.mean_logprob` — not a cumulative sum, so values from tactics of
    different lengths are directly comparable.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest.get("config", {})
    arm = cfg.get("arm", "?")
    n_candidates = cfg.get("n_candidates")
    label = f"{arm}@{format_budget(n_candidates)}" if n_candidates else arm

    per_problem: dict[str, list[float]] = {}
    for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        logprobs = [
            float(t["logprob"])
            for t in row.get("trace") or []
            if t.get("depth") == ROOT_DEPTH
            and t.get("logprob") is not None
            and math.isfinite(float(t["logprob"]))
        ]
        if logprobs:
            per_problem[str(row["problem_id"])] = logprobs
    return label, per_problem


def paired(a: dict[str, list[float]], b: dict[str, list[float]], stat,
           n_boot: int, n_perm: int, seed: int) -> dict:
    """Paired comparison of a per-problem statistic over the problems both runs reached."""
    pids = sorted(set(a) & set(b))
    d = np.array([stat(b[p]) - stat(a[p]) for p in pids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    lo, hi = bootstrap_ci(d, n_boot, rng)
    p_perm = permutation_p(d, n_perm, rng)
    return {
        "n_problems": len(pids),
        "mean_difference": float(d.mean()),
        "ci95": [lo, hi],
        "p_permutation": p_perm,
        # The project's rule: agreement required, or the honest label is `borderline`.
        "significant": bool((lo > 0 or hi < 0) and p_perm < 0.05),
        "borderline": bool((lo > 0 or hi < 0) != (p_perm < 0.05)),
    }


def main() -> int:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", required=True,
                    help="results/logs/<run_id>; repeat, control first")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if len(args.run) < 2:
        raise SystemExit("--run at least twice; the first is the control")

    arms = [root_candidates(d) for d in args.run]
    shared = set.intersection(*(set(p) for _, p in arms))
    if not shared:
        raise SystemExit("the runs share no problem with root-state candidates")

    print(f"=== root-state candidate quality over {len(shared)} shared problems ===")
    print("Every arm is prompted with the identical theorem statement here, so this is the one")
    print("place candidate quality can be compared across arms without confounding it with the")
    print("different proof states each arm goes on to explore.\n")
    print(f"  {'arm':<12} {'problems':>9} {'candidates':>11} {'mean logprob':>13} "
          f"{'best logprob':>13}")
    summary = []
    for label, per_problem in arms:
        vals = [v for p, v in per_problem.items() if p in shared]
        flat = [x for v in vals for x in v]
        means = float(np.mean(flat))
        bests = float(np.mean([max(v) for v in vals]))
        print(f"  {label:<12} {len(vals):>9} {len(flat):>11} {means:>13.4f} {bests:>13.4f}")
        summary.append({"arm": label, "n_problems": len(vals), "n_candidates": len(flat),
                        "mean_logprob": round(means, 4), "mean_best_logprob": round(bests, 4)})

    control_label, control = arms[0]
    comparisons = []
    print(f"\n  paired against {control_label!r}, per problem:\n")
    print(f"  {'contrast':<22} {'stat':<6} {'mean diff':>10} {'95% CI':>24} {'p':>8}  verdict")
    for label, per_problem in arms[1:]:
        for name, stat in (("mean", lambda v: float(np.mean(v))), ("best", max)):
            r = paired(control, per_problem, stat, args.n_boot, args.n_perm, args.seed)
            verdict = ("SIGNIFICANT" if r["significant"]
                       else "borderline" if r["borderline"] else "not significant")
            ci = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
            print(f"  {label + ' vs ' + control_label:<22} {name:<6} "
                  f"{r['mean_difference']:>+10.4f} {ci:>24} {r['p_permutation']:>8.4f}  {verdict}")
            comparisons.append({"contrast": f"{label} vs {control_label}",
                                "statistic": name, **r})

    # The arms that matter most to each other: every ordered pair, so li-vs-sv appears too.
    if len(arms) > 2:
        print("\n  and between the retrieval arms:\n")
        for i in range(1, len(arms)):
            for j in range(i + 1, len(arms)):
                for name, stat in (("mean", lambda v: float(np.mean(v))), ("best", max)):
                    r = paired(arms[i][1], arms[j][1], stat, args.n_boot, args.n_perm, args.seed)
                    verdict = ("SIGNIFICANT" if r["significant"]
                               else "borderline" if r["borderline"] else "not significant")
                    ci = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
                    contrast = f"{arms[j][0]} vs {arms[i][0]}"
                    print(f"  {contrast:<22} {name:<6} {r['mean_difference']:>+10.4f} "
                          f"{ci:>24} {r['p_permutation']:>8.4f}  {verdict}")
                    comparisons.append({"contrast": contrast, "statistic": name, **r})

    print()
    print("Read alongside the proof counts. A retrieval arm that is measurably better conditioned")
    print("here and proves no more theorems is evidence that the retrieval difference is real and")
    print("sits below the resolution of the prover — which is precisely the claim the predecessor")
    print("study could not test, having never run one.")

    out = args.json_out or Path("results/tables/root_candidate_quality.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arms": summary, "comparisons": comparisons}, indent=2),
                   encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
