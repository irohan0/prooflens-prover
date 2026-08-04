"""Paired comparison of two retrieval arms over one benchmark.

## Why this is a module and not a notebook cell

Every number in the write-up must be recomputable from `attempts.jsonl` alone. The first FATE-M
numbers I reported — "73% premise utilisation" — were read off the proof text by eye, and the
control run then showed that figure overstated retrieval's contribution by 60%: six proofs that
*used* a premise did not *need* one, because the baseline closed the same goals with a bare `simp`.
Eyeballing cannot catch that. A paired set difference can.

## What "significant" means here

Ported from the predecessor project's `scripts/significance.py`, including the rule it arrived at
the hard way: report **significant** only when the bootstrap CI excludes 0 **and** the permutation
p < 0.05. When the two disagree the honest label is `borderline`, not whichever one is nicer.

For a binary outcome the exact McNemar test is also reported, because with a paired 0/1 metric it
is the textbook test and needs no resampling — it depends only on the discordant pairs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from prooflens_prover.prover.repertoire import DEFAULT_CLOSERS

#: Statuses meaning the harness failed rather than the search: the problem was not really attempted.
HARNESS_ERROR_STATUSES = frozenset({"error"})


def is_premise_tactic(tactic: str) -> bool:
    """True if `tactic` names a retrieved premise rather than being a bare repertoire closer.

    Exact, not a heuristic: `RepertoirePolicy` emits either a key of `DEFAULT_CLOSERS` verbatim or a
    template rendered with a premise name, and those sets are disjoint.
    """
    return tactic.strip() not in DEFAULT_CLOSERS


@dataclass
class Arm:
    """One run's per-problem outcomes, keyed by problem id."""

    run_id: str
    benchmark: str
    arm: str
    proved: dict[str, bool] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)
    proof: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, run_dir: Path | str) -> Arm:
        d = Path(run_dir)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        cfg = manifest.get("config", {})
        a = cls(
            run_id=manifest["run_id"],
            benchmark=cfg.get("benchmark", "?"),
            arm=cfg.get("arm", "?"),
        )
        for line in (d / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            pid = str(r["problem_id"])
            a.proved[pid] = bool(r.get("proved"))
            a.status[pid] = r.get("status", "?")
            a.proof[pid] = list(r.get("proof") or [])
        return a

    @property
    def n_proved(self) -> int:
        return sum(self.proved.values())

    def premise_using(self) -> set[str]:
        """Problems whose proof names at least one retrieved premise."""
        return {
            pid for pid, steps in self.proof.items()
            if self.proved.get(pid) and any(is_premise_tactic(t) for t in steps)
        }


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p for discordant counts `b` (treatment-only) and `c` (baseline-only).

    Under H0 each discordant pair is a fair coin, so this is a two-sided binomial test on
    `min(b, c)` out of `b + c`. Exact rather than the chi-square approximation because the
    discordant counts here are small (10 and 0 on FATE-M), which is exactly where the approximation
    is worst.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(d: np.ndarray, n_boot: int, rng: np.random.Generator,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference (resample problems, not systems)."""
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def permutation_p(d: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided sign-flip permutation p for H0: the paired differences are symmetric about 0.

    Ties carry no information under any sign assignment, so only nonzero differences are flipped.
    The +1/(n+1) correction keeps p strictly positive — a resampling test can never license p = 0.
    """
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0
    observed = abs(nz.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(nz)))
    null = np.abs((signs * nz).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def compare(
    baseline: Arm,
    treatment: Arm,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 0,
    exclude_harness_errors: bool = False,
) -> dict[str, Any]:
    """Paired comparison over the problems both runs attempted.

    `exclude_harness_errors` drops problems where *either* arm ended in a harness error. The primary
    number keeps them (scored as unproved) so the denominator matches the published benchmark size;
    this is the sensitivity check that says whether the conclusion depends on them.
    """
    if baseline.benchmark != treatment.benchmark:
        raise ValueError(
            f"different benchmarks: {baseline.benchmark!r} vs {treatment.benchmark!r}"
        )
    if baseline.arm == treatment.arm:
        raise ValueError(f"both runs are arm={baseline.arm!r}; nothing to compare")

    pids = sorted(set(baseline.proved) & set(treatment.proved))
    if not pids:
        raise ValueError("the two runs share no problem ids")
    if exclude_harness_errors:
        pids = [
            p for p in pids
            if baseline.status[p] not in HARNESS_ERROR_STATUSES
            and treatment.status[p] not in HARNESS_ERROR_STATUSES
        ]

    only_t = [p for p in pids if treatment.proved[p] and not baseline.proved[p]]
    only_b = [p for p in pids if baseline.proved[p] and not treatment.proved[p]]

    d = np.array(
        [float(treatment.proved[p]) - float(baseline.proved[p]) for p in pids], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    lo, hi = bootstrap_ci(d, n_boot, rng)
    p_perm = permutation_p(d, n_perm, rng)
    p_mcnemar = mcnemar_exact_p(len(only_t), len(only_b))

    ci_excludes_zero = lo > 0 or hi < 0
    # The rule the predecessor project settled on: agreement is required. When the CI and the
    # permutation test disagree, saying "significant" is picking the friendlier of two answers.
    significant = bool(ci_excludes_zero and p_perm < 0.05)
    borderline = bool(ci_excludes_zero != (p_perm < 0.05))

    n_b = sum(baseline.proved[p] for p in pids)
    n_t = sum(treatment.proved[p] for p in pids)
    marginal_with_premise = [
        p for p in only_t if any(is_premise_tactic(t) for t in treatment.proof[p])
    ]

    return {
        "benchmark": baseline.benchmark,
        "baseline": {"arm": baseline.arm, "run_id": baseline.run_id, "proved": n_b},
        "treatment": {"arm": treatment.arm, "run_id": treatment.run_id, "proved": n_t},
        "n_problems": len(pids),
        "excluded_harness_errors": exclude_harness_errors,
        "delta_problems": n_t - n_b,
        "delta_rate": (n_t - n_b) / len(pids),
        "only_treatment": only_t,
        "only_baseline": only_b,
        "baseline_is_subset": not only_b,
        "ci95": [lo, hi],
        "p_permutation": p_perm,
        "p_mcnemar_exact": p_mcnemar,
        "significant": significant,
        "borderline": borderline,
        # Two different questions, and conflating them is what produced the inflated first report.
        # `used` = of all proofs found, how many name a premise. `needed` = of the proofs the
        # baseline could NOT find, how many name a premise. Only the second is causal.
        "premise_used_rate": (
            len(treatment.premise_using()) / n_t if n_t else 0.0
        ),
        "premise_needed_rate": (
            len(marginal_with_premise) / len(only_t) if only_t else 0.0
        ),
        "marginal_with_premise": marginal_with_premise,
    }


def format_report(r: dict[str, Any]) -> str:
    """Human-readable summary. The JSON is the record; this is for reading in a terminal."""
    n = r["n_problems"]
    b, t = r["baseline"], r["treatment"]
    verdict = (
        "SIGNIFICANT" if r["significant"]
        else "borderline (CI and permutation disagree)" if r["borderline"]
        else "not significant"
    )
    lines = [
        f"=== {r['benchmark']}: {t['arm']} vs {b['arm']} "
        f"({n} problems{', harness errors excluded' if r['excluded_harness_errors'] else ''}) ===",
        f"  {b['arm']:<10} {b['proved']:>4}/{n}  ({100 * b['proved'] / n:5.1f}%)",
        f"  {t['arm']:<10} {t['proved']:>4}/{n}  ({100 * t['proved'] / n:5.1f}%)",
        f"  delta      {r['delta_problems']:>+4} problems  ({100 * r['delta_rate']:+.1f} points)",
        "",
        f"  {t['arm']}-only : {len(r['only_treatment'])}  {r['only_treatment']}",
        f"  {b['arm']}-only : {len(r['only_baseline'])}  {r['only_baseline']}",
        f"  no displacement: {r['baseline_is_subset']}"
        f"{'' if r['baseline_is_subset'] else '  <-- retrieval COST problems; investigate'}",
        "",
        f"  95% CI     [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]",
        f"  p (perm)   {r['p_permutation']:.4f}",
        f"  p (McNemar exact)  {r['p_mcnemar_exact']:.5f}",
        f"  verdict    {verdict}",
        "",
        f"  premise USED  : {100 * r['premise_used_rate']:.0f}% of {t['arm']}'s proofs "
        f"name a premise",
        f"  premise NEEDED: {100 * r['premise_needed_rate']:.0f}% of the "
        f"{len(r['only_treatment'])} problems only {t['arm']} solved name a premise",
    ]
    if r["premise_used_rate"] > r["premise_needed_rate"] + 1e-9:
        lines.append(
            "                  (USED exceeds NEEDED: some premise-naming proofs were not "
            "necessary — the baseline closed the same goals unaided)"
        )
    return "\n".join(lines)
