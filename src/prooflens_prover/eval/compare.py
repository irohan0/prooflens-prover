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


def format_budget(n: int | None) -> str:
    """Render a first-stage candidate budget compactly: 1000 -> `1k`, 50000 -> `50k`.

    `None` renders as `?` rather than being assumed to be the index's stored 1,000. Filling in an
    unrecorded number is how the 0.443-recall runs came to look interchangeable with the corrected
    ones.
    """
    if n is None:
        return "?"
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


#: Policies whose proof text can be attributed to retrieval by inspection alone.
#:
#: **Only the model-free one.** `is_premise_tactic` is exact for `RepertoirePolicy` because that
#: policy emits either a `DEFAULT_CLOSERS` key verbatim or a template rendered with a premise name,
#: and those two sets are disjoint. For a language model the same test is *vacuous*: `intro x`,
#: `ring_nf` and `linarith` are all "not a closer" too, so every tactic it writes counts as naming a
#: premise, and the metric reports 100% regardless of whether retrieval contributed anything.
#:
#: That is not a hypothetical. It was reported as "100% of the 13 problems only li solved name a
#: retrieved premise" and offered as causal evidence, from exactly this vacuous test — the same
#: mistake, in the same file, as the eyeballed 73% utilisation this module's docstring was written
#: to prevent. Attributing an LLM's proof to retrieval needs the premises that were actually offered
#: for each state, which the trace does not record; until it does, the honest answer is `None`.
PREMISE_ATTRIBUTABLE_POLICIES = frozenset({"repertoire"})


def is_premise_tactic(tactic: str) -> bool:
    """True if `tactic` names a retrieved premise rather than being a bare repertoire closer.

    Exact for `RepertoirePolicy`, and **only** for it — see `PREMISE_ATTRIBUTABLE_POLICIES`. Callers
    must check `Arm.premise_attribution_available` before using this on a run's proofs.
    """
    return tactic.strip() not in DEFAULT_CLOSERS


@dataclass
class Arm:
    """One run's per-problem outcomes, keyed by problem id."""

    run_id: str
    benchmark: str
    arm: str
    #: li only: the first-stage candidate budget this run actually used, as recorded by the run
    #: itself. Two li runs differing in this are NOT the same experiment — at 1,000 the pooled
    #: first stage retained only 0.443 of its own exact top-10 — so anything that renders an li
    #: number has to be able to say which budget produced it. `None` for arms without a first
    #: stage, and for li runs predating the field.
    n_candidates: int | None = None
    #: Which generator produced these proofs. Decides whether premise attribution is measurable at
    #: all; see `PREMISE_ATTRIBUTABLE_POLICIES`. Runs predating the field were all model-free.
    policy_kind: str = "repertoire"
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
            n_candidates=cfg.get("n_candidates"),
            # Absent means the run predates the field, and every such run used the repertoire.
            policy_kind=cfg.get("policy_kind") or "repertoire",
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

    @property
    def label(self) -> str:
        """Arm name, disambiguated by first-stage budget when the run recorded one.

        `li` alone stopped being a unique identifier the moment two li runs differed only in
        `n_candidates` — which is the H1 experiment, and produced 22/141 against 31/141.
        """
        if self.n_candidates is None:
            return self.arm
        return f"{self.arm}@{format_budget(self.n_candidates)}"

    @property
    def premise_attribution_available(self) -> bool:
        """Whether this run's proof text can be attributed to retrieval by inspection."""
        return self.policy_kind in PREMISE_ATTRIBUTABLE_POLICIES

    def premise_using(self) -> set[str] | None:
        """Problems whose proof names at least one retrieved premise, or None if unmeasurable.

        `None` rather than a number, because the alternative — running the vacuous test anyway —
        returns 100% for any language model and reads as a strong causal result.
        """
        if not self.premise_attribution_available:
            return None
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


def oracle_union(a: Arm, b: Arm) -> dict[str, Any]:
    """Problems solved by *either* arm, over the problems both attempted.

    Not a fusion result and not achievable by any single retriever — it is the **ceiling** a fusion
    arm (reciprocal-rank fusion over both rankings) could reach. It exceeds both arms only when they
    disagree about *which* problems they solve, which two equal counts can hide completely: on
    ProofNet, SV and LI each solve 20 of 186 and four of those differ in each direction.
    """
    pids = sorted(set(a.proved) & set(b.proved))
    won = [p for p in pids if a.proved[p] or b.proved[p]]
    best = max(sum(a.proved[p] for p in pids), sum(b.proved[p] for p in pids))
    return {
        "n_problems": len(pids),
        "n_union": len(won),
        "n_best_single": best,
        "gain_over_best": len(won) - best,
        "union_rate": len(won) / len(pids) if pids else 0.0,
    }


def _paired_pids(
    baseline: Arm, treatment: Arm, exclude_harness_errors: bool
) -> list[str]:
    """Problems both runs attempted, in a deterministic order.

    Factored out so `compare` and `compare_pooled` cannot drift apart on which problems count — a
    pooled figure computed over a different problem set from its own per-benchmark rows would be
    indefensible and completely invisible.
    """
    pids = sorted(set(baseline.proved) & set(treatment.proved))
    if exclude_harness_errors:
        pids = [
            p for p in pids
            if baseline.status[p] not in HARNESS_ERROR_STATUSES
            and treatment.status[p] not in HARNESS_ERROR_STATUSES
        ]
    return pids


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
    # Same arm at a *different* first-stage budget is a legitimate — in fact the decisive —
    # comparison: it isolates candidate generation from the architecture, holding the encoder,
    # index, policy, search budget and seed fixed. The guard's real job is to reject a pair that
    # differs in nothing, so it tests that, not the arm name.
    if baseline.run_id == treatment.run_id:
        raise ValueError(f"both sides are the same run ({baseline.run_id!r}); nothing to compare")
    if baseline.arm == treatment.arm and baseline.n_candidates == treatment.n_candidates:
        raise ValueError(
            f"both runs are arm={baseline.arm!r} at the same first-stage budget "
            f"({format_budget(baseline.n_candidates)}); nothing to compare"
        )

    if not set(baseline.proved) & set(treatment.proved):
        raise ValueError("the two runs share no problem ids")
    pids = _paired_pids(baseline, treatment, exclude_harness_errors)

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
    both = [p for p in pids if baseline.proved[p] and treatment.proved[p]]

    # Two different questions, and conflating them produced an inflated first report. `used` = of
    # all proofs found, how many name a premise. `needed` = of the proofs the baseline could NOT
    # find, how many name a premise. Only the second is causal — and neither is measurable at all
    # unless the policy's tactic vocabulary makes "names a premise" decidable from the text.
    if treatment.premise_attribution_available:
        using = treatment.premise_using() or set()
        marginal_with_premise: list[str] | None = [
            p for p in only_t if any(is_premise_tactic(t) for t in treatment.proof[p])
        ]
        premise_used_rate: float | None = len(using) / n_t if n_t else 0.0
        premise_needed_rate: float | None = (
            len(marginal_with_premise) / len(only_t) if only_t else 0.0
        )
    else:
        marginal_with_premise = None
        premise_used_rate = premise_needed_rate = None

    return {
        "benchmark": baseline.benchmark,
        # `label` is what the report prints; `arm` stays raw so downstream keys are stable.
        "baseline": {"arm": baseline.arm, "label": baseline.label, "run_id": baseline.run_id,
                     "n_candidates": baseline.n_candidates, "proved": n_b},
        "treatment": {"arm": treatment.arm, "label": treatment.label, "run_id": treatment.run_id,
                      "n_candidates": treatment.n_candidates, "proved": n_t},
        "n_problems": len(pids),
        "excluded_harness_errors": exclude_harness_errors,
        "delta_problems": n_t - n_b,
        "delta_rate": (n_t - n_b) / len(pids),
        "only_treatment": only_t,
        "only_baseline": only_b,
        "baseline_is_subset": not only_b,
        # The union of the two arms, over the problems both attempted. Not achievable by any single
        # retriever — it is the ceiling a fusion arm could reach, and it is invisible in the deltas:
        # on FATE-M li and sv both prove 46 with a delta of exactly 0, while disagreeing about 22
        # problems, so their union is 57. A comparison that printed only the delta would report the
        # most interesting thing about that pair as "no difference".
        "n_both": len(both),
        "n_union": len(both) + len(only_t) + len(only_b),
        "union_rate": (len(both) + len(only_t) + len(only_b)) / len(pids),
        "union_gain_over_best": (len(both) + len(only_t) + len(only_b)) - max(n_b, n_t),
        "ci95": [lo, hi],
        "p_permutation": p_perm,
        "p_mcnemar_exact": p_mcnemar,
        "significant": significant,
        "borderline": borderline,
        "policy_kind": treatment.policy_kind,
        "premise_attribution_available": treatment.premise_attribution_available,
        "premise_used_rate": premise_used_rate,
        "premise_needed_rate": premise_needed_rate,
        "marginal_with_premise": marginal_with_premise,
    }


def compare_pooled(
    pairs: list[tuple[Arm, Arm]],
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 0,
    exclude_harness_errors: bool = False,
) -> dict[str, Any]:
    """One paired test over the problems of several benchmarks at once.

    ## Why pool at all

    FATE-M's li-vs-none contrast is +7 problems from 13 gained and 6 lost. Exact McNemar on 19
    discordant pairs cannot reach p < 0.05 for any split closer than 15-4, so "not significant"
    there is as much a statement about **power** as about retrieval. Problems are independent units
    and the contrast is defined identically on each benchmark — same encoder, index, policy, search
    budget and seed — so pooling them is a legitimate fixed-effects analysis that buys power without
    buying GPU time.

    ## Why the per-benchmark rows are always reported alongside

    A pooled estimate is a problem-count-weighted average over benchmarks whose base rates and
    retrieval sensitivity differ by design: FATE-M was chosen *because* it is retrieval-sensitive
    and miniF2F *because* it is not. Averaging +5.0 points on one with -1.0 on another yields a
    positive pooled figure describing neither. `per_benchmark` and `heterogeneous` exist so that
    cannot be reported without being seen.

    ## Why problem ids are namespaced

    FATE-M ids are bare integers (`326`, `1597`). A pooled union keyed on the raw id would silently
    merge two unrelated problems that happen to share a number, and the count would still look
    plausible.
    """
    if len(pairs) < 2:
        raise ValueError("pooling needs at least two benchmarks; use compare() for one")

    for baseline, treatment in pairs:
        if baseline.benchmark != treatment.benchmark:
            raise ValueError(
                f"a pair spans two benchmarks: {baseline.benchmark!r} vs {treatment.benchmark!r}"
            )

    benchmarks = [t.benchmark for _, t in pairs]
    if len(set(benchmarks)) != len(benchmarks):
        raise ValueError(
            f"a benchmark appears more than once ({benchmarks}); pooling it twice would count its "
            "problems twice and shrink the p-value for free"
        )

    # Every pair must be the SAME contrast. Pooling li-vs-none with sv-vs-none would produce a
    # number that answers no question, and the two would look identical in any summary.
    contrasts = {
        (b.arm, b.n_candidates, t.arm, t.n_candidates) for b, t in pairs
    }
    if len(contrasts) != 1:
        raise ValueError(
            f"the pairs are not the same contrast: {sorted(contrasts)}. Pooling different "
            "comparisons gives an average of unrelated effects."
        )

    per_pair = [
        compare(b, t, n_boot, n_perm, seed, exclude_harness_errors) for b, t in pairs
    ]

    diffs: list[float] = []
    only_t: list[str] = []
    only_b: list[str] = []
    for (baseline, treatment), r in zip(pairs, per_pair, strict=True):
        pids = _paired_pids(baseline, treatment, exclude_harness_errors)
        diffs += [
            float(treatment.proved[p]) - float(baseline.proved[p]) for p in pids
        ]
        only_t += [f"{treatment.benchmark}:{p}" for p in r["only_treatment"]]
        only_b += [f"{treatment.benchmark}:{p}" for p in r["only_baseline"]]

    d = np.array(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    lo, hi = bootstrap_ci(d, n_boot, rng)
    p_perm = permutation_p(d, n_perm, rng)
    p_mcnemar = mcnemar_exact_p(len(only_t), len(only_b))

    ci_excludes_zero = lo > 0 or hi < 0
    significant = bool(ci_excludes_zero and p_perm < 0.05)
    borderline = bool(ci_excludes_zero != (p_perm < 0.05))

    signs = {int(np.sign(r["delta_problems"])) for r in per_pair}
    heterogeneous = len(signs - {0}) > 1

    first_b, first_t = pairs[0]
    return {
        "benchmarks": benchmarks,
        "baseline": {"arm": first_b.arm, "label": first_b.label,
                     "n_candidates": first_b.n_candidates,
                     "run_ids": [b.run_id for b, _ in pairs]},
        "treatment": {"arm": first_t.arm, "label": first_t.label,
                      "n_candidates": first_t.n_candidates,
                      "run_ids": [t.run_id for _, t in pairs]},
        "n_problems": int(len(d)),
        "excluded_harness_errors": exclude_harness_errors,
        "delta_problems": int(d.sum()),
        "delta_rate": float(d.mean()),
        "only_treatment": only_t,
        "only_baseline": only_b,
        "ci95": [lo, hi],
        "p_permutation": p_perm,
        "p_mcnemar_exact": p_mcnemar,
        "significant": significant,
        "borderline": borderline,
        "heterogeneous": heterogeneous,
        "per_benchmark": [
            {
                "benchmark": r["benchmark"],
                "n_problems": r["n_problems"],
                "baseline_proved": r["baseline"]["proved"],
                "treatment_proved": r["treatment"]["proved"],
                "delta_problems": r["delta_problems"],
                "delta_rate": r["delta_rate"],
                "p_mcnemar_exact": r["p_mcnemar_exact"],
                "significant": r["significant"],
            }
            for r in per_pair
        ],
    }


def format_pooled_report(r: dict[str, Any]) -> str:
    """Human-readable pooled summary. The per-benchmark table is not optional decoration."""
    b, t = r["baseline"], r["treatment"]
    bl, tl = b["label"], t["label"]
    n = r["n_problems"]
    verdict = (
        "SIGNIFICANT" if r["significant"]
        else "borderline (CI and permutation disagree)" if r["borderline"]
        else "not significant"
    )
    lines = [
        f"=== POOLED: {tl} vs {bl} over {'+'.join(r['benchmarks'])} "
        f"({n} problems"
        f"{', harness errors excluded' if r['excluded_harness_errors'] else ''}) ===",
        "",
        f"  {'benchmark':<12} {'n':>5} {bl:>8} {tl:>8} {'delta':>7} {'p (McNemar)':>12}",
    ]
    for row in r["per_benchmark"]:
        lines.append(
            f"  {row['benchmark']:<12} {row['n_problems']:>5} "
            f"{row['baseline_proved']:>8} {row['treatment_proved']:>8} "
            f"{row['delta_problems']:>+7} {row['p_mcnemar_exact']:>12.4f}"
        )
    lines += [
        "",
        f"  pooled delta   {r['delta_problems']:>+4} problems  "
        f"({100 * r['delta_rate']:+.1f} points over {n})",
        f"  gained / lost  {len(r['only_treatment'])} / {len(r['only_baseline'])}",
        f"  95% CI         [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]",
        f"  p (perm)       {r['p_permutation']:.4f}",
        f"  p (McNemar)    {r['p_mcnemar_exact']:.5f}",
        f"  verdict        {verdict}",
    ]
    if r["heterogeneous"]:
        lines += [
            "",
            "  !! HETEROGENEOUS: the per-benchmark deltas do not share a sign. This total is a",
            "     problem-count-weighted average of effects pointing opposite ways, and describes",
            "     neither benchmark. Report the rows above, not the pooled figure.",
        ]
    return "\n".join(lines)


def format_report(r: dict[str, Any]) -> str:
    """Human-readable summary. The JSON is the record; this is for reading in a terminal."""
    n = r["n_problems"]
    b, t = r["baseline"], r["treatment"]
    # `.get` so a report loaded from an older table1.json still renders.
    bl, tl = b.get("label", b["arm"]), t.get("label", t["arm"])
    verdict = (
        "SIGNIFICANT" if r["significant"]
        else "borderline (CI and permutation disagree)" if r["borderline"]
        else "not significant"
    )
    lines = [
        f"=== {r['benchmark']}: {tl} vs {bl} "
        f"({n} problems{', harness errors excluded' if r['excluded_harness_errors'] else ''}) ===",
        f"  {bl:<10} {b['proved']:>4}/{n}  ({100 * b['proved'] / n:5.1f}%)",
        f"  {tl:<10} {t['proved']:>4}/{n}  ({100 * t['proved'] / n:5.1f}%)",
        f"  delta      {r['delta_problems']:>+4} problems  ({100 * r['delta_rate']:+.1f} points)",
        "",
        f"  {tl}-only : {len(r['only_treatment'])}  {r['only_treatment']}",
        f"  {bl}-only : {len(r['only_baseline'])}  {r['only_baseline']}",
    ]

    # The warning only makes sense against the no-retrieval control, where a problem the baseline
    # solved and the treatment did not really is retrieval *costing* a proof. Between two retrievers
    # it is symmetric disagreement, and calling it "cost" would frame an even split as a regression.
    if r["baseline_is_subset"]:
        lines.append("  no displacement: True")
    elif b["arm"] == "none":
        lines.append(
            "  no displacement: False  <-- retrieval COST proofs the model found unaided; "
            "investigate"
        )
    else:
        lines.append(
            f"  disagreement: {len(r['only_treatment'])} vs {len(r['only_baseline'])} — "
            "neither arm dominates; see the union below"
        )

    lines += [
        "",
        f"  union         {r['n_union']}/{n}  ({100 * r['union_rate']:5.1f}%)  "
        f"= {r['n_both']} shared + {len(r['only_treatment'])} + {len(r['only_baseline'])}",
        f"                {r['union_gain_over_best']:+d} over the better arm — the ceiling a "
        f"fusion of the two could reach",
        "",
        f"  95% CI     [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]",
        f"  p (perm)   {r['p_permutation']:.4f}",
        f"  p (McNemar exact)  {r['p_mcnemar_exact']:.5f}",
        f"  verdict    {verdict}",
        "",
    ]

    used, needed = r.get("premise_used_rate"), r.get("premise_needed_rate")
    if used is None or needed is None:
        lines += [
            f"  premise attribution: NOT MEASURABLE for policy "
            f"{r.get('policy_kind', '?')!r}",
            "    'names a premise' is decidable from proof text only for the model-free",
            "    repertoire, whose tactics are either a fixed closer or a premise template. Every",
            "    tactic a language model writes is 'not a closer', so the same test would mark all",
            "    of them as premise-naming whether or not retrieval contributed — which is why it",
            "    is withheld rather than printed. Measuring it for real needs the premises",
            "    offered at each state, which the search trace does not currently record.",
        ]
    else:
        lines += [
            f"  premise USED  : {100 * used:.0f}% of {tl}'s proofs name a premise",
            f"  premise NEEDED: {100 * needed:.0f}% of the "
            f"{len(r['only_treatment'])} problems only {tl} solved name a premise",
        ]
        if used > needed + 1e-9:
            lines.append(
                "                  (USED exceeds NEEDED: some premise-naming proofs were not "
                "necessary — the baseline closed the same goals unaided)"
            )
    return "\n".join(lines)
