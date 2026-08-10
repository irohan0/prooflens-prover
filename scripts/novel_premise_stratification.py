#!/usr/bin/env python
"""Stratify solved problems by whether their proof needs a premise the retriever never trained on.

    python scripts/novel_premise_stratification.py \
        --seen-split <prooflens>/leandojo_data/leandojo_benchmark_4/novel_premises/train.json \
        --corpus data/premises/mathlib_v4160.jsonl \
        --run results/logs/fate_m_none_vllm_... \
        --run results/logs/fate_m_sv_vllm_... \
        --run results/logs/fate_m_li_vllm_...

## The question this exists to answer

The predecessor premise-selection study's headline was a **robustness** claim: a matched
single-vector retriever degrades −17.9% from seen to novel premises, while late interaction stays
flat and wins on novel premises in 5/5 seeds. Tier 1 then found the two architectures
indistinguishable inside a live prover: +0 of 327 problems, 17 gained and 17 lost, p = 1.0000.

Both results cannot be the whole story at once. They are consistent only if either

  (a) the novel-premise advantage does not survive into proving, or
  (b) these benchmarks do not stress novel premises in the first place,

and nothing measured so far distinguishes them. If LI's 17 exclusive wins are concentrated on
problems whose proofs cite premises the retriever never trained on, while SV's 17 are concentrated
on premises it did, then the aggregate null is **two populations cancelling** rather than noise —
and the predecessor's claim lands end to end. That is the single most consequential thing the
existing logs can still be asked.

## What this measures, and what it does not

Measured: for each problem a run **solved**, does its proof cite at least one Mathlib premise absent
from the set of premises the retriever saw as a training positive?

**Not** measured: whether retrieval is what supplied that premise. A 7B model can recall a Mathlib
lemma from pretraining without any retriever offering it. So this is a property of the *problem* —
"does closing it require a lemma the retriever was never trained to rank" — and not a claim about
provenance. That distinction is why this metric is legitimate where the premise-*attribution* metric
was withheld (see `eval/compare.PREMISE_ATTRIBUTABLE_POLICIES`): attribution asserts a causal path
from retrieval to proof, which generated tactics cannot support, whereas this only classifies the
problem by a property of its own proof text.

Both arms' retrievers were fine-tuned on the *same* split (`li_ft_novel_bm25` and
`sv_ft_novel_lr3e6`), so one seen-set applies to both and the comparison is matched.

## Two deliberate conservatisms

Lean lets a tactic write an abbreviated name when a namespace is open, so a surface token can
resolve to several corpus premises. A token counts as **seen** if *any* full name it could
resolve to is in the training set. That biases the answer toward "seen" and makes an "unseen"
finding harder to obtain, which is the direction an argument resting on unseen premises needs.

Names are matched across two Mathlib versions — the predecessor traced commit `29dcec07`, this
project indexes v4.16.0 — so a premise renamed between them is misclassified as unseen. Reported as
a limitation, not silently corrected: the rename rate is unknown and estimating it would be its own
project.

Reads `attempts.jsonl` only. No GPU, no model, no Lean, no re-run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.eval.compare import format_budget  # noqa: E402
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: A Lean identifier: leading letter or underscore, then name characters. Dots are included so a
#: qualified citation (`Fintype.card_pi_const`) is captured whole rather than split into two tokens
#: that would each resolve wrongly. Subscripts and primes appear in real Mathlib names.
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.'!?₀-₉¹²³]*")

#: Tokens that are tactic syntax rather than premise citations. Kept deliberately short and used
#: only for a *sensitivity* figure, never the primary one. It barely matters which way it goes:
#: every common tactic word that doubles as a lemma name (`rfl` is the single most frequent training
#: premise, at 3,402 positives) is firmly inside the seen set, so it cannot manufacture an
#: unseen-premise finding. Excluding them changes the denominator, not the signal.
TACTIC_WORDS = frozenset("""
    exact apply rw rwa simp simpa intro intros refine constructor rcases obtain cases use have let
    show calc ring ring_nf field_simp linarith nlinarith norm_num omega decide aesop tauto trivial
    exacts induction subst unfold change conv congr ext specialize by at with this fun to and or if
    then else from rfl
""".split())


def stream_theorems(path: Path):
    """Yield theorem records from a LeanDojo split file without holding the parsed array.

    `train.json` is 365 MB; `json.load` on it costs several GB of interpreter objects for data we
    scan once and discard. `raw_decode` walks the array one element at a time, so peak memory is the
    source text plus a single record.
    """
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    i = text.index("[") + 1
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            return
        obj, i = decoder.raw_decode(text, i)
        yield obj


def seen_premise_names(split_path: Path) -> set[str]:
    """Fully-qualified names of every premise the retriever saw as a training positive.

    A premise provenance in a traced tactic carries its resolved `full_name`, which is what makes
    this portable across Mathlib versions. The predecessor resolved gold premises by *position*
    (`corpus.locate_premise(def_path, def_pos)`) against its own traced corpus; positions do not
    survive a version change, names mostly do.
    """
    seen: set[str] = set()
    for thm in stream_theorems(split_path):
        for tac in thm.get("traced_tactics") or ():
            annotated = tac.get("annotated_tactic")
            if not annotated or len(annotated) < 2:
                continue
            for prov in annotated[1] or ():
                name = prov.get("full_name")
                if name:
                    seen.add(name)
    return seen


def load_corpus(corpus_path: Path) -> tuple[set[str], dict[str, set[str]]]:
    """`(exact full names, {last component: full names})` for the premise corpus.

    The suffix index is what lets an abbreviated citation be resolved: a tactic that writes
    `card_pi_const` under `open Fintype` means `Fintype.card_pi_const`, and only the tail is
    written down.
    """
    exact: set[str] = set()
    by_suffix: dict[str, set[str]] = defaultdict(set)
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            name = json.loads(line).get("name")
            if not name:
                continue
            exact.add(name)
            by_suffix[name.rsplit(".", 1)[-1]].add(name)
    return exact, dict(by_suffix)


def resolve(token: str, exact: set[str], by_suffix: dict[str, set[str]]) -> set[str] | None:
    """Full names `token` could denote, or None if it names no premise in the corpus."""
    if token in exact:
        return {token}
    candidates = by_suffix.get(token)
    return set(candidates) if candidates else None


def cited_premises(
    tactics, exact: set[str], by_suffix: dict[str, set[str]], drop_tactic_words: bool = False
) -> dict[str, set[str]]:
    """`{surface token: full names it could denote}` over every premise a proof names."""
    out: dict[str, set[str]] = {}
    for tactic in tactics or ():
        for token in IDENTIFIER.findall(str(tactic)):
            if token in out or (drop_tactic_words and token in TACTIC_WORDS):
                continue
            resolved = resolve(token, exact, by_suffix)
            if resolved:
                out[token] = resolved
    return out


def unseen_citations(cited: dict[str, set[str]], seen: set[str]) -> set[str]:
    """Surface tokens whose every possible resolution lies outside the retriever's training set."""
    return {tok for tok, names in cited.items() if not (names & seen)}


def run_outcomes(run_dir: Path) -> tuple[str, str, dict[str, list[str] | None]]:
    """`(benchmark, arm label, {problem id: proof tactics if proved else None})` for one run."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest.get("config", {})
    n_candidates = cfg.get("n_candidates")
    arm = cfg.get("arm", "?")
    label = f"{arm}@{format_budget(n_candidates)}" if n_candidates else arm

    out: dict[str, list[str] | None] = {}
    for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["problem_id"])] = row.get("proof") if row.get("proved") else None
    return cfg.get("benchmark", "?"), label, out


def gather(run_dirs) -> tuple[dict[str, set[str]], dict[str, list[str]], list[str], int]:
    """Pool every run into `(solved per arm, proofs, benchmarks, n shared problems)`.

    Problem ids are namespaced by benchmark, so pooling FATE-M and ProofNet cannot silently pair
    two different problems that happen to share an id — the same guard `eval.compare.compare_pooled`
    applies to the proof counts. Within each benchmark only problems every arm reached are kept.

    Pooling matters here for power, not tidiness: the LI-vs-SV contrast has 11 exclusive wins per
    arm on FATE-M and 6 on ProofNet, and no test on 6 problems can conclude anything.
    """
    by_bench: dict[str, dict[str, dict]] = defaultdict(dict)
    for d in run_dirs:
        bench, label, outcomes = run_outcomes(Path(d))
        if label in by_bench[bench]:
            raise SystemExit(f"two runs for arm {label!r} on benchmark {bench!r}: pick one")
        by_bench[bench][label] = outcomes

    arm_sets = {frozenset(arms) for arms in by_bench.values()}
    if len(arm_sets) > 1:
        raise SystemExit(
            "the benchmarks do not expose the same arms — pooling would compare different "
            f"contrasts per benchmark: { {b: sorted(a) for b, a in by_bench.items()} }"
        )

    solved: dict[str, set[str]] = defaultdict(set)
    proofs: dict[str, list[str]] = {}
    n_shared = 0
    for bench, arms in sorted(by_bench.items()):
        shared = set.intersection(*(set(o) for o in arms.values()))
        n_shared += len(shared)
        for label, outcomes in arms.items():
            for pid in shared:
                if outcomes[pid]:
                    key = f"{bench}:{pid}"
                    solved[label].add(key)
                    proofs[key] = outcomes[pid]
    return dict(solved), proofs, sorted(by_bench), n_shared


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]].

    Exact rather than chi-square because the discordant sets are 17 problems each, where the
    asymptotic test is not valid. Two-sided by the total-probability convention: sum the probability
    of every table at least as extreme as the observed one.
    """
    row1, row2, col1, total = a + b, c + d, a + c, a + b + c + d
    if min(row1, row2, col1, total - col1) == 0:
        return 1.0

    def prob(x: int) -> float:
        return (
            math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(total, col1)
        )

    observed = prob(a)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    # 1e-9 absorbs floating error so a table with identical probability is not excluded by a
    # last-bit difference, which would understate p.
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def permutation_diff_means(a: list[float], b: list[float], n_perm: int, seed: int) -> float:
    """Two-sided p for `mean(b) - mean(a)` by shuffling group labels.

    Two-sample, not paired: the two arms' exclusive wins are *different problems*, so there is no
    pairing to exploit and the sign-flip test used elsewhere in this project does not apply.
    """
    import numpy as np

    if not a or not b:
        return 1.0
    pooled = np.array(a + b, dtype=np.float64)
    observed = abs(float(np.mean(b)) - float(np.mean(a)))
    rng = np.random.default_rng(seed)
    n_a = len(a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(float(pooled[n_a:].mean()) - float(pooled[:n_a].mean())) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def stratum(pids, proofs, exact, by_suffix, seen, drop_tactic_words) -> dict:
    """Unseen-premise statistics over one set of solved problems.

    The headline statistic is the **fraction** of a proof's cited premises that are unseen, not
    whether any is. With 78.2% of the corpus outside the retriever's training set and roughly five
    to ten premises named per proof, "cites at least one unseen premise" is true almost by
    construction — measured at 89-92% for every arm, which is a saturated metric rather than a
    finding. The fraction has dynamic range where the indicator has none, and it has a meaningful
    reference line: a proof drawing premises indistinguishable from the corpus at large sits at
    0.78, and only a value below that indicates the proof leans on premises the retriever knows.
    """
    rows = []
    for pid in sorted(pids):
        cited = cited_premises(proofs[pid], exact, by_suffix, drop_tactic_words)
        unseen = unseen_citations(cited, seen)
        rows.append({
            "problem_id": pid, "n_cited": len(cited), "n_unseen": len(unseen),
            "fraction_unseen": (len(unseen) / len(cited)) if cited else None,
            "unseen": sorted(unseen),
        })
    fractions = [r["fraction_unseen"] for r in rows if r["fraction_unseen"] is not None]
    n_with = sum(1 for r in rows if r["n_unseen"])
    return {
        "n_problems": len(rows),
        "n_problems_citing_any_premise": len(fractions),
        "n_citing_unseen": n_with,
        "rate": (n_with / len(rows)) if rows else None,
        "mean_fraction_unseen": (sum(fractions) / len(fractions)) if fractions else None,
        "mean_n_cited": (sum(r["n_cited"] for r in rows) / len(rows)) if rows else None,
        "mean_n_unseen": (sum(r["n_unseen"] for r in rows) / len(rows)) if rows else None,
        "fractions": fractions,
        "problems": rows,
    }


def main() -> int:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", required=True,
                    help="results/logs/<run_id>; repeat. Control first, then both retrieval arms")
    ap.add_argument("--corpus", type=Path, required=True,
                    help="the premise corpus the arms were indexed over")
    ap.add_argument("--seen-split", type=Path, default=None,
                    help="LeanDojo novel_premises/train.json — the retriever's training positives")
    ap.add_argument("--seen-cache", type=Path,
                    default=Path("data/premises/retriever_seen_premises.json"),
                    help="built from --seen-split on first use, then reused")
    ap.add_argument("--drop-tactic-words", action="store_true",
                    help="sensitivity: ignore tokens that are tactic syntax")
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path,
                    default=Path("results/tables/novel_premise_stratification.json"))
    args = ap.parse_args()

    if len(args.run) < 2:
        raise SystemExit("--run at least twice; the two retrieval arms are the comparison")

    # --- the retriever's training positives ---
    if args.seen_cache.exists():
        seen = set(json.loads(args.seen_cache.read_text(encoding="utf-8")))
        print(f"seen premises: {len(seen):,} (cached: {args.seen_cache})")
    else:
        if args.seen_split is None:
            raise SystemExit(
                f"no cache at {args.seen_cache} — pass --seen-split pointing at the predecessor's "
                "leandojo_data/leandojo_benchmark_4/novel_premises/train.json"
            )
        print(f"building seen-premise set from {args.seen_split} (365 MB, ~1 min) ...")
        seen = seen_premise_names(args.seen_split)
        args.seen_cache.parent.mkdir(parents=True, exist_ok=True)
        args.seen_cache.write_text(json.dumps(sorted(seen)), encoding="utf-8")
        print(f"seen premises: {len(seen):,} unique (cached to {args.seen_cache})")

    exact, by_suffix = load_corpus(args.corpus)
    in_corpus = seen & exact
    base_rate = 1 - len(in_corpus) / len(exact)
    print(f"premise corpus: {len(exact):,} names, {len(by_suffix):,} distinct last components")
    print(f"  {len(in_corpus):,} of the corpus ({1 - base_rate:.1%}) are premises the retriever "
          f"trained on")
    lost = len(seen) - len(in_corpus)
    print(f"  {lost:,} training premises ({lost / len(seen):.1%}) have no v4.16.0 name — "
          f"renamed or removed across the version gap")
    print(f"  => BASE RATE: an arbitrary corpus premise is unseen with probability "
          f"{base_rate:.3f}")

    solved, proofs, benchmarks, n_shared = gather(args.run)
    scope = " + ".join(benchmarks)

    print(f"\n=== unseen-premise stratification over {n_shared} shared problems ({scope}) ===")
    print("A cited premise is 'unseen' when the retriever never saw it as a training positive.")
    print("This is a property of the proof, not a claim that retrieval supplied the premise.")
    print(f"Compare every fraction against the base rate {base_rate:.3f}: at that value a proof's")
    print("premises are indistinguishable from premises drawn from the corpus at large.\n")
    print(f"  {'arm':<12} {'solved':>7} {'cited/proof':>12} {'frac unseen':>12} "
          f"{'>=1 unseen':>11}")
    per_arm = {}
    for label, pids in solved.items():
        s = stratum(pids, proofs, exact, by_suffix, seen, args.drop_tactic_words)
        per_arm[label] = s
        # An arm can solve nothing on a benchmark, which leaves every mean as None. Formatting a
        # None with a float spec raises, so each cell is rendered defensively rather than assuming
        # a non-empty stratum.
        frac = f"{s['mean_fraction_unseen']:.3f}" if s["mean_fraction_unseen"] is not None else "—"
        rate = f"{s['rate']:.1%}" if s["rate"] is not None else "—"
        cited = f"{s['mean_n_cited']:.1f}" if s["mean_n_cited"] is not None else "—"
        print(f"  {label:<12} {s['n_problems']:>7} {cited:>12} {frac:>12} {rate:>11}")
    print("\n  The last column is reported for completeness and is saturated by construction:")
    print(f"  at a {base_rate:.2f} base rate, with several premises per proof, almost every proof")
    print("  names at least one unseen premise. The fraction is the statistic with range.")

    # --- the comparison: the two retrieval arms' exclusive wins ---
    labels = list(solved)
    contrasts = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            only_a, only_b = solved[a] - solved[b], solved[b] - solved[a]
            if not (only_a and only_b):
                continue
            sa = stratum(only_a, proofs, exact, by_suffix, seen, args.drop_tactic_words)
            sb = stratum(only_b, proofs, exact, by_suffix, seen, args.drop_tactic_words)
            p_fisher = fisher_exact_two_sided(
                sb["n_citing_unseen"], sb["n_problems"] - sb["n_citing_unseen"],
                sa["n_citing_unseen"], sa["n_problems"] - sa["n_citing_unseen"],
            )
            p_frac = permutation_diff_means(
                sa["fractions"], sb["fractions"], args.n_perm, args.seed
            )
            delta = (
                sb["mean_fraction_unseen"] - sa["mean_fraction_unseen"]
                if sa["mean_fraction_unseen"] is not None
                and sb["mean_fraction_unseen"] is not None else None
            )
            contrasts.append({
                "contrast": f"{b} vs {a}",
                f"only_{a}": sa, f"only_{b}": sb,
                "delta_mean_fraction_unseen": delta,
                "p_permutation_fraction": p_frac,
                "p_fisher_two_sided_indicator": p_fisher,
                "significant": bool(p_frac < 0.05),
            })
            print(f"\n  --- {b} vs {a}: the problems only one of them solved ---")
            for lbl, s in ((f"only {a}", sa), (f"only {b}", sb)):
                frac = (f"{s['mean_fraction_unseen']:.3f}"
                        if s["mean_fraction_unseen"] is not None else "—")
                cited = f"{s['mean_n_cited']:.1f}" if s["mean_n_cited"] is not None else "—"
                print(f"  {lbl:<14} {s['n_problems']:>3} problems, "
                      f"{cited:>4} premises/proof, fraction unseen {frac}")
            if delta is not None:
                verdict = "SIGNIFICANT" if p_frac < 0.05 else "not significant"
                print(f"  difference in fraction unseen: {delta:+.4f}   "
                      f"permutation p = {p_frac:.4f}   {verdict}")
                print(f"  (indicator '>=1 unseen', Fisher exact p = {p_fisher:.4f} — saturated, "
                      f"reported for completeness)")

    print("\nIf the late-interaction arm's exclusive wins are enriched for unseen premises, the")
    print("aggregate null is two populations cancelling rather than noise, and the predecessor")
    print("study's robustness claim survives into a live prover. If the two strata look alike, the")
    print("null is a null about the retrievers and the benchmarks do not stress novel premises.")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({
        "benchmarks": benchmarks,
        "n_shared_problems": n_shared,
        "n_seen_premises": len(seen),
        "n_seen_premises_in_corpus": len(in_corpus),
        "n_corpus_premises": len(exact),
        "unseen_base_rate": base_rate,
        "drop_tactic_words": args.drop_tactic_words,
        "per_arm": per_arm,
        "contrasts": contrasts,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
