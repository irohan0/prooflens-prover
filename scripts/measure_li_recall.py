#!/usr/bin/env python
"""Measure what LI's two-stage approximation costs **on real queries**.

    python scripts/measure_li_recall.py --index data/index/li_ft_novel_bm25 \
        --benchmark fate_m --data-root ~/data/benchmarks/REAL-Prover/Realprover/data

## Why this exists

`build_dense_index.py` already prints a `recall@10 vs exact MaxSim` figure, and it reported 0.992.
That number was computed with **premise embeddings as probes** — `index.doc_tokens(i)`. A premise
retrieving itself and its neighbours is a far easier problem than a *proof state* retrieving a
premise: the probe is drawn from the same distribution as the corpus, so the mean-pooled first
stage barely has to work. It is measured on the wrong distribution and is optimistic by an unknown
margin.

This matters because it makes the headline arm comparison unbalanced in a way that is invisible in
the results:

* **SV** ranks all 276,070 premises exactly — one matrix-vector product, no approximation.
* **LI** mean-pools its token vectors, takes the top `n_candidates` (1000, i.e. **0.36%** of the
  corpus), and only then applies MaxSim.

If the first stage drops the right premise, no amount of late interaction recovers it. So a finding
that "SV beats LI" may be a finding about candidate generation rather than about multi-vector
scoring — and those are completely different claims.

Exact full-corpus MaxSim is infeasible per query (384 query tokens x 21.7M premise tokens), so this
computes it in chunks for a sample of queries. Slow and one-off, which is the right trade for a
number that decides how a headline result is interpreted.

Queries are the benchmark statements. They are not identical to the mid-search proof states the
retriever sees, but they are drawn from the same distribution as the *root* states, which is where
most of this policy's retrieval calls happen — and unlike proof states they need no Lean server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prooflens_prover.data.benchmarks import load_benchmark
from prooflens_prover.retrieval.dense import LateInteractionIndex, load_query_encoder
from prooflens_prover.utils.logging import get_logger

log = get_logger(__name__)


def exact_topk(index: LateInteractionIndex, q: np.ndarray, k: int, chunk: int) -> list[int]:
    """Exact full-corpus MaxSim top-k, computed in chunks so the score matrix fits in RAM."""
    scores = np.empty(index.n_docs, dtype=np.float32)
    for lo in range(0, index.n_docs, chunk):
        sel = np.arange(lo, min(lo + chunk, index.n_docs))
        scores[lo:lo + len(sel)] = index.maxsim_over(q, sel)
    top = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in top[np.argsort(-scores[top])]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--benchmark", default="fate_m")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-queries", type=int, default=40)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=5000, help="premises per exact-scoring chunk")
    ap.add_argument("--n-candidates", type=int, action="append", default=None,
                    help="first-stage budgets to evaluate; repeatable. Default: 1000 5000 20000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    budgets = args.n_candidates or [1000, 5000, 20000]
    index = LateInteractionIndex.load(args.index)
    log.info("index: %d premises, corpus_id=%s", index.n_docs, index.corpus_id)

    encode = load_query_encoder(
        "li", args.checkpoint or index.encoder.checkpoint, device=args.device
    )
    problems = load_benchmark(args.benchmark, args.data_root)
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(len(problems), size=min(args.n_queries, len(problems)), replace=False)
    log.info("encoding %d benchmark statements as queries", len(picked))
    queries = [encode(problems[int(i)].statement) for i in picked]
    log.info("query token counts: mean %.0f, max %d",
             float(np.mean([q.shape[0] for q in queries])), max(q.shape[0] for q in queries))

    hits: dict[int, list[float]] = {b: [] for b in budgets}
    for n, q in enumerate(queries, 1):
        gold = set(exact_topk(index, q, args.k, args.chunk))
        for b in budgets:
            approx = {i for i, _ in index.topk(q, k=args.k, n_candidates=b)}
            hits[b].append(len(gold & approx) / args.k)
        log.info("[%d/%d] %s", n, len(queries),
                 "  ".join(f"n_cand={b}: {hits[b][-1]:.2f}" for b in budgets))

    print()
    print(f"=== two-stage recall@{args.k} vs exact MaxSim, {len(queries)} real queries ===")
    print(f"{'n_candidates':>12} {'% of corpus':>12} {'recall':>8}")
    result = {}
    for b in budgets:
        r = float(np.mean(hits[b]))
        result[b] = r
        print(f"{b:>12} {100 * b / index.n_docs:>11.2f}% {r:>8.3f}")
    print()
    print("Compare against the 0.992 reported at index-build time, which used premise embeddings")
    print("as probes rather than queries. Any large gap means LI's arm was handicapped by")
    print("candidate generation, not by late interaction, and the arm comparison must say so.")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "index": str(args.index), "corpus_id": index.corpus_id,
            "benchmark": args.benchmark, "n_queries": len(queries), "k": args.k,
            "recall_by_n_candidates": result,
        }, indent=2), encoding="utf-8")
        print(f"written: {args.json_out}")


if __name__ == "__main__":
    main()
