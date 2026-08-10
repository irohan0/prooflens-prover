#!/usr/bin/env python
"""Measure what LI's two-stage approximation costs **on real queries** (hypothesis H1).

    python scripts/measure_li_recall.py --index data/index/li_ft_novel_bm25 \
        --benchmark fate_m --data-root ~/data/benchmarks/REAL-Prover/Realprover/data

## Why this exists

`build_dense_index.py` prints a `recall@10 vs exact MaxSim` figure and it reported **0.992**. That
number used **premise embeddings as probes** (`index.doc_tokens(i)`), so it measures a premise
retrieving its neighbours — a probe drawn from the same distribution as the corpus, which the
mean-pooled first stage handles easily. It is measured on the wrong distribution.

This matters because it makes the headline arm comparison unbalanced in a way invisible in the
results:

* **SV** ranks all 276,070 premises exactly — one matrix-vector product.
* **LI** mean-pools its token vectors, keeps the top `n_candidates` (1,000 = **0.36%** of the
  corpus), and only then applies MaxSim. A premise the first stage drops cannot be recovered by
  late interaction.

If recall on real queries is materially below 0.99, the measured "SV beats LI" is a statement about
candidate generation rather than about late interaction. Those are different findings.

## What it reports

Recall@k of the two-stage path against exact full-corpus MaxSim, at several first-stage budgets, so
the output is a *curve* rather than a single number: it answers both "how lossy is 1,000?" and "what
budget would fix it?" in one job.

## Queries

The benchmark statements. Not identical to the mid-search proof states the retriever sees, but drawn
from the same distribution as the *root* states, where most retrieval calls happen — and unlike
proof states they need no Lean server, which keeps this a pure-retrieval measurement.

Per-query rows are appended to `<json-out>.jsonl` as they complete, so a job killed by the walltime
still yields a usable partial answer. Two earlier jobs on this project were lost to all-or-nothing
designs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Importable from a fresh clone with no install and no exported PYTHONPATH: a login-node
# `python scripts/<this>.py` must work, because that is how the analysis scripts get run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.data.benchmarks import load_benchmark
from prooflens_prover.retrieval.dense import LateInteractionIndex, load_query_encoder
from prooflens_prover.utils.logging import ensure_utf8_output, get_logger

log = get_logger(__name__)

#: First-stage budgets to sweep. 1,000 is what every LI result so far was produced with.
DEFAULT_BUDGETS = (1_000, 5_000, 20_000, 50_000)


def main() -> None:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--benchmark", default="fate_m")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--checkpoint", default=None,
                    help="defaults to the checkpoint recorded in the index metadata, which is what "
                         "keeps queries and premises in the same embedding space")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-queries", type=int, default=0,
                    help="0 (default) uses every problem in the benchmark, removing sampling error")
    ap.add_argument("--k", type=int, default=10, help="must match the prover's --top-k")
    ap.add_argument("--chunk", type=int, default=10_000,
                    help="premises scored per chunk. Peak memory is set by this, not the corpus: "
                         "~1.2 GB of score matrix at chunk=10000 and 384 query tokens")
    ap.add_argument("--n-candidates", type=int, action="append", default=None,
                    help="first-stage budget to evaluate; repeatable. "
                         "Default: 1000 5000 20000 50000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    budgets = sorted(set(args.n_candidates or DEFAULT_BUDGETS))
    index = LateInteractionIndex.load(args.index)
    log.info("index: %d premises, %d token vectors, corpus_id=%s",
             index.n_docs, index.tokens.shape[0], index.corpus_id)
    log.info("index n_candidates (what the arm runs used): %d (%.2f%% of corpus)",
             index.n_candidates, 100 * index.n_candidates / index.n_docs)

    problems = load_benchmark(args.benchmark, args.data_root)
    if args.n_queries and args.n_queries < len(problems):
        picked = np.random.default_rng(args.seed).choice(
            len(problems), size=args.n_queries, replace=False
        )
        selected = [problems[int(i)] for i in sorted(picked)]
    else:
        selected = problems
    log.info("encoding %d/%d benchmark statements as queries", len(selected), len(problems))

    encode = load_query_encoder(
        "li", args.checkpoint or index.encoder.checkpoint, device=args.device
    )
    t0 = time.perf_counter()
    queries = [(p.id, encode(p.statement)) for p in selected]
    tok_counts = [q.shape[0] for _, q in queries]
    log.info("encoded in %.1fs — query tokens: mean %.0f, min %d, max %d",
             time.perf_counter() - t0, float(np.mean(tok_counts)),
             min(tok_counts), max(tok_counts))

    rows_path = (args.json_out.with_suffix(".jsonl") if args.json_out else None)
    hits: dict[int, list[float]] = {b: [] for b in budgets}
    t_start = time.perf_counter()

    for n, (pid, q) in enumerate(queries, 1):
        gold = {i for i, _ in index.exact_topk_chunked(q, k=args.k, chunk=args.chunk)}
        row = {"problem_id": pid, "n_query_tokens": int(q.shape[0]), "n_gold": len(gold)}
        for b in budgets:
            approx = {i for i, _ in index.topk(q, k=args.k, n_candidates=b)}
            r = len(gold & approx) / max(len(gold), 1)
            hits[b].append(r)
            row[f"recall_{b}"] = r
        if rows_path:
            with rows_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
        eta = (time.perf_counter() - t_start) / n * (len(queries) - n)
        log.info("[%d/%d] %-8s %s  (eta %.0f min)", n, len(queries), pid,
                 "  ".join(f"{b}:{row[f'recall_{b}']:.2f}" for b in budgets), eta / 60)

    print()
    print(f"=== two-stage recall@{args.k} vs exact MaxSim, {len(queries)} real queries ===")
    print(f"{'n_candidates':>12} {'% of corpus':>12} {'recall':>8} {'queries lossless':>18}")
    result = {}
    for b in budgets:
        r = float(np.mean(hits[b]))
        lossless = sum(1 for x in hits[b] if x == 1.0)
        result[str(b)] = {"recall": r, "n_lossless": lossless}
        mark = "  <-- what every LI result so far used" if b == index.n_candidates else ""
        print(f"{b:>12} {100 * b / index.n_docs:>11.2f}% {r:>8.3f} "
              f"{lossless:>10}/{len(queries)}{mark}")

    deployed = str(index.n_candidates)
    baseline = result[deployed]["recall"] if deployed in result else None
    print()
    if baseline is not None and baseline >= 0.99:
        print("H1 REJECTED: the first stage is essentially lossless on real queries, so the")
        print("arm comparison is not confounded by candidate generation. SV beats LI on merit.")
    elif baseline is not None:
        print(f"H1 SUPPORTED: recall at the deployed budget is {baseline:.3f}, not ~0.99.")
        print("The LI arm was ranking an incomplete candidate set. Re-run it at the budget where")
        print("the curve saturates before reporting LI-vs-SV as an architecture result.")
    print("Compare against the 0.992 printed at index-build time, which used premise embeddings as")
    print("probes rather than queries and therefore does not answer this question.")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "index": str(args.index), "corpus_id": index.corpus_id,
            "deployed_n_candidates": index.n_candidates,
            "benchmark": args.benchmark, "n_queries": len(queries), "k": args.k,
            "query_tokens": {"mean": float(np.mean(tok_counts)),
                             "min": int(min(tok_counts)), "max": int(max(tok_counts))},
            "recall_by_n_candidates": result,
            "elapsed_s": round(time.perf_counter() - t_start, 1),
        }, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json_out}  (per-query rows: {rows_path})")


if __name__ == "__main__":
    main()
