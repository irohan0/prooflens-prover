#!/usr/bin/env python
"""Run one retrieval arm over one benchmark with the shared best-first search harness.

    python scripts/prove_benchmark.py --benchmark fate_m --arm bm25 \
        --index data/index/bm25_mathlib_v4160 \
        --data-root ~/data/benchmarks/REAL-Prover/Realprover/data \
        --lean-project ~/lean/mathlib_v4160 --limit 10

Everything except `--arm` is held fixed across arms, and every argument lands in the run manifest,
so two runs are comparable exactly when their manifests differ only in the arm.

Writes `attempts.jsonl` with one row per problem including the full search trace. Reported numbers
must be recomputable from that file alone — the summary this prints is a convenience, not the
record.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from prooflens_prover.data.benchmarks import load_benchmark
from prooflens_prover.prover.repertoire import RepertoirePolicy
from prooflens_prover.prover.search import SearchConfig, best_first_search
from prooflens_prover.retrieval.base import DEFAULT_TOP_K, NullRetriever, RetrievalStats
from prooflens_prover.retrieval.bm25 import BM25Retriever
from prooflens_prover.utils.io import JsonlAppender
from prooflens_prover.utils.logging import get_logger
from prooflens_prover.utils.manifest import RunManifest
from prooflens_prover.utils.seed import set_global_seed

log = get_logger(__name__)


def retriever_runtime_config(arm: str) -> dict[str, int] | None:
    """Query-side settings that change results but live in code rather than in the index.

    The index metadata records the checkpoint, dimension and corpus id — everything about the
    *premise* side. It cannot record `query_length`, which is applied when the **query** is encoded
    and is therefore invisible to the index entirely.

    That gap is not hypothetical: three full benchmark runs were completed at `query_length=256`
    before the locked value of 384 was restored, and nothing in their manifests distinguishes them
    from a run at 384 against the same index. A manifest that cannot tell two runs apart cannot
    support the comparison it exists to underwrite.
    """
    if arm == "li":
        from prooflens_prover.retrieval.dense import LI_DOCUMENT_LENGTH, LI_QUERY_LENGTH

        return {"query_length": LI_QUERY_LENGTH, "document_length": LI_DOCUMENT_LENGTH}
    if arm == "sv":
        from prooflens_prover.retrieval.dense import SV_MAX_SEQ_LENGTH

        return {"max_seq_length": SV_MAX_SEQ_LENGTH}
    return None


def build_retriever(arm: str, index_dir: Path | None, stats: RetrievalStats,
                    checkpoint: str | None = None, device: str | None = None):
    if arm == "none":
        return NullRetriever()
    if index_dir is None:
        raise SystemExit(f"--arm {arm} requires --index")

    if arm == "bm25":
        log.info("loading BM25 index from %s", index_dir)
        r = BM25Retriever.from_directory(index_dir)
        r.stats = stats
    elif arm in ("li", "sv"):
        # Imported here so `--arm none/bm25` never needs torch or pylate installed.
        from prooflens_prover.retrieval.dense import load_retriever

        log.info("loading %s index from %s (this also loads the query encoder)", arm, index_dir)
        r = load_retriever(arm, index_dir, checkpoint=checkpoint, device=device, stats=stats)
        log.info("encoder: %s", r.index.encoder.to_dict())
    else:
        raise SystemExit(f"unknown arm {arm!r}")

    log.info("index: %d premises, corpus_id=%s", r.index.n_docs, r.index.corpus_id)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--arm", required=True, choices=["none", "bm25", "li", "sv"])
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="query-encoder checkpoint for --arm li/sv; defaults to the one recorded "
                         "in the index metadata, which is what keeps queries and premises in step")
    ap.add_argument("--device", default=None, help="cuda / cpu for the query encoder")
    ap.add_argument("--lean-project", type=Path, default=None)
    ap.add_argument("--lean-version", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N problems (smoke runs)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--min-closers", type=int, default=RepertoirePolicy.min_closers,
                    help="candidate slots reserved for the shared repertoire before premise "
                         "tactics compete; set >= len(closers) for an additive (non-displacing) "
                         "comparison")
    ap.add_argument("--max-expansions", type=int, default=SearchConfig.max_expansions)
    ap.add_argument("--samples-per-step", type=int, default=SearchConfig.samples_per_step)
    ap.add_argument("--max-depth", type=int, default=SearchConfig.max_depth)
    ap.add_argument("--wall-clock", type=float, default=SearchConfig.wall_clock_s)
    ap.add_argument("--tactic-timeout", type=float, default=SearchConfig.tactic_timeout)
    ap.add_argument("--results-root", type=Path, default=Path("results/logs"))
    ap.add_argument("--resume", type=Path, default=None,
                    help="an existing results/logs/<run_id> to continue: problems already in its "
                         "attempts.jsonl are skipped and new ones appended to the same run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check-count", action="store_true",
                    help="allow a benchmark whose size differs from the published one")
    args = ap.parse_args()

    set_global_seed(args.seed)

    problems = load_benchmark(
        args.benchmark, args.data_root, check_count=not args.no_check_count
    )
    selected = problems[args.offset:]
    if args.limit is not None:
        selected = selected[: args.limit]
    log.info("%s: %d problems loaded, %d selected", args.benchmark, len(problems), len(selected))
    if not selected:
        raise SystemExit("no problems selected — check --offset/--limit")

    stats = RetrievalStats()
    retriever = build_retriever(
        args.arm, args.index, stats, checkpoint=args.checkpoint, device=args.device
    )
    policy = RepertoirePolicy(
        retriever=retriever, top_k=args.top_k, min_closers=args.min_closers
    )

    cfg = SearchConfig(
        max_expansions=args.max_expansions,
        samples_per_step=args.samples_per_step,
        max_depth=args.max_depth,
        wall_clock_s=args.wall_clock,
        tactic_timeout=args.tactic_timeout,
    )

    # Resume is resolved BEFORE the Lean backend is built: if nothing is left to do, this returns
    # without paying the ~90 s `import Mathlib` (~440 s on CSF3's NFS).
    n_done_before = 0
    if args.resume is not None:
        manifest = RunManifest.load(args.resume)
        if manifest.config.get("arm") != args.arm:
            raise SystemExit(
                f"refusing to resume: that run was arm={manifest.config.get('arm')!r}, you passed "
                f"--arm {args.arm!r}. Two arms sharing one attempts.jsonl would be uninterpretable."
            )
        done: set[str] = set()
        if manifest.attempts_path.exists():
            for line in manifest.attempts_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["problem_id"])
        n_done_before = len(done)
        selected = [p for p in selected if p.id not in done]
        log.info("resuming %s: %d already recorded, %d remaining",
                 manifest.run_id, n_done_before, len(selected))
        if not selected:
            log.info("nothing left to do")
            return
    else:
        manifest = RunManifest.create(
            name=f"{args.benchmark}_{args.arm}_repertoire",
            config={
                "benchmark": args.benchmark,
                "arm": args.arm,
                "policy": policy.name,
                "top_k": args.top_k,
                "min_closers": args.min_closers,
                "index": str(args.index) if args.index else None,
                "corpus_id": getattr(getattr(retriever, "index", None), "corpus_id", None),
                "encoder": getattr(
                    getattr(getattr(retriever, "index", None), "encoder", None), "to_dict",
                    lambda: None
                )(),
                "retriever_runtime": retriever_runtime_config(args.arm),
                "n_problems": len(selected),
                "offset": args.offset,
                "search": cfg.to_dict(),
                "lean_project": str(args.lean_project) if args.lean_project else None,
                "lean_version": args.lean_version,
            },
            seed=args.seed,
            results_root=args.results_root,
            capture_lean=True,
        )

    from prooflens_prover.lean.leaninteract_backend import LeanInteractBackend

    log.info("starting Lean backend (first `import Mathlib` costs ~90-160s and ~4GB RSS)")
    t0 = time.perf_counter()
    backend = LeanInteractBackend(
        project_dir=args.lean_project,
        lean_version=args.lean_version,
        tactic_timeout=args.tactic_timeout,
    )
    log.info("backend ready in %.1fs", time.perf_counter() - t0)

    # Elaborate every distinct import set BEFORE the first timed search. A benchmark normally has
    # exactly one. Without this the first problem pays for `import Mathlib` out of its own
    # wall-clock budget and is recorded as EXHAUSTED having tried nothing — see warm_header.
    for header in dict.fromkeys(p.imports for p in selected):
        backend.warm_header(header)

    n_proved = 0
    t_start = time.perf_counter()

    # The appender fsyncs every row, so a run killed by the SLURM walltime still leaves a complete,
    # readable record of every problem it finished.
    with JsonlAppender(manifest.attempts_path) as attempts:
        for i, problem in enumerate(selected, 1):
            t = time.perf_counter()
            result = best_first_search(
                backend, policy, problem.statement, cfg, header=problem.imports
            )
            n_proved += int(result.proved)
            attempts.append(
                {
                    "problem_id": problem.id,
                    "source": problem.source,
                    "arm": args.arm,
                    **result.to_dict(),
                }
            )
            # Per-problem progress, flushed. A benchmark run is long enough that silence is
            # indistinguishable from a hang, which has cost this project real time before.
            mark = "PROVED" if result.proved else result.status.value.upper()
            print(
                f"[{i}/{len(selected)}] {problem.id:<40} {mark:<10} "
                f"{time.perf_counter() - t:6.1f}s  running {n_proved}/{i} "
                f"({100 * n_proved / i:.1f}%)",
                flush=True,
            )

    elapsed = time.perf_counter() - t_start

    # Totals are recomputed from the attempts file, not from this session's counters, so a resumed
    # run reports the whole benchmark rather than only the part that ran after the restart.
    rows = [
        json.loads(line)
        for line in manifest.attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total = len(rows)
    total_proved = sum(1 for r in rows if r.get("proved"))

    # Count the statuses that mean "not actually attempted". A run where these are non-zero is not
    # a clean measurement of the arm, and the number belongs in the manifest rather than being
    # rediscovered by hand — the first full FATE-M run lost 32/141 problems to a REPL restart and
    # still finalised as a success.
    n_error = sum(1 for r in rows if r.get("status") == "error")
    n_stale = getattr(backend, "n_stale_env_recoveries", 0)

    manifest.finalize(
        n_problems=total,
        n_proved=total_proved,
        pass_rate=round(total_proved / max(total, 1), 4),
        elapsed_s=round(elapsed, 1),
        n_this_session=len(selected),
        n_resumed=n_done_before,
        n_error=n_error,
        n_stale_env_recoveries=n_stale,
        retrieval=stats.to_dict(),
    )

    print()
    print(f"=== {args.benchmark} / arm={args.arm} / policy={policy.name} ===")
    print(f"proved      : {total_proved}/{total}  ({100 * total_proved / max(total, 1):.1f}%)")
    if n_done_before:
        print(f"              ({len(selected)} this session, {n_done_before} resumed)")
    if n_error or n_stale:
        # Loud, because the failure this guards against completed successfully and looked normal.
        print(f"!! WARNING  : {n_error} problems ended in harness ERROR "
              f"({n_stale} REPL restarts recovered). Those were not attempted; the pass rate "
              f"above uses all {total} as the denominator. Investigate before reporting.")
    print(f"wall clock  : {elapsed:.1f}s  ({elapsed / max(len(selected), 1):.1f}s per problem)")
    if stats.n_queries:
        print(f"retrieval   : {stats.to_dict()}")
    print(f"attempts    : {manifest.attempts_path}")


if __name__ == "__main__":
    main()
