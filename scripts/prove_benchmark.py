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
import sys
import time
from pathlib import Path

# Importable from a fresh clone with no install and no exported PYTHONPATH: a login-node
# `python scripts/<this>.py` must work, because that is how the analysis scripts get run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.data.benchmarks import load_benchmark

# Safe at module scope: `vllm_policy` imports vLLM only inside `VLLMGenerator`, so this costs
# nothing in the retrieval environment. Imported here rather than inside `build_policy` so the
# argparse defaults below can be *derived* from `SamplingConfig` instead of restated. They were
# restated once, and the restatement (`--temperature 1.0 --top-p 1.0`) silently overrode
# REAL-Prover's actual 1.5/0.9 on every run, because an argparse default is always passed.
from prooflens_prover.prover.prompt import DEFAULT_TEMPLATE, TEMPLATES
from prooflens_prover.prover.repertoire import RepertoirePolicy
from prooflens_prover.prover.search import SearchConfig, best_first_search
from prooflens_prover.prover.vllm_policy import SamplingConfig, VLLMGenerator, VLLMPolicy
from prooflens_prover.retrieval.base import (
    DEFAULT_TOP_K,
    PROMPT_PREMISE_LIMIT,
    NullRetriever,
    RetrievalStats,
)
from prooflens_prover.retrieval.bm25 import BM25Retriever
from prooflens_prover.retrieval.fusion import DEFAULT_FETCH_K
from prooflens_prover.retrieval.fusion import MODES as FUSION_MODES
from prooflens_prover.utils.io import JsonlAppender, read_jsonl_tolerant
from prooflens_prover.utils.logging import ensure_utf8_output, get_logger
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
    if arm == "fusion":
        # Both sub-retrievers' query-side settings, since the fusion arm encodes the query twice
        # and neither index records how. Merged rather than nested so the field keeps the flat
        # shape every other arm writes.
        return {**(retriever_runtime_config("sv") or {}), **(retriever_runtime_config("li") or {})}
    return None


def effective_n_candidates(retriever) -> int | None:
    """The first-stage budget actually in force, read back off the retriever.

    Recorded separately from the `--n-candidates` argument because the argument may be absent while
    the index still carries a stored value. What matters for interpreting a result is the budget the
    run *used*, not whether it was named on the command line — LI's measured recall@10 against exact
    MaxSim ranges from 0.443 at 1,000 to 0.979 at 50,000, so two runs differing only in this are not
    the same experiment.

    The fusion arm has no index of its own, so its sub-retrievers are walked: whichever of them has
    a first-stage budget (in practice the late-interaction one) supplies the figure. Reporting
    `None` for a fusion run would hide the single setting most able to change its result.
    """
    if (n := getattr(getattr(retriever, "index", None), "n_candidates", None)) is not None:
        return n
    for sub in getattr(retriever, "retrievers", ()):
        if (n := getattr(getattr(sub, "index", None), "n_candidates", None)) is not None:
            return n
    return None


def fused_corpus_id(retriever) -> str | None:
    """The corpus id every sub-retriever agrees on, or a hard failure if they do not.

    Fusing two rankings only means anything if both retrievers ranked the *same* premise set.
    Indices built over different corpora would still fuse — producing a plausible ranking over a
    union of two corpora that no arm in the study was ever measured against — so the mismatch is
    refused here rather than discovered in the results table. This is `--assert-corpus-id` from
    index build time, enforced again at run time where the two indices finally meet.
    """
    ids = {getattr(getattr(sub, "index", None), "corpus_id", None)
           for sub in getattr(retriever, "retrievers", ())}
    ids.discard(None)
    if len(ids) > 1:
        raise SystemExit(
            f"fusion sub-retrievers were built over different corpora: {sorted(ids)}. "
            "Rebuild both indices with --assert-corpus-id 276070:31db61c63a9b7ee1."
        )
    return ids.pop() if ids else None


def same_index_path(was: str | None, now: str | None) -> bool:
    """Whether two recorded index paths name the same directory, allowing for how it was spelled.

    Compared by identity rather than by string because the same index has more than one legitimate
    spelling: the sbatch passes a repo-relative `data/index/...` (it `cd`s to the repo first), a
    hand run may pass the absolute path, and a manifest written on one platform carries that
    platform's separator. Refusing a resume over the spelling would be a false alarm; the thing
    worth refusing is `sv_ft_novel_lr3e6` against `li_ft_novel_bm25`, which no normalisation hides.
    """
    if was is None or now is None:
        return True
    a = was.replace("\\", "/").rstrip("/")
    b = now.replace("\\", "/").rstrip("/")
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def resume_mismatches(manifest, args, cfg, n_candidates: int | None) -> list[str]:
    """Every way the requested configuration differs from the run being resumed.

    A resumed run appends to one `attempts.jsonl` and reports one pass rate over it, so the two
    halves have to be the same experiment. If they are not, the resulting number belongs to neither
    half and nothing in the record says so — the manifest still describes the *original*
    configuration, because `RunManifest.load` deliberately preserves it.

    `--seed` is the check this function was written for. The seed reaches vLLM as `LLM(seed=...)`,
    and the sbatch defaults it to 0. So resuming a seed-6 run without repeating `--seed 6` silently
    appends seed-0 draws under a manifest that says seed 6 — two different draws stitched together,
    presented to `passk_union.py` as one, which is exactly the double-counting its duplicate-seed
    refusal exists to prevent. It cannot catch this one: the seed it reads is the one that lied.

    The search budget is compared field by field rather than by a chosen subset, because every field
    in it is part of the budget. Returns human-readable lines; the caller decides to refuse.
    """
    out: list[str] = []
    cfgm = manifest.config

    def note(label: str, was, now, why: str) -> None:
        if was is not None and was != now:
            out.append(f"  {label}: the run has {was!r}, you passed {now!r} — {why}")

    note("seed", manifest.seed, args.seed,
         "the sampling draw. Two draws in one attempts.jsonl is not a draw")
    note("benchmark", cfgm.get("benchmark"), args.benchmark,
         "a different problem set, so the pass rate would have two denominators")
    note("arm", cfgm.get("arm"), args.arm, "two arms sharing one run is uninterpretable")
    note("policy_kind", cfgm.get("policy_kind"), args.policy,
         "a 7B model and a 19-tactic repertoire produce a rate belonging to neither")
    now_index = str(args.index) if args.index else None
    if not same_index_path(cfgm.get("index"), now_index):
        out.append(f"  index: the run has {cfgm.get('index')!r}, you passed {now_index!r} — "
                   "a different premise ranking for the second half of the same run")
    note("n_candidates", cfgm.get("n_candidates"), n_candidates,
         "LI recall@10 runs 0.443 to 0.979 across this range")
    note("premise_free_fraction", (cfgm.get("policy_config") or {}).get("premise_free_fraction"),
         args.premise_free_fraction, "a different prompt mix")

    was_search = cfgm.get("search") or {}
    now_search = cfg.to_dict()
    for key in sorted(set(was_search) | set(now_search)):
        note(f"search.{key}", was_search.get(key), now_search.get(key), "a different search budget")
    return out


def build_retriever(arm: str, index_dir: Path | None, stats: RetrievalStats,
                    checkpoint: str | None = None, device: str | None = None,
                    n_candidates: int | None = None):
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
        if n_candidates is not None:
            if arm != "li":
                raise SystemExit(
                    f"--n-candidates applies to the two-stage li arm only, not {arm!r} "
                    "(sv ranks the whole corpus exactly, so it has no first stage to widen)"
                )
            log.info(
                "first-stage budget: %d -> %d (%.2f%% of corpus). Measured recall@10 against exact "
                "MaxSim on real queries: 0.443 at 1000, 0.979 at 50000 — see "
                "results/tables/li_recall_fate_m.json",
                r.index.n_candidates, n_candidates, 100 * n_candidates / r.index.n_docs,
            )
            r.index.n_candidates = n_candidates
    else:
        raise SystemExit(f"unknown arm {arm!r}")

    log.info("index: %d premises, corpus_id=%s", r.index.n_docs, r.index.corpus_id)
    return r


def build_fusion_retriever(args, stats: RetrievalStats, repo_root: Path):
    """The `fusion` arm: one retriever per architecture, merged by `FusionRetriever`.

    ## Why the sub-retrievers are built separately rather than fused inside the retrieval server

    Under `--retrieval-python` each arm normally runs in its own subprocess. Fusion spawns **two**
    servers rather than teaching one server to hold both indices, and that is the cheaper design in
    every direction: the server needs no changes, each index keeps its own CUDA context, and — the
    reason that actually matters — the two can sit on **different devices**.

    That last point is not hypothetical. Late interaction's index is 5.5 GB and single-vector's is
    943 MB, and they share a GPU with a 7B model held at `gpu_memory_utilization=0.85`. Putting
    single-vector on the CPU costs about 100 ms a query against late interaction's measured 930 ms,
    which is invisible, and removes the only way this arm can fail on hardware the other arms fit.

    Each sub-retriever gets its **own** `RetrievalStats`. Sharing one would count two queries per
    `retrieve` call and report the arm as twice as busy at half the latency.
    """
    from prooflens_prover.retrieval.fusion import FusionRetriever

    if args.index_sv is None or args.index_li is None:
        raise SystemExit("--arm fusion requires both --index-sv and --index-li")

    def one(arm: str, index_dir: Path, device: str | None, n_candidates: int | None):
        sub_stats = RetrievalStats()
        if args.retrieval_python is not None:
            from prooflens_prover.retrieval.subprocess_client import spawn_retrieval_server

            return spawn_retrieval_server(
                args.retrieval_python, arm, index_dir, repo_root=repo_root,
                n_candidates=n_candidates, checkpoint=args.checkpoint, device=device,
                stats=sub_stats,
            )
        return build_retriever(arm, index_dir, sub_stats, checkpoint=args.checkpoint,
                               device=device, n_candidates=n_candidates)

    sv_device = args.fusion_sv_device or args.device
    log.info("fusion: sv on %s, li on %s, mode=%s", sv_device, args.device, args.fusion_mode)
    fused = FusionRetriever(
        retrievers=(
            one("sv", args.index_sv, sv_device, None),
            one("li", args.index_li, args.device, args.n_candidates),
        ),
        mode=args.fusion_mode,
        fetch_k=args.fusion_fetch_k,
        stats=stats,
    )
    log.info("fusion corpus_id=%s", fused_corpus_id(fused))
    return fused


#: Policy tag -> run-name segment. The run id must distinguish a model-free run from an LLM run of
#: the same arm: `build_table1.discover` keys on (benchmark, arm, policy), and without the tag a
#: 7B-model result and a 19-tactic-repertoire result would compete for the same table cell.
POLICY_TAGS = {"repertoire": "repertoire", "vllm": "vllm"}


def build_policy(args, retriever):
    """The tactic policy. `--arm` varies the retriever; `--policy` varies the generator."""
    if args.policy == "repertoire":
        if args.model is not None:
            raise SystemExit("--model applies to --policy vllm only")
        return RepertoirePolicy(
            retriever=retriever, top_k=args.top_k, min_closers=args.min_closers
        )

    if args.model is None:
        raise SystemExit("--policy vllm requires --model <path to weights>")

    informal: dict[str, str] = {}
    if args.informal_names is not None:
        from prooflens_prover.data.informal import check_coverage, load_informal_names

        informal = load_informal_names(
            args.informal_names, args.informal_formal_key, args.informal_gloss_key
        )
        # Joined against the real corpus, because a populated mapping can still match nothing:
        # `mathlib_informal_v4.16.0` stores names as path-component lists, and stringifying one
        # yields a key no premise has. That failure is invisible without this check.
        index = getattr(retriever, "index", None)
        if index is not None:
            check_coverage(informal, [r.name for r in index.records], args.min_gloss_coverage)

    sampling = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed if args.sampling_seed else None,
    )
    log.info("loading %s into vLLM (dtype=%s, gpu_mem_util=%.2f)",
             args.model, args.dtype, args.gpu_memory_utilization)
    generator = VLLMGenerator(
        args.model,
        # Both named, not folded into **llm_kwargs: their *difference* is the prompt-token budget
        # the generator truncates to, so it needs them separately rather than only forwarding
        # `max_model_len` to vLLM.
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    # `top_k=0` for the `none` arm means the retriever is never called, but the prompt shape is
    # unchanged — see `build_tactic_content`.
    #
    # Otherwise `VLLMPolicy.top_k`, which is REAL-Prover's `NUM_QUERYS = 10`, not `--prompt-limit`.
    # The rendered prompt is identical either way, since `format_premises` truncates to 6 and the
    # top 6 of 10 are the top 6 of 6 — but the manifest recorded 6 where their configuration says
    # 10, which makes two comparable runs look different.
    return VLLMPolicy(
        generator=generator,
        retriever=retriever,
        top_k=0 if args.arm == "none" else args.top_k_premises,
        prompt_limit=args.prompt_limit,
        template=args.template,
        sampling=sampling,
        informal_names=informal,
        strip_echo=args.strip_echo,
        premise_free_fraction=args.premise_free_fraction,
    )


def main() -> None:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--arm", required=True, choices=["none", "bm25", "li", "sv", "fusion"])
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--index-sv", type=Path, default=None,
                    help="--arm fusion: the single-vector index")
    ap.add_argument("--index-li", type=Path, default=None,
                    help="--arm fusion: the late-interaction index")
    ap.add_argument("--fusion-mode", default="rrf", choices=sorted(FUSION_MODES),
                    help="rrf rewards consensus; interleave guarantees each retriever reaches the "
                         "6 prompt slots. retrieval/fusion.py says why the choice is not obvious")
    ap.add_argument("--fusion-fetch-k", type=int, default=DEFAULT_FETCH_K,
                    help="premises requested from each sub-retriever before fusing")
    ap.add_argument("--fusion-sv-device", default="cpu",
                    help="device for the single-vector half of the fusion arm. Defaults to cpu: "
                         "the two indices are 5.5GB and 943MB and share a GPU with a 7B model, and "
                         "sv on cpu costs ~100ms against li's measured 930ms")
    ap.add_argument("--checkpoint", default=None,
                    help="query-encoder checkpoint for --arm li/sv; defaults to the one recorded "
                         "in the index metadata, which is what keeps queries and premises in step")
    ap.add_argument("--device", default=None, help="cuda / cpu for the query encoder")
    ap.add_argument("--n-candidates", type=int, default=None,
                    help="li only: premises kept by the pooled first stage before exact MaxSim "
                         "rerank. Defaults to the value stored in the index (1000). Measured "
                         "recall@10 vs exact MaxSim on real proof-state queries: 0.443 at 1000, "
                         "0.696 at 5000, 0.888 at 20000, 0.979 at 50000. A run at 1000 was seeing "
                         "under half its true top-10.")
    ap.add_argument("--retrieval-python", type=Path, default=None,
                    help="run retrieval in a subprocess under this interpreter instead of in "
                         "process. Required for --policy vllm on the li/sv arms: vllm needs "
                         "transformers>=5.5.3 and pylate needs <=5.3.0, which nothing satisfies. "
                         "Point it at the venv that has pylate.")
    ap.add_argument("--policy", default="repertoire", choices=sorted(POLICY_TAGS),
                    help="repertoire = model-free (Track A'); vllm = frozen LLM (Tier 1)")
    ap.add_argument("--model", default=None,
                    help="weights path or HF id, required for --policy vllm")
    ap.add_argument("--prompt-limit", type=int, default=PROMPT_PREMISE_LIMIT,
                    help="premises rendered into the prompt (REAL-Prover truncates to 6)")
    ap.add_argument("--top-k-premises", type=int, default=VLLMPolicy.top_k,
                    help="premises retrieved per state; REAL-Prover's NUM_QUERYS = 10. The prompt "
                         "renders only --prompt-limit of them, so this changes the manifest rather "
                         "than the prompt")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(TEMPLATES),
                    help="chat template. qwen_chatml is the one REAL-Prover-v1's tokenizer_config "
                         "ships; deepseek is what their code hard-codes and produced token salad "
                         "from these weights, kept only as an ablation")
    ap.add_argument("--informal-names", type=Path, default=None,
                    help="JSONL of informal premise names, e.g. mathlib_informal_v4.16.0")
    ap.add_argument("--informal-formal-key", default=None,
                    help="override the auto-detected formal-name field")
    ap.add_argument("--informal-gloss-key", default=None,
                    help="override the auto-detected informal-name field")
    ap.add_argument("--min-gloss-coverage", type=float, default=0.05,
                    help="fail if fewer than this fraction of corpus premises get an informal "
                         "name; guards against a mapping keyed on the wrong field. 0 disables.")
    # Derived from `SamplingConfig`, never restated. These are REAL-Prover's `PROVER_MODEL_PARAMS`
    # (temperature 1.5, top_p 0.9, max_tokens 256); hard-coding 1.0/1.0 here once made the corrected
    # values in `SamplingConfig` dead code, since argparse passes its default whether or not the
    # operator names the flag.
    ap.add_argument("--temperature", type=float, default=SamplingConfig.temperature)
    ap.add_argument("--top-p", type=float, default=SamplingConfig.top_p)
    ap.add_argument("--max-tokens", type=int, default=SamplingConfig.max_tokens)
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="context window; the KV cache, not the weights, is what fills the GPU")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                    help="three consumers share the GPU: the engine, the retrieval child's query "
                         "encoder, and this process's own CUDA context")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="skip CUDA-graph capture and torch.compile. Slower, but it removes most "
                         "startup compilation — the escape hatch when a compute node lacks the "
                         "toolchain some kernel wants to JIT (see ENGINE_ENV)")
    ap.add_argument("--sampling-seed", action="store_true",
                    help="seed vLLM sampling with --seed; off by default so pass@k stays honest")
    ap.add_argument("--premise-free-fraction", type=float, default=0.0,
                    help="fraction of each expansion's samples drawn from the no-retrieval prompt, "
                         "merged into the same candidate list. Targets the 12 problems late "
                         "interaction displaced from the control. 0.0 is the published behaviour")
    ap.add_argument("--no-strip-echo", dest="strip_echo", action="store_false",
                    help="keep echoed 'Assistant:' prefixes, as REAL-Prover does")
    ap.set_defaults(strip_echo=True)
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
    if args.arm == "fusion":
        retriever = build_fusion_retriever(
            args, stats, repo_root=Path(__file__).resolve().parent.parent
        )
    elif args.retrieval_python is not None and args.arm != "none":
        from prooflens_prover.retrieval.subprocess_client import spawn_retrieval_server

        retriever = spawn_retrieval_server(
            args.retrieval_python, args.arm, args.index,
            repo_root=Path(__file__).resolve().parent.parent,
            n_candidates=args.n_candidates, checkpoint=args.checkpoint,
            device=args.device, stats=stats,
        )
    else:
        retriever = build_retriever(
            args.arm, args.index, stats, checkpoint=args.checkpoint, device=args.device,
            n_candidates=args.n_candidates,
        )
    policy = build_policy(args, retriever)

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
        if bad := resume_mismatches(manifest, args, cfg, effective_n_candidates(retriever)):
            raise SystemExit(
                f"refusing to resume {manifest.run_id}: the configuration you passed is not the "
                "one that run was measuring.\n" + "\n".join(bad) +
                "\n\nA resumed run appends to one attempts.jsonl and reports one rate over it, and "
                "the manifest keeps the ORIGINAL configuration — so a mismatch here produces a "
                "number that no field in the record contradicts. Re-run with the values above, or "
                "drop --resume to start a new run."
            )
        done: set[str] = set()
        if manifest.attempts_path.exists():
            recorded, unreadable = read_jsonl_tolerant(manifest.attempts_path)
            done = {r["problem_id"] for r in recorded}
            if unreadable:
                # Tolerant rather than fatal, and the problem is deliberately NOT marked done: an
                # unreadable row is an unknown outcome, so re-attempting it is the only honest
                # option. Raising here instead would make a run with one bad row permanently
                # unresumable — the exact opposite of what an fsync per attempt is for.
                log.warning(
                    "%d row(s) in %s could not be parsed (lines %s); those problems will be "
                    "re-attempted. The bad rows REMAIN in the file and will break this run's own "
                    "totals at the end. Repair first:\n"
                    "  python scripts/repair_attempts.py %s",
                    len(unreadable), manifest.attempts_path,
                    ", ".join(str(n) for n, _ in unreadable), manifest.run_dir,
                )
        n_done_before = len(done)
        selected = [p for p in selected if p.id not in done]
        log.info("resuming %s: %d already recorded, %d remaining",
                 manifest.run_id, n_done_before, len(selected))
        if not selected:
            # A resume with nothing left to do is one of two very different situations, and the
            # difference decides whether any GPU time is owed.
            #
            # If the manifest has an outcome, the run is complete and this was a no-op. If it has
            # none, the run finished every problem and then died in the reporting block — so the
            # results are all on disk and *invisible*, because `passk_union.discover` and
            # `build_table1.discover` both skip a run without an outcome. Resubmitting cannot fix
            # that: it lands here again, every time. Measured on ProofNet / sv / seed 6 of the
            # pass@8 sweep, which recorded 186 of 186 in 4 h 59 m and exited 1 after the loop.
            if manifest.outcome is None:
                log.warning(
                    "nothing left to do, but %s has NO outcome: it finished all %d problems and "
                    "died before recording them, so every table skips it. Resuming again will not "
                    "help. Repair it on a login node in seconds:\n"
                    "  python scripts/finalize_run.py %s",
                    manifest.run_id, n_done_before, manifest.run_dir,
                )
                raise SystemExit(1)
            log.info("nothing left to do")
            return
    else:
        manifest = RunManifest.create(
            name=f"{args.benchmark}_{args.arm}_{POLICY_TAGS[args.policy]}",
            config={
                "benchmark": args.benchmark,
                "arm": args.arm,
                "policy": policy.name,
                # The *kind* separately from the name: `build_table1.discover` keys on it, so a
                # model-free run and an LLM run of the same arm land in different tables.
                "policy_kind": args.policy,
                "policy_config": policy.config() if hasattr(policy, "config") else None,
                "top_k": args.top_k,
                "min_closers": args.min_closers,
                "index": str(args.index) if args.index else None,
                # Fusion has two of everything, and neither is `--index`. Recorded as its own
                # field so a fusion run is never mistaken for a single-arm run whose `index` was
                # simply omitted — and so the mode, which changes which premises reach the prompt,
                # is in the record rather than only in the command line.
                "fusion": (
                    {**retriever.config(), "index_sv": str(args.index_sv),
                     "index_li": str(args.index_li), "sv_device": args.fusion_sv_device}
                    if args.arm == "fusion" else None
                ),
                "corpus_id": (
                    fused_corpus_id(retriever) if args.arm == "fusion"
                    else getattr(getattr(retriever, "index", None), "corpus_id", None)
                ),
                "encoder": getattr(
                    getattr(getattr(retriever, "index", None), "encoder", None), "to_dict",
                    lambda: None
                )(),
                "retriever_runtime": retriever_runtime_config(args.arm),
                # How retrieval was hosted. Not cosmetic: an in-process and a subprocess run of one
                # arm must be comparable, so the manifest has to show that the only difference was
                # the hosting — and that the corpus_id and encoder above came from the same index.
                "retrieval_transport": (
                    "subprocess" if args.retrieval_python is not None and args.arm != "none"
                    else "in_process"
                ),
                "retrieval_python": (
                    str(args.retrieval_python) if args.retrieval_python is not None else None
                ),
                "n_candidates": effective_n_candidates(retriever),
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
    # Tolerant, because this exact line once threw away a completed run. ProofNet / sv / seed 6 of
    # the pass@8 sweep printed all 186 problems, then raised JSONDecodeError here on a truncated
    # 118 KB row at line 55 — after 4 h 59 m, with every proof already durable on disk. A record the
    # aggregation cannot read is a reason to report a short denominator loudly, not to discard the
    # work: `attempts.jsonl` is appended with O_APPEND on NFS, where an append that big is not
    # atomic.
    rows, unreadable = read_jsonl_tolerant(manifest.attempts_path)
    total = len(rows)
    total_proved = sum(1 for r in rows if r.get("proved"))

    # Count the statuses that mean "not actually attempted". A run where these are non-zero is not
    # a clean measurement of the arm, and the number belongs in the manifest rather than being
    # rediscovered by hand — the first full FATE-M run lost 32/141 problems to a REPL restart and
    # still finalised as a success.
    n_error = sum(1 for r in rows if r.get("status") == "error")
    n_stale = getattr(backend, "n_stale_env_recoveries", 0)

    # Policy-side counters. Without these the run is unauditable in the one way that matters for an
    # LLM arm: if one arm's prompt induces more empty or cheating generations, its effective sample
    # count per expansion is lower than the other's and the arms are no longer running the same
    # search budget — which would present as a retrieval effect. `mean_candidates_per_expansion`
    # against `--samples-per-step` is the health check; `getattr` because the model-free
    # `RepertoirePolicy` has no counters to report.
    policy_stats = policy.stats.to_dict() if hasattr(policy, "stats") else None
    generator = getattr(policy, "generator", None)
    generator_stats = generator.stats() if hasattr(generator, "stats") else None

    manifest.finalize(
        n_problems=total,
        n_proved=total_proved,
        pass_rate=round(total_proved / max(total, 1), 4),
        elapsed_s=round(elapsed, 1),
        n_this_session=len(selected),
        n_resumed=n_done_before,
        n_error=n_error,
        # Not cosmetic: each unreadable row is a problem missing from `total`, so the rate above is
        # over a denominator smaller than the benchmark. Downstream a missing problem reads as
        # unsolved, which biases the arm downward — recorded so that bias is visible rather than
        # inferred from a count that does not match the benchmark size.
        n_unreadable_rows=len(unreadable),
        unreadable_row_lines=[n for n, _ in unreadable] or None,
        n_stale_env_recoveries=n_stale,
        retrieval=stats.to_dict(),
        # Fusion only: which half of the fused query was slow. The fused figure cannot say, and the
        # sub-retrievers' devices differ by design, so this is the arm's one real unknown.
        retrieval_components=(
            retriever.component_stats() if hasattr(retriever, "component_stats") else None
        ),
        policy_stats=policy_stats,
        generator_stats=generator_stats,
    )

    print()
    print(f"=== {args.benchmark} / arm={args.arm} / policy={policy.name} ===")
    print(f"proved      : {total_proved}/{total}  ({100 * total_proved / max(total, 1):.1f}%)")
    if n_done_before:
        print(f"              ({len(selected)} this session, {n_done_before} resumed)")
    if unreadable:
        # Louder than the error warning below, because this one changes the denominator.
        print(f"!! CORRUPT   : {len(unreadable)} row(s) unreadable at line(s) "
              f"{', '.join(str(n) for n, _ in unreadable)} — those problems are MISSING from the "
              f"{total} above, so this is not a rate over the whole benchmark. Repair and resume:\n"
              f"     python scripts/repair_attempts.py {manifest.run_dir}\n"
              f"     SEED={manifest.seed} RESUME={manifest.run_dir} ... sbatch "
              f"slurm/prove_benchmark_llm.sbatch")
    if n_error or n_stale:
        # Loud, because the failure this guards against completed successfully and looked normal.
        print(f"!! WARNING  : {n_error} problems ended in harness ERROR "
              f"({n_stale} REPL restarts recovered). Those were not attempted; the pass rate "
              f"above uses all {total} as the denominator. Investigate before reporting.")
    print(f"wall clock  : {elapsed:.1f}s  ({elapsed / max(len(selected), 1):.1f}s per problem)")
    if stats.n_queries:
        print(f"retrieval   : {stats.to_dict()}")
        # The number the fusion pilot exists to produce: single-vector's cost on the device this arm
        # puts it on. Printed beside the fused total so a smoke run answers it without any analysis.
        if hasattr(retriever, "component_stats"):
            for name, s in retriever.component_stats().items():
                print(f"              {name:>6}: {s.get('mean_latency_ms')} ms/query "
                      f"over {s.get('n_queries')} queries")
            print("              feed the sv figure to preflight_sweep.py --fusion-sv-ms")
    if policy_stats:
        print(f"policy      : {policy_stats}")
        # The one number that says whether the LLM arm is healthy. Near `--samples-per-step` means
        # the model is producing distinct, admissible tactics; near 1 means it is repeating itself
        # or almost everything is being rejected, and the search has nothing to choose between.
        per_expansion = policy_stats.get("mean_candidates_per_expansion")
        if per_expansion is not None:
            print(f"              {per_expansion} distinct usable candidates per expansion, "
                  f"out of {args.samples_per_step} sampled")
        # Read together with the count, never instead of it. A wrong prompt format produced *more*
        # distinct candidates than the right one (13 vs 8 on one state) because noise does not
        # repeat, so the count alone reported a broken run as healthy. Measured on this model:
        # about -0.34 in-distribution, about -2.78 out of it.
        quality = policy_stats.get("mean_candidate_logprob")
        if quality is not None:
            verdict = "looks in-distribution" if quality > -1.5 else "SUSPICIOUS — check --template"
            print(f"              mean candidate logprob {quality} ({verdict})")
    if generator_stats:
        print(f"generator   : {generator_stats}")
    print(f"attempts    : {manifest.attempts_path}")


if __name__ == "__main__":
    main()
