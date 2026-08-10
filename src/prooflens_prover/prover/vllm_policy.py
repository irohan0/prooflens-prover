"""A `TacticPolicy` backed by a language model, with retrieval in the prompt.

This is the Tier 1 arm: freeze a released prover (REAL-Prover-v1, 7B, `Qwen2.5-Math-7B` base) and
vary only the retriever, so the numbers are comparable to published work. The search harness needs
no changes — `propose(state, n, context) -> [(tactic, logprob)]` is the whole interface.

## Why the generator is injected

`VLLMPolicy` never imports vLLM. It talks to a `TacticGenerator` Protocol, and `VLLMGenerator`
adapts the real engine to it. Two reasons, both learned here:

* Every test in this suite is hermetic — no GPU, no weights, no network. A policy that imports vLLM
  at module scope cannot be tested at all, so the parts most likely to be wrong (deduplication,
  cheat rejection, ordering) would be exercised for the first time on an 8-hour cluster job.
* vLLM pins torch hard enough that it lives in its own virtualenv on the cluster. Importing it from
  the package would make the retrieval arms undeployable in the environment that has it.

## Scoring

The per-candidate score is the **per-token mean** log-probability,
`cumulative_logprob / max(n_tokens, 1)`, which is what REAL-Prover's `generator.py` returns to its
search. Ranking on the raw cumulative sum instead biases best-first toward short tactics, since the
sum falls monotonically with length — see `Generation.mean_logprob`. The `/ depth^0.5` penalty on
top of that is applied by the search, not here (`SearchConfig.length_penalty`).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

#: `lean.backend.TacticPolicy` is the *admissibility guard* (what the harness refuses to execute),
#: not the `search.TacticPolicy` Protocol this module implements. Two different things sharing one
#: name; aliased on import so the collision cannot be mistaken for a circular reference.
from prooflens_prover.lean.backend import ProofState
from prooflens_prover.lean.backend import TacticPolicy as TacticGuard
from prooflens_prover.prover.prompt import (
    DEFAULT_TEMPLATE,
    build_tactic_content,
    render_chat,
)
from prooflens_prover.retrieval.base import PROMPT_PREMISE_LIMIT, Premise, Retriever
from prooflens_prover.utils.logging import get_logger

log = get_logger(__name__)

#: Prompt-echo prefixes a model sometimes reproduces before its answer. Stripped rather than
#: rejected: the tactic after the echo is usually fine, and discarding it would silently shrink the
#: candidate set for one arm only if that arm's prompt happened to induce more echoing.
_ECHO_PREFIXES = ("Assistant:", "TACTIC:", "tactic:")

#: U+FFFD. Present in a generation when the tokenizer could not decode a byte sequence — vLLM emits
#: it for a partial multi-byte character at a token boundary. Lean cannot parse it, so a tactic
#: containing one cannot elaborate under any circumstances.
REPLACEMENT_CHAR = "�"


@dataclass(frozen=True)
class SamplingConfig:
    """Decoding parameters. Recorded in the run manifest, identical across arms.

    The defaults are **REAL-Prover's `PROVER_MODEL_PARAMS`, verbatim** from their `conf/config.py`:

        {"temperature": 1.5, "max_tokens": 256, "top_p": 0.9, "logprobs": 1}

    `temperature=1.5` is high enough to look like a mistake and is not one — best-first reranks the
    samples by log-probability afterwards, so the sampler's job is coverage, not precision. Guessing
    1.0 (as an earlier version of this file did) narrows the candidate set the search gets to choose
    from, and the effect would show up as the model being worse at proving.

    `logprobs=1` is **not** decoration and **not** optional. vLLM v1 only populates
    `CompletionOutput.cumulative_logprob` when logprobs were requested; without it the field is
    `None` for every candidate. An earlier version of this file recorded the opposite in this very
    docstring ("vLLM returns it regardless") and paired that with `float(o.cumulative_logprob or
    0.0)` in the generator, so every tactic would have scored exactly 0.0, best-first would have
    degenerated into alphabetical order, and the run would have finished and reported a complete,
    plausible, meaningless number. `VLLMGenerator` now raises rather than defaulting.

    `stop` deliberately does **not** include `"\\n"`. Some Lean tactics legitimately span lines
    (`induction`/`with` blocks, `calc` chains), and truncating at the first newline would convert a
    multi-line tactic into a different, shorter one that may still compile. Termination is the
    model's EOS plus a token cap; the turn markers here only catch a model that runs on into a new
    conversational turn.
    """

    temperature: float = 1.5
    top_p: float = 0.9
    max_tokens: int = 256
    stop: tuple[str, ...] = ("\nUser:", "\nSTATE:", "<｜end▁of▁sentence｜>")
    seed: int | None = None
    #: Load-bearing. See the class docstring: this is what makes `cumulative_logprob` exist.
    logprobs: int = 1
    #: Turn-end tokens to stop on, resolved to ids against the model's tokenizer by
    #: `VLLMGenerator`. Necessary because REAL-Prover-v1's `eos_token` is `<|endoftext|>` while a
    #: ChatML model ends its turn with `<|im_end|>` — vLLM stops on EOS and would otherwise run
    #: straight past the turn boundary and glue the next turn onto the tactic. A stop *string*
    #: cannot do this: `skip_special_tokens` strips special tokens before any string match.
    turn_end_tokens: tuple[str, ...] = ("<|im_end|>",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature, "top_p": self.top_p,
            "max_tokens": self.max_tokens, "stop": list(self.stop), "seed": self.seed,
            "logprobs": self.logprobs, "turn_end_tokens": list(self.turn_end_tokens),
        }


@dataclass(frozen=True)
class Generation:
    """One sampled continuation, with the raw numbers vLLM reports.

    Both fields are kept raw so the normalisation below is visible, testable, and auditable from a
    trace, rather than baked into whatever the generator happened to return.
    """

    text: str
    cumulative_logprob: float
    n_tokens: int = 0

    @property
    def mean_logprob(self) -> float:
        """Per-token mean log-probability — **the score REAL-Prover's search actually uses.**

        Their `generator.py` computes `output.cumulative_logprob / max(len(output.token_ids), 1)`
        and it is not an incidental detail. The raw cumulative logprob falls monotonically with
        length, so ranking on it makes the search systematically prefer *short* tactics regardless
        of quality — `simp` would outrank `rw [foo, bar] <;> simpa using baz` almost always.
        Dividing by the token count removes that bias.

        The `max(..., 1)` guard is theirs too: an empty generation would otherwise divide by zero.
        """
        return self.cumulative_logprob / max(self.n_tokens, 1)


@runtime_checkable
class TacticGenerator(Protocol):
    """Samples `n` continuations of one prompt. The only thing vLLM is needed for."""

    def generate(self, prompt: str, n: int, sampling: SamplingConfig) -> list[Generation]:
        ...


@dataclass
class PolicyStats:
    """Counters for the run manifest. Every rejection must be visible and attributable.

    A silent drop is the failure mode that matters: if one arm's prompt induces more empty or
    cheating generations, its effective sample count per expansion is lower than the other's, and
    the arms no longer run the same search budget. That would look like a retrieval result.

    ## Diversity is not health

    `mean_candidates_per_expansion` was the health gate, and it read **11.33 of 16 — "healthy"** on
    the run whose prompt format was wrong and whose tactics were multilingual noise. Measured
    afterwards on one state, the broken deepseek prompt produced *more* distinct tactics than the
    correct ChatML one (13 vs 8), because garbage does not repeat itself. Candidate count measures
    how much the search has to choose between, not whether any of it is worth choosing.

    `mean_candidate_logprob` is the number that separates them: −0.34 under the format the model was
    trained on, −2.78 under the one it was not. Both are needed, and neither alone is a gate.
    """

    n_prompts: int = 0
    n_generated: int = 0
    n_empty: int = 0
    n_cheats: int = 0
    n_echo_stripped: int = 0
    n_undecodable: int = 0
    n_after_dedupe: int = 0
    total_prompt_chars: int = 0
    #: Summed per-token mean log-probability over the candidates actually proposed. The quality
    #: half of the health signal; see the class docstring for why the count alone was not enough.
    total_candidate_logprob: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "n_prompts": self.n_prompts, "n_generated": self.n_generated,
            "n_empty": self.n_empty, "n_cheats": self.n_cheats,
            "n_echo_stripped": self.n_echo_stripped, "n_undecodable": self.n_undecodable,
            "n_after_dedupe": self.n_after_dedupe,
        }
        if self.n_prompts:
            d["mean_prompt_chars"] = round(self.total_prompt_chars / self.n_prompts, 1)
            d["mean_candidates_per_expansion"] = round(self.n_after_dedupe / self.n_prompts, 2)
        if self.n_after_dedupe:
            d["mean_candidate_logprob"] = round(
                self.total_candidate_logprob / self.n_after_dedupe, 4
            )
        if self.n_generated:
            d["cheat_rate"] = round(self.n_cheats / self.n_generated, 4)
            d["empty_rate"] = round(self.n_empty / self.n_generated, 4)
        return d


def clean_tactic(text: str, strip_echo: bool = True) -> tuple[str, bool]:
    """Normalise one generated tactic. Returns `(tactic, echo_was_stripped)`.

    Whitespace only — no truncation. See `SamplingConfig.stop` for why a newline is not a boundary.
    `strip_echo=False` reproduces REAL-Prover's behaviour exactly: their generator only `.strip()`s.
    """
    tactic = text.strip()
    stripped = False
    if strip_echo:
        for prefix in _ECHO_PREFIXES:
            if tactic.startswith(prefix):
                tactic = tactic[len(prefix):].strip()
                stripped = True
    return tactic, stripped


@dataclass
class VLLMPolicy:
    """Retrieval-augmented next-tactic policy over a frozen language model."""

    generator: TacticGenerator
    retriever: Retriever
    #: Premises retrieved per state. 10 is REAL-Prover's `NUM_QUERYS`; their prompt builder then
    #: truncates to 6, so the rendered block is identical to retrieving 6 (it is a prefix) and the
    #: extra four cost nothing at these budgets — the LI arm's cost is the MaxSim over 50,000
    #: candidates, not `k`. Kept at 10 so the manifest matches their configuration exactly.
    top_k: int = 10
    prompt_limit: int = PROMPT_PREMISE_LIMIT
    template: str = DEFAULT_TEMPLATE
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    guard: TacticGuard = field(default_factory=TacticGuard)
    stats: PolicyStats = field(default_factory=PolicyStats)
    #: `formal name -> informal name`, from `FrenzyMath/mathlib_informal_v4.16.0`. Applied here, in
    #: the prompt path only, because that is the only place the field is read: the retrievers encode
    #: formal statements, `corpus_id` hashes names, and the 5.5 GB index stores its own records — so
    #: joining glosses at prompt time changes no embedding, invalidates no index, and needs no
    #: rebuild. Empty means every premise renders `Informal name: `, which is a documented downward
    #: bias on the calibration gate and identical across arms.
    informal_names: dict[str, str] = field(default_factory=dict)
    #: REAL-Prover only `.strip()`s a generation. Stripping an echoed `Assistant:`/`TACTIC:` prefix
    #: is our addition — arm-neutral and counted, and it converts a guaranteed-invalid tactic into a
    #: possibly-valid one. Set False for a byte-faithful reproduction of their behaviour.
    strip_echo: bool = True
    #: Set once the first prompt has been compared against the tokenizer's own `chat_template`.
    _format_checked: bool = field(default=False, repr=False)

    @property
    def name(self) -> str:
        return f"vllm+{getattr(self.retriever, 'name', 'unknown')}"

    def config(self) -> dict[str, Any]:
        """Everything that must match between arms, for the manifest."""
        return {
            "policy": "vllm",
            "top_k": self.top_k,
            "prompt_limit": self.prompt_limit,
            "template": self.template,
            "sampling": self.sampling.to_dict(),
            "retriever": getattr(self.retriever, "name", "unknown"),
            # Whether glosses were available is a property of the run, not a detail: it is the
            # documented downward bias on the calibration gate.
            "informal_names": len(self.informal_names),
            "strip_echo": self.strip_echo,
        }

    def _with_gloss(self, premise: Premise) -> Premise:
        """Attach the informal name, if we have one for this premise.

        `Premise` is frozen, so this returns a copy. A premise absent from the mapping keeps its
        empty gloss rather than inventing one — a humanised declaration name ("mul comm") is a
        fabrication, and fabricating a field the model was trained to read is worse than leaving it
        blank.
        """
        gloss = self.informal_names.get(premise.formal_name)
        if not gloss:
            return premise
        return replace(premise, informal_name=gloss)

    def propose(
        self, state: ProofState, n: int, context: dict[str, Any] | None = None  # noqa: ARG002
    ) -> list[tuple[str, float]]:
        """Up to `n` distinct `(tactic, logprob)` candidates, best first."""
        premises: list[Premise] = (
            self.retriever.retrieve(state.pp, k=self.top_k) if self.top_k > 0 else []
        )
        if self.informal_names:
            premises = [self._with_gloss(p) for p in premises]
        content = build_tactic_content(state.pp, premises, limit=self.prompt_limit)
        prompt = render_chat([{"role": "user", "content": content}], self.template)
        self.stats.n_prompts += 1
        self.stats.total_prompt_chars += len(prompt)

        if not self._format_checked:
            # Once per run, on a real prompt. The format was wrong for an entire FATE-M run and
            # nothing in the output said so: the model emitted fluent-looking tactic-shaped noise
            # rather than failing. The checkpoint ships the authoritative answer, so ask it.
            self._format_checked = True
            checker = getattr(self.generator, "check_prompt_format", None)
            if checker is not None and (problem := checker(prompt, content)):
                log.warning("PROMPT FORMAT MISMATCH (template=%s): %s", self.template, problem)

        generations = self.generator.generate(prompt, n, self.sampling)
        self.stats.n_generated += len(generations)

        best: dict[str, float] = {}
        for g in generations:
            tactic, echoed = clean_tactic(g.text, strip_echo=self.strip_echo)
            if echoed:
                self.stats.n_echo_stripped += 1
            if not tactic:
                self.stats.n_empty += 1
                continue
            if REPLACEMENT_CHAR in tactic:
                # A corrupted decode, observed at 1 in 16 samples in a preflight
                # ('�exact CommGroup.to_commMonoid'). Lean cannot parse U+FFFD, so this tactic
                # is guaranteed to fail — and letting it through would spend one of the expansion's
                # Lean calls to find that out. Counted rather than silently dropped: it is 6% of a
                # budget, and if the rate ever differs between arms that is a confound, not a
                # curiosity.
                self.stats.n_undecodable += 1
                continue
            if self.guard.reject_reason(tactic) is not None:
                # Rejected here rather than at the Lean call: a cheat tactic would otherwise consume
                # one of the expansion's Lean calls and, if it elaborated, produce an unsound proof.
                self.stats.n_cheats += 1
                continue
            # `mean_logprob`, not `cumulative_logprob` — see the property. Ranking on the raw sum
            # would bias the search toward short tactics.
            score = g.mean_logprob
            # A tactic sampled several times keeps its best score. Not summed: these are samples
            # of one distribution, and adding logprobs would rank a repeated mediocre tactic above
            # a single strong one purely for being sampled twice.
            if tactic not in best or score > best[tactic]:
                best[tactic] = score

        self.stats.n_after_dedupe += len(best)
        # Summed over the deduped candidates — what the search is actually offered. A NaN would
        # poison the running total for the whole run, and `propose` already maps NaN to -inf below,
        # so it is excluded here rather than allowed to make the manifest unreadable.
        self.stats.total_candidate_logprob += sum(
            s for s in best.values() if not math.isnan(s) and not math.isinf(s)
        )
        if not best:
            log.warning("no usable tactic from %d samples (empty=%d cheats=%d)",
                        len(generations), self.stats.n_empty, self.stats.n_cheats)

        # Deterministic: score descending, then tactic text, so two runs given the same generations
        # explore in the same order. `math.isnan` guard because a generator that returns NaN would
        # otherwise sort unpredictably and make a run irreproducible.
        return sorted(
            ((t, s if not math.isnan(s) else -math.inf) for t, s in best.items()),
            key=lambda kv: (-kv[1], kv[0]),
        )[:n]


#: Applied to the environment before vLLM is imported. See `_configure_engine_process`.
ENGINE_ENV: dict[str, str] = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}


def _configure_engine_process() -> dict[str, str]:
    """Make the engine immune to whatever CUDA state this process is already in.

    ## The failure this prevents

    vLLM v1 runs `EngineCore` in a child process created with `fork` by default, and a forked child
    cannot use CUDA if the parent ever touched the CUDA driver — PyTorch enforces that with a
    `pthread_atfork` handler. The handler is registered by `torch.cuda.is_available()`, a call that
    initializes nothing, so a process can be fork-poisoned while `torch.cuda.is_initialized()` is
    still False. That is exactly the state vLLM's own guard cannot detect, and the result was

        RuntimeError: Cannot re-initialize CUDA in forked subprocess

    raised inside EngineCore after a full 15 GB model load. `utils.seed` no longer makes that call,
    but "no library anywhere on the import path calls `torch.cuda.is_available()` before us" is a
    hope, not an invariant, so the fork is removed as well:

    * `VLLM_ENABLE_V1_MULTIPROCESSING=0` runs EngineCore **in this process** — no child, so nothing
      to inherit and nothing to poison. For a synchronous offline `LLM.generate()` loop the
      multiprocess client buys nothing anyway: it exists to overlap an API server's event loop with
      engine stepping, and there is no event loop here.
    * `VLLM_WORKER_MULTIPROC_METHOD=spawn` is the backstop if a future vLLM ignores the first. A
      spawned child starts from a fresh interpreter and inherits no CUDA state either.

    Two independent mechanisms plus the root-cause fix, because each attempt at this costs a cluster
    round-trip. `setdefault`, so an operator debugging vLLM itself can still override either.

    ## The compiler that is not on a compute node

    `VLLM_USE_FLASHINFER_SAMPLER=0` is here for an unrelated reason with the same shape. vLLM 0.26
    routes top-k/top-p sampling through FlashInfer, which **JIT-compiles its kernel on first use**
    and so needs a CUDA toolkit. A GPU compute node has the driver, not the toolkit:

        RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist

    raised during sampler warm-up, i.e. inside `LLM(...)`, after the weights were loaded. vLLM's
    native PyTorch top-p/top-k path needs no compiler, and sampling is negligible against a 7B
    forward pass over 16 sequences — the tokens-per-second cost is unmeasurable here.

    This one is **not** merely operational: which sampler runs determines which tokens get drawn, so
    it belongs in the run manifest. That is why these live in code and are recorded through
    `VLLMGenerator.stats()` rather than being set in the sbatch, where a reader of the results would
    have no way to know which implementation produced them.
    """
    applied: dict[str, str] = {}
    for key, value in ENGINE_ENV.items():
        applied[key] = os.environ.setdefault(key, value)
    return applied


class VLLMGenerator:
    """Adapts vLLM's offline `LLM` to `TacticGenerator`. The only vLLM-dependent code here.

    Offline `LLM` rather than an OpenAI-compatible server: one process per SLURM job, no port to
    allocate, no server lifecycle to supervise, and nothing left running if the job is killed at
    the wall clock. The search is sequential per problem, so a server buys no throughput.
    """

    def __init__(
        self,
        model: str,
        max_model_len: int | None = None,
        max_tokens: int = SamplingConfig.max_tokens,
        **llm_kwargs: Any,
    ) -> None:
        self.engine_env = _configure_engine_process()      # must precede the import
        from vllm import LLM  # imported here so the package works without vLLM installed

        self.model = model
        self.n_truncated = 0
        self.max_prompt_tokens = 0
        self._tokenizer: Any = None
        self._turn_end_cache: list[int] | None = None
        # Left-truncate any prompt that would not leave room to generate. `max_model_len` is a hard
        # ceiling and vLLM raises ValueError on an over-long prompt. (REAL-Prover-v1's `config.json`
        # declares `max_position_embeddings: 8192`, so the 4096 this run passes is our choice, not
        # the model's limit — an earlier comment here asserted the opposite. 4096 is ample: the
        # longest prompt measured on real FATE-M states was 702 tokens.)
        #
        # Retrieval is what inflates prompts, so an
        # unguarded overflow would fail problems on the li/sv arms and never on `none` — an
        # arm-dependent loss that looks exactly like a retrieval result. Truncation is from the
        # left, which drops the furthest premises first and always keeps the proof state and the
        # trailing `TACTIC:` marker, because `build_tactic_prompt` puts the state last.
        #
        # `SamplingParams.truncate_prompt_tokens` was the clean way to ask for this, and the vLLM on
        # the cluster rejects the keyword outright:
        #
        #     TypeError: Unexpected keyword argument 'truncate_prompt_tokens'
        #
        # so the work happens in `_prompt_arg` instead. One code path, no version dependence, and no
        # warning telling the operator that a protection they have is a protection they do not.
        self._truncate_at = (
            max(max_model_len - max_tokens, 1) if max_model_len is not None else None
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        # `dtype="bfloat16"` and `gpu_memory_utilization` are the caller's to set; the KV cache, not
        # the 15 GB of weights, is what fills an 80 GB A100.
        try:
            self.llm = LLM(model=model, **llm_kwargs)
        except RuntimeError as exc:
            raise SystemExit(self._diagnose_startup(exc)) from exc

    @staticmethod
    def _diagnose_startup(exc: RuntimeError) -> str:
        """Turn a known startup failure into one actionable line.

        vLLM's startup tracebacks are sixty lines deep in its own internals and name a symptom
        rather than a remedy. Each one of these has already cost a cluster round-trip, so the
        remedy is written down at the point of failure instead of rediscovered.
        """
        text = str(exc)
        hint = ""
        if "nvcc" in text or "cuda_home" in text or "CUDA_HOME" in text:
            hint = (
                "\n\nThis is a JIT compiler looking for a CUDA toolkit that a GPU compute node "
                "does not have — it ships the driver, not nvcc. Something asked to compile a "
                "kernel at startup. `ENGINE_ENV` already disables the FlashInfer sampler for "
                "exactly this reason; if the trace above names a different component, either "
                "disable that one the same way or pass --enforce-eager, which skips CUDA-graph "
                "capture and torch.compile and so removes most startup compilation."
            )
        elif "out of memory" in text.lower():
            hint = (
                "\n\nThree processes share this GPU: the engine, the retrieval child's query "
                "encoder (--device cuda), and this process's own context. Lower "
                "--gpu-memory-utilization, or --max-model-len, which is what sizes the KV cache."
            )
        return (
            f"vLLM failed to start: {type(exc).__name__}: {text}{hint}\n\n"
            "Nothing was written to results/, so no partial run needs cleaning up."
        )

    def stats(self) -> dict[str, Any]:
        """Generator-side counters for the run manifest."""
        return {
            "model": self.model,
            "engine_env": self.engine_env,
            "max_prompt_tokens_seen": self.max_prompt_tokens,
            "truncate_prompt_tokens": self._truncate_at,
            "n_prompts_truncated": self.n_truncated,
        }

    def _params(self, n: int, sampling: SamplingConfig) -> Any:
        from vllm import SamplingParams

        return SamplingParams(
            n=n,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            stop=list(sampling.stop),
            stop_token_ids=self._turn_end_ids(sampling.turn_end_tokens),
            seed=sampling.seed,
            # Required, not cosmetic: vLLM v1 leaves `cumulative_logprob` as None unless logprobs
            # were requested, and the search ranks on it. REAL-Prover's PROVER_MODEL_PARAMS carries
            # `"logprobs": 1` for the same reason.
            logprobs=sampling.logprobs,
        )

    def _turn_end_ids(self, names: tuple[str, ...]) -> list[int]:
        """Token ids for the turn-end markers, resolved once against the model's own tokenizer.

        Hard-coding 151645 would work for this checkpoint and silently mis-stop any other. A name
        the tokenizer does not know is skipped rather than fatal — `deepseek` templates have no
        `<|im_end|>`, and demanding one would break the ablation arm.
        """
        if self._turn_end_cache is not None:
            return self._turn_end_cache
        ids: list[int] = []
        tokenizer = self._get_tokenizer()
        for name in names:
            token_id = None
            if tokenizer is not None:
                token_id = (tokenizer.get_vocab() or {}).get(name)
            if token_id is None:
                log.info("tokenizer does not know %r; not stopping on it", name)
                continue
            ids.append(int(token_id))
            log.info("stopping generation on %r (token id %d)", name, token_id)
        self._turn_end_cache = ids
        return ids

    def check_prompt_format(self, prompt: str, content: str) -> str | None:
        """Compare our rendered prompt against the tokenizer's own `chat_template`.

        Returns None when they agree, or a description of the difference. This exists because the
        format has now been wrong once, expensively and silently: the model produced tactic-shaped
        multilingual noise for a whole run rather than failing in any way a log would show. The
        checkpoint ships the authoritative answer in `tokenizer_config.json`, so it can simply be
        asked.

        Reported rather than raised, because a deliberate template ablation (`--template deepseek`,
        reproducing their code exactly) is a legitimate run that must not be blocked. The preflight
        turns the same check into a hard gate, which is the right division: the gate enforces, the
        long run informs.
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is None or not getattr(tokenizer, "chat_template", None):
            return None
        try:
            expected = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:                         # noqa: BLE001 — a diagnostic, not a step
            return f"could not render the tokenizer's chat_template: {type(exc).__name__}: {exc}"
        if expected == prompt:
            return None
        return (
            "the prompt does not match the chat_template shipped with these weights.\n"
            f"  ours     ({len(prompt):5d} chars): {prompt[:160]!r}\n"
            f"  shipped  ({len(expected):5d} chars): {expected[:160]!r}\n"
            "Sending a format the model was never trained on produces fluent-looking nonsense, not "
            "an error — that is exactly how a whole FATE-M run was lost. If this is a deliberate "
            "template ablation, it is expected; otherwise use --template qwen_chatml."
        )

    def _prompt_arg(self, prompt: str) -> Any:
        """The prompt as vLLM should receive it — text, unless it cannot fit.

        The string path is left **completely untouched** for any prompt that fits, which is every
        prompt in practice: a preflight measured 160 tokens against a 3,840 limit. That matters
        because handing vLLM token ids instead of text also hands it responsibility for the special
        token prefix, and this prompt deliberately carries no BOS (see `prompt.py`). Changing how
        every prompt is encoded, to guard a case that has never occurred, would be the larger risk.
        """
        if self._truncate_at is None:
            return prompt
        ids = self._encode(prompt)
        if ids is None or len(ids) <= self._truncate_at:
            return prompt

        self.n_truncated += 1
        if self.n_truncated == 1:
            log.warning(
                "a prompt was %d tokens against a %d limit and has been left-truncated, dropping "
                "the furthest premises to keep the proof state. Counted in the manifest as "
                "n_prompts_truncated; if that number is not small the arms are not seeing "
                "comparable prompts and it has to be reported.",
                len(ids), self._truncate_at,
            )
        # Keeping the *tail* keeps the state and the trailing `TACTIC:`; dropping the head drops
        # premises, which is the only part of this prompt that is safe to lose.
        return {"prompt_token_ids": list(ids[-self._truncate_at:])}

    def _get_tokenizer(self) -> Any:
        """The engine's tokenizer, or None if it will not lend us one. Cached, including failure."""
        if self._tokenizer is None:
            try:
                self._tokenizer = self.llm.get_tokenizer()
            except Exception as exc:                     # noqa: BLE001 — optional capability
                log.warning("no tokenizer available (%s); prompt length is unguarded", exc)
                self._tokenizer = False
        return None if self._tokenizer is False else self._tokenizer

    def _encode(self, prompt: str) -> list[int] | None:
        """Token ids for a length check, or None if this engine will not lend us its tokenizer.

        None degrades to "send the text and let vLLM decide", which is exactly today's behaviour:
        one problem recorded as `status=error` rather than a job that dies. A missing tokenizer must
        not be fatal to a run that would otherwise have completed.
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None
        # `add_special_tokens=False` to match this prompt's deliberate absence of a BOS. An off-by-
        # one against vLLM's own count is irrelevant against a 3,840-token budget.
        return list(tokenizer(prompt, add_special_tokens=False).input_ids)

    def generate(self, prompt: str, n: int, sampling: SamplingConfig) -> list[Generation]:
        # `use_tqdm=False` because this is called once per expansion — roughly 9,000 times in a
        # FATE-M run — and each call would otherwise draw its own progress bar straight to stderr,
        # burying the per-problem logging that is the only way to tell a slow run from a hung one.
        [result] = self.llm.generate(
            [self._prompt_arg(prompt)], self._params(n, sampling), use_tqdm=False
        )
        self.max_prompt_tokens = max(self.max_prompt_tokens, len(result.prompt_token_ids or ()))

        generations: list[Generation] = []
        for o in result.outputs:
            if o.cumulative_logprob is None:
                # SystemExit, not RuntimeError, and deliberately: `best_first_search` catches
                # `Exception` so that one bad problem cannot abort a 672-problem run. That is right
                # for a proof failure and wrong for a misconfiguration — it would turn this into 141
                # per-problem errors and burn the whole allocation. SystemExit derives from
                # BaseException, so it passes straight through and stops the job on expansion one.
                raise SystemExit(
                    "vLLM returned cumulative_logprob=None. Every candidate would score 0.0, "
                    "best-first search would collapse to alphabetical order, and the run would "
                    "finish and report a plausible, meaningless number — so it stops here. vLLM v1 "
                    "only populates that field when logprobs are requested; this request passed "
                    f"logprobs={sampling.logprobs!r}. Fix SamplingConfig.logprobs, not this check."
                )
            # `.strip()` happens in `clean_tactic`, matching theirs; `Generation.mean_logprob`
            # applies their length normalisation. Both raw numbers are carried through so a trace
            # can be re-scored later without re-running the model.
            generations.append(
                Generation(
                    text=o.text,
                    cumulative_logprob=float(o.cumulative_logprob),
                    n_tokens=len(o.token_ids or ()),
                )
            )
        return generations
