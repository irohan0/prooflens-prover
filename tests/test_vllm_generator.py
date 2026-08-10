"""`VLLMGenerator` — the vLLM adapter — tested against a fake vLLM.

Every failure pinned here was found by reading a traceback from a GPU node, and every one of them
was reachable without a GPU. `VLLMGenerator` was previously dismissed as "a thin adapter with no
logic", and the adapter is where three of the four costliest bugs in this arm lived:

1. **`cumulative_logprob` was never requested.** vLLM v1 leaves it `None` unless
   `SamplingParams.logprobs` is set, and the old code wrote `float(o.cumulative_logprob or 0.0)`.
   Every candidate would have scored 0.0, best-first would have collapsed to alphabetical order, and
   the run would have *succeeded* and reported a meaningless number. Nothing in the results would
   have looked wrong.
2. **The engine forked.** vLLM v1 forks `EngineCore`, and a forked child cannot use CUDA if the
   parent ever called `torch.cuda.is_available()`. See `_configure_engine_process`.
3. **A progress bar per expansion.** ~9,000 calls per FATE-M run, each drawing its own tqdm to
   stderr, burying the only signal that distinguishes a slow run from a hung one.

Hermetic: a stub `vllm` module is installed into `sys.modules`, so this runs anywhere.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import preflight_llm  # noqa: E402
from prooflens_prover.prover.vllm_policy import (  # noqa: E402
    ENGINE_ENV,
    SamplingConfig,
    VLLMGenerator,
)

# --------------------------------------------------------------------------------------------------
# A fake vLLM, recording everything the adapter asks of it.
# --------------------------------------------------------------------------------------------------


@dataclass
class FakeCompletion:
    text: str
    cumulative_logprob: float | None
    token_ids: tuple[int, ...] = (1, 2, 3)


@dataclass
class FakeRequestOutput:
    outputs: list[FakeCompletion]
    prompt_token_ids: tuple[int, ...] = (1, 2)


class FakeSamplingParams:
    """Records the keywords it was given.

    A real `SamplingParams` rejects `truncate_prompt_tokens` on the cluster's vLLM, which is why
    truncation is done with the tokenizer instead and why this stub no longer needs to model an
    unsupported-keyword case.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeTokenizer:
    """One token per character, plus the parts of a real tokenizer this adapter touches.

    `get_vocab` and `apply_chat_template` are here because the generator now asks the checkpoint two
    questions it used to assume the answers to: which token id ends a turn, and what the shipped
    chat template renders.
    """

    #: A ChatML-ish stand-in for what REAL-Prover-v1 ships. Rendering is done in Python rather than
    #: Jinja, which is enough to test agreement and disagreement.
    chat_template: str | None = "<chatml>"
    vocab: dict[str, int] = {"<|im_end|>": 151645, "<|endoftext|>": 151643}

    def __call__(self, text: str, add_special_tokens: bool = True) -> Any:
        assert add_special_tokens is False, "this prompt deliberately carries no BOS"
        return types.SimpleNamespace(input_ids=list(range(len(text))))

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False) -> str:
        assert tokenize is False and add_generation_prompt is True
        from prooflens_prover.prover.prompt import render_chat

        return render_chat(list(messages), "qwen_chatml")


@dataclass
class FakeLLM:
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    #: What `generate` returns. Replaced per test.
    completions: list[FakeCompletion] = field(default_factory=list)
    prompt_token_ids: tuple[int, ...] = (1, 2)
    #: None models an engine that will not lend its tokenizer, which must degrade, not crash.
    tokenizer: Any = field(default_factory=FakeTokenizer)

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer on this engine")
        return self.tokenizer

    def generate(self, prompts, params, **kwargs):
        self.calls.append({"prompts": list(prompts), "params": params, **kwargs})
        return [
            FakeRequestOutput(
                outputs=list(self.completions), prompt_token_ids=self.prompt_token_ids
            )
        ]


@pytest.fixture
def fake_vllm(monkeypatch):
    """Install a stub `vllm` module and hand back the `LLM` instance the generator will build."""
    built: dict[str, FakeLLM] = {}

    class LLM(FakeLLM):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(init_kwargs=kwargs)
            built["llm"] = self

    module = types.ModuleType("vllm")
    module.LLM = LLM
    module.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    # The engine env is process-global, so each test must start from a clean slate — otherwise
    # `setdefault` silently observes whatever a previous test left behind.
    for key in ENGINE_ENV:
        monkeypatch.delenv(key, raising=False)
    return built, module


def build(fake_vllm, **kwargs: Any) -> tuple[VLLMGenerator, FakeLLM]:
    built, _ = fake_vllm
    gen = VLLMGenerator("/weights/REAL-Prover-v1", **kwargs)
    return gen, built["llm"]


# --------------------------------------------------------------------------------------------------
# The silent-wrong-answer bug: logprobs must be requested, and None must never be defaulted.
# --------------------------------------------------------------------------------------------------


def test_it_requests_logprobs(fake_vllm):
    """Without this, vLLM v1 returns cumulative_logprob=None and every score becomes 0.0."""
    gen, llm = build(fake_vllm)
    llm.completions = [FakeCompletion("simp", -2.0)]

    gen.generate("prompt", 16, SamplingConfig())

    assert llm.calls[0]["params"].kwargs["logprobs"] == 1


def test_it_passes_the_configured_sampling_parameters(fake_vllm):
    """REAL-Prover's PROVER_MODEL_PARAMS must reach vLLM, not this module's own opinion."""
    gen, llm = build(fake_vllm)
    llm.completions = [FakeCompletion("simp", -2.0)]

    gen.generate("prompt", 16, SamplingConfig())

    kwargs = llm.calls[0]["params"].kwargs
    assert kwargs["temperature"] == 1.5
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_tokens"] == 256
    assert kwargs["n"] == 16


def test_a_missing_cumulative_logprob_stops_the_job(fake_vllm):
    """The whole run must die, not one problem.

    `best_first_search` catches `Exception` so a single bad problem cannot abort a 672-problem run.
    That is right for a proof failure and catastrophic for a misconfiguration: it would convert this
    into 141 per-problem errors, burn the allocation, and leave a results file. `SystemExit` derives
    from `BaseException` and so passes straight through that handler.
    """
    gen, llm = build(fake_vllm)
    llm.completions = [FakeCompletion("simp", None)]

    with pytest.raises(SystemExit) as exc:
        gen.generate("prompt", 16, SamplingConfig())

    # The message must name the cause, because the symptom (a plausible pass rate) is invisible.
    assert "logprobs" in str(exc.value)
    assert not isinstance(exc.value, Exception), "must bypass best_first_search's except Exception"


def test_it_does_not_default_a_missing_logprob_to_zero(fake_vllm):
    """The specific regression: `float(o.cumulative_logprob or 0.0)`.

    A score of 0.0 is not a neutral fallback — it is the *highest possible* value for a log
    probability, so every candidate would tie at the top and sort by tactic text.
    """
    gen, llm = build(fake_vllm)
    llm.completions = [FakeCompletion("simp", None), FakeCompletion("ring", -1.0)]

    with pytest.raises(SystemExit):
        gen.generate("prompt", 16, SamplingConfig())


def test_it_reports_the_numbers_raw(fake_vllm):
    """Both raw values are carried through so a trace can be re-scored without the model."""
    gen, llm = build(fake_vllm)
    llm.completions = [FakeCompletion("simp", -4.0, token_ids=(1, 2, 3, 4))]

    [g] = gen.generate("prompt", 16, SamplingConfig())

    assert g.cumulative_logprob == -4.0
    assert g.n_tokens == 4
    assert g.mean_logprob == -1.0


# --------------------------------------------------------------------------------------------------
# The fork bug.
# --------------------------------------------------------------------------------------------------


def test_it_disables_the_engine_subprocess_before_importing_vllm(fake_vllm):
    """`Cannot re-initialize CUDA in forked subprocess`, prevented two independent ways.

    Applied at construction rather than checked at call time: vLLM reads these when it builds the
    engine, so setting them after `from vllm import LLM` would be too late to matter.
    """
    gen, _ = build(fake_vllm)

    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert gen.engine_env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_an_operator_can_override_the_engine_settings(fake_vllm, monkeypatch):
    """`setdefault`, so debugging vLLM itself does not require editing this file."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    gen, _ = build(fake_vllm)

    assert gen.engine_env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"


def test_the_engine_settings_are_recorded_in_the_manifest(fake_vllm):
    """A run whose engine was configured differently is a different run.

    This matters most for the sampler: which implementation draws the tokens changes *which tokens
    are drawn*, so a reader of the results needs to be able to tell. That is the argument for these
    living in code and being recorded, rather than being set in the sbatch.
    """
    gen, _ = build(fake_vllm)
    env = gen.stats()["engine_env"]
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


# --------------------------------------------------------------------------------------------------
# The missing compiler.
# --------------------------------------------------------------------------------------------------


def test_it_disables_the_sampler_that_needs_a_cuda_toolkit(fake_vllm):
    """vLLM 0.26 routes top-p sampling through FlashInfer, which JIT-compiles its kernel.

    A GPU compute node ships the driver, not `nvcc`, so warm-up died inside `LLM(...)` after loading
    the weights. vLLM's native top-p path needs no compiler and costs nothing measurable against a
    7B forward pass.
    """
    gen, _ = build(fake_vllm)
    assert gen.engine_env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_a_missing_nvcc_is_reported_with_its_remedy(fake_vllm, monkeypatch):
    """Sixty lines of vLLM internals naming a symptom, replaced by one line naming the fix."""
    built, module = fake_vllm

    class Exploding:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError(
                "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist"
            )

    module.LLM = Exploding
    with pytest.raises(SystemExit) as exc:
        VLLMGenerator("/weights/x")

    message = str(exc.value)
    assert "nvcc" in message
    assert "--enforce-eager" in message, "the message must name a remedy, not just the symptom"


def test_an_out_of_memory_start_names_the_three_gpu_consumers(fake_vllm):
    """The non-obvious part is that the retrieval child's query encoder is on the same GPU."""
    _, module = fake_vllm

    class Exploding:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    module.LLM = Exploding
    with pytest.raises(SystemExit) as exc:
        VLLMGenerator("/weights/x")

    assert "gpu-memory-utilization" in str(exc.value)


def test_an_unrecognised_startup_failure_is_still_reported_clearly(fake_vllm):
    """No hint is fine; swallowing the message is not."""
    _, module = fake_vllm

    class Exploding:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("something entirely new")

    module.LLM = Exploding
    with pytest.raises(SystemExit) as exc:
        VLLMGenerator("/weights/x")

    assert "something entirely new" in str(exc.value)


def test_enforce_eager_reaches_the_engine(fake_vllm):
    _, llm = build(fake_vllm, enforce_eager=True)
    assert llm.init_kwargs["enforce_eager"] is True


# --------------------------------------------------------------------------------------------------
# The wrong prompt format — the failure that cost a whole FATE-M run without producing one error.
# --------------------------------------------------------------------------------------------------


def test_it_stops_on_the_turn_end_token_not_only_on_eos(fake_vllm):
    """REAL-Prover-v1's `eos_token` is `<|endoftext|>`; a ChatML model ends its turn with
    `<|im_end|>`. vLLM stops on EOS and would run straight past the turn boundary, gluing whatever
    came next onto the tactic. A stop *string* cannot catch it either, because `skip_special_tokens`
    removes special tokens from the text before any string match — so it has to be a token id.
    """
    gen, llm = build(fake_vllm)
    llm.completions = [FakeCompletion("simp", -2.0)]

    gen.generate("prompt", 16, SamplingConfig())

    assert llm.calls[0]["params"].kwargs["stop_token_ids"] == [151645]


def test_the_stop_token_id_is_resolved_from_the_tokenizer_not_hard_coded(fake_vllm):
    """151645 is right for this checkpoint and would silently mis-stop any other."""
    gen, llm = build(fake_vllm)
    llm.tokenizer.vocab = {"<|im_end|>": 777}
    llm.completions = [FakeCompletion("simp", -2.0)]

    gen.generate("prompt", 16, SamplingConfig())

    assert llm.calls[0]["params"].kwargs["stop_token_ids"] == [777]


def test_an_unknown_turn_end_token_is_skipped_not_fatal(fake_vllm):
    """A deepseek template has no `<|im_end|>`, and demanding one would break the ablation arm."""
    gen, llm = build(fake_vllm)
    llm.tokenizer.vocab = {}
    llm.completions = [FakeCompletion("simp", -2.0)]

    [g] = gen.generate("prompt", 16, SamplingConfig())

    assert g.text == "simp"
    assert llm.calls[0]["params"].kwargs["stop_token_ids"] == []


def test_the_shipped_chat_template_is_the_authority(fake_vllm):
    """Our rendering of the default template must agree with the checkpoint's own."""
    from prooflens_prover.prover.prompt import build_tactic_content, build_tactic_prompt

    gen, _ = build(fake_vllm)
    content = build_tactic_content("⊢ True", [])

    assert gen.check_prompt_format(build_tactic_prompt("⊢ True", []), content) is None


def test_a_format_mismatch_is_reported_with_both_renderings(fake_vllm):
    """The deepseek prompt against ChatML weights, which is exactly what happened."""
    from prooflens_prover.prover.prompt import build_tactic_content, build_tactic_prompt

    gen, _ = build(fake_vllm)
    content = build_tactic_content("⊢ True", [])

    problem = gen.check_prompt_format(build_tactic_prompt("⊢ True", [], "deepseek"), content)

    assert problem is not None
    assert "does not match the chat_template shipped" in problem
    # Both strings must appear, because "they differ" without showing how is not actionable.
    assert "ours" in problem and "shipped" in problem


def test_no_chat_template_means_nothing_to_check_against(fake_vllm):
    """A checkpoint that ships no template cannot contradict us, and must not block the run."""
    gen, llm = build(fake_vllm)
    llm.tokenizer.chat_template = None

    assert gen.check_prompt_format("anything at all", "content") is None


def test_a_tokenizer_that_cannot_render_is_reported_not_raised(fake_vllm):
    gen, llm = build(fake_vllm)

    def explode(*a: Any, **k: Any):
        raise ValueError("jinja is unhappy")

    llm.tokenizer.apply_chat_template = explode

    problem = gen.check_prompt_format("x", "content")
    assert problem is not None and "jinja is unhappy" in problem


# --------------------------------------------------------------------------------------------------
# Log volume and prompt length: the two ways a long run degrades rather than fails.
# --------------------------------------------------------------------------------------------------


def test_it_suppresses_the_per_call_progress_bar(fake_vllm):
    """One expansion is one call; a FATE-M run is ~9,000 of them."""
    gen, llm = build(fake_vllm)
    llm.completions = [FakeCompletion("simp", -2.0)]

    gen.generate("prompt", 16, SamplingConfig())

    assert llm.calls[0]["use_tqdm"] is False


def test_an_ordinary_prompt_is_sent_as_text_untouched(fake_vllm):
    """The path every real prompt takes, and it must stay byte-identical.

    Handing vLLM token ids also hands it responsibility for the special-token prefix, and this
    prompt deliberately carries no BOS. A preflight measured 160 tokens against a 3,840 limit, so
    re-encoding every prompt to guard a case that has never occurred would be the larger risk.
    """
    gen, llm = build(fake_vllm, max_model_len=4096, max_tokens=256)
    llm.completions = [FakeCompletion("simp", -2.0)]

    gen.generate("STATE:\n⊢ True\nTACTIC:\n", 16, SamplingConfig())

    assert llm.calls[0]["prompts"] == ["STATE:\n⊢ True\nTACTIC:\n"]
    assert gen.stats()["n_prompts_truncated"] == 0


def test_an_over_long_prompt_is_left_truncated_to_leave_room_to_generate(fake_vllm):
    """`max_model_len - max_tokens`, keeping the tail.

    vLLM raises `ValueError` on a prompt longer than the context window. Retrieval is what inflates
    prompts, so an unguarded overflow fails problems on li/sv and never on `none` — an arm-dependent
    loss indistinguishable from a retrieval effect. Keeping the tail keeps the proof state and the
    trailing `TACTIC:`, and drops premises, which is the only part safe to lose.
    """
    gen, llm = build(fake_vllm, max_model_len=4096, max_tokens=256)
    llm.completions = [FakeCompletion("simp", -2.0)]

    gen.generate("x" * 5000, 16, SamplingConfig())

    [sent] = llm.calls[0]["prompts"]
    assert isinstance(sent, dict), "an over-long prompt must go as token ids, not text"
    assert len(sent["prompt_token_ids"]) == 3840
    # The tail, not the head: the fake tokenizer numbers characters, so the last id must be 4999.
    assert sent["prompt_token_ids"][-1] == 4999
    assert gen.stats()["n_prompts_truncated"] == 1


def test_max_model_len_still_reaches_vllm(fake_vllm):
    """Naming it explicitly must not stop it being forwarded to the engine."""
    _, llm = build(fake_vllm, max_model_len=4096, max_tokens=256)

    assert llm.init_kwargs["max_model_len"] == 4096


def test_truncate_prompt_tokens_is_never_sent(fake_vllm):
    """The cluster's vLLM rejects the keyword outright, so nothing may pass it."""
    gen, llm = build(fake_vllm, max_model_len=4096, max_tokens=256)
    llm.completions = [FakeCompletion("simp", -2.0)]

    gen.generate("prompt", 16, SamplingConfig())

    assert "truncate_prompt_tokens" not in llm.calls[0]["params"].kwargs


def test_without_a_context_window_the_prompt_is_never_touched(fake_vllm):
    gen, llm = build(fake_vllm)
    llm.completions = [FakeCompletion("simp", -2.0)]

    gen.generate("x" * 5000, 16, SamplingConfig())

    assert llm.calls[0]["prompts"] == ["x" * 5000]


def test_an_engine_without_a_tokenizer_degrades_instead_of_dying(fake_vllm):
    """The guard must never itself become the failure.

    Without a tokenizer we cannot measure the prompt, so we send the text and let vLLM decide — one
    problem recorded as `status=error` rather than a job that dies. Losing a protection is not a
    reason to lose a run that would otherwise have finished.
    """
    gen, llm = build(fake_vllm, max_model_len=4096, max_tokens=256)
    llm.tokenizer = None
    llm.completions = [FakeCompletion("simp", -2.0)]

    [g] = gen.generate("x" * 5000, 16, SamplingConfig())

    assert g.text == "simp"
    assert llm.calls[0]["prompts"] == ["x" * 5000]


def test_it_records_the_longest_prompt_it_actually_sent(fake_vllm):
    """From vLLM's own count, which is authoritative, not from our approximation."""
    gen, llm = build(fake_vllm, max_model_len=4096, max_tokens=256)
    llm.completions = [FakeCompletion("simp", -2.0)]
    llm.prompt_token_ids = tuple(range(1200))

    gen.generate("prompt", 16, SamplingConfig())

    assert gen.stats()["max_prompt_tokens_seen"] == 1200


# --------------------------------------------------------------------------------------------------
# The preflight's own judgement. A gate that cannot fail is not a gate.
# --------------------------------------------------------------------------------------------------


def run_preflight(fake_vllm, monkeypatch, completions: list[FakeCompletion]) -> int:
    built, _ = fake_vllm

    class LLM(FakeLLM):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(init_kwargs=kwargs)
            self.completions = list(completions)
            built["llm"] = self

    sys.modules["vllm"].LLM = LLM
    monkeypatch.setattr(sys, "argv", ["preflight_llm.py", "--model", "/weights/x"])
    return preflight_llm.main()


def test_the_preflight_passes_on_a_healthy_engine(fake_vllm, monkeypatch, capsys):
    code = run_preflight(fake_vllm, monkeypatch, [
        FakeCompletion("simp [mul_comm]", -3.0),
        FakeCompletion("exact mul_comm a b", -1.5),
        FakeCompletion("ring", -6.0),
    ])

    assert code == 0
    assert "PREFLIGHT OK" in capsys.readouterr().out


def test_the_preflight_fails_when_the_model_repeats_itself(fake_vllm, monkeypatch, capsys):
    """A working engine and a useless search look identical in the pass rate.

    Sixteen copies of one tactic means best-first has a single option at every node, which is not a
    search. Catching it here costs five minutes; catching it from a finished FATE-M run costs the
    run and leaves a number nobody can interpret.
    """
    code = run_preflight(fake_vllm, monkeypatch, [FakeCompletion("simp", -2.0)] * 16)

    assert code == 1
    assert "distinct" in capsys.readouterr().out


def test_the_preflight_fails_on_uniformly_zero_scores(fake_vllm, monkeypatch, capsys):
    """The silent-wrong-answer bug, one layer out from `generate`'s own SystemExit.

    `generate` raises when vLLM hands back `None`. A generator that returned a literal 0.0 instead
    would slip past that and produce the same uniform, meaningless ranking, so the preflight checks
    the property that actually matters rather than the mechanism that usually violates it.
    """
    code = run_preflight(fake_vllm, monkeypatch, [
        FakeCompletion("simp", 0.0), FakeCompletion("ring", 0.0),
    ])

    assert code == 1
    assert "0.0" in capsys.readouterr().out


def test_the_preflight_runs_the_seeding_that_broke_the_arm(fake_vllm, monkeypatch):
    """It must take the dangerous path, not tiptoe around it.

    `set_global_seed` is the function whose `torch.cuda.is_available()` poisoned fork. A preflight
    that skipped it would pass while the real script still crashed.
    """
    src = (Path(preflight_llm.__file__)).read_text(encoding="utf-8")
    assert "set_global_seed(args.seed)" in src
