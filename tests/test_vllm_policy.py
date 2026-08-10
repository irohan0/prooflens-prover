"""The model-backed policy, tested without a model.

`VLLMPolicy` talks to a `TacticGenerator` Protocol, so everything that can actually be wrong —
deduplication, cheat rejection, candidate ordering, the no-retrieval path — is exercised here rather
than for the first time inside an 8-hour cluster job. The parts that need vLLM (`VLLMGenerator`) are
a thin adapter with no logic.

The counters matter as much as the candidates. If one arm's prompt induces more empty or cheating
generations, that arm gets fewer usable tactics per expansion, and the arms no longer run the same
search budget even though the config says they do. `PolicyStats` is what makes that visible.

Hermetic: no vLLM, no GPU, no weights, no network.
"""

from __future__ import annotations

import math

import pytest

from prooflens_prover.lean.backend import ProofState
from prooflens_prover.lean.backend import TacticPolicy as TacticGuard
from prooflens_prover.prover.vllm_policy import (
    REPLACEMENT_CHAR,
    Generation,
    PolicyStats,
    SamplingConfig,
    TacticGenerator,
    VLLMPolicy,
    clean_tactic,
)
from prooflens_prover.retrieval.base import Premise

STATE = ProofState(pid=1, goals=("a b : G\n⊢ a * b = b * a",))


class FakeGenerator:
    """Returns a fixed script of generations, and records what it was asked."""

    def __init__(self, generations: list[Generation]):
        self.generations = generations
        self.prompts: list[str] = []
        self.ns: list[int] = []

    def generate(self, prompt: str, n: int, sampling: SamplingConfig) -> list[Generation]:
        self.prompts.append(prompt)
        self.ns.append(n)
        return list(self.generations)


class FakeRetriever:
    name = "fake"

    def __init__(self, premises: list[Premise] | None = None):
        self.premises = premises or []
        self.queries: list[str] = []
        self.ks: list[int] = []

    def retrieve(self, query: str, k: int = 10) -> list[Premise]:
        self.queries.append(query)
        self.ks.append(k)
        return self.premises[:k]


def gen(text: str, lp: float) -> Generation:
    return Generation(text=text, cumulative_logprob=lp)


def policy(generations, premises=None, **kw) -> VLLMPolicy:
    return VLLMPolicy(
        generator=FakeGenerator(generations), retriever=FakeRetriever(premises), **kw
    )


class TestProtocolConformance:
    def test_fake_generator_satisfies_the_protocol(self):
        assert isinstance(FakeGenerator([]), TacticGenerator)

    def test_policy_satisfies_the_search_protocol(self):
        from prooflens_prover.prover.search import TacticPolicy as SearchPolicy

        # The whole design rests on this: the search harness needs no changes for the LLM arm.
        assert isinstance(policy([]), SearchPolicy)


class TestCleanTactic:
    @pytest.mark.parametrize("raw,expected", [
        ("  simp  ", "simp"),
        ("\nsimp [mul_comm]\n", "simp [mul_comm]"),
        ("Assistant: simp", "simp"),
        ("TACTIC:\napply foo", "apply foo"),
    ])
    def test_normalisation(self, raw, expected):
        assert clean_tactic(raw)[0] == expected

    def test_reports_whether_an_echo_was_stripped(self):
        assert clean_tactic("Assistant: simp")[1] is True
        assert clean_tactic("simp")[1] is False

    def test_multi_line_tactics_are_not_truncated(self):
        # Truncating at the first newline would turn a `calc` chain into a different tactic that may
        # still compile. Termination is the model's EOS plus a token cap, not post-hoc cutting.
        raw = "induction n with\n| zero => simp\n| succ k ih => simpa using ih"
        assert clean_tactic(raw)[0] == raw

    def test_whitespace_only_becomes_empty(self):
        assert clean_tactic("   \n  ")[0] == ""


class TestCandidateSelection:
    def test_returns_tactics_best_first(self):
        p = policy([gen("aesop", -3.0), gen("simp", -0.5), gen("ring", -1.5)])
        assert p.propose(STATE, 3) == [("simp", -0.5), ("ring", -1.5), ("aesop", -3.0)]

    def test_duplicates_keep_their_best_score_and_appear_once(self):
        # 32 samples at temperature 1.0 repeat constantly. Summing the logprobs would rank a
        # repeated mediocre tactic above a single strong one purely for being sampled twice.
        p = policy([gen("simp", -2.0), gen("simp", -0.5), gen("simp", -3.0), gen("ring", -1.0)])
        out = p.propose(STATE, 10)
        assert out == [("simp", -0.5), ("ring", -1.0)]

    def test_respects_n_after_deduplication(self):
        p = policy([gen(f"t{i}", -float(i)) for i in range(10)])
        assert len(p.propose(STATE, 4)) == 4

    def test_ties_break_on_tactic_text_for_reproducibility(self):
        p = policy([gen("bbb", -1.0), gen("aaa", -1.0)])
        assert [t for t, _ in p.propose(STATE, 2)] == ["aaa", "bbb"]

    def test_nan_scores_sort_last_rather_than_unpredictably(self):
        p = policy([gen("nan_one", float("nan")), gen("fine", -5.0)])
        out = p.propose(STATE, 2)
        assert out[0][0] == "fine"
        assert out[1][0] == "nan_one" and out[1][1] == -math.inf

    def test_empty_generations_give_no_candidates(self):
        assert policy([]).propose(STATE, 8) == []

    def test_n_is_forwarded_to_the_generator(self):
        p = policy([gen("simp", -1.0)])
        p.propose(STATE, 32)
        assert p.generator.ns == [32]


class TestRejection:
    @pytest.mark.parametrize("cheat", [
        "sorry", "exact sorry", "simp; admit", "exact sorryAx _", "native_decide",
    ])
    def test_cheating_tactics_never_reach_the_search(self, cheat):
        p = policy([gen(cheat, -0.1), gen("simp", -2.0)])
        assert p.propose(STATE, 5) == [("simp", -2.0)]
        assert p.stats.n_cheats == 1

    def test_a_cheat_is_dropped_even_when_it_is_the_best_scoring_sample(self):
        # `sorry` is the highest-likelihood continuation surprisingly often. Rejecting here rather
        # than at the Lean call also saves one of the expansion's Lean calls.
        p = policy([gen("sorry", -0.01)])
        assert p.propose(STATE, 5) == []

    def test_native_decide_can_be_allowed_explicitly(self):
        p = policy([gen("native_decide", -1.0)],
                   guard=TacticGuard(allow_native_decide=True))
        assert p.propose(STATE, 5) == [("native_decide", -1.0)]

    def test_empty_generations_are_counted_not_proposed(self):
        p = policy([gen("   ", -0.1), gen("\n\n", -0.2), gen("simp", -1.0)])
        assert p.propose(STATE, 5) == [("simp", -1.0)]
        assert p.stats.n_empty == 2


class TestPromptConstruction:
    def test_the_goal_is_the_retrieval_query(self):
        p = policy([gen("simp", -1.0)])
        p.propose(STATE, 4)
        assert p.retriever.queries == [STATE.pp]

    def test_retrieved_premises_appear_in_the_prompt(self):
        prem = [Premise("mul_comm", "∀ a b, a * b = b * a")]
        p = policy([gen("simp", -1.0)], premises=prem)
        p.propose(STATE, 4)
        assert "mul_comm" in p.generator.prompts[0]
        assert "STATE:" in p.generator.prompts[0]

    def test_top_k_defaults_to_their_num_querys(self):
        # REAL-Prover's `NUM_QUERYS = 10`; their prompt builder then truncates to 6, so the rendered
        # block is a prefix and identical either way. 10 keeps our manifest matching their config.
        p = policy([gen("simp", -1.0)])
        p.propose(STATE, 4)
        assert p.retriever.ks == [10]

    def test_only_six_of_the_ten_retrieved_premises_are_rendered(self):
        prem = [Premise(f"l{i}", f"s{i}") for i in range(10)]
        p = policy([gen("simp", -1.0)], premises=prem)
        p.propose(STATE, 4)
        assert p.generator.prompts[0].count("ID:") == 6
        assert "l5" in p.generator.prompts[0] and "l6" not in p.generator.prompts[0]

    def test_top_k_zero_skips_retrieval_entirely(self):
        # The `none` arm: no retriever call at all, but the prompt shape is unchanged.
        p = policy([gen("simp", -1.0)], top_k=0)
        p.propose(STATE, 4)
        assert p.retriever.queries == []
        assert "Here're some theorems that may be helpful:\n\nSTATE:" in p.generator.prompts[0]

    def test_prompt_limit_caps_premises_independently_of_top_k(self):
        prem = [Premise(f"l{i}", f"s{i}") for i in range(10)]
        p = policy([gen("simp", -1.0)], premises=prem, top_k=10, prompt_limit=2)
        p.propose(STATE, 4)
        assert p.generator.prompts[0].count("ID:") == 2


class TestStatsAndConfig:
    def test_counters_track_every_generation(self):
        p = policy([gen("simp", -1.0), gen("simp", -2.0), gen("sorry", -0.1), gen("  ", -0.2)])
        p.propose(STATE, 8)
        s = p.stats.to_dict()
        assert s["n_prompts"] == 1
        assert s["n_generated"] == 4
        assert s["n_cheats"] == 1
        assert s["n_empty"] == 1
        assert s["n_after_dedupe"] == 1          # two `simp` samples collapse to one candidate
        assert s["cheat_rate"] == 0.25

    def test_rates_are_absent_rather_than_zero_when_nothing_was_generated(self):
        # A reported 0.0 cheat rate on zero generations would read as evidence of cleanliness.
        assert "cheat_rate" not in PolicyStats().to_dict()
        assert "mean_prompt_chars" not in PolicyStats().to_dict()

    def test_config_records_everything_that_must_match_between_arms(self):
        cfg = policy([]).config()
        assert cfg["prompt_limit"] == 6
        assert cfg["top_k"] == 10
        # The checkpoint's own `tokenizer_config.json` ships ChatML; sending REAL-Prover's
        # hard-coded deepseek format to these weights produced token salad on real problems.
        assert cfg["template"] == "qwen_chatml"
        # REAL-Prover's PROVER_MODEL_PARAMS, not plausible-looking defaults.
        assert cfg["sampling"]["temperature"] == 1.5
        assert cfg["sampling"]["top_p"] == 0.9
        assert cfg["sampling"]["max_tokens"] == 256
        assert cfg["retriever"] == "fake"

    def test_name_identifies_the_arm(self):
        assert policy([]).name == "vllm+fake"

    def test_stop_sequences_exclude_a_bare_newline(self):
        assert "\n" not in SamplingConfig().stop
        assert "\nUser:" in SamplingConfig().stop


class TestLengthNormalisation:
    """The score must be the per-token MEAN log-probability, not the cumulative sum.

    REAL-Prover's `generator.py` computes `cumulative_logprob / max(len(token_ids), 1)`. That is not
    incidental: the raw sum falls monotonically with length, so ranking on it makes best-first
    systematically prefer short tactics irrespective of quality. `simp` (2 tokens) would outrank
    `rw [foo, bar] <;> simpa using baz` (~15 tokens) essentially always, and the search would
    explore a different tree — a silent, systematic bias no test of the search itself would catch.

    An earlier version of this policy ranked on the raw sum.
    """

    def test_score_is_the_per_token_mean(self):
        p = policy([Generation("simp", cumulative_logprob=-6.0, n_tokens=3)])
        assert p.propose(STATE, 1) == [("simp", -2.0)]

    def test_a_long_good_tactic_beats_a_short_mediocre_one(self):
        short = Generation("simp", cumulative_logprob=-4.0, n_tokens=2)          # mean -2.0
        long_ = Generation("rw [foo] <;> simpa", cumulative_logprob=-7.5, n_tokens=15)  # mean -0.5
        p = policy([short, long_])
        assert [t for t, _ in p.propose(STATE, 2)] == ["rw [foo] <;> simpa", "simp"]
        # On the raw cumulative sum the order would invert, which is the bug this pins.
        assert short.cumulative_logprob > long_.cumulative_logprob

    def test_zero_tokens_does_not_divide_by_zero(self):
        # Their `max(len(token_ids), 1)` guard. A generation vLLM reports with no tokens.
        assert Generation("x", cumulative_logprob=-3.0, n_tokens=0).mean_logprob == -3.0

    def test_raw_numbers_are_preserved_for_rescoring(self):
        g = Generation("simp", cumulative_logprob=-6.0, n_tokens=3)
        assert (g.cumulative_logprob, g.n_tokens) == (-6.0, 3)


class TestInformalNames:
    """`FrenzyMath/mathlib_informal_v4.16.0` matches our Mathlib pin exactly.

    Applied in the prompt path only: the retrievers encode formal statements, `corpus_id` hashes
    names, and the dense index stores its own copy of the records — so joining glosses here changes
    no embedding and needs no index rebuild.
    """

    def test_gloss_reaches_the_prompt(self):
        prem = [Premise("mul_comm", "∀ a b, a * b = b * a")]
        p = policy([gen("simp", -1.0)], premises=prem,
                   informal_names={"mul_comm": "multiplication is commutative"})
        p.propose(STATE, 4)
        assert "Informal name: multiplication is commutative" in p.generator.prompts[0]

    def test_a_premise_without_a_gloss_keeps_the_field_empty(self):
        # Never fabricate: a humanised declaration name is invented content in a field the model was
        # trained to read.
        prem = [Premise("obscure_lemma", "stmt")]
        p = policy([gen("simp", -1.0)], premises=prem, informal_names={"mul_comm": "x"})
        p.propose(STATE, 4)
        assert "Informal name: \n" in p.generator.prompts[0]

    def test_no_mapping_leaves_every_gloss_empty(self):
        prem = [Premise("mul_comm", "stmt")]
        p = policy([gen("simp", -1.0)], premises=prem)
        assert "Informal name: \n" in (p.propose(STATE, 4), p.generator.prompts[0])[1]

    def test_retrieval_is_untouched_by_the_mapping(self):
        # The query and k must not depend on whether glosses are present, or the arms would
        # differ in retrieval as well as in prompt content.
        prem = [Premise("mul_comm", "stmt")]
        a = policy([gen("simp", -1.0)], premises=prem)
        b = policy([gen("simp", -1.0)], premises=prem, informal_names={"mul_comm": "x"})
        a.propose(STATE, 4)
        b.propose(STATE, 4)
        assert a.retriever.queries == b.retriever.queries
        assert a.retriever.ks == b.retriever.ks

    def test_mapping_size_is_recorded_in_the_config(self):
        p = policy([], informal_names={"a": "x", "b": "y"})
        assert p.config()["informal_names"] == 2


class TestEchoStripping:
    def test_can_be_disabled_for_a_byte_faithful_reproduction(self):
        # REAL-Prover's generator only `.strip()`s; stripping the echo is our arm-neutral addition.
        p = policy([gen("Assistant: simp", -1.0)], strip_echo=False)
        assert p.propose(STATE, 2) == [("Assistant: simp", -1.0)]
        assert p.stats.n_echo_stripped == 0

    def test_enabled_by_default_and_counted(self):
        p = policy([gen("Assistant: simp", -1.0)])
        assert p.propose(STATE, 2) == [("simp", -1.0)]
        assert p.stats.n_echo_stripped == 1

    def test_flag_is_recorded_in_the_config(self):
        assert policy([]).config()["strip_echo"] is True


class TestCorruptedDecodes:
    """U+FFFD in a generation, observed at 1 in 16 samples in a real preflight.

    The sample was `'\ufffdexact CommGroup.to_commMonoid'` — a byte sequence the tokenizer could not
    decode. Lean cannot parse U+FFFD, so such a tactic is guaranteed to fail elaboration, and
    letting it through spends one of the expansion's Lean calls to establish that. At 1 in 16 that
    is 6% of the budget.
    """

    def test_a_corrupted_generation_is_rejected(self):
        p = policy([gen(f"{REPLACEMENT_CHAR}exact CommGroup.to_commMonoid", -1.0)])
        assert p.propose(STATE, 4) == []

    def test_it_is_counted_not_silently_dropped(self):
        """If the rate ever differs between arms, that is a confound rather than a curiosity."""
        p = policy([gen(f"{REPLACEMENT_CHAR}simp", -1.0), gen("ring", -2.0)])
        assert p.propose(STATE, 4) == [("ring", -2.0)]
        assert p.stats.n_undecodable == 1

    def test_it_is_reported_in_the_manifest(self):
        p = policy([gen(f"{REPLACEMENT_CHAR}simp", -1.0)])
        p.propose(STATE, 4)
        assert p.stats.to_dict()["n_undecodable"] == 1

    def test_corruption_anywhere_in_the_tactic_is_rejected(self):
        # Not only a leading one: a mangled character mid-tactic is equally unparseable.
        p = policy([gen(f"rw [mul{REPLACEMENT_CHAR}comm]", -1.0)])
        assert p.propose(STATE, 4) == []

    def test_legitimate_unicode_is_not_rejected(self):
        """Lean is full of real non-ASCII, and rejecting it would gut the candidate set."""
        p = policy([gen("exact fun α => rfl", -1.0), gen("simp [Set.mem_setOf_eq]", -2.0)])
        assert [t for t, _ in p.propose(STATE, 4)] == ["exact fun α => rfl",
                                                       "simp [Set.mem_setOf_eq]"]
        assert p.stats.n_undecodable == 0

    def test_a_cheat_is_still_counted_as_a_cheat(self):
        # The new check sits before the admissibility guard; it must not swallow its cases.
        p = policy([gen("sorry", -1.0)])
        assert p.propose(STATE, 4) == []
        assert p.stats.n_cheats == 1
        assert p.stats.n_undecodable == 0


class TestCandidateQualityIsMeasuredNotAssumed:
    """`mean_candidates_per_expansion` read 11.33/16 — "healthy" — on a run producing token salad.

    Measured afterwards on one proof state: the wrong prompt format produced 13 distinct tactics
    where the right one produced 8, because noise does not repeat itself. Candidate *count* measures
    how much the search has to choose between; it says nothing about whether any of it is worth
    choosing. Mean log-probability is what separated them (-0.34 in-distribution vs -2.78 out).
    """

    def test_mean_candidate_logprob_is_reported(self):
        p = policy([gen("simp", -1.0), gen("ring", -3.0)])
        p.propose(STATE, 4)
        assert p.stats.to_dict()["mean_candidate_logprob"] == -2.0

    def test_it_averages_over_the_deduped_candidates_the_search_sees(self):
        # `simp` sampled twice keeps its best score and counts once, exactly as the search sees it.
        p = policy([gen("simp", -1.0), gen("simp", -5.0), gen("ring", -3.0)])
        p.propose(STATE, 4)
        assert p.stats.n_after_dedupe == 2
        assert p.stats.to_dict()["mean_candidate_logprob"] == -2.0

    def test_it_accumulates_across_expansions(self):
        p = policy([gen("simp", -2.0)])
        p.propose(STATE, 4)
        p.propose(STATE, 4)
        assert p.stats.to_dict()["mean_candidate_logprob"] == -2.0

    def test_a_noisy_arm_is_distinguishable_from_a_healthy_one(self):
        """The property the metric exists for: both look identical by candidate count."""
        healthy = policy([gen("exact mul_comm a b", -0.3), gen("rw [mul_comm]", -0.4)])
        noisy = policy([gen("الجديد exp box", -9.4), gen("Cole Nights poking agent", -11.4)])
        healthy.propose(STATE, 4)
        noisy.propose(STATE, 4)

        assert (healthy.stats.to_dict()["mean_candidates_per_expansion"]
                == noisy.stats.to_dict()["mean_candidates_per_expansion"])
        assert healthy.stats.to_dict()["mean_candidate_logprob"] > -1.5
        assert noisy.stats.to_dict()["mean_candidate_logprob"] < -1.5

    def test_the_metric_is_absent_rather_than_dividing_by_zero(self):
        p = policy([])
        p.propose(STATE, 4)
        assert "mean_candidate_logprob" not in p.stats.to_dict()

    def test_a_nan_score_does_not_poison_the_whole_run(self):
        # `propose` maps NaN to -inf for ordering; letting either into the sum would make every
        # subsequent expansion's manifest figure unreadable.
        p = policy([gen("nan_one", float("nan")), gen("fine", -2.0)])
        p.propose(STATE, 4)
        d = p.stats.to_dict()
        assert d["mean_candidate_logprob"] == -1.0      # -2.0 summed over 2 deduped candidates
        assert math.isfinite(d["mean_candidate_logprob"])
