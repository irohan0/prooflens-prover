"""Premise-free sample mixing — sampling part of each expansion without the premise block.

## Why the feature exists

Under an LLM, retrieval *displaces* proofs. Tier 1 measured late interaction gaining 23 problems and
losing **12 the no-retrieval control had already solved**, against single-vector's 18 and 7 — the
same net +11, and the wider split is what cost late interaction its significance. Each of those 12
is a state where the model knew a tactic unaided and the premise block talked it out of it.

Splitting an expansion between the retrieval prompt and the control's prompt makes that tactic
reachable again without giving up the premises that win elsewhere.

## What these tests protect

The first and most important property is that **0.0 changes nothing**. The published Tier 1 numbers
must stay reproducible from `results/exported/`, so the default path has to issue one prompt, one
generate call, and one increment of every counter — exactly as before this feature existed.

Hermetic: no model, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.lean.backend import ProofState  # noqa: E402
from prooflens_prover.prover.vllm_policy import (  # noqa: E402
    Generation,
    SamplingConfig,
    VLLMPolicy,
)
from prooflens_prover.retrieval.base import Premise  # noqa: E402

STATE = ProofState(pid=1, goals=("a b : G\n⊢ a * b = b * a",))
PREMISES = [Premise(formal_name="mul_comm", formal_statement="∀ a b, a * b = b * a")]


class ScriptedGenerator:
    """Returns a different script per call, so the two halves can be told apart."""

    def __init__(self, scripts: list[list[Generation]]):
        self.scripts = scripts
        self.prompts: list[str] = []
        self.ns: list[int] = []

    def generate(self, prompt: str, n: int, sampling: SamplingConfig) -> list[Generation]:  # noqa: ARG002
        self.prompts.append(prompt)
        self.ns.append(n)
        return list(self.scripts[min(len(self.prompts) - 1, len(self.scripts) - 1)])


class FakeRetriever:
    name = "fake"

    def __init__(self, premises):
        self.premises = premises

    def retrieve(self, query: str, k: int = 10) -> list[Premise]:  # noqa: ARG002
        return list(self.premises)


def gen(text: str, lp: float = -1.0, n_tokens: int = 1) -> Generation:
    return Generation(text=text, cumulative_logprob=lp, n_tokens=n_tokens)


def policy(scripts, premises=PREMISES, **kw) -> VLLMPolicy:
    return VLLMPolicy(
        generator=ScriptedGenerator(scripts), retriever=FakeRetriever(premises), **kw
    )


# --- the default must be inert -----------------------------------------------------------------

def test_the_default_issues_exactly_one_prompt_and_one_generate_call():
    # THE property that keeps the published Tier 1 numbers reproducible. If this ever fails, every
    # figure in the dissertation was produced by different code than the repository contains.
    p = policy([[gen("simp")]])
    p.propose(STATE, 16)
    assert p.generator.ns == [16]
    assert len(p.generator.prompts) == 1
    assert p.stats.n_prompts == 1
    assert p.stats.n_premise_free_prompts == 0


def test_the_default_reports_no_mixing_fields_at_all():
    # Absent, not zero: a manifest key that appears on every historical run would make an old run
    # look as though it had been re-recorded under the new code.
    p = policy([[gen("simp")]])
    p.propose(STATE, 16)
    assert "n_premise_free_prompts" not in p.stats.to_dict()


def test_the_default_mean_prompt_chars_is_unchanged():
    p = policy([[gen("simp")]])
    p.propose(STATE, 16)
    d = p.stats.to_dict()
    assert d["mean_prompt_chars"] == round(p.stats.total_prompt_chars / 1, 1)


# --- splitting ---------------------------------------------------------------------------------

def test_a_quarter_fraction_splits_sixteen_samples_twelve_four():
    p = policy([[gen("simp")], [gen("aesop")]], premise_free_fraction=0.25)
    p.propose(STATE, 16)
    assert p.generator.ns == [12, 4]


def test_the_second_prompt_omits_the_premise_block():
    p = policy([[gen("simp")], [gen("aesop")]], premise_free_fraction=0.25)
    p.propose(STATE, 16)
    with_premises, without = p.generator.prompts
    assert "mul_comm" in with_premises
    assert "mul_comm" not in without


def test_both_halves_reach_the_candidate_list():
    p = policy([[gen("simp")], [gen("aesop")]], premise_free_fraction=0.25)
    tactics = [t for t, _ in p.propose(STATE, 16)]
    assert set(tactics) == {"simp", "aesop"}


def test_candidates_only_the_premise_free_half_found_are_counted():
    # The number that says whether mixing did anything. If it stays near zero the fraction is
    # spending budget on tactics the premise prompt already proposed.
    p = policy([[gen("simp")], [gen("aesop")]], premise_free_fraction=0.25)
    p.propose(STATE, 16)
    assert p.stats.n_premise_free_candidates == 1


def test_a_tactic_both_halves_propose_is_not_double_counted():
    p = policy([[gen("simp")], [gen("simp")]], premise_free_fraction=0.25)
    p.propose(STATE, 16)
    assert p.stats.n_premise_free_candidates == 0
    assert p.stats.n_after_dedupe == 1


def test_a_tactic_proposed_by_both_halves_keeps_its_better_score():
    p = policy(
        [[gen("simp", lp=-4.0, n_tokens=2)],      # mean -2.0, from the premise prompt
         [gen("simp", lp=-1.0, n_tokens=2)]],     # mean -0.5, from the premise-free prompt
        premise_free_fraction=0.5,
    )
    assert p.propose(STATE, 16) == [("simp", pytest.approx(-0.5))]


def test_the_health_gate_still_averages_over_expansions_not_prompts():
    # `mean_candidates_per_expansion` is read against --samples-per-step. Counting the extra prompt
    # in its denominator would halve it and make a mixed run look degenerate next to an unmixed one.
    p = policy([[gen("simp")], [gen("aesop")]], premise_free_fraction=0.25)
    p.propose(STATE, 16)
    assert p.stats.to_dict()["mean_candidates_per_expansion"] == 2.0


# --- degenerate splits fall back to one prompt --------------------------------------------------

@pytest.mark.parametrize("n,fraction", [
    (16, 0.0),      # feature off
    (1, 0.5),       # too few samples to split
    (16, 0.01),     # rounds to zero premise-free samples
    (16, 0.99),     # rounds to zero premise-bearing samples
])
def test_a_degenerate_split_falls_back_to_a_single_prompt(n, fraction):
    # A generate call for zero samples is an error in some backends and silent waste in the rest,
    # and an expansion that quietly used a different sample count than --samples-per-step would
    # break the only health gate this policy has.
    p = policy([[gen("simp")]], premise_free_fraction=fraction)
    p.propose(STATE, n)
    assert p.generator.ns == [n]


def test_the_none_arm_never_splits_because_it_has_no_premises_to_omit():
    # Both prompts would be identical, so the split would halve the effective sample count while
    # reporting the full budget.
    p = policy([[gen("simp")]], premises=[], premise_free_fraction=0.5)
    p.propose(STATE, 16)
    assert p.generator.ns == [16]
    assert p.stats.n_premise_free_prompts == 0


def test_the_fraction_is_recorded_in_the_policy_config():
    assert policy([[gen("simp")]], premise_free_fraction=0.25).config()[
        "premise_free_fraction"] == 0.25


def test_the_format_check_runs_on_the_prompt_the_arm_is_named_for():
    # Not on the premise-free one: a mismatch warning about a prompt the arm only uses for a
    # quarter of its samples would point the reader at the wrong thing.
    seen = []

    class Checking(ScriptedGenerator):
        def check_prompt_format(self, prompt, content):  # noqa: ARG002
            seen.append(prompt)
            return None

    p = VLLMPolicy(
        generator=Checking([[gen("simp")], [gen("aesop")]]),
        retriever=FakeRetriever(PREMISES),
        premise_free_fraction=0.25,
    )
    p.propose(STATE, 16)
    assert len(seen) == 1 and "mul_comm" in seen[0]
