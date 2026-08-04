"""A tactic policy with **no language model**: a fixed repertoire plus retrieved premises.

## Why this exists

It is the integration harness for everything before the GPU (search, retrieval, prompt seam, Lean,
acceptance predicate all exercised together on real problems, on a laptop). But it is also a
measurement worth reporting in its own right.

The predecessor study's central disappointment was a **statistical tie**: fine-tuned LI and SV
retrievers were indistinguishable downstream, because the generator did not have the resolution to
convert better premises into more proofs. That is the single most likely way this project fails too.

A model-free policy removes the generator entirely. Candidate tactics are `exact <premise>`,
`apply <premise>`, `rw [<premise>]`, `simp [<premise>]` over the retriever's own ranking,
plus a fixed list of closing tactics. So:

- the closing tactics are **identical across every arm** and contribute an identical constant;
- every remaining difference in solve rate is attributable to *which premises were retrieved and in
  what order*, with nothing in between to absorb it.

That makes this the highest-resolution retrieval comparison available anywhere in the project —
obtained without a single GPU hour. If LI beats SV here and nowhere else, that is itself the
finding: the retrieval gain is real but the generator cannot exploit it.

## What is deliberately excluded

`exact?`, `apply?`, `hint`, `aesop?` and friends are **not** in the repertoire. They perform premise
search inside Lean, using Mathlib's own index — they would solve goals regardless of what our
retriever returned, and would silently equalise every arm. (`apply?` is additionally rejected by
`TacticPolicy` in `lean/backend.py`; this list is the second, independent place that judgement is
recorded.)

## On the scores

The search harness ranks nodes by cumulative log-probability, so a policy must supply one. There is
no model here, so the numbers are a **fixed, documented prior**, not a measurement: closers get a
hand-set probability, and a premise at retrieval rank `r` is discounted by `1/(r+1)`. Because the
same table is applied to every arm, it cannot bias the comparison between retrievers — it only sets
the order in which the shared search harness explores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from prooflens_prover.lean.backend import ProofState
from prooflens_prover.retrieval.base import DEFAULT_TOP_K, Premise, Retriever

__all__ = ["RepertoirePolicy", "TacticTemplate", "DEFAULT_CLOSERS", "DEFAULT_TEMPLATES"]


#: Model-free tactics with a hand-set prior. Identical for every arm, so this table can shift the
#: prover's overall strength but never the *difference* between retrievers.
#:
#: The priors need not sum to 1 — only their order within a proposal matters, since the search
#: harness compares candidates against each other.
#:
#: **Structural tactics are ranked first, and that is not arbitrary.** A goal of the form
#: `∀ x, …` or `p → q` admits *no* closing tactic until its binders are introduced, and such goals
#: are extremely common as benchmark root states. An earlier version of this table gave `intro` a
#: prior of 0.04; at `samples_per_step=8` it fell outside the candidate list and every FATE-M
#: problem tried terminated at depth 0 with the frontier empty. This ordering is a fact about Lean,
#: not a fit to a benchmark.
DEFAULT_CLOSERS: dict[str, float] = {
    # -- structural: make progress possible on binder-headed goals --
    "intro x": 0.10,
    "intro h": 0.09,
    "constructor": 0.04,
    "assumption": 0.04,
    "ext": 0.03,
    # -- closers / simplifiers --
    "simp": 0.12,
    "aesop": 0.10,
    "norm_num": 0.08,
    "ring": 0.07,
    "linarith": 0.06,
    "omega": 0.05,
    "simp_all": 0.05,
    "rfl": 0.05,
    "positivity": 0.03,
    "nlinarith": 0.03,
    "field_simp": 0.03,
    "decide": 0.02,
    "trivial": 0.02,
    "tauto": 0.02,
}


@dataclass(frozen=True)
class TacticTemplate:
    """A way of using one retrieved premise in a tactic.

    `prior` is the template's share of probability mass; the premise's retrieval rank supplies the
    rest of the score.
    """

    pattern: str          # `{p}` is substituted with the premise's fully-qualified name
    prior: float

    def render(self, premise_name: str) -> str:
        return self.pattern.format(p=premise_name)


#: The premise-consuming tactics. `rw [<-` is included because roughly half of rewrite uses in
#: Mathlib proofs go right-to-left, and omitting it would make the retriever look worse than it is
#: on exactly the equational lemmas it is best at finding.
DEFAULT_TEMPLATES: tuple[TacticTemplate, ...] = (
    TacticTemplate("exact {p}", 0.30),
    TacticTemplate("apply {p}", 0.25),
    TacticTemplate("rw [{p}]", 0.20),
    TacticTemplate("rw [← {p}]", 0.10),
    TacticTemplate("simp [{p}]", 0.15),
)


@dataclass
class RepertoirePolicy:
    """A `TacticPolicy` that combines a retriever with a fixed tactic repertoire."""

    retriever: Retriever
    top_k: int = DEFAULT_TOP_K
    closers: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CLOSERS))
    templates: tuple[TacticTemplate, ...] = DEFAULT_TEMPLATES
    #: Set False for the `none` arm's ablation partner: repertoire only, no premise tactics. The
    #: `none` retriever already returns nothing, so this is mainly a guard for accidental misuse.
    use_premises: bool = True
    #: Candidate slots reserved for the highest-prior closers, before any premise tactic competes.
    #:
    #: **This is a correctness fix, not a tuning knob.** Without it, retrieval silently made the
    #: prover worse: `top_k=10` premises x 5 templates = 50 premise tactics, every one of which
    #: outscores `intro x` (log 0.30 - log 1 = -1.20 against log 0.10 = -2.30). At
    #: `samples_per_step=16` they consumed nearly every slot, so the `bm25` arm never tried the
    #: tactics that actually close goals. Measured on the first 30 miniF2F-test problems:
    #: `none` 10/30, `bm25` 6/30 — retrieval "hurting" for a reason with nothing to do with
    #: retrieval quality.
    #:
    #: Reserving slots keeps every arm at the same number of Lean calls per expansion (so the arms
    #: stay compute-matched) while guaranteeing the shared repertoire is always tried. It also
    #: mirrors the LLM setting this stands in for: putting premises in a prompt informs the model's
    #: tactic distribution, it does not delete the rest of it.
    min_closers: int = 8

    @property
    def name(self) -> str:
        return f"repertoire+{getattr(self.retriever, 'name', 'unknown')}"

    def propose(
        self, state: ProofState, n: int, context: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """Return up to `n` `(tactic, logprob)` candidates, best first."""
        closer_scores = {t: math.log(p) for t, p in self.closers.items()}

        premise_scores: dict[str, float] = {}
        if self.use_premises and self.top_k > 0:
            premises: list[Premise] = self.retriever.retrieve(state.pp, k=self.top_k)
            for rank, premise in enumerate(premises):
                # Rank discount, not retrieval score: BM25 scores and cosine similarities are on
                # incomparable scales, and using them directly would make the arms' candidate
                # orderings depend on a scale artefact rather than on the ranking itself.
                rank_logp = -math.log(rank + 1.0)
                for template in self.templates:
                    tactic = template.render(premise.formal_name)
                    score = math.log(template.prior) + rank_logp
                    # A tactic reachable two ways keeps its best score.
                    if tactic not in premise_scores or score > premise_scores[tactic]:
                        premise_scores[tactic] = score

        # Deterministic order everywhere: score descending, then tactic text, so two runs with the
        # same retrieval output explore in exactly the same order.
        def rank(items: dict[str, float]) -> list[tuple[str, float]]:
            return sorted(items.items(), key=lambda kv: (-kv[1], kv[0]))

        ranked_closers = rank(closer_scores)
        reserved = ranked_closers[: max(0, min(self.min_closers, n))]

        # Everything not already reserved competes for the remaining slots. When there are no
        # premises this is just the rest of the repertoire, so the `none` arm still fills its full
        # budget rather than being handicapped by the reservation.
        contested = rank({**dict(ranked_closers[len(reserved):]), **premise_scores})
        selected = reserved + contested[: max(0, n - len(reserved))]
        return rank(dict(selected))
