"""Combine two retrievers into one, so a single arm can reach what only the pair reaches.

## Why this exists

Tier 1 measured an exact tie between single-vector and late interaction — 74 of 327 each — while
their **union reached 88**. Fourteen problems in each direction, and a fusion ceiling larger than
retrieval's entire measured effect (+14 against +11). The discordance profile
(`scripts/discordance_profile.py`) says *why* the two are complementary, and the two halves point at
different fusion strategies:

* **Late interaction supplies recall.** Where LI wins, single-vector most often died with
  `no_candidates` — it had nothing left to propose. LI surfaced a premise SV never offered.
* **Single-vector supplies ranking.** Where SV wins, LI had **always** burned its entire expansion
  budget (median 64 of 64, both benchmarks). LI had plenty to try and tried the wrong things.

So the target is a retriever that keeps LI's reach without inheriting its tendency to fill the
frontier with plausible-but-wrong premises.

## Two strategies, because the prompt is narrow

Only **six** premises reach the model (`PROMPT_PREMISE_LIMIT`), and that changes which fusion rule
is appropriate. Both are implemented and selectable, because the argument for each is real and the
question is empirical.

**`rrf` — reciprocal rank fusion.** `score(p) = Σ_r 1/(K + rank_r(p))`, the standard rule, K = 60.
Consensus dominates: a premise both retrievers rank highly beats one that only a single retriever
found. That is the right prior in general — and it is exactly the wrong prior for the mechanism
above, since a premise *only LI finds* is precisely the case LI earns its keep on. With K = 60 a
premise at rank 1 from one retriever (0.0164) loses to one at rank 5 from both (0.0308).

**`interleave` — round-robin.** Take each retriever's first choice, then each retriever's second,
and so on, skipping duplicates. With six prompt slots this guarantees **three from each retriever**,
so whatever only LI found still reaches the model. It cannot express consensus at all, which is its
weakness.

Neither is obviously right, so the pilot decides. Defaulting to `rrf` because it is the published,
expected rule — a result that needs a non-standard fusion to work should have to say so.

## Cost

Both sub-retrievers run on every query, so latency is their sum: roughly 970 ms/query when late
interaction is one of them (measured 930.2 + 39.4). Requesting more than `k` from each costs
nothing extra for late interaction — its cost is the MaxSim rerank over 50,000 first-stage
candidates, which does not depend on `k`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from prooflens_prover.retrieval.base import DEFAULT_TOP_K, Premise, RetrievalStats, Retriever

#: The constant in reciprocal rank fusion. 60 is the value from the original TREC work and the one
#: every implementation uses; changing it silently would make the number incomparable to the
#: literature, so it is a named default rather than a magic number in the expression.
RRF_K = 60

#: How many premises to request from each sub-retriever before fusing. Fusion needs ranks below the
#: cut to work with: at `fetch_k == k` the two rankings would be truncated before they could
#: disagree, and fusion would degenerate to whichever retriever is listed first. Costs nothing for
#: late interaction, whose expense is the first-stage rerank rather than `k`.
DEFAULT_FETCH_K = 32

MODES = ("rrf", "interleave")


@dataclass
class FusionRetriever:
    """Merge several retrievers' rankings into one. A drop-in `Retriever`.

    `stats` counts one query per `retrieve` call and times the whole fused operation, so the run
    manifest reports what the arm actually cost. Sub-retrievers keep their own `stats` objects for
    diagnostics; those are **not** summed into this one, which would double-count `n_queries` and
    make the arm look twice as busy as it is.
    """

    retrievers: tuple[Retriever, ...]
    mode: str = "rrf"
    fetch_k: int = DEFAULT_FETCH_K
    rrf_k: int = RRF_K
    stats: RetrievalStats = field(default_factory=RetrievalStats)

    name = "fusion"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown fusion mode {self.mode!r}; expected one of {MODES}")
        if len(self.retrievers) < 2:
            raise ValueError(
                f"fusion needs at least two retrievers, got {len(self.retrievers)}. One retriever "
                "fused with nothing is that retriever, and recording it as an arm called 'fusion' "
                "would put a mislabelled row in the results table."
            )
        names = [getattr(r, "name", "?") for r in self.retrievers]
        if len(set(names)) != len(names):
            raise ValueError(
                f"fusing two retrievers of the same kind {names}: rank agreement between a "
                "retriever and itself is total, so fusion would be a no-op reported as a result."
            )

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(getattr(r, "name", "?") for r in self.retrievers)

    def component_stats(self) -> dict[str, dict]:
        """Each sub-retriever's own counters, for the run manifest.

        The fused `stats` above times the whole operation and cannot say which half was slow. That
        distinction is the single unmeasured quantity in this arm: late interaction is timed at
        936–1,071 ms/query on a GPU, but single-vector is deliberately placed on the **CPU** here,
        so the two indices need not share a card with a 7B model — and no `sv` figure in this
        project was ever taken off a GPU. Without this the only way to recover it is to subtract one
        mean from another and hope the two runs are comparable.

        It also settles a question subtraction cannot: the two servers contend for the same eight
        cores, so single-vector's CPU forward pass and late interaction's 50,000-row numpy rerank
        can each slow the other. Only per-component timing during a fused run shows that.
        """
        return {
            getattr(r, "name", f"component_{i}"): r.stats.to_dict()
            for i, r in enumerate(self.retrievers)
            if getattr(r, "stats", None) is not None
        }

    def config(self) -> dict[str, object]:
        """Recorded in the run manifest — fusion is not reproducible without these."""
        return {
            "mode": self.mode,
            "components": list(self.component_names),
            "fetch_k": self.fetch_k,
            "rrf_k": self.rrf_k if self.mode == "rrf" else None,
        }

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[Premise]:
        t0 = time.perf_counter()
        depth = max(self.fetch_k, k)
        rankings = [r.retrieve(query, k=depth) for r in self.retrievers]
        fused = (
            _interleave(rankings) if self.mode == "interleave"
            else _reciprocal_rank_fusion(rankings, self.rrf_k)
        )[:k]
        self.stats.record(time.perf_counter() - t0, len(fused))
        return fused


def _reciprocal_rank_fusion(rankings: list[list[Premise]], rrf_k: int) -> list[Premise]:
    """`Σ_r 1/(rrf_k + rank_r)`, best first. Ranks are 1-based, as the rule is defined."""
    scores: dict[str, float] = {}
    first_seen: dict[str, Premise] = {}
    for ranking in rankings:
        for rank, premise in enumerate(ranking, start=1):
            key = premise.formal_name
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            first_seen.setdefault(key, premise)
    # Ties broken on the name, not on dict order: two premises found at the same rank by disjoint
    # retrievers score identically, and an insertion-ordered tiebreak would make the arm's output
    # depend on which sub-retriever was constructed first.
    order = sorted(scores, key=lambda name: (-scores[name], name))
    return [_rescored(first_seen[name], scores[name]) for name in order]


def _interleave(rankings: list[list[Premise]]) -> list[Premise]:
    """Round-robin: every retriever's first choice, then every second choice, skipping repeats.

    Guarantees each retriever reaches the prompt. With six slots and two retrievers that is three
    each, which is the point — a premise only one retriever found is exactly what the union result
    says is worth having.
    """
    out: list[Premise] = []
    seen: set[str] = set()
    for rank in range(max((len(r) for r in rankings), default=0)):
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            premise = ranking[rank]
            if premise.formal_name in seen:
                continue
            seen.add(premise.formal_name)
            out.append(premise)
    return out


def _rescored(premise: Premise, score: float) -> Premise:
    """`Premise` is frozen; the fused score replaces the sub-retriever's incomparable one.

    An SV cosine and an LI MaxSim sum are on different scales, so carrying either through would put
    a number in the record that means nothing. The fused score is at least well defined.
    """
    return Premise(
        formal_name=premise.formal_name,
        formal_statement=premise.formal_statement,
        informal_name=premise.informal_name,
        score=score,
    )
