"""Best-first proof search — the fixed harness every retrieval arm is compared through.

## Why this is the load-bearing component

The dissertation's claim is that swapping *only* the retriever changes end-to-end proof success.
That is only true if everything else is held byte-identically fixed. This module is "everything
else": the search order, the budget, the deduplication, the stopping rules. Any nondeterminism or
arm-dependent behaviour here silently becomes part of the measured effect.

So: the search takes a `TacticPolicy` and a `LeanBackend` and knows nothing about retrieval,
models, or benchmarks. An arm is a policy; the harness never varies.

## The algorithm

Best-first over the tactic tree, scoring nodes with REAL-Prover's function so our numbers sit on
the same footing as theirs (arXiv:2505.20613 §4):

    score(node) = (Σ_t log p(a_t | s_t)) / depth^alpha,   alpha = 0.5

Cumulative log-probability rewards likely tactic sequences; dividing by `depth^alpha` stops the
search collapsing into whichever branch happens to be shallowest, since log-probabilities are
negative and simply summing them penalises depth without bound.

## Budgets, and why all four exist

`max_expansions` bounds model calls (the dominant cost), `max_depth` bounds any single line of
reasoning, `wall_clock_s` bounds the real resource a cluster allocation is measured in, and
`tactic_timeout` bounds one pathological elaboration. A run that exhausts a budget is a legitimate
`exhausted` result, not an error — and *which* budget ran out is recorded, because "we ran out of
time" and "we ran out of ideas" are different findings about a retriever.

## State deduplication

Tactics frequently produce a goal state that has already been seen (`simp` then `simp` again, or
two different routes converging). Re-expanding it wastes the budget and can loop forever. States
are keyed by their pretty-printed goals, which is the same string the model and the retriever see,
so two states are treated as identical exactly when they are indistinguishable to the rest of the
system.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from prooflens_prover.lean.backend import LeanBackend, Outcome, ProofState, TacticResult
from prooflens_prover.utils.logging import get_logger

log = get_logger("prover.search")


class SearchStatus(StrEnum):
    PROVED = "proved"
    EXHAUSTED = "exhausted"          # ran out of budget (see `limit_hit`)
    NO_CANDIDATES = "no_candidates"  # frontier emptied: every branch dead
    ERROR = "error"                  # harness failure, not a proof failure


@runtime_checkable
class TacticPolicy(Protocol):
    """Proposes candidate tactics for a proof state. **This is the only thing an arm varies.**

    Returns `[(tactic, logprob)]`, best first. The retrieval arm lives entirely inside an
    implementation of this Protocol: it retrieves premises for `state`, formats them into a
    prompt, calls the model, and returns candidates. The search harness stays identical.
    """

    def propose(self, state: ProofState, n: int, context: dict[str, Any] | None = None
                ) -> list[tuple[str, float]]:
        ...


@dataclass(frozen=True)
class SearchConfig:
    """Search budget. Identical across arms; recorded in every run manifest.

    Defaults match REAL-Prover's shipped `conf/config.py` so the frozen-prover comparison differs
    from theirs only in the retriever:

    | ours | theirs | value |
    |---|---|--:|
    | `samples_per_step` | `NUM_SAMPLES` | 16 |
    | `max_depth` | `MAX_DEPTH` | 32 |
    | `max_expansions` | `MAX_NODES` | 64 |
    | `length_penalty` | `alpha` | 0.5 |

    Their commented-out "large" variant (64 / 128 / 1024) is what the paper's Pass@64x64 used, and
    is ~4x the cost. State which of the two any reported run matches — `SearchConfig.large()`
    builds the large one.

    Caveat on `max_expansions`: theirs counts nodes *inserted* into the frontier, ours counts nodes
    *expanded*. Ours is the stricter reading (it bounds model calls directly), so at equal numbers
    our budget is the smaller one. Not silently equivalent; recorded so the write-up can say so.
    """

    max_expansions: int = 64         # nodes expanded = model calls; the dominant cost
    samples_per_step: int = 16       # candidate tactics per expansion
    max_depth: int = 32
    wall_clock_s: float = 600.0
    tactic_timeout: float = 60.0
    length_penalty: float = 0.5      # REAL-Prover's alpha
    dedupe_states: bool = True

    @classmethod
    def large(cls, **overrides: Any) -> SearchConfig:
        """REAL-Prover's commented-out large budget — what their published Pass@64x64 used."""
        return cls(**{
            "max_expansions": 1024, "samples_per_step": 64, "max_depth": 128, **overrides
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_expansions": self.max_expansions,
            "samples_per_step": self.samples_per_step,
            "max_depth": self.max_depth,
            "wall_clock_s": self.wall_clock_s,
            "tactic_timeout": self.tactic_timeout,
            "length_penalty": self.length_penalty,
            "dedupe_states": self.dedupe_states,
        }


@dataclass
class Node:
    """A search-tree node. `parent`/`tactic` walk a successful leaf back to a full proof."""

    state: ProofState
    cum_logprob: float
    depth: int
    parent: Node | None = None
    tactic: str | None = None

    def score(self, alpha: float) -> float:
        """Higher is better. Depth 0 (the root) is never scored, so `max(depth,1)` is safe."""
        return self.cum_logprob / (max(self.depth, 1) ** alpha)

    def proof_steps(self) -> list[str]:
        """Tactics from the root to this node, in order."""
        steps: list[str] = []
        node: Node | None = self
        while node is not None and node.tactic is not None:
            steps.append(node.tactic)
            node = node.parent
        return list(reversed(steps))


@dataclass
class SearchResult:
    """Outcome of one proof attempt. Serialised per attempt so any pass@k is recomputable later."""

    status: SearchStatus
    proof: list[str] | None = None
    n_expansions: int = 0
    n_tactics_tried: int = 0
    n_lean_calls: int = 0
    elapsed_s: float = 0.0
    limit_hit: str | None = None
    error: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def proved(self) -> bool:
        return self.status is SearchStatus.PROVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "proved": self.proved,
            "proof": self.proof,
            "n_expansions": self.n_expansions,
            "n_tactics_tried": self.n_tactics_tried,
            "n_lean_calls": self.n_lean_calls,
            "elapsed_s": round(self.elapsed_s, 3),
            "limit_hit": self.limit_hit,
            "error": self.error,
            "trace": self.trace,
        }


def best_first_search(
    backend: LeanBackend,
    policy: TacticPolicy,
    statement: str,
    config: SearchConfig | None = None,
    header: str | None = None,
    context: dict[str, Any] | None = None,
    record_trace: bool = True,
) -> SearchResult:
    """Search for a proof of `statement`. Never raises for an ordinary failure.

    A harness-level exception is caught and returned as `SearchStatus.ERROR` so one bad problem
    cannot abort a 672-problem benchmark run — the failure is recorded and the run continues.
    """
    cfg = config or SearchConfig()
    t0 = time.perf_counter()
    trace: list[dict[str, Any]] = []
    n_expansions = n_tactics = n_lean = 0

    def elapsed() -> float:
        return time.perf_counter() - t0

    try:
        root_state = backend.start_theorem(statement, header=header)
    except Exception as e:  # noqa: BLE001 — a statement that will not elaborate is data, not a crash
        return SearchResult(
            status=SearchStatus.ERROR, elapsed_s=elapsed(),
            error=f"start_theorem failed: {type(e).__name__}: {e}",
        )

    if root_state.is_solved:
        # The statement elaborated with no goals. Treat as a harness error rather than a win:
        # scoring this as PROVED would silently inflate results on malformed inputs.
        return SearchResult(
            status=SearchStatus.ERROR, elapsed_s=elapsed(),
            error="root state has no goals; statement may already be closed",
        )

    root = Node(state=root_state, cum_logprob=0.0, depth=0)
    # heapq is a MIN-heap and we want max score, so push -score. `counter` breaks ties in a
    # deterministic, insertion-ordered way — without it, heapq would compare Node objects and
    # raise, and any other tiebreak would make the search order depend on object identity.
    counter = itertools.count()
    root_key = (-root.score(cfg.length_penalty), next(counter), root)
    frontier: list[tuple[float, int, Node]] = [root_key]
    seen: set[str] = {root_state.pp} if cfg.dedupe_states else set()
    limit_hit: str | None = None

    while frontier:
        if elapsed() > cfg.wall_clock_s:
            limit_hit = "wall_clock_s"
            break
        if n_expansions >= cfg.max_expansions:
            limit_hit = "max_expansions"
            break

        _, _, node = heapq.heappop(frontier)
        if node.depth >= cfg.max_depth:
            continue

        try:
            candidates = policy.propose(node.state, cfg.samples_per_step, context)
        except Exception as e:  # noqa: BLE001 — a policy failure ends this attempt, not the run
            return SearchResult(
                status=SearchStatus.ERROR, n_expansions=n_expansions, n_tactics_tried=n_tactics,
                n_lean_calls=n_lean, elapsed_s=elapsed(), trace=trace,
                error=f"policy.propose failed: {type(e).__name__}: {e}",
            )
        n_expansions += 1

        for tactic, logprob in candidates:
            if elapsed() > cfg.wall_clock_s:
                limit_hit = "wall_clock_s"
                break

            n_tactics += 1
            result: TacticResult = backend.run_tactic(
                node.state, tactic, timeout=cfg.tactic_timeout
            )
            if result.outcome is not Outcome.REJECTED:
                n_lean += 1    # REJECTED never reaches Lean, so it is not a Lean call

            if record_trace:
                trace.append({
                    "depth": node.depth, "tactic": tactic, "logprob": logprob,
                    "outcome": result.outcome.value, "elapsed_s": round(result.elapsed_s, 4),
                    "error": (result.error or "")[:200] or None,
                })

            if result.outcome is Outcome.PROVED:
                proof = Node(node.state, 0.0, node.depth + 1, node, tactic).proof_steps()
                log.info("proved in %d steps (%d expansions, %.1fs)",
                         len(proof), n_expansions, elapsed())
                return SearchResult(
                    status=SearchStatus.PROVED, proof=proof, n_expansions=n_expansions,
                    n_tactics_tried=n_tactics, n_lean_calls=n_lean, elapsed_s=elapsed(),
                    trace=trace,
                )

            if result.outcome is Outcome.CRASH:
                # The REPL died. Every state handle it issued is now invalid, so the tree cannot
                # be continued; report honestly rather than silently returning a partial search
                # that looks like "no proof exists".
                return SearchResult(
                    status=SearchStatus.ERROR, n_expansions=n_expansions,
                    n_tactics_tried=n_tactics, n_lean_calls=n_lean, elapsed_s=elapsed(),
                    trace=trace, error=f"Lean backend crashed: {result.error}",
                )

            if result.outcome is Outcome.PROGRESS and result.state is not None:
                key = result.state.pp
                if cfg.dedupe_states and key in seen:
                    continue
                if cfg.dedupe_states:
                    seen.add(key)
                child = Node(
                    state=result.state,
                    cum_logprob=node.cum_logprob + logprob,
                    depth=node.depth + 1,
                    parent=node,
                    tactic=tactic,
                )
                heapq.heappush(
                    frontier, (-child.score(cfg.length_penalty), next(counter), child)
                )

        if limit_hit:
            break

    status = SearchStatus.EXHAUSTED if (limit_hit or frontier) else SearchStatus.NO_CANDIDATES
    return SearchResult(
        status=status, n_expansions=n_expansions, n_tactics_tried=n_tactics,
        n_lean_calls=n_lean, elapsed_s=elapsed(), limit_hit=limit_hit, trace=trace,
    )
