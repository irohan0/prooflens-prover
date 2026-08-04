"""Retriever interface — **the only thing that varies between experimental arms.**

Everything else in this project (search harness, Lean backend, prompt format, budget, acceptance
predicate) is held byte-identically fixed. The whole study reduces to: swap the object behind this
Protocol, change nothing else, measure the difference.

## The `Premise` shape is dictated by REAL-Prover, not chosen by us

The frozen-prover comparison substitutes our retriever for LeanSearch-PS inside their unmodified
prover. Their `LeanSearch.get_related_theorem` returns a list of dicts with exactly three keys —
`Formal name`, `Informal name`, `Formal statement` (see `manager/thirdparty/lean_search.py` and
`manager/manage/prompt_manage.py::build_theorems_str`). Matching that shape exactly is what makes
the swap a one-line change on their side and keeps our numbers comparable to their published ones.

`Informal name` has no analogue in a Mathlib premise corpus. It is emitted as an empty string, and
**identically for every arm**, so it cannot become a hidden difference between them. Whatever is
chosen there must stay constant across arms — a retriever that also supplied informal names would
be varying two things at once, and the experiment could not attribute the effect.

## Two numbers from their code that constrain everything downstream

- `NUM_QUERYS = 10` — how many premises are *requested*.
- `related_theorems[:6]` — how many actually reach the model.

So REAL-Prover's published +12.0 pt FATE-M retrieval gain comes from **six premises**. `top_k`
defaults to 10 (request) and `PROMPT_PREMISE_LIMIT` records the 6 (use), because reporting the
number offered rather than the number used would overstate retrieval depth — the exact error the
predecessor project caught in its own reporting, where "100 premises" was really ~25.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

#: Premises requested per query (REAL-Prover `conf/config.py::NUM_QUERYS`).
DEFAULT_TOP_K = 10

#: Premises that actually reach the model (REAL-Prover `build_theorems_str`: `[:6]`).
#: The comparability point for any top-k ablation.
PROMPT_PREMISE_LIMIT = 6


@dataclass(frozen=True)
class Premise:
    """One retrieved Mathlib premise, in the shape REAL-Prover's prompt builder expects."""

    formal_name: str
    formal_statement: str
    informal_name: str = ""
    score: float = 0.0

    def to_realprover_dict(self) -> dict[str, str]:
        """The exact dict shape `build_theorems_str` consumes. Key names and capitalisation are
        theirs; do not "tidy" them or the frozen-prover swap silently produces empty fields."""
        return {
            "Formal name": self.formal_name,
            "Informal name": self.informal_name,
            "Formal statement": self.formal_statement,
        }


def format_premises(premises: list[Premise], limit: int = PROMPT_PREMISE_LIMIT) -> str:
    """Render premises exactly as REAL-Prover's `build_theorems_str` does.

    Replicated verbatim rather than approximated: the predecessor project documented three separate
    ways a "reasonable" reimplementation of a prompt formatter silently invalidates a study
    (`prooflens/src/prooflens/generation/format.py`). The details that matter here are the `ID:`
    counter starting at 0, the field order, and the blank line between premises.
    """
    return "\n\n".join(
        f"ID:{i}\n"
        f"Formal name: {p.formal_name}\n"
        f"Informal name: {p.informal_name}\n"
        f"Formal statement: {p.formal_statement}"
        for i, p in enumerate(premises[:limit])
    )


@runtime_checkable
class Retriever(Protocol):
    """Ranks Mathlib premises for a proof state. One implementation per experimental arm."""

    name: str

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[Premise]:
        """Return the top-`k` premises for `query` (a pretty-printed proof state), best first."""
        ...


class NullRetriever:
    """The no-retrieval floor. Returns nothing, always.

    Not a degenerate case to be skipped — it is the baseline every other arm's gain is measured
    against, and it is how REAL-Prover's own 44.7 (no retrieval) vs 56.7 (retrieval) pair was
    produced. Cheap to run and it anchors the whole results table.
    """

    name = "none"

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[Premise]:  # noqa: ARG002
        return []


@dataclass
class RetrievalStats:
    """Per-run retrieval accounting, for the cost/latency table.

    The predecessor project left retrieval latency explicitly unmeasured (its open item #8) while
    noting late interaction costs ~69x the index footprint of single-vector. Since this project's
    argument is that multi-vector retrieval is worth adopting, "worth it" has to include what it
    costs — so latency is recorded from the start rather than retrofitted.
    """

    n_queries: int = 0
    total_latency_s: float = 0.0
    n_premises_returned: int = 0

    def record(self, latency_s: float, n_returned: int) -> None:
        self.n_queries += 1
        self.total_latency_s += latency_s
        self.n_premises_returned += n_returned

    def to_dict(self) -> dict[str, Any]:
        n = max(self.n_queries, 1)
        return {
            "n_queries": self.n_queries,
            "total_latency_s": round(self.total_latency_s, 3),
            "mean_latency_ms": round(1000 * self.total_latency_s / n, 2),
            "mean_premises_returned": round(self.n_premises_returned / n, 2),
        }
