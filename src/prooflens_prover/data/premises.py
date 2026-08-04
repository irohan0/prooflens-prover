"""The Mathlib premise corpus — the one candidate set every retrieval arm ranks over.

Produced by `scripts/extract_premises.sh` from Lean's own elaborated environment. This module only
loads and filters it.

## Why filtering happens here and not during extraction

Extraction is expensive (~an hour) and version-pinned; filtering is cheap and is a research choice
we will want to revisit. So extraction is deliberately inclusive and every judgement about what
counts as a "premise" lives here, where it is a flag with a default rather than a property baked
into a file. The corpus file's SHA-256 plus the filter arguments together identify a candidate set
exactly, and both go into the run manifest.

## The rule that must not be broken

Whatever filter is chosen, **every arm gets the identical candidate set**. A retriever that ranks
over a different corpus is not a different retriever, it is a different experiment; the difference
would show up as a retrieval-quality effect and be attributed to the architecture. `corpus_id()`
exists so a run can assert this rather than assume it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from prooflens_prover.retrieval.base import Premise

__all__ = [
    "PremiseRecord",
    "corpus_id",
    "iter_premise_records",
    "load_premise_corpus",
]

#: Declaration kinds that carry a citable statement. `rec`/`quot` are already dropped during
#: extraction; `opaque` is kept because Mathlib uses it for a handful of real definitions.
DEFAULT_KINDS = frozenset({"theorem", "axiom", "def", "inductive", "ctor", "opaque"})


@dataclass(frozen=True)
class PremiseRecord:
    """One line of the extracted corpus."""

    name: str
    kind: str
    statement: str
    module: str
    is_prop: bool

    @property
    def is_theorem(self) -> bool:
        return self.kind in ("theorem", "axiom")

    def to_premise(self, score: float = 0.0) -> Premise:
        """Convert to the retrieval-facing shape.

        `informal_name` is empty for every record, and identically so for every arm — see the
        rationale in `retrieval/base.py`. It is not a field this corpus can populate.
        """
        return Premise(
            formal_name=self.name,
            formal_statement=self.statement,
            informal_name="",
            score=score,
        )


def iter_premise_records(path: str | Path) -> Iterator[PremiseRecord]:
    """Stream the corpus without holding the raw JSON in memory."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield PremiseRecord(
                name=obj["name"],
                kind=obj["kind"],
                statement=obj["statement"],
                module=obj["module"],
                is_prop=bool(obj["is_prop"]),
            )


def load_premise_corpus(
    path: str | Path,
    *,
    props_only: bool = False,
    kinds: Iterable[str] | None = None,
    module_prefixes: Iterable[str] | None = None,
    max_statement_chars: int | None = 4000,
) -> list[PremiseRecord]:
    """Load and filter the corpus, preserving extraction order.

    Order is preserved because it is the index's row order, and a stable row order is what makes an
    index reusable across runs and its top-k tie-breaks deterministic.

    Args:
        props_only: keep only `Prop`-valued declarations (theorems and lemmas). Definitions are
            genuinely cited by tactics (`unfold`, `simp [Foo]`), so this is off by default; it
            exists to support a corpus-composition ablation.
        kinds: declaration kinds to keep. Defaults to `DEFAULT_KINDS`.
        module_prefixes: keep only declarations whose module starts with one of these (e.g.
            `("Mathlib",)` to exclude Lean core, Batteries and tactic-framework internals).
        max_statement_chars: drop pathologically long statements. A 20k-character elaborated type
            cannot fit a prompt alongside five other premises, and its token mass distorts BM25's
            length normalisation for every other document. `None` disables the cap.
    """
    keep_kinds = frozenset(kinds) if kinds is not None else DEFAULT_KINDS
    prefixes = tuple(module_prefixes) if module_prefixes is not None else None

    out: list[PremiseRecord] = []
    for rec in iter_premise_records(path):
        if rec.kind not in keep_kinds:
            continue
        if props_only and not rec.is_prop:
            continue
        if prefixes is not None and not rec.module.startswith(prefixes):
            continue
        if max_statement_chars is not None and len(rec.statement) > max_statement_chars:
            continue
        out.append(rec)
    return out


def corpus_id(records: list[PremiseRecord]) -> str:
    """A short content hash of the candidate set, for asserting arms share a corpus.

    Hashes names only, in order: two corpora with the same names in the same order are the same
    candidate set, and statements are a deterministic function of the pinned Mathlib version. Cheap
    enough to compute on every run, which is the point — a check nobody runs prevents nothing.
    """
    h = hashlib.sha256()
    for rec in records:
        h.update(rec.name.encode("utf-8"))
        h.update(b"\n")
    return f"{len(records)}:{h.hexdigest()[:16]}"
