"""Lean-aware tokenisation, shared by every lexical component.

Ported from `prooflens/src/prooflens/retrievers/bm25.py::lean_tokenize` **unchanged in
behaviour**, so this project's BM25 baseline tokenises identically to the predecessor's.
Keeping it byte-compatible matters more than improving it: the predecessor's headline finding
was that late interaction's advantage is "largely lexical", which is a claim about *this*
tokenisation. Changing it here would quietly change what that claim is re-tested against.

Why not a plain word tokenizer: a Lean proof state is mostly operators and namespaced
identifiers. `∀ {α : Type u} (a b : α), a + b = b + a` has almost no "words". Discarding
`∀`, `+`, `=` throws away most of the signal, and leaving `Nat.add_comm` unsplit makes it
unmatchable.
"""

from __future__ import annotations

import re

__all__ = ["lean_tokenize", "premise_document"]

# Identifier runs (ASCII letters/digits/underscore/prime) OR any single other non-space character.
# The second branch is what makes this Lean-aware: every unicode math operator (`∀ ∈ ⊔ ℝ ≤`) becomes
# its own token, and `.` becomes a separator so `Nat.add_comm` yields `Nat`, `.`, `add_comm`.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[^\sA-Za-z0-9_']", re.UNICODE)

# Underscore-joined identifiers are ALSO emitted as their parts when `split_underscores=True`,
# so a goal mentioning `add` can reach `Nat.add_comm`. Off by default: it changes every
# document-frequency statistic in the index, and the default must match the predecessor's.
_UNDERSCORE_SPLIT_MIN_LEN = 2


def lean_tokenize(
    text: str, lowercase: bool = False, split_underscores: bool = False
) -> list[str]:
    """Tokenise Lean source or a pretty-printed proof state.

    `lowercase=False` by default because declaration-name casing is meaningful in Mathlib
    (`Nat` the type vs `nat` in a hypothesis name; `IsUnit` vs `isUnit`).

    With `split_underscores=True`, an identifier containing `_` contributes both the whole token and
    its parts (`add_comm` -> `add_comm`, `add`, `comm`). The whole token is always retained, so this
    can only add recall, never remove an exact match.
    """
    if lowercase:
        text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    if not split_underscores:
        return tokens
    out: list[str] = []
    for tok in tokens:
        out.append(tok)
        if "_" in tok:
            out.extend(
                part for part in tok.split("_") if len(part) >= _UNDERSCORE_SPLIT_MIN_LEN
            )
    return out


def premise_document(name: str, statement: str) -> str:
    """The text a premise is indexed as: fully-qualified name, then its elaborated statement.

    The name is included because it carries real lexical signal that the statement does not — the
    Mathlib naming convention encodes the statement's shape (`add_comm`, `mul_le_mul_left`), so a
    goal that mentions those operations matches the name even when the elaborated type is stated in
    terms of `HAdd.hAdd`.
    """
    return f"{name} {statement}"
