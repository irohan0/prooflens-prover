"""Resolving the premise names a proof's tactic text cites, against a premise corpus.

Two analyses ask opposite questions of the same resolution step, which is why it lives here rather
than inside either script:

* `scripts/novel_premise_stratification.py` — is the cited premise one the retriever **trained**
  on? (Does LI win where the premises are novel?)
* `scripts/contamination_audit.py` — is the cited premise enough to close the theorem **by
  itself**? (Is the benchmark's answer sitting in the retrieval corpus?)

Both need the same conservatism about abbreviated names, and getting that wrong in one place but
not the other would make the two analyses disagree for a reason that has nothing to do with either
question.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

#: A Lean identifier: leading letter or underscore, then name characters. Dots are included so a
#: qualified citation (`Fintype.card_pi_const`) is captured whole rather than split into two tokens
#: that would each resolve wrongly. Subscripts and primes appear in real Mathlib names.
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.'!?₀-₉¹²³]*")

#: Tokens that are tactic syntax rather than premise citations. Kept deliberately short and used
#: only for *sensitivity* figures, never a primary one. It barely matters which way it goes: every
#: common tactic word that doubles as a lemma name (`rfl` is the single most frequent training
#: premise, at 3,402 positives) is firmly inside the seen set, so it cannot manufacture an
#: unseen-premise finding. Excluding them changes the denominator, not the signal.
TACTIC_WORDS = frozenset("""
    exact apply rw rwa simp simpa intro intros refine constructor rcases obtain cases use have let
    show calc ring ring_nf field_simp linarith nlinarith norm_num omega decide aesop tauto trivial
    exacts induction subst unfold change conv congr ext specialize by at with this fun to and or if
    then else from rfl
""".split())


def load_corpus(corpus_path: Path) -> tuple[set[str], dict[str, set[str]]]:
    """`(exact full names, {last component: full names})` for the premise corpus.

    The suffix index is what lets an abbreviated citation be resolved: a tactic that writes
    `card_pi_const` under `open Fintype` means `Fintype.card_pi_const`, and only the tail is
    written down.
    """
    exact: set[str] = set()
    by_suffix: dict[str, set[str]] = defaultdict(set)
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            name = json.loads(line).get("name")
            if not name:
                continue
            exact.add(name)
            by_suffix[name.rsplit(".", 1)[-1]].add(name)
    return exact, dict(by_suffix)


def resolve(token: str, exact: set[str], by_suffix: dict[str, set[str]]) -> set[str] | None:
    """Full names `token` could denote, or None if it names no premise in the corpus."""
    if token in exact:
        return {token}
    candidates = by_suffix.get(token)
    return set(candidates) if candidates else None


def cited_premises(
    tactics, exact: set[str], by_suffix: dict[str, set[str]], drop_tactic_words: bool = False
) -> dict[str, set[str]]:
    """`{surface token: full names it could denote}` over every premise a proof names."""
    out: dict[str, set[str]] = {}
    for tactic in tactics or ():
        for token in IDENTIFIER.findall(str(tactic)):
            if token in out or (drop_tactic_words and token in TACTIC_WORDS):
                continue
            resolved = resolve(token, exact, by_suffix)
            if resolved:
                out[token] = resolved
    return out
