"""Backend-agnostic types for tactic-level Lean 4 interaction.

The search loop (`prover/best_first_search.py`) depends only on the `LeanBackend` Protocol below,
never on a concrete REPL. That matters for two reasons:

1. **Two backends are required by the experiment design.** Our own prover runs on LeanInteract;
   the frozen REAL-Prover comparison must run inside *their* environment (jixia + interactive) so
   our numbers are one-to-one with their published ones. Both satisfy this Protocol.
2. **The previous project's prover died on a backend bug**, not an algorithm bug (a lean-dojo
   `Lean4Repl` ↔ Lean-4.20 stdin incompatibility). Isolating backend risk behind an interface means
   the search algorithm can be built and unit-tested against a fake before any Lean is involved.

## The correctness property this module exists to protect

**A proof counts as complete only if Lean says so with no escape hatches.** Every inflated
theorem-proving result in the literature traces back to some version of accepting a proof that
wasn't one. `verdict_from_repl` centralises that judgement so it is decided in exactly one place,
is unit-testable without Lean, and cannot drift between backends or benchmarks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

# Tokens that make a "proof" vacuous. Matched at identifier boundaries so a lemma legitimately
# named e.g. `Nat.sorry_free` cannot trigger a false positive.
#
# **`'` is deliberately treated as a quote character, not an identifier character**, even though
# Lean 4 permits it inside identifiers (`h'`, `foo'`). This was forced by a test failure:
# Lean's own warning for a vacuous proof is `declaration uses 'sorry'` — the token is *quoted*.
# Honouring the identifier reading of `'` made that message stop matching, which would have scored
# a sorry-tainted proof as PROVED.
#
# The trade-off is asymmetric and settled in favour of over-rejection: a false positive discards
# one candidate tactic out of ~32 generated for that state, while a false negative reports a
# theorem as solved when it is not, which silently inflates every headline number in the study.
# The only strings this over-matches are identifiers like `sorry'` or `h'sorry`, which do not exist
# in Mathlib and which we would want to reject anyway.
_CHEAT_TOKENS = ("sorry", "admit", "sorryAx")
_CHEAT_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"           # not directly after an identifier char or a namespace dot
    r"(" + "|".join(_CHEAT_TOKENS) + r")"
    r"(?![A-Za-z0-9_])"             # not directly before an identifier char
)

#: `native_decide` trusts the compiler rather than the kernel, so it is not a kernel-checked proof.
#: Off by default (it is legal Lean and some published results allow it), but every run records
#: which policy it used — see `TacticPolicy`.
_NATIVE_DECIDE_RE = re.compile(r"(?<![A-Za-z0-9_.'])native_decide(?![A-Za-z0-9_'])")

#: Tactics banned for parity with REAL-Prover's `ABANDON_IF_CONTAIN` (`conf/config.py`), so the
#: frozen-prover comparison differs from theirs only in the retriever. `apply?` is Lean's own
#: premise *search*: allowing it would let the prover bypass our retriever entirely and silently
#: erase the effect being measured. Substring match, as theirs is.
_BANNED_SUBSTRINGS = ("apply?",)


class Outcome(StrEnum):
    """What happened when a tactic was applied. `StrEnum` so it serialises straight to JSONL."""

    PROVED = "proved"        # goal closed, no errors, no cheats -> a real proof
    PROGRESS = "progress"    # tactic applied, goals remain
    ERROR = "error"          # tactic did not apply / Lean reported an error
    REJECTED = "rejected"    # tactic refused by policy before execution (e.g. contains `sorry`)
    TIMEOUT = "timeout"      # exceeded the per-tactic wall clock
    CRASH = "crash"          # the REPL process died; the backend needs a restart


@dataclass(frozen=True)
class ProofState:
    """One node in the proof tree.

    `pid` is the backend's opaque handle for this state. `goals` is the pretty-printed goal list;
    an empty tuple means no goals remain. `pp` is what gets shown to the model, and is also the
    retriever's query — so its formatting is experimentally load-bearing and must not drift
    between arms.
    """

    pid: int
    goals: tuple[str, ...] = ()

    @property
    def is_solved(self) -> bool:
        return len(self.goals) == 0

    @property
    def pp(self) -> str:
        """Goals joined as the model and the retriever see them (Lean's own convention)."""
        return "\n\n".join(self.goals)

    def to_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "goals": list(self.goals)}


@dataclass(frozen=True)
class TacticResult:
    """Outcome of applying one tactic, plus everything needed to audit that decision later."""

    outcome: Outcome
    state: ProofState | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    messages: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        """Did the search tree gain a usable node?"""
        return self.outcome in (Outcome.PROVED, Outcome.PROGRESS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "state": self.state.to_dict() if self.state else None,
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, 4),
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class TacticPolicy:
    """What the harness refuses to execute, recorded per run so results are comparable.

    Defaults ban `sorry`/`admit`/`sorryAx`, which is universal. `allow_native_decide` defaults to
    **False**: `native_decide` produces a proof trusted by the compiler rather than checked by the
    kernel, and whether a benchmark permits it varies. Making it an explicit, logged flag stops it
    from becoming an unstated difference between our numbers and a published baseline's.
    """

    allow_native_decide: bool = False
    max_tactic_chars: int = 2000
    ban_premise_search: bool = True     # `apply?` — see `_BANNED_SUBSTRINGS`

    def reject_reason(self, tactic: str) -> str | None:
        """Return why `tactic` is inadmissible, or None if it may be executed."""
        if not tactic.strip():
            return "empty tactic"
        if len(tactic) > self.max_tactic_chars:
            return f"tactic too long ({len(tactic)} > {self.max_tactic_chars} chars)"
        m = _CHEAT_RE.search(tactic)
        if m:
            return f"contains banned token {m.group(1)!r}"
        if not self.allow_native_decide and _NATIVE_DECIDE_RE.search(tactic):
            return "contains native_decide (disallowed by policy)"
        if self.ban_premise_search:
            for banned in _BANNED_SUBSTRINGS:
                if banned in tactic:
                    return (f"contains {banned!r} (Lean's own premise search; "
                            "would bypass the retriever)")
        return None

    def to_dict(self) -> dict[str, Any]:
        """Recorded in every run manifest: an unstated policy difference between our numbers and a
        published baseline's is exactly the kind of thing that makes a comparison meaningless."""
        return {
            "allow_native_decide": self.allow_native_decide,
            "max_tactic_chars": self.max_tactic_chars,
            "ban_premise_search": self.ban_premise_search,
        }


def contains_cheat(text: str) -> bool:
    """True if `text` contains `sorry`/`admit`/`sorryAx` as a whole Lean identifier."""
    return _CHEAT_RE.search(text) is not None


def verdict_from_repl(
    proof_status: str,
    goals: list[str] | tuple[str, ...],
    messages: list[str] | tuple[str, ...],
    error_severities: list[str] | tuple[str, ...] = (),
) -> Outcome:
    """Decide PROVED / PROGRESS / ERROR from a REPL response. **The single source of truth.**

    Pure and backend-independent so it is unit-testable without Lean, and so LeanInteract and jixia
    can never disagree about what counts as a proof.

    A step is PROVED only when *all* of these hold:

    - no message has severity ``error`` — a tactic can "apply" while Lean reports an error, and
      such a proof does not compile;
    - no goals remain;
    - `proof_status` does not report incompleteness;
    - **nothing in the status or messages mentions `sorry`/`admit`** — the REPL reports a
      sorry-containing proof as finished-with-warning, and accepting that would silently turn
      "gave up" into "solved". This is the check that protects every headline number in the study.
    """
    if any(sev == "error" for sev in error_severities):
        return Outcome.ERROR

    blob = f"{proof_status} {' '.join(messages)}"
    if contains_cheat(blob):
        return Outcome.ERROR

    status = proof_status.strip().lower()
    if status.startswith("incomplete") or "open goals" in status:
        return Outcome.PROGRESS if goals else Outcome.ERROR

    if goals:
        return Outcome.PROGRESS
    if status.startswith("completed") or status == "done" or not status:
        return Outcome.PROVED
    # Unrecognised status with no goals left: refuse to guess in the generous direction.
    return Outcome.PROGRESS


@runtime_checkable
class LeanBackend(Protocol):
    """Tactic-level interaction with a live Lean 4 process.

    Implementations must be usable as context managers and must tolerate a dead REPL (restarting
    transparently or reporting `Outcome.CRASH`) — at PutnamBench scale a long run *will* lose REPL
    processes to memory pressure, and that must degrade one attempt, not the job.
    """

    def start_theorem(self, statement: str, header: str | None = None) -> ProofState:
        """Elaborate `statement` (a theorem whose body is `sorry`) and return its root proof state.

        `header` supplies per-problem imports/opens; when None the backend's default Mathlib
        environment is used. Raises if the statement does not elaborate.
        """
        ...

    def run_tactic(
        self, state: ProofState, tactic: str, timeout: float | None = None
    ) -> TacticResult:
        """Apply `tactic` to `state`. Never raises for ordinary tactic failure — that is an
        `Outcome.ERROR` result, which is a normal and very frequent search event."""
        ...

    def close(self) -> None:
        """Release the REPL process."""
        ...
