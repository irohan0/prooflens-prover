"""Tactic-level Lean 4 interaction.

`backend` holds the Protocol and the pure verdict logic (unit-testable without Lean);
`leaninteract_backend` is the concrete REPL implementation used by our own prover.
The concrete backend is imported lazily so this package stays importable without `lean-interact`.
"""

from prooflens_prover.lean.backend import (
    LeanBackend,
    Outcome,
    ProofState,
    TacticPolicy,
    TacticResult,
    contains_cheat,
    verdict_from_repl,
)

__all__ = [
    "LeanBackend",
    "Outcome",
    "ProofState",
    "TacticPolicy",
    "TacticResult",
    "contains_cheat",
    "verdict_from_repl",
]


def get_backend(name: str = "leaninteract", **kwargs):
    """Construct a backend by name. Imported lazily so `import prooflens_prover.lean` works in the
    hermetic test environment, which has no Lean toolchain installed."""
    if name == "leaninteract":
        from prooflens_prover.lean.leaninteract_backend import LeanInteractBackend

        return LeanInteractBackend(**kwargs)
    raise ValueError(f"unknown Lean backend {name!r} (have: leaninteract)")
