"""Live Lean tests — the Tier-0 gate, as a regression suite.

Marked `lean` and skipped when no Lean project is configured, so `pytest` stays green in a
hermetic environment. Point them at a pre-built Mathlib project::

    PROOFLENS_LEAN_PROJECT=~/lean-practice/my_proofs pytest -m lean

`import Mathlib` costs ~85s and ~4GB, so the backend is a **session-scoped** fixture and every test
here shares it. That also makes these tests exercise the property that actually matters at
benchmark scale: many theorems and thousands of tactics through *one* long-lived REPL.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prooflens_prover.lean.backend import Outcome, TacticPolicy

pytestmark = pytest.mark.lean

_PROJECT = os.environ.get("PROOFLENS_LEAN_PROJECT")


@pytest.fixture(scope="session")
def backend():
    if not _PROJECT:
        pytest.skip("set PROOFLENS_LEAN_PROJECT to a pre-built Mathlib project to run Lean tests")
    project = Path(_PROJECT).expanduser()
    if not project.exists():
        pytest.skip(f"PROOFLENS_LEAN_PROJECT does not exist: {project}")

    from prooflens_prover.lean.leaninteract_backend import LeanInteractBackend

    be = LeanInteractBackend(project_dir=project, tactic_timeout=120.0)
    yield be
    be.close()


@pytest.fixture(scope="session")
def comm_root(backend):
    """Root state of `a + b = b + a` — reused across tests (states are immutable handles)."""
    return backend.start_theorem("theorem t_comm (a b : Nat) : a + b = b + a")


class TestInteraction:
    def test_root_state_has_goals(self, comm_root):
        assert not comm_root.is_solved
        assert "a + b = b + a" in comm_root.pp

    def test_closing_tactic_is_proved(self, backend, comm_root):
        r = backend.run_tactic(comm_root, "exact Nat.add_comm a b")
        assert r.outcome is Outcome.PROVED, r.error

    def test_progress_tactic_keeps_goals(self, backend, comm_root):
        r = backend.run_tactic(comm_root, "cases a")
        assert r.outcome is Outcome.PROGRESS
        assert r.state is not None and len(r.state.goals) == 2

    def test_failing_tactic_is_error_not_crash(self, backend, comm_root):
        r = backend.run_tactic(comm_root, "exact Nat.add_comm a b c d")
        assert r.outcome is Outcome.ERROR
        assert r.error

    def test_repl_survives_failure(self, backend, comm_root):
        backend.run_tactic(comm_root, "this_tactic_does_not_exist")
        r = backend.run_tactic(comm_root, "exact Nat.add_comm a b")
        assert r.outcome is Outcome.PROVED, "REPL did not survive a failed tactic"

    def test_multi_step_proof(self, backend):
        """Walk an actual two-step proof — the state threading search depends on."""
        root = backend.start_theorem("theorem t_chain (n : Nat) : n + 0 = n")
        r1 = backend.run_tactic(root, "induction n with\n| zero => rfl\n| succ k ih => simp")
        assert r1.outcome is Outcome.PROVED, r1.error


class TestProofIntegrity:
    """The properties that stop a non-proof being scored as a proof."""

    def test_sorry_rejected_by_policy(self, backend, comm_root):
        r = backend.run_tactic(comm_root, "sorry")
        assert r.outcome is Outcome.REJECTED

    def test_sorry_is_not_proved_even_if_policy_is_bypassed(self, backend, comm_root):
        """**The adversarial check.** Disable the pre-execution guard, actually run `sorry` in
        Lean, and confirm the verdict logic independently refuses to call the result PROVED.

        Two independent defences must both hold, because the policy filter alone is one edit away
        from being disabled, and a `sorry` scored as a proof would inflate every headline number
        in the dissertation with nothing downstream to catch it.
        """
        permissive = TacticPolicy(allow_native_decide=True)
        object.__setattr__(backend, "policy", _NoCheatFilter(permissive))
        try:
            r = backend.run_tactic(comm_root, "sorry")
        finally:
            object.__setattr__(backend, "policy", TacticPolicy())

        assert r.outcome is not Outcome.PROVED, (
            f"CRITICAL: Lean accepted `sorry` as a completed proof (outcome={r.outcome}). "
            "Every reported pass rate would be inflated."
        )

    def test_statement_with_real_proof_is_stripped(self, backend):
        """A statement supplied with a real proof body must still yield an OPEN root state,
        otherwise the problem would score as solved before search even begins."""
        root = backend.start_theorem("theorem t_stripped (a : Nat) : a = a := by rfl")
        assert not root.is_solved
        assert root.goals


class _NoCheatFilter:
    """Policy wrapper that allows everything — test-only, to reach the Lean-side defence."""

    def __init__(self, inner: TacticPolicy) -> None:
        self._inner = inner

    def reject_reason(self, tactic: str) -> str | None:  # noqa: ARG002
        return None
