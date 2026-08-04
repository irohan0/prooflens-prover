"""Regression tests for REPL-restart recovery.

`AutoLeanServer` restarts the Lean REPL when *system-wide* memory crosses `max_total_memory` — on a
shared cluster node that can be triggered by another user's job. Every env id and proof-state id the
old REPL issued is void afterwards.

The first full FATE-M run (141 problems, LI arm) hit this at problem 109. The cached header env
became invalid and problems 109-140 each failed in ~1 ms with `Unknown environment.` and an empty
trace — 32/141 of the benchmark recorded as failures without a single tactic being attempted. The
run *completed successfully* and finalised its manifest, so nothing flagged it; it was only visible
because errors were contiguous to the end of the file.

Two properties are tested here, and the second matters more than the first:

1. The header env is session-cached, so it survives a restart.
2. A void proof state is reported as CRASH, not ERROR. If it were ERROR the search would watch
   every candidate "fail", empty the frontier, and record NO_CANDIDATES — which reads as "the
   retriever proposed nothing that worked". That is a fabricated retrieval result, and it is
   indistinguishable from a real one in the output.

Hermetic: the REPL is a fake. No Lean, no network.
"""

from __future__ import annotations

import pytest

from prooflens_prover.lean.backend import Outcome, ProofState, TacticPolicy
from prooflens_prover.lean.leaninteract_backend import (
    DEFAULT_HEADER_TIMEOUT_S,
    STALE_HANDLE_MARKERS,
    LeanInteractBackend,
    is_stale_handle,
)

pytest.importorskip("lean_interact", reason="lean-interact not installed")

from lean_interact.interface import LeanError  # noqa: E402


class FakeMessage:
    def __init__(self, data: str, severity: str = "info") -> None:
        self.data = data
        self.severity = severity


class FakeSorry:
    def __init__(self, pid: int) -> None:
        self.proof_state = pid
        self.goal = "⊢ True"


class FakeCommandResponse:
    """Mimics `CommandResponse` structurally. Not the real class — see `_patch_isinstance`."""

    def __init__(self, env: int, sorries=(), messages=()) -> None:
        self.env = env
        self.sorries = list(sorries)
        self.messages = list(messages)


class FakeServer:
    """A REPL that can be told to forget its environments, as a restart does.

    Records every call so a test can assert on *how* the backend recovered, not just that it did.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.live_envs: set[int] = set()
        self.next_env = 0
        self.session_envs: set[int] = set()
        self.lean_version = "4.16.0"

    def restart_losing_plain_envs(self) -> None:
        """What AutoLeanServer does on memory pressure: plain envs die, session envs replay."""
        self.live_envs = set(self.session_envs)

    def run(self, request, timeout=None, add_to_session_cache=False, verbose=False):
        cmd = getattr(request, "cmd", None)
        env = getattr(request, "env", None)
        self.calls.append({"cmd": cmd, "env": env, "cached": add_to_session_cache})

        if env is not None and env not in self.live_envs:
            return LeanError(message="Unknown environment.")

        if add_to_session_cache:
            self.next_env -= 1                 # session ids are negative, per AutoLeanServer
            new_env = self.next_env
            self.session_envs.add(new_env)
        else:
            new_env = self.next_env = self.next_env + 1
        self.live_envs.add(new_env)
        return FakeCommandResponse(env=new_env, sorries=[FakeSorry(pid=100)])

    def kill(self) -> None:
        pass


@pytest.fixture
def backend(monkeypatch):
    """A real `LeanInteractBackend` wired to a fake REPL.

    `__new__` sidesteps `__init__`, which would build a `LeanREPLConfig` and want a real Lean
    project. Everything under test — the session-cache flag, the recovery path, the CRASH verdict —
    lives in the methods, so this exercises the shipping code rather than a copy of it.

    `CommandResponse` is patched because the backend narrows responses with `isinstance`, and the
    fake is structural.
    """
    import lean_interact.interface as iface

    monkeypatch.setattr(iface, "CommandResponse", FakeCommandResponse)

    b = LeanInteractBackend.__new__(LeanInteractBackend)
    b.header = "import Mathlib"
    b.policy = TacticPolicy()
    b.tactic_timeout = 60.0
    b.header_timeout = DEFAULT_HEADER_TIMEOUT_S
    b._headers = {}
    b.n_stale_env_recoveries = 0
    b.server = server = FakeServer()
    return b, server


class TestStaleHandleDetection:
    @pytest.mark.parametrize("msg", [
        "Unknown environment.",
        "unknown environment",
        "  Unknown environment.\n",
        "Unknown proof state.",
        "Unknown proof state 42.",
    ])
    def test_recognises_stale_handles(self, msg):
        assert is_stale_handle(msg)

    @pytest.mark.parametrize("msg", [
        "unknown identifier 'foo'",
        "linarith failed to find a contradiction",
        "The rfl tactic failed.",
        "unsolved goals",
        "",
        None,
    ])
    def test_does_not_swallow_ordinary_lean_errors(self, msg):
        # Over-matching here would silently convert real tactic failures into CRASHes and abort
        # attempts that should have continued searching.
        assert not is_stale_handle(msg)

    def test_markers_are_lowercase(self):
        # `is_stale_handle` lowercases the message, so an uppercase marker would never match.
        assert all(m == m.lower() for m in STALE_HANDLE_MARKERS)


class TestHeaderSurvivesRestart:
    def test_header_is_session_cached(self, backend):
        b, server = backend
        b.warm_header("import Mathlib")
        header_calls = [c for c in server.calls if c["cmd"] == "import Mathlib"]
        assert header_calls, "header was never elaborated"
        assert header_calls[0]["cached"] is True, (
            "the header env must be session-cached or it will not survive a REPL restart"
        )

    def test_session_env_id_is_negative(self, backend):
        b, _ = backend
        assert b.warm_header("import Mathlib") < 0

    def test_statement_still_elaborates_after_a_restart(self, backend):
        b, server = backend
        b.warm_header("import Mathlib")
        server.restart_losing_plain_envs()
        # This is the exact scenario that killed problems 109-140.
        state = b.start_theorem("theorem t : True")
        assert isinstance(state, ProofState)

    def test_recovers_when_the_session_cache_is_also_lost(self, backend):
        b, server = backend
        b.warm_header("import Mathlib")
        server.live_envs.clear()
        server.session_envs.clear()          # worst case: even the replay failed
        state = b.start_theorem("theorem t : True")
        assert isinstance(state, ProofState)
        assert b.n_stale_env_recoveries == 1, "the recovery must be counted, not hidden"

    def test_header_is_reelaborated_exactly_once_on_recovery(self, backend):
        b, server = backend
        b.warm_header("import Mathlib")
        server.live_envs.clear()
        server.session_envs.clear()
        b.start_theorem("theorem t : True")
        n = sum(1 for c in server.calls if c["cmd"] == "import Mathlib")
        assert n == 2, f"expected one warm + one recovery import, got {n}"

    def test_recovery_does_not_fire_for_an_ordinary_elaboration_failure(self, backend):
        b, server = backend
        b.warm_header("import Mathlib")

        def failing_run(request, timeout=None, add_to_session_cache=False, verbose=False):
            return LeanError(message="unknown identifier 'nonexistent_lemma'")

        server.run = failing_run
        with pytest.raises(RuntimeError, match="failed to elaborate"):
            b.start_theorem("theorem t : nonexistent_lemma")
        assert b.n_stale_env_recoveries == 0


class TestHeaderTimeout:
    """Job 18165549 died 20 minutes in because `import Mathlib` exceeded a hard-coded 900 s.

    The measured cost of that import on CSF3 is 439-691 s and is I/O-bound on a shared NFS mount,
    so its tail is set by other users' jobs. A ceiling near the observed worst case is a coin flip
    on whether the run survives its first minute of real work.
    """

    def test_default_clears_the_measured_worst_case_by_a_wide_margin(self):
        # 691 s is the slowest honest observation (CSF3, 1 core). Anything under ~2x that is a bet.
        assert DEFAULT_HEADER_TIMEOUT_S >= 2 * 691

    def test_timeout_is_retried_once(self, backend):
        b, server = backend
        real_run = server.run
        calls = {"n": 0}

        def flaky_run(request, timeout=None, add_to_session_cache=False, verbose=False):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("The Lean server did not respond in time and is now killed.")
            return real_run(request, timeout=timeout,
                            add_to_session_cache=add_to_session_cache, verbose=verbose)

        server.run = flaky_run
        env = b.warm_header("import Mathlib")
        assert env < 0, "the retry must still session-cache the environment"
        assert calls["n"] == 2

    def test_a_persistent_timeout_still_fails_loudly(self, backend):
        # Retrying forever would turn a genuinely hung REPL into a silent stall against the
        # SLURM walltime, which is strictly worse than failing.
        b, server = backend

        def always_timeout(request, timeout=None, add_to_session_cache=False, verbose=False):
            raise TimeoutError("hung")

        server.run = always_timeout
        with pytest.raises(TimeoutError):
            b.warm_header("import Mathlib")

    def test_configured_timeout_is_the_one_passed_to_the_repl(self, backend):
        b, server = backend
        b.header_timeout = 1234.0
        seen = {}

        def capture(request, timeout=None, add_to_session_cache=False, verbose=False):
            seen["timeout"] = timeout
            return FakeCommandResponse(env=-1)

        server.run = capture
        server.session_envs.add(-1)
        b.warm_header("import Mathlib")
        assert seen["timeout"] == 1234.0


class TestVoidProofStateIsACrash:
    """The integrity property: a restart must never be reported as a retrieval outcome."""

    def test_stale_proof_state_reports_crash_not_error(self, backend):
        b, server = backend

        def stale_run(request, timeout=None, add_to_session_cache=False, verbose=False):
            return LeanError(message="Unknown proof state.")

        server.run = stale_run
        result = b.run_tactic(ProofState(pid=100, goals=("⊢ True",)), "simp")
        assert result.outcome is Outcome.CRASH, (
            "a void proof state reported as ERROR makes the search empty its frontier and record "
            "NO_CANDIDATES — a retrieval verdict for something retrieval never influenced"
        )

    def test_ordinary_tactic_failure_is_still_error(self, backend):
        b, server = backend

        def failing_run(request, timeout=None, add_to_session_cache=False, verbose=False):
            return LeanError(message="linarith failed to find a contradiction")

        server.run = failing_run
        result = b.run_tactic(ProofState(pid=100, goals=("⊢ True",)), "linarith")
        assert result.outcome is Outcome.ERROR

    def test_crash_is_counted(self, backend):
        b, server = backend

        def stale_run(request, timeout=None, add_to_session_cache=False, verbose=False):
            return LeanError(message="Unknown proof state.")

        server.run = stale_run
        b.run_tactic(ProofState(pid=100, goals=("⊢ True",)), "simp")
        assert b.n_stale_env_recoveries == 1
