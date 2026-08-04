"""`LeanBackend` implemented on **LeanInteract** — the fix for the previous project's blocker.

## Why this backend and not lean-dojo

`prooflens/results/phase_logs/phase20.md` records the Part-4 gate failing: LeanDojo's `Dojo`
initialised and read the goal, but `run_tac` died with `DojoCrashError: Unexpected EOF` because
lean-dojo ships its **own** Lean-side shim (`Lean4Repl.lean`) that had drifted out of sync with
Lean 4.20. That shim is the unmaintained part.

LeanInteract instead wraps `leanprover-community/repl` (via `augustepoiroux/repl`), the REPL the
rest of the field actually uses, and tracks Lean releases up to v4.32. Verified against the
**installed** package (lean-interact 0.11.5) rather than documentation:

    LeanREPLConfig(project=LocalProject(directory=...), memory_hard_limit_mb=..., ...)
    AutoLeanServer(config).run(request, timeout=..., verbose=...) -> BaseREPLResponse | LeanError
    Command(cmd=str, env=int|None)  ->  CommandResponse(messages, sorries, env, ...)
    ProofStep(proof_state=int, tactic=str) -> ProofStepResponse(messages, sorries, proof_status,
                                                                proof_state, goals, traces)

## The `sorry` trick for opening a theorem

There is no "start proving this theorem" REPL command. The standard approach, used here and by
every comparable harness, is to elaborate the statement with a `sorry` body and take the resulting
sorry's `proof_state` as the search root::

    theorem putnam_1962_a1 (...) : ... := by sorry
                                             ^^^^^ -> sorries[0].proof_state

The elaborated statement is real, so the root state is exactly what a human would see. `sorry` here
is *ours*, in the statement scaffold — it never appears in a candidate tactic, which
`TacticPolicy` rejects outright, and `verdict_from_repl` independently refuses to call any
sorry-tainted result a proof.

## Memory — and a trap that costs a whole cluster job

Each REPL with Mathlib imported costs roughly 3.5-5 GB **resident**.

`memory_hard_limit_mb` is **not** an RSS cap: it constrains the process's *virtual address space*.
Lean maps far more address space than it ever resides (thread stacks, its allocator's arenas), so a
limit that looks generous against RSS starves it. Measured on this project's dev machine, same
code, only the limit changing:

| `memory_hard_limit_mb` | result |
|---|---|
| 8192  | **fails** — `lean::exception: failed to create thread` |
| 32768 | passes |
| `None` | passes |

The failure surfaces as a `ConnectionAbortedError` with "failed to create thread", which reads like
memory exhaustion and is *not* — the host had 6.9 GB free at the time. Anyone debugging that would
reasonably reach for a *smaller* limit and make it worse.

**So the default here is `None`.** The correct way to cap real memory is the layer that measures
RSS: SLURM's cgroup (`#SBATCH --mem=`) on the cluster, and `AutoLeanServer`'s own RSS monitoring,
which restarts a bloated REPL. If a belt-and-braces address-space cap is wanted anyway, pass
something generous (>= 32768); it is a backstop against runaway elaboration, not a memory budget.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from prooflens_prover.lean.backend import (
    Outcome,
    ProofState,
    TacticPolicy,
    TacticResult,
    contains_cheat,
    verdict_from_repl,
)
from prooflens_prover.utils.logging import get_logger

log = get_logger("lean.leaninteract")

DEFAULT_HEADER = "import Mathlib"

#: Measured floor for `memory_hard_limit_mb`. 8192 reproducibly kills Lean at startup; 32768 works.
#: See the module docstring for the measurement.
MIN_SAFE_MEMORY_LIMIT_MB = 32768

#: Timeout for elaborating a header (`import Mathlib`), in seconds.
#:
#: Sized from measurements, not taste. `import Mathlib` on this project has been observed at 439 s
#: (CSF3 `multicore`, 4 cores), 691 s (CSF3, 1 core) and ~90-160 s locally. It is dominated by
#: reading ~5.4 GB of `.olean` files over NFS, so it is I/O-bound and its tail is set by whatever
#: else is hitting the filesystem — i.e. by other users' jobs, which we do not control.
#:
#: The previous value of 900 s was ~1.3x the worst observation. Job 18165549 exceeded it and died
#: 20 minutes in with nothing recorded. The asymmetry is total: being generous costs nothing when
#: the import succeeds (the call returns as soon as Lean answers), while being tight costs the
#: entire run. So this is deliberately far above any plausible honest duration — it exists to catch
#: a genuinely hung REPL, not to police a slow filesystem.
DEFAULT_HEADER_TIMEOUT_S = 5400.0

#: REPL error messages meaning "the handle you passed no longer exists" — the server restarted
#: underneath us and every env / proof-state id it issued is void. Matched case-insensitively
#: against the message text because the REPL gives no structured error code for this.
STALE_HANDLE_MARKERS = ("unknown environment", "unknown proof state")


def is_stale_handle(message: str | None) -> bool:
    """True when `message` says a REPL handle is no longer valid.

    Distinguishing this from an ordinary Lean error matters: an ordinary error is *data* about the
    proof attempt, whereas a stale handle is the harness talking to a REPL that has been replaced.
    Recording the latter as a tactic failure makes a restart look like a retriever result.
    """
    m = (message or "").lower()
    return any(marker in m for marker in STALE_HANDLE_MARKERS)


class LeanInteractBackend:
    """Tactic-level Lean interaction backed by LeanInteract's `AutoLeanServer`.

    Args:
        project_dir: path to a **pre-built** Lean project that depends on Mathlib. Strongly
            preferred over `TempRequireProject` for our use: building Mathlib takes hours, and
            CSF3 compute nodes should never attempt it. Point this at a project whose `.lake` is
            already populated.
        lean_version: used only when `project_dir` is None, to construct a temporary Mathlib
            project (first call downloads and builds — laptop only, never in a batch job).
        header: default environment preamble, elaborated once and reused for every theorem.
        policy: which tactics the harness refuses to execute.
        tactic_timeout: per-tactic wall-clock ceiling in seconds.
        memory_limit_mb: **virtual address space** cap for the REPL, not an RSS budget. Defaults
            to None; see the module docstring — values below ~32 GB make Lean fail to start with a
            misleading "failed to create thread" error. Use SLURM `--mem` to cap real memory.
        header_timeout: ceiling for elaborating the header, in seconds. See
            `DEFAULT_HEADER_TIMEOUT_S` — this guards against a hung REPL, not a slow filesystem,
            and should stay well above the observed NFS worst case.
    """

    def __init__(
        self,
        project_dir: str | Path | None = None,
        lean_version: str | None = None,
        header: str = DEFAULT_HEADER,
        policy: TacticPolicy | None = None,
        tactic_timeout: float = 60.0,
        memory_limit_mb: int | None = None,
        header_timeout: float = DEFAULT_HEADER_TIMEOUT_S,
        verbose: bool = False,
    ) -> None:
        from lean_interact import AutoLeanServer, LeanREPLConfig, LocalProject, TempRequireProject

        self.header = header
        self.policy = policy or TacticPolicy()
        self.tactic_timeout = tactic_timeout
        self.header_timeout = header_timeout
        self._headers: dict[str, int] = {}   # header text -> REPL env id (elaborated once)
        #: How many times a REPL restart was detected and recovered from. Non-zero means the run
        #: hit memory pressure; report it alongside the results rather than letting it hide.
        self.n_stale_env_recoveries = 0

        if project_dir is not None:
            p = Path(project_dir).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"Lean project not found: {p}")
            project: Any = LocalProject(directory=str(p), auto_build=False)
            log.info("lean backend: LocalProject at %s", p)
        else:
            if lean_version is None:
                raise ValueError("give either project_dir or lean_version")
            project = TempRequireProject(lean_version=lean_version, require="mathlib")
            log.info("lean backend: TempRequireProject(mathlib, %s) — first run builds",
                     lean_version)

        # Guard the address-space trap documented in the module docstring: below ~32 GB Lean
        # cannot allocate its thread stacks and dies with an error that looks like OOM but is not.
        if memory_limit_mb is not None and memory_limit_mb < MIN_SAFE_MEMORY_LIMIT_MB:
            log.warning(
                "memory_limit_mb=%d is below the measured safe floor of %d. This caps VIRTUAL "
                "address space, not RSS; Lean will likely die with 'failed to create thread'. "
                "Use SLURM --mem to cap real memory instead. Proceeding as asked.",
                memory_limit_mb, MIN_SAFE_MEMORY_LIMIT_MB,
            )

        self.config = LeanREPLConfig(
            project=project,
            memory_hard_limit_mb=memory_limit_mb,
            verbose=verbose,
        )
        self.server = AutoLeanServer(self.config)
        log.info("lean backend ready (lean %s)", self.lean_version)

    # -- lifecycle -------------------------------------------------------------------------------
    @property
    def lean_version(self) -> str:
        try:
            return str(self.server.lean_version)
        except Exception:  # noqa: BLE001 - diagnostic only; never fail a run over a version string
            return "unknown"

    def __enter__(self) -> LeanInteractBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.server.kill()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass

    # -- internals -------------------------------------------------------------------------------
    def _env_for(self, header: str) -> int:
        """Elaborate `header` once and cache its env id.

        `import Mathlib` costs tens of seconds and ~4 GB. Re-elaborating it per theorem would make
        a full benchmark run impossible, so environments are cached per distinct header text.
        """
        if header in self._headers:
            return self._headers[header]

        from lean_interact import Command
        from lean_interact.interface import CommandResponse, LeanError

        t0 = time.perf_counter()
        # `add_to_session_cache=True` is load-bearing, not an optimisation. `AutoLeanServer`
        # restarts the REPL whenever *system-wide* memory crosses `max_total_memory` (0.8 by
        # default) — on a shared cluster node that can be triggered by another user's job. A restart
        # voids every positive env id the REPL ever issued, so a plainly-cached env id becomes
        # "Unknown environment." for every subsequent theorem, permanently. That is exactly what
        # happened on the first full FATE-M run: problems 0-108 ran normally, the REPL restarted,
        # and problems 109-140 each failed in ~1 ms with an empty trace — 32/141 of the benchmark
        # scored as failures without being attempted.
        #
        # A session-cached environment gets a *negative* id which `AutoLeanServer` replays into the
        # fresh REPL on restart and remaps transparently, so the handle survives.
        try:
            resp = self.server.run(
                Command(cmd=header), timeout=self.header_timeout, add_to_session_cache=True
            )
        except TimeoutError:
            # One retry, because the first attempt is not wasted: it pulled several GB of `.olean`
            # files through the page cache, so the second is normally far faster. Without this a
            # single slow filesystem moment costs the whole job — which is what happened on job
            # 18165549, 20 minutes in, with nothing recorded.
            log.warning(
                "header %r did not elaborate within %.0fs and the REPL was killed. Retrying once; "
                "the page cache is warmer now, so this attempt is usually much faster.",
                header, self.header_timeout,
            )
            resp = self.server.run(
                Command(cmd=header), timeout=self.header_timeout, add_to_session_cache=True
            )
        if isinstance(resp, LeanError):
            raise RuntimeError(f"header failed to elaborate: {header!r}\n{resp.message}")
        if not isinstance(resp, CommandResponse):
            raise RuntimeError(f"unexpected response elaborating header: {type(resp).__name__}")
        errs = [m.data for m in resp.messages if m.severity == "error"]
        if errs:
            raise RuntimeError(f"header {header!r} produced errors: {errs}")

        log.info("elaborated header %r -> env %d (%.1fs)",
                 header, resp.env, time.perf_counter() - t0)
        self._headers[header] = resp.env
        return resp.env

    def warm_header(self, header: str | None = None) -> int:
        """Elaborate and cache `header` up front. Returns the env id.

        **Call this before starting any timed search.** Header elaboration is lazy, so without it
        the first `start_theorem` of a run pays for `import Mathlib` *inside* that problem's
        wall-clock budget — 691 s on CSF3's NFS against a 600 s budget, which made the search break
        with zero expansions and record a legitimate-looking `EXHAUSTED`. The first problem of every
        benchmark run was being silently forfeited.
        """
        t0 = time.perf_counter()
        env = self._env_for(header or self.header)
        log.info("header warm (%.1fs) — search timers now exclude Mathlib import",
                 time.perf_counter() - t0)
        return env

    def _run_in_env(self, cmd: str, header: str | None, timeout: float) -> Any:
        """Run `cmd` against the header environment, re-elaborating the header once if the handle
        went stale.

        The session cache in `_env_for` should make this unreachable; it exists because the failure
        it guards against is silent and total. If the cache is ever lost, one problem pays for one
        `import Mathlib` and the run continues, rather than every remaining problem failing in a
        millisecond with an error that looks like a malformed benchmark statement.
        """
        from lean_interact import Command
        from lean_interact.interface import LeanError

        key = header or self.header
        resp = self.server.run(Command(cmd=cmd, env=self._env_for(key)), timeout=timeout)
        if isinstance(resp, LeanError) and is_stale_handle(resp.message):
            self.n_stale_env_recoveries += 1
            log.warning(
                "REPL environment for header %r is gone (%s) — the server restarted. "
                "Re-elaborating the header; this costs one import, not the rest of the run.",
                key, (resp.message or "").strip(),
            )
            self._headers.pop(key, None)
            resp = self.server.run(Command(cmd=cmd, env=self._env_for(key)), timeout=timeout)
        return resp

    # -- LeanBackend -----------------------------------------------------------------------------
    def start_theorem(self, statement: str, header: str | None = None) -> ProofState:
        """Elaborate `statement` with a `sorry` body and return the root proof state.

        `statement` may be given with or without a proof body; anything from `:=` onward is
        replaced by `:= by sorry` so the caller cannot accidentally supply a real proof (which
        would make the "root state" a solved one and silently score the problem as trivially
        proved).
        """
        from lean_interact.interface import CommandResponse, LeanError

        stmt = _as_sorry_statement(statement)
        resp = self._run_in_env(stmt, header, timeout=self.tactic_timeout * 4)
        if isinstance(resp, LeanError):
            raise RuntimeError(f"statement failed to elaborate:\n{resp.message}")
        if not isinstance(resp, CommandResponse):
            raise RuntimeError(f"unexpected response type {type(resp).__name__}")

        errs = [m.data for m in resp.messages if m.severity == "error"]
        if errs:
            raise RuntimeError(f"statement did not elaborate: {errs}")
        if not resp.sorries:
            raise RuntimeError(
                "statement elaborated but produced no `sorry` goal — it may already be proved, or "
                f"the body was not replaced correctly. Sent:\n{stmt}"
            )

        s = resp.sorries[0]
        if s.proof_state is None:
            raise RuntimeError("REPL returned a sorry without a proof_state handle")
        return ProofState(pid=int(s.proof_state), goals=(s.goal,) if s.goal else ())

    def check_command(
        self, command: str, header: str | None = None, timeout: float | None = None
    ) -> tuple[bool, list[str], list[str]]:
        """Elaborate a complete Lean command; return `(ok, error_messages, all_messages)`.

        Used to re-check a finished proof **independently of the search that produced it**. The
        search decides PROVED from the REPL's per-tactic responses; this replays the assembled
        `theorem … := by …` from scratch and asks Lean directly. The two paths share only
        `verdict_from_repl`'s notion of a cheat token, so a bug in the incremental tactic
        bookkeeping cannot pass both.

        `ok` is False if Lean reported any error **or** any message mentions a cheat token — Lean
        reports a vacuous proof as the *warning* ``declaration uses 'sorry'``, so checking severity
        alone would accept it.
        """
        from lean_interact.interface import CommandResponse, LeanError

        resp = self._run_in_env(command, header, timeout=timeout or self.tactic_timeout * 4)
        if isinstance(resp, LeanError):
            return False, [resp.message], [resp.message]
        if not isinstance(resp, CommandResponse):
            return False, [f"unexpected response type {type(resp).__name__}"], []

        messages = [m.data for m in resp.messages]
        errors = [m.data for m in resp.messages if m.severity == "error"]
        cheats = [m for m in messages if contains_cheat(m)]
        if contains_cheat(command):
            cheats.append(f"cheat token in the submitted proof text: {command!r}")
        return (not errors and not cheats), errors + cheats, messages

    def run_tactic(
        self, state: ProofState, tactic: str, timeout: float | None = None
    ) -> TacticResult:
        """Apply `tactic` to `state`. Ordinary tactic failure is a result, never an exception."""
        from lean_interact import ProofStep
        from lean_interact.interface import LeanError, ProofStepResponse

        reason = self.policy.reject_reason(tactic)
        if reason is not None:
            return TacticResult(outcome=Outcome.REJECTED, error=reason)

        t0 = time.perf_counter()
        try:
            resp = self.server.run(
                ProofStep(proof_state=state.pid, tactic=tactic),
                timeout=timeout if timeout is not None else self.tactic_timeout,
            )
        except TimeoutError as e:
            return TacticResult(
                outcome=Outcome.TIMEOUT, error=str(e) or "tactic timed out",
                elapsed_s=time.perf_counter() - t0,
            )
        except Exception as e:  # noqa: BLE001 - REPL death must degrade one attempt, not the job
            log.warning("REPL crash on tactic %r: %s: %s", tactic[:80], type(e).__name__, e)
            return TacticResult(
                outcome=Outcome.CRASH, error=f"{type(e).__name__}: {e}",
                elapsed_s=time.perf_counter() - t0,
            )
        elapsed = time.perf_counter() - t0

        if isinstance(resp, LeanError):
            if is_stale_handle(resp.message):
                # The REPL restarted, so this proof state — and every other node in the search
                # tree — no longer exists. Reporting ERROR here would make the tactic look like it
                # failed on its merits; the search would then watch every candidate "fail", empty
                # the frontier, and record NO_CANDIDATES. That is a *fabricated retrieval result*:
                # it reads as "the retriever proposed nothing that worked" when in fact nothing was
                # ever executed. CRASH tells the search the tree is void, and it returns an honest
                # ERROR with the trace intact.
                self.n_stale_env_recoveries += 1
                log.warning("proof state %d void after REPL restart: %s",
                            state.pid, (resp.message or "").strip())
                return TacticResult(
                    outcome=Outcome.CRASH, error=resp.message, elapsed_s=elapsed
                )
            return TacticResult(outcome=Outcome.ERROR, error=resp.message, elapsed_s=elapsed)
        if not isinstance(resp, ProofStepResponse):
            return TacticResult(
                outcome=Outcome.ERROR,
                error=f"unexpected response type {type(resp).__name__}",
                elapsed_s=elapsed,
            )

        msgs = tuple(m.data for m in resp.messages)
        sevs = tuple(m.severity for m in resp.messages)
        outcome = verdict_from_repl(
            proof_status=resp.proof_status, goals=resp.goals, messages=msgs, error_severities=sevs
        )

        # A tactic that introduces a fresh `sorry` leaves an unsolved hole. `verdict_from_repl`
        # already refuses to call that PROVED via the message scan; this is the structural check
        # on the same condition, from the REPL's own sorry list rather than its prose.
        if outcome is Outcome.PROVED and resp.sorries:
            outcome = Outcome.ERROR
            msgs = msgs + ("rejected: proof closed with an outstanding sorry",)

        if outcome in (Outcome.ERROR,):
            return TacticResult(
                outcome=outcome,
                error="; ".join(m for m, s in zip(msgs, sevs, strict=False) if s == "error")
                or (msgs[0] if msgs else "tactic failed"),
                elapsed_s=elapsed,
                messages=msgs,
            )

        return TacticResult(
            outcome=outcome,
            state=ProofState(pid=int(resp.proof_state), goals=tuple(resp.goals)),
            elapsed_s=elapsed,
            messages=msgs,
        )


def _as_sorry_statement(statement: str) -> str:
    """Normalise a theorem statement to end in `:= by sorry`.

    Split on the **last** top-level `:=` so statements containing `:=` inside binders or `let`
    bindings are not mangled. Idempotent: a statement already ending in `sorry` is returned with a
    single canonical body.
    """
    s = statement.strip().rstrip()
    idx = _last_toplevel_assign(s)
    if idx >= 0:
        s = s[:idx].rstrip()
    return f"{s} := by sorry"


def statement_with_proof(statement: str, tactics: list[str]) -> str:
    """Assemble `statement` and a tactic list into a complete `theorem … := by …` declaration.

    The inverse of `_as_sorry_statement`, and deliberately built from the *same* body-splitting
    logic: verification must reconstruct the identical declaration the search proved, or it is
    checking a different theorem. Every line is indented so a tactic that spans lines stays inside
    the `by` block.
    """
    s = statement.strip()
    idx = _last_toplevel_assign(s)
    if idx >= 0:
        s = s[:idx].rstrip()
    body = "\n".join(
        f"  {line}" for tactic in tactics for line in tactic.splitlines() or [""]
    )
    return f"{s} := by\n{body}"


def _last_toplevel_assign(s: str) -> int:
    """Index of the last `:=` at bracket depth 0, or -1. Ignores `:=` inside (), [], {}, ⟨⟩."""
    depth = 0
    openers, closers = "([{⟨", ")]}⟩"
    last = -1
    i = 0
    while i < len(s) - 1:
        c = s[i]
        if c in openers:
            depth += 1
        elif c in closers:
            depth -= 1
        elif depth == 0 and c == ":" and s[i + 1] == "=":
            last = i
            i += 1
        i += 1
    return last
