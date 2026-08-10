"""Logging setup for scripts, the search loop, and the eval harness.

Ported from `prooflens/src/prooflens/utils/logging.py`. One configured logger factory so every
script logs consistently to **stderr** — SLURM captures stderr to `slurm-<jobid>.err`, which is
where run diagnostics need to land on CSF3.
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT = "prooflens_prover"
_CONFIGURED = False


def ensure_utf8_output() -> None:
    """Force stdout/stderr to UTF-8 so a non-ASCII character cannot kill a finished run.

    Almost every script here prints an em-dash, a Δ, or a Lean goal containing `α`. Python picks the
    stream encoding from the locale, so under a POSIX/`cp1252` locale `print()` raises
    `UnicodeEncodeError` — *after* the work is done and the output files are written. That is the
    worst possible time to fail: the result is on disk, the exit status is non-zero, and anything
    chained after the script with `&&` silently does not run.

    `errors="replace"` rather than `strict`: a mangled em-dash in a log line is a cosmetic defect,
    an aborted 5-hour run is not.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:          # pytest's capture object, or a plain file wrapper
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):     # detached or already-written stream; nothing to do
            pass


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to stderr. Level from the LOG_LEVEL env var (default INFO)."""
    global _CONFIGURED
    if not _CONFIGURED:
        ensure_utf8_output()
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger(_ROOT)
        root.setLevel(level)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True

    # Callers pass either a short label ("prover.search") or `__name__`, which already begins with
    # the root package. Prefixing the latter produced
    # `prooflens_prover.prooflens_prover.prover.vllm_policy` on every line of every cluster log.
    if name.startswith(f"{_ROOT}."):
        name = name[len(_ROOT) + 1:]
    return logging.getLogger(f"{_ROOT}.{name}" if name != _ROOT else name)
