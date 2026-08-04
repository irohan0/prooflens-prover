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


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to stderr. Level from the LOG_LEVEL env var (default INFO)."""
    global _CONFIGURED
    if not _CONFIGURED:
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

    return logging.getLogger(f"{_ROOT}.{name}" if name != _ROOT else name)
