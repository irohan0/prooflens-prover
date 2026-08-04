"""Run manifests — the provenance header every run writes before it does any work.

**Why this exists.** The dissertation's core claim is a comparison between two retrieval arms. That
comparison is only defensible if every number can be traced to the exact code, config, seed, data
and hardware that produced it. The predecessor project enforced this and it repeatedly paid off:
its ReProver calibration check caught a *data leak in a published checkpoint* precisely because the
run headers made "which split was this trained on?" answerable after the fact.

A manifest is written at run **start**, not end, so a job killed by SLURM still leaves behind a
record of what it was attempting. `finalize()` appends the outcome.

Layout::

    results/logs/<run_id>/
        manifest.json      <- config + seed + git SHA + env + hardware, written at t=0
        attempts.jsonl     <- one record per proof attempt (the audit trail)
        run.log            <- stderr capture

`run_id` is `<name>_<UTC timestamp>_<short git sha>` — sorts chronologically, and collisions across
concurrent SLURM array tasks are impossible because the timestamp includes microseconds.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prooflens_prover.utils.logging import get_logger

log = get_logger("manifest")

_UNKNOWN = "unknown"


def git_commit(repo_root: Path | None = None) -> str:
    """Short git SHA of the working tree, with a `-dirty` suffix if there are uncommitted changes.

    Returns ``"unknown"`` rather than raising when git is unavailable or this is not a repo — a
    missing SHA must not be able to kill a 12-hour cluster job. The `-dirty` marker is load-bearing:
    a result produced from uncommitted code is not reproducible, and the manifest should say so
    rather than quietly record a commit that does not describe the code that ran.
    """
    root = str(repo_root or Path(__file__).resolve().parents[3])
    try:
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.SubprocessError, OSError):
        return _UNKNOWN


def _lean_version() -> str:
    """`lean --version`, or "unknown". Recorded because Lean/Mathlib version drift is the single
    most likely cause of a result failing to reproduce across machines — it is exactly what broke
    the predecessor project's prover harness (a lean-dojo/Lean-4.20 mismatch)."""
    try:
        out = subprocess.run(
            ["lean", "--version"], capture_output=True, text=True, timeout=30, check=True
        ).stdout.strip()
        return out
    except (subprocess.SubprocessError, OSError):
        return _UNKNOWN


def _gpu_info() -> list[dict[str, Any]]:
    """Per-device CUDA info, or [] on CPU-only hosts. Imports torch lazily."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [
        {
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
            "capability": ".".join(str(c) for c in torch.cuda.get_device_capability(i)),
        }
        for i in range(torch.cuda.device_count())
    ]


def _package_versions() -> dict[str, str]:
    """Versions of the packages whose behaviour can change a result. Absent packages are omitted
    rather than recorded as null, so the manifest reflects what was actually installed."""
    from importlib.metadata import PackageNotFoundError, version

    names = [
        "torch", "transformers", "peft", "trl", "accelerate", "datasets",
        "vllm", "pylate", "lean-interact", "numpy", "bm25s",
    ]
    out: dict[str, str] = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            continue
    return out


@dataclass
class RunManifest:
    """Provenance for one run. Construct via `RunManifest.create`, which writes it to disk."""

    run_id: str
    name: str
    seed: int
    config: dict[str, Any]
    git_commit: str
    started_utc: str
    run_dir: Path
    environment: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        name: str,
        config: dict[str, Any],
        seed: int,
        results_root: Path | str = "results/logs",
        capture_lean: bool = False,
    ) -> RunManifest:
        """Build the run directory and write `manifest.json` immediately.

        `capture_lean` shells out to `lean --version` (~1s); on for prover runs, off for pure
        training runs where Lean is irrelevant and the subprocess would just be noise.
        """
        sha = git_commit()
        now = datetime.now(UTC)
        run_id = f"{name}_{now:%Y%m%dT%H%M%S%f}_{sha}"
        run_dir = Path(results_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        env: dict[str, Any] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "packages": _package_versions(),
            "gpus": _gpu_info(),
            # SLURM context: makes a result traceable back to `seff <jobid>` for the real
            # resource accounting, which the write-up needs for the cost/latency table.
            "slurm": {
                k: v for k, v in os.environ.items()
                if k.startswith("SLURM_") and k in {
                    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_ARRAY_TASK_ID",
                    "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_CPUS_ON_NODE",
                }
            },
        }
        if capture_lean:
            env["lean"] = _lean_version()

        m = cls(
            run_id=run_id,
            name=name,
            seed=seed,
            config=config,
            git_commit=sha,
            started_utc=now.isoformat(),
            run_dir=run_dir,
            environment=env,
        )
        m.write()
        if sha.endswith("-dirty"):
            log.warning(
                "run %s started from a DIRTY working tree — this result is not reproducible from "
                "a commit. Commit before any run whose numbers will be reported.", run_id
            )
        log.info("run %s -> %s", run_id, run_dir)
        return m

    @classmethod
    def load(cls, run_dir: Path | str) -> RunManifest:
        """Reopen an existing run directory, for resuming an interrupted job.

        The original `run_id`, `git_commit` and `started_utc` are preserved: a resumed run is the
        same run, and rewriting its provenance to the restart time would misattribute the results
        to whatever the working tree happens to look like now.
        """
        d = Path(run_dir)
        payload = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            run_id=payload["run_id"],
            name=payload["name"],
            seed=payload["seed"],
            config=payload["config"],
            git_commit=payload["git_commit"],
            started_utc=payload["started_utc"],
            run_dir=d,
            environment=payload.get("environment", {}),
            outcome=payload.get("outcome"),
        )

    def write(self) -> Path:
        path = self.run_dir / "manifest.json"
        payload = {
            "run_id": self.run_id,
            "name": self.name,
            "seed": self.seed,
            "git_commit": self.git_commit,
            "started_utc": self.started_utc,
            "environment": self.environment,
            "config": self.config,
            "outcome": self.outcome,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def finalize(self, **outcome: Any) -> Path:
        """Record how the run ended (counts, metrics, wall time) and rewrite the manifest."""
        self.outcome = {
            "finished_utc": datetime.now(UTC).isoformat(),
            **outcome,
        }
        return self.write()

    @property
    def attempts_path(self) -> Path:
        """Per-attempt JSONL — the audit trail every reported number must be recomputable from."""
        return self.run_dir / "attempts.jsonl"
