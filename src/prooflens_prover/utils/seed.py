"""Global determinism: seed Python / NumPy / Torch and set deterministic flags where feasible.

Ported from `prooflens/src/prooflens/utils/seed.py`, unchanged in behaviour. The seed lands in
every run manifest (`utils.manifest`), so any reported number can be traced to the RNG state that
produced it.

Heavy libraries are seeded only if importable, so this module stays usable in the light
environment (Lean smoke tests, metrics, formatting) that has neither NumPy nor Torch installed.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed all RNGs we might touch. Idempotent; safe to call before heavy imports exist."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort determinism. Some CUDA kernels remain nondeterministic; that is acceptable
        # for retrieval scoring and for sampling-based proof search, both of which record their
        # actual outputs per attempt rather than relying on bitwise replay.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
