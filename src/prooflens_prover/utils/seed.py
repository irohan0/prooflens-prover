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

        # `torch.manual_seed` seeds the CPU generator *and* queues a seed for every CUDA device
        # through `torch.cuda.manual_seed_all`, so GPUs need nothing further here.
        #
        # It deliberately does not ask whether CUDA is available first. `torch.cuda.is_available()`
        # calls into the CUDA driver, and PyTorch answers by registering a `pthread_atfork` handler
        # that marks every subsequent forked child as unable to use CUDA — "poisoning" fork — even
        # though the call itself initializes nothing. vLLM v1 forks its `EngineCore` process, and
        # vLLM's own guard against this tests `torch.cuda.is_initialized()`, which is False here.
        # Fork-poisoned-but-not-initialized is the one state that guard cannot see, so the result
        # was
        #
        #     RuntimeError: Cannot re-initialize CUDA in forked subprocess
        #
        # raised inside EngineCore after a full 15 GB model load, caused by a seeding call three
        # hundred lines earlier in an unrelated file. See `prover.vllm_policy`, which now also
        # arranges for the engine not to fork at all.
        torch.manual_seed(seed)
        # Best-effort determinism. Some CUDA kernels remain nondeterministic; that is acceptable
        # for retrieval scoring and for sampling-based proof search, both of which record their
        # actual outputs per attempt rather than relying on bitwise replay. Both of these set a
        # context flag and touch no driver, so neither poisons fork.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
