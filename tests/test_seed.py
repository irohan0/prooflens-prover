"""Seeding — and the one call it must never make.

`set_global_seed` looks like the least interesting file in this project. It cost a cluster
round-trip and a full 15 GB model load, because `torch.cuda.is_available()` does something its name
does not
suggest: it calls into the CUDA driver, and PyTorch responds by registering a `pthread_atfork`
handler that marks every subsequent forked child as unable to use CUDA. It initializes nothing, so
`torch.cuda.is_initialized()` stays False — and that is the precise state vLLM's own guard against
this cannot detect. vLLM forked its engine, and the child raised

    RuntimeError: Cannot re-initialize CUDA in forked subprocess

from a seeding call three hundred lines earlier in an unrelated file.

The fix is invisible in a diff (`torch.manual_seed` already seeds every CUDA device lazily) and
invisible in every existing test, because the seeds it sets are all still correct. So the invariant
is pinned behaviourally here, with a fake torch that fails the test if the call is made at all.

Hermetic: no torch required, installed or otherwise.
"""

from __future__ import annotations

import sys
import types

import pytest

from prooflens_prover.utils.seed import set_global_seed


class _Cuda:
    """A `torch.cuda` whose driver-touching entry points are tripwires."""

    def __init__(self) -> None:
        self.forbidden_calls: list[str] = []
        self.manual_seed_all_calls: list[int] = []

    def is_available(self) -> bool:
        self.forbidden_calls.append("is_available")
        return True

    def device_count(self) -> int:
        self.forbidden_calls.append("device_count")
        return 1

    def manual_seed_all(self, seed: int) -> None:
        # Not forbidden — this one is lazy in real torch, queueing the seed until CUDA initializes.
        self.manual_seed_all_calls.append(seed)


@pytest.fixture
def fake_torch(monkeypatch):
    cuda = _Cuda()
    cudnn = types.SimpleNamespace(deterministic=False, benchmark=True)

    module = types.ModuleType("torch")
    module.cuda = cuda
    module.backends = types.SimpleNamespace(cudnn=cudnn)
    module.manual_seed_calls = []
    module.manual_seed = module.manual_seed_calls.append

    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_it_never_asks_whether_cuda_is_available(fake_torch):
    """The single line that broke the LLM arm.

    Any call here poisons `fork` for the rest of the process. Nothing in this project needs the
    answer: `torch.manual_seed` seeds the CPU generator and queues a seed for every CUDA device
    through `manual_seed_all` regardless of whether one exists.
    """
    set_global_seed(7)

    assert fake_torch.cuda.forbidden_calls == [], (
        "set_global_seed touched the CUDA driver; a forked child (vLLM's EngineCore) will now fail "
        f"with 'Cannot re-initialize CUDA in forked subprocess'. Calls: "
        f"{fake_torch.cuda.forbidden_calls}"
    )


def test_it_still_seeds_torch(fake_torch):
    """The fork fix must not have quietly removed the determinism it exists to provide."""
    set_global_seed(7)

    assert fake_torch.manual_seed_calls == [7]


def test_it_still_sets_the_determinism_flags(fake_torch):
    """Both set a context flag and touch no driver, so both are safe to keep."""
    set_global_seed(7)

    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cudnn.benchmark is False


def test_it_seeds_python_and_the_hash_seed(fake_torch):
    import os
    import random

    set_global_seed(1234)
    first = random.random()

    set_global_seed(1234)
    assert random.random() == first
    assert os.environ["PYTHONHASHSEED"] == "1234"


def test_it_works_with_no_torch_at_all(monkeypatch):
    """The light environment (Lean smoke tests, metrics, formatting) has neither torch nor numpy."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def no_heavy(name, *args, **kwargs):
        if name in {"torch", "numpy"}:
            raise ImportError(f"no {name} here")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr("builtins.__import__", no_heavy)

    set_global_seed(3)      # must not raise
