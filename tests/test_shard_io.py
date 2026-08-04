"""Regression tests for the shard writer.

`np.savez` appends `.npz` to any filename lacking it. A temp path of `shard.npz.tmp` therefore
lands on disk as `shard.npz.tmp.npz`, and the atomic rename fails with `FileNotFoundError` — after
the GPU has already done the encoding. That cost a cluster job (18147716) that was otherwise
working perfectly, so it gets a test rather than a comment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# `scripts/` is not a package; add it to the path so the writer can be tested where it lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_dense_index import save_shard_atomically  # noqa: E402


def test_writes_exactly_the_requested_filename(tmp_path):
    path = tmp_path / "shard_00000.npz"
    save_shard_atomically(path, np.zeros((4, 8), dtype=np.float16), np.array([2, 2]))
    assert path.exists(), "the shard must land at the requested path"


def test_leaves_no_stray_temp_files(tmp_path):
    # The original bug left `shard_00000.npz.tmp.npz` behind and no shard at all.
    path = tmp_path / "shard_00000.npz"
    save_shard_atomically(path, np.zeros((4, 8), dtype=np.float16), np.array([2, 2]))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shard_00000.npz"]


def test_roundtrips_content_and_dtype(tmp_path):
    path = tmp_path / "shard_00007.npz"
    tokens = np.random.default_rng(0).standard_normal((11, 8)).astype(np.float16)
    lengths = np.array([3, 4, 4], dtype=np.int64)
    save_shard_atomically(path, tokens, lengths)
    blob = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(blob["tokens"], tokens)
    np.testing.assert_array_equal(blob["lengths"], lengths)
    assert blob["tokens"].dtype == np.float16, "fp16 storage must survive the shard roundtrip"


def test_overwrites_an_existing_shard(tmp_path):
    path = tmp_path / "shard_00000.npz"
    save_shard_atomically(path, np.ones((2, 8), dtype=np.float16), np.array([2]))
    save_shard_atomically(path, np.zeros((2, 8), dtype=np.float16), np.array([2]))
    assert np.load(path, allow_pickle=False)["tokens"].sum() == 0.0


def test_temp_name_would_not_be_mistaken_for_a_finished_shard(tmp_path):
    # The resume check tests for `shard_NNNNN.npz` exactly; the temp name must not collide with it.
    path = tmp_path / "shard_00003.npz"
    tmp_name = f"{path.stem}.tmp.npz"
    assert tmp_name != path.name
    assert not tmp_name.startswith(path.name)


@pytest.mark.parametrize("stem", ["shard_00000", "shard_12345", "shard_99999"])
def test_works_for_any_shard_index(tmp_path, stem):
    path = tmp_path / f"{stem}.npz"
    save_shard_atomically(path, np.zeros((1, 8), dtype=np.float16), np.array([1]))
    assert path.exists()
