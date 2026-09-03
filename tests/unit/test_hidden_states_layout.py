"""Hidden-state files are sharded 1,000 per directory; older flat files still read."""

from pathlib import Path

import torch
from safetensors.torch import save_file

from hs_connectors import (
    HS_SHARD_SIZE,
    FileTransfer,
    hidden_states_candidates,
    hidden_states_file,
    iter_hidden_state_indices,
)


def _write(path: Path, idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {"token_ids": torch.tensor([idx]), "hidden_states": torch.zeros(1, 2)}
    save_file(tensors, str(path))


def test_shard_directory_is_the_index_divided_by_the_shard_size():
    root = Path("/hs")
    assert hidden_states_file(root, 0) == root / "0000" / "hs_0.safetensors"
    last_in_first = HS_SHARD_SIZE - 1
    assert hidden_states_file(root, last_in_first) == (
        root / "0000" / f"hs_{last_in_first}.safetensors"
    )
    assert hidden_states_file(root, HS_SHARD_SIZE) == (
        root / "0001" / f"hs_{HS_SHARD_SIZE}.safetensors"
    )
    assert hidden_states_file(root, 113820) == root / "0113" / "hs_113820.safetensors"


def test_candidates_prefer_the_sharded_path_and_fall_back_to_flat():
    root = Path("/hs")
    sharded, flat = hidden_states_candidates(root, 42)
    assert sharded == root / "0000" / "hs_42.safetensors"
    assert flat == root / "hs_42.safetensors"


def test_cache_writes_into_the_shard_and_get_cached_reads_it_back(tmp_path):
    staged = tmp_path / "staged.safetensors"
    _write(staged, 1234)
    transfer = FileTransfer(tmp_path / "hs")

    transfer.cache(str(staged), 1234)

    assert (tmp_path / "hs" / "0001" / "hs_1234.safetensors").is_file()
    assert not staged.exists()
    loaded = transfer.get_cached(1234)
    assert loaded is not None
    assert loaded["token_ids"].tolist() == [1234]


def test_get_cached_still_reads_a_flat_file_from_before_sharding(tmp_path):
    root = tmp_path / "hs"
    _write(root / "hs_77.safetensors", 77)
    transfer = FileTransfer(root)

    loaded = transfer.get_cached(77)

    assert loaded is not None
    assert loaded["token_ids"].tolist() == [77]
    assert transfer.get_cached(78) is None


def test_iter_indices_covers_both_layouts_without_descending_further(tmp_path):
    root = tmp_path / "hs"
    _write(root / "hs_1.safetensors", 1)
    _write(root / "0000" / "hs_2.safetensors", 2)
    # Not a layout we write, so not scanned.
    _write(root / "0000" / "deeper" / "hs_3.safetensors", 3)
    (root / "0000" / "notes.txt").touch()

    assert sorted(iter_hidden_state_indices(root)) == [1, 2]
    assert list(iter_hidden_state_indices(tmp_path / "absent")) == []
