"""The train and val splits must be exactly complementary for any train_ratio."""

from pathlib import Path
from typing import Literal

import pytest
from datasets import Dataset

from speculators.train.data import ArrowDataset


def _dataset(tmp_path: Path, n: int) -> str:
    ds = Dataset.from_dict(
        {
            "input_ids": [[i, i + 1, i + 2] for i in range(n)],
            "loss_mask": [[1, 1, 1]] * n,
            "seq_len": [3] * n,
        }
    )
    path = tmp_path / "data"
    ds.save_to_disk(str(path))
    (path / "hidden_states").mkdir()
    return str(path)


def _split(path: str, ratio: float, split: Literal["train", "val"]) -> ArrowDataset:
    return ArrowDataset(
        max_len=128,
        datapath=path,
        on_missing="skip",
        train_ratio=ratio,
        split=split,
    )


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
@pytest.mark.parametrize("n", [1000, 12501, 100000])
def test_splits_are_exactly_complementary(tmp_path, ratio, n):
    path = _dataset(tmp_path, n)
    train = _split(path, ratio, "train")
    val = _split(path, ratio, "val")

    assert len(train) + len(val) == n, "splits must partition the dataset"
    assert len(train) == int(n * ratio)
    assert len(train) > 0
    assert len(val) > 0
    train_files = {train._map_to_file_idx(i) for i in range(len(train))}
    val_files = {val._map_to_file_idx(i) for i in range(len(val))}
    assert not (train_files & val_files), "a file index is in both splits"
    assert train_files | val_files == set(range(n)), "every file index is in one split"


@pytest.mark.parametrize("ratio", [0.2, 0.9])
def test_no_row_appears_in_both_splits(tmp_path, ratio):
    n = 12501
    path = _dataset(tmp_path, n)
    train = _split(path, ratio, "train")
    val = _split(path, ratio, "val")

    train_ids = {tuple(train.data[i]["input_ids"]) for i in range(len(train))}
    val_ids = {tuple(val.data[i]["input_ids"]) for i in range(len(val))}
    assert not (train_ids & val_ids), "a row is in both train and val"


@pytest.mark.parametrize("ratio", [0.5, 0.9, 0.95])
def test_val_rows_are_spread_across_the_file_not_taken_from_its_tail(tmp_path, ratio):
    """A sharded preparation writes the file family by family, so a trailing block
    would hold out whole families. Held-out rows must come from every region."""
    n = 10_000
    path = _dataset(tmp_path, n)
    val = _split(path, ratio, "val")

    files = sorted(val._map_to_file_idx(i) for i in range(len(val)))
    stride = n / len(files)
    assert files[0] < stride, "first held-out row is within one stride of the start"
    assert files[-1] >= n - stride, "last held-out row is within one stride of the end"
    gaps = [b - a for a, b in zip(files, files[1:], strict=False)]
    assert max(gaps) <= stride + 1, (
        f"held-out rows bunch up: max gap {max(gaps)} vs stride {stride:.1f}"
    )


@pytest.mark.parametrize("split", ["train", "val"])
def test_file_indices_name_the_rows_actually_selected(tmp_path, split):
    """The fixture stores each row's original index as input_ids[0], so the mapping
    that names hidden-state files can be checked against the row it returns."""
    path = _dataset(tmp_path, 1000)
    ds = _split(path, 0.9, split)
    for i in range(len(ds)):
        assert ds.data[i]["input_ids"][0] == ds._map_to_file_idx(i)


def test_default_takes_whole_dataset(tmp_path):
    path = _dataset(tmp_path, 100)
    assert len(_split(path, 1.0, "train")) == 100


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
def test_rejects_out_of_range_ratio(tmp_path, ratio):
    path = _dataset(tmp_path, 100)
    with pytest.raises(ValueError, match="train_ratio must be in"):
        _split(path, ratio, "train")


def test_rejects_val_split_with_no_val_data(tmp_path):
    path = _dataset(tmp_path, 100)
    with pytest.raises(ValueError, match="leaves no validation split"):
        _split(path, 1.0, "val")


@pytest.mark.parametrize("ratio", [0.1, 0.5, 0.9])
def test_rejects_empty_train_from_small_dataset(tmp_path, ratio):
    path = _dataset(tmp_path, 1)
    with pytest.raises(ValueError, match="train split is empty"):
        _split(path, ratio, "train")


def test_small_dataset_both_splits_nonempty(tmp_path):
    path = _dataset(tmp_path, 2)
    train = _split(path, 0.5, "train")
    val = _split(path, 0.5, "val")
    assert len(train) == 1
    assert len(val) == 1
