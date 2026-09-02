import warnings
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from speculators.train.data_types import DatasetRecord
from speculators.train.hidden_states_source import (
    HiddenStateSample,
    HiddenStatesSource,
)
from speculators.train.recovery import (
    RECOVERY_METADATA_KEY,
    RecoveryMetadata,
    SampleUnavailable,
)

BatchType = dict[str, Any]


def create_empty_batch(
    seq_len: int, hidden_size: int, num_target_layers: int, dtype: torch.dtype
):
    # data structure: {
    #     "hidden_states": [seq_len, num_target_layers * hidden_size],
    #     "input_ids": [seq_len],
    #     "verifier_last_hidden_states": [seq_len, hidden_size],
    #     "loss_mask": [seq_len],
    #     "position_ids": [seq_len],
    #     "document_ids": [seq_len],
    # }

    return {
        "hidden_states": torch.zeros(
            1, seq_len, num_target_layers * hidden_size, dtype=dtype
        ),
        "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "verifier_last_hidden_states": torch.zeros(
            1, seq_len, hidden_size, dtype=dtype
        ),
        "loss_mask": torch.zeros(1, seq_len, dtype=torch.bool),
        "position_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "document_ids": -torch.ones(1, seq_len, dtype=torch.long),
    }


def _strided_val_indices(n: int, train_ratio: float) -> list[int]:
    """Storage indices held out for validation: ``n - int(n * train_ratio)`` of
    them, evenly spaced over ``[0, n)``, so ``len(train) == int(n * train_ratio)``
    as with a contiguous split. Spacing is at least one row, so the indices are
    distinct and ascending."""
    n_val = n - int(n * train_ratio)
    if n_val <= 0:
        return []
    return [int((k + 0.5) * n / n_val) for k in range(n_val)]


class ArrowDataset(Dataset):
    """Load preprocessed Arrow rows and expose them as dataset records."""

    def __init__(
        self,
        datapath: str | PathLike,
        train_ratio: float = 1.0,
        split: Literal["train", "val"] = "train",
    ):
        self.datapath = Path(datapath)
        self.data = load_from_disk(self.datapath)
        if not 0.0 < train_ratio <= 1.0:
            raise ValueError(f"train_ratio must be in (0.0, 1.0], got {train_ratio}")
        if split == "val" and train_ratio == 1.0:
            raise ValueError("train_ratio=1.0 leaves no validation split")

        # Validation rows are held out at an even stride, not as a trailing block.
        # A sharded preparation concatenates its parts family by family, so the
        # last rows of the file are the last few dataset families, not a sample.
        # Both splits derive from one index list, so they are exactly
        # complementary, and storage indices (which name the hidden states on
        # disk) are untouched.
        n = len(self.data)
        val_indices = _strided_val_indices(n, train_ratio)
        if split == "val":
            indices = val_indices
        else:
            held_out = set(val_indices)
            indices = [i for i in range(n) if i not in held_out]
        if not indices:
            raise ValueError(
                f"{split} split is empty (dataset has {n} rows, "
                f"train_ratio={train_ratio} holds out {len(val_indices)})"
            )
        # None when nothing is held out: the mapping is then the identity.
        self.storage_indices: list[int] | None = None if len(indices) == n else indices
        self.data = self.data.select(indices)
        self.approx_lengths = list(self.data.with_format(None)["seq_len"])

    def _map_to_storage_index(self, index: int) -> int:
        if self.storage_indices is None:
            return index
        return self.storage_indices[index]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> DatasetRecord:
        return DatasetRecord.from_arrow_row(
            self.data[index],
            storage_index=self._map_to_storage_index(index),
        )

    def __getitems__(self, indices: Sequence[int]) -> list[DatasetRecord]:
        """Load records in the same order as the requested dataset indices."""
        return [self[index] for index in indices]


class SpeculatorsDataset(Dataset):
    """Combine Arrow records with hidden states for training."""

    def __init__(
        self,
        max_len: int,
        records: ArrowDataset,
        hidden_states: HiddenStatesSource,
        hidden_size: int,
        hidden_states_dtype=torch.bfloat16,
        num_target_layers: int = 3,
        preprocess: Callable[[BatchType], BatchType] | None = None,
    ):
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_target_layers = num_target_layers
        self.preprocess = preprocess
        self.hidden_states_dtype = hidden_states_dtype
        self.records = records
        self.hidden_states = hidden_states
        self.approx_lengths = records.approx_lengths

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> BatchType | SampleUnavailable:
        record = self.records[index]
        loaded_hs = self.hidden_states.get_one(record)

        return self._build_sample(index, record, loaded_hs)

    def _build_sample(
        self,
        index: int,
        record: DatasetRecord,
        loaded_hs: HiddenStateSample | SampleUnavailable,
    ) -> BatchType | SampleUnavailable:
        """Combine one record with its loaded hidden states."""

        if isinstance(loaded_hs, SampleUnavailable):
            return loaded_hs

        # loaded_hs structure: {
        #   "hidden_states": [seq_len, num_layers, hidden_size]
        #   "token_ids": [seq_len]
        # }

        if not torch.equal(loaded_hs["token_ids"], record.input_ids):
            warnings.warn(
                f"Loaded token ids {loaded_hs['token_ids']} for index {index} don't"
                f"match input ids {record.input_ids}",
                stacklevel=1,
            )
            return SampleUnavailable(
                reason=f"Extracted token ids do not match sample {index}"
            )

        # The payload carries one entry per captured layer, ordered by layer id,
        # and its trailing entry is the verifier's final hidden state -- the input
        # of `verifier_lm_head`, i.e. the distilling target.
        #
        # The capture list normally ends one layer past the draft's target layers
        # (launch_vllm.py appends the final layer unless it is already listed), so
        # the payload is one entry wider than `fc` and the trailing entry is
        # split off. When the capture list already ends at the verifier's final
        # layer, the launcher appends nothing and the payload is exactly as wide
        # as `fc`: every entry feeds `fc` and the trailing one doubles as the
        # verifier's final hidden state. Any other width keeps the original
        # behaviour of dropping the trailing entry, so no configuration that ran
        # before this change can start failing.
        num_payload_layers = loaded_hs["hidden_states"].shape[1]
        if num_payload_layers == self.num_target_layers:
            hs_features = loaded_hs["hidden_states"]
        else:
            hs_features = loaded_hs["hidden_states"][:, :-1]

        seq_len = loaded_hs["token_ids"].shape[0]

        sample = {
            "hidden_states": hs_features.flatten(1),  # [seq_len, n_feat * h]
            "input_ids": loaded_hs["token_ids"],  # [seq_len]
            "verifier_last_hidden_states": loaded_hs["hidden_states"][
                :, -1
            ],  # [seq_len, hidden_size]
            "loss_mask": record.loss_mask,  # [seq_len]
            "lengths": torch.tensor([seq_len], dtype=torch.long),
            "position_ids": torch.arange(seq_len, dtype=torch.long),
        }

        return self.preprocess(sample) if self.preprocess else sample

    def __getitems__(self, indices: list[int]) -> BatchType:
        """Fetch and collate a DataLoader batch into its final representation."""
        records = self.records.__getitems__(indices)
        loaded_hs = self.hidden_states.get_batch(records)
        samples = [
            self._build_sample(index, record, hidden_states)
            for index, record, hidden_states in zip(
                indices, records, loaded_hs, strict=True
            )
        ]

        unavailable: list[SampleUnavailable] = []
        valid_samples: list[BatchType] = []
        for item in samples:
            if isinstance(item, SampleUnavailable):
                unavailable.append(item)
            else:
                valid_samples.append(item)

        if valid_samples:
            batch = _pad_samples(valid_samples, self.max_len, self.hidden_states_dtype)
            _add_document_ids(batch, self.max_len)
            locally_empty = False
        else:
            # Create empty batch which then gets padded to full batch size if no
            # valid samples are found. Match the configured dtype so the placeholder
            # doesn't crash downstream layers loaded at a different precision.
            batch = create_empty_batch(
                self.max_len,
                self.hidden_size,
                self.num_target_layers,
                dtype=self.hidden_states_dtype,
            )
            locally_empty = True

        batch["error_records"] = len(unavailable)
        metadata = RecoveryMetadata.from_unavailable(
            unavailable,
            locally_empty=locally_empty,
        )
        if metadata.failure_count or metadata.locally_empty:
            batch[RECOVERY_METADATA_KEY] = metadata

        return batch

    def close(self) -> None:
        """Close resources owned by the hidden-state source."""
        self.hidden_states.close()


def _pad_samples(batch: list[BatchType], max_len: int, dtype: torch.dtype) -> BatchType:
    """Copy samples into one fixed-length, padded batch."""
    collated_data: BatchType = {}
    for key in batch[0]:
        if key == "lengths":
            collated_data[key] = torch.cat([sample[key] for sample in batch], dim=0)
            continue

        # One copy per sample: preallocated buffer, hidden states cast during write.
        first = batch[0][key]
        buffer_dtype = dtype if "hidden_states" in key else first.dtype
        out = torch.zeros(
            (max_len, *first.shape[1:]), dtype=buffer_dtype, device=first.device
        )
        offset = 0
        for sample in batch:
            tensor = sample[key]
            num_rows = min(tensor.shape[0], max_len - offset)
            out[offset : offset + num_rows] = tensor[:num_rows]
            offset += num_rows
            if offset == max_len:
                break
        collated_data[key] = out.unsqueeze(0)
    return collated_data


def _add_document_ids(collated_data: BatchType, max_len: int) -> None:
    """Truncate document lengths and add document IDs for packed attention."""
    # Include lengths until they fit in max_len. The last included length is
    # truncated if necessary; any additional lengths are discarded.
    lengths = collated_data.pop("lengths")
    new_lengths = []
    cum_length = 0
    for length in lengths:
        if length + cum_length >= max_len:
            new_lengths.append(max_len - cum_length)
            break
        new_lengths.append(length)
        cum_length += length
    lengths = torch.tensor(new_lengths, dtype=torch.long)

    # Each position maps to its document, with -1 for padding.
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], dtype=torch.long), lengths
    )
    document_ids = torch.cat(
        [
            document_ids,
            -1 * torch.ones(max_len - document_ids.shape[0], dtype=torch.long),
        ]
    ).unsqueeze(0)
    collated_data["document_ids"] = document_ids
