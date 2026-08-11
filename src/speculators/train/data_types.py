"""Shared data structures for the training data pipeline."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DatasetRecord:
    """A preprocessed dataset row together with its storage identity."""

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    seq_len: int
    storage_index: int
    messages: list[dict[str, Any]] | None = None

    @classmethod
    def from_arrow_row(
        cls, row: Mapping[str, Any], *, storage_index: int
    ) -> "DatasetRecord":
        """Create a record from one formatted Arrow/Hugging Face row.

        ``messages`` is stored as a JSON string so every preprocessing shard
        infers the same Arrow schema; datasets prepared before that hold
        structured rows, and both are accepted.
        """
        messages = row.get("messages")
        if isinstance(messages, str):
            messages = json.loads(messages)
        return cls(
            input_ids=row["input_ids"],
            loss_mask=row["loss_mask"],
            seq_len=int(row["seq_len"]),
            storage_index=storage_index,
            messages=messages,
        )


@dataclass(frozen=True)
class GenerationReceipt:
    """Handle and transfer parameters returned by one vLLM request."""

    handle: str
    transfer_params: dict[str, Any]
