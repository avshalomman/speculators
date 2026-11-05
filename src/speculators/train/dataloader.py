from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hs_connectors import HiddenStatesTransfer
from speculators.train.data import (
    ArrowDataset,
    BaseDataset,
    CollateFn,
)
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.noise_transforms import AddUniformNoise

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]


@dataclass(frozen=True)
class DataSource:
    """A dataset trained alongside ``--data-path``, with its own hidden states."""

    name: str
    data_path: str
    transfer: HiddenStatesTransfer | None
    keep_window: int | None = None


class MultiSourceDataset(Dataset):
    """Several datasets behind one index, in order.

    The batch sampler shuffles and packs over the combined index, so rows of all
    sources are interleaved. Each row is read by its own dataset, which keeps its
    own hidden-states transfer, keep window and file-index mapping.
    """

    def __init__(self, datasets: list[BaseDataset]):
        self.datasets = datasets
        self.offsets = np.cumsum([0] + [len(d) for d in datasets])  # type: ignore[arg-type]
        self.approx_lengths = [n for d in datasets for n in d.approx_lengths]
        self.hidden_states_dtype = datasets[0].hidden_states_dtype

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int):
        source = int(np.searchsorted(self.offsets, index, side="right")) - 1
        return self.datasets[source][index - int(self.offsets[source])]


def _limit_worker_threads() -> None:
    """Limit per-worker thread pools to avoid thread exhaustion.

    With ``multiprocessing_context='spawn'``, each worker is a full process
    that re-imports numpy (OpenBLAS) and torch, each creating thread pools
    sized to the core count.  DataLoader workers only do I/O and tensor
    slicing — they don't benefit from intra-op parallelism.

    The env vars must be set before numpy/torch are imported to take effect
    on OpenBLAS/OMP.  Call this at the top of the training entry point,
    before DataLoader construction.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    torch.set_num_threads(1)


def _setup_dataloader(
    dataset: BaseDataset | MultiSourceDataset,
    total_seq_len: int,
    hidden_size: int,
    num_workers: int = 12,
    num_target_layers: int = 3,
    prefetch_factor: int | None = 4,
    preprocess: Callable[[BatchType], BatchType] | None = None,
    max_batches: int | None = None,
) -> DataLoader:
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
        max_batches=max_batches,
    )
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        pin_memory=True,
        collate_fn=CollateFn(
            total_seq_len,
            hidden_size,
            num_target_layers=num_target_layers,
            dtype=dataset.hidden_states_dtype,
            preprocess=preprocess,
        ),
        persistent_workers=use_workers,
        multiprocessing_context="spawn" if use_workers else None,
        worker_init_fn=_worker_init_fn if use_workers else None,
    )


def create_train_val_loaders(
    *,
    data_path: str,
    total_seq_len: int,
    hidden_states_dtype: torch.dtype,
    noise_std: float,
    transfer: HiddenStatesTransfer | None = None,
    vllm_endpoint: str,
    on_missing: Literal["generate", "skip", "warn", "raise"],
    on_generate: Literal["cache", "delete"],
    verifier_name_or_path: str,
    request_timeout: float | None,
    max_retries: int,
    generation_validation_retries: int,
    max_consecutive_generation_failures: int,
    hidden_size: int,
    num_target_layers: int,
    num_workers: int,
    prefetch_factor: int,
    preprocess: Callable[[BatchType], BatchType] | None,
    train_data_ratio: float = 0.9,
    max_train_batches: int | None = None,
    keep_window: int | None = None,
    extra_sources: Sequence[DataSource] = (),
) -> tuple[DataLoader, DataLoader, dict[str, DataLoader]]:
    """Create training and validation DataLoaders.

    Non-data SP ranks get lightweight loaders with no workers (they receive
    batches via scatter).  Reads DP/SP topology from
    :mod:`speculators.train.distributed`.

    ``extra_sources`` are split like ``data_path`` (each by ``train_data_ratio``);
    their training rows join one shuffled training set, and each gets its own
    validation loader, returned by source name.
    """
    _limit_worker_threads()
    noise_transform = AddUniformNoise(std=noise_std)

    if not (0.0 < train_data_ratio < 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1), got {train_data_ratio}")

    def make_dataset(source: DataSource, split: Literal["train", "val"]) -> BaseDataset:
        return ArrowDataset(
            datapath=source.data_path,
            max_len=total_seq_len,
            transfer=source.transfer,
            vllm_endpoint=vllm_endpoint,
            on_missing=on_missing,
            on_generate=on_generate,
            transform=noise_transform if split == "train" else None,
            train_ratio=train_data_ratio,
            split=split,
            model=verifier_name_or_path,
            hidden_states_dtype=hidden_states_dtype,
            request_timeout=request_timeout,
            max_retries=max_retries,
            generation_validation_retries=generation_validation_retries,
            max_consecutive_generation_failures=max_consecutive_generation_failures,
            keep_window=source.keep_window,
        )

    def make_loader(
        dataset: BaseDataset | MultiSourceDataset, max_batches: int | None = None
    ) -> DataLoader:
        return _setup_dataloader(
            dataset,
            total_seq_len,
            hidden_size,
            num_target_layers=num_target_layers,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            preprocess=preprocess,
            max_batches=max_batches,
        )

    primary = DataSource("primary", data_path, transfer, keep_window)
    train_dataset: BaseDataset | MultiSourceDataset = make_dataset(primary, "train")
    if extra_sources:
        train_dataset = MultiSourceDataset(
            [train_dataset] + [make_dataset(s, "train") for s in extra_sources]  # type: ignore[list-item]
        )
        logger.info(
            "Training on %s and %s, shuffled together",
            data_path,
            ", ".join(f"{s.name}={s.data_path}" for s in extra_sources),
        )

    train_loader = make_loader(train_dataset, max_batches=max_train_batches)
    val_loader = make_loader(make_dataset(primary, "val"))
    extra_val_loaders = {
        s.name: make_loader(make_dataset(s, "val")) for s in extra_sources
    }
    return train_loader, val_loader, extra_val_loaders
