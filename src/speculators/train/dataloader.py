from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from hs_connectors import FileTransfer, HiddenStatesTransfer
from speculators.data_generation.vllm_client import validate_vllm_model
from speculators.train.data import (
    ArrowDataset,
    MultiSourceDataset,
    SpeculatorsDataset,
)
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.hidden_states_source import (
    HiddenStatesSource,
    OfflineHiddenStatesSource,
    VllmHiddenStatesSource,
)

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]


@dataclass(frozen=True)
class DataSource:
    """A dataset trained alongside ``--data-path``, with its own hidden states."""

    name: str
    data_path: str
    transfer: HiddenStatesTransfer | None
    keep_window: int | None = None


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

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.accelerator.is_available():
        torch.accelerator.set_device_index(local_rank)


def identity_fn(x):
    return x


def _setup_dataloader(
    dataset: SpeculatorsDataset,
    total_seq_len: int,
    num_workers: int = 12,
    prefetch_factor: int | None = 4,
    max_batches: int | None = None,
    seed: int = 0,
) -> DataLoader:
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
        seed=seed,
        max_batches=max_batches,
    )
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        pin_memory=True,
        collate_fn=identity_fn,
        persistent_workers=use_workers,
        multiprocessing_context="spawn" if use_workers else None,
        worker_init_fn=_worker_init_fn if use_workers else None,
    )


def create_train_val_loaders(
    *,
    data_path: str,
    total_seq_len: int,
    hidden_states_dtype: torch.dtype,
    transfer: HiddenStatesTransfer | None = None,
    vllm_endpoint: str,
    on_missing: Literal["generate", "skip", "warn", "raise"],
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
    seed: int = 0,
    max_transfer_workers: int = 1,
    keep_window: int | None = None,
    extra_sources: Sequence[DataSource] = (),
) -> tuple[DataLoader, DataLoader, dict[str, DataLoader]]:
    """Create training and validation DataLoaders.

    Non-data SP ranks get lightweight loaders with no workers (they receive
    batches via scatter).  Reads DP/SP topology from
    :mod:`speculators.train.distributed`.

    ``seed`` sets the training packing order and must be the same on every
    rank. Validation keeps a fixed order, so runs with different seeds are
    validated on the same packed batches.

    ``extra_sources`` are split like ``data_path`` (each by ``train_data_ratio``);
    their training rows join one shuffled training set, and each gets its own
    validation loader, returned by source name.
    """
    _limit_worker_threads()

    if not (0.0 < train_data_ratio < 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1), got {train_data_ratio}")

    primary = DataSource("primary", data_path, transfer, keep_window)

    def make_records(source: DataSource, split: Literal["train", "val"]):
        return ArrowDataset(
            datapath=source.data_path,
            train_ratio=train_data_ratio,
            split=split,
            keep_window=source.keep_window,
        )

    train_records = make_records(primary, "train")
    val_records = make_records(primary, "val")

    # ``on_missing=generate`` is retained as the backwards-compatible switch
    # for the vLLM source. The other values select offline loading and define
    # what happens when a cached sample is absent.
    source_transfer = transfer or FileTransfer(train_records.datapath / "hidden_states")
    hidden_states_source: HiddenStatesSource
    if on_missing == "generate":
        if extra_sources:
            raise ValueError("--extra-data needs offline hidden states")
        validate_vllm_model(vllm_endpoint, verifier_name_or_path)
        hidden_states_source = VllmHiddenStatesSource(
            transfer=source_transfer,
            vllm_endpoint=vllm_endpoint,
            model=verifier_name_or_path,
            request_timeout=request_timeout,
            max_retries=max_retries,
            generation_validation_retries=generation_validation_retries,
            max_consecutive_generation_failures=max_consecutive_generation_failures,
            max_transfer_workers=max_transfer_workers,
            keep_window=keep_window,
        )
    else:
        hidden_states_source = OfflineHiddenStatesSource(
            transfer=source_transfer,
            on_missing=on_missing,
        )

    def make_dataset(records: ArrowDataset, source: HiddenStatesSource):
        return SpeculatorsDataset(
            records=records,
            max_len=total_seq_len,
            hidden_states=source,
            hidden_size=hidden_size,
            hidden_states_dtype=hidden_states_dtype,
            num_target_layers=num_target_layers,
            preprocess=preprocess,
        )

    def offline_source(source: DataSource) -> HiddenStatesSource:
        records_path = make_records(source, "val").datapath
        return OfflineHiddenStatesSource(
            transfer=source.transfer or FileTransfer(records_path / "hidden_states"),
            on_missing=on_missing,  # type: ignore[arg-type]
        )

    train_dataset: SpeculatorsDataset = make_dataset(
        train_records, hidden_states_source
    )
    val_dataset = make_dataset(val_records, hidden_states_source)
    extra_val_datasets: dict[str, SpeculatorsDataset] = {}
    if extra_sources:
        extra_train = []
        for source in extra_sources:
            extra_hs = offline_source(source)
            extra_train.append(make_dataset(make_records(source, "train"), extra_hs))
            extra_val_datasets[source.name] = make_dataset(
                make_records(source, "val"), extra_hs
            )
        train_dataset = MultiSourceDataset([train_dataset, *extra_train])
        logger.info(
            "Training on %s and %s, shuffled together",
            data_path,
            ", ".join(f"{s.name}={s.data_path}" for s in extra_sources),
        )

    train_loader = _setup_dataloader(
        train_dataset,
        total_seq_len,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        max_batches=max_train_batches,
        seed=seed,
    )
    val_loader = _setup_dataloader(
        val_dataset,
        total_seq_len,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    extra_val_loaders = {
        name: _setup_dataloader(
            dataset,
            total_seq_len,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        for name, dataset in extra_val_datasets.items()
    }

    return train_loader, val_loader, extra_val_loaders
