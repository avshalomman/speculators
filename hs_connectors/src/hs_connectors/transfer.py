"""Abstraction for hidden-states transfer between vLLM and the trainer."""

from __future__ import annotations

import dataclasses
import fcntl
import os
import shutil
import socket
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from safetensors.torch import load_file

from hs_connectors.mooncake_store import MooncakeHiddenStatesStore, MooncakeStoreConfig

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterator


# Hidden-state files live in sub-directories of HS_SHARD_SIZE files each. An
# object-store FUSE mount re-lists a directory whenever its directory cache
# expires, one page of 1,000 keys per round trip, so every stat in a flat
# directory costs O(N) in its size: generation into one such directory slowed
# 43x over eleven hours as it filled to 113k files. Bounding directory size
# bounds that cost for the writer and for the trainer, which stats one file per
# sample. Files written before this layout existed sit flat at the root; they
# are still found and still read.
HS_SHARD_SIZE = 1000


def hidden_states_file(root: Path, file_idx: int) -> Path:
    """Where sample ``file_idx`` is written.

    ``<root>/<idx // 1000>/hs_<idx>.safetensors``, zero-padded to four digits.
    """
    return root / f"{file_idx // HS_SHARD_SIZE:04d}" / f"hs_{file_idx}.safetensors"


def hidden_states_candidates(root: Path, file_idx: int) -> tuple[Path, Path]:
    """Where sample ``file_idx`` may be read from: sharded first, then the flat
    path a run from before sharding used."""
    return hidden_states_file(root, file_idx), root / f"hs_{file_idx}.safetensors"


def _index_of(name: str) -> int | None:
    if not (name.startswith("hs_") and name.endswith(".safetensors")):
        return None
    try:
        return int(name[3:-12])
    except ValueError:
        return None


def iter_hidden_state_indices(root: Path) -> Iterator[int]:
    """Every sample index with a file under ``root``, flat or one directory down.

    One listing per directory and no stat per file: on a FUSE mount a listing is
    one request per 1,000 entries, and a stat can be a listing of its own.
    """
    if not root.is_dir():
        return
    with os.scandir(root) as top:
        for entry in top:
            if entry.is_file(follow_symlinks=False):
                if (idx := _index_of(entry.name)) is not None:
                    yield idx
            elif entry.is_dir(follow_symlinks=False):
                with os.scandir(entry.path) as shard:
                    for inner in shard:
                        if not inner.is_file(follow_symlinks=False):
                            continue
                        if (idx := _index_of(inner.name)) is not None:
                            yield idx


def wait_for_lock(lock_path: str, timeout: float = 10.0, poll_interval: float = 0.1):
    fd = os.open(lock_path, os.O_RDWR)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for lock: {lock_path}"
                    ) from None
                time.sleep(poll_interval)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


class HiddenStatesTransfer(ABC):
    """Interface for reading hidden states produced by vLLM."""

    def setup(self) -> None:  # noqa: B027
        """Lazy initialization (safe to call from dataloader worker)."""

    @abstractmethod
    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        """Return a previously cached sample, or ``None``."""

    @abstractmethod
    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        """Retrieve a freshly generated sample by its vLLM-returned handle."""

    def cache(self, handle: str, file_idx: int) -> None:  # noqa: B027
        """Persist a generated sample to the cache location."""

    def delete(self, handle: str) -> None:  # noqa: B027
        """Clean up a generated sample (e.g. delete a temp file)."""


class HiddenStatesBackend(ABC):
    """Plugin interface for hidden-states transfer backends.

    Each backend registers itself via ``@HiddenStatesBackend.register(name)``
    and implements these four static hooks so that scripts (``train.py``,
    ``launch_vllm.py``) can discover and configure backends without hardcoding.
    """

    registry: ClassVar[dict[str, type[HiddenStatesBackend]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
    ) -> Callable[[type[HiddenStatesBackend]], type[HiddenStatesBackend]]:
        def decorator(
            subclass: type[HiddenStatesBackend],
        ) -> type[HiddenStatesBackend]:
            if name in cls.registry:
                raise ValueError(f"Backend '{name}' is already registered.")
            cls.registry[name] = subclass
            return subclass

        return decorator

    @staticmethod
    @abstractmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``train.py``."""
        ...

    @staticmethod
    @abstractmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``launch_vllm.py``."""
        ...

    @staticmethod
    @abstractmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> HiddenStatesTransfer:
        """Construct a :class:`HiddenStatesTransfer` from parsed train args."""
        ...

    @staticmethod
    @abstractmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        """Construct the ``kv_transfer_config`` dict for ``vllm serve``."""
        ...


# ---------------------------------------------------------------------------
# File-based backend (shared filesystem)
# ---------------------------------------------------------------------------


def _load_hs_file(file_path: Path) -> dict[str, torch.Tensor] | None:
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path)

    if file_path.exists():
        return load_file(file_path)

    return None


class FileTransfer(HiddenStatesTransfer):
    """File-system based hidden-states transfer (shared filesystem)."""

    def __init__(self, hidden_states_path: Path):
        self.hidden_states_path = hidden_states_path

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        for path in hidden_states_candidates(self.hidden_states_path, file_idx):
            loaded = _load_hs_file(path)
            if loaded is not None:
                return loaded
        return None

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return _load_hs_file(Path(handle))

    def cache(self, handle: str, file_idx: int) -> None:
        target = hidden_states_file(self.hidden_states_path, file_idx)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(handle, target)

    def delete(self, handle: str) -> None:
        Path(handle).unlink()


@HiddenStatesBackend.register("file")
class FileBackend(HiddenStatesBackend):
    """Shared-filesystem backend using safetensors files."""

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default=None,
            help=(
                "The path where cached hidden states files are stored. (Default: "
                "args.data_path / 'hidden_states')"
            ),
        )

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default="/tmp/hidden_states",  # noqa: S108
            help="The directory to save hidden states to. Default '/tmp/hidden_states'",
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> FileTransfer:
        hs_path = (
            Path(args.hidden_states_path)
            if args.hidden_states_path
            else Path(data_path) / "hidden_states"
        )
        return FileTransfer(hs_path)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        return {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }


# ---------------------------------------------------------------------------
# Mooncake-based backend (distributed store)
# ---------------------------------------------------------------------------


class MooncakeTransfer(HiddenStatesTransfer):
    """Mooncake distributed store based hidden-states transfer."""

    def __init__(self, store: MooncakeHiddenStatesStore):
        self.store = store

    def setup(self) -> None:
        if not self.store.is_setup:
            self.store.setup()

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:  # noqa: ARG002
        return None

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return self.store.get_sample(handle)

    def delete(self, handle: str) -> None:
        self.store.delete_sample(handle)


@HiddenStatesBackend.register("mooncake")
class MooncakeBackend(HiddenStatesBackend):
    """Mooncake distributed store backend (no shared filesystem required)."""

    @staticmethod
    def _add_mooncake_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mooncake-master",
            type=str,
            default="127.0.0.1:50051",
            help="Mooncake master server address. Used with backend=mooncake.",
        )
        parser.add_argument(
            "--mooncake-metadata-server",
            type=str,
            default="P2PHANDSHAKE",
            help=(
                "Mooncake metadata server (or P2PHANDSHAKE). "
                "Used with backend=mooncake."
            ),
        )
        parser.add_argument(
            "--mooncake-protocol",
            choices=["tcp", "rdma"],
            default="tcp",
            help="Mooncake transport protocol. Used with backend=mooncake.",
        )

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        MooncakeBackend._add_mooncake_args(parser)

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        MooncakeBackend._add_mooncake_args(parser)

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,  # noqa: ARG004
    ) -> MooncakeTransfer:
        local_hostname = os.environ.get(
            "MOONCAKE_LOCAL_HOSTNAME"
        ) or socket.gethostbyname(socket.gethostname())

        store = MooncakeHiddenStatesStore(
            MooncakeStoreConfig(
                local_hostname=local_hostname,
                metadata_server=args.mooncake_metadata_server,
                master_server_address=args.mooncake_master,
                protocol=args.mooncake_protocol,
            )
        )
        return MooncakeTransfer(store)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        local_hostname = os.environ.get(
            "MOONCAKE_LOCAL_HOSTNAME"
        ) or socket.gethostbyname(socket.gethostname())

        mooncake_cfg = MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=args.mooncake_metadata_server,
            master_server_address=args.mooncake_master,
            protocol=args.mooncake_protocol,
        )

        return {
            "kv_connector": "MooncakeHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_module_path": (
                "hs_connectors.mooncake_hidden_states_connector"
            ),
            "kv_connector_extra_config": {
                "mooncake": dataclasses.asdict(mooncake_cfg),
            },
        }
