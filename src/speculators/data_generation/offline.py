import logging
from pathlib import Path

import torch
from hs_connectors import hidden_states_candidates, iter_hidden_state_indices
from safetensors import safe_open

logger = logging.getLogger(__name__)


def hidden_state_offset(stored_token_ids: list[int], tokens: list[int]) -> int:
    """Index in ``tokens`` where a stored row starts.

    A row is stored whole (offset 0) or as its last N tokens when the request
    asked the connector for ``keep_last`` (offset ``len(tokens) - N``).

    Raises:
        ValueError: The stored token ids are not ``tokens`` or a suffix of it.
    """
    offset = len(tokens) - len(stored_token_ids)
    if offset < 0 or stored_token_ids != tokens[offset:]:
        raise ValueError(f"Token ids don't match expected token ids {tokens}")
    return offset


def check_hidden_states(data: dict, tokens: list[int]):
    required = {"token_ids", "hidden_states"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Hidden-state payload is missing keys: {missing}")

    t_ids = data["token_ids"].tolist()
    hidden_state_offset(t_ids, tokens)

    hs = data["hidden_states"]
    if not isinstance(hs, torch.Tensor):
        raise ValueError(f"Hidden states must be a tensor, got {type(hs).__name__}")
    if len(t_ids) != hs.shape[0]:
        raise ValueError(
            f"Sequence length of hidden states {hs.shape[0]}"
            f" doesn't match num tokens {len(t_ids)}"
        )

    # aminmax has no float8 kernel, and the FP8 connector saves float8 files.
    if hs.dtype == torch.float8_e4m3fn:
        # float8_e4m3fn has no infinities, so NaN is the only non-finite value.
        # Checking for it avoids a full-size copy in a wider dtype.
        if bool(torch.isnan(hs).any()):
            raise ValueError(
                "Hidden states contain non-finite values (min=nan, max=nan)"
            )
        return

    if hs.dtype == torch.float8_e5m2:
        # bfloat16 holds every float8_e5m2 value exactly, including infinities.
        hs = hs.to(torch.bfloat16)

    lo, hi = torch.aminmax(hs)
    if bool(torch.isfinite(torch.stack((lo, hi))).all()):
        return

    raise ValueError(f"Hidden states contain non-finite values (min={lo}, max={hi})")


def get_existing_hidden_state_indices(output_path: Path) -> list[int]:
    """Find existing `hs_i.safetensors` files (where i is the file index), flat
    at ``output_path`` or in its sharded sub-directories."""
    return sorted(set(iter_hidden_state_indices(output_path)))


def find_corrupt_hidden_state_indices(
    output_path: Path, indices: list[int]
) -> dict[int, str]:
    """Find cached `hs_i.safetensors` files that cannot be opened.

    Returns a mapping of file index to the reason that file is unreadable.

    ``safe_open`` performs safetensors' own header validation in its constructor
    -- header length, header JSON, offset contiguity, dtype/shape/offset
    consistency, and exact buffer coverage -- without reading any tensor data, so
    the cost scales with the file count rather than the tensor size. Indices
    whose file is absent are skipped; absence is already handled as "not generated".

    Note that this cannot detect corruption that preserves the file's size: the
    format carries no checksum, so bit-rot inside the tensor data still requires
    loading it (see :func:`check_hidden_states`).
    """
    corrupt: dict[int, str] = {}
    for idx in indices:
        for path in hidden_states_candidates(output_path, idx):
            try:
                with safe_open(path, framework="pt"):
                    pass
            except FileNotFoundError:
                continue
            except Exception as e:
                corrupt[idx] = str(e) or type(e).__name__
            break
    return corrupt


def get_indices_to_process(
    num_samples: int,
    max_samples: int | None,
    existing: list[int],
    world_size: int,
    rank: int,
) -> list[int]:
    """Determines which indices should be processed. If max_samples is None
    returns all dataset indices not in existing. Otherwise gets the first
    `max_samples - len(existing)` samples not already in existing.

    Args:
        num_samples: Total size of preprocessed dataset
        max_samples: (Optional) limit for number of samples to process
        existing: list of ids that have already been processed
        world_size: Number of nodes to generate on
        rank: The rank of the local node

    Returns:
        list of dataset indices to process
    """

    target = min(max_samples, num_samples) if max_samples is not None else num_samples

    if target <= 0:
        return []

    # Strided, not contiguous: rows sit in dataset order, so contiguous ranges
    # hand one rank every long trajectory and another the short single-turn
    # rows, and the barrier waits on the slowest. Files are named by global
    # index, so a resume skips what exists under either assignment.
    existing_s = set(existing)
    to_process = [i for i in range(rank, target, world_size) if i not in existing_s]

    if not to_process:
        logger.info("All samples for this rank already processed!")
        return []

    already = len(existing_s.intersection(range(rank, target, world_size)))
    if already:
        logger.info(f"Found {already} existing samples for rank {rank}.")

    return to_process
