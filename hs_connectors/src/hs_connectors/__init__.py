from hs_connectors.transfer import (
    HS_SHARD_SIZE,
    FileBackend,
    FileTransfer,
    FP8Backend,
    FP8Transfer,
    HiddenStatesBackend,
    HiddenStatesTransfer,
    MooncakeBackend,
    MooncakeTransfer,
    hidden_states_candidates,
    hidden_states_file,
    iter_hidden_state_indices,
)

__all__ = [
    "FP8Backend",
    "FP8Transfer",
    "HS_SHARD_SIZE",
    "FileBackend",
    "FileTransfer",
    "HiddenStatesBackend",
    "HiddenStatesTransfer",
    "MooncakeBackend",
    "MooncakeTransfer",
    "hidden_states_candidates",
    "hidden_states_file",
    "iter_hidden_state_indices",
]

try:
    from hs_connectors.version import version as __version__
except ImportError:
    __version__ = "unknown"
