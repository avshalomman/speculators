"""FP8-quantizing sibling of vLLM's ``ExampleHiddenStatesConnector``.

Quantizes hidden states to ``float8_e4m3fn`` (per-token scaling) before
writing them to disk, roughly halving the on-disk/on-wire size of extracted
hidden states relative to the bf16 baseline. Everything else
(async DtoH copy, thread-pool disk writes, file locking, scheduler-side
bookkeeping) is inherited unchanged from ``ExampleHiddenStatesConnector`` --
only the final tensor-write step is overridden.

Loaded out-of-tree via ``kv_connector_module_path``; must be used with the
``extract_hidden_states`` speculative method. Paired on the read side with
:class:`hs_connectors.transfer.FP8Transfer`, which dequantizes transparently
so consumers (e.g. ``speculators.train.data.ArrowDataset``) never see FP8
tensors.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector as _eh_mod,
)

from hs_connectors.fp8_utils import SCALES_KEY, quantize_tensor_to_fp8

if TYPE_CHECKING:
    from vllm.v1.request import Request


class FP8HiddenStatesConnector(_eh_mod.ExampleHiddenStatesConnector):
    """Quantizes hidden states to float8_e4m3fn before saving to safetensors.

    The output file contains three tensors::

        hidden_states        - fp8 [seq_len, num_layers, hidden_size]
        hidden_states_scales - fp32 [seq_len, 1, 1]
        token_ids            - int64 [seq_len]

    A request may carry ``kv_transfer_params={"keep_last": N}``: the file then
    holds only the last N prompt tokens, extended down to the start of the KV
    block holding token ``T - N`` so it is never shorter than N. A draft with a
    sliding window of W over the context reads nothing before ``W`` tokens
    ahead of its first supervised position, so a row stored as its final turn
    plus W context tokens trains it exactly as the full row would, at the
    cost of the tail instead of the whole prompt.
    """

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        delay_free, params = super().request_finished(request, block_ids)
        keep_last = (request.kv_transfer_params or {}).get("keep_last")
        if keep_last:
            req_id = request.request_id
            self._pending_saves[req_id] = _trim_to_tail(
                self._pending_saves[req_id], int(keep_last), self._block_size
            )
        return delay_free, params

    @staticmethod
    def _write_tensors(
        tensors: dict[str, torch.Tensor],
        event: torch.cuda.Event,
        filename: str,
        lock_fd: int | None,
    ) -> None:
        try:
            event.synchronize()
            hidden_states = tensors["hidden_states"]
            fp8_hs, scales = quantize_tensor_to_fp8(hidden_states)
            quantized_tensors = {
                **tensors,
                "hidden_states": fp8_hs,
                SCALES_KEY: scales,
            }
            save_file(quantized_tensors, filename)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)


def _trim_to_tail(
    pending: _eh_mod.PendingSave, keep_last: int, block_size: int
) -> _eh_mod.PendingSave:
    """Keep the KV blocks and token ids from the block holding token
    ``T - keep_last`` on."""
    num_tokens = int(pending.token_ids.shape[0])
    if keep_last >= num_tokens:
        return pending
    first_block = (num_tokens - keep_last) // block_size
    return replace(
        pending,
        token_ids=pending.token_ids[first_block * block_size :],
        block_ids=pending.block_ids[first_block:],
    )
