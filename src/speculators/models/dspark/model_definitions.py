"""Sequential (Markov) and confidence heads for the DSpark draft model."""

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

__all__ = [
    "ConfidenceHead",
    "MarkovHead",
    "RecurrentCorrector",
]


class MarkovHead(nn.Module):
    """Low-rank sequential logit bias ``B = W1 @ W2``.

    ``W1`` indexes the verifier vocabulary (the previous token id); ``W2`` projects
    to the draft vocabulary so the bias adds onto the DFlash logits.
    """

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        head_type: str = "vanilla",
    ) -> None:
        super().__init__()
        if markov_rank <= 0:
            raise ValueError(f"markov_rank must be > 0, got {markov_rank}")
        if head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(f"Unsupported markov_head_type: {head_type!r}")
        self.head_type = head_type
        self.markov_rank = markov_rank
        self.markov_w1 = nn.Embedding(verifier_vocab_size, markov_rank)
        nn.init.normal_(self.markov_w1.weight, std=0.01)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False)
        if head_type == "gated":
            self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)
        elif head_type == "rnn":
            # Joint [gate; candidate; output] projection over [state; prev_emb; hidden].
            self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up W1 embeddings for the given previous-token ids."""
        return self.markov_w1(token_ids.long())

    def block_bias(
        self,
        *,
        prev_token_ids: torch.Tensor,  # [N, block_size]
        hidden_states: torch.Tensor,  # [N, block_size, hidden]
        prev_emb: torch.Tensor | None = None,  # [N, block_size, r]
    ) -> torch.Tensor:
        """Return the per-position logit bias, shape [N, block_size, draft_vocab]."""
        if prev_emb is None:
            prev_emb = self.prev_embeddings(prev_token_ids)
        prev_emb = prev_emb.to(self.markov_w2.weight.dtype)

        if self.head_type == "vanilla":
            return self.markov_w2(prev_emb)

        if self.head_type == "gated":
            hidden_states = hidden_states.to(prev_emb.dtype)
            gate = torch.sigmoid(
                self.gate_proj(torch.cat([hidden_states, prev_emb], dim=-1))
            )
            return self.markov_w2(gate * prev_emb)

        # rnn: maintain a recurrent state across block positions.
        hidden_states = hidden_states.to(prev_emb.dtype)
        num_blocks, block_size, _ = prev_emb.shape
        state = prev_emb.new_zeros(num_blocks, self.markov_rank)
        outputs = []
        for k in range(block_size):
            z = torch.cat([state, prev_emb[:, k], hidden_states[:, k]], dim=-1)
            gate_raw, cand_raw, out_raw = self.joint_proj(z).chunk(3, dim=-1)
            gate = torch.sigmoid(gate_raw)
            state = gate * state + (1.0 - gate) * torch.tanh(cand_raw)
            outputs.append(self.markov_w2(torch.tanh(out_raw)))
        return torch.stack(outputs, dim=1)


class RecurrentCorrector(nn.Module):
    """Sequential hidden-space correction over the slots of each draft block.

    Slot ``k`` reads the previous slot's state, the verifier embedding of the
    previous token and the trunk's post-norm hidden state ``g_k``:

        x_k = in_proj([norm(s_{k-1}); norm(e_k); g_k])
        s_k = x_k + MLP(norm(x_k))                       (SwiGLU)
        delta_k = out_proj(norm(s_k))

    and the caller scores ``lm_head(g_k + delta_k)``. Because ``lm_head`` is
    linear, a candidate ``c`` scores ``lm_head(g_k)[c] + W_lm[c] . delta_k``, so
    serving can rescore the top-K of the trunk's logits without another
    vocabulary-wide matmul. ``out_proj`` starts at zero, so an untrained
    corrector leaves the trunk's logits unchanged. The state starts at zero in
    every block.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        state_size: int,
        intermediate_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.state_norm = Qwen3RMSNorm(state_size, eps=eps)
        self.embed_norm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.in_proj = nn.Linear(state_size + 2 * hidden_size, state_size, bias=False)
        self.mlp_norm = Qwen3RMSNorm(state_size, eps=eps)
        self.gate_proj = nn.Linear(state_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(state_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, state_size, bias=False)
        self.out_norm = Qwen3RMSNorm(state_size, eps=eps)
        self.out_proj = nn.Linear(state_size, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [N, block_size, hidden] (trunk, post-norm)
        prev_embeddings: torch.Tensor,  # [N, block_size, hidden]
    ) -> torch.Tensor:
        """Return the per-slot correction delta, shape [N, block_size, hidden]."""
        num_blocks, block_size, _ = hidden_states.shape
        dtype = self.in_proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        prev_embeddings = self.embed_norm(prev_embeddings.to(dtype))
        state = hidden_states.new_zeros(num_blocks, self.state_size)
        deltas = []
        for k in range(block_size):
            x = self.in_proj(
                torch.cat(
                    [
                        self.state_norm(state),
                        prev_embeddings[:, k],
                        hidden_states[:, k],
                    ],
                    dim=-1,
                )
            )
            h = self.mlp_norm(x)
            state = x + self.down_proj(
                nn.functional.silu(self.gate_proj(h)) * self.up_proj(h)
            )
            deltas.append(self.out_proj(self.out_norm(state)))
        return torch.stack(deltas, dim=1)


class ConfidenceHead(nn.Module):
    """Per-position acceptance-probability predictor (linear -> scalar logit)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)
