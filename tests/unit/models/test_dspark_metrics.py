"""Unit tests for the DSpark loss and metrics."""

from functools import partial

import pytest
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from speculators.losses import (
    recorded_token_accept_prob,
    resolve_loss_config,
    survival_loss_decay,
)
from speculators.losses.eager import tv_loss
from speculators.models.dspark.metrics import compute_metrics as _compute_metrics

compute_metrics = partial(_compute_metrics, tv_loss_fn=tv_loss)
_DEFAULT_LOSS = resolve_loss_config('{"ce": 0.1, "tv": 0.9}', "eager")


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


class TestComputeMetrics:
    @pytest.mark.parametrize("per_position_loss_weight", ["fixed-exp-decay", "dpace"])
    def test_empty_loss_mask_keeps_backward_graph_with_zero_gradients(
        self, per_position_loss_weight
    ):
        """An empty mask must keep both gradient paths and produce zero loss."""
        logits = torch.randn(1, 8, 16, requires_grad=True)
        targets = torch.randn(1, 8, 16)
        confidence_logits = torch.randn(1, 8, requires_grad=True)
        loss_mask = torch.zeros(1, 8)

        loss, _ = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            per_position_loss_weight=per_position_loss_weight,
        )
        loss.backward()

        assert loss.item() == 0
        assert logits.grad is not None
        assert confidence_logits.grad is not None
        assert torch.count_nonzero(logits.grad) == 0
        assert torch.count_nonzero(confidence_logits.grad) == 0

    def test_perfect_draft_low_loss_high_accept(self):
        # block_size=2; with sample_from_anchor=False, position 0 is the anchor
        # (masked) and position 1 supervised.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
            sample_from_anchor=False,
        )
        assert torch.isfinite(loss)
        # Matching distributions -> CE/TV ~ 0 and acceptance ~ 1.
        assert float(loss) < 1e-2
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) > 0.99
        # One draft slot per block accepted w.p. ~1, plus the bonus token -> ~2.
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 2.0) < 1e-2
        eal = metrics["eal_sum"] / metrics["eal_total"]
        assert abs(float(eal) - 2.0) < 1e-2

    def test_perfect_draft_anchor_sampled_includes_slot0(self):
        # sample_from_anchor=True (default): slot 0 is the first real prediction,
        # so every position is supervised and accept_len counts all draft slots.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.ones(1, 4, dtype=torch.float32)
        loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
        )
        assert torch.isfinite(loss)
        assert float(loss) < 1e-2
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) > 0.99
        # Two draft slots per block accepted w.p. ~1, plus the bonus token -> ~3.
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 3.0) < 1e-2
        eal = metrics["eal_sum"] / metrics["eal_total"]
        assert abs(float(eal) - 3.0) < 1e-2

    def test_accept_rate_equals_softmax_overlap(self):
        """accept_rate (now 1 - tv) matches the explicit softmax-overlap formula."""
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 32) * 3
        targets = torch.randn(1, 4, 32) * 3
        loss_mask = torch.ones(1, 4, dtype=torch.float32)
        _, metrics = compute_metrics(
            logits, targets, None, loss_mask, 2, loss_config=_DEFAULT_LOSS
        )
        draft_p = torch.softmax(logits.float(), dim=-1)
        target_p = torch.softmax(targets.float(), dim=-1)
        overlap = torch.minimum(draft_p, target_p).sum(dim=-1)
        assert torch.isclose(metrics["accept_rate_sum"], overlap.sum(), atol=1e-5)

    def test_confidence_target_is_overlap(self):
        # When draft == target, accept rate == 1, so a confidence logit that is
        # very positive (sigmoid -> 1) yields ~zero abs error.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
        )
        abs_err = (
            metrics["confidence_abs_error_sum"] / metrics["confidence_abs_error_total"]
        )
        assert float(abs_err) < 1e-2
        assert "confidence_loss_sum" in metrics

    def test_confidence_term_changes_loss(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_no_conf, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        # A badly-calibrated confidence head (predicts accept~1 when accept~0)
        # must add positive BCE on top of the base loss.
        confidence_logits = torch.full((1, 4), 20.0)
        loss_conf, _ = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            confidence_head_alpha=1.0,
        )
        assert float(loss_conf) > float(loss_no_conf)

    @pytest.mark.parametrize("sample_from_anchor", [False, True])
    @pytest.mark.parametrize("gamma", [1.0, 4.0])
    def test_confidence_loss_keeps_fixed_decay(self, seed, sample_from_anchor, gamma):
        """Drafter weights must not change the confidence loss or its gradients."""
        logits = torch.randn(1, 8, 16)
        targets = torch.randn(1, 8, 16)
        confidence_logits = torch.randn(1, 8, requires_grad=True)
        loss_mask = torch.tensor([[1, 1, 0, 1, 1, 1, 1, 0]], dtype=torch.float32)
        if not sample_from_anchor:
            loss_mask[:, ::4] = 0

        accept_rate = torch.minimum(
            logits.softmax(dim=-1), targets.softmax(dim=-1)
        ).sum(dim=-1)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, accept_rate, reduction="none"
        )
        positions = torch.arange(8) % 4
        offset = 0 if sample_from_anchor else 1
        weights = torch.exp(-(positions - offset).clamp_min(0) / gamma)
        expected_loss = (bce * loss_mask * weights).sum() / loss_mask.sum()
        confidence_head_alpha = 0.3
        expected_grad = torch.autograd.grad(
            confidence_head_alpha * expected_loss, confidence_logits
        )[0]

        draft_losses = []
        for mode in ("fixed-exp-decay", "dpace"):
            loss, metrics = compute_metrics(
                logits,
                targets,
                confidence_logits,
                loss_mask,
                block_size=4,
                loss_config=_DEFAULT_LOSS,
                gamma=gamma,
                confidence_head_alpha=confidence_head_alpha,
                per_position_loss_weight=mode,
                sample_from_anchor=sample_from_anchor,
            )
            torch.testing.assert_close(metrics["confidence_loss_sum"], expected_loss)
            actual_grad = torch.autograd.grad(loss, confidence_logits)[0]
            torch.testing.assert_close(actual_grad, expected_grad)
            draft_losses.append(metrics["ce_loss_sum"])

        # The option must still change the drafter's loss.
        assert not torch.isclose(draft_losses[0], draft_losses[1])

    def test_confidence_cumprod_bias_sign(self):
        # Draft != target so accept rate is ~0; an over-confident head (predicts
        # accept ~1) must show a positive cumulative-product calibration bias.
        # sample_from_anchor=False: position 0 is the anchor (masked).
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            sample_from_anchor=False,
        )
        bias = (
            metrics["confidence_cumprod_bias_sum"]
            / metrics["confidence_cumprod_bias_total"]
        )
        assert float(bias) > 0.5

    def test_alpha_weighting(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_small, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 0.1}', "eager"),
        )
        loss_large, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 1.0}', "eager"),
        )
        assert float(loss_large) > float(loss_small)

    def test_metric_keys_present(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        _, metrics = compute_metrics(
            logits,
            targets,
            torch.zeros(1, 4),
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        for key in (
            "loss_sum",
            "loss_total",
            "ce_loss_sum",
            "tv_loss_sum",
            "full_acc_sum",
            "full_acc_total",
            "position_1_acc_sum",
            "accept_len_sum",
            "accept_len_total",
            "confidence_cumprod_bias_sum",
        ):
            assert key in metrics
        # all metric values must be tensors (so dist.reduce works in the trainer)
        assert all(torch.is_tensor(v) for v in metrics.values())


def _log_probs(probs: list[list[float]]) -> torch.Tensor:
    return torch.tensor(probs).log().unsqueeze(0)


class TestSurvivalWeight:
    """Chain-survival slot weight: w_k = prod_{i<k} ((1-lam) * s_i + lam) with
    s_i = min(1, q_i(y_i) / p_i(y_i)) for the recorded token y_i."""

    def test_accept_prob_uses_recorded_token_and_caps_at_one(self):
        # Slot 0: q(y)=0.2, p(y)=0.8 -> 0.25. Slot 1: q(y)=0.9 > p(y)=0.3 -> 1.
        # The recorded token (index 1) is not the verifier's top token in slot 1.
        logits = _log_probs([[0.8, 0.2, 0.0001], [0.1, 0.9, 0.0001]])
        targets = _log_probs([[0.2, 0.8, 0.0001], [0.7, 0.3, 0.0001]])
        ids = torch.tensor([[1, 1]])
        accept = recorded_token_accept_prob(logits, targets, ids)
        assert accept.shape == (1, 2)
        assert accept[0].tolist() == pytest.approx([0.25, 1.0], abs=1e-3)

    def test_accept_prob_applies_temperature_to_both_distributions(self):
        logits = torch.tensor([[[2.0, 0.0, -1.0]]])
        targets = torch.tensor([[[0.0, 3.0, 1.0]]])
        ids = torch.tensor([[0]])
        for temperature in (0.5, 2.0):
            q = torch.softmax(logits / temperature, -1)[0, 0, 0]
            p = torch.softmax(targets / temperature, -1)[0, 0, 0]
            accept = recorded_token_accept_prob(
                logits, targets, ids, temperature=temperature
            )
            assert accept.item() == pytest.approx(min(1.0, (q / p).item()), rel=1e-5)

    def test_accept_prob_is_independent_of_chunk_size(self):
        torch.manual_seed(0)
        logits, targets = torch.randn(1, 13, 32), torch.randn(1, 13, 32)
        ids = torch.randint(0, 32, (1, 13))
        whole = recorded_token_accept_prob(logits, targets, ids, chunk=64)
        chunked = recorded_token_accept_prob(logits, targets, ids, chunk=4)
        assert torch.allclose(whole, chunked, atol=1e-6)

    def test_weight_is_exclusive_product_restarting_per_block(self):
        # Two blocks of 3 slots. The slot's own acceptance never enters its weight,
        # and the second block starts again at 1.
        accept = torch.tensor([[0.5, 0.2, 0.9, 0.0, 1.0, 1.0]])
        mask = torch.ones(1, 6)
        weight = survival_loss_decay(
            None, loss_mask=mask, block_size=3, survival_lambda=0.0, accept_prob=accept
        )
        assert weight[0].tolist() == pytest.approx([1.0, 0.5, 0.1, 1.0, 0.0, 0.0])

    def test_lambda_mixes_acceptance_with_one(self):
        accept = torch.tensor([[0.0, 0.5, 1.0]])
        mask = torch.ones(1, 3)
        half = survival_loss_decay(
            None, loss_mask=mask, block_size=3, survival_lambda=0.5, accept_prob=accept
        )
        assert half[0].tolist() == pytest.approx([1.0, 0.5, 0.375])
        uniform = survival_loss_decay(
            None, loss_mask=mask, block_size=3, survival_lambda=1.0, accept_prob=accept
        )
        assert uniform[0].tolist() == pytest.approx([1.0, 1.0, 1.0])

    def test_unsupervised_slot_neither_gets_weight_nor_cuts_the_chain(self):
        accept = torch.tensor([[0.5, 0.0, 0.5]])
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        weight = survival_loss_decay(
            None, loss_mask=mask, block_size=3, survival_lambda=0.0, accept_prob=accept
        )
        assert weight[0].tolist() == pytest.approx([1.0, 0.0, 0.5])

    def test_loss_has_no_position_decay_and_no_gradient_through_the_weight(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 6, 16, requires_grad=True)
        targets = torch.randn(1, 6, 16)
        ids = torch.randint(0, 16, (1, 6))
        mask = torch.ones(1, 6)
        ce_only = resolve_loss_config("ce", "eager")
        kwargs = {
            "block_size": 3,
            "loss_config": ce_only,
            "per_position_loss_weight": "survival",
            "recorded_token_ids": ids,
        }
        # lambda=1: every weight is 1, so the loss is the plain masked mean; a
        # gamma that would change a position decay must change nothing.
        loss, _ = compute_metrics(
            logits, targets, None, mask, survival_lambda=1.0, gamma=0.1, **kwargs
        )
        plain = resolve_loss_config("ce", "eager")["ce"][0](logits, targets).mean()
        assert loss.item() == pytest.approx(plain.item(), rel=1e-4)

        # lambda=0: the gradient equals that of the loss with the weight held fixed.
        loss, metrics = compute_metrics(
            logits, targets, None, mask, survival_lambda=0.0, **kwargs
        )
        (grad,) = torch.autograd.grad(loss, logits)
        accept = recorded_token_accept_prob(logits.detach(), targets, ids)
        weight = survival_loss_decay(
            None, loss_mask=mask, block_size=3, survival_lambda=0.0, accept_prob=accept
        )
        elementwise = ce_only["ce"][0](logits, targets)
        (expected,) = torch.autograd.grad((elementwise * weight).mean(), logits)
        assert torch.allclose(grad, expected, atol=1e-6)
        assert metrics["survival_weight_sum"] / metrics[
            "survival_weight_total"
        ] == pytest.approx(weight.mean().item(), rel=1e-5)

    def test_survival_requires_recorded_tokens(self):
        logits, targets = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        with pytest.raises(ValueError, match="recorded_token_ids"):
            compute_metrics(
                logits,
                targets,
                None,
                torch.ones(1, 4),
                block_size=2,
                loss_config=_DEFAULT_LOSS,
                per_position_loss_weight="survival",
            )

    def test_confidence_loss_uses_the_survival_weight(self):
        torch.manual_seed(0)
        logits, targets = torch.randn(1, 6, 16), torch.randn(1, 6, 16)
        confidence_logits = torch.randn(1, 6)
        ids = torch.randint(0, 16, (1, 6))
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 1.0]])
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            mask,
            block_size=3,
            loss_config=_DEFAULT_LOSS,
            per_position_loss_weight="survival",
            recorded_token_ids=ids,
            survival_lambda=0.0,
            gamma=0.1,
        )
        weight = survival_loss_decay(
            None,
            loss_mask=mask,
            block_size=3,
            survival_lambda=0.0,
            accept_prob=recorded_token_accept_prob(logits, targets, ids),
        )
        bce = binary_cross_entropy_with_logits(
            confidence_logits, 1.0 - tv_loss(logits, targets), reduction="none"
        )
        expected = (bce * weight).sum() / mask.sum()
        assert metrics["confidence_loss_sum"].item() == pytest.approx(
            expected.item(), rel=1e-5
        )

    def test_empty_loss_mask_gives_zero_loss(self):
        logits = torch.randn(1, 8, 16, requires_grad=True)
        loss, _ = compute_metrics(
            logits,
            torch.randn(1, 8, 16),
            None,
            torch.zeros(1, 8),
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            per_position_loss_weight="survival",
            recorded_token_ids=torch.zeros(1, 8, dtype=torch.long),
        )
        loss.backward()
        assert loss.item() == 0
        assert torch.count_nonzero(logits.grad) == 0
