# Loss Functions

The training loss controls what the draft model is optimized for. Select it with `--loss-fn` in [`speculators train`](../cli/train.md), which accepts either a single loss name or a JSON dict for a weighted combination. It applies to Eagle-3, P-EAGLE, DFlash, and DSpark. MTP uses its own multi-step cross-entropy loss and ignores this flag.

The fused Triton implementation is the default. Use `--loss-implementation eager` for compatibility or numerical validation; eager losses may OOM with DFlash or DSpark.

```bash
speculators train ... --loss-fn kl_div
```

## Available Losses

`p` is the target distribution, `q` the draft distribution, and `alpha = sum_v min(p_v, q_v)` the distributional overlap, which equals the acceptance rate of speculative decoding.

| Name        | Objective                              | Notes                                                                          |
| ----------- | -------------------------------------- | ------------------------------------------------------------------------------ |
| `kl_div`    | Forward KL, target to draft            | Default. Mass-covering: penalizes the draft for missing target mass.           |
| `rkl`       | Reverse KL, draft to target            | Mode-seeking: the draft concentrates on the target's dominant modes.           |
| `jsd`       | Jensen-Shannon divergence              | Symmetric, bounded by `log 2`. Balances forward and reverse KL.                |
| `ce`        | Cross-entropy against `argmax p`       | Hard labels from the target. Required by `--per-position-loss-weight dpace`.   |
| `tv`        | Total variation, `1 - alpha`           | Optimizes the acceptance rate directly. Gradients vanish when overlap is low.  |
| `nla`       | Negative log-acceptance, `-log(alpha)` | TV's target with a `1 / alpha` gradient boost, so it trains from a cold start. |
| `lk_hybrid` | Adaptive KL/TV blend                   | `lambda * KL + (1 - lambda) * TV` with `lambda = exp(-3 * alpha)`, detached.   |

All losses use the selected implementation; there is no automatic fallback.

## Choosing a Loss

- Start with `kl_div`. It is the default and the best-tested option.
- To optimize acceptance rate directly, use `lk_hybrid` or `nla`. Plain `tv` has the same objective but weak gradients early in training.
- Use `ce` when you want hard-label supervision or D-PACE position weighting.

## Weighted Combinations

Pass a JSON dict to train on a weighted sum of several losses. Weights are used as provided and are not normalized. Each term is also logged separately as `{name}_loss`, before its weight is applied.

```bash
speculators train ... --loss-fn '{"ce": 0.1, "tv": 0.9}'
```

## Survival Position Weighting

`--per-position-loss-weight survival` (DSpark, full vocabulary) weights slot `k` of a block by the probability that serving reaches it along the recorded tokens `y`:

```
w_k = prod_{i<k} ((1 - lambda) * s_i + lambda),    s_i = min(1, q_i(y_i) / p_i(y_i))
```

`s_i` is the probability that a sampled draft was accepted at slot `i`, given that the verifier emitted `y_i`. `q` and `p` are the draft and verifier distributions from the raw logits divided by `--survival-temperature`. No gradient flows through `w_k`, and the exponential position decay is not applied to the token loss. The confidence loss uses the same weight `w_k`. Any `--loss-fn` works. `--survival-lambda 1` gives every slot weight 1.

Training logs `survival_weight` (mean slot weight) and `survival_accept` (mean `s_i`) over supervised slots.

```bash
speculators train ... --speculator-type dspark --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --per-position-loss-weight survival --survival-lambda 0.5 --survival-temperature 1.0
```

## References

- Lin, "Divergence measures based on the Shannon entropy" (1991) -- Jensen-Shannon divergence
- Samarin et al., ["LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding"](https://arxiv.org/abs/2602.23881) -- `nla` and `lk_hybrid`
