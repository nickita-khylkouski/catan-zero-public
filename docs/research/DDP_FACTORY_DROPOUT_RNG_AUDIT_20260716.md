# DDP Factory Dropout RNG Audit — 2026-07-16

## Outcome

Fresh distributed BC jobs rendered by `tools/start_training_factory.py` did not
pass `--training-rng-rank-offset`. Model construction seeds each DDP process
identically so that initial parameters match. Without the post-construction rank
offset, every rank also enters training with the same PyTorch RNG state and uses
the same dropout masks. The factory now enables the existing rank-offset contract
whenever `world_size > 1` and attests the choice in `bc_training_topology`.
Single-rank factory jobs remain unchanged.

This fix changes only the PyTorch training RNG after identical model
initialization. The NumPy data trajectory remains one deterministic global order
that is sliced by rank, as required by `_epoch_order` and
`_policy_aux_epoch_order`.

## Quantitative impact

Let the per-rank dropout-induced gradient noise have variance `sigma^2`. If all
rank masks are identical, averaging the DDP gradients leaves variance
`sigma^2`. With independent masks on `W` ranks, the averaged noise variance is
`sigma^2 / W`. A fixed-seed, 200,000-draw Bernoulli simulation measured:

| DDP ranks | Identical-mask variance | Independent-mask variance | Ratio |
| ---: | ---: | ---: | ---: |
| 2 | 1.000000 | 0.501182 | 1.995x |
| 8 | 1.000000 | 0.125169 | 7.989x |

The old factory path therefore discarded essentially all dropout-noise averaging:
about 2x excess variance at two ranks and 8x at eight ranks. This is a learning
signal defect, not merely a reproducibility issue.

Archived B200 campaign recipes inspected during this audit explicitly recorded
`training_rng_rank_offset=true`; their topology already had independent streams.
The affected scope is future fresh distributed retraining rendered through the
factory (and any historical factory run that did not add the flag manually).

## H100 gate

Before any experiment, the bounded two-rank CUDA/NCCL preflight on H100
`68.209.74.159` failed on both ranks before NCCL initialization:

```text
CUDA error 802: system not yet initialized
torch.cuda.is_available() == False
exit code: 1
```

The command was bounded by a 45-second timeout and a 20-second process-group
timeout. No training or further GPU work was run after this failure. The B200
data host was kept read-only and no jobs were started there.

## Regression contract

- `world_size=1`: no `--training-rng-rank-offset`; manifest value is `false`.
- `world_size=2` or `8`: the flag appears exactly once; manifest value is `true`.
- Existing DDP RNG tests prove rank-independent, reproducible masks when enabled
  and identical masks under the historical flag-off behavior.
- A deterministic variance regression checks the expected `W`-fold difference
  for two and eight ranks.
