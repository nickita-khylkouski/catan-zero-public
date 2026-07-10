# CAT-39: HL-Gauss categorical value head

Converts the value target from scalar MSE regression to a distributional
(categorical) head trained with the **HL-Gauss** cross-entropy of Farebrother
et al. 2024 ("Stop Regressing: Training Value Functions via Classification for
Scalable Deep RL", arXiv:2403.03950). The scalar-MSE head is retained as a
selectable control arm; every default is a pure no-op.

## Why HL-Gauss, not two-hot

The head that already existed (`f76`) was **two-hot** C51-style. Farebrother's
ablations show plain two-hot *underperforms* MSE, while HL-Gauss beats it, and
the gap is largest under stochastic dynamics — the whole reason a categorical
value head is worth having in Catan (R8: "maximally stochastic"). So this ticket
verified the built head was two-hot and replaced the **target construction** with
HL-Gauss. The model head shape (a Linear stack on the CLS state + a fixed support
buffer) is unchanged in form; only the projection of scalar targets into
categorical targets changed, plus the R9 support redefinition below.

## Support (R9): win/loss + a truncation class ONLY

Per the CAT-39 R9 ruling the **primary** support is P(win)/win-loss plus one
distinct **truncation class**. VP-margin is explicitly removed from the joint
support and belongs on a separate auxiliary head (existing `final_vp_head`, and
the f75 aux-head scaffolding, task #63, when merged).

- The head emits `value_categorical_bins` logits over a uniform win-loss support
  on `[-1, 1]` (bin centres `linspace(-1, 1, bins)`), followed by **one extra
  truncation logit** when `value_categorical_truncation_class` is set (default).
- Real-outcome rows: HL-Gauss bump on the win-loss bins (a Gaussian of std
  `sigma = sigma_ratio * bin_width` integrated over atom-centred cells via the
  erf; the outer cells run to `+/-inf` so tail mass is captured, not clipped),
  zero mass on the truncation class.
- Truncated rows: one-hot on the truncation class, zero mass on the win-loss
  bins. Their VP-margin signal (F3 soft labels) is **not** fed to this head — it
  is the aux head's job now.

## Scalar readout (search backup / telemetry)

`outputs["value_categorical"]` is the **calibrated win-value**: the expectation
over the win-loss bins, renormalised to drop truncation-class mass. That is the
value the search backup should read (R9: never a blended win+margin value). The
scalar-MSE `outputs["value"]` is left bit-identical. Search, generation,
reanalysis, calibration, and head-to-head evaluation select the readout
explicitly with `value_readout={scalar,categorical}`; scalar remains the
backward-compatible default. A categorical request fails closed unless the
checkpoint metadata proves that the categorical head received positive training
mass and optimizer steps. Merely adding categorical parameters to an old scalar
checkpoint is therefore never mistaken for a trained search value.

## Distribution-space lambda blend

`--value-target-lambda` (MuZero/ReZero, arXiv:2404.16364) blends the realised
outcome `z` with the stored search root value `V_search`:

- **HL-Gauss head**: project `z` and `V_search` **each to a categorical
  distribution**, then mix at `lambda` — never blend scalars and then discretise.
- **MSE control arm**: scalar blend `lambda*z + (1-lambda)*V_search`.

The two are consistent because HL-Gauss preserves the expectation and blending is
linear: `E[lambda*d_z + (1-lambda)*d_V] = lambda*z + (1-lambda)*V` within the bin
resolution (unit-tested). `lambda = 1.0` (default) is a pure no-op, and the blend
is inert on shards with no `root_value` / `root_value_mask` column (every current
shard) — a gen-1-onward lever.

## Config / CLI

Model (`EntityGraphConfig`, appended last for positional-pickle safety; both
default to the OFF/current behaviour):

- `value_categorical_bins: int = 0` — win-loss bins; `0` disables the head (no
  new params, forward bit-identical, warm-start-safe).
- `value_categorical_truncation_class: bool = True` — add the extra truncation
  class when the head is built.

`tools/train_bc.py`:

- `--value-head-type {mse,hlgauss}` (default `mse`) — `mse` is the current
  behaviour exactly. `hlgauss` makes categorical CE the **primary** value
  objective and turns scalar MSE off, so the MSE-vs-HL-Gauss tournament changes
  one objective without silently doubling the value-loss budget.
- `--value-categorical-loss-weight` (default `0.0`) — categorical-primary
  override; in `hlgauss`, `0` falls back to `--value-loss-weight`. Supplying a
  nonzero categorical weight in `mse` mode is rejected as contradictory.
- `--hlgauss-scalar-aux-loss-weight` (default `0.0`) — optional scalar-MSE
  auxiliary in `hlgauss` mode. This is deliberately explicit and off for the
  matched tournament; a nonzero value defines a separate hybrid ablation.
- `--value-hlgauss-sigma-ratio` (default `0.75`) — `sigma / bin_width`
  (Farebrother-optimal; the CAT-39 spec says `sigma ~ bin width`).
- `--value-target-lambda` (default `1.0`).

Enable the head on an existing checkpoint with
`f69_upgrade_checkpoint_config.py --flags catbins:33 --seed <fixed-seed>`
(additive and deterministic; the scalar
`value`/`final_vp`/`q` outputs stay bit-identical, so the forward-identity
assertion still holds). The upgrade records the source checkpoint digest and
construction seed, but deliberately does **not** add categorical training
provenance; only a successful categorical-primary training run may do that.

## Recommended bins / sigma

31–64 bins over `[-1, 1]`; the reference arm uses **33 bins** (`bin_width =
2/32 = 0.0625`) with `sigma_ratio = 0.75` (`sigma ≈ 0.047`). More bins → finer
readout resolution at a modest logit-width cost.

## Decision probes, not a full production run

The local R&D plan owns the experiment contract. It deliberately separates:

1. **A0 mechanism replication:** the exact historical gen2B/gen1-init scalar
   recipe versus an otherwise matched HL-Gauss arm. This asks only whether
   classification fixes the documented three-epoch reuse failure; it does not
   tune a production recipe.
2. **A1 fresh-data candidate:** game-level held-out validation, value budget
   `0.25`, value-head LR multiplier `0.3`, and the winning readout on a fresh
   curated window. This is the first candidate eligible for strength gates.
3. **Conditional scale stress:** the ~87.85M model is not adopted until the 35M
   categorical model passes A0/A1 and at least 10M fresh diverse rows exist.

Both A0 arms use an explicit validation game-seed manifest and
`--no-resume-optimizer`, so a scalar source checkpoint's optimizer sidecar
cannot make the supposedly matched arms different. See
`docs/plans/RL_RND_EXECUTION_PLAN_20260709.md` and
`docs/plans/RL_ARCHITECTURE_SCALE_PROBE_20260709.md` for the exact commands,
hashes, gates, and stop conditions.
