# Value validation-measure audit (2026-07-15)

## 2026-07-16 correction

The component -> game -> row aggregation described below matches the composite
sampler, but it does not match the complete learner objective when
`value_player_outcome_balance_mode=sampler_balanced_v1`: that transform is fit
on training rows only, while validation intentionally retains natural outcomes.
New reports therefore emit this aggregate as `validation_natural_composite`
with `objective_matched=false`. Objective-matched consumers fail closed instead
of treating it as promotion evidence. Historical numbers below are retained as
evidence about row-mix correction, not as proof of full objective matching.
The corrected wrapper/provenance v3 contract digest-binds the full
`objective_match` declaration and its exact report key; v2 remains a
declaration-free historical compatibility path only.

## Verdict

Composite training already emitted an authenticated validation aggregate under
`validation_objective_matched`, but two science-facing consumers still read the
raw row-concatenated compatibility metric. That can misstate the value signal
because the raw held-out row mix is not the learner's component -> game -> row
sampling measure and does not apply the same value-weight density.

The fix makes the Gen2B probe and dual-arm epoch receipts prefer the matched
measure whenever it is present, require it consistently across all epochs, and
fail closed on malformed matched wrappers. Historical non-composite reports
retain their raw-validation fallback. Raw metrics remain in training reports as
calibration and compatibility diagnostics; they are not promotion evidence.

## Read-only B200 evidence

The four-arm reports under
`b200_lr_dose_four_arm_r5/arms/{A,B,C,D}/train.report.json` in the 2026-07-15
evidence bundle contain both measures over the same 840,754 held-out rows:

| Arm | Raw row-concat value MSE | Objective-matched value MSE | Raw inflation |
| --- | ---: | ---: | ---: |
| A | 0.5827035 | 0.5430040 | 0.0396995 (7.31%) |
| B | 0.5813570 | 0.5419991 | 0.0393580 (7.26%) |
| C | 0.5833706 | 0.5445782 | 0.0387924 (7.12%) |
| D | 0.5837293 | 0.5450497 | 0.0386796 (7.10%) |

Arm B exposes the mechanism directly. Forced `ROLL`/`END_TURN` states are
49.8576% of raw validation rows but only 16.4299% of effective value mass after
the configured action weights (`ROLL=0.25`, `END_TURN=0.1`). Raw row counting
therefore represents forced states at 3.03x their optimizer mass. This is a
measure mismatch, not evidence that the network learned a worse value function.

The B200 was treated as read-only; no job or mutation was performed there.

## Consumer inventory

- `train_bc` intentionally preserves `validation` as `raw_row_concat` and
  emits `validation_objective_matched` for authenticated composite reports.
- `a1_b200_batch_probe` and `a1_n256_lr_adjudicate` already require the central
  objective-matched selector for scientific comparisons.
- `a1_b200_microbatch_quality` delegates its quality-floor summary to
  `a1_b200_batch_probe`, so its named validation loss fields are matched too.
- `a1_one_dose_train` and `a1_production_temperature_replication` already
  authenticate the matched wrapper and its per-component coverage.
- `a0_gen2b_probe` now selects matched epoch metrics for science traces while
  retaining raw metrics only for the explicitly historical trace contract.
- `a1_dual_arm_train` now records matched epoch validation in receipts and
  labels the chosen measure. A partially matched multi-epoch report is rejected.
- `a1_corrective_196k_b200.sh` and `reanalyze_lite` continue reading raw values
  only for explicitly diagnostic output. `legacy_scalar_readout_attestation`
  seals the raw telemetry of one exact pre-composite historical artifact. None
  of these paths uses the value to rank a modern candidate or authorize
  promotion.

## Locked invariant

For any report that contains `validation_objective_matched`, a decision consumer
must validate schema `composite-validation-measure-v2`, require
`objective_matched=true`, and use its `metrics`. Silently falling back to raw
validation after a malformed matched wrapper would make corrupted evidence look
legacy, so the consumers fail closed instead.
