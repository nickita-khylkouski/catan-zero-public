# Scalar value-readout consistency audit (2026-07-15)

## Verdict

The scalar learner/search transform difference is real but intentional and
sealed. The scalar head is a raw linear output trained by MSE; deployed search
uses `tanh(raw * 1.0)` and finally clips to `[-1, 1]`. This is the historical
operator, not an accidental extra transform. The alternative scalar `clip`
operator is an explicit A/B arm, and the existing 1,200-game same-checkpoint
panel did not justify changing the default (`clip` 565, `tanh` 635,
pentanomial H0). Keep `tanh` until a matched strength panel establishes a
superior replacement.

One adjacent fail-open bug was confirmed and fixed: a modern checkpoint whose
`value-training-v1` provenance attests only a categorical readout still
contains the architectural scalar module. The evaluator previously returned
early for `value_readout="scalar"`, so an omitted CLI override could silently
search with that untrained scalar module. Evaluator construction now rejects
scalar selection when modern provenance does not validate scalar training.
Legacy scalar checkpoints remain admitted; their exact checkpoint/report pair
is guarded by the production legacy attestation and pre-wave contract.

No search default, backup equation, training objective, checkpoint bytes,
remote data, or promotion decision was changed by this audit.

## End-to-end evidence

### Training target and head

- The entity model emits the scalar value directly from `value_head`, with no
  activation (`src/catan_zero/rl/entity_token_policy.py:1258-1265`).
- The learner compares that raw output directly with the outcome/blended target
  using per-row MSE (`tools/train_bc.py:14894-14917`). The default
  `value_target_lambda=1` leaves the terminal target unchanged; any configured
  root-value blend is explicit (`tools/train_bc.py:14781-14806`).
- Every modern checkpoint records which readouts actually received optimizer
  updates and positive training mass, plus resolved scalar/categorical weights
  (`tools/train_bc.py:1956-2025`). Architecture presence is therefore not
  training evidence.

Training invariants:

1. Scalar-MSE optimizes the raw scalar, not `tanh(raw)`.
2. A readout is attested only after positive optimizer steps, objective weight,
   and sample mass.
3. HL-Gauss-only training may leave the scalar module present but unoptimized.

### Evaluator transform and perspective

- The configured scalar operator is explicit: `value_readout="scalar"`,
  `value_scale=1.0`, and historical `value_squash="tanh"`
  (`src/catan_zero/search/neural_rust_mcts.py:47-80`). The source documents the
  Jensen/chance-expectation consequence and says a strength A/B is required
  before replacing it.
- `_apply_value_squash` performs scale then `tanh` or identity; categorical
  expectation bypasses scalar tanh (`src/catan_zero/search/neural_rust_mcts.py:519-540`).
  Call sites then apply the two-player opponent sign flip and final clip.
- `_assert_value_readout_available` now treats the loader-validated modern
  readout list as authoritative for both scalar and categorical selection
  (`src/catan_zero/search/neural_rust_mcts.py:126-205`). It preserves the
  explicit legacy-scalar compatibility case.

Evaluator invariants:

1. Exactly one named output key is consumed; there is no cross-head fallback.
2. The selected modern readout must have validated training provenance.
3. Scalar values reach search in bounded root-player perspective.

### Gumbel/chance backup

- Ordinary action backup adds the evaluator value directly to visit sums; it
  does not apply another squash (`src/catan_zero/search/gumbel_chance_mcts.py:2483-2501`).
- Enumerated chance actions back up the probability-weighted mean of already
  transformed child values (`src/catan_zero/search/gumbel_chance_mcts.py:2711-2725`).
- The native implementation has the same direct sum/expectation behavior
  (`native/gumbel_mcts_rs/src/lib.rs:990-1010,1059-1075`).

Backup invariants:

1. The evaluator owns the readout transform; search does not transform again.
2. Chance expectation is `E[deployed child value]`, hence tanh versus clip can
   change magnitudes even though both are monotone at a single root.
3. Python and native hot loops preserve the same transformed-value semantics.

### Calibration and promotion receipts

- Calibration calls the same readout-availability guard before inference
  (`tools/phase_sliced_value_calibration.py:478-511`) and reports raw,
  scalar-tanh, and scalar-clip views separately. Its artifact names the exact
  configured effective transform and declares diagnostics non-mutating
  (`tools/phase_sliced_value_calibration.py:1089-1139`).
- The pre-wave contract binds the requested readout to checkpoint training
  provenance, or to the typed legacy scalar bridge
  (`tools/a1_pre_wave_contract.py:1625-1707`).
- Promotion verifies checkpoint identity, selected/trained readout, optimizer
  evidence, scale, squash, and the exact calibration view
  (`tools/a1_promotion_transaction.py:4714-4750,4870-4899`). A calibration of
  raw values cannot be laundered into a tanh promotion receipt.

Receipt invariants:

1. Learner objective, evaluation readout, and checkpoint provenance agree.
2. Calibration evaluates the deployed transform, not merely the raw head.
3. Promotion evidence must use the same transform for candidate and incumbent.

## Read-only B200 evidence

The following artifacts were inspected on `ubuntu@149.118.65.110` without
launching a job or modifying data:

- `/home/ubuntu/catan-zero/runs/bc/gen3_20260706/checkpoint.pt` has SHA-256
  `89aa133d629e747021bc725f2ad63e0563f3b76e71f0dd563f056c6de8f77ebb`, a
  scalar value head, no categorical bins, and predates embedded
  `value-training-v1` metadata.
- Its immutable report records `value_loss_weight=1.0`, 912 completed steps,
  train value loss `0.23851886025429092`, and validation value loss
  `0.26022181374410436`. These facts and the exact report/checkpoint hashes are
  bound by
  `/home/ubuntu/catan-zero/runs/rl_program_20260709/pre_wave/gen3_legacy_scalar_readout.attestation.json`
  (`legacy-scalar-readout-attestation-v1`).
- A recent learner report,
  `/home/ubuntu/experimental_nonpromotable/coherent-n128-trust-recovery-20260715-r2/campaign/run-8794e56/arms/LOWLR_V25/train.report.json`,
  binds `objective="mse"`, `value_readout="scalar"`, scalar weight `0.25`,
  128 optimizer steps, scalar training mass `524288`, and
  `trained_value_readouts=["scalar"]`.
- The evidence bundle's in-force science contract and champion registry bind
  `value_readout="scalar"`, `value_scale=1.0`, and `value_squash="tanh"`:
  `/home/ubuntu/catan_rl_evidence_bundle_20260715/02_champion_state/science.contract.in_force.json`
  and
  `/home/ubuntu/catan_rl_evidence_bundle_20260715/01_external_panels/champion_registry.from_handoff.json`.

This agrees with the committed direct A/B record in
`docs/audits/A1_STORED_POLICY_TEMPERATURE_WIN_20260712.md`: same-f7 scalar
`clip` versus `tanh` scored 565-635 over 1,200 games, pentanomial H0. Offline
holdout can prefer raw/clip while search strength does not; calibration alone
therefore cannot authorize a default change.

## Experimental decision

No new experiment is needed to decide today's operator: retain scalar-tanh.
Changing it requires a fresh, seat-swapped, common-random-number panel with the
same checkpoint, search configuration, and squash on both roles within each
match; compare matched all-tanh against matched all-clip cohorts. The decision
must be playing-strength evidence, with held-out calibration as a secondary
mechanism diagnostic. Until clip is superior rather than H0/underperforming,
the production contract and promotion receipts should continue to require
scalar-tanh.
