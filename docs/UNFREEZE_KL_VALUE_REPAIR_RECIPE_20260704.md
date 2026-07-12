# Unfreeze-with-KL Value-Repair Recipe (contingency f67, D1)

> **Historical diagnostic only.** Commands below preserve the old
> `loser_sample_weight=0.3` experiment for exact replay. That outcome-conditioned
> policy setting is obsolete and is not a production recommendation; current
> MCTS distillation uses `1.0`.

**Status:** LAND-READY contingency, PREP ONLY. Not launched. Runs on the f67
branch's `tools/train_bc.py`. The team lead decides whether/when to launch based
on the running H2H arm verdicts.

## Why this exists

Value-repair v2 (task #65) retrained ONLY the value head on a frozen imitation
trunk (`--train-value-only`): trunk + action_encoder + policy_head frozen, a
linear value head fit to true outcomes. Calibration improved
(corr(q,z) 0.514->0.683, E[q|loss] -0.003->-0.472), but external reviews
converged on a structural concern: **a linear head on a frozen trunk may lack
the features to separate near-tied openings.** If the trunk was trained purely
to imitate a policy, its penultimate features encode "what move looks good," not
necessarily "who is winning and by how much" -- so no linear reweighting of them
can fix a per-position value BIAS at wide placement roots (which the SNR arms
proved is the real failure mode: more search of a biased value converges
confidently to the WRONG placement).

The fix: **unfreeze the whole trunk and let the value loss reshape the shared
features**, while a policy-KL anchor to the frozen seed keeps the policy from
drifting off a distribution we already trust. This is the standard "don't let a
value objective silently destroy a good policy" guardrail.

## The mechanism (implemented in this branch)

New flag `--policy-kl-anchor-weight W` (default 0.0 = disabled, pure no-op) adds

    loss += W * mean_over_prior_rows( KL(pi_theta(s) || pi_seed(s)) )

where `pi_seed(s)` is the seed checkpoint's per-state prior distribution. **We do
not recompute it on-the-fly and we do not regenerate the corpus** -- the raw
self-play corpus already stores it: `raw_selfplay.py::_build_raw_decision_row`
writes `prior_policy` = the seed evaluator's raw prior at every state (there is no
search to "improve" it, so target==prior by construction there). Confirmed
present and populated for all 10.1M rows (the `prior_policy` column, float16).
This is the memory-sane option under the current in-RAM loader: zero extra
forward passes, zero extra columns, no second model resident in GPU.

The anchor reuses the EXACT per-row computation the existing `prior_kl` success
telemetry reports (`_prior_kl_telemetry`), un-detached, via
`_policy_kl_anchor_loss`. So the run's `prior_kl_ratio` /
`prior_kl_model_prior_mean` telemetry directly measures how hard the anchor is
binding -- no new instrumentation needed. The masked mean flows through the
DDP-correct `_weighted_mean_loss` (global-denominator all-reduce), so it behaves
under torchrun exactly like the value loss.

Gradient behavior: because the corpus rows carry `policy_weight_multiplier=0.0`
(raw argmax is not an imitation target), the ordinary policy cross-entropy loss
contributes nothing. The KL anchor is therefore the ONLY thing constraining the
policy while the value loss reshapes the trunk -- exactly the intended design.

### Verification done (f67, pre-launch)
- Unit tests (`tests/test_train_bc_policy_kl_anchor.py`, 5 tests): the anchor
  equals the mean of the telemetry's per-row KL over prior rows, is
  differentiable through the logits with nonzero gradient, returns None (adds
  nothing) when no prior rows are present, and is ~0 when model==prior.
- End-to-end CPU smoke on a fresh 24-game raw-selfplay corpus: training completes,
  loss finite, telemetry present. A/B: validation KL(model||prior) is 2.4e-5 with
  `--policy-kl-anchor-weight 1.0` vs 1.6e-4 with 0.0 (~6.7x lower) -- the anchor
  measurably pulls the policy onto the seed prior. With the real seed
  init-checkpoint the policy STARTS at the prior, so the anchor holds it there
  while the trunk moves for value -- the intended, even cleaner behavior.
- All pre-existing `train_bc` tests (prior_kl telemetry, value-only, truncated-vp,
  value-phase-weights) still green: the 0.0 default is bit-identical to before.

## STAGED COMMAND (verbatim -- do not launch without team-lead sign-off)

Corpus lives in the MAIN checkout: `/home/ubuntu/catan-zero/runs/raw_selfplay_gen1_subset10m`
(the f67 worktree's `runs/` is gitignored/empty). Seed checkpoint = the same
init used by value-repair v2 attempt-3, so the "raw hard-target policy" H2H side
is unchanged. Recipe knobs are the v2 recipe (value_loss_weight 0.25,
final_vp_loss_weight 0.05, truncated-vp-margin 0.25) plus lr 1e-4, warmup 5%
(=117 steps at this corpus/batch size, matching attempt-3), MINUS `--train-value-only`
(full trunk) PLUS `--policy-kl-anchor-weight 1.0` and `--policy-loss-weight 0.0`.

```bash
cd /home/ubuntu/catan-zero && \
PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python tools/train_bc.py \
  --data runs/raw_selfplay_gen1_subset10m \
  --init-checkpoint runs/bc/entity_graph_35m_oldbase_hardtarget_ab45_robber_opening_20260630_220320/checkpoint.pt \
  --checkpoint runs/bc/entity_graph_35m_value_repair_v3_unfreeze_kl_20260704/checkpoint.pt \
  --report   runs/bc/entity_graph_35m_value_repair_v3_unfreeze_kl_20260704/report.json \
  --arch entity_graph --track 2p_no_trade --vps-to-win 10 \
  --hidden-size 640 --graph-layers 6 --attention-heads 8 --graph-dropout 0.05 \
  --epochs 1 --batch-size 4096 --amp bf16 --optimizer adam --weight-decay 0.0 \
  --lr 1e-4 --lr-warmup-steps 117 \
  --policy-loss-weight 0.0 \
  --policy-kl-anchor-weight 1.0 \
  --value-loss-weight 0.25 --final-vp-loss-weight 0.05 --q-loss-weight 0 \
  --truncated-vp-margin-value-weight 0.25 \
  --forced-action-weight 0.1 --winner-sample-weight 1.0 --loser-sample-weight 0.3 \
  --require-35m-model --trust-curated-data-quality --skip-teacher-quality-gate \
  --validation-fraction 0.05 \
  --validation-game-seed-ranges "5006335:5006667,5106335:5106667,7006335:7006667,7106335:7106667" \
  --device cuda:0
```

Notes for the operator:
- Keep the SAME `--validation-game-seed-ranges` as attempt-3 so the calibration
  probe / holdout comparison is apples-to-apples.
- Attempts 1-2 of the frozen v2 died on the padded-concat loader memory blow-up
  (~45GB/M rows). Full-trunk training uses the same loader, so run it on a
  large-RAM host or wait for task #66's streaming/memmap loader (then add
  `--data-format memmap` pointed at a memmap slice). On the B200 (183GB) the 10.1M
  subset fit for the frozen run; the full-trunk run has the same corpus footprint
  (the extra grad memory is on the GPU, not host RAM), so it should also fit.
- Success telemetry: watch `prior_kl_ratio` in the metrics. With the anchor
  binding, `prior_kl_model_prior_mean` should stay LOW (policy close to seed)
  while `value_loss` drops -- the whole point is value improves without the policy
  wandering. If `prior_kl_model_prior_mean` climbs, the value loss is dragging the
  policy and the anchor weight should go up (try 2.0-4.0).
- Post-run: `tools/value_repair_calibration_probe.py` on the new checkpoint
  (Step 1 of the re-validation protocol), then the 3 H2H arms.
