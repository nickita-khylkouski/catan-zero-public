# CAT-12 recommended flag bundle (D1, aux heads, value-loss weight, value-LR, late-game temperature)

Companion to Linear CAT-12. This is a **documentation-only** file: it records the
CLI invocation that turns the five flags on, per the roadmap/master-plan targets.
**No code default changed** as part of this ticket (CLI-default-trap discipline --
every default below is still whatever it was before CAT-12; these flags are OFF /
no-op until a run config passes the overrides listed here).

## 1. `train_bc.py` additions

```
--value-loss-weight 0.25 \          # ALREADY the default (verified); listed for explicitness
--value-lr-mult 0.3 \               # NEW flag (this ticket) -- 0.3x torso LR for value_head/
                                     # final_vp_head/value_uncertainty_head, everything else at --lr
--value-uncertainty-head \          # NEW flag (this ticket) -- only relevant WITHOUT --init-checkpoint
                                     # (builds a fresh model with the aux head present)
--value-uncertainty-loss-weight 0.05 \   # ALREADY existed, was 0.0 (dormant) -- 0.02-0.1 per plan
--final-vp-loss-weight 0.05 \       # ALREADY the default (verified) -- final_vp_head aux weight
```

Notes:
- `--value-loss-weight` and `--final-vp-loss-weight` are **already at 0.25 / 0.05 by
  default** in `tools/train_bc.py` (verified by reading the argparse defaults) --
  both already sit inside the plan's target ranges (0.25-0.5 and 0.02-0.1
  respectively). No override is strictly required; they are listed for an
  explicit, self-documenting invocation.
- `--value-uncertainty-loss-weight` already existed as a flag (default `0.0`,
  i.e. dormant) -- this ticket does not add it, only recommends turning it on.
- `--value-uncertainty-head` and `--value-lr-mult` are **new flags added by this
  ticket** (see `tools/train_bc.py`). `--value-uncertainty-head` only matters
  when training WITHOUT `--init-checkpoint` (a fresh `--arch entity_graph`
  model); resuming from an existing checkpoint uses whatever the checkpoint's
  own saved config already has for `value_uncertainty_head` -- there is
  currently no flag to add the head to an already-built checkpoint that lacks
  it (out of scope here; would need a checkpoint-surgery tool).
- `--value-lr-mult` requires `--arch entity_graph`/`xdim_lite`/`xdim_graph` (the
  model must expose at least one of `value_head`/`final_vp_head`/
  `value_uncertainty_head` as a named submodule) and raises `SystemExit`
  otherwise (e.g. `--arch candidate`).
- Only "aux heads" that currently exist as built model heads are
  `final_vp_head` and the optional `value_uncertainty_head`. The roadmap's
  "road/army, production, belief" aux heads (roadmap line 20, master plan V3/
  §4.6) are **not built yet** -- out of scope for this ticket.

## 2. Generation (`tools/generate_gumbel_selfplay_data.py`) additions

```
--rescale-noise-floor-c 1.0 \       # NEW flag (this ticket) -- D1 noise-floor attenuation
--sigma-eval 0.79 \                 # NEW flag (this ticket) -- D1's noise-floor sigma (placeholder)
--late-temperature-decisions 150 \  # NEW flag (this ticket) -- late-game temperature window end
--late-temperature 0.3 \            # NEW flag (this ticket) -- nonzero temp inside that window
```

Notes:
- **D1 (`rescale_noise_floor_c`) was previously flag-gated on
  `GumbelChanceMCTSConfig` (task #67) but NOT plumbed into any production
  generation entrypoint** -- verified by reading `tools/generate_gumbel_selfplay_data.py`,
  `tools/modal_gumbel_factory.py`, and `tools/modal_gumbel_factory_gpu.py`'s
  `GumbelChanceMCTSConfig(...)` construction sites before this ticket: none of
  them exposed a `--rescale-noise-floor-c`/`--sigma-eval` CLI flag, so there was
  no way to turn D1 on for real generation at all (only for the offline
  calibration tools `opening_panel.py`/`ablate_search_calibration.py`). This
  ticket adds the CLI flags + wiring to `generate_gumbel_selfplay_data.py` (the
  script `tools/continuous_flywheel.py`'s `generate()` actually calls); the two
  `modal_gumbel_factory*.py` variants are UNCHANGED (still no D1 flag) -- flag
  as a follow-up if Modal-fleet generation needs D1 too.
- **`--rescale-noise-floor-c 1.0` is NOT a previously-validated production
  constant.** `tools/ablate_search_calibration.py`'s own `--d1-c` default (also
  `1.0`) carries the identical caveat in its help text: *"NOT a
  previously-validated constant (f70 doc leaves the default disabled at 0.0 and
  flags calibration as future work)"*. The master plan (`docs/plans/CATAN_ZERO_MASTER_PLAN.md`
  line 136) sequences a proper `{c_scale x D1}` re-grid AFTER root-symmetry
  averaging lands, specifically because the `c_scale=0.03` optimum was measured
  at the OLD (pre-D1) noise floor. **Do not launch a real generation batch with
  `--rescale-noise-floor-c 1.0` before that re-grid** -- use it for the CAT-12
  GPU smoke test (crash/optimizer/loss-curve check) only, per the ticket's own
  verification scope.
- **Late-game temperature is new code, not a pure flag-flip.**
  `GumbelSelfPlayConfig` only had a two-stage schedule (`temperature_high` until
  the opening cutoff, then `temperature_low`/argmax for the rest) -- verified by
  reading `src/catan_zero/rl/gumbel_self_play.py`'s `_temperature_for_decision`.
  This ticket adds a third stage (`late_temperature_move_fraction`/
  `late_temperature`, defaulting to `None`/`0.0` = exact no-op) to
  `GumbelSelfPlayConfig` plus the corresponding CLI flags. Per the ticket's own
  step 5, this should stay off in production generation until the Diagnostics
  Bundle ticket's telemetry confirms the diversity-strangulation mechanism it
  targets is actually binding -- the invocation above is for A/B testing, not a
  new production default.
- Scoped to `generate_gumbel_selfplay_data.py` only; `generate_raw_selfplay_data.py`
  (the non-search driver) has its own, separate two-stage
  `--temperature-decisions` schedule and was not touched.

## 3. Per-flag verification summary (see the completion comment on CAT-12 for the full table)

| Flag | Status before this ticket | This ticket |
|---|---|---|
| D1 noise-floor (`rescale_noise_floor_c`) | Built (task #67), flag-gated on the dataclass, but NOT wired into any generation CLI | Added `--rescale-noise-floor-c`/`--sigma-eval` to `generate_gumbel_selfplay_data.py` |
| Aux heads: `final_vp_loss_weight` | Already default 0.05 (nonzero) | No change (already "on") |
| Aux heads: `value_uncertainty_loss_weight` | Existed, default 0.0 (dormant) | No code change; documented recommended override above; added `--value-uncertainty-head` so a fresh model actually has the head |
| Value-loss weight | Already default 0.25 | No change (already in the 0.25-0.5 target range) |
| Value-head LR 0.3x | NOT supported -- single optimizer param group, confirmed by reading `_make_optimizer` | Added `--value-lr-mult` + `_build_optimizer_param_groups` (~90-line patch incl. `_apply_lr_schedule`/`_apply_lr_warmup` per-group fix); unit-tested |
| Late-game temperature | Two-stage schedule only, no window past the opening cutoff | Added `late_temperature_move_fraction`/`late_temperature` to `GumbelSelfPlayConfig` + CLI flags; unit-tested |

## 4. GPU smoke test invocation (per CAT-12's verification scope)

```bash
python tools/train_bc.py --arch entity_graph --data <small-existing-corpus> \
  --checkpoint /tmp/cat12_smoke.pt --report /tmp/cat12_smoke_report.json \
  --max-steps 50 --value-loss-weight 0.25 --final-vp-loss-weight 0.05 \
  --value-uncertainty-loss-weight 0.05 --value-uncertainty-head \
  --value-lr-mult 0.3 --mask-hidden-info
```

Expect in `train.log`/stdout: two distinct optimizer LR values logged at step 0 (base
vs. `base * 0.3`), and `value_loss`/`value_uncertainty_loss` both nonzero and moving
across steps. **This invocation was NOT executed in this environment** -- the
`catanatron_rs` Rust wheel is not installed here (a separate, pre-existing gap; see
the worktree's own background task list), so end-to-end data loading/model
construction can't run locally. Unit tests covering the new code paths (optimizer
param-group construction, LR schedule per-group application, temperature-window
arithmetic, and CLI-to-dataclass wiring) are in `tests/test_train_bc_value_lr_mult.py`,
`tests/test_gumbel_self_play_late_temperature.py`, and
`tests/test_generate_gumbel_selfplay_data_cat12_flags.py`, all passing. Running the
actual GPU smoke test above is a follow-up on GPU hardware before this bundle is
adopted in a real training run.
