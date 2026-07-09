# f69 v3b finetune — mechanical launch note

Status: launch runbook for the staged v3 experiment. v3b = **the identical v3a
recipe with the f69 action-attention flags ON**, so its purpose is to attribute
the architecture's contribution against v3a (same recipe, current architecture).
Both are evaluated on the 200-root panel (f70) + `tools/f69_ranking_probe.py` +
calibration; the winner goes to a 16-pair H2H.

## The one thing that is not mechanical in train_bc (and the fix)

`tools/train_bc.py` has **no CLI argument** for the new `EntityGraphConfig`
fields (`action_target_gather`, `action_cross_attention_layers`,
`value_attention_pool`). Its `--init-checkpoint` path for `entity_graph` calls
`EntityGraphPolicy.load(...)`, which rebuilds the module from the checkpoint's
**own pickled config** — so pointing v3a's command at the base checkpoint would
silently keep the flags OFF. Do **not** rely on a flag CLI that does not exist.

The fix requires **no change to train_bc**: pre-upgrade the config *in the init
checkpoint*, then run the identical v3a command against the upgraded checkpoint.
`train_bc`'s loader then reads the upgraded config (flags ON), builds the
upgraded module, and loads the weights cleanly (the new zero-init params are
already in the checkpoint, so strict reload has empty missing/unexpected —
verified below). The arch-mismatch preflight (`_checkpoint_config_mismatches`)
only compares `hidden_size/state_layers/attention_heads/dropout`, all unchanged,
so it passes.

## Step 0 — produce the upgraded-config init checkpoint (once)

Apply to the **same init checkpoint v3a starts from** (NOT v3a's output), so both
arms share the identical recipe and init:

```bash
PYTHONPATH=/home/ubuntu/catan-zero-f69/src \
CUDA_VISIBLE_DEVICES=<idle> \
/home/ubuntu/catan-zero/.venv/bin/python \
  /home/ubuntu/catan-zero-f69/tools/f69_upgrade_checkpoint_config.py \
  --in-checkpoint  <V3A_INIT_CHECKPOINT>.pt \
  --out-checkpoint <V3A_INIT_CHECKPOINT>.f69flags.pt \
  --flags gather,cross:2,value \
  --device cuda
```

`--flags gather,cross:2,value` sets `action_target_gather=True`,
`action_cross_attention_layers=2`, `value_attention_pool=True`. The script
asserts the upgraded model is forward-identical to the input at init
(`forward_max_diff == 0.0` on a real 54-wide placement root) before writing —
this is the warm-start guarantee, so v3b and v3a start from the same function.

Verified on the current 35M checkpoint (value_repair_v2_raw_selfplay_20260704):
upgraded checkpoint loads with flags `(True, 2, True)`, dims `640/6/8/0.05`, and
a fresh upgraded module reloads it with `strict=True` → missing `[]`,
unexpected `[]`; `forward_max_diff = 0.0`.

## Step 1 — run the identical v3a command, swapping only the init checkpoint

v3b is v3a's command with exactly two edits: `--init-checkpoint` points at the
`.f69flags.pt` from Step 0, and the run name/`--checkpoint`/`--report` get a
`v3b`/`f69flags` suffix. Everything else — corpus (`--data`), epochs, `--lr`,
loss weights, the f67 unfreeze-with-KL settings, the f72 masked-input settings,
`--hidden-size 640 --graph-layers 6 --attention-heads 8 --graph-dropout 0.05`,
seeds — is byte-for-byte the v3a command. Concretely:

```bash
# v3a (owned by the f67/f72 launch):
#   ... train_bc.py --arch entity_graph --init-checkpoint <V3A_INIT>.pt \
#       --checkpoint runs/bc/..._v3a/checkpoint.pt --report ..._v3a.json  <recipe flags>
#
# v3b == the SAME line with:
#   --init-checkpoint <V3A_INIT>.f69flags.pt
#   --checkpoint runs/bc/..._v3b/checkpoint.pt --report ..._v3b.json
```

## CRITICAL recipe caveat for the f67 owner — do the new params train?

`--freeze-modules` / `--train-value-only` freeze by **named groups**: `trunk`,
`action_encoder`, `policy_head` (and their submodules: hex/vertex/edge/player/
global/event encoders, blocks, state_norm, action_encoder, action_bias,
logit_scale). The f69 params are **not** in any of those groups, so under a
value-only freeze they would be **left trainable** while the trunk is frozen:

- `target_gather_proj.*`, `action_cross_blocks.*` — action-path params. Trainable
  under a value-only freeze; they feed both the policy (cosine logits) and q_head.
- `value_probe*`, `value_pool_head.*` — value-path params. Correctly trainable
  for a value repair.

v3 is the **unfreeze-with-KL** recipe, so the intent is presumably that all params
train under the KL anchor — in which case the f69 params train as desired and the
KL keeps the (initially identity) action paths from drifting the policy. But the
owner must decide explicitly:
1. If the KL/behavior anchor is computed against the pre-finetune policy, the
   zero-init action paths start as exact identity, so the anchor is well-defined
   and the new paths are regularised into the policy gradually — recommended.
2. If any freeze is applied, confirm whether the f69 action-path params should be
   frozen too (they are not matched by the current group names). If they should be
   frozen, `--freeze-modules` needs the new group names added — a small train_bc
   change, flagged here rather than assumed.

## Post-finetune verification (same harness for both arms)

```bash
tools/f69_ranking_probe.py --checkpoint runs/bc/..._v3b/checkpoint.pt \
  --flags gather,cross:2,value --n-states 40 --out reports/f69_probe_v3b.json
```

Compare `prior_spread`/`q_spread` against v3a (base flags OFF) and against the
pre-finetune numbers this note's Step 0 preserved (top1-top2 gap 0.0011, q range
0.037). A successful v3b should widen the q/prior spread at the opening roots
without hurting calibration or the 200-root panel; that is the attribution gate.

## Telemetry to watch in v3b (confirmed with the v3a owner)

The staged v3a/v3b recipe runs the FULL trunk unfrozen (`--policy-loss-weight
0.0`, no `--train-value-only`, no `--freeze-modules`), so the freeze-group caveat
above does NOT bite either arm as staged — all f69 params are trainable by
default and no `--freeze-modules` edit is needed. It would only matter if a freeze
were added to either run.

The KL policy anchor (`KL(pi_theta || prior_policy)`) keeps the newly-live action
attention from drifting the policy while the value loss reshapes features. Because
Step 0 asserts forward-identity at init, `pi_theta` at v3b init == the seed policy
== the distribution behind the `prior_policy` column the anchor targets, so the
anchor starts at the same near-zero KL as v3a. The anchor constrains the output
distribution, not individual weights — so the action-cross-attention params can
still move to improve internal representations as long as the policy stays
anchored (exactly the intent: features improve for value, policy does not wander).

Watch `prior_kl_model_prior_mean` in the v3b telemetry: if the unfrozen action
attention lets the policy drift despite the anchor, it climbs above the v3a
trajectory. Same knob as v3a — bump `--policy-kl-anchor-weight` from 1.0 toward
2.0-4.0. The value-path f69 params (`value_probe*`, `value_pool_head.*`) are driven
by the value loss and need no anchor. (If the merge session decides v3b should
also train the uncertainty head, the v3a owner's `--value-uncertainty-loss-weight`
is orthogonal to the f69 flags and stacks without conflict.)
