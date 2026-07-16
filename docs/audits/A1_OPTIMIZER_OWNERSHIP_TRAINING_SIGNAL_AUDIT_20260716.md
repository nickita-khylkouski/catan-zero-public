# A1 optimizer ownership and value-signal audit — 2026-07-16

## Verdict

The current learner has one concrete optimizer-ownership bug and one important
value-learning semantic trap:

1. `public_rule_state_residual` is a shared, policy-affecting input adapter, but
   it is not owned by the canonical `trunk` group. It therefore escapes
   `trunk_lr_mult`, `--train-value-only` freezing, and shared-trunk gradient
   telemetry.
2. `value_lr_mult` scales value-specific readout/tower parameters only. It does
   not scale value gradients entering the shared trunk. Historical
   `value_lr_mult=0.3` experiments therefore did not test the intended
   "protect the shared policy representation from value learning" hypothesis.

These are training-signal problems, not reporting nits. The first allows a
nominal value-only repair to change policy. The second can simultaneously
undertrain the value readout and leave the dominant shared representation
exposed to full-strength, frequently conflicting value gradients.

Machine-readable evidence is in
`docs/evidence/A1_OPTIMIZER_OWNERSHIP_TRAINING_SIGNAL_AUDIT_20260716.json`.

## Finding 1 — public rule-state residual escapes optimizer ownership

The model creates a zero-initialized `PUBLIC_RULE_STATE_FEATURE_SIZE -> hidden`
linear adapter in `src/catan_zero/rl/entity_token_policy.py` and adds its output
to the global token before the shared Transformer trunk.

The adapter is absent from
`ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["trunk"]` in `tools/train_bc.py`.
That same list is the source of truth for:

- `trunk_lr_mult` parameter assignment;
- `--train-value-only` trunk freezing;
- objective-gradient interference telemetry.

An omitted adapter falls into the optimizer's full-rate `base` group.

### Direct reproduction

With hidden size 384, base LR `1.2e-4`, and `trunk_lr_mult=0.1`:

- adapter parameters: 3,072, initialized to exactly zero;
- actual optimizer group: `base`;
- actual LR: `1.2e-4`;
- intended trunk LR: `1.2e-5`;
- LR error: 10x.

After applying the repository's value-only freeze:

- `public_rule_state_residual.weight.requires_grad` remained true;
- one synthetic value-only optimizer step produced gradient L1 `0.10997`;
- maximum adapter update was `1.2e-4`;
- maximum policy-logit change was `1.12e-5`.

Thus a run labeled value-only can alter the policy through a shared feature
path, and the telemetry intended to detect shared policy/value interference
does not include that path.

### Required repair

- Add `public_rule_state_residual` to the canonical trunk ownership surface.
- Ensure it is covered by trunk LR assignment, value-only freezing, and
  objective-gradient telemetry.
- Add optimizer-assignment and value-only policy-invariance coverage. The
  existing feature test proves zero initialization and trainability, but not
  optimizer ownership.

## Finding 2 — `value_lr_mult` does not protect the shared trunk

The trainer explicitly reports:

```text
value_lr_mult_scales_shared_trunk = false
```

`value_lr_mult` owns the value-specific tower/readout modules. Value gradients
reaching the shared representation are controlled separately by
`value_trunk_grad_scale`.

In the 35M B200 trust-recovery report:

- forward-active parameters: 33,407,431;
- Transformer blocks alone: 29,541,120 parameters;
- value head/tower telemetry group: 410,881 parameters.

Therefore `value_lr_mult=0.3` slowed a comparatively small value-specific
surface while leaving value-derived gradients in the dominant shared blocks
at their normal optimizer LR.

### Existing B200 evidence

All four coherent active-policy arms used:

```text
value_lr_mult = 0.3
value_trunk_grad_scale = 1.0
```

Across their recorded shared modules:

- mean policy/value gradient cosine ranged from `-0.0575` to `+0.0197`;
- 41.2% to 67.6% of observations had negative cosine;
- mean value-gradient norm was 0.514x to 0.891x the policy-gradient norm.

The trust-recovery arms changed only the causal trunk routing to:

```text
value_trunk_grad_scale = 0.25
```

Their mean value/policy gradient norm ratio fell to 0.175x–0.192x. This is the
control that actually reduced value pressure on the shared representation;
`value_lr_mult=0.3` alone did not.

The 60 µLR trust arm still showed:

- shared-block mean parameter delta: `0.0841223`;
- value-head/tower mean parameter delta: `0.00346031`;
- 50% of recorded shared-module gradient cosines negative.

This does not prove that 0.25 is the final production value. It proves that old
"lower value LR" experiments did not isolate the hypothesis they were named
for.

### Required semantic change

Treat these as separate controls in every recipe and report:

- `value_lr_mult`: speed of value-specific parameters;
- `value_trunk_grad_scale`: value objective's causal gradient into shared
  representation;
- `trunk_lr_mult`: optimizer LR for shared parameters from all objectives.

For short-dose recovery, select the shared-trunk setting using parent-policy
KL, layerwise drift, and policy/value gradient conflict. Do not infer shared
representation protection from `value_lr_mult`.

## Priority

The ownership escape should be fixed before interpreting any value-only,
reduced-trunk-LR, or public-rule-state commissioning experiment. After that,
the next training recipes should explicitly bind `value_trunk_grad_scale` or a
split late value tower rather than relying on `value_lr_mult` as a proxy.

