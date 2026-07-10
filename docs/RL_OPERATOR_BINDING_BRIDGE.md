# RL operator-binding bridge

The A1 handoff normally requires replayable experimental decisions for S1,
S2, and S3. The current wave has a narrower operator directive:

- retain the replayed S1 result (`c_scale=.03`, D1 off, `sigma_eval=.98`, D6
  enabled at the inclusive `>=20` gate);
- use global `n_full=128`, `n_fast=16`, and `p_full=.25` without running the
  canceled n64-versus-n128 strength experiment;
- keep adaptive n256 disabled (`n_full_wide=null`, threshold `null`,
  `wide_roots_always_full=false`) without running S3.

`tools/search_operator_binding.py` records that directive without relabeling it
as experimental evidence. It consumes an S1 decision that must replay exactly
through `search_teacher_adjudicator.py`, then emits two immutable
`rl-rnd-operator-binding-v1` artifacts. Each artifact:

- states that it is an operator choice, not strength evidence;
- records the explicit operator label, exact operator fields, reason, and UTC
  binding time;
- binds the S1 decision path, file SHA-256, and selected-field digest;
- binds the emitter's own bytes;
- carries a content digest computed over every other field;
- is created read-only and never overwrites an existing file.

The S3 hold also binds the exact S2 operator-binding file. The A1 validator
accepts this schema only for S2 and S3, only with the literal n128/no-adaptive
values above, and only when all lineage, emitter, timestamp, and self-digest
checks pass. Ordinary `rl-rnd-stage-decision-v1` evidence still follows its
existing in-process experimental replay path unchanged.

Example (CPU only; this does not seal or launch anything):

```bash
python tools/search_operator_binding.py \
  --s1-decision runs/rl_rnd_20260709/search_calibration_d6_t20_d600/adjudication/s1.decision.json \
  --s2-out runs/rl_rnd_20260709/operator_bindings/s2.n128.binding.json \
  --s3-out runs/rl_rnd_20260709/operator_bindings/s3.no_adaptive.binding.json
```

These files authorize configuration lineage for this wave. They do not show
that n128 beats n64, do not show that adaptive n256 is weak, and must not be
cited as Elo, SPRT, cost, or stability evidence.
