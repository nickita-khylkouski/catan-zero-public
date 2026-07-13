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

## Post-promotion c-scale continuity

A promoted producer can carry a deployed search identity that is newer than
the historical S1 calibration. The current v5 producer is committed at
`c_scale=.10`, while the replayable legacy S1 selected `.03`. Reverting the
next generation wave to `.03` would run the promoted checkpoint under a
different agent identity. Treating `.10` as a new strength result would be
equally wrong.

The post-promotion mode records that narrow continuity decision with an
immutable `rl-rnd-post-promotion-s1-operator-binding-v1` artifact. It:

- semantically replays the exact legacy S1 and requires its `.03` control;
- semantically replays the committed post-promotion handoff and authenticates
  its producer checkpoint, agent-identity digest, and full search-config
  digest;
- requires the deployed handoff to select `.10` and requires every other S1
  field to equal the legacy S1 result;
- states explicitly that the projection is an operator choice, not strength
  evidence;
- binds its emitter bytes and carries a self-digest;
- emits fresh S2/S3 bindings that source the exact new S1 bytes.

Example (CPU only; creates three read-only artifacts and launches nothing):

```bash
python tools/search_operator_binding.py \
  --legacy-s1-decision runs/rl_rnd_20260709/search_calibration_d6_t20_d600/adjudication/s1.decision.json \
  --post-promotion-handoff /absolute/path/to/a1-post-promotion-handoff.json \
  --emitter-path /absolute/immutable/path/to/tools/search_operator_binding.py \
  --s1-out /absolute/path/to/s1.post-promotion.binding.json \
  --s2-out /absolute/path/to/s2.n128.binding.json \
  --s3-out /absolute/path/to/s3.no-adaptive.binding.json
```

The A1 consumer accepts this S1 schema only in a v3 contract whose
`promotion_handoff.path` is exactly the handoff bound by the artifact. The
producer checkpoint must also equal the contract producer, the final S1 fields
must equal the projected fields, and ordinary legacy S1/S2/S3 evidence remains
strict. `sync-generation-guard` replays this artifact and writes its exact
path/SHA into the non-default guard receipt; there is no manual `.10` edit
path.

The optional archived emitter is a byte-identity anchor, not an executable
compatibility shim. Replay uses the checked-in implementation for this schema
and requires the bound emitter hash; a future semantic change must introduce a
new schema or an explicit authenticated historical-loader path. Merely keeping
an old emitter file cannot silently change which validation code executes.
