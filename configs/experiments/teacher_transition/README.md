# V6 checkpoint as coherent-n128 R&D teacher

This directory does not change the production champion or generation recipe.
It defines the evaluator/search surface used when a selected V8 architecture
checkpoint, trained with the V6 entity adapter, is commissioned as a
non-promotable Stage-C reanalysis teacher.

Create the immutable host-local binding after the selected checkpoint exists:

```bash
python tools/a1_rd_teacher_transition.py bind \
  --checkpoint /absolute/path/to/selected-v8-v6.pt \
  --base-operator-contract \
    configs/operations/a1-target-identity-coherent-n128-v6-history64-rd-v2/contract.json \
  --typed-generation-config \
    configs/experiments/teacher_transition/coherent_public_n128_v6_history64_teacher.schema22.json \
  --binding-id v8-v6-selected-coherent-n128-r1 \
  --output /absolute/path/to/v8-v6-selected.teacher-binding.json
```

Pass the resulting binding as `--target-operator-contract` and the same
checkpoint as `--target-checkpoint` to
`tools/a1_stage_c_teacher_alignment.py plan`.

The binding is deliberately not accepted by the fleet generation executor. It
has no seed schedule and carries `production_authority=false`,
`promotion_eligible=false`, and `diagnostic_only=true`. Stage-C fails before
search construction unless the checkpoint bytes, checkpoint-declared adapter,
typed teacher adapter, learner row adapter, meaningful-history schema, and
64-event limit are all the same exact V6 contract.

The schema-13 history64 recipe and the older 32-event schema-13 recipe remain
immutable evidence of pre-separated RNG operators. New V5 Stage-C execution
must use the schema-22 history64 recipe and v2 base contract above.
