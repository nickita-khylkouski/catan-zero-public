# V6/64 coherent-n128 reanalysis operator

This is a new immutable operator identity for V6 checkpoints. It does not
modify or reinterpret the archived V2/32-event generation evidence.

The execution block is retained only because the shared R&D contract inspector
authenticates topology along with search semantics. This contract is not a
self-play launch authority: `self_play_generation_eligible=false`. Consume it
only through `tools/a1_rd_teacher_transition.py`, which binds an exact V6
checkpoint and emits a non-promotable Stage-C reanalysis authority.

```bash
python tools/a1_rd_teacher_transition.py bind \
  --checkpoint /absolute/path/to/selected-v8-v6.pt \
  --base-operator-contract \
    configs/operations/a1-target-identity-coherent-n128-v6-history64-rd-v1/contract.json \
  --typed-generation-config \
    configs/experiments/teacher_transition/coherent_public_n128_v6_history64_teacher.schema13.json \
  --binding-id v8-v6-selected-coherent-n128-r1 \
  --output /absolute/path/to/v8-v6-selected.teacher-binding.json
```
