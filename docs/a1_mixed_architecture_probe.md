# Mixed action-relational architecture probe

`tools/a1_mixed_architecture_probe.py` seals a diagnostic-only matched A/B on
the existing ordered n256+n128 no-copy corpus. The baseline retains the dense
Transformer entity trunk. The treatment changes only the declared architecture
surface: RRT incidence encoding, action-target gather, and one action-to-board
cross-attention layer. CAT-97's direct edge-policy head is explicitly disabled
in both arms so it is not an undeclared fourth intervention.

The completed architecture-data audit is mandatory input. It must authenticate
both corpus paths, valid legal-action targets and graph incidence IDs, and zero
active event rows/targets. The launcher therefore records the event path as
excluded; this is not an event-relation experiment.

Preparation requires an explicit LR verdict and positive optimizer-step budget:

```bash
python tools/a1_mixed_architecture_probe.py \
  --lr 1.2e-4 --max-steps STEP_BUDGET \
  --n256-corpus "$ROOT/n256-early/n256.memmap" \
  --n256-validation "$ROOT/n256-early/n256.validation_seeds.json" \
  --n128-corpus "$ROOT/n128/n128.memmap" \
  --n128-validation "$ROOT/n128/n128.validation_seeds.json" \
  --initialization-checkpoint "$SHARED_INITIALIZATION" \
  --architecture-audit /tmp/architecture-target-audit.json \
  --output-root "$ROOT/training/mixed-relational-architecture"
```

This writes the sealed descriptor, experiment manifest, and per-arm command
plans without starting training. Execution remains unavailable unless the same
command is rerun with the explicit `--go` flag after GPU scheduling. Both arms
use world8, local batch 512, global batch 4096, the same warm-start source,
seed, data, validation split, optimizer, LR, and maximum optimizer steps. Every
checkpoint/report receives a separate diagnostic receipt; neither arm is
promotion eligible.
