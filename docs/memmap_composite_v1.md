# No-copy mixed memmap curriculum

`train_bc.py --data-format memmap` accepts either the historical corpus
directory or an explicit `memmap_composite_v1` descriptor. The descriptor is an
ordered, two-component view: payload files remain in their original corpus
directories, while the existing epoch-wide row permutation shuffles over the
sum of both row counts.

```json
{
  "schema_version": "memmap_composite_v1",
  "diagnostic_only": true,
  "promotion_eligible": false,
  "learner_recipe_overrides": {
    "per_game_policy_weight": true,
    "per_game_policy_weight_mode": "equal"
  },
  "learner_recipe_overrides_sha256": "sha256:...",
  "components": [
    {
      "corpus_dir": "/absolute/n256.memmap",
      "corpus_meta_sha256": "sha256:...",
      "payload_inventory_sha256": "sha256:...",
      "validation_manifest": "/absolute/n256.validation_seeds.json",
      "validation_manifest_sha256": "sha256:..."
    },
    {
      "corpus_dir": "/absolute/n128.memmap",
      "corpus_meta_sha256": "sha256:...",
      "payload_inventory_sha256": "sha256:...",
      "validation_manifest": "/absolute/n128.validation_seeds.json",
      "validation_manifest_sha256": "sha256:..."
    }
  ]
}
```

Paths must be canonical absolute paths. Each component's metadata, payload
inventory, and exact A1 validation binding are checked independently. Validation
game sets and complete component game-seed sets must be disjoint. Do not pass
`--validation-game-seed-manifest`; the descriptor binds both manifests.
The command's per-game policy settings must exactly match the canonical
`learner_recipe_overrides` object and digest. Diagnostic descriptors may also
bind the supported LR, value weighting, and value-objective fields; every field
present is compared to the effective command before data loading. This
explicitly authorizes the diagnostic learner drift without weakening
single-contract A1 runs.

The corrected mixed value-objective probe is prepared with
`tools/a1_mixed_value_objective_probe.py`. It accepts an LR verdict only from
`6e-5`, `1.2e-4`, or `2.4e-4`, binds the ordered n256+n128 corpora and their
validation manifests without copying rows, and emits matched MSE and 33-bin
HL-Gauss arm plans. Both use world8/global-batch4096, loser weight 1, equal
per-game policy mass, sqrt per-game value mass, and forced-row value weight
0.1. Preparation is inert; GPU execution requires the separate `--go` flag.

The current schema is deliberately diagnostic-only. Existing promotion receipts
bind one A1 learner contract, one selected-game identity, and one payload
inventory. Promotion support therefore requires a new contract/receipt schema
that binds the ordered component list, its canonical descriptor fingerprint,
the union validation identity, and the mixed learner recipe. Until that adapter
exists, the trainer records `diagnostic_only=true` and
`promotion_eligible=false` and cannot emit a promotion-eligible mixed run.
