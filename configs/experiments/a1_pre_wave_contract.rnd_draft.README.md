# A1 pre-wave R&D draft

Status: **near-final and deliberately unsealable**. This file materializes only
choices already supported by the local plans and immutable evidence. It does
not seal or render a contract, create output directories, allocate production
seeds, or launch generation. The checked-in JSON currently has 14 unresolved
fields: the binding S1/S2/S3 search decisions and the synchronized fleet seed
base/ledger. There is also one **conditional pre-seal action** outside that
placeholder count: if typed S1 selects `c_scale=0.1` or `0.3`, the static
generation guard must be synchronized and provenance-stamped before runtime
hashes are frozen. A `c_scale=0.03` selection requires no guard mutation.

## Resolved bindings

- A0's replayed typed decision is `retain_scalar_for_a1`. The learner objective
  and readout are scalar MSE; categorical bins and HL-Gauss sigma are `null`.
- The producer is the masked, 35,041,353-parameter gen3 entity-graph checkpoint
  with scalar search readout. Its read-only legacy scalar attestation binds the
  exact checkpoint and immutable training report without rewriting checkpoint
  bytes; it cannot authorize a categorical readout.
- The source roles are exact: gen3 is the current producer, gen2A is the recent
  promoted history checkpoint, and held gen4 is the hard negative. Gen4's
  209-191 result versus gen3 is evidence that it is challenging, not evidence
  that it was promoted.
- The production regime is two-player/no-trade at 10 VP with public-observation
  masking. `p_full=0.25`, `n_fast=16`, lazy interior chance, corrected Rust
  chance spectra, and `max_decisions=600` are fixed. Exact-budget sequential
  halving, late temperature, belief spectra, uncertainty, raw-policy fallback,
  Rust featurization, and eval-server generation are off.
- The shared output parent is
  `/home/ubuntu/catan-zero/runs/selfplay`. The renderer owns the child layout
  `/home/ubuntu/catan-zero/runs/selfplay/a1-fresh-mixed-12000games/<job_id>`.
  This path choice is operational only; no directory has been created.

## Exact learner dose

`science.learner_training_recipe` is the complete 51-field effective recipe,
not a partial CLI overlay. It is exactly equal to
`tools.a1_pre_wave_contract.EXPECTED_LEARNER_TRAINING_RECIPE` and has canonical
digest
`sha256:1be1a29e44f1742e33bbff8798365a8ef2563438e2b4864160f2180308154655`.
In particular, `graph_history_features=true` is explicit rather than inherited
from a mutable default.

The dose is one epoch on one B200, seed 1, batch/global batch 4096, accumulation
1, BF16, fresh Adam, LR `3e-5`, 100 warmup steps, flat schedule, no weight decay
or fused/resumed optimizer, value LR multiplier `.3`, and action multiplier
`1`. Policy/soft-target/value weights are `1/.9/.25`; soft-target temperature
and legal coverage are `.7/.5`; value lambda is `1`. Categorical, HL auxiliary,
final-VP, Q, policy-KL, uncertainty, subgoal, surprise, advantage, per-game, and
VP-margin objectives are off. Truncated-VP value supervision remains `.25`;
forced action/value weights are `.1/1`; winner/loser weights are `1/.3`.
Masking is on, while DDP sharding and symmetry augmentation are off. Teacher,
phase, value-phase, and freeze overlays are empty.

## Selected games and bounded attempts

The v2 contract separates attempted games from the exact complete-game corpus.
Every seed claim covers all attempts. Postflight then selects the lowest-seed
complete games in each job before row expansion; reserve, truncated,
incomplete, and unselected attempts cannot enter metrics, holdout, or training.

| category | selected/worker | attempts/worker | fleet selected | fleet attempts |
|---|---:|---:|---:|---:|
| current producer | 400 | 408 | 9,600 | 9,792 |
| recent history | 75 | 77 | 1,800 | 1,848 |
| hard negative | 25 | 26 | 600 | 624 |
| **total** | **500** | **511** | **12,000** | **12,264** |

The audit emits the immutable selected-game and validation sidecars. Memmap
ingest is bound to both those sidecars and the passing shard inventory, so the
264 predeclared reserve attempts are excluded before corpus sizing and
statistics rather than filtered later by the trainer. An A1 attestation at a
source or ancestor blocks generic conversion, every resulting flat payload is
content-addressed, and the trainer re-verifies all payload bytes before an
optimizer can be constructed.

The seal binds both the 17-file explicit learner implementation set (trainer,
converter, entity policy/features, config/guard machinery, environment schema,
symmetry, optimizer state, and directly imported policy helpers) and a complete
transitive runtime tree covering every Python module under `src/catan_zero` and
`tools` plus both static guard configs (208 files at the current snapshot).
Training re-hashes that tree and persists its aggregate digest in both report
and `value-training-v1` checkpoint provenance. The mutable shared seed ledger uses
an immutable pre-claim prefix: render emits one exact contract/job claim row,
append-only verification rejects peers/spoofs/duplicates, and post-wave audit
requires all 72 exact claims.

## Checkpoint and evidence identities

| role/artifact | SHA-256 | evidence status |
|---|---|---|
| gen3 producer checkpoint | `89aa133d629e747021bc725f2ad63e0563f3b76e71f0dd563f056c6de8f77ebb` | masked 35M promoted producer |
| gen2A recent-history checkpoint | `da7bde2e5dc428397be13fddceb25e2979d57c1b9792eec3fce9e198b95af75f` | preceding promoted champion |
| gen4 hard-negative checkpoint | `b0f939464c138d6d0dca5586585d7e71aacb7ed86183cccbc2131d95750fe1c5` | held 209-191 candidate |
| A0 binding file | `c21ba913a4b580174eb58ee71f4f4371a5a8234580ba4f42b1e050aaec94cf26` | typed scalar-retention decision |
| gen3 legacy scalar attestation file | `677205e2f5629397202e254c6a6d2e90e84651868e7d799999e2c330131fcd2f` | read-only scalar bridge |
| gen3 legacy attestation content | `fac746b3df04562be9aab76291e206ac1ad91d7d25ba84e734b5592c5a74e650` | canonical attestation digest |
| gen3 training report | `2054584a00755db242696aa78dd1af625607cc9ed0345173683ebad14f92073d` | positive scalar loss telemetry |

All three checkpoints have distinct bytes, are masked, and use the same 35M
entity-graph architecture.

## Current local integrity snapshot

These hashes describe this materialization and the stable v2 validator at the
time of inspection:

| local file/value | SHA-256 |
|---|---|
| `a1_pre_wave_contract.rnd_draft.json` | `563769f6858952a0ba539c6455ee7196780b59b4bc31878e5594b07287b2b187` |
| `a1_pre_wave_contract.template.json` | `b6e7ba32155d6e3521132fc453e74a505dc36eaf057ea020630a5c574865a260` |
| `tools/a1_pre_wave_contract.py` | `e1dbb91413616a4d323b45967068f39695a334ad229628b684509d1cdb7a78af` |
| `tools/build_memmap_corpus.py` | `c21f4a304aee19944f3882af1bd72ada7c6d8822f31a09b0d3950317987ffbec` |
| `tools/train_bc.py` | `9d787d516f6cab65a4e15a1f6a4557df04a92c958dfcb52dfbce10431aca12b0` |
| canonical learner recipe | `1be1a29e44f1742e33bbff8798365a8ef2563438e2b4864160f2180308154655` |

`inspect-template` currently returns exactly:

```json
{
  "schema_version": "a1-pre-wave-contract-draft-v2",
  "unresolved": [
    "$.science.search.c_scale",
    "$.science.search.n_full",
    "$.science.search.n_full_wide",
    "$.science.search.n_full_wide_threshold",
    "$.science.search.wide_roots_always_full",
    "$.science.search.symmetry_averaged_eval",
    "$.science.search.symmetry_averaged_eval_threshold",
    "$.science.search.rescale_noise_floor_c",
    "$.science.search.sigma_eval",
    "$.science.evidence[1].path",
    "$.science.evidence[2].path",
    "$.science.evidence[3].path",
    "$.fleet.seed_base",
    "$.fleet.seed_ledger"
  ]
}
```

Before sealing, replace only those placeholders from replayable typed S1/S2/S3
adjudicator envelopes and a synchronized production seed ledger. Do not infer a
search choice from defaults or informal probe output, and do not fabricate a
seed allocation from the stale R&D-host ledger.

After the typed S1 path and S1-selected search fields are in the draft, run:

```bash
python tools/a1_pre_wave_contract.py sync-generation-guard \
  --draft configs/experiments/a1_pre_wave_contract.rnd_draft.json
```

The command semantically replays the typed S1 adjudication. For `.03` it is a
byte-for-byte no-op and rejects a non-pristine guard. For `.1` or `.3` it
atomically changes only the provenance-declared guard's `--c-scale` expected
value and embeds the exact S1 artifact, prior guard hash, and stable repo-relative
synchronizer identity/hash inside that guard. Seal and verify reject a non-default guard with no receipt,
a manually edited guard, or receipt drift. Because the receipt lives in the
guard itself, both the explicit guard record and transitive runtime-tree hash
bind the reason for the change. Run this before seal; never hand-edit the guard.
