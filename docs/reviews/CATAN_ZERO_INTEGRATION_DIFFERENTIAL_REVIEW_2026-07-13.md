# Catan Zero integration differential review — 2026-07-13

## Executive summary

| Severity | Found | Fixed in this review |
|---|---:|---:|
| Critical | 0 | 0 |
| High | 2 | 2 |
| Medium | 3 | 3 |
| Low | 1 | 1 |

**Overall risk before remediation:** High
**Recommendation after remediation:** Approve the reviewed integration tip for the
covered CPU/source contracts, conditional on the normal GPU parity gate before a
production learner launch.

The review found no remaining correctness regression in the public-award bridge,
authenticated empty-event crop, adapter-version admission, DDP objective reductions,
or the tracked 0.1.8 wheel release transaction. It did find two identity gaps, two
ingest/launch regressions, and a release-test gap. All are repaired with fail-closed
tests in the accompanying commit.

Key metrics:

- Baseline range reviewed: `f502732..33072da` (65 commits, 133 files,
  +10,668/-575).
- Priority files read deeply: learner, typed config, optimizer sidecars, memmap
  conversion, launcher guards, native wheel builder/installer, H100 evaluator.
- Focused Python gate: 396 passed, 17 skipped.
- Native semantic gate: 4 passed (one public-award feature test and three gameplay
  temperature tests).
- No GPU, SSH, fleet, registry, tag, release, or public-main state was mutated.

## What changed

The integration range combines learner objective/DDP repairs, optimizer continuity,
public-award provenance, event-history cropping, adapter provenance, memmap selection,
and native 0.1.8 release hardening. The remediation patch changes these surfaces:

| Surface | Risk | Remediation |
|---|---|---|
| `tools/build_memmap_corpus.py` | High data weighting | Close open seed runs at each independent source boundary and reject repeated canonical shard paths. |
| `src/catan_zero/rl/pipeline_configs.py`, `tools/train_bc.py` | High trajectory identity | Bind public-award mode plus selected/excluded validation sets into typed config and optimizer-resume identity; bump schema to 12. |
| `tools/launcher_guards.py` and launchers | Medium availability/safety | Treat validated config-file values as explicit guard inputs, including boolean, append, and multi-value actions. |
| `tools/fleet/a1_h100_eval_fleet.py` | High artifact provenance | Require PEP 610 installed-wheel SHA-256 to match the sealed inventory. |
| `tools/build_catanatron_rs_wheel.sh` | Medium release acceptance | Run public-award and gameplay-temperature semantic tests before building or hashing the wheel. |
| `FLEET.md` | Low operator accuracy | Replace stale pre-release text with the completed 0.1.8 transaction. |

## Findings and remediation

### High: optimizer continuity did not bind all learner input semantics

**Evidence:** `TrainConfig` omitted the public-award feature interpretation and the
effective validation contract. `_training_resume_recipe_identity` hashes the normalized
typed config, so a resume could reuse Adam moments/RNG after changing the player-token
slot-12 interpretation or the game set excluded from training.

**Impact scenario:**

1. Epoch 1 trains with one authenticated holdout or legacy-zero award input.
2. A continuation changes the sentinel/excluded seeds or requests authoritative award
   input while pointing at the epoch-1 checkpoint.
3. The old identity compares equal and restores moments and schedule counters.
4. One reported run now contains two different optimizer trajectories/data measures.

**Fix:** schema 12 adds award mode, mixed-corpus authorization, validation-contract
file digest, evaluated seed-set digest, and complete optimizer-excluded seed-set digest.
`_validation_contract_config_identity` derives the last three before config sealing.
Parameterized resume tests prove every field changes the identity.

### High: H100 evaluation attested a wheel digest it never checked

**Evidence:** the evaluator checked the tracked inventory, installed version, and
self-reported capability names, but did not bind the installed distribution to the
planned wheel bytes. A different 0.1.8 extension could report the same names and pass.

**Impact scenario:**

1. A host retains a stale or locally rebuilt 0.1.8 wheel.
2. The checkout inventory contains the expected release digest.
3. Version/capability probes pass, so the evaluation report declares the planned
   engine identity while executing different native code.

**Fix:** `_assert_installed_native_wheel_sha256` reads PEP 610 `direct_url.json` from
the installed distribution and requires every recorded archive digest to equal the
sealed `sha256:` identity. The remote preflight calls it before launching a lane; a
unit test proves a mismatching installed digest is refused.

### Medium: seed deduplication merged games across independent sources

**Evidence:** `_GameSeedRunTracker` intentionally carries an open game over adjacent
shards, because one writer can split a game at a row boundary. The converter also
carried that open run across separately supplied source roots. With legacy rows lacking
`decision_index`, the last game of source A and first game of source B could share a
seed and be misclassified as one continuation. The generic path also allowed the same
canonical shard path twice and relied on optional seed columns to expose it.

**Impact scenario:** duplicate trajectories receive excess training weight while the
corpus reports no duplicate seed. An A1 selection could likewise accept a repeated
selected game at a source boundary.

**Fix:** preserve source identity through audit reordering, call `start_source()` on
both trackers whenever the source changes, retain the global seen-set, and reject any
repeated canonical input path before conversion.

### Medium: typed config launches were falsely rejected by CLI guards

**Evidence:** `apply_config_file` returns the destinations explicitly filled by a
validated config. Generation synthesized those values into the argv inspected by the
guard; training discarded the return and linted raw argv, reporting all five critical
values as missing even when supplied in the config.

**Fix:** centralize synthesis in `launcher_guards.argv_with_config_values` and use it
from both generation and training. The helper correctly handles boolean option pairs,
store actions, repeated append actions, empty lists, and multi-value actions. A full
TrainConfig regression proves all guarded values and two append values re-parse.

### Medium: the wheel builder advertised semantics it did not test

**Evidence:** the builder ran only the public-belief filter, then accepted an
unconditional capability-name set from the compiled extension. The corrected public
award test is behind Rust's `python` feature, and the gameplay-temperature tests live
in `gumbel_mcts_rs`; neither was part of release acceptance.

**Fix:** before `maturin build`, run the exact Python-feature public-award test and the
three temperature-filtered native MCTS tests. Contract tests require both commands to
precede compilation and hashing.

### Low: release documentation still described 0.1.8 as pending

`FLEET.md` said the inventory still named 0.1.7 even though checksum-only commit
`09f2384` seals 0.1.8. The text now records the completed transaction and preserves the
future immutable-release procedure.

## Test coverage analysis

| Contract | Evidence |
|---|---|
| Multi-source seed boundaries and repeated paths | Tracker unit tests plus end-to-end memmap refusal tests. |
| Typed config / resume identity | Every new identity field is a parameterized drift case; config namespace round trip covered. |
| Config-filled CLI guards | Real train parser/config payload, all five critical flags, append-valued option. |
| Installed wheel identity | Matching PEP 610 record passes; mismatching digest raises `FleetError`; generated remote preflight contains sealed digest. |
| Release semantics | Static release-order tests plus direct Rust execution: 1 public-award and 3 temperature tests passed. |
| Existing integration surface | 396 focused Python tests passed; 17 environment/GPU-only cases skipped. |

Coverage limit: this review did not rebuild the manylinux wheel because the local host
does not provide the sealed Ubuntu toolchain. The release source transaction was
inspected, its semantic Rust tests ran locally, and no native/build input changed after
the previously sealed wheel commit before this review. Production must continue to use
the existing deterministic build-host/release-asset verification.

## Blast radius analysis

| Function / surface | Direct production callers | Blast radius |
|---|---:|---|
| `build_memmap_corpus` | CLI plus corpus-building workflows | High: every multi-source learner corpus. |
| `_training_resume_recipe_identity` | `train_bc.main` | High: all resumable learner checkpoints. |
| `_configure_public_award_feature_training` | `train_bc.main` | High: every entity-graph learner input. |
| `argv_with_config_values` | generation and training launchers | Medium: guard admission, no learner numerics. |
| `_assert_installed_native_wheel_sha256` | every H100 evaluator host preflight | High: evaluation provenance and comparability. |
| wheel semantic commands | release builder | High but infrequent: all future 0.1.8-compatible wheel builds. |

## Historical context

- `3ef8cb7` introduced adjacent duplicate-game detection, but its shard-continuation
  invariant was too broad at a separate source boundary.
- `4bcf0d5` sealed optimizer resume semantics, but later public-award/validation
  contracts were not added to the typed identity.
- `582e3c3`, `e7d9a48`, and `33072da` hardened public-award admission and integrated
  CLI/corpus contracts. The numerical bridge itself remains correct.
- `09f2384` is the checksum-only 0.1.8 seal; the reviewed tip has no later native
  source change. The release test additions affect acceptance, not native semantics.

No validation or safety removal was found in the reviewed priority files.

## Recommendations

### Immediate

- Cherry-pick the accompanying audit-fix commit as one unit; its schema bump and tests
  belong with the behavior changes.
- Run the normal single-node GPU no-op/parity gate after integration with any concurrent
  learner-efficiency commits. This review intentionally did not use fleet hardware.

### Before production

- Re-run the canonical full repository gate on the final combined integration head.
- Keep the H100 evaluator's PEP 610 check fail-closed; do not replace it with version or
  capability checks.
- Preserve source-root boundaries if future corpus sorting or descriptor formats change.

### Tracked limitation

`optim_state.py` documents a multi-node filesystem caveat: all FSDP ranks must see the
same optimizer sidecar. Current B200 training is single-node and unaffected. A future
multi-node rollout should make shared-storage admission and rank-wide load success an
explicit collective contract before deployment.

## Analysis methodology

**Strategy:** surgical/critical-path differential review of a large integration range.

Techniques:

- Compared `f502732..33072da`, inspected commit history and post-seal native diffs.
- Followed producer → memmap → learner → checkpoint/inference public-award data flow.
- Followed selected seed → validation exclusion → typed config → optimizer sidecar flow.
- Inspected DDP/FSDP ordering around freezes, wrappers, optimizer creation, collective
  metric normalization, and sidecar save/load.
- Checked static guard configuration against live parsers.
- Checked wheel source versions, inventory, builder environment, normalization,
  installer capability enforcement, and evaluator host preflight.
- Used adversarial fixtures for repeated sources, mismatched wheel bytes, and config-only
  critical flags.

**Confidence:** high for the named source contracts; medium for whole-program behavior
until the final merged GPU parity gate runs.

## Appendix A — function micro-analysis

### `build_memmap_corpus` (`tools/build_memmap_corpus.py:1374`)

**Purpose:** Streams one or more teacher roots into a flat, aligned corpus while
preserving schema, provenance, game selection, and duplicate guarantees. It controls
the exact sample measure seen by every downstream learner.

**Inputs & assumptions:** source paths are operator-controlled but untrusted until
resolved; manifests/shards may be malformed; source order is meaningful; adjacent
shards within one source may split a game; separate source roots are independent;
selected/A1 artifacts must already authenticate their exact shard inventory.

**Outputs & effects:** writes memmap column files and metadata; returns corpus metadata;
may decompress/read every selected shard; aborts before a valid corpus is reported on
schema, provenance, selection, or duplicate drift.

**Block analysis:**

- **Attestation/selection admission:** validates paired A1 artifacts before file writes.
  **Why here:** untrusted data cannot influence output sizing first. **First principle:**
  authorization must precede transformation.
- **File inventory (lines 1465-1503):** preserves `(path, source_index)`, rejects the
  same canonical bytes twice, and applies audit order. **Why:** path aliases must not
  multiply sample weight. **Why 1:** seed columns are optional. **Why 2:** therefore
  byte identity needs an independent invariant.
- **Schema construction (lines 1506-1558):** establishes one aligned column contract.
  **Why here:** output handles need fixed dtypes/shapes. **Assumption:** first normalized
  selected shard represents the schema enforced on every later shard.
- **Streaming loop (lines 1585+):** closes open runs on source transitions, validates
  event omission before row filters, then selects and writes rows. **Why:** a shard
  boundary is continuable; a source boundary is not. **Why 3:** carrying `_current`
  across sources merges independent games, so only `_seen` may survive.

**Dependencies and invariants:** calls `_teacher_shard_files`, `_normalize_teacher_shard`,
both seed trackers, and A1 loaders; callers are CLI/tests. Invariants: one canonical
input path once; one seed run per game across independent sources; every output column
is row-aligned; selection happens before sizing/writes; authenticated audit order is
preserved. Risks are decompression/resource use, malicious metadata, and accidental
overweighting; all three fail closed in the covered path.

### `_configure_public_award_feature_training` (`tools/train_bc.py:4953`)

**Purpose:** Chooses the checkpoint-owned interpretation of public award feature slots
and performs the only allowed legacy-to-authoritative transition. It prevents a model
from silently interpreting a newly live slot with weights learned when that slot was
always zero.

**Inputs & assumptions:** policy/checkpoint metadata is validated; corpus provenance is
authenticated; requested contract comes from parsed/config-resolved input; entity
player encoder slot 12 is the longest-road feature; mixed corpora are diagnostic-only.

**Outputs & effects:** returns an auditable transition record; mutates policy contract;
zero-initializes exactly one input-weight column on upgrade; refuses downgrade,
unauthenticated authoritative data, or unacknowledged mixing.

**Block analysis:** validate requested mode first; validate aggregate corpus provenance
before model mutation; compare initializer/requested modes; zero the new column under
`no_grad`; verify it is zero; finally publish the policy contract. **First principle:**
new information should enter with zero immediate logit effect. **Why 1:** old weights
never learned its meaning. **Why 2:** random/nonzero reuse changes behavior before an
optimizer step. **Why 3:** zero initialization makes the transition continuous while
allowing subsequent learning.

**Dependencies and invariants:** coupled to corpus provenance validation,
`EntityGraphPolicy.save/load`, `_entity_batch`, and checkpoint metadata. Invariants:
authoritative mode requires wholly corrected data; authoritative checkpoints never
downgrade; mixed data never promotes; only slot-12 input weights change at transition;
DDP/FSDP wrapping and optimizer creation occur afterward. Risks are forged provenance,
wrong slot width, and transition after optimizer construction; current ordering rejects
all three.

### `_entity_batch` and `_scan_empty_event_mask` (`tools/train_bc.py:8155`, `:11999`)

**Purpose:** `_scan_empty_event_mask` proves an immutable corpus has no live event rows;
`_entity_batch` then safely returns width-zero event tensors and applies hidden/public
award input transformations. Together they remove padded work without deleting usable
information.

**Inputs & assumptions:** entity columns are row-aligned; the A1 information-surface
contract authorizes cropping; scan runs before training; global crop/award modes are set
once per `main`; batch indices are in bounds.

**Outputs & effects:** scan returns a hashed zero-count receipt or aborts on first live
chunk; batch builder returns NumPy tensors, may mask hidden player slots, and never
mutates source memmaps. The event encoder is separately frozen before DDP and optimizer
creation.

**Block analysis:** chunked scan avoids whole-corpus materialization; crop branch still
indexes/validates the full mask before making empty token axes; hidden masking precedes
award bridging; award contract is applied last. **First principle:** an optimization
may remove computation only after proving the removed signal is identically empty.
**Why 1:** metadata alone can drift. **Why 2:** an exact byte-backed scan detects drift.
**Why 3:** freezing the disconnected encoder prevents DDP unused-parameter failure and
AdamW decay on unreachable weights.

**Dependencies and invariants:** tied to A1 event-history authentication,
`_freeze_authenticated_empty_event_encoder`, model forward shapes, and DDP wrapping.
Invariants: nonzero event masks always abort; width-zero tokens retain feature width;
public hidden slots remain masked; award slot behavior matches checkpoint metadata;
frozen event parameters are excluded from optimizer groups.

### `_training_resume_recipe_identity` and
`_validation_contract_config_identity` (`tools/train_bc.py:17949`, `:2887`)

**Purpose:** defines when model weights, Adam moments, schedule counters, and RNG state
belong to one continuous trajectory. The validation helper binds both evaluated games
and all games excluded from optimizer updates.

**Inputs & assumptions:** typed config is fully resolved; input files already hashed;
validation contract is authenticated; world size/topology is known; changing only the
checkpoint filename is legitimate continuation.

**Outputs & effects:** returns deterministic JSON-serializable identities; does not
write files itself; downstream progress sidecars hash and compare the identity before
loading optimizer state.

**Block analysis:** validation helper emits empty sentinels for no contract, otherwise
canonicalizes selected and excluded seed sets; resume identity normalizes only parent
checkpoint locations and preserves all science fields; topology is added explicitly.
**First principle:** reusable optimizer state is valid only for the same objective and
sample measure. **Why 1:** Adam moments encode prior gradients. **Why 2:** changing
inputs/holdout changes those gradients. **Why 3:** therefore the contract must mismatch
before state load, not merely appear in a post-training report.

**Dependencies and invariants:** `TrainConfig.full_config_hash`,
`load_training_progress`, `_game_seed_set_sha256`, and DDP topology. Invariants: file
path changes alone do not block continuation; file content/award/holdout changes do;
selected validation and full exclusion remain distinct; world size/batch geometry stay
bound. Risks are omitted future science fields, hash collisions, and stale derived
args; schema bump, all-field hash tests, and explicit helper tests mitigate them.

### `argv_with_config_values` (`tools/launcher_guards.py:46`)

**Purpose:** converts validated typed-config inputs into an argv representation that
the existing CLI guard can parse and value-check. It preserves the distinction between
silent defaults and explicit non-CLI configuration.

**Inputs & assumptions:** `args` already contains values applied by
`apply_config_file`; `config_filled` includes only fields actually supplied; parser
actions are authoritative; raw argv retains required and explicitly overriding flags;
config schema/pipeline were validated earlier.

**Outputs & effects:** returns a new list, never mutates raw argv/parser/namespace;
adds canonical options for filled values; skips unrepresentable `None` and empty
collections; repeats append actions correctly.

**Block analysis:** map destinations to actions; branch by boolean/store/append/list/
scalar action; return the augmented sequence for a complete argparse re-parse.
**First principle:** guards should inspect effective explicit inputs, not transport
syntax. **Why 1:** config and CLI are equivalent configuration channels. **Why 2:** raw
argv sees only one. **Why 3:** action-aware synthesis lets the existing parser enforce
the same types and expected values without duplicating validation.

**Dependencies and invariants:** called by generation and training before
`run_or_refuse`; coupled to `apply_config_file` and `guard_cli_flag_lint`. Invariants:
CLI overrides remain first-class; only filled values are synthesized; booleans choose
the correct positive/negative spelling; append multiplicity is preserved; guard
expected-value checks still run. Risks are exotic argparse actions, empty nargs `+`,
and duplicate options; current parser action families are covered and regression-tested.

### `_assert_installed_native_wheel_sha256`
(`tools/fleet/a1_h100_eval_fleet.py:135`)

**Purpose:** proves the installed native evaluator code originated from the exact
sealed wheel bytes. It closes the gap between repository inventory identity and the
environment actually imported on each host.

**Inputs & assumptions:** expected digest comes from the sealed plan; pip installed the
local wheel and wrote PEP 610 metadata; distribution name is `catanatron-rs`; metadata
JSON is untrusted; SHA-256 is the release identity.

**Outputs & effects:** returns the canonical expected digest on success; reads only
installed metadata; raises `FleetError` on malformed, missing, ambiguous, or mismatched
identity; performs no install or mutation.

**Block analysis:** validate expected syntax; load/parse distribution metadata; require
archive provenance; collect legacy `hash` and modern `hashes.sha256`; require the exact
singleton expected value. **First principle:** code identity is a byte digest, not a
version label. **Why 1:** labels and capability names are self-reported. **Why 2:** a
different binary can reuse both. **Why 3:** the installer-recorded archive digest binds
the imported environment to the release artifact.

**Dependencies and invariants:** called by every host preflight before evaluation;
coupled to the plan engine identity, tracked inventory, installer, and import/version/
capability checks. Invariants: digest format is canonical; missing direct URL fails;
conflicting hash fields fail; installed digest equals planned digest; preflight exits
nonzero before lane launch. Risks are metadata deletion/tampering, editable installs,
and stale environments; all are deliberately refused.
