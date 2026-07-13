# A1 topology-residual + action-target-gather causal audit (2026-07-13)

## Decision

Do **not** spend another B200 dose rerunning the combined topology-residual plus
action-target-gather arm.  The exact experiment requested by the learner plan
already exists, used the winning TEMP learner recipe, and was statistically
indistinguishable from TEMP over 600 seat-swapped pairs:

| agent | result | decision |
|---|---:|---|
| topology+gather vs TEMP | 601-599 / 1,200 games | CONTINUE |
| ordinary superiority evidence | LLR `-1.250` | CONTINUE |
| pentanomial evidence | LLR `-0.578` | CONTINUE |

The architecture branch trained, deployed, and preserved the intended f7
starting function.  Its absence of gain is therefore not explained by a dead
branch, bad checkpoint load, DDP omission, or evaluator fallback.  At this dose,
the Pareto decision is to keep TEMP and spend the next experiment on learner
objective/data semantics rather than duplicate this architecture arm.

## Exact experiment identity

- f7 initializer:
  `sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4`
- TEMP control:
  `sha256:fefba044df58b9508de751d76d09bedeb630a2e832f6db46b70d95b5d4c77394`
- topology+gather candidate:
  `sha256:63a5608bed8b748bc75153953c235aa4f9dd7ce16023d73a675bbf2544641306`
- upgraded initializer:
  `sha256:cac132cde579d813a845c91939be2c9df413554327df7484803740c23009b315`
- candidate location:
  `/home/ubuntu/experimental_nonpromotable/f7-topology-gather-temp-recipe-20260713-r1/train1/checkpoint.pt`
- paired evaluation:
  `eval600-v-temp-d1/collected/a1-eval-bf6b1d0aaa1ef9d2/pooled/internal.json`

The learner independently reloaded f7, created fresh Adam state, and consumed
exactly `1,024 * 8 * 512 = 4,194,304` sampled rows at global batch 4,096.  It
used LR `3e-5`, 100-step warmup, value LR multiplier `0.3`, no candidate
chaining, and the same n128/n256/replay per-source policy temperatures as TEMP
(`1.0`, `1.11`, `0.52`).  Both evaluated agents used native Rust n128 search,
D6 threshold 20, `c_scale=0.1`, tanh value, public observation, and identical
P4/minimum-search settings.  The panel recorded zero errors and truncations.

The checkpoint on the B200 worker independently confirms both architecture
flags are on and all twelve new tensors are serialized.  The training report
showed all twelve changed from initialization; representative delta norms were
`1.007` for gather output weight, `0.528` for topology output weight, and
`1.245` for topology source weight.  Global initializer-to-candidate relative
L2 drift was `0.026805`, consistent with a single dose rather than an oversized
or chained update.

## Bottom-up architecture audit

### `TopologyResidualAdapter.forward`

**Purpose:** This path supplies one topology-local message before the incumbent
Transformer without replacing any mature shared layer.  Its zero output
projection makes the upgraded checkpoint exactly equal to f7 until training
changes the new parameters (`relational_trunks.py:49-113`).

**Inputs and assumptions:** `tokens` is `[B,S,H]`; `relation_ids` is
`[B,destination,source]`; the two sequence axes use the same token layout; and
`key_padding_mask`, when present, is true only for padding.  Physical/event
relations must come from `build_relation_ids`, whose local IDs are validated by
the entity-batch boundary before the tensor forward.

**Outputs and effects:** It returns `tokens + update`, has no mutation or
external I/O, and sends gradients through a new residual path.  At initialization
`update` is bitwise zero because both output weight and bias are zero.

**Block analysis and invariants:**

1. Lines 84-92 select only direct physical and event-target relations, then
   remove padded sources and destinations.  This occurs before aggregation so
   masked tokens cannot leak through degree normalization.  Invariant: SELF,
   HUB and GLOBAL edges never enter this adapter.
2. Lines 93-101 compute a per-destination mean, normalize it and project it.
   Mean rather than sum keeps scale independent of relation fanout.  Invariant:
   source projection is identity and the output is zero at upgrade time.
3. Lines 102-111 zero degree-zero and padded destinations.  This is required
   after LayerNorm/linear biases, which could otherwise turn an empty
   neighborhood into a learned generic residual.  Invariant: only a destination
   with at least one live reviewed incidence edge can change.

**Dependencies and risks:** `EntityGraphNet.encode_state` constructs relations,
applies the adapter, and only then runs the incumbent Transformer
(`entity_token_policy.py:840-891`).  D6 correctness depends on incidence IDs and
token rows being relabelled together.  The focused permutation, zero-output,
padding, gradient and checkpoint tests cover those contracts.  The remaining
experimental limitation is optimizer control: this module is classified as
the whole `trunk`, so there is no topology-only LR multiplier.

### `build_relation_ids`

**Purpose:** It converts validated local incidence columns into one directed
relation matrix shared by the residual adapter and relational trunks
(`relational_trunks.py:116-253`).

**Inputs and assumptions:** Required topology arrays have the fixed Catan board
shapes; `sequence_length` matches the assembled token stream; local IDs are
either `-1` or in their namespace; and event target columns have the fixed
hex/vertex/edge/player meaning.

**Outputs and effects:** It returns a long tensor `[B,S,S]`; it allocates only
temporary tensors and does no external I/O.  Physical links are written in
both directions with distinct relation IDs; hub/global reader edges fill only
previously unset entries; event edges are cropped to the live sequence length.

**Block analysis and invariants:** fixed offsets mirror `_state_tokens`; `_link`
maps local IDs to concatenated-token indices; hub/global relationships do not
overwrite specific physical links; event links preserve direction.  The
entity-batch validator rejects malformed IDs before this call.  The coupled
invariants are: no out-of-range indexing, every physical incidence has its
reverse relation, and D6 relabelling commutes with relation construction.

### `_gather_target_tokens` and `score_actions`

**Purpose:** The gather maps each legal action to the post-trunk entities it
targets, mean-pools them, and adds a learned zero-output residual to the action
embedding (`entity_token_policy.py:1144-1192`, `918-963`).  This exposes board-
local information directly to a policy head that previously compared only a
global state vector with static/context action features.

**Inputs and assumptions:** `legal_action_target_ids` is `[B,A,4]` with local
hex/vertex/edge/player IDs; the public numpy boundary has already rejected IDs
outside `[-1, namespace_width)` and targets attached to padded actions
(`entity_token_policy.py:1948-1997`); and `tokens` follows `_state_tokens` order.

**Outputs and effects:** The gather returns `[B,A,H]`; actions with no entity
target return exact zeros.  `score_actions` adds `target_gather_proj(pooled)` to
the encoded action and then uses the unchanged normalized dot-product policy.
There is no hidden state mutation or external call.

**Block analysis and invariants:** offsets translate local namespaces to the
concatenated sequence; invalid `-1` entries receive zero weight; valid targets
are pooled with a denominator clamped only for no-target actions.  The trailing
gather linear is zero at upgrade time, so logits are identical to f7.  Shape,
local-ID range, padding, D6 relabelling, split encode/score, CUDA-graph and
evaluator payload tests cover the full deployment call chain.

### Function-preserving upgrade and drift evidence

The allowlisted receipt replays the exact parameter-key delta, requires every
shared tensor to match in dtype/shape/value, checks the twelve deterministic
new tensors, reconstructs the effective config, and requires a reported
zero-forward difference (`a1_function_preserving_upgrade.py:273-458`).  The
receipt for this arm reports all 139 inherited tensors exact.

The audit found and fixed two fail-open provenance defects:

1. `audit_checkpoint_layer_drift.py` could import an older installed
   `catan_zero` and silently normalize away new architecture fields.  It now
   binds imports to its own checkout, verifies the resolved module path, and
   rejects config keys unknown to that checkout.
2. `a1_function_preserving_upgrade.py` also filtered unknown config keys before
   comparing effective configs.  It now rejects unknown source or upgraded
   fields instead of issuing an incomplete architecture receipt.

The layer-drift report now includes `public_award_feature_contract` in the
architecture contract and attributes topology-adapter tensors to their own
group rather than the generic shared bucket.

## Corpus coverage

The bound current n128/n256 architecture audit found no malformed targets or
incidence IDs:

| corpus | rows with any target | policy-active rows with target | actions targeted | invalid IDs |
|---|---:|---:|---:|---:|
| n128 | 20.85% | 43.02% | 59.56% | 0 |
| n256 | 20.79% | 42.92% | 59.59% | 0 |

This is sufficient to exercise the gather path, but it exposes a provenance
gap in the ad-hoc exact arm: the corresponding replay component was trained but
was not included in that architecture-coverage receipt.  It does not invalidate
the measured tie—the runtime validators still reject malformed batches—but a
future sealed replication must audit every supervised component, including
replay.

## Historical evidence and what it does not prove

- Old v3b added gather, cross-attention and value pooling while increasing the
  network from roughly 35M to 47.8M.  It trailed v3a on global correlation and
  masked evaluation.  That was a confounded bundle, not an isolated gather test.
- The fresh r3 gather-only reproduction scored `591-609` over 1,200 games
  (49.25%, CONTINUE).  It did not reproduce an earlier 52.5% cohort.
- An f7 gather-only K3 arm had small positive but inconclusive cohorts.  Its
  learner objective predates TEMP and therefore is not a matched control.
- The present exact TEMP arm is the strongest causal evidence: the combined
  topology+gather treatment neither improved nor harmed the TEMP control at one
  independent f7-started dose.

This evidence rules out “the 35M model simply cannot represent board topology”
as the proximal explanation for the bad chained learners.  It does not prove
the modules can never help at another scale or objective; it proves they are
not the current Pareto lever.

## Residual limitations and next action

The manual exact run has two sealing defects that prevent relabelling it as a
production transaction: `train1/command.json` and its launch/completion receipt
carry different command hashes (`52cf...` versus `5e57...`), and replay target
coverage was not bound.  The model/evaluation result remains useful diagnostic
evidence, but it is not a promotion artifact.

If this architecture is revisited after the learner objective is stable, use
independent f7 starts, at least two seeds, and separate gather-only and
topology-only arms.  Add a topology-only optimizer group before testing a
higher LR; do not use `--trunk-lr-mult` as a substitute because it changes the
entire mature trunk.  For now, preserve TEMP and prioritize the already-sealed
current-only/pure-target learner screens.  No B200 launch is authorized by this
audit.
