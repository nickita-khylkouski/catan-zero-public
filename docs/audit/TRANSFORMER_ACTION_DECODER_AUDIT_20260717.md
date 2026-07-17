# Transformer action-decoder audit — 2026-07-17

## Scope

This audit traced the incumbent EntityGraph Transformer policy from typed
training configuration through model construction, legal-action target joins,
meaningful public-history conditioning, D6 remapping, and function-preserving
checkpoint upgrades.

The audit intentionally did not change the selected production architecture.
It repaired the ability to commission the already-existing Transformer
action-to-board decoder as an isolated treatment.

## Execution map

| Boundary | Source of truth | Effective consumer |
|---|---|---|
| Trainer CLI | `tools/train_bc.py` | effective checkpoint-owned architecture |
| Typed learner identity | `TrainConfig` in `pipeline_configs.py` | recipe/config hash |
| Fresh construction | `EntityGraphPolicy.create` | `EntityGraphConfig` |
| Model construction | `EntityGraphNet.__init__` | `action_cross_blocks` |
| Legal-action entity join | `_gather_target_tokens` | gather/cross/direct edge paths |
| Meaningful history | event encoder + ordered pool + target gather | global policy/value state |
| D6 transform | `hex_symmetry.permute_entity_batch` | tokens, actions, targets, events |
| Warm-start upgrade | `a1_function_preserving_upgrade.py` | exact-zero residual decoder |
| Learner commissioning | `train_bc.py` | small nonzero terminal projections |

## Confirmed root issue

`EntityGraphConfig` deliberately has two different decoder-depth fields:

- `action_cross_attention_layers` for the incumbent Transformer;
- `relational_action_cross_layers` for RRT/ResRGCN.

The canonical recipes use the Transformer trunk and carry the relational field.
Before this repair, the normal trainer/fresh-construction path did not expose
or forward `action_cross_attention_layers`. A recipe could therefore appear to
request one relational decoder layer while the Transformer constructed zero
action-cross layers.

The existing function-preserving upgrader could create a Transformer cross
block, but the sealed one-dose command did not make that topology explicit in
trainer argv. This made the experiment difficult to reproduce and easy to
misreport.

## Short-dose learning issue

The Transformer `_CrossBlock` correctly zero-initializes:

- the attention output projection;
- the final feed-forward projection.

This gives exact zero-step parity. It also gives attention q/k/v and the first
feed-forward layer zero gradient on the first backward pass. In a 12-step
parent update, a meaningful part of the dose can be spent only opening those
terminal projections.

The learner now commissions an enabled, genuinely cold Transformer decoder
after checkpoint loading and before optimizer construction. It uses a bounded,
deterministic 0.01-scale identity-shaped terminal initialization. The
function-preserving upgrade artifact remains exactly identical before training,
while all inner decoder parameters receive gradient on the first learner
backward. A decoder that has already moved is preserved byte-for-byte.

## Paths audited and retained

### Legal-action target joins

The target join is structurally sound:

- robber moves bind the target hex and victim player;
- settlement/city moves bind the target vertex;
- road moves bind the target edge;
- target IDs are validated in disjoint local namespaces;
- local IDs are offset into the concatenated token sequence;
- live targets are mean-pooled per legal action;
- a parameter-free local target identity distinguishes otherwise identical
  empty board entities;
- symmetry transforms the local IDs before the gather.

No replacement mapping was warranted.

### Meaningful public history

Meaningful history is actor-public and conditions both policy and value through
the global state residual. Ordered attention preserves event order. Optional
event-target gathering binds events to their post-trunk board entity. Existing
tests prove that changing a meaningful event target changes both logits and
value after commissioning.

No direct injection of hidden authoritative state was found in this path.

### D6 and action remapping

The D6 transform remaps the complete coupled surface:

- hex/vertex/edge token rows;
- incidence tensors;
- static legal-action catalog IDs;
- legal-action entity target IDs;
- event action IDs and target IDs;
- event spatial scalars.

Legal-action row order remains stable, so policy targets stay aligned while
the action identities inside each row are transformed. The action decoder
therefore consumes the correct transformed entity and target identity.

### Upgrade parity

The checkpoint upgrader still emits zero-output action-cross blocks and
retains exact logits/value/Q parity at zero steps. Commissioning occurs only
inside the learner, after the upgrade artifact has been authenticated and
loaded. Existing trained decoder bytes are never reinitialized.

## Implemented contract

- Added `--action-cross-attention-layers`.
- Added checkpoint inheritance and mismatch refusal.
- Refused use of the Transformer knob with relational trunks.
- Forwarded the field through `EntityGraphPolicy.create`.
- Bound it into typed training identity and canonical recipe schemas.
- Added the explicit flag to the action-cross one-dose command.
- Reported effective relational decoder depth as zero on Transformer models,
  while retaining the raw configured relational knob for diagnosis.
- Added cold-path commissioning and first-backward gradient coverage.

## Focused evidence

- 30 decoder binding/report/commissioning tests passed.
- 191 config/checkpoint/upgrade/history tests passed (one skipped).
- 38 D6/action-remapping/history tests passed (one skipped).
- Python compilation, Ruff checks, and Git whitespace checks passed before
  integration.

