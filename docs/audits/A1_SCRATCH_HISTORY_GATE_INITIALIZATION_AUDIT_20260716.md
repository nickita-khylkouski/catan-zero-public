# A1 scratch history-gate initialization audit

**Date:** 2026-07-16

**Scope:** native v5 scratch initialization only

**Result:** unit initialization is unsafe; use a small nonzero gate

## Root finding

`_initialize_scratch_meaningful_history_path` initialized both the masked-mean
and ordered-history residual gates to `1.0`. Both random branches are added
after `state_norm`, with no downstream normalization. The ordered branch also
preserves event-count mass, so the perturbation grows with history occupancy.

This created a length-dependent random rotation before the first optimizer
step. It was not merely an observability issue: the initial policy and value
readouts consumed the perturbed state.

## Bounded H100 probe

No training job or B200 job was run. A deterministic forward-only probe used:

- commit `84b970b`;
- host `68-209-74-24`, one NVIDIA H100 80GB;
- PyTorch `2.11.0+cu128`;
- production width/layers/split: `632 / 6 / 1`;
- ordered history v2 with a 64-event cap;
- model seeds `1, 3, 5`, with paired synthetic-data seeds `1001, 1003, 1005`;
- both history gates set to the same tested scale.

Mean perturbation relative to the zero-gate normalized state:

| Active events | Gate 0.05 | Gate 0.10 | Gate 0.25 | Gate 1.00 |
|---:|---:|---:|---:|---:|
| 16 | 0.91% | 1.82% | 4.55% | 18.20% |
| 32 | 1.71% | 3.42% | 8.54% | 34.16% |
| 64 | 3.29% | 6.58% | 16.46% | 65.85% |

At 64 active events, mean cosine with the zero-gate state was:

- `0.9978` at gate `0.10`;
- `0.9868` at gate `0.25`;
- `0.8365` at gate `1.00`.

The result was stable across the three seeds: the gate-1.0 perturbation ranged
from `64.84%` to `66.75%` at 64 events.

## Implemented boundary

Scratch initialization now uses `0.1` for both configured history branches.
This is deliberately nonzero:

- the event encoder and ordered-history module receive gradients on the first
  backward pass;
- the gates themselves remain trainable;
- the saturated-history perturbation is reduced by roughly 10x.

Warm-start behavior is unchanged. Serialized gates are preserved exactly, and
function-preserving upgrades still initialize new gates to zero.

## Function micro-analysis

### `_initialize_scratch_meaningful_history_path`

**Purpose:** choose history residual gates after model creation and before DDP
and optimizer construction.

**Inputs and assumptions:**

- `scratch=True` means no initializer checkpoint or growth checkpoint exists;
- meaningful history may expose one or two additive branches;
- warm-started gates are authoritative checkpoint state.

**Outputs and effects:**

- mutates only scratch gate tensors;
- returns the exact initialization mode and numeric scale for the training
  information surface;
- does not alter warm-started tensors.

**Invariant:** scratch branches must be nonzero so their parameters can receive
first-step gradients, but their random output must not dominate the normalized
state.

### `EntityGraphNet.encode_state`

**Purpose:** build the shared state and optional value-tower boundary.

**History block behavior:**

1. encode retained event tokens;
2. optionally add gathered target-entity information;
3. compute masked-mean history;
4. optionally compute ordered history;
5. multiply each branch by its per-channel gate;
6. add the combined delta after `state_norm`.

**Dependency:** because step 6 has no following normalization, gate
initialization directly controls initial representation scale.

### `build_ordered_history_pool.forward`

**Purpose:** produce an order-aware bounded history summary.

**Scale behavior:** the normalized pooled vector is multiplied by
`active_events / max_events`. This correctly preserves event-count mass after
training, but it also makes unit random gates a history-length confound at
scratch initialization.

## Remaining evidence boundary

The `0.1` choice is an initialization safety fix, not a claim that it is the
optimal learned gate. Early scratch probes should still report gate trajectories
and loss/calibration by history length.
