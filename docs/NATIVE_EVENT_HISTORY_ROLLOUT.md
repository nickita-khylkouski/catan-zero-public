# Native public-event history rollout

## Current contract

The A1 corpora and native evaluator intentionally use an authenticated-empty
event surface:

- `event_tokens`: `(64, 41)` float16 zeros;
- `event_target_ids`: `(64, 4)` int16 `-1`;
- `event_mask`: `(64,)` false;
- `tools/build_memmap_corpus.py --omit-zero-events` may store the token and
  mask columns as implicit constants;
- `tools/train_bc.py --crop-authenticated-empty-event-history` may crop them
  to width zero only after the inventory-bound zero scan succeeds;
- `native_inference_event_history_capability()` reports unavailable.

Existing checkpoints were trained under this contract. Populating history at
inference for those checkpoints would be an input-distribution change, not a
backwards-compatible bug fix. The rollout therefore needs a new explicit
contract identity and fresh training data.

## Authoritative public producer

The implementation should reuse, not replace, the public reconstruction
already exercised by `tools/catanatron_player_adapter.py::_synthetic_event_log`.
Extract that logic into `catan_zero.rl.public_event_history` and make all
producers call the same pure function.

For the `2p_no_trade` track, emit one synthetic `reset` followed by the most
recent action records, right-aligned and capped at 64. Every action becomes a
`board_action` event with:

- public actor color;
- public action type;
- policy action id only when reconstructing it cannot disclose hidden data;
- robber coordinate and victim, which are public;
- rolled dice, which are public;
- a hidden-development-card marker rather than the bought card identity;
- a hidden-stolen-resource marker rather than the stolen resource;
- discard count only, never discarded resource identity.

`ActionRecord` does not retain its historical `(num_turns,
current_turn_index)`. Version 1 must therefore specify `turn_key=(0, 0)` for
every reconstructed record, matching the existing synthetic adapter. Do not
invent approximate ordinals. A later engine-state version may add public
record-time ordinals to `ActionRecord` and introduce a new contract identity.

## End-to-end contract

Use the identity `public_action_records_v1` in all four stages:

1. Producer
   - Add the identity to every shard manifest.
   - Python snapshot and direct Rust featurizers must emit schema-identical
     `(64, 41)` tokens, `(64, 4)` target ids, and `(64,)` masks.
   - Rust advertises `public_event_history_v1`; Python fails closed if the
     capability is absent.

2. Memmap
   - Preserve physical `event_tokens.dat`, `event_target_ids.dat`, and
     `event_mask.dat` for this contract.
   - Reject `--omit-zero-events` when any input component declares
     `public_action_records_v1`.
   - Seal per-column nonzero counts and the event contract in
     `corpus_meta.json` and the payload-inventory hash.
   - Reject mixing `authenticated_empty_v0` and `public_action_records_v1`
     components unless an explicit migration rewrites every row.

3. Trainer/checkpoint
   - Require the corpus contract before optimizer construction.
   - Reject `--crop-authenticated-empty-event-history` for v1.
   - Store the event contract, schema, history limit, feature width, and corpus
     inventory in checkpoint metadata.
   - Existing checkpoints without v1 metadata remain `authenticated_empty_v0`.

4. Native inference/evaluation
   - Read the checkpoint contract before constructing an evaluator.
   - Empty-v0 checkpoints continue receiving zero history.
   - V1 checkpoints require a v1-capable wheel and populated native history.
   - Include the contract in evaluation recipe/config hashes and promotion
     evidence; refuse candidate/champion comparisons with different contracts
     unless both inputs are deliberately normalized to empty-v0.

## Required parity and safety gates

- Golden public-redaction tests for buy-development-card, robber steal,
  discard, roll, build, and end-turn records.
- Python snapshot versus direct Rust bit parity for all event arrays on at
  least 200 states, both masked and unmasked.
- Truncation/right-alignment parity at 0, 1, 63, 64, and 65 records.
- Hidden-state mutation test: changing opponent hands, bought-card identity,
  or stolen/discarded resource while preserving public evidence must not
  change emitted history.
- Memmap round trip and mixed-contract rejection tests.
- Trainer fail-closed tests for empty cropping, missing metadata, and inventory
  mismatch.
- Checkpoint reload and candidate/champion evaluator contract tests.
- Native wheel capability, version, checksum, and isolated-install tests.

## Smallest safe landing slice

The first slice can land without changing any model input:

1. extract the existing public action-record reconstruction into the shared
   source module;
2. keep both production feature providers in `authenticated_empty_v0` mode;
3. add golden redaction and truncation tests against the extracted helper;
4. add the versioned shard/memmap/checkpoint contract fields and reject v1
   until both native and Python feature providers advertise readiness.

The next slice implements Python/native v1 tensor parity and the wheel
capability. Only after a v1 canary corpus passes the payload scan should a
fresh v1 checkpoint train and consume non-empty history. No existing
checkpoint changes behavior during either slice.
