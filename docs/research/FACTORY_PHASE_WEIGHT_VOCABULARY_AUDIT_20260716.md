# Factory Phase-Weight Vocabulary Audit — 2026-07-16

## Finding

`tools/start_training_factory.py` advertised phase emphasis with this default:

```text
robber=3.0,initial_build=2.0,discard=1.5
```

Current teacher shards do not store those aliases. As documented and validated
by `tools/curate_teacher_data.py`, their `phase` column is the engine's public
`current_prompt`, whose production labels include:

```text
MOVE_ROBBER
BUILD_INITIAL_SETTLEMENT
BUILD_INITIAL_ROAD
DISCARD
PLAY_TURN
```

Both `build_sample_weights` and `build_value_sample_weights` apply configured
weights with exact string equality. Consequently zero of the four production
prompts the factory intended to emphasize matched a default key. The advertised
3x robber, 2x opening, and 1.5x discard learning-signal multipliers were all
silent no-ops for current production data.

## Fix and contract

The fresh-training factory default now uses exact production labels:

```text
MOVE_ROBBER=3.0,BUILD_INITIAL_SETTLEMENT=2.0,BUILD_INITIAL_ROAD=2.0,DISCARD=1.5
```

Both initial-placement prompts are explicit because there is no single
`initial_build` prompt in the current corpus. `PLAY_TURN` retains unit weight.
Legacy-data callers can still pass old aliases explicitly; this change does not
reinterpret user-supplied keys.

The regression renders a real factory command, parses its emitted phase map,
and applies it to synthetic rows carrying the authenticated production
vocabulary. It verifies the relative policy and value weights are exactly
`[3.0, 2.0, 2.0, 1.5, 1.0]` rather than the pre-fix all-ones vector.
