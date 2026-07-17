# A1 coherent-public v4 authority

This directory is the future authority reissued after adapter-v5 corpus audits
found two learner-input contradictions. It does not amend or re-authenticate the
issued v3 contract, guard, receipts, or stored rows.

All new canonical generation and learner construction is bound to
`rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop`. Existing
adapter-v5 composites remain quarantined. Training admission requires a newly
materialized and authenticated v6 composite; relabeling an old composite does
not satisfy that requirement.

The v5 quarantine evidence currently covers:

- 959,142 rows with zeroed playable-development-card slots despite 57,640
  selected development-card-play rows.
- 4,061 resource-clipping contradictions across 1,271 games, including 1,868
  full-search rows.
- 7,440 rows at resource saturation risk.

The machine-readable admission state is in
`configs/production/training_science_admission.json`. The selected V7 parent
update and native scratch training remain fail-closed until the exact budgeted
action decoder is commissioned; the older V6 B12 result is retained as
historical evidence but cannot authorize the changed architecture.

V7 also retains the commissioned `PLAY_TURN=4.0` repair while adding the
previously audited exact-prompt weights `MOVE_ROBBER=3.0`,
`BUILD_INITIAL_ROAD=2.0`, and `DISCARD=1.5`. Replaying the complete canonical
weighting and composite-sampling operator over the sealed r5 training split
lowers `PLAY_TURN` from 71.71% to 53.00% and raises all three named
hard-decision masses. Aggregate phase fractions cannot be rescaled directly:
equal-per-game normalization changes each game's denominator after phase
weighting. This changes policy sampling only; forced rows remain policy-inactive
and value phase weights remain disabled.
