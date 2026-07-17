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
`configs/production/training_science_admission.json` and remains fail-closed.
