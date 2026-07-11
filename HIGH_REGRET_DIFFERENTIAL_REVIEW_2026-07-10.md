# High-Regret Replay Differential Review

## Executive Summary

**Reviewed commits:** `ad4e795` and `cb10353`
**Risk class:** High — promotion evidence and authoritative evaluation inputs
**Recommendation:** Approve only with the v3 fixes in this review commit and a freshly rebuilt suite/report. Existing v1/v2 suites must not promote.

| Severity | Found | Resolved |
|---|---:|---:|
| Critical | 1 | 1 |
| High | 2 | 2 |
| Medium | 1 | 1 |
| Low | 0 | 0 |

The original commits correctly rejected negative, gapped, duplicate, and partial trajectories and made evaluator/promotion consumers require v2 metadata. They did not, however, freeze the bytes or exact shard inventory that replay scans. The completed v3 contract now binds those inputs, replays source rows and complete trajectories in both consumers, validates again in evaluator workers, and binds candidate/incumbent role-specific search scales.

## Findings

### Critical — Replay scopes were path-bound but not byte-bound

**Original evidence:** `tools/high_regret_suite_contract.py` v2 resolved `shard_path` and `scope`, but the hashed regret manifest contained only path strings and row identities. `gather_game_action_sequence` recursively scans every `.npz`/`.npz.zst` below the shard parent.

**Attack scenario:**

1. Seal a valid suite and source-manifest hash.
2. Replace an authoritative shard in place, or add another shard containing the same seed.
3. Run evaluation against the altered trajectory while keeping suite and source-manifest bytes unchanged.
4. Restore the source before promotion, allowing the report to describe games that were not played from the frozen held-out states.

**Resolution:** The v3 contract records an exact canonical path/size/SHA inventory digest for every replay scope and rejects symlinks, path replacement during hashing, shard replacement, and inventory additions. See `tools/high_regret_suite_contract.py:19-96`, `tools/a1_promotion_artifacts.py:503-563`, and `tools/gumbel_search_cross_net_h2h.py:238-255`.

### High — Replay-completeness metadata was self-attested at promotion

**Original evidence:** v2 promotion checked arithmetic consistency of `replay_preflight`, then trusted its `replay_complete_states` claim. A forged manifest/suite could bind a tuple that did not match the actual shard row or a trajectory lacking `0..target`.

**Resolution:** Consumers now verify each manifest tuple against the immutable authoritative row and independently scan every bound scope for one unique contiguous `0..N` sequence covering the target. See `tools/high_regret_suite_contract.py:149-321`, evaluator enforcement at `tools/gumbel_search_cross_net_h2h.py:176-235`, and promotion enforcement around `tools/a1_promotion_transaction.py:1843-1935`.

### High — High-regret promotion did not replay routed role scales

**Original evidence:** promotion required shared sealed semantics but did not require `candidate_c_scale` and `baseline_c_scale` from the deployed agent identities. A report could therefore compare both roles at one scale and still satisfy the shared envelope.

**Resolution:** high-regret promotion now requires the candidate role scale from the candidate identity and the incumbent role scale from the champion identity, plus the role-specific budgets/readouts and complete search-pair replay. For the current adjudicated identities this is candidate `.10`, incumbent `.03`. See `tools/a1_promotion_transaction.py:1766-1801`.

### Medium — Evaluator process handoff left a validation/use gap

**Original evidence:** the parent loaded the suite before worker execution, but workers reconstructed from live paths without replaying the scope inventory.

**Resolution:** each worker validates the scope inventory after process handoff, gathers the trajectory, then validates it again before reconstructing the in-memory state. See `tools/gumbel_search_cross_net_h2h.py:238-255` and the worker replay block.

## Schema and Consumer Behavior

- `a1-held-out-high-regret-suite-v3` is the only accepted schema.
- The replay contract is `authoritative-shard-parent-hashed-unique-contiguous-trajectory-v3`.
- v1/v2 and hand-authored suites fail before GPU work.
- Source manifests load with `allow_pickle=False`.
- The builder, evaluator loader, evaluator worker, and promotion transaction all enforce the v3 inventory and trajectory contract.
- Promotion additionally replays report/suite identities, paired statistics, and candidate/incumbent deployed search identities.

## Test Coverage

Adversarial coverage includes:

- negative, missing, duplicate, and gapped decision rows;
- manifest identity and shard-path forgery;
- replaced shard bytes;
- injected same-scope shards;
- pathname replacement during source loading;
- legacy-schema rejection;
- evaluator worker inventory drift;
- promotion-time source mutation;
- candidate `.10` / incumbent `.03` role-scale laundering.

Focused result at review completion: **190 passed** across high-regret builder, evaluator, and promotion tests.

## Blast Radius

| Function | Consumers | Risk |
|---|---|---|
| `bind_state_to_manifest` | evaluator loader, promotion | High |
| `validate_replay_trajectories` | evaluator loader, promotion | High |
| `scope_inventory_sha256` | builder, evaluator parent/worker, promotion | High |
| `_verify_high_regret_source` | promotion transaction | Critical |

## Methodology

Focused high-risk differential review of all production and test changes in `ad4e795^..cb10353`, plus one-hop replay and promotion dependencies. Analysis included baseline/diff comparison, caller tracing, schema replay, path/manifest forgery modeling, TOCTOU modeling, and adversarial tests. Confidence is high for the reviewed high-regret path; GPU execution itself was not run.
