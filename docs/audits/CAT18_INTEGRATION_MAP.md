# CAT-18 Integration Map — Commit Debt (rescued untracked tooling)

Source: `origin/rescue/untracked-b200`, `origin/rescue/untracked-a100a`,
`origin/rescue/untracked-a100b` (each a read-only rsync snapshot of a live
GPU host's untracked working-tree files, per each branch's own
`rescue/README.md`). This document records, for every rescued file, where it
came from, where it landed in this branch, and why.

Note on scope vs the Linear ticket text: CAT-18's own description assumed
`modal_gumbel_factory_gpu.py` and the H2H gate tools were still untracked.
Verified against `master` before starting this work — both are **already
tracked** (`tools/modal_gumbel_factory_gpu.py` since commit `34b16d9`,
`tools/gumbel_search_vs_raw_h2h.py` / `tools/gumbel_search_vs_bot_h2h.py` /
`tools/gumbel_search_cross_net_h2h.py` / `tools/h2h_postrepair_aggregate.py`
/ `tools/h2h_v3conf_aggregate.py` all present on `master`). The actual
remaining debt was the Rust featurizer modules + their test suites, the
phase-2 window-feed tooling, and a handful of fleet ops scripts — all
captured only as untracked files on the three GPU hosts. This map covers
that set.

## Integrated (new tracked files)

| Rescued path (host) | New path | Decision |
|---|---|---|
| `rescue/b200-untracked/src/catan_zero/rl/action_context_features_rust.py` | `src/catan_zero/rl/action_context_features_rust.py` | integrated — byte-identical across b200/a100a/a100b, no existing tracked file of this name |
| `rescue/b200-untracked/src/catan_zero/rl/entity_token_features_rust.py` | `src/catan_zero/rl/entity_token_features_rust.py` | integrated — byte-identical across all 3 hosts; companion to the already-tracked `entity_token_features.py` (different name, no collision) |
| `rescue/*/tests/test_manifest_rust_featurize_provenance.py` | `tests/test_manifest_rust_featurize_provenance.py` | integrated — byte-identical across all 3 hosts |
| `rescue/*/tests/test_rust_action_context_evaluator_wiring.py` | `tests/test_rust_action_context_evaluator_wiring.py` | integrated — byte-identical across all 3 hosts |
| `rescue/*/tests/test_rust_action_context_parity.py` | `tests/test_rust_action_context_parity.py` | integrated — byte-identical across all 3 hosts |
| `rescue/*/tests/test_rust_featurize_evaluator_wiring.py` | `tests/test_rust_featurize_evaluator_wiring.py` | integrated — byte-identical across all 3 hosts |
| `rescue/*/tests/test_rust_featurize_parity.py` | `tests/test_rust_featurize_parity.py` | integrated — byte-identical across all 3 hosts |
| `rescue/a100a-untracked/tests/test_exact_budget_sh.py` (superset) | `tests/test_exact_budget_sh.py` | integrated — see **Divergent (resolved)** below, a100a's copy used (b200's is a strict subset) |
| `rescue/a100a-untracked/tools/auto_refill.sh` (seed window used) | `tools/auto_refill.sh` | integrated — see **Divergent (flagged)** below, a100a's seed window kept as the checked-in default |
| `rescue/a100a-untracked/tools/mps_rollout.sh` | `tools/mps_rollout.sh` | integrated — byte-identical a100a vs a100b |
| `rescue/a100b-untracked/tools/packing_mps_experiment.sh` | removed after completion | historical A100B experiment; canonical fleet launcher supersedes it |
| `rescue/b200-untracked/docs/PHASE2_WINDOW_FEED.md` | `docs/PHASE2_WINDOW_FEED.md` | integrated — design notes for task #94, no collision |
| `rescue/b200-untracked/launch_value_repair_v2_train.sh` | removed after completion | dated hard-coded repair invocation; current guarded trainer supersedes it |
| `rescue/b200-untracked/tests/test_concat_memmap_corpus.py` | `tests/test_concat_memmap_corpus.py` | integrated — no collision |
| `rescue/b200-untracked/tests/test_flywheel_phase2_integration.py` | `tests/test_flywheel_phase2_integration.py` | integrated — no collision |
| `rescue/b200-untracked/tests/test_value_target_lambda.py` | `tests/test_value_target_lambda.py` | integrated — no collision |
| `rescue/b200-untracked/tools/continuous_flywheel_spec_staged.py` | `tools/continuous_flywheel_spec_staged.py` | integrated — no collision with the already-tracked `tools/continuous_flywheel.py` (staged variant, different name) |
| `rescue/b200-untracked/tools/flywheel_feed_daemon.py` | `tools/flywheel_feed_daemon.py` | integrated — no collision |
| `rescue/b200-untracked-f94feed-extra/tools/deploy_phase2.sh` | removed after completion | one-time f94feed deployment script; canonical install/fleet tooling supersedes it |
| `rescue/b200-untracked-f94feed-extra/tools/feed_config.json` | `tools/flywheel_feed_daemon.example_config.json` | integrated **with rename** — see **Renamed** below |

## Skipped (duplicate / superseded)

| Rescued path (host) | Reason skipped |
|---|---|
| `rescue/a100a-untracked/tests/test_evaluator_shared_payload.py` | byte-identical to the already-tracked `tests/test_evaluator_shared_payload.py` — no-op, nothing to integrate |
| `rescue/a100b-untracked/tests/test_evaluator_shared_payload.py` | same as above |
| `rescue/a100b-untracked/tests/test_regime_fail_closed.py` | byte-identical to the already-tracked `tests/test_regime_fail_closed.py` — no-op |
| `rescue/b200-untracked/tools/gumbel_search_cross_net_h2h.py.bak.1783308135` | stale pre-commit backup of `tools/gumbel_search_cross_net_h2h.py`. Diffed: the tracked version is a strict **superset** (482 vs 547 lines; tracked version adds `_write_worker_progress` live-tally writer and `pair_errors` tracking that the `.bak` predates). Confirmed superseded, not integrated — it would be a regression if landed. |

## Divergent (resolved by picking the superset)

**`tests/test_exact_budget_sh.py`** — b200's copy is a strict prefix (184
lines) of a100a's copy (215 lines); a100a added `test_min_n_threshold_pure_python`
and `test_min_n_threshold_splits_regimes` on top, testing the
`exact_budget_sh_min_n` gate-adoption pairing dated 2026-07-07. Since b200's
content is fully contained in a100a's, a100a's copy was integrated with no
loss of coverage from either host.

## Divergent (flagged, not merged into one "true" version)

**`tools/auto_refill.sh`** — a100a and a100b are identical except for one
line, the host's assigned seed window:

```
a100a: LO=6100000000; HI=6200000000
a100b: LO=6200000000; HI=6300000000
```

These are live per-host ledgered seed ranges (see the seed-ledger discipline
already in this repo — colliding seed windows across hosts has caused real
generation-corpus problems before). a100a's window was kept as the
integrated file's default since it's arbitrary which host's copy becomes
canonical, but **this is a host-instance parameter, not a code fork** — a
host redeploying this script should update `LO`/`HI` to its own currently-
ledgered window rather than assuming the checked-in default is safe to run
as-is on a different host. Recorded here so the two concrete values that
were actually in flight (6.1–6.2B on a100a, 6.2–6.3B on a100b) aren't lost.

## Renamed

**`feed_config.json` → `flywheel_feed_daemon.example_config.json`** — the
rescued file was a live, dated run config (hardcoded `runs/selfplay/
gen3_mps_20260707*` shard globs, specific to the 2026-07-07 flywheel round),
not a reusable template. Renamed and kept as a worked example next to its
consumer (`tools/flywheel_feed_daemon.py`) rather than landed under its
original name, so nobody mistakes it for a currently-valid config to run
as-is — the shard-glob paths it references will go stale as soon as the next
round starts.

## Explicitly out of scope

- **`src/catan_zero/rl/flywheel/replay_window.py`** — historical phase-2 deployment (completed; script removed)
  (rescued) copies a modified copy of this file from the B200 dev tree
  (`~/catan-zero-f94feed`) over the live tree's tracked copy. Per the rescue
  branches' own README, only *untracked* files were captured — modifications
  to already-tracked files are working-tree diffs that were never rsynced.
  That means the actual phase-2 behavior change to `replay_window.py` is
  **not** in this integration; the completed `deploy_phase2.sh` runbook was removed
  script, but running it today would just copy the current tracked
  `replay_window.py` over itself (a no-op), not the dev tree's real diff.
  Flagging so whoever runs the phase-2 deploy knows the actual code change to
  that file needs to be sourced from the B200 host's live working tree
  directly, not from this rescue.
- **Host-side `git add`** on the B200 live tree remains a separate,
  deploy-time step — out of scope for this session (hosts are read-only from
  here; no SSH was used to produce this integration, only the rescue
  branches already pushed to `origin`).

## Verification performed

- `python3 -m py_compile` on every integrated `.py` file — all pass.
- `bash -n` on every integrated `.sh` file — all pass.
- `python3 -c "import json; json.load(...)"` on the integrated config — valid.
- `git status --short` after integration shows only the new files listed
  above as untracked additions; no existing tracked file was modified.
- No secrets, credentials, or personal (`/Users/...`, `/home/<user>`) paths
  found in any integrated file (`/home/ubuntu/...` host paths present are
  consistent with this repo's existing tracked `configs/gpu_cluster_hosts.json`).

## Known pytest-collection/run gaps against current `master` (post-CAT-18-review audit)

`py_compile` only checks syntax, not imports — actually running the rescued
test suite against current `master` (not the GPU host trees they were
authored against) surfaces four tests whose **companion source-side changes
were never landed on `master`** and so fail today, not merely skip. This is
expected given CAT-18's explicit scope ("commit what already works, don't
rewrite the tooling while committing it") — landing those companion changes
is separate, larger functional work already tracked elsewhere (task #61,
task #81, task #94) — but it wasn't previously called out here, so recording
it to avoid confusion when CI shows these red:

- **`tests/test_exact_budget_sh.py`** — `ImportError: cannot import name
  'exact_budget_sh_phases' from 'catan_zero.search.gumbel_chance_mcts'`.
  That function (task #61) exists only on the unmerged `origin/gen3-wheel-sync`
  branch, not on `master`.
- **`tests/test_value_target_lambda.py`** — `ImportError: cannot import name
  '_played_action_bootstrap_value' from 'tools.train_bc'`. That function
  exists only in the `tools/train_bc.py` copy captured in
  `origin/rescue/f94-window-feed`, not on `master`.
- **`tests/test_manifest_rust_featurize_provenance.py`** — imports fine but
  all 3 tests fail (`AssertionError`/`KeyError`): they assert
  `generate_gumbel_selfplay_data.py::_merge_worker_summaries` promotes
  `rust_featurize` to a top-level manifest field (task #81, "staged but
  unlanded" per prior audit); `master`'s `_merge_worker_summaries` doesn't do
  this yet.
- **`tests/test_flywheel_phase2_integration.py::test_ingest_feed_batches`** —
  `AttributeError: module 'continuous_flywheel_under_test' has no attribute
  'ingest_feed_batches'`. Same root cause as the already-documented
  "Explicitly out of scope" `replay_window.py` gap above: `ingest_feed_batches`
  is a phase-2 addition to the already-tracked `tools/continuous_flywheel.py`
  that lives only in the B200 dev tree (`~/catan-zero-f94feed`), never rsynced
  because the rescue branches only capture untracked files, not diffs to
  tracked ones. (The file's sibling test,
  `test_build_round_corpus_and_mixed_train_window`, correctly `skip`s — it's
  guarded by `skipif` for missing CUDA/checkpoint, per its own docstring.)

All other rescued test files (`test_concat_memmap_corpus.py`,
`test_manifest_rust_featurize_provenance.py`'s collection itself,
`test_rust_action_context_evaluator_wiring.py`,
`test_rust_action_context_parity.py`,
`test_rust_featurize_evaluator_wiring.py`, `test_rust_featurize_parity.py`)
collect and run cleanly on `master`, correctly `skip`ing (not failing) the
`catanatron_rs`-wheel-dependent cases in an environment without the Rust
wheel built.
