# Next production n128 self-play wave: sealed preparation record

**Status:** preparation only. This record does not claim seeds, stage hosts, launch a canary, or start production work.

## 1. Canonical path

The next production wave must use the existing sealed transaction:

~~~text
fresh v3 draft
  -> tools/a1_pre_wave_contract.py seal
  -> tools/a1_pre_wave_contract.py verify
  -> tools/a1_pre_wave_contract.py render
  -> tools/a1_pre_wave_contract.py claim
  -> tools/fleet/a1_live_canary.py
  -> tools/fleet/a1_production_executor.py run (dry run)
  -> the same executor command with --go, only after operator approval
~~~

Do not use `fleet_launch.sh`, `gpu_fleet.py`, `continuous_flywheel.py`, Slurm, Ray, or manually execute rendered argv for this wave. The executor stages the lock's content-addressed runtime tree and launches one detached lane supervisor per physical H100.

## 2. Authorized H100 inventory

`configs/gpu_fleet_64.json` is the topology authority. Its exact fleet is 64 H100s on 12 hosts:

| Alias | GPUs | Address |
|---|---:|---|
| c1 | 4 | 192.222.54.251 |
| c2 | 4 | 68.209.75.117 |
| c3 | 4 | 192.222.53.18 |
| c4 | 4 | 68.209.73.252 |
| c5 | 4 | 68.209.74.145 |
| c6 | 4 | 68.209.74.2 |
| c7 | 4 | 68.209.74.24 |
| c8 | 4 | 68.209.72.159 |
| h100-8a | 8 | 192.222.53.119 |
| h100-8b | 8 | 192.222.55.216 |
| h100-8c | 8 | 192.222.54.141 |
| h100-8d | 8 | 209.20.158.82 |

The private executor host file must contain exactly those 12 aliases, be mode 0600, and use a fresh wave-specific `remote_root`. The B200-origin key declared by the authority is `/home/ubuntu/.ssh/catan_fleet_ed25519`; do not route artifact transfer through an operator laptop.

## 3. Science and runtime configuration that must not change

The selected checkpoint replaces only the producer identity. The established generation configuration remains:

~~~json
{
  "track": "2p_no_trade",
  "vps_to_win": 10,
  "source_mix_selected_games": {
    "current_producer": 9600,
    "recent_history": 1800,
    "hard_negative": 600
  },
  "search": {
    "n_full": 128,
    "n_fast": 16,
    "p_full": 0.25,
    "n_full_wide": null,
    "n_full_wide_threshold": null,
    "wide_roots_always_full": false,
    "c_visit": 50.0,
    "c_scale": 0.10,
    "max_depth": 80,
    "correct_rust_chance_spectra": true,
    "lazy_interior_chance": true,
    "belief_chance_spectra": false,
    "information_set_search": true,
    "determinization_particles": 4,
    "determinization_min_simulations": 32,
    "symmetry_averaged_eval": true,
    "symmetry_averaged_eval_threshold": 20
  },
  "evaluator": {
    "public_observation": true,
    "rust_featurize": true,
    "value_squash": "tanh",
    "strict_fp32": true,
    "eval_server": false
  },
  "generation": {
    "workers_per_gpu": 16,
    "max_decisions": 600,
    "temperature_decisions": 90,
    "shard_size": 512,
    "native_mcts_hot_loop": true,
    "mps": "systemd-managed per host"
  }
}
~~~

The 80/15/5 mix is rendered as 192 deterministic jobs, not stochastic opponent selection: current producer, then recent history, then hard negative on each of 64 lanes. `recent_history` must be the exact incumbent displaced by promotion. The hard negative must be a distinct checkpoint with a replayable `a1-hard-negative-selection-v1` record. Every category retains the new producer seat and uses its deployed `c_scale=0.10`; only the opponent changes.

## 4. Safe checkpoint-swap boundary

Use these placeholders while preparing private artifacts:

~~~bash
export SELECTED_CHECKPOINT=/immutable/checkpoints/SELECTED_CHECKPOINT.pt
export PROMOTION_RECEIPT=/immutable/promotion/SELECTED_CHECKPOINT.receipt.json
export POST_PROMOTION_HANDOFF=/immutable/promotion/SELECTED_CHECKPOINT.handoff.json
export DISPLACED_INCUMBENT=/immutable/checkpoints/DISPLACED_INCUMBENT.pt
export HARD_NEGATIVE=/immutable/checkpoints/HARD_NEGATIVE.pt
export HARD_NEGATIVE_SELECTION=/immutable/evidence/HARD_NEGATIVE.selection.json

export WAVE_ID=a1-next-n128-64gpu-YYYYMMDD-r1
export PRIVATE_ROOT=/home/ubuntu/catan-zero-production/private/$WAVE_ID
export ARTIFACT_ROOT=/home/ubuntu/catan-zero-production/contracts/$WAVE_ID
export NEXT_DRAFT=$PRIVATE_ROOT/draft.json
export NEXT_LOCK=$ARTIFACT_ROOT/lock.json
export NEXT_RENDER_DIR=$ARTIFACT_ROOT/render
export NEXT_RENDER=$NEXT_RENDER_DIR/commands.json
export CLAIM_RECEIPT=$ARTIFACT_ROOT/seed-claim.receipt.json
export EXECUTOR_RECEIPT=$ARTIFACT_ROOT/executor.receipt.json
export HOSTS=$PRIVATE_ROOT/hosts.json
~~~

`SELECTED_CHECKPOINT` alone is deliberately insufficient. The canonical seal refuses a raw path swap. It must be the exact generator checkpoint committed by `PROMOTION_RECEIPT`, and the handoff must replay the live registry and `CURRENT_CHAMPION` state:

~~~bash
python tools/a1_post_promotion_handoff.py \
  --promotion-receipt "$PROMOTION_RECEIPT" \
  --out "$POST_PROMOTION_HANDOFF"
~~~

The fresh draft starts from `configs/experiments/a1_pre_wave_contract.template.json`. Resolve only the following wave-specific fields; do not edit an old sealed lock or render:

| Draft field | Required value |
|---|---|
| `contract_id` | `$WAVE_ID` |
| `promotion_handoff.path` | `$POST_PROMOTION_HANDOFF` |
| producer checkpoint path | `$SELECTED_CHECKPOINT` |
| history checkpoint path | `$DISPLACED_INCUMBENT` from the same promotion receipt |
| hard-negative path/version/evidence | `$HARD_NEGATIVE` plus `$HARD_NEGATIVE_SELECTION` |
| A0 evidence | fresh evidence for the selected learner/value objective |
| S1, S2, S3 evidence | fresh decisions whose checkpoint path and SHA-256 are exactly `$SELECTED_CHECKPOINT` |
| `fleet.seed_base` | allocator-approved next-safe base for a 64,000-seed wave block |
| `fleet.seed_ledger` | canonical append-only production ledger |
| `fleet.output_root` | a fresh immutable wave output root |

All remaining resolved science fields must equal Section 3. In particular, no n64/n196/n256 substitution, adaptive-wide override, EvalServer, belief-chance spectra, unmasked observation, or authoritative-state search is allowed.

Before sealing, the template must report no unresolved fields:

~~~bash
python tools/a1_pre_wave_contract.py inspect-template --draft "$NEXT_DRAFT"
# Accept only: {"schema_version":"a1-pre-wave-contract-draft-v3","unresolved":[]}

python tools/a1_pre_wave_contract.py sync-generation-guard --draft "$NEXT_DRAFT"
python tools/a1_pre_wave_contract.py seal --draft "$NEXT_DRAFT" --out "$NEXT_LOCK"
python tools/a1_pre_wave_contract.py verify --lock "$NEXT_LOCK"
python tools/a1_pre_wave_contract.py render \
  --lock "$NEXT_LOCK" --out-dir "$NEXT_RENDER_DIR"
~~~

`sync-generation-guard` is the existing narrow guard synchronizer; it does not launch. Inspect its diff before sealing. The render must contain exactly 64 lanes, 192 jobs, and the 9,600/1,800/600 selected quotas.

## 5. Exact private executor configuration

Create `$HOSTS` as mode 0600 with this exact schema and a fresh remote root:

~~~json
{
  "schema_version": "a1-production-hosts-v1",
  "ssh_user": "ubuntu",
  "ssh_key": "/home/ubuntu/.ssh/catan_fleet_ed25519",
  "remote_root": "/home/ubuntu/a1-production-WAVE_ID",
  "python": "/home/ubuntu/catan-zero-v1/.venv/bin/python",
  "hosts": {
    "c1": "192.222.54.251",
    "c2": "68.209.75.117",
    "c3": "192.222.53.18",
    "c4": "68.209.73.252",
    "c5": "68.209.74.145",
    "c6": "68.209.74.2",
    "c7": "68.209.74.24",
    "c8": "68.209.72.159",
    "h100-8a": "192.222.53.119",
    "h100-8b": "192.222.55.216",
    "h100-8c": "192.222.54.141",
    "h100-8d": "209.20.158.82"
  }
}
~~~

Replace only `WAVE_ID` in `remote_root`; then `chmod 0600 "$HOSTS"`. The executor rejects a missing key, a permissive host file, an alias mismatch, a relative runtime path, or a reused incompatible remote receipt tree.

## 6. Claim, canary, and executor commands

The following are the exact next commands, but they cross state boundaries and were intentionally not executed during preparation.

First claim all rendered ranges atomically after the global allocator freezes other writers:

~~~bash
python tools/a1_pre_wave_contract.py claim \
  --lock "$NEXT_LOCK" \
  --render "$NEXT_RENDER" \
  --receipt "$CLAIM_RECEIPT"
~~~

Then run both production-shape canaries through the selective canonical transaction. Use a fresh repository-declared validation-only base:

~~~bash
python tools/fleet/a1_live_canary.py run \
  --lock "$NEXT_LOCK" \
  --render "$NEXT_RENDER" \
  --hosts "$HOSTS" \
  --receipt "$ARTIFACT_ROOT/live-canary.receipt.json" \
  --canary-id "$WAVE_ID-canary" \
  --base-seed FRESH_VAL_ONLY_BASE

# Only after reviewing the 12-lane/36-job dry plan:
# repeat the identical command with --go, then:
python tools/fleet/a1_live_canary.py status \
  --lock "$NEXT_LOCK" --render "$NEXT_RENDER" --hosts "$HOSTS" \
  --receipt "$ARTIFACT_ROOT/live-canary.receipt.json" \
  --canary-id "$WAVE_ID-canary" --base-seed FRESH_VAL_ONLY_BASE
python tools/fleet/a1_live_canary.py audit \
  --lock "$NEXT_LOCK" --render "$NEXT_RENDER" --hosts "$HOSTS" \
  --receipt "$ARTIFACT_ROOT/live-canary.receipt.json" \
  --canary-id "$WAVE_ID-canary" --base-seed FRESH_VAL_ONLY_BASE
~~~

Finally inspect the full production plan. This command is read-only because it intentionally omits `--go`:

~~~bash
python tools/fleet/a1_production_executor.py run \
  --lock "$NEXT_LOCK" \
  --render "$NEXT_RENDER" \
  --hosts "$HOSTS" \
  --receipt "$EXECUTOR_RECEIPT"
~~~

Accept that dry plan only if it reports the exact lock/render, 12 hosts, 64 lanes, 192 jobs/claims, content-addressed runtime tree, checkpoint/opponent/ledger hashes, and accepted four- plus eight-GPU canary evidence. The actual production launch is the byte-identical command plus `--go`; it is outside this preparation record.

## 7. Current blockers

1. `SELECTED_CHECKPOINT` has not yet supplied a committed promotion receipt and replayable post-promotion handoff in this preparation context.
2. A new v3 seal requires A0 and S1/S2/S3 evidence tied to the exact selected checkpoint path and SHA-256. Reusing the incumbent's evidence is refused.
3. The displaced-incumbent identity and any retained hard negative must be distinct, versioned, and authenticated by the promotion/selection receipts.
4. The canonical ledger must be reconciled across all 12 hosts, and a fresh 64,000-seed block must be allocated by the one global operator.
5. The private mode-0600 host file, B200-origin SSH key, and fresh remote root must exist on the orchestration host.
6. Both the four-GPU and eight-GPU live canaries must pass with the real selected checkpoint before any full-fleet `--go`.
7. The working tree used to seal must be the reviewed canonical tree. A dirty research checkout must not become production authority merely because the seal can hash it.

Until all seven are resolved, the correct state is a prepared template and dry command, not a launch.
