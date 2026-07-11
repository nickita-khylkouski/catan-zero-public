# A1 dual-arm 56-GPU generation contract

This directory freezes the requested production science and allocation without
launching or claiming seeds. The canonical contract is intentionally marked
`blocked_pending_post_promotion_handoff` and both seal and launch authorization
are false.

The two arms preserve the deployed recipe (`p_full=.25`, `n_fast=16`) and differ
only in the requested full-search budget and operational allocation:

- `n256`: 28 logical GPU lanes, 2,000 selected games each split 1,600/300/100
  current/history/hard-negative, with 1,640/310/104 maximum attempts.
- `n128`: 28 logical GPU lanes, 5,000 selected games each split 4,000/750/250,
  with 4,080/765/255 maximum attempts.

Every logical GPU is allocated one uniform 8,192-seed block. The n256 allocation spans
`[300000168192, 300000397568)` and the adjacent n128 allocation spans
`[300000397568, 300000626944)`. Category jobs consume disjoint prefixes within
their GPU block. The ledger transaction claims the three exact maximum-attempt
prefixes, not the unused tail; the sealed 8,192 stride prevents either arm from
using another lane's tail. Any later campaign must begin at or after
`300000626944` rather than recycling those deliberately unused gaps.

Both use the A1 checkpoint with SHA-256
`f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4`,
public information-set search, `c_scale=.10`, D6 at legal width 20, and 16 MPS
workers per GPU. Output roots and logical lanes are disjoint. The contract also
binds the current generation guard, generator implementation, production
executor, and harvest implementation bytes.
The 15% history quota is the immediate prior gen3 champion (`89aa133d...`),
while the 5% hard-negative quota remains held gen4 (`b0f93946...`).
`placement.assignments.json` uses every GPU from `configs/gpu_fleet_56.json`
exactly once and never splits a host across arms: n256 owns h100-8a/h100-8c and
c1/c3/c5; n128 owns h100-8b/h100-8d and c2/c4/c6. This permits the two existing
per-arm executors to preflight and launch concurrently without seeing the other
arm as foreign compute on a shared host.

New generation waves require a committed post-promotion producer handoff. That
immutable handoff is not present in this checkout. The exact 56-GPU assignment
is checked in, but must be sealed against this campaign before materialization.
Missing either immutable input blocks materialization; neither is guessed.

Read-only validation:

```bash
python3 tools/a1_pre_wave_contract.py verify-generation-campaign \
  --contract configs/operations/a1-dual-arm-56gpu-20260710/contract.json
```

Adding `--require-ready` refuses while those inputs are absent. Once the operator
has a canonical `a1-dual-arm-generation-placement-v1` file containing all 56
unique `{logical_lane, host_alias, gpu}` assignments, the exact workflow is:

```bash
CAMPAIGN=configs/operations/a1-dual-arm-56gpu-20260710/contract.json
HANDOFF=/home/ubuntu/catan-zero-production/a1-post-promotion-handoff.json
PLACEMENT=/home/ubuntu/catan-zero-production/a1-dual-arm-placement.json
ASSIGNMENTS=configs/operations/a1-dual-arm-56gpu-20260710/placement.assignments.json
RUN=/home/ubuntu/catan-zero-production/contracts/a1-dual-arm-20260710-r1
HOSTS_N256=/home/ubuntu/catan-zero-production/private/a1-production-hosts-n256.json
HOSTS_N128=/home/ubuntu/catan-zero-production/private/a1-production-hosts-n128.json

python3 tools/a1_pre_wave_contract.py seal-generation-placement \
  --contract "$CAMPAIGN" --assignments "$ASSIGNMENTS" --out "$PLACEMENT"
python3 tools/a1_pre_wave_contract.py materialize-generation-campaign \
  --contract "$CAMPAIGN" --promotion-handoff "$HANDOFF" \
  --placement "$PLACEMENT" --out-dir "$RUN/locks"

for ARM in n256 n128; do
  HOSTS_VAR="HOSTS_${ARM^^}"
  python3 tools/a1_pre_wave_contract.py render --lock "$RUN/locks/$ARM.lock.json" \
    --out-dir "$RUN/render-$ARM"
  python3 tools/a1_pre_wave_contract.py claim --lock "$RUN/locks/$ARM.lock.json" \
    --render "$RUN/render-$ARM/commands.json" \
    --receipt "$RUN/$ARM.seed-claim.receipt.json"
  python3 tools/fleet/a1_production_executor.py run \
    --lock "$RUN/locks/$ARM.lock.json" \
    --render "$RUN/render-$ARM/commands.json" \
    --hosts "${!HOSTS_VAR}" \
    --receipt "$RUN/$ARM.executor.receipt.json"
done
```

The executor commands above are dry-run/preflight only. After reviewing both
plans, the operator repeats each with `--go`. Harvest remains per arm:

```bash
python3 tools/fleet/a1_harvest_transaction.py \
  --lock "$RUN/locks/n256.lock.json" --render "$RUN/render-n256/commands.json" \
  --destination /home/ubuntu/catan-zero-production/harvest/a1-dual-arm-n256
python3 tools/fleet/a1_harvest_transaction.py \
  --lock "$RUN/locks/n128.lock.json" --render "$RUN/render-n128/commands.json" \
  --destination /home/ubuntu/catan-zero-production/harvest/a1-dual-arm-n128

python3 tools/a1_pre_wave_contract.py audit \
  --lock "$RUN/locks/n256.lock.json" \
  --harvest-relocation /home/ubuntu/catan-zero-production/harvest/a1-dual-arm-n256/relocation_map.json \
  --out "$RUN/n256.post-wave-audit.json"
python3 tools/a1_pre_wave_contract.py audit \
  --lock "$RUN/locks/n128.lock.json" \
  --harvest-relocation /home/ubuntu/catan-zero-production/harvest/a1-dual-arm-n128/relocation_map.json \
  --out "$RUN/n128.post-wave-audit.json"
```

No checked-in command launches work.
