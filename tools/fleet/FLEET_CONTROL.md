# Fleet control scripts (CAT-123)

Canonical, robust start/stop/status for the GPU fleet. Built after a fleet stop took ~8 passes
because of fragile `pkill -f` patterns. All read the box registry from `$FLEET_CONF`
(default `~/.catan_fleet.conf`, an alias→ip bash file; copy `fleet_conf.example` and fill it —
the filled file is gitignored so no IPs land in the repo), use `ssh -i $GPU_SSH_KEY`
(default `~/.ssh/gpu_access_ed25519`, BatchMode), and are safe to run while the fleet is frozen.

## fleet_stop.sh — robust GPU-work stop
`fleet_stop.sh <alias|all> [--go]`   (default DRY-RUN; prints the plan, kills nothing)

Design (each rule fixes a real failure seen on 2026-07-09):
- **Kill by nvidia-smi compute-PID**, never `pkill -f <pattern>` — a pattern matches the operator's
  own ssh/bash shell and the pgrep command itself, dropping the connection mid-kill.
- **Climb to python/torchrun SUPERVISORS and SIGTERM them FIRST** so launchers reap their workers
  and don't respawn; the climb **stops at the first non-python ancestor** (bash/sshd/systemd) so the
  operator's shell is never a target. Only `comm=python|python3*|torchrun` are ever supervisors.
- **Orphan workers** (parent already gone) are killed directly from the compute-PID list.
- **PRESERVE** the MPS daemon (`nvidia-cuda-mps-control/-server`) and observability
  (`dcgm`, `nv-hostengine`, `prometheus`, `grafana`, `node_exporter`, `*exporter`) — excluded by process_name.
- SIGTERM supervisors → 5s → SIGKILL survivors (explicit PIDs only) → **verify 0 MiB per GPU** (retried).
- Per-box (`fleet_stop.sh c6 --go`) or fleet-wide (`fleet_stop.sh all --go`).

Validated: dry-run against a live `train_bc` (b200) selects the worker + preserves infra; mock
unit-test (two python workers under torchrun/python parents under bash, plus an mps-server) selects
both workers + both supervisors, **excludes the mps-server, and never climbs into bash**.

## fleet_status.sh — one-read fleet view
`fleet_status.sh [alias|all]`   (read-only, parallel)

Per box: `gpus`, `busy` (>50%), `util_avg`, `mem_max`, inferred **role** (TRAINING / GATE(cross-net)
/ EVAL(vs-bot) / GEN-TEACHER(nNN) / GEN-VOLUME(nNN) / idle from live cmdlines), MPS on/off, and
launcher-process count.

## fleet_launch_safe.sh — safe launch-path stub (CAT-122 builds on this)
`fleet_launch_safe.sh <alias> <gpu> <teacher|volume> <base_seed> [--go]`   (default DRY-RUN)

Enforces the three preconditions that incidents proved non-negotiable, then prints the exact command
and refuses on any guard failure:
1. **Fresh out-dir** (timestamped ⇒ fresh by construction; refuses a populated dir).
2. **Seed claim** — base seed must fall inside a CLAIMED ledger range (range-membership parse,
   tolerates commas + en-dash + open-ended rows). Wave restarts that REUSE a base produce duplicate
   game_seeds and the pooled-build dedup guard drops the whole partial wave — so never reuse; claim a
   fresh block first. (Authoritative overlap check remains the tool's own `prelaunch_guard`; this is a
   fast fail-before-ssh heuristic on the box-local ledger.)
3. **Guards ON** — the safe path runs WITH the tool's prelaunch guards. `--skip-guards` is the
   deliberate exception for in-block wave-restart self-collision only, and must be explicit.
- **`$GEN_PY` resolution**: prefer `$GEN_PY`, else `~/venv/bin/python`, else `<tree>/.venv/bin/python`.
  Never hardcode `.venv/bin/python` — on the H100 boxes it doesn't exist (they run under `~/venv`) and
  a hardcode stranded a GPU (stopped gen, failed to relaunch).

## Ops lesson baked in: one operator per box
Three two-operator collisions on 2026-07-09 (crossed c6 conversion, shared-pgid c1 kill, duplicate
c6 relaunch). Route all fleet changes through the single box owner; always post-verify a single clean
gen set after any conversion (`fleet_status.sh <box>` + `fleet_stop.sh <box>` dry-run).
