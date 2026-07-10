# Fleet control scripts (CAT-123)

Canonical, robust start/stop/status for the GPU fleet. Built after a fleet stop took ~8 passes
because of fragile `pkill -f` patterns. All read the box registry from `$FLEET_CONF`
(default `~/.catan_fleet.conf`, an alias→ip bash file; copy `fleet_conf.example` and fill it —
the filled file is gitignored so no IPs land in the repo), use `ssh -i $GPU_SSH_KEY`
(default `~/.ssh/gpu_access_ed25519`, BatchMode), and are safe to run while the fleet is frozen.
Use `fleet_launch.sh` for every generation/training start. The full operator
transaction is documented in `RL_AGENT_HANDOFF.md`.

Generation pins the validated H100 inference path explicitly: `mp_queue`
transport, zero event tokens (the live/public event tail is empty), and the
root-wave and CUDA-graph experiments disabled. Training defaults are unchanged.

`fleet_launch.sh ... --pipelines-per-gpu 2` opts into two independent
generator/EvalServer processes on each selected physical GPU. It does not
double the requested work: workers split evenly (for example, 128 becomes
64+64), while games and the corresponding per-GPU seed interval split into
contiguous, non-overlapping subranges. The default remains one pipeline and
keeps the established `gpuN/` layout. Dual mode uses `gpuN_pipeline0/` and
`gpuN_pipeline1/`, separate `run.log` files and child PID records, and records
the topology, pipeline index, and unique pipeline id in each manifest. Both
pipelines share the full GPU-local CPU affinity so Linux schedules the combined
worker population without separating SMT siblings. This topology remains an
opt-in saturation experiment until it clears the production adoption bar.

## fleet_stop.sh — robust GPU-work stop
`fleet_stop.sh <alias|all> [--go]`   (default DRY-RUN; prints the plan, kills nothing)

Design (each rule fixes a real failure seen on 2026-07-09):
- **Kill validated recorded launch sessions first**, never `pkill -f <pattern>` — each
  `launch_detached` PID is also its SID/PGID, so one exact negative-PGID signal reaches the generator,
  EvalServer, manager, and grandchildren even when MPS hides clients from NVML. PID reuse is rejected
  unless the current SID/PGID and canonical Catan command signature both validate.
- **Keep nvidia-smi compute-PID fallback** for legacy/unmanaged jobs, but admit a PID as a stop
  target only when its command or an ancestor has a canonical Catan signature. Unrelated CUDA jobs
  are reported and preserved. For admitted jobs, climb to python/torchrun supervisors but stop at
  the first non-python ancestor, so the operator shell is never a target.
- **Orphan workers** (parent already gone) are killed directly from the compute-PID list.
- **PRESERVE** the MPS daemon (`nvidia-cuda-mps-control/-server`) and observability
  (`dcgm`, `nv-hostengine`, `prometheus`, `grafana`, `node_exporter`, `*exporter`) — excluded by process_name.
- SIGTERM groups/supervisors → bounded wait → SIGKILL exact survivors → verify zero owned groups,
  zero MPS clients, and zero ancestry-validated Catan GPU PIDs. The preserved MPS server's measured idle
  footprint on H100/driver 580.105.08 is 78 MiB/GPU, so idle memory is capped at 128 MiB with MPS
  (50 MiB otherwise); a 35M evaluator context is ~1.0 GiB and still fails closed.
- Per-box (`fleet_stop.sh c6 --go`) or fleet-wide (`fleet_stop.sh all --go`).

Validated on the 8×H100 canary: a real MPS generation tree (runner + generator + resource tracker +
EvalServer + worker) was selected as one detached group; `--go` removed the full group and the live
MPS client while preserving the server. Post-stop memory was exactly 78 MiB on all eight GPUs.

## fleet_status.sh — one-read fleet view
`fleet_status.sh [alias|all]`   (read-only, parallel)

Per box: `gpus`, `busy` (>50%), `util_avg`, `mem_max`, inferred **role** (TRAINING / GATE(cross-net)
/ EVAL(vs-bot) / EVAL(vs-raw) / TEST(pytest) / GEN-TEACHER(nNN) / GEN-VOLUME(nNN) / idle from live cmdlines), MPS on/off, and
matching generation/training process count plus `gen_pipelines`, the number of
live generator processes (so dual mode is visible rather than mistaken for a
duplicate launch).

The retired `fleet_launch_safe.sh` was removed. It pointed at the old runsix tree and old MPS
recipe. Do not use it.

## Launch failure policy

`fleet_launch.sh` verifies every requested GPU is present and has no non-MPS CUDA client before
creating a run directory or appending a seed claim. Ledger append, detach-library loading, child
startup, and the early-exit check all fail the command nonzero. A successful detached PID is
verified alive with `PID == SID == PGID` before it is published. The production fleet surface also
rejects opponent-mix generation (not compatible with its mandatory shared EvalServer) and any
`c_scale` other than the certified `0.03` before SSH.

## Ops lesson baked in: one operator per box
Three two-operator collisions on 2026-07-09 (crossed c6 conversion, shared-pgid c1 kill, duplicate
c6 relaunch). Route all fleet changes through the single box owner; always post-verify a single clean
gen set after any conversion (`fleet_status.sh <box>` + `fleet_stop.sh <box>` dry-run).
