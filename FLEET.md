# FLEET.md — Catan-Zero fleet source of truth

> **No IPs in this file.** Box identity is by ALIAS; the real alias→ip map lives only in the
> uncommitted `$FLEET_CONF` (default `~/.catan_fleet.conf`), never in the repo. See §2.
> Live per-GPU job assignment is fluid; use `tools/fleet/fleet_status.sh`, not a
> committed queue snapshot.
> The end-to-end RL operator transaction is in `RL_AGENT_HANDOFF.md`.

## 1. Box inventory (aliases + stable roles)
Fleet is consolidated to **H100 + B200 only** (24× H100 across 6 boxes, NVLink,
homogeneous INTEL Xeon hosts). The prior A100 pool (`a100a`, `a100b`) and the
older `a100-legacy` box are **RETIRED** — decommissioned from the active fleet;
any useful data on them was salvaged separately before retirement. Do not
launch new work there, and drop any lingering A100 entries from your local
`$FLEET_CONF`.

| Alias | Hardware | Typical role |
|---|---|---|
| c1 | 4× H100 (NVLink) | volume gen (n64, p0.25) |
| c2 | 4× H100 | teacher gen (n128, p1.0) |
| c3 | 4× H100 | teacher gen (n128, p1.0) |
| c4 | 4× H100 | control / training (DDP/FSDP) |
| c5 | 4× H100 | volume gen (n64, p0.25) |
| c6 | 4× H100 | teacher gen (n128, p1.0) |
| h100-canary | 8× H100 (NVSwitch) | validation/performance lab; outside the 24 production H100s |
| b200 | 2× B200 | eval + orchestration hub (gates, Grafana, banking) |

## 2. Fleet config (`$FLEET_CONF`) — the IP boundary
- `FLEET_CONF="${FLEET_CONF:-$HOME/.catan_fleet.conf}"`, a **bash file that is sourced** (not JSON), **uncommitted / gitignored**.
- Defines `declare -A HOST=( [c1]=<ip> ... )` (alias→ip) and optional `GPU_SSH_KEY` (default `~/.ssh/gpu_access_ed25519`).
- Canonical resolver: **`tools/fleet/fleet_lib.sh`** — `source` it, then use `fleet_host <alias>` (echoes ip, rc 2 on unknown), `fleet_key`, `fleet_aliases`. Never hardcode ips.
- Repo commits only `tools/fleet/fleet_conf.example` (placeholder ips). Gitignored: `/.catan_fleet.conf`, `*.fleet.conf`, `/configs/gpu_cluster_hosts.json`.
- This `$FLEET_CONF` is the **single** host source of truth (CAT-137):
  `configs/gpu_cluster_hosts.example.json` is only a historical JSON example.

## 3. Canonical code + environment (CAT-117)
- Repo: **`github.com/nickita-khylkouski/catan-zero-public`** (**PUBLIC**, no
  auth required). **Release blocker:** the verified H100 changes currently live
  only in this worktree; `v1.0-deploy` predates them. Publish an immutable new
  release tag with the `catanatron_rs` wheel asset before provisioning any fleet
  box.
- Env: **Python 3.11.15**, **torch cu128** (all H100 + B200), **catanatron_rs 0.1.3 cp311**.
- Verification snapshot (2026-07-09): local full suite **1,737 passed / 200
  skipped**; H100 full suite **1,913 passed / 24 skipped**; native
  feature/context/symmetry acceptance **19/19 passed**. The final handoff delta
  passed **184/184** targeted H100 tests and returned all eight canary GPUs to
  0 MiB.
- Canonical four-GPU training smoke completed **5/5 steps** on a **21,120-row,
  352-shard memmap** in 5.76 seconds of reported train time, wrote the 35M
  masked model/report/optimizer artifacts, and returned all GPUs to 0 MiB.
- After publishing the verified tree, a fresh host downloads the installer from
  that same explicit immutable release tag (the installer intentionally has no
  stale default):
  ```
  export CATAN_REF=<published-h100-release-tag>
  curl -fsSL "https://raw.githubusercontent.com/nickita-khylkouski/catan-zero-public/${CATAN_REF}/tools/install_v1_freeze.sh" \
    | CATAN_REF="$CATAN_REF" bash
  ```
  `tools/install_v1_freeze.sh` — clone+checkout tag → py3.11 venv → torch cu128 → `pip install -e vendor/catanatron` → `pip install -e .[dev,rl]` → `catanatron_rs` 0.1.3 cp311 wheel (from `$CATAN_RS_WHEEL` if set, else auto-fetched from the tagged release) → env-doctor → rust-featurize parity smoke. A commit ref is supported only with an explicit staged `$CATAN_RS_WHEEL`; `CATAN_REPO` also accepts a local git-bundle path as an offline fallback.
- Fleet acceptance (after install, after staging the private masked champion at
  `~/bundle/champion_v0.pt`, and before the box joins rotation):
  ```
  NOOP_ATOL=1e-4 PY=<venv>/bin/python bash scripts/gate.sh --only noop
  PY=<venv>/bin/python bash scripts/gate.sh --only parity
  ```
  The public checkout does not contain that champion, so the no-op gate remains
  blocked until it is staged and its reference is verified. The fleet is
  homogeneous **INTEL Xeon**; `NOOP_ATOL=1e-4` is only the safety net for a
  future non-Intel box.

## 4. Rust engine (CAT-133)
- `catanatron-rs` canonical rev **`1400dec` (v0.1.3)** builds the shipped `catanatron_rs-0.1.3-cp311-…manylinux_2_34` wheel (maturin, from `python/`). Runtime is uniform 0.1.3 fleet-wide; the upstream is vendored under `vendor/catanatron/`.
- **Licensing posture: pending user decision — see CAT-138.**

## 5. Seed ledger (CAT-125)
- Cross-host source of truth: **`runs/SEED_LEDGER.md`** (read by `tools/prelaunch_guard.py` overlap guard). ALIAS-keyed, never ip. The launcher has no shared cross-host lock; one operator must merge, inspect, and redistribute byte-identical copies before allocating a wave.
- **Claim a fresh, disjoint base-seed block BEFORE any generation run.** Reusing a base produces duplicate `game_seed`s and the pooled-build dedup drops the whole partial wave. `fleet_launch.sh` appends the claim before starting detached per-GPU children; each child runs its guards before spawning game workers (§6/§7).
- Sync/dedupe copies: `python tools/sync_seed_ledger.py copies/*.md -o runs/SEED_LEDGER.md` (idempotent). CI/pre-commit assert canonical: `python tools/sync_seed_ledger.py runs/SEED_LEDGER.md --check`. Claim rows carry a unique `claim=<id>` token.

## 6. Guard policy (CAT-124)
- Every generation child runs `tools/prelaunch_guard.py` WITH guards on. The launcher writes its own ledger row with a unique claim id, starts the detached runner, and each per-GPU child excludes that own row by claim id while still rejecting a peer overlap. Dry-run checks remote prerequisites but does not execute the complete dynamic child guard; after `--go`, inspect every per-GPU log.
- **`--skip-guards` is RETIRED** in the canonical launcher (the self-collision that once needed it no longer happens). Do not bypass.

## 7. Launch / stop / status (CAT-122 / CAT-123) — one canonical path
Interpreter is auto-resolved (`$GEN_PY` → `~/venv/bin/python` → `<tree>/.venv/bin/python`); never a bare `torchrun`/`python3` (loads system numpy<2, crashes champion load — CAT-128) and never a hardcoded `.venv` (stranded a GPU — CAT-123). Hosts via `fleet_lib.sh` (§2).
- **Launch** (supersedes all `fire_*.sh`/`mps_rollout.sh`): `tools/fleet/fleet_launch.sh <alias> <role> --base-seed N [--gpus 0-3] [--go]` — `role ∈ {teacher, volume, train}`; `--base-seed` REQUIRED for gen roles (fresh, ledgered); **default DRY-RUN** (prints plan), `--go` to fire. Generation uses one strict-FP32 EvalServer per GPU, Rust features, cache off, immediate queue drain, GPU-local CPU affinity, and fail-fast clients. The canonical n128 production shape (≤4 selected GPUs) uses **128 workers/GPU, request collector on, max batch 96, wait 0 ms, `matmul_precision=highest`, cache 0, no MPS**; an all-8 canary launch defaults to 64 workers/GPU to preserve the same total host concurrency. Volume retains its separate 48/32 worker, batch64, collector-off recipe. MPS is diagnostic only. The detached runner survives SSH teardown. A zero launcher exit does not attest every child; reconcile manifests before harvest.
- **Stop**: `tools/fleet/fleet_stop.sh <alias|all> [--go]` — **default DRY-RUN**; terminates validated `launch_detached` process groups (so MPS-hidden clients and grandchildren cannot escape), retains explicit compute-PID fallback, PRESERVES MPS/observability, and fails unless owned groups, MPS clients, and non-infrastructure GPU PIDs are gone. Idle memory must be ≤50 MiB without MPS or ≤128 MiB for the measured 78 MiB/GPU preserved MPS-server baseline on driver 580.105.08.
- **Status**: `tools/fleet/fleet_status.sh [alias|all]` — read-only, parallel; per-box util/mem, inferred role, MPS on/off, matching job-process count.
- **Harvest → corpus**: `tools/wave1_harvest.sh {harvest-all|build-teacher|build-volume}` (parallel rsync + ControlMaster; reads `$FLEET_CONF`). Populate `DIRS` from accepted claim paths and reconcile harvested counts against remote manifests before a role-pure build. `build-pooled` is experiment-only after a predeclared mixture decision.
- **Ops rule (CAT-123):** one operator per box; always post-verify a single clean gen set after any change (`fleet_status.sh <box>` + `fleet_stop.sh <box>` dry-run).

### n128 teacher throughput lock (2026-07-09)

At w48, wait `0/0.05/0.1/0.25 ms` measured
`72.26/70.54/70.04/71.07k` rows/hour/GPU, locking wait 0. Before the collector
fix, workers `48/64/80/96` measured `68.07/74.41/74.65/75.98k`; with the fixed
collector enabled, four w96 repetitions averaged **81.93k**.

The final synthetic-checkpoint frontier measured **91.85k rows/hour/GPU** for
the canonical w128/batch96/collector recipe, about **37% above** the earlier
~67k w48 teacher baseline. Across 24 H100s this projects to approximately
**2.20M rows/hour**. Supporting paired results were w96 **83.42k** versus w128
**89.57k** (+7.4%), then batch64 **90.50k** versus batch96 **91.85k** (+1.5%)
at w128.

These are throughput-only results from a synthetic same-shape masked 35M
checkpoint. Repeat the final recipe with the real masked champion before
treating the projection as production capacity. TF32 remains rejected after
same-seed trajectory divergence; `matmul_precision=highest` is mandatory.

## 8. Bring up a new box
1. Add its `[alias]=<ip>` to your local `$FLEET_CONF` (uncommitted).
2. Publish this verified tree as an immutable release tag with the Rust wheel,
   then run the fresh-host `curl | CATAN_REF=... bash` command in §3 canary-first
   (CAT-130); env-doctor + Rust parity smoke must pass.
3. Fleet acceptance: `NOOP_ATOL=1e-4 PY=<venv> bash scripts/gate.sh --only noop` then `PY=<venv> bash scripts/gate.sh --only parity` (§3).
4. Claim a disjoint seed block in `runs/SEED_LEDGER.md` (§5).
5. Launch via `fleet_launch.sh` (§7).

## 9. Observability
- Grafana + Prometheus + DCGM on the **b200 hub**: `http://<b200 alias>:3000` (creds in `~/GRAFANA_CREDS.txt` on b200). Adding a box = one service-discovery line (label `gpumodel`, not `gpu`).
