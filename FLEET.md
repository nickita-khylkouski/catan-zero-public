# FLEET.md — Catan-Zero fleet source of truth

> **No IPs in this file.** Box identity is by ALIAS; the real alias→ip map lives only in the
> uncommitted `$FLEET_CONF` (default `~/.catan_fleet.conf`), never in the repo. See §2.
> Live per-GPU job assignment is fluid — see `tools/h100/FLEET_QUEUE.md`, not this doc.

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
| B200 | 2× B200 | eval + orchestration hub (gates, Grafana, banking) |

## 2. Fleet config (`$FLEET_CONF`) — the IP boundary
- `FLEET_CONF="${FLEET_CONF:-$HOME/.catan_fleet.conf}"`, a **bash file that is sourced** (not JSON), **uncommitted / gitignored**.
- Defines `declare -A HOST=( [c1]=<ip> ... )` (alias→ip) and optional `GPU_SSH_KEY` (default `~/.ssh/gpu_access_ed25519`).
- Canonical resolver: **`tools/fleet/fleet_lib.sh`** — `source` it, then use `fleet_host <alias>` (echoes ip, rc 2 on unknown), `fleet_key`, `fleet_aliases`. Never hardcode ips.
- Repo commits only `tools/fleet/fleet_conf.example` (placeholder ips). Gitignored: `/.catan_fleet.conf`, `*.fleet.conf`, `/configs/gpu_cluster_hosts.json`.
- This `$FLEET_CONF` is the **single** host source of truth (CAT-137): the old `configs/gpu_cluster_hosts.json` is a `.example` template only.

## 3. Canonical code + environment (CAT-117)
- Repo: **`github.com/nickita-khylkouski/catan-zero-public`** (**PUBLIC**, no auth required), tag **`v1.0-deploy`**.
- Env: **Python 3.11.15**, **torch cu128** (all H100 + B200), **catanatron_rs 0.1.3 cp311**.
- One-command install (canonical, per-box operator command — public, no auth; the rust wheel auto-downloads from the tag's release assets):
  ```
  curl -fsSL https://raw.githubusercontent.com/nickita-khylkouski/catan-zero-public/v1.0-deploy/tools/install_v1_freeze.sh | bash
  ```
  `tools/install_v1_freeze.sh` — clone+checkout tag → py3.11 venv → torch cu128 → `pip install -e .[dev,rl]` → `catanatron_rs` 0.1.3 cp311 wheel (from `$CATAN_RS_WHEEL` if set, else auto-fetched from the release) → env-doctor → rust-featurize parity smoke. `CATAN_REPO` also accepts a local git-bundle path as an offline fallback.
- Fleet acceptance (after install, before the box joins rotation):
  ```
  NOOP_ATOL=1e-4 PY=<venv>/bin/python bash scripts/gate.sh --only noop
  PY=<venv>/bin/python bash scripts/gate.sh --only parity
  ```
  The fleet is homogeneous **INTEL Xeon** (H100 boxes = Xeon 8480+, B200 hub = Xeon 8592+), so the committed byte-exact no-op reference is valid fleet-wide; `NOOP_ATOL=1e-4` is the safety net for any future non-Intel box rather than a routine requirement.

## 4. Rust engine (CAT-133)
- `catanatron-rs` canonical rev **`1400dec` (v0.1.3)** builds the shipped `catanatron_rs-0.1.3-cp311-…manylinux_2_34` wheel (maturin, from `python/`). Runtime is uniform 0.1.3 fleet-wide; the upstream is vendored under `vendor/catanatron/`.
- **Licensing posture: pending user decision — see CAT-138.**

## 5. Seed ledger (CAT-125)
- Cross-host source of truth: **`runs/SEED_LEDGER.md`** (read by `tools/prelaunch_guard.py` overlap guard). ALIAS-keyed, never ip.
- **Claim a fresh, disjoint base-seed block BEFORE any generation run.** Reusing a base produces duplicate `game_seed`s and the pooled-build dedup drops the whole partial wave. `fleet_launch.sh` auto-claims (CLAIM→GUARD→LAUNCH, §6/§7).
- Sync/dedupe copies: `python tools/sync_seed_ledger.py copies/*.md -o runs/SEED_LEDGER.md` (idempotent). CI/pre-commit assert canonical: `python tools/sync_seed_ledger.py runs/SEED_LEDGER.md --check`. Claim rows carry a unique `claim=<id>` token.

## 6. Guard policy (CAT-124)
- Every launch runs `tools/prelaunch_guard.py` WITH guards on. Launch order is **CLAIM → GUARD → LAUNCH**: the launcher writes its own ledger row with a unique claim id, then `ledger_overlap` excludes that own row (by claim id) so a peer's overlapping claim still fails closed.
- **`--skip-guards` is RETIRED** in the canonical launcher (the self-collision that once needed it no longer happens). Do not bypass.

## 7. Launch / stop / status (CAT-122 / CAT-123) — one canonical path
Interpreter is auto-resolved (`$GEN_PY` → `~/venv/bin/python` → `<tree>/.venv/bin/python`); never a bare `torchrun`/`python3` (loads system numpy<2, crashes champion load — CAT-128) and never a hardcoded `.venv` (stranded a GPU — CAT-123). Hosts via `fleet_lib.sh` (§2).
- **Launch** (supersedes all `fire_*.sh`/`mps_rollout.sh`): `tools/fleet/fleet_launch.sh <alias> <role> --base-seed N [--gpus 0-3] [--go]` — `role ∈ {teacher, volume, train}`; `--base-seed` REQUIRED for gen roles (fresh, ledgered); **default DRY-RUN** (prints plan), `--go` to fire. Does CLAIM→GUARD→LAUNCH + setsid-detach (survives SSH teardown).
- **Stop**: `tools/fleet/fleet_stop.sh <alias|all> [--go]` — **default DRY-RUN**; kills by nvidia-smi compute-PID (never `pkill -f` self-match), SIGTERMs python/torchrun supervisors first, PRESERVES the MPS daemon + observability, verifies 0 MiB/GPU.
- **Status**: `tools/fleet/fleet_status.sh [alias|all]` — read-only, parallel; per-box util/mem, inferred role, MPS on/off, launcher count.
- **Harvest → corpus**: `tools/wave1_harvest.sh {harvest-all|build-pooled}` (parallel rsync + ControlMaster; reads `$FLEET_CONF`).
- **Ops rule (CAT-123):** one operator per box; always post-verify a single clean gen set after any change (`fleet_status.sh <box>` + `fleet_stop.sh <box>` dry-run).

## 8. Bring up a new box
1. Add its `[alias]=<ip>` to your local `$FLEET_CONF` (uncommitted).
2. `curl -fsSL https://raw.githubusercontent.com/nickita-khylkouski/catan-zero-public/v1.0-deploy/tools/install_v1_freeze.sh | bash` — canary-first (CAT-130); env-doctor + rust parity smoke must pass.
3. Fleet acceptance: `NOOP_ATOL=1e-4 PY=<venv> bash scripts/gate.sh --only noop` then `PY=<venv> bash scripts/gate.sh --only parity` (§3).
4. Claim a disjoint seed block in `runs/SEED_LEDGER.md` (§5).
5. Launch via `fleet_launch.sh` (§7).

## 9. Observability
- Grafana + Prometheus + DCGM on the **B200 hub**: `http://<B200 alias>:3000` (creds in `~/GRAFANA_CREDS.txt` on B200). Adding a box = one service-discovery line (label `gpumodel`, not `gpu`).
