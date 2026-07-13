# FLEET.md — Catan-Zero fleet source of truth

> **No IPs in this file.** Box identity is by ALIAS; the real alias→ip map lives only in the
> uncommitted `$FLEET_CONF` (default `~/.catan_fleet.conf`), never in the repo. See §2.
> Live per-GPU job assignment is fluid; use `tools/fleet/fleet_status.sh`, not a
> committed queue snapshot.
> The end-to-end RL operator transaction is in `RL_AGENT_HANDOFF.md`.

## 1. Box inventory (aliases + stable roles)
Fleet is consolidated to **H100 + B200 only**. The production data lane has
**56 H100s across ten boxes**: six four-GPU nodes and four eight-GPU nodes,
all with NVLink/NVSwitch. The prior A100 pool (`a100a`, `a100b`) and the older
`a100-legacy` box are **RETIRED** — decommissioned from the active fleet; any
useful data on them was salvaged separately before retirement. Do not launch
new work there, and drop any lingering A100 entries from your local
`$FLEET_CONF`.

| Alias | Hardware | Typical role |
|---|---|---|
| c1 | 4× H100 (NVLink) | A1 generation |
| c2 | 4× H100 (NVLink) | A1 generation |
| c3 | 4× H100 (NVLink) | A1 generation |
| c4 | 4× H100 (NVLink) | A1 generation |
| c5 | 4× H100 (NVLink) | A1 generation |
| c6 | 4× H100 (NVLink) | A1 generation |
| h100-8a | 8× H100 (NVSwitch) | A1 generation; eight-GPU shape canary first |
| h100-8b | 8× H100 (NVSwitch) | A1 generation |
| h100-8c | 8× H100 (NVSwitch) | A1 generation; audited 2026-07-10 |
| h100-8d | 8× H100 (NVSwitch) | A1 generation; audited 2026-07-10 |
| b200 | 2× B200 | eval + orchestration hub (gates, Grafana, banking) |

The current A1 search decision is uniform across all 56 H100s:
`n_full=128`, `n_fast=16`, and `p_full=0.25`. There is no n64 production arm
and no adaptive or blanket n196/n256 budget in this wave. Source categories are
rendered as separate deterministic jobs from the sealed A1 contract; the box
table is not a teacher/volume role split.

## 2. Fleet config (`$FLEET_CONF`) — the IP boundary
- `FLEET_CONF="${FLEET_CONF:-$HOME/.catan_fleet.conf}"`, a **bash file that is sourced** (not JSON), **uncommitted / gitignored**.
- Defines `declare -A HOST=( [c1]=<ip> ... )` (alias→ip) and optional `GPU_SSH_KEY` (default `~/.ssh/gpu_access_ed25519`).
- Canonical resolver: **`tools/fleet/fleet_lib.sh`** — `source` it, then use `fleet_host <alias>` (echoes ip, rc 2 on unknown), `fleet_key`, `fleet_aliases`. Never hardcode ips.
- Repo commits only `tools/fleet/fleet_conf.example` (placeholder ips). Gitignored: `/.catan_fleet.conf`, `*.fleet.conf`, `/configs/gpu_cluster_hosts.json`.
- This `$FLEET_CONF` is the **single** host source of truth (CAT-137):
  `configs/gpu_cluster_hosts.example.json` is only a historical JSON example.

## 3. Canonical code + environment (CAT-117)
- Repo: **`github.com/nickita-khylkouski/catan-zero-public`** (**PUBLIC**, no
  auth required). The canonical native-MCTS release is
  **`v1.5-public-award-parity`**, created only at checksum commit B of the two-commit
  transaction below. All earlier tags predate the `catanatron_rs 0.1.8` native
  search API and must not provision or resume native-MCTS lanes.
- Env target: **Python 3.11.15**, **torch cu128** (all H100 + B200), **catanatron_rs 0.1.8 cp311**.
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
  `tools/install_v1_freeze.sh` — clone+checkout tag → install and enable the canonical foreground `nvidia-mps.service` → py3.11 venv → torch cu128 → `pip install -e vendor/catanatron` → `pip install -e .[dev,rl]` → verify and install the sealed `catanatron_rs` 0.1.8 cp311 wheel → env-doctor → rust-featurize/information-set/native-MCTS parity smoke. A commit ref is supported only with an explicit staged `$CATAN_RS_WHEEL`; `CATAN_REPO` also accepts a local git-bundle path as an offline fallback.
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

### Canonical two-commit wheel release

The native wheel and its checksum inventory are published as a two-commit
transaction. Do not edit a tag, build from an arbitrary checkout path, or fold
the checksum update into the source commit.

1. Create clean **commit A** containing every release source, builder, test, and
   documentation change. Run `tools/build_catanatron_rs_wheel.sh` twice from
   independent clean build state. The builder stages commit A at its sealed
   canonical path and emits both the wheel and
   `catanatron_rs-0.1.8-build-receipt.json`. Require byte-identical wheel
   SHA-256 values and matching sealed toolchain/environment provenance. The
   receipt schema is `catanatron-rs-wheel-build-receipt-v2`; in addition to
   both catanatron lockfiles it must bind
   `native/gumbel_mcts_rs/Cargo.lock`, `src/lib.rs`, and
   `src/python_binding.rs` by SHA-256. Before accepting the artifact, install
   it into a clean CPython 3.11 environment and assert that both
   `catanatron_rs.gumbel_search` and
   `catanatron_rs.build_entity_features_flat` are callable, then run
   `tests/test_native_gumbel_hot_loop.py` and
   `tests/test_generate_native_rollout.py` without native-path skips.
2. Create **commit B** by changing only
   `native/catanatron-rs/WHEEL_SHA256SUMS` to the verified wheel filename and
   SHA-256. Verify `git diff --name-only A..B` prints exactly that one path.
3. Rebuild clean commit B twice. The builder deliberately excludes the checksum
   inventory from native build inputs, so both wheels must reproduce commit A's
   exact SHA-256. Receipts must identify commit/tree B while retaining the same
   builder, lockfile, toolchain, environment, wheel filename, and wheel digest.
4. Create one new immutable release tag at commit B. Attach the exact verified
   commit-B wheel and build receipt; never move the tag. Confirm the release
   asset digest equals B's tracked inventory before provisioning any node.

Any byte mismatch aborts the release. Diagnose it on the build host; never
paper over it by updating the inventory to whichever build ran last.

The native-feature source changes make the currently tracked wheel digest stale
until the two-commit transaction above is completed. The exact items to
refresh are the `catanatron_rs-0.1.8` CPython 3.11 wheel asset, its build
receipt, and the single line in `native/catanatron-rs/WHEEL_SHA256SUMS`; the
inventory intentionally still names the last released 0.1.4 wheel in source
commit A and changes only in checksum commit B. The immutable release tag must
point at checksum-only commit B. `tools/install_v1_freeze.sh` already downloads
that exact filename from the selected tag and verifies it against the tracked
inventory before any privileged or environment mutation. Do not publish an
old wheel under a new tag, move an existing tag, or edit the inventory before
two clean builds agree.

## 4. Rust engine (CAT-133)
- `native/catanatron-rs` v0.1.8 is now the canonical wheel source and builds `catanatron_rs-0.1.8-cp311-…manylinux_2_34`; `native/gumbel_mcts_rs` is its linked native-search dependency. `native/catanatron-rs/WHEEL_SHA256SUMS` seals the exact release asset and the installer rejects any byte mismatch. The build receipt seals the source commit/tree, builder and lockfiles, exact toolchain/environment, and wheel digest. Fleet deployment must be uniform 0.1.8 with `sigma_reference_visits`, `belief_target_evidence`, `initial_road_d1_scope`, and `public_award_feature_parity` capabilities before corrected belief-level native MCTS, opening-road-only D1, or Rust entity featurization.
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
- **A1 launch:** seal and verify the pre-wave contract, render its exact 120
  category/GPU jobs, synchronize all 120 ledger claims to every production
  host, then use `tools/fleet/a1_production_executor.py`. It is dry-run by
  default; `--go` is the only execution boundary. The executor runs one
  category at a time per GPU under a detached resumable lane supervisor. Do
  not substitute the generic role launcher for A1.
- **A1 live canary:** before production claims or a 40-lane launch, use
  `tools/fleet/a1_live_canary.py` against the sealed lock/render and private
  host manifest. It selects exactly `c1` GPU0-3 plus `h100-8a` GPU0-7, derives
  36 validation-only jobs with the identical recipe, and writes only a private
  ledger/quarantined outputs under `/home/ubuntu/gen_out`. `run` is a dry run;
  inspect it before `run --go`, then require `status` and `audit` to pass.
  Never execute rendered argv manually and never merge canary rows or its
  ledger into production.
- **A1 runtime:** one generator per physical GPU, 16 workers/GPU,
  systemd-managed MPS, EvalServer off, strict FP32, public-observation masking,
  `n_full=128`, `n_fast=16`, `p_full=0.25`, `c_scale=0.03`, D1 rescaling off,
  and D6 averaging from legal width 20. `n_full_wide` and its threshold are
  unset and `wide_roots_always_full=false`: adaptive n256 is disabled.
- **Generic launcher:** `tools/fleet/fleet_launch.sh` remains useful for bounded
  diagnostics and historical role-shaped experiments, but it is not the A1
  production transaction. A zero launcher or executor exit does not attest
  every child; verify receipts, manifests, and postflight audit before harvest.
- **Stop**: `tools/fleet/fleet_stop.sh <alias|all> [--go]` — **default DRY-RUN**; terminates validated `launch_detached` process groups (so MPS-hidden clients and grandchildren cannot escape), retains explicit compute-PID fallback, PRESERVES MPS/observability, and fails unless owned groups, MPS clients, and non-infrastructure GPU PIDs are gone. Idle memory must be ≤50 MiB without MPS or ≤128 MiB for the measured 78 MiB/GPU preserved MPS-server baseline on driver 580.105.08.
- **Status**: `tools/fleet/fleet_status.sh [alias|all]` — read-only, parallel; per-box util/mem, inferred role, MPS on/off, matching job-process count.
- **Harvest → corpus**: `tools/wave1_harvest.sh {harvest-all|build-teacher|build-volume}` (parallel rsync + ControlMaster; reads `$FLEET_CONF`). Populate `DIRS` from accepted claim paths and reconcile harvested counts against remote manifests before a role-pure build. `build-pooled` is experiment-only after a predeclared mixture decision.
- **Ops rule (CAT-123):** one operator per box; always post-verify a single clean gen set after any change (`fleet_status.sh <box>` + `fleet_stop.sh <box>` dry-run).

### Historical n128 EvalServer throughput lock (2026-07-09)

At w48, wait `0/0.05/0.1/0.25 ms` measured
`72.26/70.54/70.04/71.07k` rows/hour/GPU, locking wait 0. Before the collector
fix, workers `48/64/80/96` measured `68.07/74.41/74.65/75.98k`; with the fixed
collector enabled, four w96 repetitions averaged **81.93k**.

The synthetic-checkpoint frontier measured **91.85k rows/hour/GPU** for
the canonical w128/batch96/collector recipe, about **37% above** the earlier
~67k w48 teacher baseline. Across 24 H100s this projects to approximately
**2.20M rows/hour**. Supporting paired results were w96 **83.42k** versus w128
**89.57k** (+7.4%), then batch64 **90.50k** versus batch96 **91.85k** (+1.5%)
at w128.

These are preserved throughput-only results from the historical 24-H100
EvalServer experiment with a synthetic same-shape masked 35M checkpoint. They
are not the 56-H100 A1 runtime recipe and must not be multiplied to claim A1
capacity. TF32 remains rejected after same-seed trajectory divergence;
`matmul_precision=highest` is mandatory.

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
