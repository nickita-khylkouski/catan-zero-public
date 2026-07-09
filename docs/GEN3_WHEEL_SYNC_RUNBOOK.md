# Gen-3 Wheel-Sync Runbook

Status: DRAFT for review. Authored by the rust-featurize agent, coordinating
with the speed-czar agent, at team-lead's request (task #81 "de-risk the
landing"). This document exists because the sync is the most complex
operation ahead of the speed program and currently exists only in two agents'
heads — it should be executable by a third party with no other context.

**Phase C prep (Python-side wiring only, flag default-off) is DONE as of this
writing** — see §8. The wheel install / flag-flip itself (§3-4) is still
pending the coordinated sync; nothing there has been executed yet. Hold for
team-lead's go-ahead before running the gate sequence's wheel-install step on
a real torch/GPU host.

## 1. What this sync is

A `catanatron_rs` wheel rebuild (rules-identical, pure performance) plus a
coordinated set of Python-side flag flips, landing together as ONE atomic
unit before any fleet (self-play generation / gate / training) host picks it
up. Per the #82 precedent (deploying 0.1.2 fleet-wide was explicitly called
out as "NOT a silent swap, it changes game rules" even though that one WAS a
rules change) — this one is rules-IDENTICAL, but still a deliberate wheel
swap with a version bump, not a silent drop-in. Treat it with the same
ceremony: announce it, gate it, version it, and keep a rollback path.

## 2. What ships (the atomic bundle)

Everything below must land TOGETHER. A wheel without the flags flipped is
inert (no measured benefit, but also no risk — every new code path is
default-OFF). Flags without the new wheel will error loudly (see §2.2) or
(for `exact_budget_sh`) silently do nothing new since that code doesn't touch
the wheel at all. The dangerous half-state is a wheel WITHOUT the matching
crate version bump getting confused for an older/newer one at a glance — see
§3's version-bump step.

### 2.1 The wheel (`catanatron_rs`)

Built from `scratch/catanatron-rs` (this local mirror's scratch clone of the
canonical crate — see §3 for provenance). Contains, all already implemented
and parity-gated as of this writing:

| Change | Owner | Status |
|---|---|---|
| `playable_action_indices` ActionSpace-rebuild fix (~50x at wide roots, bit-identical vs `decision_context_json` across 18k states) | speed-czar | Landed |
| `build_entity_features_flat`/`build_entity_features_batch` (native entity-token featurizer, no `json_snapshot`) | rust-featurize | Landed, bit-exact parity (17 tests) |
| `build_action_context_flat`/`build_action_context_batch` (native context-feature featurizer) | rust-featurize | Landed, bit-exact parity (6 tests), **now wired into all 4 evaluator call sites, see §2.3** |
| `EntityTopology` pyclass (shared board-topology object, built once per board/search) | rust-featurize | Landed |
| Byte-buffer marshalling (`Vec<u8>`/`bytes` instead of per-element Python-list boxing for every f64/i64 array) | rust-featurize | Landed, transparent to callers, no new flag |

Crate version: **0.1.2 -> 0.1.3** (bump both `Cargo.toml` and
`python/Cargo.toml`; this crate already has its own bump-per-change
precedent — `0.1.0->0.1.1->0.1.2` are each their own commit). Do this bump AS
PART of the commit that lands this bundle, not before — an uncommitted 0.1.3
floating around with no matching commit is exactly the confusion this step
exists to prevent.

### 2.2 The Python-side flags

| Flag | Location | Default | New value for this sync | Effect if flipped on an OLDER wheel |
|---|---|---|---|---|
| `EntityGraphRustEvaluatorConfig.rust_featurize` | `src/catan_zero/search/neural_rust_mcts.py` | `False` | `True` | Raises `AttributeError` on `catanatron_rs.build_entity_features_flat` at the first leaf eval — loud, not silent (see the field's own doc comment) |
| `--rust-featurize` CLI flag | `tools/generate_gumbel_selfplay_data.py` | off | on | Same as above, surfaced through the generation tool. **Added as part of this runbook** (was previously wired into the config dataclass but never exposed on this CLI — see §6) |
| `GumbelChanceMCTSConfig.exact_budget_sh`/`exact_budget_sh_min_n` | `src/catan_zero/search/gumbel_chance_mcts.py` | `False`/`0` | **No change for this sync — CLOSED, not adopted** (task #61's own gate rejected both fast-16 and full-64, see §2.4/§7 item 2). Permanently default-off/0; dormant API coherence only. | No wheel dependency at all — pure Python, works on any wheel |
| `--exact-budget-sh`/`--exact-budget-sh-min-n` CLI flags | `tools/generate_gumbel_selfplay_data.py` | off/`0` | **Never set at this sync** (see above) | n/a — already CLI-wired, pre-existing |
| Cache-gating (`EntityGraphRustEvaluatorConfig.cache_size <= 0` skips per-leaf hashing entirely, not just the store) | `src/catan_zero/search/neural_rust_mcts.py` | `cache_size=100_000` (cache ON) | **No change for this sync** — not a new flag, an internal correctness fix to an existing non-default configuration (`cache_size=0`) that no CLI tool currently exposes. Document only; nothing to flip. Production keeps the cache ON (owner's accepted verdict); the separate question of whether the cache KEY should become a Rust state fingerprint instead of the last per-leaf `json_snapshot` caller (~680us/eval) once this sync lands is a future bench, not part of this sync. | n/a |

### 2.3 Wiring status (RESOLVED — was a known gap, now closed)

`rust_featurize=True` now gates BOTH the entity-token path and the
context-feature path, consistently, at ALL FOUR `evaluate*` call sites on
`EntityGraphRustEvaluator`/`BatchedEntityGraphRustEvaluator`: `evaluate`,
`evaluate_symmetry_averaged`, `evaluate_many`, and
`BatchedEntityGraphRustEvaluator.evaluate`. Each site uses the identical
`adapter=resolved[1]` extraction pattern, and the context path
(`_context_batch_via_rust`) reuses the SAME lazily-bootstrapped
`self._rust_topology` the entity path builds — whichever of entity/context
runs first in a given call bootstraps it for both (verified by
`tests/test_rust_action_context_evaluator_wiring.py::test_context_topology_reuses_entity_bootstrap`
and its reverse-order counterpart).

Previously, `evaluate_symmetry_averaged` (the f74b wide-root symmetry
denoiser) had ZERO `rust_featurize` gating on either path — it unconditionally
used the old `rust_game_to_entity_batch`/`rust_action_context_batch` Python
paths regardless of the flag. This was latent (harmless) because
`GumbelChanceMCTSConfig.symmetry_averaged_eval` defaults `False`, but is now
fixed rather than deferred: both entity and context are gated there too.

The CONTEXT-feature native functions (`build_action_context_flat`/`batch`)
are now wired into every evaluator call site listed above, under the same
`rust_featurize` flag as entity — no separate `rust_context_features`-style
flag was needed. Wiring-level parity (evaluator OUTPUT bit-identical between
`rust_featurize` True/False, both masking regimes) is proven by
`tests/test_rust_action_context_evaluator_wiring.py` (4 tests), the same bar
as the pre-existing entity wiring test. **Not yet re-verified**: the combined
entity+context gates (b)/(c)/(d) in §4 below were run with entity-only
wiring (context wiring landed after that run) — they need a re-run with the
current fully-wired code before their numbers can be cited as the sync's real
throughput delta. See §7 item 5.

### 2.4 `exact_budget_sh` gating is SEPARATE from parity — and now CLOSED, not adopted

`exact_budget_sh` was a deliberate SEARCH-SEMANTICS change (fewer
simulations at some root widths vs. the legacy `>=1-sim-per-round`
schedule) — it was never expected to reproduce the legacy schedule's action
sequence, and its adoption was gated on a pentanomial H2H non-inferiority
test (a strength comparison, task #61, owned by speed-czar). **That gate is
now CLOSED: fast-16 REJECTED (H0, llr −3.904), and full-64's clean rerun
(seed 96M, @600-cap, no truncation confound) came back 49.25% — a wash, not
an accept** (an earlier 53.6% reading was itself a truncation artifact of a
300-cap confound in a prior run). Nothing from task #61 is adopted; see §7
item 2 for the full verdict and the hard rule that follows from it (this
flag/CLI combination is never set at the sync). The identical-seed smoke
test in §4(c) still deliberately holds `exact_budget_sh` FIXED (same value
on both sides of its A/B, i.e. `False` on both, its only production value)
specifically so it never conflates the two different kinds of gate.

## 3. Build steps

```bash
# 1. Confirm the scratch clone is the coherent, up-to-date working copy.
cd /Users/nickita/catan-zero-gpu/scratch/catanatron-rs
git status   # expect: modified Cargo.lock, src/lib.rs (uncommitted, both agents' work)
git log --oneline -3   # expect HEAD = 8b78fa4 (the B200 canonical commit this was cloned from)

# 2. Full build + test pass BEFORE touching version/commit anything.
cargo check --lib --features python
cargo check --lib                       # default (non-python) build path — the plain `catanatron` binary target
cargo clippy --lib --features python    # expect exactly 2 pre-existing warnings (filter_map + too_many_arguments in alphabeta_value), nothing new
cargo test --lib --features python      # expect 98+ passed, 0 failed (98 as of this writing; the czar's/rust-featurize's own additions may add more)

# 3. Bump the version (BOTH Cargo.toml files) — do this as part of the commit, not before.
#    Cargo.toml:        version = "0.1.2" -> "0.1.3"
#    python/Cargo.toml:  version = "0.1.2" -> "0.1.3"

# 4. Commit on the canonical B200 repo (NOT this local scratch clone — the
#    scratch clone is a read-only-provenance working copy; the real commit
#    target is /home/ubuntu/catanatron-rs on the B200 host, per the
#    established provenance chain: 6652562 -> ... -> 8b78fa4). Push the
#    version-bump + all landed changes as ONE commit (or a small stack),
#    mirroring the existing "bump version X -> Y" commit-message convention
#    already used for 0.1.0->0.1.1->0.1.2.

# 5. Build the wheel for the actual GPU host's Python version (check the
#    target host's python3 --version first; existing wheels in this project
#    have shipped for cp310/cp311/cp312 — match whichever the sync host runs).
cd /home/ubuntu/catanatron-rs/python   # on the B200 (or whichever host owns the canonical commit)
maturin build --release --features python-extension -i python3.12   # adjust -i to the target interpreter
# Wheel artifact lands at: target/wheels/catanatron_rs-0.1.3-cp312-cp312-<platform-tag>.whl

# 6. Install on EACH target host (self-play workers, gate hosts, training host):
pip install --force-reinstall target/wheels/catanatron_rs-0.1.3-*.whl
python3 -c "import catanatron_rs; print(catanatron_rs.__file__); assert hasattr(catanatron_rs, 'build_entity_features_flat'); assert hasattr(catanatron_rs, 'build_action_context_flat')"
```

Local-dev note (this exact machine, macOS/arm64): `maturin develop --release`
against `scratch/catanatron-rs/python` was used throughout development for
fast local iteration (installs directly into `.venv`, no wheel artifact) —
this is NOT the sync path, just how the parity/bench work in this runbook's
prerequisite tasks was validated. The real sync always goes through `maturin
build` -> a distributable `.whl` -> `pip install` on each target host, so
every host runs the IDENTICAL binary artifact, not N independent local builds.

## 4. Gate sequence (run on ONE torch/GPU host before ANY fleet rollout)

Run in order; each gate must fully pass before the next. All scripts below
are pre-staged in this repo now, executable at sync time.

**(a) Full test suite, including the fail-closed regime suite:**

```bash
cd /Users/nickita/catan-zero-gpu
PYTHONPATH=src python -m pytest tests/ -x -q
# Expected as of this writing (verified locally, full entity+context wiring
# including the evaluate_symmetry_averaged path, PYTHONPATH=.):
# 272 passed, 2 skipped, 0 failed
# (the 2 skips are test_rust_featurize_checkpoint_output_equality.py, gate (b) below — they need a real checkpoint)

# Specifically confirm these all pass (they exercise different slices of this bundle):
PYTHONPATH=src python -m pytest \
  tests/test_rust_featurize_parity.py \
  tests/test_rust_action_context_parity.py \
  tests/test_rust_featurize_evaluator_wiring.py \
  tests/test_rust_action_context_evaluator_wiring.py \
  tests/test_manifest_rust_featurize_provenance.py \
  tests/test_regime_fail_closed.py \
  tests/test_exact_budget_sh.py \
  -v
# Expected for this narrower subset: 176 passed, 0 skipped
```

**(b) Checkpoint-loaded output-equality** (rust-featurize's owed item —
pre-staged, needs a real gen-3 checkpoint path):

```bash
GEN3_CHECKPOINT=/path/to/gen3_checkpoint.pt \
  PYTHONPATH=src python -m pytest tests/test_rust_featurize_checkpoint_output_equality.py -v
# Proves: with the REAL checkpoint loaded and a REAL forward pass, evaluator.evaluate(...)'s
# (priors, value) are numerically identical between rust_featurize=False and True,
# on real game states, in both masking regimes the checkpoint supports.
```

**(c) 32-game identical-seed smoke** (bit-identical search on fixed seeds,
`rust_featurize` isolated — see §2.4 for why `exact_budget_sh` is held fixed,
not gated, by this script):

```bash
python3 tools/gen3_identical_seed_smoke.py \
  --checkpoint /path/to/gen3_checkpoint.pt \
  --games 32 --n-full 64 --n-fast 16 \
  --exact-budget-sh   # hold at whatever value production will actually use — same on both sides
# Expect: "GATE PASSED: 32/32 games bit-identical."
# Verified locally (this dev machine, CPU, random-init policy, tiny search budget): 3/3 passed.
# MUST be re-run on the sync host with the REAL checkpoint and production n_full=64
# before this gate counts as passed for real.
```

**(d) Integrated throughput bench** (the real games/hr delta — run twice,
diff the two JSON reports):

```bash
python3 tools/gen3_integrated_throughput_bench.py \
  --checkpoint /path/to/gen3_checkpoint.pt --device cuda \
  --games 20 --n-full 64 --label before --out /tmp/bench_before.json
  # (no --rust-featurize / --exact-budget-sh: current production defaults)

python3 tools/gen3_integrated_throughput_bench.py \
  --checkpoint /path/to/gen3_checkpoint.pt --device cuda \
  --games 20 --n-full 64 --rust-featurize --exact-budget-sh \
  --label after --out /tmp/bench_after.json

python3 -c "
import json
before = json.load(open('/tmp/bench_before.json'))
after = json.load(open('/tmp/bench_after.json'))
print(f\"games/hr: {before['games_per_hour']:.1f} -> {after['games_per_hour']:.1f} ({after['games_per_hour']/before['games_per_hour']:.2f}x)\")
"
```

**Status as of this writing (entity+context now fully wired, §2.3):**

- (a) passed with the combined entity+context wiring: 267 passed / 2 skipped,
  0 failed, on this dev machine (`PYTHONPATH=.`).
- (b) **RE-RUN and PASSED with the combined entity+context wiring**, on
  B200's isolated venv/source copy, CPU device, the real
  `runs/bc/gen2A_20260706/checkpoint.pt` checkpoint: 1 passed / 1 skipped
  (the skip is the masking regime the checkpoint wasn't trained for — the
  fail-closed guard correctly declines to test that combination). This is
  the data-integrity-critical gate and it holds for the full bundle, not
  just entity.
- (c) **RE-RUN and PASSED with the combined entity+context wiring**, same
  isolated copy, same checkpoint, reduced scale (`--n-full 6
  --max-decisions 12 --games 4`), `--public-observation` (matching the
  checkpoint's trained regime): `GATE PASSED: 4/4 games bit-identical`
  (`elapsed_sec≈1992`, thread-constrained to `OMP/MKL/OPENBLAS/NUMEXPR
  _NUM_THREADS=2` to avoid CPU contention with the host's live generation
  workers — see the note below on why that constraint mattered). As before,
  reduced scale is fine per team-lead: bit-identity is budget-independent.
- (d) integrated throughput bench is **DEFERRED per team-lead's explicit
  call** — a synchronous single-stream number under a saturated/contended
  fleet is not trustworthy and not worth the cycles. The real throughput
  number will come from a one-host B200 canary at the gen-3 sync (measure
  GPU1's current games/hr on the old path, cleanly restart it under 0.1.3 +
  `--rust-featurize`, measure the real batched+contended delta). Gate (d) in
  this runbook should be treated as superseded by that canary, not run as
  written above, unless a future need for the isolated synchronous number
  arises.

**Operational note from the (c) re-run:** the first attempt was launched
without constraining BLAS/torch thread counts and immediately grabbed ~21 of
the host's 52 cores (unconstrained OpenMP/MKL thread pool sizing to `nproc`)
against a host already at load-average 30-38 from 16 live generation-worker
processes — this WAS real, non-negligible contention and was killed within
seconds of being noticed. The re-run with
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2` bounded it to ~2 cores and ran cleanly to completion
(~33 min wall-clock at reduced scale) without measurably disturbing the host
load average. Any future CPU-device gate run on a shared host MUST set these
thread-count env vars — the gate scripts themselves do not set them.

Only after (a)+(b)+(c) all pass on the ONE torch/GPU host, WITH the current
fully-wired entity+context code, AND the gen-3-sync canary has produced a
real fleet throughput number, does the wheel+flags combination go to the
rest of the fleet.

## 4.5 Handoff to the MPS relaunch step (speed-czar composition)

Per team-lead's bundling decision (wheel + MPS ship together, ONE restart per
host at the gen-3 sync), this runbook composes with speed-czar's
`tools/mps_rollout.sh` rather than duplicating its relaunch logic. The seam:

1. **This runbook owns:** stopping generation host-wide, installing the 0.1.3
   wheel, running the gate battery above (wiring parity + fail-closed +
   checkpoint output-equality + 32-game identical-seed smoke + the
   wheel-version parity check, §7 item 6 below), and deciding which flags
   pass their gate (as of this writing, that's just `--rust-featurize` —
   `--exact-budget-sh` is CLOSED/not-adopted, §2.4/§7 item 2, so it is never
   part of that decision).
2. **`mps_rollout.sh` owns the relaunch — v2, PER-GPU-DAEMON topology:**
   once flags are decided, hand off with one line:
   ```bash
   GAMES=<ledgered_block_size> CKPT=<current_champion_checkpoint> \
   GEN_EXTRA_ARGS="--rust-featurize" \
     bash tools/mps_rollout.sh host "<gpu:seed> <gpu:seed> ..."
   ```
   **`GAMES` and `CKPT` are now REQUIRED env vars, no defaults** (v2 change,
   2026-07-07 Phase-A execution): `GAMES` must be pinned to the ledgered
   block size (750 standard) — the script's old 1500 default was a
   seed-block-overrun footgun (the #77 class). `CKPT` must be the current
   champion — a hardcoded default would silently generate against a
   dethroned champion after a promotion. The script errors immediately
   (`${VAR:?message}`) if either is unset, rather than silently defaulting.
   Ledgered seeds are supplied as `gpu:seed` args by whoever runs the sync
   (per `catan-seed-ledger`'s discipline — this runbook does not allocate
   seeds). **`--exact-budget-sh`/`--exact-budget-sh-min-n` are NEVER part of
   `GEN_EXTRA_ARGS` at this sync** — task #61 is CLOSED end-to-end, nothing
   adopted (fast-16 rejected H0; full-64's clean rerun came back a 49.25%
   wash, not an accept). See §7 item 2 for the full verdict. Both fields
   (`exact_budget_sh`, `exact_budget_sh_min_n`) exist on the
   `gen3-wheel-sync` branch (`c2ce783`) purely as dormant API coherence —
   the CLI tool passes them unconditionally so the config dataclass must
   accept them — with their defaults (`False`/`0`) as the permanent
   production values.

   **Per-GPU MPS daemons, not one host-wide daemon** (v2 replaces the
   original single-daemon design): empirically, on both A100 hosts, one
   shared MPS server hard-caps around ~80 client contexts (five 16-worker
   GPUs) — every 16w launch past that hangs forever in CUDA init. A scoped
   (per-GPU) daemon also CANNOT coexist with a host-wide one (the second
   server gets "CUDA requested but not available", since the host-wide
   server holds an idle context on every GPU). v2 starts one daemon PER GPU
   (`CUDA_VISIBLE_DEVICES=<gpu>` at daemon start, its own pipe/log dir
   `/tmp/mps_pipe_gpu<N>`/`/tmp/mps_log_gpu<N>`), keeping every server at
   <=16 clients — the topology the original packing grid used and never hit
   the ceiling on. **Migration guard:** the script REFUSES to start any
   per-GPU daemon while the legacy host-wide daemon (`/tmp/mps_pipe_host`)
   is still live, and prints the exact migration steps (stop that daemon's
   clients, quit it, then re-invoke). The migration itself only happens at
   the wheel flag-flip restart, since it requires those GPUs to already be
   stopping for the flip anyway — it's a side benefit that this also
   converts any remaining no-MPS 8-worker "straggler" GPUs to 16w.
3. **Rollback is split by concern, not by host:**
   - Wheel rollback (§5 below) is this runbook's job: `pip install
     --force-reinstall catanatron_rs==0.1.2` + relaunch.
   - MPS/packing rollback is `mps_rollout.sh rollback "<gpu:seed> ..."`
     (reverts the listed GPUs to 8 workers, no MPS, per-process env unset,
     AND quits that GPU's own daemon — v2's rollback is now per-GPU-clean,
     not "daemon left running for someone to quit manually").
   - **If both need reverting, wheel first, then the MPS rollback** — a
     wheel revert already implies a relaunch, so doing the MPS rollback
     first would just be immediately superseded by the wheel-rollback's own
     relaunch.
4. **Canary ordering is independent:** speed-czar's MPS canary (one GPU/host
   at 16 workers under its own MPS daemon) is a pure GPU-scheduling change —
   data is bit-identical, no parity gate needed — so it can run BEFORE or
   independently of this wheel sync; team-lead may trigger it earlier. This
   runbook's gate battery does not assume any particular MPS migration state
   as the starting point — none of the CPU-device gates above ((a)/(b)/(c))
   touch the GPU/MPS path at all.

**Fleet state as of the v2 rewrite (2026-07-07, Phase-A execution):** 16/16
A100 GPUs generating vs the gen-3 champion — 10 GPUs at 16 workers under the
(legacy, host-wide, pending migration) MPS daemons + 6 GPUs still at 8
workers, no-MPS ("stragglers", since the legacy daemon was already at its
client-count ceiling) — roughly 1.1M rows/hr combined. The flywheel is live
on B200. The 6 straggler GPUs convert to 16w/per-GPU-MPS as part of the
migration at the wheel flag-flip restart (an estimated +~340k rows/hr on top
of the current 1.1M).

**MPS canary evidence (speed-czar, A100A):** all 4 pass criteria met —
2.67x rows/hr vs two 8-worker sibling GPUs in the same time window, 13.9GB
GPU memory at 16 workers (comfortably under the OOM ceiling), 16+ worker
processes correctly attached to one `nvidia-cuda-mps-control` server, and
clean worker logs (zero errors). The canary exercise also caught and fixed a
real bug in `mps_rollout.sh`'s stop-pattern (it initially matched no
processes; fixed with a digit-boundary match + logging the matched cmdline
before killing, per the script's own "explicit-PID, never pattern kills"
discipline). MPS is fully validated for the sync as of this writing.

The canary ran from a100a's live venv (gpu6, seed 77.3M) during the window
that host was affected by §7 item 6's incident — its data validity under
that mixed-wheel window is covered by the same 0.1.2≡0.1.3 flag-off
bit-identity proof documented there, so no separate check was needed for
the canary's corpus.

## 5. Rollback

Per-host, if any gate fails (or a fleet host misbehaves after adoption):

1. **Flags first (fastest, zero downtime):** flip `--rust-featurize`/`--rust_featurize`
   and `--exact-budget-sh`/`exact_budget_sh` back to their defaults (`False`)
   in that host's launch command/config. Since every new code path is
   default-OFF and additive, this alone reverts behavior to exactly
   pre-sync, even with the NEW wheel still installed (the new wheel's old
   functions — `playable_action_indices`, `json_snapshot`, etc. — are
   unchanged in behavior by this sync, only faster).
2. **Wheel, if needed:** `pip install --force-reinstall catanatron_rs==0.1.2`
   (the OLD pinned version — keep the 0.1.2 wheel artifact archived
   somewhere reachable, e.g. alongside the 0.1.3 one, specifically so this
   command has something to install). Do not `pip uninstall` blind — always
   reinstall a KNOWN-GOOD pinned version.
3. Re-run gate (a) (the fast one) on that host after rollback to confirm the
   revert actually took effect before resuming any generation/gate/training
   work on it.

## 6. Changes made while authoring this runbook

Small, mechanical, additive changes made so the steps above are actually
executable (not new speculative work — see task #81 discussion):

- Added `--rust-featurize` CLI flag to `tools/generate_gumbel_selfplay_data.py`
  (mirrors the existing `--exact-budget-sh`/`--public-observation` pattern
  exactly; threaded through both `EntityGraphRustEvaluatorConfig(...)`
  construction sites, including the opponent-pool one). Previously the
  config dataclass field existed but no CLI path could set it.
- `tools/gen3_identical_seed_smoke.py` — gate (c), verified working
  end-to-end on this dev machine (CPU, random-init policy, `--games 3
  --n-full 4`): 3/3 bit-identical.
- `tools/gen3_integrated_throughput_bench.py` — gate (d), argument-parsing
  and import structure verified (`--help` runs clean); needs a real
  checkpoint + GPU host to produce real numbers, not runnable to completion
  on this dev machine.
- `tests/test_rust_featurize_checkpoint_output_equality.py` — gate (b),
  skips cleanly without `GEN3_CHECKPOINT` set (verified: 2 skipped, 0
  errors), ready to actually gate once pointed at a real checkpoint.
- Context featurizer wired into all 4 `evaluate*` call sites in
  `src/catan_zero/search/neural_rust_mcts.py` under the same `rust_featurize`
  flag as entity (`evaluate`, `evaluate_symmetry_averaged`, `evaluate_many`,
  `BatchedEntityGraphRustEvaluator.evaluate`), including retroactively
  gating `evaluate_symmetry_averaged`'s entity path too (previously
  ungated — see §2.3's history). Reuses the same lazily-bootstrapped
  `self._rust_topology` the entity path builds/reuses.
- `tests/test_rust_action_context_evaluator_wiring.py` — new 6-test suite
  proving the wired context path's evaluator-level output is bit-identical
  between `rust_featurize` True/False, both masking regimes, plus
  topology-bootstrap-ordering coverage (entity-first and context-first),
  PLUS (speed-czar's ask) an end-to-end `evaluate_symmetry_averaged()`
  comparison using a real tiny policy (`hex_symmetry.average_forward`
  consumes entity/context by shape/value, so this proves the full symmetry
  path -- not just its inputs -- is unaffected by the flag, both masking
  regimes). Same bar as the pre-existing entity wiring test.
- `tools/generate_gumbel_selfplay_data.py` -- added `rust_featurize` as its
  own top-level field in the shard-batch summary/manifest, mirroring
  `exact_budget_sh`'s existing pattern (previously only present buried
  inside the catch-all `cli_args` dict, per speed-czar's provenance-
  auditability ask). `tests/test_manifest_rust_featurize_provenance.py`
  covers this.
- `tools/wheel_version_parity_smoke.py` -- new tool, incident follow-up
  (§7 item 6): holds `rust_featurize=False` fixed and dumps action
  sequences to JSON, designed to be run under TWO DIFFERENT installed wheel
  versions (not a `rust_featurize` A/B like gate (c)) so a future wheel bump
  can prove its always-on changes are game-identical across versions, not
  just index-identical in isolation. Reusable for any future
  `catanatron_rs` version bump, not a one-off.
- (speed-czar, landed on this branch per team-lead's "pin as one unit"
  directive) `src/catan_zero/search/gumbel_chance_mcts.py` --
  `exact_budget_sh`/`exact_budget_sh_min_n` config fields, permanently
  default-off (#61 closed not-adopted at fast-16, see §2.4/§4.5), +
  `tests/test_exact_budget_sh.py` (139 tests). This completes the pairing
  the branch's `generate_gumbel_selfplay_data.py` CLI flags already
  referenced -- without it, those flags would TypeError at runtime.
- (speed-czar) `tools/mps_rollout.sh` v2 -- rewritten to per-GPU-daemon
  topology; see §4.5 for the full design rationale and migration guard.

## 7. Open items before this can be marked READY (not this agent's call to close)

1. ~~§2.3's `evaluate_symmetry_averaged` gap~~ — **RESOLVED**: both entity and
   context are now gated there, same as the other 3 call sites.
2. ~~§2.4's `exact_budget_sh` H2H gate status~~ — **RESOLVED, CLOSED
   END-TO-END, NOTHING ADOPTED** (task #61 owner speed-czar + team-lead's
   ruling): fast-16 REJECTED (H0, llr −3.904, decisively worse). Full-64's
   clean rerun (seed 96M, @600-cap, no truncation confound) came back
   49.25% — a wash, not an accept (the earlier 53.6% "continue" reading was
   itself a truncation artifact of a 300-cap confound in an earlier run).
   **Nobody sets `--exact-budget-sh` or `--exact-budget-sh-min-n` in
   `GEN_EXTRA_ARGS` at the sync — ever, absent a future re-adjudication.**
   `exact_budget_sh`/`exact_budget_sh_min_n` exist in `c2ce783` purely as
   dormant API coherence (the CLI tool passes these kwargs unconditionally,
   so the config dataclass must accept them) — both fields' defaults
   (`False`/`0`) ARE the permanent production values, and flag-off is
   tested bit-exact-legacy. This mechanism is dormant by design, not
   pending.
3. Pick and pin the actual gen-3 checkpoint path for gate (b)/(c)/(d) — not
   yet known to this agent. (Note: prior B200 runs used
   `runs/bc/gen2A_20260706/checkpoint.pt`, which may or may not be the
   intended "gen-3 checkpoint" for the final gate; confirm before treating
   that as authoritative.)
4. Confirm which hosts are "the fleet" for step 6 of §3 (self-play workers /
   gate hosts / training host list) — not enumerated here since this agent
   doesn't own fleet inventory.
5. ~~Gates (b)/(c)/(d) must be RE-RUN with the current fully-wired
   entity+context code~~ — **(b) and (c) RESOLVED**: both re-run on B200's
   isolated copy with the combined entity+context wiring and the real
   gen2A checkpoint, both PASSED (see §4). Correctness/data-integrity is now
   proven for the full bundle. **(d) remains open by design**: deferred per
   team-lead's explicit call in favor of a one-host B200 live canary at the
   gen-3 sync (real batched+contended throughput number, not a synchronous
   bench). Do that canary once gen-3 training/gates clear and a sync window
   opens — not blocked on anything else right now.
6. **RESOLVED — incident + wheel-version parity check**: while pre-staging
   the 0.1.3 wheel on a100a/a100b (isolated venvs, per team-lead's
   go-ahead), an `rsync -a`-cloned venv's `bin/pip` executed against the
   ORIGINAL live venv's Python (hardcoded absolute-path shebang, not the
   clone's own location) — `pip install --force-reinstall --no-deps <wheel>`
   run against the "isolated" clone's `bin/pip` silently installed 0.1.3
   into the LIVE venv's site-packages on BOTH a100a and a100b. Caught within
   the same session (a verification check against the intended clone showed
   stale symbols), reported immediately, no rollback attempted pending
   team-lead's call. **Team-lead's assessment (accepted): no rollback** —
   flag-off (`rust_featurize=False`, the state of every live generate
   command on both hosts, confirmed 0 uses of `--rust-featurize`) is
   behaviorally 0.1.2, the only always-on delta is the ActionSpace rebuild
   fix, and a100a/b are now correctly pre-staged rather than newly at risk.
   **Definitive proof run** (this agent, on B200): the ActionSpace fix is
   not just index-identical in isolation (already known from the crate's
   own 63,884-index audit) but GAME-identical end-to-end — `tools/
   wheel_version_parity_smoke.py` (new tool, holds `rust_featurize=False`
   fixed and is meant to be invoked under two DIFFERENT installed wheel
   versions) run once via B200's LIVE venv (confirmed still 0.1.2,
   untouched — verified directly: `catanatron_rs-0.1.2.dist-info` present,
   version 0.1.2) and once via B200's isolated 0.1.3 venv, same real gen2A
   checkpoint, same 4 seeds, `--public-observation` (the checkpoint's
   trained regime), thread-constrained (`OMP/MKL/OPENBLAS/NUMEXPR_
   NUM_THREADS=2`) to avoid repeating the earlier CPU-contention mistake.
   Result: **bit-identical action sequences on all 4 games.** Incident is
   fully closed; the a100a/b post-incident shards are safe to harvest.
   **Hard requirement still in force**: the flag stays OFF on a100a/b until
   the coordinated sync — no generate/gate command there may pass
   `--rust-featurize`.
   **Lesson for future wheel staging** (document, don't repeat): never
   `rsync`-clone a venv to "isolate" it. A venv's `bin/python` is typically
   a relocatable symlink to a stable system interpreter (correctly resolves
   via `pyvenv.cfg` detection based on the invoked path, regardless of
   where it's copied to) but `bin/pip` and other console-script entry
   points are small text files with a HARDCODED absolute-path shebang
   baked in at creation time — copying them verbatim means they keep
   executing under the ORIGINAL venv's interpreter no matter where the
   clone lives. (This also explains, after the fact, why this agent's
   EARLIER B200 isolated-venv gates worked correctly despite an identical
   clone: those installs happened to invoke `bin/python -m pip install`
   rather than `bin/pip install` directly, which sidesteps the shebang
   entirely.) The correct approach for any future wheel pre-staging: build
   with `python -m venv <fresh-path>` and reinstall dependencies from a
   lock/requirements file — never rsync an existing venv, and if a
   console-script binary must be invoked from a copied venv for any
   reason, always call it as `<venv>/bin/python -m <module>` instead of
   `<venv>/bin/<script>` directly.

## 8. Phase C prep: Python-side wiring deployed to all three live repos (flag OFF)

Per team-lead's Phase-C directive (gen-3 promoted, sync firing in phases;
MPS is Phase A): the Python wiring (rust_featurize gating, CLI flag, parity
tests) has been deployed to B200/a100a/a100b's LIVE repos AHEAD of the wheel
install, so the actual Phase-C flip later is just adding `--rust-featurize`
to launch commands at the next natural restart. Constraints honored
throughout: flag defaults `False` everywhere, no running process was
restarted, B200's live venv stays on 0.1.2 (confirmed before and after),
`python -m pip` discipline n/a here (no venv/wheel installs in this phase,
only plain source-file deploys).

### 8.1 Canonical commit (provenance, not the live deploy)

A new worktree `/home/ubuntu/catan-zero-gen3sync` on B200, branch
`gen3-wheel-sync` (based off master @ ea6ce93, the live repo's HEAD at the
time), holds the canonical commit history for this change — created via
`git worktree add`, exactly like B200's existing per-feature-branch worktree
pattern (`catan-zero-f60` .. `catan-zero-v3comb`), specifically so creating
it never touches the live checkout's working tree or its own uncommitted
changes (`tools/build_memmap_corpus.py`, `tools/train_bc.py`, the
value-lambda work). Two commits:
- `8fac032` — the wiring itself (neural_rust_mcts.py gating at all 4
  `evaluate*` sites, the two new Rust-wrapper modules, `--rust-featurize`
  CLI flag + manifest field, all 5 new test files).
- `33fc438` — two bugs found DURING live-repo deploy verification (see
  §8.3), fixed and committed as a follow-up.

### 8.2 Live-repo deploy (the actual Phase-C prep)

Files deployed via atomic `mv` from a staging directory into each live
repo's working tree (uncommitted there, matching the existing precedent from
task #61's deploy) — NOT via `git checkout`/branch switch on any live
checkout:
- `src/catan_zero/search/neural_rust_mcts.py`
- `src/catan_zero/rl/entity_token_features_rust.py` (new)
- `src/catan_zero/rl/action_context_features_rust.py` (new)
- `tools/generate_gumbel_selfplay_data.py` (B200 + a100a only, see §8.4)
- `tests/test_rust_featurize_parity.py`, `test_rust_action_context_parity.py`,
  `test_rust_featurize_evaluator_wiring.py`,
  `test_rust_action_context_evaluator_wiring.py`,
  `test_manifest_rust_featurize_provenance.py` (all new)
- `tests/test_evaluator_shared_payload.py`, `test_regime_fail_closed.py`
  (already present+identical on B200; added fresh on a100a/a100b where
  missing/behind, bringing them to the same coverage)

Before deploying, every target file was diffed against all three live repos
and cross-diffed against each other to rule out clobbering host-specific
content. Finding: `neural_rust_mcts.py`'s only non-wiring divergence was
B200's "B2 dedup" fix (shared `_resolve_entity_adapter` resolve) which
a100a/a100b lacked — deploying the tested superset file brings them to
parity with B200 on that pre-existing, already-proven fix as a side effect,
not a new risk. `generate_gumbel_selfplay_data.py`'s flag set was confirmed
a strict superset across all three hosts (no host had a flag the deployed
file lacked) — safe to overwrite wholesale EXCEPT for a100b (see §8.4).

### 8.3 Two bugs found and fixed during deploy verification

Both caught by actually running the suites on B200's live repo (still 0.1.2)
before touching a100a/a100b — exactly why "run the suites on each live repo"
matters, not just "the files copied cleanly":

1. `test_rust_featurize_parity.py`/`test_rust_action_context_parity.py` used
   only `pytest.importorskip("catanatron_rs")`, which checks the module is
   *importable*, not that it has the specific task-#81 functions. On B200's
   live 0.1.2 wheel (importable, but lacking `build_entity_features_flat`/
   `build_action_context_flat`), this caused hard FAILURES (not skips) —
   `AttributeError: module 'catanatron_rs' has no attribute 'EntityTopology'`.
   Fixed with the same `hasattr`-based `pytestmark` skip guard the newer
   wiring test files already used.
2. `test_manifest_rust_featurize_provenance.py`'s `_minimal_args()` fixture
   Namespace was missing `exact_budget_sh_min_n`, which `_merge_worker_summaries`
   now reads unconditionally (speed-czar's later addition to the shared
   file) — `AttributeError: 'Namespace' object has no attribute
   'exact_budget_sh_min_n'`. Fixed by adding the field with the CLI's own
   default (`0`).

After both fixes: B200 live repo — 19 passed (fail-closed 10 + shared-payload
6 + manifest-provenance 3), 26 correctly SKIPPED (parity/wiring tests,
wheel-gated, B200 still on 0.1.2), 0 failed.

### 8.4 a100b gap: `generate_gumbel_selfplay_data.py` NOT deployed there

a100a and B200 both import `OpponentPoolRuntime` from
`catan_zero.rl.gumbel_self_play` fine (that feature already exists on both).
a100b's `gumbel_self_play.py` has ZERO references to `OpponentPoolRuntime`/
`opponent_pool` — that whole feature is entirely absent from a100b's
codebase (confirmed: `grep -c` returns 0). Deploying the shared
`generate_gumbel_selfplay_data.py` there broke import entirely:
`ImportError: cannot import name 'OpponentPoolRuntime' from
'catan_zero.rl.gumbel_self_play'`. Caught immediately via the test run (not
left broken), reverted that ONE file on a100b back to its original
(pre-deploy) content from a saved backup — confirmed via `git status`
showing zero diff against HEAD afterward.

**Root cause + resolution (team-lead, accepted):** this is the known 3-way
repo-commit split — a100b's live repo is still on OLD commit `2ce38f4`,
which PREDATES the opponent-pool feature entirely (a100a is on `34b16d9`,
B200 on `ea6ce93`). Decoupling the `OpponentPoolRuntime` import as a
workaround was explicitly REJECTED as "a hack around the real problem." The
correct fix is to END the split: at a100b's Phase-C flip moment (its next
natural generation-restart boundary), fast-forward a100b's repo to master,
then redeploy the wiring files on top (or, if the gen-3 sync has landed on
master by then, they arrive already merged) — one operation, not two. Until
then a100b simply doesn't get the `--rust-featurize` CLI flag, which is
fine: it keeps generating flag-off (bit-identical to before), while
B200/a100a can flip ahead of it.

**Net effect:** a100b's evaluator-level wiring (`neural_rust_mcts.py` +
wrapper modules) IS deployed and fully tested there (42/42 passed, run
directly against a100b's own already-installed 0.1.3 wheel from the earlier
incident — the strongest kind of validation, a real wheel + real code, not a
skip). But the `--rust-featurize` CLI flag and its manifest field are **NOT
yet available on a100b** — that host's `generate_gumbel_selfplay_data.py`
is still the pre-Phase-C version, pending either the opponent-pool feature
landing there too, or someone decoupling the shared file's hard dependency
on `OpponentPoolRuntime`. Not this agent's call to resolve — flagged to
team-lead.

### 8.5 Test results summary (all three live repos, flag OFF, no restarts)

| Host | Wheel | Suites run | Result |
|---|---|---|---|
| B200 | 0.1.2 (unchanged, confirmed) | fail-closed + shared-payload + manifest-provenance (parity/wiring correctly skip) | 19 passed, 26 skipped, 0 failed |
| a100a | 0.1.3 (pre-staged from the earlier incident) | full set incl. parity/wiring (actually EXERCISED, not skipped) | 45 passed, 0 failed |
| a100b | 0.1.3 (pre-staged from the earlier incident) | all except manifest-provenance (CLI tool not deployed there, §8.4) | 42 passed, 0 failed |

Live worker/training process counts and GPU utilization were checked
immediately before and after every deploy step on every host; unchanged in
all cases (B200: 9 procs before/after; a100a: 118→102 procs, consistent with
natural wave completions unrelated to this deploy; a100b: 128 procs
throughout).

### 8.6 Flag-flip readiness state (as of Phase C completion)

| Host | Readiness | What's left |
|---|---|---|
| B200 | Wiring live, wheel still 0.1.2 | Needs the 0.1.3 wheel install at a flywheel round boundary (§3-4), then flip |
| a100a | FULLY ready | Wheel (0.1.3) + wiring + `--rust-featurize` CLI all in place; just needs the flag added to its next launch command |
| a100b | Wiring live, CLI blocked | Needs its repo fast-forwarded to master (§8.4) before the CLI flag lands there too |

B200 and a100a can flip ahead of a100b; a100b's flip happens at its own
natural restart boundary, bundled with the repo fast-forward (and possibly
the gen-3 sync's master-merge, if that lands first) as one operation.
