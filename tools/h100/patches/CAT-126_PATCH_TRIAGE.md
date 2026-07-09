# CAT-126 — h100 Patch-Set Triage (per-patch verdicts)

**Reviewer:** audit-fixer · **Date:** 2026-07-09 · **Scope:** `tools/h100/patches/` (13 patches → 32 findings), read-only.
**Method:** read every patch + applier + target source; verified the load-bearing claims on the fleet's torch 2.7.0 / repo (`~/cz-e2e/repo`).
**Bottom line:** do NOT run `apply_all.sh`. Only 4 findings are safe to adopt as defaults now; 2 "likely-ACCEPT" candidates are actually BROKEN or unsafe (verified), and 1 "optimization" (#20) would OOM the box.

## Legend
- **BI** = bit-identical output · **BEH** = behavioral (changes generated data / model / training trajectory) · **OPS** = operational only (no data/model change).

## Verdict table

| # | Patch file | Finding | Verdict | BI/BEH | Rationale (verified) |
|---|---|---|---|---|---|
| 1 | 11_threaded_selfplay_gen.py | threaded self-play (ref) | **REJECT** | BEH | Proven GIL-wall regression: **0 rows vs 121.6K/hr** for 16-proc+MPS; featurize is 96% GIL-bound so threads serialize. Triply confirmed (fleet bench task #78, prior branch threaded-gen-batched@3eaec27, lead). |
| 1 | apply_12_threaded_generation.py | `--use-threads` flag | **REJECT** | BEH | Same. Adds an opt-in flag (not a default), but shipping it invites a throughput-tanking footgun. Finding #1's 2-4x premise is false ⇒ #16 & #24 ("free with #1") also don't materialize. |
| 2 | 01_…bf16_compile.patch / apply_01 | bf16 autocast inference | **DEFER (gate)** | **BEH** | Changes generated policy/value targets (bf16 matmuls). The patch comment "no behaviour change" is **FALSE** — casting outputs back to fp32 doesn't undo bf16 precision loss. Marginal at batch=1 (still launch-bound; threading rejected). Also drops `no_grad` on the CUDA branch (autocast-only) → autograd graph build. Needs masked-parity + quality + throughput micro-gate first. |
| 3 | 01_…bf16_compile.patch / apply_01 | torch.compile in `__init__` | **REJECT (as default)** | BEH | **VERIFIED bug:** `torch.compile(model)` makes `state_dict()` keys `_orig_mod.*` (torch 2.7.0). `EntityGraphPolicy.save()` writes `self.model.state_dict()` ⇒ **corrupts checkpoints + changes model md5** (breaks gate baseline 8fadfb36 + fleet load compat). Plus `reduce-overhead` CUDA-graphs recompile on batch-1 dynamic legal-action shapes. Revisit only as DEFER-experimental with prefix-strip + `dynamic=True` + checkpoint-roundtrip test. |
| 10 | 01_…bf16_compile.patch / apply_01 | non_blocking H2D | **REJECT** | — | **VERIFIED crash:** `torch.as_tensor(value, device=…, non_blocking=True)` → `TypeError: as_tensor() got an unexpected keyword argument 'non_blocking'` on torch 2.7.0. Applying patch 01 **breaks the first leaf eval** = generation down. (Even fixed, it's a no-op without `.pin_memory()`, which the patch omits.) |
| 15 | apply_02_lru_cache.py | FIFO→LRU eviction | **ACCEPT (default)** | **BI** | LRU only changes *which* entries evict, never returned values (cache hits are deterministic) ⇒ bit-identical outputs. Flips default in-code (3 sites). Caveat: ~zero value for self-play (unique states never hit; OPT-1 disables the cache) — real value only in repeated-position gating/H2H. Harmless ⇒ accept. |
| 19 | 03_wave1_harvest_parallel.sh | serial→parallel rsync | **ACCEPT (ops)** | OPS | Moves identical bytes; parallel-per-box+dir + SSH ControlMaster, correct (bg+wait). Verify the hard-coded `DIRS` map still matches live out-dirs before an authoritative build (its own comment warns). |
| 4 | 06_teacher_shard_size_fix.sh | shard_size for n128 | **ACCEPT intent / REJECT wrapper** | BI (data) | Smaller n128/n256 shards = faster first-shard, data-identical. **But** the wrapper's `${@/--shard-size */…}` is a greedy substitution that **drops every arg after `--shard-size`**. Adopt as an in-code auto-default keyed on `--n-full`, not this script. |
| 30 | 10_seed_ledger_sync.sh | cross-host ledger sync | **ACCEPT goal / fix script** | OPS | Real P0 gap (no cross-host collision detection). But the script is a **non-idempotent append** (re-run/cron duplicates blocks; grep may copy header rows; assumes runsix-path ledger). Make idempotent (dedupe / replace-section) before cron-izing. |
| 5 | 08_…rustfeaturize_wrapper.sh | npz_zst compression | **REJECT (as default)** | BI (data) | `build_memmap_corpus.py` has **no zstd-read path** and the harvest include-glob is `gumbel_self_play_shard_*.npz` (not `.npz.zst`) ⇒ compressed shards would be **neither harvested nor read** = silently breaks the corpus pipeline. Defer until the reader + harvest support `.npz.zst`. |
| 18 | 08_…rustfeaturize_wrapper.sh | `--rust-featurize` | **DEFER (gate + prereq)** | BEH | Bit-exact only if the parity test passes AND every box has the `catanatron_rs.build_entity_features_flat` wheel. The cz-e2e venv **lacks** it → `--rust-featurize` fails loudly at first leaf (no fallback). Deployment prereq + parity gate before any default flip. Do NOT co-adopt with #5 (same wrapper). |
| 20 | apply_07_build_memmap_parallel.py | parallel build_memmap | **REJECT** | — | Collects **all** shards' normalized arrays into `_shard_norms` before writing → reintroduces the exact whole-corpus OOM the memmap builder exists to prevent (docstring: OOM'd 32.6M rows / 708 GB host). Order is preserved but peak RAM = entire corpus. Goal valid; needs a *bounded* producer-consumer, not load-all-then-write. |
| 8 | apply_04_optimizer_state.py | persist Adam state | **ACCEPT (DDP) / fix for FSDP** | BEH | Load fails-safe (try/except); strictly better resume for single-GPU/DDP. **But** saves `optimizer.state_dict()` rank0-only = **incorrect under FSDP** (sharded state); the target is the FSDP stack. Add an FSDP guard (skip, or `FSDP.optim_state_dict`) before it's safe there. |
| 6 | 09_training_hyperparam_wrapper.sh | value-loss-weight 0.5 | **REJECT** | BEH | Contradicts the CAT-106 14-arm sweep — tuned floor is **vlw 0.10** (run6 already lowered 0.25→0.10). 0.5 goes the wrong direction. Also bumps `loser-sample-weight` 0.3→0.5 (untuned). Finding #6's "value underweighted" premise was already explored (EXP3/CAT-106). |
| 7 | 09_training_hyperparam_wrapper.sh | cosine LR decay | **DEFER (A/B)** | BEH | `--lr-schedule cosine` exists + is wired. Plausible for longer grow runs; gate it. |
| 22 | 09_training_hyperparam_wrapper.sh | batch-size 4096 | **DEFER (A/B)** | BEH | Optimization-dynamics change; A/B on loss/val curves. |
| 23 | 09_training_hyperparam_wrapper.sh | grad-accum | **DEFER** | BEH | Only if memory-limited; effective-batch change. A/B. |
| 31 | 13_optional_heads_configs.py | distributional/uncertainty/x-attn heads | **DEFER (research)** | BEH | Warm-start-safe (zero-init) but each head needs train + gate. Config-generator only, no code change. |
| 12 | 05_a100a_relaunch.sh | a100a wrong c-scale (garbage data) | **DEFER-to-unfreeze** | (data-quality) | Finding is VALID (old `catan-zero` stack, c-scale 0.1, missing flags → corpus-incompatible data; don't mix into gen-5). But it's a GPU-ops relaunch — frozen now, and the pilot is already dead (gpu6 SIGKILL + full-stop). On unfreeze, relaunch on runsix stack w/ full flag set; never ingest the old-pilot data. |
| 13,14 | 05_a100a_relaunch.sh | GPU6 idle / workers=4 | **DEFER-to-unfreeze** | OPS | a100a-pilot utilization ops; moot during freeze. |

## Findings with no patch (informational / handled elsewhere)
- **#29 three incompatible forks (P0):** organizational — addressed by the run6-consolidated / convergence work (CAT-115), not a patch.
- **#21 npz-loads-all-RAM:** already mitigated — new stack streams via `--data-format memmap`.
- **#16 cross-worker cache, #24 checkpoint-load×16:** were "free with Finding #1" — but #1 is rejected, so no free win.
- **#9 EvalServer stub, #25 json.dumps, #26/#27/#28/#32, #11 MPS+EXCLUSIVE_PROCESS(done), #17 forced-decisions:** INFO / already-done / Rust-API / research. No action.

## Recommended Wave-2 default adoptions (after CAT-116 tags)
Only these, and only in-code (not via the wrappers/apply_all):
1. **#15 LRU** (bit-identical, drop-in).
2. **#19 parallel harvest** (ops; re-verify DIRS map).
3. **#4 shard-size auto-default by n-full** (re-implement in the gen script; do NOT use the arg-dropping wrapper).
4. **#30 ledger sync** — after making it idempotent.
Plus **#8 optimizer-state** once FSDP-guarded (DDP-safe today).

Everything else: REJECT or gate (A/B) as tabled. **`apply_all.sh` must not be run** — it applies the broken patch 01 (#10 crash, #3 checkpoint corruption) and the OOM-inducing #20.
