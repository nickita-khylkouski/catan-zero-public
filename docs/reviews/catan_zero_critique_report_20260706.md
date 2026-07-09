# Catan-Zero — External Critique & Improvement Report

**Date:** 2026-07-06
**Method:** Four parallel investigations — (1) line-by-line audit of the search / self-play core on the live B200 master tree, (2) line-by-line audit of training / gating / flywheel infra, (3) literature review of 2022–2026 AlphaZero/MuZero-family work mapped to each open problem, (4) survey of comparable open-source projects — plus my own direct reads of the search core and live-fleet verification.
**Confidence convention:** ✓ VERIFIED = code was read and traced; ? INFERRED = from pattern/grep or single-source; ✗ ABSENT = confirmed missing by find + repo-wide grep.

---

## 0. TL;DR — the five things that matter

1. **The system is real and the gen-1 result (57% over base at low search) is trustworthy.** The chance-node backup math, sequential-halving bookkeeping, per-actor value sign, soft-policy renormalization, GSPRT/pentanomial gate math, and the masked corpus loading are all ✓ VERIFIED correct. Suspected bugs in these areas were checked and **refuted**.

2. **The design doc materially overclaims the flywheel infrastructure.** Several things the narrative describes as "built" or "fixed" are ✗ ABSENT or FALSE in the actual tree: the `flywheel/` package, the KataGo window-math constants, and the "two-phase promotion journal." The continuous flywheel is *less* built than the doc implies. This is the single biggest risk to the project's own planning.

3. **The dominant search cost is architecturally un-batched.** Leaf evaluations in the main selection loop are run one-at-a-time; the batched evaluator only fires on chance fan-outs, and the self-play driver is single-threaded — so the async micro-batcher's latency tax buys *zero* batching benefit on the majority of leaf evals. This, plus per-leaf recomputation of invariant board topology, is where the ~500k-evals/game pain actually lives.

4. **Two latent leak/regularization bugs are live.** A new unmasked observation path (`rust_action_context_batch`) reintroduces the exact f72 hidden-info-leak class with zero test coverage (benign *today*), and `--weight-decay` is a silent no-op on the default `adam` optimizer while `report.json` reports it as applied.

5. **The literature points at one high-confidence, cheap win the project is not doing:** the pure Monte-Carlo outcome value target is directly contradicted by published results and the problem is *amplified* by the project's own Go-Exploit restarts. Bootstrapped/hybrid value targets are the highest-leverage research change available.

---

## 1. Fleet & live status (✓ VERIFIED directly)

- **18 GPUs:** 2× **B200** (host `B200`, H2H / verification) + 16× **A100-80GB** across two 8-GPU hosts (`a100a`, `a100b`). All at 86–100% utilization.
- **Gen-2 generation is live**, base = gen-1 checkpoint, seeds from 30,000,000, 8 workers/GPU. Read straight off the running workers' command line:
  `--n-full 64 --n-fast 16 --p-full 0.25 --c-visit 50.0 --c-scale 0.03 --max-decisions 600 --temperature-decisions 90 --lazy-interior-chance --public-observation --correct-rust-chance-spectra`.
  This **matches the gen-1 recipe** — no config drift on the live fleet. Good.
- **Stale-config hazard (local):** `modal_gumbel_factory.py` in this worktree still defaults to `c_scale=1.0`, `max_decisions=300`, `temperature_move_fraction=0.15`, and has **no masking / public-observation flag at all**. A Modal wave launched from this file as-is would generate wrong-`c_scale`, wrong-horizon, **omniscient** data. This is exactly the "CLI-default-override trap" the project already documented — it is still armed here.

---

## 2. Design-doc vs. reality discrepancies (the most important section)

The narrative document is excellent as a research diary but has drifted from the code. Each of these was confirmed by reading the actual tree:

| Claim in the doc / memory | Reality in code | Status |
|---|---|---|
| `src/catan_zero/rl/flywheel/` with `replay_window.py`, `opponent_pool.py`, `checkpoint_registry.py` exists | No such directory anywhere (find + repo-wide grep = 0). Real pipeline is `tools/selfplay_loop.py` → `generate_gumbel_selfplay_data.py` → `build_gumbel_gen_manifest.py` → `train_bc.py` → `promotion_gate_runner.py` | ✗ ABSENT |
| "Window math already matches KataGo α=0.75, β=0.4" | No `N_window = c(1+β((N/c)^α−1)/α)` formula anywhere. Actual: `--replay-fraction 0.15` + `--replay-anneal-gens 3` (fixed-fraction teacher mix-in, not a size-scaling window) | ✗ ABSENT |
| "Two-phase promotion journal landed" | `promotion_gate_runner.py` has zero atomic/flock/journal logic; verdict written by a single unguarded `write_text`. The only journal is `selfplay_loop.py`'s `loop_state.json` via tmp+`os.replace` — atomic but **single-phase, no flock** | FALSE |
| "Opponent pool built, to be wired" | `league.py` (real PFSP league) + `policy_pool.py` exist but are imported **only by the legacy PPO path**. Gumbel self-play is checkpoint-vs-itself only | ✓ (matches "not wired") |
| Training "1 epoch / batch 4096" | Defaults are **2 epochs / batch 65536** | Doc understates |
| "value-loss-weight resolved to 1.0 (AZ recipe)" | Default is still **0.25**, wired unchanged end-to-end, never overridden | FALSE / not applied |
| "Gate now masked (leak fixed)" | Entity-token gate path is masked ✓, but a **sibling Rust eval path (`rust_action_context_batch`) has no masking parameter at all** — latent re-introduction of the same leak class | Partially FALSE |
| "lazy loses 21-2 was a stale-checkpoint artifact; lazy recovers to near-parity" | Consistent with code (lazy interior = documented single-sample estimator, not a bug). Non-inferiority SPRT still "continue" | ✓ |

**Action:** treat the doc's "flywheel" section as a *design intent*, not an inventory. Before the continuous flip, the window formula, opponent-pool wiring, and a real promotion journal all have to actually be written — they are currently prose.

---

## 3. Correctness findings

### 3.1 What is verified-correct (suspicions refuted)
- **Chance-node backup is properly probability-weighted** (`gumbel_chance_mcts.py:982-1011`, `1043-1131`). Every outcome is materialized/evaluated up front; one child advances deeper per visit but the backed-up value is `Σ p_i · child_i.value` — a Rao-Blackwellized / Stochastic-MuZero-style expectation, lower-variance than naive MC. ✓ VERIFIED
- **Sequential-halving budget/candidate bookkeeping** has no off-by-one; the schedule's `count` sequence exactly mirrors `keep=max(1,count//2)` (`:138-155` vs `:615-654`). Gumbel-Top-k sampling-without-replacement is the textbook Gumbel-max trick. ✓ VERIFIED
- **Sign/perspective** always accumulates in root perspective, flips for opponent decision nodes (`:696-698, 856-858`). ✓ VERIFIED
- **`_completed_q` v_mix** uses prior-weighted inner average (F1c), matching mctx, invariant to the budget-dependent visit distribution. ✓ VERIFIED
- **Loss math:** soft policy target renormalized over legal support before CE (`train_bc.py:4193, 4231, 4370`); per-actor value sign correct by construction (`:4677-4702`); `policy_weight_multiplier` (binary 1/0 for full/fast) is persisted and applied on **both npz and memmap paths** (`gumbel_self_play.py:466-468, 99, 610-612`; `build_memmap_corpus.py:82,182`; `train_bc.py:3278-3316, 2680-2690, 4973-4974`). ✓ VERIFIED-SAFE (this was the sharpest suspected training-poisoning bug — it is **not** a bug).
- **GSPRT/pentanomial gate** matches Van den Bergh/fishtest: logistic elo→score, GSPRT mean/variance LLR (not naive binomial), correct trinomial reduction (Catan has no draws), exact Wald bounds, mean-neutral regularization prior (`sprt_gate.py:37,46-52,195-293`). ✓ VERIFIED. *One external caveat from the survey:* confirm the port re-estimates H0/H1 per-step (true GSPRT) and consider a discrete-time overshoot correction; diff against `vdbergh/pentanomial` rather than fishtest's webapp.
- **D6 dihedral group** in `hex_symmetry.py` is algebraically sound (rotation order 6, reflection order 2, inverse-via-argsort); `PUBLIC_MASK_PLAYER_SLOTS` masks only true hidden fields (hand comp, dev-card identities), correctly keeps public counts/VP. ✓ VERIFIED

### 3.2 Live/latent bugs (ranked)
1. **Unmasked Rust eval path — latent leak, zero coverage.** `neural_rust_mcts.py:817-849` (`rust_action_context_batch` / `_resolve_entity_adapter`) has **no `public_observation` parameter**; every call builds a fully unmasked payload at all four `evaluate()` sites. Benign *today* only because `action_features.py:67-107` currently reads already-public fields — but this is the f72 leak class with no test guarding it (unlike `tests/test_public_observation_masking.py`, which covers only the entity-token path). **Add the parameter + extend the masking test now**, before some future feature reads a hidden field. ✓ VERIFIED
2. **`weight_decay` silent no-op.** `_make_optimizer` (`train_bc.py:5416-5437`) passes `weight_decay` only in the `optimizer_name=="adamw"` branch; default optimizer is `adam`, so `--weight-decay X` is silently ignored — yet `report.json` records it as applied (`:1224`). Fix: apply for both, or raise at arg-parse if `weight_decay!=0 and optimizer!=adamw`. ✓ VERIFIED
3. **Truncated-game value signal defaults OFF.** `truncated_vp_margin_value_weight` defaults `0.0` everywhere (`train_bc.py:1327` …). With a historically high truncation rate, a large class of rows contributes **zero value-loss signal** unless explicitly enabled. Not a bug, but a likely-unintended recipe gap — the proxy itself is correctly scale-matched to ±1 when on (`:4636-4671`). Default it > 0 or emit a loud warning at train start. ✓ VERIFIED
4. **Seed-collision guard is entirely external.** `game_seed = base_seed + game_index`, `--base-seed` default `1`, **zero internal range-reservation/assertions** (`gumbel_self_play.py:777`, `:139`). Two workers that forget to coordinate `--base-seed` silently replay identical games — the exact task-#77 class, uncaught by this file. Add an internal claimed-range lockfile/registry. ✓ VERIFIED
5. **`build_memmap_corpus.py` duplicate-seed detector has a shard-boundary false-negative** (`:308-326`): a seed spanning a shard boundary as a continuation, then reappearing later in that same shard, is not registered in `_closed_game_seeds` → missed. And even a *caught* duplicate is a stderr WARNING only, never an abort (`:382-391`). ✓ VERIFIED
6. **`h2h_postrepair_aggregate.py` has no dedup at all**, unlike `h2h_v3conf_aggregate.py` which added `_dedupe_games` (6-field signature, `:53-70`) precisely because a seed collision once double-counted 64 games bit-for-bit into a false-significant verdict. Port `_dedupe_games` before trusting any postrepair rerun. ✓ VERIFIED

### 3.3 Robustness / provenance
- **`report.json` still omits** `args.data` (which corpus trained this checkpoint), `truncated_vp_margin_value_weight` (was the proxy active), `validation_game_seed_ranges` (true-holdout vs random split). These are the three sharpest remaining provenance gaps. Cheap to add. ✓ VERIFIED
- **Host-reimage bomb:** `tools/gumbel_search_cross_net_h2h.py` — the masked H2H tool that **decides every promotion** — is **untracked in git, zero history** (plus a stray `.bak.1783308135`). `tools/train_bc.py` and `tools/build_memmap_corpus.py` carry **uncommitted local diffs containing the very fixes the doc says landed** (`--max-steps`, dup-seed detector, arch-gating fix) — content confirmed correct, but **none is in git**. Also untracked: `build_synthetic_manifest.py`, `gumbel_search_vs_bot_h2h.py`, `launch_value_repair_v2_train.sh`. **If any host is lost, the promotion gate and the audited fixes vanish.** ✓ VERIFIED
- **Dead-code trap:** `src/catan_zero/rl/promotion_gate.py` (32 lines, its own unrelated `decide_promotion()`, zero importers) can misdirect a reader away from the real gate in `promotion_gate_runner.py:404`. `rust_mcts.py`'s standalone PUCT engine and the `reanalysis*.py` family are pre-pivot legacy (frozen at `7d08869`), not on the live path. ? INFERRED (grep-level) — grep before pruning.

---

## 4. Performance findings (where the ~500k-evals/game pain lives)

1. **[HIGHEST] Main search loop never batches leaf evals.** `_run_root_search` (`gumbel_chance_mcts.py:631-636`) calls `_simulate` → `_expand` → `evaluator.evaluate(...)` **synchronously, one leaf at a time** (`:1277-1295`). `evaluate_many()` is used **only** for chance fan-out (≤11 ROLL children, ≤5 robber/dev). The self-play driver is single-threaded (`gumbel_self_play.py:738`, no thread spawn), so `BatchedEntityGraphRustEvaluator`'s async micro-batcher (`neural_rust_mcts.py:436-495`, `max_wait_ms=3.0`) **never receives concurrent requests — its latency tax is pure overhead today.** This is the dominant, currently-serial cost path. ✓ VERIFIED
2. **`_topology()` rebuilt on every leaf.** The 19-tile incidence structure is game-invariant but recomputed from scratch every call (`entity_token_features.py:150-204`, invoked `:121`) — ~500k redundant rebuilds/game, the plausible top contributor to the 42%-featurize cost. Memoize per board layout. ✓ VERIFIED
3. **Board state reconstructed twice per leaf.** `_resolve_entity_adapter` runs the full payload build independently from both `rust_game_to_entity_batch` and `rust_action_context_batch` inside one `evaluate()` (`neural_rust_mcts.py:159-215, 724-778, 817-849`). Cache and reuse the tuple → ~halves per-leaf CPU featurization. ✓ VERIFIED
4. **No cross-move subtree reuse.** `search()` always builds a cold `_GNode` root (`:426, 462-463`); `play_one_game` never threads a warm-start root across decisions. Every decision pays for a fully cold tree — unlike standard AlphaZero/KataGo. ✓ VERIFIED
5. **`_vertex_tokens` throws away the adjacency table it's handed** (`del topology`, `:228`) and redoes an O(54×19) robber-adjacency rescan (`:568-576`) instead of an O(19) lookup. ✓ VERIFIED
6. Minor: `winning_color()` re-queried via FFI on every `_simulate` though the node is immutable (`:856`); FIFO (not LRU) transposition-cache eviction (`:239-241`); un-vectorized 72-iteration loop in `_edge_tokens` (`:283-296`). ✓ VERIFIED

**The 96%-featurize+FFI figure is real and NOT a batching problem** — batching the NN forward (item 1) helps GPU utilization but the structural fix is items 2–5 (kill redundant CPU work) and, longer-term, moving featurization + inference into Rust.

---

## 5. Architecture assessment (transformer)

Mixed verdict — *not* a blanket "wasted capacity," but real headroom (all ✓ VERIFIED against `entity_token_policy.py`):
- **Adjacency tables computed but unused at the model level:** CONFIRMED — attention is dense/unmasked, no graph-conv or adjacency bias; 4 of 6 token builders `del topology`. The graph structure is available to the featurizer and thrown away.
- **No action↔board cross-attention / value reads only CLS / policy = scaled-cosine-to-CLS + additive bias:** CONFIRMED for the production config — but a fully-built, **zero-initialized, flag-gated upgrade already exists** (`action_target_gather`, `action_cross_attention_layers`, `value_attention_pool`), backward-compatible with current checkpoints. Whether any live checkpoint enables it is unconfirmed. This is the cheapest architecture lever: it's already written and warm-start-safe.
- **Symmetry:** the value net violating D6 (the doc's ~3.3× test-time-averaging denoise) is real and the averaging is architecturally sound — **contingent on one unverified dependency**: the env-payload id-numbering and the catanatron-map id-numbering are two independently-implemented sorts that must agree bit-for-bit. This is claimed-tested but was **not** independently confirmed. Verify it explicitly; a mismatch would silently corrupt symmetry averaging (and any `--symmetry-augment` training).

---

## 6. Literature-driven improvements (2022–2026), mapped to open problems

**Highest confidence, highest leverage:**
- **Bootstrapped/hybrid value targets instead of pure MC outcome** — Willemsen, Baier, Kaisers, *Value targets in off-policy AlphaZero* (NCA 2021). Finds pure-outcome targets have **both high bias and high variance** and are beaten by soft-Z / A0GB greedy backups on Connect Four & Breakthrough. **The project's Go-Exploit/RGSC archive restarts make trajectories *more* off-policy, amplifying exactly this failure mode.** This is the single most actionable research change. Also aligns with MuZero-style **lagged target networks** (already on the roadmap).
- **UCT-V-P / variance-aware completed-Q** — arXiv:2512.21648 (2026). Replaces the hand-tuned exploration/scale constant with a per-node **value-variance**-weighted term (a ~3-line backprop change). This is a *principled* replacement for the `c_scale=0.03` hand-tune and validates promoting the already-built **D2 variance-aware arm** from "fallback, default OFF" to a primary experiment. Note: `c_scale=0.03` is not "wrong" under Gumbel's proof (which specifies no correct scale), but it is a symptom this literature exists to fix.
- **Stochastic MuZero backup semantics** — Antonoglou et al. (ICLR 2022), and *Monte-Carlo \*-Minimax* (arXiv:1304.6057). Confirms single-sample/lazy interior chance (α≈0 double-progressive-widening) is a *reasonable* choice for Catan's moderately-dense dice — enumeration's 5400-eval cost is not worth it. Raises confidence that any lazy-vs-full gap is a backup detail, not a strategy error.

**Supports current choices:**
- **AlphaZe\*\*** (PMC10213697) — plain masked/observation-only AlphaZero is "surprisingly strong" on Stratego/DarkHex. Direct evidence the public-observation masking choice is viable without belief-state machinery (ReBeL/Student of Games are the heavyweight alternative, not needed yet).
- **PIMC-with-postponing** (arXiv:2408.02380) gives a concrete **strategy-fusion diagnostic**: check whether the masked value net is overconfident in high-private-variance states (right after a trade / dev-card buy). Run this rather than assuming masking is safe by analogy — Catan sits in their "public observation" (safe) category, but verify empirically.

**Sample efficiency:**
- **ReZero** (arXiv:2404.16364, LightZero) backward-view reanalyze + entire-buffer reanalyze, and **EfficientZero-V2** (arXiv:2403.00564). Since *generation* (not training) is the throughput bottleneck, reanalyze reuse extracts more signal per already-generated game — directly reduces games-needed-per-iteration. Note the project's `reanalysis*.py` is pre-pivot legacy; true reanalyze for the Gumbel pipeline is **unimplemented** (per its own design doc).

**Asymmetric-game stability:**
- **AlphaStar PFSP** (Nature 2019) + **Minimax Exploiter** (arXiv:2311.17190): when the opponent pool is finally wired in, use **win-rate-weighted (PFSP) sampling**, not uniform-from-history. Low integration cost, directly targets cycling/forgetting — higher risk in asymmetric Catan (first-player advantage).

**Symmetry / throughput:**
- **Finite-group equivariant nets** (arXiv:2009.05027) would eliminate *both* the ~3.3× symmetry-violation noise *and* the 12× test-time-averaging cost at once — but it's unproven territory for a hex graph transformer (KataGo itself only does augmentation + test-time averaging, not architectural D6 equivariance). High payoff, nontrivial cost.
- GPU-tree-search papers (arXiv:2104.04278, 2310.05313, MCTS-NC) all assume the **NN forward** is the bottleneck — which it is **not** here (4% GPU). Don't overinvest; featurize-in-Rust remains the correct lever.

**Speculative:** **OptionZero** (arXiv:2502.16634, ICLR 2025 oral) — learned temporal options increase effective search depth per fixed budget, an indirect angle on the wide-shallow-root problem.

---

## 7. Competitive landscape

- **No prior Catan AlphaZero beats catanatron's AlphaBeta.** catanatron's strongest bot is hand-tuned `AlphaBetaPlayer` (its maintainer tried and abandoned alpha-zero-general/muzero-general). CatAnalysis is the closest AZ-for-Catan attempt but has no benchmarked strength numbers and ~50–150 sims. Henry Charlesworth's PPO agent explicitly did **not** reach superhuman. OpenSpiel Catan never shipped. **The first-mover claim holds** — a rigorously-gated Gumbel-AlphaZero Catan engine with masked public observation is genuinely new territory. That is good for novelty and bad for available reference implementations: **every integration point is un-vetted by prior art.**
- **The mctx completed-Q rescale-noise finding appears genuinely unreported** (searched mctx issues incl. #66/#79/#81/#87/#93 and general). One nuance for write-up: the **Gumbel MuZero paper itself flags c_scale sensitivity** (Beam Rider footnote) and floats variance-normalization as future work — cite that as partial prior art. The *wide-root + low-budget false-confidence mechanism* and the *James-Stein/SE-shrinkage fix* are the novel contributions. **File it as an mctx issue / short note.**

**Top "steal this" items:**
1. **KataGo NNEvaluator dedicated-eval-server-thread batching** (`cpp/neuralnet/nneval.cpp`) — the architecture that fixes finding §4.1 (search threads submit to a queue, a server thread batches). Prerequisite: parallelize search calls (there's currently no threading to batch across).
2. **In-Rust NN inference via `tch-rs`/`candle` (ZanLing-TrueZero precedent)** — reframes "featurize in Rust" as "featurize *and* infer in Rust," killing the Python round-trip (the 26% FFI + most of the 42% featurize tax). The correct gen-2+ structural target over incremental PyO3 cleanup.
3. **KataGo Playout Cap Randomization details / Forced Playouts + Policy Target Pruning** — the project already does PCR; forced-playouts + target-pruning is a complementary lever against the same wide-root false-confidence that D1/D2 attack from the other side.
4. **Fishtest per-worker residual tracking** — cheaply catches cross-host (A100A/A100B/B200) generation skew before it poisons an H2H run, as a seed collision already did once.
5. **LightZero `ctree`** (C++/pybind tree bookkeeping) — the precedent for porting Gumbel/Sequential-Halving tree logic into Rust rather than optimizing the Python tree.
- Note: **no reference implementation combines Gumbel action-selection + Stochastic-MuZero chance nodes** (open ask in mctx #66). The chance-node integration is genuinely new surface — budget review time accordingly.

---

## 8. Ranked recommendations

**Do this week (cheap, high-value, mostly hygiene):**
1. **`git add` + commit `tools/gumbel_search_cross_net_h2h.py`** and the uncommitted verified-correct diffs in `train_bc.py` / `build_memmap_corpus.py`. The promotion gate currently exists only on one untracked host file. *(Requires user approval per repo git-safety rules — flagged, not auto-run.)*
2. **Patch the local `modal_gumbel_factory.py` defaults** (`c_scale`, `max_decisions`, `temperature_move_fraction`) and add a masking/public-observation flag, or add a guard that refuses to launch unmasked. It is an armed wrong-regime-data footgun.
3. **Add `public_observation` masking to `rust_action_context_batch`** + extend the masking test to cover it. Closes the latent f72 leak class.
4. **Fix the `weight_decay` no-op** (apply for both optimizers, or raise on mismatch); stop reporting it as applied when it isn't.
5. **Add `args.data`, `truncated_vp_margin_value_weight`, `validation_game_seed_ranges` to `report.json`.**
6. **Port `_dedupe_games` into `h2h_postrepair_aggregate.py`**; fix the shard-boundary false-negative in the dup-seed detector; make a caught duplicate abort, not warn.
7. **Default `truncated_vp_margin_value_weight > 0`** or warn loudly when truncation is high and it's 0.

**Do before the continuous-flywheel flip (the doc says these are done — they are not):**
8. Actually write the replay-window size formula, the promotion journal (with flock), and wire the **PFSP opponent pool** into Gumbel generation. Add an internal seed-range registry/lockfile to `generate_gumbel_selfplay_data.py`.
9. **Verify the two id-numbering schemes agree bit-for-bit** before relying on symmetry averaging or `--symmetry-augment`.

**Research experiments (ranked by expected payoff):**
10. **Bootstrapped/hybrid value target (soft-Z / A0GB / lagged target net)** — highest-confidence win; the Go-Exploit restarts make the current pure-MC target's off-policy problem worse.
11. **Promote the D2 variance-aware completed-Q arm to a primary experiment**, framed as UCT-V-P; run the **joint `c_visit × c_scale` ablation** the roadmap already lists — the `c_scale=0.03` value likely compensates for value miscalibration rather than being intrinsically right.
12. **Batch leaf evals in the search loop** (KataGo eval-server pattern) — the dominant throughput lever once search is parallelized; pair with the topology-cache and double-reconstruction fixes for the CPU side.
13. **Enable the already-built action-cross-attention / value-attention-pool arch flags** as a warm-start-safe gen candidate.
14. **Reanalyze (ReZero backward-view)** to raise sample reuse, since generation is the bottleneck.
15. **Run the PIMC strategy-fusion diagnostic** to confirm masking is safe in high-private-variance states.

---

## 9. Publishability

Three genuinely citable pieces, in order of readiness:
1. **The mctx completed-Q rescale-noise diagnosis + variance-aware fix** — novel mechanism, novel fix, cite the Gumbel paper's Beam Rider footnote as partial prior art. File as an mctx issue first.
2. **A clean discrete-vs-continuous flywheel ablation on Catan** — no un-confounded version exists in the literature (but only if the continuous side is actually built and gated cleanly).
3. **First credible AlphaZero-class result on 2-player Catan** — if gen-N holds up under a large-sample high-search H2H vs catanatron AlphaBeta. No competing Catan-AlphaZero exists.

**Caveat that gates all three:** the strength claims are only as trustworthy as the git hygiene and provenance. Right now the gate tool is untracked and three provenance fields are missing — fix §8.1/§8.5 before publishing any number.
