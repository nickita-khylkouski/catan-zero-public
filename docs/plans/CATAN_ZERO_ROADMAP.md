# Catan-Zero Roadmap (PRD) — the whole program, step by step

**Companion to `CATAN_ZERO_MASTER_PLAN.md`** (the evidence ledger: 8 expert reviews, conflict rulings, literature). This document is the *executable* version: what we do, in what order, with what configs, on what hardware, and what result gates each step. References: [R1]–[R8] = reviews, [§x] = master-plan section, [US] = our own artifacts/tools.

---

## 0. The program in one paragraph

The measurement layer outgrew the learning loop ("over-invested in certainty, under-invested in compounding" [R7]). The loop has **three coupled bottlenecks**: (1) the **promotion valve is stuck shut** — a +30-Elo certification gate applied to a +20-Elo/turn regime blocks compounding by design; (2) the **value pipeline poisons itself** — the λ-blend trains toward the generating net's own archived root values (self-distillation, [R7]) and scalar MSE memorizes outcome noise under any reuse (rank collapse, [R8]); (3) the **search's improvement signal is decaying into a fixed noise budget** — the min-max rescale gives completed-Q constant magnitude while its signal fraction shrinks as the policy sharpens ([R8]; this predicts our +49→+49→+33→+20 curve). The build-and-shelve status is now explicit: the exact categorical formulation was wired, tested, and rejected for this wave; symmetry averaging and D1 are wired into the binding S1 calibration; opponent-pool and regret-restart work remains outside this bounded pre-wave search decision. The campaign: **open the valve → fix the value plumbing → denoise-then-crank the search → diversify the data → rebuild evaluation so we can see — and only then spend capacity (bigger net, architecture).**

**Execution update (2026-07-09):** the exact gen2B A0 replication falsified
the first HL-Gauss formulation.  Scalar MSE reproduced
`.665247 -> .809018 -> .841849`; 33-bin HL-Gauss regressed
`1.198052 -> 1.532889 -> 1.710083`.  The binding pre-wave ruling is therefore
scalar MSE/readout for A1, while the independent search sequence remains
D6 -> corrected c-scale/D1 -> n64/n128 -> adaptive n256.  The roadmap's HL
motivation remains historical context, not authorization to override that
result or retune several knobs inside this wave.

## 0.1 Architecture verdict (the explicit answer)

**There is NO big architecture rewrite in this plan. That is an 8/8-review consensus, and it's load-bearing: the data engine binds, not capacity** ("architecture doesn't matter *yet at this data scale*" [R8]; "at 3–4M rows/turn and a prior-starved policy, the data engine binds" [R7]).

What changes, by layer:

| Layer | Change | When | Size |
|---|---|---|---|
| **Loss/head** | Exact HL-Gauss A0 failed; retain scalar MSE/readout for the next fresh one-dose A1 run | Bound for this wave | complete decision |
| **Heads** | Aux heads ON (final-VP 0.02–0.1, road/army, production) + **TD-horizon heads on realized outcomes** + uncertainty/value-error head (stop-grad) + deduction/belief aux heads | W3–4 | days each |
| **Inputs** | Deduction-tracker features (near-zero-init projections, warm-start-safe) | W3–4 | ~week |
| **Trunk (cheap)** | Adjacency/graph-distance bias remains deferred until separately implemented, tested, and assigned its own causal arm | after current critical path | unscoped |
| **Trunk (real)** | Action-target cross-attention (v3b-class, done properly: ≥2 seeds, new-module LR multiplier, policy warmup, value frozen); graph-distance bias remains separate | Phase D only, after a promoted stable 35M candidate + ≥10M fresh rows | weeks |
| **Scale** | 80–100M net; production requires ≥10M fresh rows, equal exposure, and an endgame-distribution audit [R8] | Phase D, after a promoted stable 35M candidate | weeks |
| **Never (this phase)** | Full D6-equivariant transformer (augmentation + root averaging captures it [R7][R8]); ReBeL/SoG/CFR machinery; R-NaD conversion; MuZero learned model | — | — |

---

## 1. Standing rules (apply to every step below)

- Seeds: consult the **seed ledger** before any claim; VAL-ONLY range [6.19B, 6.2B) never trains. [US]
- Every gate: paired color-swapped, pentanomial, ledgered disjoint seeds; masked nets eval masked on master code. [US]
- Full CLI flag lists always (default-override trap, 7+ incidents). Typed-config migration runs in the background lane. [US]
- Anchor telemetry = **drift tripwire only**, never a promotion signal (gen-4 lesson [R8]).
- Every built feature gets a **wired-by date or a written shelf-reason** (anti build-and-shelve [R7]).
- Report language: turn-4 = "suggestive"; λ=0.5 = "adopted, unproven vs control"; compression trend = "unmeasured until WHR."

---

## 2. PHASE A — "Open the valve, install the instruments" (Days 1–4; mostly free)

### Step A0 (hour 0) — Read the 1000-game external panel
`runs/gates/v16_external/vs_value_500pairs.json` (B200). **Branch point [R8]:**
- **≥48%** → the parity contingency is live: goal reframes from "close −30" to "prove superiority with power"; promotion still proceeds (fresh distribution is needed either way).
- **≤44%** → external gap real; belief/exploiter workstreams gain priority.
- Either way the panel becomes the async tripwire baseline.

### Step A1 (day 1) — Gate re-spec + registry roles
- Flywheel/producer gate: **SPRT elo0=−10, elo1=+15, α=β=0.05, n=16, 300 games, extensions to 600** (existing extension machinery [US]). The 0/+30 spec survives only for "gen-N" turn announcements. [§2.2; R1–R8 consensus, R8's +10 noted as fallback if promotions time out]
- Registry roles written to disk: `generator_champion`, `public_champion` (stays gen-3 for now), `tournament_bot`, `opponent_pool[]`. Use the existing registry + `runs/CURRENT_CHAMPION` pointer runbook [US].
- Tripwires wired as *code, not judgment*: two consecutive external declines OR P(ΔElo_ext < −25) > 0.9 → auto-revert fleet to last externally-stable champion [R3][R6]. Every 3rd promotion: 200-game n=64 non-regression + bucketed win-rates (phase/opening/blowout) with per-bucket veto [R8].

### Step A2 (day 1) — Confirmation gate + promote
- ONE candidate (gen-4: clean provenance, artifact-verified) re-gated under the new spec (~34 min). The pooled "52.8%/1000" is *suggestive only* (mixed checkpoints, mixed n, one-sided p [R6][R8]) — this gate replaces it.
- Pass → **promote gen-4 to generator_champion** in the registry and hand the
  exact checkpoint/version update to the production data-lane owner. This
  historical roadmap step is not authority for the current bounded R&D lane to
  mutate fleet pointers.

### Step A3 (days 1–3) — Producer-vs-gen3 window A/B [R3]
- The production data lane generates matched producer and gen-3 windows on
  disjoint ledgered seed blocks; champion-init trains both on B200 and compares
  internal gate + refreshed anchor + external panel + high-regret suite. The
  current pre-wave program renders this handoff but does not launch it.
- This turns the promotion into a controlled experiment. Falsifier: producer window wins internally but loses externally twice → inbreeding confirmed → pool % up, linear promotion paused.
- Lighter fallback if fleet is contended: canary lane — candidate takes 20–30% of generation while panels run [R6].

### Step A4 (days 1–2, zero GPU) — WHR fit [R7]
- Ingest every game on disk (gates, extensions, ablation arms, panels) → Whole-History Rating (Coulom 2008) → one champion trajectory with honest error bars. Resolves whether the compression trend is even real [R8: CIs currently overlap]. This is also the arena's rating backbone.

### Step A5 (1 GPU-day) — Diagnostics bundle [R8] ← runs before any big remedy spend
1. **Search-SNR probe**: ~200 sampled full-search roots per checkpoint (v3a→gen-4); same search twice with different search seeds; report cross-seed argmax agreement, KL(π′₁‖π′₂), KL(π′‖prior). SNR-decay theory predicts KL(π′‖prior) ~constant while agreement decays.
2. **Rollout-doubling Elo**: champion n=64 vs n=128, 400 paired games. Big gap ⇒ search budget binds (mechanism B); tiny ⇒ capacity/data.
3. **Diversity telemetry** from existing corpora: unique-state fraction, opening entropy (decisions 1–30), line concentration per generation. Falling ⇒ mechanism C.
4. **Noise-vs-signal trend**: D6 orientation-noise std vs top-5 Q-spread per generation (D3 opening-panel + f74 symmetry tooling exist [US]). Spread→noise ⇒ mechanism A confirmed.
- Output: a one-pager assigning weight to (A) SNR-decay / (B) fixed-point / (C) narrowing — **directs the Phase B/C mix.**

### Step A6 (day 2) — Anchor refresh protocol [R7][R8]
- Build anchor_gen4 from the current window's held-out seeds (the reserved val machinery exists: `.valonly` ranges [US]); keep anchor_r7 as a longitudinal series; flywheel config marks anchors tripwire-only.

### Step A7 (days 2–4, CPU/parallel) — Hygiene batch
- **λ-arm vs gen2A direct H2H** if the 59.0% checkpoint survives (~34 min) [R6][R7]; champion the winner (ruthless champion ladder).
- Pentanomial pair-correlation measured once from existing gate records; fix the stated rationale [R7].
- **Exact-SH implementation audit**: was our #61 port restart-style (statistics-discarding)? If yes, the 45.9% kill-list entry is confounded [R8] — annotate before any publication.
- Chronicle fixes: Kao TAAI-2022 narrows the Gumbel+chance claim; Charlesworth/"CatAnalysis"/Deep-Catan/HexMachina corrections; "AB5 weakest" re-derived or dropped; Willemsen cite fix; c_scale framing (mechanism, not constant; cite mctx #66/#108). [R7][R8]
- **Repo hygiene** (~2 days): track the H2H gate tooling in git on B200 [US audit]; reproducibility is the price of the novelty claims.
- **Benchmark spec + finish line** doc: 2p-no-trade 10VP, catanatron @ pinned version, wall-clock budget, opponent set, ≥1000 paired games, staged finish line (a) certified vs all catanatron bots → (b) colonist.io-rules + human trials; 4p+trade extension cost audit. [R2][R3][R7][R8]

### Step A8 (day 2, hours) — Verifications that fork later work
- Steal-observability in trajectories → exact deduction tracker vs posterior tracker (§4.6 fork) [ME].
- Do shards bank unmasked hidden-state labels? → belief aux heads train on existing data or need regeneration [R4-claim, verify].
- **Completed:** the built value path was verified as the predeclared 33-bin
  HL-Gauss formulation, tested against the matched scalar control, and rejected
  for this wave by A0. Any future formulation is a new experiment, not a
  conversion inside this protocol.
- Train-time symmetry-augment: actually on in production? (f74/#91 machinery exists [US].)
- Read catanatron_value source: does it actually track cards? (1 hour; adjudicates a 3-review assertion.)

**Phase A exit criteria:** generator promoted under new gate; window A/B running; WHR + diagnostics one-pagers exist; anchors refreshed; hygiene deltas merged.

---

## 3. PHASE B — "Fix the value plumbing + change the data distribution" (Weeks 1–2; two concurrent lanes)

### Training lane (B200)

**B1 — Reanalyze-lite (V0, the defect fix) [R7][R8]**
- Batch-forward the generator champion over the stored window's states (the lr≈0 probe infrastructure — `train_bc --lr 1e-12 --max-steps 1` machinery — is ~80% of this [US]); overwrite the λ-blend's v-component in target_scores; retrain one dose champion-init; gate.
- Reanalyzer-net choice: start with current champion + anchor tripwire; if drift telemetry is ambiguous, switch to lagged/EMA net (Kumar mechanism argues lagged [R8]).
- **Decision:** anchor moves / gate >52% vs same-data control → (a) schedule full root-search reanalyze (n=16 fresh searches over stored states, on explicitly allocated between-wave compute), (b) scope the **banked 32.6M-row corpus** value-only pass (~1 fleet-day of forwards) and mix ~20% into the window [R8].

**B2 — Value-head tournament (V1) [closed for this wave by A0] [R2][R3][R4][R6][R7][R8]**
- Execution result: the exact scalar control reproduced and the 33-bin
  HL-Gauss arm failed the primary stability gate.  A1 therefore trains one
  scalar dose on the fresh mixed window; the historical proposal below is
  retained as hypothesis provenance, not a runnable instruction.
- Historical hypothesis tested: scalar MSE versus 33-bin HL-Gauss with identical
  init/corpus/game split/steps and fresh Adam.  The scalar failure reproduced;
  HL-Gauss was less stable, so the tested formulation is rejected for this wave.
- The bounded historical 87.85M stress is closed.  Categorical value may return
  only as a new, separately predeclared mechanism—not as a bin/sigma retune.

**B3 — V2 + knobs**
- Per-game value-loss weighting + forced-row downweight; LR 0.5×/2× flywheel arms [R8]; EMA/SWA checkpoint-averaging smoothing test [R4].

### Production generation lane (external to the bounded pre-wave R&D process)

**B4 — Wire the pool [R1–R8 unanimous]**
- Mix: **75–80% producer self-play / 10–15% past champions / 5–10% hard negatives** + **5% catanatron_value exploiter games** (cross-engine lockstep exists [US]; OUR search targets; own-side decision rows only [R6]).
- Simple fixed mix first; PFSP f_hard(x)=(1−x)^p weighting when per-opponent telemetry saturates (>90% vs some members) [R3][R7][ME].
- Per-opponent telemetry: win rate, KL, entropy, value calibration, separately.

**B5 — RGSC restarts [R1][R8]**
- Revive f71 (extraction + bit-exact replay built [US]); **upgrade sampling to ranking-based regret prioritization** (rlglab/rgsc reference).
- Archive: high search-vs-prior KL, high root-value variance, robber/dev swings, external losses, unstable-symmetry openings. 10% of games initially (§9-C3 ladder to 25–40%); held-out high-regret suite never trained on; value-only-rows fallback if full-game play degrades.

**B6 — Denoise → re-tune → crank (the SNR remedy, strict order) [R8]**
1. Land Rust featurize → **root 12-symmetry averaging ON** at roots wider than ~20 actions, in generation AND gates (cost ~+12–18% leaf, or ~0 via token-permutation D6 transform [R8]).
2. **Re-grid {c_scale 0.03–0.3} × {D1 on/off}** at the new noise floor (~170 games/arm; `ablate_search_calibration.py` exists [US]). Expect the optimum to move UP.
3. Winner beats cs=0.03 H2H → new production search config.
4. Then **n_full 64→96/128** globally. Test `256` only at `>=40`-legal-action opening/wide roots, independently from D6's `>=20` gate, and force those selected roots to use the full budget so playout-cap randomization cannot silently turn them into n_fast rows. Test `p_full 0.25→0.4` as the next single-dose arm. Kill: gate flat AND measured cost >1.6× (up to 1.8× only with a clear strength margin) [R7].
- Plus: policy-surprise weighting in the loader; late-game temperature A/B (small temp to ~decision 150) [R8].

### Eval lane (continuous, cheap hardware)

**B7 — Population arena + neutral harness**
- All-pairs cross-play, last 8–12 nets + v3a + catanatron bots + raw policies, n=8, few hundred games/pair (bounded separately allocated compute) [R6][R8]; Nash-averaged rating + WHR integration; sims-ladder (n=8/16/32/64/128) per champion [R2].
- **Error atlas** over arena games (per-phase/decision-type loss attribution vs catanatron_value) + **disambiguation-factor measurement** (one afternoon, replay tooling) [R1][R7].
- **Neutral-harness port**: run the definitive 1000-game VF/AB3 matches inside catanatron's own engine (CPU fleet, days) — the number the outside world judges [R8].

**Phase B exit criteria:** A0 has excluded the tested HL-Gauss formulation;
reanalyze-lite has an isolated verdict; pool + restarts are live in the mix; the
S1-S3-selected search config is in production; arena + atlas + df are running;
and two flywheel turns under the new gate are complete. For the current bounded
pre-wave slice, “done” stops earlier: A0 plus S1-S3 are adjudicated and the A1
contract is sealed/rendered for the data-lane owner, with no fleet launch.

---

## 4. PHASE C — "Structural upgrades" (Weeks 3–4)

- **C1 — Deduction tracker** (exact if steal-observable, else posterior): running-count features → net inputs (near-zero-init) + planner chance spectra + aux heads (weight ≈0.25, true-state labels legal) [R1][R2][R4][ME]. Gate ± deduction; inspect robber/dev buckets. Contested probabilistic half stays behind: df number + error-atlas evidence + "within ~10 Elo and stalled" trigger [R7].
- **C2 — Aux package**: existing heads to nonzero weight (0.02–0.1); TD-horizon heads on **realized** outcomes [§9-C6-rev]; V4 uncertainty head (stop-grad) → **backup-side weighting with a cap** (KataGo: weight=min(cap, a·err^b), start a=0.25, exp=1.0) + D2 retry with closed-form James-Stein λ*=v²/(s²+v²) [R7][R8].
- **C3 — Exploitability probe** (3–5 GPU-days): small adversary net, self-play vs FROZEN champion [R8/Wang methodology]. Exploit >70% ⇒ pool % up + R-NaD-style regularization considered; none ⇒ belief-state question closed for this phase.
- **C4 — Search target hygiene**: root candidate cap (top 16–24, symmetry-diverse) ≈ Gumbel policy-target pruning; forced-playouts analog [R1][R3][R8].
- **C5 — Adjacency/graph-distance bias** deferred until it has a runnable,
  tested implementation and its own causal arm; it is not folded into another retrain.
- **C6 — Full selective reanalyze** (if B1 moved the anchor): n=16–256 fresh searches over high-KL/wide/uncertain stored states, embarrassingly parallel between waves [R7][R8].
- **C7 — Gateless-EMA pilot** *only if* two consecutive clean promoted turns AND V1+V2 landed: fleet pulls θ_EMA (β sweep around 0.995), gate demoted to async tripwire [§9-C4; R4 staged; "Survive or Collapse" says keep a gate — the tripwire stays].

**Phase C exit criteria:** deduction gate result; aux/uncertainty heads in production recipe; belief question closed or escalated on evidence; search targets cleaned; reanalyze at scale or explicitly shelved.

---

## 5. PHASE D — "Spend capacity" (Month 2+, gated on ≥10M fresh diverse rows + a promoted stable 35M candidate)

- **D1 — 80–100M net**: fresh-data budget sized per Neumann & Gros; check endgame over-representation first (inverse-scaling warning); retain the promoted 35M objective (currently scalar), one-dose discipline, and VISA symmetry hard-negatives [R8]. Production scale requires a fresh, equal-exposure, multi-seed A/B against that unchanged 35M baseline.
- **D2 — Architecture v2 rerun**: action-target cross-attention ("settlement action sees node token; road action sees edge token" — AlphaGateau precedent [R6]) as the isolated arm; protocol: ≥2 deterministic module seeds, new-module LR multiplier, policy-only warmup, value head frozen/low-LR, equal data + wall-clock vs control [R1][R2][R8]. Graph-distance relative bias is a separate later arm only after a tested implementation exists; full D6 equivariance stays skipped.
- **D3 — Engineering** (when search is next opened): eval server / batched leaves, MCGS cross-move subtree reuse, speculative inter-decision parallelism (~2× fleet-equivalent, throughput-only) [R7]; typed configs + config-hash registry complete by here (science-corruption vector closed) [R1][US].

---

## 6. Decision-gate summary (what kills or promotes what)

| Gate | Pass ⇒ | Fail ⇒ |
|---|---|---|
| A2 confirmation SPRT (−10/+15) | promote gen-4 | hold; investigate candidate choice |
| A0/ext panel ≥48% | parity contingency: power > levers; neutral-harness sprint | gap real: belief/exploiter priority up |
| A3 window A/B internal | keep promoting | producer window worse ⇒ rollback generator |
| A3/ext two consecutive declines | — | AUTO-REVERT + pool % up, linear promotion paused |
| A5 diagnostics | weights Phase B/C mix among (A)/(B)/(C) | n/a (information-only) |
| B1 reanalyze-lite anchor/gate | full reanalyze + banked-corpus pass | λ-fix still ships (defect), reanalyze deprioritized |
| B2/A0 exact 3-epoch probe | **observed:** scalar failure reproduced; tested HL formulation rejected | run fresh one-dose scalar A1; C0 remains closed |
| B6 re-grid: cs>0.03 arm wins H2H | new production search config | denoising insufficient ⇒ mechanism-A weight down, B/C up |
| B6 global n128 | adopt only at H1 with attributable cost <1.6× (up to 1.8× for a clear margin); then screen adaptive n256 at >=40-action roots | kill if flat/inferior or above the cost bound |
| C3 exploit >70% found | pool↑, R-NaD considered, belief half (b) reopened | belief question closed this phase |
| C7 preconditions | EMA-pull pilot | stay gated |
| D1 phase-distribution check | scale | fix data mix first |

## 7. Resource map (current bounded pre-wave program)

- **B200 host (2 GPUs)**: bounded A0/S1-S3 training, search, adjudication,
  and software probes only.
- **24-GPU production data lane (six four-GPU hosts)**: owned by the separately
  sealed A1 contract and its synchronized seed ledger. This R&D lane may render
  the handoff but may not launch the wave.
- **Historical A100/Modal topology:** retained in prior run records only; it is
  not the current launch authority.
- **CPU fleet**: neutral-harness catanatron panels, WHR, error atlas, df measurement.
- Standing jobs are allowed only when they do not interfere with the bounded
  B200 critical path.

## 8-pre. ENGINEERING LEDGER — build vs reuse vs flag-flip (cited)

The build-and-shelve audit [R7] cuts both ways: a lot of this roadmap is NOT new code. Rule: **cite the existing artifact before writing a line.** Sorted by effort.

### Tier 0 — FLAG-FLIPS / CONFIG ONLY (<1 day each; zero new code)
| Item | Existing artifact [US] | What actually changes |
|---|---|---|
| Gate re-spec −10/+15 | `sprt_gate.py` (pentanomial + extension machinery, bit-exact recombination verified round-11) | elo0/elo1/α/β params + a second "certification" config |
| D1 noise-floor ON | task #67, flag-gated, dormant | flag |
| Aux heads to nonzero weight | task #63 (Catan-native aux heads, built) | loss-weight config |
| Value-loss weight 0.25–0.5, value-LR 0.3× | `train_bc.py` (has per-loss weights; verify LR-group split exists, else small patch) | config (+possible ~50-line optimizer-group patch) |
| Late-game temperature A/B | generation config (temp window param exists — T=1.0→argmax@90) | param |
| Promotion runbook | registry promote + `runs/CURRENT_CHAMPION` pointers + feed_config ckpt bump (documented runbook) | execute it |
| LR 0.5×/2× flywheel arms | `continuous_flywheel.py` round machinery | config arms |
| Anchor = tripwire-only | flywheel config (`--anchor-corpus`/`--anchor-holdout-ranges` flags exist) | policy change |

### Tier 1 — FAST (1–3 days; existing code + glue)
| Item | Reuse | New |
|---|---|---|
| Confirmation gate + window A/B | `sprt_gate.py`, generation launch scripts, seed ledger, feed daemon (`flywheel_feed_daemon.py`) | ops only |
| WHR fit | all gate/panel JSONs on disk; `whr` pip package (Coulom impl exists publicly) | ~200-line ingest script |
| Diagnostics bundle | SNR probe = run existing search twice w/ different search seeds; rollout-doubling = existing H2H tool at n=64/128; noise-vs-spread = D3 opening panel (#69) + f74 symmetry infra; diversity = corpus scan | ~1–2 days glue |
| Anchor refresh | reserved `.valonly` seed machinery + anchor-corpus build path (anchor_r7 was built this way) | script reuse |
| **Reanalyze-lite v1** | lr≈0 probe infra (`train_bc --lr 1e-12 --max-steps 1`) = "80% of this" [R7]; memmap corpus tooling (#66, `build_memmap_corpus.py`) | batch-forward + v-component column rewrite (~2–3 days) |
| **HL-Gauss conversion** | implemented and tested; exact A0 formulation rejected for this wave | reopen only as a separately predeclared future formulation |
| RGSC prioritization | f71-regret-restarts branch (cc70769+724f1c7): extract/reconstruct/generate + bit-exact replay (game_seed^0xA17E); generation unblocked since public_observation landed | ranking-based regret sampler (~1–2 days) on top; MERGE the branch, don't rewrite |
| Policy-surprise weighting | memmap loader (#66/#94 ConcatMemmapCorpus) | per-row weight column + sampler (~1–2 days) |
| Game-level validation splits | **largely already exists**: `--validation-game-seed-ranges` IS a game-level split (the round-11 leak was the random re-split path, since fixed) | verify it's the only path; done |
| EMA/SWA checkpoint averaging | checkpoints on disk | ~50-line averaging script |
| λ-vs-gen2A match, pair-correlation, hygiene | existing gate tool + gate JSONs | ops + doc edits |
| Commit debt | `modal_gumbel_factory_gpu.py` exists ONLY in local mirror `~/catan-zero-gpu`, uncommitted; H2H gate tool untracked in git on B200 | `git add` — this is Tier-0 effort, Tier-1 importance |

### Tier 2 — MEDIUM (3–7 days each; real code, but building on cited foundations)
| Item | Reuse | New |
|---|---|---|
| Opponent-pool wiring | generator (`generate_gumbel_selfplay_data.py`) + checkpoint registry + shard schema (has prior_policy/provenance columns #47/#87) | opponent-checkpoint loading, mix sampling, per-opponent shard tags, own-side-row filter [R6] |
| Exploiter lane (vs catanatron_value) | **cross-engine lockstep exists** [R7/R8 confirm]; catanatron_rs wheel 0.1.2 (#82) | catanatron-bot-as-opponent inside our generator + search-targets-our-side-only plumbing |
| Neutral-harness port | catanatron's own engine + our ONNX/torch checkpoint | our-net-as-catanatron-Player adapter (featurize from their state) — the credibility item [R8] |
| Population arena + Nash | H2H tools (`h2h_postrepair_aggregate.py` w/ dedup+pairing 276f33b) + WHR from Tier 1 | orchestrator + payoff matrix + Nash-averaging solver (small scipy job) |
| Deduction tracker | engine event stream; verify steal-observability (A8) | running-count fold + feature plumbing + aux-head labels (small-medium if exact; posterior version bigger) |
| Per-game value weighting | loader | per-game normalization in loss (medium — touches batch assembly) |
| Uncertainty/value-error head + backup weighting | aux-head scaffolding (#63); D2 code (#68) for the selection-side variant | error head + backup-weight w/ cap (KataGo constants as defaults) + closed-form JS in D2 |
| Root cap / target pruning | Gumbel search module (`gumbel_chance_mcts.py`) | considered-set cap + π′ support restriction, flag-gated |
| Banked-corpus reanalyze at scale | 32.6M-row memmap corpus (417GB, `runs/memmap_corpus_full`); explicitly allocated between-wave compute | fleet job orchestration for forwards + column rewrite |
| Exploitability probe | our own self-play loop + frozen-checkpoint opponent mode | small-net config + frozen-opponent flag (mostly config if generator supports asymmetric nets — verify) |

### Tier 3 — SLOW (1–3+ weeks; the real engineering projects)
| Item | Status | Notes |
|---|---|---|
| **Rust featurizer (finish it)** | task #81 IN PROGRESS; 20–38× measured on the featurize slice; staged, not landed | **The #1 engineering priority.** Featurize+FFI = 96% of leaf cost (NN is 4%) [US perf model]. Unblocks: symmetry averaging at ~+12–18% (or ~0 via **D6-as-token-permutation on the already-featurized tensor** [R8] — implement this variant), n_full raises, 12-symmetry batching [R3]. Also in #81's quality-gated list: **lazy-chance 65× bug** (loses 21–2 to raw; debug the depth>0 ROLL backup before trusting it) + topology-cache. |
| Typed configs + config-hash registry | #74 done (name-keyed dict + schema version) = the foundation | extend to full train/generate/gate/eval hash registry; kills the CLI-default trap (7+ incidents) |
| Eval server / batched leaves | batch API integrated (#32/#37) per-game | cross-game batching = new architecture; Phase D; profile first |
| Subtree reuse (MCGS-style) | none | Phase D; behind eval server per profiling |
| Cross-attention arch rerun + 90M | v3b checkpoint + arch code exist (benched) | Phase D protocol (multi-seed etc.) |
| 4p+trade extension audit | none | analysis doc, not code — but do the audit in Phase A while it's cheap |

### Hardware inventory (what runs where)
- **B200 host (2× B200)**: bounded R&D for A0/S1-S3, contract validation,
  and learner/search probes. It does not run a production wave or the closed C0
  91M re-probe.
- **24-GPU production data lane (six four-GPU hosts)**: receives only a sealed,
  audited, non-executing handoff after S1-S3 and seed-ledger synchronization.
  The data-lane owner, not this R&D process, performs any eventual launch.
- **Historical A100/Modal resources:** non-authoritative for the current
  pre-wave program.
- **CPU**: neutral-harness catanatron panels (their engine is CPU), WHR, error atlas, df measurement, Nash solve.
- **MPS/data-generation service details:** owned by the production data lane and
  intentionally outside this learner/search R&D boundary.

## 8. What "done" looks like (finish line, staged)

1. **Stage 0 (now measurable):** WHR trajectory + neutral-harness baseline vs catanatron_value with ≥1000-game power.
2. **Stage 1:** SPRT-certified superiority over **every** catanatron bot, 2p no-trade, in *their* harness. ("Strongest 2p no-trade agent; first leak-free, gated, AZ-class Catan system" — the defensible claim [R8].)
3. **Stage 2:** colonist.io-rules variant + invited human match series, pentanomial rigor. (Protocol TBD — open question.)
4. **Stage 3 (scope decision, deliberate):** 4p + trading — architecture audit for the extension is a Phase-A deliverable so this stays cheap to choose later.
