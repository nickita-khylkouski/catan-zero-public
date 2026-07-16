# Why self-play stalled: evidence-based diagnosis

**Date:** 2026-07-15
**Evidence:** `catan_rl_evidence_bundle_20260715/` (sealed panel reports, July-15 sweep receipts, 5 production NPZ shards).
Analysis script: `catan_rl_evidence_bundle_20260715/analyze_shards.py` — rerun it on more shards to confirm at scale (this sample: 17 games / 1,940 rows / 1,026 full-search decisions).

---

## 1. First, the claim ledger — both of you are partly right

**"The teacher springboard worked"** — confirmed. Legacy July-1 scoreboards show the BC-era models at ~30–35% vs `catanatron_value` and ~37–42% vs AB3 (e.g. `score_step1_...`: 63/200 value, 80/200 AB3). That's teacher-level play, as expected from imitation.

**"Self-play then didn't improve"** — *half* right, and the half matters:

- Self-play **did** work for three generations (gen1→gen3, +49/+49/+33 Elo internal; external vs value bot: 35.5% → 45.7%) and the July short-dose recovery added a verified +10.9pp external (376–392 vs f7's 292–476, p=1.7e-5, receipts in `01_external_panels/`).
- What's true **now**: the July-15 four-arm sweep receipts show every new candidate beating the *old* f7 baseline (52.7–59.0%) but **tying the actual incumbent v5** (A: 49.6%, B: 46.5%, C: 51.6%/47.7%, D: 49.2/48.4%; step-24 recovery arm: 52.3% vs f7, 49.0% vs v5 — all `continue` verdicts). The flywheel is *recovering to* v5, not compounding past it.

**"The model already understands the rules because it trained on Catanatron games"** — partly right on the *data*, and I'll be honest where the fresh shards vindicate him: in the 5 production shards, longest-road length is populated (87% of rows), event history is live (99%), and the deduction features are active (the early-game zeros are legitimately empty opponent hands). The historical corpora were broken; current generation is much better. **But** the turn-state gaps are structural, not learnable: `has_played_development_card_this_turn` is still absent from the 31-slot player token (the env emits it; the featurizer drops it), and there's no bought-this-turn or Road-Building-progress state. No amount of Catanatron data teaches a model to distinguish two states that produce identical inputs.

## 2. The mechanism: the improvement signal has fallen below the value-noise floor

Self-play improvement in AlphaZero = distilling (search policy − prior policy). That difference is only signal if the value function can actually rank the actions being compared. Three measurements from the production shards say it mostly can't anymore:

**(a) The value head is nearly blind in the early game — exactly where Catan is decided.**
Correlation of stored root value with the actual game outcome, by decision index:

| Game stage | n | corr(root value, outcome) |
|---|---:|---:|
| decisions < 30 (openings/first builds) | 189 | **0.32** |
| 30–80 (early development) | 177 | **0.24** |
| 80–200 (midgame) | 432 | 0.67 |
| > 200 (endgame) | 228 | 0.81 |

It knows who's winning once it's obvious. In a 2p race decided by opening placement and early engine-building, that's backwards.

**(b) The decisions left to improve are near-ties relative to that noise.**
Across 1,026 full-search roots (n=128, D6-averaged): median completed-Q spread between the best and worst legal action is **0.044** on the [−1,1] scale — right at the project's own measured value-noise floor (orientation noise σ≈0.049 *after* the 3.3× D6 denoise). Median root width 12, ~10 visits per action.

**(c) Therefore the "policy improvement" targets are mostly re-distilled noise.**
Search changes the prior's argmax on **50.7%** of policy-active rows — sounds like lots of learning signal. But among those flips, the Q-margin justifying the flip is:

- median **0.015**
- **76% below 0.05** (the noise floor)
- 88% below 0.10

So roughly three quarters of what self-play currently "teaches" the policy is the value function's coin flips. Distilling coin flips moves the policy sideways, not up — which is precisely what the sweep receipts show: candidates shuffle around v5 ±3% and never break away.

**This is why "a crap ton of games" stopped working.** Volume averages away *gradient* noise, but it cannot manufacture a preference the evaluator doesn't have. When E[search target] ≈ prior at the decisions that decide games, more games converge ever more precisely to "no change." Signal per game isn't a constant you multiply by game count — it decays as the policy approaches the value function's resolution limit. The springboard worked because imitation doesn't need a value function; gens 1–3 worked because the early value head had huge, easy errors; it stalled when the remaining errors became finer than the value head can see.

## 3. Corroborating detail from their own configs

The July-15 learner recipe (`03_learner_sweeps/coherent_active_policy_four_arm/campaign.json`) has **`aux_subgoal_loss_weight: 0.0` and `final_vp_loss_weight: 0.0`** — every auxiliary head that would give the value trunk dense early-game gradient (final VP margin, VP-in-N-turns, longest-road/largest-army attainment, next-settlement) is switched off, even though **the targets are already computed and stored in every production shard** (`aux_*` arrays are right there in the NPZ). The value function is being asked to learn 250-decision credit assignment from a single ±1 bit per game, and the dense supervision that exists is at weight zero.

## 4. What this means: ranked fixes for self-play

The objective "fix self-play" decomposes into **raising the signal** (value resolution) and **not wasting the budget** (search/data allocation). Ranked by expected value per effort:

1. **Turn on the auxiliary value targets at small weight** (final-VP margin first, then VP-in-N). Zero generation cost — targets are already in every shard. This is the standard KataGo move for exactly this pathology, it was reviewer-endorsed and planned, and it directly attacks the early-game blindness in table (a). Test as one 524k-dose arm vs control.
2. **Ship the input-correctness fixes** (see `CATAN_ZERO_INPUT_FIXES_AND_EXAMS_PROPOSAL.md`). State-aliasing (played-dev-this-turn, bought-this-turn, road-building state, robber-discounted production) is *irreducible* value noise — the same position-pair gets averaged forever no matter how much data. Lowers the noise floor in (b)/(c).
3. **Reallocate search budget to where the signal is.** In 2p, resource hands are exactly deducible; only dev-card identities are hidden. The current 4-particles × 32 sims spends 4× budget determinizing mostly-known state. Test dev-card-only determinization at P1×128 / P2×64 — effectively 2–4× deeper search per decision at the same cost, which widens Q-margins relative to noise.
4. **Give the value function informative games, not just more games.** Self vs self at parity produces maximally uninformative outcomes (every game ≈ coin flip conditional on early state). The opponent-pool/exploiter machinery exists in the repo but the current wave is three fixed self-source categories. Mixing in external-bot seats (value/AB) and prior champions creates positions where being wrong is *punished*, i.e., outcomes that carry gradient. This is their own open hypothesis H8 — the data now supports prioritizing it.
5. **Reanalysis of early-game states** (machinery exists: `generate_rust_mcts_reanalysis`): refresh stale early-decision value targets with the current net + deeper search, concentrating on decisions < 80 where corr is 0.24–0.32.
6. **Measure with exams** so the next generation shows *where* it improved (openings vs robber vs endgame) instead of a single win rate — otherwise fixes 1–5 can't be attributed. (Proposal doc, §3.)

### What NOT to conclude

- Not "self-play is fundamentally broken" — it produced 3 real generations and the short-dose recovery.
- Not "we need more games/GPUs" — the receipts show the current operator converging to v5-parity; scale multiplies a near-zero expected improvement.
- Not "the model can't learn rules from data" — fresh shards show the featurizer fixes landing. The remaining gaps are absent *inputs*, a different claim.

### Caveat

Shard-level findings rest on 5 shards / 17 games / ~1,000 full-search decisions from the bundle. The patterns are strong and internally consistent (and match the project's own SNR notes), but rerun `analyze_shards.py` across a full wave's shards before treating the exact percentages as load-bearing. The sweep/panel results are 256–1,536-game sealed receipts and stand on their own.
