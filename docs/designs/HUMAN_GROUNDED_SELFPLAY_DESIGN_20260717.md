# Human-grounded self-play: design for the next program phase

**Author:** Michael (with Claude-assisted analysis and tooling)
**Date:** 2026-07-17
**Companions:** `docs/reviews/SELFPLAY_SNR_EVIDENCE_REVIEW_20260716.md` (why self-play stalled), `tools/play_flag_ui/` (the new play-and-flag interface, shipped with this doc).

## 0. Premise

The evidence review established that the current loop's improvement targets sit at or below the value function's noise floor, and the shrinking optimal dose (524k draws → 12 optimizer steps) is the arithmetic signature of that: hygiene preserves signal, but the loop is running out of signal to preserve. This document proposes where new signal comes from. Three sources, in increasing order of novelty:

1. **dense engine-verified value labels** (rollout continuations — pure compute);
2. **human ground truth** (Michael's play, both as SFT data and as labeled exams);
3. **a self-generating curriculum** (an adaptation of Self-Guided Self-Play, arXiv:2604.20209, to a domain with a perfect verifier).

Nothing here replaces the existing engine, search, trainer, or promotion machinery — every component maps onto tools already in this repo.

## 1. The verifier (the foundation everything rests on)

For any position, the **verifier** produces a ground-truth answer with no human involvement: run the current net with a much larger search budget (n1024+), then play each top candidate action out to game completion K times with seeded dice (K≈32–64). The action with the best empirical win rate is the answer; the win-rate gap to the runner-up is the label's confidence and the cost-of-error. Positions where the gap is below a threshold are **near-ties: unanswerable, discarded, never trained on**.

Rules that keep it honest:

- the verifier must always spend ≫ the student's compute (a peer is not a teacher);
- when the student approaches verifier agreement, raise verifier compute — never train against a matched grader (that distills noise);
- verifier labels are cached by `(seed, decision_index, verifier_config_hash)` and re-used.

This is precisely the "targeted multi-seed Monte Carlo at openings/turn boundaries" the July-16 root audit called for — productized. It is where the H100 fleet's capacity should go: every GPU-hour lands on a position selected because it carries signal, instead of on game #40,000 of self-vs-self parity.

## 2. Human grounding: the play-and-flag interface

`tools/play_flag_ui/` (server + single-page board UI, stdlib + networkx only):

```bash
cd tools/play_flag_ui && python3.11 -m venv .venv && .venv/bin/pip install networkx
.venv/bin/python server.py   # → http://localhost:8765
```

- 2-player, no-domestic-trade, 10 VP — the production track — against Catanatron ValueFunction / AlphaBeta-2 / AlphaBeta-3 (a neural-champion adapter is the obvious next step: point `make_bot` at an EvalServer client).
- Every game is **seeded and fully traced** (`data/games/*.jsonl`): any moment converts to `(seed, decision_index)` — the same identity scheme as the shard/reconstruction pipeline — so flagged moments are mechanically replayable, verifiable, and trainable.
- **Flag hotkey (F):** mid-game, tag the bot's mistake with a category + short reason → `data/flags.jsonl`. Categories are the working weakness taxonomy: opening_placement, second_placement, robber_targeting, knight_timing, dev_timing, maritime_port_use, longest_road_race, largest_army_race, discard_choice, endgame_race, leader_blocking.
- Opponent's exact hand is redacted server-side (public counts only), so human games are played under honest information.
- Undo replays deterministically from the seed (bot decisions are re-run because Catanatron bots consume the global RNG stream while evaluating candidate moves — recorded knowledge for anyone touching replay).

**Protocol:** 15–20 games by a strong player, flags on every wince. Output: (a) a grounded bug list in Catan language, (b) 30–50 labeled weakness positions, (c) per-category counts that seed the weakness ledger (§4). Then run the verifier over the flagged positions and measure verifier-vs-human agreement — this is the calibration experiment that validates (or falsifies) the whole architecture for ~zero cost before any training run depends on it.

## 3. SFT on human replays

Import the player's own permissioned game replays (the repo's `data/colonist.py` pathway; own games only, no stealth automation — same line the README already draws), split into decision points on the player's turns, and fine-tune with the player's move as target. Expectations set honestly: top-1 human-move match plateaus in the 60–75% range because many Catan decisions are near-ties — evaluate with graded acceptable-set matching, not exact match. Value: strategy priors self-play demonstrably does not discover (port usage, trade sense, tempo), plus a free seed set of labeled exam positions. Decide explicitly which track this serves — human data is most valuable for the trading game, which is where the 4p benchmark ultimately lives.

## 4. The curriculum loop (SGS adapted to a verified domain)

Reference: *Scaling Self-Play with Self-Guidance* (arXiv:2604.20209) — Solver/Conjecturer/Guide, where the Conjecturer is rewarded for generating problems the Solver fails, and a Guide prevents the known collapse into hard-but-useless problems. SGS was demonstrated in Lean4, a domain with a perfect verifier. Catan is also a domain with a perfect verifier (§1) — which lets us simplify aggressively:

| SGS role | Catan instantiation |
|---|---|
| Solver | the 35M entity policy (unchanged; LLMs are ~1000× too slow for leaf evaluation) |
| Conjecturer | **a miner, not a generator**: samples candidate positions from real games (self-play, human replays, human-vs-bot flags); v2 adds a perturber that branches a real game at decision k and plays a few plies. Positions are engine-reachable by construction — the "natural/clean" half of SGS's Guide is enforced for free. |
| Guide | **a formula, not a model**: `score = weakness(category) × sweet_spot(pass@8) × verifier_confidence × novelty`. Weakness comes from the per-category ledger (exam scores updated every cycle); sweet-spot rewards pass@8 in (1/8, 5/8) — hard but solvable; verifier confidence gates out near-ties; novelty is embedding distance from the existing curriculum. Nothing learned, nothing hackable, every term inspectable. |
| Verifier | §1. Human labels serve as ground truth only on *real* positions the human actually played or flagged. |

Solver training: GRPO-style on pass@k against verified answers over the curriculum, or simply supervised distillation of verifier answers — start with the latter (it is one config away from the existing learner). The Conjecturer/Guide loop is a data-selection layer over machinery that already exists: `extract_regret_states.py` (mining), `reconstruct_state.py` (replay), `strategic_root_exam.py` (rendering), the eval fleet (verification).

## 5. Exams as promotion instrumentation

Flagged/curated positions with graded labels (best / acceptable / blunder) become per-category exams run on every candidate: report acceptable-rate and blunder-rate per category, next to (not inside) the SPRT gate. This converts "external win rate moved ±2%" into "robber play regressed, openings improved" — the per-category decomposition that would have diagnosed the n128/n256 failure in a day. Advisory for two generations, then discuss wiring blunder-rate as a tripwire.

## 6. Sequencing

1. **Now:** play-and-flag sessions (§2) + verifier-vs-human calibration on the flags. Zero GPU risk, produces the taxonomy, validates the verifier.
2. **Next learner window:** verifier-labeled dense value targets at openings/turn boundaries as a corpus component; one controlled dose arm vs control.
3. **Then:** the mechanical-Guide curriculum sampler over mined positions; exams wired into candidate evaluation.
4. **In parallel (Michael):** replay import + SFT experiment; exam label curation.
5. **Deliberately later:** learned Guide, generative Conjecturer, LLM anything — only if the mechanical versions saturate.
