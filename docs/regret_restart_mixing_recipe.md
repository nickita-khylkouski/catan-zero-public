# High-regret restart self-play: mixing recipe (task #64)

Restart training/data-generation from archived states where the agent's own
evaluation diverged from outcomes, instead of always from the initial board.
Basis: **Go-Exploit** (arXiv 2302.12359) and **Regret-Guided Search Control**
(arXiv 2602.20809). Caution from **DAGS** (arXiv 2605.14379): intermediate-start
training can bias learning in imperfect-information games — mitigated here with
explicit `start_mode` flags and separate-metrics discipline.

## Pipeline

1. `tools/extract_regret_states.py` streams a self-play shard corpus (never
   concatenates — the raw corpus is ~48M rows / 680 GB), scores every
   non-forced row, and writes a top-K regret manifest (`.npz`) + a
   `.summary.json`.
2. `tools/reconstruct_state.py` replays a manifest row's `game_seed` +
   recorded action sequence to its `decision_index`, returning the live Rust
   game (round-trip-verified against the stored featurisation).
3. `tools/generate_restart_selfplay.py` reconstructs selected states and plays
   raw-policy continuations (both seats raw policy, value rows only,
   `policy_weight_multiplier=0`), tagging every row with `start_mode`.

## Regret score

Additive, components normalised to ≈[0,1] then weighted (all weights are CLI
flags on the extractor):

| component | meaning | source |
|---|---|---|
| `value_surprise` | `\|value_or_q(taken) − z\|` | searched shards: taken action's `target_scores` (Q on the [−1,1] outcome scale); raw shards: a value-head pass over stored entity tokens. Defined only for clean-terminal rows (`z = ±1`). |
| `phase_bonus` | opening placement highest, then robber/dev | row `phase` |
| `legal_count_bonus` | branching / decision richness | legal-action count / 54 |
| `kl_disagreement` | `KL(target_policy ‖ prior_policy)` — search overruled the prior | searched shards only (0 on raw, where target==prior) |
| `argmax_mismatch_lost` | search picked a different action than the prior **and** the acting player lost | searched shards only |

`z` matches `train_bc._value_targets`: `+1` if `winner == acting player`, `−1`
otherwise, for clean terminal rows only (`winner != ""` and not truncated).

## Start-mode mixing (default recipe)

| bucket | share | `start_mode` | rationale |
|---|---|---|---|
| normal starts | **60%** | `normal` | on-distribution anchor; prevents forgetting the opening and biasing toward mid-game states (DAGS caution). |
| high-regret openings | **20%** | `archived_public_state` | placement blowouts were 74.6% of search losses — the highest-leverage states to resample. |
| robber / dev (chance-heavy) states | **10%** | `archived_public_state` | high-variance chance nodes where the value head is least calibrated. |
| random archived | **10%** | `archived_public_state` | smoothing — samples archived states uniformly regardless of regret score, so training does not overfit the regret metric's own blind spots. |

Weights are CLI flags (`--normal-fraction`, `--opening-fraction`,
`--robber-dev-fraction`, `--random-archived-fraction`); `plan_start_mix`
rounds the remainder into the normal bucket.

## Temperature handling for the continuation

Each restart continuation temperature-samples the first
`--restart-temperature-decisions` decisions (default 20) from the raw priors
for branch diversity, then plays argmax — the same absolute-count schedule as
`raw_selfplay`, counted from the restart point (not the archived depth).

## Reproducibility & the hidden-information caveat

- Reconstruction replays the archived game's **true history** using that game's
  own chance stream (`game_seed ^ 0xA17E`), so a restart state is a
  **legitimate reachable public state**, not an omniscient fabrication. The
  continuation keeps drawing from the same chance stream, so a branched game is
  reproducible from `(archived_game_seed, archived_decision_index,
  restart_select_seed)`.
- Every restart row still carries `start_mode` (+ `start_bucket`,
  `archived_game_seed`, `archived_decision_index`, `restart_select_seed`) so
  downstream training keeps **separate metrics** for intermediate-start vs
  normal-start data and can down-weight or ablate the restart rows per the DAGS
  caution.

## Public-observation / hidden-info dependency (task #71)

Restart data must be produced under the public-observation featurization fix
(no omniscient hidden-info leak). The dependency differs by stage:

- **Restart GENERATION** runs the evaluator **online**, so it must launch with
  `public_observation=True` — the produced rows are clean by construction. This
  stage is gated behind the #71 fix; do not generate restart shards before it
  lands.
- **Regret EXTRACTION** (already run on the pilot + 10% raw sample) scored raw
  shards' `value_surprise` from a value-head pass over the corpus's **stored
  omniscient tokens**. This is acceptable for state **SELECTION**: the selected
  states are legitimate reconstructions of real reachable public states (the
  bit-exact round-trip proves this) — only the *ranking signal* v(s) is
  omniscient-flavored, not the states themselves. If a post-#71 (v3) checkpoint
  materially changes which states rank highest, **re-rank** by re-running the
  extractor's value pass under `public_observation=True`; the reconstruction /
  generation code needs no change.

Net: the reconstruction and generation paths are already leak-free; only the
extraction *score* inherits the corpus featurization, and it is cheap to
recompute.

## Opening dominance is controlled at generation, not in the score

The regret score *enriches* openings (2.3–3.3x over their candidate base rate
in the top-K) but does not make them the plurality — ordinary turns and robber
states are far more numerous. Do **not** inflate `--w-phase-bonus` (kept at
0.4) to force opening dominance: the 60/20/10/10 start mix already sets the
opening share at generation time, which is the cleaner control point than
skewing the ranking. The score's job is to surface the *highest-regret*
openings within the 20% opening budget, not to win a headcount against
PLAY_TURN.

## Archive sampling: uniform vs RGSC (task CAT-43)

`tools/generate_restart_selfplay.py --restart-sampling {uniform,rgsc}` (default
`uniform`, regression-safe) controls how each start-mode bucket selects rows
from the regret manifest:

- **`uniform`** (pre-CAT-43 behaviour): opening/robber_dev take the
  highest-scoring rows in that phase (the manifest is already score-sorted
  desc, so this is a deterministic top-slice); `random_archived` samples
  uniformly across the whole manifest (smoothing, no regret bias).
- **`rgsc`**: every bucket instead samples via the ranking-based
  regret-weighted rule from *Regret-Guided Search Control* (Tsai et al.,
  ICLR 2026, [github.com/rlglab/rgsc](https://github.com/rlglab/rgsc)) --
  the one lever in the literature review with direct evidence of un-sticking
  a converged AlphaZero-style system (69.3% -> 78.2% win rate vs KataGo on a
  nearly-converged 9x9 Go model, where both vanilla AlphaZero and Go-Exploit's
  uniform sampling flatlined). Concretely (`tools/rgsc_sampler.py`), each
  candidate gets sampling probability `P(s_i) = R(s_i)^(1/tau) /
  sum_j R(s_j)^(1/tau)` (the paper's Prioritized Regret Buffer rule, Section
  3.3; `tau` = `--rgsc-temperature`, default 0.1, matching the reference
  repo's `env_buf_sampling_temperature`), drawn without replacement via
  Efraimidis-Spirakis weighted reservoir sampling. `R(s)` is this codebase's
  existing `regret_common.score_shard` additive regret_score -- we don't
  train the paper's separate regret ranking network, since that exists to
  make a hard, non-stationary, imbalanced *online* regression target
  tractable, and our regret_score is already a deterministic offline
  extraction score with none of those pathologies.

## Held-out high-regret suite (task CAT-43)

`--holdout-fraction` (default 0.0) reserves a deterministic fraction of the
manifest's states as a frozen evaluation suite that generation never draws
from -- the "held-out high-regret suite never trained on" from the roadmap
(§9 error atlas). The split is a stable hash of `(game_seed, decision_index,
--holdout-seed)` -- not a random shuffle -- so re-running generation with the
same seed always reserves the exact same states without persisting row
indices anywhere. Held-out rows are written to
`<out-dir>/holdout_manifest.npz` and excluded from all three archived buckets
regardless of `--restart-sampling` mode.

## `correct_rust_chance_spectra`

Reconstruction must use the same chance-spectrum correction the generator used
(A19/A20 verified Rust engine bugs). Manifests predating config-provenance
don't record it, so `reconstruct_state`'s round-trip harness resolves it
empirically: it tries `True` then `False` and keeps whichever reproduces the
stored featurisation exactly.
