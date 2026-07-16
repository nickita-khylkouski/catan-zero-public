# Self-Play Training

The current self-play stack is intentionally minimal and local-machine friendly.
It is not the final graph/league/search system yet, but masked PPO is now
implemented and running on remote boxes.

## What Exists

- `RandomPolicy`
- `HeuristicPolicy`
- `OnePlySearchPolicy` with copied-state heuristic rollouts
- `LinearSoftmaxPolicy`
- `NumpyMLPPolicy`
- `TorchPPOPolicy`
- `CatanatronValuePolicy` teacher/baseline
- optional `TorchPPOPolicy --architecture candidate` legal-action scorer
  with structured action features plus learned action-id embeddings
- shared-policy self-play game runner
- heuristic/search bootstrap from self-generated games
- masked policy-gradient updates from self-play outcomes
- one-learner-seat league training against fixed random/heuristic/search opponents
- all-seat shared-policy PPO self-play
- discounted per-player terminal returns
- GAE-style per-player advantages over each seat's own decision sequence
- persistent Adam optimizer state across PPO iterations
- persistent Adam optimizer for supervised teacher warmup
- actor-critic teacher warmup with terminal-return value targets
- soft value-teacher distillation targets over candidate actions
- optional value-teacher imitation anchor during PPO
- checkpoint save/load
- candidate-vs-baseline evaluation with an Elo-difference estimate
- dynamic historical league snapshots during PPO, so a learner can train
  against heuristic/value baselines plus frozen recent versions of itself
- optional old-policy KL regularization inside PPO minibatches via
  `--old-policy-kl-coef`, using the rollout policy's full legal-action
  distribution rather than only the sampled-action PPO ratio
- evaluator progress JSON via `--progress-every`, so long held-out checks can
  be monitored while training continues
- optional value-baseline evaluation for interim warmup/PPO checkpoints via
  `--warmup-checkpoint-eval-value-games` and
  `--checkpoint-eval-value-games`
- a no-human-data training ladder that runs value-teacher warmup, PPO against a
  strong mixed opponent pool, held-out promotion evaluation, and champion
  promotion only when gates pass

## Commands

```bash
.venv/bin/python tools/train_self_play.py \
  --policy linear \
  --teacher search \
  --opponents heuristic \
  --bootstrap-episodes 8 \
  --episodes 16 \
  --eval-games 8 \
  --vps-to-win 3 \
  --max-decisions 300 \
  --search-candidate-limit 24 \
  --search-rollout-decisions 4 \
  --checkpoint runs/self_play/search_bootstrap_linear.npz \
  --report runs/self_play/report.json

.venv/bin/python tools/evaluate_self_play.py \
  --candidate search \
  --opponent random \
  --games 8 \
  --vps-to-win 3 \
  --max-decisions 300 \
  --search-candidate-limit 24 \
  --search-rollout-decisions 4

.venv/bin/python tools/train_ppo.py \
  --architecture candidate \
  --teacher value \
  --warmup-games 24 \
  --warmup-epochs 3 \
  --warmup-value-coef 0.7 \
  --iterations 28 \
  --episodes-per-iteration 4 \
  --opponents self \
  --learner-seats all \
  --ppo-epochs 4 \
  --learning-rate 0.0003 \
  --entropy-coef 0.02 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --anchor-games-per-iteration 1 \
  --anchor-epochs 1 \
  --eval-games 32 \
  --vps-to-win 3 \
  --max-decisions 300 \
  --checkpoint runs/self_play/ppo_self_anchor.pt \
  --report runs/self_play/ppo_self_anchor.json

.venv/bin/python tools/train_ppo.py \
  --init-checkpoint runs/self_play/distilled_candidate.pt \
  --warmup-games 0 \
  --iterations 50 \
  --episodes-per-iteration 8 \
  --opponents mixed \
  --learner-seats one \
  --ppo-epochs 4 \
  --anchor-games-per-iteration 2 \
  --anchor-epochs 1 \
  --eval-games 64 \
  --eval-value-games 16 \
  --vps-to-win 3 \
  --max-decisions 300 \
  --checkpoint runs/self_play/distilled_candidate_ppo.pt \
  --report runs/self_play/distilled_candidate_ppo.json

.venv/bin/python tools/run_self_play_ladder.py \
  --run-dir runs/self_play/nohuman_ladder \
  --champion runs/self_play/nohuman_ladder/champion.pt \
  --cycles 1 \
  --seed 2603 \
  --vps-to-win 3 \
  --max-decisions 300 \
  --hidden-size 512 \
  --warmup-games 32 \
  --warmup-epochs 3 \
  --iterations 20 \
  --episodes-per-iteration 4 \
  --ppo-epochs 4 \
  --checkpoint-every 5 \
  --checkpoint-eval-games 8 \
  --eval-games 24 \
  --promotion-eval-games 24 \
  --min-heuristic-win-rate 0.25 \
  --no-champion-write
```

This is a historical 3-VP diagnostic workflow, so the example explicitly
disables champion writes. The ladder fails closed before training if a
champion-writing run uses anything other than 10 VP, fewer than 50 heuristic
promotion games, or fewer than 50 value-opponent promotion games. For exact
legacy reproduction, `--allow-noncanonical-champion-overwrite` restores the old
mutation behavior, but that result is not a production promotion.

The ladder does not assume a candidate is good because it beat random
opponents; it recommends promotion only after the held-out heuristic/value
gates pass and, when a champion already exists, after the candidate beats the
champion's promotion score. `--no-champion-write` records that recommendation
without creating or replacing the champion.

## Current Result Snapshot

Recent local and box runs produced:

- direct value baseline vs random: 28 wins from 32 games
- direct value baseline vs heuristic: 16 wins from 32 games
- best linear/value-distilled checkpoints so far: below the heuristic baseline
  on held-out seeds
- first one-seat PPO implementation: 18/24 and 16/24 versus random, but only
  1/24 and 4/24 versus heuristic
- first all-seat PPO self-play implementation: 22/32 and 21/32 versus random,
  but only 2/32 and 7/32 versus heuristic, so it is not promoted
- later pre-GAE controls `s905` and `s906` also failed promotion:
  8/32 and 4/32 versus heuristic
- GAE flat-policy runs `s1001`-`s1004` finished below promotion quality:
  `s1001` 7/40 versus heuristic, `s1002` 10/40, `s1003` 8/40, and
  `s1004` 10/40. They are rejected as learned-policy candidates.
- persistent flat GAE runs `s1101` and `s1102` are also rejected:
  `s1101` scored 8/48 versus heuristic and `s1102` scored 14/48 versus
  heuristic. They beat random but do not clear the baseline gate.
- persistent flat GAE run `s1103` is rejected: 35/48 versus random, 9/48
  versus heuristic, and 0/16 versus the direct value teacher.
- flat soft-distillation run `s1212b` is rejected: 43/64 versus random, but
  only 9/64 versus heuristic and 1/16 versus the direct value teacher.
- flat soft-distillation run `s1211b` is rejected: 37/64 versus random, but
  only 8/64 versus heuristic and 1/16 versus the direct value teacher.
- old candidate-scorer distillation runs `s1301` and `s1302` are rejected.
  Their final reports were weak against the heuristic/value gates
  (`s1301`: 0/64 versus heuristic and 0/16 versus value; `s1302`: 5/64
  versus heuristic and 0/16 versus value). Additional fixed-code held-out
  checks scored 5/32 and 11/32 versus heuristic, still below promotion.
- candidate-scorer PPO self-play run `s1303` is rejected: 42/64 versus
  random, 14/64 versus heuristic, and 3/16 versus the direct value teacher.
- `s1403b` was recycled after repeated mixed-opponent gates showed the same
  failure mode: it reached 10/12 versus random at `iter0020` and `iter0030`,
  but scored only 0/12 and then 2/12 versus heuristic.
- active runs now include GAE, normalized legal-action entropy, actor-critic
  teacher warmup, persistent supervised optimizers, soft value-teacher targets,
  the candidate legal-action scorer, optional PPO clipped value loss, and
  non-random `strong_mixed` opponent sampling. New replay-warmup branches train
  supervised warmup on a rolling buffer rather than only the latest teacher
  game.

Current active experiment families:

- `s1101`-`s1104`: GAE + persistent actor-critic teacher warmup runs
- `s1211b`-`s1212b`: soft value-teacher distillation with the flat PPO head
- `s1301`-`s1302`: rejected candidate-scorer soft value-teacher distillation
- `s1303`: rejected candidate-scorer warm start followed by self-play PPO
- `s1304`: candidate-scorer warm start followed by mixed-opponent PPO
- future `s14xx`: candidate scorer with learned per-action embeddings; this
  gives the model enough capacity to memorize board-location and action-id
  effects while still using structured action semantics
- `s1401b`: embedded candidate scorer, pure soft value-teacher distillation
- `s1402b`: embedded candidate scorer, teacher warm start plus PPO with
  interim checkpoints every 10 iterations. First gate: `iter0010` scored
  12/12 versus random and 4/12 versus heuristic.
- `s1403b`: recycled embedded candidate scorer, teacher warm start plus PPO against a
  mixed opponent pool, also with interim checkpoints every 10 iterations.
  Gates: `iter0010` scored 7/12 versus random and 4/12 versus heuristic;
  `iter0020` scored 10/12 versus random and 0/12 versus heuristic; `iter0030`
  scored 10/12 versus random and 2/12 versus heuristic.
- `s1404b`: h512 embedded candidate scorer, pure soft value-teacher
  distillation, replacing the rejected flat `s1211b` branch
- `s1405b`: embedded candidate scorer, PPO against a mixed opponent pool with
  tighter clipping and a larger teacher-anchor batch, replacing rejected `s1103`
- `s1501`: embedded candidate scorer, self-play PPO, and
  `--value-clip-range 0.2`
- `s1502`: embedded candidate scorer, mixed-opponent PPO, and
  `--value-clip-range 0.2`
- `s1601`: embedded candidate scorer, `strong_mixed` opponents
  (heuristic/value, no random), and `--value-clip-range 0.2`
- `s1701`: embedded candidate scorer with `--warmup-replay-size 12000`,
  `strong_mixed` opponents, and `--value-clip-range 0.2`
- `s1801`: embedded candidate scorer with both `--warmup-replay-size 12000`
  and `--anchor-replay-size 12000`, so PPO starts from a replayed teacher
  warmup and keeps a replayed teacher anchor during online updates.
- `s1802`: h512 embedded candidate scorer initialized from the completed
  `s1404b` distillation checkpoint, then continued with `strong_mixed` PPO,
  clipped value loss, and `--anchor-replay-size 16000`. Recycled after
  `iter0030` regressed to 8/12 versus random and 1/12 versus heuristic.
- `s1803`: embedded candidate scorer with sharper value-teacher distillation:
  lower teacher temperature and a hard-label blend into soft teacher targets,
  then replay warmup, replay anchor, `strong_mixed` PPO, and clipped value loss.
- `s1804`: h512 variant of the hard/soft replay-anchor branch, replacing the
  stale flat `s1104` run after it reached iteration 50 without producing a
  useful promoted checkpoint.
- `s1805`: deterministic value-teacher branch replacing stale mixed-opponent
  `s1502`; removes teacher target noise by using stable candidate pruning and
  restoring root randomness while scoring candidates, then uses hard/soft
  replay-anchor training.
- `s1901`: observation-normalization branch replacing weak `s1601`; preserves
  binary/one-hot observation features at full scale and only rescales larger
  count features, then uses deterministic hard/soft replay-anchor training.
  First comparable warmup gate: `warmup0048` scored 7/8 versus random and
  2/8 versus heuristic. Its `iter0020` checkpoint became the first later PPO
  checkpoint worth a full held-out check: the small gate was 8/12 versus
  random and 5/12 versus heuristic. The held-out check for `iter0020` scored
  46/64 versus random, 16/64 versus heuristic, and 5/32 versus the direct
  value baseline, so it is rejected for promotion.
- `s1902`: hard-teacher branch replacing weak `s1701`; same corrected
  observation normalization and deterministic value teacher as `s1901`, but
  uses `--imitation-hard-target-weight 1.0` to test whether soft value targets
  were over-smoothing important choices. First comparable warmup gate:
  `warmup0048` scored 7/8 versus random and 1/8 versus heuristic.
- `s1903`: recycled weak `s1801` into a corrected-normalization h512
  hard/soft branch with all-seat self-play after value-teacher warmup. This
  tests whether the new imitation fixes still transfer when PPO controls all
  seats instead of only one learner seat. First warmup gate: `warmup0028`
  scored 7/8 versus random and 1/8 versus heuristic.
- `s1904`: recycled stale mixed-opponent `s1405b` into corrected-normalization
  hard selected-action imitation against the broader `mixed` opponent pool.
  This tests whether the hard teacher remains useful when random opponents are
  present during online training. First warmup gate: `warmup0024` scored 5/8
  versus random and 3/8 versus heuristic. Held-out `warmup0024` eval with seed
  `99190424` scored 22/64 versus heuristic and the random check with seed
  `99190425` scored 38/64. The 32-game direct value-baseline check with seed
  `99190426` finished at 4/32, so this checkpoint is rejected for promotion
  despite being useful against random and heuristic opponents. The original
  long-running `s1904` process was stopped after `s1909` took over from its
  best warmup checkpoint.
- `s1905`: recycled stale self-play `s1402b` into a corrected-normalization
  h512 strong-mixed branch with sharper teacher temperature, larger candidate
  set, and a 0.55 hard-label blend.
- `s1906`: first dynamic historical-opponent league branch. During PPO it
  freezes the current policy every few iterations and samples those snapshots
  inside `strong_mixed`, so the learner must improve against heuristic/value
  baselines plus its own recent historical policies.
- `s1907`: recycled weak `s1802` into the best current warmup recipe:
  corrected-normalization, hard selected-action imitation, h512 capacity,
  broader `mixed` opponents, and dynamic historical league snapshots.
- `s1908`: recycled weak `s1803` into the first old-policy-KL branch. It uses
  the `s1904` hard-teacher mixed-opponent recipe plus `--old-policy-kl-coef
  0.05`, lower PPO learning rate, stronger replay anchor, and dynamic
  historical snapshots. The purpose is to test whether PPO can preserve the
  best warmup behavior instead of regressing after self-play starts. First
  warmup gate `warmup0024` scored 6/8 versus random and 1/8 versus heuristic,
  so the from-scratch KL branch is not currently ahead of `s1904`.
- `s1909`: recycled stale non-KL `s1804` into a direct continuation from
  `s1904.warmup0024`. It skips warmup, starts from the best held-out
  heuristic checkpoint, and uses tighter PPO updates (`--clip-ratio 0.08`,
  `--old-policy-kl-coef 0.08`) plus a stronger teacher replay anchor. This is
  the fastest test of whether regularized self-play can improve a useful
  learned checkpoint without destroying it. First PPO summary showed teacher
  anchor agreement near 0.895 and old-policy KL around 0.0012.
- `s1910`: recycled the rejected original `s1904` run into a high-capacity
  value-teacher distillation branch. It uses h1024 capacity, top-160 value
  candidates, sharper teacher temperature, 256 warmup games, and new interim
  value-baseline gates every 32 warmup games. It has no PPO phase until a
  distilled checkpoint proves it can compete with the direct value baseline.
  First gate `warmup0032` scored 5/8 versus random, 2/8 versus heuristic, and
  1/8 versus the value baseline, so this blended target recipe is currently
  behind the hard-only branch.
- `s1911`: recycled weak self-play branch `s1903` into a paired h1024
  distillation branch. It matches `s1910`'s value-gated setup but uses pure
  hard selected-action imitation (`--imitation-hard-target-weight 1.0`) to
  test whether the soft value target is still too diffuse for strong play.
  First gate `warmup0032` scored 6/8 versus random, 1/8 versus heuristic, and
  3/8 versus the direct value baseline. This is the best early h1024 signal,
  but it needs a larger held-out value check before promotion.
- `s1912`: first score-margin distillation branch. It adds `target_scores`
  from the value teacher and trains an auxiliary normalized action-score loss
  (`--imitation-score-coef 0.15`) in addition to hard selected-action
  imitation. The goal is to teach the fast policy the value teacher's
  per-action ranking margins instead of only its argmax. It runs on
  `bx_h7ptabx5` with the same h1024 value-gated setup as `s1911`. First gate
  `warmup0032` scored 6/8 versus random, 4/8 versus heuristic, and 0/8 versus
  the direct value baseline. Because the branch improved heuristic play while
  missing the strongest baseline completely, it was stopped and recycled.
- `s1913`: recycled weak `s1910` after its first value gate failed. This is a
  direct PPO/league continuation from `s1911.warmup0032`, the strongest early
  h1024 value-gated checkpoint so far. It uses tighter updates, old-policy KL
  (`0.10`), replay teacher anchors, `strong_mixed` opponents, dynamic
  historical snapshots, and value-baseline gates on every 5-iteration
  checkpoint. The purpose is to test whether regularized self-play can improve
  the best hard-distilled checkpoint without destroying its value-baseline
  behavior. It failed both early gates: `iter0005` scored 9/12 versus random,
  4/12 versus heuristic, and 1/8 versus the direct value baseline; `iter0010`
  regressed to 4/12 versus random, 2/12 versus heuristic, and 0/8 versus the
  direct value baseline. The branch was stopped.
- `s1914`: recycled rejected `s1901` after its held-out value-baseline eval
  failed. This is the shaped-reward A/B against `s1913`: same
  `s1911.warmup0032` initialization and league recipe, but PPO rollout
  collection adds a clipped Catanatron value-score delta
  (`--value-shaping-coef 0.03`, `--value-shaping-scale 100.0`) to each learner
  action reward. Terminal win/loss remains the main reward; the shaping term is
  a conservative credit-assignment aid. First gate `iter0005` scored 7/12
  versus random, 4/12 versus heuristic, and 0/8 versus the direct value
  baseline, so this branch was stopped and recycled.
- `s1915`: recycled failed score-margin branch `s1912` into a conservative PPO
  continuation from `s1911.warmup0032`. Compared with `s1913`, it uses lower
  learning rate, lower entropy, tighter clipping, stronger old-policy KL
  (`0.16`), fewer PPO epochs, and more teacher-anchor games. The purpose is to
  test whether PPO can improve the hard-distilled checkpoint while preserving
  the behavior that gave `s1911.warmup0032` the best early value-baseline gate.
  First gate `iter0005` scored 7/12 versus random, 3/12 versus heuristic, and
  0/8 versus the direct value baseline, so this branch was stopped. Plain PPO
  continuation, even with tighter updates, is not currently preserving the
  strongest baseline behavior.
- `s1916`: recycled the stopped `s1911` box into the first DAgger correction
  branch, starting from `s1911.warmup0032`. Normal teacher anchors label states
  the value teacher reaches; DAgger labels states reached by the current
  learner while still storing the value teacher's selected action, soft target,
  and score targets. This directly attacks the observed failure mode where
  warmup checkpoints look promising but PPO drifts into states where the
  imitation anchor has poor coverage. The branch uses `strong_mixed` opponents,
  old-policy KL `0.12`, six normal anchor games plus four DAgger games per
  iteration, and value-baseline gates every five PPO iterations. First gate
  `iter0005` scored 8/12 versus random, 4/12 versus heuristic, and 1/8 versus
  the direct value baseline, so this branch was stopped.
- `s1917`: recycled failed shaped-reward branch `s1914` into a heavier DAgger
  A/B from `s1911.warmup0032`. Compared with `s1916`, it uses fewer normal
  anchor games and more learner-visited DAgger games per iteration, tighter PPO
  clipping, stronger old-policy KL (`0.16`), and lower learning rate. The gate
  is the same: it must recover value-baseline wins, not merely show lower
  imitation loss. First gate `iter0005` looked promising at 9/12 versus random,
  4/12 versus heuristic, and 2/8 versus the direct value baseline, but the
  larger held-out check versus the value baseline scored only 9/64
  (`14.1%`, Elo estimate `-314`). This branch was stopped rather than
  promoted.
- `s1918`: recycled failed `s1913` into a DAgger plus score-margin branch from
  `s1911.warmup0032`. It uses six learner-visited DAgger games, four ordinary
  teacher-anchor games, tighter PPO settings, and a small score-margin
  coefficient (`--imitation-score-coef 0.035`) instead of the failed `s1912`
  coefficient `0.15`. The test is whether ranking-margin supervision helps
  value-baseline recovery without dominating hard action imitation. First gate
  `iter0005` scored 7/12 versus random, 2/12 versus heuristic, and 0/8 versus
  the direct value baseline, so this branch was stopped.
- `s1919`: recycled failed conservative PPO branch `s1915` into a no-PPO
  DAgger/reanalysis branch from `s1911.warmup0032`. It still visits learner
  states, but uses `--ppo-epochs 0`, ten DAgger games, six ordinary
  teacher-anchor games, two anchor epochs, and the same small score-margin
  coefficient. This isolates whether PPO updates are the source of the
  value-baseline collapse. It briefly reached 3/8 versus the value baseline at
  `iter0005`, but regressed to 1/8 at `iter0010`, so this branch was stopped.
- `s1920`: recycled failed `s1917` into the hard-only no-PPO DAgger control
  from `s1911.warmup0032`. It matches the reanalysis-only idea in `s1919` but
  disables score-margin supervision (`--imitation-score-coef 0.0`) and uses a
  faster first gate every three iterations. This isolates whether the score
  target is helping or hurting once PPO updates are removed. It scored 0/8
  versus the value baseline at both `iter0003` and `iter0006`, so this branch
  was stopped.
- `s1921`: added `ValueRolloutSearchPolicy`, a legal root-search wrapper that
  scores root actions by exact simulator rollouts where all rollout players act
  greedily under the Catanatron value function. Heuristic rollout search was
  rejected after starting 0/8 versus the direct value baseline. An initial
  value-rollout result of 21/64 was discarded because the evaluator incorrectly
  applied the candidate search candidate limit and opponent-penalty setting to
  the opponent value policy. After separating candidate and opponent evaluator
  settings, value-rollout search with candidate limit 16, rollout depth 4, and
  opponent penalty 0.00 scored 17/64 (`26.6%`) against canonical value-policy
  opponents (`candidate_limit=48`, opponent penalty `0.05`). In this
  one-candidate-versus-three-identical-opponents setup, 25% is the equal-strength
  reference, so this is approximately value-baseline strength and remains the
  strongest current legal non-learned algorithm.
- `s1922`: launched the first distillation run using the value-rollout search
  policy as the teacher. It is pure supervised warmup from scratch
  (`--iterations 0`) with h1024 candidate architecture, 24 warmup games,
  checkpoints every eight warmup games, and the search teacher setting from
  `s1921` (`candidate_limit=16`, `rollout_decisions=4`,
  `opponent_penalty=0.00`). The goal is to compress the stronger search policy
  into a fast model without PPO drift.
- `s1923`: added soft-policy and score-target outputs to
  `ValueRolloutSearchPolicy` and launched a paired search-teacher distillation
  branch. Compared with `s1922`, it uses 16 warmup games, blends hard and soft
  targets (`--imitation-hard-target-weight 0.85`), and adds a small score-margin
  loss (`--imitation-score-coef 0.03`) so the fast model can learn the search
  ranking rather than only the selected action.
- `s1924`: swept canonical value-rollout search settings after the evaluator
  fix. Candidate limit 16, rollout depth 4, opponent penalty 0.02 scored only
  7/32 versus the canonical value baseline and was rejected. Candidate limit 24,
  rollout depth 4, opponent penalty 0.00 scored 21/64 (`32.8%`) versus canonical
  value-policy opponents, restoring a real above-baseline search result. This
  becomes the current strongest legal algorithm and the next distillation
  teacher.
- `s1925`: hardened remote training reproducibility after fresh boxes imported
  the incompatible PyPI `catanatron==3.2.1` package, which lacks
  `catanatron.features`. The project importer now prefers the pinned vendored
  Catanatron tree when present and falls back to it if a Catanatron submodule is
  missing. Fresh box `bx_fzy3avks` verified the vendored import path and passed
  `tests/test_self_play.py` with 30/30 passing tests.
- `s1926`: improved value-rollout teacher throughput for distillation. The
  search teacher now caches root action scores per decision, so
  `select_action`, `target_policy`, and `target_scores` reuse one expensive
  rollout pass instead of rescoring the same root candidates multiple times.
  The rollout teacher also honors `--teacher-temperature`, making hard, soft,
  and hard/soft blended distillation branches actually comparable.
- `s1930`: launched canonical search sweep on `bx_fzy3avks`: value-rollout
  search with candidate limit 32, rollout depth 4, opponent penalty 0.00 versus
  canonical value opponents. First progress gate was 2/8 (`25.0%`), so this is
  not yet evidence of improvement over the current 24x4 setting. Later progress
  reached 5/16 (`31.25%`) and 7/24 (`29.2%`), but then fell to 24/96 (`25%`).
  Unless the final tail reverses sharply, reject 32x4 and keep 24x4 as the
  strongest verified search setting.
- `s1931`: launched hard-label distillation on `bx_w6va6xpp` from the current
  best 24x4 value-rollout teacher. Configuration: h1024 candidate scorer,
  24 warmup games, rolling replay, no PPO continuation, hard-target weight 1.0,
  no score-margin loss. Warmup-8 checkpoint scored 9/12 versus random, 1/12
  versus heuristic, and 2/8 (`25%`) versus the direct value baseline. This is not
  enough to promote but is much better than most prior learned value-baseline
  gates. Warmup-16 regressed to 0/8 versus the value baseline, so the warmup-8
  checkpoint is the only candidate worth a larger held-out check so far.
  Warmup-24 finished at 1/8 versus value, confirming the overtraining pattern.
- `s1932`: launched paired soft/score distillation on `bx_b9xj86u9` from the
  same 24x4 teacher. Configuration matches `s1931` except temperature 0.55,
  hard-target weight 0.85, and score-margin coefficient 0.03. First two teacher
  games completed and logged losses. Warmup-8 checkpoint scored 9/12 versus
  random, 2/12 versus heuristic, and 2/8 (`25%`) versus the direct value
  baseline. Warmup-16 regressed to 1/8 versus the value baseline. Continue to
  warmup-24 for the final report, but do not promote unless a larger held-out
  check confirms the warmup-8 signal. Warmup-24 also scored 1/8 versus value.
  Final evaluation of the final checkpoint scored 39/64 versus random, 15/64
  versus heuristic, and 3/32 (`9.4%`) versus value, so it is rejected as a
  deployable checkpoint.
- `s1933`: added `--select-best-warmup-checkpoint` to `tools/train_ppo.py`.
  When enabled, the trainer reloads and saves the best interim warmup checkpoint
  before final evaluation, preferring value-baseline win rate over heuristic and
  random-agent scores. This directly addresses the `s1931`/`s1932` failure mode
  where supervised loss kept improving while value-baseline strength peaked at
  warmup-8 and then regressed.
- `s1934`: ran a larger held-out evaluation of
  `s1932.warmup0008.pt`, because its tiny warmup gate was 2/8 versus value.
  The larger check finished 14/128 (`10.9%`) versus the canonical value
  baseline, so the warmup-8 soft/score checkpoint is rejected. The small 2/8
  gate was noise.
- `s1935`: launched the same soft/score distillation recipe with
  `--select-best-warmup-checkpoint` enabled. Warmup-8 was only 1/8 versus value,
  so this branch is not currently a promotion candidate. It remains useful as a
  trainer-behavior check: final report should select the best interim checkpoint
  rather than blindly saving the last warmup state. Final report selected the
  warmup-16 checkpoint, but it scored only 47/64 (`73.4%`) versus random, 15/64
  (`23.4%`) versus heuristic, and 4/64 (`6.25%`) versus value. Rejected. This
  also showed that 8-game value gates are too noisy for model selection.
- `s1936`: launched a larger held-out evaluation of
  `s1931.warmup0008.pt`. It finished 13/128 (`10.2%`) versus the canonical
  value baseline, so the hard warmup-8 checkpoint is rejected.
- `s1937`-`s1939`: launched search confirmation/sweep runs against the canonical
  value baseline. Early progress: 24x4 is 13/32 (`40.6%`), 20x4 is 13/40
  (`32.5%`), and 24x3 is 16/48 (`33.3%`). These need full 128-game results, but
  the search track remains more promising than learned distillation. Later
  progress strengthened this: 24x4 was 18/48 (`37.5%`), 20x4 was 26/64
  (`40.6%`), and 24x3 was 22/64 (`34.4%`). Later still, 20x4 was 32/88
  (`36.4%`), while 24x4 drifted to 21/72 (`29.2%`) and 24x3 to 29/96
  (`30.2%`). If 20x4 holds, it may be both stronger and cheaper than the
  previous 24x4 best setting.
  Completed results so far: 20x4 finished 40/128 (`31.25%`), which is below the
  earlier 24x4 champion. 24x3 finished 44/128 (`34.4%`), making it the strongest
  single completed sweep and cheaper than 24x4. A fresh 24x3 confirmation
  (`s1945`) is running before changing the default.
- `s1940`: added multi-sample value-rollout search. `ValueRolloutSearchPolicy`
  now supports `rollout_samples`, and evaluation/training CLIs expose
  `--search-rollout-samples` and `--teacher-rollout-samples`. First sweep is
  12 candidates x 4 rollout decisions x 2 chance samples; initial progress was
  3/8 versus value, but it finished 20/96 (`20.8%`), so this first multi-sample
  setting is rejected.
- `s1941`-`s1944`: added value-presearch candidate pruning. `ValueRolloutSearchPolicy`
  now supports a wider one-ply value-ranked candidate pool before rollout
  scoring the final shortlist, exposed by `--search-presearch-candidate-limit`
  and `--teacher-presearch-candidate-limit`. The first launch missed
  `PYTHONPATH=src` and failed immediately (`s1941`, `s1942`); corrected runs
  ran as `s1943` (16x4 presearch 96) and `s1944` (20x4 presearch 96). Both were
  stopped early as weak branches: `s1943` was 18/64 (`28.1%`) and `s1944` was
  23/88 (`26.1%`) versus value, both below plain 24x3.
- `s1945`-`s1946`: launched follow-up search sweeps: 24x3 confirmation on a
  fresh seed and cheaper 20x3. Promotion rule: keep 24x3 only if the
  confirmation remains above the older 24x4/20x4 cluster; use 20x3 only if it
  matches 24x3 at lower compute. `s1945` finished 52/128 (`40.6%`) versus the
  canonical value baseline, confirming 24x3 as the short-game search champion.
- `s1947`: relaunched the cheaper 20x3 sweep after the first `s1946` attempt
  used a directory without its own venv. Stopped at 17/88 (`19.3%`) versus
  value, so 20x3 is rejected.
- `s1948`: launched a larger actual model-training run from the current best
  teacher shape: 24x3 value-rollout teacher, h1024 candidate scorer, 64 warmup
  games, rolling replay, soft/hard blended targets, score-margin loss, and
  `--select-best-warmup-checkpoint`. This is the next serious attempt to turn
  the stronger search teacher into a fast deployable policy.
- `s1949`-`s1950`: launched the same 24x3 search teacher against heuristic and
  random opponents, not just the direct value baseline. This gives the current
  search champion a broader Elo/profile check. Results: 61/128 (`47.7%`) versus
  heuristic and 110/128 (`85.9%`) versus random.
- `s1951`: added optional root-value blending to `ValueRolloutSearchPolicy`,
  exposed as `--search-root-value-weight` and `--teacher-root-value-weight`.
  This mixes immediate one-ply Catanatron value with the short rollout terminal
  value to test whether short-rollout noise is hurting 24x3. Local Mac sweep:
  24x3 with root weight 0.25 versus the value baseline.
- `s1952`-`s1953`: launched full 128-game root-blend sweeps on boxes: 24x3 with
  root weights 0.25 and 0.50 versus value. Promotion criterion is beating the
  plain 24x3 confirmation, not merely exceeding 25%.
- `s1954`: hardened warmup checkpoint selection by ranking eval reports with a
  Wilson lower confidence bound when wins/games are available, then launched a
  conservative distillation run with 32-game value gates at each warmup
  checkpoint. This directly addresses `s1935`'s noisy 2/8 checkpoint selection.
- `s1955`: launched the confirmed 24x3 search champion on a more realistic
  10-VP benchmark versus the value bot, 32 games with a 1500-decision cap. This
  checks whether the short-game teacher transfers toward real Catan games.
- `s1956`: launched policy-only soft/score distillation from the 24x3 teacher:
  same h1024 candidate scorer and 32-game value gates, but no value-head loss
  during warmup. This tests whether critic/value loss is harming imitation.
- `s1957`: launched hard-action-only policy distillation from the 24x3 teacher:
  hard selected-action targets, no score-margin loss, no value-head loss, and
  32-game value gates. This tests whether the soft/score target mixture is the
  learned-policy failure mode.
- `s1958`: added held-out teacher-agreement checkpoint evaluation to
  `tools/train_ppo.py` and launched hard-action-only distillation selected by
  agreement on fresh 24x3 teacher games. This tests whether noisy small
  value-game gates were selecting the wrong warmup checkpoint.
- `s1959`: added a candidate-policy ablation that disables learned per-action
  ID embeddings and launched the same agreement-selected hard distillation
  recipe with `--disable-action-id-embedding`. This tests whether the model is
  memorizing action ids instead of learning structured state/action scoring.
- `s1960`: launched a 128-game held-out value-bot evaluation for the promising
  `s1954.warmup0016` checkpoint. The checkpoint looked good at its original
  32-game gate, but promotion now requires the larger independent check.
- `s1961`: launched PPO continuation from `s1954.warmup0016` against
  `strong_mixed` opponents with teacher anchors and DAgger, then stopped it
  after finding that `strong_mixed` used a weaker 32-candidate value opponent.
- `s1962`: fixed PPO opponent construction so value opponents sampled during
  training use configurable candidate limits aligned with held-out gates, then
  relaunched the continuation recipe against 48-candidate value opponents.
- `s1963`: stopped the weak `s1960` learned-policy eval at 8/56 versus value
  and launched a 128-game value-bot eval of `s1958.warmup0008`, which had
  85.6% held-out agreement on 728 fresh teacher samples. This tests whether
  teacher agreement is a better early promotion signal than noisy tiny game
  gates.
- `s1964`: attempted PPO continuation from `s1958.warmup0016`, but the recycled
  box still had an older `self_play.py` without the current rollout-search
  constructor. The run failed before training and was replaced after syncing the
  module.
- `s1965`: stopped stale weak distillation branches (`s1948`, `s1954`) and
  launched a direct 128-game value-bot eval of `s1958.warmup0016`, whose
  teacher-agreement gate improved to 89.1% on 819 fresh teacher samples.
- `s1966`: relaunched PPO continuation from `s1958.warmup0016` after syncing
  `self_play.py`, using the same strong-mixed 48-candidate value-opponent gate
  alignment as `s1962`.
- `s1967`: stopped `s1963` after it fell to 5/72 versus the value bot, then
  launched PPO continuation from the no-action-id `s1959.warmup0016`
  checkpoint. This keeps testing whether structured action features generalize
  better once the policy is pushed through strong-mixed PPO rather than judged
  only by supervised agreement.
- `s1968`/`s1969`: attempted DAgger-only continuation from the action-id and
  no-action-id agreement-selected checkpoints, but discovered that
  `--anchor-games-per-iteration 0` accidentally skipped the whole anchor/DAgger
  update block. These runs produced zero-sample iterations and were stopped.
- `s1970`/`s1971`: fixed DAgger-only training so DAgger games trigger imitation
  updates even when no fresh teacher self-play anchor games are requested, added
  `--select-best-checkpoint` so final save reloads the best measured interim
  gate, then relaunched the corrected action-id and no-action-id DAgger-only
  continuations. First iterations now collect real learner-distribution samples
  (`s1970`: 230, `s1971`: 185).
- `s1972`: stopped weak continuation branches `s1962` and `s1967` after their
  value-bot gates stayed below baseline (`s1962`: best 4/32; `s1967`: 0/32),
  then launched a 128-game independent value-bot evaluation of the strongest
  learned checkpoint so far, `s1966.iter0003` (`9/32` at its original gate).
  This was stopped and rejected at `5/48`; the original `9/32` promotion signal
  was almost certainly gate noise.
- `s1973`: launched a continuation from `s1966.iter0003`, using lower PPO
  learning rate, stronger old-policy KL, fewer PPO episodes, more DAgger games,
  and `--select-best-checkpoint`. The intent is to keep the one checkpoint that
  showed a >25% value-bot signal while correcting its learner-state
  distribution instead of continuing from later regressed checkpoints.
- `s1974`: stopped `s1970` after its first corrected DAgger-only gate fell to
  `2/32` versus the value bot (`s1971` reached only `5/32` at the same gate),
  then launched a checkpoint-league continuation from `s1966.iter0003`. This
  branch trains against the strong value bot, heuristic bot, and the best
  historical learned checkpoint, with dynamic league snapshots enabled.
- `s1975`: stopped `s1971` after its second DAgger-only value gate regressed to
  `3/32`, and stopped the older `s1966` continuation after later gates remained
  below baseline (`4/32`, `5/32` after the noisy initial `9/32`). Current learned
  policies are not yet reliable; promotion remains blocked on a real value-bot
  gate rather than teacher agreement or random-bot wins.
- `s1976`/`s1977`: stopped `s1973` after its first continuation gate reached only
  `6/32`, and stopped `s1974` after its checkpoint-league gate reached only
  `4/32`. Replaced the teacher score regression loss with a pairwise ranking
  loss over scored legal actions, then launched two fresh value-rollout
  distillation runs: action-id (`s1976`) and no-action-id (`s1977`). Both use
  `--imitation-score-coef 0.20`, value gates every eight warmup games, and
  `--select-best-min-value-win-rate 0.25`.
- `s1978`/`s1979`: launched direct value-teacher pairwise-rank distillation
  controls, action-id and no-action-id. These test whether the neural policy can
  imitate the 48-candidate value bot itself before we keep trying to distill the
  noisier rollout-search teacher. Same promotion floor: value-bot gate must
  reach at least 25%.
- Checkpoint selection note: warmup selection now has a training-wide sibling.
  Serious PPO/DAgger runs can select the best evaluated warmup or iteration
  checkpoint by value-bot Wilson lower bound before final evaluation/save,
  preventing the final artifact from regressing simply because later training
  moved away from the best held-out checkpoint.
- Promotion-floor note: `--select-best-min-value-win-rate` is now available.
  Future serious runs should use `--select-best-min-value-win-rate 0.25` so
  sub-baseline value-bot checkpoints are not selected just because every
  checkpoint in that run is weak.
- PPO stability note: advantage normalization now uses population standard
  deviation and falls back to centered advantages when a batch has too little
  variance. This removes the single-sample `std()` warning and avoids NaN
  advantages in tiny rollout/update edge cases.
- future `s18xx`: compare replay-anchor variants across model size,
  opponent mix, and teacher-anchor strength.
- `--init-checkpoint` is available for the next phase: take a completed pure
  distillation checkpoint and continue it directly with PPO/league training
  instead of rerunning supervised warmup.

That means the environment, value teacher, search wrapper, checkpointing, and
evaluation loop work end to end. The strongest verified player is currently
the canonical `ValueRolloutSearchPolicy` 24x3 setting, confirmed by `s1939` and
`s1945`. The learned fast policy is still weaker. The immediate target is to
prove the 24x3 teacher on longer 10-VP games, then distill that stronger teacher
into a fast model while keeping canonical value-baseline gates as the promotion
criterion.

## Next Upgrade

Promote only when a checkpoint clears held-out evaluation:

- at least 32 games versus random, heuristic, and value opponents,
- seat-balanced win-rate report,
- no illegal actions,
- no promotion if it only beats random,
- checkpoint league evaluation once two learned policies beat heuristic.

Use the report summarizer before recycling boxes:

```bash
.venv/bin/python tools/summarize_self_play_reports.py 'runs/self_play/*.json'
```

Near-term algorithm work:

- add vector/value heads for all seats,
- add opponent mixtures from historical learned checkpoints,
- compare flat global logits against the candidate action scorer,
- compare unclipped scalar-critic PPO against `--value-clip-range 0.2`,
- compare random-heavy `mixed` opponents against non-random `strong_mixed`,
- compare one-game teacher warmup against rolling replay warmup,
- compare current-only teacher anchors against rolling replay teacher anchors,
- compare plain clipped PPO against old-policy-KL PPO to measure whether
  policy regularization prevents the warmup-to-PPO regression,
- compare terminal-only PPO against conservative value-shaped PPO rewards,
- compare standard old-policy-KL continuation against lower-entropy,
  stronger-KL conservative continuation,
- compare purely soft teacher targets against hard/soft blended targets,
- require interim value-baseline evals on serious distillation/PPO branches so
  random/heuristic wins do not hide weakness against the strongest baseline,
- compare stochastic teacher candidate scoring against deterministic root
  teacher scoring,
- compare old global `/20` observation scaling against binary-preserving
  observation normalization,
- compare hard/soft blended value-teacher targets against hard selected-action
  imitation,
- compare noisy game-gated checkpoint selection against held-out teacher
  agreement selection,
- compare candidate action scoring with and without learned action-id
  embeddings,
- compare selected-action imitation against score-margin value-teacher
  distillation,
- compare fixed opponent mixtures against dynamic historical-snapshot leagues,
- keep PPO training value-opponent strength aligned with held-out value gates,
- compare value-presearch versus heuristic-only candidate pruning for rollout
  search,
- compare pure rollout scoring against root-value-blended rollout scoring,
- distill direct value/search teachers into the PPO policy without collapsing
  self-play improvement,
- prefer DAgger on learner-visited states against strong value opponents over
  teacher-only agreement once supervised checkpoints show high agreement but
  fail direct value-bot play,
- use `--select-best-checkpoint` on serious PPO/DAgger runs so promotion is
  tied to the best held-out gate in the run rather than the last update,
- use `--select-best-min-value-win-rate 0.25` on serious learned-policy runs
  until a better statistical promotion rule replaces it,
- prefer pairwise ranking distillation for search scores over direct normalized
  score regression when using value-rollout teacher scores.
