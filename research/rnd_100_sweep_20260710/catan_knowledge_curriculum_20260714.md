# Catan knowledge curriculum: 2-player now, 4-player trading later

Status: research proposal only. No production code or training configuration is changed.

## Recommendation

Encode expert knowledge as **questions the network must answer**, not bonuses that
tell it what move to make. Use the simulator to label counterfactual outcomes,
use expert concepts to stratify the corpus and construct opponents, and retain
win/loss as the only optimization objective that can promote a policy.

The repository already has most of the right attachment points:

- `aux_subgoal_targets.py` produces Longest Road, Largest Army, VP-at-horizon,
  next-settlement and robber-target labels.
- `regret_common.py` stratifies opening, robber/development, chance and ordinary
  build states and identifies search/prior disagreement.
- `self_play.py` contains ore/city, road-race and robber specialists.
- `flywheel/opponent_mix.py` supports a configurable population rather than
  hard-coding one roster.

The main missing piece is that several current auxiliary targets describe
**what the trajectory eventually did**, not whether that choice was good. They
are useful representation targets, but they should be supplemented with paired
counterfactual value and regret labels.

## State taxonomy

Store multi-hot tags and continuous measurements; do not force each position
into one human strategy class.

| Axis | Proposed tags or measurements | Why it matters |
|---|---|---|
| Decision phase | initial settlement, initial road, pre-roll development card, discard, robber hex/victim, main build/bank action, end turn, trade offer/response | Prevents abundant routine rows from drowning rare high-leverage decisions. |
| Game horizon | opening, expansion, engine conversion, award race, endgame/one-turn threat | A good opening label and a good endgame label answer different questions. |
| Economy | per-resource production, blocked production, resource diversity, port ratios, cards-to-settlement/city/dev/road, hand-over-seven risk | Captures production *and* convertibility; raw pip totals alone are insufficient. |
| Topology | reachable nodes, contested nodes, minimum roads to expansion, cut/block threats, longest-road length and challenger gap | Represents the parts of Catan that a flat heuristic most often misses. |
| Strategic plan | wide/settlement, tall/city, development-card/army, road award, port engine, mixed | Use soft scores derived from future action/resource deltas, never a forced one-hot identity. |
| Interaction | opponent can build now, opponent one roll away, race-to-node, award takeover threat, robber exposure | Makes denial and tempo learnable rather than treating the opponent as negative production only. |
| Information | opponent-hand entropy, dev-deck uncertainty, robber-steal uncertainty, build-threat probability | Separates a masked observation from a useful public belief state. |

Each row should retain `game_seed`, decision index, actor, seat, board hash,
opponent category, track, player count and taxonomy version. This supports
game-level splitting and avoids treating hundreds of correlated positions from
one game as independent evidence.

## Labels in the current 2-player no-trade system

### 1. Exact, cheap state labels

Derive these from the authoritative engine without rollouts:

- production probability by resource, with and without the current robber;
- port access and current bank-conversion ratios;
- minimum resource deficit for each build type;
- reachable and buildable vertices, contested vertex count and road distance;
- current Longest Road/Largest Army holder, size, lead and takeover distance;
- public VP, true terminal VP as a target only, hand size and discard exposure;
- remaining pieces and development-card deck composition.

These are descriptive targets. They may train detached auxiliary heads, but
must not be added directly to policy logits or terminal reward.

### 2. Paired counterfactual labels

Select roughly 10-15% of decision rows, oversampling openings, robber moves,
contested builds, award races, endgames and high search/prior disagreement.
For each selected state, clone the engine and evaluate legal actions under the
same opponent policy and the same chance streams. Common random numbers are
important: the difference between two moves should not mostly be different dice.

Start with 16 paired continuations per action and add another 16 for pairs whose
confidence interval overlaps. Persist:

- `q_win(a)` and Monte Carlo standard error;
- `delta_vp_8(a)` and `delta_vp_24(a)`;
- `regret(a) = max_a q_win(a) - q_win(a)`;
- probability of building settlement/city/dev/road before the next two turns;
- probability of winning or losing each special-card race;
- probability a contested node remains available at the actor's next turn;
- production and opponent-production delta after the action;
- hand-over-seven and forced-discard probability;
- for robber actions: blocked expected production, victim steal-value
  distribution and downstream win delta.

Train a legal-action Q/ranking head with confidence weighting. Apply pairwise
ranking loss only when the estimated Q difference exceeds two combined standard
errors; ambiguous pairs should not be converted into fake hard preferences.
Search visit counts remain the policy teacher. Counterfactual Q is an auxiliary
critic/diagnostic, not a replacement for the search target.

### 3. Opening solver labels

Opening placement is unusually wide and has an outsized effect on the reachable
state distribution. For a fixed board and seat order, enumerate every legal
first settlement/road continuation and evaluate it against a small population,
not only the current mirror. Labels should include paired win value, early
production by resource, resource diversity, contested-node survival, port
reachability and plan distribution. Cache by board, seat and opponent roster.

Do not turn conventional advice such as “maximize pips” or “get wood and brick”
into the target. Those are candidate explanatory features. The rollout value is
the target and must be allowed to disagree with conventional wisdom.

### 4. Public-belief labels

The trainer may inspect true hidden state to construct a supervised target, but
the model input and search state must remain public and pass hidden-state
invariance tests. Predict:

- marginal opponent resource counts and total-hand distribution;
- probability the opponent can build each object now and after the next roll;
- development-card type/count probabilities and probability of an available
  Knight/VP;
- conditional value distribution of a robber steal;
- entropy/calibration of each belief.

This is a low-cost bridge toward public-belief search. ReBeL is theoretically
appropriate to two-player zero-sum imperfect-information games, but a full
ReBeL rewrite is not required to test whether belief summaries improve Catan.

## Data mixture and opponents

Keep opponent weights in the existing manifest. A serious challenger to the
current 75/10/5/5/5 recipe is:

- 65% latest-producer self-play;
- 15% PFSP historical checkpoints, emphasizing opponents with producer win
  rates around 40-65% while retaining an old-anchor floor;
- 7.5% style specialists: ore/city, road race, robber pressure, expansion denial
  and award denial;
- 5% learned best-response exploiters against frozen recent champions;
- 5% external Catanatron bots spanning value and search budgets;
- 2.5% high-search teacher or reanalysis opponents.

This is an experiment arm, not a new default. Compare it to the current recipe
with equal games and paired evaluation. Record training rows only for the
producer seat when the opponent is frozen or scripted.

Style bots should be deliberately imperfect probes, not sources of truth.
Parameter-randomize their preferences within bounded ranges so the learner sees
a family of road/city/robber behaviors rather than memorizing three policies.
Every generation should run a population payoff matrix and promote on a robust
aggregate plus worst-opponent regression limits, not latest-vs-latest Elo.

## Sample-efficient curriculum

Avoid a purely sequential curriculum, which invites forgetting. Interleave:

1. **Foundation batches:** full-game searched self-play and high-search
   reanalysis.
2. **Opening batches:** exhaustive placement/road counterfactuals.
3. **Tactical restart batches:** high-regret robber, contested-build, award-race
   and endgame states.
4. **Belief batches:** public inputs with hidden-state posterior targets.
5. **Anti-exploit batches:** states generated against PFSP, styles and learned
   exploiters.

A reasonable screening mixture is 55/15/15/5/10 percent respectively. Sample
at most a fixed number of rows per game per taxonomy cell. Start the sum of new
auxiliary losses at 0.03 of policy-loss scale and screen 0.03 versus 0.10; keep
the existing terminal/search losses unchanged. Auxiliary-head gradients should
be monitored for conflict with policy/value gradients and each head must be
independently ablatable.

## Transfer to four players and trading

Use one shared entity trunk with explicit `player_count`, `track`, acting-seat
and relative-seat features. Transfer in four overlapping tracks:

1. 2-player no-player-trade;
2. 4-player bank/port trade only;
3. 4-player exact bilateral offers and responses;
4. 4-player wildcard offers, counters and negotiation context.

Retain 20-30% replay from earlier tracks at each transition. Four-player Catan
is general-sum and politically interactive: replace the two-player scalar
assumption with per-seat outcome/value vectors for training diagnostics and
search. A trade counterfactual needs at least:

- proposer win/VP delta versus no offer;
- accepter win/VP delta versus reject;
- each third party's delta;
- probability the trade immediately enables each build;
- leader-aid/externality and retaliation/offer-history features;
- acceptance and best-counteroffer distributions under the opponent population.

Train offer, response and counteroffer heads separately at first. Do not infer
“fairness” from card count or encode “never trade with the leader” as a rule;
the counterfactual seat-utility vector should decide whether a trade helped.
Population composition matters more in 4-player play: seat the main learner
against heterogeneous triples and randomize player permutations, avoiding a
table of four identical agents that can develop brittle conventions.

## Failure modes and gates

- **Heuristic lock-in:** expert features become a disguised reward. Prevention:
  detached auxiliary heads, no policy-logit bonus, terminal win remains the gate.
- **Hidden-information leakage:** true hands enter features through a target
  builder. Prevention: observation invariance tests and separate target-only
  storage paths.
- **Trajectory imitation:** `next_settlement` learns self-play's mistake.
  Prevention: distinguish descriptive trajectory labels from counterfactual
  quality labels.
- **Simulator exploitation:** the agent masters engine quirks. Prevention:
  cross-engine checks, rules audits and external populations.
- **2p negative transfer:** blocking and zero-sum habits hurt 4p negotiation.
  Prevention: track conditioning, earlier-track replay and per-seat values.
- **Rare-label domination or starvation:** award/trade labels are sparse.
  Prevention: taxonomy-balanced sampling and report effective labeled counts.
- **Opponent overfitting:** PFSP chases a narrow set of weaknesses. Prevention:
  historical anchors, external bots, style randomization and held-out exploiters.

Adopt a knowledge head only if it improves label calibration and either
population Elo or exploitability at equal search/compute, with no material
worst-opponent regression. Auxiliary accuracy alone is not evidence of a
stronger Catan agent.

## Public research grounding

- Gendre and Kaneko, [Playing Catan with Cross-dimensional Neural
  Network](https://arxiv.org/abs/2008.07079), motivate structured handling of
  Catan's board, cards and heterogeneous outputs.
- Driss and Cazenave, *Deep Catan* (ACG 2021), used Expert Iteration for
  multiplayer Catan, supporting search-to-policy distillation as the relevant
  teacher mechanism.
- Wu, [Accelerating Self-Play Learning in
  Go](https://arxiv.org/abs/1902.10565), reports large sample-efficiency gains
  from auxiliary ownership/score targets and other self-play improvements.
- Brown et al., [Combining Deep Reinforcement Learning and Search for
  Imperfect-Information Games](https://arxiv.org/abs/2007.13544), supplies the
  public-belief rationale for the current two-player hidden-information regime.
- Schmid et al., [Student of Games](https://arxiv.org/abs/2112.03178), combines
  guided search, self-play and game-theoretic reasoning across perfect- and
  imperfect-information games.
- DeepMind's [AlphaStar league
  description](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
  motivates main agents plus exploiters that expose specific weaknesses.
- The official [CATAN base-game FAQ](https://www.catan.com/faq/basegame) is the
  rules authority for development cards, Longest Road and Largest Army details.

