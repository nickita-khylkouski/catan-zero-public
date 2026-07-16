# A1 RL software diagnosis and repair program

**Date:** 2026-07-15
**Scope:** 2-player, no-trade, 10-VP A1 research track
**Status:** Confirmed software diagnosis; large learner and generation runs are
paused until the P0 information-state and value-routing repairs below land.

This is the collaboration entry point for the next RL implementation cycle. It
supersedes the assumption that the next useful action is simply more n128/n256
data, a longer learner dose, a larger network, or an unmodified PPO run.

The code already contains a policy/value network, coherent-public search,
distributed self-play, DDP distillation, distributed PPO components, and
evaluation infrastructure. The immediate failure is that several rules-distinct
positions collapse to the same neural state, policy training can damage the
shared representation faster than value learning repairs it, and current search
disagreement is not a sufficient target-reliability test.

## 1. Current system

```text
authoritative Rust Catan state
  -> actor-specific public observation plus exact own private information
  -> entity/action/event featurizers
  -> shared 35M entity Transformer
       -> legal-action policy scorer
       -> scalar state-value head
  -> coherent-public Gumbel MCTS
       -> policy priors
       -> leaf state values
       -> stored policy/search evidence
  -> self-play shards
       -> policy-active search targets
       -> terminal outcomes
       -> provenance and search contract
  -> DDP learner
       -> policy distillation
       -> terminal value regression
  -> searched candidate-versus-parent and external evaluation
  -> typed promotion transaction
```

The scalar value head exists in
`src/catan_zero/rl/entity_token_policy.py`. The current problem is not the
absence of a value network. It is incomplete state information, an underspecified
value input, and destructive shared-trunk optimization.

## 2. Confirmed P0 information-state defects

### 2.1 Development-card playability is missing

The authoritative Rust rules use both:

- `has_played_development_card_in_turn`;
- each development card type's `owned_at_start` count.

These determine whether the actor may play a development card. See
`native/catanatron-rs/src/lib.rs` around the development-card legality checks.

The snapshot exposes the current-turn dev-used bit, but
`src/catan_zero/search/neural_rust_mcts.py` drops it. The current player-token
surface in `src/catan_zero/rl/entity_token_features.py` contains neither that
bit nor per-type `owned_at_start`/playable counts.

This creates exact state aliasing:

```text
same board + same hand + old playable Knight
same board + same hand + Knight bought this turn
```

and:

```text
actor may still play a dev card
actor already played one this turn
```

can receive the same state representation.

This is especially damaging to the value function because the scalar value
readout consumes the encoded state, not the legal-action candidates. The policy
can partially observe the consequence through the legal mask; the value head
cannot.

### 2.2 Public multi-step rule state is missing

The Rust state also tracks:

- whether Road Building is active;
- free roads remaining;
- the current discard obligation.

The native snapshot exposes the relevant state, but the entity adapter does not
bind it into the model input. The model can therefore alias:

- first versus second free Road Building placement;
- ordinary `PLAY_TURN` versus Road Building continuation;
- discard sequences with identical current hand contents but different cards
  still required.

These are not obscure metadata. They change the legal continuation and the
meaning of the position.

### 2.3 Publicly played development cards are only partly represented

Cumulative public counts for Knight, Year of Plenty, Monopoly, and Road
Building are present. The missing information is:

- the current-turn dev-used bit;
- own card age/playability;
- exact late-game Knight count.

The legacy normalized public Knight feature saturates after five, so later
Largest Army races alias.

### 2.4 Monopoly and Year-of-Plenty actions lack semantic resource identity

The active structured-action path does not reliably bind:

- the selected Monopoly resource;
- the Year-of-Plenty resource pair.

These actions are distinguished largely by a normalized global action ID. The
serialized static action table contains richer resource-flow identity, but the
incumbent scorer does not consume that residual by default.

This is a large action-representation defect for two strategically important
development cards.

## 3. History and belief diagnosis

### 3.1 The current event surface is lossy

The meaningful-history path retains action type, actor, and a small subset of
metadata, but drops or fails to bind:

- turn boundaries;
- settlement vertex;
- road edge;
- robber destination;
- maritime resource payload;
- most event target IDs.

Native action records currently reach the Python event translator without a
usable `turn_key`. Event target IDs in the active feature path are effectively
unbound.

Events are independently encoded and mean-pooled through a zero-initialized
residual gate. This is not an ordered causal history model. Once `ROLL` and
`END_TURN` are removed, the representation cannot reliably infer when a dev
card was bought or whether it has aged into playability.

### 3.2 The live lineage has not learned history

The recovered-v5 training information surface records no usable event history:

- training event tensor width was zero;
- nonzero event-mask count was zero;
- the checkpoint has no trained history residual.

The function-preserving history/card-count upgrade correctly starts at exact
zero output. That preserves the parent function, but it also means the new
branch has learned nothing until trained on fresh history-bearing rows.

The current coherent Stage-C corpus does contain nonzero history, so the data
path can now support commissioning a repaired history encoder. Historical
zero-history rows cannot teach it.

### 3.3 Public card counting is useful but not a complete belief state

For two-player no-trade Catan, public resource conservation can often recover
the opponent's exact resource composition:

```text
starting supply - bank - actor hand = opponent resources
```

That is legitimate public inference, not hidden-truth leakage. Development-card
identity remains probabilistic.

The existing public-card residual implements useful conservation and
hypergeometric summaries, but it does not model dev-card age/playability or an
ordered, policy-conditioned belief over public history.

## 4. Value-learning diagnosis

### 4.1 The value network exists

The current model uses a scalar MLP over the shared state representation:

```text
shared state CLS -> Linear -> GELU -> Dropout -> Linear(1)
```

Training uses terminal winner outcomes in the acting-player perspective.
Perspective sign handling, terminal/truncated handling, and distributed weighted
reductions were audited and are not the current bug.

### 4.2 Policy learning overwhelms value correction

The current selected settings are approximately:

| Component | Effective scale |
|---|---:|
| policy loss | `1.0` |
| value loss | `0.25` |
| value-to-trunk gradient scale | `0.1` |
| value-head LR multiplier | `0.3` |
| shared-trunk LR multiplier | `1.0` |

The value objective therefore contributes approximately `0.025` corrective
scale into the shared trunk while policy distillation moves it at full scale.
`value_trunk_grad_scale` weakens value's upstream gradient; it does not protect
the trunk from policy gradients.

Observed learner fingerprints match this failure:

- teacher-policy KL improves monotonically;
- value MSE improves briefly, then degrades;
- value functional drift grows;
- the corresponding candidates lose externally.

This explains how a candidate can imitate search more closely while becoming a
weaker searched agent.

### 4.3 Opening value is weak

The exact-v5 opening slice has low outcome correlation and compressed
discrimination. A balanced opening value near zero is not itself a defect. The
defect is poor ranking/calibration across different opening positions.

### 4.4 Forced policy rows and forced value rows must be separated

Zero policy weight for a one-legal-action prompt is correct. Such a row cannot
teach action selection.

It does not follow that its state value is unimportant:

- pre-roll value is the expectation over dice outcomes;
- end-turn value is the position handed to the opponent;
- discard and Road Building continuation states remain strategically distinct.

The current `ROLL=0.25` and `END_TURN=0.1` value weights are unproven bundled
treatments. Restore them to `1.0` until a phase-calibrated, gradient-matched
ablation demonstrates a benefit.

### 4.5 Search-derived value targets are currently worse

The existing matched-root audit shows terminal value targets outperform the
current search-root/Q estimates:

| Readout | MSE |
|---|---:|
| raw network against terminal result | `0.5826` |
| search root against terminal result | `0.6006` |

Approximately 78% of audited completed-Q spreads lie below the estimated
visit-noise floor. Root-Q blending, TD-style self-distillation, or unqualified
search-value targets should not replace terminal outcomes yet.

### 4.6 Train/search scalar readout differs

Training fits an unbounded scalar to `[-1, 1]`; search applies `tanh`. Exact-v5
holdout slightly favors raw/clip over tanh. This is a real mismatch, but smaller
than the state-alias and trunk-interference defects.

## 5. Search-target diagnosis

### 5.1 Search surprise is not target reliability

A large divergence between parent policy and search target can mean either:

- search found a genuine policy error;
- noisy Q estimates were rescaled into false confidence.

The current surprise sampler cannot distinguish them.

High-surprise roots should be qualified using repeated search, Q-margin
uncertainty, symmetry disagreement, or a learned reliability predictor. Noisy
high-surprise rows should become lower-weight policy rows, value-only rows, or
reanalysis candidates.

### 5.2 Current D1/noise-floor evidence is actionable

The Stage-C root audit found many completed-Q spreads below the calibrated noise
floor while D1 was disabled. Tiny Q differences can become near-deterministic
targets after min-max transformation.

The next production teacher must bind its denoising/reliability rule into the
target contract. More simulations alone do not fix an unqualified target
transform.

### 5.3 Parent, producer, learner, and evaluator must be identical

Some historical Stage-C learners initialized from f7 while the producer/current
incumbent was recovered v5, then were evaluated against v5. Those were not clean
tests of improvement from the current parent.

The final replication path now correctly requires exact current-parent binding.
That invariant must become universal:

```text
learner initializer SHA
teacher checkpoint SHA
producer checkpoint SHA
evaluation incumbent SHA
```

must resolve to the intended experiment identity.

## 6. PPO judgment

PPO is viable; it is not a substitute for fixing the shared information state.

The repository already contains:

- entity-graph PPO updates;
- legal-action masking;
- GAE;
- value clipping;
- forced-action policy exclusion;
- KL controls and a frozen BC anchor;
- distributed actor/learner shards;
- seat-balanced opponent sampling.

Two software hazards remain:

1. `tools/train_selfplay_gpu.py` primarily constructs the older candidate
   architecture and is not the canonical current-champion path.
2. The distributed policy loader defaults to `xdim_graph`; current v5 is
   `entity_graph`. The architecture must be explicit and fail closed.

PPO should be commissioned after the P0 observation/value repairs:

```text
corrected parent
  -> coherent-search distillation
  -> short terminal-return PPO phase
  -> frozen seat-balanced opponent league
  -> parent-KL controlled update
```

Recommended initial contract:

- exact corrected entity-graph parent;
- true terminal reward;
- `gamma=1.0`;
- GAE lambda `0.95` to `0.98`;
- PPO clip about `0.1`;
- adaptive/early-stop parent KL around `0.005` to `0.01`;
- two to four update epochs;
- low shared-trunk LR or split late towers;
- legal mask included in both behavior and learner distributions;
- frozen opponent identity within each rollout batch;
- no stale rollout reuse outside the accepted KL/version window.

PPO is also a useful diagnostic:

- PPO improves while distillation fails: inspect teacher targets/operator.
- PPO and distillation both fail with poor critic calibration: inspect
  representation/value.
- PPO beats one opponent but not the league: opponent overfitting.

## 7. Repair work packages

Work packages are deliberately separated so agents can work concurrently
without rewriting one another's files.

### WP1: public rule state

Primary ownership:

- `native/catanatron-rs/src/lib.rs`
- `src/catan_zero/search/neural_rust_mcts.py`
- `src/catan_zero/rl/entity_token_features.py`
- feature schema/config metadata

Deliver:

- actor `has_played_dev_this_turn`;
- per-type own new/playable dev counts or exact `owned_at_start`;
- Road Building active;
- free roads remaining;
- discards remaining;
- unsaturated exact public played-dev counts.

Use a versioned, function-preserving residual so the existing checkpoint loads
with exact zero-step parity.

### WP2: action semantics and value affordances

Primary ownership:

- structured action translation;
- legal action token/context construction;
- `src/catan_zero/rl/entity_token_policy.py`.

Deliver:

- Monopoly resource identity;
- Year-of-Plenty resource-pair identity;
- structured action-to-resource and action-to-target binding;
- legal-action/affordance summary available to value;
- function-preserving initialization.

### WP3: ordered public history

Primary ownership:

- native action records;
- `src/catan_zero/rl/meaningful_history.py`;
- event feature construction;
- history encoder.

Deliver:

- monotonic turn/sequence identity;
- public target entity IDs;
- public resource payloads;
- dev-purchase timing;
- ordered causal encoder instead of unordered mean pooling;
- actor-information-set invariance.

### WP4: value/trunk repair

Primary ownership:

- `src/catan_zero/rl/entity_token_policy.py`;
- `tools/train_bc.py`;
- one-dose learner configuration.

Deliver:

- restore forced-state value weights to `1.0` control;
- lower trunk LR arms (`1.0`, `0.25`, `0.10`);
- split late policy/value tower option;
- parent-policy trust region;
- policy/value gradient norm and cosine telemetry at cadence no greater than
  eight learner steps;
- matched train/search scalar readout.

Keep terminal outcomes as the primary value target.

### WP5: reliable coherent teacher

Primary ownership:

- Gumbel target transform;
- target-quality audit;
- sampler weights;
- target contract/provenance.

Deliver:

- exact parent/operator binding;
- D1/noise-floor treatment;
- duplicate-search reliability on an audit fraction;
- surprise multiplied by reliability rather than surprise alone;
- policy-target eligibility tied to the exact search/operator hash.

### WP6: canonical PPO lane

Primary ownership:

- `src/catan_zero/rl/ppo_policy_factory.py`;
- `tools/run_local_entity_ppo_shards.py`;
- `tools/ppo_distributed_learner.py`;
- obsolete launcher retirement or hard failure.

Deliver:

- one entity-graph-only production/R&D entry point;
- exact parent and KL anchor binding;
- terminal `gamma=1` contract;
- frozen opponent league per rollout version;
- explicit stale-policy/V-trace policy;
- shared metrics with the distillation learner.

### WP7: action-local topology ceiling

Start only after WP1-WP5 are stable.

Deliver:

- canonical vertex/edge/hex identity;
- edge endpoint/incidence consumption;
- direct legal action-to-target join;
- late policy/value tower split;
- D6 consistency.

Prior gather/topology arms tied or regressed, so this must be commissioned from
the corrected exact-v5 parent and fresh information-complete data, not assumed
to win.

## 8. Implementation order

```text
WP1 public rule state ─┐
WP2 action semantics ──┼─> function-preserving corrected parent
WP3 ordered history ───┘
             |
             v
WP4 value/trunk repair + WP5 reliable teacher
             |
             v
independent exact-parent distillation arms
             |
             v
short WP6 PPO diagnostic/finisher
             |
             v
searched parent and population evaluation
             |
             v
only then: next large self-play wave or WP7 architecture scale
```

## 9. First experiment after implementation

Every arm independently reloads the exact corrected parent with fresh Adam.
No candidate chaining.

1. Corrected parent plus coherent/reliable n128 targets.
2. Trunk LR `1.0`, `0.25`, and `0.10`.
3. Late split policy/value tower.
4. Checkpoints selected by:
   - parent-policy KL;
   - value calibration by phase;
   - trunk/value drift;
   - fixed-root teacher agreement.
5. Search both candidate and parent with the same operator.
6. Run one short PPO arm from the best corrected distillation checkpoint.

The experiment is meant to identify a winning learner architecture, not to
certify a production champion.

## 10. Explicit non-bugs and rejected shortcuts

Confirmed non-bugs:

- acting-player value sign;
- terminal outcome perspective;
- truncated-game handling;
- distributed weighted reductions;
- fresh optimizer/no candidate chaining in the current final-replication path;
- current train/serve public masking in the audited paths.

Do not spend the next cycle on:

- global n256 before teacher reliability is repaired;
- a larger network before information-state aliases are fixed;
- root-Q value blending with the current noisy search values;
- PPO from scratch;
- exposing authoritative opponent hidden truth;
- another giant data wave using the incomplete schema.

## 11. Source references

Project evidence:

- `docs/audits/CATAN_RL_EXPERT_DESIGN_AUDIT_20260714.md`
- `docs/audits/A1_LEARNER_END_TO_END_FORENSICS_20260713.md`
- `docs/audits/A1_PAID_SEARCH_TARGET_SEMANTICS_20260712.md`
- `docs/audits/A1_TOPOLOGY_GATHER_CAUSAL_AUDIT_20260713.md`
- `docs/research/CATAN_ZERO_PROJECT_META_ANALYSIS_20260714.md`
- `configs/operations/a1-next-wave-coherent-public-v1/science.contract.json`

Primary external references:

- AlphaZero: <https://arxiv.org/abs/1712.01815>
- KataGo: <https://arxiv.org/abs/1902.10565>
- PPO: <https://arxiv.org/abs/1707.06347>
- Catan cross-dimensional actor-critic:
  <https://arxiv.org/abs/2008.07079>
- Open-source Catan PPO:
  <https://github.com/henrycharlesworth/settlers_of_catan_RL>
- ReBeL: <https://arxiv.org/abs/2007.13544>
- Student of Games: <https://arxiv.org/abs/2112.03178>

## 12. Collaboration rules

- Branch from the canonical collaboration branch named in the associated GitHub
  handoff.
- Claim one WP and its primary file ownership before editing.
- Do not change fleet state or launch a learner as part of WP1-WP6.
- Preserve exact parent load parity for every new residual/module.
- Bind every new feature surface to a schema and checkpoint metadata version.
- Do not merge a treatment into the production science contract until its
  independent arm produces evidence.
- Write the result and remaining uncertainty back into this document or a
  linked evidence artifact.
