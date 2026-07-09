# Colonist-Like RL Gym PRD

## Objective

Build a local Catan Gym that matches Colonist ranked 4-player base-game
dynamics closely enough for RL training and evaluation. The goal is not a visual
clone. The goal is a faithful game-mechanics, negotiation, observation, timing,
and replay surface for a specialist Catan agent.

Primary public references:

- Colonist base rules: https://colonist.io/catan-rules
- Colonist trade-system design notes: https://blog.colonist.io/improving-the-colonist-trade-system/
- Colonist timer update notes: https://blog.colonist.io/updated-timers/
- Colonist overview: https://colonist.io/discord/web

## Product Boundary

In scope:

- Base 4-player Catan to 10 victory points.
- Colonist-like player trade workflow.
- Public table chat as strategic negotiation context.
- Hidden-information constraints.
- Timers only where they affect action availability or auto-actions.
- Deterministic replay and training logs.
- Human/LLM language adapters only as wrappers around structured intents.

Out of scope:

- Copying Colonist visual assets, layout, trade dress, icons, or proprietary UI.
- Stealth browser automation or live ranked automation without permission.
- General-purpose chat moderation/reporting UI.
- Social/product features that do not alter training dynamics.
- Expansions until base 4-player is certified.

## Required Game Flow

The Gym must support:

- Room/game configuration:
  - four players,
  - base map,
  - 10 victory points,
  - standard resource/development decks,
  - standard Longest Road and Largest Army,
  - standard robber/discard behavior,
  - no house rules in the default benchmark.
- Initial build:
  - snake-order two settlements and two roads,
  - second settlement resource payout,
  - distance rule and road-adjacency constraints.
- Turn flow:
  - roll dice,
  - distribute resources,
  - handle 7 discard/robber/steal,
  - build, buy, trade, play eligible development card, end turn.
- End state:
  - first player to 10 points wins when legally reaching the win condition.

## Trading Requirements

Colonist's public trade-system notes identify open-ended trades, counteroffers,
response clarity, bank-trade clarity, and embargo/blocking as major trade
interface concerns. For the RL Gym, this becomes the following mechanics.

### Concrete Board Trades

The env must expose legal concrete trades:

- offer exact resources for exact resources,
- accept/reject,
- active player confirms one accepting counterparty,
- active player cancels,
- bank/port maritime trades.

Concrete resource exchanges must be legal under current hidden/private state.

### Negotiation Workflow

The env must also expose pre-trade negotiation state:

- exact proposal: `give brick -> want ore`,
- open-ended give: `give brick -> want any 1`,
- open-ended ask: `give any 1 -> want ore`,
- wildcard proposal: `give 1 of ore/wheat -> want brick`,
- public responses: accepted, rejected, countered,
- linked counteroffers,
- materialization of open/wildcard offers into exact concrete trade actions.

The current implementation supports this as public side-channel state and can
resolve exact materialized offers into board-trade actions for the learning
seat.

### Trade Response Visibility

Training logs must retain:

- offer id,
- actor,
- target or broadcast,
- give/want sides,
- each visible response,
- counteroffer linkage,
- eventual concrete board action if one occurs.

If Colonist hides some response information from some players in live play, the
Gym must make that an observation-policy decision rather than losing the data.

## Chat and Language Requirements

Chat is in scope only as strategic public context.

The env must represent:

- free-text table message for human/LLM adapters,
- structured intents for:
  - trade request,
  - open-ended offer,
  - counteroffer,
  - leader blocking,
  - robber negotiation/extortion,
  - social acknowledgements that appear in logs but do not affect rules.
- message caps to prevent infinite communication actions,
- public chat/event log in `info`,
- deterministic replay of chat messages.

The policy should train primarily on structured intents. Free text can be
generated or parsed by an LLM later, but the LLM must not directly emit illegal
game actions.

## Observation Requirements

Per-player observations may include:

- public board,
- public player points/counts/pieces,
- own resources,
- own development cards,
- public event log,
- public chat and negotiation log under that player's visibility policy,
- current legal action mask.

Per-player observations must not include:

- opponent resource identities,
- opponent development-card identities,
- development deck order,
- future dice,
- future robber steal result,
- hidden acceptance state unless the live-visible policy exposes it.

Tests must verify information-set equivalence: changing hidden opponent cards
without changing public history cannot change actor inputs or legal logits.

## Timer Requirements

Timers are not modelled for UI fidelity. They matter only when they alter the
training distribution.

The Gym should eventually support:

- configurable game-speed profile,
- longer initial-placement budget,
- shorter roll/robber/discard budgets,
- auto-actions on timeout:
  - auto-roll,
  - auto-discard,
  - auto-reject/expire trade,
  - auto-end where appropriate.

The default benchmark may disable wall-clock timers while still logging a
virtual deadline profile for human-parity experiments.

## Gym/API Requirements

The env must expose:

- `reset(seed=...)`,
- `step(action)` for concrete legal board actions,
- all-seat observations from the multi-agent env,
- `valid_actions()` and `action_mask()`,
- `post_chat()` and `post_chat_template()` side-channel calls,
- `propose_trade()`, `respond_to_trade()`, `counter_trade()`,
- `trade_action_for_offer()` and `step_negotiated_trade()` when an offer is
  materialized into exact resources,
- `timer_info()`, `timeout_action()`, and `step_timeout()` for virtual
  Colonist-style time-control experiments,
- `ColonistAECEnv` for PettingZoo-style turn-based self-play loops:
  - `possible_agents`,
  - `agent_selection`,
  - `observe(agent)`,
  - `last()`,
  - `step(action)`,
  - `agent_iter()`,
- public `info` fields:
  - `valid_actions`,
  - `action_mask`,
  - `chat_log`,
  - `valid_chat_templates`,
  - `negotiation_offers`,
  - `open_negotiation_offers`,
  - `timer`,
  - trade-offer cap state.

The single-seat wrapper remains useful for baseline PPO plumbing, but serious
self-play should target `ColonistMultiAgentEnv`, where every seat can use the
same action, chat, negotiation, and timer surfaces.

## Acceptance Tests

Minimum test gates:

- action mask length always equals action space size,
- every valid action index maps to `True` in the mask,
- structured domestic trade actions become valid during normal play,
- chat messages do not change board legal actions,
- chat caps are enforced,
- exact/open/wildcard negotiation offers are public and replayable,
- counteroffers link to parent offers,
- resolved open/wildcard offers can become legal concrete board trades,
- timer phase and timeout fallback are exposed in `info`,
- timeout fallback action is always legal when available,
- hidden opponent resources/development cards do not leak into actor features,
- reset with the same seed reproduces observations and legal masks,
- replay reconstructs board actions, chat, and negotiation events.
- AEC wrapper exposes only the current player's legal board actions while
  non-current agents receive empty masks.

## Current Implementation Status

Implemented:

- Catanatron-backed 4-player single-learning-seat Gym wrapper.
- Catanatron-backed 4-player current-player multi-agent env.
- PettingZoo-style AEC adapter.
- Expanded domestic trade action space.
- Per-turn concrete trade-offer cap.
- Strategic chat side channel.
- Exact/open/wildcard negotiation offers.
- Public responses and counteroffers.
- Resolution of materialized offers into concrete board-trade actions.
- Colonist-style virtual timer metadata.
- Deterministic timeout fallback actions.

Not done:

- Dedicated RLlib `MultiAgentEnv` subclass if RLlib becomes the trainer target.
- Full Colonist visibility policy for every trade response.
- Wall-clock timer enforcement and disconnect handling.
- Clean-room simulator/license-safe deployment path.
- Differential rule fixtures against independent Colonist-style logs.
- Natural-language adapter over structured intents.
