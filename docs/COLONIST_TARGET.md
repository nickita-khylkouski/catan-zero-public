# Colonist-Target Gym Contract

The immediate engineering goal is a local Gym environment that is close enough
to Colonist ranked base Catan that training failures transfer usefully.

## Current Coverage

- Four-player base Catan to ten victory points.
- True four-seat multi-agent environment:
  - no opponent auto-play,
  - every current-player decision is surfaced to the trainer,
  - each seat can use the same board action, chat, negotiation, and timer APIs.
- PettingZoo-style AEC adapter:
  - `possible_agents`,
  - `agent_selection`,
  - `observe(agent)`,
  - `last()`,
  - `step(action)`,
  - `agent_iter()`.
- Initial placements and roads.
- Dice production and robber movement.
- Player-chosen discard actions.
- Victim selection and hidden steals.
- Building roads, settlements, and cities.
- Buying and playing development cards.
- Maritime trades.
- Structured domestic player trades:
  - offer resource bundle for resource bundle,
  - target specific allowed responders,
  - accept,
  - reject,
  - confirm one accepting counterparty,
  - cancel.
- Public strategic chat side channel:
  - free-text messages for future human/LLM experiments,
  - structured negotiation templates,
  - trade request, open-ended trade, counteroffer, leader-blocking, and robber
    negotiation intents,
  - public chat log exposed in Gym `info`,
  - per-turn message cap so chat cannot become an infinite action channel.
- Colonist-style trade negotiation state:
  - exact resource proposals,
  - open-ended "make an offer" sides,
  - wildcard sides such as one of ore/wheat,
  - public response statuses,
  - linked counteroffers.
- Colonist-style trade panel snapshots:
  - eligible responders for each offer,
  - waiting, accepted, rejected, and countered players,
  - proposer-side confirm/cancel affordances,
  - responder-side accept/reject/counter affordances,
  - currently resolving concrete board trade, if any.
- Expanded strategic chat templates:
  - concise accept/reject/counter/trade-status phrases,
  - short reactions,
  - leader-blocking,
  - robber/no-steal/extortion negotiation.
- Proposal-to-action resolution:
  - exact accepted negotiation offers can resolve into concrete legal
    Catanatron trade actions,
  - open/wildcard offers require exact bundles before execution.
- Colonist-style virtual timers:
  - initial-build, roll, robber/discard, trade-response, and main-turn phases,
  - speed profiles based on public Colonist timer notes where available,
  - deterministic timeout fallback actions for RL/time-control experiments.
- Per-turn player-trade offer cap in `CatanZeroGymConfig`.
- Action masks with every valid action represented by an integer index.
- Structured action API:
  - `structured_legal_actions` mirrors `valid_actions` exactly,
  - each action carries `index`, `action_type`, `category`, normalized `args`,
    human-readable `label`, and raw descriptor,
  - `step_structured_action(action)` executes a currently legal structured
    action through the same integer mask path,
  - trade actions expose normalized give/want bundles for player trades and
    bank/port trades.
- Safe Colonist-style observation API:
  - `observation_payload(player)` exposes public board/table state plus that
    player's own exact resources and development cards,
  - opponents expose hidden-card counts, public points, pieces, and played
    development cards, not exact hidden hands,
  - non-current players receive empty legal-action lists and masks,
  - public event logs redact hidden discard resources, robber-steal resources,
    and development-card-buy results.
- Replay trace for offline RL/imitation:
  - every reset, chat message, trade proposal, trade response, counteroffer,
    timeout, invalid action, and board action creates a frame,
  - each frame contains the redacted event, safe per-seat observation payloads,
    rewards, and terminal/truncation flags,
  - observation payloads inside replay frames omit recursive event logs.
- Replay JSONL export/import:
  - `write_replay_jsonl(path)` writes one redacted replay frame per line,
  - `dump_replay_jsonl()` and `load_replay_jsonl()` support direct training
    pipeline use,
  - `tools/export_random_replay.py` generates local random-policy traces for
    smoke testing dataset ingestion.

## Known Gaps

- The multi-agent env has a PettingZoo-style AEC adapter, but not a dedicated
  RLlib `MultiAgentEnv` subclass yet.
- Chat is strategically represented, not visually cloned. We model negotiation
  signals that can affect training, but not general-purpose chat moderation,
  reporting UI, emojis, or social features.
- Negotiation offers are public side-channel state. Exact resolved offers can
  map into Catanatron's concrete board trade actions, but opponent-side
  response policy is still basic. Targeted concrete trades filter non-targeted
  responders to reject-only.
- Trade blocking/embargo is intentionally not a core RL mechanic yet. It can be
  added later if evidence shows it materially affects Colonist ranked play.
- Timers are virtual metadata and fallback policies, not wall-clock UI timers.
  Disconnect handling, ranking rules, and moderation constraints are not
  represented.
- Catanatron is GPL-licensed. It is fine as local research infrastructure, but
  final distribution and proprietary deployment need a license review or a
  clean-room simulator.
- Rule parity still needs differential fixtures against independently written
  Catan rules and logged Colonist games where legally available.

## Next Gym Work

1. Add an RLlib `MultiAgentEnv` adapter if the training stack standardizes on
   RLlib.
2. Convert the flat trade extension into a structured action head: action type,
   target player, give bundle, receive bundle, response.
3. Add a true Colonist trade-response API for all seats instead of relying on
   Catanatron bot enemies to accept/reject after a concrete offer is made.
4. Add exportable structured-action vocab stats for model heads: action type,
   target type, board target, resource bundle sizes, and response modes.
5. Expand public/private observation tests to cover deck order, future dice, and
   longer replay histories.
6. Add Colonist-style evaluation profiles: structured chat and trades first,
   then a separate language adapter after the specialist policy is strong.

Live Colonist ranking should be treated as a permissioned evaluation target.
Do not build stealth browser automation for ranked games.
