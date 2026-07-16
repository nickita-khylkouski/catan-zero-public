#!/usr/bin/env python3
"""CAT-57: our checkpoint as a real catanatron `Player`.

This is the credibility item flagged by R8 (see the Linear issue for the full
citation trail): every existing external-ladder script in this repo
(`tools/gumbel_search_vs_bot_h2h.py`, `tools/evaluate_scoreboard.py`'s
`catanatron_ab3` opponent, etc.) drives games either on OUR Rust engine with
catanatron mirrored in lockstep, or through `catan_zero.rl` envs that create
their own `Game`. None of them let our net play a match where catanatron's OWN
`Game.play()` loop is the one calling the shots end to end -- which is exactly
what "the number the outside world will judge" requires.

THE KEY REUSE THIS FILE LEANS ON
---------------------------------
`catan_zero.rl.multiagent_env.ColonistMultiAgentEnv` already contains the
entire catanatron-state -> entity-token translation used at training time
(`observation_payload()`, `_board_payload()`, `_player_payloads()`, the
`ActionCatalog` encode/decode table). It normally owns the `Game` it wraps
(built in `.reset()`). `bind_external_game()` below is the one new piece:
it points an otherwise-normal `ColonistMultiAgentEnv` at a `Game` we did NOT
create -- the one catanatron's own engine handed us via `Player.decide(game,
playable_actions)` -- so `build_entity_token_features()` and
`EntityGraphPolicy.select_action()` (both unmodified, both the exact
production code paths) work without any duplicate featurization logic.

This makes the "translation" this ticket asks for mostly a *binding*
problem, not a *reimplementation* problem -- with one real gap: the env's
event-history log is normally populated incrementally by its own `.step()`,
which we never call (catanatron's engine executes actions itself). See
`_synthetic_event_log` for how that's reconstructed, and its docstring for
the one deliberately-unfixed piece (turn-key ordinals).

MODES
-----
`CatanZeroNetPlayer` is the original raw-policy adapter.  Search is exposed as
the separate `CatanZeroSearchPlayer`: catanatron's native Python `Game.play()`
loop remains the referee and owns the real game, while a verified TOURNAMENT-
map Rust shadow is used only as the state passed to `GumbelChanceMCTS`.
Native `ActionRecord`s are replayed into that shadow with their *already
resolved* dice / robber-steal / development-card outcomes.  Every search call
checks legal-action and state parity and fails closed on a divergence; it does
not silently fall back to raw policy or pretend BASE-map RNG equivalence.
"""

# ruff: noqa: E402 -- this executable adds its sibling tools directory before imports.

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl._catanatron import import_catanatron_module
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv
from catan_zero.adapters.engine_equivalence import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    canonical_python_action_key,
    canonical_rust_action_key,
    diff_state_views,
    is_chance_action,
    legal_action_diff,
    python_state_view,
    raw_action_to_python_action,
    rust_legal_actions,
    rust_state_view,
    vendor_symbols,
)

# Importing this module registers the vendored catanatron tree on sys.path
# (see `catan_zero.rl._catanatron.ensure_catanatron`), so the direct
# `catanatron.models.player` import below resolves whether or not catanatron
# is pip-installed.
_player_module = import_catanatron_module("catanatron.models.player")
CatanatronPlayer = _player_module.Player
Color = _player_module.Color

STANDARD_COLOR_ORDER: tuple[str, ...] = ("BLUE", "RED", "ORANGE", "WHITE")

# Native catanatron ActionType names -> the (lowercased, for trade types)
# names ColonistMultiAgentEnv/entity_token_features.ACTION_TYPES use. See
# `ColonistMultiAgentEnv._decode_action`'s `_extended_actions` entries, which
# store these same lowercase kinds for the trade-negotiation action family.
_NATIVE_TO_EVENT_ACTION_TYPE: dict[str, str] = {
    "OFFER_TRADE": "offer_trade",
    "ACCEPT_TRADE": "accept_trade",
    "REJECT_TRADE": "reject_trade",
    "CONFIRM_TRADE": "confirm_trade",
    "CANCEL_TRADE": "cancel_trade",
}

SEARCH_MODE_TODO = """\
Use CatanZeroSearchPlayer for full-search native-harness games. Search cannot
be enabled by changing CatanZeroNetPlayer.mode because it additionally needs
the seating-aligned TOURNAMENT-map Rust shadow and a configured
GumbelChanceMCTS instance created by the match runner. BASE-map search is
intentionally unsupported: the Python and Rust map-shuffle RNGs do not match.
"""


class SearchEngineBoundaryError(RuntimeError):
    """The native Python referee and the Rust search shadow diverged.

    A neutral-harness result is invalid after this boundary fails.  Callers
    must record/exclude the game, never downgrade to raw policy.
    """


def standard_colors(num_players: int) -> tuple[Any, ...]:
    """Canonical BLUE/RED/ORANGE/WHITE color prefix.

    Both the bridging env (`make_bridge_env`) and whatever creates the real
    `Game` (a match runner, a test) MUST use this same prefix/order for the
    same player count: `ActionCatalog` is keyed by the exact color set
    (`action_mask.ActionCatalog.__init__`), not just the count, so a mismatch
    would silently decode actions for the wrong seats.
    """
    if not 2 <= num_players <= 4:
        raise ValueError("num_players must be between 2 and 4")
    colors = [Color[name] for name in STANDARD_COLOR_ORDER]
    return tuple(colors[:num_players])


def default_bridge_config(
    num_players: int,
    *,
    map_type: str = "BASE",
    vps_to_win: int = 10,
    max_player_trade_offers_per_turn: int = 10_000,
) -> ColonistMultiAgentConfig:
    """The `ColonistMultiAgentConfig` used for every bridging env.

    Chat/negotiation/timers only exist for the colonist.io-flavoured
    self-play variant; a real catanatron match has none of that, so they're
    disabled. `max_player_trade_offers_per_turn` is left effectively
    uncapped because we never update the env's own offer-count bookkeeping
    for an externally-driven game (see `bind_external_game`) -- the real
    trade legality gate is catanatron's own `is_valid_action`, checked
    separately by `CatanZeroNetPlayer._is_legal`.

    Exposed as its own function (rather than inlined in `make_bridge_env`)
    so tests/tools that build an `EntityGraphPolicy` for a bridging env
    (e.g. via `EntityGraphPolicy.create(env_config=...)`) can use the exact
    same config and be sure `action_space.n` / `static_action_features`
    line up with what `CatanZeroNetPlayer` will build at inference time.
    """
    return ColonistMultiAgentConfig(
        map_type=map_type,
        players=num_players,
        vps_to_win=vps_to_win,
        enable_table_chat=False,
        enable_timers=False,
        max_player_trade_offers_per_turn=max_player_trade_offers_per_turn,
    )


def make_bridge_env(
    num_players: int,
    *,
    map_type: str = "BASE",
    vps_to_win: int = 10,
    max_player_trade_offers_per_turn: int = 10_000,
) -> ColonistMultiAgentEnv:
    """Construct (but do not `.reset()`) an env sized for an external game."""
    return ColonistMultiAgentEnv(
        default_bridge_config(
            num_players,
            map_type=map_type,
            vps_to_win=vps_to_win,
            max_player_trade_offers_per_turn=max_player_trade_offers_per_turn,
        )
    )


def _color_name(color: Any) -> str:
    return getattr(color, "name", str(color))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "name") and hasattr(
        value, "value"
    ):  # enum-like (Color, Resource, ...)
        return value.name
    return value


def _reencode_action_index(env: ColonistMultiAgentEnv, action: Any) -> int | None:
    """Best-effort: recover the numeric action-index a past native action
    would have had in `env`'s space, for the `action_id` event-token feature
    (`entity_token_features._event_action_id` -> slot 35), via the same
    `ActionCatalog` used for live decoding.

    Three categories of native action are not emitted this way. Most are not
    recoverable and `try_encode` correctly misses on them (returning the same safe `None`
    default the live path already falls back to -- see
    `ColonistMultiAgentEnv.valid_actions`'s `_trade_response_indices_for`
    fallback):

      - Trade-negotiation actions (OFFER/ACCEPT/REJECT/CANCEL/CONFIRM_TRADE):
        `env._extended_actions` enumerates COLONIST's own pre-generated offer
        combos, which a native vanilla OFFER_TRADE's exact freqdeck value
        need not match; ACCEPT/REJECT/CANCEL/CONFIRM_TRADE are keyed
        dynamically off `state.current_trade` (only meaningful for the
        CURRENT trade, not a past one).
      - Publicly-redacted discard actions (DISCARD_RESOURCE): its otherwise
        recoverable catalog id uniquely identifies the hidden discarded
        resource, so emitting it would undo the value redaction through event
        token slot 35, so this case is deliberately suppressed even though it
        is technically recoverable.
      - Chance-resolved actions (ROLL, BUY_DEVELOPMENT_CARD): `ActionCatalog`
        only has ONE generic pre-resolution entry for each
        (`action_mask._action_array`: `(ActionType.ROLL, None)`,
        `(ActionType.BUY_DEVELOPMENT_CARD, None)`), but the EXECUTED action
        on `state.action_records` carries the RESOLVED value (the actual
        dice / the actual card drawn) filled in by `apply_roll`/
        `apply_buy_development_card` -- which never matches that generic
        catalog key. `MOVE_ROBBER` is NOT in this category: its catalog
        entries are already parametrized by `(coordinate, victim)`, so a
        resolved MOVE_ROBBER (a player CHOICE, not a chance outcome) encodes
        the same as any other choice-driven action.
    """
    if str(getattr(action.action_type, "name", action.action_type)) == "DISCARD_RESOURCE":
        return None
    index = env.action_catalog.try_encode(action)
    return int(index) if index is not None else None


def _synthetic_event_log(
    env: ColonistMultiAgentEnv, game: Any, *, history_limit: int
) -> list[dict[str, Any]]:
    """Rebuild recent-event tokens from a `Game` we did not create.

    `ColonistMultiAgentEnv` normally builds `_event_log` incrementally as its
    OWN `.step()` executes actions. Binding an externally-progressing
    catanatron `Game` skips `.step()` entirely -- catanatron's own engine
    calls `Player.decide()`, not us -- so without this the net would see an
    empty event history for the ENTIRE match, even deep into it, despite
    being trained with populated recent-move context.

    Reconstructed from `game.state.action_records`, the same source
    `catan_zero.adapters.catanatron.CatanatronAdapter.event_log()` uses for
    an analogous purpose on the `CatanEngine` boundary.

    One deliberately-unfixed gap: `_event_tokens` also encodes a `turn_key`
    ordinal (`entity_token_features.py`, slots 15/16) meant to be
    `(state.num_turns, state.current_turn_index)` AT RECORD TIME. We don't
    have that per-record (only the CURRENT turn is on `state`), so `turn_key`
    is left `None` (a safe default -- see `_event_tokens`: `event.get(
    "turn_key") or (0, 0)`). The primary recency signal (position in this
    slice, `event_tokens[:, 1]`) IS exact, and the action-id signal (slot 35)
    is exact for every action type `_reencode_action_index` covers. See
    `tests/test_catanatron_player_adapter.py::
    test_bind_external_game_matches_self_created_features` for the
    regression test this asymmetry is carved out of.
    """
    # Build the FULL history (reset event first, exactly as `.reset()` would
    # have recorded it, then every action taken so far) and truncate LAST --
    # `_event_tokens` truncates the same way (`event_log[-history_limit:]`),
    # so truncating before prepending the reset event would have shifted
    # every subsequent event by one slot relative to a self-created env once
    # the game runs past `history_limit` actions.
    events: list[dict[str, Any]] = [
        {
            "event_id": 0,
            "event_type": "reset",
            "turn_key": None,
            "actor": None,
            "payload": {},
        }
    ]
    for record in game.state.action_records:
        action = record.action
        native_type = str(action.action_type.name)
        action_index = _reencode_action_index(env, action)
        action_value = _jsonable(action.value)
        result = _jsonable(record.result)
        if native_type == "BUY_DEVELOPMENT_CARD":
            action_value = "hidden_development_card"
            result = "hidden_development_card"
            action_index = None
        elif native_type == "DISCARD_RESOURCE":
            action_value = "hidden_resource"
            result = "hidden_resource"
        elif native_type == "MOVE_ROBBER" and result is not None:
            # The robber destination and chosen victim are public and remain
            # in action_value. The stolen resource identity is authoritative
            # hidden truth and must not cross into the model event payload.
            result = "hidden_stolen_resource"
        events.append(
            {
                "event_id": len(events),
                "event_type": "board_action",
                "turn_key": None,
                "actor": _color_name(action.color),
                "payload": {
                    "action_index": action_index,
                    "action": {
                        "index": action_index,
                        "action_type": _NATIVE_TO_EVENT_ACTION_TYPE.get(
                            native_type, native_type
                        ),
                        "value": action_value,
                    },
                    "result": result,
                    "next_player": None,
                },
            }
        )
    return events[-history_limit:]


def bind_external_game(
    env: ColonistMultiAgentEnv, game: Any, *, history_limit: int = 64
) -> None:
    """Point `env` at an externally-owned, in-progress catanatron `Game`.

    Bypasses `.reset()` (which would create its own `Game` and dummy
    `Player`s) and instead:
      - sets `env.game` directly, after asserting the color set matches what
        `env` was built for (see `standard_colors`'s docstring for why a
        mismatch is dangerous rather than just wrong);
      - resets the small set of env-owned bookkeeping fields `.reset()`
        normally resets for a brand-new game (nothing here is safe to carry
        over from a PRIOR bound game, since it's a different `Game` object);
      - rebuilds the recent-event log from the game's own action history
        (see `_synthetic_event_log`) instead of leaving it permanently empty.

    Safe to call repeatedly on the same `env` for a whole match (once per
    `decide()` call) -- it's O(history_limit) per call, not O(game length).
    """
    # SET equality, not order: catanatron's own `State.__init__` shuffles
    # turn order from the seed, so `game.state.colors` is seated in whatever
    # order that seed produced -- NOT necessarily the order colors were
    # passed to `Game(players=...)`. Every color lookup this env does
    # (`_color_for_name`, `ActionCatalog.decode`/`try_encode`) is by color
    # VALUE (an enum singleton) or NAME, never by position in this tuple, so
    # only the SET has to match.
    env_colors = frozenset(_color_name(color) for color in env.player_colors)
    game_colors = frozenset(_color_name(color) for color in game.state.colors)
    if env_colors != game_colors:
        raise ValueError(
            "bind_external_game: env was built for colors "
            f"{sorted(env_colors)} but the external game uses "
            f"{sorted(game_colors)}. Build both from "
            "catanatron_player_adapter.standard_colors(n) (and the SAME n) "
            "to keep them in sync -- ActionCatalog is keyed by the exact "
            "color set, not just the player count."
        )
    env.game = game
    env.invalid_actions_count = 0
    env._trade_offer_counts = {}
    # `observation_payload()` only ever reads `len(self._replay_frames)`
    # (the "replay_frame_count" global-token feature, a monotonic
    # how-far-into-the-game signal -- see `entity_token_features.
    # _global_tokens`, slot 25) never its CONTENTS, so a length-matched list
    # of placeholders reproduces that feature correctly without building
    # real replay frames. `1 +` accounts for the "reset" event `.reset()`
    # always records before any board action. This undercounts relative to
    # a colonist-flavoured self-play game with chat/negotiation enabled
    # (each of those records its own frame too, and a real catanatron match
    # has neither) -- a distributional mismatch versus training, not a
    # correctness bug, and the closest true analogue available.
    # NOTE: since these are placeholders, `env.replay_trace()` /
    # `env.write_replay_jsonl()` must not be called on a bridging env.
    env._replay_frames = [None] * (1 + len(game.state.action_records))
    env._pending_trade_allowed_responders = None
    env._current_trade_allowed_responders = None
    env._event_log = _synthetic_event_log(env, game, history_limit=history_limit)
    env._next_event_id = len(env._event_log) + 1


_POLICY_CACHE: dict[tuple[str, str], EntityGraphPolicy] = {}
_POLICY_CACHE_LOCK = threading.Lock()


def _load_policy_cached(checkpoint: str, device: str) -> EntityGraphPolicy:
    """Share one loaded checkpoint across `CatanZeroNetPlayer` instances.

    A match runner naturally builds a fresh Player per game (catanatron's own
    `Player.reset_state()` contract assumes long-lived instances across many
    games, but games in this harness are cheap to set up from scratch); this
    avoids re-reading the checkpoint from disk for every game in a batch.
    """
    key = (str(checkpoint), str(device))
    with _POLICY_CACHE_LOCK:
        cached = _POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    policy = EntityGraphPolicy.load(checkpoint, device=device)
    with _POLICY_CACHE_LOCK:
        _POLICY_CACHE[key] = policy
    return policy


class CatanZeroNetPlayer(CatanatronPlayer):
    """Catanatron `Player` backed by a CatanZero checkpoint's raw policy.

    Featurizes directly from catanatron's own `Game`/`State` every
    `decide()` call -- no shadow engine, no lockstep mirroring (see module
    docstring) -- and falls back to `playable_actions[0]` (never a crash,
    never a silently-illegal move) if the policy's pick doesn't survive
    catanatron's own `is_valid_action` check. `self.stats
    ["illegal_policy_picks"]` surfaces how often that fallback fires, so a
    translation bug shows up as a visible counter in match output rather
    than a quietly-degraded win rate.
    """

    def __init__(
        self,
        color: Any,
        *,
        checkpoint: str | Path | None = None,
        policy: EntityGraphPolicy | None = None,
        device: str = "cpu",
        mode: Literal["raw_policy"] = "raw_policy",
        sample: bool = False,
        seed: int = 0,
        map_type: str = "BASE",
        vps_to_win: int = 10,
        history_limit: int = 64,
        max_player_trade_offers_per_turn: int = 0,
    ) -> None:
        super().__init__(color, is_bot=True)
        if mode != "raw_policy":
            raise NotImplementedError(SEARCH_MODE_TODO)
        if policy is None:
            if checkpoint is None:
                raise ValueError("CatanZeroNetPlayer requires checkpoint= or policy=")
            policy = _load_policy_cached(str(checkpoint), device)
        self.mode = mode
        self.sample = bool(sample)
        self.map_type = map_type
        self.vps_to_win = int(vps_to_win)
        self.history_limit = int(history_limit)
        # No-trade by default (mission spec + catanatron own bots never emit
        # OFFER_TRADE, since generate_playable_actions has no player-trade
        # branch; catanatron only accepts a proactively-constructed
        # OFFER_TRADE via is_valid_action). Leaving player-trade offers ON
        # here livelocks a greedy raw policy: bind_external_game resets the
        # per-turn offer counter every decide(), so the cap never accrues and
        # OFFER_TRADE/REJECT_TRADE can repeat forever without advancing
        # num_turns. 0 removes offer_trade indices from valid_actions.
        self.max_player_trade_offers_per_turn = int(max_player_trade_offers_per_turn)
        self._policy = policy
        self._rng = np.random.default_rng(seed)
        self._env: ColonistMultiAgentEnv | None = None
        self.stats: dict[str, int] = {"decisions": 0, "illegal_policy_picks": 0}

    def reset_state(self) -> None:
        # `env` is cheap to rebuild and its color set is tied to the PRIOR
        # game's seat count; don't assume the next game matches.
        self._env = None
        self.stats = {"decisions": 0, "illegal_policy_picks": 0}

    def decide(self, game: Any, playable_actions: Any) -> Any:
        playable_actions = list(playable_actions)
        if len(playable_actions) == 1:
            # No real decision, and sidesteps needing a legal-action-tokens
            # batch of size 1 through the net for a forced move.
            return playable_actions[0]

        env = self._env_for(game)
        bind_external_game(env, game, history_limit=self.history_limit)
        info = env.info()
        self.stats["decisions"] += 1

        native_action: Any | None
        try:
            action_index = self._policy.select_action(
                env, None, info, self._rng, training=self.sample
            )
            native_action = env._decode_action(action_index)
        except Exception:  # noqa: BLE001 - fall back rather than crash the match.
            native_action = None

        if native_action is None or not self._is_legal(
            env, game, native_action, playable_actions
        ):
            self.stats["illegal_policy_picks"] += 1
            native_action = playable_actions[0]
        return native_action

    def _env_for(self, game: Any) -> ColonistMultiAgentEnv:
        if self._env is None:
            self._env = make_bridge_env(
                len(game.state.colors),
                map_type=self.map_type,
                vps_to_win=self.vps_to_win,
                max_player_trade_offers_per_turn=self.max_player_trade_offers_per_turn,
            )
        return self._env

    @staticmethod
    def _is_legal(
        env: ColonistMultiAgentEnv,
        game: Any,
        native_action: Any,
        playable_actions: list[Any],
    ) -> bool:
        try:
            return bool(
                env.is_valid_action(playable_actions, game.state, native_action)
            )
        except Exception:  # noqa: BLE001 - be conservative: treat any check failure as illegal.
            return False


def _enum_name(value: Any) -> str:
    """Stable name for native enum/string chance outcomes."""
    return str(getattr(value, "name", value))


def _recorded_roll_outcome_index(rust_game: Any, raw_action: Any, result: Any) -> int:
    try:
        die_a, die_b = (int(value) for value in result)
    except (TypeError, ValueError) as error:
        raise SearchEngineBoundaryError(
            f"native ROLL record has invalid result {result!r}"
        ) from error
    total = die_a + die_b
    if not (1 <= die_a <= 6 and 1 <= die_b <= 6 and 2 <= total <= 12):
        raise SearchEngineBoundaryError(
            f"native ROLL record has impossible dice {result!r}"
        )
    spectrum = json.loads(rust_game.spectrum_json(json.dumps(raw_action)))
    outcome_index = total - 2
    if not 0 <= outcome_index < len(spectrum):
        raise SearchEngineBoundaryError(
            f"Rust ROLL spectrum cannot represent native dice {result!r}: {spectrum!r}"
        )
    return outcome_index


def _recorded_robber_outcome_index(rust_game: Any, raw_action: Any, result: Any) -> int:
    wanted_resource = _enum_name(result)
    if wanted_resource not in RESOURCES:
        raise SearchEngineBoundaryError(
            f"native MOVE_ROBBER record has invalid stolen resource {result!r}"
        )
    victim_name = str(raw_action[2][1])
    before = json.loads(rust_game.json_snapshot())
    colors = [str(color) for color in before["colors"]]
    try:
        victim_index = colors.index(victim_name)
    except ValueError as error:
        raise SearchEngineBoundaryError(
            f"Rust MOVE_ROBBER victim {victim_name!r} is absent from seats {colors!r}"
        ) from error
    before_hand = before["player_state"][victim_index]["resources"]
    spectrum = json.loads(rust_game.spectrum_json(json.dumps(raw_action)))
    matching: list[int] = []
    for outcome_index in range(len(spectrum)):
        candidate = rust_game.apply_chance_outcome(
            json.dumps(raw_action), outcome_index
        )
        after = json.loads(candidate.json_snapshot())
        after_hand = after["player_state"][victim_index]["resources"]
        stolen = [
            resource
            for resource in RESOURCES
            if int(after_hand[resource]) - int(before_hand[resource]) == -1
        ]
        clean = len(stolen) == 1 and all(
            int(after_hand[resource]) - int(before_hand[resource]) in (0, -1)
            for resource in RESOURCES
        )
        if clean and stolen[0] == wanted_resource:
            matching.append(outcome_index)
    if not matching:
        raise SearchEngineBoundaryError(
            "Rust MOVE_ROBBER spectrum cannot reproduce native stolen resource "
            f"{wanted_resource!r} for action {raw_action!r}"
        )
    return matching[0]


def _recorded_development_card_outcome_index(
    rust_game: Any, raw_action: Any, result: Any
) -> int:
    wanted_card = _enum_name(result)
    if wanted_card not in DEVELOPMENT_CARDS:
        raise SearchEngineBoundaryError(
            f"native BUY_DEVELOPMENT_CARD record has invalid card {result!r}"
        )
    actor_name = str(raw_action[0])
    before = json.loads(rust_game.json_snapshot())
    colors = [str(color) for color in before["colors"]]
    try:
        actor_index = colors.index(actor_name)
    except ValueError as error:
        raise SearchEngineBoundaryError(
            f"Rust BUY_DEVELOPMENT_CARD actor {actor_name!r} is absent from seats {colors!r}"
        ) from error
    before_cards = before["player_state"][actor_index]["dev_cards"]
    before_deck_count = int(before["development_deck_count"])
    spectrum = json.loads(rust_game.spectrum_json(json.dumps(raw_action)))
    matching: list[int] = []
    for outcome_index in range(len(spectrum)):
        candidate = rust_game.apply_chance_outcome(
            json.dumps(raw_action), outcome_index
        )
        after = json.loads(candidate.json_snapshot())
        after_cards = after["player_state"][actor_index]["dev_cards"]
        gained = [
            card
            for card in DEVELOPMENT_CARDS
            if int(after_cards[card]) - int(before_cards[card]) == 1
        ]
        if (
            gained == [wanted_card]
            and int(after["development_deck_count"]) == before_deck_count - 1
        ):
            matching.append(outcome_index)
    if not matching:
        raise SearchEngineBoundaryError(
            "Rust BUY_DEVELOPMENT_CARD spectrum cannot reproduce native card "
            f"{wanted_card!r} for action {raw_action!r}"
        )
    return matching[0]


def apply_native_action_record_to_rust(
    rust_game: Any,
    action_record: Any,
    *,
    seated_colors: tuple[str, ...],
    map_kind: str,
) -> Any:
    """Replay one authoritative native-catanatron ActionRecord in Rust.

    Chance is never re-sampled.  The concrete result already chosen by the
    native referee is located in the Rust spectrum and applied exactly.
    Returns the new Rust game because `apply_chance_outcome` is functional.
    """
    ids, raw_actions = rust_legal_actions(rust_game, seated_colors, map_kind)
    wanted_key = canonical_python_action_key(action_record.action)
    matches = [
        index
        for index, raw_action in enumerate(raw_actions)
        if canonical_rust_action_key(raw_action) == wanted_key
    ]
    if len(matches) != 1:
        raise SearchEngineBoundaryError(
            "native action has no unique Rust legal equivalent: "
            f"key={wanted_key!r} matches={len(matches)}"
        )
    position = matches[0]
    selected_id, raw_action = ids[position], raw_actions[position]
    if not is_chance_action(raw_action):
        rust_game.execute_action_index(int(selected_id), list(seated_colors), map_kind)
        return rust_game

    action_type = str(raw_action[1])
    if action_type == "ROLL":
        outcome_index = _recorded_roll_outcome_index(
            rust_game, raw_action, action_record.result
        )
    elif action_type == "MOVE_ROBBER":
        outcome_index = _recorded_robber_outcome_index(
            rust_game, raw_action, action_record.result
        )
    elif action_type == "BUY_DEVELOPMENT_CARD":
        outcome_index = _recorded_development_card_outcome_index(
            rust_game, raw_action, action_record.result
        )
    else:  # pragma: no cover - guarded by engine_equivalence.is_chance_action.
        raise SearchEngineBoundaryError(
            f"unsupported chance action at native/Rust boundary: {raw_action!r}"
        )
    return rust_game.apply_chance_outcome(json.dumps(raw_action), outcome_index)


class CatanZeroSearchPlayer(CatanatronPlayer):
    """Gumbel-search player inside catanatron's native Game.play referee.

    `rust_game` must be the seating-aligned TOURNAMENT shadow created alongside
    the native game by `engine_equivalence.build_paired_games`.  Search never
    changes the native referee state; only the selected action crosses back.
    """

    def __init__(
        self,
        color: Any,
        *,
        rust_game: Any,
        search: Any,
        seated_colors: tuple[str, ...],
        map_kind: str = "TOURNAMENT",
    ) -> None:
        super().__init__(color, is_bot=True)
        if str(map_kind) != "TOURNAMENT":
            raise ValueError(
                "CatanZeroSearchPlayer requires map_kind='TOURNAMENT': BASE map "
                "shuffle parity is not established between native Python and Rust"
            )
        self._rust_game = rust_game
        self._search = search
        self.seated_colors = tuple(str(color_name) for color_name in seated_colors)
        self.map_kind = str(map_kind)
        self._synced_action_records = 0
        self.stats: dict[str, int] = {
            "decisions": 0,
            "forced_decisions": 0,
            "search_decisions": 0,
            "simulations_used": 0,
            "shadow_records_synced": 0,
            "engine_divergences": 0,
            "illegal_policy_picks": 0,
        }

    def reset_state(self) -> None:
        # A search player owns a game-specific shadow and must not be reused.
        raise RuntimeError(
            "CatanZeroSearchPlayer is game-scoped; construct a new player"
        )

    @property
    def rust_game(self) -> Any:
        return self._rust_game

    def _fail_boundary(self, detail: str) -> None:
        self.stats["engine_divergences"] += 1
        raise SearchEngineBoundaryError(detail)

    def sync_from_native(
        self,
        game: Any,
        *,
        check_legal_actions: bool = True,
        native_playable_actions: Any | None = None,
    ) -> tuple[list[int], list[Any]] | None:
        """Bring the Rust shadow to the authoritative native state.

        ``Game.play`` has already generated the native legal list before it
        calls ``Player.decide``.  Accepting that exact list here avoids asking
        the Python referee to generate it a second time.  Return the Rust legal
        list used for the parity check so ``decide`` can map the search result
        without a second Rust/Python JSON boundary crossing.  Neither reuse
        weakens the check: the same native and Rust lists are compared at the
        same decision boundary, and Gumbel search treats its root as immutable.
        """
        records = game.state.action_records
        if len(records) < self._synced_action_records:
            self._fail_boundary(
                "native action history shrank; search player was reused across games"
            )
        divergences_before = self.stats["engine_divergences"]
        try:
            for action_record in records[self._synced_action_records :]:
                self._rust_game = apply_native_action_record_to_rust(
                    self._rust_game,
                    action_record,
                    seated_colors=self.seated_colors,
                    map_kind=self.map_kind,
                )
                self._synced_action_records += 1
                self.stats["shadow_records_synced"] += 1

            symbols = vendor_symbols()
            mismatches = diff_state_views(
                rust_state_view(self._rust_game), python_state_view(game, symbols)
            )
            if mismatches:
                self._fail_boundary(
                    "state divergence after replaying native records "
                    f"through index {self._synced_action_records}: "
                    + "; ".join(mismatches[:5])
                )

            rust_legals: tuple[list[int], list[Any]] | None = None
            if check_legal_actions and game.winning_color() is None:
                rust_legals = rust_legal_actions(
                    self._rust_game, self.seated_colors, self.map_kind
                )
                _ids, raw_actions = rust_legals
                native_actions = (
                    game.playable_actions
                    if native_playable_actions is None
                    else native_playable_actions
                )
                only_rust, only_native = legal_action_diff(
                    raw_actions, native_actions
                )
                if only_rust or only_native:
                    self._fail_boundary(
                        "legal-action divergence at native decision boundary: "
                        f"only_rust={sorted(only_rust)[:5]!r} "
                        f"only_native={sorted(only_native)[:5]!r}"
                    )
            return rust_legals
        except SearchEngineBoundaryError:
            if self.stats["engine_divergences"] == divergences_before:
                self.stats["engine_divergences"] += 1
            raise
        except Exception as error:  # noqa: BLE001 - convert any bridge failure to fail-closed.
            self._fail_boundary(
                f"exception while syncing native referee into Rust shadow: {error!r}"
            )

    def audit_current_game(self, game: Any) -> None:
        """Replay any final actions and verify terminal state parity."""
        self.sync_from_native(game, check_legal_actions=False)

    def decide(self, game: Any, playable_actions: Any) -> Any:
        playable_actions = list(playable_actions)
        rust_legals = self.sync_from_native(
            game, native_playable_actions=playable_actions
        )
        self.stats["decisions"] += 1
        if len(playable_actions) == 1:
            self.stats["forced_decisions"] += 1
            return playable_actions[0]

        result = self._search.search(self._rust_game, force_full=True)
        self.stats["search_decisions"] += 1
        self.stats["simulations_used"] += int(result.simulations_used)
        if rust_legals is None:  # pragma: no cover - a live decision is nonterminal.
            self._fail_boundary("native decision boundary has no Rust legal actions")
        ids, raw_actions = rust_legals
        try:
            position = ids.index(int(result.selected_action))
        except ValueError as error:
            self.stats["illegal_policy_picks"] += 1
            raise SearchEngineBoundaryError(
                f"Gumbel search selected illegal Rust action {result.selected_action!r}"
            ) from error
        native_action = raw_action_to_python_action(
            raw_actions[position], vendor_symbols()
        )
        native_keys = {
            canonical_python_action_key(action) for action in playable_actions
        }
        if canonical_python_action_key(native_action) not in native_keys:
            self.stats["illegal_policy_picks"] += 1
            raise SearchEngineBoundaryError(
                "Gumbel search action has no native playable equivalent: "
                f"{native_action!r}"
            )
        return native_action


__all__ = [
    "CatanZeroNetPlayer",
    "CatanZeroSearchPlayer",
    "SearchEngineBoundaryError",
    "SEARCH_MODE_TODO",
    "apply_native_action_record_to_rust",
    "bind_external_game",
    "default_bridge_config",
    "make_bridge_env",
    "standard_colors",
]
