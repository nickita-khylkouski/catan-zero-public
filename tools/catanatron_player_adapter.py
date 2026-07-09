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

MODE
----
Only `mode="raw_policy"` is implemented (net's policy head, no search) -- see
`SEARCH_MODE_TODO` for a precise account of what full-search mode needs and
why it's a separate, larger piece of work.
"""

from __future__ import annotations

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
Full search mode (GumbelChanceMCTS) is not wired into this adapter yet.
Concretely, what's missing:

  1. A shadow `catanatron_rs.Game` kept in lockstep with the REAL catanatron
     `Game` this Player is handed each `decide()` call -- driven in the
     OPPOSITE direction from the existing lockstep bridge in
     `tools/gumbel_search_vs_bot_h2h.py` / `catan_zero.adapters.
     engine_equivalence`. There, our own search picks every move and the two
     engines are advanced together move-by-move. Here, catanatron's real
     engine (and a real opponent Player, e.g. AlphaBetaPlayer) is what
     advances state on the OTHER seats' turns, so every already-resolved
     chance outcome (dice roll, robber-steal victim, dev-card draw) has to be
     forced into the shadow Rust game via `apply_chance_step` /
     `raw_action_to_python_action` instead of being re-sampled.
  2. On this Player's own turns, hand the (kept-in-sync) shadow Rust game to
     `GumbelChanceMCTS.search(...)`, exactly as
     `tools/gumbel_search_vs_bot_h2h.py` does for its candidate side, then
     translate the chosen action back to catanatron's native `Action` --
     this file's `env._decode_action` covers that same translation for raw
     policy, but the Rust engine's action-id space needs the
     `rust_legal_actions` / `canonical_rust_action_key` matching from
     `engine_equivalence` instead of `ActionCatalog`.
  3. `engine_equivalence`'s board-parity guarantee is only established for
     the TOURNAMENT map (its docstring: the vendored Python engine's
     board-shuffle RNG doesn't match Rust's). Search mode would need to
     either accept that map restriction, or extend the equivalence bridge to
     BASE's random board shuffle -- which raw-policy mode does NOT need,
     since it never touches a second engine at all.
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
    if hasattr(value, "name") and hasattr(value, "value"):  # enum-like (Color, Resource, ...)
        return value.name
    return value


def _reencode_action_index(env: ColonistMultiAgentEnv, action: Any) -> int | None:
    """Best-effort: recover the numeric action-index a past native action
    would have had in `env`'s space, for the `action_id` event-token feature
    (`entity_token_features._event_action_id` -> slot 35), via the same
    `ActionCatalog` used for live decoding.

    Two categories of native action are NOT recoverable this way, and
    `try_encode` correctly misses on them (returning the same safe `None`
    default the live path already falls back to -- see
    `ColonistMultiAgentEnv.valid_actions`'s `_trade_response_indices_for`
    fallback):

      - Trade-negotiation actions (OFFER/ACCEPT/REJECT/CANCEL/CONFIRM_TRADE):
        `env._extended_actions` enumerates COLONIST's own pre-generated offer
        combos, which a native vanilla OFFER_TRADE's exact freqdeck value
        need not match; ACCEPT/REJECT/CANCEL/CONFIRM_TRADE are keyed
        dynamically off `state.current_trade` (only meaningful for the
        CURRENT trade, not a past one).
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
                        "action_type": _NATIVE_TO_EVENT_ACTION_TYPE.get(native_type, native_type),
                        "value": _jsonable(action.value),
                    },
                    "result": _jsonable(record.result),
                    "next_player": None,
                },
            }
        )
    return events[-history_limit:]


def bind_external_game(env: ColonistMultiAgentEnv, game: Any, *, history_limit: int = 64) -> None:
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

        if native_action is None or not self._is_legal(env, game, native_action, playable_actions):
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
    def _is_legal(env: ColonistMultiAgentEnv, game: Any, native_action: Any, playable_actions: list[Any]) -> bool:
        try:
            return bool(env.is_valid_action(playable_actions, game.state, native_action))
        except Exception:  # noqa: BLE001 - be conservative: treat any check failure as illegal.
            return False


__all__ = [
    "CatanZeroNetPlayer",
    "SEARCH_MODE_TODO",
    "bind_external_game",
    "default_bridge_config",
    "make_bridge_env",
    "standard_colors",
]
