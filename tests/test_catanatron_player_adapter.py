"""Translation-correctness + smoke tests for CAT-57's catanatron Player adapter.

Two distinct correctness questions, tested separately:

  1. Does binding an EXTERNALLY-created `Game` into a `ColonistMultiAgentEnv`
     (`bind_external_game`) produce the SAME entity-token features as the
     env's own self-created game, for an identical trajectory? This is the
     "translation-correctness" check the ticket asks for, adapted to this
     codebase's actual architecture: the production featurizer already
     speaks catanatron-native state (`ColonistMultiAgentEnv` wraps the same
     vendored `catanatron.game.Game` used everywhere else), so the risk
     isn't "two different featurizers disagree" -- it's "does bypassing
     `.reset()` change anything observable". `test_bind_external_game_
     matches_self_created_features` answers that directly, key-by-key.

  2. Does a full match, played end to end through `CatanZeroNetPlayer`
     inside catanatron's OWN `Game.play()` loop, complete without any
     illegal-move fallback firing? `test_catan_zero_net_player_full_game_*`
     answers that using a freshly-initialized (untrained) policy -- legality
     doesn't depend on policy quality, only on the action translation being
     correct, and an untrained net still exercises every decision type
     (build, robber, discard, roll, trade-response) over a full game.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.entity_token_features import ACTION_TYPES
from catan_zero.rl.entity_token_features import build_entity_token_features
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv

from catanatron_player_adapter import (  # type: ignore  # noqa: E402
    CatanZeroNetPlayer,
    _reencode_action_index,
    _synthetic_event_log,
    bind_external_game,
    default_bridge_config,
    make_bridge_env,
    standard_colors,
)


def test_synthetic_discard_history_never_reemits_hidden_resource_id() -> None:
    action_type = SimpleNamespace(name="DISCARD_RESOURCE")
    action = SimpleNamespace(
        action_type=action_type,
        color=SimpleNamespace(name="RED"),
        value=SimpleNamespace(name="ORE", value="ORE"),
    )
    env = SimpleNamespace(action_catalog=SimpleNamespace(try_encode=lambda _action: 123))
    game = SimpleNamespace(
        state=SimpleNamespace(
            action_records=[SimpleNamespace(action=action, result=action.value)]
        )
    )

    assert _reencode_action_index(env, action) is None
    event = _synthetic_event_log(env, game, history_limit=64)[-1]
    assert event["payload"]["action_index"] is None
    assert event["payload"]["action"]["index"] is None
    assert event["payload"]["action"]["value"] == "hidden_resource"
    assert event["payload"]["result"] == "hidden_resource"

# entity_token_features._event_tokens slots 15/16 encode a per-event
# `turn_key` ordinal that `_synthetic_event_log` deliberately does not
# reconstruct (see its docstring). Zeroed out before comparison below.
_EVENT_TURN_KEY_SLOTS = (15, 16)

# entity_token_features._event_tokens slot 35 encodes the event's numeric
# action-id (`_event_action_id`). `_reencode_action_index`'s docstring
# documents the categories of native action that are NOT re-encoded:
# trade-negotiation actions (colonist-specific offer combos / dynamic
# `current_trade` state) and chance-resolved actions ROLL/BUY_DEVELOPMENT_
# CARD (ActionCatalog only has one generic pre-resolution entry for each;
# the executed action carries the resolved dice/card), plus DISCARD_RESOURCE,
# whose otherwise recoverable index is deliberately redacted because it
# identifies the hidden resource. Masked ONLY for rows matching one of those,
# so this test still strictly verifies action-id fidelity for every OTHER
# action type (build, end_turn, move_robber, maritime_trade, play_*).
_ACTION_ID_SLOT = 35
_UNRECODABLE_ACTION_TYPE_COLUMNS = tuple(
    17 + ACTION_TYPES.index(kind)
    for kind in (
        "offer_trade",
        "accept_trade",
        "reject_trade",
        "cancel_trade",
        "confirm_trade",
        "ROLL",
        "BUY_DEVELOPMENT_CARD",
        "DISCARD_RESOURCE",
    )
)


def _require_catanatron() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")


def _assert_features_match(feats_a: dict, feats_c: dict) -> None:
    assert set(feats_a) == set(feats_c)
    for key in feats_a:
        if key == "schema":
            assert str(feats_a[key]) == str(feats_c[key])
            continue
        if key == "event_tokens":
            a = np.array(feats_a[key], copy=True)
            c = np.array(feats_c[key], copy=True)
            a[:, list(_EVENT_TURN_KEY_SLOTS)] = 0
            c[:, list(_EVENT_TURN_KEY_SLOTS)] = 0
            unrecodable_row = (a[:, _UNRECODABLE_ACTION_TYPE_COLUMNS].sum(axis=1) > 0) | (
                c[:, _UNRECODABLE_ACTION_TYPE_COLUMNS].sum(axis=1) > 0
            )
            a[unrecodable_row, _ACTION_ID_SLOT] = 0
            c[unrecodable_row, _ACTION_ID_SLOT] = 0
            np.testing.assert_array_equal(
                a, c, err_msg=f"mismatch in {key} (turn_key, and action_id for trade/ROLL/BUY_DEV rows, excluded)"
            )
            continue
        np.testing.assert_array_equal(feats_a[key], feats_c[key], err_msg=f"mismatch in {key}")


@pytest.mark.parametrize("reset_seed,rng_seed", [(7, 0), (23, 5)])
def test_bind_external_game_matches_self_created_features(reset_seed: int, rng_seed: int) -> None:
    """A `Game` driven independently of `ColonistMultiAgentEnv.reset()` must
    featurize identically to the SAME trajectory played through the env's
    own self-created game -- i.e. `bind_external_game` is behaviorally
    invisible to the featurizer, which is the whole premise this adapter
    relies on to avoid re-deriving `build_entity_token_features` from
    scratch. Checked at every ply (not just a sparse sample) over a long,
    randomly-acting trajectory so any latent mismatch (a rare action type,
    a boundary in the event-history window) actually gets exercised."""
    _require_catanatron()
    rng = np.random.default_rng(rng_seed)

    env_a = ColonistMultiAgentEnv(default_bridge_config(2))
    env_a.reset(seed=reset_seed)
    # Pristine copy of the SAME initial state (identical board/seating),
    # taken via catanatron's own `Game.copy()` before any moves are played,
    # so game_b tracks env_a's game move-for-move without depending on any
    # board-generation RNG matching a second time.
    game_b = env_a.game.copy()
    env_c = make_bridge_env(2)

    checked_any = False
    for _step in range(250):
        valid = env_a.valid_actions()
        if not valid:
            break
        action_index = int(rng.choice(valid))
        _obs, _reward, terminated, truncated, _info = env_a.step(action_index)
        last_record = env_a.game.state.action_records[-1]
        # `validate_action=False` + `action_record=last_record` is
        # catanatron's own replay mechanism (see `apply_roll`/
        # `apply_move_robber`/`apply_buy_development_card`'s `action_record`
        # param): it forces the SAME already-resolved chance outcome (dice,
        # robber-steal resource, dev-card draw) instead of re-sampling one,
        # which is what keeps game_b a byte-identical mirror of env_a's game
        # rather than diverging the moment a ROLL happens. This is a
        # test-only concern -- the real adapter never needs it, since
        # catanatron's own engine resolves chance exactly once, in the one
        # real `Game` object, with no second copy to keep in sync.
        game_b.execute(last_record.action, validate_action=False, action_record=last_record)

        bind_external_game(env_c, game_b)
        for actor in ("BLUE", "RED"):
            feats_a = build_entity_token_features(env_a, actor=actor)
            feats_c = build_entity_token_features(env_c, actor=actor)
            _assert_features_match(feats_a, feats_c)
        checked_any = True

        if terminated or truncated:
            break

    assert checked_any, "no checkpoints were exercised; increase step budget"


def test_decode_action_round_trips_to_a_legal_native_action() -> None:
    """Every action index `bind_external_game` + `env._decode_action` can
    produce for a bound game must independently pass catanatron's own
    `is_valid_action` -- the exact gate `Game.execute` uses -- not just look
    plausible."""
    _require_catanatron()
    rng = np.random.default_rng(1)

    env_a = ColonistMultiAgentEnv(default_bridge_config(2))
    env_a.reset(seed=11)
    game_b = env_a.game.copy()

    env_c = make_bridge_env(2)
    checked_any_multi_action_state = False
    for _ in range(30):
        valid = env_a.valid_actions()
        if not valid:
            break

        bind_external_game(env_c, game_b)
        if len(valid) > 1:
            checked_any_multi_action_state = True
            for action_index in valid:
                decoded = env_c._decode_action(int(action_index))
                assert decoded is not None
                assert env_c.is_valid_action(game_b.playable_actions, game_b.state, decoded)

        action_index = int(rng.choice(valid))
        _obs, _reward, terminated, truncated, _info = env_a.step(action_index)
        last_record = env_a.game.state.action_records[-1]
        # `validate_action=False` + `action_record=last_record` is
        # catanatron's own replay mechanism (see `apply_roll`/
        # `apply_move_robber`/`apply_buy_development_card`'s `action_record`
        # param): it forces the SAME already-resolved chance outcome (dice,
        # robber-steal resource, dev-card draw) instead of re-sampling one,
        # which is what keeps game_b a byte-identical mirror of env_a's game
        # rather than diverging the moment a ROLL happens. This is a
        # test-only concern -- the real adapter never needs it, since
        # catanatron's own engine resolves chance exactly once, in the one
        # real `Game` object, with no second copy to keep in sync.
        game_b.execute(last_record.action, validate_action=False, action_record=last_record)
        if terminated or truncated:
            break

    assert checked_any_multi_action_state


def _untrained_policy_for(num_players: int, *, vps_to_win: int) -> EntityGraphPolicy:
    return EntityGraphPolicy.create(
        env_config=default_bridge_config(num_players, vps_to_win=vps_to_win),
        hidden_size=32,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )


def test_catan_zero_net_player_full_game_no_illegal_fallback() -> None:
    """End-to-end: catanatron's OWN `Game.play()` loop drives a full match
    with `CatanZeroNetPlayer` in one seat -- no shadow engine, no lockstep
    mirroring. An untrained (randomly-initialized) policy is enough to
    exercise every decision type over a full game; legality of the
    reverse-translation doesn't depend on the policy being any good."""
    _require_catanatron()
    from catan_zero.rl._catanatron import import_catanatron_module

    game_module = import_catanatron_module("catanatron.game")
    map_module = import_catanatron_module("catanatron.models.map")
    player_module = import_catanatron_module("catanatron.models.player")
    vps_to_win = 5  # keep the smoke test fast; legality doesn't depend on game length.
    policy = _untrained_policy_for(2, vps_to_win=vps_to_win)

    for seed in (101, 202):
        colors = standard_colors(2)
        candidate = CatanZeroNetPlayer(
            colors[0], policy=policy, seed=seed, vps_to_win=vps_to_win,
        )
        opponent = player_module.RandomPlayer(colors[1])

        catan_map = map_module.build_map("BASE")
        game = game_module.Game(
            players=[candidate, opponent],
            seed=seed,
            catan_map=catan_map,
            vps_to_win=vps_to_win,
        )
        winner = game.play()

        assert winner is not None or game.state.num_turns >= game_module.TURNS_LIMIT
        assert candidate.stats["decisions"] > 0
        assert candidate.stats["illegal_policy_picks"] == 0, (
            "policy pick failed catanatron's own is_valid_action at least once "
            f"(seed={seed}); this indicates a real translation bug, not a "
            "policy-quality issue"
        )
