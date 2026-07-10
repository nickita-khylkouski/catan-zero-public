"""Regression tests for the f72 public-observation (hidden-info leak) fix.

Three layers:
  1. Featurization masking (fast, deterministic, no model): the online payload
     mask and the load-time token mask agree slot-for-slot; opponent hidden
     slots are zeroed; the actor and all public slots survive; and the UNMASKED
     tokens genuinely leaked (guards against a no-op mask).
  2. Planner belief chance spectra (drives real robber/dev nodes from seeded
     random play): belief robber weights are uniform; belief dev weights follow
     the belief deck; BASE_DEVELOPMENT_DECK matches the engine's initial deck.
  3. Model invariance (guarded, needs a checkpoint + CUDA): evaluate() output is
     invariant to an opponent hidden-hand permutation when public_observation is
     ON, and NOT invariant when OFF (documents the leak the fix removes).
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import pytest

pytest.importorskip("catanatron_rs")

from catan_zero.rl.entity_token_features import (
    PLAYERS,
    PUBLIC_MASK_PLAYER_SLOTS,
    _player_tokens,
    mask_player_tokens_public,
)
from catan_zero.search.neural_rust_mcts import _mask_players_to_public
from catan_zero.search.gumbel_chance_mcts import (
    BASE_DEVELOPMENT_DECK,
    belief_buy_development_card_outcomes,
    belief_move_robber_outcome_weights,
    buy_development_card_real_outcomes,
    is_move_robber_with_victim,
    move_robber_victim_outcome_weights,
)
from catan_zero.search.rust_mcts import _require_rust_module

ACTOR = "BLUE"
OPP = "RED"
ACTOR_IDX = PLAYERS.index(ACTOR)
OPP_IDX = PLAYERS.index(OPP)

# Slot layout landmarks (see entity_token_features._player_tokens).
SLOT_PUBLIC_VP = 3
SLOT_RESOURCE_COUNT = 6
SLOT_DEV_COUNT = 7
SLOT_BRICK = 17  # resources base slot 16 + brick offset 1
SLOT_DEV_VP = 26  # dev base slot 22 + VICTORY_POINT offset 4


def _rust():
    """Skip only live-engine tests when the optional Rust wheel is absent."""
    pytest.importorskip("catanatron_rs")
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def _players_fixture():
    """Actor + one opponent, opponent holding a hidden VP dev card and a lopsided
    hand -- exactly the kind of hidden state the leak exposed."""
    return {
        ACTOR: {
            "public_victory_points": 4,
            "actual_victory_points": 6,
            "resource_card_count": 5,
            "development_card_count": 3,
            "resources": {"wood": 2, "brick": 1, "sheep": 0, "wheat": 1, "ore": 1},
            "development_cards": {"KNIGHT": 1, "VICTORY_POINT": 2},
            "played_development_cards": {"KNIGHT": 1},
            "roads_left": 10,
            "settlements_left": 3,
            "cities_left": 4,
            "has_largest_army": False,
            "has_longest_road": False,
            "has_rolled": True,
            "longest_road_length": 3,
        },
        OPP: {
            "public_victory_points": 3,
            "actual_victory_points": 5,  # 2 hidden VP cards
            "resource_card_count": 4,
            "development_card_count": 2,
            "resources": {"wood": 0, "brick": 4, "sheep": 0, "wheat": 0, "ore": 0},
            "development_cards": {"VICTORY_POINT": 2},
            "played_development_cards": {},
            "roads_left": 11,
            "settlements_left": 4,
            "cities_left": 4,
            "has_largest_army": False,
            "has_longest_road": False,
            "has_rolled": False,
            "longest_road_length": 2,
        },
    }


def _payload(players):
    return {"players": players, "current_player": ACTOR}


# --------------------------------------------------------------------------
# Layer 1: featurization masking
# --------------------------------------------------------------------------
def test_unmasked_tokens_actually_leak():
    tokens = _player_tokens(_payload(_players_fixture()), ACTOR)
    assert tokens[OPP_IDX, SLOT_BRICK] > 0.0, "opponent brick composition should leak when unmasked"
    assert tokens[OPP_IDX, SLOT_DEV_VP] > 0.0, "opponent hidden VP dev card should leak when unmasked"
    assert tokens[OPP_IDX, 5] > 0.0, "opponent actual VP should leak when unmasked"


def test_mask_zeroes_opponent_hidden_slots_only():
    tokens = _player_tokens(_payload(_players_fixture()), ACTOR)
    masked = mask_player_tokens_public(tokens)
    for slot in PUBLIC_MASK_PLAYER_SLOTS:
        assert masked[OPP_IDX, slot] == 0.0, f"opponent slot {slot} must be masked to 0"
    # Public opponent slots survive.
    assert masked[OPP_IDX, SLOT_RESOURCE_COUNT] == tokens[OPP_IDX, SLOT_RESOURCE_COUNT]
    assert masked[OPP_IDX, SLOT_DEV_COUNT] == tokens[OPP_IDX, SLOT_DEV_COUNT]
    assert masked[OPP_IDX, SLOT_PUBLIC_VP] == tokens[OPP_IDX, SLOT_PUBLIC_VP]


def test_actor_row_untouched_by_mask():
    tokens = _player_tokens(_payload(_players_fixture()), ACTOR)
    masked = mask_player_tokens_public(tokens)
    assert np.array_equal(masked[ACTOR_IDX], tokens[ACTOR_IDX]), "actor's own hand must stay visible"


def test_online_and_loadtime_mask_routes_agree():
    """The online payload mask (neural_rust_mcts._mask_players_to_public feeding
    _player_tokens) and the load-time token mask (mask_player_tokens_public on
    unmasked tokens) MUST produce byte-identical player tokens."""
    unmasked = _player_tokens(_payload(_players_fixture()), ACTOR)
    loadtime = mask_player_tokens_public(unmasked)
    masked_players = _mask_players_to_public(_players_fixture(), ACTOR)
    online = _player_tokens(_payload(masked_players), ACTOR)
    assert np.array_equal(online, loadtime), "online and load-time masks diverge"


def test_mask_is_batched_and_copies():
    tokens = _player_tokens(_payload(_players_fixture()), ACTOR)
    batch = np.stack([tokens, tokens], axis=0)
    masked = mask_player_tokens_public(batch)
    assert masked.shape == batch.shape
    assert masked[0, OPP_IDX, SLOT_BRICK] == 0.0
    # Input not mutated.
    assert tokens[OPP_IDX, SLOT_BRICK] > 0.0


def test_none_perspective_masks_everyone():
    masked = _mask_players_to_public(_players_fixture(), None)
    assert masked[ACTOR]["resources"] is None
    assert masked[OPP]["resources"] is None


def test_gumbel_row_writer_persists_public_observation(monkeypatch):
    """The shard writer must request masking itself, independent of training.

    This is a contract test at the call boundary: the feature implementation's
    slot-level masking is covered above, while this catches the production bug
    where online search was masked but `_build_decision_row` silently called
    both feature builders with their unsafe default.
    """
    from catan_zero.rl import gumbel_self_play as self_play
    from catan_zero.search.gumbel_chance_mcts import SearchResult

    calls: list[tuple[str, bool]] = []

    class _Game:
        def current_color(self):
            return ACTOR

        def json_snapshot(self):
            return json.dumps({"current_prompt": "MAIN"})

        def playable_action_indices(self, _colors, _filter):
            return [10, 11]

        def playable_actions_json(self):
            return json.dumps([[ACTOR, "END_TURN"], [ACTOR, "ROLL"]])

    def _entity(*_args, public_observation=False, **_kwargs):
        calls.append(("entity", bool(public_observation)))
        return {"player_tokens": np.zeros((1, 4, 31), dtype=np.float16)}

    def _context(*_args, public_observation=False, **_kwargs):
        calls.append(("context", bool(public_observation)))
        return np.zeros((1, 2, 18), dtype=np.float32)

    monkeypatch.setattr(self_play, "rust_policy_action_ids", lambda *_a, **_k: (3, 4))
    monkeypatch.setattr(self_play, "rust_game_to_entity_batch", _entity)
    monkeypatch.setattr(self_play, "rust_action_context_batch", _context)

    self_play._build_decision_row(
        _Game(),
        result=SearchResult(
            selected_action=10,
            improved_policy={10: 0.75, 11: 0.25},
            visit_counts={10: 3, 11: 1},
            q_values={10: 0.2, 11: -0.1},
            priors={10: 0.6, 11: 0.4},
            root_value=0.1,
            used_full_search=True,
            simulations_used=4,
        ),
        action_size=8,
        colors=(ACTOR, OPP),
        game_seed=7,
        decision_index=0,
        obs_width=1,
    )

    assert calls == [("entity", True), ("context", True)]


# --------------------------------------------------------------------------
# Layer 2: planner belief chance spectra
# --------------------------------------------------------------------------
def test_base_dev_deck_matches_engine_initial_count():
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=0)
    snap = json.loads(game.json_snapshot())
    assert sum(BASE_DEVELOPMENT_DECK.values()) == int(snap["development_deck_count"]), (
        "BASE_DEVELOPMENT_DECK must match the engine's initial development_deck_count"
    )


def _find_chance_nodes(catanatron_rs, seed: int, max_steps: int = 400):
    """Seeded random rollout; return the first (game, action_json) that is a
    MOVE_ROBBER-with-victim and the first that is a BUY_DEVELOPMENT_CARD."""
    colors = ["RED", "BLUE"]
    game = catanatron_rs.Game.simple(colors, seed=seed)
    rng = random.Random(seed)
    robber = None
    buy_dev = None
    for _ in range(max_steps):
        if game.winning_color() is not None:
            break
        ids = [int(a) for a in game.playable_action_indices(colors, None)]
        if not ids:
            break
        raw = json.loads(game.playable_actions_json())
        by_id = dict(zip(ids, raw))
        for aid, aj in by_id.items():
            if robber is None and is_move_robber_with_victim(aj):
                # only useful if the victim actually holds cards
                vi = colors.index(str(aj[2][1]))
                if sum(json.loads(game.json_snapshot())["player_state"][vi]["resources"].values()) > 0:
                    robber = (game, aj)
            if buy_dev is None and len(aj) > 1 and str(aj[1]) == "BUY_DEVELOPMENT_CARD":
                buy_dev = (game, aj)
        if robber is not None and buy_dev is not None:
            break
        choice = rng.choice(ids)
        aj = by_id[choice]
        spectrum = json.loads(game.spectrum_json(json.dumps(aj)))
        if spectrum:
            k = rng.randrange(len(spectrum))
            game = game.apply_chance_outcome(json.dumps(aj), k)
        else:
            game.execute_action_index(choice, colors, None)
    return robber, buy_dev


def test_belief_robber_weights_are_uniform_over_true_steal_set():
    catanatron_rs = _rust()
    robber = None
    for seed in range(0, 40):
        robber, _ = _find_chance_nodes(catanatron_rs, seed)
        if robber is not None:
            break
    if robber is None:
        pytest.skip("no MOVE_ROBBER-with-nonempty-victim node found in probe rollouts")
    game, action_json = robber
    true = move_robber_victim_outcome_weights(game, action_json)
    belief = belief_move_robber_outcome_weights(game, action_json)
    assert belief, "belief robber produced no candidates for a non-empty victim"
    # All belief weights uniform.
    assert all(abs(w - 1.0) < 1e-9 for _i, w, _g in belief)
    # Same outcome-index SET as the true (materialized) steal candidates, when
    # the true path materialized (legacy-shape) rather than passing through.
    if true:
        assert {i for i, _w, _g in belief} == {i for i, _w, _g in true}


def test_belief_dev_weights_follow_belief_deck():
    catanatron_rs = _rust()
    buy_dev = None
    for seed in range(0, 40):
        _, buy_dev = _find_chance_nodes(catanatron_rs, seed)
        if buy_dev is not None:
            break
    if buy_dev is None:
        pytest.skip("no BUY_DEVELOPMENT_CARD node found in probe rollouts")
    game, action_json = buy_dev
    real = buy_development_card_real_outcomes(game, action_json)
    belief = belief_buy_development_card_outcomes(game, action_json)
    assert belief, "belief dev produced no candidates"
    # Belief weights are integer belief-deck counts (>=1), not the engine's
    # fractional native probabilities.
    assert all(float(w).is_integer() and w >= 1.0 for _i, w, _g in belief)
    # Belief keeps a subset of (or all) the real drawable outcomes.
    assert {i for i, _w, _g in belief}.issubset({i for i, _w, _g in real})


# --------------------------------------------------------------------------
# Layer 3: model invariance (guarded -- needs a checkpoint + CUDA)
# --------------------------------------------------------------------------
_CKPT = os.environ.get("CATAN_ZERO_CKPT", "")


@pytest.mark.skipif(
    not _CKPT or not os.path.exists(_CKPT), reason="set CATAN_ZERO_CKPT to a checkpoint to run"
)
def test_model_invariant_to_hidden_hand_when_public_observation_on():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    catanatron_rs = _rust()
    from catan_zero.search.neural_rust_mcts import (
        EntityGraphRustEvaluator,
        EntityGraphRustEvaluatorConfig,
        rust_game_to_entity_batch,
        rust_policy_action_ids,
    )

    colors = ("RED", "BLUE")

    class _Proxy:
        def __init__(self, game, target, new_state):
            object.__setattr__(self, "_g", game)
            object.__setattr__(self, "_t", str(target))
            object.__setattr__(self, "_s", new_state)

        def player_state_json(self, color):
            return self._s if str(color) == self._t else self._g.player_state_json(color)

        def __getattr__(self, n):
            return getattr(object.__getattribute__(self, "_g"), n)

    def permute(state_json):
        st = json.loads(state_json)
        for key in ("resources", "dev_cards"):
            vec = st.get(key)
            if isinstance(vec, list) and sum(int(x) for x in vec) > 0:
                total = sum(int(x) for x in vec)
                hi = max(range(len(vec)), key=lambda i: vec[i])
                tgt = next((i for i in range(len(vec)) if i != hi), hi)
                st[key] = [0] * len(vec)
                st[key][tgt] = total
        return json.dumps(st)

    def outputs(evaluator, game, legal, actor):
        pids = rust_policy_action_ids(game, legal, colors=colors, action_size=int(evaluator.policy.action_size))
        ent = rust_game_to_entity_batch(
            game, legal, actor=actor, colors=colors,
            action_size=int(evaluator.policy.action_size), policy_action_ids=pids,
            public_observation=bool(evaluator.config.public_observation),
        )
        out = evaluator.policy.forward_legal_np(
            ent, np.asarray(pids, dtype=np.int64)[None, :],
            _context(evaluator, game, legal, actor, pids), return_q=False,
        )
        return (out["value"].detach().float().cpu().numpy()[0],
                out["logits"].detach().float().cpu().numpy()[0][: len(legal)])

    def _context(evaluator, game, legal, actor, pids):
        from catan_zero.search.neural_rust_mcts import rust_action_context_batch
        return rust_action_context_batch(
            game, legal, actor=actor, colors=colors,
            action_size=int(evaluator.policy.action_size), policy_action_ids=pids,
        )

    game = catanatron_rs.Game.simple(list(colors), seed=12345)
    crng = random.Random(1)
    for _ in range(46):
        if game.winning_color() is not None:
            break
        ids = [int(a) for a in game.playable_action_indices(list(colors), None)]
        if not ids:
            break
        aj = json.loads(game.playable_actions_json())[ids.index(ids[crng.randrange(len(ids))])]
        spec = json.loads(game.spectrum_json(json.dumps(aj)))
        if spec:
            game = game.apply_chance_outcome(json.dumps(aj), crng.randrange(len(spec)))
        else:
            game.execute_action_index(ids[0], list(colors), None)
    actor = str(game.current_color())
    opp = [c for c in colors if c != actor][0]
    legal = tuple(int(a) for a in game.playable_action_indices(list(colors), None))
    new_state = permute(game.player_state_json(opp))
    proxy = _Proxy(game, opp, new_state)

    for flag, invariant in ((True, True), (False, False)):
        ev = EntityGraphRustEvaluator.from_checkpoint(
            _CKPT, device="cuda:0", config=EntityGraphRustEvaluatorConfig(public_observation=flag)
        )
        v0, l0 = outputs(ev, game, legal, actor)
        v1, l1 = outputs(ev, proxy, legal, actor)
        vdiff = abs(float(v0) - float(v1))
        ldiff = float(np.max(np.abs(l0 - l1)))
        if invariant:
            assert vdiff < 1e-5 and ldiff < 1e-4, f"public_observation=ON should be invariant: v={vdiff} l={ldiff}"
        else:
            assert vdiff > 1e-5 or ldiff > 1e-4, "public_observation=OFF should still leak (documents current behavior)"
