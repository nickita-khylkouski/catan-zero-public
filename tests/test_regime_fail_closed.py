"""Fail-closed regime test suite (independent-audit follow-up to f72/#76).

Five guarantees, enforced together so no future change can silently regress
any one of them:
  1. A masked-trained checkpoint (`trained_with_masked_hidden_info=True`)
     paired with `public_observation=False` must fail closed.
  2. A known-omniscient checkpoint (`trained_with_masked_hidden_info=False`)
     paired with `public_observation=True` must fail closed.
  3. Under `public_observation=True`, permuting an OPPONENT's hidden hand
     leaves every masked feature path bit-identical -- including
     `rust_action_context_batch`, which an independent audit found built its
     players payload without ever applying `_mask_players_to_public`
     (fixed alongside this suite -- see the `public_observation` parameter
     added to `rust_action_context_batch` in neural_rust_mcts.py).
  4. Permuting the ACTING player's own hand DOES change features (guards
     against a vacuous mask that "passes" #3 by zeroing everyone).
  5. The planner belief-chance-spectra path is actually dispatched to when
     `belief_chance_spectra=True` (not silently falling through to the
     true-hand-weighted / native-spectrum path).

Crib source for fixtures: tests/test_public_observation_masking.py (Layer 2/3
helpers), tests/test_mask_hidden_info_checkpoint_safety.py (tiny-policy
save/load), tests/test_gumbel_chance_mcts.py (direct `_traverse_robber_or_dev`
node construction), tests/test_value_squash.py (`EntityGraphPolicy.create`
tiny-but-real-action-space fixture).
"""
from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import pytest

import catan_zero.search.gumbel_chance_mcts as gcm
import catan_zero.search.neural_rust_mcts as neural_rust_mcts
from catan_zero.rl.action_mask import ActionCatalog
from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy
from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    HeuristicRustEvaluator,
    _GNode,
)
from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
    rust_action_context_batch,
    rust_game_to_entity_batch,
    rust_policy_action_ids,
)
from catan_zero.search.rust_mcts import _require_rust_module

COLORS = ("RED", "BLUE")
ACTION_SIZE = 8
STATIC_FEATURE_SIZE = 4


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def _tiny_policy() -> EntityGraphPolicy:
    """Minimal policy for regime-metadata tests that never run a forward
    pass -- action/context sizes are arbitrary here (see
    tests/test_mask_hidden_info_checkpoint_safety.py, same pattern)."""
    config = EntityGraphConfig(
        action_size=ACTION_SIZE,
        static_action_feature_size=STATIC_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    static = np.zeros((ACTION_SIZE, STATIC_FEATURE_SIZE), dtype=np.float32)
    return EntityGraphPolicy(config, static, device="cpu")


def _advance_to_multi_action_state(catanatron_rs, *, seed: int, min_legal: int = 2):
    game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
    for _ in range(300):
        game.play_tick()
        if game.winning_color() is not None:
            break
        playable = json.loads(game.playable_actions_json())
        if len(playable) >= min_legal:
            return game
    raise AssertionError(f"did not reach a state with >= {min_legal} legal actions")


# ---------------------------------------------------------------------------
# 1 + 2: checkpoint-regime mismatch fails closed, both directions
# ---------------------------------------------------------------------------


def test_masked_checkpoint_with_public_observation_false_raises(tmp_path):
    """A checkpoint trained with --mask-hidden-info (metadata
    trained_with_masked_hidden_info=True) loaded with public_observation=False
    must raise: feeding it omniscient inputs it never learned to use is a
    silent, undetectable-at-runtime misconfiguration."""
    policy = _tiny_policy()
    path = tmp_path / "masked_checkpoint.pt"
    policy.save(path, mask_hidden_info=True)
    loaded = EntityGraphPolicy.load(path, device="cpu")
    assert loaded.trained_with_masked_hidden_info is True

    with pytest.raises(ValueError, match="mismatch"):
        EntityGraphRustEvaluator(
            loaded, config=EntityGraphRustEvaluatorConfig(public_observation=False)
        )


def test_known_omniscient_checkpoint_with_public_observation_true_raises(tmp_path):
    """A checkpoint trained WITHOUT masking (the known-omniscient control)
    loaded with public_observation=True must raise: this is the exact f72
    leak (regenerating the leaked-hidden-info corpus) the guard exists to
    prevent."""
    policy = _tiny_policy()
    path = tmp_path / "omniscient_checkpoint.pt"
    policy.save(path, mask_hidden_info=False)
    loaded = EntityGraphPolicy.load(path, device="cpu")
    assert loaded.trained_with_masked_hidden_info is False

    with pytest.raises(ValueError, match="mismatch"):
        EntityGraphRustEvaluator(
            loaded, config=EntityGraphRustEvaluatorConfig(public_observation=True)
        )


def test_matching_regimes_construct_without_raising(tmp_path):
    """Sanity bound on 1+2: the guard must not be so aggressive it rejects
    every combination -- both agreeing pairs must construct cleanly."""
    policy = _tiny_policy()

    masked_path = tmp_path / "masked.pt"
    policy.save(masked_path, mask_hidden_info=True)
    masked = EntityGraphPolicy.load(masked_path, device="cpu")
    ev_masked = EntityGraphRustEvaluator(
        masked, config=EntityGraphRustEvaluatorConfig(public_observation=True)
    )
    assert ev_masked.config.public_observation is True

    omniscient_path = tmp_path / "omniscient.pt"
    policy.save(omniscient_path, mask_hidden_info=False)
    omniscient = EntityGraphPolicy.load(omniscient_path, device="cpu")
    ev_omniscient = EntityGraphRustEvaluator(
        omniscient, config=EntityGraphRustEvaluatorConfig(public_observation=False)
    )
    assert ev_omniscient.config.public_observation is False


# ---------------------------------------------------------------------------
# 3 + 4: feature-level invariance/sensitivity under public_observation=True,
# through BOTH rust_game_to_entity_batch and rust_action_context_batch.
# ---------------------------------------------------------------------------


class _HandPermutedProxy:
    """Wraps a real `catanatron_rs.Game`, substituting `player_state_json` for
    exactly one color with a permuted hand -- everything else (board, other
    players, legal actions) delegates to the real game untouched."""

    def __init__(self, game: Any, target_color: str, permuted_state_json: str) -> None:
        object.__setattr__(self, "_game", game)
        object.__setattr__(self, "_target", str(target_color))
        object.__setattr__(self, "_state_json", permuted_state_json)

    def player_state_json(self, color: str) -> str:
        if str(color) == self._target:
            return self._state_json
        return self._game.player_state_json(color)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_game"), name)


def _permute_hand(state_json: str) -> str:
    """Move all of a hand's resource/dev-card mass onto a single different
    slot -- any masking of that hand must be invariant to this; any
    unmasked read of it must not be."""
    state = json.loads(state_json)
    for key in ("resources", "dev_cards"):
        vector = state.get(key)
        if isinstance(vector, list) and sum(int(v) for v in vector) > 0:
            total = sum(int(v) for v in vector)
            high = max(range(len(vector)), key=lambda i: vector[i])
            target = next((i for i in range(len(vector)) if i != high), high)
            state[key] = [0] * len(vector)
            state[key][target] = total
    return json.dumps(state)


def _masked_feature_bundle(game: Any, *, actor: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    action_size = ActionCatalog(COLORS).size
    legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
    mapped = rust_policy_action_ids(game, legal, colors=COLORS, action_size=action_size)
    entity = rust_game_to_entity_batch(
        game, legal, actor=actor, colors=COLORS, action_size=action_size,
        policy_action_ids=mapped, public_observation=True,
    )
    context = rust_action_context_batch(
        game, legal, actor=actor, colors=COLORS, action_size=action_size,
        policy_action_ids=mapped, public_observation=True,
    )
    return entity, context


def test_opponent_hand_permutation_is_invariant_when_public_observation_on():
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=11)
    actor = str(game.current_color())
    opp = next(c for c in COLORS if c != actor)

    permuted = _permute_hand(game.player_state_json(opp))
    proxy = _HandPermutedProxy(game, opp, permuted)

    entity_a, context_a = _masked_feature_bundle(game, actor=actor)
    entity_b, context_b = _masked_feature_bundle(proxy, actor=actor)

    assert set(entity_a) == set(entity_b)
    for key in entity_a:
        assert np.array_equal(entity_a[key], entity_b[key]), (
            f"public_observation=True must be invariant to opponent hand "
            f"permutation, but entity feature {key!r} differs"
        )
    assert np.array_equal(context_a, context_b), (
        "public_observation=True must be invariant to opponent hand "
        "permutation through rust_action_context_batch too (the path fixed "
        "alongside this suite)"
    )


def _advance_until_actor_holds_cards(catanatron_rs, *, seed: int, min_legal: int = 2, max_ticks: int = 300):
    """Like `_advance_to_multi_action_state`, but keeps ticking until the
    current actor's hand (resources or dev cards) is non-empty -- the early
    placement phase leaves every hand empty, which would make a permutation
    test on the actor's own hand vacuously pass (nothing to permute)."""
    game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
    for _ in range(max_ticks):
        game.play_tick()
        if game.winning_color() is not None:
            break
        playable = json.loads(game.playable_actions_json())
        if len(playable) < min_legal:
            continue
        actor = str(game.current_color())
        state = json.loads(game.player_state_json(actor))
        if sum(int(v) for v in state.get("resources", ())) > 0 or sum(
            int(v) for v in state.get("dev_cards", ())
        ) > 0:
            return game
    raise AssertionError(f"actor never held cards within {max_ticks} ticks (seed={seed})")


def test_actor_own_hand_permutation_changes_entity_features():
    """Guards against a vacuous mask that zeroes EVERY player (including the
    actor), which would otherwise make the test above pass trivially."""
    catanatron_rs = _rust()
    game = _advance_until_actor_holds_cards(catanatron_rs, seed=11)
    actor = str(game.current_color())

    original_state = game.player_state_json(actor)
    permuted = _permute_hand(original_state)
    assert json.loads(permuted) != json.loads(original_state), (
        "test fixture requires the actor to hold at least one card to permute"
    )
    proxy = _HandPermutedProxy(game, actor, permuted)

    entity_a, _ = _masked_feature_bundle(game, actor=actor)
    entity_b, _ = _masked_feature_bundle(proxy, actor=actor)

    assert any(
        not np.array_equal(entity_a[key], entity_b[key]) for key in entity_a
    ), "permuting the ACTING player's own hand must change SOME entity feature"


def test_rust_action_context_batch_applies_masking_gate_when_configured():
    """Directly proves the Task-1 fix: `rust_action_context_batch` must call
    `_mask_players_to_public` when `public_observation=True`, and must NOT
    call it when False/omitted (matching the pre-fix behavior still relied on
    by callers that intentionally record the omniscient corpus for later
    load-time masking, e.g. gumbel_self_play.py's shard writer)."""
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=3)
    actor = str(game.current_color())
    action_size = ActionCatalog(COLORS).size
    legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
    mapped = rust_policy_action_ids(game, legal, colors=COLORS, action_size=action_size)

    calls: list[Any] = []
    real_mask = neural_rust_mcts._mask_players_to_public

    def _spy(players, perspective):
        calls.append(perspective)
        return real_mask(players, perspective)

    neural_rust_mcts._mask_players_to_public = _spy
    try:
        rust_action_context_batch(
            game, legal, actor=actor, colors=COLORS, action_size=action_size,
            policy_action_ids=mapped, public_observation=True,
        )
        assert calls == [actor], (
            "public_observation=True must mask via _mask_players_to_public(perspective=actor)"
        )

        calls.clear()
        rust_action_context_batch(
            game, legal, actor=actor, colors=COLORS, action_size=action_size,
            policy_action_ids=mapped,
        )
        assert calls == [], "public_observation defaulting to False must NOT invoke the masking gate"
    finally:
        neural_rust_mcts._mask_players_to_public = real_mask


def test_logits_invariant_to_opponent_hand_permutation_when_masked():
    """Cheap (CPU, tiny hidden_size, real action space) end-to-end check that
    the invariance in #3 survives an actual forward pass, not just the raw
    feature tensors -- monkeypatches the regime metadata so no real trained
    checkpoint is required (this is the known-omniscient-vs-masked CONTROL
    pattern from tests/test_masked_vs_unmasked_calibration_check.py, applied
    to a from-scratch policy instead of a saved checkpoint)."""
    catanatron_rs = _rust()
    from catan_zero.rl.self_play import make_env_config

    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    policy.model.eval()  # create() leaves train mode; active Dropout would break equality.
    policy.trained_with_masked_hidden_info = True
    evaluator = EntityGraphRustEvaluator(
        policy, config=EntityGraphRustEvaluatorConfig(public_observation=True, cache_size=0)
    )

    game = _advance_to_multi_action_state(catanatron_rs, seed=5)
    actor = str(game.current_color())
    opp = next(c for c in COLORS if c != actor)
    legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
    permuted = _permute_hand(game.player_state_json(opp))
    proxy = _HandPermutedProxy(game, opp, permuted)

    priors_a, value_a = evaluator.evaluate(game, legal, root_color=actor, colors=COLORS)
    priors_b, value_b = evaluator.evaluate(proxy, legal, root_color=actor, colors=COLORS)

    assert priors_a == pytest.approx(priors_b, abs=1e-6)
    assert value_a == pytest.approx(value_b, abs=1e-6)


# ---------------------------------------------------------------------------
# 5: belief-chance-spectra dispatch actually happens when configured, and
# never falls back to a true-hidden-state read while it's on.
# ---------------------------------------------------------------------------


def _find_chance_nodes(seed: int, max_steps: int = 400):
    """Seeded random rollout; return the first (game, action_json) that is a
    MOVE_ROBBER-with-victim and the first that is a BUY_DEVELOPMENT_CARD.
    Cribbed from tests/test_public_observation_masking.py."""
    catanatron_rs = _rust()
    colors = list(COLORS)
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
            if robber is None and gcm.is_move_robber_with_victim(aj):
                victim_index = colors.index(str(aj[2][1]))
                snapshot = json.loads(game.json_snapshot())
                if sum(snapshot["player_state"][victim_index]["resources"].values()) > 0:
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


def test_belief_chance_spectra_dispatches_to_belief_robber_not_true_hand(monkeypatch):
    robber = None
    for seed in range(0, 40):
        robber, _ = _find_chance_nodes(seed)
        if robber is not None:
            break
    if robber is None:
        pytest.skip("no MOVE_ROBBER-with-nonempty-victim node found in probe rollouts")
    game, _action_json = robber

    belief_calls: list[Any] = []
    true_calls: list[Any] = []
    real_belief = gcm.belief_move_robber_outcome_weights
    real_true = gcm.move_robber_victim_outcome_weights

    def _belief_spy(g, aj, *, cached_spectrum=None):
        belief_calls.append(aj)
        return real_belief(g, aj, cached_spectrum=cached_spectrum)

    def _true_spy(g, aj, *, cached_spectrum=None):
        true_calls.append(aj)
        return real_true(g, aj, cached_spectrum=cached_spectrum)

    monkeypatch.setattr(gcm, "belief_move_robber_outcome_weights", _belief_spy)
    monkeypatch.setattr(gcm, "move_robber_victim_outcome_weights", _true_spy)

    node = _GNode(game=game.copy(), root_color=str(game.current_color()))
    mcts = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=1, belief_chance_spectra=True),
        HeuristicRustEvaluator(score_actions=False),
    )
    mcts._expand(node)
    action_id = next(
        aid for aid, stored in node.action_json.items()
        if stored[1] == "MOVE_ROBBER" and stored[2][1] is not None
    )
    stats = node.actions[action_id]
    mcts._traverse_robber_or_dev(node, action_id, stats, depth=0)

    assert belief_calls, "belief_chance_spectra=True must dispatch to belief_move_robber_outcome_weights"
    assert not true_calls, "belief_chance_spectra=True must NOT also consult the true-hand-weighted path"


def test_belief_chance_spectra_off_never_calls_belief_robber(monkeypatch):
    robber = None
    for seed in range(0, 40):
        robber, _ = _find_chance_nodes(seed)
        if robber is not None:
            break
    if robber is None:
        pytest.skip("no MOVE_ROBBER-with-nonempty-victim node found in probe rollouts")
    game, _action_json = robber

    belief_calls: list[Any] = []
    real_belief = gcm.belief_move_robber_outcome_weights

    def _belief_spy(g, aj, *, cached_spectrum=None):
        belief_calls.append(aj)
        return real_belief(g, aj, cached_spectrum=cached_spectrum)

    monkeypatch.setattr(gcm, "belief_move_robber_outcome_weights", _belief_spy)

    node = _GNode(game=game.copy(), root_color=str(game.current_color()))
    mcts = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=1, belief_chance_spectra=False, correct_rust_chance_spectra=True),
        HeuristicRustEvaluator(score_actions=False),
    )
    mcts._expand(node)
    action_id = next(
        aid for aid, stored in node.action_json.items()
        if stored[1] == "MOVE_ROBBER" and stored[2][1] is not None
    )
    stats = node.actions[action_id]
    mcts._traverse_robber_or_dev(node, action_id, stats, depth=0)

    assert not belief_calls, "belief_chance_spectra=False must never consult the belief path"


def test_belief_chance_spectra_dispatches_to_belief_dev_deck_not_true_deck(monkeypatch):
    buy_dev = None
    for seed in range(0, 40):
        _, buy_dev = _find_chance_nodes(seed)
        if buy_dev is not None:
            break
    if buy_dev is None:
        pytest.skip("no BUY_DEVELOPMENT_CARD node found in probe rollouts")
    game, _action_json = buy_dev

    belief_calls: list[Any] = []
    true_calls: list[Any] = []
    real_belief = gcm.belief_buy_development_card_outcomes
    real_true = gcm.buy_development_card_real_outcomes

    def _belief_spy(g, aj, *, cached_spectrum=None):
        belief_calls.append(aj)
        return real_belief(g, aj, cached_spectrum=cached_spectrum)

    def _true_spy(g, aj, *, cached_spectrum=None):
        true_calls.append(aj)
        return real_true(g, aj, cached_spectrum=cached_spectrum)

    monkeypatch.setattr(gcm, "belief_buy_development_card_outcomes", _belief_spy)
    monkeypatch.setattr(gcm, "buy_development_card_real_outcomes", _true_spy)

    node = _GNode(game=game.copy(), root_color=str(game.current_color()))
    mcts = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=1, belief_chance_spectra=True),
        HeuristicRustEvaluator(score_actions=False),
    )
    mcts._expand(node)
    action_id = next(
        aid for aid, stored in node.action_json.items()
        if stored[1] == "BUY_DEVELOPMENT_CARD"
    )
    stats = node.actions[action_id]
    mcts._traverse_robber_or_dev(node, action_id, stats, depth=0)

    assert belief_calls, "belief_chance_spectra=True must dispatch to belief_buy_development_card_outcomes"
    assert not true_calls, "belief_chance_spectra=True must NOT also consult the true-remaining-deck path"
