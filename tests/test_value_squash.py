"""Tests for `EntityGraphRustEvaluatorConfig.value_squash` (#60, default "tanh").

The value head is a raw Linear trained with MSE on z in {-1,+1}; "tanh"
(default) keeps the historical inference-time tanh(raw * value_scale)
bit-for-bit, "clip" drops the tanh and leaves the existing post-sign-flip
np.clip(-1, 1) as the only squash. tanh is odd and clip symmetric, so the
two modes must agree on SIGN and on the ORDERING of states; they differ only
in magnitude (|tanh(x)| <= |x|).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from catan_zero.search.rust_mcts import _require_rust_module


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def tiny_policy():
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    # create() leaves the model in train mode (unlike load(), which calls
    # eval()) -- active Dropout makes forwards nondeterministic, which would
    # break every value-equality assertion below.
    policy.model.eval()
    return policy


def _states(catanatron_rs, count: int = 6):
    """Fixed multi-action states at varying depths (deterministic)."""
    states = []
    for seed in range(3, 40):
        game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=seed)
        for step in range(240):
            if game.winning_color() is not None:
                break
            legal = [int(a) for a in game.playable_action_indices(["RED", "BLUE"], None)]
            if not legal:
                break
            if step > 15 and len(legal) >= 2:
                states.append(game.copy())
                break
            game.execute_action_index(legal[0], ["RED", "BLUE"], None)
        if len(states) >= count:
            break
    if len(states) < count:
        pytest.skip("not enough fixed states")
    return states


def _evaluate(policy, game, *, squash: str, root_color: str | None = None):
    evaluator = EntityGraphRustEvaluator(
        policy,
        config=EntityGraphRustEvaluatorConfig(value_squash=squash, cache_size=0),
    )
    legal = tuple(int(a) for a in game.playable_action_indices(["RED", "BLUE"], None))
    root = root_color if root_color is not None else str(game.current_color())
    return evaluator.evaluate(game, legal, root_color=root, colors=("RED", "BLUE"))


def test_default_mode_is_tanh():
    assert EntityGraphRustEvaluatorConfig().value_squash == "tanh"


def test_default_config_bit_identical_to_explicit_tanh(tiny_policy):
    """The default-constructed config must reproduce the historical values
    exactly: default == explicit 'tanh' == manual tanh(raw * value_scale)."""
    catanatron_rs = _rust()
    game = _states(catanatron_rs, count=1)[0]
    legal = tuple(int(a) for a in game.playable_action_indices(["RED", "BLUE"], None))
    root = str(game.current_color())

    default_eval = EntityGraphRustEvaluator(
        tiny_policy, config=EntityGraphRustEvaluatorConfig(cache_size=0)
    )
    priors_default, value_default = default_eval.evaluate(
        game, legal, root_color=root, colors=("RED", "BLUE")
    )
    priors_tanh, value_tanh = _evaluate(tiny_policy, game, squash="tanh")
    assert value_default == value_tanh
    assert priors_default == priors_tanh

    # Manual recomputation of the historical formula from the raw head output.
    captured = {}
    original_forward = tiny_policy.forward_legal_np

    def capturing_forward(*args, **kwargs):  # noqa: ANN002, ANN003
        out = original_forward(*args, **kwargs)
        captured["raw"] = float(out["value"].detach().float().cpu().numpy()[0])
        return out

    tiny_policy.forward_legal_np = capturing_forward
    try:
        _priors, value_again = _evaluate(tiny_policy, game, squash="tanh")
    finally:
        tiny_policy.forward_legal_np = original_forward
    expected = float(np.clip(math.tanh(captured["raw"] * 1.0), -1.0, 1.0))
    assert value_again == pytest.approx(expected, abs=0.0)


def test_clip_mode_preserves_sign_and_state_ordering(tiny_policy):
    catanatron_rs = _rust()
    states = _states(catanatron_rs)
    tanh_values, clip_values = [], []
    for game in states:
        root = str(game.current_color())
        _p, v_tanh = _evaluate(tiny_policy, game, squash="tanh", root_color=root)
        _p, v_clip = _evaluate(tiny_policy, game, squash="clip", root_color=root)
        tanh_values.append(v_tanh)
        clip_values.append(v_clip)
        # Same sign (tanh is odd, clip symmetric).
        assert math.copysign(1, v_tanh) == math.copysign(1, v_clip) or v_tanh == v_clip == 0.0
        # tanh compresses magnitude relative to the unsquashed value
        # (pre-clip); with |clip| capped at 1 the invariant is
        # |v_tanh| <= min(|raw|, 1) == |v_clip| whenever |raw| <= 1, and
        # v_clip saturates at +-1 otherwise -- either way |v_tanh| <= |v_clip|
        # can fail only if clip saturated BELOW tanh, impossible since
        # tanh(x) < 1 for finite x. So:
        assert abs(v_tanh) <= abs(v_clip) + 1e-12
    # Monotonic-equal ordering across the state set.
    assert np.argsort(tanh_values).tolist() == np.argsort(clip_values).tolist()
    # And priors are untouched by the squash mode.
    for game in states[:2]:
        p_tanh, _v = _evaluate(tiny_policy, game, squash="tanh")
        p_clip, _v = _evaluate(tiny_policy, game, squash="clip")
        assert p_tanh == p_clip


@pytest.mark.parametrize("squash", ["tanh", "clip"])
def test_opponent_sign_flip_respected_in_both_modes(tiny_policy, squash):
    catanatron_rs = _rust()
    game = _states(catanatron_rs, count=1)[0]
    acting = str(game.current_color())
    other = "BLUE" if acting == "RED" else "RED"
    _p, value_own = _evaluate(tiny_policy, game, squash=squash, root_color=acting)
    _p, value_opp = _evaluate(tiny_policy, game, squash=squash, root_color=other)
    assert value_opp == pytest.approx(-value_own, abs=1e-12)


def test_unknown_squash_mode_raises(tiny_policy):
    catanatron_rs = _rust()
    game = _states(catanatron_rs, count=1)[0]
    with pytest.raises(ValueError, match="value_squash"):
        _evaluate(tiny_policy, game, squash="sigmoid")


def test_batched_evaluate_many_matches_single_path_in_both_modes(tiny_policy):
    """evaluate_many (the batch path, second tanh call site) must agree with
    the single-state path value-for-value in both squash modes."""
    catanatron_rs = _rust()
    states = _states(catanatron_rs, count=3)
    for squash in ("tanh", "clip"):
        evaluator = EntityGraphRustEvaluator(
            tiny_policy,
            config=EntityGraphRustEvaluatorConfig(value_squash=squash, cache_size=0),
        )
        requests = []
        singles = []
        root = str(states[0].current_color())
        for game in states:
            legal = tuple(int(a) for a in game.playable_action_indices(["RED", "BLUE"], None))
            requests.append((game, legal))
            singles.append(
                evaluator.evaluate(game, legal, root_color=root, colors=("RED", "BLUE"))
            )
        batched = evaluator.evaluate_many(requests, root_color=root, colors=("RED", "BLUE"))
        for (p_single, v_single), (p_batch, v_batch) in zip(singles, batched):
            assert v_batch == pytest.approx(v_single, abs=1e-6)
            for action in p_single:
                assert p_batch[action] == pytest.approx(p_single[action], abs=1e-6)
