"""Wiring parity for the CONTEXT half of `EntityGraphRustEvaluatorConfig.rust_featurize`
(task #81 phase 4/8: context wiring into the evaluator).

`tests/test_rust_action_context_parity.py` proves the Rust context-feature
FUNCTION (`build_action_context_rust`) is bit-exact vs `_context_vector`. This
suite proves the EVALUATOR WIRING is: `_context_batch_via_rust` (flag ON) must
hand the forward pass an array bit-identical -- same dtype, shape, values --
to what the flag-OFF path (`rust_action_context_batch`) builds on the same
real game states, in BOTH masking regimes, including reuse of the SAME lazily-
bootstrapped `self._rust_topology` the entity path builds (whichever of
entity/context runs first in a given evaluate() call bootstraps it for both).

Same torch-free pattern as `test_rust_featurize_evaluator_wiring.py`: this is
the core of the end-to-end gate; the full checkpoint-loaded output check and
identical-seed smoke still run on a GPU host before fleet adoption.

Needs the catanatron_rs extension WITH `build_action_context_flat`; skips
cleanly on older wheels.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

try:
    import catanatron_rs

    _HAS_RUST_CONTEXT = hasattr(catanatron_rs, "build_action_context_flat")
except ImportError:
    catanatron_rs = None  # type: ignore[assignment]
    _HAS_RUST_CONTEXT = False

needs_rust_context = pytest.mark.skipif(
    not _HAS_RUST_CONTEXT,
    reason="catanatron_rs with build_action_context_flat (task #81) not installed",
)

COLORS: tuple[str, ...] = ("RED", "BLUE")


def _make_evaluator(public_observation: bool):
    """EntityGraphRustEvaluator with a torch-free dummy policy: this suite
    only exercises the featurize seam, never `forward_legal_np`."""
    from catan_zero.rl.action_mask import ActionCatalog
    from catan_zero.search.neural_rust_mcts import (
        EntityGraphRustEvaluator,
        EntityGraphRustEvaluatorConfig,
    )

    policy = SimpleNamespace(
        action_size=ActionCatalog(COLORS).size,
        trained_with_masked_hidden_info=public_observation,
    )
    return EntityGraphRustEvaluator(
        policy,  # type: ignore[arg-type]
        config=EntityGraphRustEvaluatorConfig(
            public_observation=public_observation, rust_featurize=True
        ),
    )


def _collect_states(seed: int, count: int) -> list:
    games = []
    game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
    step = 0
    while len(games) < count and game.winning_color() is None and step < 400:
        legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
        if legal and step % 7 == 0:
            games.append(game.copy())
        game.play_tick()
        step += 1
    return games


@needs_rust_context
@pytest.mark.parametrize("public_observation", [False, True])
def test_rust_context_wiring_matches_python_path(public_observation: bool) -> None:
    from catan_zero.search.neural_rust_mcts import (
        _fetch_leaf_decision_inputs,
        _resolve_entity_adapter,
        rust_action_context_batch,
        rust_policy_action_ids,
    )
    import json

    evaluator = _make_evaluator(public_observation)
    states = _collect_states(seed=17, count=25)
    assert len(states) >= 15, "trajectory too short to be a meaningful sample"

    compared = 0
    for game in states:
        acting_color = str(game.current_color())
        snapshot_text, action_by_id = _fetch_leaf_decision_inputs(game, COLORS)
        legal_actions = tuple(action_by_id.keys())
        policy_action_ids = rust_policy_action_ids(
            game,
            legal_actions,
            colors=COLORS,
            action_size=int(evaluator.policy.action_size),
            action_by_id=action_by_id,
        )
        resolved = _resolve_entity_adapter(
            game,
            legal_actions,
            colors=COLORS,
            action_size=int(evaluator.policy.action_size),
            policy_action_ids=policy_action_ids,
            snapshot=json.loads(snapshot_text),
            action_by_id=action_by_id,
            public_observation=public_observation,
            perspective=acting_color,
        )
        python_context = rust_action_context_batch(
            game,
            legal_actions,
            actor=acting_color,
            colors=COLORS,
            action_size=int(evaluator.policy.action_size),
            fill=float(evaluator.config.context_fill),
            policy_action_ids=policy_action_ids,
            public_observation=public_observation,
            resolved=resolved,
        )
        rust_context = evaluator._context_batch_via_rust(
            game,
            acting_color=acting_color,
            adapter=resolved[1],
        )

        expected = np.asarray(python_context)
        got = np.asarray(rust_context)
        assert got.dtype == expected.dtype, f"dtype {got.dtype} != {expected.dtype}"
        assert got.shape == expected.shape, f"shape {got.shape} != {expected.shape}"
        assert np.array_equal(got, expected), "context values differ"
        compared += 1
    assert compared > 0


@needs_rust_context
def test_context_topology_reuses_entity_bootstrap() -> None:
    """Whichever featurizer runs first in a call should bootstrap
    `self._rust_topology` for both -- this proves the context path does not
    silently rebuild its own topology when the entity path already primed it."""
    import json

    from catan_zero.search.neural_rust_mcts import (
        _fetch_leaf_decision_inputs,
        _resolve_entity_adapter,
        rust_policy_action_ids,
    )

    evaluator = _make_evaluator(False)
    assert evaluator._rust_topology is None
    states = _collect_states(seed=23, count=3)

    for game in states:
        acting_color = str(game.current_color())
        snapshot_text, action_by_id = _fetch_leaf_decision_inputs(game, COLORS)
        legal_actions = tuple(action_by_id.keys())
        policy_action_ids = rust_policy_action_ids(
            game,
            legal_actions,
            colors=COLORS,
            action_size=int(evaluator.policy.action_size),
            action_by_id=action_by_id,
        )
        resolved = _resolve_entity_adapter(
            game,
            legal_actions,
            colors=COLORS,
            action_size=int(evaluator.policy.action_size),
            policy_action_ids=policy_action_ids,
            snapshot=json.loads(snapshot_text),
            action_by_id=action_by_id,
            public_observation=False,
            perspective=acting_color,
        )
        # Entity path first (primes the shared topology)...
        evaluator._entity_batch_via_rust(
            game,
            colors=COLORS,
            policy_action_ids=policy_action_ids,
            acting_color=acting_color,
            adapter=resolved[1],
        )
        topology_after_entity = evaluator._rust_topology
        assert topology_after_entity is not None
        # ...then context path must reuse it, not rebuild.
        evaluator._context_batch_via_rust(
            game,
            acting_color=acting_color,
            adapter=resolved[1],
        )
        assert evaluator._rust_topology is topology_after_entity


@needs_rust_context
def test_context_bootstraps_topology_when_run_first() -> None:
    """The reverse ordering: if context runs BEFORE entity in a given call
    (not the production order, but must still be correct), it must bootstrap
    the shared topology itself rather than crashing or building a throwaway
    one that entity then silently discards."""
    import json

    from catan_zero.search.neural_rust_mcts import (
        _fetch_leaf_decision_inputs,
        _resolve_entity_adapter,
        rust_policy_action_ids,
    )

    evaluator = _make_evaluator(False)
    assert evaluator._rust_topology is None
    game = _collect_states(seed=29, count=1)[0]
    acting_color = str(game.current_color())
    snapshot_text, action_by_id = _fetch_leaf_decision_inputs(game, COLORS)
    legal_actions = tuple(action_by_id.keys())
    policy_action_ids = rust_policy_action_ids(
        game,
        legal_actions,
        colors=COLORS,
        action_size=int(evaluator.policy.action_size),
        action_by_id=action_by_id,
    )
    resolved = _resolve_entity_adapter(
        game,
        legal_actions,
        colors=COLORS,
        action_size=int(evaluator.policy.action_size),
        policy_action_ids=policy_action_ids,
        snapshot=json.loads(snapshot_text),
        action_by_id=action_by_id,
        public_observation=False,
        perspective=acting_color,
    )
    evaluator._context_batch_via_rust(
        game,
        acting_color=acting_color,
        adapter=resolved[1],
    )
    topology_after_context = evaluator._rust_topology
    assert topology_after_context is not None

    evaluator._entity_batch_via_rust(
        game,
        colors=COLORS,
        policy_action_ids=policy_action_ids,
        acting_color=acting_color,
        adapter=resolved[1],
    )
    assert evaluator._rust_topology is topology_after_context


# ---------------------------------------------------------------------------
# `evaluate_symmetry_averaged` end-to-end wiring (speed-czar's ask, task #81
# gap #1): the helper-level tests above prove `_entity_batch_via_rust`/
# `_context_batch_via_rust` are bit-identical in isolation, but
# `evaluate_symmetry_averaged` feeds their output through
# `hex_symmetry.average_forward` (12 D6 board-orientation permutations) and a
# REAL forward pass before producing (priors, value) -- this proves the
# method's actual OUTPUT is unaffected by the `rust_featurize` flag, not just
# its inputs. Needs a real (if tiny) policy since `average_forward` requires
# an actual `forward_fn` to call per symmetry -- same `_tiny_real_policy`
# fixture pattern as `tests/test_evaluator_shared_payload.py`.
# ---------------------------------------------------------------------------


def _tiny_real_policy():
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    policy.model.eval()  # create() leaves train mode; active Dropout would break equality.
    return policy


def _advance_to_wide_root(game, *, min_legal: int = 3, max_steps: int = 300):
    for _ in range(max_steps):
        if game.winning_color() is not None:
            break
        legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
        if len(legal) >= min_legal:
            return game
        game.play_tick()
    raise AssertionError(f"did not reach a state with >= {min_legal} legal actions")


@needs_rust_context
@pytest.mark.parametrize("public_observation", [False, True])
def test_evaluate_symmetry_averaged_matches_between_rust_featurize_flag(
    public_observation: bool,
) -> None:
    from catan_zero.search.neural_rust_mcts import (
        EntityGraphRustEvaluator,
        EntityGraphRustEvaluatorConfig,
    )

    policy = _tiny_real_policy()
    # This fixture's masking-regime attribute isn't tied to the tiny model's
    # architecture (it's checkpoint metadata elsewhere) -- set it directly so
    # BOTH regimes actually exercise the fail-closed guard's matching branch,
    # rather than skipping one parametrization.
    policy.trained_with_masked_hidden_info = public_observation

    evaluator_off = EntityGraphRustEvaluator(
        policy,
        config=EntityGraphRustEvaluatorConfig(
            public_observation=public_observation, rust_featurize=False
        ),
    )
    evaluator_on = EntityGraphRustEvaluator(
        policy,
        config=EntityGraphRustEvaluatorConfig(
            public_observation=public_observation, rust_featurize=True
        ),
    )

    compared = 0
    for seed in (101, 102, 103):
        game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
        game = _advance_to_wide_root(game, min_legal=3)
        actor = str(game.current_color())
        legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))

        priors_off, value_off = evaluator_off.evaluate_symmetry_averaged(
            game.copy(), legal, root_color=actor, colors=COLORS
        )
        priors_on, value_on = evaluator_on.evaluate_symmetry_averaged(
            game.copy(), legal, root_color=actor, colors=COLORS
        )

        assert set(priors_off) == set(priors_on), (
            f"seed={seed}: prior key mismatch {set(priors_off) ^ set(priors_on)}"
        )
        for action in priors_off:
            assert np.isclose(priors_off[action], priors_on[action], atol=1e-6, rtol=1e-5), (
                f"seed={seed} action={action}: prior {priors_off[action]!r} != "
                f"{priors_on[action]!r}"
            )
        assert np.isclose(value_off, value_on, atol=1e-6, rtol=1e-5), (
            f"seed={seed}: value {value_off!r} != {value_on!r}"
        )
        compared += 1
    assert compared > 0
