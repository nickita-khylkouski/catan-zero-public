"""Bit-exact parity suite: Rust action-context featurizer (task #81 "context
lever") vs the existing Python `neural_rust_mcts.rust_action_context_batch`
path, on the same live `catanatron_rs.Game` states.

Requires the local `catanatron-rs` Python extension to be built WITH the
task-#81 functions (`build_action_context_flat`), not just importable -- an
OLDER installed wheel (e.g. 0.1.2) still imports fine but lacks this specific
function, so `pytest.importorskip("catanatron_rs")` alone is not enough to
skip cleanly there; see `scratch/catanatron-rs/python` + `maturin develop` to
build a wheel with it.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

catanatron_rs = pytest.importorskip("catanatron_rs")

pytestmark = pytest.mark.skipif(
    not hasattr(catanatron_rs, "build_action_context_flat"),
    reason="catanatron_rs with build_action_context_flat (task #81) not installed",
)

from catan_zero.rl.action_context_features_rust import (  # noqa: E402
    build_action_context_batch_rust,
    build_action_context_rust,
)
from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V6,
)
from catan_zero.rl.entity_token_features_rust import compute_rust_topology  # noqa: E402
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    _RustEntityFeatureEnv,
    _resolve_entity_adapter,
    rust_action_context_batch,
    rust_policy_action_ids,
)

COLORS = ("RED", "BLUE")
ACTION_SIZE = 400


def _collect_states(num_games: int, max_ticks: int, seed_base: int) -> list:
    states = []
    for game_index in range(num_games):
        game = catanatron_rs.Game.random(colors=list(COLORS), seed=seed_base + game_index)
        ticks = 0
        while game.winning_color() is None and ticks < max_ticks:
            legal = game.playable_action_indices(list(COLORS), None)
            if legal:
                states.append(game.copy())
            game.play_tick()
            ticks += 1
    return states


@pytest.fixture(scope="module")
def states():
    collected = _collect_states(num_games=30, max_ticks=200, seed_base=4000)
    random.Random(1).shuffle(collected)
    return collected[:260]


def _reference_context(game, actor, legal_action_ids, policy_action_ids, public_observation):
    batched = rust_action_context_batch(
        game,
        legal_action_ids,
        actor=actor,
        colors=COLORS,
        action_size=ACTION_SIZE,
        policy_action_ids=policy_action_ids,
        public_observation=public_observation,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V2,
    )
    return batched[0]


@pytest.mark.parametrize("public_observation", [False, True])
def test_bit_exact_parity_across_diverse_states(states, public_observation):
    # public_observation shouldn't affect context features at all (they never
    # read opponent hand composition), but the reference call still accepts
    # the flag -- assert parity in both regimes to prove that's actually true
    # rather than assumed.
    assert len(states) >= 200

    mismatches: list[str] = []
    for game in states:
        actor = game.current_color()
        legal_action_ids = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
        if not legal_action_ids:
            continue
        policy_action_ids = rust_policy_action_ids(
            game, legal_action_ids, colors=COLORS, action_size=ACTION_SIZE
        )
        adapter_env = _RustEntityFeatureEnv(
            _resolve_entity_adapter(
                game,
                legal_action_ids,
                colors=COLORS,
                action_size=ACTION_SIZE,
                policy_action_ids=policy_action_ids,
                snapshot=None,
                action_by_id=None,
                public_observation=public_observation,
                perspective=actor,
            )[0],
            action_size=ACTION_SIZE,
        )
        topology = compute_rust_topology(adapter_env, actor)

        reference = _reference_context(game, actor, legal_action_ids, policy_action_ids, public_observation)
        rust = build_action_context_rust(
            game,
            topology=topology,
            entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V2,
        )

        if reference.shape != rust.shape or not np.array_equal(reference, rust):
            mismatches.append(
                f"actor={actor} public_observation={public_observation}: "
                f"shapes ref={reference.shape} rust={rust.shape} "
                f"equal={np.array_equal(reference, rust) if reference.shape == rust.shape else 'shape-mismatch'}"
            )

    assert not mismatches, "bit-exact context parity FAILED for:\n" + "\n".join(mismatches[:40])


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("wave_size", [1, 24])
def test_batch_matches_single_item(states, parallel, wave_size):
    wave = states[:wave_size]
    actor = wave[0].current_color()
    adapter_env = _RustEntityFeatureEnv(
        _resolve_entity_adapter(
            wave[0],
            tuple(int(a) for a in wave[0].playable_action_indices(list(COLORS), None)),
            colors=COLORS,
            action_size=ACTION_SIZE,
            policy_action_ids=None,
            snapshot=None,
            action_by_id=None,
            public_observation=False,
            perspective=actor,
        )[0],
        action_size=ACTION_SIZE,
    )
    topology = compute_rust_topology(adapter_env, actor)

    per_game_single = [
        build_action_context_rust(
            game,
            topology=topology,
            entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V2,
        )
        for game in wave
    ]
    batch, widths = build_action_context_batch_rust(
        wave,
        topology=topology,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V2,
        parallel=parallel,
    )

    assert widths == [single.shape[0] for single in per_game_single]
    assert batch.shape[0] == len(wave)

    mismatches: list[str] = []
    for row, (single, width) in enumerate(zip(per_game_single, widths)):
        batch_row = batch[row, :width]
        if batch_row.shape != single.shape or not np.array_equal(batch_row, single):
            mismatches.append(f"row={row}: batch vs single mismatch")
        if width < batch.shape[1]:
            assert (batch[row, width:] == 0.0).all(), "context padding must be 0.0"

    assert not mismatches, "batch vs single-item context parity FAILED for:\n" + "\n".join(mismatches[:40])


def test_v6_initial_road_two_hop_python_native_single_and_batch_parity():
    if not hasattr(catanatron_rs, "supported_action_context_adapter_versions"):
        pytest.skip("local catanatron_rs wheel predates adapter-v6 context support")
    if RUST_ENTITY_ADAPTER_V6 not in set(
        catanatron_rs.supported_action_context_adapter_versions()
    ):
        pytest.skip("local catanatron_rs wheel does not implement adapter v6")

    game = catanatron_rs.Game.random(colors=list(COLORS), seed=500_003)
    settlement_ids = tuple(
        int(action) for action in game.playable_action_indices(list(COLORS), None)
    )
    settlement_actions = json.loads(game.playable_actions_json())
    settlement_action_id = next(
        action_id
        for action_id, raw in zip(settlement_ids, settlement_actions)
        if raw[1] == "BUILD_SETTLEMENT" and int(raw[2]) == 15
    )
    game.execute_action_index(settlement_action_id, list(COLORS), None)

    actor = game.current_color()
    legal_action_ids = tuple(
        int(action) for action in game.playable_action_indices(list(COLORS), None)
    )
    raw_actions = json.loads(game.playable_actions_json())
    policy_action_ids = rust_policy_action_ids(
        game, legal_action_ids, colors=COLORS, action_size=ACTION_SIZE
    )
    resolved = _resolve_entity_adapter(
        game,
        legal_action_ids,
        colors=COLORS,
        action_size=ACTION_SIZE,
        policy_action_ids=policy_action_ids,
        snapshot=None,
        action_by_id=None,
        public_observation=True,
        perspective=actor,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )
    adapter_env = _RustEntityFeatureEnv(resolved[0], action_size=ACTION_SIZE)
    topology = compute_rust_topology(adapter_env, actor)

    python_v6 = rust_action_context_batch(
        game,
        legal_action_ids,
        actor=actor,
        colors=COLORS,
        action_size=ACTION_SIZE,
        policy_action_ids=policy_action_ids,
        public_observation=True,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
        resolved=resolved,
    )[0]
    native_v6 = build_action_context_rust(
        game,
        topology=topology,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )
    assert np.array_equal(python_v6, native_v6)

    native_batch, widths = build_action_context_batch_rust(
        [game.copy(), game.copy()],
        topology=topology,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
        parallel=True,
    )
    assert widths == [len(legal_action_ids), len(legal_action_ids)]
    assert np.array_equal(native_batch[0, : widths[0]], native_v6)
    assert np.array_equal(native_batch[1, : widths[1]], native_v6)

    road_rows = {
        tuple(sorted(map(int, raw[2]))): row
        for row, raw in enumerate(raw_actions)
        if raw[1] == "BUILD_ROAD"
    }
    assert python_v6[road_rows[(4, 15)], 16] == pytest.approx(9.0 / 18.0)
    assert python_v6[road_rows[(15, 17)], 16] == pytest.approx(7.0 / 18.0)
