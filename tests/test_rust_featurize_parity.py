"""Bit-exact parity suite: Rust featurizer (task #81 phases 1-2, single-item +
batch) vs the existing Python `entity_token_features.build_entity_token_features`
path, on the same live `catanatron_rs.Game` states.

Requires the local `catanatron-rs` Python extension to be built WITH the
task-#81 functions (`build_entity_features_flat`/`EntityTopology`), not just
importable -- an OLDER installed wheel (e.g. 0.1.2) still imports fine but
lacks these specific functions, so `pytest.importorskip("catanatron_rs")`
alone is not enough to skip cleanly there; see
`scratch/catanatron-rs/python` + `maturin develop` to build a wheel with them.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

catanatron_rs = pytest.importorskip("catanatron_rs")

pytestmark = pytest.mark.skipif(
    not hasattr(catanatron_rs, "build_entity_features_flat"),
    reason="catanatron_rs with build_entity_features_flat (task #81) not installed",
)

from catan_zero.rl.entity_token_features import (  # noqa: E402
    PLAYER_ACTOR_FLAG_SLOT,
    PUBLIC_MASK_PLAYER_SLOTS,
)
from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V3,
    RUST_ENTITY_ADAPTER_V4,
    RUST_ENTITY_ADAPTER_V5,
    RUST_ENTITY_ADAPTER_V6,
)
from catan_zero.deduction_tracker import DEDUCTION_FEATURES_KEY  # noqa: E402
from catan_zero.rl.entity_token_features_rust import (  # noqa: E402
    build_entity_features_batch_rust,
    build_entity_features_rust,
    compute_rust_topology,
)
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    _RustEntityFeatureEnv,
    _resolve_entity_adapter,
    rust_game_to_entity_batch,
    rust_policy_action_ids,
)


COLORS = ("RED", "BLUE")
ACTION_SIZE = 400  # generous upper bound; only used to scale a feature, not to bound legal ids


def _collect_states(num_games: int, max_ticks: int, seed_base: int) -> list[tuple]:
    """Drive real random-vs-random games, capturing (game, colors) at every
    decision point across the whole game (opening/mid/late/robber/dev-card),
    by snapshotting a COPY of the game before each tick."""
    states: list[tuple] = []
    for game_index in range(num_games):
        game = catanatron_rs.Game.random(colors=list(COLORS), seed=seed_base + game_index)
        ticks = 0
        while game.winning_color() is None and ticks < max_ticks:
            legal = game.playable_action_indices(list(COLORS), None)
            if legal:
                states.append((game.copy(),))
            game.play_tick()
            ticks += 1
    return states


def _reference_entity(
    game,
    actor,
    legal_action_ids,
    policy_action_ids,
    public_observation,
    adapter_version=CURRENT_RUST_ENTITY_ADAPTER_VERSION,
):
    resolved = _resolve_entity_adapter(
        game,
        legal_action_ids,
        colors=COLORS,
        action_size=ACTION_SIZE,
        policy_action_ids=policy_action_ids,
        snapshot=None,
        action_by_id=None,
        public_observation=public_observation,
        perspective=actor,
        entity_feature_adapter_version=adapter_version,
    )
    batched = rust_game_to_entity_batch(
        game,
        legal_action_ids,
        actor=actor,
        colors=COLORS,
        action_size=ACTION_SIZE,
        policy_action_ids=policy_action_ids,
        public_observation=public_observation,
        entity_feature_adapter_version=adapter_version,
        resolved=resolved,
    )
    return {key: value[0] for key, value in batched.items()}


def _rust_entity(
    game,
    policy_action_ids,
    public_observation,
    topology,
    adapter_version=CURRENT_RUST_ENTITY_ADAPTER_VERSION,
):
    # NOTE: no `perspective`/`legal_action_ids` params anymore -- the Rust
    # builder derives perspective from `game.current_color()` and requires
    # `policy_action_ids` index-aligned to `game.playable_actions`'s native
    # order, which `legal_action_ids = game.playable_action_indices(...)`
    # (used to compute `policy_action_ids` below) always satisfies.
    return build_entity_features_rust(
        game,
        colors=COLORS,
        policy_action_ids=policy_action_ids,
        action_size=ACTION_SIZE,
        topology=topology,
        public_observation=public_observation,
        entity_feature_adapter_version=adapter_version,
    )


ALL_KEYS = (
    "hex_tokens",
    "hex_vertex_ids",
    "hex_edge_ids",
    "vertex_tokens",
    "edge_tokens",
    "edge_vertex_ids",
    "player_tokens",
    DEDUCTION_FEATURES_KEY,
    "global_tokens",
    "legal_action_tokens",
    "legal_action_target_ids",
    "event_tokens",
    "event_target_ids",
    "hex_mask",
    "vertex_mask",
    "edge_mask",
    "player_mask",
    "legal_action_mask",
    "event_mask",
)


@pytest.fixture(scope="module")
def states():
    # 30 full random-vs-random games up to 200 ticks each covers opening
    # (BUILD_INITIAL_SETTLEMENT/ROAD), mid/late PLAY_TURN, MOVE_ROBBER,
    # DISCARD, and (empirically verified) BUY_DEVELOPMENT_CARD/
    # PLAY_KNIGHT_CARD/PLAY_YEAR_OF_PLENTY/PLAY_MONOPOLY/PLAY_ROAD_BUILDING/
    # MARITIME_TRADE/BUILD_CITY legal actions among the sampled states.
    collected = _collect_states(num_games=30, max_ticks=200, seed_base=3000)
    random.Random(0).shuffle(collected)
    return collected[:260]


@pytest.mark.parametrize("public_observation", [False, True])
def test_bit_exact_parity_across_diverse_states(states, public_observation):
    assert len(states) >= 200, f"need >=200 states for the parity gate, got {len(states)}"

    mismatches: list[str] = []
    longest_road_award_states = 0
    for (game,) in states:
        if any(
            bool(json.loads(game.player_state_json(color))["has_road"])
            for color in COLORS
        ):
            longest_road_award_states += 1
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

        reference = _reference_entity(game, actor, legal_action_ids, policy_action_ids, public_observation)
        rust = _rust_entity(game, policy_action_ids, public_observation, topology)

        for key in ALL_KEYS:
            ref_arr = np.asarray(reference[key])
            rust_arr = np.asarray(rust[key])
            if ref_arr.shape != rust_arr.shape or not np.array_equal(ref_arr, rust_arr):
                mismatches.append(
                    f"actor={actor} public_observation={public_observation} key={key}: "
                    f"shapes ref={ref_arr.shape} rust={rust_arr.shape} equal={np.array_equal(ref_arr, rust_arr) if ref_arr.shape == rust_arr.shape else 'shape-mismatch'}"
                )

    assert not mismatches, "bit-exact parity FAILED for:\n" + "\n".join(mismatches[:40])
    assert longest_road_award_states > 0, (
        "parity corpus did not exercise public longest-road ownership"
    )


@pytest.mark.parametrize("public_observation", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("wave_size", [1, 24])
def test_batch_matches_single_item(states, public_observation, parallel, wave_size):
    """Batch API parity: for a wave of DISTINCT games sharing one board/color
    set, `build_entity_features_batch_rust` must produce, for every row, the
    SAME arrays (after un-padding to that row's own width) as calling
    `build_entity_features_rust` on that game individually -- proving the
    padding/stacking is correct and `parallel=True` (rayon) doesn't change
    results vs sequential. `wave_size=1` exercises the batch-1 fast path
    (skips the Rust batch call entirely -- see
    `build_entity_features_batch_rust`'s docstring).
    """
    # A "wave" = several distinct game states sharing colors/topology, mimicking
    # a Sequential-Halving round or chance-node expansion batch.
    wave = [game for (game,) in states[:wave_size]]
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
            public_observation=public_observation,
            perspective=actor,
        )[0],
        action_size=ACTION_SIZE,
    )
    topology = compute_rust_topology(adapter_env, actor)

    per_game_policy_ids = []
    per_game_single = []
    for game in wave:
        legal_action_ids = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
        assert legal_action_ids, "sampled states always have >=1 legal action"
        policy_action_ids = rust_policy_action_ids(
            game, legal_action_ids, colors=COLORS, action_size=ACTION_SIZE
        )
        per_game_policy_ids.append(policy_action_ids)
        per_game_single.append(_rust_entity(game, policy_action_ids, public_observation, topology))

    batch, widths = build_entity_features_batch_rust(
        wave,
        colors=COLORS,
        policy_action_ids=per_game_policy_ids,
        action_size=ACTION_SIZE,
        topology=topology,
        public_observation=public_observation,
        parallel=parallel,
    )

    assert widths == [single["legal_action_mask"].shape[0] for single in per_game_single]
    assert batch["hex_tokens"].shape[0] == len(wave)

    mismatches: list[str] = []
    for row, (single, width) in enumerate(zip(per_game_single, widths)):
        for key in ALL_KEYS:
            if key.startswith("legal_action_"):
                batch_row = batch[key][row, :width]
            else:
                batch_row = batch[key][row]
            single_arr = single[key]
            if batch_row.shape != single_arr.shape or not np.array_equal(batch_row, single_arr):
                mismatches.append(f"row={row} key={key}: batch vs single mismatch")
        # Padding region must be the documented fill value, not garbage.
        if width < batch["legal_action_mask"].shape[1]:
            assert not batch["legal_action_mask"][row, width:].any(), "mask padding must be False"
            assert not batch["legal_action_target_ids"][row, width:].any() or (
                batch["legal_action_target_ids"][row, width:] == -1
            ).all(), "target-id padding must be -1"
            assert (batch["legal_action_tokens"][row, width:] == 0.0).all(), "token padding must be 0.0"

    assert not mismatches, "batch vs single-item parity FAILED for:\n" + "\n".join(mismatches[:40])


def test_v6_resource_scales_python_native_single_and_batch_parity(states):
    game = max(
        (candidate for (candidate,) in states),
        key=lambda candidate: sum(
            map(
                int,
                json.loads(
                    candidate.player_state_json(candidate.current_color())
                )["resources"],
            )
        ),
    )
    actor = game.current_color()
    legal_action_ids = tuple(
        int(action) for action in game.playable_action_indices(list(COLORS), None)
    )
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
    topology = compute_rust_topology(
        _RustEntityFeatureEnv(resolved[0], action_size=ACTION_SIZE), actor
    )

    python_v6 = _reference_entity(
        game,
        actor,
        legal_action_ids,
        policy_action_ids,
        True,
        RUST_ENTITY_ADAPTER_V6,
    )
    native_v6 = _rust_entity(
        game,
        policy_action_ids,
        True,
        topology,
        RUST_ENTITY_ADAPTER_V6,
    )
    for key in ALL_KEYS:
        assert np.array_equal(python_v6[key], native_v6[key]), key

    player_tokens = np.asarray(native_v6["player_tokens"], dtype=np.float32)
    actor_row = player_tokens[player_tokens[:, PLAYER_ACTOR_FLAG_SLOT] > 0.5]
    assert actor_row.shape[0] == 1
    decoded_total = int(np.rint(actor_row[0, 6] * 95.0))
    decoded_composition = np.rint(actor_row[0, 16:21] * 19.0).astype(int)
    authoritative_resources = json.loads(game.player_state_json(actor))["resources"]
    expected_composition = np.asarray(authoritative_resources, dtype=int)
    assert decoded_total > 0
    assert decoded_total == int(expected_composition.sum())
    assert np.array_equal(decoded_composition, expected_composition)
    assert decoded_total == int(decoded_composition.sum())

    batch, widths = build_entity_features_batch_rust(
        [game.copy(), game.copy()],
        colors=COLORS,
        policy_action_ids=[policy_action_ids, policy_action_ids],
        action_size=ACTION_SIZE,
        topology=topology,
        public_observation=True,
        parallel=True,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )
    assert widths == [len(legal_action_ids), len(legal_action_ids)]
    for key in ALL_KEYS:
        expected = native_v6[key]
        if key.startswith("legal_action_"):
            assert np.array_equal(batch[key][0, : widths[0]], expected), key
            assert np.array_equal(batch[key][1, : widths[1]], expected), key
        else:
            assert np.array_equal(batch[key][0], expected), key
            assert np.array_equal(batch[key][1], expected), key

    legacy_player_rows = []
    for adapter_version in (
        RUST_ENTITY_ADAPTER_V2,
        RUST_ENTITY_ADAPTER_V3,
        RUST_ENTITY_ADAPTER_V4,
        RUST_ENTITY_ADAPTER_V5,
    ):
        legacy_player_rows.append(
            _rust_entity(
                game,
                policy_action_ids,
                True,
                topology,
                adapter_version,
            )["player_tokens"]
        )
    for rows in legacy_player_rows[1:]:
        assert np.array_equal(
            legacy_player_rows[0][:, [6, 16, 17, 18, 19, 20]],
            rows[:, [6, 16, 17, 18, 19, 20]],
        )
    legacy_actor_row = legacy_player_rows[-1][
        legacy_player_rows[-1][:, PLAYER_ACTOR_FLAG_SLOT] > 0.5
    ][0]
    assert not np.array_equal(
        legacy_actor_row[[6, 16, 17, 18, 19, 20]],
        actor_row[0, [6, 16, 17, 18, 19, 20]],
    )


def test_public_observation_masks_opponent_hidden_slots(states):
    """Masked-regime invariance check: for every non-actor player row, the
    slots `mask_player_tokens_public` zeroes must be exactly zero in the Rust
    output, regardless of the opponent's true (hidden) hand contents -- the
    property that matters for the hidden-information leak fix (f72/#71).
    """
    checked = 0
    for (game,) in states:
        actor = game.current_color()
        legal_action_ids = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
        if not legal_action_ids:
            continue
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
        )
        adapter_env = _RustEntityFeatureEnv(resolved[0], action_size=ACTION_SIZE)
        topology = compute_rust_topology(adapter_env, actor)
        rust = _rust_entity(game, policy_action_ids, True, topology)

        player_tokens = rust["player_tokens"]
        actor_rows = player_tokens[:, PLAYER_ACTOR_FLAG_SLOT] > 0.5
        for row in range(player_tokens.shape[0]):
            if player_tokens[row, 0] < 0.5:  # not a present player row
                continue
            if actor_rows[row]:
                continue
            for slot in PUBLIC_MASK_PLAYER_SLOTS:
                assert player_tokens[row, slot] == 0.0, (
                    f"masked slot {slot} not zeroed for non-actor row {row}"
                )
        checked += 1
    assert checked >= 50, f"expected to check >=50 states, checked {checked}"
