from __future__ import annotations

import dataclasses
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.action_mask import ActionCatalog
from catan_zero.rl.aux_subgoal_targets import (
    AUX_SUBGOAL_TARGET_SEMANTIC,
    AUX_SUBGOAL_TARGET_VERSION,
    AUX_SUBGOAL_TARGET_VERSION_KEY,
    AUX_TARGET_KEYS,
)
from catan_zero.rl.entity_feature_adapter import (
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V3,
)
from catan_zero.rl.flywheel.opponent_mix import (
    MixCategory,
    MixCheckpointRef,
    OpponentMixConfig,
)
from catan_zero.rl.gumbel_self_play import (
    COLORS,
    GumbelSelfPlayConfig,
    MixRuntime,
    TARGET_INFORMATION_REGIME_AUTHORITATIVE,
    TARGET_INFORMATION_REGIME_PUBLIC,
    TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
    _apply_selected_action,
    _full_search_simulation_accounting,
    _is_n128_reliability_result,
    _search_execution_contract,
    _target_information_regime_for_search,
    action_size_for_evaluator,
    learner_entity_adapter_for_generation,
    play_one_game,
    run_worker_games,
)
from catan_zero.search.gumbel_chance_mcts import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    HeuristicRustEvaluator,
)
from catan_zero.search.rust_mcts import _require_rust_module

# `tools/train_bc.py` does bare sibling imports (`from factory_common import ...`),
# so it only works with the `tools/` directory itself on sys.path (not just the
# repo root, which pytest.ini already puts `src/` on but not `tools/`).
_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from train_bc import _load_npz, _normalize_teacher_shard  # type: ignore  # noqa: E402


@pytest.mark.parametrize(
    ("legal_width", "expected"),
    ((3, 127), (16, 128), (26, 112), (40, 141), (54, 154)),
)
def test_n128_reliability_accepts_width_dependent_legacy_sh_counts(
    legal_width: int, expected: int
) -> None:
    config = GumbelChanceMCTSConfig(n_full=128, exact_budget_sh=False)
    result = SimpleNamespace(used_full_search=True, simulations_used=expected)

    assert _full_search_simulation_accounting(config, legal_width) == (128, expected)
    assert _is_n128_reliability_result(config, legal_width=legal_width, result=result)


def test_n128_reliability_refuses_wrong_count_or_non_n128_wide_budget() -> None:
    legacy = GumbelChanceMCTSConfig(n_full=128, exact_budget_sh=False)
    assert not _is_n128_reliability_result(
        legacy,
        legal_width=26,
        result=SimpleNamespace(used_full_search=True, simulations_used=128),
    )
    assert not _is_n128_reliability_result(
        legacy,
        legal_width=26,
        result=SimpleNamespace(used_full_search=False, simulations_used=112),
    )
    adaptive = GumbelChanceMCTSConfig(n_full=128, n_full_wide=256)
    nominal, expected = _full_search_simulation_accounting(adaptive, 54)
    assert nominal == 256
    assert not _is_n128_reliability_result(
        adaptive,
        legal_width=54,
        result=SimpleNamespace(used_full_search=True, simulations_used=expected),
    )


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def _catanatron_rs_version() -> tuple[int, ...]:
    try:
        from importlib.metadata import version

        raw = version("catanatron_rs")
    except Exception:
        return (0, 0, 0)
    parts: list[int] = []
    for part in raw.split(".")[:3]:
        digits = "".join(char for char in part if char.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


# A20 (BUY_DEVELOPMENT_CARD phantom outcome bug) was fixed in catanatron_rs
# 0.1.1; see the same gate in tests/test_gumbel_chance_mcts.py.
_A19_A20_FIXED_ON_WHEEL = _catanatron_rs_version() >= (0, 1, 1)


def _fast_config(
    *,
    max_decisions: int,
    n_full: int = 4,
    n_fast: int = 2,
    p_full: float = 1.0,
    seed: int = 0,
):
    self_play_config = GumbelSelfPlayConfig(max_decisions=max_decisions)
    search_config = GumbelChanceMCTSConfig(
        seed=seed, n_full=n_full, n_fast=n_fast, p_full=p_full, max_depth=40
    )
    return self_play_config, search_config


# ---------------------------------------------------------------------------
# action_size_for_evaluator
# ---------------------------------------------------------------------------


def test_action_size_for_evaluator_falls_back_to_action_catalog_for_heuristic():
    _rust()
    size = action_size_for_evaluator(HeuristicRustEvaluator(), COLORS)
    assert size == ActionCatalog(COLORS).size


def test_learner_row_adapter_can_explicitly_advance_beyond_teacher_adapter():
    evaluator = SimpleNamespace(
        config=SimpleNamespace(entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V2)
    )

    assert (
        learner_entity_adapter_for_generation(GumbelSelfPlayConfig(), evaluator)
        == RUST_ENTITY_ADAPTER_V2
    )
    assert (
        learner_entity_adapter_for_generation(
            GumbelSelfPlayConfig(
                learner_entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V3
            ),
            evaluator,
        )
        == RUST_ENTITY_ADAPTER_V3
    )


# ---------------------------------------------------------------------------
# Outcome-label zero-sum property (tested against a real, naturally-terminal
# game reached via fast random play_tick(), decoupled from the (much slower)
# Gumbel-search-driven game loop so this stays a fast, deterministic check of
# the labeling logic itself).
# ---------------------------------------------------------------------------


def test_game_outcome_fields_are_zero_sum_across_seats():
    catanatron_rs = _rust()
    from catan_zero.rl.gumbel_self_play import _game_outcome_fields

    game = catanatron_rs.Game.simple(list(COLORS), seed=3)
    for _ in range(6000):
        if game.winning_color() is not None:
            break
        game.play_tick()
    assert game.winning_color() is not None, (
        "expected a natural game completion within budget"
    )

    outcome = _game_outcome_fields(game, terminal=True, colors=COLORS)
    assert outcome["winner"] in COLORS
    assert outcome["has_final_public_vps"] is True
    assert outcome["has_final_actual_vps"] is True

    def _implied_value(player: str) -> float:
        return 1.0 if outcome["winner"] == player else -1.0

    red_value = _implied_value("RED")
    blue_value = _implied_value("BLUE")
    assert red_value == -blue_value
    assert {red_value, blue_value} == {1.0, -1.0}


def test_initial_settlement_and_road_change_perspective_only_after_road():
    """Opening settlement and its road are consecutive prompts for one actor.

    Search values are expressed from the root actor's perspective.  A sign
    change between the settlement root and its road child would therefore be
    wrong; perspective changes only after the road hands control to the next
    player.  This guards the exact transition behind the opening phase-value
    calibration audit.
    """
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(list(COLORS), seed=7)
    root_actor = str(game.current_color())

    def apply_first_action_with_type(position, action_type: str):
        action_ids = [
            int(action)
            for action in position.playable_action_indices(list(COLORS), None)
        ]
        actions = json.loads(position.playable_actions_json())
        action_id, action_json = next(
            (action_id, action_json)
            for action_id, action_json in zip(action_ids, actions)
            if str(action_json[1]) == action_type
        )
        return _apply_selected_action(
            position,
            action_id,
            colors=COLORS,
            rng=random.Random(0),
            correct_rust_chance_spectra=True,
            action_json=action_json,
        )

    after_settlement = apply_first_action_with_type(game, "BUILD_SETTLEMENT")
    settlement_snapshot = json.loads(after_settlement.json_snapshot())
    assert settlement_snapshot["current_prompt"] == "BUILD_INITIAL_ROAD"
    assert str(after_settlement.current_color()) == root_actor
    assert (1.0 if str(after_settlement.current_color()) == root_actor else -1.0) == 1.0

    after_road = apply_first_action_with_type(after_settlement, "BUILD_ROAD")
    road_snapshot = json.loads(after_road.json_snapshot())
    assert road_snapshot["current_prompt"] == "BUILD_INITIAL_SETTLEMENT"
    assert str(after_road.current_color()) != root_actor
    assert (1.0 if str(after_road.current_color()) == root_actor else -1.0) == -1.0


# ---------------------------------------------------------------------------
# Regression: "seat" must index into PLAYER_NAMES order, not `colors` order.
# COLORS=("RED","BLUE") gives RED=0 under colors.index(), but PLAYER_NAMES=
# ("BLUE","RED","ORANGE","WHITE") gives BLUE=0 -- final_public_vps/
# final_actual_vps are built in PLAYER_NAMES order, so using colors.index()
# would point every row's seat at the WRONG (opponent's) VP slot.
# ---------------------------------------------------------------------------


def test_decision_row_seat_matches_player_names_index_not_colors_index():
    _rust()
    self_play_config, search_config = _fast_config(max_decisions=6)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    mcts = GumbelChanceMCTS(search_config, evaluator)

    record = play_one_game(
        mcts,
        evaluator,
        config=self_play_config,
        game_seed=11,
        game_index=0,
        action_size=ActionCatalog(COLORS).size,
    )

    assert record.decisions, "expected at least one recorded decision"
    seen_players = set()
    for decision in record.decisions:
        player = str(decision.row["player"])
        seen_players.add(player)
        expected_seat = ("BLUE", "RED", "ORANGE", "WHITE").index(player)
        assert int(decision.row["seat"]) == expected_seat, (
            f"player={player!r} got seat={decision.row['seat']}, expected {expected_seat} "
            "(PLAYER_NAMES order) -- colors.index() would give a different, wrong answer "
            "here since COLORS=('RED','BLUE') != PLAYER_NAMES order"
        )
    # Sanity: both colors should appear across a multi-decision 2p game, and
    # colors.index() would disagree with PLAYER_NAMES.index() for RED
    # specifically (0 vs 1), so this test is only meaningful if RED shows up.
    assert "RED" in seen_players


def test_winner_row_seat_indexes_its_own_final_vp():
    catanatron_rs = _rust()
    from catan_zero.rl.gumbel_self_play import _game_outcome_fields

    # Fast random play to a REAL natural completion (not search-driven --
    # this only needs to exercise _game_outcome_fields' PLAYER_NAMES-ordered
    # VP arrays together with the same PLAYER_NAMES.index() seat convention
    # _build_decision_row uses, independent of trajectory).
    game = catanatron_rs.Game.simple(list(COLORS), seed=3)
    for _ in range(6000):
        if game.winning_color() is not None:
            break
        game.play_tick()
    assert game.winning_color() is not None

    outcome = _game_outcome_fields(game, terminal=True, colors=COLORS)
    winner = outcome["winner"]
    winner_seat = ("BLUE", "RED", "ORANGE", "WHITE").index(winner)
    assert int(outcome["final_actual_vps"][winner_seat]) >= 10
    # The loser's seat must NOT also show >=10 (guards against a degenerate
    # all-same-value array masking the seat-index bug).
    loser = "BLUE" if winner == "RED" else "RED"
    loser_seat = ("BLUE", "RED", "ORANGE", "WHITE").index(loser)
    assert int(outcome["final_actual_vps"][loser_seat]) < 10


# ---------------------------------------------------------------------------
# Truncated games must not fabricate outcome labels.
# ---------------------------------------------------------------------------


def test_truncated_game_produces_no_outcome_labels():
    _rust()
    self_play_config, search_config = _fast_config(max_decisions=3)
    mcts = GumbelChanceMCTS(search_config, HeuristicRustEvaluator(score_actions=False))

    record = play_one_game(
        mcts,
        HeuristicRustEvaluator(score_actions=False),
        config=self_play_config,
        game_seed=4242,
        game_index=0,
        action_size=ActionCatalog(COLORS).size,
    )

    assert record.truncated is True
    assert record.terminal is False
    assert record.winner == ""
    for decision in record.decisions:
        assert decision.row["winner"] == ""
        assert decision.row["truncated"] is True
        assert decision.row["terminated"] is False
        assert decision.row["has_final_public_vps"] is False
        assert decision.row["has_final_actual_vps"] is False


def test_play_one_game_materializes_aux_targets_from_full_rust_trajectory(monkeypatch):
    """Exercise production wiring without requiring a particular local Rust wheel.

    The real engine integration tests are version-gated above; this compact fake
    verifies that play_one_game captures every ply, reuses native action payloads,
    and stamps the recorded rows after the terminal state is known.
    """
    import catan_zero.rl.gumbel_self_play as gsp

    actions = (
        ["RED", "BUILD_SETTLEMENT", 5],
        ["BLUE", "ROLL", None],
        ["RED", "MOVE_ROBBER", [[-2, 0, 2], "BLUE"]],
    )

    class FakeGame:
        def __init__(self):
            self.step = 0

        def winning_color(self):
            return "RED" if self.step >= len(actions) else None

        def current_color(self):
            return ("RED", "BLUE", "RED")[self.step]

        def playable_action_indices(self, _colors, _map_kind):
            return [100]

        def playable_actions_json(self):
            return json.dumps([actions[self.step]])

        def json_snapshot(self):
            red_vp = (0, 1, 1, 3)[self.step]
            return json.dumps(
                {
                    "colors": ["RED", "BLUE"],
                    "current_prompt": "PLAY_TURN",
                    "player_state": [
                        {
                            "actual_victory_points": red_vp,
                            "has_road": self.step >= 3,
                            "has_army": False,
                        },
                        {
                            "actual_victory_points": 2,
                            "has_road": False,
                            "has_army": True,
                        },
                    ],
                    "tiles": [
                        {
                            "coordinate": [-2, 0, 2],
                            "tile": {"id": 11, "type": "RESOURCE_TILE"},
                        }
                    ],
                }
            )

        def player_state_json(self, color):
            if color == "RED":
                return json.dumps({"victory_points": 3, "actual_victory_points": 3})
            return json.dumps({"victory_points": 2, "actual_victory_points": 2})

    game = FakeGame()

    class FakeModule:
        class Game:
            @staticmethod
            def simple(_colors, seed):
                assert seed == 123
                return game

    class FakeMCTS:
        def __init__(self):
            self.config = GumbelChanceMCTSConfig(
                forced_root_target_mode="trajectory_only"
            )
            self.evaluator = None

        def search(self, _game, *, force_full):
            assert force_full is None
            return SimpleNamespace(selected_action=100, simulations_used=4)

    recorded_adapter_versions: list[str] = []

    def fake_build(game, **kwargs):
        recorded_adapter_versions.append(
            str(kwargs["entity_feature_adapter_version"])
        )
        return {
            "player": str(game.current_color()),
            "policy_weight_multiplier": np.float32(0.0),
            "value_weight_multiplier": np.float32(1.0),
            "root_value": np.float32(np.nan),
            "root_value_mask": np.bool_(False),
        }, {}

    def fake_apply(game, _action_index, **kwargs):
        assert kwargs["action_json"] == actions[game.step]
        game.step += 1
        return game

    monkeypatch.setattr(
        gsp._gumbel_chance_mcts, "_require_rust_module", lambda: FakeModule
    )
    monkeypatch.setattr(gsp, "_build_decision_row", fake_build)
    monkeypatch.setattr(gsp, "_apply_selected_action", fake_apply)

    record = play_one_game(
        FakeMCTS(),
        SimpleNamespace(
            config=SimpleNamespace(
                entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V2
            )
        ),
        config=GumbelSelfPlayConfig(
            max_decisions=3,
            learner_entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V3,
        ),
        game_seed=123,
        game_index=0,
        action_size=1,
    )

    assert record.terminal is True
    assert recorded_adapter_versions == [RUST_ENTITY_ADAPTER_V3] * 3
    assert record.forced_decisions == 3
    assert len(record.decisions) == 3
    for decision in record.decisions:
        assert decision.row["winner"] == "RED"
        assert decision.row["terminated"] is True
        assert decision.row["truncated"] is False
        assert float(decision.row["policy_weight_multiplier"]) == 0.0
        assert float(decision.row["value_weight_multiplier"]) == 1.0
        assert not bool(decision.row["root_value_mask"])
        assert np.isnan(decision.row["root_value"])
    red0 = record.decisions[0].row
    assert red0["aux_longest_road"] == np.float32(1.0)
    assert red0["aux_largest_army"] == np.float32(0.0)
    assert red0["aux_vp_in_n"] == np.float32(3.0)
    # Current-row actions are excluded from the strict-future aux contract.
    assert red0["aux_next_settlement"] == np.int16(-1)
    assert red0["aux_robber_target"] == np.int16(11)
    assert red0[AUX_SUBGOAL_TARGET_VERSION_KEY] == np.uint8(AUX_SUBGOAL_TARGET_VERSION)
    blue = record.decisions[1].row
    assert blue["aux_largest_army"] == np.float32(1.0)
    assert blue["aux_next_settlement"] == np.int16(-1)
    assert blue["aux_robber_target"] == np.int16(-1)


# ---------------------------------------------------------------------------
# Playout-cap-randomized fast rows advance trajectories and train value only.
# Production policy distillation remains exact n_full teacher supervision.
# ---------------------------------------------------------------------------


def test_fast_search_rows_are_value_only():
    _rust()
    self_play_config, search_config = _fast_config(max_decisions=5, p_full=0.0)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    mcts = GumbelChanceMCTS(search_config, evaluator)

    record = play_one_game(
        mcts,
        evaluator,
        config=self_play_config,
        game_seed=99,
        game_index=0,
        action_size=ActionCatalog(COLORS).size,
    )

    assert record.decisions, "expected at least one non-forced decision to be recorded"
    for decision in record.decisions:
        assert decision.row["used_full_search"] is False
        assert float(decision.row["policy_weight_multiplier"]) == pytest.approx(0.0)
        assert float(decision.row["value_weight_multiplier"]) == pytest.approx(1.0)
        assert not bool(decision.row["root_value_mask"])
        assert np.isnan(decision.row["root_value"])


def test_full_search_rows_have_nonzero_policy_weight():
    _rust()
    self_play_config, search_config = _fast_config(max_decisions=5, p_full=1.0)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    mcts = GumbelChanceMCTS(search_config, evaluator)

    record = play_one_game(
        mcts,
        evaluator,
        config=self_play_config,
        game_seed=99,
        game_index=0,
        action_size=ActionCatalog(COLORS).size,
    )

    assert record.decisions, "expected at least one non-forced decision to be recorded"
    for decision in record.decisions:
        assert decision.row["used_full_search"] is True
        assert float(decision.row["policy_weight_multiplier"]) == pytest.approx(1.0)
        if not decision.row["is_forced"]:
            assert bool(decision.row["root_value_mask"])
            assert np.isfinite(decision.row["root_value"])


# ---------------------------------------------------------------------------
# Forced (single-legal-action) decisions are not recorded as training rows.
# ---------------------------------------------------------------------------


def test_forced_decisions_are_recorded_with_zero_policy_weight_and_real_value_weight():
    _rust()
    # 2p initial placement is exactly 8 decisions (2 settlements + 2 roads per
    # player); the first ROLL (always forced/single-legal-action) follows, so
    # max_decisions must exceed 8 to guarantee at least one forced decision.
    self_play_config, search_config = _fast_config(max_decisions=14)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    mcts = GumbelChanceMCTS(search_config, evaluator)

    record = play_one_game(
        mcts,
        evaluator,
        config=self_play_config,
        game_seed=7,
        game_index=0,
        action_size=ActionCatalog(COLORS).size,
    )

    assert record.forced_decisions >= 1
    # Forced decisions are recorded as rows now (not skipped) -- every
    # decision (forced or not) gets exactly one row.
    assert len(record.decisions) == record.total_decisions

    forced_rows = [
        decision.row for decision in record.decisions if decision.row["is_forced"]
    ]
    assert len(forced_rows) == record.forced_decisions
    for row in forced_rows:
        assert float(row["policy_weight_multiplier"]) == pytest.approx(0.0)
        assert float(row["value_weight_multiplier"]) == pytest.approx(1.0)
        assert not bool(row["root_value_mask"])
        assert np.isnan(row["root_value"])


def test_trajectory_only_forced_rows_emit_without_search_or_policy_authority():
    _rust()
    self_play_config, _ = _fast_config(max_decisions=14)
    search_config = GumbelChanceMCTSConfig(
        seed=0,
        n_full=4,
        n_fast=2,
        p_full=1.0,
        max_depth=40,
        forced_root_target_mode="trajectory_only",
    )
    evaluator = HeuristicRustEvaluator(score_actions=False)
    record = play_one_game(
        GumbelChanceMCTS(search_config, evaluator),
        evaluator,
        config=self_play_config,
        game_seed=7,
        game_index=0,
        action_size=ActionCatalog(COLORS).size,
    )

    forced_rows = [
        decision.row for decision in record.decisions if decision.row["is_forced"]
    ]
    assert forced_rows
    for row in forced_rows:
        assert row["used_full_search"] is False
        assert int(row["simulations_used"]) == 0
        assert float(row["policy_weight_multiplier"]) == 0.0
        assert float(row["value_weight_multiplier"]) == 1.0
        assert not bool(row["root_value_mask"])
        assert np.isnan(row["root_value"])
        assert not np.any(row["afterstate_target_mask"])
        assert "_search_visit_counts" not in row
        assert "_search_completed_q" not in row


def test_explicit_legacy_opt_out_still_omits_forced_transition_rows():
    _rust()
    self_play_config, search_config = _fast_config(max_decisions=14)
    self_play_config = dataclasses.replace(
        self_play_config, record_automatic_transitions=False
    )
    evaluator = HeuristicRustEvaluator(score_actions=False)
    record = play_one_game(
        GumbelChanceMCTS(search_config, evaluator),
        evaluator,
        config=self_play_config,
        game_seed=7,
        game_index=0,
        action_size=ActionCatalog(COLORS).size,
    )

    assert record.total_decisions > len(record.decisions)
    assert record.forced_decisions == 0
    assert not any(decision.row["is_forced"] for decision in record.decisions)


def test_forced_roll_rows_carry_real_afterstate_targets():
    _rust()
    self_play_config, search_config = _fast_config(max_decisions=14)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    mcts = GumbelChanceMCTS(search_config, evaluator)

    record = play_one_game(
        mcts,
        evaluator,
        config=self_play_config,
        game_seed=7,
        game_index=0,
        action_size=ActionCatalog(COLORS).size,
    )

    roll_rows = [
        decision.row
        for decision in record.decisions
        if decision.row["is_forced"]
        and bool(np.any(decision.row["afterstate_target_mask"]))
    ]
    assert roll_rows, (
        "expected at least one forced ROLL row with a real afterstate target"
    )
    for row in roll_rows:
        assert row["used_full_search"] is True
        assert row["simulations_used"] == 0
        mask = row["afterstate_target_mask"]
        values = row["afterstate_target"][mask]
        assert values.size > 0
        assert np.all(np.isfinite(values))
        assert np.all((values >= -1.0) & (values <= 1.0))


# ---------------------------------------------------------------------------
# Shard writing + schema round-trip through train_bc.py's loader.
# ---------------------------------------------------------------------------


def test_run_worker_games_writes_valid_shards_that_round_trip_through_train_bc(
    tmp_path,
):
    _rust()
    self_play_config, search_config = _fast_config(max_decisions=6)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    summary = run_worker_games(
        out_dir=tmp_path / "out",
        games=2,
        game_index_start=0,
        base_seed=500,
        worker_seed=3,
        config=self_play_config,
        search_config=search_config,
        evaluator=evaluator,
    )

    assert summary["games_completed"] == 2
    assert summary["games_failed"] == 0
    assert summary["shards"], "expected at least one shard to be written"

    total_rows = 0
    for shard_path in summary["shards"]:
        path = Path(shard_path)
        assert path.exists()
        raw = _load_npz(path)
        normalized = _normalize_teacher_shard(raw, path)
        n = len(normalized["action_taken"])
        total_rows += n
        # Required schema fields train_bc.py's loader hard-requires.
        assert normalized["obs"].shape[0] == n
        assert normalized["legal_action_context"].shape[0] == n
        assert normalized["legal_action_context"].ndim == 3
        assert normalized["legal_action_ids"].shape[0] == n
        assert normalized["target_policy"].shape == normalized["legal_action_ids"].shape
        # F4: target_policy is fp32 in the shard, not fp16 -- fp16's ~1e-3
        # relative precision was silently flushing small-but-real completed
        # probabilities to zero (worst at 54-action placement), tripping
        # train_bc.py's soft-target coverage gate into a spurious one-hot
        # hard-CE fallback on a fully-covered soft target.
        assert normalized["target_policy"].dtype == np.float32
        assert (
            normalized["target_score_source"].tolist().count("gumbel_mcts_visit_q") <= n
        )
        # policy_weight_multiplier/value_weight_multiplier must survive the
        # round trip (they drive train_bc.py's loss weighting).
        assert "policy_weight_multiplier" in normalized
        assert "value_weight_multiplier" in normalized
        # Future self-play persists the search root value needed by the
        # already-wired value-target blend.  The column is scalar/optional so
        # legacy shards without it continue to normalize unchanged.
        assert "root_value" in raw.files
        assert "root_value_mask" in raw.files
        assert normalized["root_value"].dtype == np.float32
        assert normalized["root_value_mask"].dtype == np.bool_
        root_mask = normalized["root_value_mask"]
        expected_root_mask = (
            ~normalized["is_forced"].astype(np.bool_)
            & normalized["used_full_search"].astype(np.bool_)
            & np.isfinite(normalized["root_value"])
        )
        assert np.array_equal(root_mask, expected_root_mask)
        assert np.any(root_mask)
        assert np.all(np.isfinite(normalized["root_value"][root_mask]))
        assert np.all(np.isnan(normalized["root_value"][~root_mask]))
        assert np.all(np.abs(normalized["root_value"][root_mask]) <= 1.0)
        # CAT-100 production wiring: the raw shard AND train_bc's normalized
        # view retain every aux target. Numeric targets use NaN for an
        # unobserved horizon; categorical targets use -1 as ignore_index.
        assert set(AUX_TARGET_KEYS).issubset(raw.files)
        assert set(AUX_TARGET_KEYS).issubset(normalized)
        assert AUX_SUBGOAL_TARGET_VERSION_KEY in raw.files
        assert AUX_SUBGOAL_TARGET_VERSION_KEY in normalized
        assert normalized[AUX_SUBGOAL_TARGET_VERSION_KEY].dtype == np.uint8
        assert set(normalized[AUX_SUBGOAL_TARGET_VERSION_KEY].tolist()) == {
            AUX_SUBGOAL_TARGET_VERSION
        }
        for key in ("aux_longest_road", "aux_largest_army", "aux_vp_in_n"):
            assert normalized[key].dtype == np.float32
        for key, upper in (("aux_next_settlement", 54), ("aux_robber_target", 19)):
            assert normalized[key].dtype == np.int16
            assert np.all(
                (normalized[key] == -1)
                | ((normalized[key] >= 0) & (normalized[key] < upper))
            )

    assert total_rows == int(summary["rows"])


def test_shard_prior_policy_round_trips_and_supports_kl_to_improved_policy(tmp_path):
    """`prior_policy` (root priors, pre-search) is persisted parallel to
    `target_policy` (the post-search improved policy) so KL(improved_policy
    || prior) is computable directly from shards without re-running the
    evaluator -- this only checks the shard round trip, not the KL math
    itself."""
    _rust()
    self_play_config, search_config = _fast_config(max_decisions=6)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    summary = run_worker_games(
        out_dir=tmp_path / "out",
        games=2,
        game_index_start=0,
        base_seed=501,
        worker_seed=4,
        config=self_play_config,
        search_config=search_config,
        evaluator=evaluator,
    )

    assert summary["shards"], "expected at least one shard to be written"

    saw_any_row = False
    for shard_path in summary["shards"]:
        raw = _load_npz(Path(shard_path))
        assert "prior_policy" in raw
        assert raw["prior_policy"].shape == raw["target_policy"].shape
        legal_mask = raw["legal_action_ids"] != -1
        for row_index in range(raw["prior_policy"].shape[0]):
            row_mask = legal_mask[row_index]
            priors = raw["prior_policy"][row_index][row_mask].astype(np.float64)
            assert priors.size > 0
            assert np.all(priors >= 0.0)
            # Priors form a real probability distribution over legal
            # actions (this is what makes KL(improved_policy || prior)
            # well-defined) -- padded/illegal slots are excluded via
            # legal_mask, not folded into the sum.
            assert priors.sum() == pytest.approx(1.0, abs=1.0e-2)
            saw_any_row = True
    assert saw_any_row


def test_run_worker_games_writes_a_real_manifest_json(tmp_path):
    # Regression: the docstring promised a per-worker manifest.json that was
    # never actually written; `_merge_worker_summaries` (CLI) records worker
    # manifest paths that must genuinely exist.
    _rust()
    self_play_config, search_config = _fast_config(max_decisions=6)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    out_dir = tmp_path / "out"

    summary = run_worker_games(
        out_dir=out_dir,
        games=1,
        game_index_start=0,
        base_seed=42,
        worker_seed=7,
        config=self_play_config,
        search_config=search_config,
        evaluator=evaluator,
    )

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists(), "run_worker_games must write out_dir/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["games_completed"] == summary["games_completed"]
    assert manifest["rows"] == summary["rows"]
    assert manifest["shards"] == summary["shards"]
    assert (
        manifest["target_information_regime"] == TARGET_INFORMATION_REGIME_AUTHORITATIVE
    )
    assert manifest[AUX_SUBGOAL_TARGET_VERSION_KEY] == AUX_SUBGOAL_TARGET_VERSION
    assert manifest["aux_subgoal_target_semantic"] == AUX_SUBGOAL_TARGET_SEMANTIC
    for shard_path in manifest["shards"]:
        assert Path(shard_path).exists()
        with np.load(shard_path, allow_pickle=False) as shard:
            assert "target_information_regime" in shard.files
            assert set(shard["target_information_regime"].astype(str)) == {
                TARGET_INFORMATION_REGIME_AUTHORITATIVE
            }


def test_target_information_regime_requires_explicit_search_and_native_support():
    legacy = SimpleNamespace(
        information_set_search=False,
        belief_chance_spectra=True,
        public_observation=True,
    )
    assert (
        _target_information_regime_for_search(
            legacy, engine_supports_determinization=True
        )
        == TARGET_INFORMATION_REGIME_AUTHORITATIVE
    )

    requested = SimpleNamespace(
        information_set_search=True,
        determinization_particles=4,
    )
    with pytest.raises(RuntimeError, match="determinize_for_player"):
        _target_information_regime_for_search(
            requested, engine_supports_determinization=False
        )
    assert (
        _target_information_regime_for_search(
            requested, engine_supports_determinization=True
        )
        == TARGET_INFORMATION_REGIME_PUBLIC
    )

    coherent = SimpleNamespace(
        coherent_public_belief_search=True,
        information_set_search=False,
        belief_chance_spectra=False,
    )
    with pytest.raises(RuntimeError, match="apply_public_belief_development_draws"):
        _target_information_regime_for_search(
            coherent,
            engine_supports_determinization=True,
            engine_supports_public_belief_development_draws=False,
        )
    assert (
        _target_information_regime_for_search(
            coherent,
            engine_supports_determinization=True,
            engine_supports_public_belief_development_draws=True,
        )
        == TARGET_INFORMATION_REGIME_PUBLIC_COHERENT
    )

    with pytest.raises(ValueError, match="belief_chance_spectra"):
        _target_information_regime_for_search(
            SimpleNamespace(
                information_set_search=True,
                determinization_particles=4,
                belief_chance_spectra=True,
            ),
            engine_supports_determinization=True,
        )


def test_search_execution_contract_attests_effective_particle_budget_semantics():
    pimc = SimpleNamespace(information_set_search=True, exact_budget_sh=False)
    assert _search_execution_contract(pimc, native_mcts_hot_loop=True) == {
        "budget_scope": "total_before_determinization_division",
        "configured_exact_budget_sh": False,
        "information_set_particle_subbudgets_exact": True,
        "forced_root_target_mode": "full",
        "native_mcts_hot_loop": True,
    }

    single_world = SimpleNamespace(information_set_search=False, exact_budget_sh=False)
    assert _search_execution_contract(single_world, native_mcts_hot_loop=False) == {
        "budget_scope": "single_authoritative_world",
        "configured_exact_budget_sh": False,
        "information_set_particle_subbudgets_exact": False,
        "forced_root_target_mode": "full",
        "native_mcts_hot_loop": False,
    }

    coherent = SimpleNamespace(
        information_set_search=False,
        coherent_public_belief_search=True,
        exact_budget_sh=False,
        forced_root_target_mode="trajectory_only",
    )
    assert _search_execution_contract(coherent, native_mcts_hot_loop=True) == {
        "budget_scope": "single_public_belief_tree",
        "configured_exact_budget_sh": False,
        "information_set_particle_subbudgets_exact": False,
        "forced_root_target_mode": "trajectory_only",
        "native_mcts_hot_loop": True,
    }


def test_run_worker_games_isolates_one_bad_game_from_the_worker(tmp_path, monkeypatch):
    _rust()
    self_play_config, search_config = _fast_config(max_decisions=4)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    import catan_zero.rl.gumbel_self_play as gumbel_self_play

    real_play_one_game = gumbel_self_play.play_one_game
    call_count = {"n": 0}

    def _flaky_play_one_game(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("synthetic failure for isolation test")
        return real_play_one_game(*args, **kwargs)

    monkeypatch.setattr(gumbel_self_play, "play_one_game", _flaky_play_one_game)

    summary = run_worker_games(
        out_dir=tmp_path / "out",
        games=2,
        game_index_start=0,
        base_seed=900,
        worker_seed=1,
        config=self_play_config,
        search_config=search_config,
        evaluator=evaluator,
    )

    assert summary["games_failed"] == 1
    assert summary["games_completed"] == 1
    assert len(summary["errors"]) == 1
    assert "synthetic failure" in summary["errors"][0]["error"]


# ---------------------------------------------------------------------------
# CAT-54 opponent MIX: tag/md5 shard tagging + own-side-row filter, exercised
# through the same `run_worker_games` entry point the H2 binary pool uses.
# ---------------------------------------------------------------------------


def _mix_runtime_two_categories() -> MixRuntime:
    """A 2-category mix (roughly even split so a handful of games exercises
    both) that never touches a real checkpoint file -- `evaluator_factory`
    hands back a second (distinct) HeuristicRustEvaluator instance regardless
    of the path string, decoupling this test from real checkpoint loading."""
    categories = (
        MixCategory(name="producer_self_play", weight=1.0, source="self"),
        MixCategory(
            name="hard_experimental",
            weight=1.0,
            source="checkpoint_list",
            checkpoints=(
                MixCheckpointRef(
                    path="fake-hard-negative.pt", version=7, md5="deadbeef"
                ),
            ),
        ),
    )
    return MixRuntime(
        config=OpponentMixConfig(categories=categories),
        evaluator_factory=lambda _path: HeuristicRustEvaluator(score_actions=False),
    )


def test_opponent_mix_stamps_tag_and_md5_and_never_records_opponent_seat_rows(tmp_path):
    _rust()
    self_play_config, search_config = _fast_config(max_decisions=6)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    mix = _mix_runtime_two_categories()

    summary = run_worker_games(
        out_dir=tmp_path / "out",
        games=12,
        game_index_start=0,
        base_seed=7000,
        worker_seed=5,
        config=self_play_config,
        search_config=search_config,
        evaluator=evaluator,
        opponent_mix=mix,
    )

    assert summary["opponent_mix_enabled"] is True
    # `run_worker_games` (this test's entry point) reports raw per-tag counts
    # under `opponent_mix_per_tag_stats` -- the CLI-level "_tags_used" list is
    # only computed by `generate_gumbel_selfplay_data.py`'s
    # `_merge_worker_summaries`, mirroring the existing opponent_pool split
    # (per-worker "_per_version_stats" vs. CLI-merged "_versions_used").
    assert summary["opponent_mix_per_tag_stats"]
    # Both configured categories should show up over 12 games at a ~50/50
    # nominal split (small-N slack: just assert neither is entirely absent).
    assert set(summary["opponent_mix_per_tag_stats"]) <= {
        "producer_self_play",
        "hard_experimental",
    }

    saw_pool_row = False
    saw_mirror_row = False
    for shard_path in summary["shards"]:
        raw = _load_npz(Path(shard_path))
        assert "opponent_tag" in raw
        assert "opponent_checkpoint_md5" in raw
        assert "is_pool_game" in raw
        assert "opponent_version" in raw

        tags = raw["opponent_tag"]
        is_pool = raw["is_pool_game"]
        versions = raw["opponent_version"]
        md5s = raw["opponent_checkpoint_md5"]
        players = raw["player"]
        game_seeds = raw["game_seed"]

        for row_index in range(len(tags)):
            tag = str(tags[row_index])
            assert tag in ("producer_self_play", "hard_experimental")
            if tag == "hard_experimental":
                assert bool(is_pool[row_index]) is True
                assert int(versions[row_index]) == 7
                assert str(md5s[row_index]) == "deadbeef"
                saw_pool_row = True
            else:
                assert bool(is_pool[row_index]) is False
                saw_mirror_row = True

        # Own-side-row filter: within a single pool game (one game_seed, tag
        # "hard_experimental"), every RECORDED decision must belong to the
        # SAME acting color -- the champion never switches seats mid-game,
        # and the opponent seat's own decisions must never have been
        # recorded at all. A mirror game ("producer_self_play") has no such
        # constraint (both seats are legitimately recorded).
        by_seed: dict[int, set[str]] = {}
        for row_index in range(len(tags)):
            if str(tags[row_index]) != "hard_experimental":
                continue
            seed = int(game_seeds[row_index])
            by_seed.setdefault(seed, set()).add(str(players[row_index]))
        for seed, seen_players in by_seed.items():
            assert len(seen_players) == 1, (
                f"pool game_seed={seed} recorded decisions from {seen_players} -- "
                "the opponent seat's own decisions must never be recorded"
            )

    assert saw_pool_row, (
        "expected at least one hard_experimental (pool) row across 12 games"
    )
    assert saw_mirror_row, (
        "expected at least one producer_self_play (mirror) row across 12 games"
    )


def test_opponent_pool_and_opponent_mix_are_mutually_exclusive(tmp_path):
    """`run_worker_games` must refuse to run with both `opponent_pool` and
    `opponent_mix` set -- they both resolve into the exact same
    `PoolGameAssignment`, so running both would leave it ambiguous which
    policy governs a given game_index. This must fail before any game is
    played (cheap: no rust engine call needed to observe it), which is what
    makes it safe to assert here even though most of this file needs
    `catanatron_rs` -- but exercising it end-to-end through the real
    `run_worker_games` entry point (rather than only unit-testing the guard
    in isolation) is the point of keeping this test in this file.
    """
    from catan_zero.rl.flywheel import ChampionRef, OpponentPolicy
    from catan_zero.rl.gumbel_self_play import OpponentPoolRuntime

    self_play_config, search_config = _fast_config(max_decisions=4)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    pool = OpponentPoolRuntime(
        policy=OpponentPolicy(pool_fraction=0.2),
        champion=ChampionRef(version=1, path="", promoted_at=0.0),
        archive=(),
        evaluator_factory=lambda _path: evaluator,
    )
    mix = _mix_runtime_two_categories()

    with pytest.raises(ValueError, match="opponent_pool and opponent_mix"):
        run_worker_games(
            out_dir=tmp_path / "out",
            games=1,
            game_index_start=0,
            base_seed=1,
            worker_seed=1,
            config=self_play_config,
            search_config=search_config,
            evaluator=evaluator,
            opponent_pool=pool,
            opponent_mix=mix,
        )


# ---------------------------------------------------------------------------
# _apply_selected_action must use the same corrected chance spectra as the
# search (A19/A20 mitigation) -- this is the codepath that actually decides
# the recorded game trajectory, independent of the search's internal
# simulation logic tested in test_gumbel_chance_mcts.py.
# ---------------------------------------------------------------------------


def _find_move_robber_with_victim(catanatron_rs, *, seed: int, max_ticks: int = 300):
    game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
    for _ in range(max_ticks):
        playable = json.loads(game.playable_actions_json())
        candidates = [
            action
            for action in playable
            if action[1] == "MOVE_ROBBER" and action[2][1] is not None
        ]
        if candidates:
            ids = [int(a) for a in game.playable_action_indices(list(COLORS), None)]
            action_index = ids[playable.index(candidates[0])]
            return game, action_index, candidates[0]
        game.play_tick()
    raise AssertionError(
        "did not reach a MOVE_ROBBER-with-victim decision within budget"
    )


def test_apply_selected_action_steals_only_resources_the_victim_actually_holds():
    catanatron_rs = _rust()
    game, action_index, action_json = _find_move_robber_with_victim(
        catanatron_rs, seed=0
    )

    victim_name = action_json[2][1]
    snapshot = json.loads(game.json_snapshot())
    colors = [str(color) for color in snapshot["colors"]]
    victim_hand = snapshot["player_state"][colors.index(victim_name)]["resources"]
    held = {resource for resource in RESOURCES if victim_hand[resource] > 0}
    assert held and held != set(RESOURCES)

    rng = random.Random(123)
    for _ in range(50):
        result_game = _apply_selected_action(
            game.copy(),
            action_index,
            colors=COLORS,
            rng=rng,
            correct_rust_chance_spectra=True,
        )
        after = json.loads(result_game.json_snapshot())
        after_hand = after["player_state"][colors.index(victim_name)]["resources"]
        delta = {r: int(victim_hand[r]) - int(after_hand[r]) for r in RESOURCES}
        stolen = [r for r, d in delta.items() if d == 1]
        assert len(stolen) == 1, (
            f"expected exactly one resource stolen, got delta={delta}"
        )
        assert stolen[0] in held, f"stole {stolen[0]!r} which the victim did not hold"


def _find_buy_development_card_with_phantom_outcome(
    catanatron_rs, *, seed: int, max_ticks: int = 4000
):
    game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
    for _ in range(max_ticks):
        if game.winning_color() is not None:
            break
        playable = json.loads(game.playable_actions_json())
        candidates = [
            action for action in playable if action[1] == "BUY_DEVELOPMENT_CARD"
        ]
        if candidates:
            action_json = candidates[0]
            ids = [int(a) for a in game.playable_action_indices(list(COLORS), None)]
            action_index = ids[playable.index(action_json)]
            spectrum = json.loads(game.spectrum_json(json.dumps(action_json)))
            snapshot = json.loads(game.json_snapshot())
            colors = [str(color) for color in snapshot["colors"]]
            actor_index = colors.index(str(action_json[0]))
            before_cards = snapshot["player_state"][actor_index]["dev_cards"]
            before_deck = int(snapshot["development_deck_count"])
            has_phantom = False
            for outcome_index, _entry in enumerate(spectrum):
                candidate_game = game.apply_chance_outcome(
                    json.dumps(action_json), outcome_index
                )
                candidate_snapshot = json.loads(candidate_game.json_snapshot())
                candidate_cards = candidate_snapshot["player_state"][actor_index][
                    "dev_cards"
                ]
                gained = [
                    card
                    for card in DEVELOPMENT_CARDS
                    if int(candidate_cards.get(card, 0))
                    - int(before_cards.get(card, 0))
                    == 1
                ]
                deck_decreased = (
                    int(candidate_snapshot["development_deck_count"]) == before_deck - 1
                )
                if not (len(gained) == 1 and deck_decreased):
                    has_phantom = True
            if has_phantom:
                return game, action_index, action_json
        game.play_tick()
    raise AssertionError(
        "did not reach a BUY_DEVELOPMENT_CARD decision with a phantom outcome within budget"
    )


def test_apply_selected_action_dev_card_draw_always_gains_exactly_one_real_card():
    # Property test: unconditional, regardless of whether the installed wheel
    # still has phantom BUY_DEVELOPMENT_CARD outcomes to correct (if it
    # doesn't, per bug A20's fix, the correction is a no-op that still holds).
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(list(COLORS), seed=42)
    for _ in range(2000):
        playable = json.loads(game.playable_actions_json())
        candidates = [a for a in playable if a[1] == "BUY_DEVELOPMENT_CARD"]
        if candidates:
            action_json = candidates[0]
            ids = [int(a) for a in game.playable_action_indices(list(COLORS), None)]
            action_index = ids[playable.index(action_json)]
            break
        game.play_tick()
    else:
        raise AssertionError(
            "did not reach a BUY_DEVELOPMENT_CARD decision within budget"
        )

    snapshot = json.loads(game.json_snapshot())
    colors = [str(color) for color in snapshot["colors"]]
    actor_index = colors.index(str(action_json[0]))
    before_cards = snapshot["player_state"][actor_index]["dev_cards"]
    before_deck = int(snapshot["development_deck_count"])

    rng = random.Random(7)
    for _ in range(50):
        result_game = _apply_selected_action(
            game.copy(),
            action_index,
            colors=COLORS,
            rng=rng,
            correct_rust_chance_spectra=True,
        )
        after = json.loads(result_game.json_snapshot())
        after_cards = after["player_state"][actor_index]["dev_cards"]
        gained = [
            card
            for card in DEVELOPMENT_CARDS
            if int(after_cards.get(card, 0)) - int(before_cards.get(card, 0)) == 1
        ]
        assert len(gained) == 1, (
            "every applied outcome must actually draw exactly one card"
        )
        assert int(after["development_deck_count"]) == before_deck - 1


@pytest.mark.skipif(
    _A19_A20_FIXED_ON_WHEEL,
    reason="A20 (BUY_DEVELOPMENT_CARD phantom outcome) was fixed in catanatron_rs "
    "0.1.1; this fixture is stale by design on newer wheels.",
)
def test_buy_development_card_phantom_bug_is_present_before_0_1_1():
    catanatron_rs = _rust()
    _game, _action_index, _action_json = (
        _find_buy_development_card_with_phantom_outcome(catanatron_rs, seed=6)
    )


# ---------------------------------------------------------------------------
# CLI worker-level error isolation: `_worker_entry` must never raise (a raised
# exception there aborts `pool.map` for ALL workers, discarding every OTHER
# worker's already-written shards/results before the top-level manifest is
# ever written).
# ---------------------------------------------------------------------------


def test_worker_entry_never_raises_and_reports_a_failed_worker(monkeypatch):
    from tools.generate_gumbel_selfplay_data import _worker_entry
    import tools.generate_gumbel_selfplay_data as cli_module

    def _boom(_worker_args):
        raise RuntimeError("synthetic checkpoint load failure")

    monkeypatch.setattr(cli_module, "_run_worker", _boom)

    result = _worker_entry({"worker_index": 3, "out_dir": "/nonexistent", "games": 2})

    assert result["worker_index"] == 3
    assert result["games_completed"] == 0
    assert result["games_failed"] == 2
    assert result["shards"] == []
    assert result["errors"]
    assert "synthetic checkpoint load failure" in result["errors"][0]["error"]
