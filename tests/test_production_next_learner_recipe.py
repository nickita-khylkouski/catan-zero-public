from __future__ import annotations

import numpy as np
import pytest

from catan_zero.rl.flywheel.config import FlywheelConfig
from tools.build_memmap_corpus import build_memmap_corpus
from tools import train_bc
from tools.continuous_flywheel import Runner


def test_forced_policy_rows_have_exactly_zero_effective_mass() -> None:
    data = {
        "action_taken": np.asarray([1, 2, 3, 4], dtype=np.int16),
        "legal_action_ids": np.asarray(
            [[1, -1], [2, 9], [3, -1], [4, 8]], dtype=np.int16
        ),
        # This persisted multiplier is the authoritative full-search policy
        # coverage boundary.  Forced rows remain zero even before the new
        # launcher redundantly pins --forced-action-weight=0.
        "policy_weight_multiplier": np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    }
    weights = train_bc.build_sample_weights(
        data,
        teacher_weights={},
        phase_weights={},
        forced_action_weight=0.1,
        winner_sample_weight=1.0,
        loser_sample_weight=1.0,
        vp_margin_weight=0.0,
        vps_to_win=10,
    )
    assert weights.tolist() == pytest.approx([0.0, 2.0, 0.0, 2.0])
    assert float(weights[[0, 2]].sum()) == 0.0
    assert float(weights.sum()) == pytest.approx(4.0)


def test_production_next_recipe_is_explicit_and_keeps_unproven_bootstraps_off() -> None:
    cfg = FlywheelConfig().validate()
    argv = cfg.resolve_learner_argv()

    def value(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert value("--forced-action-weight") == "0.0"
    assert value("--forced-row-value-weight") == "1.0"
    assert "--per-game-policy-weight" in argv
    assert value("--per-game-policy-weight-mode") == "sqrt"
    assert "--per-game-value-weight" in argv
    assert value("--per-game-value-weight-mode") == "sqrt"
    assert value("--loser-sample-weight") == "1.0"
    assert value("--policy-kl-anchor-weight") == "0.0"
    assert value("--value-target-lambda") == "1.0"
    assert value("--q-loss-weight") == "0.0"


def test_v2_config_refuses_omitted_production_learner_fields() -> None:
    payload = FlywheelConfig().to_dict()
    del payload["learner_per_game_value_weight"]
    with pytest.raises(ValueError, match="missing production learner recipe"):
        FlywheelConfig.from_dict(payload)


def test_production_next_training_refuses_single_component_without_anchor(tmp_path) -> None:
    runner = Runner(
        FlywheelConfig().validate(),
        tmp_path,
        dry_run=True,
        workers=1,
        device="cpu",
        base_seed=1,
    )
    result = runner.train_window(["current-only"], "champion.pt", 0, 100)
    assert result["ok"] is False
    assert "requires at least two replay components" in result["note"]


def test_search_afterstate_and_budget_survive_npz_and_memmap_loaders(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    shard = source / "shard.npz"
    np.savez(
        shard,
        obs=np.zeros((2, 3), dtype=np.float16),
        legal_action_ids=np.asarray([[1, -1], [2, 3]], dtype=np.int16),
        legal_action_context=np.zeros((2, 2, 1), dtype=np.float16),
        action_taken=np.asarray([1, 2], dtype=np.int16),
        afterstate_target=np.asarray([[0.25, np.nan], [0.5, -0.5]], dtype=np.float32),
        afterstate_target_mask=np.asarray([[True, False], [True, True]], dtype=np.bool_),
        simulations_used=np.asarray([0, 128], dtype=np.int32),
    )
    (source / "manifest.json").write_text(
        '{"shards":["' + str(shard) + '"]}', encoding="utf-8"
    )

    loaded = train_bc.load_teacher_data(source)
    np.testing.assert_array_equal(
        loaded["afterstate_target_mask"],
        np.asarray([[True, False], [True, True]], dtype=np.bool_),
    )
    np.testing.assert_array_equal(loaded["simulations_used"], [0, 128])

    corpus_path = tmp_path / "corpus"
    build_memmap_corpus(source, corpus_path, progress_every=0)
    corpus = train_bc.MemmapCorpus(corpus_path)
    np.testing.assert_array_equal(
        np.asarray(corpus["afterstate_target_mask"]),
        np.asarray([[True, False], [True, True]], dtype=np.bool_),
    )
    np.testing.assert_array_equal(np.asarray(corpus["simulations_used"]), [0, 128])
