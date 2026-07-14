from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.flywheel import ShardMeta, WindowedReplay, ensure_dirs
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
    assert value("--per-game-policy-weight-mode") == "equal"
    assert "--no-per-game-value-weight" in argv
    assert value("--per-game-value-weight-mode") == "equal"
    assert value("--loser-sample-weight") == "1.0"
    assert value("--policy-kl-anchor-weight") == "0.0"
    assert value("--policy-kl-anchor-direction") == "forward"
    assert value("--value-target-lambda") == "1.0"
    assert value("--q-loss-weight") == "0.0"


def test_v2_config_refuses_omitted_production_learner_fields() -> None:
    payload = FlywheelConfig().to_dict()
    del payload["learner_per_game_value_weight"]
    with pytest.raises(ValueError, match="missing production learner recipe"):
        FlywheelConfig.from_dict(payload)


def test_production_refuses_duplicate_value_game_length_correction() -> None:
    with pytest.raises(ValueError, match="duplicate per-game value weighting"):
        FlywheelConfig(learner_per_game_value_weight=True).validate()


def test_production_refuses_replay_ratio_drift_from_exact_twenty_percent() -> None:
    with pytest.raises(ValueError, match="exact historical replay ratio 0.20"):
        FlywheelConfig(learner_min_replay_ratio=0.25).validate()


def test_production_next_training_refuses_single_component_without_anchor(tmp_path) -> None:
    runner = Runner(
        FlywheelConfig().validate(),
        tmp_path,
        dry_run=True,
        workers=1,
        device="cpu",
        base_seed=1,
    )
    result = runner.train_window(
        ["current-only"],
        "champion.pt",
        0,
        100,
        current_ckpt_version=3,
    )
    assert result["ok"] is False
    assert "requires ShardMeta provenance" in result["note"]


def test_production_composite_uses_replay_contract_without_a1_sentinel() -> None:
    args = SimpleNamespace(a1_contract_sha256="")
    train_bc._bind_composite_validation_provenance(  # noqa: SLF001
        args,
        object(),
        validation_seed_contract=None,
        composite_meta={
            "schema_version": "memmap_composite_v2",
            "diagnostic_only": False,
            "promotion_eligible": True,
            "flywheel_replay_contract": {"schema_version": "flywheel-replay-composite-v2"},
        },
        ddp={"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
    )
    assert args.a1_contract_sha256 == ""


def test_authenticated_diagnostic_derivative_inherits_production_split() -> None:
    args = SimpleNamespace(a1_contract_sha256="")
    train_bc._bind_composite_validation_provenance(  # noqa: SLF001
        args,
        object(),
        validation_seed_contract=None,
        composite_meta={
            "schema_version": "memmap_composite_v2",
            "diagnostic_only": True,
            "promotion_eligible": False,
            "flywheel_diagnostic_derivative": True,
            "diagnostic_derivation_authority": {
                "schema_version": "flywheel-diagnostic-descriptor-derivation-v1"
            },
            "flywheel_replay_contract": {
                "schema_version": "flywheel-replay-composite-v2"
            },
        },
        ddp={"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
    )
    assert args.a1_contract_sha256 == ""


def test_composite_without_any_validation_authority_fails_closed() -> None:
    with pytest.raises(SystemExit, match="must carry the exact"):
        train_bc._bind_composite_validation_provenance(  # noqa: SLF001
            SimpleNamespace(a1_contract_sha256=""),
            object(),
            validation_seed_contract=None,
            composite_meta={
                "schema_version": "memmap_composite_v2",
                "diagnostic_only": True,
                "promotion_eligible": False,
                "flywheel_replay_contract": None,
            },
            ddp={"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
        )


def test_non_dry_train_window_delegates_canonical_composite_construction(
    tmp_path, monkeypatch
) -> None:
    loop_dir = tmp_path / "loop"
    ensure_dirs(loop_dir)
    current_checkpoint = tmp_path / "current.pt"
    replay_checkpoint = tmp_path / "replay.pt"
    current_checkpoint.write_bytes(b"current checkpoint")
    replay_checkpoint.write_bytes(b"replay checkpoint")

    def shard(path, base_seed: int) -> None:
        np.savez(
            path,
            obs=np.zeros((4, 3), dtype=np.float16),
            legal_action_ids=np.asarray(
                [[1, 2], [1, 2], [1, 2], [1, 2]], dtype=np.int16
            ),
            legal_action_context=np.zeros((4, 2, 1), dtype=np.float16),
            action_taken=np.asarray([1, 2, 1, 2], dtype=np.int16),
            game_seed=np.asarray(
                [base_seed, base_seed, base_seed + 1, base_seed + 1],
                dtype=np.int64,
            ),
            decision_index=np.asarray([0, 1, 0, 1], dtype=np.int32),
            policy_weight_multiplier=np.asarray(
                [1.0, 0.0, 1.0, 0.0], dtype=np.float32
            ),
        )

    current_shard = tmp_path / "current.npz"
    recent_shard = tmp_path / "recent.npz"
    hard_shard = tmp_path / "hard.npz"
    replay_shard = tmp_path / "replay.npz"
    shard(current_shard, 100)
    shard(recent_shard, 200)
    shard(hard_shard, 300)
    shard(replay_shard, 400)
    from tools import continuous_flywheel as flywheel

    window = [
        ShardMeta(
            path=str(current_shard.resolve()),
            rows=4,
            order=2,
            ckpt_version=5,
            producer_checkpoint_path=str(current_checkpoint.resolve()),
            producer_checkpoint_sha256=flywheel._file_sha256(current_checkpoint),
            source_id="round-5",
            source_category="current_producer",
        ),
        ShardMeta(
            path=str(recent_shard.resolve()),
            rows=4,
            order=3,
            ckpt_version=5,
            producer_checkpoint_path=str(current_checkpoint.resolve()),
            producer_checkpoint_sha256=flywheel._file_sha256(current_checkpoint),
            source_id="round-5-recent",
            source_category="recent_history",
        ),
        ShardMeta(
            path=str(hard_shard.resolve()),
            rows=4,
            order=4,
            ckpt_version=5,
            producer_checkpoint_path=str(current_checkpoint.resolve()),
            producer_checkpoint_sha256=flywheel._file_sha256(current_checkpoint),
            source_id="round-5-hard",
            source_category="hard_negative",
        ),
        ShardMeta(
            path=str(replay_shard.resolve()),
            rows=4,
            order=1,
            ckpt_version=4,
            producer_checkpoint_path=str(replay_checkpoint.resolve()),
            producer_checkpoint_sha256=flywheel._file_sha256(replay_checkpoint),
            source_id="round-4",
            source_category="current_producer",
        ),
    ]
    commands: list[list[str]] = []

    def run(command: list[str], _log: Path) -> int:
        commands.append(command)
        if command[1] == "tools/build_memmap_corpus.py":
            source = Path(command[command.index("--source") + 1])
            output = Path(command[command.index("--out") + 1])
            build_memmap_corpus(source, output, progress_every=0)
            return 0
        if command[1] == "tools/train_bc.py":
            descriptor = Path(command[command.index("--data") + 1])
            verified = train_bc._preflight_memmap_composite_descriptor(descriptor)
            assert verified["diagnostic_only"] is False
            assert verified["promotion_eligible"] is True
            assert verified["component_ids"] == [
                "current_producer",
                "recent_history",
                "hard_negative",
                "historical_replay",
            ]
            assert verified["component_game_sampling_ratios"] == pytest.approx(
                [0.64, 0.12, 0.04, 0.2]
            )
            assert verified["flywheel_replay_contract"][
                "checkpoint_versions"
            ] == [4, 5]
            receipt = verified["production_mix_contract"]["sampling_receipt"]
            assert receipt["sampler_order"] == ["source", "game", "row"]
            assert receipt["aggregate"]["game_count"] == 8
            assert receipt["aggregate"]["row_count"] == 16
            assert receipt["aggregate"]["policy_active_row_count"] == 8
            checkpoint_path = Path(command[command.index("--checkpoint") + 1])
            report_path = Path(command[command.index("--report") + 1])
            initializer_path = Path(command[command.index("--init-checkpoint") + 1])
            steps = int(command[command.index("--max-steps") + 1])
            checkpoint_path.write_bytes(b"candidate")
            Path(str(checkpoint_path) + ".optimizer.pt").write_bytes(b"optimizer")
            report_path.write_text(
                json.dumps(
                    {
                        "diagnostic_only": False,
                        "promotion_eligible": True,
                        "data": str(descriptor.resolve()),
                        "data_fingerprint": verified["descriptor_fingerprint"],
                        "data_format": "memmap",
                        "checkpoint": str(checkpoint_path),
                        "init_checkpoint": str(initializer_path),
                        "init_checkpoint_sha256": flywheel._file_sha256(
                            initializer_path
                        ),
                        "max_steps": steps,
                        "steps_completed": steps,
                        "total_training_steps": steps,
                        "resume_optimizer": False,
                        "optimizer_restored": False,
                        "mask_hidden_info": True,
                        "training_rng_rank_offset": True,
                        "validation_max_samples": 0,
                        "metrics": [
                            {
                                "validation": {
                                    "schema_version": "raw-row-compatibility-metric",
                                    "objective_matched": False,
                                },
                                "validation_objective_matched": {
                                    "schema_version": "composite-validation-measure-v2",
                                    "objective_matched": True,
                                    "component_sampling_ratios": verified[
                                        "flywheel_replay_contract"
                                    ]["effective_component_sampling_ratios"],
                                    "components": {
                                        component_id: {
                                            "authenticated_sampling_ratio": ratio
                                        }
                                        for component_id, ratio in verified[
                                            "flywheel_replay_contract"
                                        ][
                                            "effective_component_sampling_ratios"
                                        ].items()
                                    },
                                }
                            }
                        ],
                        "memmap_composite": {
                            "descriptor_path": str(descriptor.resolve()),
                            "descriptor_file_sha256": verified[
                                "descriptor_file_sha256"
                            ],
                            "descriptor_fingerprint": verified[
                                "descriptor_fingerprint"
                            ],
                            "flywheel_replay_contract": verified[
                                "flywheel_replay_contract"
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            return 0
        raise AssertionError(command)

    monkeypatch.setattr(flywheel, "_run", run)
    runner = Runner(
        FlywheelConfig(train_batch_size=4).validate(),
        loop_dir,
        dry_run=False,
        workers=1,
        device="cpu",
        base_seed=1,
    )
    result = runner.train_window(
        window,
        str(current_checkpoint.resolve()),
        6,
        4,
        current_ckpt_version=5,
    )

    assert result["ok"] is False
    assert "canonical authenticated composite required" in result["note"]
    assert all(command[1] != "tools/train_bc.py" for command in commands)


def test_replay_guard_rejects_two_shards_from_same_generation(tmp_path) -> None:
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_bytes(b"checkpoint")
    shard_a = tmp_path / "a.npz"
    shard_b = tmp_path / "b.npz"
    shard_a.write_bytes(b"a")
    shard_b.write_bytes(b"b")
    from tools import continuous_flywheel as flywheel

    digest = flywheel._file_sha256(checkpoint)
    shards = [
        ShardMeta(
            path=str(path.resolve()),
            rows=1,
            order=index,
            ckpt_version=7,
            producer_checkpoint_path=str(checkpoint.resolve()),
            producer_checkpoint_sha256=digest,
            source_id=f"round-{index}",
            source_category="current_producer",
        )
        for index, path in enumerate((shard_a, shard_b))
    ]
    runner = Runner(
        FlywheelConfig().validate(),
        tmp_path / "loop",
        dry_run=False,
        workers=1,
        device="cpu",
        base_seed=1,
    )

    result = runner.train_window(
        shards,
        str(checkpoint),
        8,
        2,
        current_ckpt_version=7,
    )

    assert result["ok"] is False
    assert "distinct authenticated historical" in result["note"]


def test_replay_guard_refuses_flattened_fresh_source_wave(tmp_path) -> None:
    checkpoint = tmp_path / "current.pt"
    replay_checkpoint = tmp_path / "replay.pt"
    checkpoint.write_bytes(b"current checkpoint")
    replay_checkpoint.write_bytes(b"replay checkpoint")
    current = tmp_path / "current.npz"
    replay = tmp_path / "replay.npz"
    current.write_bytes(b"current")
    replay.write_bytes(b"replay")
    from tools import continuous_flywheel as flywheel

    shards = [
        ShardMeta(
            path=str(current.resolve()),
            rows=1,
            order=2,
            ckpt_version=5,
            producer_checkpoint_path=str(checkpoint.resolve()),
            producer_checkpoint_sha256=flywheel._file_sha256(checkpoint),
            source_id="wave-5",
            source_category="current_producer",
        ),
        ShardMeta(
            path=str(replay.resolve()),
            rows=1,
            order=1,
            ckpt_version=4,
            producer_checkpoint_path=str(replay_checkpoint.resolve()),
            producer_checkpoint_sha256=flywheel._file_sha256(replay_checkpoint),
            source_id="wave-4",
            source_category="current_producer",
        ),
    ]
    runner = Runner(
        FlywheelConfig().validate(),
        tmp_path / "loop",
        dry_run=False,
        workers=1,
        device="cpu",
        base_seed=1,
    )
    result = runner.train_window(
        shards,
        str(checkpoint.resolve()),
        6,
        1,
        current_ckpt_version=5,
    )
    assert result["ok"] is False
    assert "must preserve separate current/recent/hard sources" in result["note"]
    assert "recent_history" in result["note"]
    assert "hard_negative" in result["note"]


def test_replay_guard_rejects_missing_producer_identity_before_build(tmp_path) -> None:
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_bytes(b"checkpoint")
    current = tmp_path / "current.npz"
    recent = tmp_path / "recent.npz"
    hard = tmp_path / "hard.npz"
    replay = tmp_path / "replay.npz"
    current.write_bytes(b"current")
    recent.write_bytes(b"recent")
    hard.write_bytes(b"hard")
    replay.write_bytes(b"replay")
    shards = [
        ShardMeta(
            str(current.resolve()), 1, 2, ckpt_version=2, source_id="r2",
            source_category="current_producer",
        ),
        ShardMeta(
            str(recent.resolve()), 1, 3, ckpt_version=2, source_id="r2-recent",
            source_category="recent_history",
        ),
        ShardMeta(
            str(hard.resolve()), 1, 4, ckpt_version=2, source_id="r2-hard",
            source_category="hard_negative",
        ),
        ShardMeta(
            str(replay.resolve()), 1, 1, ckpt_version=1, source_id="r1",
            source_category="current_producer",
        ),
    ]
    runner = Runner(
        FlywheelConfig().validate(),
        tmp_path / "loop",
        dry_run=False,
        workers=1,
        device="cpu",
        base_seed=1,
    )

    result = runner.train_window(
        shards,
        str(checkpoint),
        3,
        2,
        current_ckpt_version=2,
    )

    assert result["ok"] is False
    assert "lacks producer checkpoint provenance" in result["note"]


def test_replay_registry_restart_preserves_source_and_checkpoint_provenance(
    tmp_path,
) -> None:
    state = tmp_path / "window.json"
    window = WindowedReplay(state, c=10)
    original = window.register(
        "/data/current.npz",
        12,
        ckpt_version=9,
        producer_checkpoint_path="/checkpoints/champion_v9.pt",
        producer_checkpoint_sha256="sha256:" + "a" * 64,
        source_id="a1-wave-9-current",
        source_category="current_producer",
    )
    window.save()

    resumed = WindowedReplay(state, c=10)
    restored = resumed.select().in_window[0]
    assert restored == original
    # Blank values on idempotent rediscovery may not erase the durable binding.
    rediscovered = resumed.register("/data/current.npz", 12, ckpt_version=9)
    assert rediscovered.producer_checkpoint_path == original.producer_checkpoint_path
    assert rediscovered.producer_checkpoint_sha256 == original.producer_checkpoint_sha256
    assert rediscovered.source_id == original.source_id
    assert rediscovered.source_category == original.source_category
    with pytest.raises(ValueError, match="refusing to reattribute"):
        resumed.register("/data/current.npz", 12, ckpt_version=10)
    with pytest.raises(ValueError, match="change source category"):
        resumed.register(
            "/data/current.npz",
            12,
            ckpt_version=9,
            source_category="hard_negative",
        )


def test_replay_guard_rejects_future_generation_before_corpus_build(tmp_path) -> None:
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_bytes(b"checkpoint")
    future = tmp_path / "future.npz"
    future.write_bytes(b"future")
    runner = Runner(
        FlywheelConfig().validate(),
        tmp_path / "loop",
        dry_run=False,
        workers=1,
        device="cpu",
        base_seed=1,
    )
    result = runner.train_window(
        [
            ShardMeta(
                path=str(future.resolve()),
                rows=1,
                order=1,
                ckpt_version=10,
                producer_checkpoint_path=str(checkpoint.resolve()),
                producer_checkpoint_sha256="sha256:" + "b" * 64,
                source_id="future-wave",
                source_category="current_producer",
            )
        ],
        str(checkpoint),
        1,
        1,
        current_ckpt_version=9,
    )
    assert result == {"ok": False, "note": "window contains future checkpoint data"}


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
