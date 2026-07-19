from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from catan_zero.rl.production_recipe_catalog import production_recipes
from catan_zero.rl.pipeline_configs import CONFIG_SCHEMA_VERSION
from tools import train, train_bc


REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "configs/training/a1_current_35m_b200.schema1.json"
PARENT_RECIPE = REPO / "configs/training/a1_parent_update_35m_b200.schema1.json"
HARD_DECISION_EVIDENCE = (
    REPO / "docs/evidence/A1_V7_HARD_DECISION_POLICY_MASS_CORRECTION_20260717.json"
)
HARD_DECISION_POLICY_PHASE_WEIGHTS = (
    "PLAY_TURN=4.0,MOVE_ROBBER=3.0,BUILD_INITIAL_ROAD=2.0,DISCARD=1.5"
)


def _payload() -> dict[str, object]:
    return json.loads(RECIPE.read_text(encoding="utf-8"))


def _fields() -> dict[str, object]:
    return _payload()["train_config"]["fields"]


def _public_args(*, init_checkpoint: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        data="/tmp/data",
        checkpoint="/tmp/candidate.pt",
        report="/tmp/train.json",
        init_checkpoint=init_checkpoint,
        parent_checkpoint="",
        information_contract_migration_receipt="",
        device="auto",
        host_lock_file="/tmp/catan-test.lock",
        allow_concurrent_bc=False,
    )


def test_canonical_coverage_recipe_can_reach_composite_training() -> None:
    """Keep the public recipe compatible with train_bc's composite admission.

    Authenticated composites require a complete whole-game validation split,
    represented by the zero row-cap sentinel.  A nonzero cap is rejected before
    the first optimizer step.
    """

    fields = _fields()
    assert _payload()["engine_settings"]["base_sampler"] == "coverage_importance_v1"
    assert (
        "minimum_policy_effective_rows_per_global_batch"
        not in _payload()["engine_settings"]
    )
    assert fields["minimum_policy_effective_rows_per_global_batch"] == 32.0
    assert fields["data_format"] == "memmap"
    assert fields["validation_max_samples"] == 0


def test_production_hard_decision_mass_contract_is_commissioned() -> None:
    engine = _payload()["engine_settings"]

    assert train._require_production_hard_decision_policy_mass_contract(engine) == {
        "minimum_initial_settlement_policy_mass_fraction": 0.02,
        "minimum_initial_road_policy_mass_fraction": 0.02,
        "minimum_discard_policy_mass_fraction": 0.02,
        "minimum_move_robber_policy_mass_fraction": 0.02,
    }


def test_parent_production_recipe_can_collect_two_signal_observations() -> None:
    config, engine = train._load_recipe(PARENT_RECIPE)  # noqa: SLF001

    train._require_exact_cap_feature_observability(config, engine)  # noqa: SLF001

    assert config.max_steps == 12
    assert engine["train_diagnostics_every_batches"] == 6
    assert engine["minimum_feature_learning_signal_observations"] == 2


def test_engine_projection_defaults_optional_value_only_child_receipt() -> None:
    """Canonical recipe replay omits one-off child-receipt CLI fields.

    train_bc replays the checked-in parent recipe with this minimal namespace
    while checking immutable authority.  An optional experimental receipt must
    therefore behave like its parser default rather than abort every DDP rank.
    """

    config, engine = train._load_recipe(PARENT_RECIPE)  # noqa: SLF001
    resolved = train._engine_namespace(  # noqa: SLF001
        config=config,
        engine_settings=engine,
        public_args=_public_args(init_checkpoint="/tmp/parent.pt"),
    )
    assert resolved.a1_value_only_child_receipt == ""


def test_completed_q_opt_in_binds_reliability_inputs() -> None:
    config, engine = train._load_recipe(PARENT_RECIPE)  # noqa: SLF001
    baseline_args = train._engine_namespace(  # noqa: SLF001
        config=config,
        engine_settings=engine,
        public_args=_public_args(init_checkpoint="/tmp/parent.pt"),
    )
    baseline = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        baseline_args,
        {"enabled": True, "world_size": 8, "rank": 0, "local_rank": 0},
    )
    assert "completed_q_loss_weight" not in baseline

    opt_in_args = train._engine_namespace(  # noqa: SLF001
        config=dataclasses.replace(
            config,
            completed_q_loss_weight=0.25,
            target_reliability_confidence_floor=0.4,
            target_reliability_confidence_weighting=False,
        ),
        engine_settings=engine,
        public_args=_public_args(init_checkpoint="/tmp/parent.pt"),
    )
    effective = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        opt_in_args,
        {"enabled": True, "world_size": 8, "rank": 0, "local_rank": 0},
    )

    assert effective["completed_q_loss_weight"] == pytest.approx(0.25)
    assert effective["target_reliability_confidence_floor"] == pytest.approx(0.4)
    assert effective["target_reliability_confidence_weighting"] is False


def test_parent_report_retains_canonical_launch_authority(tmp_path) -> None:
    report = tmp_path / "train.report.json"
    report.write_text(
        json.dumps(
            {
                "steps_completed": 12,
                "training_row_draws": 6_144,
                "training_row_draws_semantics": (
                    "base_sampler_draw_events; may repeat rows; excludes_policy_aux"
                ),
                "base_training_row_draws": 6_144,
                "policy_aux_training_row_draws": 0,
                "policy_base_active_training_row_draws": 1_399,
                "policy_active_training_row_draws": 1_399,
                "value_active_training_row_draws": 6_144,
                "total_training_row_draws": 6_144,
                "policy_base_active_rows": 1_399,
                "policy_aux_active_rows": 0,
                "policy_total_active_rows": 1_399,
                "value_active_rows": 6_144,
                "policy_kl_anchor_eligible_rows": 0,
                # train_bc cannot bind this for an ordinary diagnostic corpus,
                # but the canonical wrapper still owns the recipe identity.
                "a1_canonical_parent_update_authority": None,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_sha = "sha256:" + ("a" * 64)
    initialization = {
        "schema_version": "a1-canonical-parent-initializer-v1",
        "mode": "exact_parent",
        "parent": {"sha256": checkpoint_sha},
        "initializer": {"sha256": checkpoint_sha},
        "information_contract_migration": None,
    }
    authority = {
        "schema_version": "a1-canonical-parent-update-runtime-authority-v1",
        "config": str(PARENT_RECIPE),
        "config_file_sha256": "sha256:" + ("b" * 64),
        "diagnostic_only": False,
        "promotion_eligible": False,
    }

    train._bind_parent_report(  # noqa: SLF001
        report,
        initialization=initialization,
        canonical_authority=authority,
    )

    bound = json.loads(report.read_text(encoding="utf-8"))
    assert bound["a1_canonical_parent_update_authority"] == authority
    assert bound["a1_parent_update_initialization"] == initialization
    assert bound["promotion_eligible"] is False
    assert bound["a1_lineage_dose"]["objective_exposure"] == {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": 6_144,
        "policy_base_active_sampled_rows": 1_399,
        "policy_aux_active_sampled_rows": 0,
        "policy_active_sampled_rows": 1_399,
        "value_active_sampled_rows": 6_144,
        "anchor_eligible_sampled_rows": 0,
        "completed_q_base_effective_weight_exposure": 0.0,
        "completed_q_aux_effective_weight_exposure": 0.0,
        "completed_q_base_active_rows": 0,
        "completed_q_aux_active_rows": 0,
        "completed_q_exposure_measurement_status": "bound_exactly",
    }


def test_parent_report_refuses_inconsistent_objective_counters(
    tmp_path,
) -> None:
    report = tmp_path / "train.report.json"
    report.write_text(
        json.dumps(
            {
                "steps_completed": 12,
                "training_row_draws": 6_144,
                "training_row_draws_semantics": (
                    "base_sampler_draw_events; may repeat rows; excludes_policy_aux"
                ),
                "base_training_row_draws": 6_144,
                "policy_aux_training_row_draws": 0,
                "policy_base_active_training_row_draws": 1_399,
                "policy_active_training_row_draws": 1_400,
                "value_active_training_row_draws": 6_144,
                "total_training_row_draws": 6_144,
                "policy_base_active_rows": 1_399,
                "policy_aux_active_rows": 0,
                "policy_total_active_rows": 1_400,
                "value_active_rows": 6_144,
                "policy_kl_anchor_eligible_rows": 0,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_sha = "sha256:" + ("a" * 64)

    with pytest.raises(SystemExit, match="canonical parent lineage refused"):
        train._bind_parent_report(  # noqa: SLF001
            report,
            initialization={
                "mode": "exact_parent",
                "parent": {"sha256": checkpoint_sha},
                "initializer": {"sha256": checkpoint_sha},
                "information_contract_migration": None,
            },
            canonical_authority={"diagnostic_only": False},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("a1_lineage_dose", {"forged": True}),
        ("a1_parent_update_initialization", {"mode": "other"}),
        ("a1_canonical_parent_update_authority", {"diagnostic_only": True}),
        ("promotion_eligible", True),
    ),
)
def test_parent_report_refuses_conflicting_child_provenance(
    tmp_path,
    field: str,
    value: object,
) -> None:
    report = tmp_path / "train.report.json"
    payload = {
        "steps_completed": 12,
        "training_row_draws": 6_144,
        "training_row_draws_semantics": (
            "base_sampler_draw_events; may repeat rows; excludes_policy_aux"
        ),
        "base_training_row_draws": 6_144,
        "policy_aux_training_row_draws": 0,
        "policy_base_active_training_row_draws": 1_399,
        "policy_active_training_row_draws": 1_399,
        "value_active_training_row_draws": 6_144,
        "total_training_row_draws": 6_144,
        "policy_base_active_rows": 1_399,
        "policy_aux_active_rows": 0,
        "policy_total_active_rows": 1_399,
        "value_active_rows": 6_144,
        "policy_kl_anchor_eligible_rows": 0,
        field: value,
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint_sha = "sha256:" + ("a" * 64)

    with pytest.raises(SystemExit, match="canonical parent report"):
        train._bind_parent_report(  # noqa: SLF001
            report,
            initialization={
                "mode": "exact_parent",
                "parent": {"sha256": checkpoint_sha},
                "initializer": {"sha256": checkpoint_sha},
                "information_contract_migration": None,
            },
            canonical_authority={"diagnostic_only": False},
        )


def test_raw_validation_semantics_are_consistent_across_frontier_steps() -> None:
    source = {"loss": 1.25}

    bound = train_bc._bind_raw_validation_semantics(  # noqa: SLF001
        source,
        training_value_player_outcome_balance_mode="sampler_balanced_v1",
    )

    assert source == {"loss": 1.25}
    assert bound == {
        "loss": 1.25,
        "measure": "raw_row_concat",
        "objective_matched": False,
        "training_value_player_outcome_balance_mode": "sampler_balanced_v1",
        "validation_value_player_outcome_balance_mode": "none",
        "warning": (
            "compatibility metric: raw held-out rows do not follow the "
            "authenticated component->game->row training measure, and validation "
            "uses natural outcomes rather than fitting training-only outcome balance"
        ),
    }


def test_canonical_memmap_binds_authenticated_validation_manifest(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = tmp_path / "selected-games.json"
    manifest.write_text('{"schema_version":"selected-games-v1"}\n', encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
    (corpus / "corpus_meta.json").write_text(
        json.dumps(
            {
                "a1_post_wave_audit": {
                    "validation_holdout": {
                        "path": str(manifest),
                        "file_sha256": digest,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert train._validation_manifest_from_memmap(corpus) == str(manifest.resolve())


def test_canonical_memmap_rejects_changed_validation_manifest(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = tmp_path / "selected-games.json"
    manifest.write_text("{}\n", encoding="utf-8")
    (corpus / "corpus_meta.json").write_text(
        json.dumps(
            {
                "a1_post_wave_audit": {
                    "validation_holdout": {
                        "path": str(manifest),
                        "file_sha256": "sha256:" + ("0" * 64),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="digest mismatch"):
        train._validation_manifest_from_memmap(corpus)


def _feature_observability_engine() -> dict[str, object]:
    return {
        "require_feature_learning_signal_modules": "event_encoder,value_head",
        "minimum_feature_learning_signal_observations": 2,
        "train_diagnostics_every_batches": 8,
    }


@pytest.mark.parametrize("max_steps", (12, 15))
def test_exact_cap_rejects_insufficient_feature_observations(max_steps: int) -> None:
    config = train.TrainConfig(
        max_steps=max_steps,
        exact_max_steps=True,
        grad_accum_steps=1,
    )

    with pytest.raises(
        SystemExit,
        match=(
            f"max_steps={max_steps} .*maximum_feature_learning_signal_observations=1 "
            ".*minimum_feature_learning_signal_observations=2"
        ),
    ):
        train._require_exact_cap_feature_observability(
            config, _feature_observability_engine()
        )


def test_exact_cap_accepts_feature_observation_boundary() -> None:
    config = train.TrainConfig(
        max_steps=16,
        exact_max_steps=True,
        grad_accum_steps=1,
    )

    train._require_exact_cap_feature_observability(
        config, _feature_observability_engine()
    )


def test_exact_cap_counts_only_cadence_hits_that_close_accumulation() -> None:
    config = train.TrainConfig(
        max_steps=4,
        exact_max_steps=True,
        grad_accum_steps=3,
    )
    engine = {
        "require_feature_learning_signal_modules": "event_encoder,value_head",
        "minimum_feature_learning_signal_observations": 2,
        "train_diagnostics_every_batches": 4,
    }

    with pytest.raises(
        SystemExit,
        match="maximum_feature_learning_signal_observations=1",
    ):
        train._require_exact_cap_feature_observability(config, engine)


def test_epoch_mode_skips_exact_cap_feature_observation_arithmetic() -> None:
    config = train.TrainConfig(
        max_steps=0,
        exact_max_steps=True,
        grad_accum_steps=1,
    )

    train._require_exact_cap_feature_observability(
        config, _feature_observability_engine()
    )


def test_production_hard_decision_mass_contract_requires_every_phase() -> None:
    settlement = "minimum_initial_settlement_policy_mass_fraction"
    road = "minimum_initial_road_policy_mass_fraction"
    discard = "minimum_discard_policy_mass_fraction"
    robber = "minimum_move_robber_policy_mass_fraction"

    with pytest.raises(SystemExit, match="missing=.*initial_road"):
        train._require_production_hard_decision_policy_mass_contract({settlement: 0.01})

    with pytest.raises(SystemExit, match="missing=.*discard.*move_robber"):
        train._require_production_hard_decision_policy_mass_contract(
            {settlement: 0.01, road: 0.02}
        )

    minima = train._require_production_hard_decision_policy_mass_contract(
        {settlement: 0.01, road: 0.02, discard: 0.03, robber: 0.04}
    )
    assert minima == {
        settlement: 0.01,
        road: 0.02,
        discard: 0.03,
        robber: 0.04,
    }


def test_parent_recipe_commissions_every_hard_decision_mass_floor() -> None:
    _config, engine = train._load_recipe(PARENT_RECIPE)  # noqa: SLF001

    minima = train._require_production_hard_decision_policy_mass_contract(engine)

    assert minima == {
        "minimum_initial_settlement_policy_mass_fraction": 0.02,
        "minimum_initial_road_policy_mass_fraction": 0.02,
        "minimum_discard_policy_mass_fraction": 0.02,
        "minimum_move_robber_policy_mass_fraction": 0.02,
    }


def test_canonical_non_moe_recipe_has_no_phantom_moe_objective() -> None:
    """Coverage normalization must not bind an objective with no live head."""

    fields = _fields()
    assert fields["moe_routed_experts"] == 0
    assert fields["moe_expert_ff_size"] == 0
    assert fields["moe_balance_loss_weight"] == 0.0


def test_canonical_forced_value_baseline_preserves_boundary_evidence() -> None:
    """Forced policy is inert while turn-boundary value evidence is retained."""

    fields = _fields()
    assert fields["forced_action_weight"] == 0.0
    assert fields["forced_row_value_weight"] == 1.0
    assert fields["forced_row_value_action_type_weights"] == ("END_TURN=1.0,ROLL=1.0")


def test_canonical_v7_value_routing_protects_only_the_shared_trunk() -> None:
    """V7 commissions a 0.1 shared boundary with a fully live private tower."""

    fields = _fields()
    engine = _payload()["engine_settings"]
    assert fields["value_trunk_grad_scale"] == 0.1
    assert engine["value_tower_split_layers"] == 1

    parent = json.loads(PARENT_RECIPE.read_text(encoding="utf-8"))
    assert parent["train_config"]["fields"]["value_trunk_grad_scale"] == 0.1
    assert parent["engine_settings"]["value_tower_split_layers"] == 1


@pytest.mark.parametrize("recipe", (RECIPE, PARENT_RECIPE))
def test_canonical_recipe_emphasizes_hard_decisions_without_weighting_value(
    recipe: Path,
) -> None:
    """V7 keeps the proven PLAY_TURN repair but restores hard-decision mass."""

    fields = json.loads(recipe.read_text(encoding="utf-8"))["train_config"]["fields"]

    assert fields["phase_weights"] == HARD_DECISION_POLICY_PHASE_WEIGHTS
    assert fields["value_phase_weights"] == "none"
    assert fields["forced_action_weight"] == 0.0


def test_hard_decision_mass_replay_uses_per_game_runtime_operator() -> None:
    evidence = json.loads(HARD_DECISION_EVIDENCE.read_text(encoding="utf-8"))
    replay = evidence["exact_runtime_replay"]
    assert replay["aggregate_projection_valid"] is False
    assert replay["training_row_count"] == 2_392_241
    assert replay["policy_objective_mass_fraction"] == pytest.approx(
        {
            "BUILD_INITIAL_ROAD": 0.05617698802311918,
            "BUILD_INITIAL_SETTLEMENT": 0.02808849401155959,
            "DISCARD": 0.05581640524684239,
            "MOVE_ROBBER": 0.3299306883373039,
            "PLAY_TURN": 0.5299874243811756,
        }
    )

    phases = np.asarray(
        [
            "DISCARD",
            "PLAY_TURN",
            "BUILD_INITIAL_SETTLEMENT",
            "BUILD_INITIAL_ROAD",
            "MOVE_ROBBER",
            "PLAY_TURN",
            "PLAY_TURN",
        ]
    )
    data = {
        "action_taken": np.arange(phases.size, dtype=np.int16),
        "phase": phases,
        "game_seed": np.asarray([11, 11, 22, 22, 22, 22, 22], dtype=np.int64),
        "legal_action_ids": np.tile(
            np.asarray([[0, 1]], dtype=np.int16), (phases.size, 1)
        ),
    }
    common = {
        "teacher_weights": {},
        "forced_action_weight": 0.0,
        "winner_sample_weight": 1.0,
        "loser_sample_weight": 1.0,
        "vp_margin_weight": 0.0,
        "vps_to_win": 10,
        "per_game_policy_weight": True,
        "per_game_policy_weight_mode": "equal",
    }

    def runtime_mass(
        phase_weights: dict[str, float], *, enforce_minima: bool = False
    ) -> tuple[dict[str, float], dict[str, object]]:
        weights = train_bc.build_sample_weights(
            data, phase_weights=phase_weights, **common
        )
        report = train_bc._policy_phase_objective_mass_admission(
            data,
            np.arange(phases.size, dtype=np.int64),
            policy_sample_weights=weights,
            sampling_weights=None,
            minimum_phase_mass_fractions=(
                {phase: 0.02 for phase in train_bc.HARD_DECISION_POLICY_MASS_PHASES}
                if enforce_minima
                else None
            ),
            objective_measure="synthetic_uniform_row_probability_x_policy_loss_weight",
        )
        return (
            {
                phase: row["policy_objective_mass_fraction"]
                for phase, row in report["per_phase"].items()
            },
            report,
        )

    baseline, _ = runtime_mass({"PLAY_TURN": 4.0})
    corrected, corrected_report = runtime_mass(
        train_bc._parse_weight_map(HARD_DECISION_POLICY_PHASE_WEIGHTS),
        enforce_minima=True,
    )
    factors = evidence["correction"]["relative_to_baseline_recipe"]
    denominator = sum(baseline[phase] * factors[phase] for phase in baseline)
    invalid_aggregate_projection = {
        phase: baseline[phase] * factors[phase] / denominator for phase in baseline
    }

    assert corrected["DISCARD"] == pytest.approx(0.1363636352)
    assert invalid_aggregate_projection["DISCARD"] == pytest.approx(0.1264367816)
    assert corrected["DISCARD"] != pytest.approx(
        invalid_aggregate_projection["DISCARD"]
    )
    assert corrected_report["admitted"] is True
    for phase in ("BUILD_INITIAL_ROAD", "DISCARD", "MOVE_ROBBER"):
        assert corrected[phase] > baseline[phase]
    assert corrected["PLAY_TURN"] < baseline["PLAY_TURN"]
    assert evidence["isolation"] == {
        "forced_action_weight": 0.0,
        "value_phase_weights": "none",
        "forced_rows_remain_value_only": True,
        "value_objective_phase_distribution_changed": False,
    }


def test_hard_decision_recipe_weights_only_active_policy_rows() -> None:
    fields = _fields()
    phases = np.asarray(
        ["BUILD_INITIAL_ROAD", "DISCARD", "MOVE_ROBBER", "PLAY_TURN", "DISCARD"]
    )
    data = {
        "action_taken": np.arange(phases.size, dtype=np.int16),
        "phase": phases,
        "legal_action_ids": np.asarray(
            [[0, 1], [0, 1], [0, 1], [0, 1], [0, -1]], dtype=np.int16
        ),
    }
    policy_weights = train_bc.build_sample_weights(
        data,
        teacher_weights={},
        phase_weights=train_bc._parse_weight_map(fields["phase_weights"]),
        forced_action_weight=float(fields["forced_action_weight"]),
        winner_sample_weight=1.0,
        loser_sample_weight=1.0,
        vp_margin_weight=0.0,
        vps_to_win=10,
    )
    value_weights = train_bc.build_value_sample_weights(data, phase_weights={})

    assert policy_weights / policy_weights[3] == pytest.approx(
        [0.5, 0.375, 0.75, 1.0, 0.0]
    )
    assert value_weights == pytest.approx(np.ones(phases.size))


def test_scratch_horizon_is_not_relabelled_as_the_parent_update_frontier() -> None:
    """The proven 32-step result was a parent update, not scratch evidence."""

    fields = _fields()
    engine = _payload()["engine_settings"]
    assert fields["init_checkpoint"] == ""
    assert fields["resume_optimizer"] is False
    assert fields["topology_residual_adapter"] is True
    assert "topology_residual_adapter" in engine[
        "require_feature_learning_signal_modules"
    ].split(",")
    assert engine["min_35m_params"] == 42_500_000
    assert engine["max_35m_params"] == 43_000_000
    assert fields["max_steps"] == 0
    assert fields["epochs"] == 3


def test_canonical_scratch_recipe_rejects_candidate_chaining() -> None:
    config, engine = train._load_recipe(RECIPE)
    assert engine["initialization_mode"] == "scratch_fresh_optimizer"
    with pytest.raises(SystemExit, match="forbids parent/grow checkpoints"):
        train._engine_namespace(
            config=config,
            engine_settings=engine,
            public_args=_public_args(init_checkpoint="/tmp/previous-candidate.pt"),
        )


def test_canonical_scratch_recipe_binds_hard_decision_mass_minima() -> None:
    config, engine = train._load_recipe(RECIPE)

    resolved = train._engine_namespace(
        config=config,
        engine_settings=engine,
        public_args=_public_args(),
    )

    assert resolved.minimum_initial_settlement_policy_mass_fraction == 0.02
    assert resolved.minimum_initial_road_policy_mass_fraction == 0.02
    assert resolved.minimum_discard_policy_mass_fraction == 0.02
    assert resolved.minimum_move_robber_policy_mass_fraction == 0.02


@pytest.mark.parametrize(
    ("recipe", "role"),
    (
        (RECIPE, "scratch_fresh_optimizer"),
        (PARENT_RECIPE, "parent_fresh_optimizer"),
    ),
)
def test_canonical_recipe_catalog_is_bound_to_role(recipe: Path, role: str) -> None:
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    assert payload["engine_settings"]["initialization_mode"] == role
    catalog_name = train.require_production_recipe(
        entrypoint="train", path=recipe, payload=payload
    )
    assert catalog_name
    assert role in train.CANONICAL_CONFIG_ROLES
    train._load_recipe(recipe)


def test_every_cataloged_training_recipe_resolves_without_name_registry() -> None:
    for entry in production_recipes("train"):
        path = Path(entry["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        role = payload["engine_settings"]["initialization_mode"]

        assert role in train.CANONICAL_CONFIG_ROLES
        _config, engine = train._load_recipe(path)
        assert engine["initialization_mode"] == role


@pytest.mark.parametrize("recipe", (RECIPE, PARENT_RECIPE))
def test_canonical_scalar_value_objective_uses_stable_binary_win_bce(
    recipe: Path,
) -> None:
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    engine = payload["engine_settings"]

    assert engine["scalar_value_objective"] == "binary_win_bce"
    assert engine["scalar_value_loss_readout"] == "deployed_tanh"
    assert engine["scalar_value_loss_scale"] == 1.0


def test_parent_update_recipe_reproduces_split1_selected_step12() -> None:
    payload = json.loads(PARENT_RECIPE.read_text(encoding="utf-8"))
    engine = payload["engine_settings"]
    fields = payload["train_config"]["fields"]

    assert engine["initialization_mode"] == "parent_fresh_optimizer"
    assert engine["base_sampler"] == "weighted_replacement_v1"
    assert engine["checkpoint_steps"] == "8,10"
    assert fields["batch_size"] == 64
    assert 8 * fields["batch_size"] == 512
    assert fields["epochs"] == 999
    assert fields["max_steps"] == 12
    assert fields["exact_max_steps"] is True
    assert fields["lr"] == 6e-5
    assert fields["lr_schedule"] == "flat"
    assert fields["lr_warmup_steps"] == 16
    assert fields["validation_fraction"] == 0.05
    # The parent-update recipe accepts arbitrary authenticated coherent corpora;
    # validation is the deterministic game-level 5% split above, not stale
    # seed ranges from one historical wave.
    assert fields["validation_game_seed_ranges"] == ""
    assert fields["validation_max_samples"] == 0
    assert fields["minimum_policy_effective_rows_per_global_batch"] == 0.0
    assert fields["value_trunk_grad_scale"] == 0.1
    assert fields["post_policy_dose_value_trunk_grad_scale"] == 1.0
    assert fields["policy_dose_lr_area"] == 0.0
    assert fields["policy_kl_target"] is None
    assert fields["policy_kl_anchor_weight"] == 0.0
    assert fields["freeze_modules"] == ""
    assert fields["forced_action_weight"] == 0.0
    assert fields["forced_row_value_weight"] == 1.0
    assert fields["forced_row_value_action_type_weights"] == ("END_TURN=1.0,ROLL=1.0")
    assert fields["resume_optimizer"] is False
    assert fields["init_checkpoint"] == ""
    assert fields["grow_from_checkpoint"] == ""
    assert engine["value_tower_split_layers"] == 1
    assert payload["train_config"]["schema_version"] == CONFIG_SCHEMA_VERSION


def test_parent_update_requires_exact_parent_and_uses_corpus_target_identity() -> None:
    config, engine = train._load_recipe(PARENT_RECIPE)
    with pytest.raises(SystemExit, match="requires --init-checkpoint"):
        train._engine_namespace(
            config=config,
            engine_settings=engine,
            public_args=_public_args(),
        )

    resolved = train._engine_namespace(
        config=config,
        engine_settings=engine,
        public_args=_public_args(init_checkpoint="/tmp/f7.pt"),
    )
    assert resolved.init_checkpoint == "/tmp/f7.pt"
    assert resolved.resume_optimizer is False
    assert resolved.grow_from_checkpoint == ""
    assert resolved.accepted_policy_target_identity_sha256 == []

    from tools import train_bc

    assert train_bc._parse_checkpoint_steps(
        resolved.checkpoint_steps, max_steps=resolved.max_steps
    ) == (8, 10)
    train_bc._validate_coverage_sampler_configuration(
        resolved, categorical_value_loss_weight=0.0
    )


def test_parent_initializer_requires_exact_incumbent_upgrade_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "f7.pt"
    initializer = tmp_path / "f7-current-v5-split1.pt"
    receipt = tmp_path / "upgrade.json"
    parent.write_bytes(b"legacy-parent")
    initializer.write_bytes(b"current-v5-split1")
    receipt.write_text("{}", encoding="utf-8")
    args = _public_args(init_checkpoint=str(initializer))
    args.parent_checkpoint = str(parent)

    with pytest.raises(SystemExit, match="migration-receipt is required"):
        train._parent_initializer_binding(args)

    from tools import a1_information_contract_migration as migration

    parent_ref = train._checkpoint_ref(str(parent), where="parent")
    initializer_ref = train._checkpoint_ref(str(initializer), where="initializer")
    receipt_parent_ref = {
        **parent_ref,
        "path": "/producer-host/original/f7.pt",
    }
    receipt_initializer_ref = {
        **initializer_ref,
        "path": "/migration-host/original/f7-current-v6-split1.pt",
    }
    receipt_ref = {
        "path": str(receipt.resolve()),
        "sha256": "sha256:" + "a" * 64,
    }
    monkeypatch.setattr(
        migration,
        "verify_receipt",
        lambda _path: {
            "migration": migration.MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1,
            "source": receipt_parent_ref,
            "migrated_initializer": receipt_initializer_ref,
            "receipt": receipt_ref,
            "forward_identical": False,
            "promotion_eligible": False,
        },
    )
    args.information_contract_migration_receipt = str(receipt)
    binding = train._parent_initializer_binding(args)

    assert binding["parent"] == parent_ref
    assert binding["initializer"] == initializer_ref
    assert binding["information_contract_migration"] == {
        "schema_version": "a1-lineage-information-contract-migration-v1",
        "migration": migration.MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1,
        "receipt": receipt_ref["path"],
        "receipt_sha256": receipt_ref["sha256"],
        "source_checkpoint_sha256": parent_ref["sha256"],
        "migrated_initializer_sha256": initializer_ref["sha256"],
        "forward_identical": False,
        "promotion_eligible": False,
    }


def test_parent_initializer_accepts_exact_v5_to_v7_compatibility_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "f7-v5.pt"
    initializer = tmp_path / "f7-v7.pt"
    receipt = tmp_path / "v7-migration.json"
    parent.write_bytes(b"exact-v5-parent")
    initializer.write_bytes(b"exact-v7-initializer")
    receipt.write_text("{}", encoding="utf-8")
    args = _public_args(init_checkpoint=str(initializer))
    args.parent_checkpoint = str(parent)
    args.information_contract_migration_receipt = str(receipt)

    from tools import a1_information_contract_migration as migration

    parent_ref = train._checkpoint_ref(str(parent), where="parent")
    initializer_ref = train._checkpoint_ref(str(initializer), where="initializer")
    receipt_ref = {
        "path": str(receipt.resolve()),
        "sha256": "sha256:" + "b" * 64,
    }
    monkeypatch.setattr(
        migration,
        "verify_receipt",
        lambda _path: {
            "migration": migration.MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY,
            "source": {**parent_ref, "path": "/producer/f7-v5.pt"},
            "migrated_initializer": {
                **initializer_ref,
                "path": "/learner/f7-v7.pt",
            },
            "receipt": receipt_ref,
            "forward_identical": True,
            "promotion_eligible": False,
        },
    )

    binding = train._parent_initializer_binding(args)

    assert binding["parent"] == parent_ref
    assert binding["initializer"] == initializer_ref
    assert binding["information_contract_migration"] == {
        "schema_version": "a1-lineage-information-contract-migration-v1",
        "migration": migration.MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY,
        "receipt": receipt_ref["path"],
        "receipt_sha256": receipt_ref["sha256"],
        "source_checkpoint_sha256": parent_ref["sha256"],
        "migrated_initializer_sha256": initializer_ref["sha256"],
        "forward_identical": True,
        "promotion_eligible": False,
    }


def test_parent_update_rejects_growth_or_optimizer_resume() -> None:
    base, engine = train._load_recipe(PARENT_RECIPE)
    variants = (
        dataclasses.replace(base, grow_from_checkpoint="/tmp/grow.pt"),
        dataclasses.replace(base, resume_optimizer=True),
    )
    for variant in variants:
        with pytest.raises(SystemExit, match="fresh optimizer"):
            train._engine_namespace(
                config=variant,
                engine_settings=engine,
                public_args=_public_args(init_checkpoint="/tmp/f7.pt"),
            )


def test_canonical_train_raises_its_own_fd_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[tuple[int, int]] = []
    monkeypatch.setattr(
        train.resource,
        "getrlimit",
        lambda _kind: (1024, 1_048_576),
    )
    monkeypatch.setattr(
        train.resource,
        "setrlimit",
        lambda _kind, value: applied.append(value),
    )

    train._ensure_runtime_limits()

    assert applied == [(65_536, 1_048_576)]
