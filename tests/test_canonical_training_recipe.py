from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pytest

from tools import train


REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "configs/training/a1_current_35m_b200.schema1.json"
PARENT_RECIPE = REPO / "configs/training/a1_parent_update_35m_b200.schema1.json"


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
        architecture_upgrade_receipt="",
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
    assert "minimum_policy_effective_rows_per_global_batch" not in _payload()[
        "engine_settings"
    ]
    assert fields["minimum_policy_effective_rows_per_global_batch"] == 32.0
    assert fields["data_format"] == "memmap"
    assert fields["validation_max_samples"] == 0


def test_production_opening_mass_contract_is_fail_closed_until_reviewed() -> None:
    engine = _payload()["engine_settings"]

    with pytest.raises(SystemExit, match="remains fail-closed"):
        train._require_production_opening_policy_mass_contract(engine)


def test_parent_production_launcher_refuses_impossible_observability_first() -> None:
    with pytest.raises(
        SystemExit,
        match=(
            "max_steps=12 .*maximum_feature_learning_signal_observations=1 "
            ".*minimum_feature_learning_signal_observations=2"
        ),
    ):
        train.main(
            [
                "--config",
                str(PARENT_RECIPE),
                "--data",
                "/tmp/unopened-data",
                "--checkpoint",
                "/tmp/unwritten-candidate.pt",
                "--report",
                "/tmp/unwritten-train.json",
                "--init-checkpoint",
                "/tmp/unopened-initializer.pt",
            ]
        )


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


def test_epoch_mode_skips_exact_cap_feature_observation_arithmetic() -> None:
    config = train.TrainConfig(
        max_steps=0,
        exact_max_steps=True,
        grad_accum_steps=1,
    )

    train._require_exact_cap_feature_observability(
        config, _feature_observability_engine()
    )


def test_production_opening_mass_contract_accepts_only_complete_reviewed_pair() -> None:
    settlement = "minimum_initial_settlement_policy_mass_fraction"
    road = "minimum_initial_road_policy_mass_fraction"

    with pytest.raises(SystemExit, match="missing=.*initial_road"):
        train._require_production_opening_policy_mass_contract({settlement: 0.01})

    minima = train._require_production_opening_policy_mass_contract(
        {settlement: 0.01, road: 0.02}
    )
    assert minima == {settlement: 0.01, road: 0.02}


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
    assert fields["forced_row_value_action_type_weights"] == (
        "END_TURN=1.0,ROLL=1.0"
    )


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


@pytest.mark.parametrize(
    ("recipe", "role"),
    (
        (RECIPE, "scratch_fresh_optimizer"),
        (PARENT_RECIPE, "parent_fresh_optimizer"),
    ),
)
def test_canonical_recipe_catalog_is_bound_to_role(
    recipe: Path, role: str
) -> None:
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    assert payload["engine_settings"]["initialization_mode"] == role
    catalog_name = train.require_production_recipe(
        entrypoint="train", path=recipe, payload=payload
    )
    assert train.CANONICAL_CONFIG_ROLES_BY_CATALOG_NAME[catalog_name] == role
    train._load_recipe(recipe)


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
    assert engine["checkpoint_steps"] == "8"
    assert fields["batch_size"] == 64
    assert 8 * fields["batch_size"] == 512
    assert fields["epochs"] == 999
    assert fields["max_steps"] == 12
    assert fields["exact_max_steps"] is True
    assert fields["lr"] == 6e-5
    assert fields["lr_schedule"] == "flat"
    assert fields["lr_warmup_steps"] == 16
    assert fields["validation_fraction"] == 0.125
    # The parent-update recipe accepts arbitrary authenticated coherent corpora;
    # validation is the deterministic game-level 12.5% split above, not stale
    # seed ranges from one historical wave.
    assert fields["validation_game_seed_ranges"] == ""
    assert fields["validation_max_samples"] == 0
    assert fields["minimum_policy_effective_rows_per_global_batch"] == 0.0
    assert fields["value_trunk_grad_scale"] == 1.0
    assert fields["post_policy_dose_value_trunk_grad_scale"] == 1.0
    assert fields["policy_dose_lr_area"] == 0.0
    assert fields["policy_kl_target"] is None
    assert fields["policy_kl_anchor_weight"] == 0.0
    assert fields["freeze_modules"] == ""
    assert fields["forced_action_weight"] == 0.0
    assert fields["forced_row_value_weight"] == 1.0
    assert fields["forced_row_value_action_type_weights"] == (
        "END_TURN=1.0,ROLL=1.0"
    )
    assert fields["resume_optimizer"] is False
    assert fields["init_checkpoint"] == ""
    assert fields["grow_from_checkpoint"] == ""
    assert engine["value_tower_split_layers"] == 1
    assert payload["train_config"]["schema_version"] == 20


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
    ) == (8,)
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

    with pytest.raises(SystemExit, match="upgrade-receipt is required"):
        train._parent_initializer_binding(args)

    from tools import a1_function_preserving_upgrade as upgrade

    parent_ref = train._checkpoint_ref(str(parent), where="parent")
    initializer_ref = train._checkpoint_ref(str(initializer), where="initializer")
    receipt_ref = {
        "path": str(receipt.resolve()),
        "sha256": "sha256:" + "a" * 64,
    }
    monkeypatch.setattr(
        upgrade,
        "verify_receipt",
        lambda _path: {
            "module": upgrade.MODULE_CURRENT_V5_TOPOLOGY_VALUE_TOWER_SPLIT_1,
            "source": parent_ref,
            "upgraded_initializer": initializer_ref,
            "receipt": receipt_ref,
        },
    )
    args.architecture_upgrade_receipt = str(receipt)
    binding = train._parent_initializer_binding(args)

    assert binding["parent"] == parent_ref
    assert binding["initializer"] == initializer_ref
    assert binding["function_preserving_upgrade"] == {
        "schema_version": "a1-lineage-function-preserving-upgrade-v1",
        "module": upgrade.MODULE_CURRENT_V5_TOPOLOGY_VALUE_TOWER_SPLIT_1,
        "receipt": receipt_ref["path"],
        "receipt_sha256": receipt_ref["sha256"],
        "source_checkpoint_sha256": parent_ref["sha256"],
        "upgraded_initializer_sha256": initializer_ref["sha256"],
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
