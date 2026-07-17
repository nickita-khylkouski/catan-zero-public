from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import stat

import pytest
import torch

from catan_zero.rl.entity_feature_adapter import (
    RUST_ENTITY_ADAPTER_V5,
    RUST_ENTITY_ADAPTER_V6,
    checkpoint_entity_feature_adapter_metadata,
)
from catan_zero.rl.entity_token_policy import EntityGraphConfig
from tools import a1_information_contract_migration as migration


def _evidence(tmp_path: Path) -> dict:
    source = tmp_path / "source.pt"
    target = tmp_path / "target.pt"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    return {
        "migration": migration.MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1,
        "source": migration._ref(source),  # noqa: SLF001
        "migrated_initializer": migration._ref(target),  # noqa: SLF001
        "source_adapter": "rust_entity_adapter_v2_actor_private_only",
        "target_adapter": (
            "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
        ),
        "forward_identical": False,
        "promotion_eligible": False,
        "commissioning_status": "non_promotable_architecture_treatment",
        "step0_anchor_evidence": {"bound": True},
        "topology_replay": {"bound": True},
    }


def test_migration_receipt_replays_and_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path)
    monkeypatch.setattr(migration, "inspect_migration", lambda *_a, **_k: evidence)
    receipt = tmp_path / "migration.json"

    issued = migration.issue_receipt(
        Path(evidence["source"]["path"]),
        Path(evidence["migrated_initializer"]["path"]),
        receipt,
    )
    replayed = migration.verify_receipt(receipt)

    assert issued["forward_identical"] is False
    assert issued["promotion_eligible"] is False
    assert replayed["receipt"]["path"] == str(receipt.resolve())
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o444
    with pytest.raises(migration.MigrationError, match="refusing to overwrite"):
        migration.issue_receipt(
            Path(evidence["source"]["path"]),
            Path(evidence["migrated_initializer"]["path"]),
            receipt,
        )


def test_migration_artifacts_refuse_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.pt"
    target.write_bytes(b"checkpoint")
    alias = tmp_path / "alias.pt"
    alias.symlink_to(target)

    with pytest.raises(migration.MigrationError, match="must not be a symlink"):
        migration._ref(alias)  # noqa: SLF001


def test_anchor_replay_tolerates_only_float32_reduction_noise() -> None:
    expected = {
        "schema_version": migration.ANCHOR_SCHEMA,
        "count": 4,
        "changed": True,
        "measurements": [0.17797088623046875, 0.014811873435974121],
    }
    replayed = {
        **expected,
        "measurements": [0.17797112464904785, 0.014811880886554718],
    }

    assert migration._anchor_replay_matches(expected, replayed)  # noqa: SLF001
    assert not migration._anchor_replay_matches(  # noqa: SLF001
        expected, {**replayed, "count": 5}
    )
    assert not migration._anchor_replay_matches(  # noqa: SLF001
        expected, {**replayed, "measurements": [0.18, replayed["measurements"][1]]}
    )


def _v7_anchor_evidence() -> dict:
    identity = "sha256:" + "a" * 64
    anchor = {
        "label": "resource_initial_road",
        "phase": "BUILD_INITIAL_ROAD",
        "actor": "RED",
        "actor_resource_total": 2,
        "legal_width": 3,
        "migration_output_max_abs_diff": 0.0,
        "migration_output_max_abs_diff_by_key": {
            "logits": 0.0,
            "value": 0.0,
            "final_vp": 0.0,
            "q_values": 0.0,
        },
        "legal_policy_forward_kl": 0.0,
        "legal_policy_reverse_kl": 0.0,
        "legal_policy_top1_flip": False,
        "scalar_value_abs_error": 0.0,
        "feature_max_abs_diff": 0.25,
        "feature_changed_value_count": 1,
        "anchor_identity_sha256": identity,
    }
    anchor_set = (
        "sha256:"
        + hashlib.sha256(
            json.dumps([identity], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return {
        "schema_version": migration.V7_INPUT_ANCHOR_SCHEMA,
        "device": "cpu",
        "source_adapter": RUST_ENTITY_ADAPTER_V5,
        "target_adapter": RUST_ENTITY_ADAPTER_V6,
        "public_observation": True,
        "separate_adapter_specific_entity_features": True,
        "separate_adapter_specific_action_contexts": True,
        "adapter_features_identical": False,
        "forward_identical": True,
        "promotion_eligible": False,
        "topology_construction_proof": "deterministic_parameter_replay_in_receipt",
        "migration_output_max_abs_diff": 0.0,
        "feature_max_abs_diff": 0.25,
        "feature_changed_value_count": 1,
        "legal_policy_forward_kl_mean": 0.0,
        "legal_policy_forward_kl_max": 0.0,
        "legal_policy_reverse_kl_mean": 0.0,
        "legal_policy_reverse_kl_max": 0.0,
        "legal_policy_top1_flip_count": 0,
        "legal_policy_top1_flip_rate": 0.0,
        "scalar_value_rmse": 0.0,
        "scalar_value_max_abs_error": 0.0,
        "anchor_count": 1,
        "anchor_set_sha256": anchor_set,
        "anchors": [anchor],
    }


def test_v7_anchor_requires_real_feature_change_and_exact_forward_identity() -> None:
    evidence = _v7_anchor_evidence()

    assert (
        migration._verify_anchor_evidence(  # noqa: SLF001
            evidence,
            migration=migration.MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY,
        )
        == evidence
    )

    output_drift = copy.deepcopy(evidence)
    output_drift["migration_output_max_abs_diff"] = 0.1
    output_drift["anchors"][0]["migration_output_max_abs_diff"] = 0.1
    with pytest.raises(migration.MigrationError, match="exact forward"):
        migration._verify_anchor_evidence(  # noqa: SLF001
            output_drift,
            migration=migration.MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY,
        )


def test_v7_replay_does_not_apply_float_tolerance_to_exact_forward_claim() -> None:
    evidence = _v7_anchor_evidence()
    replayed = copy.deepcopy(evidence)
    replayed["migration_output_max_abs_diff"] = 5.0e-7
    replayed["anchors"][0]["migration_output_max_abs_diff"] = 5.0e-7

    # Historical diagnostic replay tolerance alone would accept this delta.
    assert migration._anchor_replay_matches(evidence, replayed)  # noqa: SLF001
    with pytest.raises(migration.MigrationError, match="exact forward"):
        migration._verify_anchor_replay(  # noqa: SLF001
            evidence,
            replayed,
            migration=migration.MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY,
        )


def test_v7_receipt_replay_uses_v5_to_v7_anchor_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "f7-v5.pt"
    migrated = tmp_path / "f7-v7.pt"
    source.write_bytes(b"source")
    migrated.write_bytes(b"migrated")
    source_ref = migration._ref(source)  # noqa: SLF001
    evidence = _v7_anchor_evidence()
    before = {
        "entity_feature_adapter": checkpoint_entity_feature_adapter_metadata(
            RUST_ENTITY_ADAPTER_V5
        )
    }
    after = {
        "entity_feature_adapter": checkpoint_entity_feature_adapter_metadata(
            RUST_ENTITY_ADAPTER_V6
        ),
        "information_contract_migration_provenance": {
            "schema_version": migration.CHECKPOINT_PROVENANCE_SCHEMA,
            "migration": "v5_to_v7_input_compatibility",
            "source_checkpoint_sha256": source_ref["sha256"].removeprefix("sha256:"),
            "source_adapter": RUST_ENTITY_ADAPTER_V5,
            "target_adapter": RUST_ENTITY_ADAPTER_V6,
            "source_input_routing": (
                "v5_legacy_resource_and_initial_road_inputs"
            ),
            "target_input_routing": (
                "v6_raw_inputs_with_legacy_encoder_views_plus_zero_output_residuals"
            ),
            "forward_identical": True,
            "promotion_eligible": False,
            "commissioning_status": "non_promotable_architecture_treatment",
            "step0_anchor_evidence": evidence,
        },
    }
    monkeypatch.setattr(
        torch,
        "load",
        lambda path, **_kwargs: before if Path(path) == source.resolve() else after,
    )

    class _Model:
        def eval(self) -> None:
            return None

    class _Policy:
        model = _Model()

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from tools import f69_upgrade_checkpoint_config as checkpoint_migration

    monkeypatch.setattr(
        EntityGraphPolicy, "load", lambda *_args, **_kwargs: _Policy()
    )
    observed: list[str] = []

    def _anchors(*_args: object, migration: str) -> dict:
        observed.append(migration)
        return evidence

    monkeypatch.setattr(
        checkpoint_migration, "_migration_anchor_evidence", _anchors
    )
    monkeypatch.setattr(
        migration,
        "_verify_v7_input_routing_delta",
        lambda *_args: {"shared_parameters_bit_identical": True},
    )

    inspected = migration.inspect_migration(
        source,
        migrated,
        migration=migration.MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY,
    )

    assert observed == ["v5_to_v7_input_compatibility"]
    assert inspected["forward_identical"] is True


def _v7_delta() -> tuple[dict, dict]:
    before_config = EntityGraphConfig(action_size=4, static_action_feature_size=3)
    after_config = EntityGraphConfig(
        action_size=4,
        static_action_feature_size=3,
        v6_compatibility_preserving_inputs=True,
    )
    before = {
        "policy_type": "entity_graph",
        "config": {
            field: getattr(before_config, field)
            for field in before_config.__dataclass_fields__
        },
        "entity_feature_adapter": checkpoint_entity_feature_adapter_metadata(
            RUST_ENTITY_ADAPTER_V5
        ),
        "model": {"player_encoder.0.weight": torch.ones((2, 3))},
        "mask_hidden_info": True,
    }
    after = copy.deepcopy(before)
    after["config"] = {
        field: getattr(after_config, field)
        for field in after_config.__dataclass_fields__
    }
    after["entity_feature_adapter"] = checkpoint_entity_feature_adapter_metadata(
        RUST_ENTITY_ADAPTER_V6
    )
    after["model"]["v6_exact_resource_residual.weight"] = torch.zeros((2, 7))
    after["model"]["v6_initial_road_residual.weight"] = torch.zeros((2, 1))
    after["information_contract_migration_provenance"] = {"bound": True}
    return before, after


def test_v7_delta_replay_accepts_only_zero_residual_and_one_config_flag() -> None:
    before, after = _v7_delta()

    replay = migration._verify_v7_input_routing_delta(before, after)  # noqa: SLF001

    assert replay["shared_parameters_bit_identical"] is True
    assert replay["new_parameter_initialization"] == {
        "v6_exact_resource_residual.weight": "zeros",
        "v6_initial_road_residual.weight": "zeros",
    }

    nonzero = copy.deepcopy(after)
    nonzero["model"]["v6_exact_resource_residual.weight"][0, 0] = 1.0
    with pytest.raises(migration.MigrationError, match="not initialized to zero"):
        migration._verify_v7_input_routing_delta(before, nonzero)  # noqa: SLF001

    metadata_drift = copy.deepcopy(after)
    metadata_drift["mask_hidden_info"] = False
    with pytest.raises(migration.MigrationError, match="metadata"):
        migration._verify_v7_input_routing_delta(  # noqa: SLF001
            before, metadata_drift
        )
