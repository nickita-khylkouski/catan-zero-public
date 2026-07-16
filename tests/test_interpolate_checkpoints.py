from __future__ import annotations

from pathlib import Path

import pytest
import torch

from catan_zero.rl.entity_feature_adapter import (
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
    LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION,
    checkpoint_entity_feature_adapter_metadata,
)
from tools.interpolate_checkpoints import (
    interpolate_checkpoints,
    write_interpolation_receipt,
)


def _write_checkpoint(
    path: Path,
    *,
    bias: float,
    hidden_size: int = 2,
    action_feature_bias: float = 7.0,
) -> None:
    torch.save(
        {
            "observation_size": 3,
            "action_size": 4,
            "hidden_size": hidden_size,
            "architecture": "candidate",
            "use_action_id_embedding": True,
            "context_action_feature_size": 1,
            "action_features": torch.full((4, 2), action_feature_bias),
            "model": {"0.weight": torch.full((2, 3), bias)},
            "actor": {"weight": torch.full((2, 2), bias)},
            "critic": {"weight": torch.full((1, 2), bias)},
            "q_head": None,
            "q_state": {"weight": torch.full((2, 2), bias)},
            "q_action_encoder": {"0.weight": torch.full((2, 3), bias)},
            "q_action_bias": {"weight": torch.full((1, 3), bias)},
            "action_encoder": {"0.weight": torch.full((2, 3), bias)},
            "action_id_embedding": {"weight": torch.full((4, 2), bias)},
            "action_bias": {"weight": torch.full((1, 3), bias)},
            "step": torch.tensor(7, dtype=torch.int64),
        },
        path,
    )


def _append_metadata(path: Path) -> None:
    value = torch.load(path, map_location="cpu", weights_only=False)
    value["value_training"] = {"trained_value_readouts": ["scalar"]}
    torch.save(value, path)


def test_interpolate_checkpoints_writes_multiple_alpha_outputs(tmp_path: Path) -> None:
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(base, bias=0.0)
    _write_checkpoint(candidate, bias=1.0)
    _append_metadata(candidate)

    outputs = interpolate_checkpoints(
        base=base,
        candidate=candidate,
        alphas=(0.1, 0.25),
        output_template=str(tmp_path / "blend_a{alpha}.pt"),
    )

    assert [path.name for path in outputs] == ["blend_a0p1.pt", "blend_a0p25.pt"]
    first = torch.load(outputs[0], map_location="cpu")
    second = torch.load(outputs[1], map_location="cpu")
    assert torch.allclose(first["model"]["0.weight"], torch.full((2, 3), 0.1))
    assert torch.allclose(second["action_bias"]["weight"], torch.full((1, 3), 0.25))
    assert torch.equal(first["action_features"], torch.full((4, 2), 7.0))
    assert first["hidden_size"] == 2
    assert first["step"].item() == 7
    assert first["checkpoint_interpolation"]["diagnostic_only"] is True
    assert first["checkpoint_interpolation"]["promotion_eligible"] is False
    assert first["checkpoint_interpolation"]["alpha"] == pytest.approx(0.1)
    assert first["checkpoint_interpolation"]["base"]["path"] == str(base.resolve())
    assert first["checkpoint_interpolation"]["candidate"]["path"] == str(
        candidate.resolve()
    )

    receipt = tmp_path / "interpolation.receipt.json"
    value = write_interpolation_receipt(
        base=base,
        candidate=candidate,
        alphas=(0.1, 0.25),
        outputs=outputs,
        receipt=receipt,
    )
    assert receipt.is_file()
    assert value["diagnostic_only"] is True
    assert value["promotion_eligible"] is False
    assert value["non_floating_source"] == "base"
    assert value["output_metadata_source"] == "base"
    assert value["candidate_only_metadata_ignored"] == ["value_training"]
    assert [row["alpha"] for row in value["outputs"]] == [0.1, 0.25]
    assert all(row["sha256"].startswith("sha256:") for row in value["outputs"])
    assert any(
        row["path"] == "checkpoint.step" and row["floating"] is False
        for row in value["tensor_schema"]
    )


def test_interpolate_checkpoints_rejects_incompatible_metadata(tmp_path: Path) -> None:
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(base, bias=0.0, hidden_size=2)
    _write_checkpoint(candidate, bias=1.0, hidden_size=4)

    with pytest.raises(ValueError, match="hidden_size"):
        interpolate_checkpoints(
            base=base,
            candidate=candidate,
            alphas=(0.1,),
            output_template=str(tmp_path / "blend.pt"),
        )


def test_interpolate_checkpoints_rejects_changed_immutable_tensor(tmp_path: Path) -> None:
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(base, bias=0.0, action_feature_bias=7.0)
    _write_checkpoint(candidate, bias=1.0, action_feature_bias=8.0)

    with pytest.raises(ValueError, match="immutable checkpoint tensor differs"):
        interpolate_checkpoints(
            base=base,
            candidate=candidate,
            alphas=(0.1,),
            output_template=str(tmp_path / "blend.pt"),
        )


def test_interpolate_checkpoints_allows_asymmetric_nontensor_metadata(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(base, bias=0.0)
    _write_checkpoint(candidate, bias=1.0)

    base_value = torch.load(base, map_location="cpu", weights_only=False)
    base_value["training_information_surface"] = {"event_tensor_width": 41}
    torch.save(base_value, base)
    candidate_value = torch.load(candidate, map_location="cpu", weights_only=False)
    candidate_value["config"] = {"fields": {"belief_resource_head": False}}
    torch.save(candidate_value, candidate)

    outputs = interpolate_checkpoints(
        base=base,
        candidate=candidate,
        alphas=(0.5,),
        output_template=str(tmp_path / "blend.pt"),
    )

    blended = torch.load(outputs[0], map_location="cpu", weights_only=False)
    assert torch.allclose(blended["model"]["0.weight"], torch.full((2, 3), 0.5))
    assert blended["training_information_surface"] == {"event_tensor_width": 41}
    assert "config" not in blended


@pytest.mark.parametrize(
    ("key", "left", "right"),
    [
        ("mask_hidden_info", True, False),
        (
            "entity_feature_adapter",
            checkpoint_entity_feature_adapter_metadata(
                CURRENT_RUST_ENTITY_ADAPTER_VERSION
            ),
            {
                "schema_version": ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
                "version": "obsolete-v1",
            },
        ),
        ("public_award_feature_contract", "legacy_zero_v0", "authoritative_v1"),
        ("static_action_features_sha256", "catalog-a", "catalog-b"),
    ],
)
def test_entity_graph_interpolation_refuses_semantic_mismatch(
    tmp_path: Path,
    key: str,
    left: object,
    right: object,
) -> None:
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(base, bias=0.0)
    _write_checkpoint(candidate, bias=1.0)
    base_value = torch.load(base, map_location="cpu", weights_only=False)
    candidate_value = torch.load(candidate, map_location="cpu", weights_only=False)
    shared_config = {"fields": {"action_target_gather": False}}
    for value, semantic in ((base_value, left), (candidate_value, right)):
        value["policy_type"] = "entity_graph"
        value["config"] = shared_config
        value["action_mask_version"] = "mask-v1"
        value.setdefault("mask_hidden_info", True)
        value.setdefault(
            "entity_feature_adapter",
            checkpoint_entity_feature_adapter_metadata(
                CURRENT_RUST_ENTITY_ADAPTER_VERSION
            ),
        )
        value.setdefault("public_award_feature_contract", "legacy_zero_v0")
        value.setdefault("static_action_features_sha256", "catalog-a")
        value[key] = semantic
    torch.save(base_value, base)
    torch.save(candidate_value, candidate)

    error_pattern = (
        "entity feature adapter" if key == "entity_feature_adapter" else key
    )
    with pytest.raises(ValueError, match=error_pattern):
        interpolate_checkpoints(
            base=base,
            candidate=candidate,
            alphas=(0.5,),
            output_template=str(tmp_path / "blend.pt"),
        )


def test_entity_graph_interpolation_accepts_missing_legacy_and_explicit_v2(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(base, bias=0.0)
    _write_checkpoint(candidate, bias=1.0)
    base_value = torch.load(base, map_location="cpu", weights_only=False)
    candidate_value = torch.load(candidate, map_location="cpu", weights_only=False)
    for value in (base_value, candidate_value):
        value["policy_type"] = "entity_graph"
        value["config"] = {"fields": {"action_target_gather": False}}
        value["action_mask_version"] = "mask-v1"
        value["mask_hidden_info"] = True
        value["public_award_feature_contract"] = "legacy_zero_v0"
        value["static_action_features_sha256"] = "catalog-a"
    candidate_value["entity_feature_adapter"] = (
        checkpoint_entity_feature_adapter_metadata(
            LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION
        )
    )
    torch.save(base_value, base)
    torch.save(candidate_value, candidate)

    outputs = interpolate_checkpoints(
        base=base,
        candidate=candidate,
        alphas=(0.5,),
        output_template=str(tmp_path / "blend.pt"),
    )

    blended = torch.load(outputs[0], map_location="cpu", weights_only=False)
    assert "entity_feature_adapter" not in blended
