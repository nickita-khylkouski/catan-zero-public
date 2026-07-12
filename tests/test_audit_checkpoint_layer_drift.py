from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy
from tools.audit_checkpoint_layer_drift import (
    DriftAuditError,
    audit_checkpoints,
    main,
)


def _checkpoint(*, config_width: int = 8, overrides=None):
    model = {
        "hex_encoder.0.weight": torch.ones(2),
        "blocks.0.attn.in_proj_weight": torch.ones(2),
        "blocks.1.ff.0.weight": torch.ones(2),
        "action_encoder.0.weight": torch.ones(2),
        "value_head.0.weight": torch.ones(2),
        "final_vp_head.0.weight": torch.ones(2),
        "q_head.0.weight": torch.ones(2),
        "state_norm.weight": torch.ones(2),
    }
    for name, value in (overrides or {}).items():
        if value is None:
            model.pop(name, None)
        else:
            model[name] = value
    static = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return {
        "policy_type": "entity_graph",
        "config": {
            "__config_dataclass__": "EntityGraphConfig",
            "__config_schema__": 1,
            "fields": {
                "action_size": 16,
                "static_action_feature_size": 3,
                "hidden_size": config_width,
                "state_layers": 2,
            },
        },
        "action_mask_version": "v1",
        "mask_hidden_info": True,
        "soft_target_source": "policy",
        "static_action_features_sha256": "fixture-static",
        "static_action_features": static,
        "model": model,
    }


def _save(path: Path, payload) -> None:
    torch.save(payload, path)


def test_audit_attributes_energy_and_reports_provenance(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    _save(baseline_path, _checkpoint())
    _save(
        candidate_path,
        _checkpoint(
            overrides={
                "blocks.0.attn.in_proj_weight": torch.full((2,), 2.0),
                "action_encoder.0.weight": torch.full((2,), 3.0),
            }
        ),
    )

    report = audit_checkpoints(baseline_path, candidate_path, top_tensors=3)

    assert report["thresholds"] is None
    assert report["global"]["parameter_count"] == 16
    assert report["global"]["delta_energy"] == pytest.approx(10.0)
    assert report["global"]["relative_l2"] == pytest.approx((10.0 / 16.0) ** 0.5)
    assert report["groups"]["transformer_block_000"][
        "delta_energy_share"
    ] == pytest.approx(0.2)
    assert report["groups"]["policy"]["delta_energy_share"] == pytest.approx(0.8)
    assert report["groups"]["input_encoders"]["parameter_count"] == 2
    assert report["groups"]["value"]["parameter_count"] == 2
    assert report["groups"]["final_vp"]["parameter_count"] == 2
    assert report["groups"]["q"]["parameter_count"] == 2
    assert report["groups"]["shared"]["parameter_count"] == 2
    assert (
        report["top_tensor_outliers"]["by_delta_energy"][0]["name"]
        == "action_encoder.0.weight"
    )
    assert (
        report["baseline"]["sha256"]
        == "sha256:" + hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    )
    assert report["candidate"]["checkpoint_metadata"]["soft_target_source"] == "policy"


def test_audit_accepts_real_tiny_entity_graph_checkpoints(tmp_path: Path) -> None:
    config = EntityGraphConfig(
        action_size=16,
        static_action_feature_size=3,
        hidden_size=8,
        state_layers=2,
        attention_heads=2,
        dropout=0.0,
        action_mask_version="v1",
    )
    static = np.zeros((16, 3), dtype=np.float32)
    baseline = EntityGraphPolicy(config, static, seed=7, device="cpu")
    candidate = EntityGraphPolicy(config, static, seed=7, device="cpu")
    with torch.no_grad():
        candidate.model.blocks[1].ff[0].weight.add_(0.01)
        candidate.model.value_head[0].weight.add_(0.02)
    baseline_path = tmp_path / "real_baseline.pt"
    candidate_path = tmp_path / "real_candidate.pt"
    baseline.save(baseline_path, mask_hidden_info=True, soft_target_source="policy")
    candidate.save(candidate_path, mask_hidden_info=True, soft_target_source="policy")

    report = audit_checkpoints(baseline_path, candidate_path)
    assert report["groups"]["transformer_block_001"]["delta_energy"] > 0.0
    assert report["groups"]["value"]["delta_energy"] > 0.0
    assert report["groups"]["transformer_block_000"]["delta_energy"] == 0.0


def test_zero_baseline_norm_has_explicitly_undefined_relative_metrics(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    _save(
        baseline_path,
        _checkpoint(overrides={"value_head.0.weight": torch.zeros(2)}),
    )
    _save(
        candidate_path,
        _checkpoint(overrides={"value_head.0.weight": torch.ones(2)}),
    )
    report = audit_checkpoints(baseline_path, candidate_path)
    tensor = next(
        item
        for item in report["top_tensor_outliers"]["by_delta_energy"]
        if item["name"] == "value_head.0.weight"
    )
    assert tensor["relative_l2"] is None
    assert tensor["cosine_similarity"] is None


def test_legacy_and_durable_config_representations_compare_by_effective_architecture(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    baseline = _checkpoint()
    baseline["config"] = EntityGraphConfig(
        action_size=16,
        static_action_feature_size=3,
        hidden_size=8,
        state_layers=2,
    )
    _save(baseline_path, baseline)
    _save(candidate_path, _checkpoint())
    report = audit_checkpoints(baseline_path, candidate_path)
    assert report["compatibility"]["exact_architecture_metadata"] is True


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (_checkpoint(config_width=16), "architecture metadata differs"),
        (_checkpoint(overrides={"q_head.0.weight": None}), "state_dict keys differ"),
        (
            _checkpoint(overrides={"q_head.0.weight": torch.ones(3)}),
            "shape mismatch",
        ),
        (
            _checkpoint(
                overrides={"q_head.0.weight": torch.ones(2, dtype=torch.float64)}
            ),
            "dtype mismatch",
        ),
    ],
)
def test_audit_refuses_incompatible_checkpoints(
    tmp_path: Path, candidate, message: str
) -> None:
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    _save(baseline_path, _checkpoint())
    _save(candidate_path, candidate)
    with pytest.raises(DriftAuditError, match=message):
        audit_checkpoints(baseline_path, candidate_path)


def test_audit_refuses_static_feature_or_policy_type_mismatch(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    _save(baseline_path, _checkpoint())
    candidate = _checkpoint()
    candidate["static_action_features"] = torch.zeros_like(
        candidate["static_action_features"]
    )
    _save(candidate_path, candidate)
    with pytest.raises(DriftAuditError, match="architecture metadata differs"):
        audit_checkpoints(baseline_path, candidate_path)

    candidate = _checkpoint()
    candidate["policy_type"] = "xdim_graph"
    _save(candidate_path, candidate)
    with pytest.raises(DriftAuditError, match="policy_type='entity_graph'"):
        audit_checkpoints(baseline_path, candidate_path)


def test_audit_refuses_changed_non_floating_state(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    _save(
        baseline_path,
        _checkpoint(overrides={"counter": torch.tensor(1, dtype=torch.int64)}),
    )
    _save(
        candidate_path,
        _checkpoint(overrides={"counter": torch.tensor(2, dtype=torch.int64)}),
    )
    with pytest.raises(DriftAuditError, match="non-floating state tensor"):
        audit_checkpoints(baseline_path, candidate_path)


def test_audit_refuses_non_finite_parameters(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    _save(baseline_path, _checkpoint())
    _save(
        candidate_path,
        _checkpoint(
            overrides={"value_head.0.weight": torch.tensor([1.0, float("nan")])}
        ),
    )
    with pytest.raises(DriftAuditError, match="non-finite value"):
        audit_checkpoints(baseline_path, candidate_path)


def test_cli_writes_json_without_modifying_checkpoints(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    output = tmp_path / "drift.json"
    _save(baseline_path, _checkpoint())
    _save(candidate_path, _checkpoint())
    before = (baseline_path.read_bytes(), candidate_path.read_bytes())

    main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(output),
            "--top-tensors",
            "2",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "entity-graph-checkpoint-layer-drift-v1"
    assert report["global"]["delta_energy"] == 0.0
    assert before == (baseline_path.read_bytes(), candidate_path.read_bytes())
    with pytest.raises(DriftAuditError, match="already exists"):
        main(
            [
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--output",
                str(output),
            ]
        )
