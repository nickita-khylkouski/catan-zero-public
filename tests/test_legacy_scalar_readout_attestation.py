from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import torch

from catan_zero.rl.entity_token_policy import EntityGraphConfig
from tools import legacy_scalar_readout_attestation as legacy


def _legacy_checkpoint(
    path: Path, *, categorical_bins: int = 0, marker: int = 1
) -> Path:
    """Small fixture with the same top-level/config shape as the real gen3 artifact."""

    torch.save(
        {
            "policy_type": "entity_graph",
            "config": EntityGraphConfig(
                action_size=290,
                static_action_feature_size=16,
                hidden_size=640,
                state_layers=6,
                attention_heads=8,
                dropout=0.05,
                value_categorical_bins=categorical_bins,
            ),
            "mask_hidden_info": True,
            "model": {"value_head.2.bias": torch.tensor([float(marker)])},
            "marker": marker,
            # Intentionally no value_training: this is the compatibility case.
        },
        path,
    )
    return path


def _legacy_report(
    path: Path,
    checkpoint: Path,
    *,
    declared_checkpoint: str | None = None,
    value_head_type: str = "scalar",
) -> Path:
    # This mirrors train_bc's completed legacy report shape: top-level recipe
    # plus a list of epoch metrics with nested validation telemetry.
    payload = {
        "arch": "entity_graph",
        "checkpoint": declared_checkpoint or str(checkpoint),
        "mask_hidden_info": True,
        "epochs": 2,
        "steps_completed": 812,
        "value_loss_weight": 0.25,
        "value_head_type": value_head_type,
        "value_categorical_loss_weight": 0.0,
        "metrics": [
            {
                "epoch": 1,
                "loss": 1.0,
                "policy_loss": 0.5,
                "value_loss": 0.72,
                "validation": {"value_loss": 0.74, "policy_loss": 0.51},
            },
            {
                "epoch": 2,
                "loss": 0.9,
                "policy_loss": 0.45,
                "value_loss": 0.65,
                "validation": {"value_loss": 0.68, "policy_loss": 0.47},
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_real_shape_legacy_pair_creates_readonly_typed_attestation(
    tmp_path: Path,
) -> None:
    checkpoint = _legacy_checkpoint(tmp_path / "checkpoint.pt")
    report = _legacy_report(
        tmp_path / "report.json",
        checkpoint,
        declared_checkpoint="runs/bc/gen3_20260706/checkpoint.pt",
    )
    # Recreate the real suffix expected by the historical repo-relative report.
    real_dir = tmp_path / "runs" / "bc" / "gen3_20260706"
    real_dir.mkdir(parents=True)
    real_checkpoint = real_dir / "checkpoint.pt"
    checkpoint.replace(real_checkpoint)
    report_payload = json.loads(report.read_text())
    report_payload["checkpoint"] = "runs/bc/gen3_20260706/checkpoint.pt"
    report.write_text(json.dumps(report_payload), encoding="utf-8")

    output = tmp_path / "legacy_scalar.attestation.json"
    payload = legacy.write_attestation(real_checkpoint, report, output)
    verified = legacy.verify_attestation(
        output,
        expected_checkpoint_path=real_checkpoint,
        expected_checkpoint_sha256=payload["checkpoint"]["sha256"],
    )

    assert verified["schema_version"] == legacy.SCHEMA_VERSION
    assert verified["claims"]["value_readout"] == "scalar"
    assert verified["claims"]["checkpoint_value_categorical_bins"] == 0
    assert len(verified["claims"]["value_loss_telemetry"]) == 2
    assert output.stat().st_mode & stat.S_IWUSR == 0
    with pytest.raises(legacy.AttestationError, match="refusing to overwrite"):
        legacy.write_attestation(real_checkpoint, report, output)


def test_attestation_rejects_wrong_report_checkpoint_identity(tmp_path: Path) -> None:
    checkpoint = _legacy_checkpoint(tmp_path / "checkpoint.pt")
    wrong = _legacy_checkpoint(tmp_path / "other.pt", marker=2)
    report = _legacy_report(
        tmp_path / "report.json", checkpoint, declared_checkpoint=str(wrong)
    )
    with pytest.raises(legacy.AttestationError, match="checkpoint identity mismatch"):
        legacy.build_attestation(checkpoint, report)


def test_attestation_verification_rejects_bound_report_tamper(tmp_path: Path) -> None:
    checkpoint = _legacy_checkpoint(tmp_path / "checkpoint.pt")
    report = _legacy_report(tmp_path / "report.json", checkpoint)
    output = tmp_path / "attestation.json"
    legacy.write_attestation(checkpoint, report, output)

    payload = json.loads(report.read_text())
    payload["metrics"][0]["value_loss"] = 999.0
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(legacy.AttestationError, match="report hash drift"):
        legacy.verify_attestation(output)


def test_attestation_rejects_semantic_tamper_even_with_rehashed_envelope(
    tmp_path: Path,
) -> None:
    checkpoint = _legacy_checkpoint(tmp_path / "checkpoint.pt")
    report = _legacy_report(tmp_path / "report.json", checkpoint)
    output = tmp_path / "attestation.json"
    legacy.write_attestation(checkpoint, report, output)

    output.chmod(0o600)
    payload = json.loads(output.read_text())
    payload["claims"]["report_value_loss_weight"] = 99.0
    unhashed = dict(payload)
    unhashed.pop("attestation_sha256")
    payload["attestation_sha256"] = legacy._digest_value(unhashed)
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        legacy.AttestationError, match="semantic claims do not reconstruct"
    ):
        legacy.verify_attestation(output)


@pytest.mark.parametrize(
    ("categorical_bins", "head_type", "message"),
    [
        (33, "scalar", "no categorical head"),
        (0, "hlgauss", "contradictory categorical objective"),
    ],
)
def test_attestation_cannot_authorize_categorical_training(
    tmp_path: Path, categorical_bins: int, head_type: str, message: str
) -> None:
    checkpoint = _legacy_checkpoint(
        tmp_path / "checkpoint.pt", categorical_bins=categorical_bins
    )
    report = _legacy_report(
        tmp_path / "report.json", checkpoint, value_head_type=head_type
    )
    with pytest.raises(legacy.AttestationError, match=message):
        legacy.build_attestation(checkpoint, report)


def test_attestation_requires_positive_scalar_weight_and_real_telemetry(
    tmp_path: Path,
) -> None:
    checkpoint = _legacy_checkpoint(tmp_path / "checkpoint.pt")
    report = _legacy_report(tmp_path / "report.json", checkpoint)
    payload = json.loads(report.read_text())
    payload["value_loss_weight"] = 0.0
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(legacy.AttestationError, match="positive finite"):
        legacy.build_attestation(checkpoint, report)

    payload["value_loss_weight"] = 0.25
    payload["metrics"][0].pop("value_loss")
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(legacy.AttestationError, match="actual value_loss telemetry"):
        legacy.build_attestation(checkpoint, report)


def test_attestation_rejects_contradictory_resolved_scalar_weight(
    tmp_path: Path,
) -> None:
    checkpoint = _legacy_checkpoint(tmp_path / "checkpoint.pt")
    report = _legacy_report(tmp_path / "report.json", checkpoint)
    payload = json.loads(report.read_text())
    payload["resolved_scalar_value_loss_weight"] = 0.0
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(legacy.AttestationError, match="non-positive resolved scalar"):
        legacy.build_attestation(checkpoint, report)
