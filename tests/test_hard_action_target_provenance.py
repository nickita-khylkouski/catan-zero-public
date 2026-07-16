from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import generate_teacher_data, train_bc
from tools.factory_common import (
    HARD_ACTION_SCOPE_PUBLIC,
    HARD_ACTION_TARGET_INFORMATION_SCHEMA,
    classical_teacher_hard_action_target_information,
    propagated_hard_action_target_information,
)


def _write_manifest(path: Path, contract: dict | None) -> None:
    path.mkdir()
    payload = {}
    if contract is not None:
        payload["hard_action_target_information"] = contract
    (path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _admit(path: Path, *, production: bool, acknowledged: bool) -> dict[str, object]:
    return train_bc._validate_hard_action_target_admission(  # noqa: SLF001
        {
            "action_taken": np.asarray([1, 2], dtype=np.int16),
            "policy_weight_multiplier": np.asarray([1.0, 0.0], dtype=np.float32),
        },
        path,
        mask_hidden_info=True,
        policy_loss_weight=1.0,
        train_value_only=False,
        production=production,
        acknowledged_authoritative_targets=acknowledged,
    )


def test_generator_declares_classical_hard_actions_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "raw"
    monkeypatch.setattr(generate_teacher_data, "make_named_policy", lambda _name: object())
    monkeypatch.setattr(
        generate_teacher_data,
        "parse_track",
        lambda *_args, **_kwargs: SimpleNamespace(players=2),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_teacher_data.py",
            "--games",
            "0",
            "--teachers",
            "test_teacher",
            "--out",
            str(out),
        ],
    )

    generate_teacher_data.main()

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hard_action_target_information"] == (
        classical_teacher_hard_action_target_information()
    )


def test_masked_diagnostic_replay_requires_explicit_acknowledgement(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_manifest(data, classical_teacher_hard_action_target_information())

    with pytest.raises(SystemExit, match="acknowledge-authoritative-hard-action-targets"):
        _admit(data, production=False, acknowledged=False)

    report = _admit(data, production=False, acknowledged=True)
    assert report["public_information_authenticated"] is False
    assert report["diagnostic_acknowledgement"] is True


def test_acknowledgement_cannot_override_masked_production(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_manifest(data, classical_teacher_hard_action_target_information())

    with pytest.raises(SystemExit, match="masked production training refused"):
        _admit(data, production=True, acknowledged=True)


def test_public_authenticated_hard_actions_pass_masked_production(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    public_contract = {
        "schema_version": HARD_ACTION_TARGET_INFORMATION_SCHEMA,
        "target_column": "action_taken",
        "information_scope": HARD_ACTION_SCOPE_PUBLIC,
        "public_information_authenticated": True,
        "producer": "public_information_set_teacher_v1",
    }
    _write_manifest(data, public_contract)

    report = _admit(data, production=True, acknowledged=False)
    assert report["public_information_authenticated"] is True


def test_value_only_training_does_not_require_hard_action_admission(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_manifest(data, classical_teacher_hard_action_target_information())

    report = train_bc._validate_hard_action_target_admission(  # noqa: SLF001
        {"action_taken": np.asarray([1], dtype=np.int16)},
        data,
        mask_hidden_info=True,
        policy_loss_weight=1.0,
        train_value_only=True,
        production=True,
        acknowledged_authoritative_targets=False,
    )
    assert report["hard_action_objective_active"] is False


def test_transformation_never_upgrades_missing_lineage_to_public() -> None:
    contract = propagated_hard_action_target_information(
        [{"manifest.json": {"track": "2p_no_trade"}}]
    )

    assert contract["information_scope"] == "unknown"
    assert contract["public_information_authenticated"] is False
