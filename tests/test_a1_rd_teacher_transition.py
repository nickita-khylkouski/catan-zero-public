from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_zero.rl.entity_feature_adapter import RUST_ENTITY_ADAPTER_V6
from catan_zero.rl.meaningful_history import MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
from tools import a1_rd_teacher_transition as transition
from tools import a1_target_eligibility_inventory as inventory


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    typed_limit: int,
    checkpoint_history: tuple[bool, int, str, str, bool] = (
        True,
        64,
        MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
        "ordered_attention_v2",
        True,
    ),
) -> dict[str, object]:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    base = _write_json(
        tmp_path / "contract.json",
        {
            "operator": {
                "meaningful_public_history": True,
                "event_history_limit": 64,
            },
            "target_information_regime": "public_belief_single_tree_v1",
        },
    )
    typed = _write_json(
        tmp_path / "typed.json",
        {
            "pipeline": "generate",
            "schema_version": 13,
            "fields": {
                "meaningful_public_history": True,
                "event_history_limit": typed_limit,
                "teacher_entity_feature_adapter_version": RUST_ENTITY_ADAPTER_V6,
                "learner_entity_feature_adapter_version": RUST_ENTITY_ADAPTER_V6,
            },
        },
    )
    monkeypatch.setattr(
        transition.inventory,
        "inspect_rd_contract",
        lambda _path: {"contract_sha256": "sha256:" + "1" * 64},
    )
    monkeypatch.setattr(
        transition.train_bc,
        "_checkpoint_entity_feature_adapter_version",
        lambda _path: RUST_ENTITY_ADAPTER_V6,
    )
    monkeypatch.setattr(
        transition.train_bc,
        "_checkpoint_meaningful_public_history",
        lambda _path: checkpoint_history,
    )
    monkeypatch.setattr(transition.alignment, "_file_sha256", lambda _path: "sha256:" + "2" * 64)
    monkeypatch.setattr(
        transition.alignment,
        "_rd_teacher_transition_authority",
        lambda *_args, **_kwargs: {"status": "accepted"},
    )
    return transition.bind(
        checkpoint=checkpoint,
        base_operator_contract=base,
        typed_generation_config=typed,
        binding_id="test-v6-teacher",
        output=tmp_path / "binding.json",
    )


def test_bind_rejects_v6_history_limit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(
        transition.BindingError,
        match="checkpoint and typed generator history contracts differ",
    ):
        _bind(tmp_path, monkeypatch, typed_limit=32)


def test_bind_records_matching_v6_history_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _bind(tmp_path, monkeypatch, typed_limit=64)
    assert result["teacher_feature_contract"] == {
        "schema_version": "entity-feature-adapter-v1",
        "entity_feature_adapter_version": RUST_ENTITY_ADAPTER_V6,
        "meaningful_public_history": True,
        "meaningful_public_history_schema": MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
        "event_history_limit": 64,
    }


def test_v6_history64_operator_contract_is_self_consistent() -> None:
    contract = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "operations"
        / "a1-target-identity-coherent-n128-v6-history64-rd-v1"
        / "contract.json"
    )
    inspected = inventory.inspect_rd_contract(contract)
    assert (
        inspected["contract_id"]
        == "a1-v6-history64-coherent-n128-reanalysis-operator-20260717-r1"
    )
    typed = json.loads(
        Path(inspected["typed_generation_config"]["path"]).read_text(encoding="utf-8")
    )
    assert typed["fields"]["event_history_limit"] == 64
    assert (
        typed["fields"]["teacher_entity_feature_adapter_version"]
        == RUST_ENTITY_ADAPTER_V6
    )
    assert (
        typed["fields"]["learner_entity_feature_adapter_version"]
        == RUST_ENTITY_ADAPTER_V6
    )
