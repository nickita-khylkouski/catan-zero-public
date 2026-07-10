from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest

from tools import a1_promotion_transaction as promotion
from tools.champion_registry import ChampionRegistry


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _contract(*, n_full: int = 128, n_full_wide=None) -> dict:
    recipe = {"world_size": 1, "optimizer": "adam", "mask_hidden_info": True}
    return {
        "contract_sha256": "sha256:" + "a" * 64,
        "science": {
            "search_operator": {
                "n_full": n_full,
                "n_full_wide": n_full_wide,
                "wide_roots_always_full": n_full_wide is not None,
            },
            "learner_training_recipe": recipe,
            "learner_training_recipe_sha256": promotion._digest_value(recipe),
        },
    }


def _fixture(tmp_path: Path, *, promotion_count: int = 0, n_full: int = 128) -> dict:
    champion = tmp_path / "champion.pt"
    candidate = tmp_path / "candidate.pt"
    champion.write_bytes(b"incumbent checkpoint")
    candidate.write_bytes(b"candidate checkpoint")
    registry_path = tmp_path / "registry.json"
    registry = ChampionRegistry(registry_path)
    registry.set_role(
        "generator_champion",
        champion,
        expected_md5=promotion._md5(champion),
        version=4,
        reason="fixture",
    )
    registry.set_role(
        "public_champion",
        champion,
        expected_md5=promotion._md5(champion),
        version=4,
        reason="fixture",
    )
    for _ in range(promotion_count):
        registry.record_promotion()
    registry.save()
    pointer = tmp_path / "CURRENT_CHAMPION"
    pointer.write_text(str(champion.resolve()) + "\n", encoding="utf-8")
    contract_path = tmp_path / "contract.lock.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    contract = _contract(n_full=n_full)
    report_path = tmp_path / "report.json"
    _write_json(
        report_path,
        {
            "a1_contract_sha256": contract["contract_sha256"],
            "a1_learner_training_recipe_sha256": contract["science"][
                "learner_training_recipe_sha256"
            ],
            "a1_bound_learner_training_recipe": contract["science"][
                "learner_training_recipe"
            ],
            "arch": "entity_graph",
            "mask_hidden_info": True,
            "track": "2p_no_trade",
            "vps_to_win": 10,
            "steps_completed": 7,
            "epochs": 1,
        },
    )
    evidence = []
    for kind in sorted(promotion.REQUIRED_EVIDENCE_KINDS):
        evidence_path = tmp_path / f"{kind}.json"
        _write_json(evidence_path, {"kind": kind, "passed": True})
        evidence.append(
            {"kind": kind, "path": str(evidence_path), "sha256": promotion._sha256(evidence_path)}
        )
    next_count = promotion_count + 1
    nth_required = next_count % 3 == 0
    adjudication = {
        "schema_version": promotion.ADJUDICATION_SCHEMA,
        "passed": True,
        "decision": "promote",
        "contract_sha256": contract["contract_sha256"],
        "candidate": {
            "path": str(candidate),
            "sha256": promotion._sha256(candidate),
            "version": 5,
            "training_report": {
                "path": str(report_path),
                "sha256": promotion._sha256(report_path),
            },
        },
        "champion": {
            "path": str(champion),
            "sha256": promotion._sha256(champion),
            "version": 4,
        },
        "checks": {name: True for name in promotion.REQUIRED_CHECKS},
        "nth_confirmation_required": nth_required,
        "nth_confirmation_passed": True if nth_required else False,
        "evidence": evidence,
    }
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    adjudication_path = tmp_path / "adjudication.json"
    _write_json(adjudication_path, adjudication)
    return {
        "champion": champion,
        "candidate": candidate,
        "registry": registry_path,
        "pointer": pointer,
        "contract_path": contract_path,
        "contract": contract,
        "adjudication": adjudication_path,
        "report": report_path,
        "receipt": tmp_path / "promotion.receipt.json",
        "lock": tmp_path / "promotion.lock",
    }


def _verify(fixture: dict):
    def verify(path: Path, *, require_all_job_claims: bool = False):
        assert path == fixture["contract_path"]
        assert require_all_job_claims is True
        return fixture["contract"]

    return verify


def _execute(fixture: dict, *, go: bool):
    return promotion.execute_promotion(
        registry_path=fixture["registry"],
        current_pointer=fixture["pointer"],
        contract_lock=fixture["contract_path"],
        adjudication_path=fixture["adjudication"],
        receipt_path=fixture["receipt"],
        reason="A1 typed promotion",
        lock_path=fixture["lock"],
        go=go,
        verify_lock_fn=_verify(fixture),
    )


def test_dry_run_is_read_only_and_attests_global_n128(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before_registry = fixture["registry"].read_bytes()
    before_pointer = fixture["pointer"].read_bytes()

    result = _execute(fixture, go=False)

    assert result["status"] == "dry_run"
    assert result["contract"]["n_full"] == 128
    assert result["contract"]["n_full_wide"] is None
    assert result["fleet_ckpt_updated"] is False
    assert fixture["registry"].read_bytes() == before_registry
    assert fixture["pointer"].read_bytes() == before_pointer
    assert not fixture["receipt"].exists()


def test_go_updates_generator_and_pointer_with_committed_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    public_before = ChampionRegistry.load(fixture["registry"]).get_role("public_champion")

    receipt = _execute(fixture, go=True)

    assert receipt["status"] == "committed"
    assert receipt["fleet_ckpt_updated"] is False
    registry = ChampionRegistry.load(fixture["registry"])
    generator = registry.get_role("generator_champion")
    assert generator is not None
    assert Path(generator.checkpoint_path).resolve() == fixture["candidate"].resolve()
    assert generator.version == 5
    assert registry.promotion_count() == 1
    assert any(
        Path(entry.checkpoint_path).resolve() == fixture["champion"].resolve()
        for entry in registry.opponent_pool()
    )
    assert registry.get_role("public_champion") == public_before
    assert fixture["pointer"].read_text().strip() == str(fixture["candidate"].resolve())
    saved = json.loads(fixture["receipt"].read_text())
    assert saved["status"] == "committed"
    assert Path(saved["rollback"]["registry_backup"]).is_file()
    assert Path(saved["rollback"]["current_backup"]).is_file()


def test_recovery_is_dry_run_then_restores_exact_before_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registry_before = fixture["registry"].read_bytes()
    pointer_before = fixture["pointer"].read_bytes()
    _execute(fixture, go=True)

    dry = promotion.recover_transaction(
        receipt_path=fixture["receipt"], lock_path=fixture["lock"], go=False
    )
    assert dry["status"] == "recovery_dry_run"
    assert fixture["registry"].read_bytes() != registry_before

    recovered = promotion.recover_transaction(
        receipt_path=fixture["receipt"], lock_path=fixture["lock"], go=True
    )
    assert recovered["status"] == "recovered"
    assert fixture["registry"].read_bytes() == registry_before
    assert fixture["pointer"].read_bytes() == pointer_before
    assert json.loads(fixture["receipt"].read_text())["status"] == "recovered"


def test_global_n196_contract_is_rejected_before_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, n_full=196)
    before = fixture["registry"].read_bytes()

    with pytest.raises(promotion.PromotionError, match="n_full=128"):
        _execute(fixture, go=True)

    assert fixture["registry"].read_bytes() == before
    assert not fixture["receipt"].exists()


def test_candidate_hash_drift_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["candidate"].write_bytes(b"mutated after adjudication")

    with pytest.raises(promotion.PromotionError, match="artifact drift"):
        _execute(fixture, go=False)


def test_every_third_confirmation_is_derived_from_registry(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, promotion_count=2)
    payload = json.loads(fixture["adjudication"].read_text())
    payload["nth_confirmation_required"] = False
    payload["nth_confirmation_passed"] = False
    payload.pop("adjudication_sha256")
    payload["adjudication_sha256"] = promotion._digest_value(payload)
    _write_json(fixture["adjudication"], payload)

    with pytest.raises(promotion.PromotionError, match="every-third"):
        _execute(fixture, go=False)


def test_exclusive_lock_refuses_a_second_writer(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    descriptor = os.open(fixture["lock"], os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(promotion.PromotionError, match="already held"):
            _execute(fixture, go=False)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_failed_second_replace_rolls_registry_and_pointer_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    registry_before = fixture["registry"].read_bytes()
    pointer_before = fixture["pointer"].read_bytes()
    real_write = promotion._atomic_write_bytes
    failed = False

    def fail_once(path: Path, data: bytes) -> None:
        nonlocal failed
        if path == fixture["pointer"] and not failed and data != pointer_before:
            failed = True
            raise OSError("synthetic pointer replace failure")
        real_write(path, data)

    monkeypatch.setattr(promotion, "_atomic_write_bytes", fail_once)
    with pytest.raises(promotion.PromotionError, match="original.*restored"):
        _execute(fixture, go=True)

    assert fixture["registry"].read_bytes() == registry_before
    assert fixture["pointer"].read_bytes() == pointer_before
    assert json.loads(fixture["receipt"].read_text())["status"] == "rolled_back"
