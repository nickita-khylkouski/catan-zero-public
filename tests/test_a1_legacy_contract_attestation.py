from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_pre_wave_contract as contract_tool
from tools import a1_promotion_transaction as promotion


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "historical.draft.json"
    _write(
        source,
        {
            "schema_version": contract_tool.LEGACY_DRAFT_SCHEMA,
            "contract_id": "historical-a1",
            "science": {},
        },
    )
    lock = tmp_path / "historical.lock.json"
    payload = {
        "schema_version": contract_tool.LEGACY_LOCK_SCHEMA,
        "contract_id": "historical-a1",
        "source_draft": {"path": str(source), "sha256": promotion._sha256(source)},
        "science": {
            "search_operator": {
                "n_full": 128,
                "n_full_wide": None,
                "wide_roots_always_full": False,
            }
        },
    }
    payload["contract_sha256"] = promotion._digest_value(payload)
    _write(lock, payload)
    receipt = tmp_path / "training.receipt.json"
    receipt_payload = {
        "schema_version": promotion.one_dose.RETRY_RECEIPT_SCHEMA,
        "status": "complete",
        "returncode": 0,
        "contract_sha256": payload["contract_sha256"],
        "lock": str(lock),
        "lock_file_sha256": promotion._sha256(lock),
    }
    receipt_payload["receipt_sha256"] = promotion._digest_value(receipt_payload)
    _write(receipt, receipt_payload)
    allowlist = {
        "contract_id": payload["contract_id"],
        "contract_sha256": payload["contract_sha256"],
        "lock_file_sha256": promotion._sha256(lock),
        "source_draft_sha256": promotion._sha256(source),
        "training_receipt_sha256": promotion._sha256(receipt),
        "training_receipt_digest": receipt_payload["receipt_sha256"],
    }
    monkeypatch.setattr(
        promotion, "HISTORICAL_MARKERLESS_A1_CONTRACT", allowlist
    )
    return {
        "source": source,
        "lock": lock,
        "lock_payload": payload,
        "receipt": receipt,
        "receipt_payload": receipt_payload,
    }


def _attestation(fixture: dict, tmp_path: Path) -> Path:
    value = promotion.build_legacy_contract_attestation(
        fixture["lock"], fixture["receipt"]
    )
    path = tmp_path / "legacy.attestation.json"
    _write(path, value)
    return path


def test_explicit_attestation_allows_only_promotion_contract_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    attestation = _attestation(fixture, tmp_path)

    verified = promotion._verify_contract(  # noqa: SLF001
        fixture["lock"], legacy_contract_attestation=attestation
    )
    assert verified["contract_sha256"] == fixture["lock_payload"]["contract_sha256"]

    with pytest.raises(promotion.PromotionError, match="sealed A1 contract"):
        promotion._verify_contract(fixture["lock"])  # noqa: SLF001
    with pytest.raises(contract_tool.ContractError, match="promotion handoff"):
        contract_tool.verify_lock(fixture["lock"])


@pytest.mark.parametrize("field", ["contract_id", "contract_sha256"])
def test_wrong_contract_identity_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    allowlist = dict(promotion.HISTORICAL_MARKERLESS_A1_CONTRACT)
    allowlist[field] = "wrong" if field == "contract_id" else "sha256:" + "0" * 64
    monkeypatch.setattr(promotion, "HISTORICAL_MARKERLESS_A1_CONTRACT", allowlist)
    with pytest.raises(promotion.PromotionError, match="not the allowlisted"):
        promotion.build_legacy_contract_attestation(
            fixture["lock"], fixture["receipt"]
        )


def test_wrong_lock_hash_or_receipt_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    allowlist = dict(promotion.HISTORICAL_MARKERLESS_A1_CONTRACT)
    allowlist["lock_file_sha256"] = "sha256:" + "1" * 64
    monkeypatch.setattr(promotion, "HISTORICAL_MARKERLESS_A1_CONTRACT", allowlist)
    with pytest.raises(promotion.PromotionError, match="not the allowlisted"):
        promotion.build_legacy_contract_attestation(
            fixture["lock"], fixture["receipt"]
        )

    fixture = _fixture(tmp_path / "second", monkeypatch)
    receipt = json.loads(fixture["receipt"].read_text())
    receipt["contract_sha256"] = "sha256:" + "2" * 64
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = promotion._digest_value(receipt)
    _write(fixture["receipt"], receipt)
    with pytest.raises(promotion.PromotionError, match="training receipt"):
        promotion.build_legacy_contract_attestation(
            fixture["lock"], fixture["receipt"]
        )


def test_markerless_new_contract_cannot_self_attest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    new_lock = tmp_path / "new.lock.json"
    value = dict(fixture["lock_payload"])
    value["contract_id"] = "new-markerless-contract"
    value.pop("contract_sha256")
    value["contract_sha256"] = promotion._digest_value(value)
    _write(new_lock, value)
    with pytest.raises(promotion.PromotionError, match="not the allowlisted"):
        promotion.build_legacy_contract_attestation(new_lock, fixture["receipt"])


def test_attestation_tamper_and_post_promotion_use_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    attestation = _attestation(fixture, tmp_path)
    value = json.loads(attestation.read_text())
    value["training_receipt"]["receipt_sha256"] = "sha256:" + "3" * 64
    value.pop("attestation_sha256")
    value["attestation_sha256"] = promotion._digest_value(value)
    _write(attestation, value)
    with pytest.raises(promotion.PromotionError, match="does not replay"):
        promotion._verify_contract(  # noqa: SLF001
            fixture["lock"], legacy_contract_attestation=attestation
        )

    # The compatibility input is not accepted by the generation verifier at
    # all, so it cannot authorize a current/post-promotion producer contract.
    with pytest.raises(contract_tool.ContractError, match="promotion handoff"):
        contract_tool.verify_lock(fixture["lock"])


def test_promotion_cannot_substitute_a_different_training_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    attestation = _attestation(fixture, tmp_path)
    other = tmp_path / "other.receipt.json"
    other.write_text(fixture["receipt"].read_text())
    with pytest.raises(promotion.PromotionError, match="different training receipt"):
        promotion._verify_contract(  # noqa: SLF001
            fixture["lock"],
            legacy_contract_attestation=attestation,
            expected_training_receipt=other,
        )


def test_contract_replacement_after_attestation_replay_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    attestation = _attestation(fixture, tmp_path)
    original = promotion.build_legacy_contract_attestation
    replaced = False

    def replace_after_replay(lock: Path, receipt: Path) -> dict:
        nonlocal replaced
        value = original(lock, receipt)
        if not replaced:
            replaced = True
            malicious = dict(fixture["lock_payload"])
            malicious["contract_id"] = "substituted-markerless-contract"
            malicious.pop("contract_sha256")
            malicious["contract_sha256"] = promotion._digest_value(malicious)
            replacement = tmp_path / "replacement.lock.json"
            _write(replacement, malicious)
            replacement.replace(lock)
        return value

    monkeypatch.setattr(
        promotion, "build_legacy_contract_attestation", replace_after_replay
    )
    with pytest.raises(promotion.PromotionError, match="changed after attestation replay"):
        promotion._verify_contract(  # noqa: SLF001
            fixture["lock"], legacy_contract_attestation=attestation
        )
