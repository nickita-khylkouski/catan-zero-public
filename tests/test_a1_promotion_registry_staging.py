"""Determinism contracts for promotion registry staging."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tools import a1_promotion_transaction as promotion
from tools.champion_registry import ChampionRegistry


def _checkpoint(path: Path, payload: bytes) -> dict:
    path.write_bytes(payload)
    return {
        "path": str(path),
        "md5": hashlib.md5(payload).hexdigest(),
        "version": 1,
        "agent_identity": {
            "agent_identity_sha256": "sha256:" + "a" * 64,
            "search_config": {"n_full": 128},
        },
    }


def test_replayed_promotion_staging_has_identical_registry_bytes(
    tmp_path: Path,
) -> None:
    champion = _checkpoint(tmp_path / "champion.pt", b"champion")
    candidate = _checkpoint(tmp_path / "candidate.pt", b"candidate")
    registry_path = tmp_path / "registry.json"
    registry = ChampionRegistry(registry_path)
    registry.set_role(
        "generator_champion",
        champion["path"],
        expected_md5=champion["md5"],
        version=0,
        _timestamp=1000.0,
    )
    registry.save()
    verified = {
        "champion": champion,
        "candidate": candidate,
        "candidate_lineage": None,
        "training_receipt": {
            "path": str(tmp_path / "training.receipt.json"),
            "sha256": "sha256:" + "b" * 64,
            "execution_binding_sha256": "sha256:" + "c" * 64,
        },
        "adjudication_sha256": "sha256:" + "d" * 64,
        "promotion_mode": "promotion_parent",
        "next_promotion_count": 1,
    }
    kwargs = {
        "verified": verified,
        "contract_sha256": "sha256:" + "e" * 64,
        "adjudication_path": tmp_path / "adjudication.json",
        "receipt_path": tmp_path / "promotion.receipt.json",
        "reason": "passed",
        "mutation_timestamp": 1234.5,
    }

    first, first_count = promotion._stage_registry(registry_path, **kwargs)  # noqa: SLF001
    second, second_count = promotion._stage_registry(registry_path, **kwargs)  # noqa: SLF001

    assert first_count == second_count == 1
    assert first == second
    assert promotion._sha256_bytes(first) == promotion._sha256_bytes(second)  # noqa: SLF001


def test_standalone_go_requires_and_parses_dry_run_timestamp(monkeypatch) -> None:
    common = [
        "promote",
        "--registry",
        "registry.json",
        "--current-pointer",
        "CURRENT_CHAMPION",
        "--contract-lock",
        "contract.json",
        "--adjudication",
        "adjudication.json",
        "--training-receipt",
        "training.json",
        "--cohort-exclusions",
        "cohorts.json",
        "--receipt",
        "promotion.json",
        "--reason",
        "passed",
        "--go",
    ]
    called = []
    monkeypatch.setattr(
        promotion,
        "execute_promotion",
        lambda **kwargs: called.append(kwargs) or {},
    )

    assert promotion.main(common) == 2
    assert called == []
    assert promotion.main(
        [*common, "--registry-mutation-timestamp", "1234.5"]
    ) == 0
    assert called[0]["registry_mutation_timestamp"] == 1234.5
