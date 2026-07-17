from __future__ import annotations

from pathlib import Path
import stat

import pytest

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
