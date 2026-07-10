from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_registry_bootstrap as bootstrap
from tools.champion_registry import ChampionRegistry


def _write(path: Path, data: bytes) -> dict[str, str]:
    path.write_bytes(data)
    return {"path": str(path.resolve()), "sha256": bootstrap._sha256(path)}


def _fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    incumbent = tmp_path / "gen3.pt"
    history = tmp_path / "gen2a.pt"
    hard = tmp_path / "gen4.pt"
    report = tmp_path / "gen3-report.json"
    incumbent_ref = _write(incumbent, b"gen3")
    history_ref = _write(history, b"gen2a")
    hard_ref = _write(hard, b"gen4")
    report_ref = _write(report, b"{}\n")
    lock = {
        "contract_sha256": "sha256:" + "a" * 64,
        "checkpoints": [
            {
                **incumbent_ref,
                "role": "producer",
                "metadata": {
                    "legacy_scalar_readout_attestation": {"report": report_ref}
                },
            },
            {**history_ref, "role": "history"},
            {**hard_ref, "role": "hard_negative"},
        ],
    }
    lock_path = tmp_path / "contract.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return lock, lock_path, incumbent


def _verify(lock: dict):
    def verify(_path: Path, *, require_all_job_claims: bool):
        assert require_all_job_claims is True
        return lock

    return verify


def test_dry_run_is_read_only_and_binds_new_a1_lineage(tmp_path: Path) -> None:
    lock, lock_path, incumbent = _fixture(tmp_path)
    registry = tmp_path / "registry.json"
    pointer = tmp_path / "CURRENT_CHAMPION"
    receipt = tmp_path / "bootstrap.json"

    plan = bootstrap.build_plan(
        lock_path=lock_path,
        registry_path=registry,
        pointer_path=pointer,
        receipt_path=receipt,
        incumbent=incumbent,
        verify_lock_fn=_verify(lock),
    )

    assert plan["lineage"] == {
        "name": "a1",
        "promotion_count": 0,
        "basis": "new_registry_lineage_no_persisted_pre_a1_registry",
    }
    assert plan["incumbent"]["sha256"] == bootstrap._sha256(incumbent)
    assert len(plan["opponent_pool"]) == 2
    assert not registry.exists() and not pointer.exists() and not receipt.exists()


def test_commit_publishes_roles_pool_pointer_and_receipt_once(tmp_path: Path) -> None:
    lock, lock_path, incumbent = _fixture(tmp_path)
    registry_path = tmp_path / "registry.json"
    pointer = tmp_path / "CURRENT_CHAMPION"
    receipt_path = tmp_path / "bootstrap.json"
    plan = bootstrap.build_plan(
        lock_path=lock_path,
        registry_path=registry_path,
        pointer_path=pointer,
        receipt_path=receipt_path,
        incumbent=incumbent,
        verify_lock_fn=_verify(lock),
    )

    receipt = bootstrap.commit(plan)
    registry = ChampionRegistry.load(registry_path)

    assert receipt["mode"] == "committed"
    assert registry.promotion_count("generator_champion") == 0
    assert set(registry.roles()) == {
        "generator_champion",
        "public_champion",
        "tournament_bot",
    }
    assert all(
        pointer_value.checkpoint_path == str(incumbent.resolve())
        for pointer_value in registry.roles().values()
    )
    assert len(registry.opponent_pool()) == 2
    assert pointer.read_text() == str(incumbent.resolve()) + "\n"
    assert receipt_path.stat().st_mode & 0o222 == 0
    journal = Path(plan["destinations"]["prepared_journal"])
    assert journal.is_file() and journal.stat().st_mode & 0o222 == 0
    assert bootstrap.commit(plan) == receipt


def test_refuses_nonproducer_incumbent_and_report_drift(tmp_path: Path) -> None:
    lock, lock_path, incumbent = _fixture(tmp_path)
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"wrong")
    kwargs = {
        "lock_path": lock_path,
        "registry_path": tmp_path / "registry.json",
        "pointer_path": tmp_path / "CURRENT_CHAMPION",
        "receipt_path": tmp_path / "receipt.json",
        "verify_lock_fn": _verify(lock),
    }
    with pytest.raises(bootstrap.BootstrapError, match="not the sealed A1 producer"):
        bootstrap.build_plan(incumbent=wrong, **kwargs)

    report_path = Path(
        lock["checkpoints"][0]["metadata"]["legacy_scalar_readout_attestation"][
            "report"
        ]["path"]
    )
    report_path.write_bytes(b"drift")
    with pytest.raises(bootstrap.BootstrapError, match="training report hash drift"):
        bootstrap.build_plan(incumbent=incumbent, **kwargs)


def test_refuses_preexisting_destination(tmp_path: Path) -> None:
    lock, lock_path, incumbent = _fixture(tmp_path)
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    with pytest.raises(bootstrap.BootstrapError, match="non-fresh registry"):
        bootstrap.build_plan(
            lock_path=lock_path,
            registry_path=registry,
            pointer_path=tmp_path / "CURRENT_CHAMPION",
            receipt_path=tmp_path / "receipt.json",
            incumbent=incumbent,
            verify_lock_fn=_verify(lock),
        )


def test_commit_resumes_after_hard_kill_between_registry_and_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, lock_path, incumbent = _fixture(tmp_path)
    registry = tmp_path / "registry.json"
    pointer = tmp_path / "CURRENT_CHAMPION"
    receipt = tmp_path / "bootstrap.json"
    plan = bootstrap.build_plan(
        lock_path=lock_path,
        registry_path=registry,
        pointer_path=pointer,
        receipt_path=receipt,
        incumbent=incumbent,
        verify_lock_fn=_verify(lock),
    )
    original = bootstrap._publish_exact

    def kill_at_pointer(path: Path, payload: bytes, **kwargs) -> None:
        if path == pointer:
            raise KeyboardInterrupt("simulated hard kill")
        original(path, payload, **kwargs)

    monkeypatch.setattr(bootstrap, "_publish_exact", kill_at_pointer)
    with pytest.raises(KeyboardInterrupt, match="hard kill"):
        bootstrap.commit(plan)
    journal = Path(plan["destinations"]["prepared_journal"])
    assert journal.is_file()
    assert registry.is_file()
    assert not pointer.exists() and not receipt.exists()

    monkeypatch.setattr(bootstrap, "_publish_exact", original)
    committed = bootstrap.commit(plan)
    assert committed["mode"] == "committed"
    assert pointer.read_text() == str(incumbent.resolve()) + "\n"
    assert receipt.is_file()


def test_resume_from_prepared_journal_rejects_unknown_partial_bytes(
    tmp_path: Path,
) -> None:
    lock, lock_path, incumbent = _fixture(tmp_path)
    registry = tmp_path / "registry.json"
    pointer = tmp_path / "CURRENT_CHAMPION"
    receipt = tmp_path / "bootstrap.json"
    plan = bootstrap.build_plan(
        lock_path=lock_path,
        registry_path=registry,
        pointer_path=pointer,
        receipt_path=receipt,
        incumbent=incumbent,
        verify_lock_fn=_verify(lock),
    )
    journal = bootstrap._journal_payload(plan)
    journal_path = Path(plan["destinations"]["prepared_journal"])
    bootstrap._write_exclusive(
        journal_path,
        json.dumps(journal, indent=2, sort_keys=True).encode() + b"\n",
        mode=0o444,
    )
    pointer.write_text("attacker-controlled\n", encoding="utf-8")
    pointer.chmod(0o600)

    with pytest.raises(bootstrap.BootstrapError, match="differs from prepared bytes"):
        bootstrap.commit(plan)
    assert pointer.read_text() == "attacker-controlled\n"
    assert not receipt.exists()


def test_commit_resumes_after_hard_kill_before_committed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, lock_path, incumbent = _fixture(tmp_path)
    registry = tmp_path / "registry.json"
    pointer = tmp_path / "CURRENT_CHAMPION"
    receipt = tmp_path / "bootstrap.json"
    plan = bootstrap.build_plan(
        lock_path=lock_path,
        registry_path=registry,
        pointer_path=pointer,
        receipt_path=receipt,
        incumbent=incumbent,
        verify_lock_fn=_verify(lock),
    )
    original = bootstrap._publish_exact

    def kill_at_receipt(path: Path, payload: bytes, **kwargs) -> None:
        if path == receipt:
            raise KeyboardInterrupt("simulated receipt kill")
        original(path, payload, **kwargs)

    monkeypatch.setattr(bootstrap, "_publish_exact", kill_at_receipt)
    with pytest.raises(KeyboardInterrupt, match="receipt kill"):
        bootstrap.commit(plan)
    assert registry.is_file() and pointer.is_file()
    assert not receipt.exists()

    monkeypatch.setattr(bootstrap, "_publish_exact", original)
    committed = bootstrap.commit(plan)
    assert committed["mode"] == "committed"
    assert receipt.is_file()


def test_resume_plan_from_journal_binds_original_arguments(tmp_path: Path) -> None:
    lock, lock_path, incumbent = _fixture(tmp_path)
    registry = tmp_path / "registry.json"
    pointer = tmp_path / "CURRENT_CHAMPION"
    receipt = tmp_path / "bootstrap.json"
    plan = bootstrap.build_plan(
        lock_path=lock_path,
        registry_path=registry,
        pointer_path=pointer,
        receipt_path=receipt,
        incumbent=incumbent,
        verify_lock_fn=_verify(lock),
    )
    journal = bootstrap._journal_payload(plan)
    journal_path = Path(plan["destinations"]["prepared_journal"])
    bootstrap._write_exclusive(
        journal_path,
        json.dumps(journal, indent=2, sort_keys=True).encode() + b"\n",
        mode=0o444,
    )

    resumed = bootstrap._resume_plan_from_journal(
        journal_path,
        lock_path=lock_path,
        registry_path=registry,
        pointer_path=pointer,
        receipt_path=receipt,
        incumbent=incumbent,
    )
    assert resumed == plan
    with pytest.raises(bootstrap.BootstrapError, match="destinations"):
        bootstrap._resume_plan_from_journal(
            journal_path,
            lock_path=lock_path,
            registry_path=tmp_path / "other.json",
            pointer_path=pointer,
            receipt_path=receipt,
            incumbent=incumbent,
        )
