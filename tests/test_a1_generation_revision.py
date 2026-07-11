from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_pre_wave_contract as contract


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "configs/operations/a1-dual-arm-56gpu-20260710/contract.json"
)


def _arm_lock(path: Path, arm_id: str) -> dict[str, str]:
    value = {
        "schema_version": contract.GENERATION_ARM_LOCK_SCHEMA,
        "contract_id": f"issued-{arm_id}",
        "game_contract": {"arm_id": arm_id},
    }
    value["contract_sha256"] = contract._digest_value(value)  # noqa: SLF001
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return {
        "contract_id": value["contract_id"],
        "contract_sha256": value["contract_sha256"],
        "file_sha256": contract._sha256(path),  # noqa: SLF001
    }


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    locks = []
    expected = {}
    for arm_id in ("n128", "n256"):
        path = tmp_path / f"{arm_id}.lock.json"
        expected[arm_id] = _arm_lock(path, arm_id)
        locks.append(path)
    monkeypatch.setattr(contract, "GENERATION_CAMPAIGN_R1_LOCKS", expected)
    out = tmp_path / "revision.json"
    payload = contract.build_generation_campaign_revision(
        SOURCE,
        superseded_lock_paths=locks,
        contract_id="a1-dual-arm-n256-n128-56gpu-20260711-r2",
        output_root=tmp_path / "fresh-r2",
        out_path=out,
    )
    return out, payload


def test_revision_is_fresh_native_and_preserves_nonimplementation_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, payload = _build(tmp_path, monkeypatch)
    source = json.loads(SOURCE.read_text())
    arms = {arm["id"]: arm for arm in payload["arms"]}

    assert out.stat().st_mode & 0o222 == 0
    assert payload["schema_version"] == contract.GENERATION_CAMPAIGN_REVISION_SCHEMA
    assert payload["implementation_commit"] == contract.GENERATION_CAMPAIGN_REVISION_IMPLEMENTATION_COMMIT
    assert payload["common_recipe"]["native_mcts_hot_loop"] is True
    assert payload["common_recipe"]["rust_featurize"] is True
    for key, value in source["common_recipe"].items():
        if key != "rust_featurize":
            assert payload["common_recipe"][key] == value
    assert arms["n256"]["seed_start"] == contract.GENERATION_CAMPAIGN_R1_NEXT_SEED_FLOOR
    assert arms["n256"]["seed_end"] == arms["n128"]["seed_start"]
    assert payload["fleet"]["next_campaign_seed_floor"] == arms["n128"]["seed_end"]
    assert {row["name"] for row in payload["source_categories"]} == {
        "current_producer", "recent_history", "hard_negative"
    }
    assert all(Path(arm["output_root"]).parent == tmp_path / "fresh-r2" for arm in arms.values())
    assert contract.validate_generation_campaign(out) == payload
    search, evaluator, generation = contract._campaign_science(  # noqa: SLF001
        payload, n_full=128
    )
    assert search["n_full"] == 128
    assert evaluator["rust_featurize"] is True
    assert generation["native_mcts_hot_loop"] is True


def test_revision_rejects_any_superseded_lock_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = _build(tmp_path, monkeypatch)
    out.unlink()
    lock = tmp_path / "n128.lock.json"
    lock.write_bytes(lock.read_bytes() + b" ")
    with pytest.raises(contract.ContractError, match="not issued r1 bytes"):
        contract.build_generation_campaign_revision(
            SOURCE,
            superseded_lock_paths=[lock, tmp_path / "n256.lock.json"],
            contract_id="a1-dual-arm-n256-n128-56gpu-20260711-r2",
            output_root=tmp_path / "other",
            out_path=out,
        )


def test_revision_requires_absolute_fresh_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = _build(tmp_path, monkeypatch)
    out.unlink()
    with pytest.raises(contract.ContractError, match="output root must be absolute"):
        contract.build_generation_campaign_revision(
            SOURCE,
            superseded_lock_paths=[
                tmp_path / "n128.lock.json", tmp_path / "n256.lock.json"
            ],
            contract_id="a1-dual-arm-n256-n128-56gpu-20260711-r2",
            output_root=Path("relative-output"),
            out_path=out,
        )


@pytest.mark.parametrize("mutation", ["native", "seed", "output"])
def test_revision_rejects_recomputed_contract_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    out, payload = _build(tmp_path, monkeypatch)
    if mutation == "native":
        payload["common_recipe"]["native_mcts_hot_loop"] = False
    elif mutation == "seed":
        payload["arms"][0]["seed_start"] += 8192
    else:
        payload["arms"][0]["output_root"] = payload["arms"][1]["output_root"]
    payload.pop("contract_sha256")
    payload["contract_sha256"] = contract._digest_value(payload)  # noqa: SLF001
    out.chmod(0o644)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(contract.ContractError):
        contract.validate_generation_campaign(out)


def test_revision_never_becomes_ready_without_new_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = _build(tmp_path, monkeypatch)
    with pytest.raises(contract.ContractError, match="new committed promotion"):
        contract.validate_generation_campaign(out, require_ready=True)


@pytest.mark.parametrize(
    "record",
    [
        {"transaction_id": contract.GENERATION_CAMPAIGN_R1_TRANSACTION_ID},
        {"handoff_sha256": contract.GENERATION_CAMPAIGN_R1_HANDOFF_SHA256},
    ],
)
def test_revision_refuses_to_reuse_issued_r1_handoff(record: dict[str, str]) -> None:
    with pytest.raises(contract.ContractError, match="cannot authorize r2"):
        contract._require_fresh_revision_handoff(record)  # noqa: SLF001

    contract._require_fresh_revision_handoff(  # noqa: SLF001
        {"transaction_id": "new-transaction", "handoff_sha256": "sha256:new"}
    )
