from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools import a1_dual_learner_contract as contract


SHA = "sha256:" + "a" * 64


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": contract._sha256(path)}  # noqa: SLF001


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    arm_lock = tmp_path / "arm.lock.json"
    arm_lock.write_text("{}")
    files = {}
    for name in ("corpus_meta", "selected", "audit", "validation", "producer"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        files[name] = _ref(path)
    recipe = {
        "world_size": 8, "batch_size": 512, "grad_accum_steps": 1,
        "global_batch_size": 4096, "ddp_shard_data": False,
    }
    objective = {"objective": "mse", "value_readout": "scalar"}
    spec = {
        "schema_version": contract.SPEC_SCHEMA,
        "arm_id": "n128", "subset_id": "matched-56k",
        "objective": objective, "recipe": recipe, "topology": contract.TOPOLOGY,
    }
    spec_path = tmp_path / "learner.spec.json"
    spec_path.write_text(json.dumps(spec))
    artifacts = {
        "arm_id": "n128", "subset_id": "matched-56k",
        "contract_sha256": SHA, "objective": objective, "recipe": recipe,
        "learner_code_sha256": SHA, "runtime_code_tree_sha256": SHA,
        "corpus_meta": files["corpus_meta"], "selected_manifest": files["selected"],
        "audit": files["audit"], "validation": files["validation"],
        "producer": files["producer"], "payload_inventory_sha256": SHA,
        "data_fingerprint": SHA, "corpus_rows": 100,
        "training_rows": 90, "validation_rows": 10,
        "selected_game_seed_set_sha256": SHA,
        "training_game_seed_set_sha256": SHA,
        "validation_game_seed_set_sha256": SHA,
    }
    monkeypatch.setattr(contract, "inspect_artifacts", lambda **_kwargs: artifacts)
    arm = {
        "contract_sha256": SHA,
        "game_contract": {"arm_id": "n128"},
        "checkpoints": [{"role": "producer", "sha256": files["producer"]["sha256"]}],
    }
    value = contract.build_lock(
        arm_lock=arm_lock, learner_spec=spec_path, data=tmp_path,
        validation=Path(files["validation"]["path"]),
        producer_checkpoint=Path(files["producer"]["path"]),
        verify_arm_lock=lambda *_args, **_kwargs: arm,
    )
    lock_path = tmp_path / "learner.lock.json"
    contract._write_new(lock_path, value)  # noqa: SLF001
    monkeypatch.setattr(
        contract.generation_contract,
        "verify_lock",
        lambda *_args, **_kwargs: arm,
    )
    return lock_path, value


def test_reviewed_lock_binds_trainer_guard_runtime_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, value = _fixture(tmp_path, monkeypatch)
    paths = {record["path"] for record in value["runtime"]}
    assert "tools/train_bc.py" in paths
    assert "configs/guards/train_bc.json" in paths
    assert "tools/a1_dual_arm_train.py" in paths
    loaded = contract.verify_lock(
        lock_path, reviewed_file_sha256=contract._sha256(lock_path)  # noqa: SLF001
    )
    assert loaded["topology"] == contract.TOPOLOGY


def test_review_spec_can_explicitly_authorize_known_two_b200_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lock, value = _fixture(tmp_path, monkeypatch)
    artifacts = {
        "arm_id": value["arm_id"], "subset_id": value["subset_id"],
        "objective": value["objective"], "recipe": value["recipe"],
    }
    monkeypatch.setattr(contract, "inspect_artifacts", lambda **_kwargs: artifacts)
    spec = contract.render_spec(
        data=tmp_path, validation=tmp_path / "validation.bin",
        producer_checkpoint=tmp_path / "producer.bin", world_size=2,
    )
    assert spec["topology"] == contract.TOPOLOGIES[2]
    assert spec["topology"]["global_batch_size"] == 4096


def test_reviewed_lock_rejects_recipe_and_runtime_mutation_even_if_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, value = _fixture(tmp_path, monkeypatch)
    os.chmod(lock_path, 0o644)
    value["recipe"]["batch_size"] = 1024
    value["lock_sha256"] = contract._digest(  # noqa: SLF001
        {key: item for key, item in value.items() if key != "lock_sha256"}
    )
    lock_path.write_text(json.dumps(value))
    with pytest.raises(contract.LearnerContractError, match="spec/lock semantics"):
        contract.verify_lock(
            lock_path, reviewed_file_sha256=contract._sha256(lock_path)  # noqa: SLF001
        )

    value["recipe"]["batch_size"] = 512
    value["runtime"][0]["sha256"] = SHA
    value["runtime_sha256"] = contract._digest(value["runtime"])  # noqa: SLF001
    value["lock_sha256"] = contract._digest(  # noqa: SLF001
        {key: item for key, item in value.items() if key != "lock_sha256"}
    )
    lock_path.write_text(json.dumps(value))
    with pytest.raises(contract.LearnerContractError, match="runtime byte drift"):
        contract.verify_lock(
            lock_path, reviewed_file_sha256=contract._sha256(lock_path)  # noqa: SLF001
        )


def test_reviewed_lock_requires_explicit_raw_file_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, _value = _fixture(tmp_path, monkeypatch)
    with pytest.raises(contract.LearnerContractError, match="explicitly reviewed"):
        contract.verify_lock(lock_path, reviewed_file_sha256=SHA)


@pytest.mark.parametrize("relative", ["tools/train_bc.py", "configs/guards/train_bc.json"])
def test_reviewed_lock_rejects_trainer_or_guard_digest_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    lock_path, value = _fixture(tmp_path, monkeypatch)
    os.chmod(lock_path, 0o644)
    record = next(item for item in value["runtime"] if item["path"] == relative)
    record["sha256"] = SHA
    value["runtime_sha256"] = contract._digest(value["runtime"])  # noqa: SLF001
    value["lock_sha256"] = contract._digest(  # noqa: SLF001
        {key: item for key, item in value.items() if key != "lock_sha256"}
    )
    lock_path.write_text(json.dumps(value))
    with pytest.raises(contract.LearnerContractError, match="runtime byte drift"):
        contract.verify_lock(
            lock_path, reviewed_file_sha256=contract._sha256(lock_path)  # noqa: SLF001
        )
