from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.rnd_topology_run_manifest import (
    ManifestError,
    RUN_SCHEMA,
    build_run_manifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, object]:
    training = _write(
        tmp_path / "training.json",
        {"schema_version": "a1-selected-training-games-v1"},
    )
    experiment_payload = {
        "schema_version": "catan-zero-topology-real-train/v1",
        "config_sha256_scope": "canonical_json_without_config_sha256",
        "arms": [{"arm_id": "candidate"}, {"arm_id": "control"}],
        "learning_gate": {
            "seeds": [1, 2],
            "training_manifest_sha256": _sha(training),
        },
    }
    experiment_payload["config_sha256"] = _canonical_sha(experiment_payload)
    experiment = _write(tmp_path / "experiment.json", experiment_payload)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    sidecar = Path(str(checkpoint) + ".optimizer.pt")
    sidecar.write_bytes(b"optimizer")
    report = _write(
        tmp_path / "report.json",
        {
            "seed": 1,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": "sha256:" + _sha(checkpoint),
            "optimizer_sidecar": str(sidecar),
            "optimizer_sidecar_sha256": "sha256:" + _sha(sidecar),
        },
    )
    return {
        "arm": "candidate",
        "training_seed": 1,
        "training_manifest": training,
        "training_report": report,
        "experiment_config": experiment,
        "checkpoint": checkpoint,
        "optimizer_sidecar": sidecar,
        "output": tmp_path / "run.json",
        "repo_root": tmp_path,
    }


def test_builds_exact_consumer_schema_and_hashes_every_input(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    result = build_run_manifest(**inputs)
    on_disk = json.loads(inputs["output"].read_text(encoding="utf-8"))
    assert on_disk == result
    assert set(result) == {
        "schema_version",
        "arm",
        "training_seed",
        "training_manifest_sha256",
        "training_report",
        "experiment_config",
        "checkpoint",
        "optimizer_sidecar",
    }
    assert result["schema_version"] == RUN_SCHEMA
    assert result["training_manifest_sha256"] == _sha(inputs["training_manifest"])
    for name in ("training_report", "experiment_config", "checkpoint", "optimizer_sidecar"):
        assert set(result[name]) == {"path", "file_sha256"}
        assert result[name]["file_sha256"] == _sha(Path(result[name]["path"]))
        assert Path(result[name]["path"]).is_absolute()


def test_refuses_overwrite_without_changing_existing_bytes(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs["output"].write_bytes(b"keep me")
    with pytest.raises(ManifestError, match="refusing to overwrite"):
        build_run_manifest(**inputs)
    assert inputs["output"].read_bytes() == b"keep me"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda values: values.update(arm="missing"), "exactly one arm"),
        (lambda values: values.update(training_seed=9), "not registered"),
    ],
)
def test_rejects_unregistered_arm_or_seed(tmp_path: Path, mutation, message: str) -> None:
    inputs = _fixture(tmp_path)
    mutation(inputs)
    with pytest.raises(ManifestError, match=message):
        build_run_manifest(**inputs)
    assert not inputs["output"].exists()


def test_rejects_report_seed_mismatch(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    report = json.loads(inputs["training_report"].read_text())
    report["seed"] = 2
    _write(inputs["training_report"], report)
    with pytest.raises(ManifestError, match="report seed differs"):
        build_run_manifest(**inputs)


@pytest.mark.parametrize("field", ["checkpoint", "optimizer_sidecar"])
def test_rejects_report_path_or_digest_drift(tmp_path: Path, field: str) -> None:
    inputs = _fixture(tmp_path)
    report = json.loads(inputs["training_report"].read_text())
    report[f"{field}_sha256"] = "sha256:" + "0" * 64
    _write(inputs["training_report"], report)
    with pytest.raises(ManifestError, match="digest differs"):
        build_run_manifest(**inputs)


def test_rejects_noncanonical_optimizer_sidecar(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    other = tmp_path / "other.optimizer.pt"
    other.write_bytes(b"optimizer")
    inputs["optimizer_sidecar"] = other
    with pytest.raises(ManifestError, match="canonical sidecar"):
        build_run_manifest(**inputs)


def test_rejects_experiment_or_training_registration_drift(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs["training_manifest"].write_text(
        json.dumps({"schema_version": "a1-selected-training-games-v1", "drift": True})
    )
    with pytest.raises(ManifestError, match="differs from the experiment"):
        build_run_manifest(**inputs)


def test_rejects_invalid_experiment_self_hash(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    experiment = json.loads(inputs["experiment_config"].read_text())
    experiment["arms"].append({"arm_id": "extra"})
    _write(inputs["experiment_config"], experiment)
    with pytest.raises(ManifestError, match="self-hash is invalid"):
        build_run_manifest(**inputs)


def test_report_paths_resolve_from_repo_root_not_nested_report_dir(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    run_dir = tmp_path / "runs" / "candidate"
    report_dir = run_dir / "reports" / "nested"
    report_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"real-layout checkpoint")
    sidecar = Path(str(checkpoint) + ".optimizer.pt")
    sidecar.write_bytes(b"real-layout optimizer")
    report = _write(
        report_dir / "report.json",
        {
            "seed": 1,
            "checkpoint": "runs/candidate/checkpoint.pt",
            "checkpoint_sha256": "sha256:" + _sha(checkpoint),
            "optimizer_sidecar": "runs/candidate/checkpoint.pt.optimizer.pt",
            "optimizer_sidecar_sha256": "sha256:" + _sha(sidecar),
        },
    )
    inputs.update(
        training_report=report,
        checkpoint=checkpoint,
        optimizer_sidecar=sidecar,
    )

    result = build_run_manifest(**inputs)

    assert result["checkpoint"]["path"] == str(checkpoint.resolve())
    assert result["optimizer_sidecar"]["path"] == str(sidecar.resolve())


@pytest.mark.parametrize("field", ["checkpoint", "optimizer_sidecar"])
def test_rejects_report_path_that_escapes_repo_root(tmp_path: Path, field: str) -> None:
    inputs = _fixture(tmp_path)
    report = json.loads(inputs["training_report"].read_text())
    report[field] = "../outside.pt"
    _write(inputs["training_report"], report)

    with pytest.raises(ManifestError, match=rf"{field} path escapes"):
        build_run_manifest(**inputs)
    assert not inputs["output"].exists()
