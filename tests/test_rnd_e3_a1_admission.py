from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from tools.rnd_e3_a1_admission import (
    ADMISSION_SCHEMA,
    ARMS,
    ARTIFACT_ROLES,
    AdmissionError,
    IDENTITY_SCHEMA,
    RUN_KEYS,
    SEEDS,
    SOURCE_FILES,
    admit_run,
    admit_all,
    register_experiment,
)
from tools import rnd_e3_a1_admission as admission_module


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/rnd/e3_a1_screen_20260710/experiment.template.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    template = tmp_path / "experiment.template.json"
    shutil.copyfile(TEMPLATE, template)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    payload = corpus / "action_taken.dat"
    payload.write_bytes(b"payload")
    inventory_records = [
        {
            "filename": payload.name,
            "size_bytes": payload.stat().st_size,
            "sha256": "sha256:" + _sha(payload),
        }
    ]
    inventory = hashlib.sha256(
        json.dumps(inventory_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (corpus / "corpus_meta.json").write_text(
        json.dumps(
            {
                "payload_inventory_schema": "memmap-payload-inventory-v1",
                "payload_inventory": inventory_records,
                "payload_inventory_sha256": "sha256:" + inventory,
            }
        )
    )
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    artifacts = {}
    for role in ARTIFACT_ROLES:
        path = artifacts_dir / f"{role}.json"
        path.write_text(json.dumps({"role": role}))
        artifacts[role] = path
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"schema_version": "a1-selected-training-games-v1"}))
    validation = artifacts["validation_manifest"]
    source_root = tmp_path / "source"
    for relative in SOURCE_FILES:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n")
    checkpoints = {}
    seed_rows = []
    for seed in SEEDS:
        base_state_sha = hashlib.sha256(f"base:{seed}".encode()).hexdigest()
        expanded_state_sha = hashlib.sha256(f"expanded:{seed}".encode()).hexdigest()
        arm_rows = []
        for arm, (steps, params, _capacity) in ARMS.items():
            key = f"{arm}@{seed}"
            checkpoint = tmp_path / "initialization" / f"seed_{seed}" / f"{arm}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"checkpoint:{key}".encode())
            checkpoints[key] = checkpoint
            arm_rows.append(
                {
                    "arm_id": arm,
                    "latent_deliberation_steps": steps,
                    "parameter_count": params,
                    "checkpoint_sha256": _sha(checkpoint),
                    "model_state_sha256": base_state_sha if arm == "rrt-k0" else expanded_state_sha,
                    "shared_base_state_sha256": base_state_sha,
                    "compared_to": f"rrt-k0@{seed}",
                    "exact_identity": True,
                    "max_abs_logit_diff": 0.0,
                    "max_abs_value_diff": 0.0,
                    "max_abs_final_vp_diff": 0.0,
                }
            )
        seed_rows.append(
            {"training_seed": seed, "probe_batch_sha256": "2" * 64, "arms": arm_rows}
        )
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "schema_version": IDENTITY_SCHEMA,
                "reference_arm": "rrt-k0",
                "seeds": seed_rows,
            }
        )
    )
    return {
        "template": template,
        "corpus_dir": corpus,
        "training_manifest": training,
        "validation_manifest": validation,
        "artifact_paths": artifacts,
        "identity_report": identity,
        "checkpoint_paths": checkpoints,
        "source_root": source_root,
        "output": tmp_path / "experiment.registered.json",
    }


def test_template_is_self_hashed_and_capacity_aware() -> None:
    payload = json.loads(TEMPLATE.read_text())
    declared = payload.pop("config_sha256")
    actual = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    assert declared == actual
    arms = {arm["arm_id"]: arm for arm in payload["arms"]}
    assert arms["rrt-k0"]["promotion_eligible"] is False
    assert arms["rrt-k0"]["expected_parameters"] == 20_070_932
    assert {arms[name]["expected_parameters"] for name in ARMS if name != "rrt-k0"} == {
        22_146_068
    }
    assert payload["comparison_contract"]["primary_reference_arm"] == "think-rrt-k1"
    assert payload["comparison_contract"]["primary_candidate_arms"] == [
        "think-rrt-k2",
        "think-rrt-k4",
    ]


def test_registers_all_exact_inputs_then_admits_only_exact_output(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    registered = register_experiment(**values)
    assert registered["status"] == "registered_ready"
    frozen = registered["registration"]
    assert set(frozen["initial_checkpoint_sha256_by_arm_seed"]) == set(RUN_KEYS)
    assert set(frozen["executing_learner_source_sha256"]) == set(SOURCE_FILES)

    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    output = repo_root / "runs/rnd_e3_a1_screen_20260710/think-rrt-k2/seed_29/admission.json"
    admitted = admit_run(
        experiment=values["output"],
        arm="think-rrt-k2",
        training_seed=29,
        corpus_dir=values["corpus_dir"],
        training_manifest=values["training_manifest"],
        validation_manifest=values["validation_manifest"],
        artifact_paths=values["artifact_paths"],
        identity_report=values["identity_report"],
        checkpoint_paths=values["checkpoint_paths"],
        source_root=values["source_root"],
        output=output,
        repo_root=repo_root,
    )
    assert admitted["schema_version"] == ADMISSION_SCHEMA
    assert admitted["expected_parameters"] == 22_146_068
    argv = admitted["train_argv"]
    for flag in (
        "--graph-history-features",
        "--rnd-allow-a1-learner-override",
        "--skip-teacher-quality-gate",
        "--trust-curated-data-quality",
        "--no-symmetry-augment",
        "--soft-target-weight",
        "--value-target-lambda",
        "--device",
    ):
        assert flag in argv
    assert argv[argv.index("--latent-deliberation-steps") + 1] == "2"
    assert argv[argv.index("--init-checkpoint") + 1] == str(
        values["checkpoint_paths"]["think-rrt-k2@29"].resolve()
    )
    from tools import train_bc

    parsed = train_bc.build_parser().parse_args(argv[2:])
    assert parsed.graph_history_features is True
    assert parsed.latent_deliberation_steps == 2
    assert parsed.grad_accum_steps == 4
    assert parsed.symmetry_augment is False


def test_registration_rejects_nonidentical_expanded_family(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    report = json.loads(values["identity_report"].read_text())
    report["seeds"][0]["arms"][2]["model_state_sha256"] = "f" * 64
    values["identity_report"].write_text(json.dumps(report))
    with pytest.raises(AdmissionError, match="expanded model weights differ"):
        register_experiment(**values)


def test_admission_rejects_source_drift_and_wrong_output(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    register_experiment(**values)
    source = values["source_root"] / SOURCE_FILES[0]
    source.write_text("drift")
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    common = dict(
        experiment=values["output"],
        arm="think-rrt-k4",
        training_seed=47,
        corpus_dir=values["corpus_dir"],
        training_manifest=values["training_manifest"],
        validation_manifest=values["validation_manifest"],
        artifact_paths=values["artifact_paths"],
        identity_report=values["identity_report"],
        checkpoint_paths=values["checkpoint_paths"],
        source_root=values["source_root"],
        output=tmp_path / "wrong.json",
        repo_root=repo_root,
    )
    with pytest.raises(AdmissionError, match="sources differ"):
        admit_run(**common)
    source.write_text(f"source:{SOURCE_FILES[0]}\n")
    with pytest.raises(AdmissionError, match="output must be exactly"):
        admit_run(**common)


def test_admit_all_fingerprints_corpus_once_and_publishes_complete_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    register_experiment(**values)
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    original = admission_module._corpus_fingerprint
    calls = 0

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(admission_module, "_corpus_fingerprint", counted)
    published = admit_all(
        experiment=values["output"],
        corpus_dir=values["corpus_dir"],
        training_manifest=values["training_manifest"],
        validation_manifest=values["validation_manifest"],
        artifact_paths=values["artifact_paths"],
        identity_report=values["identity_report"],
        checkpoint_paths=values["checkpoint_paths"],
        source_root=values["source_root"],
        repo_root=repo_root,
    )
    assert calls == 1
    assert len(published) == 15
    actual = {
        f"{item['arm_id']}@{item['training_seed']}"
        for item in published
    }
    assert actual == set(RUN_KEYS)
    for item in published:
        manifest = Path(item["run_directory"]) / "admission.json"
        assert manifest.is_file()
        assert json.loads(manifest.read_text()) == item


def test_admit_all_preflights_every_destination_before_publishing(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    register_experiment(**values)
    repo_root = tmp_path / "checkout"
    occupied = repo_root / "runs/rnd_e3_a1_screen_20260710/think-rrt-k8/seed_47/admission.json"
    occupied.parent.mkdir(parents=True)
    occupied.write_bytes(b"keep")
    with pytest.raises(AdmissionError, match="refusing to overwrite"):
        admit_all(
            experiment=values["output"],
            corpus_dir=values["corpus_dir"],
            training_manifest=values["training_manifest"],
            validation_manifest=values["validation_manifest"],
            artifact_paths=values["artifact_paths"],
            identity_report=values["identity_report"],
            checkpoint_paths=values["checkpoint_paths"],
            source_root=values["source_root"],
            repo_root=repo_root,
        )
    assert occupied.read_bytes() == b"keep"
    assert not (
        repo_root
        / "runs/rnd_e3_a1_screen_20260710/rrt-k0/seed_11/admission.json"
    ).exists()
