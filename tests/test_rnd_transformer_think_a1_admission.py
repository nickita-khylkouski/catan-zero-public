from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

import tools.rnd_transformer_think_a1_admission as admission
from tools.rnd_transformer_think_a1_admission import (
    ADMISSION_SCHEMA,
    ARMS,
    ARTIFACT_ROLES,
    AdmissionError,
    IDENTITY_SCHEMA,
    RUN_KEYS,
    SEEDS,
    SOURCE_FILES,
    admit_all,
    admit_run,
    register_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/rnd/transformer_think_a1_screen_20260711/experiment.template.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    template = tmp_path / "template.json"
    shutil.copyfile(TEMPLATE, template)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    payload = corpus / "action_taken.dat"
    payload.write_bytes(b"payload")
    records = [{
        "filename": payload.name,
        "size_bytes": payload.stat().st_size,
        "sha256": "sha256:" + _sha(payload),
    }]
    inventory = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (corpus / "corpus_meta.json").write_text(json.dumps({
        "payload_inventory_schema": "memmap-payload-inventory-v1",
        "payload_inventory": records,
        "payload_inventory_sha256": "sha256:" + inventory,
    }))
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    artifacts: dict[str, Path] = {}
    for role in ARTIFACT_ROLES:
        path = artifact_dir / f"{role}.json"
        path.write_text(json.dumps({"role": role}))
        artifacts[role] = path
    training = tmp_path / "training.json"
    training.write_text('{"schema_version":"a1-selected-training-games-v1"}')
    source_root = tmp_path / "source"
    for relative in SOURCE_FILES:
        source = source_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"source:{relative}\n")
    checkpoints: dict[str, Path] = {}
    identity_seeds = []
    teacher_state_sha = hashlib.sha256(b"fixture-teacher-state").hexdigest()
    for seed in SEEDS:
        base_sha = teacher_state_sha
        expanded_sha = hashlib.sha256(f"expanded:{seed}".encode()).hexdigest()
        arm_rows = []
        for arm, (steps, params, _capacity) in ARMS.items():
            key = f"{arm}@{seed}"
            checkpoint = tmp_path / "init" / f"seed_{seed}" / f"{arm}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(
                b"fixture-frozen-teacher"
                if arm == "transformer-k0"
                else f"checkpoint:{key}".encode()
            )
            checkpoints[key] = checkpoint
            arm_rows.append({
                "arm_id": arm,
                "latent_deliberation_steps": steps,
                "parameter_count": params,
                "checkpoint_sha256": _sha(checkpoint),
                "model_state_sha256": base_sha if arm == "transformer-k0" else expanded_sha,
                "shared_base_state_sha256": base_sha,
                "compared_to": f"transformer-k0@{seed}",
                "exact_identity": True,
                "max_abs_logit_diff": 0.0,
                "max_abs_value_diff": 0.0,
                "max_abs_final_vp_diff": 0.0,
            })
        identity_seeds.append({
            "training_seed": seed,
            "probe_batch_sha256": "a" * 64,
            "arms": arm_rows,
        })
    teacher_sha = _sha(checkpoints[f"transformer-k0@{SEEDS[0]}"])
    monkeypatch.setattr(
        admission, "FROZEN_INCUMBENT_CHECKPOINT_SHA256", teacher_sha
    )
    template_payload = json.loads(template.read_text())
    template_payload["common"]["frozen_incumbent_checkpoint_sha256"] = teacher_sha
    template_payload.pop("config_sha256")
    template_payload["config_sha256"] = hashlib.sha256(
        json.dumps(
            template_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    template.write_text(json.dumps(template_payload))
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({
        "schema_version": IDENTITY_SCHEMA,
        "reference_arm": "transformer-k0",
        "source_teacher_checkpoint_sha256": teacher_sha,
        "seeds": identity_seeds,
    }))
    return {
        "template": template,
        "corpus_dir": corpus,
        "training_manifest": training,
        "validation_manifest": artifacts["validation_manifest"],
        "artifact_paths": artifacts,
        "identity_report": identity,
        "checkpoint_paths": checkpoints,
        "source_root": source_root,
        "output": tmp_path / "registered.json",
    }


def test_template_is_self_hashed_and_exact_family() -> None:
    config = json.loads(TEMPLATE.read_text())
    declared = config.pop("config_sha256")
    assert declared == hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode()
    ).hexdigest()
    assert config["run_matrix"]["seeds"] == [101, 103, 107]
    assert config["run_matrix"]["required_run_count"] == 12
    assert (
        config["comparison_contract"][
            "maximum_nonforced_decision_micro_ce_regression"
        ]
        == 0.005
    )
    assert "maximum_overall_ce_regression" not in config["comparison_contract"]
    assert {row["arm_id"]: row["expected_parameters"] for row in config["arms"]} == {
        arm: values[1] for arm, values in ARMS.items()
    }


def test_register_and_admit_exact_transformer_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path, monkeypatch)
    registered = register_experiment(**values)
    assert registered["status"] == "registered_ready"
    assert set(registered["registration"]["executing_learner_source_sha256"]) == set(SOURCE_FILES)
    assert set(registered["registration"]["initial_checkpoint_sha256_by_arm_seed"]) == set(RUN_KEYS)
    frozen_bytes = values["output"].read_bytes()
    with pytest.raises(AdmissionError, match="refusing to overwrite"):
        register_experiment(**values)
    assert values["output"].read_bytes() == frozen_bytes
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    output = repo_root / "runs/rnd_transformer_think_a1_screen_20260711/think-transformer-k2/seed_103/admission.json"
    admitted = admit_run(
        experiment=values["output"], arm="think-transformer-k2", training_seed=103,
        repo_root=repo_root, output=output,
        **{key: value for key, value in values.items() if key not in {"template", "output"}},
    )
    assert admitted["schema_version"] == ADMISSION_SCHEMA
    assert admitted["expected_parameters"] == 40_793_673
    argv = admitted["train_argv"]
    assert argv[argv.index("--hidden-size") + 1] == "640"
    assert argv[argv.index("--graph-layers") + 1] == "6"
    assert argv[argv.index("--attention-heads") + 1] == "8"
    assert argv[argv.index("--entity-state-trunk") + 1] == "transformer"
    assert argv[argv.index("--latent-deliberation-steps") + 1] == "2"
    from tools import train_bc

    parsed = train_bc.build_parser().parse_args(argv[2:])
    assert parsed.grad_accum_steps == 4
    assert parsed.max_steps == 250
    assert parsed.entity_state_trunk == "transformer"


def test_identity_rejects_nonidentical_expanded_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path, monkeypatch)
    report = json.loads(values["identity_report"].read_text())
    report["seeds"][0]["arms"][2]["model_state_sha256"] = "f" * 64
    values["identity_report"].write_text(json.dumps(report))
    with pytest.raises(AdmissionError, match="expanded weights differ"):
        register_experiment(**values)


def test_admit_all_is_complete_atomic_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path, monkeypatch)
    register_experiment(**values)
    repo_root = tmp_path / "checkout"
    occupied = repo_root / "runs/rnd_transformer_think_a1_screen_20260711/think-transformer-k4/seed_107/admission.json"
    occupied.parent.mkdir(parents=True)
    occupied.write_bytes(b"keep")
    common = {key: value for key, value in values.items() if key not in {"template", "output"}}
    with pytest.raises(AdmissionError, match="refusing to overwrite"):
        admit_all(experiment=values["output"], repo_root=repo_root, **common)
    assert occupied.read_bytes() == b"keep"
    assert not (repo_root / "runs/rnd_transformer_think_a1_screen_20260711/transformer-k0/seed_101/admission.json").exists()

    occupied.unlink()
    rows = admit_all(experiment=values["output"], repo_root=repo_root, **common)
    assert len(rows) == 12
    assert {f"{row['arm_id']}@{row['training_seed']}" for row in rows} == set(RUN_KEYS)
