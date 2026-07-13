from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_selected_dose_diagnostic_completion as completion
from tools import a1_selected_dose_value_axis_arm as value_axis
from test_a1_selected_dose_pure_soft_arm import _args as pure_soft_args


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _verified(tmp_path: Path, *, arm_id: str) -> dict:
    root = tmp_path / arm_id.lower()
    root.mkdir()
    trainer = _write_json(tmp_path / "runtime" / "tools" / "train_bc.py", {})
    initializer = tmp_path / "f7.pt"
    initializer.write_bytes(b"f7-parent")
    data = _write_json(tmp_path / f"{arm_id}.descriptor.json", {})
    sentinel = _write_json(tmp_path / "validation.json", {})
    manifest_path = _write_json(root / "manifest.json", {"manifest": arm_id})
    command = ["python", str(trainer), "--axis", arm_id]
    kind = "PURE_SEARCH_TARGET" if arm_id == "PURE_SEARCH_TARGET" else "VALUE_AXIS"
    expected_value = (
        list(value_axis.CURRENT_COMPONENT_IDS)
        if arm_id == value_axis.CURRENT_VALUE_SCOPE
        else list(value_axis.EXPECTED_COMPONENT_IDS)
    )
    manifest = {
        "command_sha256": completion.value_axis.bridge.corrected._digest(command),  # noqa: SLF001
        "source_temperature_manifest": {
            "path": str(tmp_path / "source.json"),
            "sha256": "sha256:" + "1" * 64,
        },
        "selected_geometry_evidence": {"evidence": True},
        "only_declared_causal_delta": {"axis": arm_id},
    }
    return {
        "manifest": manifest,
        "manifest_ref": completion._compact_ref(manifest_path),  # noqa: SLF001
        "kind": kind,
        "arm_id": arm_id,
        "claim_schema": "claim-v1",
        "submission_schema": "submission-v1",
        "source": {
            "initialization": completion._compact_ref(initializer),  # noqa: SLF001
            "validation_sentinel": completion._compact_ref(sentinel),  # noqa: SLF001
            "selected_geometry_runtime_repo": str((tmp_path / "runtime").resolve()),
            "selected_geometry_trainer": str(trainer.resolve()),
        },
        "output_root": root,
        "data_path": data.resolve(),
        "data_ref": completion._compact_ref(data),  # noqa: SLF001
        "treatment_meta": {
            "policy_distillation_component_ids": list(
                value_axis.EXPECTED_COMPONENT_IDS
            ),
            "value_training_component_ids": expected_value,
        },
        "command": command,
        "selected_trainer": trainer.resolve(),
    }


def _write_report(verified: dict) -> None:
    root = verified["output_root"]
    (root / "candidate.pt").write_bytes(b"trained-candidate")
    expected = completion._report_expected(verified)  # noqa: SLF001
    policy = list(value_axis.EXPECTED_COMPONENT_IDS)
    value = (
        list(value_axis.CURRENT_COMPONENT_IDS)
        if verified["arm_id"] == value_axis.CURRENT_VALUE_SCOPE
        else policy
    )
    report = {
        **expected,
        "checkpoint": str((root / "candidate.pt").resolve()),
        "init_checkpoint": verified["source"]["initialization"]["path"],
        "init_checkpoint_sha256": verified["source"]["initialization"]["sha256"],
        "data": str(verified["data_path"]),
        "input_validation_game_sentinel_manifest": verified["source"][
            "validation_sentinel"
        ]["path"],
        "checkout_runtime_binding": {
            "trainer": str(verified["selected_trainer"]),
            "trainer_sha256": completion.value_axis.bridge.corrected._file_sha(  # noqa: SLF001
                verified["selected_trainer"]
            ),
        },
        "memmap_composite": {
            "descriptor_path": str(verified["data_path"]),
            "descriptor_file_sha256": verified["data_ref"]["sha256"],
            "component_ids": policy,
            "policy_distillation_component_ids": policy,
            "value_training_component_ids": value,
            "policy_kl_anchor_component_ids": [value_axis.REPLAY_COMPONENT_ID],
        },
        "stored_policy_component_temperatures": (
            value_axis.bridge.production_temp.COMPONENT_TEMPERATURES
        ),
        "policy_distillation_scope": {"component_ids": policy},
        "value_training_scope": {"component_ids": value},
        "metrics": [
            {
                "validation_objective_matched": {
                    "schema_version": "composite-validation-measure-v2",
                    "objective_matched": True,
                    "components": {component: {} for component in policy},
                }
            }
        ],
        "value_active_rows": (
            0 if verified["arm_id"] == value_axis.VALUE_LOSS_OFF else 123
        ),
    }
    _write_json(root / "train.report.json", report)


def _write_submission(verified: dict) -> None:
    root = verified["output_root"]
    unit = "a1-selected-test"
    claim = {
        "schema_version": verified["claim_schema"],
        "created_at_unix_ns": 1,
        "manifest": verified["manifest_ref"],
        "unit": unit,
    }
    claim["claim_sha256"] = completion.value_axis.bridge.corrected._digest(claim)  # noqa: SLF001
    claim_path = _write_json(root / "diagnostic-execution.claim.json", claim)
    receipt = {
        "schema_version": verified["submission_schema"],
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": 2,
        "manifest": verified["manifest_ref"],
        "claim": completion._compact_ref(claim_path),  # noqa: SLF001
        "unit": unit,
        "command_sha256": verified["manifest"]["command_sha256"],
        "systemd_command_sha256": completion.value_axis.bridge.corrected._digest(  # noqa: SLF001
            completion._systemd_command(verified, unit=unit)  # noqa: SLF001
        ),
        "systemd_stdout": "Running as unit.",
    }
    receipt["receipt_sha256"] = completion.value_axis.bridge.corrected._digest(receipt)  # noqa: SLF001
    _write_json(root / "diagnostic-execution.receipt.json", receipt)
    (root / "diagnostic-execution.status.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "stdout.log").write_text("done\n", encoding="utf-8")
    (root / "stderr.log").write_text("", encoding="utf-8")


@pytest.mark.parametrize(
    "arm_id",
    ["PURE_SEARCH_TARGET", value_axis.CURRENT_VALUE_SCOPE, value_axis.VALUE_LOSS_OFF],
)
def test_report_contract_seals_each_value_axis(tmp_path: Path, arm_id: str) -> None:
    verified = _verified(tmp_path, arm_id=arm_id)
    _write_report(verified)
    checkpoint, report = completion._verify_report(verified)  # noqa: SLF001
    assert checkpoint["sha256"].startswith("sha256:")
    assert report["path"].endswith("train.report.json")


def test_completion_receipt_replays_all_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path, arm_id=value_axis.CURRENT_VALUE_SCOPE)
    _write_report(verified)
    _write_submission(verified)
    monkeypatch.setattr(completion, "verify_manifest", lambda _path: verified)
    payload = completion.build_completion(
        Path(verified["manifest_ref"]["path"]), created_at_unix_ns=99
    )
    receipt = verified["output_root"] / completion.COMPLETION_NAME
    completion._write_exclusive(receipt, payload)  # noqa: SLF001
    assert completion.verify_completion(receipt) == payload


def test_pure_soft_manifest_can_be_verified_after_outputs_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = pure_soft_args(tmp_path, monkeypatch)
    manifest, path = completion.pure_soft.prepare(args)
    repo = Path(completion.pure_soft.__file__).resolve().parents[1]
    files = {
        relative: completion.pure_soft.bridge.corrected._file_ref(repo / relative)  # noqa: SLF001
        for relative in completion.pure_soft.SOURCE_FILES
    }
    manifest["source_binding"] = {
        "repository_root": str(repo),
        "git_commit": "ignored-by-posthoc-finalizer",
        "files": files,
        "files_sha256": completion.pure_soft.bridge.corrected._digest(files),  # noqa: SLF001
    }
    manifest["diagnostic_executor"] = files[
        completion.pure_soft.EXECUTOR_RELATIVE_PATH
    ]
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = completion.pure_soft.bridge.corrected._digest(  # noqa: SLF001
        manifest
    )
    path.chmod(0o644)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    (args.output_root / "candidate.pt").write_bytes(b"already-completed")
    (args.output_root / "train.report.json").write_text("{}", encoding="utf-8")
    source_ref = manifest["descriptor"]
    monkeypatch.setattr(
        completion.value_axis,
        "_source_descriptor_contract",
        lambda _source: (
            {},
            {
                "component_ids": list(value_axis.EXPECTED_COMPONENT_IDS),
                "policy_distillation_component_ids": list(
                    value_axis.EXPECTED_COMPONENT_IDS
                ),
                "value_training_component_ids": list(
                    value_axis.EXPECTED_COMPONENT_IDS
                ),
                "policy_kl_anchor_component_ids": [value_axis.REPLAY_COMPONENT_ID],
                "stored_policy_component_temperatures": (
                    value_axis.bridge.production_temp.COMPONENT_TEMPERATURES
                ),
            },
            source_ref,
        ),
    )
    verified = completion.verify_manifest(path)
    assert verified["kind"] == "PURE_SEARCH_TARGET"
    assert (args.output_root / "candidate.pt").exists()
