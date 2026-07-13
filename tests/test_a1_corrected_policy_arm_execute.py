from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import a1_corrected_policy_arm as arm
from tools import a1_corrected_policy_arm_execute as executor
from test_a1_corrected_policy_arm import _args


def test_direct_cli_help_works_outside_repository(tmp_path: Path) -> None:
    script = Path(executor.__file__).resolve()
    completed = subprocess.run(
        (sys.executable, str(script), "--help"),
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--manifest" in completed.stdout


def _manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    future_two_component: bool = False,
) -> tuple[Path, dict]:
    args = _args(
        tmp_path, monkeypatch, future_two_component=future_two_component
    )
    trainer = tmp_path / "tools" / "train_bc.py"
    trainer.parent.mkdir(exist_ok=True)
    trainer.write_text("# exact trainer\n", encoding="utf-8")
    manifest, path = arm.prepare(args)
    manifest["source_binding"] = {
        "repository_root": str(tmp_path),
        "git_commit": "abc123",
        "files": {"tools/train_bc.py": arm._file_ref(trainer)},
    }
    trainer_index = next(
        index for index, value in enumerate(manifest["command"])
        if Path(value).name == "train_bc.py"
    )
    manifest["command"][trainer_index] = str(trainer)
    manifest["command_sha256"] = arm._digest(manifest["command"])
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = arm._digest(manifest)
    path.chmod(0o644)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(executor, "_git_head", lambda repo: "abc123")
    return path, manifest


def test_verify_replays_manifest_inputs_and_real_sentinel_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _manifest_payload = _manifest(tmp_path, monkeypatch)
    verified = executor.verify(path)
    assert "--validation-game-sentinel-manifest" in verified["command"]
    assert "--validation-game-seed-manifest" not in verified["command"]
    assert verified["manifest"]["evaluation_baseline"] == verified["manifest"][
        "initialization"
    ]


def test_future_n128_plus_predecessor_operator_replays_through_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(
        tmp_path, monkeypatch, future_two_component=True
    )
    assert payload["supervision_contract"]["component_ids"] == [
        "n128_current", "predecessor_replay"
    ]
    assert payload["supervision_contract"]["component_game_sampling_ratios"] == [
        0.8, 0.2
    ]
    verified = executor.verify(path)
    assert verified["manifest_ref"]["sha256"] == arm._file_sha(path)


def test_verify_refuses_reused_validation_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch, future_two_component=True)
    fresh = Path(payload["validation_sentinel"]["path"])
    value = json.loads(fresh.read_text(encoding="utf-8"))
    value["game_seeds"] = [1]
    fresh.write_text(json.dumps(value), encoding="utf-8")
    payload["validation_sentinel"] = arm._file_ref(fresh)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = arm._digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(executor.ExecutionError, match="fresh disjoint games"):
        executor.verify(path)


def test_verify_refuses_evaluation_baseline_different_from_initializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    other = tmp_path / "other-baseline.pt"
    other.write_bytes(b"other")
    payload["evaluation_baseline"] = arm._file_ref(other)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = arm._digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(executor.ExecutionError, match="differs from learner initializer"):
        executor.verify(path)


def _write_realized_report(payload: dict) -> Path:
    report = Path(arm._option(payload["command"], "--report"))
    report.parent.mkdir(parents=True, exist_ok=True)
    current = list(payload["supervision_contract"]["component_ids"])
    expected_active = payload["supervision_contract"]["policy_active_row_dose"][
        "reference_base_active_rows"
    ]
    ratios = payload["supervision_contract"]["component_game_sampling_ratios"]
    component_metrics = {
        component_id: {
            "games": 3,
            "rows": 9,
            "metrics": {
                "loss": 1.0,
                "policy_loss": 0.8,
                "value_loss": 0.8,
                "accuracy": 0.6,
                "active_policy_teacher_gap_closure": 0.1,
            },
        }
        for component_id in current
    }
    report.write_text(
        json.dumps(
            {
                "init_checkpoint_sha256": payload["initialization"]["sha256"],
                "optimizer_restored": False,
                "resume_optimizer": False,
                "resumed_optimizer_step": None,
                "world_size": 8,
                "batch_size": 512,
                "grad_accum_steps": 1,
                "effective_global_batch_size": arm.GLOBAL_BATCH_SIZE,
                "training_row_draws": arm.GLOBAL_ROW_DOSE,
                "max_steps": arm.OPTIMIZER_STEPS,
                "steps_completed": arm.OPTIMIZER_STEPS,
                "total_training_steps": arm.OPTIMIZER_STEPS,
                "data_fingerprint": payload["descriptor_fingerprint"],
                "input_validation_game_seed_manifest": payload["validation_sentinel"]["path"],
                "input_validation_game_seed_manifest_sha256": payload["validation_sentinel"]["sha256"],
                "input_validation_game_sentinel_manifest": payload["validation_sentinel"]["path"],
                "validation_game_seed_set_sha256": payload[
                    "validation_sentinel_selection_sha256"
                ],
                "epochs": 1,
                "lr": 3e-5,
                "lr_warmup_steps": 100,
                "lr_schedule": "flat",
                "value_loss_weight": 0.25,
                "value_lr_mult": 0.3,
                "value_target_lambda": 1.0,
                "forced_action_weight": 0.0,
                "forced_row_value_weight": 1.0,
                "policy_loss_weight": 1.0,
                "soft_target_temperature": 0.7,
                "soft_target_min_legal_coverage": 0.5,
                "mask_hidden_info": True,
                "policy_distillation_scope": {"component_ids": current},
                "value_training_scope": {"component_ids": current},
                "memmap_composite": {
                    "policy_distillation_component_ids": current,
                    "value_training_component_ids": current,
                    "policy_kl_anchor_component_ids": current,
                    "policy_distillation_scope_explicit": True,
                    "value_training_scope_explicit": True,
                },
                "soft_target_source": "policy",
                "soft_target_weight": 0.9,
                "policy_aux_active_batch_size": 0,
                "policy_kl_anchor_direction": "forward",
                "policy_kl_anchor_weight": arm.REPLAY_ANCHOR_WEIGHT,
                "winner_sample_weight": 1.0,
                "loser_sample_weight": 1.0,
                "policy_base_active_rows": expected_active,
                "policy_aux_active_rows": arm.EXPECTED_POLICY_AUX_ACTIVE_ROWS,
                "policy_total_active_rows": expected_active,
                "metrics": [
                    {
                        "epoch": 1,
                        "validation_objective_matched": {
                            "schema_version": "composite-validation-measure-v2",
                            "objective_matched": True,
                            "component_sampling_ratios": dict(zip(current, ratios, strict=True)),
                            "components": component_metrics,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return report


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("init_checkpoint_sha256", "sha256:" + "0" * 64),
        ("optimizer_restored", True),
        ("resume_optimizer", True),
        ("resumed_optimizer_step", 7),
        ("world_size", 4),
        ("batch_size", 256),
        ("grad_accum_steps", 2),
        ("effective_global_batch_size", 8192),
        ("training_row_draws", arm.GLOBAL_ROW_DOSE - 1),
        ("max_steps", 2048),
        ("steps_completed", 1023),
        ("data_fingerprint", "sha256:" + "1" * 64),
        ("input_validation_game_seed_manifest_sha256", "sha256:" + "2" * 64),
        ("validation_game_seed_set_sha256", "sha256:" + "3" * 64),
    ],
)
def test_verify_training_report_refuses_one_dose_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    report = _write_realized_report(payload)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_payload[field] = value
    report.write_text(json.dumps(report_payload), encoding="utf-8")
    with pytest.raises(executor.ExecutionError, match="one-dose execution identity drift"):
        executor.verify_training_report(path, report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 2),
        ("lr", 6e-5),
        ("lr_warmup_steps", 0),
        ("lr_schedule", "cosine"),
        ("value_loss_weight", 1.0),
        ("value_lr_mult", 1.0),
        ("value_target_lambda", 0.5),
        ("forced_action_weight", 0.1),
        ("forced_row_value_weight", 0.0),
        ("policy_loss_weight", 0.5),
        ("soft_target_temperature", 1.0),
        ("soft_target_min_legal_coverage", 0.0),
        ("mask_hidden_info", False),
    ],
)
def test_verify_training_report_refuses_command_bound_operator_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    report = _write_realized_report(payload)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_payload[field] = value
    report.write_text(json.dumps(report_payload), encoding="utf-8")
    with pytest.raises(executor.ExecutionError, match="command-bound operator drift"):
        executor.verify_training_report(path, report)


def test_verify_training_report_replays_supervision_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    report = _write_realized_report(payload)
    verified = executor.verify_training_report(path, report)
    assert verified["verified"] is True
    assert verified["supervision_contract_sha256"] == payload[
        "supervision_contract"
    ]["contract_sha256"]
    expected_active = payload["supervision_contract"]["policy_active_row_dose"][
        "reference_base_active_rows"
    ]
    assert verified["policy_active_row_dose"] == {
        "base": expected_active,
        "aux": 0,
        "total": expected_active,
    }


def test_verify_training_report_requires_objective_matched_component_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    report = _write_realized_report(payload)
    value = json.loads(report.read_text(encoding="utf-8"))
    del value["metrics"][0]["validation_objective_matched"]
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(executor.ExecutionError, match="objective-matched validation"):
        executor.verify_training_report(path, report)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy_base_active_rows", 1, "base policy-active dose"),
        ("policy_aux_active_rows", 1, "auxiliary policy-active dose"),
        ("policy_total_active_rows", 1, "does not add up"),
    ],
)
def test_verify_training_report_refuses_policy_active_dose_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
    message: str,
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    report = _write_realized_report(payload)
    result = json.loads(report.read_text())
    result[field] = value
    report.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(executor.ExecutionError, match=message):
        executor.verify_training_report(path, report)


def test_verify_training_report_refuses_scope_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    report = _write_realized_report(payload)
    value = json.loads(report.read_text())
    value["value_training_scope"]["component_ids"].append(arm.REPLAY_COMPONENT_ID)
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(executor.ExecutionError, match="scope provenance drift"):
        executor.verify_training_report(path, report)


def test_verify_refuses_manifest_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    payload["recipe"]["replay_forward_kl_weight"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(executor.ExecutionError, match="semantic digest drift"):
        executor.verify(path)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--soft-target-weight", "1.0"),
        ("--policy-aux-active-batch-size", "128"),
        ("--policy-kl-anchor-weight", "0.006"),
        ("--loser-sample-weight", "0.3"),
    ],
)
def test_verify_refuses_rehashed_supervision_operator_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    index = payload["command"].index(flag)
    payload["command"][index + 1] = value
    payload["command_sha256"] = arm._digest(payload["command"])
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = arm._digest(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(executor.ExecutionError, match="supervision drift"):
        executor.verify(path)


@pytest.mark.parametrize("missing", ["ack", "crop", "manifest"])
def test_verify_refuses_missing_event_history_command_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    command = payload["command"]
    if missing == "ack":
        index = command.index(arm.EVENT_HISTORY_ACK_FLAG)
        del command[index : index + 2]
    elif missing == "crop":
        command.remove(arm.EVENT_HISTORY_CROP_FLAG)
    else:
        payload.pop("event_history_training_contract")
    payload["command_sha256"] = arm._digest(command)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = arm._digest(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(executor.ExecutionError, match="event-history|crop flag"):
        executor.verify(path)


def test_explicit_execute_submits_exact_command_and_writes_append_only_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Running as unit.", stderr="")

    receipt = executor.execute(
        path, unit="a1-corrected-anchor-k3-test", runner=runner,
        conflict_probe=lambda: [],
    )
    assert receipt["diagnostic_only"] is True
    assert len(calls) == 1
    submitted = calls[0]
    assert submitted[:3] == ["sudo", "-n", "systemd-run"]
    assert "--property=RemainAfterExit=yes" in submitted
    assert "--collect" not in submitted
    assert "--property=LimitNOFILE=65536" in submitted
    separator = submitted.index("--")
    assert submitted[separator + 1 :] == payload["command"]
    out = Path(arm._option(payload["command"], "--checkpoint")).parent
    assert (out / "diagnostic-execution.claim.json").is_file()
    assert (out / "diagnostic-execution.receipt.json").is_file()
    events = [json.loads(line)["event"] for line in (
        out / "diagnostic-execution.status.jsonl"
    ).read_text().splitlines()]
    assert events == ["authorized", "submitted"]
    with pytest.raises(executor.ExecutionError, match="already exists"):
        executor.execute(
            path, unit="a1-corrected-anchor-k3-test", runner=runner,
            conflict_probe=lambda: [],
        )


def test_execute_refuses_conflicting_b200_compute_without_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    with pytest.raises(executor.ExecutionError, match="not idle"):
        executor.execute(
            path, unit="a1-corrected-anchor-k3-test",
            conflict_probe=lambda: ["1234, python"],
        )
    out = Path(arm._option(payload["command"], "--checkpoint")).parent
    assert not (out / "diagnostic-execution.claim.json").exists()
