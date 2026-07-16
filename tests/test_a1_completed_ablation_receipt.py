from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import a1_b200_stage_c_learner_campaign as campaign
from tools import a1_one_dose_train as one_dose


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _feature_observability(*, observed_steps: int) -> dict[str, object]:
    module_row = {
        "mean_pre_clip_grad_norm": 0.4,
        "max_pre_clip_grad_norm": 0.6,
        "mean_parameter_delta_norm": 0.02,
        "mean_parameter_update_rms": 0.001,
        "mean_relative_parameter_delta": 0.03,
        "parameter_count": 8,
    }
    return {
        "schema_version": "module-optimizer-observability-v1",
        "observed_steps": observed_steps,
        "cadence_batches": campaign.TRAIN_DIAGNOSTIC_CADENCE_BATCHES,
        "norm_scope": "global_replicated",
        "modules": {
            module_name: dict(module_row)
            for module_name in campaign.FEATURE_SIGNAL_MODULES
        },
    }


def _checkpoint_dose_trajectory() -> dict[str, object]:
    checkpoints = []
    for step in campaign.CHECKPOINT_STEPS:
        observed_steps = step // campaign.TRAIN_DIAGNOSTIC_CADENCE_BATCHES
        checkpoints.append(
            {
                "schema_version": "train-bc-checkpoint-dose-telemetry-v1",
                "optimizer_step": step,
                "module_optimizer_observability": (
                    None
                    if observed_steps == 0
                    else _feature_observability(observed_steps=observed_steps)
                ),
                "feature_path_gradients": {
                    "public_card": {
                        "enabled": True,
                        "status": (
                            "observed"
                            if observed_steps > 0
                            else "awaiting_diagnostic_cadence"
                        ),
                    },
                    "meaningful_history": {
                        "enabled": True,
                        "status": (
                            "observed"
                            if observed_steps > 0
                            else "awaiting_diagnostic_cadence"
                        ),
                    },
                },
            }
        )
    return {
        "schema_version": "train-bc-checkpoint-dose-trajectory-v1",
        "checkpoint_steps": list(campaign.CHECKPOINT_STEPS),
        "checkpoints": checkpoints,
    }


def _completed_feature_signal_report() -> dict[str, object]:
    objective_observations = [
        {
            "available": True,
            "optimizer_step": step,
            "policy_trunk_grad_norm": 0.8,
            "policy_base_trunk_grad_norm": 0.6,
            "policy_aux_trunk_grad_norm": 0.2,
            "value_trunk_grad_norm": 0.3,
            "trunk_gradient_cosine": 0.1,
            "policy_base_aux_gradient_cosine": -0.1,
        }
        for step in (
            campaign.OBJECTIVE_GRADIENT_CADENCE_BATCHES,
            campaign.MAX_STEPS,
        )
    ]
    return {
        **campaign.EFFECTIVE_FEATURE_CONTRACT,
        "train_diagnostics_every_batches": (
            campaign.TRAIN_DIAGNOSTIC_CADENCE_BATCHES
        ),
        "module_optimizer_observability": _feature_observability(
            observed_steps=campaign.MINIMUM_FEATURE_SIGNAL_OBSERVATIONS
        ),
        "checkpoint_dose_trajectory": _checkpoint_dose_trajectory(),
        "objective_gradient_interference_every_batches": (
            campaign.OBJECTIVE_GRADIENT_CADENCE_BATCHES
        ),
        "objective_gradient_interference": {
            "cadence_batches": campaign.OBJECTIVE_GRADIENT_CADENCE_BATCHES,
            "observations": objective_observations,
        },
    }


def _cached_functional_payload(expected: dict[str, object]) -> dict[str, object]:
    shared_semantics = {
        "schema_version": "posthoc-shared-holdout-identity/v1",
        "memmap_fingerprint": expected["memmap"]["fingerprint"],
        "memmap_payload_inventory_sha256": expected["memmap"][
            "payload_inventory_sha256"
        ],
        "validation_manifest_semantic_sha256": expected["validation_manifest"][
            "manifest_sha256"
        ],
        "validation_game_seed_set_sha256": expected[
            "validation_game_seed_set_sha256"
        ],
        "validation_rows": 10,
    }
    shared = {
        **shared_semantics,
        "identity_sha256": campaign._value_sha256(shared_semantics),  # noqa: SLF001
        "training_report": expected["training_report"],
        "memmap": expected["memmap"],
        "validation_manifest": expected["validation_manifest"],
    }
    return {
        "schema_version": "posthoc-checkpoint-teacher-gap/v1",
        "arch": "entity_graph",
        "batch_size": 16,
        "validation_rows": 10,
        "validation_game_seed_set_sha256": expected[
            "validation_game_seed_set_sha256"
        ],
        "inputs": {
            key: expected[key]
            for key in (
                "checkpoint",
                "training_report",
                "memmap",
                "validation_manifest",
            )
        },
        "shared_holdout": shared,
        "paired_parent_teacher_gap": {
            "schema_version": campaign.PAIRED_PARENT_GAP_SCHEMA,
        },
    }


def test_stage_c_feature_learning_signal_is_evidence_backed() -> None:
    report = _completed_feature_signal_report()
    campaign._verify_completed_feature_learning_signal(report)  # noqa: SLF001

    disabled = copy.deepcopy(report)
    disabled["meaningful_public_history"] = False
    with pytest.raises(campaign.CampaignError, match="feature contract drifted"):
        campaign._verify_completed_feature_learning_signal(  # noqa: SLF001
            disabled
        )

    for module_name in campaign.FEATURE_SIGNAL_MODULES:
        missing_signal = copy.deepcopy(report)
        missing_signal["module_optimizer_observability"]["modules"][  # type: ignore[index]
            module_name
        ]["mean_parameter_update_rms"] = 0.0
        with pytest.raises(
            campaign.CampaignError,
            match=f"positive commissioned feature.*{module_name}",
        ):
            campaign._verify_completed_feature_learning_signal(  # noqa: SLF001
                missing_signal
            )

    no_observations = copy.deepcopy(report)
    no_observations["module_optimizer_observability"] = None
    with pytest.raises(campaign.CampaignError, match="observation cadence"):
        campaign._verify_completed_feature_learning_signal(  # noqa: SLF001
            no_observations
        )


def _functional_bindings(
    *,
    checkpoint: Path,
    report: Path,
    data: Path,
    manifest: Path,
    manifest_semantic: str,
) -> dict[str, object]:
    return {
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": campaign._file_sha256(checkpoint),  # noqa: SLF001
        },
        "training_report": {
            "path": str(report.resolve()),
            "sha256": campaign._file_sha256(report),  # noqa: SLF001
        },
        "memmap": {
            "path": str(data.resolve()),
            "fingerprint": "sha256:" + "a" * 64,
            "payload_inventory_sha256": "sha256:" + "b" * 64,
        },
        "validation_manifest": {
            "path": str(manifest.resolve()),
            "sha256": campaign._file_sha256(manifest),  # noqa: SLF001
            "manifest_sha256": manifest_semantic,
        },
        "validation_game_seed_set_sha256": "sha256:" + "c" * 64,
    }


def test_stage_c_rejects_cached_functional_for_stale_candidate(tmp_path: Path) -> None:
    output = tmp_path / "fingerprints"
    output.mkdir()
    checkpoint = tmp_path / "candidate.pt"
    report = tmp_path / "report.json"
    data = tmp_path / "data"
    manifest = tmp_path / "holdout.json"
    checkpoint.write_bytes(b"old candidate")
    report.write_text("{}\n", encoding="utf-8")
    data.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")
    old = _functional_bindings(
        checkpoint=checkpoint,
        report=report,
        data=data,
        manifest=manifest,
        manifest_semantic="sha256:" + "d" * 64,
    )
    cached = output / "step0004.functional.fresh-parent.json"
    _write_json(cached, _cached_functional_payload(old))
    checkpoint.write_bytes(b"current candidate")
    current = _functional_bindings(
        checkpoint=checkpoint,
        report=report,
        data=data,
        manifest=manifest,
        manifest_semantic="sha256:" + "d" * 64,
    )

    with pytest.raises(campaign.CampaignError, match="stale or misbound"):
        campaign._functional_artifact_path(  # noqa: SLF001
            output,
            4,
            allow_separate_parent=False,
            expected_bindings=current,
        )


def test_stage_c_rejects_cached_functional_for_stale_holdout(tmp_path: Path) -> None:
    output = tmp_path / "fingerprints"
    output.mkdir()
    checkpoint = tmp_path / "candidate.pt"
    report = tmp_path / "report.json"
    data = tmp_path / "data"
    manifest = tmp_path / "holdout.json"
    checkpoint.write_bytes(b"candidate")
    report.write_text("{}\n", encoding="utf-8")
    data.mkdir()
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    old = _functional_bindings(
        checkpoint=checkpoint,
        report=report,
        data=data,
        manifest=manifest,
        manifest_semantic="sha256:" + "d" * 64,
    )
    cached = output / "step0004.functional.fresh-parent.json"
    _write_json(cached, _cached_functional_payload(old))
    manifest.write_text('{"version": 2}\n', encoding="utf-8")
    current = _functional_bindings(
        checkpoint=checkpoint,
        report=report,
        data=data,
        manifest=manifest,
        manifest_semantic="sha256:" + "e" * 64,
    )

    with pytest.raises(campaign.CampaignError, match="stale or misbound"):
        campaign._functional_artifact_path(  # noqa: SLF001
            output,
            4,
            allow_separate_parent=False,
            expected_bindings=current,
        )


def test_stage_c_objective_gradient_signal_is_evidence_backed() -> None:
    report = _completed_feature_signal_report()
    campaign._verify_completed_objective_gradient_signal(report)  # noqa: SLF001

    no_observations = copy.deepcopy(report)
    no_observations["objective_gradient_interference"]["observations"] = []  # type: ignore[index]
    with pytest.raises(campaign.CampaignError, match="gradient observation cadence"):
        campaign._verify_completed_objective_gradient_signal(  # noqa: SLF001
            no_observations
        )

    missing_value = copy.deepcopy(report)
    missing_value["objective_gradient_interference"]["observations"][0][  # type: ignore[index]
        "value_trunk_grad_norm"
    ] = 0.0
    with pytest.raises(campaign.CampaignError, match="policy-base/AUX/value"):
        campaign._verify_completed_objective_gradient_signal(  # noqa: SLF001
            missing_value
        )


def test_completed_code_binding_authenticates_historical_checkout(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "historical").resolve()
    relative_kinds = {
        "tools/train_bc.py": "learner_code",
        "src/example_runtime.py": "runtime_code",
        "src/catan_zero/rl/entity_token_policy.py": "learner_code",
        "tools/a1_ddp_epoch_canary.py": "learner_code",
        "tools/a1_function_preserving_upgrade.py": "learner_code",
        "tools/a1_one_dose_train.py": "learner_code",
    }
    records = []
    for relative, kind in sorted(relative_kinds.items()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
        records.append(
            {
                "kind": kind,
                "relative_path": relative,
                "path": str(path),
                "sha256": one_dose._file_sha256(path),  # noqa: SLF001
            }
        )
    binding = {
        "schema_version": "a1-learner-ablation-code-binding-v1",
        "repository_root": str(root),
        "records": records,
    }
    binding["code_tree_sha256"] = one_dose._value_sha256(binding)  # noqa: SLF001
    lock = {
        "provenance": {
            "learner_code": [{"path": "/sealed/tools/train_bc.py"}],
            "runtime_code_tree": [{"path": "/sealed/src/example_runtime.py"}],
        }
    }

    assert (
        one_dose._verify_completed_ablation_code_binding(  # noqa: SLF001
            binding, lock=lock
        )
        == binding
    )
    (root / "tools" / "train_bc.py").write_text("drift\n", encoding="utf-8")
    with pytest.raises(one_dose.ExecutorError, match="bytes/path drift"):
        one_dose._verify_completed_ablation_code_binding(  # noqa: SLF001
            binding, lock=lock
        )


def test_generic_completed_loader_does_not_accept_matched_aux(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    payload: dict[str, object] = {
        "schema_version": one_dose.ABLATION_RECEIPT_SCHEMA,
        "status": "complete",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "learner_ablation": {"matched_aux_regularization": {}},
    }
    payload["receipt_sha256"] = one_dose._value_sha256(payload)  # noqa: SLF001
    _write_json(receipt, payload)
    with pytest.raises(one_dose.ExecutorError, match="not a generic"):
        one_dose._load_authenticated_completed_ablation_receipt(  # noqa: SLF001
            receipt
        )


def test_stage_c_run_adopts_authenticated_completed_receipt_without_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "one-dose.receipt.json"
    report_path = tmp_path / "train.report.json"
    execution_path = tmp_path / "execution.receipt.json"
    checkpoint = tmp_path / "candidate.pt"
    receipt_path.write_text("sealed receipt\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    aux_draws = (
        campaign.POLICY_AUX_ACTIVE_BATCH_SIZE * campaign.WORLD_SIZE * campaign.MAX_STEPS
    )
    unique_rows = 128
    report = {
        **_completed_feature_signal_report(),
        "value_trunk_grad_scale": 0.1,
        "freeze_modules": "",
        "training_information_surface": {},
        "policy_aux_active_rows": aux_draws,
        "policy_aux_unique_source_rows": unique_rows,
        "policy_aux_reuse_factor": aux_draws / unique_rows,
        "policy_base_active_rows": 7,
    }
    _write_json(report_path, report)
    authenticated = {
        "receipt_sha256": "sha256:" + "1" * 64,
        "outputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": one_dose._file_sha256(checkpoint),  # noqa: SLF001
            "report": str(report_path),
            "report_sha256": one_dose._file_sha256(report_path),  # noqa: SLF001
        },
    }
    plan = {
        "campaign_sha256": "sha256:" + "2" * 64,
        "policy_target_contract": {
            "selected_unique_training_roots": 256,
            "selected_unique_roots_total": 300,
        },
        "expected_artifacts": {
            "one_dose_receipt": str(receipt_path),
            "report": str(report_path),
            "terminal_checkpoint": str(checkpoint),
            "execution_receipt": str(execution_path),
        },
    }
    monkeypatch.setattr(campaign, "_verify_inputs", lambda _plan: None)
    monkeypatch.setattr(campaign, "_one_dose_command", lambda _plan: ["trainer"])
    monkeypatch.setattr(
        one_dose,
        "_load_authenticated_completed_ablation_receipt",
        lambda _path: copy.deepcopy(authenticated),
    )

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("completed receipt must not relaunch the trainer")

    monkeypatch.setattr(campaign.subprocess, "run", forbidden_run)
    result = campaign._run(plan, go=True)  # noqa: SLF001
    assert result["mode"] == "finalize-existing"
    assert (
        json.loads(execution_path.read_text(encoding="utf-8"))[
            "existing_completed_dose_adopted"
        ]
        is True
    )


def test_stage_c_fingerprint_binds_report_emitted_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "train.report.json"
    input_manifest = tmp_path / "selection-holdout.json"
    emitted_manifest = tmp_path / "train.report.validation_seeds.json"
    parent = tmp_path / "parent.pt"
    data = tmp_path / "data"
    output_root = tmp_path / "arm"
    data.mkdir()
    parent.write_bytes(b"parent")
    input_manifest.write_text("{}\n", encoding="utf-8")
    seed_set = "sha256:" + "3" * 64
    report = {
        **_completed_feature_signal_report(),
        "value_trunk_grad_scale": 0.1,
        "freeze_modules": "",
        "training_information_surface": {},
        "a1_contract_sha256": "sha256:" + "4" * 64,
        "data": str(data),
        "data_fingerprint": "sha256:" + "5" * 64,
        "validation_game_seed_manifest": str(emitted_manifest),
        "validation_game_seed_count": 2,
        "validation_game_seed_set_sha256": seed_set,
        "training_excluded_game_seed_count": 2,
        "training_excluded_game_seed_set_sha256": seed_set,
        "input_validation_game_seed_manifest_sha256": one_dose._file_sha256(  # noqa: SLF001
            input_manifest
        ),
    }
    emitted = {
        "schema_version": "train-validation-game-seeds-v1",
        "a1_contract_sha256": report["a1_contract_sha256"],
        "data": str(data),
        "data_fingerprint": report["data_fingerprint"],
        "validation_game_seed_count": 2,
        "validation_game_seed_set_sha256": seed_set,
        "training_excluded_game_seed_count": 2,
        "training_excluded_game_seed_set_sha256": seed_set,
        "input_validation_game_seed_manifest": str(input_manifest),
        "input_validation_game_seed_manifest_sha256": report[
            "input_validation_game_seed_manifest_sha256"
        ],
    }
    _write_json(report_path, report)
    _write_json(emitted_manifest, emitted)
    authority_path = tmp_path / "parent.authority.json"
    _write_json(
        authority_path,
        {
            "function_preserving_upgrade": {
                "upgraded_initializer": {"path": str(parent)}
            }
        },
    )
    for step in campaign.CHECKPOINT_STEPS:
        path = (
            output_root / "learner" / "candidate.pt"
            if step == campaign.MAX_STEPS
            else output_root / "learner" / f"candidate_step{step:04d}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(step).encode("ascii"))
    receipt_path = output_root / "learner" / "one-dose.receipt.json"
    receipt_path.write_text("sealed receipt\n", encoding="utf-8")
    terminal_checkpoint = output_root / "learner" / "candidate.pt"
    report["checkpoint"] = str(terminal_checkpoint)
    report["intermediate_checkpoints"] = [
        {
            "schema_version": "train-bc-intermediate-checkpoint-v1",
            "optimizer_step": step,
            "checkpoint": str(
                output_root / "learner" / f"candidate_step{step:04d}.pt"
            ),
            "checkpoint_sha256": one_dose._file_sha256(  # noqa: SLF001
                output_root / "learner" / f"candidate_step{step:04d}.pt"
            ),
            "size_bytes": (
                output_root / "learner" / f"candidate_step{step:04d}.pt"
            ).stat().st_size,
            "same_training_trajectory": True,
            "optimizer_sidecar": None,
        }
        for step in campaign.INTERMEDIATE_CHECKPOINT_STEPS
    ]
    _write_json(report_path, report)
    authenticated = {
        "receipt_sha256": "sha256:" + "6" * 64,
        "outputs": {
            "checkpoint": str(terminal_checkpoint),
            "checkpoint_sha256": one_dose._file_sha256(  # noqa: SLF001
                terminal_checkpoint
            ),
            "report": str(report_path),
            "report_sha256": one_dose._file_sha256(report_path),  # noqa: SLF001
        },
    }
    plan = {
        "output_root": str(output_root),
        "inputs": {
            "python": "/usr/bin/python3",
            "data": str(data),
            "validation_manifest": str(input_manifest),
            "independent_parent_authority": str(authority_path),
        },
        "expected_artifacts": {
            "one_dose_receipt": str(receipt_path),
            "report": str(report_path),
            "terminal_checkpoint": str(terminal_checkpoint),
        },
    }
    monkeypatch.setattr(campaign, "_verify_inputs", lambda _plan: None)
    monkeypatch.setattr(
        one_dose,
        "_load_authenticated_completed_ablation_receipt",
        lambda _path: copy.deepcopy(authenticated),
    )
    result = campaign._fingerprint(  # noqa: SLF001
        tmp_path / "campaign.json", plan, go=False, device="cpu"
    )
    assert result["validation_holdout"]["path"] == str(emitted_manifest)
    assert result["validation_holdout"]["validation_game_seed_set_sha256"] == seed_set
    assert all(
        command["functional"][command["functional"].index("--validation-manifest") + 1]
        == str(emitted_manifest)
        for command in result["commands"]
    )


def test_stage_c_fingerprint_refuses_frozen_adapter_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "train.report.json"
    checkpoint = tmp_path / "candidate.pt"
    receipt_path = tmp_path / "one-dose.receipt.json"
    checkpoint.write_bytes(b"checkpoint")
    receipt_path.write_text("sealed receipt\n", encoding="utf-8")
    report = {
        **_completed_feature_signal_report(),
        "value_trunk_grad_scale": 0.1,
        "freeze_modules": "",
        "training_information_surface": {
            "explicit_module_freeze": {
                "frozen_submodules": [
                    "meaningful_history_residual_gate",
                    "public_card_count_residual",
                ]
            }
        },
    }
    _write_json(report_path, report)
    authenticated = {
        "receipt_sha256": "sha256:" + "7" * 64,
        "outputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": one_dose._file_sha256(checkpoint),  # noqa: SLF001
            "report": str(report_path),
            "report_sha256": one_dose._file_sha256(report_path),  # noqa: SLF001
        },
    }
    plan = {
        "expected_artifacts": {
            "one_dose_receipt": str(receipt_path),
            "report": str(report_path),
            "terminal_checkpoint": str(checkpoint),
        }
    }
    monkeypatch.setattr(campaign, "_verify_inputs", lambda _plan: None)
    monkeypatch.setattr(
        one_dose,
        "_load_authenticated_completed_ablation_receipt",
        lambda _path: copy.deepcopy(authenticated),
    )

    with pytest.raises(
        campaign.CampaignError, match="did not keep both feature adapters trainable"
    ):
        campaign._fingerprint(  # noqa: SLF001
            tmp_path / "campaign.json", plan, go=False, device="cpu"
        )


def test_stage_c_selector_ignores_misleading_stored_generation_prior_closure() -> None:
    records = [
        {
            "step": 8,
            "feature_learning_signal_authenticated": False,
            "parent_kl": 0.01,
            "trunk_relative_l2": 0.01,
            # This looks excellent only against the stale generation prior.
            "legacy_stored_generation_prior_teacher_gap_closure": 0.99,
            "teacher_gap_closure": 0.99,
            "fresh_parent_teacher_gap_absolute_closure": -0.02,
            "fresh_parent_teacher_gap_relative_closure": -0.10,
            "value_quality_gate": {"passed": True},
        },
        {
            "step": 16,
            "feature_learning_signal_authenticated": True,
            "parent_kl": 0.02,
            "trunk_relative_l2": 0.02,
            # The legacy metric is deliberately worse, while the exact fresh
            # parent comparison proves real movement toward the teacher.
            "legacy_stored_generation_prior_teacher_gap_closure": -0.50,
            "teacher_gap_closure": -0.50,
            "fresh_parent_teacher_gap_absolute_closure": 0.04,
            "fresh_parent_teacher_gap_relative_closure": 0.25,
            "value_quality_gate": {"passed": True},
        },
    ]

    selected = campaign._select_fingerprint_winner(records)  # noqa: SLF001

    assert selected is not None
    assert selected["step"] == 16
    assert selected["fresh_parent_teacher_gap_relative_closure"] == pytest.approx(0.25)


def test_stage_c_selector_rejects_early_checkpoint_without_feature_signal() -> None:
    report = _completed_feature_signal_report()
    early = campaign._checkpoint_feature_learning_signal(  # noqa: SLF001
        report, step=8
    )
    commissioned = campaign._checkpoint_feature_learning_signal(  # noqa: SLF001
        report, step=16
    )
    records = [
        {
            "step": 8,
            "feature_learning_signal_authenticated": early["authenticated"],
            "parent_kl": 0.001,
            "trunk_relative_l2": 0.001,
            "fresh_parent_teacher_gap_absolute_closure": 0.5,
            "fresh_parent_teacher_gap_relative_closure": 0.5,
            "value_quality_gate": {"passed": True},
        },
        {
            "step": 16,
            "feature_learning_signal_authenticated": commissioned[
                "authenticated"
            ],
            "parent_kl": 0.02,
            "trunk_relative_l2": 0.02,
            "fresh_parent_teacher_gap_absolute_closure": 0.01,
            "fresh_parent_teacher_gap_relative_closure": 0.01,
            "value_quality_gate": {"passed": True},
        },
    ]

    selected = campaign._select_fingerprint_winner(records)  # noqa: SLF001

    assert early == {
        "authenticated": False,
        "reason": "awaiting_feature_optimizer_observation_cadence",
        "optimizer_step": 8,
    }
    assert commissioned["authenticated"] is True
    assert selected is not None
    assert selected["step"] == 16


def test_stage_c_rejects_policy_improvement_when_value_regresses() -> None:
    records = [
        {
            "step": 16,
            "feature_learning_signal_authenticated": True,
            "parent_kl": 0.002,
            "trunk_relative_l2": 0.001,
            "fresh_parent_teacher_gap_absolute_closure": 0.02,
            "fresh_parent_teacher_gap_relative_closure": 0.03,
            "value_quality_gate": {"passed": False},
        },
        {
            "step": 24,
            "feature_learning_signal_authenticated": True,
            "parent_kl": 0.02,
            "trunk_relative_l2": 0.002,
            "fresh_parent_teacher_gap_absolute_closure": 0.05,
            "fresh_parent_teacher_gap_relative_closure": 0.06,
            "value_quality_gate": {"passed": False},
        },
    ]

    assert campaign._select_fingerprint_winner(records) is None  # noqa: SLF001


def test_stage_c_accepts_earlier_mid_epoch_value_safe_checkpoint() -> None:
    records = [
        {
            "step": 16,
            "feature_learning_signal_authenticated": True,
            "parent_kl": 0.002,
            "trunk_relative_l2": 0.001,
            "fresh_parent_teacher_gap_absolute_closure": 0.02,
            "fresh_parent_teacher_gap_relative_closure": 0.03,
            "value_quality_gate": {"passed": True},
        },
        {
            "step": 24,
            "feature_learning_signal_authenticated": True,
            "parent_kl": 0.02,
            "trunk_relative_l2": 0.002,
            "fresh_parent_teacher_gap_absolute_closure": 0.05,
            "fresh_parent_teacher_gap_relative_closure": 0.06,
            "value_quality_gate": {"passed": False},
        },
    ]

    selected = campaign._select_fingerprint_winner(records)  # noqa: SLF001
    assert selected is not None
    assert selected["step"] == 16


def test_stage_c_value_gate_replays_b200_parent_comparison() -> None:
    def functional(candidate_value: float) -> dict:
        parent_value = 0.6638134101444333

        def projection(value: float) -> dict:
            return {
                "schema_version": campaign.VALUE_QUALITY_SCHEMA,
                "selection_authority": True,
                "surface": "same_reconstructed_holdout_and_value_weight_measure",
                "metric": "primary_value_loss",
                "metric_kind": "scalar_mse",
                "value": value,
                "scalar_value_mse_diagnostic": value,
                "value_weight_mass": 100.0,
            }

        return {
            "metrics": {
                "primary_value_loss": candidate_value,
                "primary_value_loss_kind": "scalar_mse",
                "scalar_value_mse_diagnostic": candidate_value,
                "value_loss": candidate_value,
                "loss_denominators": {"value_loss": 100.0},
            },
            "value_quality": projection(candidate_value),
            "parent_value_quality": projection(parent_value),
            "paired_parent_value_quality": {
                "schema_version": campaign.PAIRED_PARENT_VALUE_SCHEMA,
                "selection_authority": True,
                "surface": (
                    "same_holdout_same_objective_weights_fresh_exact_parent_forward"
                ),
                "metric": "primary_value_loss",
                "metric_kind": "scalar_mse",
                "value_weight_mass": 100.0,
                "parent_value": parent_value,
                "candidate_value": candidate_value,
                "candidate_minus_parent": candidate_value - parent_value,
            },
        }

    early = campaign._paired_value_quality(  # noqa: SLF001
        functional(0.6618989103258975),
        parent_functional=None,
        policy=campaign.VALUE_GATE_POLICY,
        max_absolute_regression=0.0,
    )
    regressed = campaign._paired_value_quality(  # noqa: SLF001
        functional(0.6842854594003515),
        parent_functional=None,
        policy=campaign.VALUE_GATE_POLICY,
        max_absolute_regression=0.0,
    )
    assert early["passed"] is True
    assert regressed["passed"] is False
    selected = campaign._select_fingerprint_winner(  # noqa: SLF001
        [
            {
                "step": 4,
                "feature_learning_signal_authenticated": False,
                "parent_kl": 0.0022811808808186323,
                "trunk_relative_l2": 0.0006498816281192034,
                "fresh_parent_teacher_gap_absolute_closure": 0.02064811311465664,
                "fresh_parent_teacher_gap_relative_closure": 0.025638264338820396,
                "value_quality_gate": early,
            },
            {
                "step": 32,
                "feature_learning_signal_authenticated": True,
                "parent_kl": 0.1130148750044209,
                "trunk_relative_l2": 0.010375720545963748,
                "fresh_parent_teacher_gap_absolute_closure": 0.1080338176634672,
                "fresh_parent_teacher_gap_relative_closure": 0.13414298727479415,
                "value_quality_gate": regressed,
            },
        ]
    )
    assert selected is None

    diagnostic = campaign._paired_value_quality(  # noqa: SLF001
        functional(0.6842854594003515),
        parent_functional=None,
        policy="diagnostic_record_only_allow_regression",
        max_absolute_regression=0.0,
    )
    assert diagnostic["passed"] is False
    assert diagnostic["selection_admitted"] is True
    assert diagnostic["promotion_authority"] is False


def test_checkpoint_signal_is_bound_to_exact_intermediate_bytes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate_step0016.pt"
    checkpoint.write_bytes(b"step-16-original")
    terminal = tmp_path / "candidate.pt"
    terminal.write_bytes(b"terminal")
    report = _completed_feature_signal_report()
    report["checkpoint"] = str(terminal)
    report["intermediate_checkpoints"] = [
        {
            "schema_version": "train-bc-intermediate-checkpoint-v1",
            "optimizer_step": 16,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": one_dose._file_sha256(checkpoint),  # noqa: SLF001
            "size_bytes": checkpoint.stat().st_size,
            "same_training_trajectory": True,
            "optimizer_sidecar": None,
        }
    ]

    binding = campaign._authenticate_checkpoint_snapshot(  # noqa: SLF001
        report,
        step=16,
        checkpoint=checkpoint,
        terminal_checkpoint=terminal,
    )
    assert binding["checkpoint_sha256"] == one_dose._file_sha256(  # noqa: SLF001
        checkpoint
    )

    checkpoint.write_bytes(b"step-16-replaced")
    with pytest.raises(campaign.CampaignError, match="bytes differ"):
        campaign._authenticate_checkpoint_snapshot(  # noqa: SLF001
            report,
            step=16,
            checkpoint=checkpoint,
            terminal_checkpoint=terminal,
        )


def test_stage_c_rejects_paired_value_block_that_contradicts_raw_metrics() -> None:
    candidate_value = 0.6618989103258975
    parent_value = 0.6638134101444333

    def projection(value: float) -> dict:
        return {
            "schema_version": campaign.VALUE_QUALITY_SCHEMA,
            "selection_authority": True,
            "surface": "same_reconstructed_holdout_and_value_weight_measure",
            "metric": "primary_value_loss",
            "metric_kind": "scalar_mse",
            "value": value,
            "scalar_value_mse_diagnostic": value,
            "value_weight_mass": 100.0,
        }

    functional = {
        "metrics": {
            "primary_value_loss": candidate_value,
            "primary_value_loss_kind": "scalar_mse",
            "scalar_value_mse_diagnostic": candidate_value,
            "value_loss": candidate_value,
            "loss_denominators": {"value_loss": 100.0},
        },
        "value_quality": projection(candidate_value),
        "parent_value_quality": projection(parent_value),
        "paired_parent_value_quality": {
            "schema_version": campaign.PAIRED_PARENT_VALUE_SCHEMA,
            "selection_authority": True,
            "surface": (
                "same_holdout_same_objective_weights_fresh_exact_parent_forward"
            ),
            "metric": "primary_value_loss",
            "metric_kind": "scalar_mse",
            "value_weight_mass": 100.0,
            "parent_value": parent_value,
            # Internally consistent, but contradicts candidate projection/raw metrics.
            "candidate_value": 0.70,
            "candidate_minus_parent": 0.70 - parent_value,
        },
    }

    with pytest.raises(campaign.CampaignError, match="inconsistent"):
        campaign._paired_value_quality(  # noqa: SLF001
            functional,
            parent_functional=None,
            policy=campaign.VALUE_GATE_POLICY,
            max_absolute_regression=0.0,
        )


def test_stage_c_reuses_authenticated_transitional_fresh_parent_evidence() -> None:
    parent_kl = 0.8053631416613536
    candidate_kl = 0.7514737349905842
    stored_prior_kl = 0.6984388223118919
    absolute = parent_kl - candidate_kl
    functional = {
        "teacher_gap": {
            "active_policy_teacher_gap_rows": 100,
            "active_policy_kl_target_model_mean": candidate_kl,
            "active_policy_kl_target_prior_mean": stored_prior_kl,
            "active_policy_teacher_gap_closure": 1.0 - candidate_kl / stored_prior_kl,
        },
        "parent_teacher_gap": {
            "active_policy_teacher_gap_rows": 100,
            "active_policy_kl_target_model_mean": parent_kl,
            "active_policy_kl_target_prior_mean": stored_prior_kl,
            "active_policy_teacher_gap_closure": 1.0 - parent_kl / stored_prior_kl,
        },
        "paired_parent_teacher_gap": {
            "schema_version": campaign.TRANSITIONAL_PAIRED_PARENT_GAP_SCHEMA,
            "surface": "same_holdout_same_targets_fresh_exact_parent_forward",
            "rows": 100,
            "parent_active_policy_kl_target_model_mean": parent_kl,
            "candidate_active_policy_kl_target_model_mean": candidate_kl,
            "absolute_target_kl_improvement": absolute,
            "relative_teacher_gap_closure": absolute / parent_kl,
            "improved_over_exact_parent": True,
            "stored_prior_active_policy_kl_target_mean": stored_prior_kl,
            "stored_prior_closure_is_legacy_diagnostic_only": True,
        },
    }

    result = campaign._fresh_parent_teacher_gap(functional)  # noqa: SLF001

    assert result["evidence_schema_version"] == (
        campaign.TRANSITIONAL_PAIRED_PARENT_GAP_SCHEMA
    )
    assert result["relative_closure"] == pytest.approx(0.0669131767)
    assert result["legacy_stored_prior_closure"] < 0.0
