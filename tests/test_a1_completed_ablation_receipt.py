from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import a1_b200_stage_c_learner_campaign as campaign
from tools import a1_one_dose_train as one_dose


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


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
        "value_trunk_grad_scale": 0.1,
        "training_information_surface": {
            "explicit_module_freeze": {
                "frozen_groups": sorted(campaign.FROZEN_ADAPTER_GROUPS.split(",")),
                "frozen_submodules": [
                    "meaningful_history_residual_gate",
                    "public_card_count_residual",
                ],
                "all_require_grad_false": True,
                "optimizer_excluded_parameter_tensors": 2,
            }
        },
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
    plan = {
        "output_root": str(output_root),
        "inputs": {
            "python": "/usr/bin/python3",
            "data": str(data),
            "validation_manifest": str(input_manifest),
            "independent_parent_authority": str(authority_path),
        },
        "expected_artifacts": {"report": str(report_path)},
    }
    monkeypatch.setattr(campaign, "_verify_inputs", lambda _plan: None)
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


def test_stage_c_selector_ignores_misleading_stored_generation_prior_closure() -> None:
    records = [
        {
            "step": 8,
            "parent_kl": 0.01,
            "trunk_relative_l2": 0.01,
            # This looks excellent only against the stale generation prior.
            "legacy_stored_generation_prior_teacher_gap_closure": 0.99,
            "teacher_gap_closure": 0.99,
            "fresh_parent_teacher_gap_absolute_closure": -0.02,
            "fresh_parent_teacher_gap_relative_closure": -0.10,
        },
        {
            "step": 12,
            "parent_kl": 0.02,
            "trunk_relative_l2": 0.02,
            # The legacy metric is deliberately worse, while the exact fresh
            # parent comparison proves real movement toward the teacher.
            "legacy_stored_generation_prior_teacher_gap_closure": -0.50,
            "teacher_gap_closure": -0.50,
            "fresh_parent_teacher_gap_absolute_closure": 0.04,
            "fresh_parent_teacher_gap_relative_closure": 0.25,
        },
    ]

    selected = campaign._select_fingerprint_winner(records)  # noqa: SLF001

    assert selected is not None
    assert selected["step"] == 12
    assert selected["fresh_parent_teacher_gap_relative_closure"] == pytest.approx(0.25)


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
