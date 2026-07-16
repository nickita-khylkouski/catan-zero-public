from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_current_science_contract as current_science
from tools import a1_scratch_value_routing_campaign as campaign
from tools import train_bc


def _verified(tmp_path: Path) -> dict:
    data = tmp_path / "composite.json"
    lock = tmp_path / "staged.lock.json"
    data.write_text("{}\n", encoding="utf-8")
    lock.write_text("{}\n", encoding="utf-8")
    recipe = current_science.learner_training_recipe()
    return {
        "data_kind": "test",
        "lock_path": lock,
        "data_path": data,
        "recipe": recipe,
        "logical_recipe": recipe,
        "initialization": current_science.learner_initialization(),
        "model_construction": current_science.learner_model_construction(),
        "execution_topology": current_science.learner_execution_topology(),
        "accepted_policy_target_identity_sha256": "sha256:" + "1" * 64,
        "policy_target_quality_admission": {
            "path": str((tmp_path / "quality.json").resolve()),
            "file_sha256": "sha256:" + "2" * 64,
            "receipt_sha256": "sha256:" + "3" * 64,
            "identity_sha256": "sha256:" + "4" * 64,
            "metrics": {"admitted": True},
        },
        "trainer_authority": {
            "path": str((campaign.REPO_ROOT / "tools/train_bc.py").resolve())
        },
        "event_history_training_contract": {
            "empty_payload_inventory_acknowledgements": [],
            "training_event_history_trainable": True,
        },
    }


def _python(tmp_path: Path) -> Path:
    path = tmp_path / "python"
    path.write_bytes(b"python")
    path.chmod(0o755)
    return path


def _stub_scratch_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign.scratch,
        "_scratch_plan_authority",
        lambda _verified: {
            "schema_version": "a1-coherent-scratch-plan-authority-v2",
            "test": True,
        },
    )


def test_generic_ablation_reports_unit_value_scale_without_changing_history() -> None:
    parser = train_bc.build_parser()
    required = [
        "--arch",
        "entity_graph",
        "--data",
        "data.json",
        "--checkpoint",
        "candidate.pt",
        "--report",
        "report.json",
    ]
    historical = parser.parse_args(required)
    historical_effective = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        historical,
        {"world_size": 1, "rank": 0, "local_rank": 0, "enabled": False},
    )
    assert "value_trunk_grad_scale" not in historical_effective

    treatment = parser.parse_args(
        [
            *required,
            "--a1-learner-ablation-id",
            "scratch-value-routing-v100",
            "--value-trunk-grad-scale",
            "1.0",
        ]
    )
    treatment_effective = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        treatment,
        {"world_size": 1, "rank": 0, "local_rank": 0, "enabled": False},
    )
    assert treatment_effective["value_trunk_grad_scale"] == pytest.approx(1.0)


def test_bounded_diagnostic_authority_allows_only_exact_short_arm() -> None:
    parser = train_bc.build_parser()
    science = {
        "learner_training_recipe": current_science.learner_training_recipe(),
        "learner_execution_topology": (current_science.learner_execution_topology()),
    }
    code_sha = "sha256:" + "a" * 64
    authority = {
        "schema_version": "a1-scratch-bounded-diagnostic-authority-v1",
        "campaign_id": "scratch-value-routing-v0-v25-v100",
        "arm_id": "V100",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "exact_max_steps": True,
        "max_steps": 128,
        "epochs": 1,
        "checkpoint_steps": [8, 16, 32, 64],
        "value_trunk_grad_scale": 1.0,
        "source_recipe_sha256": train_bc._canonical_json_sha256(  # noqa: SLF001
            science["learner_training_recipe"]
        ),
        "source_execution_topology_sha256": (
            train_bc._canonical_json_sha256(  # noqa: SLF001
                science["learner_execution_topology"]
            )
        ),
        "code_tree_sha256": code_sha,
    }
    args = parser.parse_args(
        [
            "--data",
            "data.json",
            "--checkpoint",
            "candidate.pt",
            "--report",
            "report.json",
            "--epochs",
            "1",
            "--max-steps",
            "128",
            "--exact-max-steps",
            "--checkpoint-steps",
            "8,16,32,64",
            "--value-trunk-grad-scale",
            "1.0",
            "--a1-learner-ablation-id",
            "scratch-value-routing-v100",
            "--a1-ablation-code-tree-sha256",
            code_sha,
        ]
    )
    validated = train_bc._validate_a1_scratch_diagnostic_authority(  # noqa: SLF001
        json.dumps(authority),
        args=args,
        science=science,
    )
    train_bc._require_a1_scratch_execution_schedule(  # noqa: SLF001
        science["learner_execution_topology"],
        diagnostic_authority=validated,
    )

    authority["max_steps"] = 257
    with pytest.raises(SystemExit, match="value drift"):
        train_bc._validate_a1_scratch_diagnostic_authority(  # noqa: SLF001
            json.dumps(authority),
            args=args,
            science=science,
        )


def test_derive_arms_changes_only_value_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_scratch_authority(monkeypatch)
    verified = _verified(tmp_path)
    code_binding = campaign._code_binding()  # noqa: SLF001
    root = tmp_path / "campaign"
    arms = {
        arm_id: campaign._derive_arm(  # noqa: SLF001
            verified=verified,
            python=_python(tmp_path),
            root=root,
            arm_id=arm_id,
            scale=scale,
            diagnostic_max_steps=128,
            code_binding=code_binding,
        )
        for arm_id, scale in campaign.ARMS.items()
    }

    baseline = arms["V25"]["effective_recipe"]
    for arm_id, scale in campaign.ARMS.items():
        campaign._assert_one_axis_recipe(  # noqa: SLF001
            baseline, arms[arm_id]["effective_recipe"], scale=scale
        )
        command = arms[arm_id]["command"]
        assert "--init-checkpoint" not in command
        assert "--grow-from-checkpoint" not in command
        assert "--resume-optimizer" not in command
        assert command.count("--no-resume-optimizer") == 1
        assert arms[arm_id]["independent_initialization"] == {
            "mode": "from_scratch",
            "seed": baseline["seed"],
            "fresh_optimizer": True,
            "candidate_chaining": False,
        }
    assert arms["V0"]["causal_recipe_delta"] == {
        "field": "value_trunk_grad_scale",
        "baseline": 0.25,
        "treatment": 0.0,
    }
    assert arms["V100"]["causal_recipe_delta"] == {
        "field": "value_trunk_grad_scale",
        "baseline": 0.25,
        "treatment": 1.0,
    }


def test_prepare_and_verify_emit_nonpromotable_machine_readable_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_scratch_authority(monkeypatch)
    verified = _verified(tmp_path)
    monkeypatch.setattr(
        campaign.scratch,
        "verify_inputs",
        lambda **_kwargs: verified,
    )
    output_root = tmp_path / "campaign"
    plan_path = tmp_path / "campaign.plan.json"

    planned = campaign.prepare(
        lock=verified["lock_path"],
        data=verified["data_path"],
        composite_build_receipt=tmp_path / "build.json",
        policy_target_quality_receipt=tmp_path / "quality.json",
        output_root=output_root,
        plan_path=plan_path,
        python=_python(tmp_path),
        diagnostic_max_steps=128,
    )
    replay = campaign.verify(plan_path, require_fresh_outputs=True)

    assert planned["promotion_eligible"] is False
    assert planned["evaluation_eligible"] is False
    assert planned["causal_axis"] == "value_trunk_grad_scale"
    assert planned["diagnostic_max_steps"] == 128
    assert set(replay["arms"]) == {"V0", "V25", "V100"}
    assert replay["arms"]["V100"]["effective_recipe"][
        "value_trunk_grad_scale"
    ] == pytest.approx(1.0)
    assert all(arm["promotion_eligible"] is False for arm in replay["arms"].values())


@pytest.mark.parametrize("max_steps", [0, 257])
def test_prepare_refuses_unbounded_or_full_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_steps: int,
) -> None:
    _stub_scratch_authority(monkeypatch)
    verified = _verified(tmp_path)
    monkeypatch.setattr(
        campaign.scratch,
        "verify_inputs",
        lambda **_kwargs: verified,
    )
    with pytest.raises(campaign.ValueRoutingCampaignError, match=r"\[1, 256\]"):
        campaign.prepare(
            lock=verified["lock_path"],
            data=verified["data_path"],
            composite_build_receipt=tmp_path / "build.json",
            policy_target_quality_receipt=tmp_path / "quality.json",
            output_root=tmp_path / "campaign",
            plan_path=tmp_path / "campaign.plan.json",
            python=_python(tmp_path),
            diagnostic_max_steps=max_steps,
        )
