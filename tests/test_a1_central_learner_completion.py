from __future__ import annotations

from tools import a1_central_learner_completion as completion
from tools import a1_aux_pair_coordinator as coordinator


SHA = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def _outputs() -> dict:
    return {
        "sampled_rows": 524288,
        "steps_completed": 128,
        "checkpoint_sha256": SHA,
        "optimizer_sidecar_sha256": SHA_B,
        "report_sha256": SHA_C,
        "sample_order_sha256": SHA_B,
        "row_set_sha256": SHA_C,
    }


def _base(stage: str) -> dict:
    return {
        "recipe": {"amp": "none"},
        "training_topology": {
            "world_size": 8,
            "local_batch_size": 512,
            "global_batch_size": 4096,
        },
        "central_learner_binding": {
            "stage": stage,
            "initializer_sha256": SHA,
            "sample_binding": {"sampler_identity_sha256": SHA_C},
        },
    }


def test_p1_result_is_derived_only_from_authority_and_outputs(monkeypatch):
    monkeypatch.setattr(coordinator, "_repo_tool_sha256", lambda _path: SHA)
    verified = _base("P1")
    verified["p1_arm_executor_authority"] = {
        "sweep_id": SHA_B,
        "arm_id": "K3",
        "arm": {
            "policy_kl_anchor_weight_decimal": "0.03",
            "effective_recipe_sha256": SHA_C,
        },
        "composite": {
            "payload_inventory_sha256": SHA,
            "validation_split_receipt_sha256": SHA_B,
        },
    }
    result = completion.derive_terminal_result(
        verified, {"outputs": _outputs()}
    )
    assert set(result) == {
        "schema_version",
        "status",
        "sweep_id",
        "arm_id",
        "policy_kl_anchor_weight_decimal",
        "initializer_sha256",
        "sampled_rows",
        "optimizer_steps",
        "world_size",
        "local_batch_size",
        "global_batch_size",
        "amp",
        "fresh_adam",
        "optimizer_restored",
        "effective_recipe_sha256",
        "payload_inventory_sha256",
        "validation_split_receipt_sha256",
        "sampler_identity_sha256",
        "sample_order_sha256",
        "checkpoint_sha256",
        "optimizer_sidecar_sha256",
        "report_sha256",
        "origin_tool_sha256",
    }
    assert result["arm_id"] == "K3"
    assert result["sampled_rows"] == 524288
    assert result["checkpoint_sha256"] == SHA


def test_aux_result_has_no_caller_metrics_or_topology_fields(monkeypatch):
    monkeypatch.setattr(coordinator, "_repo_tool_sha256", lambda _path: SHA)
    verified = _base(coordinator.ARM_TREATMENT)
    verified["aux_pair_executor_authority"] = {
        "arm": {
            "aux_subgoal_loss_weight_decimal": "0.007",
        },
        "aux_pair_contract": {
            "pair_id": SHA_B,
            "experiment_id": SHA_C,
            "joint": {
                "effective_recipe_sha256": SHA,
                "composite": {
                    "payload_inventory_sha256": SHA_B,
                    "validation_split_receipt_sha256": SHA_C,
                },
            },
        },
    }
    result = completion.derive_terminal_result(
        verified, {"outputs": _outputs()}
    )
    assert result["arm_id"] == coordinator.ARM_TREATMENT
    assert result["aux_subgoal_loss_weight_decimal"] == "0.007"
    assert "world_size" not in result
    assert "amp" not in result
    assert "loss" not in result


def test_final_result_reloads_parent_and_never_a_diagnostic_checkpoint(monkeypatch):
    monkeypatch.setattr(coordinator, "_repo_tool_sha256", lambda _path: SHA)
    verified = _base("FINAL")
    verified["final_replication_binding"] = {"bound": True}
    verified["final_replication_executor_authority"] = {
        "final_replication_authority": {
            "experiment_id": SHA_C,
            "final_replication_id": SHA_B,
            "selected_aux_decision": coordinator.ARM_CONTROL,
            "selected_aux_coefficient_decimal": "0",
            "initializer_authority": {
                "exact_current_parent_authority": {"checkpoint_sha256": SHA_C},
                "reference_warmup_terminal": None,
            },
            "component_routing_state_sha256": SHA,
            "sampling_state_sha256": SHA_B,
            "sampling_receipt": {"sampler_identity_sha256": SHA_C},
            "effective_recipe_sha256": SHA,
        }
    }
    result = completion.derive_terminal_result(
        verified, {"outputs": _outputs()}
    )
    assert result["initializer_parent_checkpoint_sha256"] == SHA_C
    assert result["diagnostic_checkpoint_loaded"] is False
    assert result["selected_aux_decision"] == coordinator.ARM_CONTROL
    assert result["shared_warmup_initializer_consumed"] is False
    assert result["full_gate_entry_eligible"] is True
    assert result["candidate_slot12_nonzero_count"] is None


def test_completion_refuses_noncentral_inputs():
    try:
        completion.derive_terminal_result({}, {"outputs": _outputs()})
    except completion.CompletionError as error:
        assert "central learner binding" in str(error)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("noncentral receipt was accepted")
