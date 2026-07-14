#!/usr/bin/env python3
"""Derive central learner terminals from authenticated one-dose artifacts.

The coordinator deliberately accepts a narrow result object, but operators must
never be allowed to author that object.  This module is the only adapter between
``a1_one_dose_train`` and the P1/AUX/FINAL terminal transitions: it replays the
immutable one-dose receipt, its terminal claim, command/environment, report,
checkpoint, optimizer sidecar, and training-progress marker, then derives the
coordinator result from those artifacts.

The functions accept an already reconstructed ``verified`` input authority.  A
normal launch has that object in memory; crash recovery reconstructs the same
object from the published executor authority before calling this module.
"""

from __future__ import annotations

import copy
from pathlib import Path
import stat
from typing import Any, Mapping

from tools import a1_aux_pair_coordinator as coordinator
from tools import a1_one_dose_train as one_dose
from tools import a1_scientific_evidence as scientific_evidence


class CompletionError(RuntimeError):
    """Raised when training artifacts cannot authorize a central terminal."""


def _expected_receipt_schema(verified: Mapping[str, Any]) -> str:
    if isinstance(verified.get("central_learner_binding"), dict):
        return one_dose.CENTRAL_RECEIPT_SCHEMA
    if verified.get("retry_contract") is not None:
        return one_dose.RETRY_RECEIPT_SCHEMA
    if verified.get("learner_ablation") is not None:
        return one_dose.ABLATION_RECEIPT_SCHEMA
    if verified.get("function_preserving_upgrade") is not None:
        return one_dose.UPGRADE_RECEIPT_SCHEMA
    return one_dose.RECEIPT_SCHEMA


def _canonical_receipt(path: Path) -> tuple[Path, dict[str, Any]]:
    lexical = path.expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise CompletionError("one-dose receipt must be a regular file")
    try:
        resolved = lexical.resolve(strict=True)
        payload, _file_sha256, _identity = (
            coordinator._stable_read_json_artifact(  # noqa: SLF001
                resolved, where="completed one-dose receipt"
            )
        )
    except (OSError, coordinator.CoordinatorError) as error:
        raise CompletionError(f"cannot load one-dose receipt: {error}") from error
    if resolved.stat().st_mode & 0o222:
        raise CompletionError("completed one-dose receipt must be immutable")
    if not isinstance(payload, dict):
        raise CompletionError("one-dose receipt is not an object")
    unsigned = dict(payload)
    stated = unsigned.pop("receipt_sha256", None)
    if stated != one_dose._value_sha256(unsigned):  # noqa: SLF001
        raise CompletionError("one-dose receipt digest drift")
    return resolved, payload


def authenticate_completed_receipt(
    receipt_path: Path,
    *,
    verified: dict[str, Any],
) -> dict[str, Any]:
    """Replay a completed receipt and all realized training outputs exactly."""

    resolved, payload = _canonical_receipt(receipt_path)
    if (
        payload.get("schema_version") != _expected_receipt_schema(verified)
        or payload.get("status") != "complete"
        or payload.get("returncode") != 0
        or payload.get("failure") is not None
    ):
        raise CompletionError("one-dose receipt schema/status drift")

    claim_ref = payload.get("claim")
    if not isinstance(claim_ref, str) or not claim_ref:
        raise CompletionError("one-dose receipt has no terminal claim")
    claim_lexical = Path(claim_ref).expanduser()
    if claim_lexical.is_symlink() or not claim_lexical.is_file():
        raise CompletionError("one-dose terminal claim must be a regular file")
    try:
        claim_path = claim_lexical.resolve(strict=True)
        claim_stat = claim_path.stat()
    except OSError as error:
        raise CompletionError(f"cannot resolve one-dose claim: {error}") from error
    if stat.S_IMODE(claim_stat.st_mode) != 0o444 or (
        claim_stat.st_uid != resolved.stat().st_uid
    ):
        raise CompletionError("one-dose terminal claim mode/owner drift")
    expected_claim_path = one_dose._claim_path(verified)  # noqa: SLF001
    if claim_path != expected_claim_path:
        raise CompletionError("one-dose terminal claim is outside its durable anchor")
    contract_sha = str(verified.get("contract_sha256", ""))
    claim_identity = str(verified.get("claim_identity_sha256", contract_sha))
    try:
        claim, _claim_file_sha, _claim_inode = (
            coordinator._stable_read_immutable_json(  # noqa: SLF001
                claim_path, where="completed one-dose terminal claim"
            )
        )
    except coordinator.CoordinatorError as error:
        raise CompletionError(f"one-dose claim replay refused: {error}") from error
    expected_claim_schema = one_dose.CENTRAL_CLAIM_SCHEMA
    if (
        claim.get("schema_version") != expected_claim_schema
        or claim.get("contract_sha256") != contract_sha
        or claim.get("claim_identity_sha256") != claim_identity
    ):
        raise CompletionError("one-dose terminal claim schema/identity drift")
    target_ref = claim.get("receipt_target")
    if not isinstance(target_ref, str) or not target_ref:
        raise CompletionError("one-dose terminal claim lacks its receipt target")
    target_lexical = Path(target_ref).expanduser()
    if target_lexical.is_symlink():
        raise CompletionError("one-dose terminal receipt target is a symlink")
    try:
        target = target_lexical.resolve(strict=True)
    except OSError as error:
        raise CompletionError(f"cannot resolve terminal receipt target: {error}") from error
    if (
        claim.get("status") != "complete"
        or payload.get("claim_state_sha256") != claim.get("state_sha256")
        or target != resolved
    ):
        raise CompletionError("one-dose claim/receipt terminal identity drift")

    claim_projection = dict(claim)
    claim_projection.pop("state_sha256", None)
    claim_projection.pop("receipt_target", None)
    claim_projection["schema_version"] = payload["schema_version"]
    receipt_projection = dict(payload)
    receipt_projection.pop("receipt_sha256", None)
    receipt_projection.pop("claim", None)
    receipt_projection.pop("claim_state_sha256", None)
    if claim_projection != receipt_projection:
        raise CompletionError("one-dose receipt differs from its terminal claim")

    published = verified.get("central_published_executor_authority")
    if not isinstance(published, dict) or not isinstance(published.get("path"), str):
        raise CompletionError("central completion lacks published executor authority")
    try:
        replayed_published = coordinator.verify_published_executor_authority(
            Path(published["path"])
        )
    except coordinator.CoordinatorError as error:
        raise CompletionError(f"published executor authority replay refused: {error}") from error
    if replayed_published != published:
        raise CompletionError("published executor authority wrapper drift")

    canary = payload.get("ddp_canary")
    canary_path = canary.get("path") if isinstance(canary, dict) else None
    started_unix_ns = claim.get("started_unix_ns")
    if not isinstance(canary_path, str) or type(started_unix_ns) is not int:
        raise CompletionError("completed central dose lacks canary/start evidence")
    try:
        replayed_canary = one_dose._verify_ddp_canary_receipt(  # noqa: SLF001
            Path(canary_path), reference_time_ns=started_unix_ns
        )
    except one_dose.ExecutorError as error:
        raise CompletionError(f"DDP canary replay refused: {error}") from error
    if replayed_canary != canary or verified.get("ddp_canary") != canary:
        raise CompletionError("DDP canary differs at authenticated claim time")

    outputs = payload.get("outputs")
    command = payload.get("command")
    if not isinstance(outputs, dict) or not isinstance(command, list) or not command:
        raise CompletionError("one-dose receipt lacks command/output artifacts")
    if not all(isinstance(item, str) and item for item in command):
        raise CompletionError("one-dose command is malformed")
    checkpoint = Path(str(outputs.get("checkpoint", "")))
    report = Path(str(outputs.get("report", "")))
    optimizer = Path(str(outputs.get("optimizer_sidecar", "")))
    progress = Path(str(outputs.get("training_progress", "")))
    for where, path in (
        ("checkpoint", checkpoint),
        ("optimizer", optimizer),
        ("progress", progress),
        ("report", report),
    ):
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            raise CompletionError(f"one-dose {where} output is not canonical/regular")
    canonical_command = one_dose.build_train_command(
        verified,
        python=Path(command[0]),
        checkpoint=checkpoint,
        report=report,
    )
    environment = one_dose._child_environment(  # noqa: SLF001
        one_dose._selected_gpus(verified, fallback_gpu=0)  # noqa: SLF001
    )
    execution_binding = one_dose._execution_binding(  # noqa: SLF001
        command=canonical_command, environment=environment
    )
    input_binding = one_dose._input_binding(verified)  # noqa: SLF001
    training_transaction_sha256 = one_dose._training_transaction_sha256(  # noqa: SLF001
        command=canonical_command, input_binding=input_binding
    )
    if (
        command != canonical_command
        or payload.get("command_sha256")
        != one_dose._value_sha256(canonical_command)  # noqa: SLF001
        or payload.get("execution_binding") != execution_binding
        or payload.get("input_binding") != input_binding
        or payload.get("training_transaction_sha256")
        != training_transaction_sha256
        or payload.get("trainer_authority")
        != verified.get("trainer_authority")
        or payload.get("lock_verifier_authority")
        != verified.get("lock_verifier_authority")
    ):
        raise CompletionError("one-dose command/environment/input replay drift")
    try:
        reverified = one_dose._verify_training_outputs(  # noqa: SLF001
            checkpoint=checkpoint,
            report=report,
            verified=verified,
            execution_binding=execution_binding,
            command=canonical_command,
        )
    except one_dose.ExecutorError as error:
        raise CompletionError(f"one-dose output replay refused: {error}") from error
    if reverified != outputs:
        raise CompletionError("one-dose output artifact replay drift")
    return payload


def _common_result(
    verified: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    outputs = receipt["outputs"]
    recipe = verified["recipe"]
    topology = verified["training_topology"]
    return {
        "sampled_rows": outputs["sampled_rows"],
        "optimizer_steps": outputs["steps_completed"],
        "world_size": topology["world_size"],
        "local_batch_size": topology["local_batch_size"],
        "global_batch_size": topology["global_batch_size"],
        "amp": recipe["amp"],
        "fresh_adam": True,
        "optimizer_restored": False,
        "checkpoint_sha256": outputs["checkpoint_sha256"],
        "optimizer_sidecar_sha256": outputs["optimizer_sidecar_sha256"],
        "report_sha256": outputs["report_sha256"],
        "origin_tool_sha256": coordinator._repo_tool_sha256(  # noqa: SLF001
            "tools/a1_one_dose_train.py"
        ),
    }


def derive_terminal_result(
    verified: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive the exact coordinator terminal result; accept no metric inputs."""

    central = verified.get("central_learner_binding")
    if not isinstance(central, dict):
        raise CompletionError("training inputs have no central learner binding")
    stage = central.get("stage")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise CompletionError("completed receipt has no output artifacts")
    common = _common_result(verified, receipt)
    if stage == "P1":
        authority = verified.get("p1_arm_executor_authority")
        if not isinstance(authority, dict):
            raise CompletionError("P1 completion lacks executor authority")
        arm = authority["arm"]
        composite = authority["composite"]
        return {
            "schema_version": "a1-p1-central-arm-result-v1",
            "status": "complete",
            "sweep_id": authority["sweep_id"],
            "arm_id": authority["arm_id"],
            "policy_kl_anchor_weight_decimal": arm[
                "policy_kl_anchor_weight_decimal"
            ],
            "initializer_sha256": central["initializer_sha256"],
            **common,
            "effective_recipe_sha256": arm["effective_recipe_sha256"],
            "payload_inventory_sha256": composite["payload_inventory_sha256"],
            "validation_split_receipt_sha256": composite[
                "validation_split_receipt_sha256"
            ],
            "sampler_identity_sha256": central["sample_binding"][
                "sampler_identity_sha256"
            ],
            "sample_order_sha256": outputs["sample_order_sha256"],
        }
    if stage in {coordinator.ARM_CONTROL, coordinator.ARM_TREATMENT}:
        authority = verified.get("aux_pair_executor_authority")
        if not isinstance(authority, dict):
            raise CompletionError("AUX completion lacks executor authority")
        pair = authority["aux_pair_contract"]
        arm = authority["arm"]
        composite = pair["joint"]["composite"]
        # AUX terminals intentionally omit topology fields; the pair contract
        # already fixes them and the authenticated receipt still proves them.
        for field in ("world_size", "local_batch_size", "global_batch_size", "amp"):
            common.pop(field)
        return {
            "schema_version": "a1-aux-joint-arm-result-v1",
            "status": "complete",
            "pair_id": pair["pair_id"],
            "arm_id": stage,
            "aux_subgoal_loss_weight_decimal": arm[
                "aux_subgoal_loss_weight_decimal"
            ],
            "initializer_sha256": central["initializer_sha256"],
            **common,
            "effective_recipe_sha256": pair["joint"]["effective_recipe_sha256"],
            "payload_inventory_sha256": composite["payload_inventory_sha256"],
            "validation_split_receipt_sha256": composite[
                "validation_split_receipt_sha256"
            ],
            "sampler_identity_sha256": central["sample_binding"][
                "sampler_identity_sha256"
            ],
            "sample_order_sha256": outputs["sample_order_sha256"],
        }
    if stage == "FINAL":
        authority = verified.get("final_replication_executor_authority")
        binding = verified.get("final_replication_binding")
        if not isinstance(authority, dict) or not isinstance(binding, dict):
            raise CompletionError("FINAL completion lacks executor authority/binding")
        final = authority["final_replication_authority"]
        initializer = final["initializer_authority"]
        treatment = final["selected_aux_decision"] == coordinator.ARM_TREATMENT
        expected_warmup = (
            initializer["reference_warmup_terminal"]["result"][
                "warmed_checkpoint_sha256"
            ]
            if treatment
            else None
        )
        sampling = final["sampling_receipt"]
        return {
            "schema_version": "a1-final-replication-result-v1",
            "status": "complete",
            "final_replication_id": final["final_replication_id"],
            "initializer_parent_checkpoint_sha256": initializer[
                "exact_current_parent_authority"
            ]["checkpoint_sha256"],
            "diagnostic_checkpoint_loaded": False,
            "selected_aux_decision": final["selected_aux_decision"],
            "selected_aux_coefficient_decimal": final[
                "selected_aux_coefficient_decimal"
            ],
            "pointer_upgrade_replayed": False,
            "pointer_upgrade_initializer_sha256": None,
            "warmed_checkpoint_sha256": expected_warmup,
            "shared_warmup_initializer_consumed": treatment,
            # Filled from replayed tensor receipts in complete_central_terminal.
            "initializer_slot12_zero_receipt_state_sha256": None,
            "trained_slot12_delta_receipt_state_sha256": None,
            "candidate_slot12_finite": True,
            "candidate_slot12_nonzero_count": None,
            "learned_signal_observed": None,
            "component_routing_state_sha256": final[
                "component_routing_state_sha256"
            ],
            "sampling_state_sha256": final["sampling_state_sha256"],
            **common,
            "effective_recipe_sha256": final["effective_recipe_sha256"],
            "sampler_seed": coordinator.FINAL_SAMPLER_SEED,
            "sampler_identity_sha256": sampling["sampler_identity_sha256"],
            "sample_order_sha256": outputs["sample_order_sha256"],
            "row_set_sha256": outputs["row_set_sha256"],
            "full_gate_entry_eligible": True,
        }
    raise CompletionError(f"unsupported central learner stage {stage!r}")


def complete_central_terminal(
    root: Path,
    *,
    verified: dict[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    """Authenticate one dose, derive its result, and append its terminal."""

    receipt = authenticate_completed_receipt(receipt_path, verified=verified)
    result = derive_terminal_result(verified, receipt)
    commitment_reference = receipt.get("central_execution_commitment")
    if not isinstance(commitment_reference, dict):
        raise CompletionError("central receipt has no execution commitment")
    try:
        execution_evidence = coordinator.central_terminal_execution_evidence(
            commitment_reference, receipt_path
        )
    except coordinator.CoordinatorError as error:
        raise CompletionError(f"central terminal execution evidence refused: {error}") from error
    central = verified["central_learner_binding"]
    stage = central["stage"]
    if stage == "P1":
        authority = verified["p1_arm_executor_authority"]
        return coordinator.complete_p1_arm(
            root,
            authority["sweep_id"],
            arm_id=authority["arm_id"],
            result=result,
            execution_evidence=execution_evidence,
        )
    if stage in {coordinator.ARM_CONTROL, coordinator.ARM_TREATMENT}:
        authority = verified["aux_pair_executor_authority"]
        pair = authority["aux_pair_contract"]
        return coordinator.complete_arm(
            root,
            pair["experiment_id"],
            arm_id=stage,
            result=result,
            execution_evidence=execution_evidence,
        )
    if stage != "FINAL":
        raise CompletionError(f"unsupported central learner stage {stage!r}")

    authority = verified["final_replication_executor_authority"]
    final = authority["final_replication_authority"]
    initializer = Path(str(verified["architecture_initializer"]["path"]))
    candidate = Path(str(receipt["outputs"]["checkpoint"]))
    directory = coordinator._artifact_dir(  # noqa: SLF001
        root.expanduser().resolve(strict=True),
        final["experiment_id"],
        create=False,
    )
    zero_path = directory / "94-final-initializer-slot12-zero.json"
    delta_path = directory / "94-final-trained-slot12-delta.json"
    try:
        zero = scientific_evidence.build_initializer_slot12_zero_receipt(initializer)
        delta = scientific_evidence.build_trained_slot12_delta_receipt(
            initializer, candidate
        )
        scientific_evidence._atomic_write(zero_path, zero)  # noqa: SLF001
        scientific_evidence._atomic_write(delta_path, delta)  # noqa: SLF001
    except (scientific_evidence.EvidenceError, OSError, ValueError) as error:
        raise CompletionError(f"FINAL slot12 evidence refused: {error}") from error
    result = copy.deepcopy(result)
    result.update(
        {
            "initializer_slot12_zero_receipt_state_sha256": zero["state_sha256"],
            "trained_slot12_delta_receipt_state_sha256": delta["state_sha256"],
            "candidate_slot12_finite": delta["candidate_slot12_finite"],
            "candidate_slot12_nonzero_count": delta[
                "candidate_slot12_nonzero_count"
            ],
            "learned_signal_observed": delta["learned_signal_observed"],
        }
    )
    return coordinator.complete_final_replication(
        root,
        final["experiment_id"],
        result=result,
        initializer_checkpoint_path=initializer,
        candidate_checkpoint_path=candidate,
        initializer_slot12_zero_receipt_path=zero_path,
        trained_slot12_delta_receipt_path=delta_path,
        execution_evidence=execution_evidence,
    )


__all__ = [
    "CompletionError",
    "authenticate_completed_receipt",
    "complete_central_terminal",
    "derive_terminal_result",
]
