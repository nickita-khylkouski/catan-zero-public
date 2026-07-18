from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import pytest

from tools import a1_aux_pair_coordinator as coordinator


_REAL_VERIFY_PUBLIC_AWARD_TRANSITION_AUTHORITY = (
    coordinator.verify_public_award_transition_authority
)
_REAL_VERIFY_PUBLIC_AWARD_TRANSITION_RECEIPT = (
    coordinator.scientific_evidence.verify_public_award_transition_receipt
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _geometry_rng_transaction(cuda_device: int = 0) -> dict:
    state = {
        "schema_version": "a1-aux-geometry-rng-state-v1",
        "python_random_sha256": _sha("1"),
        "numpy_generator_sha256": {"sampler": _sha("2")},
        "torch_cpu_sha256": _sha("3"),
        "cuda_device": cuda_device,
        "torch_cuda_sha256": _sha("4"),
    }
    state["state_sha256"] = coordinator._digest(state)
    after_probe = copy.deepcopy(state)
    after_probe["torch_cuda_sha256"] = _sha("5")
    after_probe["state_sha256"] = coordinator._digest(
        {
            key: value
            for key, value in after_probe.items()
            if key != "state_sha256"
        }
    )
    return {
        "schema_version": "a1-aux-geometry-rng-transaction-v1",
        "scope": "one_complete_ordered_five_batch_probe",
        "restore_frequency": "once_after_all_five_batches",
        "before": copy.deepcopy(state),
        "after_probe": after_probe,
        "after_restore": copy.deepcopy(state),
        "restored_exactly": True,
    }


_P1_PAYLOAD_SHA = _sha("0")


def _eligibility_rows():
    for index in range(coordinator.SHORT_SAMPLE_DOSE):
        eligible = index % 2 == 0
        facts = {
            "payload_member_sha256": _P1_PAYLOAD_SHA,
            "row_offset": index,
            "component_id": "current_producer",
            "prior_policy_present": eligible,
            "legal_action_count": 2 if eligible else 1,
        }
        yield {
            "row_identity_sha256": coordinator.canonical_p1_row_identity(**facts),
            **facts,
        }


_P1_SAMPLE_ORDER_SHA = coordinator.canonical_p1_sample_order_sha256(
    row["row_identity_sha256"] for row in _eligibility_rows()
)


def _execution(character: str) -> dict:
    return {
        "schema_version": coordinator.EXECUTION_SCHEMA,
        "command_sha256": _sha(character),
        "environment_sha256": _sha(chr(ord(character) + 1)),
        "output_namespace_sha256": _sha(chr(ord(character) + 2)),
    }


def _stage_execution_material(
    root: Path, experiment_id: str, stage: str
) -> tuple[dict, list[str], dict[str, str], dict[str, str], dict]:
    directory = root.resolve() / experiment_id.removeprefix("sha256:")
    authority_name = (
        "15-warmup-executor-authority.json"
        if stage == "WARMUP"
        else "35-geometry-executor-authority.json"
    )
    authority_path = directory / authority_name
    if authority_path.exists():
        authority, authority_file_sha, _identity = (
            coordinator._stable_read_immutable_json(  # noqa: SLF001
                authority_path, where="test stage authority"
            )
        )
        authority_state_sha = authority["state_sha256"]
    else:
        authority_file_sha = _sha("d")
        authority_state_sha = _sha("e")
    outputs = {
        "checkpoint": f"/tmp/a1-{experiment_id[-8:]}-{stage.lower()}.pt",
        "report": f"/tmp/a1-{experiment_id[-8:]}-{stage.lower()}.json",
        "optimizer_sidecar": (
            f"/tmp/a1-{experiment_id[-8:]}-{stage.lower()}.pt.optimizer.pt"
        ),
    }
    binding = {
        "schema_version": "a1-aux-stage-training-binding-v1",
        "stage": stage,
        "experiment_id": experiment_id,
        "executor_authority_path": str(authority_path),
        "executor_authority_file_sha256": authority_file_sha,
        "executor_authority_state_sha256": authority_state_sha,
        "output_checkpoint": outputs["checkpoint"],
        "output_report": outputs["report"],
    }
    command = [
        "python",
        "train_bc.py",
        "--a1-aux-stage-binding-json",
        json.dumps(binding, sort_keys=True, separators=(",", ":")),
        "--a1-aux-stage-executor-authority",
        str(authority_path),
        "--a1-aux-stage-executor-authority-sha256",
        authority_file_sha,
    ]
    environment = {"TEST_A1_STAGE": stage}
    execution = {
        "schema_version": coordinator.EXECUTION_SCHEMA,
        "command_sha256": coordinator._digest(
            coordinator.canonical_stage_command_intent(command)
        ),
        "environment_sha256": coordinator._digest(environment),
        "output_namespace_sha256": coordinator._digest(outputs),
    }
    return execution, command, environment, outputs, binding


def _commit_test_stage(root: Path, experiment_id: str, stage: str) -> dict:
    experiment = coordinator.load_experiment(root, experiment_id)
    if stage == "WARMUP":
        coordinator.load_warmup_executor_authority(
            root,
            experiment_id,
            observed_allocation=experiment["allocations"]["WARMUP"],
        )
    else:
        coordinator.load_geometry_executor_authority(
            root,
            experiment_id,
            observed_allocation=experiment["allocations"]["GEOMETRY"],
        )
    _execution_value, command, environment, outputs, binding = (
        _stage_execution_material(root, experiment_id, stage)
    )
    return coordinator.commit_stage_execution(
        root,
        experiment_id,
        stage=stage,
        command=command,
        environment=environment,
        output_namespace=outputs,
        training_binding=binding,
    )


def _central_execution_material(
    root: Path,
    experiment_id: str,
    stage: str,
    *,
    arm_id: str | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[dict, dict]:
    directory = root.resolve() / experiment_id.removeprefix("sha256:")
    label = (arm_id or stage).lower()
    if stage == "P1":
        if arm_id not in coordinator.P1_ARMS:
            raise AssertionError("P1 test material requires an arm")
        authority_name = f"p1-15-{arm_id.lower()}-executor-authority.json"
    elif stage in coordinator.ARMS:
        authority_name = f"65-{label}-executor-authority.json"
    else:
        authority_name = "93-final-executor-authority.json"
    authority_path = directory / authority_name
    stem = f"central-{experiment_id[-8:]}-{label}"
    checkpoint = checkpoint_path or (root.resolve() / f"{stem}.pt")
    outputs = {
        "checkpoint": str(checkpoint.resolve(strict=False)),
        "optimizer_sidecar": str(Path(str(checkpoint.resolve(strict=False)) + ".optimizer.pt")),
        "training_progress": str(Path(str(checkpoint.resolve(strict=False)) + ".training-progress.json")),
        "report": str(root.resolve() / f"{stem}-report.json"),
        "receipt": str(root.resolve() / f"{stem}-receipt.json"),
        "one_dose_claim": str(root.resolve() / f"{stem}-claim.json"),
    }
    central_binding = {
        "stage": stage,
        "central_authority_sha256": _sha("1"),
        "executor_authority_path": str(authority_path),
        "executor_authority_file_sha256": _sha("2"),
        "executor_authority_state_sha256": _sha("3"),
    }
    command = [
        "python",
        "train_bc.py",
        "--a1-central-learner-binding-json",
        json.dumps(central_binding, sort_keys=True, separators=(",", ":")),
        "--a1-central-executor-authority",
        str(authority_path),
        "--a1-central-executor-authority-sha256",
        _sha("2"),
    ]
    if stage in coordinator.ARMS:
        command.extend(
            [
                "--a1-aux-regularization-binding-json",
                json.dumps(
                    {"aux_pair_authority_sha256": _sha("4")},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    environment = {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
    execution = {
        "schema_version": coordinator.EXECUTION_SCHEMA,
        "command_sha256": coordinator._digest(
            coordinator.canonical_central_command_intent(command)
        ),
        "environment_sha256": coordinator._digest(environment),
        "output_namespace_sha256": coordinator._digest(
            coordinator.canonical_central_output_namespace_intent(outputs)
        ),
    }
    return execution, {
        "authority_path": authority_path,
        "central_binding": central_binding,
        "command": command,
        "environment": environment,
        "outputs": outputs,
        "claim_identity_sha256": _sha("9"),
    }


def _publish_central_execution_evidence(
    root: Path,
    *,
    stage: str,
    material: dict,
    result: dict,
) -> dict:
    published = coordinator.verify_published_executor_authority(
        material["authority_path"]
    )
    binding = dict(material["central_binding"])
    binding.update(
        {
            "central_authority_sha256": published["authority"]["authority_sha256"],
            "executor_authority_path": published["path"],
            "executor_authority_file_sha256": published["file_sha256"],
            "executor_authority_state_sha256": published["authority"]["state_sha256"],
        }
    )
    command = list(material["command"])
    binding_index = command.index("--a1-central-learner-binding-json") + 1
    command[binding_index] = json.dumps(
        binding, sort_keys=True, separators=(",", ":")
    )
    authority_index = command.index("--a1-central-executor-authority") + 1
    command[authority_index] = published["path"]
    authority_sha_index = (
        command.index("--a1-central-executor-authority-sha256") + 1
    )
    command[authority_sha_index] = published["file_sha256"]
    if stage in coordinator.ARMS:
        aux_index = command.index("--a1-aux-regularization-binding-json") + 1
        command[aux_index] = json.dumps(
            {
                "aux_pair_authority_sha256": published["authority"][
                    "authority_sha256"
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        aux_binding = json.loads(command[aux_index])
    else:
        aux_binding = None
    reference = coordinator.commit_central_learner_execution(
        published_executor_authority=published,
        command=command,
        environment=material["environment"],
        output_namespace=material["outputs"],
        central_binding=binding,
        input_binding={},
        one_dose_claim_identity_sha256=material["claim_identity_sha256"],
        aux_regularization_binding=aux_binding,
    )
    realized = {
        "checkpoint_sha256": material["outputs"]["checkpoint"],
        "optimizer_sidecar_sha256": material["outputs"]["optimizer_sidecar"],
        "training_progress_sha256": material["outputs"]["training_progress"],
        "report_sha256": material["outputs"]["report"],
    }
    for field, path_value in realized.items():
        path = Path(path_value)
        if path.exists():
            raw = path.read_bytes()
        else:
            raw = f"{stage}:{field}:{path.name}\n".encode("utf-8")
            path.write_bytes(raw)
        path.chmod(0o444)
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if field in result:
            result[field] = digest
    outputs = {
        "checkpoint": material["outputs"]["checkpoint"],
        "optimizer_sidecar": material["outputs"]["optimizer_sidecar"],
        "training_progress": material["outputs"]["training_progress"],
        "report": material["outputs"]["report"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "optimizer_sidecar_sha256": result["optimizer_sidecar_sha256"],
        "report_sha256": result["report_sha256"],
        "training_progress_sha256": "sha256:"
        + hashlib.sha256(
            Path(material["outputs"]["training_progress"]).read_bytes()
        ).hexdigest(),
    }
    common = {
        "status": "complete",
        "claim_identity_sha256": material["claim_identity_sha256"],
        "command": command,
        "command_sha256": coordinator._digest(command),
        "execution_binding": {
            "environment_sha256": coordinator._digest(material["environment"])
        },
        "input_binding": {},
        "central_execution_commitment": reference,
        "returncode": 0,
        "failure": None,
        "outputs": outputs,
    }
    claim_payload = {
        "schema_version": "a1-central-learner-training-claim-v1",
        **common,
        "receipt_target": material["outputs"]["receipt"],
    }
    claim = coordinator._write_once(
        Path(material["outputs"]["one_dose_claim"]), claim_payload
    )
    receipt_payload = {
        "schema_version": "a1-central-learner-training-receipt-v1",
        **common,
        "claim": material["outputs"]["one_dose_claim"],
        "claim_state_sha256": claim["state_sha256"],
    }
    receipt_payload["receipt_sha256"] = coordinator._digest(receipt_payload)
    receipt_path = Path(material["outputs"]["receipt"])
    receipt_path.write_text(
        json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)
    return coordinator.central_terminal_execution_evidence(reference, receipt_path)


def _allocation(host: str, offset: int) -> dict:
    return {
        "schema_version": coordinator.ALLOCATION_SCHEMA,
        "host_id": host,
        "hostname": coordinator.B200_LEARNER_HOSTNAME,
        "machine_id": coordinator.B200_LEARNER_MACHINE_ID,
        "ssh_host_key_sha256": _sha(f"{offset % 10}"),
        "checkout_tree_sha256": _sha(f"{(offset + 1) % 10}"),
        "tool_sha256": coordinator._repo_tool_sha256(
            "tools/a1_scientific_evidence.py"
        ),
        "physical_gpu_indices": list(range(8)),
        "gpu_names": ["NVIDIA B200"] * 8,
        "gpu_uuids": list(coordinator.B200_LEARNER_GPU_UUIDS),
        "pci_bus_ids": [f"0000:{offset + index:02x}:00.0" for index in range(8)],
    }


def _category_semantics() -> dict:
    recovery = _recovery()
    return {
        "current_producer": {
            "scheduler_category": "current_producer",
            "semantic": "current_producer",
            "relation": "self_play",
            "checkpoint": {
                "id": "recovered-generator",
                "path": recovery["recovered_generator"]["path"],
                "sha256": recovery["recovered_generator"]["sha256"],
                "version": recovery["recovered_generator"][
                    "historical_generation_version_claim"
                ],
            },
        },
        "recent_history": {
            "scheduler_category": "recent_history",
            "semantic": "recovery_reference",
            "relation": "safety_reference_unproven_predecessor",
            "causal_parent_proven": False,
            "promotion_proof_recreated": False,
            "checkpoint": {
                "id": "f7-safety-reference",
                "path": recovery["safety_reference_unproven_predecessor"]["path"],
                "sha256": recovery["safety_reference_unproven_predecessor"]["sha256"],
                "version": recovery["safety_reference_unproven_predecessor"][
                    "historical_generation_version_claim"
                ],
            },
            "recovery_lineage_id": recovery["recovery_lineage_id"],
        },
        "hard_negative": {
            "scheduler_category": "hard_negative",
            "semantic": "hard_negative",
            "relation": "sealed_hard_negative_selection",
            "checkpoint": {
                "id": "hard-negative",
                "path": "/sealed/hard-negative.pt",
                "sha256": _sha("7"),
                "version": 3,
            },
        },
    }


def _composite(
    *,
    descriptor_sha256: str | None = None,
    source_authority: dict | None = None,
) -> dict:
    semantics = _category_semantics()
    return {
        "schema_version": "a1-typed-fresh-80-15-5-composite-v1",
        "component_ids": list(coordinator.COMPONENT_IDS),
        "component_sampling_ratios": list(coordinator.COMPONENT_RATIOS),
        "descriptor_sha256": descriptor_sha256 or _sha("a"),
        "data_fingerprint": _sha("b"),
        "payload_inventory_sha256": _sha("c"),
        "production_sampling_receipt_sha256": _sha("d"),
        "validation_split_receipt_sha256": _sha("e"),
        "sampler_identity_sha256": _sha("f"),
        "sample_order_sha256": _P1_SAMPLE_ORDER_SHA,
        "training_game_seed_set_sha256": _sha("1"),
        "validation_game_seed_set_sha256": _sha("2"),
        "truncation_surface_sha256": _sha("3"),
        "truncated_rows": 0,
        "complete_game_inputs": True,
        "category_semantics": semantics,
        "category_semantics_sha256": coordinator._digest(semantics),
        "source_authority": source_authority
        or {
            "path": "/sealed/source-authority.json",
            "file_sha256": _sha("4"),
            "authority_sha256": _sha("5"),
        },
        "learner_recipe_overrides_sha256": _sha("6"),
        "aux_subgoal_target_contract_sha256": _sha("7"),
        "public_award_feature_transition_contract_sha256": _sha("8"),
        "source_authority_semantic_sha256": _sha("9"),
    }


def _base_recipe() -> dict:
    return copy.deepcopy(coordinator.canonical_p1_final_lock_authority()["base_recipe"])


def _final_lock() -> dict:
    return coordinator.canonical_p1_final_lock_authority()


def _recovery() -> dict:
    return {
        "schema_version": "a1-v5-disaster-recovery-authority-v1",
        "recovery_receipt": {
            "path": "/sealed/recovery.json",
            "sha256": _sha("8"),
            "recovery_receipt_sha256": _sha("9"),
        },
        "recovery_lineage_id": _sha("a"),
        "recovered_generator": {
            "path": "/sealed/checkpoint-6817.pt",
            "sha256": (
                "sha256:6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c"
            ),
            "md5": "6817ab054506f962a758ebf48addce5c",
            "historical_generation_version_claim": 5,
        },
        "safety_reference_unproven_predecessor": {
            "path": "/sealed/checkpoint-f7.pt",
            "sha256": (
                "sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4"
            ),
            "md5": "f7e93dfb8cdb713d647b3e142c949d5",
            "historical_generation_version_claim": 4,
            "relation": "unproven_predecessor_safety_reference",
            "causal_parent_proven": False,
        },
        "producer_identity": {
            "schema_version": "a1-surviving-producer-identity-v1",
            "checkpoint": {
                "path": "/sealed/checkpoint-6817.pt",
                "sha256": (
                    "sha256:6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c"
                ),
            },
            "search_config": {"n_full": 128, "c_scale": 0.1},
            "agent_identity_sha256": _sha("b"),
        },
        "promotion_proof_recreated": False,
        "dual_baseline_fresh_gate_required": True,
        "promotion_eligible": False,
        "training_proof": False,
        "wave_lineage_mode": "recovery_reference",
        "authority_sha256": _sha("c"),
    }


def _parent() -> dict:
    return coordinator.current_parent_authority_from_recovery(_recovery())


def _sample_receipt(*, final: bool = False, composite: dict | None = None) -> dict:
    dose = coordinator.SHORT_SAMPLE_DOSE
    composite = _composite() if composite is None else composite
    prior = _sample_receipt(final=False, composite=composite) if final else None
    seed = coordinator.FINAL_SAMPLER_SEED if final else coordinator.P1_SAMPLER_SEED
    return coordinator._sealed(
        {
            "schema_version": "a1-authenticated-sample-evidence-v3",
            "status": "complete",
            "sample_dose": dose,
            "sampler_seed": seed,
            "sampler_algorithm": coordinator.scientific_evidence.SAMPLER_ALGORITHM,
            "sampler_identity_sha256": _sha("5") if final else _sha("f"),
            "sample_order_sha256": _sha("6") if final else _P1_SAMPLE_ORDER_SHA,
            "row_set_sha256": _sha("7") if final else _sha("4"),
            "unique_row_count": 500_000,
            "prior_rows_file_sha256": None if prior is None else prior["rows_file_sha256"],
            "prior_row_set_sha256": None if prior is None else prior["row_set_sha256"],
            "prior_unique_row_count": 0 if prior is None else prior["unique_row_count"],
            "observed_unique_overlap_count": 8_192 if final else 0,
            "analytic_expected_unique_overlap_decimal": "8192" if final else "0",
            "overlap_excess_bound_decimal": "7000",
            "overlap_alpha_decimal": "0.000000001",
            "overlap_within_independent_bound": True,
            "component_overlap": {
                component: {
                    "draw_count": dose // 4,
                    "unique_row_count": 125_000,
                    "prior_unique_row_count": 0 if prior is None else 125_000,
                    "observed_unique_overlap_count": 0 if prior is None else 2_048,
                    "analytic_expected_unique_overlap_decimal": (
                        "0" if prior is None else "2048"
                    ),
                }
                for component in coordinator.COMPONENT_IDS
            },
            "kl_eligible_rows": 0,
            "kl_eligible_mass_decimal": "0",
            "kl_ordered_evidence_sha256": _sha("1"),
            "kl_eligible_evidence_sha256": _sha("2"),
            "policy_kl_anchor_component_ids": [],
            "descriptor_sha256": composite["descriptor_sha256"],
            "payload_inventory_sha256": composite["payload_inventory_sha256"],
            "category_semantics": copy.deepcopy(composite["category_semantics"]),
            "category_semantics_sha256": composite[
                "category_semantics_sha256"
            ],
            "source_authority": copy.deepcopy(composite["source_authority"]),
            "rows_file_sha256": _sha("3") if final else _sha("0"),
            "origin_tool_sha256": coordinator._repo_tool_sha256(
                "tools/a1_scientific_evidence.py"
            ),
            "replay_verified": True,
        }
    )


_ELIGIBILITY = coordinator.build_p1_kl_eligibility_authority(
    composite=_composite(), sampled_row_evidence=_eligibility_rows()
)


def _eligibility() -> dict:
    return copy.deepcopy(_ELIGIBILITY)


def _native_admission() -> dict:
    hosts = {}
    for host, offset in ((coordinator.B200_LEARNER_HOST_ID, 10),):
        allocation = _allocation(host, offset)
        hosts[host] = {
            "host_id": host,
            "hostname": allocation["hostname"],
            "machine_id": allocation["machine_id"],
            "ssh_host_key_sha256": allocation["ssh_host_key_sha256"],
            "checkout_tree_sha256": allocation["checkout_tree_sha256"],
            "tool_sha256": allocation["tool_sha256"],
            "gpu_indices": allocation["physical_gpu_indices"],
            "gpu_names": allocation["gpu_names"],
            "gpu_uuids": allocation["gpu_uuids"],
            "pci_bus_ids": allocation["pci_bus_ids"],
            "python": coordinator.production_executor.PRODUCTION_RUNTIME[
                "python_version"
            ],
            "torch_version": coordinator.production_executor.PRODUCTION_RUNTIME[
                "torch_version"
            ],
            "torch_cuda_version": coordinator.production_executor.PRODUCTION_RUNTIME[
                "torch_cuda_version"
            ],
            "catanatron_rs_version": coordinator.production_executor.PRODUCTION_RUNTIME[
                "catanatron_rs_version"
            ],
            "native_wheel_sha256": coordinator.NATIVE_WHEEL_SHA256,
            "native_mcts_capabilities": list(coordinator.NATIVE_CAPABILITIES),
            "nofile_soft": 65_536,
            "nofile_hard": 1_048_576,
        }
    return coordinator._sealed(
        {
            "schema_version": "a1-b200-learner-runtime-admission-v1",
            "status": "complete",
            "hosts": hosts,
            "origin_tool_sha256": coordinator._repo_tool_sha256(
                "tools/a1_scientific_evidence.py"
            ),
        }
    )


def _native() -> dict:
    return coordinator.build_native_runtime_authority(
        learner_admission_receipt=_native_admission()
    )


@pytest.fixture(autouse=True)
def _isolate_external_evidence_verifiers(monkeypatch) -> None:
    """Coordinator tests isolate orchestration; producer modules test replay."""

    monkeypatch.setattr(
        coordinator.v5_recovery,
        "verify_committed_receipt",
        lambda _path: {"authority": _recovery(), "receipt": {"status": "committed"}},
    )
    monkeypatch.setattr(
        coordinator.scientific_evidence,
        "verify_runtime_admission_receipt",
        lambda path, **_kwargs: coordinator._load_json(path, where="test runtime"),
    )
    monkeypatch.setattr(
        coordinator.scientific_evidence,
        "verify_sample_evidence",
        lambda path, **_kwargs: coordinator._load_json(path, where="test sample"),
    )
    monkeypatch.setattr(
        coordinator.scientific_evidence,
        "verify_component_routing_receipt",
        lambda path, **_kwargs: coordinator._load_json(path, where="test routing"),
    )
    monkeypatch.setattr(
        coordinator.scientific_evidence,
        "verify_initializer_slot12_zero_receipt",
        lambda path, **_kwargs: coordinator._load_json(path, where="test slot12"),
    )
    monkeypatch.setattr(
        coordinator.scientific_evidence,
        "verify_trained_slot12_delta_receipt",
        lambda path, **_kwargs: coordinator._load_json(path, where="test delta"),
    )
    def _verify_test_transition(value, *, expected_parent):
        if value["source_checkpoint"] != {
            "path": expected_parent["checkpoint_path"],
            "sha256": expected_parent["checkpoint_sha256"],
        }:
            raise coordinator.CoordinatorError(
                "test transition lost its causal parent"
            )
        return copy.deepcopy(value)

    monkeypatch.setattr(
        coordinator,
        "verify_public_award_transition_authority",
        _verify_test_transition,
    )
    monkeypatch.setattr(
        coordinator.v5_recovery_gate,
        "verify_recovery_gate_authority",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        coordinator.h100_eval_fleet,
        "verify_fixed_panel_receipt",
        lambda path, **_kwargs: coordinator._load_json(
            path, where="test fixed panel"
        ),
    )


def _p1_input_paths(root: Path) -> tuple[dict[str, Path], dict]:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = root / "composite.json"
    source_authority_path = root / "source-authority.json"
    rows = root / "p1-rows.jsonl"
    recovery = root / "recovery.json"
    runtime = root / "runtime.json"
    sample = root / "p1-sample.json"
    semantics = _category_semantics()
    source_unsigned = {
        "schema_version": "a1-portable-composite-authority-v1",
        "category_semantics": semantics,
    }
    source_payload = {
        **source_unsigned,
        "authority_sha256": coordinator._digest(source_unsigned),
    }
    source_authority_path.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_ref = {
        "path": str(source_authority_path.resolve(strict=True)),
        "file_sha256": "sha256:"
        + hashlib.sha256(source_authority_path.read_bytes()).hexdigest(),
        "authority_sha256": source_payload["authority_sha256"],
    }
    descriptor_payload = {
        "category_semantics": semantics,
        "policy_kl_anchor_component_ids": [],
        "source_authority_manifest": source_ref["path"],
        "source_authority_manifest_sha256": source_ref["file_sha256"],
        "source_authority_sha256": source_ref["authority_sha256"],
    }
    descriptor.write_text(
        json.dumps(descriptor_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    descriptor_sha = "sha256:" + hashlib.sha256(descriptor.read_bytes()).hexdigest()
    composite = _composite(
        descriptor_sha256=descriptor_sha,
        source_authority=source_ref,
    )
    rows.write_text("{}\n", encoding="utf-8")
    recovery.write_text("{}\n", encoding="utf-8")
    # _write_once seals; pass the unsigned measured fixtures.
    native_unsigned = _native_admission()
    native_unsigned.pop("state_sha256")
    coordinator._write_once(runtime, native_unsigned)
    sample_unsigned = _sample_receipt(composite=composite)
    sample_unsigned.pop("state_sha256")
    coordinator._write_once(sample, sample_unsigned)
    return (
        {
            "composite_descriptor_path": descriptor,
            "p1_sample_receipt_path": sample,
            "p1_sample_rows_path": rows,
            "v5_recovery_receipt_path": recovery,
            "native_learner_admission_receipt_path": runtime,
        },
        composite,
    )


def _issue_p1_sweep(root: Path) -> dict:
    pytest.skip(
        "the historical-replay P1 sweep and its dependent auxiliary workflow "
        "were retired when production moved to the fresh-only 80/15/5 composite"
    )


def _p1_evaluation_plan() -> dict:
    return coordinator.canonical_p1_evaluation_plan(
        baseline_checkpoint_sha256=_parent()["checkpoint_sha256"]
    )


def _p1_allocations() -> dict:
    allocations = {
        "K0": _allocation(coordinator.B200_LEARNER_HOST_ID, 10),
        "K3": _allocation(coordinator.B200_LEARNER_HOST_ID, 10),
    }
    allocations["K10"] = copy.deepcopy(allocations["K0"])
    return allocations


def _p1_result(sweep: dict, arm_id: str, character: str) -> dict:
    arm = sweep["arms"][arm_id]
    composite = sweep["composite"]
    alphabet = "0123456789abcdef"
    start = sum(ord(value) for value in character) % (len(alphabet) - 3)
    return {
        "schema_version": "a1-p1-central-arm-result-v1",
        "status": "complete",
        "sweep_id": sweep["sweep_id"],
        "arm_id": arm_id,
        "policy_kl_anchor_weight_decimal": arm["policy_kl_anchor_weight_decimal"],
        "policy_kl_anchor_eligible_rows": arm["training_descriptor_authority"][
            "expected_policy_kl_anchor_eligible_rows"
        ],
        "initializer_sha256": _parent()["checkpoint_sha256"],
        "sampled_rows": 524_288,
        "optimizer_steps": 128,
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "amp": "none",
        "fresh_adam": True,
        "optimizer_restored": False,
        "effective_recipe_sha256": arm["effective_recipe_sha256"],
        "payload_inventory_sha256": composite["payload_inventory_sha256"],
        "validation_split_receipt_sha256": composite["validation_split_receipt_sha256"],
        "sampler_identity_sha256": composite["sampler_identity_sha256"],
        "sample_order_sha256": composite["sample_order_sha256"],
        "checkpoint_sha256": _sha(alphabet[start]),
        "optimizer_sidecar_sha256": _sha(alphabet[start + 1]),
        "report_sha256": _sha(alphabet[start + 2]),
        "origin_tool_sha256": coordinator._repo_tool_sha256(
            "tools/a1_one_dose_train.py"
        ),
    }


def _write_panel_receipt(
    root: Path,
    *,
    family: str,
    panel_kind: str,
    authority_id: str,
    arms: tuple[str, ...],
    checkpoints: dict,
    baseline_checkpoint_sha256: str,
    cohort_sha256: str,
    search_operator_sha256: str,
    origin_tool_sha256: str,
    points_milli: dict,
) -> Path:
    path = root / f"{family.lower()}-{panel_kind}-panel.json"
    coordinator._write_once(
        path,
        {
            "schema_version": "a1-fixed-panel-receipt-v2",
            "family": family,
            "panel_kind": panel_kind,
            "authority_id": authority_id,
            "arms": list(arms),
            "arm_checkpoint_sha256": checkpoints,
            "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
            "cohort_sha256": cohort_sha256,
            "search_operator_sha256": search_operator_sha256,
            "common_random_numbers": True,
            "seat_swapped": True,
            "points_milli": points_milli,
            "origin_tool_sha256": origin_tool_sha256,
        },
    )
    return path


def _p1_panel_receipts(root: Path, sweep: dict, selected_arm: str) -> tuple[Path, Path]:
    plan = sweep["evaluation_plan"]
    internal = {
        arm: [1500 if arm == selected_arm else 1000] * plan["internal_pairs_per_arm"]
        for arm in coordinator.P1_ARMS
    }
    external = {
        arm: [500] * plan["external_games_per_arm"] for arm in coordinator.P1_ARMS
    }
    checkpoints = {
        arm: coordinator._artifact(  # noqa: SLF001
            root,
            sweep["sweep_id"],
            f"p1-20-{arm.lower()}-terminal.json",
        )["result"]["checkpoint_sha256"]
        for arm in coordinator.P1_ARMS
    }
    shared = {
        "family": "P1",
        "authority_id": sweep["sweep_id"],
        "arms": coordinator.P1_ARMS,
        "checkpoints": checkpoints,
        "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
        "search_operator_sha256": plan["search_operator_sha256"],
        "origin_tool_sha256": plan["panel_origin_tool_sha256"],
    }
    return (
        _write_panel_receipt(
            root,
            panel_kind="internal",
            cohort_sha256=plan["internal_cohort_sha256"],
            points_milli=internal,
            **shared,
        ),
        _write_panel_receipt(
            root,
            panel_kind="external",
            cohort_sha256=plan["external_cohort_sha256"],
            points_milli=external,
            **shared,
        ),
    )


def _complete_p1(root: Path, *, selected_arm: str = "K3") -> tuple[dict, dict]:
    sweep = _issue_p1_sweep(root)
    for index, arm_id in enumerate(coordinator.P1_ARMS):
        result = _p1_result(sweep, arm_id, chr(ord("d") + index * 4))
        execution, material = _central_execution_material(
            root, sweep["sweep_id"], "P1", arm_id=arm_id
        )
        coordinator.claim_p1_arm(
            root,
            sweep["sweep_id"],
            arm_id=arm_id,
            observed_allocation=sweep["allocations"][arm_id],
            execution=execution,
        )
        coordinator.load_p1_arm_executor_authority(
            root,
            sweep["sweep_id"],
            arm_id=arm_id,
            observed_allocation=sweep["allocations"][arm_id],
        )
        evidence = _publish_central_execution_evidence(
            root, stage="P1", material=material, result=result
        )
        coordinator.complete_p1_arm(
            root,
            sweep["sweep_id"],
            arm_id=arm_id,
            result=result,
            execution_evidence=evidence,
        )
    coordinator.claim_p1_evaluation(root, sweep["sweep_id"], execution=_execution("7"))
    internal_panel, external_panel = _p1_panel_receipts(root, sweep, selected_arm)
    coordinator.complete_p1_evaluation(
        root,
        sweep["sweep_id"],
        internal_panel_receipt_path=internal_panel,
        external_panel_receipt_path=external_panel,
    )
    coordinator.adjudicate_p1_sweep(
        root,
        sweep["sweep_id"],
        selected_arm=selected_arm,
        applied_selection_rule_sha256=sweep["selection_rule_sha256"],
    )
    return sweep, coordinator.load_selected_p1_recipe_data_authority(
        root, sweep["sweep_id"]
    )


def _pointer_upgrade() -> dict:
    return {
        "schema_version": coordinator.POINTER_UPGRADE_AUTHORITY_SCHEMA,
        "module": coordinator.POINTER_MODULE,
        "source_checkpoint_sha256": _sha("9"),
        "upgraded_initializer_sha256": _sha("a"),
        "receipt_sha256": _sha("b"),
        "receipt_replay_sha256": _sha("c"),
        "flags": dict(coordinator.POINTER_FLAGS),
        "new_parameter_set_sha256": _sha("d"),
        "main_output_max_diff": 0.0,
        "shared_parameters_bit_identical": True,
    }


def _public_award_transition() -> dict:
    evidence = coordinator._sealed(
        {
            "schema_version": "a1-public-award-initializer-transition-evidence-v1",
            "status": "complete",
            "source_checkpoint_sha256": _parent()["checkpoint_sha256"],
            "transitioned_checkpoint_sha256": _sha("9"),
            "source_public_award_feature_contract": "legacy_zero_v0",
            "transitioned_public_award_feature_contract": "authoritative_v1",
            "changed_parameter_name": "player_encoder.0.weight",
            "changed_input_column_index": 12,
            "source_slot12_column_sha256": _sha("1"),
            "transitioned_slot12_column_sha256": _sha("2"),
            "transitioned_slot12_max_abs_decimal": "0",
            "unchanged_parameter_count": 127,
            "unchanged_parameter_identity_sha256": _sha("3"),
            "unchanged_parameters_bit_identical": True,
            "unrelated_metadata_bit_identical": True,
            "legacy_zero_input_function_preserving": True,
            "optimizer_steps": 0,
            "origin_tool_sha256": coordinator._repo_tool_sha256(
                "tools/a1_scientific_evidence.py"
            ),
        }
    )
    return {
        "schema_version": coordinator.PUBLIC_AWARD_TRANSITION_AUTHORITY_SCHEMA,
        "source_checkpoint": {
            "path": _parent()["checkpoint_path"],
            "sha256": _parent()["checkpoint_sha256"],
        },
        "transitioned_checkpoint": {
            "path": "/sealed/checkpoint-6817-authoritative-v1.pt",
            "sha256": _sha("9"),
        },
        "receipt": {
            "path": "/sealed/public-award-transition.json",
            "file_sha256": _sha("8"),
            "evidence": evidence,
        },
    }


def _warmup_recipe(composite: dict | None = None) -> dict:
    steps = coordinator.SHORT_OPTIMIZER_STEPS
    composite = _composite() if composite is None else composite
    seed = 1
    return {
        "schema_version": coordinator.WARMUP_RECIPE_SCHEMA,
        "sample_dose": coordinator.SHORT_SAMPLE_DOSE,
        "optimizer_steps": steps,
        "max_steps": steps,
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "grad_accum_steps": 1,
        "amp": "none",
        "optimizer": "fresh_adam",
        "resume_optimizer": False,
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1.0e-8,
        "weight_decay": 0.0,
        "fused_optimizer": False,
        "lr": 3.0e-5,
        "lr_warmup_steps": 100,
        "lr_schedule": "flat",
        "seed": seed,
        "training_rng_rank_offset": True,
        "effective_torch_seeds": [seed + rank for rank in range(8)],
        "max_grad_norm": 1.0,
        "gradient_clipping": True,
        "head_only": True,
        "inherited_parameters_frozen": True,
        "trainable_prefixes": list(coordinator.POINTER_TRAINABLE_PREFIXES),
        "aux_subgoal_loss_weight": 1.0,
        "main_objective_coefficients": dict(coordinator.WARMUP_MAIN_OBJECTIVE_ZERO),
        "descriptor_sha256": composite["descriptor_sha256"],
        "data_fingerprint": composite["data_fingerprint"],
        "payload_inventory_sha256": composite["payload_inventory_sha256"],
        "validation_split_receipt_sha256": composite["validation_split_receipt_sha256"],
        "sampler_identity_sha256": composite["sampler_identity_sha256"],
        "sample_order_sha256": composite["sample_order_sha256"],
        "sampler_seed": 424242,
        "target_scope": "authenticated_aux_component_scope",
        "target_scope_sha256": coordinator._digest(
            {
                "scope": "authenticated_aux_component_scope",
                "module": coordinator.POINTER_MODULE,
                "target_version": 1,
            }
        ),
        "target_version": 1,
        "checkpoint_selection": {
            "rule": "fixed_terminal_step",
            "optimizer_step": steps,
            "adaptive_best_checkpoint": False,
        },
    }


def _selector_rule() -> dict:
    return {
        "schema_version": coordinator.SELECTOR_RULE_SCHEMA,
        "formula": (
            "min(max_aux_to_main_ratio/r,"
            "max_opposing_projection/max(-r*cos,epsilon)_if_cos_negative,"
            "maximum_coefficient)"
        ),
        "maximum_aux_to_main_ratio_decimal": "0.05",
        "maximum_opposing_projection_decimal": "0.01",
        "maximum_coefficient_decimal": "0.05",
        "minimum_coefficient_decimal": "0.001",
        "quantum_decimal": "0.001",
        "rounding": "ROUND_DOWN",
        "out_of_range": "refuse",
        "probe_manifest_sha256": _sha("e"),
        "probe_sampler_seed": 424242,
        "probe_row_order_sha256": _sha("f"),
        "probe_batches": 5,
        "probe_batch_size": 512,
        "shared_parameter_surface": "inherited_trunk_only",
        "shared_parameter_set_sha256": _sha("1"),
        "same_forward_graph": True,
        "global_ddp_aggregation": True,
        "ddp_reduction": "global_numerator_and_denominator_sum",
        "cross_batch_aggregation": "concatenated_batch_gradient_geometry",
        "no_optimizer_step": True,
        "no_persistent_mutation": True,
    }


def _aux_evaluation_plan() -> dict:
    return coordinator.canonical_aux_evaluation_plan(
        baseline_checkpoint_sha256=_parent()["checkpoint_sha256"]
    )


def _aux_allocations() -> dict:
    return {
        "WARMUP": _allocation(coordinator.B200_LEARNER_HOST_ID, 10),
        "GEOMETRY": _allocation(coordinator.B200_LEARNER_HOST_ID, 10),
        "AUX0": _allocation(coordinator.B200_LEARNER_HOST_ID, 10),
        "AUXT": _allocation(coordinator.B200_LEARNER_HOST_ID, 10),
    }


def _prepare_aux(root: Path) -> tuple[dict, dict]:
    _sweep, p1 = _complete_p1(root)
    experiment = coordinator.prepare_experiment(
        root,
        p1_recipe_data_authority=p1,
        public_award_transition_authority=_public_award_transition(),
        pointer_upgrade_authority=_pointer_upgrade(),
        warmup_recipe=_warmup_recipe(p1["composite"]),
        selector_rule=_selector_rule(),
        portable_code_identity_sha256=_sha("4"),
        allocations=_aux_allocations(),
    )
    return experiment, p1


def _warmup_result(experiment: dict) -> dict:
    science = experiment["portable_science_identity"]
    return {
        "schema_version": "a1-aux-pointer-warmup-result-v1",
        "status": "complete",
        "sampled_rows": science["warmup_recipe"]["sample_dose"],
        "optimizer_steps": science["warmup_recipe"]["optimizer_steps"],
        "input_initializer_sha256": science["pointer_upgrade_authority"][
            "upgraded_initializer_sha256"
        ],
        "warmed_checkpoint_sha256": _sha("6"),
        "optimizer_sidecar_sha256": _sha("7"),
        "optimizer_sidecar_discarded_for_joint": True,
        "changed_parameter_prefixes": list(coordinator.POINTER_TRAINABLE_PREFIXES),
        "changed_parameter_set_sha256": science["pointer_upgrade_authority"][
            "new_parameter_set_sha256"
        ],
        "inherited_parameter_identity_sha256": _sha("8"),
        "inherited_parameters_bit_identical": True,
        "main_output_max_diff": 0.0,
        "report_sha256": _sha("9"),
        "origin_tool_sha256": _sha("a"),
    }


def _geometry_evidence(experiment: dict) -> dict:
    rule = experiment["portable_science_identity"]["selector_rule"]
    return {
        "schema_version": "a1-aux-gradient-geometry-evidence-v1",
        "status": "complete",
        "warmed_checkpoint_sha256": _sha("6"),
        "probe_manifest_sha256": rule["probe_manifest_sha256"],
        "probe_sampler_seed": rule["probe_sampler_seed"],
        "probe_row_order_sha256": rule["probe_row_order_sha256"],
        "probe_batches": 5,
        "probe_batch_size": 512,
        "shared_parameter_set_sha256": rule["shared_parameter_set_sha256"],
        "batch_shared_parameter_set_sha256": [rule["shared_parameter_set_sha256"]] * 5,
        "per_batch_geometry": [
            {
                "batch_index": index,
                "shared_parameter_set_sha256": rule["shared_parameter_set_sha256"],
                "main_squared_norm_decimal": "20",
                "unit_aux_squared_norm_decimal": "2000",
                "gradient_dot_decimal": "40",
            }
            for index in range(5)
        ],
        "same_forward_graph": True,
        "global_ddp_aggregation": True,
        "optimizer_steps": 0,
        "persistent_state_mutated": False,
        "rng_transactions_by_rank": [
            _geometry_rng_transaction(rank) for rank in range(8)
        ],
        "report_sha256": _sha("b"),
        "origin_tool_sha256": _sha("c"),
    }


def _issue_aux_pair(root: Path) -> tuple[dict, dict]:
    experiment, _p1 = _prepare_aux(root)
    experiment_id = experiment["experiment_id"]
    coordinator.claim_warmup(
        root,
        experiment_id,
        observed_allocation=experiment["allocations"]["WARMUP"],
        execution=_stage_execution_material(root, experiment_id, "WARMUP")[0],
    )
    _commit_test_stage(root, experiment_id, "WARMUP")
    coordinator.complete_warmup(root, experiment_id, result=_warmup_result(experiment))
    coordinator.claim_geometry(
        root,
        experiment_id,
        observed_allocation=experiment["allocations"]["GEOMETRY"],
        execution=_stage_execution_material(root, experiment_id, "GEOMETRY")[0],
    )
    _commit_test_stage(root, experiment_id, "GEOMETRY")
    coordinator.complete_geometry(
        root, experiment_id, evidence=_geometry_evidence(experiment)
    )
    return experiment, coordinator.issue_pair(root, experiment_id)


def _claim_geometry_for_test(root: Path, experiment: dict) -> None:
    experiment_id = experiment["experiment_id"]
    coordinator.claim_warmup(
        root,
        experiment_id,
        observed_allocation=experiment["allocations"]["WARMUP"],
        execution=_stage_execution_material(root, experiment_id, "WARMUP")[0],
    )
    _commit_test_stage(root, experiment_id, "WARMUP")
    coordinator.complete_warmup(
        root, experiment_id, result=_warmup_result(experiment)
    )
    coordinator.claim_geometry(
        root,
        experiment_id,
        observed_allocation=experiment["allocations"]["GEOMETRY"],
        execution=_stage_execution_material(root, experiment_id, "GEOMETRY")[0],
    )
    _commit_test_stage(root, experiment_id, "GEOMETRY")


def _arm_result(pair: dict, arm_id: str, character: str) -> dict:
    composite = pair["joint"]["composite"]
    alphabet = "0123456789abcdef"
    start = sum(ord(value) for value in character) % (len(alphabet) - 3)
    return {
        "schema_version": "a1-aux-joint-arm-result-v1",
        "status": "complete",
        "pair_id": pair["pair_id"],
        "arm_id": arm_id,
        "aux_subgoal_loss_weight_decimal": pair["arms"][arm_id][
            "aux_subgoal_loss_weight_decimal"
        ],
        "initializer_sha256": pair["joint"]["initializer_sha256"],
        "sampled_rows": 524_288,
        "optimizer_steps": 128,
        "fresh_adam": True,
        "optimizer_restored": False,
        "effective_recipe_sha256": pair["joint"]["effective_recipe_sha256"],
        "payload_inventory_sha256": composite["payload_inventory_sha256"],
        "validation_split_receipt_sha256": composite["validation_split_receipt_sha256"],
        "sampler_identity_sha256": composite["sampler_identity_sha256"],
        "sample_order_sha256": composite["sample_order_sha256"],
        "checkpoint_sha256": _sha(alphabet[start]),
        "optimizer_sidecar_sha256": _sha(alphabet[start + 1]),
        "report_sha256": _sha(alphabet[start + 2]),
        "origin_tool_sha256": coordinator._repo_tool_sha256(
            "tools/a1_one_dose_train.py"
        ),
    }


def _aux_panel_receipts(
    root: Path, pair: dict, *, passed: bool = True
) -> tuple[Path, Path]:
    plan = pair["portable_science_identity"]["evaluation_plan"]
    internal = {
        "AUX0": [1000] * plan["internal_pairs_per_arm"],
        "AUXT": [1500 if passed else 900] * plan["internal_pairs_per_arm"],
    }
    external = {arm: [500] * plan["external_games_per_arm"] for arm in coordinator.ARMS}
    shared = {
        "family": "AUX",
        "authority_id": pair["pair_id"],
        "arms": coordinator.ARMS,
        "checkpoints": {
            arm: coordinator._artifact(  # noqa: SLF001
                root,
                pair["experiment_id"],
                f"70-{arm.lower()}-terminal.json",
            )["result"]["checkpoint_sha256"]
            for arm in coordinator.ARMS
        },
        "baseline_checkpoint_sha256": plan["baseline_checkpoint_sha256"],
        "search_operator_sha256": plan["search_operator_sha256"],
        "origin_tool_sha256": plan["panel_origin_tool_sha256"],
    }
    return (
        _write_panel_receipt(
            root,
            panel_kind="internal",
            cohort_sha256=plan["internal_cohort_sha256"],
            points_milli=internal,
            **shared,
        ),
        _write_panel_receipt(
            root,
            panel_kind="external",
            cohort_sha256=plan["external_cohort_sha256"],
            points_milli=external,
            **shared,
        ),
    )


def _complete_aux_pair(root: Path, *, passed: bool = True) -> tuple[dict, dict, dict]:
    experiment, pair = _issue_aux_pair(root)
    experiment_id = experiment["experiment_id"]
    for index, arm_id in enumerate(coordinator.ARMS):
        result = _arm_result(pair, arm_id, "d" if arm_id == "AUX0" else "h")
        execution, material = _central_execution_material(
            root, experiment_id, arm_id
        )
        coordinator.claim_arm(
            root,
            experiment_id,
            arm_id=arm_id,
            observed_allocation=pair["allocations"][arm_id],
            execution=execution,
        )
        coordinator.load_aux_pair_executor_authority(
            root,
            experiment_id,
            arm_id=arm_id,
            observed_allocation=pair["allocations"][arm_id],
        )
        evidence = _publish_central_execution_evidence(
            root, stage=arm_id, material=material, result=result
        )
        coordinator.complete_arm(
            root,
            experiment_id,
            arm_id=arm_id,
            result=result,
            execution_evidence=evidence,
        )
    coordinator.claim_pair_evaluation(root, experiment_id, execution=_execution("7"))
    internal, external = _aux_panel_receipts(root, pair, passed=passed)
    coordinator.complete_pair_evaluation(
        root,
        experiment_id,
        internal_panel_receipt_path=internal,
        external_panel_receipt_path=external,
    )
    terminal = coordinator.finalize_pair(root, experiment_id)
    return experiment, pair, terminal


def _final_receipts(root: Path, pair: dict) -> dict[str, Path]:
    composite = pair["joint"]["composite"]
    routing_path = root / "final-routing.json"
    coordinator._write_once(
        routing_path,
        {
            "schema_version": "a1-mixed-component-routing-authority-v2",
            "status": "complete",
            "descriptor_sha256": composite["descriptor_sha256"],
            "payload_inventory_sha256": composite["payload_inventory_sha256"],
            "component_ids": list(coordinator.COMPONENT_IDS),
            "component_routes": {
                "current_producer": "authoritative_v1",
                "recent_history": "authoritative_v1",
                "hard_negative": "authoritative_v1",
                "historical_replay": "legacy_zero_v0",
            },
            "component_row_counts": {
                component: 1_000 for component in coordinator.COMPONENT_IDS
            },
            "legacy_slot12_all_zero": True,
            "legacy_slot12_nonzero_count": 0,
            "legacy_slot12_evidence_sha256": _sha("9"),
            "authoritative_slot12_evidence_sha256": _sha("a"),
            "ordered_row_routing_evidence_sha256": _sha("b"),
            "per_row_component_authenticated": True,
            "mixed_authoritative_transition_approved": True,
            "model_slot12_zero_initialization_required": True,
            "origin_tool_sha256": coordinator._repo_tool_sha256(
                "tools/a1_scientific_evidence.py"
            ),
        },
    )
    sampling_path = root / "final-sampling.json"
    sample = _sample_receipt(final=True, composite=composite)
    sample.pop("state_sha256")
    coordinator._write_once(sampling_path, sample)
    descriptor_path = root / "composite.json"
    sampling_rows_path = root / "final-rows.jsonl"
    p1_rows_path = root / "p1-rows.jsonl"
    if not descriptor_path.exists():
        descriptor_path.write_text("{}\n", encoding="utf-8")
    sampling_rows_path.write_text("{}\n", encoding="utf-8")
    if not p1_rows_path.exists():
        p1_rows_path.write_text("{}\n", encoding="utf-8")
    return {
        "composite_descriptor_path": descriptor_path,
        "component_routing_receipt_path": routing_path,
        "sampling_receipt_path": sampling_path,
        "sampling_rows_path": sampling_rows_path,
        "p1_sample_rows_path": p1_rows_path,
    }


def _slot12_evidence(
    root: Path,
    final: dict,
    *,
    candidate_sha256: str = _sha("f"),
    learned_signal: bool = True,
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    treatment = final["selected_aux_decision"] == "AUXT"
    initializer_sha = (
        final["initializer_authority"]["reference_warmup_terminal"]["result"][
            "warmed_checkpoint_sha256"
        ]
        if treatment
        else final["initializer_authority"]["public_award_transition_authority"][
            "transitioned_checkpoint"
        ]["sha256"]
    )
    initializer_path = root / f"initializer-{candidate_sha256[-8:]}.pt"
    candidate_path = root / f"candidate-{candidate_sha256[-8:]}.pt"
    initializer_path.write_bytes(b"initializer")
    candidate_path.write_bytes(b"candidate")
    candidate_sha256 = "sha256:" + hashlib.sha256(b"candidate").hexdigest()
    initializer_path.chmod(0o444)
    candidate_path.chmod(0o444)
    zero_path = root / f"slot12-zero-{candidate_sha256[-8:]}.json"
    zero = {
        "schema_version": "a1-initializer-slot12-zero-evidence-v1",
        "status": "complete",
        "measurement_phase": "pre_optimizer",
        "initializer_checkpoint_sha256": initializer_sha,
        "model_slot12_parameter_set_sha256": _sha("a"),
        "model_slot12_parameter_count": 256,
        "initializer_slot12_max_abs_decimal": "0",
        "initializer_slot12_zero_evidence_sha256": _sha("b"),
        "origin_tool_sha256": coordinator._repo_tool_sha256(
            "tools/a1_scientific_evidence.py"
        ),
    }
    coordinator._write_once(zero_path, zero)
    delta_path = root / f"slot12-delta-{candidate_sha256[-8:]}.json"
    delta = {
        "schema_version": "a1-trained-model-slot12-delta-evidence-v1",
        "status": "complete",
        "measurement_phase": "post_optimizer",
        "initializer_checkpoint_sha256": initializer_sha,
        "candidate_checkpoint_sha256": candidate_sha256,
        "model_slot12_parameter_set_sha256": _sha("a"),
        "model_slot12_parameter_count": 256,
        "initializer_slot12_max_abs_decimal": "0",
        "candidate_slot12_max_abs_decimal": "0.125" if learned_signal else "0",
        "slot12_delta_max_abs_decimal": "0.125" if learned_signal else "0",
        "slot12_delta_l2_decimal": "0.5" if learned_signal else "0",
        "candidate_slot12_nonzero_count": 128 if learned_signal else 0,
        "candidate_slot12_finite": True,
        "learned_signal_observed": learned_signal,
        "slot12_delta_evidence_sha256": _sha("c"),
        "origin_tool_sha256": coordinator._repo_tool_sha256(
            "tools/a1_scientific_evidence.py"
        ),
    }
    coordinator._write_once(delta_path, delta)
    return {
        "initializer_checkpoint_path": initializer_path,
        "candidate_checkpoint_path": candidate_path,
        "initializer_slot12_zero_receipt_path": zero_path,
        "trained_slot12_delta_receipt_path": delta_path,
        "zero_receipt": coordinator._load_json(zero_path, where="test zero"),
        "delta_receipt": coordinator._load_json(delta_path, where="test delta"),
    }


def _recovery_gate_path(
    root: Path, final: dict, *, verdict: str = "continue"
) -> Path:
    recovery = final["diagnostic_p1_selection_authority"]["recovery_authority"]
    payload = {
        "schema_version": "a1-v5-recovery-full-gate-authority-v1",
        "inputs": {},
        "recovery_authority": recovery,
        "contract": {},
        "candidate": {
            "sha256": coordinator._artifact(  # noqa: SLF001
                root,
                final["experiment_id"],
                "95-final-terminal.json",
            )["result"]["checkpoint_sha256"]
        },
        "strict_h1_parent_gate": {
            "passed": True,
            "baseline": recovery["recovered_generator"],
        },
        "f7_non_regression_veto": {
            "passed": True,
            "baseline": recovery["safety_reference_unproven_predecessor"],
            "verdict": verdict,
        },
        "policy": {
            "dual_baseline_conjunctive": True,
            "strict_h1_over_recovered_parent": True,
            "f7_h0_veto": True,
            "fresh_cohorts_required": True,
            "promotion_eligible": True,
            "auto_promotion": False,
        },
    }
    payload["authority_sha256"] = coordinator._digest(payload)
    path = root / f"recovery-gate-authority-{verdict}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _final_result(final: dict, evidence: dict[str, object]) -> dict:
    initializer = final["initializer_authority"]
    treatment = final["selected_aux_decision"] == "AUXT"
    zero = evidence["zero_receipt"]
    delta = evidence["delta_receipt"]
    assert isinstance(zero, dict) and isinstance(delta, dict)
    return {
        "schema_version": "a1-final-replication-result-v1",
        "status": "complete",
        "final_replication_id": final["final_replication_id"],
        "initializer_parent_checkpoint_sha256": _parent()["checkpoint_sha256"],
        "diagnostic_checkpoint_loaded": False,
        "selected_aux_decision": final["selected_aux_decision"],
        "selected_aux_coefficient_decimal": final["selected_aux_coefficient_decimal"],
        "pointer_upgrade_replayed": False,
        "pointer_upgrade_initializer_sha256": None,
        "warmed_checkpoint_sha256": (
            initializer["reference_warmup_terminal"]["result"][
                "warmed_checkpoint_sha256"
            ]
            if treatment
            else None
        ),
        "shared_warmup_initializer_consumed": treatment,
        "initializer_slot12_zero_receipt_state_sha256": zero["state_sha256"],
        "trained_slot12_delta_receipt_state_sha256": delta["state_sha256"],
        "candidate_slot12_finite": delta["candidate_slot12_finite"],
        "candidate_slot12_nonzero_count": delta["candidate_slot12_nonzero_count"],
        "learned_signal_observed": delta["learned_signal_observed"],
        "component_routing_state_sha256": final["component_routing_state_sha256"],
        "sampling_state_sha256": final["sampling_state_sha256"],
        "sampled_rows": coordinator.SHORT_SAMPLE_DOSE,
        "optimizer_steps": coordinator.SHORT_OPTIMIZER_STEPS,
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "amp": "none",
        "fresh_adam": True,
        "optimizer_restored": False,
        "effective_recipe_sha256": final["effective_recipe_sha256"],
        "sampler_seed": coordinator.FINAL_SAMPLER_SEED,
        "sampler_identity_sha256": final["sampling_receipt"]["sampler_identity_sha256"],
        "sample_order_sha256": final["sampling_receipt"]["sample_order_sha256"],
        "row_set_sha256": final["sampling_receipt"]["row_set_sha256"],
        "checkpoint_sha256": delta["candidate_checkpoint_sha256"],
        "optimizer_sidecar_sha256": _sha("e"),
        "report_sha256": _sha("d"),
        "origin_tool_sha256": coordinator._repo_tool_sha256(
            "tools/a1_one_dose_train.py"
        ),
        "full_gate_entry_eligible": True,
    }


def test_legacy_geometry_hash_claim_is_quarantined() -> None:
    authority = coordinator.canonical_geometry_dose_authority(
        dose_receipt_sha256=_sha("a"), dose_replay_sha256=_sha("b")
    )
    assert authority["sample_dose"] == 524_288
    assert authority["optimizer_steps"] == 128
    assert authority["amp"] == "none"
    assert authority["historical_only"] is True
    assert authority["current_central_authority"] is False
    assert authority["replay_verified"] is False


def test_p1_final_lock_must_seal_fp32_short_dose_and_truncated_vp() -> None:
    authority = coordinator.verify_p1_final_lock_authority(_final_lock())
    assert authority["base_recipe"]["sampler_seed"] == 424242
    for field, bad in (
        ("amp", "bf16"),
        ("max_steps", 0),
        ("truncated_vp_margin_value_weight", 0.0),
    ):
        lock = _final_lock()
        lock["base_recipe"][field] = bad
        lock["base_recipe_sha256"] = coordinator._digest(lock["base_recipe"])
        with pytest.raises(coordinator.CoordinatorError, match="selected FP32"):
            coordinator.verify_p1_final_lock_authority(lock)


def test_current_promoted_parent_is_required_and_candidate_chaining_refused() -> None:
    parent = _parent()
    assert parent["checkpoint_sha256"] == (
        "sha256:6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c"
    )
    coordinator.verify_current_parent_authority(
        parent, recovery_authority=_recovery()
    )
    for forbidden in (
        "sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4",
        _sha("d"),
    ):
        drift = copy.deepcopy(parent)
        drift["checkpoint_sha256"] = forbidden
        with pytest.raises(
            coordinator.CoordinatorError, match="exact recovered generator"
        ):
            coordinator.verify_current_parent_authority(
                drift, recovery_authority=_recovery()
            )
    coordinator.verify_pointer_upgrade_authority(
        _pointer_upgrade(),
        expected_parent_sha256=_public_award_transition()[
            "transitioned_checkpoint"
        ]["sha256"],
    )
    wrong_upgrade = _pointer_upgrade()
    wrong_upgrade["source_checkpoint_sha256"] = _sha("d")
    with pytest.raises(coordinator.CoordinatorError, match="legacy/aliased"):
        coordinator.verify_pointer_upgrade_authority(
            wrong_upgrade,
            expected_parent_sha256=_public_award_transition()[
                "transitioned_checkpoint"
            ]["sha256"],
        )


def test_public_award_transition_authority_replays_exact_checkpoint_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    source = (tmp_path / "legacy.pt").resolve()
    transitioned = (tmp_path / "authoritative.pt").resolve()
    receipt_path = (tmp_path / "transition.json").resolve()
    torch.save(
        {
            "model": {
                "entity_graph.player_encoder.0.weight": torch.arange(
                    64, dtype=torch.float32
                ).reshape(4, 16),
                "entity_graph.trunk.weight": torch.eye(4),
            },
            "public_award_feature_contract": "legacy_zero_v0",
            "optimizer_steps": 17,
        },
        source,
    )
    source.chmod(0o444)
    transition_evidence = (
        coordinator.scientific_evidence.build_public_award_transition_initializer(
            source, transitioned
        )
    )
    coordinator.scientific_evidence._atomic_write(  # noqa: SLF001
        receipt_path, transition_evidence
    )
    source_sha = coordinator.scientific_evidence._file_sha256(source)  # noqa: SLF001
    transitioned_sha = coordinator.scientific_evidence._file_sha256(  # noqa: SLF001
        transitioned
    )
    authority = {
        "schema_version": coordinator.PUBLIC_AWARD_TRANSITION_AUTHORITY_SCHEMA,
        "source_checkpoint": {"path": str(source), "sha256": source_sha},
        "transitioned_checkpoint": {
            "path": str(transitioned),
            "sha256": transitioned_sha,
        },
        "receipt": {
            "path": str(receipt_path),
            "file_sha256": coordinator.scientific_evidence._file_sha256(  # noqa: SLF001
                receipt_path
            ),
            "evidence": transition_evidence,
        },
    }
    monkeypatch.setattr(
        coordinator.scientific_evidence,
        "verify_public_award_transition_receipt",
        _REAL_VERIFY_PUBLIC_AWARD_TRANSITION_RECEIPT,
    )
    verified = _REAL_VERIFY_PUBLIC_AWARD_TRANSITION_AUTHORITY(
        authority,
        expected_parent={
            "checkpoint_path": str(source),
            "checkpoint_sha256": source_sha,
        },
    )
    assert verified == authority
    assert (
        verified["receipt"]["evidence"]["legacy_zero_input_function_preserving"]
        is True
    )
    assert verified["receipt"]["evidence"]["optimizer_steps"] == 0


def test_evaluation_design_is_canonical_and_has_no_operator_knobs() -> None:
    for canonical, verifier in (
        (
            coordinator.canonical_p1_evaluation_plan,
            coordinator.verify_p1_evaluation_plan,
        ),
        (
            coordinator.canonical_aux_evaluation_plan,
            coordinator.verify_aux_evaluation_plan,
        ),
    ):
        baseline = _parent()["checkpoint_sha256"]
        plan = canonical(baseline_checkpoint_sha256=baseline)
        assert plan["internal_pairs_per_arm"] == 300
        assert plan["external_games_per_arm"] == 500
        assert plan["search_operator"]["n_full"] == 128
        assert plan["internal_cohort"]["seat_swapped"] is True
        verifier(plan, baseline_checkpoint_sha256=baseline)
        mutations = []
        changed_count = copy.deepcopy(plan)
        changed_count["internal_pairs_per_arm"] = 1
        mutations.append(changed_count)
        changed_cohort = copy.deepcopy(plan)
        changed_cohort["internal_cohort"]["base_seed"] += 1
        changed_cohort["internal_cohort_sha256"] = coordinator._digest(
            changed_cohort["internal_cohort"]
        )
        mutations.append(changed_cohort)
        changed_search = copy.deepcopy(plan)
        changed_search["search_operator"]["n_full"] = 64
        changed_search["search_operator_sha256"] = coordinator._digest(
            changed_search["search_operator"]
        )
        mutations.append(changed_search)
        changed_verifier = copy.deepcopy(plan)
        changed_verifier["panel_origin_tool_sha256"] = _sha("0")
        mutations.append(changed_verifier)
        for mutation in mutations:
            with pytest.raises(coordinator.CoordinatorError, match="canonical"):
                verifier(mutation, baseline_checkpoint_sha256=baseline)


def test_p1_historical_anchor_sweep_is_retired_for_fresh_only_data(
    tmp_path: Path,
) -> None:
    paths, composite = _p1_input_paths(tmp_path)
    with pytest.raises(
        coordinator.CoordinatorError,
        match="historical-anchor sweep is retired.*fresh-only",
    ):
        coordinator.prepare_p1_sweep(
            tmp_path,
            final_lock_authority=_final_lock(),
            composite=composite,
            portable_code_identity_sha256=_sha("a"),
            allocations=_p1_allocations(),
            **paths,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sample_dose", coordinator.SHORT_SAMPLE_DOSE - 1),
        ("sample_order_sha256", _sha("9")),
        ("sampler_seed", coordinator.FINAL_SAMPLER_SEED),
    ),
)
def test_p1_measured_dose_order_seed_drift_refused(
    tmp_path: Path, field: str, value: object
) -> None:
    paths, composite = _p1_input_paths(tmp_path)
    sample = coordinator._load_json(
        paths["p1_sample_receipt_path"], where="test P1 sample"
    )
    sample.pop("state_sha256")
    sample[field] = value
    bad = tmp_path / f"bad-{field}.json"
    coordinator._write_once(bad, sample)
    paths["p1_sample_receipt_path"] = bad
    with pytest.raises(coordinator.CoordinatorError, match="sample|order"):
        coordinator.prepare_p1_sweep(
            tmp_path,
            final_lock_authority=_final_lock(),
            composite=composite,
            portable_code_identity_sha256=_sha("a"),
            allocations=_p1_allocations(),
            **paths,
        )


def test_p1_refuses_cross_recovery_descriptor_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, composite = _p1_input_paths(tmp_path)
    other = _recovery()
    other["recovery_lineage_id"] = _sha("d")
    other["authority_sha256"] = _sha("e")
    monkeypatch.setattr(
        coordinator.v5_recovery,
        "verify_committed_receipt",
        lambda _path: {"authority": other, "receipt": {"status": "committed"}},
    )
    with pytest.raises(coordinator.CoordinatorError, match="recovery authority"):
        coordinator.prepare_p1_sweep(
            tmp_path,
            final_lock_authority=_final_lock(),
            composite=composite,
            portable_code_identity_sha256=_sha("a"),
            allocations=_p1_allocations(),
            **paths,
        )


@pytest.mark.parametrize("mutation", ("strip", "swap"))
def test_p1_refuses_laundered_category_semantics(
    tmp_path: Path, mutation: str
) -> None:
    paths, composite = _p1_input_paths(tmp_path)
    if mutation == "strip":
        composite.pop("category_semantics")
    else:
        composite["category_semantics"] = copy.deepcopy(
            composite["category_semantics"]
        )
        composite["category_semantics"]["recent_history"]["semantic"] = (
            "recent_history"
        )
        composite["category_semantics_sha256"] = coordinator._digest(
            composite["category_semantics"]
        )
    with pytest.raises(
        coordinator.CoordinatorError, match="shape drift|category semantics"
    ):
        coordinator.prepare_p1_sweep(
            tmp_path,
            final_lock_authority=_final_lock(),
            composite=composite,
            portable_code_identity_sha256=_sha("a"),
            allocations=_p1_allocations(),
            **paths,
        )


def test_published_executor_authority_is_immutable_and_replayed(
    tmp_path: Path, monkeypatch
) -> None:
    sweep = _issue_p1_sweep(tmp_path)
    coordinator.claim_p1_arm(
        tmp_path,
        sweep["sweep_id"],
        arm_id="K0",
        observed_allocation=sweep["allocations"]["K0"],
        execution=_execution("1"),
    )
    authority = coordinator.load_p1_arm_executor_authority(
        tmp_path,
        sweep["sweep_id"],
        arm_id="K0",
        observed_allocation=sweep["allocations"]["K0"],
    )
    path = (
        tmp_path
        / sweep["sweep_id"].removeprefix("sha256:")
        / "p1-15-k0-executor-authority.json"
    )
    published = coordinator.verify_published_executor_authority(path)
    assert published["authority"] == authority
    assert published["file_sha256"] == coordinator.production_executor._sha256(path)

    original = coordinator.load_p1_arm_executor_authority

    def replace_during_replay(*args, **kwargs):
        replay = original(*args, **kwargs)
        raw = path.read_bytes()
        path.unlink()
        path.write_bytes(raw)
        path.chmod(0o444)
        return replay

    monkeypatch.setattr(
        coordinator, "load_p1_arm_executor_authority", replace_during_replay
    )
    with pytest.raises(coordinator.CoordinatorError, match="replaced|changed"):
        coordinator.verify_published_executor_authority(path)


def test_p1_claim_is_o_excl_idempotent_and_cannot_be_relabelled(
    tmp_path: Path,
) -> None:
    sweep = _issue_p1_sweep(tmp_path)
    with pytest.raises(coordinator.CoordinatorError, match="cannot start before"):
        coordinator.claim_p1_arm(
            tmp_path,
            sweep["sweep_id"],
            arm_id="K3",
            observed_allocation=sweep["allocations"]["K3"],
            execution=_execution("4"),
        )
    arguments = {
        "arm_id": "K0",
        "observed_allocation": sweep["allocations"]["K0"],
        "execution": _execution("1"),
    }
    first = coordinator.claim_p1_arm(tmp_path, sweep["sweep_id"], **arguments)
    assert coordinator.claim_p1_arm(tmp_path, sweep["sweep_id"], **arguments) == first
    changed = dict(arguments)
    changed["execution"] = _execution("4")
    with pytest.raises(coordinator.CoordinatorError, match="already issued"):
        coordinator.claim_p1_arm(tmp_path, sweep["sweep_id"], **changed)
    with pytest.raises(coordinator.CoordinatorError, match="K0/K3/K10"):
        coordinator.claim_p1_arm(
            tmp_path,
            sweep["sweep_id"],
            arm_id="K0-retry",
            observed_allocation=sweep["allocations"]["K0"],
            execution=_execution("1"),
        )


def test_concurrent_conflicting_p1_claim_has_one_winner(tmp_path: Path) -> None:
    sweep = _issue_p1_sweep(tmp_path)

    def claim(character: str):
        try:
            return coordinator.claim_p1_arm(
                tmp_path,
                sweep["sweep_id"],
                arm_id="K0",
                observed_allocation=sweep["allocations"]["K0"],
                execution=_execution(character),
            )
        except coordinator.CoordinatorError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("1", "4")))
    assert sum(isinstance(value, dict) for value in results) == 1
    assert (
        sum(isinstance(value, coordinator.CoordinatorError) for value in results) == 1
    )


def test_p1_adjudication_requires_all_fixed_terminals(tmp_path: Path) -> None:
    sweep = _issue_p1_sweep(tmp_path)
    with pytest.raises(coordinator.CoordinatorError, match="all K0/K3/K10"):
        coordinator.adjudicate_p1_sweep(
            tmp_path,
            sweep["sweep_id"],
            selected_arm="K0",
            applied_selection_rule_sha256=sweep["selection_rule_sha256"],
        )


def test_typed_composite_requires_exact_mix_and_truncation_proof() -> None:
    for mutation in ("ratio", "scope"):
        composite = _composite()
        if mutation == "ratio":
            composite["component_sampling_ratios"] = [0.80, 0.10, 0.05, 0.05]
        else:
            composite["complete_game_inputs"] = False
        with pytest.raises(coordinator.CoordinatorError, match="80/15/5|truncation"):
            coordinator._verify_composite(composite)


def test_kl_row_identity_binds_mask_facts_and_native_preflight_is_replayed() -> None:
    rows = _eligibility_rows()
    first = next(rows)
    first["prior_policy_present"] = not first["prior_policy_present"]

    def mutated_rows():
        yield first
        yield from rows

    with pytest.raises(coordinator.CoordinatorError, match="bind mask facts"):
        coordinator.build_p1_kl_eligibility_authority(
            composite=_composite(), sampled_row_evidence=mutated_rows()
        )
    receipt = _native_admission()
    receipt["hosts"][coordinator.B200_LEARNER_HOST_ID]["native_wheel_sha256"] = _sha(
        "0"
    )
    unsigned = dict(receipt)
    unsigned.pop("state_sha256")
    receipt["state_sha256"] = coordinator._digest(unsigned)
    with pytest.raises(coordinator.CoordinatorError, match="native preflight"):
        coordinator.build_native_runtime_authority(learner_admission_receipt=receipt)

    identity_drift = _native_admission()
    identity_drift["hosts"][coordinator.B200_LEARNER_HOST_ID]["machine_id"] = "0" * 32
    unsigned = dict(identity_drift)
    unsigned.pop("state_sha256")
    identity_drift["state_sha256"] = coordinator._digest(unsigned)
    with pytest.raises(coordinator.CoordinatorError, match="native preflight"):
        coordinator.build_native_runtime_authority(
            learner_admission_receipt=identity_drift
        )


def test_native_authority_derives_from_single_runtime_contract(monkeypatch) -> None:
    receipt = _native_admission()
    expected = coordinator.production_executor._native_wheel_release_identity()
    monkeypatch.setattr(coordinator, "NATIVE_WHEEL_SHA256", _sha("0"))
    authority = coordinator.build_native_runtime_authority(
        learner_admission_receipt=receipt
    )
    assert authority["wheel_sha256"] == expected["sha256"]
    assert authority["capabilities"] == expected["required_capabilities"]


def test_all_stages_must_reuse_the_exact_solo_learner() -> None:
    p1 = _p1_allocations()
    p1["K3"]["pci_bus_ids"][0] = "0000:ff:00.0"
    with pytest.raises(coordinator.CoordinatorError, match="sequentially reuse"):
        coordinator._verify_p1_scheduled_allocations(p1)
    aux = _aux_allocations()
    aux["AUXT"]["pci_bus_ids"][0] = "0000:ff:00.0"
    with pytest.raises(coordinator.CoordinatorError, match="sequentially reuse"):
        coordinator._verify_aux_scheduled_allocations(aux)


def test_aux_refuses_fabricated_noncentral_p1_authority(tmp_path: Path) -> None:
    fake = {
        "schema_version": coordinator.P1_RECIPE_DATA_AUTHORITY_SCHEMA,
        "sweep_id": _sha("a"),
        "selected_arm": "K0",
        "central_authority": True,
        "selection_receipt_sha256": _sha("b"),
        "selection_replay_sha256": _sha("c"),
        "replay_verified": True,
        "scientific_role": "diagnostic_recipe_selection",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "requires_independent_final_replication": True,
        "effective_recipe": _base_recipe(),
        "effective_recipe_sha256": coordinator._digest(_base_recipe()),
        "composite": _composite(),
        "recovery_authority": _recovery(),
        "current_parent_authority": _parent(),
        "native_runtime_authority": _native(),
        "native_learner_admission_receipt": _native_admission(),
        "p1_sample_evidence_receipt": _sample_receipt(),
        "recovery_component_semantics": coordinator.recovery_component_semantics(
            _recovery(), _category_semantics()
        ),
    }
    with pytest.raises(coordinator.CoordinatorError, match="experiment directory"):
        coordinator.prepare_experiment(
            tmp_path,
            p1_recipe_data_authority=fake,
            public_award_transition_authority=_public_award_transition(),
            pointer_upgrade_authority=_pointer_upgrade(),
            warmup_recipe=_warmup_recipe(),
            selector_rule=_selector_rule(),
            portable_code_identity_sha256=_sha("f"),
            allocations=_aux_allocations(),
        )


def test_legacy_cls_upgrade_and_adaptive_warmup_are_refused(tmp_path: Path) -> None:
    pointer = _pointer_upgrade()
    pointer["module"] = "entity_graph.aux_subgoal_heads.v1"
    with pytest.raises(coordinator.CoordinatorError, match="legacy/aliased"):
        coordinator.verify_pointer_upgrade_authority(
            pointer,
            expected_parent_sha256=_public_award_transition()[
                "transitioned_checkpoint"
            ]["sha256"],
        )
    warmup = _warmup_recipe()
    warmup["checkpoint_selection"]["adaptive_best_checkpoint"] = True
    with pytest.raises(coordinator.CoordinatorError, match="preregistered"):
        coordinator.verify_warmup_recipe(warmup)


def test_aux_warmup_crash_resume_is_exact_and_mutation_refused(tmp_path: Path) -> None:
    experiment, _p1 = _prepare_aux(tmp_path)
    experiment_id = experiment["experiment_id"]
    claim = coordinator.claim_warmup(
        tmp_path,
        experiment_id,
        observed_allocation=experiment["allocations"]["WARMUP"],
        execution=_stage_execution_material(
            tmp_path, experiment_id, "WARMUP"
        )[0],
    )
    state = coordinator.inspect_state(tmp_path, experiment_id)
    assert state["warmup_claimed"] is True
    assert state["warmup_terminal"] is False
    assert (
        coordinator.claim_warmup(
            tmp_path,
            experiment_id,
            observed_allocation=experiment["allocations"]["WARMUP"],
            execution=_stage_execution_material(
                tmp_path, experiment_id, "WARMUP"
            )[0],
        )
        == claim
    )
    with pytest.raises(coordinator.CoordinatorError, match="already issued"):
        coordinator.claim_warmup(
            tmp_path,
            experiment_id,
            observed_allocation=experiment["allocations"]["WARMUP"],
            execution=_execution("6"),
        )


def test_warmup_terminal_proves_only_pointer_heads_changed(tmp_path: Path) -> None:
    experiment, _p1 = _prepare_aux(tmp_path)
    experiment_id = experiment["experiment_id"]
    coordinator.claim_warmup(
        tmp_path,
        experiment_id,
        observed_allocation=experiment["allocations"]["WARMUP"],
        execution=_stage_execution_material(
            tmp_path, experiment_id, "WARMUP"
        )[0],
    )
    _commit_test_stage(tmp_path, experiment_id, "WARMUP")
    bad = _warmup_result(experiment)
    bad["inherited_parameters_bit_identical"] = False
    with pytest.raises(coordinator.CoordinatorError, match="head-only"):
        coordinator.complete_warmup(tmp_path, experiment_id, result=bad)
    result = coordinator.complete_warmup(
        tmp_path, experiment_id, result=_warmup_result(experiment)
    )
    assert result["result"]["optimizer_sidecar_discarded_for_joint"] is True


def test_gradient_selector_is_deterministic_dynamic_and_not_hardcoded_dot02(
    tmp_path: Path,
) -> None:
    experiment, pair = _issue_aux_pair(tmp_path)
    assert pair["selected_aux_coefficient_decimal"] == "0.005"
    assert pair["selected_aux_coefficient"] == pytest.approx(0.005)
    assert set(pair["arms"]) == {"AUX0", "AUXT"}
    assert pair["arms"]["AUX0"]["aux_subgoal_loss_weight"] == 0.0
    assert pair["arms"]["AUXT"]["aux_subgoal_loss_weight"] == pytest.approx(0.005)
    assert pair["joint"]["sample_dose"] == 524_288
    assert pair["joint"]["optimizer_steps"] == 128
    assert pair["joint"]["initializer_sha256"] == _sha("6")
    assert pair["joint"]["warmup_optimizer_sidecar_discarded"] is True
    assert (
        pair["portable_science_identity"]["effective_recipe"]
        == experiment["portable_science_identity"]["effective_recipe"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("same_forward_graph", False),
        ("global_ddp_aggregation", False),
        ("optimizer_steps", 1),
        ("batch_shared_parameter_set_sha256", [_sha("1")] * 4 + [_sha("2")]),
    ),
)
def test_gradient_geometry_mutations_refuse_before_pair(
    tmp_path: Path, field: str, value
) -> None:
    experiment, _p1 = _prepare_aux(tmp_path)
    experiment_id = experiment["experiment_id"]
    _claim_geometry_for_test(tmp_path, experiment)
    evidence = _geometry_evidence(experiment)
    evidence[field] = value
    with pytest.raises(coordinator.CoordinatorError, match="preregistered probe"):
        coordinator.complete_geometry(tmp_path, experiment_id, evidence=evidence)


def test_gradient_selector_refuses_below_minimum(tmp_path: Path) -> None:
    experiment, _p1 = _prepare_aux(tmp_path)
    experiment_id = experiment["experiment_id"]
    _claim_geometry_for_test(tmp_path, experiment)
    evidence = _geometry_evidence(experiment)
    for batch in evidence["per_batch_geometry"]:
        batch["unit_aux_squared_norm_decimal"] = "200000000000"
    with pytest.raises(coordinator.CoordinatorError, match="outside safe range"):
        coordinator.complete_geometry(tmp_path, experiment_id, evidence=evidence)


def test_gradient_geometry_derives_cosine_and_refuses_impossible_dot(
    tmp_path: Path,
) -> None:
    experiment, _p1 = _prepare_aux(tmp_path)
    experiment_id = experiment["experiment_id"]
    _claim_geometry_for_test(tmp_path, experiment)
    evidence = _geometry_evidence(experiment)
    evidence["per_batch_geometry"][0]["gradient_dot_decimal"] = "1000000"
    with pytest.raises(coordinator.CoordinatorError, match="preregistered probe"):
        coordinator.complete_geometry(tmp_path, experiment_id, evidence=evidence)


@pytest.mark.parametrize("mutation", ("not_restored", "different_after_state"))
def test_gradient_geometry_refuses_unrestored_rng_transaction(
    tmp_path: Path, mutation: str
) -> None:
    experiment, _p1 = _prepare_aux(tmp_path)
    experiment_id = experiment["experiment_id"]
    _claim_geometry_for_test(tmp_path, experiment)
    evidence = _geometry_evidence(experiment)
    transaction = evidence["rng_transactions_by_rank"][3]
    if mutation == "not_restored":
        transaction["restored_exactly"] = False
    else:
        after = transaction["after_restore"]
        after["torch_cpu_sha256"] = _sha("9")
        after["state_sha256"] = coordinator._digest(
            {key: value for key, value in after.items() if key != "state_sha256"}
        )
    with pytest.raises(coordinator.CoordinatorError, match="not isolated"):
        coordinator.complete_geometry(tmp_path, experiment_id, evidence=evidence)


@pytest.mark.parametrize(
    ("name", "batch_dot", "expected_cosine", "expected_coefficient"),
    (
        ("aligned", "1", "1", "0.05"),
        ("orthogonal", "0", "0", "0.05"),
    ),
)
def test_gradient_geometry_exact_aggregate_cosine_and_coefficient(
    tmp_path: Path,
    name: str,
    batch_dot: str,
    expected_cosine: str,
    expected_coefficient: str,
) -> None:
    root = tmp_path / name
    root.mkdir()
    experiment, _p1 = _prepare_aux(root)
    experiment_id = experiment["experiment_id"]
    _claim_geometry_for_test(root, experiment)
    evidence = _geometry_evidence(experiment)
    for batch in evidence["per_batch_geometry"]:
        batch["main_squared_norm_decimal"] = "1"
        batch["unit_aux_squared_norm_decimal"] = "1"
        batch["gradient_dot_decimal"] = batch_dot
    terminal = coordinator.complete_geometry(root, experiment_id, evidence=evidence)
    assert terminal["derived_geometry"]["main_gradient_norm_decimal"] == "2.236067977499789696409173669"
    assert terminal["derived_geometry"]["unit_aux_gradient_norm_decimal"] == "2.236067977499789696409173669"
    assert terminal["derived_geometry"]["gradient_cosine_decimal"] == expected_cosine
    assert terminal["raw_coefficient_decimal"] == expected_coefficient
    assert terminal["selected_coefficient_decimal"] == expected_coefficient


def test_arm_claims_are_fixed_allocation_bound_and_executor_echoes_science(
    tmp_path: Path,
) -> None:
    experiment, pair = _issue_aux_pair(tmp_path)
    experiment_id = experiment["experiment_id"]
    with pytest.raises(coordinator.CoordinatorError, match="cannot start before AUX0"):
        coordinator.claim_arm(
            tmp_path,
            experiment_id,
            arm_id="AUXT",
            observed_allocation=pair["allocations"]["AUXT"],
            execution=_execution("6"),
        )
    for index, arm_id in enumerate(coordinator.ARMS):
        result = _arm_result(pair, arm_id, "d" if arm_id == "AUX0" else "h")
        execution, material = _central_execution_material(
            tmp_path, experiment_id, arm_id
        )
        claim = coordinator.claim_arm(
            tmp_path,
            experiment_id,
            arm_id=arm_id,
            observed_allocation=pair["allocations"][arm_id],
            execution=execution,
        )
        authority = coordinator.load_aux_pair_executor_authority(
            tmp_path,
            experiment_id,
            arm_id=arm_id,
            observed_allocation=pair["allocations"][arm_id],
        )
        assert authority["arm_claim"] == claim
        assert (
            authority["aux_pair_contract"]["portable_science_identity"]["composite"]
            == experiment["portable_science_identity"]["composite"]
        )
        assert authority["arm"]["arm_id"] == arm_id
        execution_evidence = _publish_central_execution_evidence(
            tmp_path, stage=arm_id, material=material, result=result
        )
        coordinator.complete_arm(
            tmp_path,
            experiment_id,
            arm_id=arm_id,
            result=result,
            execution_evidence=execution_evidence,
        )
    with pytest.raises(coordinator.CoordinatorError, match="AUX0 and AUXT"):
        coordinator.claim_arm(
            tmp_path,
            experiment_id,
            arm_id="AUXT-retry",
            observed_allocation=pair["allocations"]["AUXT"],
            execution=_execution("7"),
        )
    wrong = copy.deepcopy(pair["allocations"]["AUX0"])
    wrong["gpu_uuids"][0] = "GPU-wrong-0"
    with pytest.raises(coordinator.CoordinatorError, match="exact 8xB200"):
        coordinator.load_aux_pair_executor_authority(
            tmp_path,
            experiment_id,
            arm_id="AUX0",
            observed_allocation=wrong,
        )


def test_pair_terminal_requires_both_exact_fresh_adam_doses(tmp_path: Path) -> None:
    experiment, pair = _issue_aux_pair(tmp_path)
    experiment_id = experiment["experiment_id"]
    control_result = _arm_result(pair, "AUX0", "d")
    control_execution, control_material = _central_execution_material(
        tmp_path, experiment_id, "AUX0"
    )
    coordinator.claim_arm(
        tmp_path,
        experiment_id,
        arm_id="AUX0",
        observed_allocation=pair["allocations"]["AUX0"],
        execution=control_execution,
    )
    coordinator.load_aux_pair_executor_authority(
        tmp_path,
        experiment_id,
        arm_id="AUX0",
        observed_allocation=pair["allocations"]["AUX0"],
    )
    control_evidence = _publish_central_execution_evidence(
        tmp_path, stage="AUX0", material=control_material, result=control_result
    )
    bad = copy.deepcopy(control_result)
    bad["optimizer_restored"] = True
    with pytest.raises(coordinator.CoordinatorError, match="terminal drifted"):
        coordinator.complete_arm(
            tmp_path,
            experiment_id,
            arm_id="AUX0",
            result=bad,
            execution_evidence=control_evidence,
        )
    control = coordinator.complete_arm(
        tmp_path,
        experiment_id,
        arm_id="AUX0",
        result=control_result,
        execution_evidence=control_evidence,
    )
    treatment_result = _arm_result(pair, "AUXT", "h")
    treatment_execution, treatment_material = _central_execution_material(
        tmp_path, experiment_id, "AUXT"
    )
    coordinator.claim_arm(
        tmp_path,
        experiment_id,
        arm_id="AUXT",
        observed_allocation=pair["allocations"]["AUXT"],
        execution=treatment_execution,
    )
    coordinator.load_aux_pair_executor_authority(
        tmp_path,
        experiment_id,
        arm_id="AUXT",
        observed_allocation=pair["allocations"]["AUXT"],
    )
    treatment_evidence = _publish_central_execution_evidence(
        tmp_path,
        stage="AUXT",
        material=treatment_material,
        result=treatment_result,
    )
    with pytest.raises(coordinator.CoordinatorError):
        coordinator.finalize_pair(tmp_path, experiment_id)
    treatment = coordinator.complete_arm(
        tmp_path,
        experiment_id,
        arm_id="AUXT",
        result=treatment_result,
        execution_evidence=treatment_evidence,
    )
    coordinator.claim_pair_evaluation(
        tmp_path, experiment_id, execution=_execution("7")
    )
    internal_panel, external_panel = _aux_panel_receipts(tmp_path, pair, passed=True)
    evaluation = coordinator.complete_pair_evaluation(
        tmp_path,
        experiment_id,
        internal_panel_receipt_path=internal_panel,
        external_panel_receipt_path=external_panel,
    )
    terminal = coordinator.finalize_pair(tmp_path, experiment_id)
    assert terminal["arm_terminal_sha256"] == {
        "AUX0": control["state_sha256"],
        "AUXT": treatment["state_sha256"],
    }
    assert terminal["evaluation_terminal_sha256"] == evaluation["state_sha256"]
    assert terminal["passed"] is True
    assert coordinator.finalize_pair(tmp_path, experiment_id) == terminal


def test_only_independent_final_replication_can_enter_full_gate(
    tmp_path: Path,
) -> None:
    experiment, pair, pair_terminal = _complete_aux_pair(tmp_path, passed=True)
    assert pair["diagnostic_only"] is True
    assert pair["promotion_eligible"] is False
    assert pair_terminal["selected_aux_decision"] == "AUXT"
    receipts = _final_receipts(tmp_path, pair)

    bad_sampling = coordinator._load_json(
        receipts["sampling_receipt_path"], where="test FINAL sampling"
    )
    bad_unsigned = copy.deepcopy(bad_sampling)
    bad_unsigned.pop("state_sha256")
    bad_unsigned["overlap_within_independent_bound"] = False
    bad_path = tmp_path / "bad-final-sampling.json"
    coordinator._write_once(bad_path, bad_unsigned)
    bad_receipts = dict(receipts)
    bad_receipts["sampling_receipt_path"] = bad_path
    with pytest.raises(coordinator.CoordinatorError, match="evidence|independent"):
        coordinator.issue_final_replication(
            tmp_path,
            experiment["experiment_id"],
            **bad_receipts,
        )

    final = coordinator.issue_final_replication(
        tmp_path,
        experiment["experiment_id"],
        **receipts,
    )
    assert final["initializer_authority"]["base_parent_lineage_reloaded"] is True
    assert (
        final["initializer_authority"]["warmup_initializer_role"]
        == "shared_immutable_architecture_initializer"
    )
    assert final["initializer_authority"]["exact_reference_warmup_bytes_reused"] is True
    assert final["selected_aux_decision"] == "AUXT"
    assert final["training"]["sampler_seed"] == 424243
    assert final["final_policy_kl_anchor_weight_decimal"] == "0.0075"
    assert final["effective_recipe"]["policy_kl_anchor_weight"] == 0.0075
    assert final["diagnostic_only"] is False
    evidence = _slot12_evidence(tmp_path, final)
    good_result = _final_result(final, evidence)
    final_execution, final_material = _central_execution_material(
        tmp_path,
        experiment["experiment_id"],
        "FINAL",
        checkpoint_path=evidence["candidate_checkpoint_path"],
    )
    coordinator.claim_final_replication(
        tmp_path,
        experiment["experiment_id"],
        observed_allocation=final["allocation"],
        execution=final_execution,
    )
    executor = coordinator.load_final_replication_executor_authority(
        tmp_path,
        experiment["experiment_id"],
        observed_allocation=final["allocation"],
    )
    assert executor["final_replication_authority"] == final
    final_execution_evidence = _publish_central_execution_evidence(
        tmp_path,
        stage="FINAL",
        material=final_material,
        result=good_result,
    )
    evidence_paths = {
        key: value
        for key, value in evidence.items()
        if key.endswith("_path")
    }
    bad_result = copy.deepcopy(good_result)
    bad_result["checkpoint_sha256"] = coordinator._artifact(  # noqa: SLF001
        tmp_path,
        experiment["experiment_id"],
        "70-auxt-terminal.json",
    )["result"]["checkpoint_sha256"]
    with pytest.raises(coordinator.CoordinatorError, match="promotion-safe"):
        coordinator.complete_final_replication(
            tmp_path,
            experiment["experiment_id"],
            result=bad_result,
            execution_evidence=final_execution_evidence,
            **evidence_paths,
        )
    zero_signal_evidence = _slot12_evidence(
        tmp_path / "zero-signal-slot12",
        final,
        learned_signal=False,
    )
    zero_signal_paths = {
        key: value
        for key, value in zero_signal_evidence.items()
        if key.endswith("_path")
    }
    with pytest.raises(coordinator.CoordinatorError, match="promotion-safe"):
        coordinator.complete_final_replication(
            tmp_path,
            experiment["experiment_id"],
            result=_final_result(final, zero_signal_evidence),
            execution_evidence=final_execution_evidence,
            **zero_signal_paths,
        )
    terminal = coordinator.complete_final_replication(
        tmp_path,
        experiment["experiment_id"],
        result=good_result,
        execution_evidence=final_execution_evidence,
        **evidence_paths,
    )
    assert terminal["promotion_eligible"] is False
    assert terminal["eligible_for_full_gate"] is True
    assert terminal["full_gate_required"] is True
    with pytest.raises(coordinator.CoordinatorError, match="dual-baseline"):
        coordinator.load_final_gate_entry_authority(
            tmp_path,
            experiment["experiment_id"],
            recovery_gate_authority_path=_recovery_gate_path(
                tmp_path, final, verdict="H0"
            ),
        )
    gate = coordinator.load_final_gate_entry_authority(
        tmp_path,
        experiment["experiment_id"],
        recovery_gate_authority_path=_recovery_gate_path(tmp_path, final),
    )
    assert gate["candidate_checkpoint_sha256"] == good_result["checkpoint_sha256"]
    assert gate["auto_promotion"] is False


def test_failed_aux_selects_control_for_independent_final(tmp_path: Path) -> None:
    experiment, pair, pair_terminal = _complete_aux_pair(tmp_path, passed=False)
    assert pair_terminal["selected_aux_decision"] == "AUX0"
    receipts = _final_receipts(tmp_path, pair)
    final = coordinator.issue_final_replication(
        tmp_path,
        experiment["experiment_id"],
        **receipts,
    )
    assert final["selected_aux_coefficient_decimal"] == "0"
    assert final["initializer_authority"]["pointer_upgrade_authority"] is None
    assert final["initializer_authority"]["public_award_transition_authority"] == (
        experiment["portable_science_identity"][
            "public_award_transition_authority"
        ]
    )
    assert final["effective_recipe"]["aux_subgoal_heads"] is False


def test_portable_identity_is_path_invariant_and_allocation_drift_is_refused(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    first, _ = _prepare_aux(first_root)
    second, _ = _prepare_aux(second_root)
    assert first["experiment_id"] == second["experiment_id"]
    assert (
        first["portable_science_identity_sha256"]
        == second["portable_science_identity_sha256"]
    )

    _sweep, p1 = _complete_p1(tmp_path / "three")
    allocations = _aux_allocations()
    allocations["AUXT"]["pci_bus_ids"][0] = "0000:ff:00.0"
    with pytest.raises(coordinator.CoordinatorError, match="sequentially reuse"):
        coordinator.prepare_experiment(
            tmp_path / "three",
            p1_recipe_data_authority=p1,
            public_award_transition_authority=_public_award_transition(),
            pointer_upgrade_authority=_pointer_upgrade(),
            warmup_recipe=_warmup_recipe(p1["composite"]),
            selector_rule=_selector_rule(),
            portable_code_identity_sha256=_sha("4"),
            allocations=allocations,
        )


def test_portable_projection_ignores_locations_but_binds_science(
    tmp_path: Path,
) -> None:
    experiment, _p1 = _prepare_aux(tmp_path)
    science = experiment["portable_science_identity"]
    expected = coordinator._portable_science_digest(science)

    relocated = copy.deepcopy(science)
    for composite in (
        relocated["composite"],
        relocated["p1_recipe_data_authority"]["composite"],
    ):
        composite["descriptor_sha256"] = _sha("0")
        composite["data_fingerprint"] = _sha("1")
        composite["source_authority"] = {
            "path": "/different/canonical/root/source-authority.json",
            "file_sha256": _sha("2"),
            "authority_sha256": _sha("3"),
        }
    for sample in (
        relocated["p1_sample_evidence_receipt"],
        relocated["p1_recipe_data_authority"]["p1_sample_evidence_receipt"],
    ):
        sample["descriptor_sha256"] = _sha("0")
        sample["sampler_identity_sha256"] = _sha("4")
        sample["source_authority"] = copy.deepcopy(
            relocated["composite"]["source_authority"]
        )
        sample["state_sha256"] = _sha("5")
    relocated["p1_recipe_data_authority"]["sweep_id"] = _sha("6")
    relocated["p1_recipe_data_authority"]["selection_receipt_sha256"] = _sha("7")
    relocated["p1_recipe_data_authority"]["selection_replay_sha256"] = _sha("8")
    relocated["geometry_dose_authority"][
        "source_p1_recipe_data_authority_sha256"
    ] = _sha("9")
    relocated["geometry_dose_authority"]["source_p1_sample_state_sha256"] = _sha(
        "a"
    )
    relocated["warmup_recipe"]["descriptor_sha256"] = _sha("0")
    relocated["warmup_recipe"]["data_fingerprint"] = _sha("1")
    relocated["warmup_recipe"]["sampler_identity_sha256"] = _sha("4")
    relocated["recovery_authority"]["recovered_generator"]["path"] = (
        "/different/checkpoints/recovered.pt"
    )
    relocated["recovery_authority"][
        "safety_reference_unproven_predecessor"
    ]["path"] = "/different/checkpoints/safety.pt"
    relocated["recovery_authority"]["producer_identity"]["checkpoint"]["path"] = (
        "/different/checkpoints/producer.pt"
    )
    assert coordinator._portable_science_digest(relocated) == expected

    semantic_mutations = []
    payload = copy.deepcopy(science)
    payload["composite"]["payload_inventory_sha256"] = _sha("0")
    semantic_mutations.append(payload)
    for field in (
        "learner_recipe_overrides_sha256",
        "aux_subgoal_target_contract_sha256",
        "public_award_feature_transition_contract_sha256",
        "source_authority_semantic_sha256",
    ):
        contract = copy.deepcopy(science)
        contract["composite"][field] = _sha("0")
        semantic_mutations.append(contract)
    order = copy.deepcopy(science)
    order["p1_sample_evidence_receipt"]["sample_order_sha256"] = _sha("0")
    semantic_mutations.append(order)
    relation = copy.deepcopy(science)
    relation["recovery_component_semantics"]["recent_history"]["semantic"] = (
        "recent_history"
    )
    semantic_mutations.append(relation)
    recipe = copy.deepcopy(science)
    recipe["effective_recipe"]["lr"] = 9.0e-4
    semantic_mutations.append(recipe)
    for mutation in semantic_mutations:
        assert coordinator._portable_science_digest(mutation) != expected


def test_hash_chain_corruption_is_detected(tmp_path: Path) -> None:
    experiment, _p1 = _prepare_aux(tmp_path)
    experiment_id = experiment["experiment_id"]
    coordinator.claim_warmup(
        tmp_path,
        experiment_id,
        observed_allocation=experiment["allocations"]["WARMUP"],
        execution=_stage_execution_material(
            tmp_path, experiment_id, "WARMUP"
        )[0],
    )
    directory = tmp_path / experiment_id.removeprefix("sha256:")
    claim = directory / "10-warmup-claim.json"
    claim.chmod(0o644)
    claim.write_text(
        claim.read_text().replace('"stage": "WARMUP"', '"stage": "BROKEN"')
    )
    with pytest.raises(coordinator.CoordinatorError, match="immutable|digest drift"):
        coordinator.inspect_state(tmp_path, experiment_id)
