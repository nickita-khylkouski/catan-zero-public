from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from tools import a1_aux_pair_coordinator as coordinator


def _sha(character: str) -> str:
    return "sha256:" + character * 64


_P1_PAYLOAD_SHA = _sha("0")


def _eligibility_rows():
    for index in range(coordinator.SHORT_SAMPLE_DOSE):
        eligible = index % 2 == 0
        facts = {
            "payload_member_sha256": _P1_PAYLOAD_SHA,
            "row_offset": index,
            "component_id": "historical_replay" if eligible else "current_producer",
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


def _composite() -> dict:
    return {
        "schema_version": "a1-typed-64-12-4-20-composite-v1",
        "component_ids": list(coordinator.COMPONENT_IDS),
        "component_sampling_ratios": list(coordinator.COMPONENT_RATIOS),
        "descriptor_sha256": _sha("a"),
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
        "producer_identity": {"tool_sha256": _sha("b")},
        "promotion_proof_recreated": False,
        "dual_baseline_fresh_gate_required": True,
        "promotion_eligible": False,
        "training_proof": False,
        "wave_lineage_mode": "recovery_reference",
        "authority_sha256": _sha("c"),
    }


def _parent() -> dict:
    return coordinator.current_parent_authority_from_recovery(_recovery())


def _sample_receipt(*, final: bool = False) -> dict:
    dose = coordinator.SHORT_SAMPLE_DOSE
    prior = _sample_receipt(final=False) if final else None
    seed = coordinator.FINAL_SAMPLER_SEED if final else coordinator.P1_SAMPLER_SEED
    return coordinator._sealed(
        {
            "schema_version": "a1-authenticated-sample-evidence-v1",
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
            "kl_eligible_rows": dose // 2 if not final else dose // 4,
            "kl_eligible_mass_decimal": "0.5" if not final else "0.25",
            "kl_ordered_evidence_sha256": _sha("1"),
            "kl_eligible_evidence_sha256": _sha("2"),
            "descriptor_sha256": _composite()["descriptor_sha256"],
            "payload_inventory_sha256": _composite()["payload_inventory_sha256"],
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
        "verify_mixed_routing_receipt",
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
    monkeypatch.setattr(
        coordinator.v5_recovery_gate,
        "verify_recovery_gate_authority",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )


def _p1_input_paths(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = root / "composite.json"
    rows = root / "p1-rows.jsonl"
    recovery = root / "recovery.json"
    runtime = root / "runtime.json"
    sample = root / "p1-sample.json"
    descriptor.write_text("{}\n", encoding="utf-8")
    rows.write_text("{}\n", encoding="utf-8")
    recovery.write_text("{}\n", encoding="utf-8")
    # _write_once seals; pass the unsigned measured fixtures.
    native_unsigned = _native_admission()
    native_unsigned.pop("state_sha256")
    coordinator._write_once(runtime, native_unsigned)
    sample_unsigned = _sample_receipt()
    sample_unsigned.pop("state_sha256")
    coordinator._write_once(sample, sample_unsigned)
    return {
        "composite_descriptor_path": descriptor,
        "p1_sample_receipt_path": sample,
        "p1_sample_rows_path": rows,
        "v5_recovery_receipt_path": recovery,
        "native_learner_admission_receipt_path": runtime,
    }


def _issue_p1_sweep(root: Path) -> dict:
    return coordinator.prepare_p1_sweep(
        root,
        final_lock_authority=_final_lock(),
        composite=_composite(),
        portable_code_identity_sha256=_sha("a"),
        allocations=_p1_allocations(),
        **_p1_input_paths(root),
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
    cohort_sha256: str,
    search_operator_sha256: str,
    origin_tool_sha256: str,
    points_milli: dict,
) -> Path:
    path = root / f"{family.lower()}-{panel_kind}-panel.json"
    coordinator._write_once(
        path,
        {
            "schema_version": "a1-fixed-panel-receipt-v1",
            "family": family,
            "panel_kind": panel_kind,
            "authority_id": authority_id,
            "arms": list(arms),
            "arm_checkpoint_sha256": checkpoints,
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
        arm: _p1_result(sweep, arm, chr(ord("d") + index * 4))["checkpoint_sha256"]
        for index, arm in enumerate(coordinator.P1_ARMS)
    }
    shared = {
        "family": "P1",
        "authority_id": sweep["sweep_id"],
        "arms": coordinator.P1_ARMS,
        "checkpoints": checkpoints,
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
        coordinator.claim_p1_arm(
            root,
            sweep["sweep_id"],
            arm_id=arm_id,
            observed_allocation=sweep["allocations"][arm_id],
            execution=_execution(str(index + 1)),
        )
        coordinator.complete_p1_arm(
            root,
            sweep["sweep_id"],
            arm_id=arm_id,
            result=_p1_result(sweep, arm_id, chr(ord("d") + index * 4)),
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
        "source_checkpoint_sha256": _parent()["checkpoint_sha256"],
        "upgraded_initializer_sha256": _sha("a"),
        "receipt_sha256": _sha("b"),
        "receipt_replay_sha256": _sha("c"),
        "flags": dict(coordinator.POINTER_FLAGS),
        "new_parameter_set_sha256": _sha("d"),
        "main_output_max_diff": 0.0,
        "shared_parameters_bit_identical": True,
    }


def _warmup_recipe() -> dict:
    steps = coordinator.SHORT_OPTIMIZER_STEPS
    composite = _composite()
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
        pointer_upgrade_authority=_pointer_upgrade(),
        warmup_recipe=_warmup_recipe(),
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
        execution=_execution("3"),
    )
    coordinator.complete_warmup(root, experiment_id, result=_warmup_result(experiment))
    coordinator.claim_geometry(
        root,
        experiment_id,
        observed_allocation=experiment["allocations"]["GEOMETRY"],
        execution=_execution("4"),
    )
    coordinator.complete_geometry(
        root, experiment_id, evidence=_geometry_evidence(experiment)
    )
    return experiment, coordinator.issue_pair(root, experiment_id)


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
            "AUX0": _arm_result(pair, "AUX0", "d")["checkpoint_sha256"],
            "AUXT": _arm_result(pair, "AUXT", "h")["checkpoint_sha256"],
        },
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
        coordinator.claim_arm(
            root,
            experiment_id,
            arm_id=arm_id,
            observed_allocation=pair["allocations"][arm_id],
            execution=_execution(str(index + 5)),
        )
        coordinator.complete_arm(
            root,
            experiment_id,
            arm_id=arm_id,
            result=_arm_result(pair, arm_id, "d" if arm_id == "AUX0" else "h"),
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
    sample = _sample_receipt(final=True)
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
        else _parent()["checkpoint_sha256"]
    )
    initializer_path = root / f"initializer-{candidate_sha256[-8:]}.pt"
    candidate_path = root / f"candidate-{candidate_sha256[-8:]}.pt"
    initializer_path.write_bytes(b"initializer")
    candidate_path.write_bytes(b"candidate")
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
        "candidate": {"sha256": _sha("f")},
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
        _pointer_upgrade(), expected_parent_sha256=parent["checkpoint_sha256"]
    )
    wrong_upgrade = _pointer_upgrade()
    wrong_upgrade["source_checkpoint_sha256"] = _sha("d")
    with pytest.raises(coordinator.CoordinatorError, match="legacy/aliased"):
        coordinator.verify_pointer_upgrade_authority(
            wrong_upgrade, expected_parent_sha256=parent["checkpoint_sha256"]
        )


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


def test_p1_sweep_has_fixed_one_axis_arms_and_central_selection(tmp_path: Path) -> None:
    sweep, selected = _complete_p1(tmp_path)
    assert list(sweep["arms"]) == ["K0", "K3", "K10"]
    assert [
        sweep["arms"][arm]["policy_kl_anchor_weight_decimal"]
        for arm in coordinator.P1_ARMS
    ] == ["0", "0.015", "0.05"]
    base = sweep["final_lock_authority"]["base_recipe"]
    control_recipe = sweep["arms"]["K0"]["effective_recipe"]
    assert control_recipe["world_size"] == 8
    assert control_recipe["batch_size"] == 512
    assert control_recipe["global_batch_size"] == 4096
    assert {
        key for key in control_recipe if control_recipe.get(key) != base.get(key)
    } == {"world_size", "batch_size"}
    for arm in coordinator.P1_ARMS:
        recipe = sweep["arms"][arm]["effective_recipe"]
        changed = {key for key in recipe if recipe.get(key) != control_recipe.get(key)}
        if arm == "K0":
            assert changed == set()
        else:
            assert changed == {"policy_kl_anchor_weight"}
    assert selected["selected_arm"] == "K3"
    assert selected["effective_recipe"]["policy_kl_anchor_weight"] == 0.015
    assert selected["effective_recipe"]["truncated_vp_margin_value_weight"] == 0.25


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
    paths = _p1_input_paths(tmp_path)
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
            composite=_composite(),
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
        with pytest.raises(coordinator.CoordinatorError, match="64/12/4/20|truncation"):
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
            _recovery()
        ),
    }
    with pytest.raises(coordinator.CoordinatorError, match="experiment directory"):
        coordinator.prepare_experiment(
            tmp_path,
            p1_recipe_data_authority=fake,
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
            pointer, expected_parent_sha256=_parent()["checkpoint_sha256"]
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
        execution=_execution("3"),
    )
    state = coordinator.inspect_state(tmp_path, experiment_id)
    assert state["warmup_claimed"] is True
    assert state["warmup_terminal"] is False
    assert (
        coordinator.claim_warmup(
            tmp_path,
            experiment_id,
            observed_allocation=experiment["allocations"]["WARMUP"],
            execution=_execution("3"),
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
        execution=_execution("3"),
    )
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
    coordinator.claim_warmup(
        tmp_path,
        experiment_id,
        observed_allocation=experiment["allocations"]["WARMUP"],
        execution=_execution("3"),
    )
    coordinator.complete_warmup(
        tmp_path, experiment_id, result=_warmup_result(experiment)
    )
    coordinator.claim_geometry(
        tmp_path,
        experiment_id,
        observed_allocation=experiment["allocations"]["GEOMETRY"],
        execution=_execution("4"),
    )
    evidence = _geometry_evidence(experiment)
    evidence[field] = value
    with pytest.raises(coordinator.CoordinatorError, match="preregistered probe"):
        coordinator.complete_geometry(tmp_path, experiment_id, evidence=evidence)


def test_gradient_selector_refuses_below_minimum(tmp_path: Path) -> None:
    experiment, _p1 = _prepare_aux(tmp_path)
    experiment_id = experiment["experiment_id"]
    coordinator.claim_warmup(
        tmp_path,
        experiment_id,
        observed_allocation=experiment["allocations"]["WARMUP"],
        execution=_execution("3"),
    )
    coordinator.complete_warmup(
        tmp_path, experiment_id, result=_warmup_result(experiment)
    )
    coordinator.claim_geometry(
        tmp_path,
        experiment_id,
        observed_allocation=experiment["allocations"]["GEOMETRY"],
        execution=_execution("4"),
    )
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
    coordinator.claim_warmup(
        tmp_path,
        experiment_id,
        observed_allocation=experiment["allocations"]["WARMUP"],
        execution=_execution("3"),
    )
    coordinator.complete_warmup(
        tmp_path, experiment_id, result=_warmup_result(experiment)
    )
    coordinator.claim_geometry(
        tmp_path,
        experiment_id,
        observed_allocation=experiment["allocations"]["GEOMETRY"],
        execution=_execution("4"),
    )
    evidence = _geometry_evidence(experiment)
    evidence["per_batch_geometry"][0]["gradient_dot_decimal"] = "1000000"
    with pytest.raises(coordinator.CoordinatorError, match="preregistered probe"):
        coordinator.complete_geometry(tmp_path, experiment_id, evidence=evidence)


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
        claim = coordinator.claim_arm(
            tmp_path,
            experiment_id,
            arm_id=arm_id,
            observed_allocation=pair["allocations"][arm_id],
            execution=_execution(str(index + 5)),
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
            == _composite()
        )
        assert authority["arm"]["arm_id"] == arm_id
        coordinator.complete_arm(
            tmp_path,
            experiment_id,
            arm_id=arm_id,
            result=_arm_result(pair, arm_id, "d" if arm_id == "AUX0" else "h"),
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
    coordinator.claim_arm(
        tmp_path,
        experiment_id,
        arm_id="AUX0",
        observed_allocation=pair["allocations"]["AUX0"],
        execution=_execution("5"),
    )
    bad = _arm_result(pair, "AUX0", "d")
    bad["optimizer_restored"] = True
    with pytest.raises(coordinator.CoordinatorError, match="terminal drifted"):
        coordinator.complete_arm(tmp_path, experiment_id, arm_id="AUX0", result=bad)
    control = coordinator.complete_arm(
        tmp_path,
        experiment_id,
        arm_id="AUX0",
        result=_arm_result(pair, "AUX0", "d"),
    )
    coordinator.claim_arm(
        tmp_path,
        experiment_id,
        arm_id="AUXT",
        observed_allocation=pair["allocations"]["AUXT"],
        execution=_execution("6"),
    )
    with pytest.raises(coordinator.CoordinatorError):
        coordinator.finalize_pair(tmp_path, experiment_id)
    treatment = coordinator.complete_arm(
        tmp_path,
        experiment_id,
        arm_id="AUXT",
        result=_arm_result(pair, "AUXT", "h"),
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
    coordinator.claim_final_replication(
        tmp_path,
        experiment["experiment_id"],
        observed_allocation=final["allocation"],
        execution=_execution("7"),
    )
    executor = coordinator.load_final_replication_executor_authority(
        tmp_path,
        experiment["experiment_id"],
        observed_allocation=final["allocation"],
    )
    assert executor["final_replication_authority"] == final

    evidence = _slot12_evidence(tmp_path, final)
    evidence_paths = {
        key: value
        for key, value in evidence.items()
        if key.endswith("_path")
    }
    bad_result = _final_result(final, evidence)
    bad_result["checkpoint_sha256"] = _arm_result(pair, "AUXT", "h")[
        "checkpoint_sha256"
    ]
    with pytest.raises(coordinator.CoordinatorError, match="promotion-safe"):
        coordinator.complete_final_replication(
            tmp_path,
            experiment["experiment_id"],
            result=bad_result,
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
            **zero_signal_paths,
        )
    terminal = coordinator.complete_final_replication(
        tmp_path,
        experiment["experiment_id"],
        result=_final_result(final, evidence),
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
    assert gate["candidate_checkpoint_sha256"] == _sha("f")
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
            pointer_upgrade_authority=_pointer_upgrade(),
            warmup_recipe=_warmup_recipe(),
            selector_rule=_selector_rule(),
            portable_code_identity_sha256=_sha("4"),
            allocations=allocations,
        )


def test_hash_chain_corruption_is_detected(tmp_path: Path) -> None:
    experiment, _p1 = _prepare_aux(tmp_path)
    experiment_id = experiment["experiment_id"]
    coordinator.claim_warmup(
        tmp_path,
        experiment_id,
        observed_allocation=experiment["allocations"]["WARMUP"],
        execution=_execution("3"),
    )
    directory = tmp_path / experiment_id.removeprefix("sha256:")
    claim = directory / "10-warmup-claim.json"
    claim.chmod(0o644)
    claim.write_text(
        claim.read_text().replace('"stage": "WARMUP"', '"stage": "BROKEN"')
    )
    with pytest.raises(coordinator.CoordinatorError, match="immutable|digest drift"):
        coordinator.inspect_state(tmp_path, experiment_id)
