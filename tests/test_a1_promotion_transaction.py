from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
import numpy as np

from tools import a1_promotion_transaction as promotion
from tools import a1_evaluation_pool as evaluation_pool
from tools import a1_promotion_artifacts as artifacts
from tools import a1_one_dose_train as one_dose
from tools import a1_production_l1_rerun as production_l1
from tools import a1_production_gather_retrain as production_gather
from tools.champion_registry import ChampionRegistry
from tools.high_regret_suite_contract import REPLAY_CONTRACT, scope_inventory_sha256


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _contract(
    *,
    n_full: int = 128,
    n_full_wide=None,
    producer: Path | None = None,
    c_scale: float = 0.03,
) -> dict:
    recipe = {
        "world_size": 1,
        "optimizer": "adam",
        "mask_hidden_info": True,
        "symmetry_augment": False,
        "epochs": 1,
        "max_steps": 0,
    }
    producer = producer or Path("/producer.pt")
    search = {
        "belief_chance_spectra": False,
        "c_scale": c_scale,
        "c_visit": 50.0,
        "correct_rust_chance_spectra": True,
        "determinization_min_simulations": 32,
        "determinization_particles": 4,
        "exact_budget_sh": False,
        "exact_budget_sh_min_n": 0,
        "information_set_search": True,
        "lazy_interior_chance": True,
        "max_depth": 80,
        "n_fast": 16,
        "n_full": n_full,
        "n_full_wide": n_full_wide,
        "n_full_wide_threshold": None,
        "p_full": 0.25,
        "play_sh_winner": False,
        "policy_target_min_visits": 0,
        "prior_temperature": 1.0,
        "raw_policy_above_width": None,
        "rescale_noise_floor_c": 0.0,
        "root_wave_batching": False,
        "sigma_eval": 0.98,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
        "uncertainty_backup_a": 0.25,
        "uncertainty_backup_cap": 1.0,
        "uncertainty_backup_exp": 1.0,
        "uncertainty_backup_weighting": False,
        "use_batch_api": True,
        "variance_aware_closed_form_js": False,
        "variance_aware_k": 1.0,
        "variance_aware_q": False,
        "wide_candidates_threshold": 24,
        "wide_roots_always_full": n_full_wide is not None,
        "max_root_candidates": 16,
        "max_root_candidates_wide": 54,
    }
    evaluator = {
        "cache_size": 0,
        "context_fill": 0.0,
        "emit_uncertainty": False,
        "prior_temperature": 1.0,
        "public_observation": True,
        "rust_featurize": False,
        "value_readout": "scalar",
        "value_scale": 1.0,
        "value_squash": "tanh",
    }
    return {
        "contract_id": promotion.HISTORICAL_MARKERLESS_A1_CONTRACT["contract_id"],
        "contract_sha256": promotion.HISTORICAL_MARKERLESS_A1_CONTRACT[
            "contract_sha256"
        ],
        "science": {
            "search_operator": search,
            "effective_search_config": search,
            "evaluator": evaluator,
            "learner_training_recipe": recipe,
            "learner_training_recipe_sha256": promotion._digest_value(recipe),
            "learner_value_objective": {"value_readout": "scalar"},
        },
        "generation": {"max_decisions": 600},
        "checkpoints": [
            {
                "role": "producer",
                "path": str(producer),
                "sha256": promotion._sha256(producer)
                if producer.is_file()
                else "sha256:" + "f" * 64,
            }
        ],
    }


def _checkpoint_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": promotion._sha256(path)}


def _write_matched_quick_screen(
    path: Path,
    *,
    intermediate: list[dict],
    terminal: Path,
    baseline: Path,
    evaluation_binding: dict,
    points_by_step: dict[int, float],
) -> dict:
    checkpoints = {
        64: {
            "path": intermediate[0]["checkpoint"],
            "sha256": intermediate[0]["checkpoint_sha256"],
        },
        96: {
            "path": intermediate[1]["checkpoint"],
            "sha256": intermediate[1]["checkpoint_sha256"],
        },
        128: _checkpoint_ref(terminal),
    }
    ordered_game_keys = [
        {"game_seed": 9_000_000 + pair, "orientation": orientation}
        for pair in range(100)
        for orientation in ("candidate_blue", "candidate_red")
    ]
    game_keys_sha256 = promotion._digest_value(ordered_game_keys)
    effective_search_config = {
        "mode": "cross_net",
        "n_full": 128,
        "public_observation": True,
        "information_set_search": True,
        "determinization_particles": 4,
        "symmetry_averaged_eval": True,
    }
    planned_engine_identity = promotion._canonical_internal_h2h_engine_identity()  # noqa: SLF001
    runtime_engine_identity = dict(planned_engine_identity)
    search_configuration = {
        "schema_version": "a1-matched-quick-screen-search-v2",
        "training_parent": _checkpoint_ref(baseline),
        "baseline_checkpoint": _checkpoint_ref(baseline),
        "effective_search_config": effective_search_config,
        "search_rng_contract": promotion.INTERNAL_H2H_SEARCH_RNG_CONTRACT,
        "evaluation_binding": evaluation_binding,
        "planned_engine_identity": planned_engine_identity,
        "engine_identity": runtime_engine_identity,
    }
    search_sha256 = promotion._digest_value(search_configuration)
    half_points = {step: int(round(points_by_step[step] * 2)) for step in (64, 96, 128)}
    best_half_points = max(half_points.values())
    selected_step = min(
        step
        for step in (64, 96, 128)
        if (best_half_points - half_points[step]) * 10_000
        <= promotion.MATCHED_QUICK_SCREEN_SELECTION_RULE[
            "max_absolute_win_rate_gap_basis_points"
        ]
        * 2
        * len(ordered_game_keys)
    )
    reports: dict[int, Path] = {}
    for step in (64, 96, 128):
        wins = int(points_by_step[step])
        games = []
        for index, key in enumerate(ordered_game_keys):
            candidate_won = index < wins
            games.append(
                {
                    "pair_id": index // 2,
                    **key,
                    "candidate_won": candidate_won,
                    "search_won": candidate_won,
                    "terminated": True,
                    "truncated": False,
                    "error": None,
                    "engine_divergence": False,
                }
            )
        _scores, diagnostics = promotion.pair_scores_from_h2h_games(games)
        report = {
            "evaluation_binding": evaluation_binding,
            "planned_engine_identity": planned_engine_identity,
            "engine_identity": runtime_engine_identity,
            "candidate_checkpoint": checkpoints[step]["path"],
            "candidate_checkpoint_sha256": checkpoints[step]["sha256"],
            "baseline_checkpoint": str(baseline),
            "baseline_checkpoint_sha256": promotion._sha256(baseline),
            "effective_search_config": effective_search_config,
            "search_rng_contract": promotion.INTERNAL_H2H_SEARCH_RNG_CONTRACT,
            "games_played": len(games),
            "games_with_winner": len(games),
            "complete_pairs": len(games) // 2,
            "games_truncated": 0,
            "errors": [],
            "candidate_wins": wins,
            "baseline_wins": len(games) - wins,
            "pair_diagnostics": diagnostics,
            "games": games,
            "fleet_merge": {
                "schema_version": "a1-fleet-evaluation-pool-v1",
                "kind": "internal_h2h",
                "candidate": checkpoints[step],
                "champion": _checkpoint_ref(baseline),
                "effective_search_config_sha256": promotion._digest_value(
                    effective_search_config
                ),
            },
        }
        report_path = path.with_name(f"{path.stem}.step{step}.report.json")
        _write_json(report_path, report)
        reports[step] = report_path
    payload = {
        "schema_version": promotion.MATCHED_QUICK_SCREEN_SCHEMA,
        "eligible_optimizer_steps": [64, 96, 128],
        "checkpoints": [
            {"optimizer_step": step, "checkpoint": checkpoints[step]}
            for step in (64, 96, 128)
        ],
        "ordered_game_keys": ordered_game_keys,
        "ordered_game_keys_sha256": game_keys_sha256,
        "search_configuration": search_configuration,
        "search_configuration_sha256": search_sha256,
        "results": [
            {
                "optimizer_step": step,
                "checkpoint_sha256": checkpoints[step]["sha256"],
                "evaluation_report": _checkpoint_ref(reports[step]),
                "candidate_half_points": half_points[step],
                "games_played": len(ordered_game_keys),
                "ordered_game_keys_sha256": game_keys_sha256,
                "search_configuration_sha256": search_sha256,
            }
            for step in (64, 96, 128)
        ],
        "selection_rule": promotion.MATCHED_QUICK_SCREEN_SELECTION_RULE,
        "candidate_checkpoint_sha256": checkpoints[selected_step]["sha256"],
    }
    payload["screen_sha256"] = promotion._digest_value(payload)
    _write_json(path, payload)
    return payload


def _make_internal_pool_shard_complete(
    source: dict, *, candidate: Path, champion: Path
) -> None:
    source["candidate_checkpoint_sha256"] = promotion._sha256(candidate)
    source["baseline_checkpoint_sha256"] = promotion._sha256(champion)
    source["gate_config"] = "flywheel"
    for game in source["games"]:
        game.update(
            terminated=True,
            truncated=False,
            error=None,
            engine_divergence=False,
        )
    source["search_telemetry"] = {
        "by_role": {
            role: {
                "search_calls": 10,
                "non_forced_search_calls": 8,
                "search_elapsed_sec": 2.0,
                "simulations_used": 1_280,
                "wide_root_calls": 0,
                "wide_root_simulations_used": 0,
                "selected_vs_prior_disagreement_calls": 3,
                "wide_selected_vs_prior_disagreement_calls": 0,
            }
            for role in ("candidate", "baseline")
        }
    }


def _make_external_pool_shard_complete(source: dict, *, checkpoint: Path) -> None:
    source["candidate_checkpoint_sha256"] = promotion._sha256(checkpoint)
    source["base_seed"] = min(game["game_seed"] for game in source["games"])
    source.setdefault("referee_engine", "vendored_python_catanatron")
    source.setdefault("games_errored", 0)
    for game in source["games"]:
        game.update(
            search_won=game["candidate_won"],
            terminated=True,
            truncated=False,
            error=None,
            engine_divergence=False,
        )


def _write_one_dose_receipt(
    tmp_path: Path,
    *,
    contract_path: Path,
    contract: dict,
    candidate: Path,
    report: Path,
    command: list[str],
    execution_binding: dict,
) -> Path:
    optimizer = Path(str(candidate) + ".optimizer.pt")
    optimizer.write_bytes(b"fresh adam optimizer")
    outputs = {
        "checkpoint": str(candidate),
        "checkpoint_sha256": promotion._sha256(candidate),
        "optimizer_sidecar": str(optimizer),
        "optimizer_sidecar_sha256": promotion._sha256(optimizer),
        "report": str(report),
        "report_sha256": promotion._sha256(report),
        "execution_binding_sha256": promotion._digest_value(execution_binding),
        "steps_completed": 7,
        "corpus_row_count": 100,
        "training_row_count": 95,
        "validation_row_count": 5,
    }
    common = {
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "lock": str(contract_path),
        "lock_file_sha256": promotion._sha256(contract_path),
        "corpus": str(tmp_path / "audited-corpus"),
        "corpus_meta_file_sha256": "sha256:" + "1" * 64,
        "payload_inventory_sha256": "sha256:" + "2" * 64,
        "validation_manifest": str(tmp_path / "validation.json"),
        "validation_manifest_file_sha256": "sha256:" + "3" * 64,
        "producer_checkpoint_sha256": contract["checkpoints"][0]["sha256"],
        "learner_training_recipe_sha256": contract["science"][
            "learner_training_recipe_sha256"
        ],
        "command": command,
        "command_sha256": promotion._digest_value(command),
        "execution_binding": execution_binding,
        "world_size": 1,
        "gpu": 0,
        "gpu_name": "NVIDIA B200",
        "started_unix_ns": 10,
        "finished_unix_ns": 20,
        "returncode": 0,
        "outputs": outputs,
        "failure": None,
    }
    receipt_path = tmp_path / "one-dose.receipt.json"
    claim_path = tmp_path / "one-dose.claim.json"
    claim = {
        "schema_version": one_dose.CLAIM_SCHEMA,
        **common,
        "receipt_target": str(receipt_path),
    }
    claim["state_sha256"] = one_dose._value_sha256(claim)
    _write_json(claim_path, claim)
    receipt = {
        "schema_version": one_dose.RECEIPT_SCHEMA,
        **common,
        "claim": str(claim_path),
        "claim_state_sha256": claim["state_sha256"],
    }
    receipt["receipt_sha256"] = one_dose._value_sha256(receipt)
    _write_json(receipt_path, receipt)
    return receipt_path


def _rewrite_one_dose_receipt(receipt_path: Path, receipt: dict) -> None:
    claim_path = Path(receipt["claim"])
    claim = {
        key: value
        for key, value in receipt.items()
        if key not in {"claim", "claim_state_sha256", "receipt_sha256"}
    }
    claim["schema_version"] = one_dose.CLAIM_SCHEMA
    claim["receipt_target"] = str(receipt_path)
    claim["state_sha256"] = one_dose._value_sha256(claim)
    _write_json(claim_path, claim)
    receipt["claim_state_sha256"] = claim["state_sha256"]
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = one_dose._value_sha256(receipt)
    _write_json(receipt_path, receipt)


def _modernize_one_dose_receipt(
    fixture: dict,
    *,
    world_size: int = 8,
    with_intermediate: bool = False,
) -> tuple[dict, dict]:
    receipt_path = fixture["training_receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report_path = fixture["report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = fixture["candidate"]
    gpus = list(range(8)) if world_size == 8 else [0]
    local_batch = 512 if world_size == 8 else 4096
    topology = {
        "schema_version": "a1-one-dose-training-topology-v1",
        "name": (
            one_dose.B200_8GPU_DDP_TOPOLOGY
            if world_size == 8
            else one_dose.LEGACY_SINGLE_GPU_TOPOLOGY
        ),
        "world_size": world_size,
        "physical_gpus": gpus,
        "local_batch_size": local_batch,
        "grad_accum_steps": 1,
        "global_batch_size": 4096,
        "dose_preserving": True,
    }
    command = (
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            str(Path(one_dose.train_bc.__file__).resolve()),
            "--sealed-a1",
        ]
        if world_size == 8
        else [
            sys.executable,
            str(Path(one_dose.train_bc.__file__).resolve()),
            "--sealed-a1",
        ]
    )
    execution_binding = one_dose._execution_binding(
        command=command, environment=one_dose._child_environment(gpus)
    )
    canary_path = fixture["contract_path"].parent / "ddp-canary.json"
    canary_path.write_text("{}\n", encoding="utf-8")
    canary = (
        {
            "path": str(canary_path),
            "file_sha256": promotion._sha256(canary_path),
            "receipt_sha256": "sha256:" + "c" * 64,
            "global_draw_sha256": "sha256:" + "d" * 64,
            "rank_slice_sha256": ["sha256:" + f"{rank:x}" * 64 for rank in range(8)],
            "semantic_identity": {"world_size": 8},
            "semantic_identity_sha256": "sha256:" + "e" * 64,
        }
        if world_size == 8
        else None
    )
    lineage_dose = {"schema_version": "test-lineage-dose-v1", "current_sampled_rows": 524_288}
    outputs = receipt["outputs"]
    progress = Path(str(candidate) + ".training-progress.json")
    progress.write_text('{"progress_sha256":"sha256:test"}\n', encoding="utf-8")
    outputs.update(
        {
            "training_progress": str(progress),
            "training_progress_sha256": promotion._sha256(progress),
            "training_progress_payload_sha256": "sha256:" + "4" * 64,
            "sample_receipt_state_sha256": None,
            "sample_order_sha256": None,
            "row_set_sha256": None,
            "realized_sample_evidence_sha256": None,
            "unique_training_rows": None,
            "base_sampler_draw_events": 524_288,
            "sampler_draw_events": 524_288,
            "sampled_rows": 524_288,
            "lineage_dose": lineage_dose,
            "production_sampling_receipt_sha256": None,
            "validation_split_receipt_sha256": None,
        }
    )
    outputs["steps_completed"] = 128 if with_intermediate else 7
    intermediate_records = []
    if with_intermediate:
        import torch

        for step in (64, 96):
            checkpoint = candidate.with_name(f"candidate_step{step:04d}.pt")
            torch.save(
                {
                    "policy_type": "entity_graph",
                    "model": {"weight": torch.tensor([float(step)])},
                    "value_training": {
                        "optimizer_steps": step,
                        "completed_epochs": 0,
                        "trained_value_readouts": ["scalar"],
                        "intermediate_checkpoint": {
                            "schema_version": promotion.INTERMEDIATE_CHECKPOINT_SCHEMA,
                            "optimizer_step": step,
                            "same_training_trajectory": True,
                            "optimizer_sidecar_intentionally_omitted": True,
                        },
                    },
                },
                checkpoint,
            )
            intermediate_records.append(
                {
                    "schema_version": promotion.INTERMEDIATE_CHECKPOINT_SCHEMA,
                    "optimizer_step": step,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": promotion._sha256(checkpoint),
                    "size_bytes": checkpoint.stat().st_size,
                    "same_training_trajectory": True,
                    "optimizer_sidecar": None,
                }
            )
        outputs["intermediate_checkpoints"] = intermediate_records
        report["checkpoint_steps_requested"] = [64, 96]
        report["intermediate_checkpoints"] = intermediate_records
        report["steps_completed"] = 128
    input_binding = {
        "schema_version": one_dose.REPORT_INPUT_BINDING_SCHEMA,
        "contract_sha256": fixture["contract"]["contract_sha256"],
        "data": receipt["corpus"],
        "data_kind": "a1_memmap_v1",
        "data_fingerprint": "sha256:" + "0" * 64,
        "payload_inventory_sha256": receipt["payload_inventory_sha256"],
        "corpus_row_count": outputs["corpus_row_count"],
        "training_row_count": outputs["training_row_count"],
        "validation_row_count": outputs["validation_row_count"],
        "sealed_learner_recipe_sha256": receipt["learner_training_recipe_sha256"],
        "effective_learner_recipe_sha256": promotion._digest_value(
            {
                **fixture["contract"]["science"]["learner_training_recipe"],
                "world_size": world_size,
                "batch_size": local_batch,
                "grad_accum_steps": 1,
                "global_batch_size": 4096,
            }
        ),
        "training_topology": topology,
        "ddp_canary": canary,
        "aux_subgoal_preclaim_contract": None,
        "aux_pair_executor_authority_sha256": None,
        "p1_arm_executor_authority_sha256": None,
        "final_replication_executor_authority_sha256": None,
        "central_published_executor_authority": None,
        "validation_manifest": receipt["validation_manifest"],
        "validation_manifest_file_sha256": receipt[
            "validation_manifest_file_sha256"
        ],
        "selected_game_seed_set_sha256": "sha256:" + "5" * 64,
        "training_game_seed_set_sha256": "sha256:" + "6" * 64,
        "validation_game_seed_set_sha256": "sha256:" + "7" * 64,
    }
    input_binding["binding_sha256"] = promotion._digest_value(input_binding)
    outputs["input_binding_sha256"] = input_binding["binding_sha256"]
    outputs["execution_binding_sha256"] = promotion._digest_value(execution_binding)
    report.update(
        {
            one_dose.REPORT_EXECUTION_BINDING_FIELD: execution_binding,
            one_dose.REPORT_INPUT_BINDING_FIELD: input_binding,
            "world_size": world_size,
            "batch_size": local_batch,
            "grad_accum_steps": 1,
            "effective_global_batch_size": 4096,
            "a1_lineage_dose": lineage_dose,
        }
    )
    _write_json(report_path, report)
    outputs["report_sha256"] = promotion._sha256(report_path)
    receipt.update(
        {
            "command": command,
            "command_sha256": promotion._digest_value(command),
            "execution_binding": execution_binding,
            "input_binding": input_binding,
            "training_transaction_sha256": one_dose._training_transaction_sha256(
                command=command, input_binding=input_binding
            ),
            "trainer_authority": None,
            "lock_verifier_authority": None,
            "world_size": world_size,
            "gpu": 0,
            "gpus": gpus,
            "gpu_name": "NVIDIA B200",
            "gpu_names": ["NVIDIA B200"] * world_size,
            "training_topology": topology,
            "ddp_canary": canary,
            "production_sampling_receipt_sha256": None,
            "validation_split_receipt_sha256": None,
            "lineage_dose": lineage_dose,
            "outputs": outputs,
        }
    )
    _rewrite_one_dose_receipt(receipt_path, receipt)
    adjudication = json.loads(fixture["adjudication"].read_text(encoding="utf-8"))
    adjudication["candidate"]["training_report"]["sha256"] = promotion._sha256(
        report_path
    )
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)
    return receipt, {"canary": canary, "intermediate": intermediate_records}


def _write_evidence_envelope(
    path: Path,
    *,
    kind: str,
    contract: dict,
    candidate: Path,
    champion: Path,
    sources: list[tuple[str, Path]],
    verdict: str,
    result: dict,
) -> None:
    payload = {
        "schema_version": promotion.EVIDENCE_SCHEMA,
        "kind": kind,
        "passed": True,
        "verdict": verdict,
        "contract_sha256": contract["contract_sha256"],
        "candidate": _checkpoint_ref(candidate),
        "champion": _checkpoint_ref(champion),
        "sources": [
            {"role": role, "path": str(source), "sha256": promotion._sha256(source)}
            for role, source in sources
        ],
        "result": result,
    }
    payload["evidence_sha256"] = promotion._digest_value(payload)
    _write_json(path, payload)


def _fixture(
    tmp_path: Path,
    *,
    promotion_count: int = 0,
    n_full: int = 128,
    champion: Path | None = None,
    branch_parent: Path | None = None,
) -> dict:
    champion = champion or tmp_path / "champion.pt"
    candidate = tmp_path / "candidate.pt"
    champion.parent.mkdir(parents=True, exist_ok=True)
    champion.write_bytes(b"incumbent checkpoint")
    candidate.write_bytes(b"candidate checkpoint")
    if branch_parent is not None:
        branch_parent.parent.mkdir(parents=True, exist_ok=True)
        if not branch_parent.exists():
            branch_parent.write_bytes(b"older authenticated initializer")
    registry_path = tmp_path / "registry.json"
    registry = ChampionRegistry(registry_path)
    registry.set_role(
        "generator_champion",
        champion,
        expected_md5=promotion._md5(champion),
        version=4,
        reason="fixture",
    )
    registry.set_role(
        "public_champion",
        champion,
        expected_md5=promotion._md5(champion),
        version=4,
        reason="fixture",
    )
    for _ in range(promotion_count):
        registry.record_promotion()
    registry.save()
    pointer = tmp_path / "CURRENT_CHAMPION"
    pointer.write_text(str(champion.resolve()) + "\n", encoding="utf-8")
    contract_path = tmp_path / "contract.lock.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    producer = branch_parent or champion
    promotion_mode = "branch_challenge" if branch_parent is not None else "promotion_parent"
    contract = _contract(n_full=n_full, producer=producer)
    # The n196 negative fixture must still be constructible so execution can
    # prove the contract is rejected before any mutation/evidence processing.
    evidence_contract = (
        contract if n_full == 128 else _contract(n_full=128, producer=producer)
    )
    evidence_semantics = promotion._sealed_evaluation_semantics(evidence_contract)
    candidate_search_config = promotion._candidate_search_config(evidence_contract)
    if branch_parent is not None:
        seeded_incumbent_config = dict(evidence_semantics)
        seeded_incumbent_config["c_scale"] = candidate_search_config["c_scale"]
        seeded_incumbent_identity = promotion._agent_identity(
            _checkpoint_ref(champion), seeded_incumbent_config
        )
        registry.set_role(
            "generator_champion",
            champion,
            expected_md5=promotion._md5(champion),
            version=4,
            provenance={
                "a1_candidate_agent_identity_sha256": seeded_incumbent_identity[
                    "agent_identity_sha256"
                ],
                "a1_candidate_search_config": seeded_incumbent_config,
            },
            reason="fixture branch incumbent identity",
        )
        registry.save()
    champion_search_config = promotion._incumbent_search_config(
        evidence_contract,
        registry=registry,
        champion_path=champion.resolve(),
        champion_sha256=promotion._sha256(champion),
    )
    champion_identity = promotion._agent_identity(
        _checkpoint_ref(champion), champion_search_config
    )
    registry.set_role(
        "generator_champion",
        champion,
        expected_md5=promotion._md5(champion),
        version=4,
        provenance={
            "a1_candidate_agent_identity_sha256": champion_identity[
                "agent_identity_sha256"
            ],
            "a1_candidate_search_config": champion_search_config,
        },
        reason="fixture agent identity",
    )
    registry.save()
    evaluation_binding = {
        "schema_version": (
            "a1-evaluation-baseline-binding-v2"
            if branch_parent is not None
            else "a1-evaluation-baseline-binding-v1"
        ),
        "comparison_mode": promotion_mode,
        "promotion_eligible": True,
        "historical_comparison_reason": None,
        "candidate_parent": _checkpoint_ref(producer),
        "baseline": _checkpoint_ref(champion),
        "registry": _checkpoint_ref(registry_path),
        "authoritative_incumbent": {
            **_checkpoint_ref(champion),
            "version": 4,
            "agent_identity_sha256": champion_identity[
                "agent_identity_sha256"
            ],
            "search_config": champion_search_config,
        },
    }
    report_path = tmp_path / "report.json"
    command = ["/usr/bin/python3", "tools/train_bc.py", "--sealed-a1"]
    execution_binding = one_dose._execution_binding(
        command=command, environment=one_dose._child_environment(0)
    )
    _write_json(
        report_path,
        {
            "a1_contract_sha256": contract["contract_sha256"],
            "a1_learner_training_recipe_sha256": contract["science"][
                "learner_training_recipe_sha256"
            ],
            "a1_bound_learner_training_recipe": contract["science"][
                "learner_training_recipe"
            ],
            "arch": "entity_graph",
            "mask_hidden_info": True,
            "symmetry_augment": False,
            "track": "2p_no_trade",
            "vps_to_win": 10,
            "steps_completed": 7,
            "epochs": 1,
            "max_steps": 0,
            "checkpoint": str(candidate),
            "init_checkpoint": str(producer),
            "init_checkpoint_sha256": contract["checkpoints"][0]["sha256"],
            one_dose.REPORT_EXECUTION_BINDING_FIELD: execution_binding,
        },
    )
    training_receipt = _write_one_dose_receipt(
        tmp_path,
        contract_path=contract_path,
        contract=contract,
        candidate=candidate,
        report=report_path,
        command=command,
        execution_binding=execution_binding,
    )
    calibration_sources = []
    for role, checkpoint, rmse in (
        ("candidate_calibration", candidate, 0.20),
        ("champion_calibration", champion, 0.21),
    ):
        source = tmp_path / f"{role}.json"
        _write_json(
            source,
            {
                "schema_version": "phase-sliced-value-calibration-v2",
                "checkpoint": str(checkpoint),
                "shard_dir": str(tmp_path / "shared_validation_corpus"),
                "value_readout": "scalar",
                "readout_provenance": {
                    "requested_readout": "scalar",
                    "trained_value_readouts": ["scalar"],
                    "optimizer_steps": 7,
                    "completed_epochs": 1,
                },
                "row_selection": {
                    "mode": "validation_seed_manifest",
                    "held_out_filter_applied": True,
                    "validation_fraction": 0.05,
                    "validation_seed": 17,
                    "validation_game_seed_ranges": [],
                    "seed_manifest_sha256": "sha256:" + "9" * 64,
                    "configured_game_seed_count": 256,
                    "configured_game_seed_set_sha256": "sha256:" + "a" * 64,
                    "observed_game_seed_count": 256,
                    "observed_game_seed_set_sha256": "sha256:" + "a" * 64,
                    "observed_row_count": 4096,
                },
                "deployed_readout_diagnostics": {
                    "diagnostic_only": True,
                    "changes_operator_default": False,
                    "value_scale": 1.0,
                    "configured_value_squash": "tanh",
                    "configured_effective_transform": "scalar_tanh",
                    "categorical_bypasses_scalar_tanh": False,
                    "views": {
                        "raw_training_readout": {
                            "global": {"n": 4096, "value_rmse": rmse}
                        },
                        "scalar_tanh": {
                            "global": {"n": 4096, "value_rmse": rmse}
                        },
                        "scalar_clip": {
                            "global": {"n": 4096, "value_rmse": rmse}
                        },
                    },
                },
                "global": {"n": 4096, "value_rmse": rmse},
            },
        )
        calibration_sources.append((role, source))
    internal_games = []
    for pair in range(200):
        game_seed = 7_000_000 + pair
        for orientation in ("candidate_red", "candidate_blue"):
            candidate_color = "RED" if orientation == "candidate_red" else "BLUE"
            baseline_color = "BLUE" if candidate_color == "RED" else "RED"
            internal_games.append(
                {
                    "pair_id": pair,
                    "game_seed": game_seed,
                    "orientation": orientation,
                    "search_won": True,
                    "candidate_won": True,
                    "search_seeds_by_role": {
                        "candidate": promotion._internal_h2h_search_seed(  # noqa: SLF001
                            game_seed=game_seed, seat_color=candidate_color
                        ),
                        "baseline": promotion._internal_h2h_search_seed(  # noqa: SLF001
                            game_seed=game_seed, seat_color=baseline_color
                        ),
                    },
                }
            )
    pair_scores, pair_diagnostics = promotion.pair_scores_from_h2h_games(internal_games)
    pentanomial = promotion.evaluate_pentanomial_sprt(
        pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    superiority_pentanomial = promotion.evaluate_pentanomial_sprt(
        pair_scores, elo0=0.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    assert pentanomial["decision"] == "H1"
    assert superiority_pentanomial["decision"] == "H1"
    typed_config = {
        "pipeline": "eval",
        "schema_version": 6,
        "fields": {
            **evidence_semantics,
            "mode": "cross_net",
            "map_kind": "BASE",
            "candidate": str(candidate),
            "baseline": str(champion),
            "public_observation": True,
            "information_set_search": True,
            "determinization_particles": 4,
            "determinization_min_simulations": 32,
            "candidate_n_full": 128,
            "baseline_n_full": 128,
            "candidate_c_scale": candidate_search_config["c_scale"],
            "baseline_c_scale": champion_search_config["c_scale"],
            "candidate_n_full_wide": None,
            "baseline_n_full_wide": None,
            "candidate_n_full_wide_threshold": None,
            "baseline_n_full_wide_threshold": None,
            "candidate_value_readout": "scalar",
            "baseline_value_readout": "scalar",
        },
    }
    config_digest = hashlib.sha256(promotion._canonical_bytes(typed_config)).hexdigest()
    internal_source = tmp_path / "internal_h2h.raw.json"
    _write_json(
        internal_source,
        {
            "evaluation_binding": evaluation_binding,
            "planned_engine_identity": promotion._canonical_internal_h2h_engine_identity(),  # noqa: SLF001
            "engine_identity": promotion._canonical_internal_h2h_engine_identity(),  # noqa: SLF001
            "candidate_checkpoint": str(candidate),
            "baseline_checkpoint": str(champion),
            "typed_config": typed_config,
            "config_hash": "sha256:" + config_digest[:16],
            "full_config_hash": "sha256:" + config_digest,
            "search_rng_contract": promotion.INTERNAL_H2H_SEARCH_RNG_CONTRACT,
            "base_seed": 7_000_000,
            "pairs_requested": 200,
            "candidate_value_readout": "scalar",
            "baseline_value_readout": "scalar",
            "public_observation": True,
            "information_set_search": True,
            "determinization_particles": 4,
            "determinization_min_simulations": 32,
            "search_budgets_by_role": {
                role: {
                    "n_full": 128,
                    "n_full_wide": None,
                    "n_full_wide_threshold": None,
                }
                for role in ("candidate", "baseline")
            },
            "complete_pairs": 200,
            "games_played": 400,
            "games_with_winner": 400,
            "games_truncated": 0,
            "candidate_wins": 400,
            "baseline_wins": 0,
            "candidate_win_rate": 1.0,
            "errors": [],
            "games": internal_games,
            "pair_diagnostics": pair_diagnostics,
            "pentanomial_sprt": pentanomial,
            "verdict": "H1",
            "superiority_pentanomial_sprt": superiority_pentanomial,
            "superiority_verdict": "H1",
        },
    )
    branch_internal_sources: list[tuple[str, Path]] | None = None
    if branch_parent is not None:
        second_internal_source = tmp_path / "internal_h2h.cohort2.raw.json"
        second_payload = json.loads(internal_source.read_text(encoding="utf-8"))
        second_payload["base_seed"] += 10_000
        for game in second_payload["games"]:
            game["game_seed"] += 10_000
            candidate_color = (
                "RED" if game["orientation"] == "candidate_red" else "BLUE"
            )
            baseline_color = "BLUE" if candidate_color == "RED" else "RED"
            game["search_seeds_by_role"] = {
                "candidate": promotion._internal_h2h_search_seed(  # noqa: SLF001
                    game_seed=game["game_seed"], seat_color=candidate_color
                ),
                "baseline": promotion._internal_h2h_search_seed(  # noqa: SLF001
                    game_seed=game["game_seed"], seat_color=baseline_color
                ),
            }
        _write_json(second_internal_source, second_payload)
        branch_internal_sources = [
            ("internal_h2h_cohort_1", internal_source),
            ("internal_h2h_cohort_2", second_internal_source),
        ]

    external_sources = []
    planned_engine_identity = {
        "schema_version": "a1-neutral-engine-identity-v1",
        "repo_commit": "a" * 40,
        "native_wheel_sha256": "sha256:" + "b" * 64,
        "python_referee_sha256": "sha256:" + "c" * 64,
    }
    runtime_engine_identity = {
        **planned_engine_identity,
        "native_runtime_sha256": "sha256:" + "d" * 64,
    }
    for role, checkpoint, win_rate, external_search_config in (
        ("candidate_panel", candidate, 0.55, candidate_search_config),
        ("champion_panel", champion, 0.54, champion_search_config),
    ):
        external_games = [
            {
                "pair_id": pair,
                "game_seed": 8_100_000 + pair,
                "orientation": orientation,
                "candidate_won": game_index < int(win_rate * 1_000),
            }
            for game_index, (pair, orientation) in enumerate(
                (pair, orientation)
                for pair in range(500)
                for orientation in ("candidate_first", "candidate_second")
            )
        ]
        normalized_external_games = [
            {**game, "search_won": game["candidate_won"]} for game in external_games
        ]
        external_pair_scores, external_pair_diagnostics = (
            promotion.pair_scores_from_h2h_games(normalized_external_games)
        )
        external_pentanomial = promotion.evaluate_pentanomial_sprt(
            external_pair_scores,
            elo0=-10.0,
            elo1=15.0,
            alpha=0.05,
            beta=0.05,
        )
        candidate_wins = sum(bool(game["candidate_won"]) for game in external_games)
        source = tmp_path / f"{role}.raw.json"
        _write_json(
            source,
            {
                "evaluation_binding": evaluation_binding,
                "planned_engine_identity": planned_engine_identity,
                "engine_identity": runtime_engine_identity,
                "stratum": "neutral-harness",
                "harness": "catanatron_native_engine",
                "baseline_bot": "catanatron_value",
                "mode": "search",
                "public_observation": True,
                "information_set_search": True,
                "determinization_particles": 4,
                "determinization_min_simulations": 32,
                "candidate_value_readout": "scalar",
                "trained_value_readouts": ["scalar"],
                "n_full": 128,
                "n_full_wide": None,
                "map_kind": "TOURNAMENT",
                "search_config": external_search_config,
                "gate_config": "flywheel",
                "pairs_requested": 500,
                "games_requested": 1000,
                "games_played": 1000,
                "games_with_winner": 1000,
                "games_truncated": 0,
                "games": external_games,
                "candidate_checkpoint": str(checkpoint),
                "candidate_checkpoint_md5": promotion._md5(checkpoint),
                "complete_pairs": 500,
                "candidate_win_rate": win_rate,
                "candidate_wins": candidate_wins,
                "baseline_wins": 1_000 - candidate_wins,
                "pair_diagnostics": external_pair_diagnostics,
                "pentanomial_sprt": external_pentanomial,
                "verdict": external_pentanomial["decision"],
                "errors": [],
                "worker_errors": [],
                "games_engine_divergence": 0,
            },
        )
        external_sources.append((role, source))

    high_regret_scope = tmp_path / "high-regret-worker"
    high_regret_scope.mkdir()
    high_regret_source_shard = high_regret_scope / "source-shard.npz"
    np.savez(
        high_regret_source_shard,
        game_seed=np.arange(7_000_000, 7_000_240, dtype=np.int64),
        decision_index=np.zeros(240, dtype=np.int32),
        action_taken=np.arange(240, dtype=np.int32),
    )
    high_regret_source_manifest = tmp_path / "high_regret.source.npz"
    validation_seeds = np.arange(7_000_000, 7_000_240, dtype=np.int64)
    validation_seed_manifest = tmp_path / "high_regret.validation-seeds.json"
    validation_seed_digest = "sha256:" + hashlib.sha256(
        validation_seeds.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    _write_json(
        validation_seed_manifest,
        {
            "schema_version": "train-validation-game-seeds-v1",
            "game_seeds": validation_seeds.tolist(),
            "validation_game_seed_count": len(validation_seeds),
            "validation_game_seed_set_sha256": validation_seed_digest,
        },
    )
    validation_binding = {
        "path": str(validation_seed_manifest.resolve()),
        "sha256": promotion._sha256(validation_seed_manifest),
        "schema_version": "train-validation-game-seeds-v1",
        "game_seed_count": len(validation_seeds),
        "game_seed_set_sha256": validation_seed_digest,
    }
    np.savez(
        high_regret_source_manifest,
        shard_paths=np.asarray([str(high_regret_source_shard)]),
        shard_id=np.zeros(240, dtype=np.int32),
        row_index=np.arange(240, dtype=np.int32),
        game_seed=validation_seeds,
        decision_index=np.zeros(240, dtype=np.int32),
        held_out_only=np.asarray(True),
        validation_seed_manifest_path=np.asarray(
            str(validation_seed_manifest.resolve())
        ),
        validation_seed_manifest_sha256=np.asarray(validation_binding["sha256"]),
        validation_seed_manifest_schema_version=np.asarray(
            validation_binding["schema_version"]
        ),
        validation_game_seed_count=np.asarray(
            validation_binding["game_seed_count"], dtype=np.int64
        ),
        validation_game_seed_set_sha256=np.asarray(validation_seed_digest),
    )
    scope_digest, scope_count = scope_inventory_sha256(high_regret_scope)
    high_regret_suite = tmp_path / "high_regret.suite.json"
    high_regret_suite_payload = {
        "schema_version": promotion.HIGH_REGRET_SUITE_SCHEMA,
        "suite": "held_out_high_regret",
        "held_out": True,
        "source_manifest": _checkpoint_ref(high_regret_source_manifest),
        "validation_seed_manifest": validation_binding,
        "selection": {
            "algorithm": "trainer-validation-stratified-regret-unique-game-v3",
            "selection_scope": "full_authenticated_training_validation_manifest",
            "holdout_fraction": 1.0,
            "holdout_seed": 17,
            "eligible_unique_states": 240,
            "eligible_unique_games": 240,
            "replay_complete_unique_games": 240,
            "selected_unique_games": 240,
            "selected_pairs": 240,
            "stratum_min_pairs": 24,
            "selected_by_stratum": {
                "phase:opening": 24,
                "phase:robber_dev": 24,
                "phase:chance": 24,
                "phase:build_trade": 24,
                "41+": 24,
            },
            "replay_preflight": {
                "contract": REPLAY_CONTRACT,
                "candidate_states": 240,
                "replay_complete_states": 240,
                "rejected_bad_source": 0,
                "rejected_noncontiguous": 0,
            },
        },
        "states": [
            {
                "pair_id": pair,
                "shard_path": str(high_regret_source_shard),
                "shard_id": 0,
                "row_index": pair,
                "game_seed": 7_000_000 + pair,
                "decision_index": 0,
                "phase": (
                    "BUILD_INITIAL_SETTLEMENT",
                    "MOVE_ROBBER",
                    "ROLL",
                    "BUILD_ROAD",
                )[pair % 4],
                "legal_count": 54 if pair < 24 else 12,
                "regret_score": 1.0,
                "replay_source": {
                    "contract": REPLAY_CONTRACT,
                    "scope": str(high_regret_scope),
                    "scope_inventory_sha256": scope_digest,
                    "scope_shard_count": scope_count,
                },
            }
            for pair in range(240)
        ],
    }
    high_regret_suite_payload["suite_sha256"] = promotion._digest_value(
        high_regret_suite_payload
    )
    _write_json(high_regret_suite, high_regret_suite_payload)
    high_regret_games = [
        {
            "pair_id": pair,
            "orientation": orientation,
            "candidate_won": True,
            "truncated": False,
            "archived_game_seed": 7_000_000 + pair,
            "archived_decision_index": 0,
            "buckets": ["phase:BUILD", "close"],
        }
        for pair in range(240)
        for orientation in ("candidate_first", "candidate_second")
    ]
    normalized_high_regret_games = [
        {**game, "search_won": game["candidate_won"]} for game in high_regret_games
    ]
    high_pair_scores, high_pair_diagnostics = promotion.pair_scores_from_h2h_games(
        normalized_high_regret_games
    )
    high_pentanomial = promotion.evaluate_pentanomial_sprt(
        high_pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    assert high_pentanomial["decision"] == "H1"
    high_regret_report = tmp_path / "high_regret.report.json"
    _write_json(
        high_regret_report,
        {
            "schema_version": promotion.HIGH_REGRET_REPORT_SCHEMA,
            "suite": "held_out_high_regret",
            "held_out": True,
            "suite_manifest": _checkpoint_ref(high_regret_suite),
            "candidate": _checkpoint_ref(candidate),
            "champion": _checkpoint_ref(champion),
            "evaluation_config": {
                **evidence_semantics,
                "evaluator_rust_featurize": True,
                "native_mcts_hot_loop": True,
                "c_scale": candidate_search_config["c_scale"],
                "candidate_c_scale": candidate_search_config["c_scale"],
                "baseline_c_scale": champion_search_config["c_scale"],
                "candidate_n_full": 128,
                "baseline_n_full": 128,
                "candidate_n_full_wide": None,
                "baseline_n_full_wide": None,
                "candidate_n_full_wide_threshold": None,
                "baseline_n_full_wide_threshold": None,
                "candidate_value_readout": "scalar",
                "baseline_value_readout": "scalar",
            },
            "planned_engine_identity": {
                "schema_version": promotion.HIGH_REGRET_ENGINE_IDENTITY_SCHEMA,
                "repo_commit": "a" * 40,
                "native_wheel_sha256": "sha256:" + "b" * 64,
                "evaluator_sha256": "sha256:" + "c" * 64,
                "replay_sha256": "sha256:" + "d" * 64,
            },
            "engine_identity": {
                "schema_version": promotion.HIGH_REGRET_ENGINE_IDENTITY_SCHEMA,
                "repo_commit": "a" * 40,
                "native_wheel_sha256": "sha256:" + "b" * 64,
                "evaluator_sha256": "sha256:" + "c" * 64,
                "replay_sha256": "sha256:" + "d" * 64,
                "native_runtime_sha256": "sha256:" + "e" * 64,
            },
            "archived_state_reconstruction": {
                "schema_version": promotion.ARCHIVED_STATE_RECONSTRUCTION_SCHEMA,
                "constructor": "catanatron_rs.Game.simple",
                "map_kind": "BASE",
                "action_prefix": "[0,target_decision)",
                "chance_stream": "random.Random(game_seed ^ 0xA17E)",
                "replay_contract": REPLAY_CONTRACT,
            },
            "errors": [],
            "games": high_regret_games,
            "pentanomial_sprt": high_pentanomial,
            "pair_diagnostics": high_pair_diagnostics,
        },
    )
    high_regret_source = tmp_path / "high_regret.raw.json"
    _write_json(
        high_regret_source,
        {
            "schema_version": promotion.HIGH_REGRET_SCHEMA,
            "suite": "held_out_high_regret",
            "held_out": True,
            "candidate": _checkpoint_ref(candidate),
            "champion": _checkpoint_ref(champion),
            "passed": True,
            "verdict": "H1",
            "complete_pairs": 240,
            "errors": [],
            "report": _checkpoint_ref(high_regret_report),
            "suite_manifest": _checkpoint_ref(high_regret_suite),
            "pentanomial_sprt": high_pentanomial,
            "pair_diagnostics": high_pair_diagnostics,
        },
    )
    phase_buckets = ("opening", "robber_dev", "chance", "build_trade")

    def fixture_buckets(pair: int) -> list[str]:
        phase = phase_buckets[pair % len(phase_buckets)]
        labels = [f"phase:{phase}", "blowout" if pair % 2 == 0 else "close"]
        if phase == "opening":
            labels.append("opening")
        if pair < 50:
            labels.append("41+")
        return sorted(labels)

    bucket_games = [
        {
            "pair_id": pair,
            "orientation": orientation,
            "candidate_won": True,
            "buckets": fixture_buckets(pair),
        }
        for pair in range(100)
        for orientation in ("candidate_first", "candidate_second")
    ]
    bucket_report = tmp_path / "bucket_veto.report.json"
    _write_json(
        bucket_report,
        {
            "schema_version": promotion.BUCKET_GAME_REPORT_SCHEMA,
            "candidate": _checkpoint_ref(candidate),
            "champion": _checkpoint_ref(champion),
            "errors": [],
            "games": bucket_games,
        },
    )
    bucket_source = tmp_path / "bucket_veto.raw.json"
    _write_json(
        bucket_source,
        {
            "schema_version": promotion.BUCKET_VETO_SCHEMA,
            "candidate": _checkpoint_ref(candidate),
            "champion": _checkpoint_ref(champion),
            "veto": False,
            "veto_buckets": [],
            "per_bucket": {
                "41+": {"status": "pass", "n": 100, "winrate": 1.0},
                "blowout": {"status": "pass", "n": 100, "winrate": 1.0},
                "close": {"status": "pass", "n": 100, "winrate": 1.0},
                "opening": {"status": "pass", "n": 50, "winrate": 1.0},
                "phase:build_trade": {
                    "status": "pass",
                    "n": 50,
                    "winrate": 1.0,
                },
                "phase:chance": {"status": "pass", "n": 50, "winrate": 1.0},
                "phase:opening": {"status": "pass", "n": 50, "winrate": 1.0},
                "phase:robber_dev": {
                    "status": "pass",
                    "n": 50,
                    "winrate": 1.0,
                },
            },
            "report": _checkpoint_ref(bucket_report),
        },
    )
    evidence_specs = {
        "mechanism_calibration": (
            calibration_sources,
            "pass",
            {"value_readout": "scalar", "max_rmse_regression": 0.02},
        ),
        "internal_h2h": (
            (
                [("internal_h2h", internal_source)]
                if branch_internal_sources is None
                else branch_internal_sources
            ),
            "H1",
            (
                dict(promotion.INTERNAL_STRENGTH_RESULT)
                if branch_parent is None
                else {
                    **promotion.INTERNAL_STRENGTH_RESULT,
                    "required_fresh_cohorts": 2,
                    "strict_superiority": True,
                }
            ),
        ),
        "external_panel": (
            external_sources,
            "pass",
            {"max_win_rate_regression": 0.02},
        ),
        "high_regret": ([("high_regret", high_regret_source)], "pass", {}),
        "bucket_veto": ([("bucket_veto", bucket_source)], "pass", {}),
    }
    evidence = []
    for kind in sorted(promotion.REQUIRED_EVIDENCE_KINDS):
        sources, verdict, result = evidence_specs[kind]
        evidence_path = tmp_path / f"{kind}.json"
        _write_evidence_envelope(
            evidence_path,
            kind=kind,
            contract=contract,
            candidate=candidate,
            champion=champion,
            sources=sources,
            verdict=verdict,
            result=result,
        )
        evidence.append(
            {
                "kind": kind,
                "path": str(evidence_path),
                "sha256": promotion._sha256(evidence_path),
            }
        )
    next_count = promotion_count + 1
    nth_required = next_count % 3 == 0
    nth_confirmation = None
    if nth_required:
        nth_source = tmp_path / "nth_confirmation_n64.raw.json"
        nth_payload = json.loads(internal_source.read_text(encoding="utf-8"))
        fields = nth_payload["typed_config"]["fields"]
        for key in ("n_full", "n_fast", "candidate_n_full", "baseline_n_full"):
            fields[key] = 64
        digest = hashlib.sha256(
            promotion._canonical_bytes(nth_payload["typed_config"])
        ).hexdigest()
        nth_payload["config_hash"] = "sha256:" + digest[:16]
        nth_payload["full_config_hash"] = "sha256:" + digest
        for budget in nth_payload["search_budgets_by_role"].values():
            budget["n_full"] = 64
        _write_json(nth_source, nth_payload)
        nth_confirmation = {
            "path": str(nth_source),
            "sha256": promotion._sha256(nth_source),
        }
    adjudication = {
        "schema_version": (
            promotion.BRANCH_CHALLENGE_ADJUDICATION_SCHEMA
            if branch_parent is not None
            else promotion.ADJUDICATION_SCHEMA
        ),
        "passed": True,
        "decision": "promote",
        "contract_sha256": contract["contract_sha256"],
        "candidate": {
            "path": str(candidate),
            "sha256": promotion._sha256(candidate),
            "version": 5,
            "agent_identity": promotion._agent_identity(
                _checkpoint_ref(candidate), candidate_search_config
            ),
            "training_report": {
                "path": str(report_path),
                "sha256": promotion._sha256(report_path),
            },
        },
        "champion": {
            "path": str(champion),
            "sha256": promotion._sha256(champion),
            "version": 4,
            "agent_identity": promotion._agent_identity(
                _checkpoint_ref(champion), champion_search_config
            ),
        },
        "checks": {name: True for name in promotion.REQUIRED_CHECKS},
        "nth_confirmation_required": nth_required,
        "nth_confirmation": nth_confirmation,
        "evidence": evidence,
    }
    if branch_parent is not None:
        adjudication["promotion_mode"] = "branch_challenge"
        adjudication["candidate_lineage"] = {
            "schema_version": promotion.BRANCH_CHALLENGE_LINEAGE_SCHEMA,
            "initializer": _checkpoint_ref(branch_parent),
            "displaced_incumbent": {
                **_checkpoint_ref(champion),
                "version": 4,
                "agent_identity_sha256": champion_identity[
                    "agent_identity_sha256"
                ],
            },
        }
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    adjudication_path = tmp_path / "adjudication.json"
    _write_json(adjudication_path, adjudication)
    diagnostic_source = tmp_path / "prior-diagnostic-cohort.json"
    _write_json(
        diagnostic_source,
        {
            "kind": "arm_selection",
            "candidate_checkpoint": str(candidate),
            "candidate_checkpoint_sha256": promotion._sha256(candidate),
            "games": [
                {"game_seed": seed} for seed in range(9_000_000, 9_000_200)
            ],
        },
    )
    cohort_exclusions = {
        "schema_version": promotion.COHORT_EXCLUSIONS_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "candidate_sha256": promotion._sha256(candidate),
        "cohorts": [
            {
                "label": "p1-arm-selection",
                "kind": "internal_h2h",
                "source": _checkpoint_ref(diagnostic_source),
                "seed_intervals": [
                    {"base_seed": 9_000_000, "end_seed": 9_000_200}
                ],
            }
        ],
    }
    cohort_exclusions["manifest_sha256"] = promotion._digest_value(
        cohort_exclusions
    )
    cohort_exclusions_path = tmp_path / "promotion-cohort-exclusions.json"
    _write_json(cohort_exclusions_path, cohort_exclusions)
    return {
        "champion": champion,
        "candidate": candidate,
        "registry": registry_path,
        "pointer": pointer,
        "contract_path": contract_path,
        "contract": contract,
        "evaluation_binding": evaluation_binding,
        "adjudication": adjudication_path,
        "report": report_path,
        "training_receipt": training_receipt,
        "branch_parent": branch_parent,
        "cohort_exclusions": cohort_exclusions_path,
        "receipt": tmp_path / "promotion.receipt.json",
        "lock": registry_path.with_suffix(registry_path.suffix + ".a1.lock"),
    }


def test_promoted_candidate_operator_becomes_next_cycle_incumbent_operator(
    tmp_path: Path,
) -> None:
    """A .10 promotion must never be silently reclassified as champion@.03."""
    old = tmp_path / "gen3.pt"
    promoted = tmp_path / "f7.pt"
    old.write_bytes(b"gen3")
    promoted.write_bytes(b"f7")
    first_contract = _contract(producer=old, c_scale=0.03)
    promoted_config = promotion._candidate_search_config(first_contract)
    assert promoted_config["c_scale"] == 0.10
    promoted_identity = promotion._agent_identity(
        _checkpoint_ref(promoted), promoted_config
    )

    registry = ChampionRegistry(tmp_path / "registry.json")
    registry.set_role(
        "generator_champion",
        promoted,
        expected_md5=promotion._md5(promoted),
        version=5,
        provenance={
            "a1_candidate_agent_identity_sha256": promoted_identity[
                "agent_identity_sha256"
            ],
            "a1_candidate_search_config": promoted_config,
        },
        reason="simulated committed first promotion",
    )
    registry.record_promotion("generator_champion")
    registry.save()

    # The next candidate may tune its operator; that must not reclassify the
    # incumbent, whose exact promoted .10 identity remains authoritative.
    next_contract = _contract(producer=promoted, c_scale=0.15)
    next_contract["contract_id"] = "a1-next-cycle"
    next_contract["contract_sha256"] = "sha256:" + "2" * 64
    incumbent_config = promotion._incumbent_search_config(
        next_contract,
        registry=ChampionRegistry.load(registry.path),
        champion_path=promoted.resolve(),
        champion_sha256=promotion._sha256(promoted),
    )
    assert incumbent_config["c_scale"] == 0.10
    assert promotion._candidate_search_config(next_contract)["c_scale"] == 0.15

    checkpoint_ref = _checkpoint_ref(promoted)
    silently_downgraded = dict(incumbent_config, c_scale=0.03)
    identity = promotion._agent_identity(checkpoint_ref, silently_downgraded)
    with pytest.raises(
        promotion.PromotionError, match="search_config sealed A1 semantic drift"
    ):
        promotion._verify_agent_identity(
            identity,
            expected_search_config=incumbent_config,
            checkpoint_path=promoted.resolve(),
            checkpoint_sha256=checkpoint_ref["sha256"],
            base=tmp_path,
            where="next-cycle champion.agent_identity",
        )


def test_next_cycle_incumbent_without_bound_operator_identity_fails_closed(
    tmp_path: Path,
) -> None:
    """Only the exact historical bootstrap may lack incumbent provenance."""
    promoted = tmp_path / "f7.pt"
    promoted.write_bytes(b"f7")
    registry = ChampionRegistry(tmp_path / "registry.json")
    registry.set_role(
        "generator_champion",
        promoted,
        expected_md5=promotion._md5(promoted),
        version=5,
        provenance={},
        reason="simulated malformed promotion",
    )
    registry.save()

    next_contract = _contract(producer=promoted, c_scale=0.10)
    next_contract["contract_id"] = "a1-next-cycle"
    next_contract["contract_sha256"] = "sha256:" + "2" * 64
    with pytest.raises(
        promotion.PromotionError, match="no bound agent search identity"
    ):
        promotion._incumbent_search_config(
            next_contract,
            registry=ChampionRegistry.load(registry.path),
            champion_path=promoted.resolve(),
            champion_sha256=promotion._sha256(promoted),
        )


def _verify(fixture: dict):
    def verify(path: Path, *, require_all_job_claims: bool = False):
        assert path == fixture["contract_path"]
        assert require_all_job_claims is True
        return fixture["contract"]

    return verify


def _execute(fixture: dict, *, go: bool):
    return promotion.execute_promotion(
        registry_path=fixture["registry"],
        current_pointer=fixture["pointer"],
        contract_lock=fixture["contract_path"],
        adjudication_path=fixture["adjudication"],
        training_receipt=fixture["training_receipt"],
        cohort_exclusions=fixture["cohort_exclusions"],
        receipt_path=fixture["receipt"],
        reason="A1 typed promotion",
        lock_path=fixture["lock"],
        go=go,
        verify_lock_fn=_verify(fixture),
    )


def _mutate_training_receipt(fixture: dict, mutate) -> dict:
    path = fixture["training_receipt"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("receipt_sha256")
    mutate(payload)
    payload["receipt_sha256"] = promotion._digest_value(payload)
    _write_json(path, payload)
    return payload


def _convert_training_receipt_to_sealed_retry(
    fixture: dict, *, repair_kind: str = one_dose.RETRY_REPAIR_KIND
) -> None:
    receipt_path = fixture["training_receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    claim_path = Path(receipt["claim"])
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    identity_evidence = {
        "schema_version": one_dose.RETRY_IDENTITY_SCHEMA,
        "repair_kind": repair_kind,
        "parent_contract_sha256": fixture["contract"]["contract_sha256"],
        "parent": {
            "claim": "/sealed/failed-r1.claim.json",
            "claim_file_sha256": "sha256:" + "4" * 64,
            "claim_state_sha256": "sha256:" + "5" * 64,
            "receipt": "/sealed/failed-r1.receipt.json",
            "receipt_file_sha256": "sha256:" + "6" * 64,
            "receipt_sha256": "sha256:" + "7" * 64,
            "command_sha256": "sha256:" + "8" * 64,
            "returncode": 1,
            "failure": "ExecutorError: train_bc exited nonzero: 1",
        },
    }
    identity_sha = promotion._digest_value(identity_evidence)
    retry_contract = {
        "schema_version": one_dose.RETRY_CONTRACT_SCHEMA,
        "retry_identity": identity_evidence,
        "retry_identity_sha256": identity_sha,
        "parent": {
            **identity_evidence["parent"],
            "pre_optimizer_proof": {
                "kind": "replayed_init_checkpoint_architecture_preflight",
                "mismatches": ["graph_layers checkpoint=6 cli=4"],
                "optimizer_steps": 0,
                "outputs": None,
            },
        },
        "preserved_bindings": {},
        "retry": {},
    }
    retry_contract["retry_contract_sha256"] = promotion._digest_value(retry_contract)
    retry_contract_path = receipt_path.with_name("learner-retry.contract.json")
    _write_json(retry_contract_path, retry_contract)
    reference = {
        "path": str(retry_contract_path),
        "file_sha256": promotion._sha256(retry_contract_path),
        "retry_contract_sha256": retry_contract["retry_contract_sha256"],
    }

    claim.pop("state_sha256")
    claim["schema_version"] = one_dose.RETRY_CLAIM_SCHEMA
    claim["claim_identity_sha256"] = identity_sha
    claim["retry_contract"] = reference
    claim["state_sha256"] = one_dose._value_sha256(claim)
    _write_json(claim_path, claim)

    receipt.pop("receipt_sha256")
    receipt["schema_version"] = one_dose.RETRY_RECEIPT_SCHEMA
    receipt["claim_identity_sha256"] = identity_sha
    receipt["retry_contract"] = reference
    receipt["claim_state_sha256"] = claim["state_sha256"]
    receipt["receipt_sha256"] = one_dose._value_sha256(receipt)
    _write_json(receipt_path, receipt)


def test_dry_run_binds_v3_one_dose_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    plan = _execute(fixture, go=False)

    assert plan["schema_version"] == promotion.RECEIPT_SCHEMA
    assert plan["training_receipt"] == {
        "path": str(fixture["training_receipt"]),
        "sha256": promotion._sha256(fixture["training_receipt"]),
        "receipt_sha256": json.loads(
            fixture["training_receipt"].read_text(encoding="utf-8")
        )["receipt_sha256"],
        "claim": str(tmp_path / "one-dose.claim.json"),
        "claim_state_sha256": json.loads(
            fixture["training_receipt"].read_text(encoding="utf-8")
        )["claim_state_sha256"],
        "execution_binding_sha256": json.loads(
            fixture["training_receipt"].read_text(encoding="utf-8")
        )["outputs"]["execution_binding_sha256"],
    }
    isolation = plan["promotion_cohort_disjointness"]
    assert isolation["overlap_count"] == 0
    assert isolation["manifest"]["path"] == str(fixture["cohort_exclusions"])
    assert {item["kind"] for item in isolation["final_seed_intervals"]} == {
        "internal_h2h",
        "external_panel",
    }


def test_dry_run_accepts_exact_authenticated_eight_rank_one_dose_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _receipt, modern = _modernize_one_dose_receipt(fixture, world_size=8)
    monkeypatch.setattr(
        one_dose,
        "_verify_ddp_canary_receipt",
        lambda _path, *, reference_time_ns: modern["canary"],
    )

    plan = _execute(fixture, go=False)

    assert plan["candidate"]["sha256"] == promotion._sha256(fixture["candidate"])
    assert plan["training_receipt"]["execution_binding_sha256"] == json.loads(
        fixture["training_receipt"].read_text(encoding="utf-8")
    )["outputs"]["execution_binding_sha256"]


def test_eight_rank_one_dose_topology_fails_closed_on_gpu_ownership_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    receipt, modern = _modernize_one_dose_receipt(fixture, world_size=8)
    monkeypatch.setattr(
        one_dose,
        "_verify_ddp_canary_receipt",
        lambda _path, *, reference_time_ns: modern["canary"],
    )
    receipt["gpus"] = list(range(7)) + [9]
    _rewrite_one_dose_receipt(fixture["training_receipt"], receipt)

    with pytest.raises(promotion.PromotionError, match="topology attestation"):
        _execute(fixture, go=False)


def test_post_wave_composite_report_uses_authenticated_outer_ddp_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    receipt, modern = _modernize_one_dose_receipt(fixture, world_size=8)
    monkeypatch.setattr(
        one_dose,
        "_verify_ddp_canary_receipt",
        lambda _path, *, reference_time_ns: modern["canary"],
    )
    sampling_sha = "sha256:" + "8" * 64
    split_sha = "sha256:" + "9" * 64
    trainer_authority = one_dose._current_production_trainer_authority()
    event_history_acknowledgements = [
        "sha256:" + f"{index:x}" * 64 for index in range(6, 10)
    ]
    event_history_training_contract = {
        "schema": "a1-training-event-history-contract-v1",
        "training_event_history_trainable": False,
        "event_history_end_to_end_usable": False,
        "status": "empty_payloads_acknowledged",
        "empty_payload_inventory_acknowledgements": (
            event_history_acknowledgements
        ),
    }
    event_history_component_authority = [
        {
            "component_id": component_id,
            "payload_inventory_sha256": inventory_sha256,
        }
        for component_id, inventory_sha256 in zip(
            (
                "current_producer",
                "recent_history",
                "hard_negative",
                "historical_replay",
            ),
            event_history_acknowledgements,
            strict=True,
        )
    ]
    lock_authority = {
        "schema_version": "a1-frozen-lock-verifier-authority-v1",
        "lock": str(fixture["contract_path"]),
        "lock_file_sha256": promotion._sha256(fixture["contract_path"]),
        "contract_sha256": fixture["contract"]["contract_sha256"],
        "frozen_repo": str(tmp_path / "frozen-repo"),
        "verifier": str(tmp_path / "frozen-repo/tools/a1_pre_wave_contract.py"),
        "verifier_sha256": "sha256:" + "c" * 64,
        "require_all_job_claims": True,
        "verified_lock_sha256": promotion._digest_value(fixture["contract"]),
        "authority_sha256": "sha256:" + "d" * 64,
    }
    monkeypatch.setattr(
        one_dose.frozen_lock_verifier,
        "verify_frozen_lock",
        lambda *_args, **_kwargs: (fixture["contract"], lock_authority),
    )
    input_binding = receipt["input_binding"]
    for key in (
        "validation_manifest",
        "validation_manifest_file_sha256",
        "selected_game_seed_set_sha256",
        "training_game_seed_set_sha256",
        "validation_game_seed_set_sha256",
        "binding_sha256",
    ):
        input_binding.pop(key)
    input_binding.update(
        {
            "data_kind": "production_composite_v2",
            "trainer_authority": trainer_authority,
            "lock_verifier_authority": lock_authority,
            "event_history_training_contract": event_history_training_contract,
            "event_history_component_authority": (
                event_history_component_authority
            ),
            "production_mix_contract_sha256": "sha256:" + "a" * 64,
            "production_sampling_receipt_sha256": sampling_sha,
            "validation_split_receipt": {"schema_version": "test-split-v1"},
            "validation_split_receipt_sha256": split_sha,
            "composite_build_receipt": {"schema_version": "test-build-v1"},
            "source_authority": None,
            "category_semantics": {"current_producer": "current"},
            "category_semantics_sha256": "sha256:" + "b" * 64,
        }
    )
    input_binding["binding_sha256"] = promotion._digest_value(input_binding)
    receipt["trainer_authority"] = trainer_authority
    receipt["lock_verifier_authority"] = lock_authority
    receipt["training_transaction_sha256"] = (
        one_dose._training_transaction_sha256(
            command=receipt["command"], input_binding=input_binding
        )
    )
    receipt["production_sampling_receipt_sha256"] = sampling_sha
    receipt["validation_split_receipt_sha256"] = split_sha
    receipt["outputs"]["production_sampling_receipt_sha256"] = sampling_sha
    receipt["outputs"]["validation_split_receipt_sha256"] = split_sha
    receipt["outputs"]["input_binding_sha256"] = input_binding["binding_sha256"]
    validation_seeds = [880_001, 880_002]
    validation_seed_digest = one_dose.train_bc._game_seed_set_sha256(  # noqa: SLF001
        np.asarray(validation_seeds, dtype=np.int64)
    )
    validation_seed_manifest = tmp_path / "train.validation_seeds.json"
    _write_json(
        validation_seed_manifest,
        {
            "schema_version": "train-validation-game-seeds-v1",
            "validation_game_seed_count": len(validation_seeds),
            "validation_game_seed_set_sha256": validation_seed_digest,
            "game_seeds": validation_seeds,
        },
    )
    receipt["outputs"].update(
        {
            "validation_seed_manifest": str(validation_seed_manifest.resolve()),
            "validation_seed_manifest_sha256": promotion._sha256(
                validation_seed_manifest
            ),
            "validation_game_seed_count": len(validation_seeds),
            "validation_game_seed_set_sha256": validation_seed_digest,
        }
    )
    report = json.loads(fixture["report"].read_text(encoding="utf-8"))
    report.update(
        {
            "a1_contract_sha256": None,
            "a1_learner_training_recipe_sha256": None,
            "a1_bound_learner_training_recipe": None,
            one_dose.REPORT_INPUT_BINDING_FIELD: input_binding,
            "validation_game_seed_manifest": str(
                validation_seed_manifest.resolve()
            ),
            "validation_game_seed_count": len(validation_seeds),
            "validation_game_seed_set_sha256": validation_seed_digest,
            "checkout_runtime_binding": {},
            "training_information_surface": {},
            "public_award_feature_contract": "authoritative_v1",
            "public_award_feature_training": {},
        }
    )
    _write_json(fixture["report"], report)
    receipt["outputs"]["report_sha256"] = promotion._sha256(fixture["report"])
    _rewrite_one_dose_receipt(fixture["training_receipt"], receipt)
    adjudication = json.loads(fixture["adjudication"].read_text(encoding="utf-8"))
    adjudication["candidate"]["training_report"]["sha256"] = promotion._sha256(
        fixture["report"]
    )
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    runtime_binding = {"test": "runtime"}
    monkeypatch.setattr(
        one_dose,
        "_verify_production_checkout_runtime_binding",
        lambda *_args, **_kwargs: runtime_binding,
    )
    monkeypatch.setattr(
        one_dose,
        "_require_production_event_history_surface",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        one_dose,
        "_require_production_public_award_transition",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        promotion,
        "_require_calibration_matches_training_validation_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        promotion,
        "_require_high_regret_matches_training_validation_manifest",
        lambda *_args, **_kwargs: None,
    )
    import torch

    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {
            "public_award_feature_contract": "authoritative_v1",
            "training_information_surface": {},
            "value_training": {"checkout_runtime_binding": runtime_binding},
        },
    )

    plan = _execute(fixture, go=False)

    assert plan["candidate"]["sha256"] == promotion._sha256(fixture["candidate"])


def test_explicit_same_trajectory_checkpoint_selection_replays_64_96_128(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    receipt, modern = _modernize_one_dose_receipt(
        fixture, world_size=8, with_intermediate=True
    )
    monkeypatch.setattr(
        one_dose,
        "_verify_ddp_canary_receipt",
        lambda _path, *, reference_time_ns: modern["canary"],
    )
    screen = tmp_path / "matched-quick-screen.json"
    _write_matched_quick_screen(
        screen,
        intermediate=modern["intermediate"],
        terminal=fixture["candidate"],
        baseline=fixture["champion"],
        evaluation_binding=fixture["evaluation_binding"],
        points_by_step={64: 100.0, 96: 130.0, 128: 110.0},
    )
    selected = modern["intermediate"][1]
    selection = {
        "schema_version": promotion.CHECKPOINT_SELECTION_SCHEMA,
        "training_receipt": {
            "path": str(fixture["training_receipt"]),
            "sha256": promotion._sha256(fixture["training_receipt"]),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "training_report": {
            "path": str(fixture["report"]),
            "sha256": promotion._sha256(fixture["report"]),
        },
        "training_parent": _checkpoint_ref(fixture["champion"]),
        "terminal_checkpoint": _checkpoint_ref(fixture["candidate"]),
        "eligible_optimizer_steps": [64, 96, 128],
        "selected_optimizer_step": 96,
        "selected_checkpoint": {
            "path": selected["checkpoint"],
            "sha256": selected["checkpoint_sha256"],
        },
        "screen_evidence": _checkpoint_ref(screen),
        "selection_basis": "matched_quick_screen",
        "matched_common_random_numbers": True,
        "full_gate_requires_disjoint_cohort": True,
        "fresh_adam_single_trajectory": True,
        "resume_or_candidate_chaining": False,
    }
    selection["selection_sha256"] = promotion._digest_value(selection)
    selection_path = tmp_path / "checkpoint-selection.json"
    _write_json(selection_path, selection)

    verified = promotion._verify_one_dose_training_receipt(
        fixture["training_receipt"],
        contract_lock=fixture["contract_path"],
        contract=fixture["contract"],
        candidate_path=Path(selected["checkpoint"]),
        candidate_sha256=selected["checkpoint_sha256"],
        training_report_path=fixture["report"],
        training_report_sha256=promotion._sha256(fixture["report"]),
        checkpoint_selection_path=selection_path,
        checkpoint_selection_sha256=promotion._sha256(selection_path),
    )

    assert verified["checkpoint_selection"]["selected_optimizer_step"] == 96
    assert verified["report_checkpoint"] == _checkpoint_ref(fixture["candidate"])


def test_checkpoint_selection_rejects_candidate_chaining_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    receipt, modern = _modernize_one_dose_receipt(
        fixture, world_size=8, with_intermediate=True
    )
    monkeypatch.setattr(
        one_dose,
        "_verify_ddp_canary_receipt",
        lambda _path, *, reference_time_ns: modern["canary"],
    )
    screen = tmp_path / "matched-quick-screen.json"
    _write_matched_quick_screen(
        screen,
        intermediate=modern["intermediate"],
        terminal=fixture["candidate"],
        baseline=fixture["champion"],
        evaluation_binding=fixture["evaluation_binding"],
        points_by_step={64: 120.0, 96: 120.0, 128: 120.0},
    )
    selected = modern["intermediate"][0]
    selection = {
        "schema_version": promotion.CHECKPOINT_SELECTION_SCHEMA,
        "training_receipt": {
            "path": str(fixture["training_receipt"]),
            "sha256": promotion._sha256(fixture["training_receipt"]),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "training_report": _checkpoint_ref(fixture["report"]),
        "training_parent": _checkpoint_ref(fixture["champion"]),
        "terminal_checkpoint": _checkpoint_ref(fixture["candidate"]),
        "eligible_optimizer_steps": [64, 96, 128],
        "selected_optimizer_step": 64,
        "selected_checkpoint": {
            "path": selected["checkpoint"],
            "sha256": selected["checkpoint_sha256"],
        },
        "screen_evidence": _checkpoint_ref(screen),
        "selection_basis": "matched_quick_screen",
        "matched_common_random_numbers": True,
        "full_gate_requires_disjoint_cohort": True,
        "fresh_adam_single_trajectory": True,
        "resume_or_candidate_chaining": True,
    }
    selection["selection_sha256"] = promotion._digest_value(selection)
    selection_path = tmp_path / "checkpoint-selection.json"
    _write_json(selection_path, selection)

    with pytest.raises(promotion.PromotionError, match="selection policy drifted"):
        promotion._verify_one_dose_training_receipt(
            fixture["training_receipt"],
            contract_lock=fixture["contract_path"],
            contract=fixture["contract"],
            candidate_path=Path(selected["checkpoint"]),
            candidate_sha256=selected["checkpoint_sha256"],
            training_report_path=fixture["report"],
            training_report_sha256=promotion._sha256(fixture["report"]),
            checkpoint_selection_path=selection_path,
            checkpoint_selection_sha256=promotion._sha256(selection_path),
        )


def test_select_dose_builder_seals_same_trajectory_choice(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _receipt, modern = _modernize_one_dose_receipt(
        fixture, world_size=8, with_intermediate=True
    )
    screen = tmp_path / "matched-quick-screen.json"
    _write_matched_quick_screen(
        screen,
        intermediate=modern["intermediate"],
        terminal=fixture["candidate"],
        baseline=fixture["champion"],
        evaluation_binding=fixture["evaluation_binding"],
        points_by_step={64: 120.0, 96: 121.0, 128: 123.0},
    )
    output = tmp_path / "checkpoint-selection.json"

    selection = promotion.create_same_trajectory_checkpoint_selection(
        training_receipt_path=fixture["training_receipt"],
        training_report_path=fixture["report"],
        screen_evidence_path=screen,
        selected_optimizer_step=64,
        output_path=output,
    )

    assert selection["selected_optimizer_step"] == 64
    assert selection["selected_checkpoint"]["path"].endswith(
        "candidate_step0064.pt"
    )
    assert selection["file_sha256"] == promotion._sha256(output)


def test_select_dose_derives_step_and_rejects_caller_override(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _receipt, modern = _modernize_one_dose_receipt(
        fixture, world_size=8, with_intermediate=True
    )
    screen = tmp_path / "matched-quick-screen.json"
    _write_matched_quick_screen(
        screen,
        intermediate=modern["intermediate"],
        terminal=fixture["candidate"],
        baseline=fixture["champion"],
        evaluation_binding=fixture["evaluation_binding"],
        points_by_step={64: 100.0, 96: 130.0, 128: 110.0},
    )

    with pytest.raises(promotion.PromotionError, match="caller-selected.*differs"):
        promotion.create_same_trajectory_checkpoint_selection(
            training_receipt_path=fixture["training_receipt"],
            training_report_path=fixture["report"],
            screen_evidence_path=screen,
            selected_optimizer_step=64,
            output_path=tmp_path / "checkpoint-selection.json",
        )


def test_select_dose_rejects_result_not_bound_to_common_games(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _receipt, modern = _modernize_one_dose_receipt(
        fixture, world_size=8, with_intermediate=True
    )
    screen = tmp_path / "matched-quick-screen.json"
    payload = _write_matched_quick_screen(
        screen,
        intermediate=modern["intermediate"],
        terminal=fixture["candidate"],
        baseline=fixture["champion"],
        evaluation_binding=fixture["evaluation_binding"],
        points_by_step={64: 120.0, 96: 120.0, 128: 120.0},
    )
    payload["results"][1]["ordered_game_keys_sha256"] = "sha256:" + "d" * 64
    payload.pop("screen_sha256")
    payload["screen_sha256"] = promotion._digest_value(payload)
    _write_json(screen, payload)

    with pytest.raises(promotion.PromotionError, match="common checkpoint/games/search"):
        promotion.create_same_trajectory_checkpoint_selection(
            training_receipt_path=fixture["training_receipt"],
            training_report_path=fixture["report"],
            screen_evidence_path=screen,
            output_path=tmp_path / "checkpoint-selection.json",
        )


def test_build_dose_screen_from_three_pooled_reports(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _receipt, modern = _modernize_one_dose_receipt(
        fixture, world_size=8, with_intermediate=True
    )
    fixture_screen = tmp_path / "fixture-screen.json"
    fixture_payload = _write_matched_quick_screen(
        fixture_screen,
        intermediate=modern["intermediate"],
        terminal=fixture["candidate"],
        baseline=fixture["champion"],
        evaluation_binding=fixture["evaluation_binding"],
        points_by_step={64: 120.0, 96: 121.0, 128: 123.0},
    )
    reports = {
        result["optimizer_step"]: Path(result["evaluation_report"]["path"])
        for result in fixture_payload["results"]
    }

    screen = promotion.create_matched_quick_screen(
        step64_report_path=reports[64],
        step96_report_path=reports[96],
        step128_report_path=reports[128],
        output_path=tmp_path / "operator-screen.json",
    )

    assert screen["selected_optimizer_step"] == 64
    assert screen["candidate_checkpoint_sha256"] == modern["intermediate"][0][
        "checkpoint_sha256"
    ]
    assert screen["file_sha256"] == promotion._sha256(
        tmp_path / "operator-screen.json"
    )


def test_build_dose_screen_rejects_cross_report_search_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _receipt, modern = _modernize_one_dose_receipt(
        fixture, world_size=8, with_intermediate=True
    )
    fixture_screen = tmp_path / "fixture-screen.json"
    fixture_payload = _write_matched_quick_screen(
        fixture_screen,
        intermediate=modern["intermediate"],
        terminal=fixture["candidate"],
        baseline=fixture["champion"],
        evaluation_binding=fixture["evaluation_binding"],
        points_by_step={64: 120.0, 96: 121.0, 128: 123.0},
    )
    reports = {
        result["optimizer_step"]: Path(result["evaluation_report"]["path"])
        for result in fixture_payload["results"]
    }
    changed = json.loads(reports[96].read_text(encoding="utf-8"))
    changed["effective_search_config"]["n_full"] = 256
    changed["fleet_merge"]["effective_search_config_sha256"] = (
        promotion._digest_value(changed["effective_search_config"])
    )
    _write_json(reports[96], changed)

    with pytest.raises(promotion.PromotionError, match="identical search configuration"):
        promotion.create_matched_quick_screen(
            step64_report_path=reports[64],
            step96_report_path=reports[96],
            step128_report_path=reports[128],
            output_path=tmp_path / "operator-screen.json",
        )


def test_internal_screen_rejects_forged_runtime_and_registry_search_identity(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _receipt, modern = _modernize_one_dose_receipt(
        fixture, world_size=8, with_intermediate=True
    )
    fixture_payload = _write_matched_quick_screen(
        tmp_path / "fixture-screen.json",
        intermediate=modern["intermediate"],
        terminal=fixture["candidate"],
        baseline=fixture["champion"],
        evaluation_binding=fixture["evaluation_binding"],
        points_by_step={64: 120.0, 96: 121.0, 128: 123.0},
    )
    reports = {
        result["optimizer_step"]: Path(result["evaluation_report"]["path"])
        for result in fixture_payload["results"]
    }

    runtime_drift = json.loads(reports[64].read_text(encoding="utf-8"))
    runtime_drift["engine_identity"]["native_runtime_sha256"] = "sha256:" + "0" * 64
    _write_json(reports[64], runtime_drift)
    with pytest.raises(promotion.PromotionError, match="runtime internal evaluator"):
        promotion.create_matched_quick_screen(
            step64_report_path=reports[64],
            step96_report_path=reports[96],
            step128_report_path=reports[128],
            output_path=tmp_path / "runtime-drift-screen.json",
        )

    _write_matched_quick_screen(
        tmp_path / "fixture-screen.json",
        intermediate=modern["intermediate"],
        terminal=fixture["candidate"],
        baseline=fixture["champion"],
        evaluation_binding=fixture["evaluation_binding"],
        points_by_step={64: 120.0, 96: 121.0, 128: 123.0},
    )
    forged = json.loads(reports[64].read_text(encoding="utf-8"))
    incumbent = forged["evaluation_binding"]["authoritative_incumbent"]
    incumbent["search_config"]["c_scale"] = 0.3
    incumbent["agent_identity_sha256"] = promotion._agent_identity(  # noqa: SLF001
        {"path": incumbent["path"], "sha256": incumbent["sha256"]},
        incumbent["search_config"],
    )["agent_identity_sha256"]
    _write_json(reports[64], forged)
    with pytest.raises(promotion.PromotionError, match="sealed A1 semantic drift"):
        promotion.create_matched_quick_screen(
            step64_report_path=reports[64],
            step96_report_path=reports[96],
            step128_report_path=reports[128],
            output_path=tmp_path / "search-drift-screen.json",
        )


def test_same_trajectory_parent_uses_causal_sha_not_transformed_initializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformed = tmp_path / "transformed-init.pt"
    transformed.write_bytes(b"function-preserving transformed bytes")
    report = tmp_path / "typed-training-report.json"
    _write_json(report, {"init_checkpoint": str(transformed)})
    causal_sha256 = "sha256:" + "a" * 64
    monkeypatch.setattr(
        promotion,
        "_training_evaluation_parent_sha256",
        lambda _report, _receipt: causal_sha256,
    )

    assert (
        promotion._same_trajectory_training_parent_sha256(  # noqa: SLF001
            training_receipt={}, training_report_path=report
        )
        == causal_sha256
    )


def test_build_dose_screen_rejects_candidate_checkpoint_hash_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _receipt, modern = _modernize_one_dose_receipt(
        fixture, world_size=8, with_intermediate=True
    )
    fixture_screen = tmp_path / "fixture-screen.json"
    fixture_payload = _write_matched_quick_screen(
        fixture_screen,
        intermediate=modern["intermediate"],
        terminal=fixture["candidate"],
        baseline=fixture["champion"],
        evaluation_binding=fixture["evaluation_binding"],
        points_by_step={64: 120.0, 96: 121.0, 128: 123.0},
    )
    reports = {
        result["optimizer_step"]: Path(result["evaluation_report"]["path"])
        for result in fixture_payload["results"]
    }
    changed = json.loads(reports[64].read_text(encoding="utf-8"))
    changed["candidate_checkpoint_sha256"] = "sha256:" + "e" * 64
    _write_json(reports[64], changed)

    with pytest.raises(promotion.PromotionError, match="checkpoint bytes drifted"):
        promotion.create_matched_quick_screen(
            step64_report_path=reports[64],
            step96_report_path=reports[96],
            step128_report_path=reports[128],
            output_path=tmp_path / "operator-screen.json",
        )


def _mutate_cohort_exclusions(fixture: dict, mutate) -> dict:
    path = fixture["cohort_exclusions"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    mutate(payload)
    payload["manifest_sha256"] = promotion._digest_value(payload)
    _write_json(path, payload)
    return payload


@pytest.mark.parametrize(
    ("base_seed", "end_seed"),
    [
        (7_000_050, 7_000_060),
        (8_100_050, 8_100_060),
    ],
)
def test_promotion_refuses_declared_ranges_not_derived_from_diagnostic_report(
    tmp_path: Path, base_seed: int, end_seed: int
) -> None:
    fixture = _fixture(tmp_path)

    def overlap(payload: dict) -> None:
        payload["cohorts"][0]["seed_intervals"] = [
            {"base_seed": base_seed, "end_seed": end_seed}
        ]

    _mutate_cohort_exclusions(fixture, overlap)
    with pytest.raises(
        promotion.PromotionError,
        match="do not exactly match",
    ):
        _execute(fixture, go=False)


def test_promotion_refuses_mutated_bound_diagnostic_source(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    exclusions = json.loads(fixture["cohort_exclusions"].read_text(encoding="utf-8"))
    source = Path(exclusions["cohorts"][0]["source"]["path"])
    source.write_text("mutated\n", encoding="utf-8")

    with pytest.raises(promotion.PromotionError, match="artifact drift"):
        _execute(fixture, go=False)


def test_promotion_recomputes_candidate_hash_from_bound_diagnostic_report(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    exclusions = json.loads(fixture["cohort_exclusions"].read_text(encoding="utf-8"))
    source = Path(exclusions["cohorts"][0]["source"]["path"])
    report = json.loads(source.read_text(encoding="utf-8"))
    report["candidate_checkpoint_sha256"] = "sha256:" + "e" * 64
    _write_json(source, report)
    exclusions["cohorts"][0]["source"]["sha256"] = promotion._sha256(source)
    exclusions.pop("manifest_sha256")
    exclusions["manifest_sha256"] = promotion._digest_value(exclusions)
    _write_json(fixture["cohort_exclusions"], exclusions)

    with pytest.raises(promotion.PromotionError, match="candidate checkpoint hash drifted"):
        _execute(fixture, go=False)


def test_promotion_refuses_final_internal_cohort_without_exact_seeds(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def remove_seed(source: dict) -> None:
        source["games"][0].pop("game_seed")

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=remove_seed,
    )
    with pytest.raises(promotion.PromotionError, match="invalid cohort identity"):
        _execute(fixture, go=False)


def test_ordinary_promotion_refuses_unpaired_internal_orientation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def duplicate_orientation(source: dict) -> None:
        first, second = source["games"][:2]
        second["orientation"] = first["orientation"]
        second["search_seeds_by_role"] = dict(first["search_seeds_by_role"])

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=duplicate_orientation,
    )
    with pytest.raises(promotion.PromotionError, match="repeats a paired orientation"):
        _execute(fixture, go=False)


def test_promotion_requires_corrected_schedule_invariant_search_rng(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def remove_contract(source: dict) -> None:
        source.pop("search_rng_contract")

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=remove_contract,
    )
    with pytest.raises(promotion.PromotionError, match="corrected per-game/seat"):
        _execute(fixture, go=False)


def test_promotion_replays_each_role_to_seat_search_seed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def drift_seed(source: dict) -> None:
        source["games"][0]["search_seeds_by_role"]["candidate"] += 1

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=drift_seed,
    )
    with pytest.raises(promotion.PromotionError, match="role/seat binding"):
        _execute(fixture, go=False)


def test_promotion_refuses_partial_exit_zero_internal_report(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def claim_larger_assignment(source: dict) -> None:
        source["pairs_requested"] = source["complete_pairs"] + 100

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=claim_larger_assignment,
    )
    with pytest.raises(promotion.PromotionError, match="completed 200 of 300"):
        _execute(fixture, go=False)


def test_promotion_receipt_binds_deployed_agent_identities(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    plan = _execute(fixture, go=False)

    candidate = plan["candidate"]["agent_identity"]
    champion = plan["champion"]["agent_identity"]
    assert candidate["search_config"]["c_scale"] == 0.10
    assert champion["search_config"]["c_scale"] == 0.03
    assert {
        key: value
        for key, value in candidate["search_config"].items()
        if key != "c_scale"
    } == {
        key: value
        for key, value in champion["search_config"].items()
        if key != "c_scale"
    }
    for identity in (candidate, champion):
        unhashed = dict(identity)
        declared = unhashed.pop("agent_identity_sha256")
        assert declared == promotion._digest_value(unhashed)


def test_adjudication_cannot_rebind_candidate_to_champion_search(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    adjudication = json.loads(fixture["adjudication"].read_text())
    identity = adjudication["candidate"]["agent_identity"]
    identity["search_config"]["c_scale"] = 0.03
    identity.pop("agent_identity_sha256")
    identity["agent_identity_sha256"] = promotion._digest_value(identity)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="sealed A1 semantic drift"):
        _execute(fixture, go=False)


@pytest.mark.parametrize(
    "repair_kind",
    [
        one_dose.RETRY_REPAIR_KIND,
        one_dose.PRODUCTION_PREFLIGHT_RETRY_REPAIR_KIND,
        one_dose.PRODUCTION_PREFLIGHT_TRANSPORT_RETRY_REPAIR_KIND,
    ],
)
def test_dry_run_accepts_schema_separated_sealed_retry_receipt(
    tmp_path: Path, repair_kind: str
) -> None:
    fixture = _fixture(tmp_path)
    _convert_training_receipt_to_sealed_retry(fixture, repair_kind=repair_kind)

    plan = _execute(fixture, go=False)

    assert plan["status"] == "dry_run"
    assert plan["training_receipt"]["path"] == str(fixture["training_receipt"])


def test_dry_run_rejects_unknown_sealed_retry_repair_kind(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _convert_training_receipt_to_sealed_retry(
        fixture, repair_kind="untyped_retry_escape_hatch"
    )

    with pytest.raises(promotion.PromotionError, match="retry identity is invalid"):
        _execute(fixture, go=False)


def test_direct_promotion_rejects_legacy_one_dose_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _mutate_training_receipt(
        fixture,
        lambda payload: payload.__setitem__(
            "schema_version", "a1-one-dose-training-receipt-v2"
        ),
    )

    with pytest.raises(promotion.PromotionError, match="receipt schema"):
        _execute(fixture, go=False)


def test_direct_promotion_rejects_environment_binding_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def drift_environment(payload: dict) -> None:
        environment = payload["execution_binding"]["environment"]
        environment["PYTHONHASHSEED"] = "123"
        payload["execution_binding"]["environment_sha256"] = promotion._digest_value(
            environment
        )
        payload["outputs"]["execution_binding_sha256"] = promotion._digest_value(
            payload["execution_binding"]
        )

    _mutate_training_receipt(fixture, drift_environment)

    with pytest.raises(promotion.PromotionError, match="exact allowlist"):
        _execute(fixture, go=False)


def test_direct_promotion_rejects_candidate_not_created_by_dose(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    other = tmp_path / "other.pt"
    other.write_bytes(b"other candidate")

    def swap_candidate(payload: dict) -> None:
        payload["outputs"]["checkpoint"] = str(other)
        payload["outputs"]["checkpoint_sha256"] = promotion._sha256(other)

    _mutate_training_receipt(fixture, swap_candidate)

    with pytest.raises(promotion.PromotionError, match="candidate differs"):
        _execute(fixture, go=False)


def test_direct_promotion_rejects_missing_durable_claim(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (tmp_path / "one-dose.claim.json").unlink()

    with pytest.raises(promotion.PromotionError, match="durable claim"):
        _execute(fixture, go=False)


def _mutate_evidence_source(fixture: dict, *, kind: str, role: str, mutate) -> None:
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == kind
    )
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    source_ref = next(item for item in envelope["sources"] if item["role"] == role)
    source_path = Path(source_ref["path"])
    source = json.loads(source_path.read_text())
    mutate(source)
    _write_json(source_path, source)
    source_ref["sha256"] = promotion._sha256(source_path)
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)


def test_historical_comparison_cannot_be_used_as_promotion_baseline(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def make_historical(source: dict) -> None:
        binding = source["evaluation_binding"]
        binding["comparison_mode"] = "historical_comparison"
        binding["promotion_eligible"] = False
        binding["historical_comparison_reason"] = "diagnostic gen3 comparison"

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=make_historical,
    )
    with pytest.raises(
        promotion.PromotionError, match="not a promotion-parent evaluation binding"
    ):
        _execute(fixture, go=False)


def test_branch_challenge_dry_run_authenticates_older_parent_and_incumbent(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "f7-parent.pt"
    fixture = _fixture(tmp_path, branch_parent=parent)

    plan = _execute(fixture, go=False)

    assert plan["status"] == "dry_run"
    adjudication = json.loads(fixture["adjudication"].read_text(encoding="utf-8"))
    assert adjudication["promotion_mode"] == "branch_challenge"
    assert adjudication["candidate_lineage"]["initializer"] == _checkpoint_ref(parent)
    assert adjudication["candidate_lineage"]["displaced_incumbent"]["sha256"] == (
        promotion._sha256(fixture["champion"])
    )


def test_branch_challenge_rejects_unbound_initializer(tmp_path: Path) -> None:
    parent = tmp_path / "f7-parent.pt"
    fixture = _fixture(tmp_path, branch_parent=parent)
    unrelated = tmp_path / "unrelated.pt"
    unrelated.write_bytes(b"not the candidate initializer")
    adjudication = json.loads(fixture["adjudication"].read_text(encoding="utf-8"))
    adjudication["candidate_lineage"]["initializer"] = _checkpoint_ref(unrelated)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(
        promotion.PromotionError,
        match="candidate_lineage.initializer does not bind",
    ):
        _execute(fixture, go=False)


def test_branch_challenge_rejects_historical_evaluation_binding(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, branch_parent=tmp_path / "f7-parent.pt")

    def make_historical(source: dict) -> None:
        binding = source["evaluation_binding"]
        binding["schema_version"] = "a1-evaluation-baseline-binding-v1"
        binding["comparison_mode"] = "historical_comparison"
        binding["promotion_eligible"] = False
        binding["historical_comparison_reason"] = "diagnostic only"

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h_cohort_1",
        mutate=make_historical,
    )
    with pytest.raises(
        promotion.PromotionError,
        match="not a promotion-eligible branch-challenge binding",
    ):
        _execute(fixture, go=False)


def test_branch_challenge_rejects_baseline_other_than_current_incumbent(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, branch_parent=tmp_path / "f7-parent.pt")

    def rebind_baseline_to_parent(source: dict) -> None:
        source["evaluation_binding"]["baseline"] = _checkpoint_ref(
            fixture["branch_parent"]
        )

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h_cohort_1",
        mutate=rebind_baseline_to_parent,
    )
    with pytest.raises(
        promotion.PromotionError,
        match="evaluation_binding.baseline does not bind",
    ):
        _execute(fixture, go=False)


def test_branch_challenge_rejects_registry_bytes_changed_after_evaluation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, branch_parent=tmp_path / "f7-parent.pt")
    fixture["registry"].write_text(
        fixture["registry"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(promotion.PromotionError, match="artifact drift"):
        _execute(fixture, go=False)


def test_branch_challenge_external_regression_is_a_hard_veto(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, branch_parent=tmp_path / "f7-parent.pt")
    _set_external_panel_outcomes(fixture, role="candidate_panel", wins=390)
    _set_external_panel_outcomes(fixture, role="champion_panel", wins=420)

    with pytest.raises(promotion.PromotionError, match="noninferiority is unresolved"):
        _execute(fixture, go=False)


def test_branch_challenge_rejects_unpaired_external_panel(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, branch_parent=tmp_path / "f7-parent.pt")

    def change_one_matched_seed(source: dict) -> None:
        for game in source["games"]:
            if game["pair_id"] == 0:
                game["game_seed"] += 999_999

    _mutate_evidence_source(
        fixture,
        kind="external_panel",
        role="candidate_panel",
        mutate=change_one_matched_seed,
    )
    with pytest.raises(
        promotion.PromotionError, match="different cohorts/configs"
    ):
        _execute(fixture, go=False)


def test_branch_challenge_rejects_missing_or_tampered_external_panel(
    tmp_path: Path,
) -> None:
    missing = _fixture(tmp_path / "missing", branch_parent=tmp_path / "missing/f7.pt")
    adjudication = json.loads(missing["adjudication"].read_text(encoding="utf-8"))
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == "external_panel"
    )
    envelope_path = Path(evidence_ref["path"])
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["sources"] = [
        source for source in envelope["sources"] if source["role"] != "candidate_panel"
    ]
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(envelope_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(envelope_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(missing["adjudication"], adjudication)
    with pytest.raises(promotion.PromotionError, match="source roles mismatch"):
        _execute(missing, go=False)

    tampered_root = tmp_path / "tampered"
    tampered = _fixture(tampered_root, branch_parent=tampered_root / "f7.pt")
    adjudication = json.loads(tampered["adjudication"].read_text(encoding="utf-8"))
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == "external_panel"
    )
    envelope = json.loads(Path(evidence_ref["path"]).read_text(encoding="utf-8"))
    candidate_source = Path(
        next(
            source
            for source in envelope["sources"]
            if source["role"] == "candidate_panel"
        )["path"]
    )
    candidate_source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(promotion.PromotionError, match="artifact drift"):
        _execute(tampered, go=False)


def test_role_specific_value_squash_diagnostic_cannot_be_promotion_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        source["typed_config"]["fields"]["candidate_value_squash"] = "clip"
        source["typed_config"]["fields"]["baseline_value_squash"] = "tanh"
        digest = hashlib.sha256(
            promotion._canonical_bytes(source["typed_config"])
        ).hexdigest()
        source["config_hash"] = "sha256:" + digest[:16]
        source["full_config_hash"] = "sha256:" + digest

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=mutate,
    )

    with pytest.raises(promotion.PromotionError, match="candidate_value_squash"):
        _execute(fixture, go=False)


@pytest.mark.parametrize(
    ("field", "diagnostic_value"),
    (
        ("candidate_wide_roots_always_full", True),
        (
            "candidate_gameplay_policy_aggregation",
            "aggregate_q_then_improve",
        ),
        ("candidate_rescale_noise_floor_c", 0.25),
        ("candidate_sigma_eval", 0.5),
        ("candidate_sigma_reference_visits", 8),
    ),
)
def test_role_specific_search_diagnostic_cannot_be_promotion_evidence(
    tmp_path: Path,
    field: str,
    diagnostic_value: object,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        source["typed_config"]["fields"][field] = diagnostic_value
        digest = hashlib.sha256(
            promotion._canonical_bytes(source["typed_config"])
        ).hexdigest()
        source["config_hash"] = "sha256:" + digest[:16]
        source["full_config_hash"] = "sha256:" + digest

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=mutate,
    )

    with pytest.raises(promotion.PromotionError, match=field):
        _execute(fixture, go=False)


def test_dry_run_is_read_only_and_attests_global_n128(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before_registry = fixture["registry"].read_bytes()
    before_pointer = fixture["pointer"].read_bytes()

    result = _execute(fixture, go=False)

    assert result["status"] == "dry_run"
    assert result["contract"]["n_full"] == 128
    assert result["contract"]["n_full_wide"] is None
    assert result["fleet_ckpt_updated"] is False
    assert fixture["registry"].read_bytes() == before_registry
    assert fixture["pointer"].read_bytes() == before_pointer
    assert not fixture["receipt"].exists()


def test_dry_run_accepts_hash_replayed_fleet_pooled_internal_source(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    shard_path = tmp_path / "internal-h2h-shard.json"

    def make_pooled(source: dict) -> None:
        _make_internal_pool_shard_complete(
            source,
            candidate=fixture["candidate"],
            champion=fixture["champion"],
        )
        _write_json(shard_path, source)
        pooled = evaluation_pool.pool_internal(
            [shard_path],
            candidate=fixture["candidate"],
            champion=fixture["champion"],
        )
        source.clear()
        source.update(pooled)

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=make_pooled,
    )

    assert _execute(fixture, go=False)["status"] == "dry_run"


def test_fleet_pooled_internal_refuses_effective_config_digest_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    shard_path = tmp_path / "internal-h2h-shard.json"

    def make_bad_pool(source: dict) -> None:
        _make_internal_pool_shard_complete(
            source,
            candidate=fixture["candidate"],
            champion=fixture["champion"],
        )
        _write_json(shard_path, source)
        pooled = evaluation_pool.pool_internal(
            [shard_path],
            candidate=fixture["candidate"],
            champion=fixture["champion"],
        )
        pooled["fleet_merge"]["effective_search_config_sha256"] = (
            "sha256:" + "0" * 64
        )
        source.clear()
        source.update(pooled)

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=make_bad_pool,
    )

    with pytest.raises(promotion.PromotionError, match="effective-search config"):
        _execute(fixture, go=False)


def test_fleet_pooled_internal_replays_raw_games_from_sources(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    shard_path = tmp_path / "internal-h2h-shard.json"

    def forge_pooled_outcome(source: dict) -> None:
        _make_internal_pool_shard_complete(
            source,
            candidate=fixture["candidate"],
            champion=fixture["champion"],
        )
        _write_json(shard_path, source)
        pooled = evaluation_pool.pool_internal(
            [shard_path],
            candidate=fixture["candidate"],
            champion=fixture["champion"],
        )
        pooled["games"][0]["candidate_won"] = False
        pooled["games"][0]["search_won"] = False
        outcomes = [bool(game["candidate_won"]) for game in pooled["games"]]
        scores, diagnostics = promotion.pair_scores_from_h2h_games(pooled["games"])
        pooled["candidate_wins"] = sum(outcomes)
        pooled["baseline_wins"] = len(outcomes) - sum(outcomes)
        pooled["candidate_win_rate"] = sum(outcomes) / len(outcomes)
        pooled["pair_diagnostics"] = diagnostics
        pooled["pentanomial_sprt"] = promotion.evaluate_pentanomial_sprt(
            scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
        )
        pooled["verdict"] = pooled["pentanomial_sprt"]["decision"]
        pooled["superiority_pentanomial_sprt"] = (
            promotion.evaluate_pentanomial_sprt(
                scores, elo0=0.0, elo1=15.0, alpha=0.05, beta=0.05
            )
        )
        pooled["superiority_verdict"] = pooled[
            "superiority_pentanomial_sprt"
        ]["decision"]
        source.clear()
        source.update(pooled)

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=forge_pooled_outcome,
    )
    with pytest.raises(promotion.PromotionError, match="pooled games do not replay"):
        _execute(fixture, go=False)


def test_go_updates_generator_and_pointer_with_committed_receipt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    public_before = ChampionRegistry.load(fixture["registry"]).get_role(
        "public_champion"
    )

    receipt = _execute(fixture, go=True)

    assert receipt["status"] == "committed"
    assert receipt["fleet_ckpt_updated"] is False
    registry = ChampionRegistry.load(fixture["registry"])
    generator = registry.get_role("generator_champion")
    assert generator is not None
    assert Path(generator.checkpoint_path).resolve() == fixture["candidate"].resolve()
    assert generator.version == 5
    assert generator.provenance["a1_one_dose_training_receipt"] == str(
        fixture["training_receipt"]
    )
    assert generator.provenance[
        "a1_one_dose_training_receipt_sha256"
    ] == promotion._sha256(fixture["training_receipt"])
    assert (
        generator.provenance["a1_one_dose_execution_binding_sha256"]
        == json.loads(fixture["training_receipt"].read_text(encoding="utf-8"))[
            "outputs"
        ]["execution_binding_sha256"]
    )
    assert registry.promotion_count() == 1
    assert any(
        Path(entry.checkpoint_path).resolve() == fixture["champion"].resolve()
        for entry in registry.opponent_pool()
    )
    assert registry.get_role("public_champion") == public_before
    assert fixture["pointer"].read_text().strip() == str(fixture["candidate"].resolve())
    saved = json.loads(fixture["receipt"].read_text())
    assert saved["status"] == "committed"
    assert Path(saved["rollback"]["registry_backup"]).is_file()
    assert Path(saved["rollback"]["current_backup"]).is_file()


def test_recovery_is_dry_run_then_restores_exact_before_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registry_before = fixture["registry"].read_bytes()
    pointer_before = fixture["pointer"].read_bytes()
    _execute(fixture, go=True)

    dry = promotion.recover_transaction(
        receipt_path=fixture["receipt"], lock_path=fixture["lock"], go=False
    )
    assert dry["status"] == "recovery_dry_run"
    assert fixture["registry"].read_bytes() != registry_before

    recovered = promotion.recover_transaction(
        receipt_path=fixture["receipt"], lock_path=fixture["lock"], go=True
    )
    assert recovered["status"] == "recovered"
    assert fixture["registry"].read_bytes() == registry_before
    assert fixture["pointer"].read_bytes() == pointer_before
    assert json.loads(fixture["receipt"].read_text())["status"] == "recovered"


def test_recovery_accepts_pre_v2_promotion_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _execute(fixture, go=True)
    receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    receipt.pop("training_receipt")
    receipt.pop("promotion_cohort_disjointness")
    receipt["schema_version"] = promotion.LEGACY_RECEIPT_SCHEMA
    receipt["receipt_sha256"] = promotion._digest_value(receipt)
    _write_json(fixture["receipt"], receipt)

    result = promotion.recover_transaction(receipt_path=fixture["receipt"], go=False)

    assert result["status"] == "recovery_dry_run"


def test_recovery_accepts_v2_promotion_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _execute(fixture, go=True)
    receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    receipt.pop("promotion_cohort_disjointness")
    receipt["schema_version"] = promotion.PREVIOUS_RECEIPT_SCHEMA
    receipt["receipt_sha256"] = promotion._digest_value(receipt)
    _write_json(fixture["receipt"], receipt)

    result = promotion.recover_transaction(receipt_path=fixture["receipt"], go=False)

    assert result["status"] == "recovery_dry_run"


def test_global_n196_contract_is_rejected_before_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, n_full=196)
    before = fixture["registry"].read_bytes()

    with pytest.raises(promotion.PromotionError, match="n_full=128"):
        _execute(fixture, go=True)

    assert fixture["registry"].read_bytes() == before
    assert not fixture["receipt"].exists()


def test_external_panel_without_information_set_attestation_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source = tmp_path / "candidate_panel.raw.json"
    payload = json.loads(source.read_text())
    payload["information_set_search"] = False

    with pytest.raises(promotion.PromotionError, match="information-set recipe"):
        promotion._verify_external_panel_source(
            payload,
            checkpoint=fixture["candidate"],
            checkpoint_md5=promotion._md5(fixture["candidate"]),
            where="candidate external panel",
            sealed_semantics=promotion._sealed_evaluation_semantics(
                fixture["contract"]
            ),
            role="candidate",
            deployed_search_config=payload["search_config"],
        )


def test_candidate_hash_drift_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["candidate"].write_bytes(b"mutated after adjudication")

    with pytest.raises(promotion.PromotionError, match="artifact drift"):
        _execute(fixture, go=False)


def test_training_report_must_name_exact_candidate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = json.loads(fixture["report"].read_text())
    report["checkpoint"] = str(fixture["champion"])
    _write_json(fixture["report"], report)
    adjudication = json.loads(fixture["adjudication"].read_text())
    adjudication["candidate"]["training_report"]["sha256"] = promotion._sha256(
        fixture["report"]
    )
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="report checkpoint differs"):
        _execute(fixture, go=False)


def test_bucket_insufficient_data_is_a_binding_veto(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == "bucket_veto"
    )
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    source_path = Path(envelope["sources"][0]["path"])
    source = json.loads(source_path.read_text())
    source["per_bucket"]["41+"] = {
        "status": "insufficient_data",
        "n": 4,
        "winrate": 0.75,
    }
    _write_json(source_path, source)
    envelope["sources"][0]["sha256"] = promotion._sha256(source_path)
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="do not replay"):
        _execute(fixture, go=False)


def test_bucket_cannot_launder_regression_with_pass_status(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == "bucket_veto"
    )
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    source_path = Path(envelope["sources"][0]["path"])
    source = json.loads(source_path.read_text())
    source["per_bucket"]["41+"] = {"status": "pass", "n": 80, "winrate": 0.44}
    _write_json(source_path, source)
    envelope["sources"][0]["sha256"] = promotion._sha256(source_path)
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="do not replay"):
        _execute(fixture, go=False)


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("mechanism_calibration", "max_rmse_regression"),
        ("external_panel", "max_win_rate_regression"),
    ],
)
def test_evidence_cannot_launder_regression_with_its_own_tolerance(
    tmp_path: Path, kind: str, field: str
) -> None:
    fixture = _fixture(tmp_path)
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == kind
    )
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    envelope["result"][field] = 1.0
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="fixed policy"):
        _execute(fixture, go=False)


def test_calibration_comparison_rejects_different_validation_seed_cohorts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def change_seed_manifest(source: dict) -> None:
        source["row_selection"]["seed_manifest_sha256"] = "sha256:" + "8" * 64

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="candidate_calibration",
        mutate=change_seed_manifest,
    )

    with pytest.raises(promotion.PromotionError, match="different cohorts"):
        _execute(fixture, go=False)


def test_calibration_gate_scores_the_sealed_deployed_transform(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def regress_only_after_tanh(source: dict) -> None:
        # The historical raw-readout check would accept 0.20 versus the
        # incumbent's 0.21.  The actual sealed MCTS operator consumes tanh,
        # whose deliberately regressed RMSE must be the binding value.
        assert source["global"]["value_rmse"] == pytest.approx(0.20)
        source["deployed_readout_diagnostics"]["views"]["scalar_tanh"][
            "global"
        ]["value_rmse"] = 0.50

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="candidate_calibration",
        mutate=regress_only_after_tanh,
    )

    with pytest.raises(promotion.PromotionError, match="allowed RMSE regression"):
        _execute(fixture, go=False)


def test_calibration_gate_rejects_deployed_transform_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def drift_transform(source: dict) -> None:
        source["deployed_readout_diagnostics"]["configured_effective_transform"] = (
            "scalar_clip"
        )

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="candidate_calibration",
        mutate=drift_transform,
    )

    with pytest.raises(promotion.PromotionError, match="deployed value transform"):
        _execute(fixture, go=False)


def test_calibration_accepts_trained_step_checkpoint_before_first_completed_epoch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mark_step_sliced(source: dict) -> None:
        source["readout_provenance"]["optimizer_steps"] = 64
        source["readout_provenance"]["completed_epochs"] = 0

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="candidate_calibration",
        mutate=mark_step_sliced,
    )

    plan = _execute(fixture, go=False)

    assert plan["status"] == "dry_run"


@pytest.mark.parametrize("invalid", [True, -1, None])
def test_calibration_rejects_invalid_completed_epochs_for_trained_checkpoint(
    tmp_path: Path,
    invalid: object,
) -> None:
    fixture = _fixture(tmp_path)

    def corrupt_completed_epochs(source: dict) -> None:
        source["readout_provenance"]["completed_epochs"] = invalid

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="candidate_calibration",
        mutate=corrupt_completed_epochs,
    )

    with pytest.raises(promotion.PromotionError, match="completed_epochs"):
        _execute(fixture, go=False)


def test_calibration_rejects_missing_completed_epochs_for_trained_checkpoint(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def remove_completed_epochs(source: dict) -> None:
        del source["readout_provenance"]["completed_epochs"]

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="candidate_calibration",
        mutate=remove_completed_epochs,
    )

    with pytest.raises(promotion.PromotionError, match="completed_epochs"):
        _execute(fixture, go=False)


def _install_legacy_incumbent_bridge(fixture: dict, tmp_path: Path) -> Path:
    champion = fixture["champion"]
    historical = tmp_path / "gen3-historical-training-report.json"
    _write_json(
        historical,
        {
            "checkpoint": str(champion.resolve()),
            "checkpoint_sha256": promotion._sha256(champion),
            "steps_completed": 912,
            "epochs": 1,
        },
    )

    def add_bridge(source: dict) -> None:
        source["readout_provenance"]["optimizer_steps"] = None
        source["readout_provenance"]["completed_epochs"] = None
        source["legacy_incumbent_provenance"] = {
            "schema_version": promotion.LEGACY_INCUMBENT_PROVENANCE_SCHEMA,
            "contract_sha256": fixture["contract"]["contract_sha256"],
            "checkpoint_sha256": promotion._sha256(champion),
            "historical_training_report": _checkpoint_ref(historical),
        }

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="champion_calibration",
        mutate=add_bridge,
    )
    return historical


def test_exact_contract_incumbent_accepts_typed_legacy_calibration_bridge(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _install_legacy_incumbent_bridge(fixture, tmp_path)

    plan = _execute(fixture, go=False)

    assert plan["status"] == "dry_run"


def test_artifact_built_real_gen3_relative_bridge_passes_promotion_from_other_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    champion = tmp_path / "runs" / "bc" / "gen3_20260706" / "checkpoint.pt"
    fixture = _fixture(tmp_path, champion=champion)
    historical = tmp_path / "reports" / "archive" / "gen3-training-report.json"
    historical.parent.mkdir(parents=True)
    _write_json(
        historical,
        {
            "checkpoint": "runs/bc/gen3_20260706/checkpoint.pt",
            "checkpoint_sha256": promotion._sha256(champion),
            "steps_completed": 912,
            "epochs": 1,
        },
    )
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item
        for item in adjudication["evidence"]
        if item["kind"] == "mechanism_calibration"
    )
    envelope = json.loads(Path(evidence_ref["path"]).read_text())
    source_ref = next(
        item for item in envelope["sources"] if item["role"] == "champion_calibration"
    )
    source_path = Path(source_ref["path"])
    calibration = json.loads(source_path.read_text())
    calibration["readout_provenance"]["optimizer_steps"] = None
    calibration["readout_provenance"]["completed_epochs"] = None
    _write_json(source_path, calibration)
    built = artifacts.build_legacy_incumbent_calibration_source(
        calibration_path=source_path,
        historical_training_report=historical,
        contract=fixture["contract"],
        champion=champion,
    )

    def install_built_bridge(source: dict) -> None:
        source.clear()
        source.update(built)

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="champion_calibration",
        mutate=install_built_bridge,
    )
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    plan = _execute(fixture, go=False)

    assert plan["status"] == "dry_run"


def test_legacy_calibration_bridge_rejects_mutated_historical_report(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    historical = _install_legacy_incumbent_bridge(fixture, tmp_path)
    report = json.loads(historical.read_text())
    report["steps_completed"] = 913
    _write_json(historical, report)

    with pytest.raises(promotion.PromotionError, match="artifact drift"):
        _execute(fixture, go=False)


def test_candidate_cannot_use_legacy_incumbent_calibration_bridge(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    historical = tmp_path / "candidate-historical.json"
    _write_json(
        historical,
        {
            "checkpoint": str(fixture["candidate"].resolve()),
            "steps_completed": 1,
            "epochs": 1,
        },
    )

    def add_bridge(source: dict) -> None:
        source["readout_provenance"]["optimizer_steps"] = None
        source["readout_provenance"]["completed_epochs"] = None
        source["legacy_incumbent_provenance"] = {
            "schema_version": promotion.LEGACY_INCUMBENT_PROVENANCE_SCHEMA,
            "contract_sha256": fixture["contract"]["contract_sha256"],
            "checkpoint_sha256": promotion._sha256(fixture["candidate"]),
            "historical_training_report": _checkpoint_ref(historical),
        }

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="candidate_calibration",
        mutate=add_bridge,
    )
    with pytest.raises(promotion.PromotionError, match="optimizer_steps"):
        _execute(fixture, go=False)


def test_external_comparison_rejects_different_pair_seed_cohorts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def change_pair_seed(source: dict) -> None:
        source["games"][0]["game_seed"] += 1_000_000

    _mutate_evidence_source(
        fixture,
        kind="external_panel",
        role="candidate_panel",
        mutate=change_pair_seed,
    )

    with pytest.raises(promotion.PromotionError, match="different cohorts/configs"):
        _execute(fixture, go=False)


def test_internal_promotion_h2h_must_attest_randomized_base_map(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def use_fixed_tournament_map(source: dict) -> None:
        source["typed_config"]["fields"]["map_kind"] = "TOURNAMENT"
        digest = hashlib.sha256(
            promotion._canonical_bytes(source["typed_config"])
        ).hexdigest()
        source["config_hash"] = "sha256:" + digest[:16]
        source["full_config_hash"] = "sha256:" + digest

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=use_fixed_tournament_map,
    )

    with pytest.raises(promotion.PromotionError, match="randomized BASE"):
        _execute(fixture, go=False)


def test_external_comparison_rejects_different_search_configs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def change_search_config(source: dict) -> None:
        source["search_config"]["c_scale"] = 0.3

    _mutate_evidence_source(
        fixture,
        kind="external_panel",
        role="candidate_panel",
        mutate=change_search_config,
    )

    with pytest.raises(promotion.PromotionError, match="sealed A1 semantic drift"):
        _execute(fixture, go=False)


def test_external_comparison_rejects_other_cohort_metadata_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    _mutate_evidence_source(
        fixture,
        kind="external_panel",
        role="candidate_panel",
        mutate=lambda source: source.__setitem__("gate_config", "drifted-gate"),
    )

    with pytest.raises(promotion.PromotionError, match="different cohorts/configs"):
        _execute(fixture, go=False)


def test_external_reports_cannot_omit_d6_threshold_attestation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    for role in ("candidate_panel", "champion_panel"):
        _mutate_evidence_source(
            fixture,
            kind="external_panel",
            role=role,
            mutate=lambda source: source["search_config"].pop(
                "symmetry_averaged_eval_threshold"
            ),
        )

    with pytest.raises(promotion.PromotionError, match="keys differ"):
        _execute(fixture, go=False)


def test_high_regret_report_cannot_launder_sealed_search_config(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["evaluation_config"]["c_scale"] = 0.3
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="sealed A1 semantic drift"):
        _execute(fixture, go=False)


def test_high_regret_rust_featurize_requires_native_runtime_binding(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["evaluation_config"]["native_mcts_hot_loop"] = False
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )
    with pytest.raises(promotion.PromotionError, match="bound native MCTS runtime"):
        _execute(fixture, go=False)


def test_high_regret_report_binds_candidate_and_incumbent_role_scales(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["evaluation_config"]["baseline_c_scale"] = report[
            "evaluation_config"
        ]["candidate_c_scale"]
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="baseline_c_scale"):
        _execute(fixture, go=False)


def test_high_regret_report_rejects_forged_candidate_role_scale(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["evaluation_config"]["candidate_c_scale"] = 0.03
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="candidate_c_scale"):
        _execute(fixture, go=False)


@pytest.mark.parametrize(
    ("field", "diagnostic_value"),
    (
        ("candidate_wide_roots_always_full", True),
        (
            "candidate_gameplay_policy_aggregation",
            "aggregate_q_then_improve",
        ),
        ("candidate_rescale_noise_floor_c", 0.25),
        ("candidate_sigma_eval", 0.5),
        ("candidate_sigma_reference_visits", 8),
        ("candidate_value_squash", "clip"),
    ),
)
def test_high_regret_report_rejects_role_specific_diagnostic_operator(
    tmp_path: Path,
    field: str,
    diagnostic_value: object,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["evaluation_config"][field] = diagnostic_value
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match=field):
        _execute(fixture, go=False)


def _install_truncated_high_regret_pair(fixture: dict) -> None:
    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        for row in report["games"]:
            if row["orientation"] == "candidate_first":
                row["orientation"] = "candidate_red"
                row["candidate_color"] = "RED"
                row["baseline_color"] = "BLUE"
            else:
                row["orientation"] = "candidate_blue"
                row["candidate_color"] = "BLUE"
                row["baseline_color"] = "RED"
        game = next(
            game
            for game in report["games"]
            if game["pair_id"] == 0
            and game["orientation"] == "candidate_red"
        )
        game["candidate_won"] = None
        game["truncated"] = True
        normalized = [
            {**row, "search_won": row["candidate_won"]}
            for row in report["games"]
        ]
        scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
        pentanomial = promotion.evaluate_pentanomial_sprt(
            scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
        )
        report["pair_diagnostics"] = diagnostics
        report["pentanomial_sprt"] = pentanomial
        _write_json(report_path, report)
        source["complete_pairs"] = 239
        source["pair_diagnostics"] = diagnostics
        source["pentanomial_sprt"] = pentanomial
        source["verdict"] = pentanomial["decision"]
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )


def test_transaction_independently_accepts_legitimate_truncated_high_regret_pair(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _install_truncated_high_regret_pair(fixture)

    plan = _execute(fixture, go=False)

    assert plan["status"] == "dry_run"


def test_transaction_rejects_none_outcome_without_truncation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["games"][0]["candidate_won"] = None
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="inconsistent truncation"):
        _execute(fixture, go=False)


def test_transaction_rejects_forged_high_regret_orientation_color(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _install_truncated_high_regret_pair(fixture)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["games"][0]["candidate_color"] = "BLUE"
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="orientation/color mismatch"):
        _execute(fixture, go=False)


def test_transaction_rejects_internal_candidate_search_outcome_alias_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        source["games"][0]["candidate_won"] = False

    _mutate_evidence_source(
        fixture, kind="internal_h2h", role="internal_h2h", mutate=mutate
    )

    with pytest.raises(
        promotion.PromotionError, match="candidate_won/search_won alias drift"
    ):
        _execute(fixture, go=False)


@pytest.mark.parametrize(
    ("counts", "expected_decision"),
    [
        ((19, 150, 31), "continue"),
        ((2, 292, 6), "H0"),
    ],
)
def test_transaction_rejects_regression_h1_without_superiority_h1(
    tmp_path: Path,
    counts: tuple[int, int, int],
    expected_decision: str,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        games: list[dict] = []
        pair_id = 0
        for pair_count, outcomes in zip(
            counts,
            ((False, False), (True, False), (True, True)),
            strict=True,
        ):
            for _ in range(pair_count):
                for orientation, won in zip(
                    ("candidate_red", "candidate_blue"),
                    outcomes,
                    strict=True,
                ):
                    candidate_color = (
                        "RED" if orientation == "candidate_red" else "BLUE"
                    )
                    baseline_color = "BLUE" if candidate_color == "RED" else "RED"
                    game_seed = 7_000_000 + pair_id
                    games.append(
                        {
                            "pair_id": pair_id,
                            "game_seed": game_seed,
                            "orientation": orientation,
                            "search_won": won,
                            "candidate_won": won,
                            "search_seeds_by_role": {
                                "candidate": promotion._internal_h2h_search_seed(  # noqa: SLF001
                                    game_seed=game_seed,
                                    seat_color=candidate_color,
                                ),
                                "baseline": promotion._internal_h2h_search_seed(  # noqa: SLF001
                                    game_seed=game_seed,
                                    seat_color=baseline_color,
                                ),
                            },
                        }
                    )
                pair_id += 1
        pair_scores, diagnostics = promotion.pair_scores_from_h2h_games(games)
        regression = promotion.evaluate_pentanomial_sprt(
            pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
        )
        superiority = promotion.evaluate_pentanomial_sprt(
            pair_scores, elo0=0.0, elo1=15.0, alpha=0.05, beta=0.05
        )
        assert regression["decision"] == "H1"
        assert superiority["decision"] == expected_decision
        wins = sum(game["candidate_won"] for game in games)
        source.update(
            {
                    "games": games,
                    "base_seed": 7_000_000,
                    "pairs_requested": len(games) // 2,
                    "games_played": len(games),
                "games_with_winner": len(games),
                "complete_pairs": len(games) // 2,
                "candidate_wins": wins,
                "baseline_wins": len(games) - wins,
                "candidate_win_rate": wins / len(games),
                "pair_diagnostics": diagnostics,
                "pentanomial_sprt": regression,
                "verdict": "H1",
                "superiority_pentanomial_sprt": superiority,
                "superiority_verdict": expected_decision,
            }
        )

    _mutate_evidence_source(
        fixture, kind="internal_h2h", role="internal_h2h", mutate=mutate
    )

    with pytest.raises(
        promotion.PromotionError, match="does not prove positive-Elo superiority"
    ):
        _execute(fixture, go=False)


def test_transaction_rejects_legacy_v1_evidence_for_new_promotion(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == "internal_h2h"
    )
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    envelope["schema_version"] = promotion.LEGACY_EVIDENCE_SCHEMA
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="evidence schema/kind mismatch"):
        _execute(fixture, go=False)


def test_transaction_rejects_mismatched_external_panel_engine_identities(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        source["planned_engine_identity"]["repo_commit"] = "f" * 40
        source["engine_identity"]["repo_commit"] = "f" * 40

    _mutate_evidence_source(
        fixture, kind="external_panel", role="champion_panel", mutate=mutate
    )

    with pytest.raises(
        promotion.PromotionError, match="different engine identities"
    ):
        _execute(fixture, go=False)


def test_transaction_rejects_mixed_high_regret_orientation_encodings(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _install_truncated_high_regret_pair(fixture)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        game = report["games"][2]
        game["orientation"] = "candidate_first"
        game["candidate_color"] = "RED"
        game["baseline_color"] = "BLUE"
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="mixes orientation encodings"):
        _execute(fixture, go=False)


def test_transaction_rejects_half_high_regret_pair(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["games"].pop(1)
        normalized = [
            {**row, "search_won": row["candidate_won"]}
            for row in report["games"]
        ]
        scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
        pentanomial = promotion.evaluate_pentanomial_sprt(
            scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
        )
        report["pair_diagnostics"] = diagnostics
        report["pentanomial_sprt"] = pentanomial
        _write_json(report_path, report)
        source["complete_pairs"] = 199
        source["pair_diagnostics"] = diagnostics
        source["pentanomial_sprt"] = pentanomial
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="cover every suite pair twice"):
        _execute(fixture, go=False)


def test_transaction_rejects_inconsistent_truncated_complete_pair_count(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _install_truncated_high_regret_pair(fixture)
    _mutate_evidence_source(
        fixture,
        kind="high_regret",
        role="high_regret",
        mutate=lambda source: source.__setitem__("complete_pairs", 240),
    )

    with pytest.raises(promotion.PromotionError, match="paired statistics"):
        _execute(fixture, go=False)


def test_transaction_rejects_incomplete_bucket_pair(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        report["games"].pop(0)
        _write_json(report_path, report)
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="bucket_veto", role="bucket_veto", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="incomplete bucket pair"):
        _execute(fixture, go=False)


def test_high_regret_games_must_match_frozen_suite_state_identities(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(source: dict) -> None:
        report_path = Path(source["report"]["path"])
        report = json.loads(report_path.read_text())
        suite_path = Path(report["suite_manifest"]["path"])
        suite = json.loads(suite_path.read_text())
        suite["states"][0]["game_seed"] += 1
        suite.pop("suite_sha256")
        suite["suite_sha256"] = promotion._digest_value(suite)
        _write_json(suite_path, suite)
        suite_sha = promotion._sha256(suite_path)
        report["suite_manifest"]["sha256"] = suite_sha
        _write_json(report_path, report)
        source["suite_manifest"]["sha256"] = suite_sha
        source["report"]["sha256"] = promotion._sha256(report_path)

    _mutate_evidence_source(
        fixture, kind="high_regret", role="high_regret", mutate=mutate
    )

    with pytest.raises(promotion.PromotionError, match="not bound to source manifest"):
        _execute(fixture, go=False)


def test_high_regret_promotion_rejects_replaced_replay_scope_bytes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_shard = tmp_path / "high-regret-worker/source-shard.npz"
    source_shard.write_bytes(b"forged easier archived trajectory")

    with pytest.raises(promotion.PromotionError, match="scope inventory drifted"):
        _execute(fixture, go=False)


def _set_external_panel_outcomes(
    fixture: dict,
    *,
    role: str,
    wins: int,
    elo0: float = -10.0,
    elo1: float = 15.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> None:
    def mutate(source: dict) -> None:
        for index, game in enumerate(source["games"]):
            game["candidate_won"] = index < wins
        source["candidate_wins"] = wins
        source["baseline_wins"] = len(source["games"]) - wins
        source["candidate_win_rate"] = wins / len(source["games"])
        normalized = [
            {**game, "search_won": game["candidate_won"]} for game in source["games"]
        ]
        scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
        source["pair_diagnostics"] = diagnostics
        source["pentanomial_sprt"] = promotion.evaluate_pentanomial_sprt(
            scores, elo0=elo0, elo1=elo1, alpha=alpha, beta=beta
        )
        source["verdict"] = source["pentanomial_sprt"]["decision"]

    _mutate_evidence_source(fixture, kind="external_panel", role=role, mutate=mutate)


def test_external_absolute_h0_panels_pass_when_candidate_is_nonregressing(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _set_external_panel_outcomes(fixture, role="candidate_panel", wins=410)
    _set_external_panel_outcomes(fixture, role="champion_panel", wins=420)
    adjudication = json.loads(fixture["adjudication"].read_text())
    envelope_path = Path(
        next(
            item
            for item in adjudication["evidence"]
            if item["kind"] == "external_panel"
        )["path"]
    )
    envelope = json.loads(envelope_path.read_text())
    for source_ref in envelope["sources"]:
        source = json.loads(Path(source_ref["path"]).read_text())
        assert source["verdict"] == "H0"

    assert _execute(fixture, go=False)["status"] == "dry_run"


def test_external_comparative_regression_fails_at_fixed_two_percent_limit(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _set_external_panel_outcomes(fixture, role="candidate_panel", wins=390)
    _set_external_panel_outcomes(fixture, role="champion_panel", wins=420)

    with pytest.raises(promotion.PromotionError, match="noninferiority is unresolved"):
        _execute(fixture, go=False)


@pytest.mark.parametrize(("candidate_wins", "passes"), [(410, True), (390, False)])
def test_external_sprt_threshold_overrides_cannot_change_comparative_eligibility(
    tmp_path: Path, candidate_wins: int, passes: bool
) -> None:
    fixture = _fixture(tmp_path)
    _set_external_panel_outcomes(
        fixture,
        role="candidate_panel",
        wins=candidate_wins,
        elo0=-400.0,
        elo1=-300.0,
        alpha=0.2,
        beta=0.2,
    )
    _set_external_panel_outcomes(
        fixture,
        role="champion_panel",
        wins=420,
        elo0=300.0,
        elo1=400.0,
        alpha=0.01,
        beta=0.01,
    )

    if passes:
        assert _execute(fixture, go=False)["status"] == "dry_run"
    else:
        with pytest.raises(promotion.PromotionError, match="noninferiority is unresolved"):
            _execute(fixture, go=False)


def test_external_comparison_accepts_hash_replayed_fleet_pools(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    for role, checkpoint in (
        ("candidate_panel", fixture["candidate"]),
        ("champion_panel", fixture["champion"]),
    ):
        shard_path = tmp_path / f"{role}.shard.json"

        def make_pooled(source: dict, *, _shard=shard_path, _checkpoint=checkpoint):
            _make_external_pool_shard_complete(source, checkpoint=_checkpoint)
            _write_json(_shard, source)
            pooled = evaluation_pool.pool_neutral(
                [_shard],
                checkpoint=_checkpoint,
            )
            source.clear()
            source.update(pooled)

        _mutate_evidence_source(
            fixture,
            kind="external_panel",
            role=role,
            mutate=make_pooled,
        )

    assert _execute(fixture, go=False)["status"] == "dry_run"


def test_external_point_delta_cannot_pass_without_paired_noninferiority(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    joint = (
        [(True, True)] * 233
        + [(True, False)] * 229
        + [(False, True)] * 219
        + [(False, False)] * 319
    )

    def set_vector(role: str, column: int) -> None:
        def mutate(source: dict) -> None:
            outcomes = [row[column] for row in joint]
            for game, outcome in zip(source["games"], outcomes, strict=True):
                game["candidate_won"] = outcome
            wins = sum(outcomes)
            source["candidate_wins"] = wins
            source["baseline_wins"] = len(outcomes) - wins
            source["candidate_win_rate"] = wins / len(outcomes)
            normalized = [
                {**game, "search_won": game["candidate_won"]}
                for game in source["games"]
            ]
            scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
            source["pair_diagnostics"] = diagnostics
            source["pentanomial_sprt"] = promotion.evaluate_pentanomial_sprt(
                scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
            )
            source["verdict"] = source["pentanomial_sprt"]["decision"]

        _mutate_evidence_source(
            fixture, kind="external_panel", role=role, mutate=mutate
        )

    set_vector("candidate_panel", 0)
    set_vector("champion_panel", 1)

    with pytest.raises(promotion.PromotionError, match="noninferiority is unresolved"):
        _execute(fixture, go=False)


def _different_json_value(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if value is None:
        return 7
    if isinstance(value, str):
        return value + "_drift"
    raise AssertionError(f"no mutation strategy for {value!r}")


@pytest.mark.parametrize(
    "field",
    sorted(promotion._sealed_evaluation_semantics(_contract()).keys()),
)
def test_internal_h2h_rejects_every_sealed_semantic_drift(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)

    def drift_and_rehash(source: dict) -> None:
        fields = source["typed_config"]["fields"]
        fields[field] = _different_json_value(fields[field])
        digest = hashlib.sha256(
            promotion._canonical_bytes(source["typed_config"])
        ).hexdigest()
        source["config_hash"] = "sha256:" + digest[:16]
        source["full_config_hash"] = "sha256:" + digest

    _mutate_evidence_source(
        fixture,
        kind="internal_h2h",
        role="internal_h2h",
        mutate=drift_and_rehash,
    )

    with pytest.raises(promotion.PromotionError, match="sealed A1 semantic drift"):
        _execute(fixture, go=False)


@pytest.mark.parametrize(
    "field",
    sorted(promotion._sealed_evaluation_semantics(_contract()).keys()),
)
def test_external_panels_cannot_jointly_launder_sealed_semantic_drift(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)

    def drift(source: dict) -> None:
        config = source["search_config"]
        config[field] = _different_json_value(config[field])

    for role in ("candidate_panel", "champion_panel"):
        _mutate_evidence_source(
            fixture,
            kind="external_panel",
            role=role,
            mutate=drift,
        )

    with pytest.raises(promotion.PromotionError, match="sealed A1 semantic drift"):
        _execute(fixture, go=False)


def test_promotion_fails_closed_when_contract_omits_search_semantic(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    del fixture["contract"]["science"]["effective_search_config"]["c_scale"]

    with pytest.raises(
        promotion.PromotionError,
        match=r"omits effective_search_config\.c_scale",
    ):
        _execute(fixture, go=False)


def test_every_third_confirmation_is_derived_from_registry(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, promotion_count=2)
    payload = json.loads(fixture["adjudication"].read_text())
    payload["nth_confirmation_required"] = False
    payload.pop("adjudication_sha256")
    payload["adjudication_sha256"] = promotion._digest_value(payload)
    _write_json(fixture["adjudication"], payload)

    with pytest.raises(promotion.PromotionError, match="every-third"):
        _execute(fixture, go=False)


def test_every_third_confirmation_accepts_replayed_n64_artifact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, promotion_count=2)

    plan = _execute(fixture, go=False)

    assert plan["promotion_count"] == 3
    assert plan["nth_confirmation_required"] is True


def test_every_third_confirmation_requires_an_immutable_artifact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, promotion_count=2)
    payload = json.loads(fixture["adjudication"].read_text())
    payload["nth_confirmation"] = None
    payload.pop("adjudication_sha256")
    payload["adjudication_sha256"] = promotion._digest_value(payload)
    _write_json(fixture["adjudication"], payload)

    with pytest.raises(promotion.PromotionError, match="immutable evidence"):
        _execute(fixture, go=False)


def test_every_third_confirmation_must_be_global_n64(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, promotion_count=2)
    payload = json.loads(fixture["adjudication"].read_text())
    confirmation = Path(payload["nth_confirmation"]["path"])
    source = json.loads(confirmation.read_text())
    source["typed_config"]["fields"]["candidate_n_full"] = 128
    digest = hashlib.sha256(
        promotion._canonical_bytes(source["typed_config"])
    ).hexdigest()
    source["config_hash"] = "sha256:" + digest[:16]
    source["full_config_hash"] = "sha256:" + digest
    _write_json(confirmation, source)
    payload["nth_confirmation"]["sha256"] = promotion._sha256(confirmation)
    payload.pop("adjudication_sha256")
    payload["adjudication_sha256"] = promotion._digest_value(payload)
    _write_json(fixture["adjudication"], payload)

    with pytest.raises(promotion.PromotionError, match="expected 64"):
        _execute(fixture, go=False)


def test_non_third_promotion_rejects_confirmation_artifact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = json.loads(fixture["adjudication"].read_text())
    internal = next(
        item for item in payload["evidence"] if item["kind"] == "internal_h2h"
    )
    payload["nth_confirmation"] = {
        "path": internal["path"],
        "sha256": internal["sha256"],
    }
    payload.pop("adjudication_sha256")
    payload["adjudication_sha256"] = promotion._digest_value(payload)
    _write_json(fixture["adjudication"], payload)

    with pytest.raises(promotion.PromotionError, match="must be null"):
        _execute(fixture, go=False)


def test_historical_v1_non_third_false_confirmation_remains_replayable(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    payload = json.loads(fixture["adjudication"].read_text())
    payload["schema_version"] = promotion.PREVIOUS_ADJUDICATION_SCHEMA
    payload.pop("nth_confirmation")
    payload["nth_confirmation_passed"] = False
    payload.pop("adjudication_sha256")
    payload["adjudication_sha256"] = promotion._digest_value(payload)
    _write_json(fixture["adjudication"], payload)

    assert _execute(fixture, go=False)["status"] == "dry_run"


def test_historical_v1_cannot_authorize_every_third_promotion(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, promotion_count=2)
    payload = json.loads(fixture["adjudication"].read_text())
    payload["schema_version"] = promotion.PREVIOUS_ADJUDICATION_SCHEMA
    payload.pop("nth_confirmation")
    payload["nth_confirmation_passed"] = True
    payload.pop("adjudication_sha256")
    payload["adjudication_sha256"] = promotion._digest_value(payload)
    _write_json(fixture["adjudication"], payload)

    with pytest.raises(promotion.PromotionError, match="cannot authorize"):
        _execute(fixture, go=False)


def test_every_third_confirmation_rejects_split_pair_seed_identity(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, promotion_count=2)
    payload = json.loads(fixture["adjudication"].read_text())
    confirmation = Path(payload["nth_confirmation"]["path"])
    source = json.loads(confirmation.read_text())
    source["games"][1]["game_seed"] += 1_000_000
    _write_json(confirmation, source)
    payload["nth_confirmation"]["sha256"] = promotion._sha256(confirmation)
    payload.pop("adjudication_sha256")
    payload["adjudication_sha256"] = promotion._digest_value(payload)
    _write_json(fixture["adjudication"], payload)

    with pytest.raises(promotion.PromotionError, match="different seeds"):
        _execute(fixture, go=False)


def test_symmetry_event_provenance_requires_explicit_on_recipe_and_report() -> None:
    recipe = {"symmetry_augment": True, "symmetry_augment_events": True}
    report = {"symmetry_augment": True, "symmetry_augment_events": True}

    promotion._verify_symmetry_training_provenance(  # noqa: SLF001
        report, recipe, where="test report"
    )
    for missing_from in ("recipe", "report"):
        bad_recipe = dict(recipe)
        bad_report = dict(report)
        (bad_recipe if missing_from == "recipe" else bad_report).pop(
            "symmetry_augment_events"
        )
        with pytest.raises(promotion.PromotionError, match="omits"):
            promotion._verify_symmetry_training_provenance(  # noqa: SLF001
                bad_report, bad_recipe, where="test report"
            )


def test_symmetry_event_provenance_rejects_mismatch_and_limits_legacy_compat() -> None:
    with pytest.raises(promotion.PromotionError, match="differs"):
        promotion._verify_symmetry_training_provenance(  # noqa: SLF001
            {"symmetry_augment": True, "symmetry_augment_events": False},
            {"symmetry_augment": True, "symmetry_augment_events": True},
            where="test report",
        )

    # Historical omission is accepted only when augmentation itself was off.
    promotion._verify_symmetry_training_provenance(  # noqa: SLF001
        {"symmetry_augment": False},
        {"symmetry_augment": False},
        where="historical report",
    )
    with pytest.raises(promotion.PromotionError, match="active"):
        promotion._verify_symmetry_training_provenance(  # noqa: SLF001
            {"symmetry_augment": False, "symmetry_augment_events": True},
            {"symmetry_augment": False},
            where="historical report",
        )


def test_exclusive_lock_refuses_a_second_writer(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    descriptor = os.open(fixture["lock"], os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(promotion.PromotionError, match="already held"):
            _execute(fixture, go=False)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_exclusive_lock_allows_same_thread_nested_replay(tmp_path: Path) -> None:
    lock = tmp_path / "promotion.lock"
    with promotion._exclusive_lock(lock):
        with promotion._exclusive_lock(lock):
            assert lock.is_file()


def test_exclusive_lock_reentrancy_is_shared_across_module_identities(
    tmp_path: Path,
) -> None:
    """The CLI is __main__ while handoff replay imports the tools module."""
    source = Path(promotion.__file__)
    spec = importlib.util.spec_from_file_location(
        "_promotion_transaction_duplicate_for_test", source
    )
    assert spec is not None and spec.loader is not None
    duplicate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = duplicate
    try:
        spec.loader.exec_module(duplicate)
    finally:
        sys.modules.pop(spec.name, None)

    lock = tmp_path / "promotion.lock"
    with promotion._exclusive_lock(lock):
        with duplicate._exclusive_lock(lock):
            assert lock.is_file()


def test_execute_dry_run_allows_verifier_to_reenter_registry_lock(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def verify(path: Path, *, require_all_job_claims: bool = False) -> dict:
        assert path == fixture["contract_path"]
        assert require_all_job_claims is True
        with promotion._exclusive_lock(fixture["lock"]):
            assert fixture["lock"].is_file()
        return fixture["contract"]

    result = promotion.execute_promotion(
        registry_path=fixture["registry"],
        current_pointer=fixture["pointer"],
        contract_lock=fixture["contract_path"],
        adjudication_path=fixture["adjudication"],
        training_receipt=fixture["training_receipt"],
        cohort_exclusions=fixture["cohort_exclusions"],
        receipt_path=fixture["receipt"],
        reason="A1 typed promotion",
        lock_path=fixture["lock"],
        go=False,
        verify_lock_fn=verify,
    )

    assert result["status"] == "dry_run"
    assert not fixture["receipt"].exists()


def test_execute_go_allows_verifier_to_reenter_registry_lock(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def verify(path: Path, *, require_all_job_claims: bool = False) -> dict:
        assert path == fixture["contract_path"]
        assert require_all_job_claims is True
        with promotion._exclusive_lock(fixture["lock"]):
            pass
        return fixture["contract"]

    result = promotion.execute_promotion(
        registry_path=fixture["registry"],
        current_pointer=fixture["pointer"],
        contract_lock=fixture["contract_path"],
        adjudication_path=fixture["adjudication"],
        training_receipt=fixture["training_receipt"],
        cohort_exclusions=fixture["cohort_exclusions"],
        receipt_path=fixture["receipt"],
        reason="A1 typed promotion",
        lock_path=fixture["lock"],
        go=True,
        verify_lock_fn=verify,
    )

    assert result["status"] == "committed"
    assert fixture["receipt"].is_file()


def test_exclusive_lock_nested_exception_does_not_leak_depth(tmp_path: Path) -> None:
    lock = tmp_path / "promotion.lock"
    with promotion._exclusive_lock(lock):
        with pytest.raises(RuntimeError, match="nested failure"):
            with promotion._exclusive_lock(lock):
                raise RuntimeError("nested failure")
        result: list[str] = []

        def contend() -> None:
            try:
                with promotion._exclusive_lock(lock):
                    result.append("acquired")
            except promotion.PromotionError as error:
                result.append(str(error))

        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(result) == 1
        assert "already held" in result[0]
        with promotion._exclusive_lock(lock):
            pass
    with promotion._exclusive_lock(lock):
        pass


def test_exclusive_lock_still_refuses_competing_thread(tmp_path: Path) -> None:
    lock = tmp_path / "promotion.lock"
    result: list[str] = []

    def contend() -> None:
        try:
            with promotion._exclusive_lock(lock):
                result.append("acquired")
        except promotion.PromotionError as error:
            result.append(str(error))

    with promotion._exclusive_lock(lock):
        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(result) == 1
    assert "already held" in result[0]


def test_nested_exclusive_lock_revalidates_named_inode(tmp_path: Path) -> None:
    lock = tmp_path / "promotion.lock"
    with pytest.raises(promotion.PromotionError, match="identity drifted"):
        with promotion._exclusive_lock(lock):
            with promotion._exclusive_lock(lock):
                replacement = tmp_path / "replacement.lock"
                replacement.write_text("", encoding="utf-8")
                os.replace(replacement, lock)


def test_exclusive_lock_rejects_symlink_and_revalidates_named_inode(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.lock"
    target.write_text("", encoding="utf-8")
    symlink = tmp_path / "symlink.lock"
    symlink.symlink_to(target)
    with pytest.raises(OSError):
        with promotion._exclusive_lock(symlink):
            pass

    lock = tmp_path / "promotion.lock"
    with pytest.raises(promotion.PromotionError, match="identity drifted"):
        with promotion._exclusive_lock(lock):
            replacement = tmp_path / "replacement.lock"
            replacement.write_text("", encoding="utf-8")
            os.replace(replacement, lock)


def test_alternate_lock_path_is_forbidden(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(promotion.PromotionError, match="alternate promotion lock"):
        promotion.execute_promotion(
            registry_path=fixture["registry"],
            current_pointer=fixture["pointer"],
            contract_lock=fixture["contract_path"],
            adjudication_path=fixture["adjudication"],
            training_receipt=fixture["training_receipt"],
            cohort_exclusions=fixture["cohort_exclusions"],
            receipt_path=fixture["receipt"],
            reason="A1 typed promotion",
            lock_path=tmp_path / "bypass.lock",
            go=False,
            verify_lock_fn=_verify(fixture),
        )


def test_symlink_registry_is_rejected_before_lock_or_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    alias = tmp_path / "registry.alias.json"
    alias.symlink_to(fixture["registry"])

    with pytest.raises(promotion.PromotionError, match="must not contain symlinks"):
        promotion.execute_promotion(
            registry_path=alias,
            current_pointer=fixture["pointer"],
            contract_lock=fixture["contract_path"],
            adjudication_path=fixture["adjudication"],
            training_receipt=fixture["training_receipt"],
            cohort_exclusions=fixture["cohort_exclusions"],
            receipt_path=fixture["receipt"],
            reason="A1 typed promotion",
            go=False,
            verify_lock_fn=_verify(fixture),
        )


def test_failed_second_replace_rolls_registry_and_pointer_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    registry_before = fixture["registry"].read_bytes()
    pointer_before = fixture["pointer"].read_bytes()
    real_write = promotion._atomic_write_bytes
    failed = False

    def fail_once(path: Path, data: bytes) -> None:
        nonlocal failed
        if path == fixture["pointer"] and not failed and data != pointer_before:
            failed = True
            raise OSError("synthetic pointer replace failure")
        real_write(path, data)

    monkeypatch.setattr(promotion, "_atomic_write_bytes", fail_once)
    with pytest.raises(promotion.PromotionError, match="original.*restored"):
        _execute(fixture, go=True)

    assert fixture["registry"].read_bytes() == registry_before
    assert fixture["pointer"].read_bytes() == pointer_before
    assert json.loads(fixture["receipt"].read_text())["status"] == "rolled_back"


def test_recovery_refuses_tampered_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _execute(fixture, go=True)
    receipt = json.loads(fixture["receipt"].read_text())
    receipt["reason"] = "tampered"
    _write_json(fixture["receipt"], receipt)

    with pytest.raises(promotion.PromotionError, match="semantic digest mismatch"):
        promotion.recover_transaction(receipt_path=fixture["receipt"], go=True)


def test_failed_recovery_restores_pre_recovery_committed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _execute(fixture, go=True)
    committed_registry = fixture["registry"].read_bytes()
    committed_pointer = fixture["pointer"].read_bytes()
    before_pointer = Path(str(fixture["receipt"]) + ".current.before").read_bytes()
    real_write = promotion._atomic_write_bytes
    failed = False

    def fail_once(path: Path, data: bytes) -> None:
        nonlocal failed
        if path == fixture["pointer"] and data == before_pointer and not failed:
            failed = True
            raise OSError("synthetic recovery pointer failure")
        real_write(path, data)

    monkeypatch.setattr(promotion, "_atomic_write_bytes", fail_once)
    with pytest.raises(promotion.PromotionError, match="pre-recovery.*restored"):
        promotion.recover_transaction(receipt_path=fixture["receipt"], go=True)

    assert fixture["registry"].read_bytes() == committed_registry
    assert fixture["pointer"].read_bytes() == committed_pointer
    assert json.loads(fixture["receipt"].read_text())["status"] == "committed"


def _legacy_snapshot_for_fixture(fixture: dict, tmp_path: Path):
    source = tmp_path / "legacy-source.json"
    attestation = tmp_path / "legacy-attestation.json"
    _write_json(source, {"source": "pinned"})
    _write_json(attestation, {"attestation": "pinned"})
    return promotion._LegacyPromotionSnapshot(  # noqa: SLF001
        contract_lock=promotion._stable_json_snapshot(  # noqa: SLF001
            fixture["contract_path"], where="test contract"
        ),
        source_draft=promotion._stable_json_snapshot(  # noqa: SLF001
            source, where="test source"
        ),
        training_receipt=promotion._stable_json_snapshot(  # noqa: SLF001
            fixture["training_receipt"], where="test receipt"
        ),
        attestation=promotion._stable_json_snapshot(  # noqa: SLF001
            attestation, where="test attestation"
        ),
    )


def test_legacy_receipt_replacement_during_full_validation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    snapshot = _legacy_snapshot_for_fixture(fixture, tmp_path)
    monkeypatch.setattr(
        promotion,
        "_verify_contract_with_snapshot",
        lambda *_args, **_kwargs: (fixture["contract"], snapshot),
    )
    original = promotion._verify_one_dose_training_receipt  # noqa: SLF001
    replaced = False

    def replace_then_validate(path: Path, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            replacement = tmp_path / "replacement-receipt.json"
            replacement.write_bytes(Path(path).read_bytes())
            replacement.replace(path)
        return original(path, **kwargs)

    monkeypatch.setattr(
        promotion, "_verify_one_dose_training_receipt", replace_then_validate
    )
    with pytest.raises(
        promotion.PromotionError, match="historical training receipt pathname changed"
    ):
        _execute(fixture, go=False)


def test_legacy_attestation_replacement_before_plan_construction_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    snapshot = _legacy_snapshot_for_fixture(fixture, tmp_path)
    monkeypatch.setattr(
        promotion,
        "_verify_contract_with_snapshot",
        lambda *_args, **_kwargs: (fixture["contract"], snapshot),
    )
    original = promotion._verify_adjudication  # noqa: SLF001

    def replace_after_adjudication(*args, **kwargs):
        result = original(*args, **kwargs)
        assert snapshot.attestation is not None
        replacement = tmp_path / "replacement-attestation.json"
        replacement.write_bytes(snapshot.attestation.path.read_bytes())
        replacement.replace(snapshot.attestation.path)
        return result

    monkeypatch.setattr(promotion, "_verify_adjudication", replace_after_adjudication)
    with pytest.raises(
        promotion.PromotionError, match="legacy contract attestation pathname changed"
    ):
        _execute(fixture, go=False)


def test_legacy_snapshot_is_revalidated_at_commit_mutation_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    snapshot = _legacy_snapshot_for_fixture(fixture, tmp_path)
    monkeypatch.setattr(
        promotion,
        "_verify_contract_with_snapshot",
        lambda *_args, **_kwargs: (fixture["contract"], snapshot),
    )
    original_prepare = promotion.prepare_promotion

    def replace_after_prepare(**kwargs):
        plan = original_prepare(**kwargs)
        assert snapshot.attestation is not None
        replacement = tmp_path / "replacement-at-mutation-boundary.json"
        replacement.write_bytes(snapshot.attestation.data)
        replacement.replace(snapshot.attestation.path)
        return plan

    monkeypatch.setattr(promotion, "prepare_promotion", replace_after_prepare)
    registry_before = fixture["registry"].read_bytes()
    pointer_before = fixture["pointer"].read_bytes()
    with pytest.raises(
        promotion.PromotionError, match="legacy contract attestation pathname changed"
    ):
        _execute(fixture, go=True)
    assert fixture["registry"].read_bytes() == registry_before
    assert fixture["pointer"].read_bytes() == pointer_before
    assert not fixture["receipt"].exists()
    assert not fixture["receipt"].with_name(
        fixture["receipt"].name + ".registry.before"
    ).exists()


def _production_l1_receipt_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    f7 = tmp_path / "f7.pt"
    f7.write_bytes(b"f7")
    report = tmp_path / "train.report.json"
    _write_json(
        report,
        {
            "steps_completed": 1024,
            "world_size": 8,
            "batch_size": 512,
            "init_checkpoint_sha256": promotion._sha256(f7),
        },
    )
    optimizer = tmp_path / "candidate.pt.optimizer.pt"
    optimizer.write_bytes(b"optimizer")
    progress_path = tmp_path / "candidate.pt.training-progress.json"
    progress = {
        "checkpoint": {"path": candidate.name, "sha256": promotion._sha256(candidate)},
        "optimizer": {"path": optimizer.name, "sha256": promotion._sha256(optimizer)},
        "optimizer_step": 1024,
        "completed_epochs": 1,
        "rank_torch_rng_states": [{} for _ in range(8)],
    }
    progress["progress_sha256"] = promotion._digest_value(progress)
    _write_json(progress_path, progress)
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, {"sealed": True})
    manifest_ref = {"path": str(manifest_path), "sha256": promotion._sha256(manifest_path)}
    command_sha = "sha256:" + "1" * 64
    manifest = {
        "selected_dose": {
            "optimizer_steps": 1024, "world_size": 8,
            "per_rank_batch_size": 512, "global_samples": 4_194_304,
            "policy_aux_active_batch_size": 0,
        },
        "f7_parent": {"path": str(f7), "sha256": promotion._sha256(f7)},
        "command_sha256": command_sha,
    }
    monkeypatch.setattr(
        production_l1, "verify",
        lambda path: {"manifest": manifest, "manifest_ref": manifest_ref},
    )
    claim_path = tmp_path / "execution.claim.json"
    claim = {
        "schema_version": production_l1.CLAIM_SCHEMA,
        "created_at_unix_ns": 1,
        "manifest": manifest_ref,
        "unit": "production-l1",
    }
    claim["claim_sha256"] = promotion._digest_value(claim)
    _write_json(claim_path, claim)
    submission_path = tmp_path / "submission.receipt.json"
    submission = {
        "schema_version": production_l1.SUBMISSION_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "manifest": manifest_ref,
        "claim": {"path": str(claim_path), "sha256": promotion._sha256(claim_path)},
        "command_sha256": command_sha,
        "unit": "production-l1",
    }
    submission["receipt_sha256"] = promotion._digest_value(submission)
    _write_json(submission_path, submission)
    completion_path = tmp_path / "completion.receipt.json"
    completion = {
        "schema_version": production_l1.COMPLETION_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "created_at_unix_ns": 2,
        "manifest": manifest_ref,
        "submission": {
            "path": str(submission_path), "sha256": promotion._sha256(submission_path),
        },
        "checkpoint": {
            "path": str(candidate), "sha256": promotion._sha256(candidate),
        },
        "report": {"path": str(report), "sha256": promotion._sha256(report)},
        "unit_state": {"ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"},
    }
    completion["receipt_sha256"] = promotion._digest_value(completion)
    _write_json(completion_path, completion)
    contract = {
        "checkpoints": [
            {"role": "producer", "path": str(f7), "sha256": promotion._sha256(f7)}
        ]
    }
    return {
        "candidate": candidate, "f7": f7, "report": report,
        "optimizer": optimizer, "progress": progress_path,
        "completion": completion_path, "contract": contract,
    }


def test_production_l1_completion_replays_eight_rank_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _production_l1_receipt_fixture(tmp_path, monkeypatch)

    verified = promotion._verify_production_l1_completion_receipt(
        fixture["completion"], contract=fixture["contract"],
        candidate_path=fixture["candidate"],
        candidate_sha256=promotion._sha256(fixture["candidate"]),
        training_report_path=fixture["report"],
        training_report_sha256=promotion._sha256(fixture["report"]),
    )

    assert verified["receipt_sha256"].startswith("sha256:")
    assert verified["execution_binding_sha256"].startswith("sha256:")


def test_production_l1_completion_refuses_optimizer_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _production_l1_receipt_fixture(tmp_path, monkeypatch)
    fixture["optimizer"].write_bytes(b"replaced")

    with pytest.raises(promotion.PromotionError, match="optimizer sidecar artifact drift"):
        promotion._verify_production_l1_completion_receipt(
            fixture["completion"], contract=fixture["contract"],
            candidate_path=fixture["candidate"],
            candidate_sha256=promotion._sha256(fixture["candidate"]),
            training_report_path=fixture["report"],
            training_report_sha256=promotion._sha256(fixture["report"]),
        )


def _production_gather_receipt_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy_aux_active_batch_size: int = 0,
):
    import torch

    source = tmp_path / "r3.pt"
    torch.save({"model": {"shared.weight": torch.tensor([1.0, 2.0])}}, source)
    init = tmp_path / "r3-gather-init.pt"
    new_parameters = [
        "target_gather_proj.0.bias",
        "target_gather_proj.0.weight",
        "target_gather_proj.1.bias",
        "target_gather_proj.1.weight",
    ]
    init_model = {"shared.weight": torch.tensor([1.0, 2.0])}
    init_model.update({key: torch.zeros(1) for key in new_parameters})
    torch.save({"model": init_model}, init)
    candidate = tmp_path / "candidate.pt"
    candidate_model = {"shared.weight": torch.tensor([1.0, 2.0])}
    candidate_model.update({key: torch.ones(1) for key in new_parameters})
    torch.save({"model": candidate_model}, candidate)
    f7 = tmp_path / "f7.pt"
    f7.write_bytes(b"f7 corpus producer")
    upgrade_receipt = tmp_path / "architecture-upgrade.json"
    _write_json(upgrade_receipt, {"sealed": True})
    upgrade = {
        "schema_version": "a1-function-preserving-architecture-upgrade-v1",
        "module": "entity_graph.action_target_gather.v1",
        "source": {"path": str(source), "sha256": promotion._sha256(source)},
        "upgraded_initializer": {
            "path": str(init), "sha256": promotion._sha256(init),
        },
        "new_parameters": new_parameters,
        "receipt_sha256": "sha256:" + "a" * 64,
        "receipt": {
            "path": str(upgrade_receipt),
            "sha256": promotion._sha256(upgrade_receipt),
        },
    }
    monkeypatch.setattr(
        promotion.one_dose.architecture_upgrade,
        "verify_receipt",
        lambda path: upgrade,
    )
    report = tmp_path / "train.report.json"
    report_payload = {
            "arch": "entity_graph", "mask_hidden_info": True,
            "track": "2p_no_trade", "vps_to_win": 10,
            "world_size": 4, "batch_size": 512,
            "effective_global_batch_size": 2048, "epochs": 1,
            "max_steps": 2048, "steps_completed": 2048,
            "training_row_draws": 4_194_304, "optimizer": "adam",
            "resume_optimizer": False, "optimizer_restored": False,
            "soft_target_weight": 0.9, "value_loss_weight": 0.25,
            "loser_sample_weight": 1.0,
            "policy_aux_active_batch_size": policy_aux_active_batch_size,
            "action_module_lr_mult": 4.0, "value_lr_mult": 1.0,
            "freeze_modules": "trunk,action_encoder,policy_head,value_heads",
            "require_only_trainable_prefixes": "target_gather_proj",
            "action_target_gather": True, "ddp_find_unused_parameters": True,
            "ddp_shard_data": False, "value_target_lambda": 1.0,
            "forced_action_weight": 0.0, "forced_row_value_weight": 1.0,
            "winner_sample_weight": 1.0, "lr_schedule": "flat",
            "lr_warmup_steps": 100, "weight_decay": 0.0, "seed": 1,
            "max_grad_norm": 1.0, "gradient_clipping_enabled": True,
            "checkpoint": str(candidate), "init_checkpoint": str(init),
            "init_checkpoint_sha256": promotion._sha256(init),
        }
    if policy_aux_active_batch_size:
        report_payload.update(
            {
                "policy_aux_training_row_draws": 524_288,
                "total_training_row_draws": 4_718_592,
            }
        )
    _write_json(report, report_payload)
    optimizer = tmp_path / "candidate.pt.optimizer.pt"
    optimizer.write_bytes(b"fresh optimizer")
    progress_path = tmp_path / "candidate.pt.training-progress.json"
    progress = {
        "checkpoint": {"path": str(candidate), "sha256": promotion._sha256(candidate)},
        "optimizer": {"path": str(optimizer), "sha256": promotion._sha256(optimizer)},
        "optimizer_step": 2048,
        "completed_epochs": 1,
        "rank_torch_rng_states": [{} for _ in range(4)],
    }
    progress["progress_sha256"] = promotion._digest_value(progress)
    _write_json(progress_path, progress)
    manifest_path = tmp_path / "gather.manifest.json"
    _write_json(manifest_path, {"sealed": True})
    manifest_ref = {"path": str(manifest_path), "sha256": promotion._sha256(manifest_path)}
    operator = {
        "world_size": 4, "per_rank_batch_size": 512,
        "optimizer_steps": 2048, "global_base_draws": 4_194_304,
        "current_fraction": 0.8, "current_n128_fraction": 5.0 / 7.0,
        "current_n256_fraction": 2.0 / 7.0,
        "exact_predecessor_replay_fraction": 0.2,
        "soft_target_weight": 0.9, "value_loss_weight": 0.25,
        "loser_sample_weight": 1.0, "action_module_lr_mult": 4.0,
        "freeze_modules": ["trunk", "action_encoder", "policy_head", "value_heads"],
        "required_trainable_prefixes": ["target_gather_proj"],
        "fresh_optimizer": True,
        "ddp_find_unused_parameters": True,
    }
    if policy_aux_active_batch_size:
        operator.update(
            {
                "policy_aux_active_batch_size_per_rank":
                    policy_aux_active_batch_size,
                "global_policy_aux_active_draws": 524_288,
            }
        )
    manifest = {
        "operator": operator,
        "operator_sha256": promotion._digest_value(operator),
        "command_sha256": "sha256:" + "b" * 64,
        "runtime_python": {"lexical_path": "/usr/bin/python3"},
        "repo_binding": {"public_main_commit": "a" * 40},
        "learner_source_incumbent": upgrade["source"],
        "corpus_producer": {"path": str(f7), "sha256": promotion._sha256(f7)},
        "function_preserving_upgrade": upgrade,
        "visible_devices": list(
            production_gather.AUX64_VISIBLE_DEVICES
            if policy_aux_active_batch_size
            else production_gather.DEFAULT_VISIBLE_DEVICES
        ),
    }
    verified_manifest = {
        "manifest": manifest, "manifest_ref": manifest_ref,
        "repo": tmp_path, "command": ["/usr/bin/python3", "train.py"],
        "output_root": tmp_path,
    }
    monkeypatch.setattr(
        production_gather,
        "verify",
        lambda path: verified_manifest,
    )
    claim_path = tmp_path / "execution.claim.json"
    claim = {
        "schema_version": production_gather.CLAIM_SCHEMA,
        "created_at_unix_ns": 1, "manifest": manifest_ref, "unit": "gather-unit",
    }
    claim["claim_sha256"] = promotion._digest_value(claim)
    _write_json(claim_path, claim)
    submission_path = tmp_path / "submission.receipt.json"
    submission = {
        "schema_version": production_gather.SUBMISSION_SCHEMA,
        "diagnostic_only": False, "production_eligible": True,
        "created_at_unix_ns": 2, "manifest": manifest_ref,
        "claim": {"path": str(claim_path), "sha256": promotion._sha256(claim_path)},
        "unit": "gather-unit", "command_sha256": manifest["command_sha256"],
        "systemd_command_sha256": promotion._digest_value(
            production_gather._systemd_command(verified_manifest, "gather-unit")  # noqa: SLF001
        ),
        "execution_binding": production_gather._execution_binding(  # noqa: SLF001
            verified_manifest
        ),
        "systemd_stdout": "submitted",
    }
    submission["execution_binding_sha256"] = promotion._digest_value(
        submission["execution_binding"]
    )
    submission["receipt_sha256"] = promotion._digest_value(submission)
    _write_json(submission_path, submission)
    completion_path = tmp_path / "completion.receipt.json"
    completion = {
        "schema_version": production_gather.COMPLETION_SCHEMA,
        "diagnostic_only": False, "production_eligible": True,
        "created_at_unix_ns": 3, "manifest": manifest_ref,
        "submission": {"path": str(submission_path), "sha256": promotion._sha256(submission_path)},
        "checkpoint": {"path": str(candidate), "sha256": promotion._sha256(candidate)},
        "report": {"path": str(report), "sha256": promotion._sha256(report)},
        "operator_sha256": manifest["operator_sha256"],
        "progress": {"path": str(progress_path), "sha256": promotion._sha256(progress_path)},
        "optimizer": {"path": str(optimizer), "sha256": promotion._sha256(optimizer)},
        "model_delta": production_gather._verify_adapter_only_model_delta(  # noqa: SLF001
            init, candidate
        ),
        "unit_state": {"ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"},
    }
    completion["receipt_sha256"] = promotion._digest_value(completion)
    _write_json(completion_path, completion)
    return {
        "source": source, "init": init, "candidate": candidate, "f7": f7,
        "report": report, "progress": progress_path, "optimizer": optimizer,
        "completion": completion_path, "manifest": manifest,
        "contract": {"checkpoints": [{"role": "producer", **manifest["corpus_producer"]}]},
    }


def test_production_target_gather_completion_binds_r3_not_corpus_f7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _production_gather_receipt_fixture(tmp_path, monkeypatch)

    verified = promotion._verify_production_target_gather_completion_receipt(
        fixture["completion"], contract=fixture["contract"],
        candidate_path=fixture["candidate"],
        candidate_sha256=promotion._sha256(fixture["candidate"]),
        training_report_path=fixture["report"],
        training_report_sha256=promotion._sha256(fixture["report"]),
    )

    assert verified["evaluation_parent_sha256"] == promotion._sha256(fixture["source"])
    assert verified["evaluation_parent_sha256"] != promotion._sha256(fixture["f7"])


def test_production_target_gather_completion_accepts_sealed_aux64_dose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _production_gather_receipt_fixture(
        tmp_path, monkeypatch, policy_aux_active_batch_size=64
    )
    verified = promotion._verify_production_target_gather_completion_receipt(
        fixture["completion"],
        contract=fixture["contract"],
        candidate_path=fixture["candidate"],
        candidate_sha256=promotion._sha256(fixture["candidate"]),
        training_report_path=fixture["report"],
        training_report_sha256=promotion._sha256(fixture["report"]),
    )
    assert verified["evaluation_parent_sha256"] == promotion._sha256(
        fixture["source"]
    )


def test_production_target_gather_completion_refuses_inherited_tensor_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _production_gather_receipt_fixture(tmp_path, monkeypatch)
    import torch

    raw = torch.load(fixture["candidate"], map_location="cpu", weights_only=False)
    raw["model"]["shared.weight"] = torch.tensor([9.0, 2.0])
    torch.save(raw, fixture["candidate"])
    completion = json.loads(fixture["completion"].read_text())
    completion["checkpoint"]["sha256"] = promotion._sha256(fixture["candidate"])
    completion["receipt_sha256"] = promotion._digest_value(
        {key: value for key, value in completion.items() if key != "receipt_sha256"}
    )
    _write_json(fixture["completion"], completion)
    progress = json.loads(fixture["progress"].read_text())
    progress["checkpoint"]["sha256"] = promotion._sha256(fixture["candidate"])
    progress["progress_sha256"] = promotion._digest_value(
        {key: value for key, value in progress.items() if key != "progress_sha256"}
    )
    _write_json(fixture["progress"], progress)
    completion["progress"]["sha256"] = promotion._sha256(fixture["progress"])
    completion["receipt_sha256"] = promotion._digest_value(
        {key: value for key, value in completion.items() if key != "receipt_sha256"}
    )
    _write_json(fixture["completion"], completion)

    with pytest.raises(promotion.PromotionError, match="model delta refused"):
        promotion._verify_production_target_gather_completion_receipt(
            fixture["completion"], contract=fixture["contract"],
            candidate_path=fixture["candidate"],
            candidate_sha256=promotion._sha256(fixture["candidate"]),
            training_report_path=fixture["report"],
            training_report_sha256=promotion._sha256(fixture["report"]),
        )


def test_production_target_gather_completion_requires_four_rank_rng_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _production_gather_receipt_fixture(tmp_path, monkeypatch)
    progress = json.loads(fixture["progress"].read_text())
    progress["rank_torch_rng_states"].append({})
    progress["progress_sha256"] = promotion._digest_value(
        {key: value for key, value in progress.items() if key != "progress_sha256"}
    )
    _write_json(fixture["progress"], progress)
    completion = json.loads(fixture["completion"].read_text())
    completion["progress"]["sha256"] = promotion._sha256(fixture["progress"])
    completion["receipt_sha256"] = promotion._digest_value(
        {key: value for key, value in completion.items() if key != "receipt_sha256"}
    )
    _write_json(fixture["completion"], completion)

    with pytest.raises(promotion.PromotionError, match="progress/dose topology"):
        promotion._verify_production_target_gather_completion_receipt(
            fixture["completion"], contract=fixture["contract"],
            candidate_path=fixture["candidate"],
            candidate_sha256=promotion._sha256(fixture["candidate"]),
            training_report_path=fixture["report"],
            training_report_sha256=promotion._sha256(fixture["report"]),
        )


def test_production_target_gather_report_requires_exact_four_rank_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _production_gather_receipt_fixture(tmp_path, monkeypatch)
    contract = {
        **fixture["contract"],
        "science": {"learner_training_recipe": {"symmetry_augment": False}},
    }
    replay = promotion._verify_training_report(
        fixture["report"], contract=contract,
        contract_sha256="sha256:" + "0" * 64,
        candidate_path=fixture["candidate"],
        candidate_sha256=promotion._sha256(fixture["candidate"]),
        production_target_gather_completion=True,
    )
    assert replay["training_row_draws"] == 4_194_304

    report = json.loads(fixture["report"].read_text())
    report["ddp_find_unused_parameters"] = False
    _write_json(fixture["report"], report)
    with pytest.raises(promotion.PromotionError, match="ddp_find_unused_parameters"):
        promotion._verify_training_report(
            fixture["report"], contract=contract,
            contract_sha256="sha256:" + "0" * 64,
            candidate_path=fixture["candidate"],
            candidate_sha256=promotion._sha256(fixture["candidate"]),
            production_target_gather_completion=True,
        )


def test_production_target_gather_completion_refuses_wrong_corpus_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _production_gather_receipt_fixture(tmp_path, monkeypatch)
    other = tmp_path / "other-producer.pt"
    other.write_bytes(b"other")
    fixture["contract"]["checkpoints"][0] = {
        "role": "producer", "path": str(other), "sha256": promotion._sha256(other)
    }
    with pytest.raises(promotion.PromotionError, match="corpus producer differs"):
        promotion._verify_production_target_gather_completion_receipt(
            fixture["completion"], contract=fixture["contract"],
            candidate_path=fixture["candidate"],
            candidate_sha256=promotion._sha256(fixture["candidate"]),
            training_report_path=fixture["report"],
            training_report_sha256=promotion._sha256(fixture["report"]),
        )


def test_production_target_gather_completion_replays_execution_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _production_gather_receipt_fixture(tmp_path, monkeypatch)
    completion = json.loads(fixture["completion"].read_text())
    submission_path = Path(completion["submission"]["path"])
    submission = json.loads(submission_path.read_text())
    submission["execution_binding"]["environment"]["CUDA_VISIBLE_DEVICES"] = "0,1"
    submission["execution_binding_sha256"] = promotion._digest_value(
        submission["execution_binding"]
    )
    submission["receipt_sha256"] = promotion._digest_value(
        {key: value for key, value in submission.items() if key != "receipt_sha256"}
    )
    _write_json(submission_path, submission)
    completion["submission"]["sha256"] = promotion._sha256(submission_path)
    completion["receipt_sha256"] = promotion._digest_value(
        {key: value for key, value in completion.items() if key != "receipt_sha256"}
    )
    _write_json(fixture["completion"], completion)

    with pytest.raises(promotion.PromotionError, match="submission drifted"):
        promotion._verify_production_target_gather_completion_receipt(
            fixture["completion"], contract=fixture["contract"],
            candidate_path=fixture["candidate"],
            candidate_sha256=promotion._sha256(fixture["candidate"]),
            training_report_path=fixture["report"],
            training_report_sha256=promotion._sha256(fixture["report"]),
        )


def test_production_l1_report_requires_exact_selected_recipe(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    f7 = tmp_path / "f7.pt"
    f7.write_bytes(b"f7")
    report = {
        "arch": "entity_graph", "mask_hidden_info": True, "track": "2p_no_trade",
        "vps_to_win": 10, "world_size": 8, "batch_size": 512, "epochs": 1,
        "max_steps": 1024, "steps_completed": 1024, "optimizer": "adam",
        "resume_optimizer": False, "optimizer_restored": False,
        "loser_sample_weight": 1.0, "soft_target_weight": 0.9,
        "policy_aux_active_batch_size": 0, "max_grad_norm": 1.0,
        "gradient_clipping_enabled": True, "checkpoint": str(candidate),
        "init_checkpoint": str(f7), "init_checkpoint_sha256": promotion._sha256(f7),
    }
    report_path = tmp_path / "report.json"
    _write_json(report_path, report)
    contract = {"checkpoints": [{"role": "producer", "sha256": promotion._sha256(f7)}]}

    replay = promotion._verify_training_report(
        report_path, contract=contract, contract_sha256="sha256:" + "0" * 64,
        candidate_path=candidate, candidate_sha256=promotion._sha256(candidate),
        production_l1_completion=True,
    )
    assert replay["steps_completed"] == 1024

    report["loser_sample_weight"] = 0.3
    _write_json(report_path, report)
    with pytest.raises(promotion.PromotionError, match="loser_sample_weight"):
        promotion._verify_training_report(
            report_path, contract=contract, contract_sha256="sha256:" + "0" * 64,
            candidate_path=candidate, candidate_sha256=promotion._sha256(candidate),
            production_l1_completion=True,
        )


def test_contract_value_readout_accepts_bootstrap_and_post_promotion_shapes() -> None:
    assert promotion._contract_value_readout(
        {"science": {"learner_value_objective": {"value_readout": "scalar"}}}
    ) == "scalar"
    assert promotion._contract_value_readout(
        {"science": {"value_readout": "scalar"}}
    ) == "scalar"


def test_contract_value_readout_rejects_ambiguous_or_missing_binding() -> None:
    with pytest.raises(promotion.PromotionError, match="representations disagree"):
        promotion._contract_value_readout(
            {
                "science": {
                    "value_readout": "scalar",
                    "learner_value_objective": {"value_readout": "categorical"},
                }
            }
        )
    with pytest.raises(promotion.PromotionError, match="unsupported value_readout"):
        promotion._contract_value_readout({"science": {}})


def test_role_search_config_accepts_only_complete_native_runtime_binding() -> None:
    expected = {"n_full": 128, "c_scale": 0.1, "evaluator_rust_featurize": False}
    raw = {
        **expected,
        "evaluator_rust_featurize": True,
        "native_mcts_hot_loop": True,
        "mcts_implementation": "rust_native_hot_loop_v1",
    }
    assert promotion._verify_role_search_config(
        raw, expected_search_config=expected, where="panel"
    ) == expected

    for drifted in (
        {**expected, "native_mcts_hot_loop": True},
        {**expected, "evaluator_rust_featurize": True},
        {
            **expected,
            "native_mcts_hot_loop": False,
            "mcts_implementation": "rust_native_hot_loop_v1",
        },
        {
            **expected,
            "native_mcts_hot_loop": True,
            "mcts_implementation": "python",
        },
    ):
        with pytest.raises(
            promotion.PromotionError,
            match="native MCTS runtime binding|sealed A1 semantic drift",
        ):
            promotion._verify_role_search_config(
                drifted, expected_search_config=expected, where="panel"
            )


def _typed_final_training_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected_aux: str,
) -> tuple[Path, Path, dict]:
    raw = tmp_path / f"{selected_aux}-raw.pt"
    transitioned = tmp_path / f"{selected_aux}-transitioned.pt"
    pointer = tmp_path / f"{selected_aux}-pointer.pt"
    warmed = tmp_path / f"{selected_aux}-warmed.pt"
    candidate = tmp_path / f"{selected_aux}-candidate.pt"
    for path, payload in (
        (raw, b"raw producer"),
        (transitioned, b"public award transitioned"),
        (pointer, b"pointer initializer"),
        (warmed, b"warmed pointer initializer"),
        (candidate, b"trained final candidate"),
    ):
        path.write_bytes(payload)
    transition_receipt = tmp_path / f"{selected_aux}-transition.json"
    transition_receipt.write_text("{}", encoding="utf-8")
    transition_state = "sha256:" + "a" * 64
    transition_evidence = {
        "state_sha256": transition_state,
        "source_checkpoint_sha256": promotion._sha256(raw),
        "transitioned_checkpoint_sha256": promotion._sha256(transitioned),
        "optimizer_steps": 0,
        "legacy_zero_input_function_preserving": True,
    }
    transition_row = one_dose._initializer_transition_record(  # noqa: SLF001
        kind="public_award_zero_initialization",
        role="feature_schema_zero_initialization",
        source_checkpoint_sha256=promotion._sha256(raw),
        output_checkpoint_sha256=promotion._sha256(transitioned),
        sampled_rows=0,
        optimizer_steps=0,
        optimizer_state_terminal="not_constructed",
        receipt_path=str(transition_receipt),
        receipt_file_sha256=promotion._sha256(transition_receipt),
        receipt_state_sha256=transition_state,
    )
    chain = [transition_row]
    pointer_receipt = tmp_path / f"{selected_aux}-pointer.json"
    pointer_receipt.write_text("{}", encoding="utf-8")
    pointer_state = "sha256:" + "b" * 64
    pointer_evidence = {
        "module": one_dose.AUX_REGULARIZATION_MODULE,
        "source": {
            "path": str(transitioned),
            "sha256": promotion._sha256(transitioned),
        },
        "upgraded_initializer": {
            "path": str(pointer),
            "sha256": promotion._sha256(pointer),
        },
        "receipt": {
            "path": str(pointer_receipt),
            "sha256": promotion._sha256(pointer_receipt),
        },
        "receipt_sha256": pointer_state,
    }
    warmup_path = tmp_path / f"{selected_aux}-warmup-terminal.json"
    if selected_aux == "AUXT":
        pointer_row = one_dose._initializer_transition_record(  # noqa: SLF001
            kind="function_preserving_pointer_upgrade",
            role="architecture_zero_diff_upgrade",
            source_checkpoint_sha256=promotion._sha256(transitioned),
            output_checkpoint_sha256=promotion._sha256(pointer),
            sampled_rows=0,
            optimizer_steps=0,
            optimizer_state_terminal="not_constructed",
            receipt_path=str(pointer_receipt),
            receipt_file_sha256=promotion._sha256(pointer_receipt),
            receipt_state_sha256=pointer_state,
        )
        warmup = {
            "schema_version": "a1-aux-pointer-warmup-terminal-v1",
            "result": {
                "status": "complete",
                "input_initializer_sha256": promotion._sha256(pointer),
                "warmed_checkpoint_sha256": promotion._sha256(warmed),
                "sampled_rows": 524_288,
                "optimizer_steps": 128,
                "optimizer_sidecar_discarded_for_joint": True,
                "inherited_parameters_bit_identical": True,
                "main_output_max_diff": 0.0,
            },
        }
        warmup["state_sha256"] = promotion._digest_value(warmup)
        _write_json(warmup_path, warmup)
        warmup_path.chmod(0o444)
        warmup_row = one_dose._initializer_transition_record(  # noqa: SLF001
            kind="head_only_auxiliary_warmup",
            role="head_only_auxiliary_commissioning",
            source_checkpoint_sha256=promotion._sha256(pointer),
            output_checkpoint_sha256=promotion._sha256(warmed),
            sampled_rows=524_288,
            optimizer_steps=128,
            optimizer_state_terminal="discarded_before_joint_training",
            receipt_path=str(warmup_path),
            receipt_file_sha256=promotion._sha256(warmup_path),
            receipt_state_sha256=warmup["state_sha256"],
        )
        chain.extend([pointer_row, warmup_row])
    init = warmed if selected_aux == "AUXT" else transitioned
    lineage_dose = one_dose.lineage.direct_lineage_dose(
        declared_producer_sha256=promotion._sha256(raw),
        init_checkpoint_sha256=promotion._sha256(init),
        initializer_transition_chain=chain,
        current_sampled_rows=524_288,
        current_optimizer_steps=128,
    )
    contract = _contract(producer=raw)
    report = {
        "a1_contract_sha256": None,
        "a1_learner_training_recipe_sha256": None,
        "a1_bound_learner_training_recipe": None,
        "arch": "entity_graph",
        "mask_hidden_info": True,
        "symmetry_augment": False,
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "steps_completed": 128,
        "epochs": 1,
        "max_steps": 0,
        "checkpoint": str(candidate),
        "init_checkpoint": str(init),
        "init_checkpoint_sha256": promotion._sha256(init),
        "a1_lineage_dose": lineage_dose,
        "a1_central_learner_binding": {
            "stage": "FINAL",
            "selected_aux_decision": selected_aux,
            "diagnostic_only": False,
            "promotion_eligible": False,
            "eligible_for_full_gate": True,
            "full_gate_required": True,
            "immutable_contract_recipe": contract["science"][
                "learner_training_recipe"
            ],
            "immutable_contract_recipe_sha256": promotion._digest_value(
                contract["science"]["learner_training_recipe"]
            ),
        },
    }
    report_path = tmp_path / f"{selected_aux}-report.json"
    _write_json(report_path, report)

    def verify_transition(
        path: Path,
        *,
        source_checkpoint: Path,
        transitioned_checkpoint: Path,
        expected_origin_tool_sha256: str,
    ):
        assert path == transition_receipt
        assert source_checkpoint == raw
        assert transitioned_checkpoint == transitioned
        assert expected_origin_tool_sha256
        return transition_evidence

    monkeypatch.setattr(
        one_dose.aux_coordinator.scientific_evidence,
        "verify_public_award_transition_receipt",
        verify_transition,
    )
    monkeypatch.setattr(
        one_dose.architecture_upgrade,
        "verify_receipt",
        lambda path: pointer_evidence,
    )
    return report_path, candidate, contract


@pytest.mark.parametrize("selected_aux", ["AUX0", "AUXT"])
def test_promotion_training_report_accepts_only_independent_final_typed_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_aux: str,
) -> None:
    report, candidate, contract = _typed_final_training_report(
        tmp_path, monkeypatch, selected_aux=selected_aux
    )
    verified = promotion._verify_training_report(
        report,
        contract=contract,
        contract_sha256=contract["contract_sha256"],
        candidate_path=candidate,
        candidate_sha256=promotion._sha256(candidate),
    )
    chain = verified["a1_lineage_dose"]["initializer_transition_chain"]
    assert chain[0]["source_checkpoint_sha256"] == contract["checkpoints"][0][
        "sha256"
    ]
    assert chain[-1]["output_checkpoint_sha256"] == verified[
        "init_checkpoint_sha256"
    ]
    raw_parent_sha = contract["checkpoints"][0]["sha256"]
    assert promotion._training_evaluation_parent_sha256(verified, {}) == (  # noqa: SLF001
        raw_parent_sha
    )
    with pytest.raises(promotion.PromotionError, match="typed chain source"):
        promotion._training_evaluation_parent_sha256(  # noqa: SLF001
            verified, {"evaluation_parent_sha256": verified["init_checkpoint_sha256"]}
        )

    verified["a1_central_learner_binding"]["stage"] = selected_aux
    _write_json(report, verified)
    with pytest.raises(promotion.PromotionError, match="gate-eligible FINAL"):
        promotion._verify_training_report(
            report,
            contract=contract,
            contract_sha256=contract["contract_sha256"],
            candidate_path=candidate,
            candidate_sha256=promotion._sha256(candidate),
        )


def _central_final_receipt_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected_aux: str,
) -> dict:
    report, candidate, contract = _typed_final_training_report(
        tmp_path, monkeypatch, selected_aux=selected_aux
    )
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    lineage = report_payload["a1_lineage_dose"]
    contract_path = tmp_path / f"{selected_aux}-contract.lock.json"
    _write_json(contract_path, contract)
    optimizer = Path(str(candidate) + ".optimizer.pt")
    optimizer.write_bytes(b"fresh central optimizer")
    progress = Path(str(candidate) + ".training-progress.json")
    progress.write_text("{}\n", encoding="utf-8")
    command = ["/usr/bin/python3", "-m", "torch.distributed.run", "--nproc-per-node=8"]
    execution_binding = one_dose._execution_binding(  # noqa: SLF001
        command=command, environment=one_dose._child_environment(list(range(8)))  # noqa: SLF001
    )
    outputs = {
        "checkpoint": str(candidate),
        "checkpoint_sha256": promotion._sha256(candidate),
        "optimizer_sidecar": str(optimizer),
        "optimizer_sidecar_sha256": promotion._sha256(optimizer),
        "training_progress": str(progress),
        "training_progress_sha256": promotion._sha256(progress),
        "training_progress_payload_sha256": "sha256:" + "1" * 64,
        "report": str(report),
        "report_sha256": promotion._sha256(report),
        "sample_receipt_state_sha256": "sha256:" + "2" * 64,
        "sample_order_sha256": "sha256:" + "3" * 64,
        "row_set_sha256": "sha256:" + "4" * 64,
        "realized_sample_evidence_sha256": "sha256:" + "5" * 64,
        "execution_binding_sha256": promotion._digest_value(execution_binding),
        "input_binding_sha256": "sha256:" + "6" * 64,
        "steps_completed": 128,
        "unique_training_rows": None,
        "base_sampler_draw_events": 524_288,
        "sampler_draw_events": 524_288,
        "sampled_rows": 524_288,
        "lineage_dose": lineage,
        "corpus_row_count": 1_000_000,
        "training_row_count": 950_000,
        "validation_row_count": 50_000,
        "production_sampling_receipt_sha256": "sha256:" + "7" * 64,
        "validation_split_receipt_sha256": "sha256:" + "8" * 64,
    }
    experiment_id = "sha256:" + "9" * 64
    root = tmp_path / f"{selected_aux}-coordinator"
    experiment_dir = root / experiment_id.removeprefix("sha256:")
    experiment_dir.mkdir(parents=True)
    published_path = experiment_dir / "93-final-executor-authority.json"
    published_path.write_text("{}\n", encoding="utf-8")
    central = {
        **report_payload["a1_central_learner_binding"],
        "reviewed_lock_file_sha256": promotion._sha256(contract_path),
        "code_tree_sha256": "sha256:" + "a" * 64,
    }
    published = {
        "schema_version": "a1-published-executor-authority-v1",
        "path": str(published_path),
        "file_sha256": promotion._sha256(published_path),
        "authority": {"schema_version": "a1-final-replication-executor-authority-v1"},
    }
    central_input_binding = {"data_kind": "production_composite_v2"}
    value = {
        "schema_version": one_dose.CENTRAL_RECEIPT_SCHEMA,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "lock": str(contract_path),
        "lock_file_sha256": promotion._sha256(contract_path),
        "corpus": str(tmp_path / "composite.json"),
        "corpus_meta_file_sha256": "sha256:" + "b" * 64,
        "payload_inventory_sha256": "sha256:" + "c" * 64,
        "validation_manifest": str(tmp_path / "composite.json"),
        "validation_manifest_file_sha256": "sha256:" + "d" * 64,
        "producer_checkpoint_sha256": contract["checkpoints"][0]["sha256"],
        "learner_training_recipe_sha256": contract["science"][
            "learner_training_recipe_sha256"
        ],
        "command": command,
        "command_sha256": promotion._digest_value(command),
        "execution_binding": execution_binding,
        "input_binding": central_input_binding,
        "training_transaction_sha256": one_dose._training_transaction_sha256(
            command=command, input_binding=central_input_binding
        ),
        "trainer_authority": None,
        "lock_verifier_authority": None,
        "world_size": 8,
        "gpu": 0,
        "gpus": list(range(8)),
        "gpu_name": "NVIDIA B200",
        "gpu_names": ["NVIDIA B200"] * 8,
        "training_topology": {"world_size": 8, "physical_gpus": list(range(8))},
        "ddp_canary": {"path": str(tmp_path / "ddp-canary.json")},
        "production_sampling_receipt_sha256": "sha256:" + "7" * 64,
        "validation_split_receipt_sha256": "sha256:" + "8" * 64,
        "started_unix_ns": 10,
        "finished_unix_ns": 20,
        "returncode": 0,
        "outputs": outputs,
        "lineage_dose": lineage,
        "failure": None,
        "claim_identity_sha256": "sha256:" + "e" * 64,
        "central_learner_binding": central,
        "central_published_executor_authority": published,
        "central_execution_commitment": {"path": str(tmp_path / "commitment.json")},
        "claim": str(tmp_path / "central.claim.json"),
        "claim_state_sha256": "sha256:" + "f" * 64,
    }
    if selected_aux == "AUXT":
        value["function_preserving_upgrade"] = {"receipt": {"path": "upgrade.json"}}
    receipt_path = tmp_path / f"{selected_aux}-central.receipt.json"
    value["receipt_sha256"] = promotion._digest_value(value)
    _write_json(receipt_path, value)
    receipt_path.chmod(0o444)

    terminal = {
        "schema_version": "a1-final-replication-terminal-v1",
        "execution_evidence": {"receipt_path": str(receipt_path)},
        "result": {
            "status": "complete",
            "checkpoint_sha256": promotion._sha256(candidate),
            "full_gate_entry_eligible": True,
        },
        "diagnostic_only": False,
        "promotion_eligible": False,
        "eligible_for_full_gate": True,
        "full_gate_required": True,
        "auto_promotion": False,
    }
    terminal["state_sha256"] = promotion._digest_value(terminal)
    terminal_path = experiment_dir / "95-final-terminal.json"
    _write_json(terminal_path, terminal)
    terminal_path.chmod(0o444)

    monkeypatch.setattr(
        promotion,
        "_reconstruct_completed_central_final",
        lambda *_args, **_kwargs: (
            {"bound_recipe": contract["science"]["learner_training_recipe"]},
            root,
            experiment_id,
        ),
    )
    from tools import a1_central_learner_completion as central_completion
    from tools import a1_aux_pair_coordinator as coordinator

    monkeypatch.setattr(
        central_completion,
        "authenticate_completed_receipt",
        lambda path, *, verified: json.loads(path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        coordinator,
        "_verify_central_terminal_execution_evidence",
        lambda *_args, **_kwargs: {"state_sha256": "sha256:" + "0" * 64},
    )
    return {
        "receipt": receipt_path,
        "contract_path": contract_path,
        "contract": contract,
        "candidate": candidate,
        "report": report,
        "terminal": terminal_path,
    }


@pytest.mark.parametrize("selected_aux", ["AUX0", "AUXT"])
def test_central_final_receipt_reaches_promotion_preflight_with_raw_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_aux: str,
) -> None:
    fixture = _central_final_receipt_fixture(
        tmp_path, monkeypatch, selected_aux=selected_aux
    )
    verified = promotion._verify_one_dose_training_receipt(
        fixture["receipt"],
        contract_lock=fixture["contract_path"],
        contract=fixture["contract"],
        candidate_path=fixture["candidate"],
        candidate_sha256=promotion._sha256(fixture["candidate"]),
        training_report_path=fixture["report"],
        training_report_sha256=promotion._sha256(fixture["report"]),
    )
    assert verified["world_size"] == 8
    assert verified["evaluation_parent_sha256"] == fixture["contract"][
        "checkpoints"
    ][0]["sha256"]
    assert verified["central_final"]["terminal"]["path"] == str(
        fixture["terminal"]
    )


def test_central_final_receipt_refuses_nonfinal_or_single_gpu_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _central_final_receipt_fixture(
        tmp_path, monkeypatch, selected_aux="AUX0"
    )
    value = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    value["world_size"] = 1
    value["receipt_sha256"] = promotion._digest_value(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    fixture["receipt"].chmod(0o644)
    _write_json(fixture["receipt"], value)
    fixture["receipt"].chmod(0o444)
    with pytest.raises(promotion.PromotionError, match="successful 8xB200"):
        promotion._verify_one_dose_training_receipt(
            fixture["receipt"],
            contract_lock=fixture["contract_path"],
            contract=fixture["contract"],
            candidate_path=fixture["candidate"],
            candidate_sha256=promotion._sha256(fixture["candidate"]),
            training_report_path=fixture["report"],
            training_report_sha256=promotion._sha256(fixture["report"]),
        )


def test_recovery_gate_authority_binds_every_final_promotion_input(
    tmp_path: Path,
) -> None:
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "contract_lock",
            "standard_adjudication",
            "training_receipt",
            "cohort_exclusions",
            "registry",
            "current_pointer",
        )
    }
    for name, path in paths.items():
        path.write_text(name + "\n", encoding="utf-8")
    candidate = {"path": str(tmp_path / "candidate.pt"), "sha256": "sha256:" + "a" * 64}
    verifier_authority = {
        "schema_version": promotion.frozen_lock_verifier.AUTHORITY_SCHEMA,
        "authority_sha256": "sha256:" + "b" * 64,
    }
    gate_authority = {
        "inputs": {
            name: {"path": str(path), "sha256": promotion._sha256(path)}
            for name, path in paths.items()
        },
        "candidate": candidate,
        "contract_verifier": verifier_authority,
        "policy": {"promotion_eligible": True},
        "authority_sha256": "sha256:" + "c" * 64,
    }
    authority_path = tmp_path / "full-gate.json"
    authority_path.write_text("authority\n", encoding="utf-8")

    bound = promotion._verify_recovery_gate_for_promotion(
        gate_authority,
        authority_ref={
            "path": str(authority_path),
            "sha256": promotion._sha256(authority_path),
        },
        verifier_authority=verifier_authority,
        contract_lock=paths["contract_lock"],
        adjudication_path=paths["standard_adjudication"],
        training_receipt=paths["training_receipt"],
        cohort_exclusions=paths["cohort_exclusions"],
        registry_path=paths["registry"],
        current_pointer=paths["current_pointer"],
        verified={"candidate": candidate},
    )
    assert bound["authority_sha256"] == "sha256:" + "c" * 64

    gate_authority["inputs"]["training_receipt"]["sha256"] = "sha256:" + "d" * 64
    with pytest.raises(promotion.PromotionError, match="training_receipt"):
        promotion._verify_recovery_gate_for_promotion(
            gate_authority,
            authority_ref={
                "path": str(authority_path),
                "sha256": promotion._sha256(authority_path),
            },
            verifier_authority=verifier_authority,
            contract_lock=paths["contract_lock"],
            adjudication_path=paths["standard_adjudication"],
            training_receipt=paths["training_receipt"],
            cohort_exclusions=paths["cohort_exclusions"],
            registry_path=paths["registry"],
            current_pointer=paths["current_pointer"],
            verified={"candidate": candidate},
        )


def test_execute_promotion_propagates_frozen_and_recovery_gate_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    frozen_repo = tmp_path / "frozen"
    verifier = frozen_repo / "tools" / "a1_pre_wave_contract.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("# frozen\n", encoding="utf-8")
    verifier_authority = {
        "schema_version": promotion.frozen_lock_verifier.AUTHORITY_SCHEMA,
        "frozen_repo": str(frozen_repo),
        "verifier": str(verifier),
        "verifier_sha256": promotion._sha256(verifier),
        "authority_sha256": "sha256:" + "e" * 64,
    }
    full_gate_path = tmp_path / "full-gate.json"
    _write_json(full_gate_path, {"authority": "fixture"})
    full_gate = {
        "recovery_authority": {"schema_version": "fixture"},
        "contract_verifier": verifier_authority,
    }
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        promotion.frozen_lock_verifier,
        "build_frozen_lock_verifier",
        lambda **_kwargs: (_verify(fixture), verifier_authority),
    )
    from tools import a1_v5_recovery_gate as recovery_gate

    monkeypatch.setattr(
        recovery_gate,
        "verify_recovery_gate_authority",
        lambda path: full_gate if path == full_gate_path.resolve() else None,
    )

    def fake_prepare(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {"status": "dry_run"}

    monkeypatch.setattr(promotion, "prepare_promotion", fake_prepare)
    result = promotion.execute_promotion(
        registry_path=fixture["registry"],
        current_pointer=fixture["pointer"],
        contract_lock=fixture["contract_path"],
        adjudication_path=fixture["adjudication"],
        training_receipt=fixture["training_receipt"],
        cohort_exclusions=fixture["cohort_exclusions"],
        receipt_path=fixture["receipt"],
        reason="recovery promotion",
        recovery_gate_authority=full_gate_path,
        frozen_repo=frozen_repo,
        frozen_verifier_sha256=promotion._sha256(verifier),
    )

    assert result == {"status": "dry_run"}
    assert observed["recovery_gate_authority"] is full_gate
    assert observed["recovery_gate_authority_ref"] == {
        "path": str(full_gate_path.resolve()),
        "sha256": promotion._sha256(full_gate_path),
    }
    assert observed["verifier_authority"] == verifier_authority
