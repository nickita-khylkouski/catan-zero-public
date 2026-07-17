from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools import a1_one_dose_train as executor
from tools import a1_pre_wave_contract as contract
from catan_zero.rl.entity_token_policy import EntityGraphConfig
from catan_zero.rl.entity_feature_adapter import CURRENT_RUST_ENTITY_ADAPTER_VERSION
from catan_zero.rl.optim_state import save_training_progress


_SHA = "sha256:" + "a" * 64


def _reviewed_ablation_code_binding() -> dict[str, object]:
    """Build the same authenticated code closure used by a real ablation."""

    repo = Path(executor.__file__).resolve().parents[1]
    return executor._current_ablation_code_binding(  # noqa: SLF001
        {
            "provenance": {
                "learner_code": [{"path": str(repo / "tools/train_bc.py")}],
                "runtime_code_tree": [
                    {"path": str(repo / "tools/a1_one_dose_train.py")}
                ],
            }
        }
    )


def _objective() -> dict[str, object]:
    return {
        "objective": "mse",
        "value_readout": "scalar",
        "value_categorical_bins": None,
        "hlgauss_sigma_ratio": None,
    }


def _search_evidence_meta(schema: str) -> dict[str, object]:
    columns = {
        name: {}
        for name in (
            "search_evidence_version",
            "search_evidence_mask",
            "search_evidence_offsets",
            "search_visit_counts_flat",
            "search_completed_q_flat",
        )
    }
    if schema == executor.SEARCH_EVIDENCE_V2_SCHEMA:
        columns["search_prior_policy_flat"] = {}
    return {"search_evidence": {"schema": schema}, "columns": columns}


def test_coherent_direct_search_evidence_accepts_archived_v1_and_bound_v2() -> None:
    executor._verify_coherent_search_evidence_memmap(  # noqa: SLF001
        corpus={
            "search_evidence_schema": executor.SEARCH_EVIDENCE_V1_SCHEMA,
            "search_evidence_storage": "receipt_bound_source_npz_only",
        },
        meta={},
    )
    executor._verify_coherent_search_evidence_memmap(  # noqa: SLF001
        corpus={
            "search_evidence_schema": executor.SEARCH_EVIDENCE_V2_SCHEMA,
            "search_evidence_storage": "training_memmap",
        },
        meta=_search_evidence_meta(executor.SEARCH_EVIDENCE_V2_SCHEMA),
    )

    missing_prior = _search_evidence_meta(executor.SEARCH_EVIDENCE_V2_SCHEMA)
    del missing_prior["columns"]["search_prior_policy_flat"]  # type: ignore[index]
    with pytest.raises(executor.ExecutorError, match="differs from admission"):
        executor._verify_coherent_search_evidence_memmap(  # noqa: SLF001
            corpus={
                "search_evidence_schema": executor.SEARCH_EVIDENCE_V2_SCHEMA,
                "search_evidence_storage": "training_memmap",
            },
            meta=missing_prior,
        )
    with pytest.raises(executor.ExecutorError, match="must be stored"):
        executor._verify_coherent_search_evidence_memmap(  # noqa: SLF001
            corpus={
                "search_evidence_schema": executor.SEARCH_EVIDENCE_V2_SCHEMA,
                "search_evidence_storage": "receipt_bound_source_npz_only",
            },
            meta={},
        )


def _lock(
    *,
    n_full: int = 128,
    producer: Path = Path("/producer.pt"),
    ledger: Path = Path("/seed-ledger.tsv"),
) -> dict:
    return {
        "contract_sha256": _SHA,
        "science": {
            "search_operator": {"n_full": n_full, "p_full": 0.25},
            "learner_training_recipe": dict(contract.EXPECTED_LEARNER_TRAINING_RECIPE),
            "learner_value_objective": _objective(),
        },
        "checkpoints": [
            {
                "role": "producer",
                "path": str(producer),
                "sha256": "sha256:" + "b" * 64,
            }
        ],
        "fleet": {"seed_ledger": {"path": str(ledger)}},
    }


def _verified(tmp_path: Path) -> dict:
    producer = tmp_path / "producer.pt"
    producer.write_bytes(b"producer")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}")
    validation = tmp_path / "validation.json"
    validation.write_text("{}")
    data = tmp_path / "corpus"
    data.mkdir()
    (data / "corpus_meta.json").write_text("{}")
    ledger = tmp_path / "seed-ledger.tsv"
    ledger.write_text("sealed ledger\n")
    lock = _lock(producer=producer, ledger=ledger)
    lock["checkpoints"][0]["sha256"] = executor._file_sha256(producer)
    return {
        "lock": lock,
        "lock_path": lock_path,
        "lock_file_sha256": executor._file_sha256(lock_path),
        "contract_sha256": _SHA,
        "recipe": {
            **contract.EXPECTED_LEARNER_TRAINING_RECIPE,
            "policy_target_blend_semantics": (
                executor.train_bc.POLICY_TARGET_BLEND_LEGACY_V1
            ),
        },
        "objective": _objective(),
        "producer": lock["checkpoints"][0],
        "data_path": data,
        "corpus_meta_file_sha256": executor._file_sha256(data / "corpus_meta.json"),
        "payload_inventory_sha256": "sha256:" + "c" * 64,
        "data_fingerprint": "sha256:" + "1" * 64,
        "corpus_row_count": 5000,
        "training_row_count": 4097,
        "validation_row_count": 903,
        "selected_game_seed_set_sha256": "sha256:" + "d" * 64,
        "training_game_seed_set_sha256": "sha256:" + "e" * 64,
        "validation_path": validation,
        "validation_file_sha256": executor._file_sha256(validation),
        "validation_game_seed_set_sha256": "sha256:" + "f" * 64,
    }


def _production_trainer_verified(tmp_path: Path) -> dict:
    verified = _verified(tmp_path)
    event_history_acknowledgements = [
        "sha256:" + f"{index:x}" * 64 for index in range(6, 10)
    ]
    verified.update(
        {
            "data_kind": "production_composite_v2",
            "trainer_authority": executor._current_production_trainer_authority(),
            "production_mix_contract": {},
            "production_sampling_receipt_sha256": "sha256:" + "6" * 64,
            "validation_split_receipt": {},
            "validation_split_receipt_sha256": "sha256:" + "7" * 64,
            "composite_build_receipt": {},
            "source_authority_ref": {},
            "category_semantics": {},
            "category_semantics_sha256": "sha256:" + "8" * 64,
            "event_history_training_contract": {
                "schema": "a1-training-event-history-contract-v1",
                "training_event_history_trainable": False,
                "event_history_end_to_end_usable": False,
                "status": "empty_payloads_acknowledged",
                "empty_payload_inventory_acknowledgements": (
                    event_history_acknowledgements
                ),
            },
            "event_history_component_authority": [
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
            ],
        }
    )
    return verified


def _fake_aux_upgrade(verified: dict, tmp_path: Path) -> dict:
    initializer = tmp_path / "shared-aux-initializer.pt"
    initializer.write_bytes(b"one shared aux initializer")
    receipt = tmp_path / "shared-aux-upgrade.receipt.json"
    receipt.write_text("{}")
    spec = executor.architecture_upgrade.ALLOWLIST[
        executor.AUX_REGULARIZATION_MODULE
    ]
    seeded = {
        name: "sha256:" + f"{index + 1:064x}"
        for index, (name, kind) in enumerate(
            sorted(spec["new_parameter_initialization"].items())
        )
        if kind == "seeded_torch_default"
    }
    return {
        "module": executor.AUX_REGULARIZATION_MODULE,
        "source": dict(verified["producer"]),
        "upgraded_initializer": {
            "path": str(initializer.resolve()),
            "sha256": executor._file_sha256(initializer),
        },
        "receipt_sha256": "sha256:" + "4" * 64,
        "receipt": {
            "path": str(receipt.resolve()),
            "sha256": executor._file_sha256(receipt),
        },
        "flags": dict(spec["flags"]),
        "initialization_seed": 20260713,
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
        "shared_parameters_bit_identical": True,
        "shared_parameter_count": 100,
        "new_parameters": sorted(spec["new_parameter_initialization"]),
        "new_parameter_initialization": dict(
            spec["new_parameter_initialization"]
        ),
        "effective_source_config_sha256": "sha256:" + "5" * 64,
        "effective_upgraded_config_sha256": "sha256:" + "6" * 64,
        "seeded_parameter_sha256": seeded,
    }


def _fake_public_card_upgrade(verified: dict, tmp_path: Path) -> dict:
    initializer = tmp_path / "public-card-initializer.pt"
    initializer.write_bytes(b"function-preserving public-card residual")
    receipt = tmp_path / "public-card-upgrade.receipt.json"
    receipt.write_text("{}")
    return {
        "module": executor.architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES_V2,
        "source": dict(verified["producer"]),
        "upgraded_initializer": {
            "path": str(initializer.resolve()),
            "sha256": executor._file_sha256(initializer),
        },
        "receipt_sha256": "sha256:" + "4" * 64,
        "receipt": {
            "path": str(receipt.resolve()),
            "sha256": executor._file_sha256(receipt),
        },
    }


def _patch_valid_aux_admission(
    monkeypatch: pytest.MonkeyPatch, verified: dict
) -> dict[str, np.ndarray]:
    rows = int(verified["training_row_count"])
    data = {
        "aux_longest_road": np.zeros(rows, dtype=np.float32),
        "aux_largest_army": np.ones(rows, dtype=np.float32),
        "aux_vp_in_n": np.zeros(rows, dtype=np.float32),
        "aux_next_settlement": np.arange(rows, dtype=np.int16) % 54,
        "aux_robber_target": np.arange(rows, dtype=np.int16) % 19,
        executor.train_bc.AUX_SUBGOAL_TARGET_VERSION_KEY: np.full(
            rows,
            executor.train_bc.AUX_SUBGOAL_TARGET_VERSION,
            dtype=np.uint8,
        ),
    }
    monkeypatch.setattr(
        executor.train_bc, "load_teacher_data_memmap", lambda _path: data
    )
    monkeypatch.setattr(
        executor.train_bc,
        "_load_validation_game_seed_manifest_for_training",
        lambda *args, **kwargs: {"game_seeds": np.asarray([99], dtype=np.int64)},
    )
    monkeypatch.setattr(
        executor.train_bc,
        "split_train_validation_indices",
        lambda *args, **kwargs: {
            "train": np.arange(rows, dtype=np.int64),
            "validation": np.empty(0, dtype=np.int64),
        },
    )
    return data


def _training_report(
    verified: dict, checkpoint: Path, *, steps_completed: int = 2
) -> dict:
    recipe = verified["recipe"]
    resume_identity = {
        "schema_version": "train-bc-resume-recipe-v1",
        "normalized_train_config_sha256": "sha256:" + "9" * 64,
        "grad_accum_steps": int(recipe["grad_accum_steps"]),
        "world_size": int(recipe["world_size"]),
        "ddp_shard_data": False,
        "fsdp": False,
        "policy_aux_active_batch_size": int(
            recipe.get("policy_aux_active_batch_size", 0)
        ),
        "policy_aux_loss_weight": float(recipe.get("policy_aux_loss_weight", 1.0)),
    }
    payload = {
        "arch": "entity_graph",
        **executor.SEALED_A1_MODEL_REPORT,
        "a1_contract_sha256": verified["contract_sha256"],
        "a1_bound_learner_training_recipe": recipe,
        "a1_bound_learner_value_objective": verified["objective"],
        "a1_learner_training_recipe_sha256": executor._value_sha256(recipe),
        "a1_memmap_payload_inventory_sha256": verified["payload_inventory_sha256"],
        "a1_selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
        "a1_training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
        "world_size": 1,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "fused_optimizer": False,
        "epochs": 1,
        "max_steps": recipe["max_steps"],
        "exact_max_steps": int(recipe["max_steps"]) > 0,
        "batch_size": recipe["batch_size"],
        "grad_accum_steps": recipe["grad_accum_steps"],
        "effective_global_batch_size": recipe["global_batch_size"],
        "ddp_shard_data": False,
        "amp": recipe["amp"],
        "lr": recipe["lr"],
        "weight_decay": recipe["weight_decay"],
        "seed": recipe["seed"],
        "training_rng_rank_offset": bool(
            recipe.get("training_rng_rank_offset", False)
        ),
        "mask_hidden_info": True,
        "symmetry_augment": False,
        "data": str(verified["data_path"]),
        "data_format": "memmap",
        "data_fingerprint": verified["data_fingerprint"],
        "samples": verified["corpus_row_count"],
        "global_samples": verified["corpus_row_count"],
        "train_samples": verified["training_row_count"],
        "validation_samples": verified["validation_row_count"],
        "track": recipe["track"],
        "vps_to_win": recipe["vps_to_win"],
        "checkpoint": str(checkpoint),
        "init_checkpoint": str(verified["producer"]["path"]),
        "init_checkpoint_sha256": verified["producer"]["sha256"],
        "input_validation_game_seed_manifest": str(verified["validation_path"]),
        "input_validation_game_seed_manifest_sha256": verified[
            "validation_file_sha256"
        ],
        "validation_game_seed_set_sha256": verified["validation_game_seed_set_sha256"],
        "forced_action_weight": float(recipe["forced_action_weight"]),
        "forced_row_value_weight": float(recipe["forced_row_value_weight"]),
        "per_game_policy_weight": bool(
            recipe.get("per_game_policy_weight", False)
        ),
        "per_game_policy_weight_mode": str(
            recipe.get("per_game_policy_weight_mode", "equal")
        ),
        "per_game_value_weight": bool(recipe["per_game_value_weight"]),
        "value_loss_weight": float(recipe["value_loss_weight"]),
        "truncated_vp_margin_value_weight": float(
            recipe["truncated_vp_margin_value_weight"]
        ),
        "steps_completed": steps_completed,
        "total_training_steps": steps_completed,
        "training_resume_recipe_identity": resume_identity,
        "training_resume_recipe_identity_sha256": executor._value_sha256(
            resume_identity
        ),
        "require_35m_model": True,
        "parameter_count": 35_000_000,
        "value_training": {
            "primary_readout": "scalar",
            "trained_value_readouts": ["scalar"],
            "optimizer_steps": steps_completed,
            "completed_epochs": 1,
            "a1_contract_sha256": verified["contract_sha256"],
            "a1_selected_game_seed_set_sha256": verified[
                "selected_game_seed_set_sha256"
            ],
            "a1_training_game_seed_set_sha256": verified[
                "training_game_seed_set_sha256"
            ],
            "a1_learner_training_recipe_sha256": executor._value_sha256(recipe),
            "a1_memmap_payload_inventory_sha256": verified["payload_inventory_sha256"],
        },
        "metrics": [
            {
                "epoch": 1,
                "loss": 1.0,
                "policy_loss": 0.8,
                "value_loss": 0.2,
                "loss_denominators": {
                    "value_loss": float(steps_completed * recipe["batch_size"]),
                    "policy_kl_anchor_loss": 0.0,
                },
                "validation": {
                    "samples": verified["validation_row_count"],
                    "loss": 1.1,
                },
            }
        ],
    }
    if "policy_aux_active_batch_size" in recipe:
        # The legacy single-rank iterator keeps its final partial batch instead
        # of padding it.  Report the rows actually consumed, not nominal
        # optimizer_steps * batch_size capacity.  (The 8-rank DDP topology has
        # an authenticated padding receipt and therefore reports padded global
        # draw events.)
        base_draws = verified["training_row_count"]
        aux_draws = steps_completed * recipe["policy_aux_active_batch_size"]
        policy_base = min(base_draws, 1_000)
        payload.update(
            {
                "policy_aux_active_batch_size": recipe[
                    "policy_aux_active_batch_size"
                ],
                "policy_aux_loss_weight": recipe["policy_aux_loss_weight"],
                "training_row_draws": base_draws,
                "base_training_row_draws": base_draws,
                "policy_aux_training_row_draws": aux_draws,
                "total_training_row_draws": base_draws + aux_draws,
                "policy_base_active_rows": policy_base,
                "policy_aux_active_rows": aux_draws,
                "policy_total_active_rows": policy_base + aux_draws,
                "value_active_rows": base_draws,
                "policy_kl_anchor_eligible_rows": 0,
            }
        )
    return payload


def _write_training_progress(
    checkpoint: Path,
    report_payload: dict,
    *,
    steps_completed: int = 2,
) -> None:
    """Commit a structurally real checkpoint-set marker for executor tests."""

    identity = report_payload["training_resume_recipe_identity"]
    world_size = int(identity["world_size"])
    numpy_states = [
        {"bit_generator": "PCG64", "state": {"state": rank, "inc": rank + 1}}
        for rank in range(world_size)
    ]
    torch_states = [
        {"rank": rank, "cpu": [rank], "cuda": None}
        for rank in range(world_size)
    ]
    assert save_training_progress(
        checkpoint,
        optimizer_step=steps_completed,
        completed_epochs=1,
        recipe_identity=identity,
        rng_state=numpy_states[0],
        rank_numpy_rng_states=numpy_states,
        symmetry_rng_state=None,
        rank_torch_rng_states=torch_states,
        scalar_training_weight_sum=1.0,
        categorical_training_weight_sum=0.0,
        checkpoint_role="terminal_admitted",
        ddp={"rank": 0},
    ) is not None


def _option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _replace_option(command: list[str], flag: str, value: str) -> list[str]:
    changed = list(command)
    changed[changed.index(flag) + 1] = value
    return changed


def _remove_option(command: list[str], flag: str) -> list[str]:
    changed = list(command)
    index = changed.index(flag)
    del changed[index : index + 2]
    return changed


class _FakeCompositeCorpus:
    def __init__(self, component_seed_rows: list[list[int]]) -> None:
        self.corpora = [
            {"game_seed": np.asarray(seeds, dtype=np.int64)}
            for seeds in component_seed_rows
        ]
        self.component_offsets = np.cumsum(
            [0, *(len(seeds) for seeds in component_seed_rows)], dtype=np.int64
        )

    def __len__(self) -> int:
        return int(self.component_offsets[-1])


def _production_composite_meta(tmp_path: Path, producer_sha256: str) -> dict:
    component_ids = [
        "current_producer",
        "recent_history",
        "hard_negative",
    ]
    ratios = {
        "current_producer": 0.80,
        "recent_history": 0.15,
        "hard_negative": 0.05,
    }
    sampling_receipt = {"effective_component_sampling_ratios": ratios}
    contract_payload = {
        "schema_version": "flywheel-replay-composite-v3",
        "fresh_component_ids": component_ids,
        "replay_component_ids": [],
        "fresh_source_game_ratios": {
            "current_producer": 0.8,
            "recent_history": 0.15,
            "hard_negative": 0.05,
        },
        "effective_component_sampling_ratios": ratios,
        "realized_replay_ratio": 0.0,
        "initializer_checkpoint_sha256": producer_sha256,
        "sampling_receipt": sampling_receipt,
        "sampling_receipt_sha256": executor._value_sha256(sampling_receipt),
    }
    return {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": False,
        "promotion_eligible": True,
        "component_ids": component_ids,
        "component_game_sampling_ratios": list(ratios.values()),
        "stored_policy_component_temperatures": dict(
            executor.composite_builder.STORED_POLICY_COMPONENT_TEMPERATURES
        ),
        "entity_feature_adapter_component_versions": {
            component_id: CURRENT_RUST_ENTITY_ADAPTER_VERSION
            for component_id in component_ids
        },
        "production_mix_contract": contract_payload,
        "components": [
            {
                "component_id": component_id,
                "source_category": component_id,
                "corpus_dir": str(tmp_path / component_id),
                "payload_inventory_sha256": (
                    "sha256:" + f"{index + 6:x}" * 64
                ),
                "corpus_meta": {
                    "payload_inventory_sha256": (
                        "sha256:" + f"{index + 6:x}" * 64
                    ),
                    "implicit_zero_columns": ["event_tokens", "event_mask"],
                },
            }
            for index, component_id in enumerate(component_ids)
        ],
        "descriptor_file_sha256": "sha256:" + "2" * 64,
        "descriptor_fingerprint": "sha256:" + "3" * 64,
        "learner_recipe_overrides": dict(
            executor.composite_builder.LEARNER_RECIPE_OVERRIDES
        ),
        "learner_recipe_overrides_sha256": "sha256:" + "4" * 64,
        "policy_kl_anchor_component_ids": [],
        "policy_distillation_component_ids": component_ids[:3],
        "value_training_component_ids": component_ids[:1],
        "aux_subgoal_target_contract_sha256": "sha256:" + "a" * 64,
        "public_award_feature_transition_contract_sha256": "sha256:" + "b" * 64,
        "source_authority_semantic_sha256": "sha256:" + "c" * 64,
        "payload_inventory_sha256": "sha256:" + "5" * 64,
    }


def _production_composite_build_receipt(
    tmp_path: Path,
    *,
    verified: dict,
    descriptor: Path,
    meta: dict,
) -> Path:
    authority_ref = {
        "path": str((tmp_path / "source-authority.json").resolve()),
        "file_sha256": "sha256:" + "7" * 64,
        "authority_sha256": "sha256:" + "8" * 64,
    }
    meta["source_authority_ref"] = authority_ref
    activation = {"passed": True, "target_activation_sha256": "sha256:" + "9" * 64}
    meta["source_authority"] = {
        "current_contract": {
            "file_sha256": executor._file_sha256(verified["lock_path"]),
            "contract_sha256": verified["contract_sha256"],
        },
        "fresh_source_bindings": [],
        "fresh_target_activation": activation,
    }
    payload = {
        "schema_version": "a1-post-wave-composite-build-v3",
        "contract": {
            "path": str(verified["lock_path"]),
            "file_sha256": executor._file_sha256(verified["lock_path"]),
            "contract_sha256": verified["contract_sha256"],
        },
        "selected_game_manifest": {},
        "post_wave_audit": {
            "target_activation_sha256": activation["target_activation_sha256"]
        },
        "fresh_target_activation": activation,
        "source_bindings": [],
        "source_bindings_sha256": executor._value_sha256([]),
        "source_authority": authority_ref,
        "descriptor": {
            "path": str(descriptor.resolve()),
            "file_sha256": meta["descriptor_file_sha256"],
            "fingerprint": meta["descriptor_fingerprint"],
        },
        "sampling_receipt": meta["production_mix_contract"]["sampling_receipt"],
        "verified_descriptor_fingerprint": meta["descriptor_fingerprint"],
    }
    payload["receipt_sha256"] = executor._value_sha256(payload)
    path = tmp_path / "build-receipt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _failed_architecture_attempt(tmp_path: Path) -> tuple[dict, Path, list[str]]:
    torch = pytest.importorskip("torch")

    verified = _verified(tmp_path)
    producer = Path(verified["producer"]["path"])
    torch.save(
        {
            "policy_type": "entity_graph",
            "config": {
                "__config_dataclass__": "EntityGraphConfig",
                "__config_schema__": 1,
                "fields": {
                    "hidden_size": 640,
                    "state_layers": 6,
                    "attention_heads": 8,
                    "dropout": 0.05,
                    "state_trunk": "transformer",
                    "relational_block_pattern": "",
                    "relational_ff_size": 0,
                    "relational_bases": 4,
                    "relational_action_cross_layers": 1,
                    "latent_deliberation_steps": 0,
                    "latent_deliberation_slots": 8,
                    "moe_routed_experts": 0,
                    "moe_top_k": 2,
                    "moe_expert_ff_size": 0,
                    "value_categorical_bins": 0,
                },
            },
        },
        producer,
    )
    producer_sha = executor._file_sha256(producer)
    verified["producer"]["sha256"] = producer_sha
    parent_checkpoint = tmp_path / "attempt-r1" / "candidate.pt"
    parent_report = tmp_path / "attempt-r1" / "training.report.json"
    parent_receipt = tmp_path / "attempt-r1" / "training.receipt.json"
    parent_command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=parent_checkpoint,
        report=parent_report,
    )
    # Reproduce the exact production defect: the legacy argv omitted both
    # fields. train_bc resolved entity_graph hidden_size to 640, while its old
    # graph_layers parser default remained 4 and mismatched the 6-layer parent.
    parent_command = _remove_option(parent_command, "--graph-layers")
    parent_command = _remove_option(parent_command, "--hidden-size")
    with pytest.raises(executor.ExecutorError, match="exited nonzero"):
        executor.execute(
            verified=verified,
            command=parent_command,
            checkpoint=parent_checkpoint,
            report=parent_report,
            receipt=parent_receipt,
            gpu=0,
            runner=lambda *args, **kwargs: subprocess.CompletedProcess(
                parent_command, 1
            ),
            probe=lambda _gpu: "NVIDIA B200",
        )
    return verified, executor._claim_path(verified), parent_command


def _trainer_authority_with_train_bc_sha(
    authority: dict, trainer_sha256: str
) -> dict:
    rebound = copy.deepcopy(authority)
    rebound["sha256"] = trainer_sha256
    for record in rebound["code_surface"]:
        if record["relative_path"] == "tools/train_bc.py":
            record["sha256"] = trainer_sha256
    rebound["code_surface_sha256"] = executor._value_sha256(
        rebound["code_surface"]
    )
    rebound.pop("authority_sha256", None)
    rebound["authority_sha256"] = executor._value_sha256(rebound)
    return rebound


def _failed_production_preflight_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, Path, list[str]]:
    verified = executor.bind_training_topology(
        _production_trainer_verified(tmp_path),
        topology=executor.B200_8GPU_DDP_TOPOLOGY,
        gpu=0,
    )
    live_authority = executor._current_production_trainer_authority()
    fixed_authority = _trainer_authority_with_train_bc_sha(
        live_authority, executor.FIXED_PRODUCTION_PREFLIGHT_TRAINER_SHA256
    )
    buggy_authority = _trainer_authority_with_train_bc_sha(
        live_authority, executor.BUGGY_PRODUCTION_PREFLIGHT_TRAINER_SHA256
    )
    verified["trainer_authority"] = buggy_authority
    canary_semantics = {
        "schema_version": "a1-ddp-canary-semantic-identity-v1",
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "train_bc_sha256": executor.BUGGY_PRODUCTION_PREFLIGHT_TRAINER_SHA256,
    }
    verified["ddp_canary"] = {
        "semantic_identity": canary_semantics,
        "semantic_identity_sha256": executor._value_sha256(canary_semantics),
    }
    parent_checkpoint = tmp_path / "production-r1" / "candidate.pt"
    parent_report = tmp_path / "production-r1" / "training.report.json"
    parent_receipt = tmp_path / "production-r1" / "training.receipt.json"
    with monkeypatch.context() as patch:
        patch.setattr(
            executor,
            "_current_production_trainer_authority",
            lambda: copy.deepcopy(buggy_authority),
        )
        parent_command = executor.build_train_command(
            verified,
            python=Path(sys.executable),
            checkpoint=parent_checkpoint,
            report=parent_report,
        )
        with pytest.raises(executor.ExecutorError, match="exited nonzero"):
            executor.execute(
                verified=verified,
                command=parent_command,
                checkpoint=parent_checkpoint,
                report=parent_report,
                receipt=parent_receipt,
                gpu=0,
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(
                    parent_command, 1
                ),
                probe=lambda _gpu: "NVIDIA B200",
            )
    verified["trainer_authority"] = fixed_authority
    fixed_canary_semantics = {
        **canary_semantics,
        "train_bc_sha256": executor.FIXED_PRODUCTION_PREFLIGHT_TRAINER_SHA256,
    }
    verified["ddp_canary"] = {
        "semantic_identity": fixed_canary_semantics,
        "semantic_identity_sha256": executor._value_sha256(
            fixed_canary_semantics
        ),
    }
    monkeypatch.setattr(
        executor,
        "_current_production_trainer_authority",
        lambda: copy.deepcopy(fixed_authority),
    )
    return verified, executor._claim_path(verified), parent_command


def _failed_production_transport_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, Path, list[str], Path]:
    """Build an exact failed first typed retry, then bind fixed transport bytes."""

    verified, original_claim, _ = _failed_production_preflight_attempt(
        tmp_path, monkeypatch
    )
    first_checkpoint = tmp_path / "production-r2" / "candidate.pt"
    first_report = tmp_path / "production-r2" / "training.report.json"
    first_receipt = tmp_path / "production-r2" / "training.receipt.json"
    first_contract = tmp_path / "production-r2" / "retry.contract.json"
    first_command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=first_checkpoint,
        report=first_report,
    )
    first_derived = executor.authorize_failed_before_optimizer_retry(
        verified=verified,
        parent_claim=original_claim,
        retry_command=first_command,
        checkpoint=first_checkpoint,
        report=first_report,
        receipt=first_receipt,
        retry_contract_path=first_contract,
        publish=True,
    )
    with pytest.raises(executor.ExecutorError, match="exited nonzero"):
        executor.execute(
            verified=first_derived,
            command=first_command,
            checkpoint=first_checkpoint,
            report=first_report,
            receipt=first_receipt,
            gpu=0,
            runner=lambda *args, **kwargs: subprocess.CompletedProcess(
                first_command, 1
            ),
            probe=lambda _gpu: "NVIDIA B200",
        )
    first_claim = executor._claim_path(first_derived)

    final_authority = _trainer_authority_with_train_bc_sha(
        verified["trainer_authority"],
        executor.FIXED_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256,
    )
    verified["trainer_authority"] = final_authority
    final_canary_semantics = copy.deepcopy(
        verified["ddp_canary"]["semantic_identity"]
    )
    final_canary_semantics["train_bc_sha256"] = (
        executor.FIXED_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256
    )
    verified["ddp_canary"] = {
        "semantic_identity": final_canary_semantics,
        "semantic_identity_sha256": executor._value_sha256(
            final_canary_semantics
        ),
    }
    monkeypatch.setattr(
        executor,
        "_current_production_trainer_authority",
        lambda: copy.deepcopy(final_authority),
    )
    return verified, first_claim, first_command, first_contract


def test_current_a1_requires_global_n128_and_exact_scalar_dose() -> None:
    recipe, objective = executor._require_a1_science(_lock())
    assert recipe == contract.EXPECTED_LEARNER_TRAINING_RECIPE
    assert objective == _objective()

    with pytest.raises(executor.ExecutorError, match="n_full=128"):
        executor._require_a1_science(_lock(n_full=64))
    with pytest.raises(executor.ExecutorError, match="n_full=128"):
        executor._require_a1_science(_lock(n_full=256))


def test_current_coherent_one_dose_requires_selected_parent_update_authority(
) -> None:
    initialization = executor.current_science.learner_initialization()
    lock = {
        "science": {
            "search_operator": executor.current_science.search(),
            "evaluator": executor.current_science.evaluator(),
            "learner_training_recipe": (
                executor.current_science.learner_training_recipe()
            ),
            "learner_value_objective": (
                executor.current_science.learner_value_objective()
            ),
            "learner_initialization": initialization,
            "learner_initialization_sha256": executor._value_sha256(initialization),
        },
        "generation": executor.current_science.generation(),
        "post_wave_acceptance": {
            "require_target_information_regime": (
                executor.current_science.target_information_regime()
            )
        },
    }

    with pytest.raises(
        executor.ExecutorError,
        match="selects the canonical parent update",
    ):
        executor._require_a1_science(lock)


def test_command_is_direct_one_b200_fresh_unfused_adam(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )

    assert "torch.distributed" not in " ".join(command)
    assert _option(command, "--optimizer") == "adam"
    assert "--no-resume-optimizer" in command
    assert "--no-fused-optimizer" in command
    assert "--fused-optimizer" not in command
    assert _option(command, "--value-lr-mult") == "0.3"
    assert _option(command, "--batch-size") == "4096"
    assert _option(command, "--grad-accum-steps") == "1"
    assert _option(command, "--data-loader-workers") == "2"
    assert _option(command, "--data-loader-prefetch") == "2"
    assert _option(command, "--epochs") == "1"
    assert _option(command, "--value-head-type") == "mse"
    assert _option(command, "--value-categorical-bins") == "0"
    assert "--mask-hidden-info" in command
    assert "--no-symmetry-augment" in command
    assert "--validation-game-seed-manifest" in command
    assert "--p-full" not in command  # generation choice remains contract-bound.
    # Historical source-bound production commands remain causally immutable.
    assert executor.EVENT_HISTORY_ACK_FLAG not in command
    assert executor.EVENT_HISTORY_CROP_FLAG not in command
    for flag, value in executor.SEALED_A1_MODEL_CLI.items():
        assert _option(command, flag) == value


def test_b200_8gpu_topology_preserves_global_batch_and_renders_torchrun(
    tmp_path: Path,
) -> None:
    verified = _production_trainer_verified(tmp_path)
    verified["recipe"].update(
        {
            "training_rng_rank_offset": True,
            "per_game_policy_weight": True,
            "per_game_policy_weight_mode": "equal",
        }
    )
    bound = executor.bind_training_topology(
        verified,
        topology=executor.B200_8GPU_DDP_TOPOLOGY,
        gpu=0,
    )
    command = executor.build_train_command(
        bound,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )

    assert command[:7] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(executor._REPO_ROOT / "tools" / "train_bc.py"),
        "--arch",
    ]
    assert bound["training_topology"] == {
        "schema_version": "a1-one-dose-training-topology-v1",
        "name": executor.B200_8GPU_DDP_TOPOLOGY,
        "world_size": 8,
        "physical_gpus": list(range(8)),
        "local_batch_size": 512,
        "grad_accum_steps": 1,
        "global_batch_size": 4096,
        "dose_preserving": True,
    }
    assert _option(command, "--batch-size") == "512"
    assert _option(command, "--grad-accum-steps") == "1"
    assert "--ddp-shard-data" not in command
    assert "--training-rng-rank-offset" in command
    assert "--per-game-policy-weight" in command
    assert _option(command, "--per-game-policy-weight-mode") == "equal"
    assert "--validation-game-seed-manifest" not in command
    assert executor._child_environment(range(8))["CUDA_VISIBLE_DEVICES"] == (
        "0,1,2,3,4,5,6,7"
    )
    assert executor._effective_global_batch_size(bound["recipe"]) == 4096
    assert executor._expected_optimizer_steps(bound) == 2


def test_production_one_dose_emits_same_trajectory_dose_snapshots(
    tmp_path: Path,
) -> None:
    verified = _production_trainer_verified(tmp_path)
    verified["recipe"] = {
        **verified["recipe"],
        "max_steps": 128,
        "batch_size": 512,
        "world_size": 8,
        "global_batch_size": 4096,
        "training_rng_rank_offset": True,
        "per_game_policy_weight": True,
        "per_game_policy_weight_mode": "equal",
    }
    command = executor._build_direct_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )

    assert _option(command, "--checkpoint-steps") == "64,96"
    assert command.count("--exact-max-steps") == 1
    assert executor.train_bc._parse_checkpoint_steps(
        _option(command, "--checkpoint-steps"), max_steps=128
    ) == (64, 96)


def test_action_cross_upgrade_is_explicit_in_one_dose_trainer_argv(
    tmp_path: Path,
) -> None:
    verified = _production_trainer_verified(tmp_path)
    verified["function_preserving_upgrade"] = {
        "module": (
            executor.architecture_upgrade.MODULE_ACTION_CROSS_ATTENTION_1
        )
    }

    command = executor._build_direct_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )

    assert _option(command, "--action-cross-attention-layers") == "1"
    assert _option(command, "--relational-action-cross-layers") == "1"


def test_positive_cap_refuses_a_training_partition_shorter_than_the_sealed_dose(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    verified["recipe"] = {
        **verified["recipe"],
        "max_steps": 3,
    }

    with pytest.raises(
        executor.ExecutorError,
        match=r"can realize only 2/3 steps",
    ):
        executor._expected_optimizer_steps(verified)


def test_positive_cap_report_must_attest_exact_max_steps(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    verified["recipe"] = {
        **verified["recipe"],
        "max_steps": 2,
    }
    checkpoint = tmp_path / "candidate.pt"
    optimizer = Path(str(checkpoint) + ".optimizer.pt")
    report = tmp_path / "report.json"
    checkpoint.write_bytes(b"candidate")
    optimizer.write_bytes(b"optimizer")
    payload = _training_report(verified, checkpoint)
    payload["exact_max_steps"] = False
    report.write_text(json.dumps(payload))
    _write_training_progress(checkpoint, payload)
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )
    execution_binding = executor._execution_binding(
        command=command, environment=executor._child_environment(0)
    )
    executor._bind_training_report(
        report,
        verified=verified,
        execution_binding=execution_binding,
    )

    with pytest.raises(executor.ExecutorError, match="report invariant drift"):
        executor._verify_training_outputs(
            checkpoint=checkpoint,
            report=report,
            verified=verified,
            execution_binding=execution_binding,
            command=command,
        )


def test_coherent_one_dose_renders_deployed_scalar_value_objective(
    tmp_path: Path,
) -> None:
    verified = _production_trainer_verified(tmp_path)
    verified["recipe"] = dict(contract.COHERENT_PUBLIC_LEARNER_TRAINING_RECIPE)
    command = executor._build_direct_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )

    assert _option(command, "--scalar-value-loss-readout") == "deployed_tanh"
    assert _option(command, "--scalar-value-loss-scale") == "1.0"
    assert _option(command, "--data-loader-workers") == "2"
    assert _option(command, "--data-loader-prefetch") == "2"
    assert _option(command, "--base-sampler") == "coverage_importance_v1"
    assert "--minimum-initial-settlement-policy-mass-fraction" not in command
    assert "--minimum-initial-road-policy-mass-fraction" not in command
    assert "--minimum-discard-policy-mass-fraction" not in command
    assert "--minimum-move-robber-policy-mass-fraction" not in command
    assert _option(command, "--train-diagnostics-every-batches") == "16"
    assert (
        _option(command, "--objective-gradient-interference-every-batches")
        == "16"
    )
    assert (
        _option(command, "--minimum-feature-learning-signal-observations") == "2"
    )
    assert "topology_residual_adapter" in _option(
        command, "--require-feature-learning-signal-modules"
    )
    assert _option(command, "--max-steps") == "0"
    assert "--exact-max-steps" not in command
    assert "--checkpoint-steps" not in command


def test_one_dose_refuses_missing_policy_target_blend_semantics(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    verified["recipe"].pop("policy_target_blend_semantics")

    with pytest.raises(
        executor.ExecutorError,
        match="refusing to silently revive sampled-action interpolation",
    ):
        executor._build_direct_train_command(
            verified,
            python=Path(sys.executable),
            checkpoint=tmp_path / "candidate.pt",
            report=tmp_path / "report.json",
        )


def test_one_dose_passes_explicit_policy_target_blend_semantics(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    verified["recipe"]["policy_target_blend_semantics"] = (
        executor.train_bc.POLICY_TARGET_BLEND_FALLBACK_V2
    )

    command = executor._build_direct_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )

    assert _option(command, "--policy-target-blend-semantics") == (
        executor.train_bc.POLICY_TARGET_BLEND_FALLBACK_V2
    )


def test_intermediate_checkpoint_steps_fail_closed() -> None:
    assert executor.train_bc._parse_checkpoint_steps(
        "64,96", max_steps=128
    ) == (64, 96)
    for raw in ("0", "64,64", "96,64", "128", "x"):
        with pytest.raises(SystemExit):
            executor.train_bc._parse_checkpoint_steps(raw, max_steps=128)


def test_topology_wrapper_refuses_nested_torchrun(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    direct = executor._build_direct_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    distributed = executor._topologize_train_command(direct, world_size=8)

    assert distributed.count("torch.distributed.run") == 1
    assert sum(Path(value).name == "train_bc.py" for value in distributed) == 1
    with pytest.raises(executor.ExecutorError, match="unwrapped direct"):
        executor._topologize_train_command(distributed, world_size=8)


def test_b200_8gpu_topology_requires_same_host_canary_receipt(
    tmp_path: Path,
) -> None:
    bound = executor.bind_training_topology(
        _verified(tmp_path),
        topology=executor.B200_8GPU_DDP_TOPOLOGY,
        gpu=0,
    )

    with pytest.raises(
        executor.ExecutorError,
        match="requires --ddp-canary-receipt",
    ):
        executor.bind_ddp_canary(bound, None)


def _ddp_canary_payload(*, created_unix_ns: int) -> dict:
    identities = [
        {
            "physical_index": rank,
            "uuid": f"GPU-{rank:02d}",
            "pci_bus_id": f"00000000:{rank:02x}:00.0",
            "name": "NVIDIA B200",
        }
        for rank in range(8)
    ]
    payload = {
        "schema_version": executor.ddp_canary.SCHEMA,
        "passed": True,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "hostname": executor.socket.gethostname(),
        "created_unix_ns": created_unix_ns,
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "ddp_shard_data": False,
        "training_rng_rank_offset": True,
        "training_rng_contracts": [
            {
                "effective_torch_seed": executor.ddp_canary.SEED + rank,
                "rank_offset_enabled": True,
            }
            for rank in range(8)
        ],
        "dropout_probe_sha256_by_rank": [
            "sha256:" + f"{rank + 1:064x}" for rank in range(8)
        ],
        "distributed_backend": "nccl",
        "cuda_collective": {
            "operation": "all_reduce_sum",
            "expected": 36.0,
            "actual_by_rank": [36.0] * 8,
            "passed": True,
        },
        "padded_global_draws": 12_288,
        "local_draws_per_rank": 1_536,
        "gpu_names": ["NVIDIA B200"] * 8,
        "gpu_identities": identities,
        "runtime_identity": {
            "schema_version": "a1-b200-learner-runtime-identity-v1",
            "python": {
                "implementation": "CPython",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "executable_sha256": executor._file_sha256(
                    Path(sys.executable).resolve()
                ),
            },
            "torch_version": "test-torch",
            "torch_cuda_version": "test-cuda",
            "cudnn_version": 90000,
            "numpy_version": np.__version__,
            "nvidia_driver_version": "test-driver",
        },
        "global_draw_sha256": "sha256:" + "a" * 64,
        "rank_slice_sha256": ["sha256:" + f"{rank + 9:064x}" for rank in range(8)],
        "tool": {
            "path": str(Path(executor.ddp_canary.__file__).resolve()),
            "sha256": executor._file_sha256(
                Path(executor.ddp_canary.__file__).resolve()
            ),
        },
        "train_bc": {
            "path": str(Path(executor.train_bc.__file__).resolve()),
            "sha256": executor._file_sha256(Path(executor.train_bc.__file__).resolve()),
        },
    }
    payload["receipt_sha256"] = executor._value_sha256(payload)
    return payload


def test_b200_8gpu_topology_accepts_exact_local_canary_receipt(
    tmp_path: Path,
) -> None:
    bound = executor.bind_training_topology(
        _verified(tmp_path),
        topology=executor.B200_8GPU_DDP_TOPOLOGY,
        gpu=0,
    )
    payload = _ddp_canary_payload(created_unix_ns=executor.time.time_ns())
    receipt = tmp_path / "ddp-canary.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    result = executor.bind_ddp_canary(bound, receipt)

    assert result["ddp_canary"]["receipt_sha256"] == payload["receipt_sha256"]
    assert result["ddp_canary"]["global_draw_sha256"] == payload[
        "global_draw_sha256"
    ]


def test_b200_8gpu_topology_rejects_stale_canary_receipt(
    tmp_path: Path,
) -> None:
    bound = executor.bind_training_topology(
        _verified(tmp_path),
        topology=executor.B200_8GPU_DDP_TOPOLOGY,
        gpu=0,
    )
    payload = _ddp_canary_payload(
        created_unix_ns=(
            executor.time.time_ns() - executor.MAX_DDP_CANARY_AGE_NS - 1
        )
    )
    receipt = tmp_path / "stale-canary.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(executor.ExecutorError, match="exact B200 DDP topology"):
        executor.bind_ddp_canary(bound, receipt)


def test_legacy_topology_rejects_ddp_canary_receipt(tmp_path: Path) -> None:
    bound = executor.bind_training_topology(
        _verified(tmp_path),
        topology=executor.LEGACY_SINGLE_GPU_TOPOLOGY,
        gpu=3,
    )

    with pytest.raises(executor.ExecutorError, match="valid only for 8-GPU"):
        executor.bind_ddp_canary(bound, tmp_path / "canary.json")


def test_production_composite_receipts_component_whole_game_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified(tmp_path)
    producer = verified["producer"]
    descriptor = tmp_path / "production-composite.json"
    descriptor.write_text("{}", encoding="utf-8")
    corpus = _FakeCompositeCorpus(
        [
                [100, 100, 101, 101],
                [200, 200, 201, 201],
                [300, 300, 301, 301],
        ]
    )
    monkeypatch.setattr(
        executor.train_bc,
        "load_teacher_data_memmap",
        lambda *_args, **_kwargs: corpus,
    )
    monkeypatch.setattr(
        executor.train_bc,
        "split_train_validation_indices",
        lambda *_args, **_kwargs: {
            "train": np.asarray([0, 1, 4, 5, 8, 9], dtype=np.int64),
            "validation": np.asarray(
                [2, 3, 6, 7, 10, 11], dtype=np.int64
            ),
        },
    )
    monkeypatch.setattr(
        executor.train_bc,
        "_training_data_fingerprint",
        lambda *_args, **_kwargs: "sha256:" + "6" * 64,
    )
    meta = _production_composite_meta(tmp_path, producer["sha256"])
    build_receipt = _production_composite_build_receipt(
        tmp_path, verified=verified, descriptor=descriptor, meta=meta
    )
    frozen_authority = {
        "schema_version": "a1-frozen-lock-verifier-authority-v1",
        "authority_sha256": "sha256:" + "7" * 64,
    }

    result = executor._verify_production_composite_inputs(
        lock=verified["lock"],
        lock_path=verified["lock_path"],
        reviewed_lock_file_sha256=None,
        recipe=verified["recipe"],
        objective=verified["objective"],
        producer=producer,
        data_path=descriptor,
        meta=meta,
        validation_path=None,
        build_receipt_path=build_receipt,
        lock_verifier_authority=frozen_authority,
    )

    split = result["validation_split_receipt"]
    assert split["aggregate"] == {
        "selected_game_count": 6,
        "training_game_count": 3,
        "validation_game_count": 3,
        "row_count": 12,
        "training_row_count": 6,
        "validation_row_count": 6,
    }
    assert [row["component_id"] for row in split["components"]] == [
        "current_producer",
        "recent_history",
        "hard_negative",
    ]
    assert result["training_row_count"] == 6
    assert result["validation_row_count"] == 6
    assert result["lock_verifier_authority"] == frozen_authority
    assert executor._input_binding(result)["lock_verifier_authority"] == (
        frozen_authority
    )
    command = executor._build_direct_train_command(
        result,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert _option(command, "--policy-kl-anchor-weight") == "0.0"
    assert _option(command, "--policy-kl-anchor-direction") == "forward"
    args = executor.train_bc.build_parser().parse_args(command[2:])
    executor.train_bc._validate_composite_learner_recipe_authorization(  # noqa: SLF001
        args, meta
    )


def test_production_composite_uses_current_trainer_not_frozen_lock_provenance(
    tmp_path: Path,
) -> None:
    verified = _production_trainer_verified(tmp_path)
    frozen_trainer = tmp_path / "frozen/tools/train_bc.py"
    frozen_trainer.parent.mkdir(parents=True)
    frozen_trainer.write_text("raise RuntimeError('historical trainer')\n")
    verified["lock"]["provenance"] = {
        "learner_code": [{"path": str(frozen_trainer)}]
    }

    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    authority = verified["trainer_authority"]
    assert command[1] == authority["path"]
    assert str(frozen_trainer) not in command
    assert authority["sha256"] == executor._file_sha256(Path(command[1]))
    assert command.count(executor.EVENT_HISTORY_ACK_FLAG) == 4
    assert command.count(executor.EVENT_HISTORY_CROP_FLAG) == 1
    assert [
        command[index + 1]
        for index, token in enumerate(command)
        if token == executor.EVENT_HISTORY_ACK_FLAG
    ] == verified["event_history_training_contract"][
        "empty_payload_inventory_acknowledgements"
    ]

    input_binding = executor._input_binding(verified)
    transaction = executor._training_transaction_sha256(
        command=command, input_binding=input_binding
    )
    assert input_binding["trainer_authority"] == authority
    changed = dict(input_binding)
    changed["trainer_authority"] = {
        **authority,
        "sha256": "sha256:" + "0" * 64,
    }
    changed["binding_sha256"] = executor._value_sha256(
        {key: value for key, value in changed.items() if key != "binding_sha256"}
    )
    assert executor._training_transaction_sha256(
        command=command, input_binding=changed
    ) != transaction


def test_production_trainer_byte_drift_is_rejected_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _production_trainer_verified(tmp_path)
    checkpoint = tmp_path / "candidate.pt"
    report = tmp_path / "report.json"
    receipt = tmp_path / "receipt.json"
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )
    trainer = Path(verified["trainer_authority"]["path"])
    original_sha = executor._file_sha256

    def drifted_sha(path: Path) -> str:
        if path.resolve() == trainer:
            return "sha256:" + "0" * 64
        return original_sha(path)

    monkeypatch.setattr(executor, "_file_sha256", drifted_sha)
    with pytest.raises(executor.ExecutorError, match="trainer authority drifted"):
        executor.execute(
            verified=verified,
            command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            gpu=0,
            runner=lambda *_args, **_kwargs: pytest.fail("trainer was launched"),
            probe=lambda _gpu: pytest.fail("hardware was probed"),
        )
    assert not executor._claim_path(verified).exists()
    assert not receipt.exists()


def test_production_trainer_authority_binds_current_science_import() -> None:
    authority = executor._current_production_trainer_authority()
    records = {
        record["relative_path"]: record for record in authority["code_surface"]
    }
    science_path = "tools/a1_current_science_contract.py"
    assert science_path in executor.PRODUCTION_TRAINER_CODE_SURFACE
    assert records[science_path]["path"] == str(
        Path(executor.current_science.__file__).resolve(strict=True)
    )
    assert "tools/a1_feature_signal_admission.py" in records


def test_canonical_parent_update_binds_12_step_8x64_recipe(tmp_path: Path) -> None:
    producer = {"path": "/checkpoint/f7.pt", "sha256": "sha256:" + "1" * 64}
    verified = {
        "contract_sha256": "sha256:" + "2" * 64,
        "data_kind": "production_composite_v2",
        "producer": producer,
        "corpus_meta_file_sha256": "sha256:" + "3" * 64,
        "composite_build_receipt": {"file_sha256": "sha256:" + "4" * 64},
        "information_contract_migration": {
            "migration": executor.information_migration.MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1,
            "source": producer,
            "receipt": {"sha256": "sha256:" + "5" * 64},
            "receipt_sha256": "sha256:" + "6" * 64,
            "forward_identical": False,
            "promotion_eligible": False,
        },
    }

    bound = executor.bind_canonical_parent_update_recipe(
        verified, executor.CANONICAL_PARENT_UPDATE_CONFIG
    )
    bound = executor.bind_training_topology(
        bound, topology=executor.B200_8GPU_DDP_TOPOLOGY, gpu=0
    )

    assert bound["recipe"]["max_steps"] == 12
    assert bound["recipe"]["optimizer"] == "adamw"
    assert bound["recipe"]["world_size"] == 8
    assert bound["recipe"]["batch_size"] == 64
    assert bound["recipe"]["global_batch_size"] == 512
    assert bound["recipe"]["trunk_lr_mult"] == 0.25
    assert bound["recipe"]["min_35m_params"] == 42_500_000
    assert bound["recipe"]["max_35m_params"] == 43_000_000
    assert bound["recipe"]["scalar_value_objective"] == "binary_win_bce"
    assert bound["recipe"]["scalar_value_loss_readout"] == "deployed_tanh"
    assert bound["recipe"]["scalar_value_loss_scale"] == 1.0
    assert bound["canonical_parent_update"]["parent_checkpoint_sha256"] == (
        producer["sha256"]
    )
    assert executor._training_report_runtime_contract(bound["recipe"]) == {  # noqa: SLF001
        "optimizer": "adamw",
        "fused_optimizer": True,
        "epochs": 999,
        "symmetry_augment": True,
    }
    assert bound["canonical_parent_update"]["checkpoint_steps"] == [8, 10]
    assert bound["training_science_commissioning"]["authorized"] is False
    assert bound["training_science_commissioning"]["go_authorized"] is False
    assert bound["training_science_commissioning"]["reason"] == (
        "v7_action_decoder_requires_fresh_commissioning"
    )
    assert bound["promotion_eligible"] is False
    assert bound["eligible_for_full_gate"] is False
    assert bound["promotion_block_reason"] == (
        "training_science_admission_unauthorized"
    )

    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"initializer")
    bound.update(
        {
            "data_path": tmp_path / "corpus",
            "trainer_authority": executor._current_production_trainer_authority(),
            "architecture_initializer": {
                "path": str(initializer),
                "sha256": executor._file_sha256(initializer),
            },
            "event_history_training_contract": {
                "training_event_history_trainable": False,
                "event_history_end_to_end_usable": False,
                "empty_payload_inventory_acknowledgements": [
                    "sha256:" + f"{index:x}" * 64 for index in range(6, 10)
                ],
            },
        }
    )
    command = executor._build_direct_train_command(  # noqa: SLF001
        bound,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert _option(command, "--checkpoint-steps") == "8,10"
    assert _option(command, "--scalar-value-objective") == "binary_win_bce"
    assert _option(command, "--scalar-value-loss-readout") == "deployed_tanh"
    assert _option(command, "--scalar-value-loss-scale") == "1.0"
    assert _option(command, "--min-35m-params") == "42500000"
    assert _option(command, "--max-35m-params") == "43000000"
    assert _option(
        command, "--minimum-initial-road-policy-mass-fraction"
    ) == "0.02"
    assert _option(command, "--minimum-discard-policy-mass-fraction") == "0.02"
    assert _option(
        command, "--minimum-move-robber-policy-mass-fraction"
    ) == "0.02"


def test_migration_claim_schema_replays_as_derived_identity(tmp_path: Path) -> None:
    contract = "sha256:" + "1" * 64
    identity = "sha256:" + "2" * 64
    claim = tmp_path / "migration.claim.json"
    payload = {
        "schema_version": executor.MIGRATION_CLAIM_SCHEMA,
        "contract_sha256": contract,
        "claim_identity_sha256": identity,
        "status": "claimed",
    }
    payload["state_sha256"] = executor._value_sha256(payload)  # noqa: SLF001
    claim.write_text(json.dumps(payload), encoding="utf-8")

    assert executor._load_claim_state(  # noqa: SLF001
        claim,
        contract_sha256=contract,
        claim_identity_sha256=identity,
    )["schema_version"] == executor.MIGRATION_CLAIM_SCHEMA


def test_canonical_parent_update_rejects_malformed_checkpoint_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_config, loaded_engine = executor.canonical_train._load_recipe(  # noqa: SLF001
        executor.CANONICAL_PARENT_UPDATE_CONFIG
    )
    drifted_engine = {**loaded_engine, "checkpoint_steps": "12,8"}
    monkeypatch.setattr(
        executor.canonical_train,
        "_load_recipe",
        lambda _path: (loaded_config, drifted_engine),
    )
    producer = {"path": "/checkpoint/f7.pt", "sha256": "sha256:" + "1" * 64}
    verified = {
        "contract_sha256": "sha256:" + "2" * 64,
        "data_kind": "production_composite_v2",
        "producer": producer,
        "corpus_meta_file_sha256": "sha256:" + "3" * 64,
        "composite_build_receipt": {"file_sha256": "sha256:" + "4" * 64},
        "information_contract_migration": {
            "migration": executor.information_migration.MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1,
            "source": producer,
            "receipt": {"sha256": "sha256:" + "5" * 64},
            "receipt_sha256": "sha256:" + "6" * 64,
            "forward_identical": False,
            "promotion_eligible": False,
        },
    }

    with pytest.raises(executor.ExecutorError, match="frontier is malformed"):
        executor.bind_canonical_parent_update_recipe(
            verified, executor.CANONICAL_PARENT_UPDATE_CONFIG
        )


def test_canonical_checkpoint_frontier_supports_twelve_step_terminal(
    tmp_path: Path,
) -> None:
    verified = _production_trainer_verified(tmp_path)
    verified["recipe"] = {**verified["recipe"], "max_steps": 12}
    verified["canonical_parent_update"] = {
        "schema_version": "a1-canonical-parent-update-authority-v2",
        "checkpoint_steps": [8],
    }
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )

    assert _option(command, "--max-steps") == "12"
    assert _option(command, "--checkpoint-steps") == "8"


def test_canonical_checkpoint_frontier_is_bound_into_failure_receipt(
    tmp_path: Path,
) -> None:
    verified = _production_trainer_verified(tmp_path)
    verified["canonical_parent_update"] = {
        "schema_version": "a1-canonical-parent-update-authority-v2",
        "checkpoint_steps": [8, 12, 16, 24, 32],
    }
    checkpoint = tmp_path / "candidate.pt"
    report = tmp_path / "report.json"
    receipt = tmp_path / "receipt.json"
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )

    with pytest.raises(executor.ExecutorError, match="exited nonzero"):
        executor.execute(
            verified=verified,
            command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            gpu=0,
            runner=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 1),
            probe=lambda _gpu: "NVIDIA B200",
        )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert _option(payload["command"], "--checkpoint-steps") == "8,12,16,24,32"
    assert payload["canonical_parent_update"]["checkpoint_steps"] == [
        8,
        12,
        16,
        24,
        32,
    ]
    assert payload["input_binding"]["canonical_parent_update"] == payload[
        "canonical_parent_update"
    ]
def test_production_failure_receipt_binds_current_trainer_authority(
    tmp_path: Path,
) -> None:
    verified = _production_trainer_verified(tmp_path)
    checkpoint = tmp_path / "candidate.pt"
    report = tmp_path / "report.json"
    receipt = tmp_path / "receipt.json"
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )

    with pytest.raises(executor.ExecutorError, match="exited nonzero"):
        executor.execute(
            verified=verified,
            command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            gpu=0,
            runner=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 1),
            probe=lambda _gpu: "NVIDIA B200",
        )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    authority = verified["trainer_authority"]
    assert payload["trainer_authority"] == authority
    assert payload["input_binding"]["trainer_authority"] == authority
    assert payload["training_transaction_sha256"] == (
        executor._training_transaction_sha256(
            command=command, input_binding=payload["input_binding"]
        )
    )


def test_production_composite_requires_atomic_build_receipt(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    descriptor = tmp_path / "production-composite.json"
    descriptor.write_text("{}", encoding="utf-8")

    with pytest.raises(executor.ExecutorError, match="requires its atomic"):
        executor._verify_production_composite_inputs(
            lock=verified["lock"],
            lock_path=verified["lock_path"],
            reviewed_lock_file_sha256=None,
            recipe=verified["recipe"],
            objective=verified["objective"],
            producer=verified["producer"],
            data_path=descriptor,
            meta=_production_composite_meta(
                tmp_path, verified["producer"]["sha256"]
            ),
            validation_path=None,
            build_receipt_path=None,
        )


def test_production_composite_rejects_tampered_build_receipt(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    descriptor = tmp_path / "production-composite.json"
    descriptor.write_text("{}", encoding="utf-8")
    meta = _production_composite_meta(tmp_path, verified["producer"]["sha256"])
    receipt = _production_composite_build_receipt(
        tmp_path, verified=verified, descriptor=descriptor, meta=meta
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["source_bindings_sha256"] = "sha256:" + "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(executor.ExecutorError, match="semantic digest drift"):
        executor._verify_production_composite_inputs(
            lock=verified["lock"],
            lock_path=verified["lock_path"],
            reviewed_lock_file_sha256=None,
            recipe=verified["recipe"],
            objective=verified["objective"],
            producer=verified["producer"],
            data_path=descriptor,
            meta=meta,
            validation_path=None,
            build_receipt_path=receipt,
        )


def test_production_composite_rejects_cross_component_game_seed_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified(tmp_path)
    producer = verified["producer"]
    descriptor = tmp_path / "production-composite.json"
    descriptor.write_text("{}", encoding="utf-8")
    corpus = _FakeCompositeCorpus(
        [
            [100, 100, 101, 101],
            [100, 100, 201, 201],
            [300, 300, 301, 301],
            [400, 400, 401, 401],
        ]
    )
    monkeypatch.setattr(
        executor.train_bc,
        "load_teacher_data_memmap",
        lambda *_args, **_kwargs: corpus,
    )
    monkeypatch.setattr(
        executor.train_bc,
        "split_train_validation_indices",
        lambda *_args, **_kwargs: {
            "train": np.asarray([0, 1, 4, 5, 8, 9, 12, 13], dtype=np.int64),
            "validation": np.asarray(
                [2, 3, 6, 7, 10, 11, 14, 15], dtype=np.int64
            ),
        },
    )

    meta = _production_composite_meta(tmp_path, producer["sha256"])
    build_receipt = _production_composite_build_receipt(
        tmp_path, verified=verified, descriptor=descriptor, meta=meta
    )
    with pytest.raises(executor.ExecutorError, match="reuses game seeds"):
        executor._verify_production_composite_inputs(
            lock=verified["lock"],
            lock_path=verified["lock_path"],
            reviewed_lock_file_sha256=None,
            recipe=verified["recipe"],
            objective=verified["objective"],
            producer=producer,
            data_path=descriptor,
            meta=meta,
            validation_path=None,
            build_receipt_path=build_receipt,
        )


def test_future_production_per_game_value_mode_is_explicit(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    verified["recipe"]["per_game_value_weight"] = True
    verified["recipe"]["per_game_value_weight_mode"] = "sqrt"
    verified["recipe"]["value_player_outcome_balance_mode"] = (
        "sampler_balanced_v1"
    )

    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )

    assert "--per-game-value-weight" in command
    assert _option(command, "--per-game-value-weight-mode") == "sqrt"
    assert (
        _option(command, "--value-player-outcome-balance-mode")
        == "sampler_balanced_v1"
    )


def test_latest_main_ablation_command_binds_inventory_ack_and_crop(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    verified["recipe"]["per_game_value_weight_mode"] = "equal"
    code_binding = _reviewed_ablation_code_binding()
    verified["learner_ablation"] = {
        "ablation_id": "new-main-arm",
        "code_binding": code_binding,
        "code_tree_sha256": code_binding["code_tree_sha256"],
        "reviewed_lock_file_sha256": "sha256:" + "9" * 64,
    }
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert command.count(executor.EVENT_HISTORY_ACK_FLAG) == 1
    assert _option(command, executor.EVENT_HISTORY_ACK_FLAG) == verified[
        "payload_inventory_sha256"
    ]
    assert command.count(executor.EVENT_HISTORY_CROP_FLAG) == 1


def test_policy_aux_active_dose_is_typed_and_command_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    code_binding = _reviewed_ablation_code_binding()
    code_sha = code_binding["code_tree_sha256"]
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda lock: code_binding,
    )
    derived = executor.bind_learner_ablation(
        verified,
        ablation_id="policy-aux-128",
        overrides_json=(
            '{"policy_aux_active_batch_size":128,'
            '"policy_aux_loss_weight":0.25}'
        ),
        reviewed_code_tree_sha256=code_sha,
    )

    assert derived["recipe"]["policy_aux_active_batch_size"] == 128
    assert derived["recipe"]["policy_aux_loss_weight"] == 0.25
    assert derived["learner_ablation"]["recipe_drift"][
        "policy_aux_active_batch_size"
    ] == {
        "contract": 0,
        "effective": 128,
    }
    command = executor.build_train_command(
        derived,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert _option(command, "--policy-aux-active-batch-size") == "128"
    assert _option(command, "--policy-aux-loss-weight") == "0.25"
    parsed = executor.train_bc.build_parser().parse_args(command[2:])
    assert executor.train_bc.TrainConfig.from_namespace(
        parsed
    ).policy_aux_active_batch_size == 128


def test_policy_game_weight_can_be_isolated_as_a_generic_ablation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    verified["recipe"]["per_game_policy_weight"] = True
    verified["recipe"]["per_game_policy_weight_mode"] = "equal"
    code_binding = _reviewed_ablation_code_binding()
    code_sha = code_binding["code_tree_sha256"]
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: code_binding,
    )

    derived = executor.bind_learner_ablation(
        verified,
        ablation_id="policy-game-weight-off",
        overrides_json=(
            '{"per_game_policy_weight":false,'
            '"per_game_policy_weight_mode":"equal"}'
        ),
        reviewed_code_tree_sha256=code_sha,
    )

    assert derived["recipe"]["per_game_policy_weight"] is False
    assert derived["learner_ablation"]["recipe_drift"][
        "per_game_policy_weight"
    ] == {"contract": True, "effective": False}
    command = executor.build_train_command(
        derived,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert "--per-game-policy-weight" not in command
    assert "--per-game-policy-weight-mode" not in command


def _descriptor_bound_production_verified(tmp_path: Path) -> tuple[dict, Path, dict]:
    verified = _production_trainer_verified(tmp_path)
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    base = {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": False,
        "promotion_eligible": True,
        "components": [],
        "learner_recipe_overrides": dict(
            executor.composite_builder.LEARNER_RECIPE_OVERRIDES
        ),
        "learner_recipe_overrides_sha256": executor._value_sha256(
            executor.composite_builder.LEARNER_RECIPE_OVERRIDES
        ),
        "policy_kl_anchor_component_ids": [],
        "policy_distillation_component_ids": list(
            executor.FRESH_POLICY_DISTILLATION_COMPONENT_IDS
        ),
        "value_training_component_ids": list(
            executor.FRESH_VALUE_TRAINING_COMPONENT_IDS
        ),
    }
    descriptor = tmp_path / "production-composite.json"
    descriptor.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    verified.update(
        {
            "data_path": descriptor.resolve(),
            "validation_path": descriptor.resolve(),
            "corpus_meta_file_sha256": executor._file_sha256(descriptor),
            "validation_file_sha256": executor._file_sha256(descriptor),
            "descriptor_fingerprint": executor._value_sha256(base),
            "data_fingerprint": executor._value_sha256(base),
            "learner_recipe_overrides": dict(
                executor.composite_builder.LEARNER_RECIPE_OVERRIDES
            ),
            "learner_recipe_overrides_sha256": executor._value_sha256(
                executor.composite_builder.LEARNER_RECIPE_OVERRIDES
            ),
        }
    )
    verified["bound_recipe"] = dict(verified["recipe"])
    verified["recipe"].update(
        executor.composite_builder.LEARNER_RECIPE_OVERRIDES
    )
    return verified, descriptor, base


def test_policy_game_weight_ablation_derives_authenticated_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, base_path, base = _descriptor_bound_production_verified(tmp_path)
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    arm = executor.bind_learner_ablation(
        verified,
        ablation_id="policy-game-weight-off-dose-1024",
        overrides_json=(
            '{"max_steps":1024,"per_game_policy_weight":false,'
            '"per_game_policy_weight_mode":"equal"}'
        ),
        reviewed_code_tree_sha256=code_sha,
    )
    derived_path = tmp_path / "run" / "candidate.pt.training-descriptor.json"
    arm = executor.bind_diagnostic_training_descriptor(
        arm, descriptor_path=derived_path
    )

    authority = arm["diagnostic_training_descriptor_authority"]
    assert set(authority["semantic_delta"]) == {"learner_recipe_overrides"}
    assert authority["learner_recipe_overrides"]["per_game_policy_weight"] is False
    assert arm["data_path"] == derived_path
    assert not derived_path.exists()
    executor._materialize_diagnostic_training_descriptor(arm)
    derived = json.loads(derived_path.read_text())
    assert derived["learner_recipe_overrides"]["per_game_policy_weight"] is False
    assert derived["learner_recipe_overrides_sha256"] == executor._value_sha256(
        derived["learner_recipe_overrides"]
    )
    assert derived["policy_distillation_component_ids"] == list(
        executor.FRESH_POLICY_DISTILLATION_COMPONENT_IDS
    )
    assert base_path.read_text() == json.dumps(base, indent=2, sort_keys=True) + "\n"
    assert derived_path.stat().st_mode & 0o777 == 0o444


def test_forced_action_type_value_map_derives_and_replays_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, base_path, _base = _descriptor_bound_production_verified(tmp_path)
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    arm = executor.bind_learner_ablation(
        verified,
        ablation_id="forced-type-value-dose",
        overrides_json=(
            '{"forced_row_value_action_type_weights":'
            '"roll=0.2,END_TURN=0.5"}'
        ),
        reviewed_code_tree_sha256=code_sha,
    )
    derived_path = tmp_path / "forced-type.training-descriptor.json"
    arm = executor.bind_diagnostic_training_descriptor(
        arm, descriptor_path=derived_path
    )
    executor._materialize_diagnostic_training_descriptor(arm)
    derived = json.loads(derived_path.read_text())
    canonical = "END_TURN=0.5,ROLL=0.2"
    assert derived["learner_recipe_overrides"][
        "forced_row_value_action_type_weights"
    ] == canonical

    monkeypatch.setattr(
        executor.train_bc,
        "_preflight_memmap_composite_descriptor",
        lambda path: {
            "diagnostic_only": False,
            "promotion_eligible": True,
            "descriptor_file_sha256": executor._file_sha256(base_path),
            "descriptor_fingerprint": executor._value_sha256(
                json.loads(base_path.read_text())
            ),
            "learner_recipe_overrides": dict(
                executor.composite_builder.LEARNER_RECIPE_OVERRIDES
            ),
        },
    )
    replayed = executor.train_bc._preflight_flywheel_diagnostic_derivative(
        derived_path.resolve(), derived
    )
    assert replayed is not None
    assert replayed["learner_recipe_overrides"][
        "forced_row_value_action_type_weights"
    ] == canonical


def test_reviewed_public_card_one_dose_renders_exact_eight_b200_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, _base_path, _base = _descriptor_bound_production_verified(tmp_path)
    verified["training_row_count"] = 600_000
    upgrade = _fake_public_card_upgrade(verified, tmp_path)
    monkeypatch.setattr(
        executor.architecture_upgrade, "verify_receipt", lambda _path: upgrade
    )
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    overrides_json = json.dumps(
        {
            "forced_action_weight": 0.0,
            "forced_row_value_action_type_weights": "END_TURN=1,ROLL=1",
            "forced_row_value_weight": 1.0,
            "max_steps": 128,
            "per_game_policy_surprise_weighting": True,
            "public_card_lr_mult": 4.0,
            "value_loss_weight": 0.25,
        }
    )

    upgraded = executor.bind_function_preserving_upgrade(
        verified, Path(upgrade["receipt"]["path"])
    )
    arm = executor.bind_learner_ablation(
        upgraded,
        ablation_id="coherent-public-card-count-v2",
        overrides_json=overrides_json,
        reviewed_code_tree_sha256=code_sha,
    )
    arm = executor.bind_diagnostic_training_descriptor(
        arm,
        descriptor_path=tmp_path / "public-card.training-descriptor.json",
    )
    arm = executor.bind_training_topology(
        arm, topology=executor.B200_8GPU_DDP_TOPOLOGY, gpu=0
    )
    command = executor.build_train_command(
        arm,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )

    assert arm["function_preserving_upgrade"]["module"] == (
        executor.architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES_V2
    )
    assert arm["function_preserving_upgrade"]["module"] != (
        executor.architecture_upgrade.MODULE_TARGET_GATHER
    )
    assert arm["recipe"]["max_steps"] == 128
    assert arm["recipe"]["world_size"] == 8
    assert arm["recipe"]["batch_size"] == 512
    assert arm["recipe"]["global_batch_size"] == 4096
    assert arm["recipe"]["resume_optimizer"] is False
    assert arm["recipe"]["public_card_lr_mult"] == pytest.approx(4.0)
    assert arm["recipe"]["moe_balance_loss_weight"] == pytest.approx(0.01)
    assert arm["recipe"]["per_game_policy_surprise_weighting"] is True
    assert arm["learner_ablation"]["recipe_drift"]["public_card_lr_mult"] == {
        "contract": 1.0,
        "effective": 4.0,
    }
    assert arm["learner_ablation"]["recipe_drift"][
        "per_game_policy_surprise_weighting"
    ] == {"contract": False, "effective": True}
    dose = executor._direct_lineage_dose(arm)
    assert dose["current_optimizer_steps"] == 128
    assert dose["current_sampled_rows"] == 524_288

    assert command[:5] == [
        str(Path(sys.executable)),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
    ]
    assert _option(command, "--max-steps") == "128"
    assert _option(command, "--batch-size") == "512"
    assert _option(command, "--public-card-lr-mult") == "4.0"
    assert _option(command, "--moe-balance-loss-weight") == "0.01"
    assert "--public-card-count-features" in command
    assert "--per-game-policy-surprise-weighting" in command
    assert "--no-resume-optimizer" in command

    # Replay the exact train_bc-side recipe comparison used by the real DDP
    # launch.  The immutable parent recipe predates these two fields, whereas
    # one_dose expands their legacy defaults before applying the interventions.
    # Both sides must normalize to the same declared shape.
    trainer_index = command.index(str(Path(executor.train_bc.__file__).resolve()))
    parsed = executor.train_bc.build_parser().parse_args(command[trainer_index + 1 :])
    trainer_path = Path(executor.train_bc.__file__).resolve()
    code_binding = {
        "schema_version": "test-a1-ablation-code-binding-v1",
        "repository_root": str(trainer_path.parents[1]),
        "records": [
            {
                "kind": "learner_code",
                "relative_path": "tools/train_bc.py",
                "path": str(trainer_path),
                "sha256": executor._file_sha256(trainer_path),
            }
        ],
    }
    code_sha = executor.train_bc._canonical_json_sha256(code_binding)
    code_binding["code_tree_sha256"] = code_sha
    parsed.a1_ablation_code_binding_json = json.dumps(
        code_binding, sort_keys=True, separators=(",", ":")
    )
    parsed.a1_ablation_code_tree_sha256 = code_sha
    parsed.a1_reviewed_lock_file_sha256 = "sha256:" + "8" * 64
    bound_recipe = dict(arm["bound_recipe"])
    bound = {
        "learner_training_recipe": bound_recipe,
        "learner_training_recipe_sha256": (
            executor.train_bc._canonical_json_sha256(bound_recipe)
        ),
        "coherent_direct_corpus": True,
        "coherent_topology": {
            "name": "b200-8gpu-ddp",
            "world_size": 8,
            "local_batch_size": 512,
            "grad_accum_steps": 1,
            "global_batch_size": 4096,
        },
    }
    effective = executor.train_bc._validate_a1_learner_training_recipe(
        parsed,
        {"enabled": True, "world_size": 8, "rank": 0, "local_rank": 0},
        bound,
    )
    assert effective == arm["recipe"]
    assert bound["learner_ablation"]["recipe_drift"]["public_card_lr_mult"] == {
        "contract": 1.0,
        "effective": 4.0,
    }


def test_public_card_lr_multiplier_refuses_every_non_card_initializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, _base_path, _base = _descriptor_bound_production_verified(tmp_path)
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    with pytest.raises(executor.ExecutorError, match="public-card"):
        executor.bind_learner_ablation(
            verified,
            ablation_id="missing-card-initializer",
            overrides_json='{"public_card_lr_mult":4.0}',
            reviewed_code_tree_sha256=code_sha,
        )

    target_gather = {
        **_fake_public_card_upgrade(verified, tmp_path),
        "module": executor.architecture_upgrade.MODULE_TARGET_GATHER,
    }
    monkeypatch.setattr(
        executor.architecture_upgrade,
        "verify_receipt",
        lambda _path: target_gather,
    )
    target_gather_bound = executor.bind_function_preserving_upgrade(
        verified, Path(target_gather["receipt"]["path"])
    )
    with pytest.raises(executor.ExecutorError, match="public-card"):
        executor.bind_learner_ablation(
            target_gather_bound,
            ablation_id="target-gather-is-not-card",
            overrides_json='{"public_card_lr_mult":4.0}',
            reviewed_code_tree_sha256=code_sha,
        )


@pytest.mark.parametrize(
    "legacy_module",
    [
        executor.architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES,
        executor.architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY,
    ],
)
def test_public_card_lr_multiplier_refuses_legacy_biased_initializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_module: str,
) -> None:
    verified, _base_path, _base = _descriptor_bound_production_verified(tmp_path)
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    legacy = {
        **_fake_public_card_upgrade(verified, tmp_path),
        "module": legacy_module,
    }
    monkeypatch.setattr(
        executor.architecture_upgrade,
        "verify_receipt",
        lambda _path: legacy,
    )
    bound = executor.bind_function_preserving_upgrade(
        verified, Path(legacy["receipt"]["path"])
    )

    with pytest.raises(executor.ExecutorError, match="bias-free"):
        executor.bind_learner_ablation(
            bound,
            ablation_id="legacy-biased-card4",
            overrides_json='{"public_card_lr_mult":4.0}',
            reviewed_code_tree_sha256=code_sha,
        )


def test_generic_ablation_can_bind_trunk_lr_and_adaptive_parent_kl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    code_binding = _reviewed_ablation_code_binding()
    code_sha = code_binding["code_tree_sha256"]
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: code_binding,
    )

    trunk = executor.bind_learner_ablation(
        verified,
        ablation_id="trunk25",
        overrides_json='{"trunk_lr_mult":0.25}',
        reviewed_code_tree_sha256=code_sha,
    )
    assert trunk["recipe"]["trunk_lr_mult"] == pytest.approx(0.25)
    assert trunk["learner_ablation"]["recipe_drift"]["trunk_lr_mult"] == {
        "contract": 1.0,
        "effective": 0.25,
    }

    shared_action = executor.bind_learner_ablation(
        verified,
        ablation_id="shared-action25",
        overrides_json='{"shared_action_lr_mult":0.25}',
        reviewed_code_tree_sha256=code_sha,
    )
    assert shared_action["recipe"]["shared_action_lr_mult"] == pytest.approx(0.25)
    assert shared_action["learner_ablation"]["recipe_drift"][
        "shared_action_lr_mult"
    ] == {
        "contract": 1.0,
        "effective": 0.25,
    }
    assert "--shared-action-lr-mult" in executor.build_train_command(
        shared_action,
        python=Path(sys.executable),
        checkpoint=tmp_path / "shared-action.pt",
        report=tmp_path / "shared-action.json",
    )

    trust = executor.bind_learner_ablation(
        verified,
        ablation_id="adaptive-parent-kl",
        overrides_json=json.dumps(
            {
                "policy_kl_anchor_direction": "forward",
                "policy_kl_target": 0.012,
                "policy_kl_dual_lr": 1.0,
                "policy_kl_max_weight": 1.0,
            }
        ),
        reviewed_code_tree_sha256=code_sha,
    )
    command = executor.build_train_command(
        trust,
        python=Path(sys.executable),
        checkpoint=tmp_path / "trust.pt",
        report=tmp_path / "trust.report.json",
    )
    assert _option(command, "--policy-kl-target") == "0.012"
    assert _option(command, "--policy-kl-dual-lr") == "1.0"
    assert _option(command, "--policy-kl-max-weight") == "1.0"
    assert _option(command, "--policy-kl-anchor-direction") == "forward"


def test_generic_ablation_accepts_v8_public_resource_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V8 can take part in causal learner comparisons after its migration."""

    verified = _verified(tmp_path)
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    code_binding = _reviewed_ablation_code_binding()
    code_sha = code_binding["code_tree_sha256"]
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: code_binding,
    )
    v8 = {
        **_fake_public_card_upgrade(verified, tmp_path),
        "module": (
            executor.architecture_upgrade.MODULE_V7_PUBLIC_CARD_EXACT_RESOURCE_RESIDUAL
        ),
    }
    monkeypatch.setattr(
        executor.architecture_upgrade,
        "verify_receipt",
        lambda _path: v8,
    )
    upgraded = executor.bind_function_preserving_upgrade(
        verified, Path(v8["receipt"]["path"])
    )

    arm = executor.bind_learner_ablation(
        upgraded,
        ablation_id="v8-value-trunk10",
        overrides_json='{"value_trunk_grad_scale":0.1}',
        reviewed_code_tree_sha256=code_sha,
        diagnostic_dose_curve=True,
    )

    assert arm["recipe"]["value_trunk_grad_scale"] == pytest.approx(0.1)
    assert arm["learner_ablation"]["recipe_drift"]["value_trunk_grad_scale"] == {
        "contract": 1.0,
        "effective": 0.1,
    }
    # A V8 experiment is meaningful only if its newly added exact-public
    # resource path is observed learning, rather than merely loading a
    # function-preserving zero-output module.
    assert arm["recipe"]["require_feature_learning_signal_modules"] == (
        "public_card_exact_resource_residual"
    )
    assert arm["recipe"]["minimum_feature_learning_signal_observations"] == 2

def test_generic_ablation_preserves_explicit_parent_moe_balance_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    historical = _verified(tmp_path)
    historical["reviewed_lock_file_sha256"] = historical["lock_file_sha256"]
    code_binding = _reviewed_ablation_code_binding()
    code_sha = code_binding["code_tree_sha256"]
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: code_binding,
    )

    historical_arm = executor.bind_learner_ablation(
        historical,
        ablation_id="historical-moe-default",
        overrides_json='{"trunk_lr_mult":0.25}',
        reviewed_code_tree_sha256=code_sha,
    )
    assert "moe_balance_loss_weight" not in historical_arm["bound_recipe"]
    assert historical_arm["recipe"]["moe_balance_loss_weight"] == pytest.approx(0.01)
    assert "moe_balance_loss_weight" not in historical_arm["learner_ablation"][
        "recipe_drift"
    ]
    historical_command = executor.build_train_command(
        historical_arm,
        python=Path(sys.executable),
        checkpoint=tmp_path / "historical.pt",
        report=tmp_path / "historical.report.json",
    )
    assert _option(historical_command, "--moe-balance-loss-weight") == "0.01"

    current = copy.deepcopy(historical)
    current["recipe"]["moe_balance_loss_weight"] = 0.0
    current_arm = executor.bind_learner_ablation(
        current,
        ablation_id="current-moe-zero",
        overrides_json='{"trunk_lr_mult":0.25}',
        reviewed_code_tree_sha256=code_sha,
    )
    assert current_arm["bound_recipe"]["moe_balance_loss_weight"] == pytest.approx(0.0)
    assert current_arm["recipe"]["moe_balance_loss_weight"] == pytest.approx(0.0)
    assert "moe_balance_loss_weight" not in current_arm["learner_ablation"][
        "recipe_drift"
    ]
    current_command = executor.build_train_command(
        current_arm,
        python=Path(sys.executable),
        checkpoint=tmp_path / "current.pt",
        report=tmp_path / "current.report.json",
    )
    assert _option(current_command, "--moe-balance-loss-weight") == "0.0"


def test_adaptive_parent_kl_ablation_requires_complete_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    with pytest.raises(executor.ExecutorError, match="complete controller"):
        executor.bind_learner_ablation(
            verified,
            ablation_id="incomplete-parent-kl",
            overrides_json='{"policy_kl_target":0.012}',
            reviewed_code_tree_sha256=code_sha,
        )


def test_target_gather_upgrade_combines_with_typed_forced_value_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, _base_path, _base = _descriptor_bound_production_verified(tmp_path)
    initializer = tmp_path / "target-gather-initializer.pt"
    initializer.write_bytes(b"function-preserving target gather")
    receipt = tmp_path / "target-gather.receipt.json"
    receipt.write_text("{}")
    upgrade = {
        "module": executor.architecture_upgrade.MODULE_TARGET_GATHER,
        "source": dict(verified["producer"]),
        "upgraded_initializer": {
            "path": str(initializer.resolve()),
            "sha256": executor._file_sha256(initializer),
        },
        "receipt_sha256": "sha256:" + "4" * 64,
        "receipt": {
            "path": str(receipt.resolve()),
            "sha256": executor._file_sha256(receipt),
        },
    }
    monkeypatch.setattr(
        executor.architecture_upgrade, "verify_receipt", lambda _path: upgrade
    )
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )

    upgraded = executor.bind_function_preserving_upgrade(verified, receipt)
    derived = executor.bind_learner_ablation(
        upgraded,
        ablation_id="coherent-public-action-gather-v1",
        overrides_json=json.dumps(
            {
                "forced_action_weight": 0.0,
                "forced_row_value_action_type_weights": "ROLL=0.25,END_TURN=0.1",
                "forced_row_value_weight": 1.0,
                "value_loss_weight": 0.25,
            }
        ),
        reviewed_code_tree_sha256=code_sha,
    )

    assert derived["architecture_initializer"] == upgrade["upgraded_initializer"]
    assert derived["function_preserving_upgrade"]["module"] == (
        executor.architecture_upgrade.MODULE_TARGET_GATHER
    )
    assert derived["recipe"]["forced_row_value_action_type_weights"] == (
        "END_TURN=0.1,ROLL=0.25"
    )
    assert derived["recipe"]["value_loss_weight"] == 0.25
    assert derived["recipe"]["resume_optimizer"] is False
    assert derived["learner_ablation"]["diagnostic_only"] is True
    assert derived["learner_ablation"]["promotion_eligible"] is False

    derived = executor.bind_diagnostic_training_descriptor(
        derived,
        descriptor_path=tmp_path / "target-gather.training-descriptor.json",
    )
    derived["trainer_authority"] = executor._current_production_trainer_authority()
    command = executor.build_train_command(
        derived,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert _option(command, "--init-checkpoint") == str(initializer.resolve())
    assert _option(command, "--forced-row-value-action-type-weights") == (
        "END_TURN=0.1,ROLL=0.25"
    )
    assert "--no-resume-optimizer" in command


def test_fresh_policy_scope_retains_selected_policy_weight_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, _base_path, _base = _descriptor_bound_production_verified(tmp_path)
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    arm = executor.bind_learner_ablation(
        verified,
        ablation_id="fresh-policy-only",
        overrides_json=(
            '{"max_steps":1024,"per_game_policy_weight":false,'
            '"per_game_policy_weight_mode":"equal"}'
        ),
        reviewed_code_tree_sha256=code_sha,
    )
    arm = executor.bind_diagnostic_training_descriptor(
        arm,
        descriptor_path=tmp_path / "fresh-policy.training-descriptor.json",
        fresh_policy_distillation_only=True,
    )

    authority = arm["diagnostic_training_descriptor_authority"]
    assert set(authority["semantic_delta"]) == {"learner_recipe_overrides"}
    assert authority["learner_recipe_overrides"]["per_game_policy_weight"] is False
    assert authority["policy_distillation_component_ids"] == list(
        executor.FRESH_POLICY_DISTILLATION_COMPONENT_IDS
    )
    assert authority["value_training_component_ids"] == list(
        executor.FRESH_VALUE_TRAINING_COMPONENT_IDS
    )
    assert arm["learner_ablation"]["training_descriptor_authority"] == authority


def test_fresh_value_scope_is_already_the_production_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, base_path, _base = _descriptor_bound_production_verified(tmp_path)
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    arm = executor.bind_learner_ablation(
        verified,
        ablation_id="fresh-value-only",
        overrides_json=(
            '{"max_steps":1024,"per_game_policy_weight":false,'
            '"per_game_policy_weight_mode":"equal"}'
        ),
        reviewed_code_tree_sha256=code_sha,
    )
    derived_path = tmp_path / "fresh-value.training-descriptor.json"
    arm = executor.bind_diagnostic_training_descriptor(
        arm,
        descriptor_path=derived_path,
        fresh_value_training_only=True,
    )

    authority = arm["diagnostic_training_descriptor_authority"]
    assert set(authority["semantic_delta"]) == {"learner_recipe_overrides"}
    assert authority["learner_recipe_overrides"]["per_game_policy_weight"] is False
    assert authority["policy_distillation_component_ids"] == list(
        executor.FRESH_POLICY_DISTILLATION_COMPONENT_IDS
    )
    assert authority["value_training_component_ids"] == list(
        executor.FRESH_VALUE_TRAINING_COMPONENT_IDS
    )

    executor._materialize_diagnostic_training_descriptor(arm)
    derived = json.loads(derived_path.read_text())
    monkeypatch.setattr(
        executor.train_bc,
        "_preflight_memmap_composite_descriptor",
        lambda path: {
            "diagnostic_only": False,
            "promotion_eligible": True,
            "descriptor_file_sha256": executor._file_sha256(base_path),
            "descriptor_fingerprint": executor._value_sha256(
                json.loads(base_path.read_text())
            ),
            "policy_distillation_component_ids": list(
                executor.FRESH_POLICY_DISTILLATION_COMPONENT_IDS
            ),
            "value_training_component_ids": list(
                executor.FRESH_VALUE_TRAINING_COMPONENT_IDS
            ),
        },
    )
    replayed = executor.train_bc._preflight_flywheel_diagnostic_derivative(
        derived_path.resolve(), derived
    )
    assert replayed is not None
    assert replayed["policy_distillation_component_ids"] == list(
        executor.FRESH_POLICY_DISTILLATION_COMPONENT_IDS
    )
    assert replayed["value_training_component_ids"] == list(
        executor.FRESH_VALUE_TRAINING_COMPONENT_IDS
    )


def test_fresh_value_scope_requires_policy_game_weight_off_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, _base_path, _base = _descriptor_bound_production_verified(tmp_path)
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    arm = executor.bind_learner_ablation(
        verified,
        ablation_id="fresh-value-with-wrong-baseline",
        overrides_json='{"max_steps":1024}',
        reviewed_code_tree_sha256=code_sha,
    )
    with pytest.raises(executor.ExecutorError, match="policy-weight-off"):
        executor.bind_diagnostic_training_descriptor(
            arm,
            descriptor_path=tmp_path / "wrong.training-descriptor.json",
            fresh_value_training_only=True,
        )


def test_generic_production_ablation_report_binding_does_not_require_aux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, _base_path, _base = _descriptor_bound_production_verified(tmp_path)
    verified["learner_ablation"] = {
        "schema_version": "a1-learner-ablation-v1",
        "ablation_id": "plain-recipe-arm",
    }
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "value_training": {},
                "a1_aux_regularization_binding": None,
                "a1_learner_ablation": None,
                "a1_effective_learner_training_recipe": None,
            }
        )
    )
    execution = {
        "schema_version": executor.REPORT_EXECUTION_BINDING_SCHEMA,
        "command_sha256": "sha256:" + "1" * 64,
        "environment": {
            key: "test" for key in executor.CHILD_ENVIRONMENT_KEYS
        },
    }
    execution["environment_sha256"] = executor._value_sha256(
        execution["environment"]
    )
    monkeypatch.setattr(executor, "_input_binding", lambda _verified: {})
    monkeypatch.setattr(
        executor,
        "_direct_lineage_dose",
        lambda _verified, report_payload=None: {"dose": "test"},
    )

    executor._bind_training_report(
        report, verified=verified, execution_binding=execution
    )
    payload = json.loads(report.read_text())
    assert payload["a1_learner_ablation"] == verified["learner_ablation"]
    assert payload["diagnostic_only"] is True
    assert payload["promotion_eligible"] is False


def test_long_policy_weight_ablation_retains_dose_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["data_kind"] = "production_composite_v2"
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    verified["recipe"]["per_game_policy_weight"] = True
    verified["recipe"]["per_game_policy_weight_mode"] = "equal"
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    derived = executor.bind_learner_ablation(
        verified,
        ablation_id="policy-game-weight-off-dose-1024",
        overrides_json=(
            '{"max_steps":1024,"per_game_policy_weight":false,'
            '"per_game_policy_weight_mode":"equal"}'
        ),
        reviewed_code_tree_sha256=code_sha,
    )
    derived["trainer_authority"] = executor._current_production_trainer_authority()
    derived["event_history_training_contract"] = {
        "empty_payload_inventory_acknowledgements": [
            derived["payload_inventory_sha256"]
        ],
        "training_event_history_trainable": False,
        "event_history_end_to_end_usable": False,
    }

    command = executor.build_train_command(
        derived,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert _option(command, "--max-steps") == "1024"
    assert _option(command, "--checkpoint-steps") == "128,256,512"


def test_pointer_aux_cannot_use_generic_ablation_labels_or_coefficients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    upgrade = _fake_aux_upgrade(verified, tmp_path)
    monkeypatch.setattr(
        executor.architecture_upgrade, "verify_receipt", lambda _path: upgrade
    )
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    upgraded = executor.bind_function_preserving_upgrade(
        verified, Path(upgrade["receipt"]["path"])
    )
    for label, weight in (("aux0", 0.0), ("aux2", 0.02), ("auxt", 0.013)):
        with pytest.raises(
            executor.ExecutorError,
            match="central warmup/geometry/pair authority",
        ):
            executor.bind_learner_ablation(
                upgraded,
                ablation_id=label,
                overrides_json=json.dumps(
                    {"aux_subgoal_loss_weight": weight}
                ),
                reviewed_code_tree_sha256=code_sha,
            )


@pytest.mark.parametrize("weight", (0.01, 0.05))
def test_matched_aux_rejects_unreviewed_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weight: float
) -> None:
    verified = _verified(tmp_path)
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    upgrade = _fake_aux_upgrade(verified, tmp_path)
    monkeypatch.setattr(
        executor.architecture_upgrade, "verify_receipt", lambda _path: upgrade
    )
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    upgraded = executor.bind_function_preserving_upgrade(
        verified, Path(upgrade["receipt"]["path"])
    )
    with pytest.raises(
        executor.ExecutorError,
        match="central warmup/geometry/pair authority",
    ):
        executor.bind_learner_ablation(
            upgraded,
            ablation_id="aux-bad",
            overrides_json=json.dumps({"aux_subgoal_loss_weight": weight}),
            reviewed_code_tree_sha256=code_sha,
        )


def test_aux_loss_ablation_without_shared_upgrade_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["reviewed_lock_file_sha256"] = verified["lock_file_sha256"]
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )
    with pytest.raises(executor.ExecutorError, match="shared aux-head initializer"):
        executor.bind_learner_ablation(
            verified,
            ablation_id="aux2",
            overrides_json='{"aux_subgoal_loss_weight":0.02}',
            reviewed_code_tree_sha256=code_sha,
        )


def _p1_executor_authority(verified: dict, *, code_sha: str) -> dict:
    recipe = dict(
        executor.aux_coordinator.canonical_p1_final_lock_authority()["base_recipe"]
    )
    recipe.update(
        {
            "world_size": 8,
            "batch_size": 512,
            "global_batch_size": 4096,
            "policy_kl_anchor_weight": 0.03,
            "sampler_seed": executor.aux_coordinator.P1_SAMPLER_SEED,
            "amp": "none",
            "epochs": 1,
            "max_steps": 128,
            "resume_optimizer": False,
        }
    )
    derived_descriptor = {"policy_kl_anchor_component_ids": ["historical_replay"]}
    derived_bytes = (
        json.dumps(derived_descriptor, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    descriptor_authority = {
        "schema_version": executor.aux_coordinator.P1_TRAINING_DESCRIPTOR_SCHEMA,
        "kind": "derived_historical_anchor",
        "filename": "p1-05-k3-training-descriptor.json",
        "base_descriptor_sha256": verified["corpus_meta_file_sha256"],
        "descriptor_file_sha256": "sha256:"
        + hashlib.sha256(derived_bytes).hexdigest(),
        "descriptor_fingerprint": executor._value_sha256(derived_descriptor),
        "policy_kl_anchor_component_ids": ["historical_replay"],
        "expected_policy_kl_anchor_eligible_rows": 100_000,
    }
    arm = {
        "arm_id": "K3",
        "target_global_equivalent_decimal": "0.03",
        "eligible_mass_decimal": "1",
        "policy_kl_anchor_weight_decimal": "0.03",
        "policy_kl_anchor_weight": 0.03,
        "effective_recipe": recipe,
        "effective_recipe_sha256": executor._value_sha256(recipe),
        "training_descriptor_authority": descriptor_authority,
    }
    allocation = {"allocation": "test"}
    claim = {
        "schema_version": "a1-p1-central-arm-claim-v1",
        "sweep_id": "sha256:" + "1" * 64,
        "arm_id": "K3",
        "prior_authority_sha256": "sha256:" + "2" * 64,
        "arm": arm,
        "allocation": allocation,
        "execution": {},
    }
    composite = {
        "descriptor_sha256": verified["corpus_meta_file_sha256"],
        "data_fingerprint": verified["data_fingerprint"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "production_sampling_receipt_sha256": verified[
            "production_sampling_receipt_sha256"
        ],
        "validation_split_receipt_sha256": verified[
            "validation_split_receipt_sha256"
        ],
        "training_game_seed_set_sha256": verified[
            "training_game_seed_set_sha256"
        ],
        "validation_game_seed_set_sha256": verified[
            "validation_game_seed_set_sha256"
        ],
        "complete_game_inputs": True,
        "category_semantics": {
            "current_producer": {"semantic": "current_producer"},
            "recent_history": {"semantic": "recovery_reference"},
            "hard_negative": {"semantic": "hard_negative"},
        },
    }
    authority = {
        "schema_version": "a1-p1-arm-executor-authority-v1",
        "sweep_id": claim["sweep_id"],
        "arm_id": "K3",
        "sweep_state_sha256": claim["prior_authority_sha256"],
        "arm_claim": claim,
        "arm": arm,
        "current_parent_authority": {"checkpoint_sha256": verified["producer"]["sha256"]},
        "composite": composite,
        "kl_eligibility_authority": {
            "sampler_identity_sha256": "sha256:" + "3" * 64,
            "sample_order_sha256": "sha256:" + "4" * 64,
            "eligible_rows": 100_000,
        },
        "training_descriptor_authority": descriptor_authority,
        "p1_sample_evidence_receipt": {
            "state_sha256": "sha256:" + "a" * 64,
            "descriptor_sha256": verified["corpus_meta_file_sha256"],
            "payload_inventory_sha256": verified["payload_inventory_sha256"],
            "category_semantics": {"semantic": "test-recovery"},
            "category_semantics_sha256": executor._value_sha256(
                {"semantic": "test-recovery"}
            ),
            "source_authority": {
                "path": "/srv/composite/source_authority.json",
                "file_sha256": "sha256:" + "8" * 64,
                "authority_sha256": "sha256:" + "9" * 64,
            },
            "sampler_identity_sha256": "sha256:" + "3" * 64,
            "sample_order_sha256": "sha256:" + "4" * 64,
            "row_set_sha256": "sha256:" + "b" * 64,
            "unique_row_count": 500_000,
            "rows_file_sha256": "sha256:" + "c" * 64,
            "sample_dose": 524_288,
            "sampler_seed": 424_242,
            "prior_rows_file_sha256": None,
            "prior_row_set_sha256": None,
            "kl_eligible_rows": 100_000,
            "kl_eligible_mass_decimal": "0.190734863281",
            "kl_ordered_evidence_sha256": "sha256:" + "d" * 64,
            "kl_eligible_evidence_sha256": "sha256:" + "e" * 64,
        },
        "recovery_authority": {"authority_sha256": "sha256:" + "f" * 64},
        "recovery_component_semantics": {"semantic": "test-recovery"},
        "native_runtime_authority": {},
        "native_learner_admission_receipt": {},
        "portable_code_identity_sha256": code_sha,
        "portable_runtime_identity_sha256": "sha256:" + "5" * 64,
        "allocation": allocation,
    }
    authority["authority_sha256"] = executor._value_sha256(authority)
    authority["state_sha256"] = executor._value_sha256(authority)
    return authority


def _published_authority(tmp_path: Path, authority: dict) -> dict:
    descriptor_authority = authority["training_descriptor_authority"]
    descriptor = tmp_path / descriptor_authority["filename"]
    descriptor.write_text(
        json.dumps(
            {"policy_kl_anchor_component_ids": ["historical_replay"]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    descriptor.chmod(0o444)
    path = (tmp_path / "p1-15-k3-executor-authority.json").resolve()
    path.write_text(json.dumps(authority, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return {
        "schema_version": executor.aux_coordinator.PUBLISHED_EXECUTOR_AUTHORITY_SCHEMA,
        "path": str(path),
        "file_sha256": executor._file_sha256(path),
        "authority": authority,
    }


def test_central_p1_binds_exact_current_parent_sampler_and_mixed_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _production_trainer_verified(tmp_path)
    verified.update(
        {
            "data_kind": "production_composite_v2",
            "trainer_authority": executor._current_production_trainer_authority(),
            "production_sampling_receipt_sha256": "sha256:" + "6" * 64,
            "validation_split_receipt_sha256": "sha256:" + "7" * 64,
            "reviewed_lock_file_sha256": verified["lock_file_sha256"],
        }
    )
    train_path = Path(executor.train_bc.__file__).resolve()
    code_binding = {
        "records": [
            {
                "kind": "learner_code",
                "relative_path": "tools/train_bc.py",
                "path": str(train_path),
                "sha256": executor._file_sha256(train_path),
            }
        ]
    }
    code_binding["code_tree_sha256"] = executor._value_sha256(code_binding)
    code_sha = code_binding["code_tree_sha256"]
    authority = _p1_executor_authority(verified, code_sha=code_sha)
    published = _published_authority(tmp_path, authority)
    monkeypatch.setattr(
        executor.aux_coordinator,
        "verify_current_parent_authority",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        executor.aux_coordinator,
        "recovery_component_semantics",
        lambda _value, _semantics: authority["recovery_component_semantics"],
    )
    monkeypatch.setattr(
        executor.aux_coordinator,
        "verify_published_executor_authority",
        lambda _path: published,
    )
    monkeypatch.setattr(
        executor.aux_coordinator, "_verify_composite", lambda value: value
    )
    monkeypatch.setattr(
        executor.aux_coordinator, "verify_allocation", lambda value: value
    )
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: code_binding,
    )
    bound = executor.bind_p1_arm(
        verified,
        authority=authority,
        published_executor_authority=published,
        reviewed_code_tree_sha256=code_sha,
    )
    assert bound["recipe"]["sampler_seed"] == 424242
    assert bound["data_path"].name == "p1-05-k3-training-descriptor.json"
    assert bound["p1_training_descriptor_authority"][
        "policy_kl_anchor_component_ids"
    ] == ["historical_replay"]
    assert bound["learner_ablation"]["diagnostic_only"] is True
    assert bound["learner_ablation"]["promotion_eligible"] is False
    assert (
        bound["learner_ablation"]["promotion_block_reason"]
        == "requires_independent_final_replication"
    )
    command = executor._build_direct_train_command(
        bound,
        python=Path(sys.executable),
        checkpoint=tmp_path / "p1.pt",
        report=tmp_path / "p1.json",
    )
    assert _option(command, "--sampler-seed") == "424242"
    assert _option(command, "--data") == str(bound["data_path"])
    assert _option(command, "--public-award-feature-contract") == "authoritative_v1"
    assert command.count("--allow-mixed-public-award-feature-contracts") == 1
    assert command.count("--no-resume-optimizer") == 1
    assert command.count("--a1-central-learner-binding-json") == 1
    assert "--a1-learner-ablation-id" not in command
    args = executor.train_bc.build_parser().parse_args(command[2:])
    args.init_checkpoint_sha256 = executor._file_sha256(
        Path(args.init_checkpoint)
    )
    immutable_recipe = bound["central_learner_binding"][
        "immutable_contract_recipe"
    ]
    train_bound = {
        "learner_training_recipe": dict(immutable_recipe),
        "learner_training_recipe_sha256": executor._value_sha256(
            immutable_recipe
        ),
    }
    effective = executor.train_bc._validate_a1_learner_training_recipe(
        args,
        {"world_size": 8, "rank": 0, "local_rank": 0, "enabled": True},
        train_bound,
    )
    assert effective == bound["recipe"]
    assert train_bound["central_learner_binding"]["stage"] == "P1"
    assert train_bound["learner_ablation"] is None


def test_central_p1_rejects_candidate_chaining_and_sampler_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified.update(
        {
            "data_kind": "production_composite_v2",
            "production_sampling_receipt_sha256": "sha256:" + "6" * 64,
            "validation_split_receipt_sha256": "sha256:" + "7" * 64,
            "reviewed_lock_file_sha256": verified["lock_file_sha256"],
        }
    )
    code_sha = "sha256:" + "8" * 64
    authority = _p1_executor_authority(verified, code_sha=code_sha)
    published = _published_authority(tmp_path, authority)
    monkeypatch.setattr(
        executor.aux_coordinator,
        "verify_current_parent_authority",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        executor.aux_coordinator,
        "recovery_component_semantics",
        lambda _value, _semantics: authority["recovery_component_semantics"],
    )
    monkeypatch.setattr(
        executor.aux_coordinator, "_verify_composite", lambda value: value
    )
    monkeypatch.setattr(
        executor.aux_coordinator, "verify_allocation", lambda value: value
    )
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda _lock: {"code_tree_sha256": code_sha, "records": []},
    )

    chained = dict(verified)
    chained["producer"] = {**verified["producer"], "sha256": "sha256:" + "9" * 64}
    with pytest.raises(executor.ExecutorError, match="exact current promoted parent"):
        executor.bind_p1_arm(
            chained,
            authority=authority,
            published_executor_authority=published,
            reviewed_code_tree_sha256=code_sha,
        )

    drifted = json.loads(json.dumps(authority))
    drifted["arm"]["effective_recipe"]["sampler_seed"] = 424244
    drifted["arm"]["effective_recipe_sha256"] = executor._value_sha256(
        drifted["arm"]["effective_recipe"]
    )
    drifted["arm_claim"]["arm"] = drifted["arm"]
    unsigned = dict(drifted)
    unsigned.pop("state_sha256")
    unsigned.pop("authority_sha256")
    drifted["authority_sha256"] = executor._value_sha256(unsigned)
    state_unsigned = dict(drifted)
    state_unsigned.pop("state_sha256")
    drifted["state_sha256"] = executor._value_sha256(state_unsigned)
    with pytest.raises(executor.ExecutorError, match="exact FP32"):
        executor.bind_p1_arm(
            verified,
            authority=drifted,
            published_executor_authority=published,
            reviewed_code_tree_sha256=code_sha,
        )


def test_report_binding_seals_exact_base_value_and_policy_doses(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    verified["recipe"]["policy_aux_active_batch_size"] = 128
    verified["recipe"]["policy_aux_loss_weight"] = 0.25
    checkpoint = tmp_path / "candidate.pt"
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_training_report(verified, checkpoint)))
    binding = executor._execution_binding(
        command=[sys.executable, "train_bc.py"],
        environment=executor._child_environment(0),
    )

    executor._bind_training_report(
        report,
        verified=verified,
        execution_binding=binding,
    )

    dose = json.loads(report.read_text())["a1_lineage_dose"]
    exposure = dose["objective_exposure"]
    assert exposure == {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": 4_097,
        "policy_base_active_sampled_rows": 1_000,
        "policy_aux_active_sampled_rows": 256,
        "policy_active_sampled_rows": 1_256,
        "value_active_sampled_rows": 4_097,
        "anchor_eligible_sampled_rows": 0,
    }
    assert dose["current_sampled_rows"] == 4_097


def test_report_binding_seals_exact_ordinary_production_objective_dose(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    checkpoint = tmp_path / "candidate.pt"
    payload = _training_report(verified, checkpoint)
    payload.update(
        {
            "training_row_draws": 4_097,
            "base_training_row_draws": 4_097,
            "policy_aux_training_row_draws": 0,
            "total_training_row_draws": 4_097,
            "policy_base_active_rows": 777,
            "policy_aux_active_rows": 0,
            "policy_total_active_rows": 777,
            "value_active_rows": 4_097,
            "policy_kl_anchor_eligible_rows": 0,
        }
    )
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload))
    binding = executor._execution_binding(
        command=[sys.executable, "train_bc.py"],
        environment=executor._child_environment(0),
    )

    executor._bind_training_report(
        report,
        verified=verified,
        execution_binding=binding,
    )

    dose = json.loads(report.read_text())["a1_lineage_dose"]
    assert dose["objective_exposure"] == {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": 4_097,
        "policy_base_active_sampled_rows": 777,
        "policy_aux_active_sampled_rows": 0,
        "policy_active_sampled_rows": 777,
        "value_active_sampled_rows": 4_097,
        "anchor_eligible_sampled_rows": 0,
    }


def test_exact_policy_dose_binding_refuses_counter_drift(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    verified["recipe"]["policy_aux_active_batch_size"] = 128
    verified["recipe"]["policy_aux_loss_weight"] = 0.25
    checkpoint = tmp_path / "candidate.pt"
    payload = _training_report(verified, checkpoint)
    payload["policy_total_active_rows"] += 1
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload))
    binding = executor._execution_binding(
        command=[sys.executable, "train_bc.py"],
        environment=executor._child_environment(0),
    )

    with pytest.raises(executor.ExecutorError, match="objective-dose arithmetic drift"):
        executor._bind_training_report(
            report,
            verified=verified,
            execution_binding=binding,
        )


def test_real_gen3_architecture_cannot_fall_back_to_cli_defaults(
    tmp_path: Path,
) -> None:
    """The real incumbent must match every checkpoint-compared CLI field."""

    verified = _verified(tmp_path)
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    parser = executor.train_bc.build_parser()
    parsed = parser.parse_args(command[2:])
    gen3 = EntityGraphConfig(
        action_size=567,
        static_action_feature_size=50,
        hidden_size=640,
        state_layers=6,
        attention_heads=8,
        dropout=0.05,
    )

    assert parser.get_default("graph_layers") == 4
    assert parsed.hidden_size == gen3.hidden_size
    assert parsed.graph_layers == gen3.state_layers == 6
    assert parsed.attention_heads == gen3.attention_heads
    assert parsed.graph_dropout == gen3.dropout
    assert (
        executor.train_bc._checkpoint_config_mismatches(
            policy_type="entity_graph", config=gen3, args=parsed
        )
        == []
    )

    omitted = list(command)
    graph_layers_index = omitted.index("--graph-layers")
    del omitted[graph_layers_index : graph_layers_index + 2]
    defaulted = parser.parse_args(omitted[2:])
    mismatches = executor.train_bc._checkpoint_config_mismatches(
        policy_type="entity_graph", config=gen3, args=defaulted
    )
    assert "graph_layers checkpoint=6 cli=4" in mismatches


def test_all_checkpoint_compared_gen3_model_knobs_are_explicit(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )

    expected = {
        "--hidden-size": "640",
        "--graph-layers": "6",
        "--attention-heads": "8",
        "--graph-dropout": "0.05",
        "--entity-state-trunk": "transformer",
        "--relational-block-pattern": "",
        "--relational-ff-size": "0",
        "--relational-bases": "4",
        "--relational-action-cross-layers": "1",
        "--latent-deliberation-steps": "0",
        "--latent-deliberation-slots": "8",
        "--moe-routed-experts": "0",
        "--moe-top-k": "2",
        "--moe-expert-ff-size": "0",
        "--value-categorical-bins": "0",
    }
    assert (
        expected.items() <= {flag: _option(command, flag) for flag in expected}.items()
    )


def test_loader_prefetch_is_outside_sealed_recipe_and_runtime_inventory(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    baseline = list(command)
    for flag in ("--data-loader-workers", "--data-loader-prefetch"):
        index = baseline.index(flag)
        del baseline[index : index + 2]

    parser = executor.train_bc.build_parser()
    parsed = parser.parse_args(command[2:])
    baseline_parsed = parser.parse_args(baseline[2:])
    effective = executor.train_bc._effective_a1_learner_training_recipe(
        parsed,
        {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
    )
    baseline_effective = executor.train_bc._effective_a1_learner_training_recipe(
        baseline_parsed,
        {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
    )
    assert effective == baseline_effective == verified["recipe"]
    assert (
        executor.train_bc.TrainConfig.from_namespace(parsed).canonical_payload()
        == executor.train_bc.TrainConfig.from_namespace(
            baseline_parsed
        ).canonical_payload()
    )
    assert "tools/a1_one_dose_train.py" not in contract.REQUIRED_RUNTIME_CODE_SUFFIXES


def test_future_rank_offset_rng_recipe_is_command_and_provenance_bound(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    verified["recipe"]["training_rng_rank_offset"] = True
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert command.count("--training-rng-rank-offset") == 1
    parsed = executor.train_bc.build_parser().parse_args(command[2:])
    effective = executor.train_bc._effective_a1_learner_training_recipe(
        parsed,
        {"enabled": True, "world_size": 8, "rank": 0, "local_rank": 0},
    )
    assert effective["training_rng_rank_offset"] is True


def test_learner_python_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    base = tmp_path / "base-python"
    base.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    base.chmod(0o755)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.symlink_to(base)

    selected = executor._lexical_python_executable(python)

    assert selected == python.absolute()
    assert selected != python.resolve()


def test_dry_run_plan_invokes_lexical_virtualenv_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    verified = _verified(tmp_path)
    base = tmp_path / "base-python"
    base.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    base.chmod(0o755)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.symlink_to(base)
    claim = tmp_path / "claim.json"
    monkeypatch.setattr(executor, "verify_training_inputs", lambda **_kwargs: verified)
    monkeypatch.setattr(executor, "_claim_path", lambda _verified: claim)
    monkeypatch.setattr(
        executor, "_require_fresh_outputs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        executor, "_require_unconsumed_contract", lambda _verified: None
    )

    result = executor.main(
        [
            "--lock",
            str(tmp_path / "lock.json"),
            "--data",
            str(tmp_path / "corpus"),
            "--validation-manifest",
            str(tmp_path / "validation.json"),
            "--checkpoint",
            str(tmp_path / "candidate.pt"),
            "--report",
            str(tmp_path / "report.json"),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--python",
            str(python),
        ]
    )

    plan = json.loads(capsys.readouterr().out)
    assert result == 0
    assert plan["command"][0] == str(python.absolute())
    assert plan["command"][0] != str(python.resolve())


def test_loader_prefetch_materializes_byte_identical_batches() -> None:
    row_count = 6
    corpus = object.__new__(executor.train_bc.MemmapCorpus)
    corpus.row_count = row_count
    corpus._eager = {
        "game_seed": np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int64),
    }
    fixed = np.arange(row_count * 4, dtype=np.float16).reshape(row_count, 2, 2)
    offsets = np.asarray([0, 2, 3, 6, 6, 7, 9], dtype=np.int64)
    ragged = np.arange(9, dtype=np.int16)
    corpus._lazy = {
        "fixed": executor.train_bc._MemmapFixedColumn(fixed, row_count),
        "ragged": executor.train_bc._MemmapRaggedColumn(
            ragged, offsets, 3, -1, np.int16, None
        ),
    }
    train_indices = np.asarray([5, 2, 0, 4, 1, 3], dtype=np.int64)
    order = np.asarray([2, 4, 0, 5, 1, 3], dtype=np.int64)
    policy_weights = np.linspace(0.1, 0.6, row_count, dtype=np.float32)
    value_weights = np.linspace(1.1, 1.6, row_count, dtype=np.float32)

    def collect(
        workers: int,
    ) -> list[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        batches = []
        for data, batch, policy, value in executor.train_bc._iterate_training_batches(
            corpus,
            order,
            train_indices,
            2,
            policy_weights,
            value_weights,
            num_workers=workers,
            prefetch=3,
        ):
            if isinstance(data, executor.train_bc.MemmapCorpus):
                materialized = {key: data[key][batch] for key in data.keys()}
                policy = policy[batch]
                value = value[batch]
            else:
                materialized = data
            batches.append((materialized, policy, value))
        return batches

    synchronous = collect(0)
    prefetched = collect(2)
    assert len(synchronous) == len(prefetched) == 3
    for batch_index, (
        (sync_data, sync_policy, sync_value),
        (prefetch_data, prefetch_policy, prefetch_value),
    ) in enumerate(zip(synchronous, prefetched, strict=True)):
        internal_prefetch_keys = {
            key for key in prefetch_data if key.startswith("_source_")
        }
        assert set(sync_data) == set(prefetch_data) - internal_prefetch_keys
        assert np.array_equal(
            prefetch_data["_source_global_row_indices"],
            train_indices[order[batch_index * 2 : (batch_index + 1) * 2]],
        )
        for key in sync_data:
            assert sync_data[key].dtype == prefetch_data[key].dtype
            assert sync_data[key].shape == prefetch_data[key].shape
            assert sync_data[key].tobytes() == prefetch_data[key].tobytes()
        assert sync_policy.tobytes() == prefetch_policy.tobytes()
        assert sync_value.tobytes() == prefetch_value.tobytes()


def test_child_environment_is_exact_allowlisted_and_ambient_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poisoned = {
        "AWS_SECRET_ACCESS_KEY": "secret",
        "CUDA_VISIBLE_DEVICES": "7",
        "LD_PRELOAD": "/tmp/inject.so",
        "LOCAL_RANK": "5",
        "PYTHONPATH": "/tmp/inject",
        "RANK": "5",
        "WORLD_SIZE": "8",
    }
    for key, value in poisoned.items():
        monkeypatch.setenv(key, value)

    environment = executor._child_environment(3)
    assert set(environment) == executor.CHILD_ENVIRONMENT_KEYS
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["PYTHONPATH"] == (
        f"{executor._REPO_ROOT / 'src'}:{executor._REPO_ROOT}"
    )
    assert not (set(environment) & set(poisoned)) - {
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
    }
    for forbidden in (
        "AWS_SECRET_ACCESS_KEY",
        "LD_PRELOAD",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
    ):
        assert forbidden not in environment


def _gpu_query_runner(
    *,
    identity: str = "NVIDIA B200, Default, 0\n",
    processes: str = "",
):
    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        output = processes if "--query-compute-apps=" in " ".join(command) else identity
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    return run


def test_b200_preflight_accepts_idle_default_or_exclusive_process() -> None:
    assert (
        executor._probe_b200(0, runner=_gpu_query_runner(), mps_probe=lambda: [])
        == "NVIDIA B200"
    )
    assert (
        executor._probe_b200(
            7,
            runner=_gpu_query_runner(identity="NVIDIA B200, Exclusive Process, 16\n"),
            mps_probe=lambda: [],
        )
        == "NVIDIA B200"
    )


@pytest.mark.parametrize(
    ("identity", "processes", "match"),
    [
        ("NVIDIA H100, Default, 0\n", "", "not exactly one B200"),
        ("NVIDIA B200, Prohibited, 0\n", "", "unsafe compute mode"),
        ("NVIDIA B200, Default, 65\n", "", "not idle"),
        ("NVIDIA B200, Default, N/A\n", "", "unparseable memory"),
        ("NVIDIA B200, Default, 0\n", "123, python, 1024\n", "compute process"),
    ],
)
def test_b200_preflight_rejects_unsafe_or_occupied_gpu(
    identity: str, processes: str, match: str
) -> None:
    with pytest.raises(executor.ExecutorError, match=match):
        executor._probe_b200(
            0,
            runner=_gpu_query_runner(identity=identity, processes=processes),
            mps_probe=lambda: [],
        )


def test_b200_preflight_rejects_manual_or_service_managed_mps() -> None:
    with pytest.raises(executor.ExecutorError, match="CUDA MPS is active"):
        executor._probe_b200(
            0,
            runner=_gpu_query_runner(),
            mps_probe=lambda: ["pid=123 executable=nvidia-cuda-mps-control"],
        )


def test_mps_process_scan_reads_exact_executable_names(tmp_path: Path) -> None:
    for pid, command in {
        "101": b"/usr/bin/nvidia-cuda-mps-control\0-f\0",
        "102": b"/usr/bin/nvidia-cuda-mps-server\0",
        "103": b"python3\0worker.py\0",
    }.items():
        process = tmp_path / pid
        process.mkdir()
        (process / "cmdline").write_bytes(command)
    (tmp_path / "not-a-pid").mkdir()

    assert executor._active_mps_processes(tmp_path) == [
        "pid=101 executable=nvidia-cuda-mps-control",
        "pid=102 executable=nvidia-cuda-mps-server",
    ]


def test_hardware_refusal_does_not_consume_sealed_dose(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    claim = executor._claim_path(verified)
    runner_called = False

    def runner(*_args, **_kwargs):
        nonlocal runner_called
        runner_called = True
        return subprocess.CompletedProcess([], 0)

    with pytest.raises(executor.ExecutorError, match="occupied"):
        executor.execute(
            verified=verified,
            command=[sys.executable, "train_bc.py"],
            checkpoint=tmp_path / "candidate.pt",
            report=tmp_path / "report.json",
            receipt=tmp_path / "receipt.json",
            gpu=0,
            runner=runner,
            probe=lambda _gpu: (_ for _ in ()).throw(
                executor.ExecutorError("occupied B200")
            ),
        )

    assert runner_called is False
    assert not claim.exists()


def test_verification_replays_lock_payload_and_validation_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}")
    data = tmp_path / "corpus"
    data.mkdir()
    (data / "corpus_meta.json").write_text("{}")
    validation_path = tmp_path / "validation.json"
    validation_path.write_text("{}")
    producer = tmp_path / "producer.pt"
    producer.write_bytes(b"producer")
    lock = _lock(producer=producer)
    calls: dict[str, object] = {}

    def fake_verify(path: Path, *, require_all_job_claims: bool = False) -> dict:
        calls["require_all_job_claims"] = require_all_job_claims
        return lock

    meta = {
        "payload_inventory_sha256": "sha256:" + "c" * 64,
        "row_count": 5000,
    }
    validation = {
        "a1_contract_sha256": _SHA,
        "file_sha256": executor._file_sha256(validation_path),
        "validation_game_seed_set_sha256": "sha256:" + "f" * 64,
        "validation_row_count": 903,
    }
    bound = {
        "learner_training_recipe": dict(contract.EXPECTED_LEARNER_TRAINING_RECIPE),
        "learner_value_objective": _objective(),
        "producer_checkpoint_sha256": lock["checkpoints"][0]["sha256"],
        "selected_game_seed_set_sha256": "sha256:" + "d" * 64,
        "training_game_seed_set_sha256": "sha256:" + "e" * 64,
    }
    monkeypatch.setattr(executor.a1_contract, "verify_lock", fake_verify)
    monkeypatch.setattr(
        executor.train_bc, "_preflight_a1_memmap_metadata", lambda *a, **k: meta
    )
    monkeypatch.setattr(
        executor.train_bc,
        "_load_validation_game_seed_manifest_for_training",
        lambda *a, **k: validation,
    )
    monkeypatch.setattr(
        executor.train_bc,
        "_validate_a1_validation_manifest_corpus_binding",
        lambda *a, **k: calls.setdefault("binding_checked", True),
    )
    monkeypatch.setattr(
        executor.train_bc,
        "load_teacher_data_memmap",
        lambda *a, **k: {"game_seed": np.asarray([1, 1, 2])},
    )
    monkeypatch.setattr(
        executor.train_bc,
        "_validate_a1_corpus_artifacts_and_seeds",
        lambda *a, **k: bound,
    )

    verified = executor.verify_training_inputs(
        lock_path=lock_path, data_path=data, validation_path=validation_path
    )
    assert calls == {"require_all_job_claims": True, "binding_checked": True}
    assert verified["contract_sha256"] == _SHA
    assert verified["recipe"]["value_lr_mult"] == 0.3


def test_frozen_training_lock_authority_bypasses_ambient_verifier_and_binds_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    frozen_repo = tmp_path / "frozen"
    frozen_repo.mkdir()
    lock = _lock()
    authority = {
        "schema_version": "a1-frozen-lock-verifier-authority-v1",
        "lock": str(lock_path.resolve()),
        "lock_file_sha256": executor._file_sha256(lock_path),
        "contract_sha256": lock["contract_sha256"],
        "frozen_repo": str(frozen_repo.resolve()),
        "verifier": str((frozen_repo / "tools/a1_pre_wave_contract.py").resolve()),
        "verifier_sha256": "sha256:" + "8" * 64,
        "require_all_job_claims": True,
        "verified_lock_sha256": "sha256:" + "9" * 64,
        "authority_sha256": "sha256:" + "1" * 64,
    }
    calls: dict[str, object] = {}

    def fake_frozen_verify(path: Path, **kwargs: object):
        calls["path"] = path
        calls.update(kwargs)
        return lock, authority

    monkeypatch.setattr(
        executor.frozen_lock_verifier, "verify_frozen_lock", fake_frozen_verify
    )
    monkeypatch.setattr(
        executor.a1_contract,
        "verify_lock",
        lambda *_args, **_kwargs: pytest.fail("ambient verifier was called"),
    )

    verified_lock, observed = executor._verify_training_lock(
        lock_path,
        frozen_repo=frozen_repo,
        frozen_verifier_sha256=authority["verifier_sha256"],
    )
    assert verified_lock == lock
    assert observed == authority
    assert calls == {
        "path": lock_path,
        "frozen_repo": frozen_repo,
        "expected_verifier_sha256": authority["verifier_sha256"],
    }

    verified = _verified(tmp_path)
    verified["lock_verifier_authority"] = authority
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    binding = executor._input_binding(verified)
    identity = executor._training_transaction_sha256(
        command=command, input_binding=binding
    )
    assert binding["lock_verifier_authority"] == authority
    assert identity.startswith("sha256:")
    drifted = dict(binding)
    drifted["lock_verifier_authority"] = {
        **authority,
        "verifier_sha256": "sha256:" + "0" * 64,
    }
    drifted["binding_sha256"] = executor._value_sha256(
        {key: value for key, value in drifted.items() if key != "binding_sha256"}
    )
    assert executor._training_transaction_sha256(
        command=command, input_binding=drifted
    ) != identity


def test_frozen_training_lock_flags_are_atomic(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    with pytest.raises(executor.ExecutorError, match="required together"):
        executor._verify_training_lock(
            lock_path,
            frozen_repo=tmp_path,
            frozen_verifier_sha256=None,
        )


def test_frozen_training_lock_can_bind_reviewed_post_wave_ablation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    lock_sha256 = executor._file_sha256(lock_path)
    frozen_repo = tmp_path / "frozen"
    frozen_repo.mkdir()
    lock = _lock()
    authority = {
        "schema_version": "a1-frozen-lock-verifier-authority-v1",
        "lock_file_sha256": lock_sha256,
    }
    monkeypatch.setattr(
        executor.frozen_lock_verifier,
        "verify_frozen_lock",
        lambda *_args, **_kwargs: (lock, authority),
    )

    verified_lock, observed = executor._verify_training_lock(
        lock_path,
        reviewed_lock_file_sha256=lock_sha256,
        frozen_repo=frozen_repo,
        frozen_verifier_sha256="sha256:" + "8" * 64,
    )
    assert verified_lock == lock
    assert observed == authority

    with pytest.raises(executor.ExecutorError, match="disagree on A1 lock bytes"):
        executor._verify_training_lock(
            lock_path,
            reviewed_lock_file_sha256="sha256:" + "0" * 64,
            frozen_repo=frozen_repo,
            frozen_verifier_sha256="sha256:" + "8" * 64,
        )


def test_execute_writes_atomic_success_receipt_and_never_resumes(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    checkpoint = tmp_path / "out" / "candidate.pt"
    report = tmp_path / "out" / "report.json"
    receipt = tmp_path / "out" / "receipt.json"
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )
    captured: dict[str, object] = {}

    def fake_runner(
        command_arg: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess:
        captured["command"] = command_arg
        captured["env"] = kwargs["env"]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"candidate")
        Path(str(checkpoint) + ".optimizer.pt").write_bytes(b"fresh adam state")
        payload = _training_report(verified, checkpoint)
        report.write_text(json.dumps(payload))
        _write_training_progress(checkpoint, payload)
        return subprocess.CompletedProcess(command_arg, 0)

    result = executor.execute(
        verified=verified,
        command=command,
        checkpoint=checkpoint,
        report=report,
        receipt=receipt,
        gpu=1,
        runner=fake_runner,
        probe=lambda gpu: "NVIDIA B200",
    )
    payload = json.loads(receipt.read_text())
    assert result["status"] == payload["status"] == "complete"
    assert payload["world_size"] == 1
    assert payload["outputs"]["steps_completed"] == 2
    assert payload["receipt_sha256"].startswith("sha256:")
    claim = executor._claim_path(verified)
    assert claim.exists()
    claim_payload = json.loads(claim.read_text())
    assert claim_payload["status"] == "complete"
    assert payload["claim"] == str(claim)
    assert payload["claim_state_sha256"] == claim_payload["state_sha256"]
    expected_environment = executor._child_environment(1)
    expected_binding = executor._execution_binding(
        command=command, environment=expected_environment
    )
    assert captured["env"] == expected_environment
    assert payload["execution_binding"] == expected_binding
    assert claim_payload["execution_binding"] == expected_binding
    report_payload = json.loads(report.read_text())
    assert report_payload[executor.REPORT_EXECUTION_BINDING_FIELD] == expected_binding
    assert payload["outputs"]["execution_binding_sha256"] == executor._value_sha256(
        expected_binding
    )
    assert payload["outputs"]["report_sha256"] == executor._file_sha256(report)
    with pytest.raises(executor.ExecutorError, match="non-fresh|already"):
        executor.execute(
            verified=verified,
            command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            gpu=1,
            runner=fake_runner,
            probe=lambda gpu: "NVIDIA B200",
        )


def test_failure_is_receipted_and_claim_remains_terminal(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    checkpoint = tmp_path / "failed" / "candidate.pt"
    report = tmp_path / "failed" / "report.json"
    receipt = tmp_path / "failed" / "receipt.json"
    command = [sys.executable, "train_bc.py"]

    with pytest.raises(executor.ExecutorError, match="exited nonzero"):
        executor.execute(
            verified=verified,
            command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            gpu=0,
            runner=lambda *a, **k: subprocess.CompletedProcess(command, 7),
            probe=lambda gpu: "NVIDIA B200",
        )
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "failed"
    assert payload["returncode"] == 7
    assert "train_bc exited nonzero" in payload["failure"]
    claim = executor._claim_path(verified)
    claim_payload = json.loads(claim.read_text())
    assert claim_payload["status"] == "failed"
    assert payload["claim"] == str(claim)
    expected_binding = executor._execution_binding(
        command=command, environment=executor._child_environment(0)
    )
    assert payload["execution_binding"] == expected_binding
    assert claim_payload["execution_binding"] == expected_binding


def test_failed_before_optimizer_derives_new_immutable_retry_claim(
    tmp_path: Path,
) -> None:
    verified, parent_claim, _parent_command = _failed_architecture_attempt(tmp_path)
    parent_claim_before = parent_claim.read_bytes()
    parent = executor._load_claim_state(
        parent_claim, contract_sha256=verified["contract_sha256"]
    )
    parent_receipt = Path(parent["receipt_target"])
    parent_receipt_before = parent_receipt.read_bytes()
    checkpoint = tmp_path / "attempt-r2" / "candidate.pt"
    report = tmp_path / "attempt-r2" / "training.report.json"
    receipt = tmp_path / "attempt-r2" / "training.receipt.json"
    retry_contract = tmp_path / "attempt-r2" / "learner-retry.contract.json"
    retry_command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )

    derived = executor.authorize_failed_before_optimizer_retry(
        verified=verified,
        parent_claim=parent_claim,
        retry_command=retry_command,
        checkpoint=checkpoint,
        report=report,
        receipt=receipt,
        retry_contract_path=retry_contract,
        publish=True,
    )

    assert derived["contract_sha256"] == verified["contract_sha256"]
    assert derived["claim_identity_sha256"] != verified["contract_sha256"]
    assert executor._claim_path(derived) != parent_claim
    assert not executor._claim_path(derived).exists()
    assert (
        derived["retry_contract"]["preserved_bindings"]["parent_contract_sha256"]
        == verified["contract_sha256"]
    )
    assert derived["retry_contract"]["parent"]["pre_optimizer_proof"] == {
        "kind": "replayed_init_checkpoint_architecture_preflight",
        "mismatches": ["graph_layers checkpoint=6 cli=4"],
        "optimizer_steps": 0,
        "outputs": None,
    }
    assert json.loads(retry_contract.read_text()) == derived["retry_contract"]
    assert parent_claim.read_bytes() == parent_claim_before
    assert parent_receipt.read_bytes() == parent_receipt_before

    # The new identity can acquire exactly one independent claim; the parent
    # remains terminal and cannot be overwritten or mistaken for the retry.
    retry_claim = executor._claim_attempt(
        derived,
        {
            "schema_version": executor.RETRY_CLAIM_SCHEMA,
            "status": "claimed",
            "contract_sha256": verified["contract_sha256"],
            "claim_identity_sha256": derived["claim_identity_sha256"],
        },
    )
    retry_state = executor._load_claim_state(
        retry_claim,
        contract_sha256=verified["contract_sha256"],
        claim_identity_sha256=derived["claim_identity_sha256"],
    )
    assert retry_state["status"] == "claimed"
    assert retry_claim != parent_claim


def test_production_preflight_serialization_failure_derives_typed_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, parent_claim, _ = _failed_production_preflight_attempt(
        tmp_path, monkeypatch
    )
    parent_claim_before = parent_claim.read_bytes()
    parent = executor._load_claim_state(
        parent_claim, contract_sha256=verified["contract_sha256"]
    )
    parent_receipt = Path(parent["receipt_target"])
    parent_receipt_before = parent_receipt.read_bytes()
    checkpoint = tmp_path / "production-r2" / "candidate.pt"
    report = tmp_path / "production-r2" / "training.report.json"
    receipt = tmp_path / "production-r2" / "training.receipt.json"
    retry_contract = tmp_path / "production-r2" / "retry.contract.json"
    retry_command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )

    derived = executor.authorize_failed_before_optimizer_retry(
        verified=verified,
        parent_claim=parent_claim,
        retry_command=retry_command,
        checkpoint=checkpoint,
        report=report,
        receipt=receipt,
        retry_contract_path=retry_contract,
        publish=True,
    )

    contract_payload = derived["retry_contract"]
    assert contract_payload["retry_identity"]["repair_kind"] == (
        executor.PRODUCTION_PREFLIGHT_RETRY_REPAIR_KIND
    )
    assert contract_payload["parent"]["pre_optimizer_proof"][
        "buggy_train_bc_sha256"
    ] == executor.BUGGY_PRODUCTION_PREFLIGHT_TRAINER_SHA256
    assert contract_payload["retry"]["fixed_train_bc_sha256"] == (
        executor.FIXED_PRODUCTION_PREFLIGHT_TRAINER_SHA256
    )
    assert contract_payload["preserved_bindings"]["parent_ddp_canary"] != (
        contract_payload["retry"]["ddp_canary"]
    )
    assert contract_payload["retry"]["allowed_argv_drift"] == [
        "checkpoint",
        "report",
    ]
    assert executor._claim_path(derived) != parent_claim
    assert not executor._claim_path(derived).exists()
    assert json.loads(retry_contract.read_text()) == contract_payload
    assert parent_claim.read_bytes() == parent_claim_before
    assert parent_receipt.read_bytes() == parent_receipt_before

    retry_claim = executor._claim_attempt(
        derived,
        {
            "schema_version": executor.RETRY_CLAIM_SCHEMA,
            "status": "claimed",
            "contract_sha256": verified["contract_sha256"],
            "claim_identity_sha256": derived["claim_identity_sha256"],
        },
    )
    assert retry_claim != parent_claim
    with pytest.raises(executor.ExecutorError, match="already exists"):
        executor._claim_attempt(
            derived,
            {
                "schema_version": executor.RETRY_CLAIM_SCHEMA,
                "status": "claimed",
                "contract_sha256": verified["contract_sha256"],
                "claim_identity_sha256": derived["claim_identity_sha256"],
            },
        )


def test_production_preflight_retry_refuses_learner_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, parent_claim, _ = _failed_production_preflight_attempt(
        tmp_path, monkeypatch
    )
    checkpoint = tmp_path / "production-r2" / "candidate.pt"
    report = tmp_path / "production-r2" / "training.report.json"
    retry_command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )
    retry_command = _replace_option(retry_command, "--lr", "0.00012")

    with pytest.raises(executor.ExecutorError, match="learner semantics"):
        executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=retry_command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / "production-r2" / "receipt.json",
            retry_contract_path=tmp_path / "production-r2" / "retry.json",
            publish=False,
        )


def test_production_preflight_retry_refuses_any_parent_training_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, parent_claim, parent_command = _failed_production_preflight_attempt(
        tmp_path, monkeypatch
    )
    parent_args = executor._train_command_namespace(parent_command)
    parent_progress = Path(str(parent_args.checkpoint) + ".training-progress.json")
    parent_progress.write_text("partial", encoding="utf-8")
    checkpoint = tmp_path / "production-r2" / "candidate.pt"
    report = tmp_path / "production-r2" / "training.report.json"
    retry_command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )

    with pytest.raises(executor.ExecutorError, match="zero-output/zero-step"):
        executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=retry_command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / "production-r2" / "receipt.json",
            retry_contract_path=tmp_path / "production-r2" / "retry.json",
            publish=False,
        )


def test_transport_failure_derives_one_exact_chained_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, parent_claim, _, first_contract = (
        _failed_production_transport_attempt(tmp_path, monkeypatch)
    )
    parent_before = parent_claim.read_bytes()
    first_contract_before = first_contract.read_bytes()

    def authorize(suffix: str) -> dict:
        checkpoint = tmp_path / suffix / "candidate.pt"
        report = tmp_path / suffix / "training.report.json"
        command = executor.build_train_command(
            verified,
            python=Path(sys.executable),
            checkpoint=checkpoint,
            report=report,
        )
        return executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / suffix / "training.receipt.json",
            retry_contract_path=tmp_path / suffix / "retry.contract.json",
            publish=False,
        )

    first = authorize("production-r3-a")
    second = authorize("production-r3-b")
    contract = first["retry_contract"]
    assert contract["retry_identity"]["repair_kind"] == (
        executor.PRODUCTION_PREFLIGHT_TRANSPORT_RETRY_REPAIR_KIND
    )
    assert contract["parent"]["causal_parent_retry_contract"][
        "retry_identity_sha256"
    ] == contract["preserved_bindings"]["first_retry_identity_sha256"]
    assert contract["retry"]["fixed_train_bc_sha256"] == (
        executor.FIXED_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256
    )
    assert contract["retry"]["transport"] == {
        "schema_version": executor.train_bc.A1_PREFLIGHT_STORE_PACKET_SCHEMA,
        "chunk_bytes": executor.train_bc.A1_PREFLIGHT_STORE_CHUNK_BYTES,
        "publish_order": "chunks_then_authenticated_manifest",
    }
    assert first["claim_identity_sha256"] == second["claim_identity_sha256"]
    assert executor._claim_path(first) == executor._claim_path(second)
    assert parent_claim.read_bytes() == parent_before
    assert first_contract.read_bytes() == first_contract_before


def test_transport_retry_refuses_learner_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, parent_claim, _, _ = _failed_production_transport_attempt(
        tmp_path, monkeypatch
    )
    checkpoint = tmp_path / "production-r3" / "candidate.pt"
    report = tmp_path / "production-r3" / "training.report.json"
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )
    command = _replace_option(command, "--lr", "0.00012")
    with pytest.raises(executor.ExecutorError, match="learner semantics"):
        executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / "production-r3" / "receipt.json",
            retry_contract_path=tmp_path / "production-r3" / "retry.json",
            publish=False,
        )


def test_transport_retry_refuses_parent_training_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, parent_claim, parent_command, _ = (
        _failed_production_transport_attempt(tmp_path, monkeypatch)
    )
    parent_args = executor._train_command_namespace(parent_command)
    progress = Path(str(parent_args.checkpoint) + ".training-progress.json")
    progress.write_text("partial", encoding="utf-8")
    checkpoint = tmp_path / "production-r3" / "candidate.pt"
    report = tmp_path / "production-r3" / "training.report.json"
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )
    with pytest.raises(executor.ExecutorError, match="zero-output/zero-step"):
        executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / "production-r3" / "receipt.json",
            retry_contract_path=tmp_path / "production-r3" / "retry.json",
            publish=False,
        )


def test_transport_retry_refuses_parent_contract_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, parent_claim, _, first_contract = (
        _failed_production_transport_attempt(tmp_path, monkeypatch)
    )
    first_contract.chmod(0o644)
    first_contract.write_bytes(first_contract.read_bytes() + b"\n")
    checkpoint = tmp_path / "production-r3" / "candidate.pt"
    report = tmp_path / "production-r3" / "training.report.json"
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )
    with pytest.raises(executor.ExecutorError, match="bytes/digest drift"):
        executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / "production-r3" / "receipt.json",
            retry_contract_path=tmp_path / "production-r3" / "retry.json",
            publish=False,
        )


def test_retry_identity_is_stable_across_output_path_changes(tmp_path: Path) -> None:
    verified, parent_claim, _ = _failed_architecture_attempt(tmp_path)

    def authorize(suffix: str) -> dict:
        checkpoint = tmp_path / suffix / "candidate.pt"
        report = tmp_path / suffix / "report.json"
        command = executor.build_train_command(
            verified,
            python=Path(sys.executable),
            checkpoint=checkpoint,
            report=report,
        )
        return executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / suffix / "receipt.json",
            retry_contract_path=tmp_path / suffix / "retry.json",
            publish=False,
        )

    first = authorize("r2-a")
    second = authorize("r2-b")
    assert first["claim_identity_sha256"] == second["claim_identity_sha256"]
    assert executor._claim_path(first) == executor._claim_path(second)
    assert (
        first["retry_contract"]["retry_contract_sha256"]
        != second["retry_contract"]["retry_contract_sha256"]
    )


def test_retry_refuses_loader_drift(tmp_path: Path) -> None:
    verified, parent_claim, _ = _failed_architecture_attempt(tmp_path)
    checkpoint = tmp_path / "r2" / "candidate.pt"
    report = tmp_path / "r2" / "report.json"
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )
    command = _replace_option(command, "--data-loader-workers", "3")
    with pytest.raises(executor.ExecutorError, match="non-architecture learner"):
        executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / "r2" / "receipt.json",
            retry_contract_path=tmp_path / "r2" / "retry.json",
            publish=False,
        )


def test_retry_requires_literal_parent_omission_and_corrected_six(
    tmp_path: Path,
) -> None:
    verified, parent_claim, _ = _failed_architecture_attempt(tmp_path)
    checkpoint = tmp_path / "r2" / "candidate.pt"
    report = tmp_path / "r2" / "report.json"
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )
    command = _replace_option(command, "--graph-layers", "06")
    with pytest.raises(executor.ExecutorError, match="literal --graph-layers 6"):
        executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / "r2" / "receipt.json",
            retry_contract_path=tmp_path / "r2" / "retry.json",
            publish=False,
        )


def test_retry_refuses_parent_with_any_output_artifact(tmp_path: Path) -> None:
    verified, parent_claim, _ = _failed_architecture_attempt(tmp_path)
    parent = executor._load_claim_state(
        parent_claim, contract_sha256=verified["contract_sha256"]
    )
    parent_checkpoint = Path(_option(parent["command"], "--checkpoint"))
    parent_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    parent_checkpoint.write_bytes(b"partial output")
    retry_checkpoint = tmp_path / "r2" / "candidate.pt"
    retry_report = tmp_path / "r2" / "report.json"
    retry_command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=retry_checkpoint,
        report=retry_report,
    )

    with pytest.raises(executor.ExecutorError, match="zero-output/zero-step"):
        executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=retry_command,
            checkpoint=retry_checkpoint,
            report=retry_report,
            receipt=tmp_path / "r2" / "receipt.json",
            retry_contract_path=tmp_path / "r2" / "retry.json",
            publish=False,
        )


def test_retry_refuses_non_architecture_semantic_change(tmp_path: Path) -> None:
    verified, parent_claim, _ = _failed_architecture_attempt(tmp_path)
    checkpoint = tmp_path / "r2" / "candidate.pt"
    report = tmp_path / "r2" / "report.json"
    retry_command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )
    retry_command = _replace_option(retry_command, "--lr", "0.123")

    with pytest.raises(executor.ExecutorError, match="non-architecture learner"):
        executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=retry_command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / "r2" / "receipt.json",
            retry_contract_path=tmp_path / "r2" / "retry.json",
            publish=False,
        )


def test_retry_refuses_command_that_still_fails_architecture_preflight(
    tmp_path: Path,
) -> None:
    verified, parent_claim, parent_command = _failed_architecture_attempt(tmp_path)
    checkpoint = tmp_path / "r2" / "candidate.pt"
    report = tmp_path / "r2" / "report.json"
    retry_command = list(parent_command)
    retry_command = _replace_option(retry_command, "--checkpoint", str(checkpoint))
    retry_command = _replace_option(retry_command, "--report", str(report))

    with pytest.raises(executor.ExecutorError, match="still fails architecture"):
        executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=retry_command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / "r2" / "receipt.json",
            retry_contract_path=tmp_path / "r2" / "retry.json",
            publish=False,
        )


def test_retry_refuses_parent_lock_or_data_binding_drift(tmp_path: Path) -> None:
    verified, parent_claim, _ = _failed_architecture_attempt(tmp_path)
    Path(verified["lock_path"]).write_text("mutated lock")
    checkpoint = tmp_path / "r2" / "candidate.pt"
    report = tmp_path / "r2" / "report.json"
    retry_command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )

    with pytest.raises(executor.ExecutorError, match="sealed retry binding drift"):
        executor.authorize_failed_before_optimizer_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=retry_command,
            checkpoint=checkpoint,
            report=report,
            receipt=tmp_path / "r2" / "receipt.json",
            retry_contract_path=tmp_path / "r2" / "retry.json",
            publish=False,
        )


def test_contract_claim_blocks_a_second_receipt_and_output_set(tmp_path: Path) -> None:
    verified = _verified(tmp_path)

    def run_once(suffix: str) -> None:
        checkpoint = tmp_path / suffix / "candidate.pt"
        report = tmp_path / suffix / "report.json"
        receipt = tmp_path / suffix / "receipt.json"
        command = executor.build_train_command(
            verified,
            python=Path(sys.executable),
            checkpoint=checkpoint,
            report=report,
        )

        def runner(command_arg, **_kwargs):
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"candidate")
            Path(str(checkpoint) + ".optimizer.pt").write_bytes(b"optimizer")
            payload = _training_report(verified, checkpoint)
            report.write_text(json.dumps(payload))
            _write_training_progress(checkpoint, payload)
            return subprocess.CompletedProcess(command_arg, 0)

        executor.execute(
            verified=verified,
            command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            gpu=0,
            runner=runner,
            probe=lambda _gpu: "NVIDIA B200",
        )

    run_once("first")
    with pytest.raises(executor.ExecutorError, match="claim already exists"):
        run_once("second")


def test_receipt_publication_failure_leaves_terminal_complete_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    checkpoint = tmp_path / "out" / "candidate.pt"
    report = tmp_path / "out" / "report.json"
    receipt = tmp_path / "out" / "receipt.json"
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )

    def runner(command_arg, **_kwargs):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"candidate")
        Path(str(checkpoint) + ".optimizer.pt").write_bytes(b"optimizer")
        payload = _training_report(verified, checkpoint)
        report.write_text(json.dumps(payload))
        _write_training_progress(checkpoint, payload)
        return subprocess.CompletedProcess(command_arg, 0)

    receipt_writer = executor._write_receipt_no_clobber
    monkeypatch.setattr(
        executor,
        "_write_receipt_no_clobber",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("receipt disk error")),
    )
    with pytest.raises(OSError, match="receipt disk error"):
        executor.execute(
            verified=verified,
            command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            gpu=0,
            runner=runner,
            probe=lambda _gpu: "NVIDIA B200",
        )

    claim = executor._claim_path(verified)
    state = executor._load_claim_state(
        claim, contract_sha256=verified["contract_sha256"]
    )
    assert state["status"] == "complete"
    assert state["outputs"]["checkpoint_sha256"] == executor._file_sha256(checkpoint)
    monkeypatch.setattr(executor, "_write_receipt_no_clobber", receipt_writer)

    recovered = executor.execute(
        verified=verified,
        command=command,
        checkpoint=checkpoint,
        report=report,
        receipt=receipt,
        gpu=0,
        runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal receipt recovery must not rerun training")
        ),
        probe=lambda _gpu: (_ for _ in ()).throw(
            AssertionError("terminal receipt recovery must not reprobe a GPU")
        ),
    )
    assert recovered["status"] == "complete"
    assert recovered["claim_state_sha256"] == state["state_sha256"]
    assert receipt.is_file()
    assert receipt.stat().st_mode & 0o777 == 0o444
    with pytest.raises(executor.ExecutorError, match="claim already exists"):
        executor._claim_attempt(
            verified,
            {
                "schema_version": executor.CLAIM_SCHEMA,
                "status": "claimed",
                "contract_sha256": verified["contract_sha256"],
            },
        )


def test_retry_contract_publication_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "retry.contract.json"
    payload = {
        "schema_version": executor.RETRY_CONTRACT_SCHEMA,
        "retry_contract_sha256": "sha256:" + "1" * 64,
    }

    executor._write_retry_contract_no_clobber(path, payload)

    assert json.loads(path.read_text()) == payload
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(executor.ExecutorError, match="refusing to overwrite"):
        executor._write_retry_contract_no_clobber(path, payload)


def test_production_strict_checkpoint_reconstruction_is_mandatory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"not-used-by-mocked-loader")

    class _Parameter:
        def numel(self) -> int:
            return 35_041_353

    class _Model:
        def parameters(self):
            return [_Parameter()]

    class _Policy:
        model = _Model()

    observed: dict[str, object] = {}

    def strict_load(path, **kwargs):
        observed.update({"path": path, **kwargs})
        return _Policy()

    monkeypatch.setattr(EntityGraphPolicy, "load", strict_load)
    executor._strict_load_production_entity_checkpoint(
        checkpoint, where="test production checkpoint"
    )
    assert observed == {
        "path": checkpoint,
        "device": "cpu",
        "strict_metadata": True,
        "allow_missing_optional_parameters": False,
    }

    monkeypatch.setattr(
        EntityGraphPolicy,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("missing model tensor")
        ),
    )
    with pytest.raises(executor.ExecutorError, match="cannot strict-load"):
        executor._strict_load_production_entity_checkpoint(
            checkpoint, where="test production checkpoint"
        )


def test_production_acceptance_rejects_self_consistent_validation_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"durable checkpoint")
    runtime = {"repo_commit": "current", "trainer": "bound"}
    actual_model_sha256 = "sha256:" + "1" * 64
    replayed_model_sha256 = "sha256:" + "2" * 64
    monkeypatch.setattr(
        executor.train_bc,
        "_checkpoint_model_tensor_state_sha256",
        lambda _path: actual_model_sha256,
    )
    expected = executor.train_bc.objective_matched_validation_evaluation_identity(
        model_state_sha256=actual_model_sha256,
        runtime_binding=runtime,
        epoch=1,
        optimizer_step=32,
    )
    executor._verify_objective_matched_evaluation_identity(  # noqa: SLF001
        provenance=expected,
        checkpoint=checkpoint,
        runtime_binding=runtime,
        epoch=1,
        optimizer_step=32,
    )

    replayed = executor.train_bc.objective_matched_validation_evaluation_identity(
        model_state_sha256=replayed_model_sha256,
        runtime_binding=runtime,
        epoch=1,
        optimizer_step=32,
    )
    with pytest.raises(
        executor.ExecutorError,
        match="durable checkpoint/runtime/step identity",
    ):
        executor._verify_objective_matched_evaluation_identity(  # noqa: SLF001
            provenance=replayed,
            checkpoint=checkpoint,
            runtime_binding=runtime,
            epoch=1,
            optimizer_step=32,
        )


@pytest.mark.parametrize("alias", ["checkpoint", "report", "receipt"])
def test_output_paths_cannot_alias_the_contract_claim(
    tmp_path: Path, alias: str
) -> None:
    verified = _verified(tmp_path)
    claim = executor._claim_path(verified)
    checkpoint = tmp_path / "out" / "candidate.pt"
    report = tmp_path / "out" / "report.json"
    receipt = tmp_path / "out" / "receipt.json"
    if alias == "checkpoint":
        checkpoint = claim
    elif alias == "report":
        report = claim
    else:
        receipt = claim

    runner_called = False

    def runner(*_args, **_kwargs):
        nonlocal runner_called
        runner_called = True
        return subprocess.CompletedProcess([], 0)

    with pytest.raises(executor.ExecutorError, match="claim path"):
        executor.execute(
            verified=verified,
            command=[sys.executable, "train_bc.py"],
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            gpu=0,
            runner=runner,
            probe=lambda _gpu: "NVIDIA B200",
        )
    assert runner_called is False
    assert not claim.exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("checkpoint", "/wrong/candidate.pt"),
        ("init_checkpoint_sha256", "sha256:" + "0" * 64),
        ("data_fingerprint", "sha256:" + "0" * 64),
        ("steps_completed", 1),
        ("a1_training_game_seed_set_sha256", "sha256:" + "0" * 64),
        ("hidden_size", 512),
        ("graph_layers", 4),
        ("attention_heads", 4),
        ("graph_dropout", 0.1),
    ],
)
def test_output_report_must_semantically_prove_the_sealed_dose(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    verified = _verified(tmp_path)
    checkpoint = tmp_path / "candidate.pt"
    optimizer = Path(str(checkpoint) + ".optimizer.pt")
    report = tmp_path / "report.json"
    checkpoint.write_bytes(b"candidate")
    optimizer.write_bytes(b"optimizer")
    payload = _training_report(verified, checkpoint)
    payload[field] = bad_value
    report.write_text(json.dumps(payload))
    _write_training_progress(checkpoint, payload)
    command = [sys.executable, "train_bc.py"]
    execution_binding = executor._execution_binding(
        command=command, environment=executor._child_environment(0)
    )
    executor._bind_training_report(
        report,
        verified=verified,
        execution_binding=execution_binding,
    )

    with pytest.raises(executor.ExecutorError, match="report invariant drift"):
        executor._verify_training_outputs(
            checkpoint=checkpoint,
            report=report,
            verified=verified,
            execution_binding=execution_binding,
        )


def test_report_binding_rejects_child_spoof_and_verifier_rejects_drift(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    checkpoint = tmp_path / "candidate.pt"
    optimizer = Path(str(checkpoint) + ".optimizer.pt")
    report = tmp_path / "report.json"
    checkpoint.write_bytes(b"candidate")
    optimizer.write_bytes(b"optimizer")
    command = [sys.executable, "train_bc.py"]
    binding = executor._execution_binding(
        command=command, environment=executor._child_environment(0)
    )

    spoofed = _training_report(verified, checkpoint)
    spoofed[executor.REPORT_EXECUTION_BINDING_FIELD] = binding
    report.write_text(json.dumps(spoofed))
    with pytest.raises(executor.ExecutorError, match="pre-populated"):
        executor._bind_training_report(
            report,
            verified=verified,
            execution_binding=binding,
        )

    payload = _training_report(verified, checkpoint)
    report.write_text(json.dumps(payload))
    _write_training_progress(checkpoint, payload)
    executor._bind_training_report(
        report,
        verified=verified,
        execution_binding=binding,
    )
    drifted = dict(binding)
    drifted["command_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(executor.ExecutorError, match="does not bind"):
        executor._verify_training_outputs(
            checkpoint=checkpoint,
            report=report,
            verified=verified,
            execution_binding=drifted,
        )


def test_cli_is_dry_run_by_default() -> None:
    args = executor.build_parser().parse_args(
        [
            "--lock",
            "lock.json",
            "--data",
            "corpus",
            "--validation-manifest",
            "validation.json",
            "--checkpoint",
            "candidate.pt",
            "--report",
            "report.json",
            "--receipt",
            "receipt.json",
        ]
    )
    assert args.go is False


def test_direct_composite_parent_upgrade_renders_complete_topology() -> None:
    command = ["python", "train_bc.py"]
    executor._append_current_parent_topology_cli(  # noqa: SLF001
        command,
        executor.architecture_upgrade.MODULE_CURRENT_V5_SPLIT1_TOPOLOGY_ONLY,
    )

    assert command == [
        "python",
        "train_bc.py",
        "--action-target-gather",
        "--static-action-residual",
        "--legal-action-value-residual",
        "--legal-action-value-set-statistics",
        "--public-card-count-features",
        "--no-public-card-count-residual-bias",
        "--meaningful-public-history",
        "--event-history-limit",
        "64",
        "--meaningful-public-history-pooling",
        "ordered_attention_v2",
        "--meaningful-public-history-target-gather",
        "--public-rule-state-features",
        "--entity-feature-adapter-version",
        "rust_entity_adapter_v5_meaningful_history_v2",
        "--value-tower-split-layers",
        "1",
        "--topology-residual-adapter",
    ]
