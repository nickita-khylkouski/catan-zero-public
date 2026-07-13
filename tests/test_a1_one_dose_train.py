from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools import a1_one_dose_train as executor
from tools import a1_pre_wave_contract as contract
from catan_zero.rl.entity_token_policy import EntityGraphConfig


_SHA = "sha256:" + "a" * 64


def _objective() -> dict[str, object]:
    return {
        "objective": "mse",
        "value_readout": "scalar",
        "value_categorical_bins": None,
        "hlgauss_sigma_ratio": None,
    }


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
    return {
        "lock": lock,
        "lock_path": lock_path,
        "lock_file_sha256": executor._file_sha256(lock_path),
        "contract_sha256": _SHA,
        "recipe": dict(contract.EXPECTED_LEARNER_TRAINING_RECIPE),
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


def _training_report(
    verified: dict, checkpoint: Path, *, steps_completed: int = 2
) -> dict:
    recipe = verified["recipe"]
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
        "max_steps": 0,
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
        self.offsets = np.cumsum(
            [0, *(len(seeds) for seeds in component_seed_rows)], dtype=np.int64
        )

    def __len__(self) -> int:
        return int(self.offsets[-1])


def _production_composite_meta(tmp_path: Path, producer_sha256: str) -> dict:
    component_ids = [
        "current_producer",
        "recent_history",
        "hard_negative",
        "historical_replay",
    ]
    ratios = {
        "current_producer": 0.64,
        "recent_history": 0.12,
        "hard_negative": 0.04,
        "historical_replay": 0.20,
    }
    sampling_receipt = {"effective_component_sampling_ratios": ratios}
    contract_payload = {
        "schema_version": "flywheel-replay-composite-v2",
        "fresh_component_ids": component_ids[:3],
        "replay_component_ids": component_ids[3:],
        "fresh_source_game_ratios": {
            "current_producer": 0.8,
            "recent_history": 0.15,
            "hard_negative": 0.05,
        },
        "effective_component_sampling_ratios": ratios,
        "realized_replay_ratio": 0.20,
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
        "production_mix_contract": contract_payload,
        "components": [
            {
                "source_category": component_id,
                "corpus_dir": str(tmp_path / component_id),
            }
            for component_id in component_ids
        ],
        "descriptor_file_sha256": "sha256:" + "2" * 64,
        "descriptor_fingerprint": "sha256:" + "3" * 64,
        "learner_recipe_overrides": {
            "per_game_policy_weight": True,
            "per_game_policy_weight_mode": "equal",
        },
        "learner_recipe_overrides_sha256": "sha256:" + "4" * 64,
        "payload_inventory_sha256": "sha256:" + "5" * 64,
    }


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


def test_current_a1_requires_global_n128_and_exact_scalar_dose() -> None:
    recipe, objective = executor._require_a1_science(_lock())
    assert recipe == contract.EXPECTED_LEARNER_TRAINING_RECIPE
    assert objective == _objective()

    with pytest.raises(executor.ExecutorError, match="n_full=128"):
        executor._require_a1_science(_lock(n_full=64))
    with pytest.raises(executor.ExecutorError, match="n_full=128"):
        executor._require_a1_science(_lock(n_full=256))


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
    verified = _verified(tmp_path)
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
    bound["data_kind"] = "production_composite_v2"

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
    monkeypatch.setattr(
        executor.train_bc,
        "_training_data_fingerprint",
        lambda *_args, **_kwargs: "sha256:" + "6" * 64,
    )
    meta = _production_composite_meta(tmp_path, producer["sha256"])

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
    )

    split = result["validation_split_receipt"]
    assert split["aggregate"] == {
        "selected_game_count": 8,
        "training_game_count": 4,
        "validation_game_count": 4,
        "row_count": 16,
        "training_row_count": 8,
        "validation_row_count": 8,
    }
    assert [row["component_id"] for row in split["components"]] == [
        "current_producer",
        "recent_history",
        "hard_negative",
        "historical_replay",
    ]
    assert result["training_row_count"] == 8
    assert result["validation_row_count"] == 8


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

    with pytest.raises(executor.ExecutorError, match="reuses game seeds"):
        executor._verify_production_composite_inputs(
            lock=verified["lock"],
            lock_path=verified["lock_path"],
            reviewed_lock_file_sha256=None,
            recipe=verified["recipe"],
            objective=verified["objective"],
            producer=producer,
            data_path=descriptor,
            meta=_production_composite_meta(tmp_path, producer["sha256"]),
            validation_path=None,
        )


def test_future_production_per_game_value_mode_is_explicit(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    verified["recipe"]["per_game_value_weight"] = True
    verified["recipe"]["per_game_value_weight_mode"] = "sqrt"

    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )

    assert "--per-game-value-weight" in command
    assert _option(command, "--per-game-value-weight-mode") == "sqrt"


def test_latest_main_ablation_command_binds_inventory_ack_and_crop(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    verified["recipe"]["per_game_value_weight_mode"] = "equal"
    verified["learner_ablation"] = {
        "ablation_id": "new-main-arm",
        "code_binding": {},
        "code_tree_sha256": "sha256:" + "8" * 64,
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
    code_sha = "sha256:" + "7" * 64
    monkeypatch.setattr(
        executor,
        "_current_ablation_code_binding",
        lambda lock: {"code_tree_sha256": code_sha, "records": []},
    )
    derived = executor.bind_learner_ablation(
        verified,
        ablation_id="policy-aux-128",
        overrides_json='{"policy_aux_active_batch_size":128}',
        reviewed_code_tree_sha256=code_sha,
    )

    assert derived["recipe"]["policy_aux_active_batch_size"] == 128
    assert derived["learner_ablation"]["recipe_drift"][
        "policy_aux_active_batch_size"
    ] == {
        "contract": "0 (implicit historical train_bc default)",
        "effective": 128,
    }
    command = executor.build_train_command(
        derived,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert _option(command, "--policy-aux-active-batch-size") == "128"
    parsed = executor.train_bc.build_parser().parse_args(command[2:])
    assert executor.train_bc.TrainConfig.from_namespace(
        parsed
    ).policy_aux_active_batch_size == 128


def test_report_binding_seals_exact_base_value_and_policy_doses(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    verified["recipe"]["policy_aux_active_batch_size"] = 128
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


def test_exact_policy_dose_binding_refuses_counter_drift(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    verified["recipe"]["policy_aux_active_batch_size"] = 128
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
    for (sync_data, sync_policy, sync_value), (
        prefetch_data,
        prefetch_policy,
        prefetch_value,
    ) in zip(synchronous, prefetched, strict=True):
        assert set(sync_data) == set(prefetch_data)
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
        report.write_text(json.dumps(_training_report(verified, checkpoint)))
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
            report.write_text(json.dumps(_training_report(verified, checkpoint)))
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
        report.write_text(json.dumps(_training_report(verified, checkpoint)))
        return subprocess.CompletedProcess(command_arg, 0)

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
    with pytest.raises(executor.ExecutorError, match="claim already exists"):
        executor._claim_attempt(
            verified,
            {
                "schema_version": executor.CLAIM_SCHEMA,
                "status": "claimed",
                "contract_sha256": verified["contract_sha256"],
            },
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

    report.write_text(json.dumps(_training_report(verified, checkpoint)))
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
