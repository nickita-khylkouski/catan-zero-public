from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import a1_dual_arm_train as dual
from tools import a1_promotion_transaction as promotion
from tools import a1_dual_learner_contract as learner_contract


SHA = "sha256:" + "a" * 64
DUAL_RUNTIME = [
    {"path": "src/catan_zero/rl/entity_token_policy.py"},
    {"path": "tools/a1_ddp_epoch_canary.py"},
    {"path": "tools/a1_function_preserving_upgrade.py"},
    {"path": "tools/a1_stage_c_final_replication.py"},
    {"path": "tools/train_bc.py"},
    {"path": "tools/a1_dual_arm_train.py"},
    {"path": "tools/a1_dual_learner_contract.py"},
    {"path": "tools/a1_one_dose_train.py"},
]


def _verified(tmp_path: Path) -> dict:
    producer = tmp_path / "producer.pt"
    producer.write_bytes(b"producer")
    validation = tmp_path / "validation.json"
    validation.write_text("{}")
    data = tmp_path / "corpus"
    data.mkdir()
    corpus_meta = _touch_ref(tmp_path / "corpus_meta.json")
    selected = _touch_ref(tmp_path / "selected.json")
    audit = _touch_ref(tmp_path / "audit.json")
    learner_lock = _touch_ref(tmp_path / "learner.lock.json")
    recipe = {
        "track": "2p_no_trade", "vps_to_win": 10,
        "graph_history_features": True, "seed": 1, "epochs": 1,
        "max_steps": 0, "batch_size": 512, "grad_accum_steps": 1,
        "world_size": 8, "global_batch_size": 4096, "optimizer": "adam",
        "resume_optimizer": False, "lr": 3e-5, "lr_warmup_steps": 100,
        "lr_schedule": "flat", "weight_decay": 0.0, "fused_optimizer": False,
        "value_lr_mult": 0.3, "action_module_lr_mult": 1.0,
        "trunk_lr_mult": 1.0,
        "policy_loss_weight": 1.0, "soft_target_source": "policy",
        "soft_target_weight": 0.9, "soft_target_temperature": 0.7,
        "soft_target_min_legal_coverage": 0.5, "value_loss_weight": 0.25,
        "value_target_lambda": 1.0, "value_categorical_loss_weight": 0.0,
        "hlgauss_scalar_aux_loss_weight": 0.0, "final_vp_loss_weight": 0.0,
        "q_loss_weight": 0.0, "policy_kl_anchor_weight": 0.0,
        "value_uncertainty_loss_weight": 0.0, "aux_subgoal_loss_weight": 0.0,
        "train_value_only": False, "freeze_modules": "",
        "policy_surprise_weight": 0.0, "advantage_policy_weighting": "none",
        "per_game_value_weight": False, "vp_margin_weight": 0.0,
        "truncated_vp_margin_value_weight": 0.25, "amp": "bf16",
        "mask_hidden_info": True, "symmetry_augment": False,
        "forced_action_weight": 0.1, "forced_row_value_weight": 1.0,
        "winner_sample_weight": 1.0, "loser_sample_weight": 0.3,
        "teacher_weights": "", "phase_weights": "", "value_phase_weights": "",
        "ddp_shard_data": False,
    }
    return {
        "arm_id": "n256", "subset_id": "full-56k",
        "contract_sha256": SHA, "data": data,
        "corpus_meta": corpus_meta, "selected_manifest": selected, "audit": audit,
        "learner_lock": learner_lock,
        "reviewed_lock_file_sha256": learner_lock["sha256"],
        "validation": {"path": str(validation.resolve()), "sha256": dual._sha256(validation)},  # noqa: SLF001
        "producer": {"path": str(producer.resolve()), "sha256": dual._sha256(producer)},  # noqa: SLF001
        "recipe": recipe,
        "bound_recipe": dict(recipe),
        "topology": learner_contract.TOPOLOGY,
        "objective": {"objective": "mse", "value_readout": "scalar"},
        "learner_code_sha256": SHA, "runtime_code_tree_sha256": SHA,
        "selected_game_seed_set_sha256": SHA,
        "training_game_seed_set_sha256": SHA,
        "validation_game_seed_set_sha256": SHA,
        "payload_inventory_sha256": SHA, "data_fingerprint": SHA,
        "corpus_rows": 10_000, "validation_rows": 1_000, "training_rows": 9_000,
    }


def _touch_ref(path: Path) -> dict[str, str]:
    path.write_text("{}")
    return {"path": str(path.resolve()), "sha256": dual._sha256(path)}  # noqa: SLF001


def _bind_corrective_ablation(
    verified: dict, *, overrides: dict | None = None
) -> dict:
    repo = Path(dual.__file__).resolve().parents[1]
    verified["ablation_code_lock"] = {
        "provenance": {
            "learner_code": [{"path": str(repo / "tools/train_bc.py")}],
            "runtime_code_tree": [
                {"path": str(repo / "tools/a1_dual_arm_train.py")},
                {"path": str(repo / "tools/a1_dual_learner_contract.py")},
                {"path": str(repo / "tools/a1_one_dose_train.py")},
            ],
        }
    }
    code = dual.one_dose._current_ablation_code_binding(  # noqa: SLF001
        verified["ablation_code_lock"]
    )
    return dual.bind_learner_ablation(
        verified,
        ablation_id="all-196k-corrective-v1",
        overrides_json=json.dumps(
            overrides or {"lr": 0.00012, "loser_sample_weight": 1.0}
        ),
        reviewed_code_tree_sha256=code["code_tree_sha256"],
    )


def test_dry_run_command_is_direct_eight_rank_memmap_ddp(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    command = dual.build_command(
        verified,
        python=Path("/venv/bin/python"),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert command[:5] == [
        "/venv/bin/python", "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=8",
    ]
    assert command.count("torch.distributed.run") == 1
    assert sum(Path(value).name == "train_bc.py" for value in command) == 1
    assert command[command.index("--data-format") + 1] == "memmap"
    assert command[command.index("--batch-size") + 1] == "512"
    assert "--ddp-shard-data" not in command
    assert command[command.index("--device") + 1] == "cuda"
    assert dual._execution_binding(command)["gpu_ids"] == list(range(8))  # noqa: SLF001


def test_curriculum_command_warm_starts_from_authenticated_parent(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    plain_identity = dual._claim_identity(verified)  # noqa: SLF001
    parent = tmp_path / "n256-candidate.pt"
    parent.write_bytes(b"parent")
    receipt = tmp_path / "n256.receipt.json"
    receipt.write_text("{}")
    parent_dose = dual.lineage.direct_lineage_dose(
        declared_producer_sha256=verified["producer"]["sha256"],
        init_checkpoint_sha256=verified["producer"]["sha256"],
        current_sampled_rows=56_000,
        current_optimizer_steps=14,
    )
    verified["curriculum_parent"] = {
        "schema_version": "a1-curriculum-parent-binding-v1",
        "receipt_path": str(receipt.resolve()),
        "receipt_sha256": dual._sha256(receipt),  # noqa: SLF001
        "parent_arm_id": "n256",
        "parent_subset_id": "full-56k",
        "parent_checkpoint": {
            "path": str(parent.resolve()),
            "sha256": dual._sha256(parent),  # noqa: SLF001
        },
        "generation_producer_sha256": verified["producer"]["sha256"],
    }
    verified["curriculum_declaration"] = {
        "schema_version": "a1-curriculum-declaration-v1",
        "kind": "sequential_checkpoint_curriculum",
        "parent_receipt_path": str(receipt.resolve()),
        "parent_receipt_sha256": dual._sha256(receipt),  # noqa: SLF001
        "parent_arm_id": "n256",
        "parent_subset_id": "full-56k",
        "parent_checkpoint": {
            "path": str(parent.resolve()),
            "sha256": dual._sha256(parent),  # noqa: SLF001
        },
        "generation_producer_sha256": verified["producer"]["sha256"],
        "parent_lineage_dose": parent_dose,
        "parent_cumulative_sampled_rows": 56_000,
        "parent_cumulative_optimizer_steps": 14,
        "child_arm_id": "n128",
        "child_subset_id": "full-140k",
    }
    command = dual.build_command(
        verified,
        python=Path("/venv/bin/python"),
        checkpoint=tmp_path / "combined.pt",
        report=tmp_path / "combined.json",
    )
    assert command[command.index("--init-checkpoint") + 1] == str(parent.resolve())
    assert command[command.index("--a1-curriculum-parent-receipt") + 1] == str(
        receipt.resolve()
    )
    assert dual._claim_identity(verified) != plain_identity  # noqa: SLF001
    dose = dual._lineage_dose(verified)  # noqa: SLF001
    assert dose["mode"] == "typed_curriculum"
    assert dose["cumulative_sampled_rows"] == 65_000
    assert dose["cumulative_optimizer_steps"] == 17


def test_dual_curriculum_supports_sealed_diagnostic_learner_ablation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified = _bind_corrective_ablation(verified)
    command = dual.build_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "corrective.pt",
        report=tmp_path / "corrective.json",
    )
    assert "--a1-dual-learner-lock" in command
    assert "--a1-learner-ablation-id" in command
    assert command[command.index("--lr") + 1] == "0.00012"
    assert command[command.index("--loser-sample-weight") + 1] == "1.0"

    args = dual.train_bc.build_parser().parse_args(command[6:])
    bound = {
        "dual_arm": True,
        "arm_id": verified["arm_id"],
        "subset_id": verified["subset_id"],
        "learner_training_recipe": verified["bound_recipe"],
        "learner_training_recipe_sha256": dual._digest(verified["bound_recipe"]),  # noqa: SLF001
        "learner_value_objective": verified["objective"],
    }
    monkeypatch.setattr(
        learner_contract,
        "verify_lock",
        lambda *_args, **_kwargs: {
            "arm_id": verified["arm_id"],
            "subset_id": verified["subset_id"],
            "recipe": verified["bound_recipe"],
            "objective": verified["objective"],
            "topology": learner_contract.TOPOLOGY,
            "runtime": DUAL_RUNTIME,
        },
    )
    effective = dual.train_bc._validate_a1_learner_training_recipe(  # noqa: SLF001
        args,
        {"world_size": 8, "rank": 0, "local_rank": 0, "enabled": False},
        bound,
    )
    assert effective["lr"] == pytest.approx(0.00012)
    assert effective["loser_sample_weight"] == pytest.approx(1.0)
    assert bound["learner_ablation"]["diagnostic_only"] is True
    assert bound["learner_topology_authorization"]["topology"] == (
        learner_contract.TOPOLOGY
    )

    bad_args = dual.train_bc.build_parser().parse_args(command[6:])
    declared = json.loads(bad_args.a1_effective_learner_recipe_json)
    declared["value_loss_weight"] = 1.0
    bad_args.value_loss_weight = 1.0
    bad_args.a1_effective_learner_recipe_json = dual._canonical(declared).decode()  # noqa: SLF001
    bad_args.a1_effective_learner_recipe_sha256 = dual._digest(declared)  # noqa: SLF001
    with pytest.raises(
        SystemExit, match="only permits epochs, lr, and loser_sample_weight"
    ):
        dual.train_bc._validate_a1_learner_training_recipe(  # noqa: SLF001
            bad_args,
            {"world_size": 8, "rank": 0, "local_rank": 0, "enabled": False},
            {
                "dual_arm": True,
                "arm_id": verified["arm_id"],
                "subset_id": verified["subset_id"],
                "learner_training_recipe": verified["bound_recipe"],
                "learner_training_recipe_sha256": dual._digest(verified["bound_recipe"]),  # noqa: SLF001
                "learner_value_objective": verified["objective"],
            },
        )

    wrong_lock_args = dual.train_bc.build_parser().parse_args(command[6:])
    wrong_lock_args.a1_reviewed_lock_file_sha256 = "sha256:" + "b" * 64
    with pytest.raises(SystemExit, match="bind the reviewed dual learner lock"):
        dual.train_bc._validate_a1_learner_training_recipe(  # noqa: SLF001
            wrong_lock_args,
            {"world_size": 8, "rank": 0, "local_rank": 0, "enabled": False},
            {
                "dual_arm": True,
                "arm_id": verified["arm_id"],
                "subset_id": verified["subset_id"],
                "learner_training_recipe": verified["bound_recipe"],
                "learner_training_recipe_sha256": dual._digest(verified["bound_recipe"]),  # noqa: SLF001
                "learner_value_objective": verified["objective"],
            },
        )

    short_code_args = dual.train_bc.build_parser().parse_args(command[6:])
    code = json.loads(short_code_args.a1_ablation_code_binding_json)
    code["records"] = code["records"][:-1]
    unhashed = dict(code)
    unhashed.pop("code_tree_sha256")
    code["code_tree_sha256"] = dual._digest(unhashed)  # noqa: SLF001
    short_code_args.a1_ablation_code_binding_json = dual._canonical(code).decode()  # noqa: SLF001
    short_code_args.a1_ablation_code_tree_sha256 = code["code_tree_sha256"]
    with pytest.raises(SystemExit, match="differs from reviewed runtime closure"):
        dual.train_bc._validate_a1_learner_training_recipe(  # noqa: SLF001
            short_code_args,
            {"world_size": 8, "rank": 0, "local_rank": 0, "enabled": False},
            {
                "dual_arm": True,
                "arm_id": verified["arm_id"],
                "subset_id": verified["subset_id"],
                "learner_training_recipe": verified["bound_recipe"],
                "learner_training_recipe_sha256": dual._digest(verified["bound_recipe"]),  # noqa: SLF001
                "learner_value_objective": verified["objective"],
            },
        )


def test_two_b200_fallback_preserves_global_batch_4096(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    verified["topology"] = learner_contract.TOPOLOGIES[2]
    verified["recipe"] = dict(verified["bound_recipe"])
    verified["recipe"].update(
        {"world_size": 2, "batch_size": 512, "grad_accum_steps": 4}
    )
    command = dual.build_command(
        verified, python=Path("/venv/bin/python"),
        checkpoint=tmp_path / "candidate.pt", report=tmp_path / "report.json",
    )
    assert "--nproc_per_node=2" in command
    assert command.count("torch.distributed.run") == 1
    assert sum(Path(value).name == "train_bc.py" for value in command) == 1
    assert command[command.index("--grad-accum-steps") + 1] == "4"
    binding = dual._execution_binding(command, verified)  # noqa: SLF001
    assert binding["gpu_ids"] == [0, 1]
    assert binding["global_batch_size"] == 4096


def test_train_bc_accepts_only_reviewed_two_rank_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    (tmp_path / "lock").write_text("{}")
    bound = {
        "dual_arm": True, "arm_id": verified["arm_id"],
        "subset_id": verified["subset_id"],
        "learner_training_recipe": verified["bound_recipe"],
        "learner_value_objective": verified["objective"],
    }
    effective = dict(verified["bound_recipe"])
    effective.update({"world_size": 2, "batch_size": 512, "grad_accum_steps": 4})
    args = SimpleNamespace(
        **{key: value for key, value in effective.items() if key != "world_size"},
        per_game_value_weight_mode="equal", a1_learner_ablation_id="",
        a1_effective_learner_recipe_json="",
        a1_effective_learner_recipe_sha256="",
        a1_ablation_code_binding_json="", a1_ablation_code_tree_sha256="",
        a1_reviewed_lock_file_sha256="", a1_dual_learner_lock=str(tmp_path / "lock"),
        a1_dual_reviewed_lock_file_sha256=SHA,
    )
    monkeypatch.setattr(
        learner_contract, "verify_lock",
        lambda *_args, **_kwargs: {
            "arm_id": verified["arm_id"], "subset_id": verified["subset_id"],
            "recipe": verified["bound_recipe"], "objective": verified["objective"],
            "topology": learner_contract.TOPOLOGIES[2],
            "runtime": DUAL_RUNTIME,
        },
    )
    actual = dual.train_bc._validate_a1_learner_training_recipe(  # noqa: SLF001
        args, {"world_size": 2}, bound
    )
    assert actual["global_batch_size"] == 4096
    assert bound["learner_topology_authorization"]["topology"]["world_size"] == 2


def test_train_bc_ddp_replays_dual_authority_once_on_rank0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nonzero ranks consume rank 0's replay instead of racing its file lock."""

    import torch.distributed as dist

    verified = _verified(tmp_path)
    (tmp_path / "lock").write_text("{}")
    authority = {
        "arm_id": verified["arm_id"],
        "subset_id": verified["subset_id"],
        "recipe": verified["bound_recipe"],
        "objective": verified["objective"],
        "topology": learner_contract.TOPOLOGIES[2],
        "runtime": DUAL_RUNTIME,
    }
    effective = dict(verified["bound_recipe"])
    effective.update(
        {"world_size": 2, "batch_size": 512, "grad_accum_steps": 4}
    )
    args = SimpleNamespace(
        **{key: value for key, value in effective.items() if key != "world_size"},
        per_game_value_weight_mode="equal",
        a1_learner_ablation_id="",
        a1_effective_learner_recipe_json="",
        a1_effective_learner_recipe_sha256="",
        a1_ablation_code_binding_json="",
        a1_ablation_code_tree_sha256="",
        a1_reviewed_lock_file_sha256="",
        a1_dual_learner_lock=str(tmp_path / "lock"),
        a1_dual_reviewed_lock_file_sha256=SHA,
    )
    verify_calls: list[int] = []
    published: list[dict[str, object] | None] = [None]
    active_rank = {"value": 0}

    def verify_once(*_args, **_kwargs):
        verify_calls.append(active_rank["value"])
        return authority

    def broadcast(payload, *, src):
        assert src == 0
        if active_rank["value"] == 0:
            published[0] = payload[0]
        else:
            payload[0] = published[0]

    monkeypatch.setattr(learner_contract, "verify_lock", verify_once)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)

    for rank in (0, 1):
        active_rank["value"] = rank
        bound = {
            "dual_arm": True,
            "arm_id": verified["arm_id"],
            "subset_id": verified["subset_id"],
            "learner_training_recipe": verified["bound_recipe"],
            "learner_value_objective": verified["objective"],
        }
        actual = dual.train_bc._validate_a1_learner_training_recipe(  # noqa: SLF001
            args,
            {"enabled": True, "world_size": 2, "rank": rank},
            bound,
        )
        assert actual["global_batch_size"] == 4096
        assert bound["learner_topology_authorization"]["topology"]["world_size"] == 2

    assert verify_calls == [0]


def test_train_bc_ddp_broadcasts_leaked_promotion_lock_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rank-0 PromotionError reaches peers instead of stranding the collective."""

    import torch.distributed as dist
    from tools import a1_promotion_transaction as promotion

    verified = _verified(tmp_path)
    (tmp_path / "lock").write_text("{}")
    effective = dict(verified["bound_recipe"])
    effective.update({"world_size": 2, "batch_size": 512, "grad_accum_steps": 4})
    args = SimpleNamespace(
        **{key: value for key, value in effective.items() if key != "world_size"},
        per_game_value_weight_mode="equal", a1_learner_ablation_id="",
        a1_effective_learner_recipe_json="",
        a1_effective_learner_recipe_sha256="",
        a1_ablation_code_binding_json="", a1_ablation_code_tree_sha256="",
        a1_reviewed_lock_file_sha256="", a1_dual_learner_lock=str(tmp_path / "lock"),
        a1_dual_reviewed_lock_file_sha256=SHA,
    )
    published: list[dict[str, object] | None] = [None]
    active_rank = {"value": 0}
    verify_calls: list[int] = []

    def locked(*_args, **_kwargs):
        verify_calls.append(active_rank["value"])
        raise promotion.PromotionError("lock held at champion_registry.json.a1.lock")

    def broadcast(payload, *, src):
        assert src == 0
        if active_rank["value"] == 0:
            published[0] = payload[0]
        else:
            payload[0] = published[0]

    monkeypatch.setattr(learner_contract, "verify_lock", locked)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)
    for rank in (0, 1):
        active_rank["value"] = rank
        bound = {
            "dual_arm": True,
            "arm_id": verified["arm_id"],
            "subset_id": verified["subset_id"],
            "learner_training_recipe": verified["bound_recipe"],
            "learner_value_objective": verified["objective"],
        }
        with pytest.raises(
            SystemExit,
            match="dual learner topology lock refused: lock held at champion_registry",
        ):
            dual.train_bc._validate_a1_learner_training_recipe(  # noqa: SLF001
                args,
                {"enabled": True, "world_size": 2, "rank": rank},
                bound,
            )
    assert verify_calls == [0]


def test_completed_receipt_replay_never_acquires_gpu_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    sentinel = {"status": "complete"}
    monkeypatch.setattr(dual, "verify_receipt", lambda *_args, **_kwargs: sentinel)
    monkeypatch.setattr(
        dual.one_dose, "_physical_gpu_lock",
        lambda _gpu: (_ for _ in ()).throw(AssertionError("GPU lock acquired")),
    )
    assert dual.execute(
        verified=verified, command=["train"], checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json", receipt=receipt,
    ) == sentinel


def _write_outputs(
    tmp_path: Path, verified: dict, binding: dict, *, world_size: int = 8
) -> tuple[Path, Path]:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    Path(str(checkpoint) + ".optimizer.pt").write_bytes(b"optimizer")
    report = tmp_path / "report.json"
    epochs = int(verified["recipe"].get("epochs", 1))
    steps = math.ceil(verified["training_rows"] / 4096) * epochs
    lineage_dose = dual._lineage_dose(verified)  # noqa: SLF001
    parent = verified.get("curriculum_parent")
    payload = {
        "arch": "entity_graph", "hidden_size": 640, "graph_layers": 6,
        "attention_heads": 8, "graph_dropout": 0.05, "world_size": world_size,
        "batch_size": 512, "ddp_shard_data": False, "steps_completed": steps,
        "total_training_steps": steps, "epochs": epochs, "max_steps": 0,
        "samples": verified["corpus_rows"], "global_samples": verified["corpus_rows"],
        "train_samples": verified["training_rows"],
        "validation_samples": verified["validation_rows"],
        "data": str(verified["data"]), "data_format": "memmap",
        "data_fingerprint": verified["data_fingerprint"],
        "track": "2p_no_trade", "vps_to_win": 10,
        "checkpoint": str(checkpoint),
        "init_checkpoint": (
            verified["producer"]["path"]
            if parent is None
            else parent["parent_checkpoint"]["path"]
        ),
        "init_checkpoint_sha256": (
            verified["producer"]["sha256"]
            if parent is None
            else parent["parent_checkpoint"]["sha256"]
        ),
        "a1_curriculum_parent": parent,
        "a1_curriculum_declaration": verified.get("curriculum_declaration"),
        "a1_lineage_dose": lineage_dose,
        "a1_contract_sha256": verified["contract_sha256"],
        "a1_selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
        "a1_training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
        "validation_game_seed_set_sha256": verified["validation_game_seed_set_sha256"],
        "input_validation_game_seed_manifest": verified["validation"]["path"],
        "input_validation_game_seed_manifest_sha256": verified["validation"]["sha256"],
        "a1_bound_learner_training_recipe": verified["bound_recipe"],
        "a1_bound_learner_value_objective": verified["objective"],
        "a1_learner_training_recipe_sha256": dual._digest(verified["bound_recipe"]),  # noqa: SLF001
        "a1_learner_code_sha256": verified["learner_code_sha256"],
        "a1_runtime_code_tree_sha256": verified["runtime_code_tree_sha256"],
        "a1_memmap_payload_inventory_sha256": verified["payload_inventory_sha256"],
        "mask_hidden_info": True, "require_35m_model": True,
        "optimizer": "adam", "resume_optimizer": False,
        "optimizer_restored": False, "fused_optimizer": False, "amp": "bf16",
        dual.REPORT_BINDING_FIELD: binding, "parameter_count": 35_000_000,
        "metrics": [
            {
                "epoch": epoch, "loss": 1.0, "policy_loss": 0.7,
                "value_loss": 0.3,
                "validation": {"samples": verified["validation_rows"], "loss": 1.1},
            }
            for epoch in range(1, epochs + 1)
        ],
        "value_training": {
            "optimizer_steps": steps, "completed_epochs": epochs,
            "trained_value_readouts": ["scalar"],
            "a1_contract_sha256": verified["contract_sha256"],
            "a1_selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
            "a1_training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
            "a1_learner_training_recipe_sha256": dual._digest(verified["bound_recipe"]),  # noqa: SLF001
            "a1_memmap_payload_inventory_sha256": verified["payload_inventory_sha256"],
        },
    }
    if verified["recipe"] != verified["bound_recipe"]:
        payload.update({
            "a1_effective_learner_training_recipe": verified["recipe"],
            "a1_effective_learner_training_recipe_sha256": dual._digest(verified["recipe"]),  # noqa: SLF001
            "a1_learner_topology_authorization": {
                "schema_version": "a1-dual-learner-topology-authorization-v1",
                "learner_lock": verified["learner_lock"]["path"],
                "learner_lock_file_sha256": verified["reviewed_lock_file_sha256"],
                "topology": verified["topology"],
                "effective_recipe": verified["recipe"],
                "effective_recipe_sha256": dual._digest(verified["recipe"]),  # noqa: SLF001
            },
        })
    if verified.get("learner_ablation") is not None:
        payload.update({
            "a1_learner_ablation": verified["learner_ablation"],
            "diagnostic_only": True,
            "promotion_eligible": False,
        })
        payload["value_training"]["learner_ablation"] = verified["learner_ablation"]
    if epochs > 1:
        for epoch in range(1, epochs + 1):
            epoch_checkpoint = dual.train_bc._epoch_checkpoint_path(  # noqa: SLF001
                str(checkpoint), epoch
            )
            epoch_checkpoint.write_bytes(f"candidate-{epoch}".encode())
            Path(str(epoch_checkpoint) + ".optimizer.pt").write_bytes(
                f"optimizer-{epoch}".encode()
            )
    report.write_text(json.dumps(payload))
    return checkpoint, report


def test_output_verifier_accepts_exact_ddp_and_rejects_world_size_drift(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    binding = dual._execution_binding(["train"])  # noqa: SLF001
    checkpoint, report = _write_outputs(tmp_path, verified, binding)
    outputs = dual.verify_outputs(
        verified=verified, checkpoint=checkpoint, report=report, binding=binding
    )
    assert outputs["steps_completed"] == 3
    assert outputs["lineage_dose"]["cumulative_sampled_rows"] == 9_000

    report.unlink()
    _checkpoint, report = _write_outputs(tmp_path, verified, binding, world_size=1)
    with pytest.raises(dual.DualTrainError, match="world_size"):
        dual.verify_outputs(
            verified=verified, checkpoint=checkpoint, report=report, binding=binding
        )


def test_output_verifier_rejects_accidental_n256_to_n128_chaining_without_declaration(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    verified["arm_id"] = "n128"
    verified["subset_id"] = "full-140k"
    binding = dual._execution_binding(["train"], verified)  # noqa: SLF001
    checkpoint, report = _write_outputs(tmp_path, verified, binding)
    payload = json.loads(report.read_text())
    accidental_parent = tmp_path / "n256-candidate.pt"
    accidental_parent.write_bytes(b"previous-dose")
    payload["init_checkpoint"] = str(accidental_parent)
    payload["init_checkpoint_sha256"] = dual._sha256(accidental_parent)  # noqa: SLF001
    report.write_text(json.dumps(payload))
    with pytest.raises(dual.DualTrainError, match="init_checkpoint"):
        dual.verify_outputs(
            verified=verified,
            checkpoint=checkpoint,
            report=report,
            binding=binding,
        )


def test_output_verifier_accepts_separately_bound_effective_ablation(
    tmp_path: Path,
) -> None:
    verified = _bind_corrective_ablation(_verified(tmp_path))
    binding = dual._execution_binding(["train"], verified)  # noqa: SLF001
    checkpoint, report = _write_outputs(tmp_path, verified, binding)

    outputs = dual.verify_outputs(
        verified=verified, checkpoint=checkpoint, report=report, binding=binding
    )

    assert outputs["steps_completed"] == 3


def test_three_epoch_diagnostic_is_one_clean_trajectory_with_sealed_checkpoints(
    tmp_path: Path,
) -> None:
    verified = _bind_corrective_ablation(
        _verified(tmp_path),
        overrides={"epochs": 3, "lr": 0.00012, "loser_sample_weight": 1.0},
    )
    command = dual.build_command(
        verified,
        python=Path("/venv/bin/python"),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert command[command.index("--epochs") + 1] == "3"
    assert "--no-resume-optimizer" in command
    assert "--save-each-epoch" in command
    assert command[command.index("--train-diagnostics-every-batches") + 1] == "100"

    binding = dual._execution_binding(command, verified)  # noqa: SLF001
    checkpoint, report = _write_outputs(tmp_path, verified, binding)
    outputs = dual.verify_outputs(
        verified=verified, checkpoint=checkpoint, report=report, binding=binding
    )

    assert outputs["steps_completed"] == 9
    assert set(outputs["epoch_checkpoints"]) == {"1", "2", "3"}
    assert [
        outputs["epoch_checkpoints"][str(epoch)]["exposures"]
        for epoch in range(1, 4)
    ] == [1.0, 2.0, 3.0]

    epoch_two = dual.train_bc._epoch_checkpoint_path(str(checkpoint), 2)  # noqa: SLF001
    Path(str(epoch_two) + ".optimizer.pt").unlink()
    with pytest.raises(dual.DualTrainError, match="epoch 2 optimizer sidecar"):
        dual.verify_outputs(
            verified=verified, checkpoint=checkpoint, report=report, binding=binding
        )


def test_epoch_outputs_prefer_objective_matched_validation_when_present(
    tmp_path: Path,
) -> None:
    verified = _bind_corrective_ablation(
        _verified(tmp_path),
        overrides={"epochs": 3, "lr": 0.00012, "loser_sample_weight": 1.0},
    )
    binding = dual._execution_binding(["train"], verified)  # noqa: SLF001
    checkpoint, report = _write_outputs(tmp_path, verified, binding)
    payload = json.loads(report.read_text(encoding="utf-8"))
    for epoch, metric in enumerate(payload["metrics"], start=1):
        metric["validation"]["loss"] = 90.0 + epoch
        metric["validation_objective_matched"] = {
            "schema_version": "composite-validation-measure-v2",
            "objective_matched": True,
            "samples": verified["validation_rows"],
            "metrics": {
                "loss": 1.0 + epoch / 10.0,
            },
        }
    report.write_text(json.dumps(payload), encoding="utf-8")

    outputs = dual.verify_outputs(
        verified=verified, checkpoint=checkpoint, report=report, binding=binding
    )

    assert [
        outputs["epoch_checkpoints"][str(epoch)]["validation"]["loss"]
        for epoch in range(1, 4)
    ] == [1.1, 1.2, 1.3]
    assert all(
        row["validation_measure"] == "objective_matched"
        for row in outputs["epoch_checkpoints"].values()
    )


def test_epoch_outputs_reject_malformed_objective_matched_validation(
    tmp_path: Path,
) -> None:
    verified = _bind_corrective_ablation(
        _verified(tmp_path),
        overrides={"epochs": 3, "lr": 0.00012, "loser_sample_weight": 1.0},
    )
    binding = dual._execution_binding(["train"], verified)  # noqa: SLF001
    checkpoint, report = _write_outputs(tmp_path, verified, binding)
    payload = json.loads(report.read_text(encoding="utf-8"))
    for metric in payload["metrics"]:
        metric["validation_objective_matched"] = {
            "objective_matched": True,
            "samples": verified["validation_rows"],
            "metrics": {"loss": 1.0},
        }
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(dual.DualTrainError, match="malformed objective-matched"):
        dual.verify_outputs(
            verified=verified, checkpoint=checkpoint, report=report, binding=binding
        )


def test_promotion_receipt_verifier_accepts_dual_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_lock = tmp_path / "contract.lock.json"
    contract_lock.write_text("{}")
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    optimizer = tmp_path / "candidate.pt.optimizer.pt"
    optimizer.write_bytes(b"optimizer")
    binding = {"schema_version": "a1-dual-arm-execution-binding-v1"}
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        dual.REPORT_BINDING_FIELD: binding,
        "a1_bound_learner_training_recipe": {"world_size": 8},
        "a1_bound_learner_value_objective": {"objective": "mse"},
    }))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "contract_sha256": SHA, "contract_path": str(contract_lock.resolve())
    }))
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"schema_version": dual.RECEIPT_SCHEMA}))
    producer = tmp_path / "producer.pt"
    producer.write_bytes(b"producer")
    learner_lock_path = tmp_path / "learner.lock.json"
    learner_lock_path.write_text("{}")
    corpus_meta = tmp_path / "corpus_meta.json"
    corpus_meta.write_text("{}")
    validation = tmp_path / "validation.json"
    validation.write_text("{}")
    value = {
        "schema_version": dual.RECEIPT_SCHEMA,
        "contract_sha256": SHA,
        "receipt_sha256": SHA,
        "arm_id": "n128", "subset_id": "matched-56k",
        "claim": {"path": str(tmp_path / "claim.json"), "sha256": SHA},
        "claim_completion": {"path": str(tmp_path / "complete.json"), "sha256": SHA},
        "execution_binding": binding,
        "inputs": {
            "producer": {"path": str(producer), "sha256": dual._sha256(producer)},  # noqa: SLF001
            "audit": {"path": str(audit), "sha256": dual._sha256(audit)},  # noqa: SLF001
                "learner_lock": {
                "path": str(learner_lock_path),
                "sha256": dual._sha256(learner_lock_path),  # noqa: SLF001
                },
                "corpus_meta": {
                    "path": str(corpus_meta), "sha256": dual._sha256(corpus_meta),  # noqa: SLF001
                },
                "validation": {
                    "path": str(validation), "sha256": dual._sha256(validation),  # noqa: SLF001
                },
        },
        "outputs": {
            "checkpoint": {"path": str(candidate), "sha256": dual._sha256(candidate)},  # noqa: SLF001
            "optimizer": {"path": str(optimizer), "sha256": dual._sha256(optimizer)},  # noqa: SLF001
            "report": {"path": str(report), "sha256": dual._sha256(report)},  # noqa: SLF001
        },
    }
    replay_calls = []
    monkeypatch.setattr(
        dual,
        "verify_inputs",
        lambda **kwargs: replay_calls.append(("inputs", kwargs)) or {"sealed": True},
    )
    monkeypatch.setattr(
        dual,
        "verify_receipt",
        lambda _path, **kwargs: replay_calls.append(("receipt", kwargs)) or value,
    )
    monkeypatch.setattr(
        learner_contract,
        "verify_lock",
        lambda *_args, **_kwargs: {
            "generation_arm_lock": {
                "path": str(contract_lock.resolve()),
                "sha256": dual._sha256(contract_lock),  # noqa: SLF001
            },
            "generation_contract_sha256": SHA,
            "arm_id": "n128", "subset_id": "matched-56k",
            "recipe": {"world_size": 8},
            "objective": {"objective": "mse"},
        },
    )
    result = promotion._verify_one_dose_training_receipt(  # noqa: SLF001
        receipt_path,
        contract_lock=contract_lock.resolve(),
        contract={
            "contract_sha256": SHA,
            "checkpoints": [{"role": "producer", "sha256": dual._sha256(producer)}],  # noqa: SLF001
        },
        candidate_path=candidate.resolve(),
        candidate_sha256=dual._sha256(candidate),  # noqa: SLF001
        training_report_path=report.resolve(),
        training_report_sha256=dual._sha256(report),  # noqa: SLF001
    )
    assert result["receipt_sha256"] == SHA
    assert replay_calls[-1] == ("receipt", {"verified": {"sealed": True}})


def test_promotion_training_report_accepts_bound_eight_rank_recipe(
    tmp_path: Path,
) -> None:
    verified = _verified(tmp_path)
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "a1_dual_arm_execution_binding": {"schema_version": "binding"},
        "a1_contract_sha256": SHA,
        "a1_learner_training_recipe_sha256": dual._digest(verified["recipe"]),  # noqa: SLF001
        "a1_bound_learner_training_recipe": verified["recipe"],
        "arch": "entity_graph", "mask_hidden_info": True,
        "track": "2p_no_trade", "vps_to_win": 10,
        "world_size": 8, "batch_size": 512,
        "checkpoint": str(candidate),
        "init_checkpoint_sha256": verified["producer"]["sha256"],
        "steps_completed": 3, "epochs": 1, "max_steps": 0,
    }))
    result = promotion._verify_training_report(  # noqa: SLF001
        report,
        contract={
            "checkpoints": [{"role": "producer", "sha256": verified["producer"]["sha256"]}],
        },
        contract_sha256=SHA,
        candidate_path=candidate.resolve(),
        candidate_sha256=dual._sha256(candidate),  # noqa: SLF001
    )
    assert result["world_size"] == 8


def _runner_that_writes_outputs(
    tmp_path: Path, verified: dict, binding: dict, calls: list[int]
):
    def runner(*_args, **_kwargs):
        calls.append(1)
        checkpoint, report = _write_outputs(tmp_path, verified, binding)
        payload = json.loads(report.read_text())
        payload.pop(dual.REPORT_BINDING_FIELD)
        payload.pop("a1_lineage_dose")
        payload.pop("a1_curriculum_declaration")
        report.write_text(json.dumps(payload))
        assert checkpoint == tmp_path / "candidate.pt"
        return subprocess.CompletedProcess([], 0)

    return runner


def test_recovery_quarantines_claim_only_then_replays_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    command = ["train"]
    binding = dual._execution_binding(command)  # noqa: SLF001
    claim = dual._claim_path(verified)  # noqa: SLF001
    dual._write_new(claim, {"schema_version": dual.CLAIM_SCHEMA, "status": "claimed"})  # noqa: SLF001
    monkeypatch.setattr(dual.one_dose, "_physical_gpu_lock", lambda _gpu: _nullcontext())
    calls: list[int] = []
    receipt = dual.execute(
        verified=verified, command=command,
        checkpoint=tmp_path / "candidate.pt", report=tmp_path / "report.json",
        receipt=tmp_path / "receipt.json",
        runner=_runner_that_writes_outputs(tmp_path, verified, binding, calls),
        probe=lambda gpu: f"NVIDIA B200 gpu{gpu}",
    )
    assert calls == [1] and receipt["status"] == "complete"
    recoveries = list((claim.parent / "quarantine" / claim.stem).glob("recovery-*.json"))
    assert len(recoveries) == 1


def test_recovery_quarantines_unbound_child_outputs_then_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    command = ["train"]
    binding = dual._execution_binding(command)  # noqa: SLF001
    claim = dual._claim_path(verified)  # noqa: SLF001
    dual._write_new(claim, {"schema_version": dual.CLAIM_SCHEMA, "status": "claimed"})  # noqa: SLF001
    _checkpoint, report = _write_outputs(tmp_path, verified, binding)
    payload = json.loads(report.read_text())
    payload.pop(dual.REPORT_BINDING_FIELD)
    payload.pop("a1_lineage_dose")
    payload.pop("a1_curriculum_declaration")
    report.write_text(json.dumps(payload))
    monkeypatch.setattr(dual.one_dose, "_physical_gpu_lock", lambda _gpu: _nullcontext())
    calls: list[int] = []
    result = dual.execute(
        verified=verified, command=command,
        checkpoint=tmp_path / "candidate.pt", report=report,
        receipt=tmp_path / "receipt.json",
        runner=_runner_that_writes_outputs(tmp_path, verified, binding, calls),
        probe=lambda gpu: f"NVIDIA B200 gpu{gpu}",
    )
    assert calls == [1] and result["status"] == "complete"


def test_recovery_finalizes_bound_outputs_without_retraining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    command = ["train"]
    binding = dual._execution_binding(command)  # noqa: SLF001
    claim = dual._claim_path(verified)  # noqa: SLF001
    dual._write_new(claim, {"schema_version": dual.CLAIM_SCHEMA, "status": "claimed"})  # noqa: SLF001
    checkpoint, report = _write_outputs(tmp_path, verified, binding)
    monkeypatch.setattr(dual.one_dose, "_physical_gpu_lock", lambda _gpu: _nullcontext())

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("valid bound outputs must finalize without retraining")

    receipt_path = tmp_path / "receipt.json"
    result = dual.execute(
        verified=verified, command=command, checkpoint=checkpoint, report=report,
        receipt=receipt_path, runner=must_not_run,
        probe=lambda gpu: f"NVIDIA B200 gpu{gpu}",
    )
    assert result == dual.verify_receipt(receipt_path, verified=verified)


def test_recovery_finalizes_existing_completion_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    command = ["train"]
    binding = dual._execution_binding(command)  # noqa: SLF001
    claim = dual._claim_path(verified)  # noqa: SLF001
    dual._write_new(claim, {"schema_version": dual.CLAIM_SCHEMA, "status": "claimed"})  # noqa: SLF001
    checkpoint, report = _write_outputs(tmp_path, verified, binding)
    outputs = dual.verify_outputs(
        verified=verified, checkpoint=checkpoint, report=report, binding=binding
    )
    completion = {
        "schema_version": dual.CLAIM_SCHEMA, "status": "complete",
        "claim": dual._file_ref(claim, where="fixture"),  # noqa: SLF001
        "claim_identity_sha256": dual._claim_identity(verified),  # noqa: SLF001
        "receipt": str(tmp_path / "receipt.json"),
        "command_sha256": dual._digest(command),  # noqa: SLF001
        "execution_binding": binding,
        "gpu_names": [f"NVIDIA B200 gpu{i}" for i in range(8)],
        "outputs": outputs,
        "lineage_dose": outputs["lineage_dose"],
        "finished_unix_ns": 1,
    }
    dual._write_new(dual._completion_path(claim), completion)  # noqa: SLF001
    monkeypatch.setattr(dual.one_dose, "_physical_gpu_lock", lambda _gpu: _nullcontext())
    result = dual.execute(
        verified=verified, command=command, checkpoint=checkpoint, report=report,
        receipt=tmp_path / "receipt.json",
        runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no rerun")),
        probe=lambda gpu: f"NVIDIA B200 gpu{gpu}",
    )
    assert result["finished_unix_ns"] == 1


def test_identity_lock_rejects_concurrent_same_corpus_attempt(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    with dual._identity_lock(verified):  # noqa: SLF001
        with pytest.raises(dual.DualTrainError, match="already active"):
            with dual._identity_lock(verified):  # noqa: SLF001
                pass


def test_failed_child_writes_durable_attempt_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    monkeypatch.setattr(dual.one_dose, "_physical_gpu_lock", lambda _gpu: _nullcontext())
    with pytest.raises(dual.DualTrainError, match="nonzero"):
        dual.execute(
            verified=verified, command=["train"],
            checkpoint=tmp_path / "candidate.pt", report=tmp_path / "report.json",
            receipt=tmp_path / "receipt.json",
            runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 7),
            probe=lambda gpu: f"NVIDIA B200 gpu{gpu}",
        )
    claim = dual._claim_path(verified)  # noqa: SLF001
    attempts = list((claim.parent / "attempt-receipts" / claim.stem).glob("failed-*.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text())["returncode"] == 7


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False
