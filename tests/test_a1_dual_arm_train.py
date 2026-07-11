from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools import a1_dual_arm_train as dual
from tools import a1_promotion_transaction as promotion
from tools import a1_dual_learner_contract as learner_contract


SHA = "sha256:" + "a" * 64


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
    assert command[command.index("--data-format") + 1] == "memmap"
    assert command[command.index("--batch-size") + 1] == "512"
    assert "--ddp-shard-data" not in command
    assert command[command.index("--device") + 1] == "cuda"
    assert dual._execution_binding(command)["gpu_ids"] == list(range(8))  # noqa: SLF001


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
        },
    )
    actual = dual.train_bc._validate_a1_learner_training_recipe(  # noqa: SLF001
        args, {"world_size": 2}, bound
    )
    assert actual["global_batch_size"] == 4096
    assert bound["learner_topology_authorization"]["topology"]["world_size"] == 2


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
    steps = math.ceil(verified["training_rows"] / 4096)
    payload = {
        "arch": "entity_graph", "hidden_size": 640, "graph_layers": 6,
        "attention_heads": 8, "graph_dropout": 0.05, "world_size": world_size,
        "batch_size": 512, "ddp_shard_data": False, "steps_completed": steps,
        "total_training_steps": steps, "epochs": 1, "max_steps": 0,
        "samples": verified["corpus_rows"], "global_samples": verified["corpus_rows"],
        "train_samples": verified["training_rows"],
        "validation_samples": verified["validation_rows"],
        "data": str(verified["data"]), "data_format": "memmap",
        "data_fingerprint": verified["data_fingerprint"],
        "track": "2p_no_trade", "vps_to_win": 10,
        "checkpoint": str(checkpoint),
        "init_checkpoint": verified["producer"]["path"],
        "init_checkpoint_sha256": verified["producer"]["sha256"],
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
        "metrics": [{
            "epoch": 1, "loss": 1.0, "policy_loss": 0.7, "value_loss": 0.3,
            "validation": {"samples": verified["validation_rows"], "loss": 1.1},
        }],
        "value_training": {
            "optimizer_steps": steps, "completed_epochs": 1,
            "trained_value_readouts": ["scalar"],
            "a1_contract_sha256": verified["contract_sha256"],
            "a1_selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
            "a1_training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
            "a1_learner_training_recipe_sha256": dual._digest(verified["recipe"]),  # noqa: SLF001
            "a1_memmap_payload_inventory_sha256": verified["payload_inventory_sha256"],
        },
    }
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

    report.unlink()
    _checkpoint, report = _write_outputs(tmp_path, verified, binding, world_size=1)
    with pytest.raises(dual.DualTrainError, match="world_size"):
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
        "outputs": outputs, "finished_unix_ns": 1,
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
