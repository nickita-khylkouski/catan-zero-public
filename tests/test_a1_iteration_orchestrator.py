from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import a1_iteration_orchestrator as iteration
from tools import a1_pre_wave_contract as contract


def _write(path: Path, value: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _verified(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    lock = _write(tmp_path / "contract.lock.json", "lock")
    validation = _write(tmp_path / "validation.json", "validation")
    ledger = _write(tmp_path / "seed_ledger.md", "ledger")
    data = tmp_path / "corpus"
    data.mkdir()
    meta = _write(data / "corpus_meta.json", "meta")
    producer = _write(tmp_path / "producer.pt", "producer")
    contract_sha = "sha256:" + "a" * 64
    recipe = dict(contract.EXPECTED_LEARNER_TRAINING_RECIPE)
    objective = {
        "objective": "mse",
        "value_readout": "scalar",
        "value_categorical_bins": None,
        "hlgauss_sigma_ratio": None,
    }
    value = {
        "lock": {"fleet": {"seed_ledger": {"path": str(ledger)}}},
        "lock_path": lock,
        "lock_file_sha256": iteration._file_sha256(lock),
        "contract_sha256": contract_sha,
        "recipe": recipe,
        "objective": objective,
        "producer": {
            "role": "producer",
            "path": str(producer),
            "sha256": iteration._file_sha256(producer),
        },
        "data_path": data,
        "corpus_meta_file_sha256": iteration._file_sha256(meta),
        "payload_inventory_sha256": "sha256:" + "b" * 64,
        "data_fingerprint": "sha256:" + "c" * 64,
        "corpus_row_count": 100,
        "training_row_count": 95,
        "validation_row_count": 5,
        "selected_game_seed_set_sha256": "sha256:" + "d" * 64,
        "training_game_seed_set_sha256": "sha256:" + "e" * 64,
        "validation_path": validation,
        "validation_file_sha256": iteration._file_sha256(validation),
        "validation_game_seed_set_sha256": "sha256:" + "f" * 64,
    }
    return value, {
        "lock": lock,
        "validation": validation,
        "ledger": ledger,
        "data": data,
        "meta": meta,
        "producer": producer,
    }


def _initialize(tmp_path: Path) -> tuple[Path, dict]:
    verified, paths = _verified(tmp_path)
    state_path = tmp_path / "iteration.json"
    state = iteration.initialize(
        state_path=state_path,
        lock_path=paths["lock"],
        data_path=paths["data"],
        validation_path=paths["validation"],
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "train.json",
        training_receipt=tmp_path / "train.receipt.json",
        python=Path(sys.executable),
        gpu=0,
        bootstrap_history=True,
        verify_fn=lambda **_kwargs: verified,
    )
    return state_path, state


def _dry_plan(state: dict, *, mode: str = "dry-run") -> dict:
    return {
        "schema_version": iteration.one_dose.PLAN_SCHEMA,
        "mode": mode,
        "contract_sha256": state["training"]["contract_sha256"],
        "global_n_full": 128,
        "world_size": 1,
        "gpu": 0,
        "command": ["python", "train_bc.py"],
        "command_sha256": "sha256:" + "1" * 64,
        "execution_binding": {
            "schema_version": iteration.one_dose.REPORT_EXECUTION_BINDING_SCHEMA,
            "command_sha256": "sha256:" + "1" * 64,
            "environment": {
                key: "value" for key in iteration.one_dose.CHILD_ENVIRONMENT_KEYS
            },
            "environment_sha256": "sha256:" + "8" * 64,
        },
        "checkpoint": state["training"]["checkpoint"],
        "report": state["training"]["report"],
        "receipt": state["training"]["receipt"],
    }


def _runner(payload: dict, *, returncode: int = 0, stderr: str = ""):
    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=json.dumps(payload), stderr=stderr
        )

    return run


def _advance_to_dry(tmp_path: Path) -> tuple[Path, dict]:
    state_path, state = _initialize(tmp_path)
    state = iteration.dose_dry_run(
        state_path=state_path, runner=_runner(_dry_plan(state))
    )
    return state_path, state


def _advance_to_complete(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    state_path, state = _advance_to_dry(tmp_path)
    training = state["training"]
    checkpoint = _write(Path(training["checkpoint"]), "candidate")
    optimizer = _write(Path(str(checkpoint) + ".optimizer.pt"), "optimizer")
    report = _write(Path(training["report"]), "report")
    receipt = _write(Path(training["receipt"]), "receipt")
    outputs = {
        "checkpoint": str(checkpoint),
        "optimizer_sidecar": str(optimizer),
        "report": str(report),
    }
    monkeypatch.setattr(
        iteration,
        "_load_complete_training_receipt",
        lambda _state: {"outputs": outputs},
    )
    state = iteration.dose_go(
        state_path=state_path,
        runner=lambda *_args, **_kwargs: pytest.fail("resume must not rerun training"),
    )
    assert state["training_outputs"]["receipt"]["path"] == str(receipt)
    return state_path, state


def test_initialize_seals_verified_corpus_and_fresh_outputs(tmp_path: Path) -> None:
    state_path, state = _initialize(tmp_path)

    assert state["stage"] == "corpus_verified"
    assert state["training"]["corpus_row_count"] == 100
    assert state["training"]["training_row_count"] == 95
    assert state["training"]["gpu"] == 0
    assert (
        iteration.status(state_path=state_path)["state_sha256"] == state["state_sha256"]
    )


def test_initialize_preserves_virtualenv_python_path(tmp_path: Path) -> None:
    verified, paths = _verified(tmp_path)
    base = _write(tmp_path / "base-python", "#!/bin/sh\nexit 0\n")
    base.chmod(0o755)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.symlink_to(base)

    state = iteration.initialize(
        state_path=tmp_path / "iteration.json",
        lock_path=paths["lock"],
        data_path=paths["data"],
        validation_path=paths["validation"],
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
        training_receipt=tmp_path / "training.receipt.json",
        python=python,
        gpu=0,
        bootstrap_history=True,
        verify_fn=lambda **_kwargs: verified,
    )

    assert state["training"]["python"]["path"] == str(python.absolute())
    assert state["training"]["python"]["target_path"] == str(base.resolve())
    assert iteration._dose_argv(state, go=False)[-3] == str(python.absolute())


def _fake_turn(tmp_path: Path, verified: dict) -> dict:
    parent = _write(tmp_path / "next-parent.pt", "parent")
    handoff = _write(tmp_path / "handoff.json", "handoff")
    campaign = _write(tmp_path / "campaign.json", "campaign")
    audit = _write(tmp_path / "audit.json", "audit")
    value = {
        "schema_version": iteration.flywheel.SCHEMA,
        "promotion": {
            "handoff": iteration._file_ref(handoff, where="test handoff"),
            "handoff_sha256": "sha256:" + "1" * 64,
            "receipt": iteration._file_ref(handoff, where="test receipt"),
            "transaction_id": "turn-1",
            "dethroned_champion_sha256": "sha256:" + "2" * 64,
        },
        "generator": {
            "checkpoint": iteration._file_ref(parent, where="test parent"),
            "registry_version": 2,
            "agent_identity_sha256": "sha256:" + "3" * 64,
            "search_config_sha256": "sha256:" + "4" * 64,
        },
        "generation": {
            "campaign": iteration._file_ref(campaign, where="test campaign"),
            "campaign_contract_sha256": verified["contract_sha256"],
            "audit": iteration._file_ref(audit, where="test audit"),
            "audit_sha256": "sha256:" + "5" * 64,
            "shard_inventory_sha256": "sha256:" + "6" * 64,
            "selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
        },
        "corpus": {
            "meta": iteration._file_ref(
                tmp_path / "corpus/corpus_meta.json", where="test meta"
            )
        },
        "learner_parent": iteration._file_ref(parent, where="test learner parent"),
        "evaluation_parent": iteration._file_ref(parent, where="test eval parent"),
        "initializer": {
            "mode": "exact_parent",
            "checkpoint": iteration._file_ref(parent, where="test init"),
            "receipt": None,
        },
    }
    value["turn_sha256"] = iteration.flywheel._digest(value)  # noqa: SLF001
    return value


def _initialize_next_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, state_name: str = "next.json"
) -> tuple[Path, dict, dict]:
    verified, paths = _verified(tmp_path)
    turn = _fake_turn(tmp_path, verified)
    monkeypatch.setattr(
        iteration.flywheel, "verify_turn", lambda _path, *, verified: turn
    )
    state_path = tmp_path / state_name
    state = iteration.initialize_next(
        state_path=state_path,
        turn_path=tmp_path / f"{state_name}.turn.json",
        handoff_path=tmp_path / "handoff.json",
        campaign_path=tmp_path / "campaign.json",
        audit_path=tmp_path / "audit.json",
        lock_path=paths["lock"],
        data_path=paths["data"],
        validation_path=paths["validation"],
        learner_parent=tmp_path / "next-parent.pt",
        evaluation_parent=tmp_path / "next-parent.pt",
        initializer=tmp_path / "next-parent.pt",
        architecture_upgrade_receipt=None,
        checkpoint=tmp_path / f"{state_name}.candidate.pt",
        report=tmp_path / f"{state_name}.report.json",
        training_receipt=tmp_path / f"{state_name}.receipt.json",
        python=Path(sys.executable),
        gpu=0,
        verify_fn=lambda **_kwargs: verified,
        turn_builder=lambda **_kwargs: turn,
    )
    return state_path, state, verified


def test_initialize_next_claims_fresh_corpus_once_and_rejects_cross_turn_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, state, verified = _initialize_next_fake(tmp_path, monkeypatch)
    assert state["training"]["initialization_mode"] == "next_turn"
    assert state["history"][0]["action"] == "initialize_next"
    same = iteration.initialize_next(
        state_path=tmp_path / "next.json",
        turn_path=tmp_path / "next.json.turn.json",
        handoff_path=tmp_path / "handoff.json",
        campaign_path=tmp_path / "campaign.json",
        audit_path=tmp_path / "audit.json",
        lock_path=tmp_path / "contract.lock.json",
        data_path=tmp_path / "corpus",
        validation_path=tmp_path / "validation.json",
        learner_parent=tmp_path / "next-parent.pt",
        evaluation_parent=tmp_path / "next-parent.pt",
        initializer=tmp_path / "next-parent.pt",
        architecture_upgrade_receipt=None,
        checkpoint=tmp_path / "next.json.candidate.pt",
        report=tmp_path / "next.json.report.json",
        training_receipt=tmp_path / "next.json.receipt.json",
        python=Path(sys.executable),
        gpu=0,
        verify_fn=lambda **_kwargs: verified,
        turn_builder=lambda **_kwargs: _fake_turn(tmp_path, verified),
    )
    assert same["iteration_id"] == state["iteration_id"]

    with pytest.raises(iteration.IterationError, match="already consumed"):
        iteration.initialize_next(
            state_path=tmp_path / "other-state.json",
            turn_path=tmp_path / "other-turn.json",
            handoff_path=tmp_path / "handoff.json",
            campaign_path=tmp_path / "campaign.json",
            audit_path=tmp_path / "audit.json",
            lock_path=tmp_path / "contract.lock.json",
            data_path=tmp_path / "corpus",
            validation_path=tmp_path / "validation.json",
            learner_parent=tmp_path / "next-parent.pt",
            evaluation_parent=tmp_path / "next-parent.pt",
            initializer=tmp_path / "next-parent.pt",
            architecture_upgrade_receipt=None,
            checkpoint=tmp_path / "other.pt",
            report=tmp_path / "other.report.json",
            training_receipt=tmp_path / "other.receipt.json",
            python=Path(sys.executable),
            gpu=0,
            verify_fn=lambda **_kwargs: verified,
            turn_builder=lambda **_kwargs: _fake_turn(tmp_path, verified),
        )


def test_initialize_next_resumes_exact_turn_and_claim_after_state_write_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified, paths = _verified(tmp_path)
    turn = _fake_turn(tmp_path, verified)
    state_path = tmp_path / "next.json"
    turn_path = tmp_path / "next.turn.json"
    kwargs = {
        "state_path": state_path,
        "turn_path": turn_path,
        "handoff_path": tmp_path / "handoff.json",
        "campaign_path": tmp_path / "campaign.json",
        "audit_path": tmp_path / "audit.json",
        "lock_path": paths["lock"],
        "data_path": paths["data"],
        "validation_path": paths["validation"],
        "learner_parent": tmp_path / "next-parent.pt",
        "evaluation_parent": tmp_path / "next-parent.pt",
        "initializer": tmp_path / "next-parent.pt",
        "architecture_upgrade_receipt": None,
        "checkpoint": tmp_path / "candidate.pt",
        "report": tmp_path / "report.json",
        "training_receipt": tmp_path / "receipt.json",
        "python": Path(sys.executable),
        "gpu": 0,
        "verify_fn": lambda **_kwargs: verified,
        "turn_builder": lambda **_kwargs: turn,
    }
    monkeypatch.setattr(
        iteration.flywheel, "verify_turn", lambda _path, *, verified: turn
    )
    real_write_state = iteration._write_state
    monkeypatch.setattr(
        iteration,
        "_write_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("simulated crash")),
    )

    with pytest.raises(OSError, match="simulated crash"):
        iteration.initialize_next(**kwargs)

    assert turn_path.is_file()
    assert (paths["data"] / ".a1-flywheel-corpus-consumption.json").is_file()
    assert not state_path.exists()

    monkeypatch.setattr(iteration, "_write_state", real_write_state)
    resumed = iteration.initialize_next(**kwargs)

    assert resumed["stage"] == "corpus_verified"
    assert resumed["training"]["flywheel_turn"]["path"] == str(turn_path)


def test_next_turn_tamper_refuses_before_dose_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path, state, verified = _initialize_next_fake(tmp_path, monkeypatch)
    turn_path = Path(state["training"]["flywheel_turn"]["path"])
    turn_path.chmod(0o644)
    turn_path.write_text("tampered", encoding="utf-8")
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("runner must not launch")

    with pytest.raises(iteration.IterationError, match="hash drift"):
        iteration.dose_dry_run(
            state_path=state_path,
            runner=runner,
            verify_fn=lambda **_kwargs: verified,
        )
    assert called is False


def test_next_turn_upgrade_receipt_is_forwarded_to_one_dose(tmp_path: Path) -> None:
    receipt = _write(tmp_path / "upgrade.receipt.json", "receipt")
    turn_path = tmp_path / "turn.json"
    turn_path.write_text(
        json.dumps({"initializer": {"receipt": {"path": str(receipt)}}}),
        encoding="utf-8",
    )
    state = {
        "training": {
            "initialization_mode": "next_turn",
            "flywheel_turn": {"path": str(turn_path)},
            "lock": {"path": "lock"},
            "data": "data",
            "validation_manifest": {"path": "validation"},
            "checkpoint": "candidate",
            "report": "report",
            "receipt": "training-receipt",
            "python": {"path": "python"},
            "gpu": 0,
        }
    }
    argv = iteration._dose_argv(state, go=False)
    index = argv.index("--architecture-upgrade-receipt")
    assert argv[index + 1] == str(receipt)


def test_bootstrap_initialize_is_explicit(tmp_path: Path) -> None:
    verified, paths = _verified(tmp_path)
    with pytest.raises(iteration.IterationError, match="explicit"):
        iteration.initialize(
            state_path=tmp_path / "iteration.json",
            lock_path=paths["lock"],
            data_path=paths["data"],
            validation_path=paths["validation"],
            checkpoint=tmp_path / "candidate.pt",
            report=tmp_path / "report.json",
            training_receipt=tmp_path / "receipt.json",
            python=Path(sys.executable),
            gpu=0,
            bootstrap_history=False,
            verify_fn=lambda **_kwargs: verified,
        )


def test_state_tampering_is_rejected(tmp_path: Path) -> None:
    state_path, _ = _initialize(tmp_path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["training"]["gpu"] = 7
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(iteration.IterationError, match="digest mismatch"):
        iteration.status(state_path=state_path)


def test_dose_dry_run_records_exact_authoritative_plan(tmp_path: Path) -> None:
    state_path, state = _initialize(tmp_path)
    expected = _dry_plan(state)

    state = iteration.dose_dry_run(state_path=state_path, runner=_runner(expected))

    assert state["stage"] == "dose_dry_run"
    assert state["training_plan"] == expected
    # The completed stage is idempotent and does not invoke the tool again.
    resumed = iteration.dose_dry_run(
        state_path=state_path,
        runner=lambda *_args, **_kwargs: pytest.fail("must not rerun dry-run"),
    )
    assert resumed["state_sha256"] == state["state_sha256"]


def test_tool_refusal_does_not_advance_state(tmp_path: Path) -> None:
    state_path, state = _initialize(tmp_path)

    with pytest.raises(iteration.IterationError, match="exit 2"):
        iteration.dose_dry_run(
            state_path=state_path,
            runner=_runner({}, returncode=2, stderr="REFUSED: unsafe"),
        )

    assert iteration.status(state_path=state_path)["stage"] == "corpus_verified"


def test_dose_go_adopts_completed_receipt_after_orchestrator_crash(
    tmp_path: Path, monkeypatch
) -> None:
    state_path, state = _advance_to_complete(tmp_path, monkeypatch)

    assert state["stage"] == "dose_complete"
    assert state["training_outputs"]["checkpoint"]["sha256"] == iteration._file_sha256(
        Path(state["training"]["checkpoint"])
    )
    assert state["history"][-1]["action"] == "dose_go_or_resume"


def test_dose_go_accepts_noisy_trainer_stdout_and_verifies_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    state_path, state = _advance_to_dry(tmp_path)
    training = state["training"]
    checkpoint = Path(training["checkpoint"])
    optimizer = Path(str(checkpoint) + ".optimizer.pt")
    report = Path(training["report"])
    receipt = Path(training["receipt"])
    outputs = {
        "checkpoint": str(checkpoint),
        "optimizer_sidecar": str(optimizer),
        "report": str(report),
    }

    def noisy_go(*_args, **_kwargs):
        _write(checkpoint, "candidate")
        _write(optimizer, "optimizer")
        _write(report, "report")
        _write(receipt, "receipt")
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                json.dumps({"mode": "go"})
                + "\n"
                + json.dumps({"progress": "epoch", "step": 1})
                + "\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(
        iteration,
        "_load_complete_training_receipt",
        lambda _state: {"outputs": outputs},
    )

    completed = iteration.dose_go(state_path=state_path, runner=noisy_go)

    assert completed["stage"] == "dose_complete"
    assert completed["training_outputs"]["checkpoint"]["path"] == str(checkpoint)


def test_training_receipt_must_match_dry_run_execution_environment(
    tmp_path: Path,
) -> None:
    _state_path, state = _advance_to_dry(tmp_path)
    payload = {
        "schema_version": iteration.one_dose.RECEIPT_SCHEMA,
        "status": "complete",
        "contract_sha256": state["training"]["contract_sha256"],
        "command_sha256": state["training_plan"]["command_sha256"],
        "execution_binding": {
            **state["training_plan"]["execution_binding"],
            "command_sha256": "sha256:" + "9" * 64,
        },
    }
    payload["receipt_sha256"] = iteration.one_dose._value_sha256(payload)
    Path(state["training"]["receipt"]).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(iteration.IterationError, match="environment/command differs"):
        iteration._load_complete_training_receipt(state)


def _remove_option(command: list[str], flag: str) -> list[str]:
    changed = list(command)
    index = changed.index(flag)
    del changed[index : index + 2]
    return changed


def _training_report(verified: dict, checkpoint: Path) -> dict:
    recipe = verified["recipe"]
    return {
        "arch": "entity_graph",
        **iteration.one_dose.SEALED_A1_MODEL_REPORT,
        "a1_contract_sha256": verified["contract_sha256"],
        "a1_bound_learner_training_recipe": recipe,
        "a1_bound_learner_value_objective": verified["objective"],
        "a1_learner_training_recipe_sha256": iteration.one_dose._value_sha256(recipe),
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
        "steps_completed": 1,
        "total_training_steps": 1,
        "require_35m_model": True,
        "parameter_count": 35_000_000,
        "value_training": {
            "primary_readout": "scalar",
            "trained_value_readouts": ["scalar"],
            "optimizer_steps": 1,
            "completed_epochs": 1,
            "a1_contract_sha256": verified["contract_sha256"],
            "a1_selected_game_seed_set_sha256": verified[
                "selected_game_seed_set_sha256"
            ],
            "a1_training_game_seed_set_sha256": verified[
                "training_game_seed_set_sha256"
            ],
            "a1_learner_training_recipe_sha256": iteration.one_dose._value_sha256(
                recipe
            ),
            "a1_memmap_payload_inventory_sha256": verified["payload_inventory_sha256"],
        },
        "metrics": [
            {
                "epoch": 1,
                "loss": 1.0,
                "policy_loss": 0.8,
                "value_loss": 0.2,
                "validation": {
                    "samples": verified["validation_row_count"],
                    "loss": 1.1,
                },
            }
        ],
    }


def _completed_retry(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    torch = pytest.importorskip("torch")
    verified, paths = _verified(tmp_path)
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
        paths["producer"],
    )
    verified["producer"]["sha256"] = iteration._file_sha256(paths["producer"])
    parent_checkpoint = tmp_path / "r1" / "candidate.pt"
    parent_report = tmp_path / "r1" / "report.json"
    parent_receipt = tmp_path / "r1" / "training.receipt.json"
    parent_command = iteration.one_dose.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=parent_checkpoint,
        report=parent_report,
    )
    parent_command = _remove_option(parent_command, "--graph-layers")
    parent_command = _remove_option(parent_command, "--hidden-size")
    with pytest.raises(iteration.one_dose.ExecutorError, match="exited nonzero"):
        iteration.one_dose.execute(
            verified=verified,
            command=parent_command,
            checkpoint=parent_checkpoint,
            report=parent_report,
            receipt=parent_receipt,
            gpu=0,
            runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                parent_command, 1
            ),
            probe=lambda _gpu: "NVIDIA B200",
        )
    parent_claim = iteration.one_dose._claim_path(verified)
    checkpoint = tmp_path / "r2" / "candidate.pt"
    report = tmp_path / "r2" / "report.json"
    receipt = tmp_path / "r2" / "training.receipt.json"
    retry_contract = tmp_path / "r2" / "learner-retry.contract.json"
    command = iteration.one_dose.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=checkpoint,
        report=report,
    )
    derived = iteration.one_dose.authorize_failed_before_optimizer_retry(
        verified=verified,
        parent_claim=parent_claim,
        retry_command=command,
        checkpoint=checkpoint,
        report=report,
        receipt=receipt,
        retry_contract_path=retry_contract,
        publish=True,
    )

    def run_retry(*_args, **_kwargs):
        _write(checkpoint, "candidate")
        _write(Path(str(checkpoint) + ".optimizer.pt"), "optimizer")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(_training_report(verified, checkpoint)))
        return subprocess.CompletedProcess(command, 0)

    iteration.one_dose.execute(
        verified=derived,
        command=command,
        checkpoint=checkpoint,
        report=report,
        receipt=receipt,
        gpu=0,
        runner=run_retry,
        probe=lambda _gpu: "NVIDIA B200",
    )
    paths.update(
        {
            "parent_claim": parent_claim,
            "parent_receipt": parent_receipt,
            "retry_contract": retry_contract,
            "retry_receipt": receipt,
            "checkpoint": checkpoint,
        }
    )
    return verified, paths


def _adopt_retry(tmp_path: Path, verified: dict, paths: dict[str, Path]) -> dict:
    return iteration.adopt_completed_retry(
        state_path=tmp_path / "iteration.json",
        lock_path=paths["lock"],
        data_path=paths["data"],
        validation_path=paths["validation"],
        parent_claim=paths["parent_claim"],
        retry_contract=paths["retry_contract"],
        retry_receipt=paths["retry_receipt"],
        python=Path(sys.executable),
        gpu=0,
        verify_fn=lambda **_kwargs: verified,
    )


def test_adopt_completed_v4_retry_without_rerunning_training(tmp_path: Path) -> None:
    verified, paths = _completed_retry(tmp_path)

    state = _adopt_retry(tmp_path, verified, paths)
    resumed = _adopt_retry(tmp_path, verified, paths)

    assert state["stage"] == "dose_complete"
    assert state["training"]["attempt_kind"] == "derived-retry-v4"
    assert state["training_outputs"]["parent_claim"]["path"] == str(
        paths["parent_claim"]
    )
    assert state["training_outputs"]["retry_contract"]["path"] == str(
        paths["retry_contract"]
    )
    assert state["training_outputs"]["receipt"]["path"] == str(paths["retry_receipt"])
    assert resumed["state_sha256"] == state["state_sha256"]
    assert (
        iteration.status(state_path=tmp_path / "iteration.json")["state_sha256"]
        == state["state_sha256"]
    )


@pytest.mark.parametrize(
    "drift",
    ["retry_contract", "retry_identity", "parent", "command", "output"],
)
def test_adopt_retry_refuses_every_evidence_link_drift(
    tmp_path: Path, drift: str
) -> None:
    verified, paths = _completed_retry(tmp_path)
    if drift in {"retry_contract", "retry_identity"}:
        path = paths["retry_contract"]
        path.chmod(0o644)
        payload = json.loads(path.read_text())
        if drift == "retry_contract":
            payload["preserved_bindings"]["data_fingerprint"] = "sha256:" + "9" * 64
        else:
            payload["retry_identity_sha256"] = "sha256:" + "9" * 64
        payload.pop("retry_contract_sha256")
        payload["retry_contract_sha256"] = iteration.one_dose._value_sha256(payload)
        path.write_text(json.dumps(payload))
    elif drift == "parent":
        path = paths["parent_receipt"]
        path.chmod(0o644)
        payload = json.loads(path.read_text())
        payload["failure"] = "ExecutorError: different failure"
        payload.pop("receipt_sha256")
        payload["receipt_sha256"] = iteration.one_dose._value_sha256(payload)
        path.write_text(json.dumps(payload))
    elif drift == "command":
        path = paths["retry_receipt"]
        path.chmod(0o644)
        payload = json.loads(path.read_text())
        index = payload["command"].index("--lr") + 1
        payload["command"][index] = "0.123"
        payload["command_sha256"] = iteration.one_dose._value_sha256(payload["command"])
        payload["execution_binding"]["command_sha256"] = payload["command_sha256"]
        payload.pop("receipt_sha256")
        payload["receipt_sha256"] = iteration.one_dose._value_sha256(payload)
        path.write_text(json.dumps(payload))
    else:
        paths["checkpoint"].chmod(0o644)
        paths["checkpoint"].write_text("mutated candidate")

    with pytest.raises(iteration.IterationError):
        _adopt_retry(tmp_path, verified, paths)


def _promotion_plan(state: dict, adjudication: Path) -> dict:
    return {
        "schema_version": iteration.promotion.RECEIPT_SCHEMA,
        "status": "dry_run",
        "transaction_id": "tx-dry",
        "contract": {"contract_sha256": state["training"]["contract_sha256"]},
        "adjudication": {
            "path": str(adjudication),
            "adjudication_sha256": "sha256:" + "2" * 64,
        },
        "training_receipt": state["training_outputs"]["receipt"],
        "candidate": {
            "path": state["training_outputs"]["checkpoint"]["path"],
            "sha256": state["training_outputs"]["checkpoint"]["sha256"],
            "training_report": state["training_outputs"]["report"],
        },
        "champion": {"path": "/champion.pt", "sha256": "sha256:" + "3" * 64},
        "evidence": [{"kind": "mechanism_calibration"}],
        "promotion_cohort_disjointness": {
            "manifest": {"sha256": "sha256:" + "6" * 64},
            "status": "disjoint",
        },
        "registry": {"before_sha256": "sha256:" + "4" * 64},
        "current_pointer": {"before_sha256": "sha256:" + "5" * 64},
    }


def test_evaluation_stage_reuses_promotion_verifier_and_binds_this_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    state_path, state = _advance_to_complete(tmp_path, monkeypatch)
    registry = _write(tmp_path / "registry.json", "registry")
    pointer = _write(tmp_path / "CURRENT_CHAMPION", "champion\n")
    adjudication = _write(tmp_path / "adjudication.json", "adjudication")
    cohort_exclusions = _write(
        tmp_path / "cohort-exclusions.json", "cohort exclusions"
    )
    promotion_receipt = tmp_path / "promotion.receipt.json"
    plan = _promotion_plan(state, adjudication)
    calls: list[dict] = []

    def verify(**kwargs):
        calls.append(kwargs)
        assert kwargs["go"] is False
        return plan

    state = iteration.verify_evaluation(
        state_path=state_path,
        registry_path=registry,
        current_pointer=pointer,
        adjudication_path=adjudication,
        cohort_exclusions=cohort_exclusions,
        promotion_receipt=promotion_receipt,
        reason="candidate passed all typed A1 gates",
        promotion_fn=verify,
    )

    assert state["stage"] == "evaluation_verified"
    assert len(calls) == 1
    assert calls[0]["cohort_exclusions"] == cohort_exclusions
    assert state["evaluation"]["dry_run_plan"] == plan
    assert state["evaluation"]["cohort_exclusions"] == {
        "path": str(cohort_exclusions.resolve()),
        "sha256": iteration._file_sha256(cohort_exclusions),
    }


def test_evaluation_refuses_candidate_from_a_different_dose(
    tmp_path: Path, monkeypatch
) -> None:
    state_path, state = _advance_to_complete(tmp_path, monkeypatch)
    registry = _write(tmp_path / "registry.json", "registry")
    pointer = _write(tmp_path / "CURRENT_CHAMPION", "champion\n")
    adjudication = _write(tmp_path / "adjudication.json", "adjudication")
    cohort_exclusions = _write(
        tmp_path / "cohort-exclusions.json", "cohort exclusions"
    )
    plan = _promotion_plan(state, adjudication)
    plan["candidate"]["sha256"] = "sha256:" + "9" * 64

    with pytest.raises(iteration.IterationError, match="other than this dose"):
        iteration.verify_evaluation(
            state_path=state_path,
            registry_path=registry,
            current_pointer=pointer,
            adjudication_path=adjudication,
            cohort_exclusions=cohort_exclusions,
            promotion_receipt=tmp_path / "promotion.receipt.json",
            reason="wrong candidate",
            promotion_fn=lambda **_kwargs: plan,
        )

    assert iteration.status(state_path=state_path)["stage"] == "dose_complete"


def test_evaluation_stage_refuses_cohort_exclusion_drift(
    tmp_path: Path, monkeypatch
) -> None:
    state_path, state = _advance_to_complete(tmp_path, monkeypatch)
    registry = _write(tmp_path / "registry.json", "registry")
    pointer = _write(tmp_path / "CURRENT_CHAMPION", "champion\n")
    adjudication = _write(tmp_path / "adjudication.json", "adjudication")
    cohort_exclusions = _write(
        tmp_path / "cohort-exclusions.json", "cohort exclusions"
    )
    state = iteration.verify_evaluation(
        state_path=state_path,
        registry_path=registry,
        current_pointer=pointer,
        adjudication_path=adjudication,
        cohort_exclusions=cohort_exclusions,
        promotion_receipt=tmp_path / "promotion.receipt.json",
        reason="passed",
        promotion_fn=lambda **_kwargs: _promotion_plan(state, adjudication),
    )
    cohort_exclusions.write_text("drifted", encoding="utf-8")

    with pytest.raises(iteration.IterationError, match="cohort_exclusions hash drift"):
        iteration.status(state_path=state_path)


def test_promotion_go_reuses_exact_preflight_cohort_exclusions(
    tmp_path: Path, monkeypatch
) -> None:
    state_path, state = _advance_to_complete(tmp_path, monkeypatch)
    registry = _write(tmp_path / "registry.json", "registry")
    pointer = _write(tmp_path / "CURRENT_CHAMPION", "champion\n")
    adjudication = _write(tmp_path / "adjudication.json", "adjudication")
    cohort_exclusions = _write(
        tmp_path / "cohort-exclusions.json", "cohort exclusions"
    )
    promotion_receipt = tmp_path / "promotion.receipt.json"
    plan = _promotion_plan(state, adjudication)
    state = iteration.verify_evaluation(
        state_path=state_path,
        registry_path=registry,
        current_pointer=pointer,
        adjudication_path=adjudication,
        cohort_exclusions=cohort_exclusions,
        promotion_receipt=promotion_receipt,
        reason="passed",
        promotion_fn=lambda **_kwargs: plan,
    )
    calls: list[dict] = []

    def commit(**kwargs):
        calls.append(kwargs)
        _write(promotion_receipt, "committed receipt")
        return {}

    committed = {
        "transaction_id": "tx-committed",
        "registry": {"after_sha256": "sha256:" + "6" * 64},
        "current_pointer": {"after_sha256": "sha256:" + "7" * 64},
        "promotion_count": 4,
    }
    monkeypatch.setattr(
        iteration, "_adopt_committed_promotion", lambda _state: committed
    )

    promoted = iteration.promote(state_path=state_path, promotion_fn=commit)

    assert promoted["stage"] == "promoted"
    assert calls[0]["go"] is True
    assert calls[0]["cohort_exclusions"] == cohort_exclusions.resolve()


def test_promote_adopts_committed_transaction_after_orchestrator_crash(
    tmp_path: Path, monkeypatch
) -> None:
    state_path, state = _advance_to_complete(tmp_path, monkeypatch)
    registry = _write(tmp_path / "registry.json", "registry")
    pointer = _write(tmp_path / "CURRENT_CHAMPION", "champion\n")
    adjudication = _write(tmp_path / "adjudication.json", "adjudication")
    cohort_exclusions = _write(
        tmp_path / "cohort-exclusions.json", "cohort exclusions"
    )
    promotion_receipt = tmp_path / "promotion.receipt.json"
    plan = _promotion_plan(state, adjudication)
    state = iteration.verify_evaluation(
        state_path=state_path,
        registry_path=registry,
        current_pointer=pointer,
        adjudication_path=adjudication,
        cohort_exclusions=cohort_exclusions,
        promotion_receipt=promotion_receipt,
        reason="passed",
        promotion_fn=lambda **_kwargs: plan,
    )
    _write(promotion_receipt, "committed receipt")
    committed = {
        "transaction_id": "tx-committed",
        "registry": {"after_sha256": "sha256:" + "6" * 64},
        "current_pointer": {"after_sha256": "sha256:" + "7" * 64},
        "promotion_count": 4,
    }
    monkeypatch.setattr(
        iteration, "_adopt_committed_promotion", lambda _state: committed
    )

    state = iteration.promote(
        state_path=state_path,
        promotion_fn=lambda **_kwargs: pytest.fail(
            "must not repeat committed promotion"
        ),
    )

    assert state["stage"] == "promoted"
    assert state["promotion"]["transaction_id"] == "tx-committed"
    assert state["promotion"]["promotion_count"] == 4


def test_committed_promotion_must_match_dry_run_mutation_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    registry = _write(tmp_path / "registry.json", "after-registry")
    pointer = _write(tmp_path / "CURRENT_CHAMPION", "candidate\n")
    receipt_path = _write(tmp_path / "promotion.receipt.json", "receipt")
    expected = {
        "registry": {
            "before_sha256": "sha256:" + "1" * 64,
            "after_sha256": iteration.promotion._sha256(registry),
        },
        "current_pointer": {
            "before_sha256": "sha256:" + "2" * 64,
            "after_sha256": iteration.promotion._sha256(pointer),
        },
        "contract": {"contract_sha256": "sha256:" + "a" * 64},
        "adjudication": {"adjudication_sha256": "sha256:" + "b" * 64},
        "candidate": {"sha256": "sha256:" + "c" * 64},
        "champion": {"sha256": "sha256:" + "d" * 64},
        "evidence": [],
        "promotion_cohort_disjointness": {
            "manifest": {"sha256": "sha256:" + "e" * 64},
            "status": "disjoint",
        },
        "promotion_count": 1,
        "nth_confirmation_required": False,
        "reason": "passed",
        "fleet_ckpt_updated": False,
    }
    receipt = {
        **expected,
        "status": "committed",
        "registry": {**expected["registry"], "before_sha256": "sha256:" + "9" * 64},
    }
    monkeypatch.setattr(
        iteration.promotion,
        "_load_recovery_receipt",
        lambda _path: (receipt, receipt_path, registry, pointer, registry, pointer),
    )
    state = {
        "evaluation": {
            "promotion_receipt": str(receipt_path),
            "dry_run_plan": expected,
        }
    }

    with pytest.raises(iteration.IterationError, match="dry-run field 'registry'"):
        iteration._adopt_committed_promotion(state)


def test_committed_promotion_must_match_preflight_cohort_disjointness(
    tmp_path: Path, monkeypatch
) -> None:
    registry = _write(tmp_path / "registry.json", "after-registry")
    pointer = _write(tmp_path / "CURRENT_CHAMPION", "candidate\n")
    receipt_path = _write(tmp_path / "promotion.receipt.json", "receipt")
    expected = {
        "registry": {"after_sha256": iteration.promotion._sha256(registry)},
        "current_pointer": {"after_sha256": iteration.promotion._sha256(pointer)},
        "contract": {},
        "adjudication": {},
        "training_receipt": {},
        "candidate": {},
        "champion": {},
        "evidence": [],
        "promotion_cohort_disjointness": {"status": "disjoint"},
        "promotion_count": 1,
        "nth_confirmation_required": False,
        "reason": "passed",
        "fleet_ckpt_updated": False,
    }
    receipt = {
        **expected,
        "status": "committed",
        "promotion_cohort_disjointness": {"status": "different"},
    }
    monkeypatch.setattr(
        iteration.promotion,
        "_load_recovery_receipt",
        lambda _path: (receipt, receipt_path, registry, pointer, registry, pointer),
    )
    state = {
        "evaluation": {
            "promotion_receipt": str(receipt_path),
            "dry_run_plan": expected,
        }
    }

    with pytest.raises(
        iteration.IterationError,
        match="dry-run field 'promotion_cohort_disjointness'",
    ):
        iteration._adopt_committed_promotion(state)
