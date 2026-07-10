from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools import a1_one_dose_train as executor
from tools import a1_pre_wave_contract as contract


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
            "learner_training_recipe": dict(
                contract.EXPECTED_LEARNER_TRAINING_RECIPE
            ),
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
        "corpus_meta_file_sha256": executor._file_sha256(
            data / "corpus_meta.json"
        ),
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
    return {
        "arch": "entity_graph",
        "a1_contract_sha256": verified["contract_sha256"],
        "a1_bound_learner_training_recipe": recipe,
        "a1_bound_learner_value_objective": verified["objective"],
        "a1_learner_training_recipe_sha256": executor._value_sha256(recipe),
        "a1_memmap_payload_inventory_sha256": verified[
            "payload_inventory_sha256"
        ],
        "a1_selected_game_seed_set_sha256": verified[
            "selected_game_seed_set_sha256"
        ],
        "a1_training_game_seed_set_sha256": verified[
            "training_game_seed_set_sha256"
        ],
        "world_size": 1,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "fused_optimizer": False,
        "epochs": 1,
        "max_steps": 0,
        "batch_size": recipe["batch_size"],
        "amp": recipe["amp"],
        "lr": recipe["lr"],
        "weight_decay": recipe["weight_decay"],
        "seed": recipe["seed"],
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
        "validation_game_seed_set_sha256": verified[
            "validation_game_seed_set_sha256"
        ],
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
            "a1_memmap_payload_inventory_sha256": verified[
                "payload_inventory_sha256"
            ],
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


def _option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


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
    assert _option(command, "--epochs") == "1"
    assert _option(command, "--value-head-type") == "mse"
    assert _option(command, "--value-categorical-bins") == "0"
    assert "--mask-hidden-info" in command
    assert "--no-symmetry-augment" in command
    assert "--validation-game-seed-manifest" in command
    assert "--p-full" not in command  # generation choice remains contract-bound.


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

    def fake_runner(command_arg: list[str], **kwargs: object) -> subprocess.CompletedProcess:
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
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert "WORLD_SIZE" not in captured["env"]
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
    assert state["outputs"]["checkpoint_sha256"] == executor._file_sha256(
        checkpoint
    )
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

    with pytest.raises(executor.ExecutorError, match="report invariant drift"):
        executor._verify_training_outputs(
            checkpoint=checkpoint, report=report, verified=verified
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
