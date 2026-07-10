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


def _lock(*, n_full: int = 128, producer: Path = Path("/producer.pt")) -> dict:
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
    return {
        "lock": _lock(producer=producer),
        "lock_path": lock_path,
        "lock_file_sha256": executor._file_sha256(lock_path),
        "contract_sha256": _SHA,
        "recipe": dict(contract.EXPECTED_LEARNER_TRAINING_RECIPE),
        "objective": _objective(),
        "producer": _lock(producer=producer)["checkpoints"][0],
        "data_path": data,
        "corpus_meta_file_sha256": executor._file_sha256(
            data / "corpus_meta.json"
        ),
        "payload_inventory_sha256": "sha256:" + "c" * 64,
        "selected_game_seed_set_sha256": "sha256:" + "d" * 64,
        "training_game_seed_set_sha256": "sha256:" + "e" * 64,
        "validation_path": validation,
        "validation_file_sha256": executor._file_sha256(validation),
        "validation_game_seed_set_sha256": "sha256:" + "f" * 64,
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

    meta = {"payload_inventory_sha256": "sha256:" + "c" * 64}
    validation = {
        "a1_contract_sha256": _SHA,
        "file_sha256": executor._file_sha256(validation_path),
        "validation_game_seed_set_sha256": "sha256:" + "f" * 64,
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
        report.write_text(
            json.dumps(
                {
                    "a1_contract_sha256": _SHA,
                    "a1_bound_learner_training_recipe": verified["recipe"],
                    "world_size": 1,
                    "optimizer": "adam",
                    "resume_optimizer": False,
                    "optimizer_restored": False,
                    "fused_optimizer": False,
                    "epochs": 1,
                    "steps_completed": 743,
                }
            )
        )
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
    assert payload["outputs"]["steps_completed"] == 743
    assert payload["receipt_sha256"].startswith("sha256:")
    assert not Path(str(receipt) + ".claim").exists()
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


def test_failure_is_receipted_and_claim_is_released(tmp_path: Path) -> None:
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
    assert not Path(str(receipt) + ".claim").exists()


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
