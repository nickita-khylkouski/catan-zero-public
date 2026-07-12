from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from tools import a1_one_dose_train as executor
from tools import a1_pre_wave_contract as contract
from tools import train_bc


def _verified() -> dict[str, object]:
    train_path = Path(train_bc.__file__).resolve()
    selfplay_path = (
        Path(__file__).resolve().parents[1]
        / "src/catan_zero/rl/gumbel_self_play.py"
    ).resolve()
    lock = {
        "provenance": {
            "learner_code": [{"path": str(train_path)}],
            "runtime_code_tree": [{"path": str(selfplay_path)}],
        }
    }
    return {
        "recipe": dict(contract.EXPECTED_LEARNER_TRAINING_RECIPE),
        "contract_sha256": "sha256:" + "a" * 64,
        "lock": lock,
        "producer": {"path": "/sealed/champion.pt", "sha256": "sha256:" + "b" * 64},
        "data_path": Path("/sealed/a1-memmap"),
        "validation_path": Path("/sealed/holdout.json"),
        "payload_inventory_sha256": "sha256:" + "d" * 64,
        "lock_file_sha256": "sha256:" + "c" * 64,
        "reviewed_lock_file_sha256": "sha256:" + "c" * 64,
    }


def _bind(**overrides: object) -> dict[str, object]:
    verified = _verified()
    code = executor._current_ablation_code_binding(verified["lock"])
    return executor.bind_learner_ablation(
        verified,
        ablation_id="pure-distill-value-balanced",
        overrides_json=json.dumps(overrides),
        reviewed_code_tree_sha256=code["code_tree_sha256"],
    )


def test_ablation_reuses_existing_weighting_knobs_and_binds_exact_drift() -> None:
    result = _bind(
        soft_target_weight=1.0,
        loser_sample_weight=1.0,
        value_loss_weight=1.0,
        value_lr_mult=1.0,
        per_game_value_weight=True,
        per_game_value_weight_mode="sqrt",
        forced_row_value_weight=0.1,
    )
    ablation = result["learner_ablation"]
    assert result["bound_recipe"] == contract.EXPECTED_LEARNER_TRAINING_RECIPE
    assert result["recipe"]["per_game_value_weight"] is True
    assert set(ablation["recipe_drift"]) == {
        "forced_row_value_weight",
        "per_game_value_weight",
        "per_game_value_weight_mode",
        "soft_target_weight",
        "value_loss_weight",
        "value_lr_mult",
    }
    assert ablation["diagnostic_only"] is True
    assert ablation["promotion_eligible"] is False
    assert ablation["code_tree_sha256"].startswith("sha256:")
    assert result["claim_identity_sha256"] != result["contract_sha256"]


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"graph_layers": 7},
        {"mask_hidden_info": False},
        {"batch_size": 8192},
        {"value_loss_weight": 1},  # int must not impersonate a float
        {"value_loss_weight": 0.25},  # explicit no-op
        {"soft_target_weight": 2.0},
        {"forced_row_value_weight": -1.0},
        {"value_target_lambda": 1.1},
        {"lr_schedule": "banana"},
        {"per_game_value_weight_mode": "sqrt"},
        {"soft_target_temperature": 1.0},
        {"q_loss_weight": 0.1},
        {
            "policy_loss_weight": 0.0,
            "value_loss_weight": 0.0,
            "final_vp_loss_weight": 0.0,
            "policy_kl_anchor_weight": 0.0,
        },
    ],
)
def test_ablation_rejects_empty_forbidden_type_drift_and_noop(
    overrides: dict[str, object]
) -> None:
    verified = _verified()
    code = executor._current_ablation_code_binding(verified["lock"])
    with pytest.raises(executor.ExecutorError):
        executor.bind_learner_ablation(
            verified,
            ablation_id="probe",
            overrides_json=json.dumps(overrides),
            reviewed_code_tree_sha256=code["code_tree_sha256"],
        )


def test_ablation_requires_explicit_reviewed_current_code_digest() -> None:
    with pytest.raises(executor.ExecutorError, match="reviewed digest"):
        executor.bind_learner_ablation(
            _verified(),
            ablation_id="probe",
            overrides_json=json.dumps({"value_loss_weight": 1.0}),
            reviewed_code_tree_sha256="sha256:" + "0" * 64,
        )


def test_ablation_rejects_nonfinite_json_number() -> None:
    verified = _verified()
    code = executor._current_ablation_code_binding(verified["lock"])
    with pytest.raises(executor.ExecutorError, match="non-finite"):
        executor.bind_learner_ablation(
            verified,
            ablation_id="probe",
            overrides_json='{"lr": NaN}',
            reviewed_code_tree_sha256=code["code_tree_sha256"],
        )


def test_code_binding_deduplicates_real_learner_runtime_overlap() -> None:
    verified = _verified()
    train_path = str(Path(train_bc.__file__).resolve())
    verified["lock"]["provenance"]["runtime_code_tree"].append(
        {"path": train_path}
    )
    binding = executor._current_ablation_code_binding(verified["lock"])
    relative = [row["relative_path"] for row in binding["records"]]
    assert len(relative) == len(set(relative))
    train_record = next(
        row for row in binding["records"] if row["relative_path"] == "tools/train_bc.py"
    )
    assert train_record["kind"] == "learner_code"


def test_untrusted_lock_verifier_path_is_never_imported(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    malicious = tmp_path / "tools" / "a1_pre_wave_contract.py"
    malicious.parent.mkdir()
    malicious.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
    )
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "contract_sha256": "sha256:" + "a" * 64,
                "provenance": {
                    "runtime_code_tree": [
                        {
                            "path": str(malicious),
                            # Internally self-consistent malicious path/hash;
                            # the independent lineage trust anchor must still
                            # reject it before import.
                            "sha256": executor._file_sha256(malicious),
                        }
                    ]
                },
            }
        )
    )
    with pytest.raises(executor.ExecutorError, match="trust anchor"):
        executor._verify_lock_with_sealed_runtime(
            lock, reviewed_lock_file_sha256=executor._file_sha256(lock)
        )
    assert not marker.exists()


def test_tampered_sealed_dependency_refuses_before_verifier_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sealed"
    tools = root / "tools"
    tools.mkdir(parents=True)
    marker = tmp_path / "verifier-imported"
    verifier = tools / "a1_pre_wave_contract.py"
    verifier.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "def verify_lock(path, require_all_job_claims=False): return {}\n"
    )
    dependency = root / "src" / "dependency.py"
    dependency.parent.mkdir()
    dependency.write_text("SAFE = True\n")
    safe_dependency_sha = executor._file_sha256(dependency)
    lock = tmp_path / "contract.lock.json"
    payload = {
        "contract_sha256": "sha256:" + "a" * 64,
        "provenance": {
            "learner_code": [
                {"path": str(dependency), "sha256": safe_dependency_sha}
            ],
            "runtime_code_tree": [
                {
                    "path": str(verifier),
                    "sha256": executor._file_sha256(verifier),
                }
            ],
        },
    }
    lock.write_text(json.dumps(payload))
    monkeypatch.setattr(executor, "TRUSTED_A1_LOCK_PATH", lock)
    monkeypatch.setattr(
        executor, "TRUSTED_A1_LOCK_FILE_SHA256", executor._file_sha256(lock)
    )
    monkeypatch.setattr(executor, "TRUSTED_A1_VERIFIER_PATH", verifier)
    monkeypatch.setattr(
        executor, "TRUSTED_A1_VERIFIER_SHA256", executor._file_sha256(verifier)
    )
    dependency.write_text("TAMPERED = True\n")
    with pytest.raises(executor.ExecutorError, match="dependency drift before import"):
        executor._verify_lock_with_sealed_runtime(
            lock, reviewed_lock_file_sha256=executor._file_sha256(lock)
        )
    assert not marker.exists()


def test_physical_gpu_lock_refuses_same_gpu_concurrently(tmp_path: Path) -> None:
    with executor._physical_gpu_lock(3, lock_root=tmp_path) as path:
        assert path.name == "catan_zero_a1_b200_gpu3.lock"
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_mode & 0o077 == 0
        with pytest.raises(executor.ExecutorError, match="already reserved"):
            with executor._physical_gpu_lock(3, lock_root=tmp_path):
                raise AssertionError("same-GPU lock unexpectedly acquired")


def test_physical_gpu_lock_allows_distinct_gpus(tmp_path: Path) -> None:
    with executor._physical_gpu_lock(0, lock_root=tmp_path) as first:
        with executor._physical_gpu_lock(1, lock_root=tmp_path) as second:
            assert first != second


def test_physical_gpu_lock_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("")
    executor._gpu_lock_path(2, lock_root=tmp_path).symlink_to(target)
    with pytest.raises(executor.ExecutorError, match="cannot open"):
        with executor._physical_gpu_lock(2, lock_root=tmp_path):
            raise AssertionError("symlink lock unexpectedly acquired")


def test_train_command_and_child_validation_preserve_bound_recipe() -> None:
    result = _bind(
        loser_sample_weight=1.0,
        per_game_value_weight=True,
        per_game_value_weight_mode="sqrt",
        forced_row_value_weight=0.1,
    )
    command = executor.build_train_command(
        result,
        python=Path(sys.executable),
        checkpoint=Path("/tmp/diagnostic.pt"),
        report=Path("/tmp/diagnostic.json"),
    )
    assert "--allow-concurrent-bc" in command
    assert "--per-game-value-weight" in command
    assert command[command.index("--per-game-value-weight-mode") + 1] == "sqrt"
    assert command[command.index("--loser-sample-weight") + 1] == "1.0"
    args = train_bc.build_parser().parse_args(command[2:])
    bound = {
        "learner_training_recipe": dict(contract.EXPECTED_LEARNER_TRAINING_RECIPE),
        "learner_training_recipe_sha256": executor._value_sha256(
            contract.EXPECTED_LEARNER_TRAINING_RECIPE
        ),
    }
    effective = train_bc._validate_a1_learner_training_recipe(
        args,
        {"world_size": 1, "rank": 0, "local_rank": 0, "enabled": False},
        bound,
    )
    assert effective == result["recipe"]
    assert bound["learner_ablation"] == result["learner_ablation"]

    args.a1_learner_ablation = bound["learner_ablation"]
    metadata = train_bc._value_training_metadata(
        args,
        scalar_weight=1.0,
        categorical_weight=0.0,
        categorical_bins=0,
        optimizer_steps=1,
        completed_epochs=1,
        scalar_training_weight_sum=1.0,
        categorical_training_weight_sum=0.0,
    )
    assert metadata["learner_ablation"] == result["learner_ablation"]


def test_default_command_never_disables_the_host_training_lock(tmp_path: Path) -> None:
    verified = _verified()
    sealed_train = tmp_path / "tools" / "train_bc.py"
    sealed_train.parent.mkdir()
    sealed_train.write_text("# sealed fixture\n", encoding="utf-8")
    verified["lock"]["provenance"]["learner_code"] = [
        {"path": str(sealed_train)}
    ]
    command = executor.build_train_command(
        verified,
        python=Path(sys.executable),
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
    )
    assert "--allow-concurrent-bc" not in command
    assert command[1] == str(sealed_train.resolve())
