from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import a0_binding_verdict as binding  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _slice(
    *,
    n: int,
    brier: float,
    rmse: float,
    categorical: bool,
) -> dict:
    return {
        "n": n,
        "n_win": n // 2,
        "n_loss": n - n // 2,
        "brier": brier,
        "value_rmse": rmse,
        "win_probability_ece": 0.05,
        "reliability_bins": [{"n": n}],
        "corr_q_z": 0.4,
        "categorical_score_n": n if categorical else 0,
    }


def _calibration(
    *,
    readout: str,
    checkpoint: Path,
    seed_manifest: Path,
    seed_manifest_sha: str,
) -> dict:
    categorical = readout == "categorical"
    return {
        "schema_version": binding.CALIBRATION_SCHEMA,
        "value_readout": readout,
        "checkpoint": str(checkpoint),
        "shard_dir": "/sealed/raw/shards",
        "readout_provenance": {
            "requested_readout": readout,
            "model_output_key": "value_categorical" if categorical else "value",
            "categorical_training_verified": categorical,
            "trained_value_readouts": [readout],
            "value_training_schema_version": "value-training-v1",
            "categorical_bins": 33 if categorical else 0,
            "categorical_truncation_class": categorical,
            "categorical_objective_weight": 1.0 if categorical else 0.0,
            "categorical_training_weight_sum": 1000.0 if categorical else 0.0,
            "hlgauss_sigma_ratio": 0.75 if categorical else None,
            "optimizer_steps": 30,
            "completed_epochs": 3,
        },
        "row_selection": {
            "mode": "validation_seed_manifest",
            "held_out_filter_applied": True,
            "seed_manifest_path": str(seed_manifest),
            "seed_manifest_sha256": seed_manifest_sha.removeprefix("sha256:"),
            "configured_game_seed_count": 3,
            "observed_game_seed_count": 3,
            "observed_row_count": 100,
        },
        "global": _slice(
            n=100, brier=0.20, rmse=0.80, categorical=categorical
        ),
        "by_phase": {
            "opening_placement": _slice(
                n=40, brier=0.22, rmse=0.84, categorical=categorical
            )
        },
        "by_legal_count_bucket": {
            "41+": _slice(n=20, brier=0.24, rmse=0.88, categorical=categorical)
        },
    }


@pytest.fixture
def sealed_inputs(tmp_path: Path, monkeypatch) -> dict:
    seeds = [7, 9, 11]
    seed_sha = binding.a0._int64_set_sha(np.asarray(seeds, dtype=np.int64))
    scalar_seed_manifest = tmp_path / "scalar.validation_seeds.json"
    hl_seed_manifest = tmp_path / "hl.validation_seeds.json"
    for path in (scalar_seed_manifest, hl_seed_manifest):
        _write_json(
            path,
            {
                "schema_version": "train-validation-game-seeds-v1",
                "validation_game_seed_count": len(seeds),
                "validation_game_seed_set_sha256": seed_sha,
                "game_seeds": seeds,
            },
        )

    scalar_checkpoint = tmp_path / "scalar.pt"
    hl_checkpoint = tmp_path / "hl.pt"
    scalar_checkpoint.write_bytes(b"scalar checkpoint")
    hl_checkpoint.write_bytes(b"HL checkpoint")
    scalar_stages = binding.policy_probe._checkpoint_stages(scalar_checkpoint)
    hl_stages = binding.policy_probe._checkpoint_stages(hl_checkpoint)
    for stage in ("epoch1", "epoch2", "epoch3"):
        scalar_stages[stage].write_bytes(f"scalar {stage}".encode())
        hl_stages[stage].write_bytes(f"HL {stage}".encode())

    def report(policy_loss: float, prior_kl: float) -> dict:
        epochs = []
        for epoch in range(3):
            validation = {"primary_value_loss": 1.0 - 0.01 * epoch}
            if epoch == 2:
                validation.update(
                    {
                        "policy_loss": policy_loss,
                        "accuracy_active_count": 80,
                        "prior_kl_rows": 60,
                        "prior_kl_model_prior_mean": prior_kl,
                    }
                )
            epochs.append({"validation": validation})
        return {"metrics": epochs}

    scalar_report = tmp_path / "scalar.report.json"
    hl_report = tmp_path / "hl.report.json"
    _write_json(scalar_report, report(1.0, 0.50))
    _write_json(hl_report, report(1.01, 0.505))

    lock_path = tmp_path / "a0.lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    lock = {
        "experiment_id": "a0-test",
        "artifact_root_at_seal": str(tmp_path),
        "input_contract_sha256": "input",
        "recipe_sha256": "recipe",
        "seed_contract_sha256": "seed",
        "validation": {
            "validation_game_seed_set_sha256": seed_sha,
            "validation_game_seed_count_after_row_cap": len(seeds),
        },
        "arm_contracts": {
            "matched_common_sha256": "matched",
            "scalar": {
                "report": str(scalar_report),
                "checkpoint": str(scalar_checkpoint),
            },
            "hlgauss33": {
                "report": str(hl_report),
                "checkpoint": str(hl_checkpoint),
            },
        },
    }
    training_result = {
        "schema_version": binding.a0.RESULT_SCHEMA,
        "scalar_reproduces_historical_failure": True,
        "a0_interpretable": True,
        "hl_training_stable": True,
        "a0_training_loss_gate_pass": True,
        "scalar_primary_validation_trace": [0.6652, 0.8090, 0.8418],
        "historical_scalar_validation_trace": [0.6652, 0.8090, 0.8418],
        "hl_primary_validation_trace": [1.0, 0.99, 0.98],
        "artifacts": {
            "scalar": {
                "report_sha256": _sha(scalar_report),
                "checkpoint_sha256": _sha(scalar_checkpoint),
            },
            "hlgauss33": {
                "report_sha256": _sha(hl_report),
                "checkpoint_sha256": _sha(hl_checkpoint),
            },
        },
    }
    result_path = tmp_path / "a0.result.json"
    _write_json(result_path, training_result)

    scalar_calibration_path = tmp_path / "scalar.calibration.json"
    hl_calibration_path = tmp_path / "hl.calibration.json"
    scalar_calibration = _calibration(
        readout="scalar",
        checkpoint=scalar_checkpoint,
        seed_manifest=scalar_seed_manifest,
        seed_manifest_sha=_sha(scalar_seed_manifest),
    )
    hl_calibration = _calibration(
        readout="categorical",
        checkpoint=hl_checkpoint,
        seed_manifest=hl_seed_manifest,
        seed_manifest_sha=_sha(hl_seed_manifest),
    )
    # A small global improvement and stable critical slices.
    hl_calibration["global"]["brier"] = 0.198
    hl_calibration["global"]["value_rmse"] = 0.804
    hl_calibration["by_phase"]["opening_placement"]["brier"] = 0.225
    hl_calibration["by_phase"]["opening_placement"]["value_rmse"] = 0.85
    hl_calibration["by_legal_count_bucket"]["41+"]["brier"] = 0.245
    hl_calibration["by_legal_count_bucket"]["41+"]["value_rmse"] = 0.89
    _write_json(scalar_calibration_path, scalar_calibration)
    _write_json(hl_calibration_path, hl_calibration)

    policy_stages = {}
    policy_gates = {}
    for stage in binding.policy_probe._STAGES:
        scalar_metrics = {
            "checkpoint": str(scalar_stages[stage]),
            "checkpoint_sha256": _sha(scalar_stages[stage]),
            "samples": 100,
            "accuracy_active_count": 80,
            "prior_kl_rows": 60,
            "policy_loss": 1.0,
            "prior_kl_model_prior_mean": 0.50,
            "prior_kl_target_prior_mean": 0.70,
        }
        hl_metrics = {
            "checkpoint": str(hl_stages[stage]),
            "checkpoint_sha256": _sha(hl_stages[stage]),
            "samples": 100,
            "accuracy_active_count": 80,
            "prior_kl_rows": 60,
            "policy_loss": 1.01,
            "prior_kl_model_prior_mean": 0.505,
            "prior_kl_target_prior_mean": 0.70,
        }
        comparison = binding.policy_probe.compare_stage_metrics(
            scalar_metrics, hl_metrics
        )
        policy_stages[stage] = {
            "scalar": scalar_metrics,
            "hlgauss33": hl_metrics,
            "comparison": comparison,
        }
        policy_gates[stage] = comparison["pass"]
    policy_artifact = {
        "schema_version": binding.policy_probe.SCHEMA,
        "experiment_id": lock["experiment_id"],
        "lock_sha256": _sha(lock_path),
        "input_contract_sha256": lock["input_contract_sha256"],
        "recipe_sha256": lock["recipe_sha256"],
        "seed_contract_sha256": lock["seed_contract_sha256"],
        "matched_common_sha256": lock["arm_contracts"]["matched_common_sha256"],
        "corpus_tree_sha256": "corpus",
        "validation": {
            "scalar": {
                "manifest": str(scalar_seed_manifest),
                "manifest_sha256": _sha(scalar_seed_manifest),
            },
            "hlgauss33": {
                "manifest": str(hl_seed_manifest),
                "manifest_sha256": _sha(hl_seed_manifest),
            },
            "validation_game_seed_set_sha256": seed_sha,
            "validation_game_seed_count": len(seeds),
        },
        "thresholds": {"max_absolute_relative_policy_drift": 0.02},
        "stages": policy_stages,
        "gates": policy_gates,
        "policy_drift_pass": all(policy_gates.values()),
    }
    lock["corpus_tree_sha256"] = "corpus"
    policy_drift_path = tmp_path / "a0.policy_drift.json"
    _write_json(policy_drift_path, policy_artifact)

    monkeypatch.setattr(
        binding.a0, "_load_and_verify_lock", lambda _path, _repo: lock
    )
    monkeypatch.setattr(binding.a0, "_postflight", lambda _lock, _repo: training_result)
    return {
        "lock": lock,
        "lock_path": lock_path,
        "result": training_result,
        "result_path": result_path,
        "scalar_report": scalar_report,
        "hl_report": hl_report,
        "scalar_checkpoint": scalar_checkpoint,
        "hl_checkpoint": hl_checkpoint,
        "scalar_seed_manifest": scalar_seed_manifest,
        "hl_seed_manifest": hl_seed_manifest,
        "scalar_calibration": scalar_calibration,
        "hl_calibration": hl_calibration,
        "scalar_calibration_path": scalar_calibration_path,
        "hl_calibration_path": hl_calibration_path,
        "policy_artifact": policy_artifact,
        "policy_drift_path": policy_drift_path,
        "repo_root": tmp_path,
    }


def _build(inputs: dict) -> dict:
    return binding.build_binding_verdict(
        lock_path=inputs["lock_path"],
        result_path=inputs["result_path"],
        scalar_calibration_path=inputs["scalar_calibration_path"],
        hl_calibration_path=inputs["hl_calibration_path"],
        policy_drift_path=inputs["policy_drift_path"],
        repo_root=inputs["repo_root"],
    )


def test_binding_verdict_passes_only_with_all_sealed_evidence(sealed_inputs: dict) -> None:
    verdict = _build(sealed_inputs)
    assert verdict["a0_binding_pass"] is True
    assert verdict["a0_stage_complete"] is True
    assert verdict["hlgauss_adoption_pass"] is True
    assert all(verdict["gates"].values())
    assert (
        verdict["calibration_artifacts"]["scalar"][
            "validation_game_seed_set_sha256"
        ]
        == sealed_inputs["lock"]["validation"][
            "validation_game_seed_set_sha256"
        ]
    )
    assert verdict["calibration_comparison"]["global"][
        "at_least_one_metric_improves"
    ] is True
    assert verdict["decision"]["learner_objective"] == "hlgauss"


def test_binding_refuses_training_result_not_equal_to_recomputed_postflight(
    sealed_inputs: dict,
) -> None:
    payload = dict(sealed_inputs["result"])
    payload["extra_unsealed_claim"] = True
    _write_json(sealed_inputs["result_path"], payload)
    with pytest.raises(binding.a0.ContractError, match="recomputed postflight"):
        _build(sealed_inputs)


def test_binding_refuses_calibration_checkpoint_hash_drift(sealed_inputs: dict) -> None:
    rogue = sealed_inputs["repo_root"] / "rogue.pt"
    rogue.write_bytes(b"rogue")
    payload = sealed_inputs["hl_calibration"]
    payload["checkpoint"] = str(rogue)
    _write_json(sealed_inputs["hl_calibration_path"], payload)
    with pytest.raises(binding.a0.ContractError, match="sealed final checkpoint"):
        _build(sealed_inputs)


def test_binding_refuses_validation_seed_drift(sealed_inputs: dict) -> None:
    manifest_path = sealed_inputs["hl_seed_manifest"]
    seeds = [7, 9, 13]
    _write_json(
        manifest_path,
        {
            "schema_version": "train-validation-game-seeds-v1",
            "validation_game_seed_count": 3,
            "validation_game_seed_set_sha256": binding.a0._int64_set_sha(
                np.asarray(seeds, dtype=np.int64)
            ),
            "game_seeds": seeds,
        },
    )
    payload = sealed_inputs["hl_calibration"]
    payload["row_selection"]["seed_manifest_sha256"] = _sha(
        manifest_path
    ).removeprefix("sha256:")
    _write_json(sealed_inputs["hl_calibration_path"], payload)
    with pytest.raises(binding.a0.ContractError, match="game-seed set drift"):
        _build(sealed_inputs)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("global", None),
        ("by_phase", "opening_placement"),
        ("by_legal_count_bucket", "41+"),
    ],
)
def test_binding_rejects_calibration_regression_in_every_binding_slice(
    sealed_inputs: dict, section: str, key: str | None
) -> None:
    scalar = sealed_inputs["scalar_calibration"]
    hl = sealed_inputs["hl_calibration"]
    scalar_slice = scalar[section] if key is None else scalar[section][key]
    hl_slice = hl[section] if key is None else hl[section][key]
    hl_slice["brier"] = scalar_slice["brier"] * 1.10
    _write_json(sealed_inputs["hl_calibration_path"], hl)
    verdict = _build(sealed_inputs)
    assert verdict["gates"]["calibration"] is False
    assert verdict["a0_binding_pass"] is True
    assert verdict["hlgauss_adoption_pass"] is False
    assert verdict["decision"]["status"] == "retain_scalar_for_a1"


def test_binding_requires_one_global_metric_to_improve(sealed_inputs: dict) -> None:
    scalar = sealed_inputs["scalar_calibration"]["global"]
    hl = sealed_inputs["hl_calibration"]["global"]
    hl["brier"] = scalar["brier"]
    hl["value_rmse"] = scalar["value_rmse"]
    _write_json(sealed_inputs["hl_calibration_path"], sealed_inputs["hl_calibration"])
    verdict = _build(sealed_inputs)
    assert verdict["gates"]["calibration"] is False


def test_binding_rejects_more_than_two_percent_policy_drift(sealed_inputs: dict) -> None:
    artifact = sealed_inputs["policy_artifact"]
    stage = artifact["stages"]["final"]
    stage["hlgauss33"]["policy_loss"] = 1.03
    comparison = binding.policy_probe.compare_stage_metrics(
        stage["scalar"], stage["hlgauss33"]
    )
    stage["comparison"] = comparison
    artifact["gates"]["final"] = comparison["pass"]
    artifact["policy_drift_pass"] = all(artifact["gates"].values())
    _write_json(sealed_inputs["policy_drift_path"], artifact)
    verdict = _build(sealed_inputs)
    assert verdict["gates"]["policy_drift"] is False
    assert verdict["a0_binding_pass"] is True
    assert verdict["hlgauss_adoption_pass"] is False
    assert verdict["decision"]["learner_objective"] == "mse"


def test_valid_primary_hl_rejection_completes_without_posthoc_artifacts(
    sealed_inputs: dict,
) -> None:
    result = sealed_inputs["result"]
    result["hl_training_stable"] = False
    result["a0_training_loss_gate_pass"] = False
    result["hl_primary_validation_trace"] = [1.19, 1.53, 1.60]
    _write_json(sealed_inputs["result_path"], result)

    verdict = binding.build_binding_verdict(
        lock_path=sealed_inputs["lock_path"],
        result_path=sealed_inputs["result_path"],
        scalar_calibration_path=None,
        hl_calibration_path=None,
        policy_drift_path=None,
        repo_root=sealed_inputs["repo_root"],
    )

    assert verdict["a0_interpretable"] is True
    assert verdict["a0_stage_complete"] is True
    assert verdict["a0_binding_pass"] is True
    assert verdict["hlgauss_adoption_pass"] is False
    assert verdict["decision"]["status"] == "retain_scalar_for_a1"
    assert verdict["decision"]["learner_objective"] == "mse"
    assert verdict["decision"]["learner_value_readout"] == "scalar"
    assert verdict["calibration_artifacts"] is None
    assert verdict["policy_drift"] is None


def test_scalar_non_reproduction_remains_a_blocked_invalid_experiment(
    sealed_inputs: dict,
) -> None:
    result = sealed_inputs["result"]
    result["scalar_reproduces_historical_failure"] = False
    result["a0_interpretable"] = False
    result["a0_training_loss_gate_pass"] = False
    _write_json(sealed_inputs["result_path"], result)
    with pytest.raises(binding.a0.ContractError, match="scalar control did not reproduce"):
        binding.build_binding_verdict(
            lock_path=sealed_inputs["lock_path"],
            result_path=sealed_inputs["result_path"],
            scalar_calibration_path=None,
            hl_calibration_path=None,
            policy_drift_path=None,
            repo_root=sealed_inputs["repo_root"],
        )


def test_cli_writes_complete_scalar_decision_before_non_adoption_exit(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "binding.json"
    verdict = {
        "schema_version": binding.SCHEMA,
        "a0_binding_pass": True,
        "a0_stage_complete": True,
        "hlgauss_adoption_pass": False,
        "decision": {
            "status": "retain_scalar_for_a1",
            "learner_objective": "mse",
            "learner_value_readout": "scalar",
        },
    }
    monkeypatch.setattr(binding, "build_binding_verdict", lambda **_kwargs: verdict)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "a0_binding_verdict.py",
            "--lock",
            str(tmp_path / "lock.json"),
            "--result",
            str(tmp_path / "result.json"),
            "--repo-root",
            str(tmp_path),
            "--out",
            str(output),
        ],
    )
    with pytest.raises(binding.a0.ContractError, match="typed scalar-retention"):
        binding.main()
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["a0_stage_complete"] is True
    assert written["decision"]["status"] == "retain_scalar_for_a1"


def test_binding_refuses_random_categorical_readout_provenance(
    sealed_inputs: dict,
) -> None:
    payload = sealed_inputs["hl_calibration"]
    payload["readout_provenance"]["categorical_training_verified"] = False
    _write_json(sealed_inputs["hl_calibration_path"], payload)
    with pytest.raises(binding.a0.ContractError, match="not verified"):
        _build(sealed_inputs)


def test_binding_requires_opening_and_41_plus_slices(sealed_inputs: dict) -> None:
    payload = sealed_inputs["hl_calibration"]
    del payload["by_legal_count_bucket"]["41+"]
    _write_json(sealed_inputs["hl_calibration_path"], payload)
    with pytest.raises(binding.a0.ContractError, match=r"by_legal_count_bucket.41\+"):
        _build(sealed_inputs)
