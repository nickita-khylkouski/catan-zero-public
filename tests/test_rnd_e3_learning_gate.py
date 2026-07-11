from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from tools.rnd_e3_learning_gate import GateInputError, main, score_learning_gate


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = ROOT / "configs/rnd/e3_a1_screen_20260710/experiment.registered.json"
GATE_PATH = ROOT / "configs/rnd/e3_a1_screen_20260710/learning_gate.v1.json"


def _canonical_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def _contracts() -> tuple[dict, dict, str]:
    experiment = json.loads(EXPERIMENT_PATH.read_text())
    gate = json.loads(GATE_PATH.read_text())
    gate["minimum_holdout_games"] = 3
    gate["minimum_nonforced_decisions"] = 6
    gate["bootstrap_samples"] = 200
    semantic = dict(gate)
    semantic.pop("config_sha256")
    gate["config_sha256"] = _canonical_sha(semantic)
    return experiment, gate, hashlib.sha256(EXPERIMENT_PATH.read_bytes()).hexdigest()


def _resolved(arm: dict) -> dict:
    fields = {
        "hidden_size": 384,
        "graph_layers": 9,
        "attention_heads": 6,
        "entity_state_trunk": "rrt",
        "relational_block_pattern": "RRTRRTRRT",
        "relational_ff_size": 1024,
        "relational_bases": 4,
        "relational_action_cross_layers": 1,
        "latent_deliberation_slots": 8,
        "latent_deliberation_steps": arm["latent_deliberation_steps"],
        "max_steps": 250,
        "batch_size": 1024,
        "grad_accum_steps": 4,
        "mask_hidden_info": True,
    }
    return {"schema_version": 6, "pipeline": "train", "fields": fields}


def _rows(experiment: dict, file_sha: str, ce_fn=None, game_sizes=(2, 2, 2)) -> list[dict]:
    registration = experiment["registration"]
    arms = {item["arm_id"]: item for item in experiment["arms"]}
    default_ce = {
        "rrt-k0": 0.80,
        "think-rrt-k1": 1.00,
        "think-rrt-k2": 0.96,
        "think-rrt-k4": 0.97,
        "think-rrt-k8": 0.95,
    }
    rows = []
    for arm, arm_config in arms.items():
        for seed in (11, 29, 47):
            resolved = _resolved(arm_config)
            provenance = {
                "checkpoint_sha256": hashlib.sha256(f"trained:{arm}:{seed}".encode()).hexdigest(),
                "training_report_sha256": hashlib.sha256(f"report:{arm}:{seed}".encode()).hexdigest(),
                "admission_manifest_sha256": hashlib.sha256(f"admit:{arm}:{seed}".encode()).hexdigest(),
                "initial_checkpoint_sha256": registration[
                    "initial_checkpoint_sha256_by_arm_seed"
                ][f"{arm}@{seed}"],
                "resolved_train_config": resolved,
                "resolved_train_config_sha256": _canonical_sha(resolved),
                "graph_history_features": True,
                "parameter_count": arm_config["expected_parameters"],
                "optimizer_steps": 250,
                "global_batch_size": 4096,
                "sample_presentations": 1_024_000,
            }
            for game_index, decision_count in enumerate(game_sizes):
                game = f"g{game_index}"
                for decision_index in range(decision_count):
                    ce = (
                        ce_fn(arm, game_index, decision_index)
                        if ce_fn is not None
                        else default_ce[arm]
                    )
                    rows.append(
                        {
                            "arm_id": arm,
                            "training_seed": seed,
                            "game_id": game,
                            "decision_id": f"d{decision_index}",
                            "forced": False,
                            "soft_target_policy_ce": ce,
                            "evaluation_split": "holdout",
                            "is_training_game": False,
                            "experiment_config_sha256": file_sha,
                            "corpus_fingerprint": registration["corpus_fingerprint"],
                            "training_manifest_sha256": registration["training_manifest_sha256"],
                            "validation_manifest_sha256": registration["validation_manifest_sha256"],
                            "run_provenance": provenance,
                        }
                    )
    return rows


def test_primary_roles_thresholds_and_k0_exclusion() -> None:
    experiment, gate, file_sha = _contracts()
    report = score_learning_gate(
        _rows(experiment, file_sha),
        experiment,
        gate,
        experiment_config_sha256=file_sha,
        bootstrap_samples=200,
    )
    assert report["status"] == "pass"
    assert report["promotion_eligible_passed_arms"] == ["think-rrt-k2", "think-rrt-k4"]
    assert report["comparisons"]["think-rrt-k8"]["comparison_role"] == "secondary"
    assert report["descriptive_compute_control"]["arm"] == "rrt-k0"
    assert report["descriptive_compute_control"]["promotion_eligible"] is False
    assert report["support"]["runs"] == 15


def test_nonforced_safety_is_decision_micro_not_game_macro() -> None:
    experiment, gate, file_sha = _contracts()

    def ce(arm: str, game: int, _decision: int) -> float:
        if arm == "think-rrt-k1":
            return 1.0
        if arm == "think-rrt-k2":
            return 0.0 if game == 0 else 1.02
        if arm == "think-rrt-k4":
            return 0.96
        if arm == "think-rrt-k8":
            return 0.95
        return 0.8

    report = score_learning_gate(
        _rows(experiment, file_sha, ce_fn=ce, game_sizes=(1, 100, 100)),
        experiment,
        gate,
        experiment_config_sha256=file_sha,
        bootstrap_samples=200,
    )
    k2 = report["comparisons"]["think-rrt-k2"]
    # Equal game weighting sees a large improvement; decision weighting exposes
    # that the two large games regress beyond the registered 0.5% safety bound.
    assert k2["primary_nonforced_game_macro"]["relative_improvement"] > 0.30
    assert k2["nonforced_decision_micro_safety"]["relative_regression"] > 0.005
    assert k2["overall_safety_pass"] is False
    assert k2["passed"] is False


def test_rejects_missing_run_or_decision_support() -> None:
    experiment, gate, file_sha = _contracts()
    rows = _rows(experiment, file_sha)
    missing_run = [
        row for row in rows
        if not (row["arm_id"] == "think-rrt-k8" and row["training_seed"] == 47)
    ]
    with pytest.raises(GateInputError, match="exactly all 15"):
        score_learning_gate(
            missing_run, experiment, gate,
            experiment_config_sha256=file_sha, bootstrap_samples=200,
        )
    missing_decision = copy.deepcopy(rows)
    missing_decision.pop()
    with pytest.raises(GateInputError, match="support differs"):
        score_learning_gate(
            missing_decision, experiment, gate,
            experiment_config_sha256=file_sha, bootstrap_samples=200,
        )


def test_rejects_provenance_or_gate_binding_drift() -> None:
    experiment, gate, file_sha = _contracts()
    rows = _rows(experiment, file_sha)
    rows[0]["run_provenance"] = dict(rows[0]["run_provenance"])
    rows[0]["run_provenance"]["optimizer_steps"] = 249
    with pytest.raises(GateInputError, match="optimizer_steps"):
        score_learning_gate(
            rows, experiment, gate,
            experiment_config_sha256=file_sha, bootstrap_samples=200,
        )
    bad_gate = copy.deepcopy(gate)
    bad_gate["experiment_file_sha256"] = "f" * 64
    semantic = dict(bad_gate)
    semantic.pop("config_sha256")
    bad_gate["config_sha256"] = _canonical_sha(semantic)
    with pytest.raises(GateInputError, match="file SHA"):
        score_learning_gate(
            _rows(experiment, file_sha), experiment, bad_gate,
            experiment_config_sha256=file_sha, bootstrap_samples=200,
        )
    with pytest.raises(GateInputError, match="bootstrap settings"):
        score_learning_gate(
            _rows(experiment, file_sha), experiment, gate,
            experiment_config_sha256=file_sha, bootstrap_samples=201,
        )


def test_checked_in_gate_contract_is_self_hashed_and_pre_outcome_locked() -> None:
    gate = json.loads(GATE_PATH.read_text())
    declared = gate.pop("config_sha256")
    assert declared == _canonical_sha(gate)
    assert gate["experiment_file_sha256"] == hashlib.sha256(
        EXPERIMENT_PATH.read_bytes()
    ).hexdigest()
    assert gate["bootstrap_samples"] == 10_000
    assert gate["bootstrap_seed"] == 20260710


def test_cli_hashes_exact_experiment_bytes_and_streams_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, gate, file_sha = _contracts()
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        "\n".join(json.dumps(row) for row in _rows(experiment, file_sha)) + "\n"
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rnd_e3_learning_gate.py",
            "--evidence", str(evidence),
            "--experiment", str(EXPERIMENT_PATH),
            "--gate-contract", str(gate_path),
            "--bootstrap-samples", "200",
            "--output", str(output),
        ],
    )
    assert main() == 0
    assert json.loads(output.read_text())["status"] == "pass"

    alternate = tmp_path / "experiment-whitespace.json"
    alternate.write_text(json.dumps(experiment, indent=4) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rnd_e3_learning_gate.py",
            "--evidence", str(evidence),
            "--experiment", str(alternate),
            "--gate-contract", str(gate_path),
            "--bootstrap-samples", "200",
        ],
    )
    with pytest.raises(SystemExit, match="file SHA"):
        main()
