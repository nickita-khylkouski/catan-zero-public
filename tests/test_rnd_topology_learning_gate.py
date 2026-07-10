from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from tools.rnd_topology_learning_gate import GateInputError, main, score_learning_gate


ARMS = {
    "c0": "reference",
    "capacity": "capacity_compute_control",
    "rewired": "geometry_control",
    "candidate": "primary_candidate",
}
HASHES = {
    "topology_mask_registration_artifact_sha256": "1" * 64,
    "training_manifest_sha256": "2" * 64,
    "holdout_manifest_sha256": "3" * 64,
    "training_data_sha256": "4" * 64,
}
COMMON = {"hidden_size": 16, "state_layers": 2, "attention_heads": 2}


def _canonical_sha(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _config() -> dict:
    return {
        "common": copy.deepcopy(COMMON),
        "arms": [
            {
                "arm_id": arm,
                "role": role,
                "adapter_kind": "none" if arm == "c0" else "local_attention_v2",
                "edge_control": (
                    "self_message"
                    if arm == "capacity"
                    else "type_degree_preserving_rewire"
                    if arm == "rewired"
                    else "true_topology"
                ),
                "expected_parameters": 1000 if arm == "c0" else 1200,
            }
            for arm, role in ARMS.items()
        ],
        "learning_gate": {
            "seeds": [1, 2, 3],
            "optimizer_steps": 250,
            "global_batch_size": 4096,
            "sample_presentations_per_arm_seed": 1_024_000,
            "minimum_holdout_games": 3,
            "minimum_topology_sensitive_decisions": 3,
            "minimum_relative_improvement_vs_incumbent": 0.02,
            "minimum_relative_improvement_vs_capacity_control": 0.02,
            "maximum_overall_ce_regression": 0.005,
            **HASHES,
        },
    }


def _resolved_config(config: dict, arm_id: str) -> dict:
    arm = next(item for item in config["arms"] if item["arm_id"] == arm_id)
    return {
        **config["common"],
        "adapter_kind": arm["adapter_kind"],
        "edge_control": arm["edge_control"],
    }


def _run_provenance(config: dict, arm: str, seed: int) -> dict:
    resolved = _resolved_config(config, arm)
    arm_config = next(item for item in config["arms"] if item["arm_id"] == arm)
    return {
        "checkpoint_sha256": hashlib.sha256(f"{arm}/{seed}".encode()).hexdigest(),
        "resolved_config": resolved,
        "resolved_config_sha256": _canonical_sha(resolved),
        "parameter_count": arm_config["expected_parameters"],
        "training_data_sha256": HASHES["training_data_sha256"],
        "optimizer_steps": 250,
        "global_batch_size": 4096,
        "sample_presentations": 1_024_000,
        "training_report_sha256": hashlib.sha256(
            f"report/{arm}/{seed}".encode()
        ).hexdigest(),
        "experiment_config_sha256": _canonical_sha(config),
        "optimizer_sidecar_sha256": hashlib.sha256(
            f"optimizer/{arm}/{seed}".encode()
        ).hexdigest(),
        "train_config_hash": "sha256:" + hashlib.sha256(
            f"config/{arm}/{seed}".encode()
        ).hexdigest()[:16],
    }


def _records(*, candidate_ce: float = 0.7, config: dict | None = None) -> list[dict]:
    config = config or _config()
    ce = {"c0": 1.0, "capacity": 0.95, "rewired": 1.05, "candidate": candidate_ce}
    rows = []
    for arm in ARMS:
        for seed in (1, 2, 3):
            provenance = _run_provenance(config, arm, seed)
            for game in ("g1", "g2", "g3", "g4"):
                for decision, sensitive in (("d1", True), ("d2", False)):
                    rows.append(
                        {
                            "arm": arm,
                            "training_seed": seed,
                            "game_id": game,
                            "decision_id": decision,
                            "policy_ce": ce[arm],
                            "forced": False,
                            "topology_sensitive": sensitive,
                            "evaluation_split": "holdout",
                            "is_training_game": False,
                            "topology_mask_registration_artifact_sha256": HASHES[
                                "topology_mask_registration_artifact_sha256"
                            ],
                            "training_manifest_sha256": HASHES[
                                "training_manifest_sha256"
                            ],
                            "holdout_manifest_sha256": HASHES[
                                "holdout_manifest_sha256"
                            ],
                            "experiment_config_sha256": _canonical_sha(config),
                            "run_provenance": copy.deepcopy(provenance),
                        }
                    )
    return rows


def test_passes_with_config_bound_crossed_evidence_deterministically() -> None:
    first = score_learning_gate(
        _records(), _config(), bootstrap_samples=300, bootstrap_seed=9
    )
    second = score_learning_gate(
        _records(), _config(), bootstrap_samples=300, bootstrap_seed=9
    )

    assert first == second
    assert first["status"] == "pass"
    assert all(first["checks"].values())
    assert first["comparisons"]["vs_incumbent"][
        "relative_improvement"
    ] == pytest.approx(0.3)
    assert first["metric_contract"]["ci"].startswith("paired crossed")
    assert first["evidence"]["holdout_games"] == 4
    assert len(first["run_provenance"]) == 12


def test_target_probability_is_converted_and_weak_candidate_fails() -> None:
    rows = _records(candidate_ce=1.01)
    for row in rows:
        row["target_probability"] = math.exp(-row.pop("policy_ce"))
    report = score_learning_gate(rows, _config(), bootstrap_samples=200)

    assert report["status"] == "fail"
    assert "primary_vs_incumbent_point_threshold" in report["failed_checks"]
    assert "primary_vs_capacity_point_threshold" in report["failed_checks"]
    assert "overall_ce_regression_upper_ci_within_limit" in report["failed_checks"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda rows: rows.append(copy.deepcopy(rows[0])),
            "duplicate decision overlap",
        ),
        (
            lambda rows: rows.__setitem__(0, {**rows[0], "is_training_game": True}),
            "leakage",
        ),
        (
            lambda rows: rows.__setitem__(0, {**rows[0], "topology_sensitive": None}),
            "must be booleans",
        ),
        (
            lambda rows: rows.__setitem__(
                0, {**rows[0], "topology_mask_registration_artifact_sha256": "f" * 64}
            ),
            "does not match experiment config",
        ),
    ],
)
def test_fails_closed_on_invalid_or_unbound_evidence(mutation, match: str) -> None:
    rows = _records()
    mutation(rows)
    with pytest.raises(GateInputError, match=match):
        score_learning_gate(rows, _config(), bootstrap_samples=100)


def test_enforces_common_support_across_arms_and_training_seeds() -> None:
    rows = [
        row
        for row in _records()
        if not (
            row["arm"] == "candidate"
            and row["training_seed"] == 3
            and row["game_id"] == "g4"
        )
    ]
    with pytest.raises(GateInputError, match="support differs"):
        score_learning_gate(rows, _config(), bootstrap_samples=100)

    rows = [row for row in _records() if row["training_seed"] != 3]
    with pytest.raises(GateInputError, match="support differs"):
        score_learning_gate(rows, _config(), bootstrap_samples=100)


def test_enforces_minimum_independent_game_and_sensitive_decision_support() -> None:
    rows = [row for row in _records() if row["game_id"] in {"g1", "g2"}]
    with pytest.raises(GateInputError, match="insufficient holdout games"):
        score_learning_gate(rows, _config(), bootstrap_samples=100)

    rows = _records()
    for row in rows:
        if row["game_id"] in {"g2", "g3", "g4"} and row["decision_id"] == "d1":
            row["topology_sensitive"] = False
    with pytest.raises(
        GateInputError, match="insufficient topology-sensitive decisions"
    ):
        score_learning_gate(rows, _config(), bootstrap_samples=100)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("parameter_count", 999, "parameter_count"),
        ("optimizer_steps", 251, "optimizer_steps"),
        ("training_data_sha256", "f" * 64, "training_data_sha256"),
    ],
)
def test_run_provenance_must_match_matrix(field: str, value, match: str) -> None:
    rows = _records()
    rows[0]["run_provenance"][field] = value
    with pytest.raises(GateInputError, match=match):
        score_learning_gate(rows, _config(), bootstrap_samples=100)


def test_resolved_config_hash_and_values_are_bound() -> None:
    rows = _records()
    rows[0]["run_provenance"]["resolved_config"]["hidden_size"] = 99
    rows[0]["run_provenance"]["resolved_config_sha256"] = _canonical_sha(
        rows[0]["run_provenance"]["resolved_config"]
    )
    with pytest.raises(GateInputError, match="does not match experiment matrix"):
        score_learning_gate(rows, _config(), bootstrap_samples=100)

    rows = _records()
    rows[0]["run_provenance"]["resolved_config_sha256"] = "f" * 64
    with pytest.raises(GateInputError, match="does not match resolved_config"):
        score_learning_gate(rows, _config(), bootstrap_samples=100)


def test_checkpoint_cannot_be_reused_across_runs() -> None:
    rows = _records()
    first = rows[0]["run_provenance"]["checkpoint_sha256"]
    for row in rows:
        if row["arm"] == "capacity" and row["training_seed"] == 1:
            row["run_provenance"]["checkpoint_sha256"] = first
    with pytest.raises(GateInputError, match="checkpoint reused"):
        score_learning_gate(rows, _config(), bootstrap_samples=100)


def test_crossed_bootstrap_preserves_common_game_effect_uncertainty() -> None:
    rows = _records()
    candidate_by_game = {"g1": 0.7, "g2": 0.7, "g3": 0.7, "g4": 1.7}
    for row in rows:
        if row["arm"] == "candidate" and row["topology_sensitive"]:
            row["policy_ce"] = candidate_by_game[row["game_id"]]
    report = score_learning_gate(
        rows, _config(), bootstrap_samples=4000, bootstrap_seed=17
    )

    comparison = report["comparisons"]["vs_incumbent"]
    assert comparison["relative_improvement"] == pytest.approx(0.05)
    # The adverse g4 effect is common to all model seeds and must not be averaged
    # away by independently resampling a different game set for every seed.
    assert comparison["difference_ci95"][1] > 0
    assert report["status"] == "fail"


def test_overall_regression_uses_upper_confidence_bound() -> None:
    rows = _records()
    for row in rows:
        if row["arm"] == "candidate" and not row["topology_sensitive"]:
            row["policy_ce"] = 2.1 if row["game_id"] == "g4" else 1.0
    report = score_learning_gate(
        rows, _config(), bootstrap_samples=2000, bootstrap_seed=5
    )

    overall = report["overall_vs_incumbent"]
    assert overall["relative_regression"] < 0.005
    assert overall["relative_regression_ci95"][1] > 0.005
    assert not report["checks"]["overall_ce_regression_upper_ci_within_limit"]


def test_zero_reference_game_is_invalid_not_nonfinite_output() -> None:
    rows = _records()
    for row in rows:
        if row["arm"] == "c0" and row["game_id"] == "g1" and row["topology_sensitive"]:
            row["policy_ce"] = 0.0
    with pytest.raises(GateInputError, match="positive for every game cluster"):
        score_learning_gate(rows, _config(), bootstrap_samples=100)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config["learning_gate"].__setitem__(
            "minimum_relative_improvement_vs_incumbent", float("nan")
        ),
        lambda config: config["learning_gate"].__setitem__("seeds", [1, "2"]),
        lambda config: config.__setitem__("arms", None),
        lambda config: config["learning_gate"].__setitem__(
            "topology_mask_registration_artifact_sha256", "not-a-hash"
        ),
    ],
)
def test_cli_normalizes_malformed_config_to_invalid_json(
    tmp_path: Path, capsys, mutation
) -> None:
    config_value = _config()
    mutation(config_value)
    records = tmp_path / "records.json"
    config = tmp_path / "config.json"
    records.write_text(json.dumps(_records()), encoding="utf-8")
    config.write_text(json.dumps(config_value), encoding="utf-8")

    code = main(
        [
            "--records",
            str(records),
            "--experiment-config",
            str(config),
            "--bootstrap-samples",
            "100",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert report["status"] == "invalid"


def test_cli_emits_json_and_nonzero_for_failed_valid_evidence(
    tmp_path: Path, capsys
) -> None:
    records = tmp_path / "records.json"
    config = tmp_path / "config.json"
    records.write_text(json.dumps(_records(candidate_ce=1.01)), encoding="utf-8")
    config.write_text(json.dumps(_config()), encoding="utf-8")
    config_file_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    rows = json.loads(records.read_text())
    for row in rows:
        row["experiment_config_sha256"] = config_file_sha
        row["run_provenance"]["experiment_config_sha256"] = config_file_sha
    records.write_text(json.dumps(rows), encoding="utf-8")

    code = main(
        [
            "--records",
            str(records),
            "--experiment-config",
            str(config),
            "--bootstrap-samples",
            "100",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert report["status"] == "fail"
