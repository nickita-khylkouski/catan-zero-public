from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from catan_zero.rl.pipeline_configs import EvalConfig
from tools import a1_evaluation_decomposition as decomposition


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _games(seed: int = 100) -> list[dict[str, object]]:
    return [
        {
            "game_seed": seed,
            "orientation": "candidate_red",
            "candidate_won": True,
            "terminated": True,
            "truncated": False,
        },
        {
            "game_seed": seed,
            "orientation": "candidate_blue",
            "candidate_won": False,
            "terminated": True,
            "truncated": False,
        },
    ]


def _report(
    path: Path,
    *,
    candidate: Path,
    baseline: Path,
    contract: str,
    candidate_raw: int | None,
    baseline_raw: int | None,
    verdict: str = "continue",
    superiority_verdict: str = "H1",
    seed: int = 100,
    candidate_win_rate: float = 0.5,
) -> Path:
    config = EvalConfig(
        mode="cross_net",
        candidate=str(candidate),
        baseline=str(baseline),
        public_observation=True,
        coherent_public_belief_search=True,
        native_mcts_hot_loop=True,
        n_full=128,
        candidate_n_full=128,
        baseline_n_full=128,
        candidate_raw_policy_above_width=candidate_raw,
        baseline_raw_policy_above_width=baseline_raw,
        c_scale=0.03,
        candidate_c_scale=0.03,
        baseline_c_scale=0.03,
        candidate_value_readout="scalar",
        baseline_value_readout="scalar",
        candidate_value_squash="tanh",
        baseline_value_squash="tanh",
        candidate_gameplay_policy_aggregation="mean_improved_policy",
        baseline_gameplay_policy_aggregation="mean_improved_policy",
        candidate_rescale_noise_floor_c=0.0,
        baseline_rescale_noise_floor_c=0.0,
        candidate_sigma_eval=0.79,
        baseline_sigma_eval=0.79,
    )
    games = _games(seed)
    payload = {
        "candidate_checkpoint": str(candidate),
        "candidate_checkpoint_sha256": _sha256(candidate),
        "baseline_checkpoint": str(baseline),
        "baseline_checkpoint_sha256": _sha256(baseline),
        "comparison_contract": contract,
        "planned_engine_identity": {"repo_commit": "a" * 40},
        "engine_identity": {"repo_commit": "a" * 40},
        "typed_config": config.canonical_payload(),
        "config_hash": config.config_hash(),
        "full_config_hash": config.full_config_hash(),
        "games": games,
        "errors": [],
        "games_truncated": 0,
        "games_played": 2,
        "games_with_winner": 2,
        "complete_pairs": 1,
        "candidate_win_rate": candidate_win_rate,
        "verdict": verdict,
        "superiority_verdict": superiority_verdict,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _panels(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate = tmp_path / "candidate.pt"
    parent = tmp_path / "parent.pt"
    candidate.write_bytes(b"candidate")
    parent.write_bytes(b"parent")
    raw = _report(
        tmp_path / "raw.json",
        candidate=candidate,
        baseline=parent,
        contract=decomposition.RAW_CONTRACT,
        candidate_raw=0,
        baseline_raw=0,
    )
    searched = _report(
        tmp_path / "searched.json",
        candidate=candidate,
        baseline=parent,
        contract=decomposition.SEARCHED_CONTRACT,
        candidate_raw=None,
        baseline_raw=None,
    )
    uplift = _report(
        tmp_path / "uplift.json",
        candidate=candidate,
        baseline=candidate,
        contract=decomposition.UPLIFT_CONTRACT,
        candidate_raw=None,
        baseline_raw=0,
    )
    return raw, searched, uplift


def test_build_decomposition_authenticates_three_matched_panels(
    tmp_path: Path,
) -> None:
    raw, searched, uplift = _panels(tmp_path)
    receipt = decomposition.build_decomposition(
        raw_cross=raw,
        searched_cross=searched,
        candidate_search_vs_raw=uplift,
    )

    assert receipt["schema_version"] == decomposition.SCHEMA
    assert receipt["cohort"]["complete_pairs"] == 1
    assert receipt["diagnosis"]["searched_checkpoint_superiority_proven"] is True
    assert receipt["diagnosis"]["raw_network_nonregression_resolved"] is False
    assert receipt["diagnosis"]["candidate_search_uplift_resolved"] is False
    assert receipt["ready_for_promotion_adjudication"] is False
    assert receipt["receipt_sha256"].startswith("sha256:")


def test_decomposition_is_ready_only_after_raw_and_uplift_resolve(
    tmp_path: Path,
) -> None:
    raw, searched, uplift = _panels(tmp_path)
    for path in (raw, uplift):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["verdict"] = "H1"
        payload["candidate_win_rate"] = 0.6
        path.write_text(json.dumps(payload), encoding="utf-8")

    receipt = decomposition.build_decomposition(
        raw_cross=raw,
        searched_cross=searched,
        candidate_search_vs_raw=uplift,
    )

    assert receipt["diagnosis"]["search_compensation_risk"] is False
    assert receipt["ready_for_promotion_adjudication"] is True


def test_decomposition_surfaces_search_compensating_for_worse_raw_network(
    tmp_path: Path,
) -> None:
    raw, searched, uplift = _panels(tmp_path)
    payload = json.loads(raw.read_text(encoding="utf-8"))
    payload["candidate_win_rate"] = 0.4
    payload["verdict"] = "H0"
    raw.write_text(json.dumps(payload), encoding="utf-8")

    receipt = decomposition.build_decomposition(
        raw_cross=raw,
        searched_cross=searched,
        candidate_search_vs_raw=uplift,
    )

    assert receipt["diagnosis"]["raw_network_material_regression_detected"] is True
    assert receipt["diagnosis"]["search_compensation_risk"] is True
    assert receipt["ready_for_promotion_adjudication"] is False


def test_decomposition_refuses_different_seed_cohorts(tmp_path: Path) -> None:
    raw, searched, uplift = _panels(tmp_path)
    payload = json.loads(uplift.read_text(encoding="utf-8"))
    payload["games"] = _games(101)
    uplift.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(decomposition.DecompositionError, match="same paired seed"):
        decomposition.build_decomposition(
            raw_cross=raw,
            searched_cross=searched,
            candidate_search_vs_raw=uplift,
        )


def test_decomposition_refuses_checkpoint_byte_drift(tmp_path: Path) -> None:
    raw, searched, uplift = _panels(tmp_path)
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"changed")

    with pytest.raises(decomposition.DecompositionError, match="bytes drifted"):
        decomposition.build_decomposition(
            raw_cross=raw,
            searched_cross=searched,
            candidate_search_vs_raw=uplift,
        )


def test_decomposition_authenticates_boundary_value_particle_operator(
    tmp_path: Path,
) -> None:
    raw, searched, uplift = _panels(tmp_path)
    for path in (raw, searched, uplift):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path in (raw, uplift):
            payload["verdict"] = "H1"
            payload["candidate_win_rate"] = 0.6
        config = EvalConfig(
            **{
                **payload["typed_config"]["fields"],
                "boundary_value_particles": 4,
            }
        )
        payload["typed_config"] = config.canonical_payload()
        payload["config_hash"] = config.config_hash()
        payload["full_config_hash"] = config.full_config_hash()
        path.write_text(json.dumps(payload), encoding="utf-8")

    receipt = decomposition.build_decomposition(
        raw_cross=raw,
        searched_cross=searched,
        candidate_search_vs_raw=uplift,
    )

    assert receipt["ready_for_promotion_adjudication"] is True


def test_decomposition_rejects_boundary_value_particle_operator_drift(
    tmp_path: Path,
) -> None:
    raw, searched, uplift = _panels(tmp_path)
    payload = json.loads(uplift.read_text(encoding="utf-8"))
    config = EvalConfig(
        **{
            **payload["typed_config"]["fields"],
            "boundary_value_particles": 4,
        }
    )
    payload["typed_config"] = config.canonical_payload()
    payload["config_hash"] = config.config_hash()
    payload["full_config_hash"] = config.full_config_hash()
    uplift.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        decomposition.DecompositionError, match="changed the candidate search operator"
    ):
        decomposition.build_decomposition(
            raw_cross=raw,
            searched_cross=searched,
            candidate_search_vs_raw=uplift,
        )
