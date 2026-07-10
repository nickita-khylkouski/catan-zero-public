from __future__ import annotations

import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import search_teacher_adjudicator as adjudicator
from tools.fixed_root_search_stability import (
    aggregate_report_slices,
    summarize_cross_seed_runs,
)


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return path


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": adjudicator._sha256(path)}


def _operator(**overrides: Any) -> dict[str, Any]:
    result = {
        "max_depth": 80,
        "c_visit": 50.0,
        "c_scale": 0.03,
        "prior_temperature": 1.0,
        "n_full": 64,
        "n_fast": 16,
        "p_full": 0.25,
        "n_full_wide": None,
        "n_full_wide_threshold": None,
        "wide_roots_always_full": False,
        "raw_policy_above_width": None,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
        "wide_candidates_threshold": 24,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": True,
        "exact_budget_sh": True,
        "exact_budget_sh_min_n": 48,
        "belief_chance_spectra": False,
        "rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.98,
    }
    result.update(overrides)
    return result


def _evaluator() -> dict[str, Any]:
    return {
        "value_scale": 1.0,
        "prior_temperature": 1.0,
        "context_fill": 0.0,
        "cache_size": 0,
        "value_squash": "tanh",
        "value_readout": "scalar",
        "public_observation": True,
        "rust_featurize": False,
        "emit_uncertainty": False,
    }


def _pent(
    counts: tuple[int, int, int], *, elo0: float = -10.0, elo1: float = 15.0
) -> dict[str, Any]:
    ll, split, ww = counts
    pairs = sum(counts)
    observed_mean = (0.5 * split + ww) / pairs
    regularized = [ll + 1.0, split, ww + 1.0]
    total = sum(regularized)
    mean = (
        sum(count * value for count, value in zip(regularized, (0.0, 0.5, 1.0))) / total
    )
    variance = (
        sum(
            count * (value - mean) ** 2
            for count, value in zip(regularized, (0.0, 0.5, 1.0))
        )
        / total
    )
    s0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    s1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))
    llr = total / (2.0 * variance) * (s1 - s0) * (2.0 * mean - s0 - s1)
    alpha = beta = 0.05
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    decision = "H1" if llr >= upper else "H0" if llr <= lower else "continue"
    return {
        "model": "pentanomial",
        "elo0": elo0,
        "elo1": elo1,
        "s0": s0,
        "s1": s1,
        "alpha": alpha,
        "beta": beta,
        "lower_bound": lower,
        "upper_bound": upper,
        "pairs": pairs,
        "ll_pairs": ll,
        "split_pairs": split,
        "ww_pairs": ww,
        "mean_pair_score": observed_mean,
        "llr": llr,
        "decision": decision,
    }


def _calibration(path: Path, checkpoint: Path, *, sigma: float = 0.98) -> Path:
    return _write(
        path,
        {
            "schema_version": adjudicator.CALIBRATION_SCHEMA,
            "value_readout": "scalar",
            "checkpoint": str(checkpoint),
            "by_phase": {"opening_placement": {"value_rmse": sigma}},
        },
    )


def _s1_arm(
    path: Path,
    checkpoint: Path,
    name: str,
    *,
    counts: tuple[int, int, int] = (20, 45, 20),
    threshold: int | None = 20,
    pairs: int = 85,
) -> Path:
    scale, d1 = adjudicator.S1_ARMS[name]
    baseline = {
        "colors": ["RED", "BLUE"],
        "n_full": 64,
        "n_fast": 64,
        "p_full": 1.0,
        "max_depth": 80,
        "temperature": 0.0,
        "c_visit": 50.0,
        "c_scale": 0.03,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": True,
        "max_root_candidates": 16,
        "max_root_candidates_wide": 54,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": threshold,
    }
    candidate = {
        **baseline,
        "c_scale": scale,
    }
    if d1:
        candidate.update({"rescale_noise_floor_c": 1.0, "sigma_eval": 0.98})
    pent = _pent(counts)
    assert pent["pairs"] == pairs
    games = []
    outcomes = [0] * counts[0] + [1] * counts[1] + [2] * counts[2]
    for pair_id, wins in enumerate(outcomes):
        pair_outcomes = (False, False) if wins == 0 else (True, False) if wins == 1 else (True, True)
        for orientation, won in zip(("candidate_red", "candidate_blue"), pair_outcomes):
            games.append(
                {
                    "pair_id": pair_id,
                    "game_seed": 10_000 + pair_id,
                    "orientation": orientation,
                    "candidate_won": won,
                    "terminated": True,
                    "truncated": False,
                }
            )
    overrides = {"c_visit": 50.0, "c_scale": scale}
    if d1:
        overrides.update({"rescale_noise_floor_c": 1.0, "sigma_eval": 0.98})
    return _write(
        path,
        {
            "arm": name,
            "checkpoint": str(checkpoint),
            "masked": True,
            "max_depth": 80,
            "max_decisions": 600,
            "correct_rust_chance_spectra": True,
            "lazy_interior_chance": True,
            "prior_temperature": 1.0,
            "value_scale": 1.0,
            "value_squash": "tanh",
            "value_readout": "scalar",
            "d1_c": 1.0,
            "d1_sigma_eval": 0.98,
            "symmetry_averaged_eval": True,
            "seed_block_base": 10_000,
            "arm_config_overrides": overrides,
            "arm_mcts_cls_key": "stock",
            "baseline_search_config": baseline,
            "candidate_search_config": candidate,
            "errors": [],
            "games_truncated": 0,
            "pairs_requested": pairs,
            "complete_pairs": pairs,
            "games_played": 2 * pairs,
            "pentanomial_sprt": pent,
            "games": games,
        },
    )


def _manifest(
    path: Path,
    *,
    stage: str,
    checkpoint: Path,
    base: dict[str, Any],
    candidate: dict[str, Any] | None,
    predecessors: list[dict[str, str]],
    evidence: dict[str, Any],
) -> Path:
    return _write(
        path,
        {
            "schema_version": adjudicator.MANIFEST_SCHEMA,
            "stage": stage,
            "checkpoint": _ref(checkpoint),
            "base_search_operator": base,
            "candidate_search_operator": candidate,
            "teacher_evaluator": _evaluator(),
            "predecessors": predecessors,
            "evidence": evidence,
        },
    )


def _s1_manifest(
    tmp_path: Path,
    *,
    h1_arm: str | None = None,
    old_threshold_arm: str | None = None,
    sigma: float = 0.98,
) -> tuple[Path, Path]:
    checkpoint = _write(tmp_path / "champion.pt", b"checkpoint")
    calibration = _calibration(tmp_path / "calibration.json", checkpoint, sigma=sigma)
    refs = []
    for name in adjudicator.S1_ARMS:
        counts = (5, 20, 60) if name == h1_arm else (20, 45, 20)
        threshold = None if name == old_threshold_arm else 20
        refs.append(
            _ref(
                _s1_arm(
                    tmp_path / f"{name}.json",
                    checkpoint,
                    name,
                    counts=counts,
                    threshold=threshold,
                )
            )
        )
    manifest = _manifest(
        tmp_path / "s1.manifest.json",
        stage="s1",
        checkpoint=checkpoint,
        base=_operator(),
        candidate=None,
        predecessors=[],
        evidence={"arms": refs, "sigma_eval": _ref(calibration)},
    )
    return manifest, checkpoint


def _source_backed_decision(
    path: Path,
    *,
    stage: str,
    decision: str,
    selected: dict[str, Any],
) -> Path:
    source = _write(path.with_suffix(".source.json"), {"source": stage})
    payload = {
        "schema_version": adjudicator.DECISION_SCHEMA,
        "stage": stage,
        "passed": True,
        "decision": decision,
        "source_artifacts": [_ref(source)],
        "selected_fields": selected,
        "selected_fields_sha256": adjudicator._digest_value(selected),
    }
    return _write(path, payload)


def _h2h(
    path: Path,
    checkpoint: Path,
    *,
    base_n: int,
    candidate_n: int,
    counts: tuple[int, int, int],
    candidate_wide: int | None = None,
    candidate_wide_threshold: int | None = None,
    baseline_wide: int | None = None,
    overhead: float = 1.1,
) -> Path:
    pairs = sum(counts)
    fields = {
        "mode": "cross_net",
        "candidate": str(checkpoint),
        "baseline": str(checkpoint),
        "public_observation": True,
        "belief_chance_spectra": False,
        "pairs": pairs,
        "base_seed": 20_000,
        "n_full": base_n,
        "candidate_n_full": candidate_n,
        "baseline_n_full": base_n,
        "n_full_wide": baseline_wide,
        "candidate_n_full_wide": candidate_wide,
        "baseline_n_full_wide": baseline_wide,
        "n_full_wide_threshold": None,
        "candidate_n_full_wide_threshold": candidate_wide_threshold,
        "baseline_n_full_wide_threshold": None,
        "raw_policy_above_width": None,
        "max_depth": 80,
        "max_decisions": 600,
        "c_visit": 50.0,
        "c_scale": 0.03,
        "rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.98,
        "max_root_candidates": 16,
        "max_root_candidates_wide": 54,
        "wide_candidates_threshold": 24,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": True,
        "prior_temperature": 1.0,
        "value_scale": 1.0,
        "value_squash": "tanh",
        "value_readout": "scalar",
        "candidate_value_readout": "scalar",
        "baseline_value_readout": "scalar",
        "elo0": -10.0,
        "elo1": 15.0,
    }
    typed = {"pipeline": "eval", "schema_version": 4, "fields": fields}
    full_hash = adjudicator._digest_value(typed)
    games = []
    outcomes = [0] * counts[0] + [1] * counts[1] + [2] * counts[2]
    for pair_id, wins in enumerate(outcomes):
        pair_outcomes = (False, False) if wins == 0 else (True, False) if wins == 1 else (True, True)
        for orientation, won in zip(("candidate_red", "candidate_blue"), pair_outcomes):
            games.append(
                {
                    "pair_id": pair_id,
                    "game_seed": 20_000 + pair_id,
                    "orientation": orientation,
                    "candidate_won": won,
                    "terminated": True,
                    "truncated": False,
                }
            )
    return _write(
        path,
        {
            "candidate_checkpoint": str(checkpoint),
            "baseline_checkpoint": str(checkpoint),
            "public_observation": True,
            "symmetry_averaged_eval": True,
            "symmetry_averaged_eval_threshold": 20,
            "c_visit": 50.0,
            "c_scale": 0.03,
            "rescale_noise_floor_c": 0.0,
            "sigma_eval": 0.98,
            "candidate_n_full": candidate_n,
            "baseline_n_full": base_n,
            "candidate_n_full_wide": candidate_wide,
            "candidate_n_full_wide_threshold": candidate_wide_threshold,
            "baseline_n_full_wide": baseline_wide,
            "baseline_n_full_wide_threshold": None,
            "errors": [],
            "games_truncated": 0,
            "pairs_requested": pairs,
            "complete_pairs": pairs,
            "games_played": 2 * pairs,
            "pentanomial_sprt": _pent(counts),
            "typed_config": typed,
            "full_config_hash": full_hash,
            "config_hash": "sha256:" + full_hash.removeprefix("sha256:")[:16],
            "games": games,
            "search_telemetry": {
                "candidate_over_baseline_elapsed_ratio": overhead,
                "by_role": {
                    "candidate": {"search_elapsed_sec": overhead * 10.0},
                    "baseline": {"search_elapsed_sec": 10.0},
                },
            },
        },
    )


def _fixed_root(
    path: Path,
    checkpoint: Path,
    *,
    base: dict[str, Any],
    candidate: dict[str, Any],
    stage: str,
    baseline_role: str,
    candidate_role: str,
    cost: float = 1.5,
    js_reduction: float = 0.0,
    top1_delta: float = 0.0,
) -> Path:
    config_a = _write(path.parent / f"{path.stem}.a.json", {"config": "a"})
    config_b = _write(path.parent / f"{path.stem}.b.json", {"config": "b"})
    panel = _write(path.parent / f"{path.stem}.panel.json", {"roots": 40})
    base_config = dict(base)
    candidate_config = dict(candidate)
    differences = (
        {"n_full"} if stage == "s2" else set(adjudicator.S3_KEYS)
    )
    per_root = []
    for root_index in range(40):
        base_runs = []
        candidate_runs = []
        for repeat in range(4):
            base_policy = (
                {0: 0.7, 1: 0.3} if repeat % 2 == 0 else {0: 0.55, 1: 0.45}
            )
            candidate_policy = (
                {0: 0.625, 1: 0.375}
                if js_reduction > 0.0
                else base_policy
            )
            common = {
                "simulations_used": 64,
                "logical_leaf_evaluations": 64,
                "orientation_evaluation_rows": 64,
                "evaluator_method_calls": 64,
            }
            base_runs.append(
                {
                    "search_seed": 100_000 + root_index * 10 + repeat,
                    "selected_action": 0,
                    "improved_policy": base_policy,
                    "wall_sec": 1.0,
                    **common,
                }
            )
            candidate_runs.append(
                {
                    "search_seed": 200_000 + root_index * 10 + repeat,
                    "selected_action": 1 if top1_delta < 0.0 and repeat % 2 else 0,
                    "improved_policy": candidate_policy,
                    "wall_sec": cost,
                    **common,
                }
            )
        per_root.append(
            {
                "root_index": root_index,
                "root_sha256": "sha256:" + f"{root_index:064x}"[-64:],
                "game_seed": 30_000 + root_index,
                "decision_index": root_index,
                "legal_width": 40 if root_index < 10 else 20,
                "legal_width_bucket": "41+" if root_index < 10 else "17-24",
                "wide_ge_40": root_index < 10,
                "phase": "MAIN",
                "phase_raw": "MAIN",
                "roles": {
                    baseline_role: {
                        "runs": base_runs,
                        "stability": summarize_cross_seed_runs(base_runs),
                    },
                    candidate_role: {
                        "runs": candidate_runs,
                        "stability": summarize_cross_seed_runs(candidate_runs),
                    },
                },
            }
        )
    slices = aggregate_report_slices(per_root, baseline_role, candidate_role)
    search_seed_manifests = {}
    for role in (baseline_role, candidate_role):
        seeds_by_root = [
            [int(run["search_seed"]) for run in root["roles"][role]["runs"]]
            for root in per_root
        ]
        flat = sorted(seed for row in seeds_by_root for seed in row)
        search_seed_manifests[role] = {
            "base_seed": flat[0],
            "seeds_by_root": seeds_by_root,
            "seed_set_sha256": adjudicator._digest_value(flat),
        }
    payload = {
        "schema_version": adjudicator.FIXED_ROOT_SCHEMA,
        "measurement": "fixed_root_search_stability_cost",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": adjudicator._sha256(checkpoint),
        },
        "root_panel": {
            "path": str(panel),
            "file_sha256": adjudicator._sha256(panel),
            "content_sha256": "sha256:" + "a" * 64,
            "root_count": 40,
            "wide_ge_40_count": 10,
            "root_sha256s": ["sha256:" + "b" * 64] * 40,
        },
        "evaluator": {"effective_evaluator_config": _evaluator()},
        "evaluator_runtime": {
            "device": "cuda:0",
            "max_batch_size": 64,
            "max_wait_ms": 3.0,
        },
        "roles": {
            baseline_role: {"effective_search_config": base_config},
            candidate_role: {"effective_search_config": candidate_config},
        },
        "search_config_differences": {key: {} for key in differences},
        "allowed_search_config_differences": sorted(differences),
        "search_seed_manifests": search_seed_manifests,
        "locked_input_file_hashes": {
            str(checkpoint): adjudicator._sha256(checkpoint),
            str(config_a): adjudicator._sha256(config_a),
            str(config_b): adjudicator._sha256(config_b),
            str(panel): adjudicator._sha256(panel),
        },
        "protocol": {
            "force_full": True,
            "repeats_per_root_per_role": 4,
            "wide_slice": "legal_width>=40",
        },
        "probe_elapsed_sec": 10.0,
        "slices": slices,
        "per_root": per_root,
    }
    payload["report_content_sha256"] = adjudicator._digest_value(payload)
    return _write(path, payload)


def _s2_manifest(
    tmp_path: Path,
    *,
    counts: tuple[int, int, int] = (5, 100, 95),
    cost: float = 1.5,
) -> tuple[Path, Path, Path]:
    checkpoint = _write(tmp_path / "champion.pt", b"checkpoint")
    base = _operator()
    candidate = _operator(n_full=128)
    s1 = _source_backed_decision(
        tmp_path / "s1.decision.json",
        stage="s1",
        decision="hold",
        selected={key: base[key] for key in sorted(adjudicator.S1_KEYS)},
    )
    h2h = _h2h(
        tmp_path / "s2.h2h.json",
        checkpoint,
        base_n=64,
        candidate_n=128,
        counts=counts,
    )
    fixed = _fixed_root(
        tmp_path / "s2.fixed.json",
        checkpoint,
        base=base,
        candidate=candidate,
        stage="s2",
        baseline_role="n64",
        candidate_role="n128",
        cost=cost,
    )
    manifest = _manifest(
        tmp_path / "s2.manifest.json",
        stage="s2",
        checkpoint=checkpoint,
        base=base,
        candidate=candidate,
        predecessors=[_ref(s1)],
        evidence={
            "h2h": _ref(h2h),
            "fixed_root": _ref(fixed),
            "baseline_role": "n64",
            "candidate_role": "n128",
        },
    )
    return manifest, checkpoint, s1


def _s3_manifest(
    tmp_path: Path,
    *,
    counts: tuple[int, int, int] = (50, 100, 50),
    js_reduction: float = 0.20,
    top1_delta: float = 0.0,
    overhead: float = 1.1,
    s2_decision: str = "adopt",
) -> Path:
    checkpoint = _write(tmp_path / "champion.pt", b"checkpoint")
    base = _operator(n_full=128) if s2_decision == "adopt" else _operator()
    s1 = _source_backed_decision(
        tmp_path / "s1.decision.json",
        stage="s1",
        decision="hold",
        selected={key: base[key] for key in sorted(adjudicator.S1_KEYS)},
    )
    s2 = _source_backed_decision(
        tmp_path / "s2.decision.json",
        stage="s2",
        decision=s2_decision,
        selected={key: base[key] for key in sorted(adjudicator.S2_KEYS)},
    )
    if s2_decision == "hold":
        return _manifest(
            tmp_path / "s3.manifest.json",
            stage="s3",
            checkpoint=checkpoint,
            base=base,
            candidate=None,
            predecessors=[_ref(s1), _ref(s2)],
            evidence={},
        )
    candidate = _operator(
        n_full=128,
        n_full_wide=256,
        n_full_wide_threshold=40,
        wide_roots_always_full=True,
    )
    h2h = _h2h(
        tmp_path / "s3.h2h.json",
        checkpoint,
        base_n=128,
        candidate_n=128,
        counts=counts,
        candidate_wide=256,
        candidate_wide_threshold=40,
        overhead=overhead,
    )
    fixed = _fixed_root(
        tmp_path / "s3.fixed.json",
        checkpoint,
        base=base,
        candidate=candidate,
        stage="s3",
        baseline_role="n128",
        candidate_role="adaptive_n256",
        js_reduction=js_reduction,
        top1_delta=top1_delta,
    )
    return _manifest(
        tmp_path / "s3.manifest.json",
        stage="s3",
        checkpoint=checkpoint,
        base=base,
        candidate=candidate,
        predecessors=[_ref(s1), _ref(s2)],
        evidence={
            "h2h": _ref(h2h),
            "fixed_root": _ref(fixed),
            "baseline_role": "n128",
            "candidate_role": "adaptive_n256",
        },
    )


def test_s1_complete_grid_without_h1_emits_locked_fallback(tmp_path: Path) -> None:
    manifest, _ = _s1_manifest(tmp_path)
    result = adjudicator.adjudicate(manifest)
    assert result["schema_version"] == adjudicator.DECISION_SCHEMA
    assert result["stage"] == "s1"
    assert result["passed"] is True
    assert result["decision"] == "hold"
    assert result["selected_fields"] == {
        "c_scale": 0.03,
        "rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.98,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
    }
    assert result["selected_fields_sha256"] == adjudicator._digest_value(
        result["selected_fields"]
    )
    assert result["adjudicator"]["sha256"] == adjudicator._sha256(
        Path(adjudicator.__file__)
    )
    assert result["source_artifacts"]


def test_s1_selects_only_h1_winner_and_binds_sigma_artifact(tmp_path: Path) -> None:
    manifest, _ = _s1_manifest(tmp_path, h1_arm="cv50_cs0.1+D1")
    result = adjudicator.adjudicate(manifest)
    assert result["decision"] == "adopt"
    assert result["metrics"]["winner"] == "cv50_cs0.1+D1"
    assert result["selected_fields"]["c_scale"] == 0.1
    assert result["selected_fields"]["rescale_noise_floor_c"] == 1.0
    assert result["selected_fields"]["sigma_eval"] == 0.98


def test_s1_rejects_legacy_d6_fallback_threshold_and_bad_sigma(tmp_path: Path) -> None:
    old_manifest, _ = _s1_manifest(tmp_path / "old", old_threshold_arm="D1")
    with pytest.raises(adjudicator.AdjudicationError, match="inclusive >=20"):
        adjudicator.adjudicate(old_manifest)
    bad_sigma_manifest, _ = _s1_manifest(tmp_path / "sigma", sigma=0.79)
    with pytest.raises(adjudicator.AdjudicationError, match="value_rmse"):
        adjudicator.adjudicate(bad_sigma_manifest)


def test_s1_rejects_hash_drift_and_fabricated_pentanomial_decision(
    tmp_path: Path,
) -> None:
    manifest, _ = _s1_manifest(tmp_path / "drift")
    payload = json.loads(manifest.read_text())
    first = Path(payload["evidence"]["arms"][0]["path"])
    first.write_text(first.read_text() + " ", encoding="utf-8")
    with pytest.raises(adjudicator.AdjudicationError, match="artifact drift"):
        adjudicator.adjudicate(manifest)

    manifest2, _ = _s1_manifest(tmp_path / "fake")
    payload2 = json.loads(manifest2.read_text())
    first2 = Path(payload2["evidence"]["arms"][0]["path"])
    report = json.loads(first2.read_text())
    report["pentanomial_sprt"]["decision"] = "H1"
    _write(first2, report)
    payload2["evidence"]["arms"][0] = _ref(first2)
    _write(manifest2, payload2)
    with pytest.raises(adjudicator.AdjudicationError, match="inconsistent"):
        adjudicator.adjudicate(manifest2)


def test_s1_rejects_nonproduction_decision_cap_even_with_rehashed_artifact(
    tmp_path: Path,
) -> None:
    manifest, _ = _s1_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    first = Path(payload["evidence"]["arms"][0]["path"])
    report = json.loads(first.read_text())
    report["max_decisions"] = 300
    _write(first, report)
    payload["evidence"]["arms"][0] = _ref(first)
    _write(manifest, payload)
    with pytest.raises(adjudicator.AdjudicationError, match="max_decisions"):
        adjudicator.adjudicate(manifest)


def test_s2_adopts_confirmed_n128_below_cost_bound(tmp_path: Path) -> None:
    manifest, _, _ = _s2_manifest(tmp_path, cost=1.5)
    result = adjudicator.adjudicate(manifest)
    assert result["decision"] == "adopt"
    assert result["selected_fields"] == {"n_fast": 16, "n_full": 128, "p_full": 0.25}
    assert result["metrics"]["cost_pass"] is True


def test_s2_holds_without_h1_or_when_cost_is_too_high(tmp_path: Path) -> None:
    no_h1, _, _ = _s2_manifest(tmp_path / "flat", counts=(50, 100, 50), cost=1.5)
    assert adjudicator.adjudicate(no_h1)["decision"] == "hold"
    expensive, _, _ = _s2_manifest(tmp_path / "cost", cost=1.81)
    result = adjudicator.adjudicate(expensive)
    assert result["decision"] == "hold"
    assert result["selected_fields"]["n_full"] == 64


def test_s2_extended_cost_requires_predeclared_clear_margin(tmp_path: Path) -> None:
    low_margin, _, _ = _s2_manifest(tmp_path / "low", counts=(0, 191, 9), cost=1.7)
    low = adjudicator.adjudicate(low_margin)
    assert low["metrics"]["pentanomial_decision"] == "H1"
    assert low["metrics"]["clear_margin"] is False
    assert low["decision"] == "hold"

    clear_margin, _, _ = _s2_manifest(tmp_path / "clear", counts=(0, 180, 20), cost=1.7)
    clear = adjudicator.adjudicate(clear_margin)
    assert clear["metrics"]["clear_margin"] is True
    assert clear["decision"] == "adopt"


def test_s2_positive_screen_cannot_impersonate_confirmation(tmp_path: Path) -> None:
    manifest, _, _ = _s2_manifest(tmp_path, counts=(0, 20, 30), cost=1.5)
    with pytest.raises(adjudicator.AdjudicationError, match="200-pair confirmation"):
        adjudicator.adjudicate(manifest)


def test_s2_rejects_typed_h2h_config_drift_and_raw_pair_fabrication(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _s2_manifest(tmp_path / "config")
    payload = json.loads(manifest.read_text())
    h2h_path = Path(payload["evidence"]["h2h"]["path"])
    h2h = json.loads(h2h_path.read_text())
    h2h["typed_config"]["fields"]["max_decisions"] = 300
    h2h["full_config_hash"] = adjudicator._digest_value(h2h["typed_config"])
    h2h["config_hash"] = "sha256:" + h2h["full_config_hash"].split(":", 1)[1][:16]
    _write(h2h_path, h2h)
    payload["evidence"]["h2h"] = _ref(h2h_path)
    _write(manifest, payload)
    with pytest.raises(adjudicator.AdjudicationError, match="max_decisions"):
        adjudicator.adjudicate(manifest)

    manifest2, _, _ = _s2_manifest(tmp_path / "raw")
    payload2 = json.loads(manifest2.read_text())
    h2h_path2 = Path(payload2["evidence"]["h2h"]["path"])
    h2h2 = json.loads(h2h_path2.read_text())
    h2h2["games"][0]["candidate_won"] = not h2h2["games"][0]["candidate_won"]
    _write(h2h_path2, h2h2)
    payload2["evidence"]["h2h"] = _ref(h2h_path2)
    _write(manifest2, payload2)
    with pytest.raises(adjudicator.AdjudicationError, match="do not reconstruct"):
        adjudicator.adjudicate(manifest2)


def test_s3_adopts_confirmed_stability_path_and_emits_final_teacher(
    tmp_path: Path,
) -> None:
    manifest = _s3_manifest(tmp_path, counts=(50, 100, 50), js_reduction=0.2)
    result = adjudicator.adjudicate(manifest)
    assert result["decision"] == "adopt"
    assert result["metrics"]["strength_pass"] is False
    assert result["metrics"]["stability_pass"] is True
    assert result["selected_fields"] == {
        "n_full_wide": 256,
        "n_full_wide_threshold": 40,
        "wide_roots_always_full": True,
    }
    assert result["final_search_operator"]["n_full"] == 128
    assert result["final_search_operator"]["n_full_wide"] == 256
    assert result["final_search_operator_sha256"] == adjudicator._digest_value(
        result["final_search_operator"]
    )
    assert result["teacher_evaluator_sha256"] == adjudicator._digest_value(
        result["teacher_evaluator"]
    )


def test_s3_h1_path_still_requires_overhead_and_confirmation(tmp_path: Path) -> None:
    too_slow = _s3_manifest(
        tmp_path / "slow",
        counts=(5, 100, 95),
        js_reduction=0.0,
        overhead=1.21,
    )
    assert adjudicator.adjudicate(too_slow)["decision"] == "hold"

    screen = _s3_manifest(
        tmp_path / "screen",
        counts=(0, 20, 30),
        js_reduction=0.0,
        overhead=1.2,
    )
    with pytest.raises(adjudicator.AdjudicationError, match="200-pair confirmation"):
        adjudicator.adjudicate(screen)


def test_s3_non_worse_top1_is_binding_and_global_n256_is_forbidden(
    tmp_path: Path,
) -> None:
    worse_top1 = _s3_manifest(
        tmp_path / "top1",
        counts=(50, 100, 50),
        js_reduction=0.3,
        top1_delta=-0.001,
    )
    assert adjudicator.adjudicate(worse_top1)["decision"] == "hold"

    manifest = _s3_manifest(tmp_path / "global")
    payload = json.loads(manifest.read_text())
    payload["candidate_search_operator"]["n_full"] = 256
    _write(manifest, payload)
    with pytest.raises(adjudicator.AdjudicationError, match="global n256"):
        adjudicator.adjudicate(manifest)


def test_s3_is_stage_complete_hold_when_s2_retains_n64(tmp_path: Path) -> None:
    manifest = _s3_manifest(tmp_path, s2_decision="hold")
    result = adjudicator.adjudicate(manifest)
    assert result["passed"] is True
    assert result["decision"] == "hold"
    assert result["metrics"]["reason"] == "s2_held_n64_s3_ineligible"
    assert result["final_search_operator"]["n_full"] == 64
    assert result["final_search_operator"]["n_full_wide"] is None


def test_lineage_and_fixed_root_config_drift_fail_closed(tmp_path: Path) -> None:
    manifest, _, _ = _s2_manifest(tmp_path / "lineage")
    payload = json.loads(manifest.read_text())
    payload["base_search_operator"]["c_scale"] = 0.1
    _write(manifest, payload)
    with pytest.raises(adjudicator.AdjudicationError, match="does not inherit S1"):
        adjudicator.adjudicate(manifest)

    manifest2, _, _ = _s2_manifest(tmp_path / "fixed")
    payload2 = json.loads(manifest2.read_text())
    fixed_path = Path(payload2["evidence"]["fixed_root"]["path"])
    fixed = json.loads(fixed_path.read_text())
    fixed["allowed_search_config_differences"].append("c_scale")
    fixed.pop("report_content_sha256")
    fixed["report_content_sha256"] = adjudicator._digest_value(fixed)
    _write(fixed_path, fixed)
    payload2["evidence"]["fixed_root"] = _ref(fixed_path)
    _write(manifest2, payload2)
    with pytest.raises(adjudicator.AdjudicationError, match="allowed-differences"):
        adjudicator.adjudicate(manifest2)

    manifest3, _, _ = _s2_manifest(tmp_path / "aggregate")
    payload3 = json.loads(manifest3.read_text())
    fixed_path3 = Path(payload3["evidence"]["fixed_root"]["path"])
    fixed3 = json.loads(fixed_path3.read_text())
    fixed3["slices"]["global"]["comparison"][
        "role_b_over_role_a_wall_ratio"
    ] = 0.01
    fixed3.pop("report_content_sha256")
    fixed3["report_content_sha256"] = adjudicator._digest_value(fixed3)
    _write(fixed_path3, fixed3)
    payload3["evidence"]["fixed_root"] = _ref(fixed_path3)
    _write(manifest3, payload3)
    with pytest.raises(adjudicator.AdjudicationError, match="reconstruct from per_root"):
        adjudicator.adjudicate(manifest3)


def test_cli_creates_read_only_decision_and_refuses_overwrite(tmp_path: Path) -> None:
    manifest, _ = _s1_manifest(tmp_path)
    out = tmp_path / "s1.decision.json"
    command = [
        sys.executable,
        str(Path(adjudicator.__file__)),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
    ]
    first = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    assert out.is_file()
    assert stat.S_IMODE(out.stat().st_mode) == 0o444
    second = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
    os.chmod(out, 0o644)
