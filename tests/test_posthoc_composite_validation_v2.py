from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import posthoc_composite_validation_v2 as posthoc


class _Composite:
    component_ids = ("current", "replay")
    component_game_sampling_ratios = (0.8, 0.2)
    corpora = (object(), object())

    def __init__(self) -> None:
        self.seeds = np.asarray([11, 11, 12, 12, 21, 21], dtype=np.int64)

    def __getitem__(self, key: str):
        if key == "game_seed":
            return self.seeds
        if key == "action_taken":
            return np.zeros(len(self.seeds), dtype=np.int64)
        raise KeyError(key)

    def component_indices_for_rows(self, rows) -> np.ndarray:
        rows = np.asarray(rows)
        return (rows >= 4).astype(np.int64)


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _report(seed_sha: str) -> dict:
    return {
        "data_format": "memmap",
        "data_fingerprint": "sha256:descriptor",
        "a1_memmap_payload_inventory_sha256": "sha256:inventory",
        "validation_game_seed_set_sha256": seed_sha,
        "validation_samples": 6,
        "arch": "entity_graph",
        "batch_size": 16,
        "teacher_weights": {},
        "phase_weights": {},
        "value_phase_weights": {},
        "forced_action_weight": 0.1,
        "winner_sample_weight": 1.0,
        "loser_sample_weight": 0.3,
        "vp_margin_weight": 0.0,
        "vps_to_win": 10,
        "per_game_policy_weight": False,
        "per_game_policy_weight_mode": "equal",
        "forced_row_value_weight": 1.0,
        "per_game_value_weight": False,
        "per_game_value_weight_mode": "equal",
        "mask_hidden_info": True,
        "soft_target_temperature": 0.7,
        "soft_target_weight": 0.9,
        "soft_target_source": "policy",
        "soft_target_min_legal_coverage": 0.5,
        "policy_loss_weight": 1.0,
        "value_loss_weight": 0.25,
        "value_categorical_loss_weight": 0.0,
        "final_vp_loss_weight": 0.0,
        "q_loss_weight": 0.0,
        "q_skip_teacher_prefixes": [],
        "advantage_policy_weighting": "none",
        "advantage_temperature": 1.0,
        "advantage_weight_cap": 5.0,
        "advantage_weight_floor": 0.05,
        "amp": "bf16",
        "truncated_vp_margin_value_weight": 0.25,
        "policy_kl_anchor_weight": 0.0,
        "policy_kl_anchor_direction": "forward",
        "value_uncertainty_loss_weight": 0.0,
        "aux_subgoal_loss_weight": 0.0,
        "moe_balance_loss_weight": 0.0,
        "value_hlgauss_sigma_ratio": 0.75,
        "value_target_lambda": 1.0,
        "value_root_blend_regime": {"mode": "disabled", "phases": []},
        "checkout_runtime_binding": {"training": "old"},
    }


def test_locked_validation_indices_bind_seed_set_and_all_components(tmp_path: Path) -> None:
    data = _Composite()
    seeds = np.asarray([11, 12, 21], dtype=np.int64)
    seed_sha = posthoc.train_bc._game_seed_set_sha256(seeds)
    manifest = _write(tmp_path / "validation.json", {"game_seeds": seeds.tolist()})
    indices, binding = posthoc._locked_validation_indices(
        data, manifest, _report(seed_sha)
    )
    assert np.array_equal(indices, np.arange(6))
    assert binding["game_seed_set_sha256"] == seed_sha
    assert binding["row_count"] == 6


def test_locked_validation_indices_reject_seed_identity_drift(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "validation.json", {"game_seeds": [11, 12, 21]})
    with pytest.raises(SystemExit, match="seed identity differs"):
        posthoc._locked_validation_indices(
            _Composite(), manifest, _report("sha256:wrong")
        )


def test_run_rescore_is_read_only_and_emits_exact_v2(tmp_path: Path, monkeypatch) -> None:
    data = _Composite()
    seeds = np.asarray([11, 12, 21], dtype=np.int64)
    seed_sha = posthoc.train_bc._game_seed_set_sha256(seeds)
    report_path = _write(tmp_path / "report.json", _report(seed_sha))
    checkpoint = (tmp_path / "checkpoint.pt")
    checkpoint.write_bytes(b"checkpoint")
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["checkpoint"] = str(checkpoint)
    _write(report_path, report_payload)
    descriptor = _write(tmp_path / "descriptor.json", {"schema_version": "fake"})
    manifest = _write(tmp_path / "validation.json", {"game_seeds": seeds.tolist()})

    monkeypatch.setattr(
        posthoc.train_bc,
        "_preflight_memmap_composite_descriptor",
        lambda path: {
            "schema_version": "memmap_composite_v2",
            "payload_inventory_sha256": "sha256:inventory",
            "component_ids": ["current", "replay"],
            "component_game_sampling_ratios": [0.8, 0.2],
        },
    )
    monkeypatch.setattr(
        posthoc.train_bc,
        "_training_data_fingerprint",
        lambda path, fmt: "sha256:descriptor",
    )
    monkeypatch.setattr(posthoc.train_bc, "load_teacher_data_memmap", lambda *a, **k: data)
    monkeypatch.setattr(
        posthoc.train_bc,
        "build_sample_weights",
        lambda *a, **k: np.ones(6, dtype=np.float32),
    )
    monkeypatch.setattr(
        posthoc.train_bc,
        "build_value_sample_weights",
        lambda *a, **k: np.ones(6, dtype=np.float32),
    )
    monkeypatch.setattr(posthoc, "_load_policy", lambda *a, **k: SimpleNamespace())

    def evaluate(policy, corpus, indices, *args, **kwargs):
        del policy, corpus, args, kwargs
        value = float(np.asarray(indices).mean() + 1.0)
        return {
            "samples": int(len(indices)),
            "loss": value,
            "policy_loss": value,
            "loss_denominators": {"policy_loss": float(len(indices))},
            "objective_coefficients": {"policy_loss": 1.0},
        }

    monkeypatch.setattr(posthoc.train_bc, "evaluate_bc_batches", evaluate)
    monkeypatch.setattr(
        posthoc.train_bc,
        "_assert_checkout_runtime_binding",
        lambda: {"evaluation": "current"},
    )
    monkeypatch.setattr(posthoc, "_git_commit", lambda: "abc123")
    before = {path: posthoc._sha256(path) for path in (report_path, checkpoint, descriptor, manifest)}

    result = posthoc.run_rescore(
        report_path=report_path,
        checkpoint_path=checkpoint,
        descriptor_path=descriptor,
        validation_manifest_path=manifest,
        device="cpu",
    )

    assert result["read_only"] is True
    assert result["optimizer_steps"] == 0
    assert result["checkpoint_mutated"] is False
    assert result["evaluation_repo_commit"] == "abc123"
    assert result["evaluation_tool_sha256"].startswith("sha256:")
    assert result["exact_validation"]["schema_version"] == "composite-validation-measure-v2"
    assert before == {
        path: posthoc._sha256(path)
        for path in (report_path, checkpoint, descriptor, manifest)
    }
