from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1] / "tools" / "posthoc_teacher_gap_probe.py"
    )
    spec = importlib.util.spec_from_file_location("posthoc_teacher_gap_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report() -> dict:
    return {
        "arch": "entity_graph",
        "data_format": "memmap",
        "data_fingerprint": "sha256:data",
        "a1_memmap_payload_inventory_sha256": "sha256:inventory",
        "validation_fraction": 0.05,
        "validation_seed": 17,
        "validation_max_samples": 0,
        "validation_game_seed_ranges": None,
        "allow_missing_game_seed_validation_split": False,
        "teacher_weights": {"teacher": 2.0},
        "phase_weights": {"main": 3.0},
        "value_phase_weights": {"main": 4.0},
        "forced_action_weight": 0.1,
        "per_game_policy_weight": True,
        "per_game_policy_weight_mode": "sqrt",
        "forced_row_value_weight": 0.7,
        "per_game_value_weight": True,
        "per_game_value_weight_mode": "equal",
        "winner_sample_weight": 1.2,
        "loser_sample_weight": 0.3,
        "vp_margin_weight": 0.4,
        "vps_to_win": 10,
        "mask_hidden_info": True,
        "batch_size": 512,
        "soft_target_temperature": 0.7,
        "soft_target_weight": 0.9,
        "soft_target_source": "policy",
        "soft_target_min_legal_coverage": 0.5,
        "policy_loss_weight": 1.0,
        "value_loss_weight": 0.25,
        "resolved_scalar_value_loss_weight": 0.25,
        "final_vp_loss_weight": 0.0,
        "q_loss_weight": 0.0,
        "q_skip_teacher_prefixes": ["fast"],
        "advantage_policy_weighting": "none",
        "advantage_temperature": 1.0,
        "advantage_weight_cap": 5.0,
        "advantage_weight_floor": 0.05,
        "amp": "bf16",
        "truncated_vp_margin_value_weight": 0.25,
        "policy_kl_anchor_weight": 0.0,
        "value_uncertainty_loss_weight": 0.0,
        "aux_subgoal_loss_weight": 0.0,
        "moe_balance_loss_weight": 0.01,
        "value_categorical_loss_weight": 0.0,
        "resolved_categorical_value_loss_weight": 0.0,
        "value_hlgauss_sigma_ratio": 0.75,
        "value_target_lambda": 1.0,
    }


class _FakeTrainBC:
    def __init__(self):
        self.calls = {}
        self._MASK_HIDDEN_INFO_PLAYER_TOKENS = False

    def _training_data_fingerprint(self, path, data_format):
        self.calls["fingerprint"] = (path, data_format)
        return "sha256:data"

    def _load_validation_game_seed_manifest_for_training(self, path, **kwargs):
        self.calls["manifest"] = (path, kwargs)
        return {
            "game_seeds": np.asarray([101, 103], dtype=np.int64),
            "validation_row_count": 2,
            "validation_game_seed_set_sha256": "sha256:seeds",
            "manifest_sha256": "sha256:manifest-semantic",
        }

    def MemmapCorpus(self, path):
        self.calls["corpus"] = path
        return {"action_taken": np.arange(5), "game_seed": np.arange(5)}

    def split_train_validation_indices(self, data, **kwargs):
        self.calls["split"] = (data, kwargs)
        return {"train": np.asarray([0, 2, 4]), "validation": np.asarray([1, 3])}

    def build_sample_weights(self, data, **kwargs):
        self.calls["policy_weights"] = kwargs
        return np.asarray([1, 2, 3, 4, 5], dtype=np.float32)

    def build_value_sample_weights(self, data, **kwargs):
        self.calls["value_weights"] = kwargs
        return np.asarray([5, 4, 3, 2, 1], dtype=np.float32)

    def evaluate_bc_batches(self, *args, **kwargs):
        self.calls["evaluate"] = (args, kwargs)
        return {
            "samples": 2,
            "active_policy_teacher_gap_rows": 2,
            "active_policy_kl_target_model_mean": 0.2,
            "active_policy_kl_target_prior_mean": 0.5,
            "active_policy_teacher_gap_closure": 0.6,
            "prior_kl_rows": 4,
            "prior_kl_model_prior_mean": 0.3,
            "prior_kl_target_prior_mean": 0.4,
            "prior_kl_ratio": 0.75,
        }


def _paths(tmp_path: Path, report: dict):
    report_path = tmp_path / "report.json"
    checkpoint = tmp_path / "candidate.pt"
    data = tmp_path / "memmap"
    manifest = tmp_path / "validation.json"
    checkpoint.write_bytes(b"checkpoint")
    data.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, checkpoint, data, manifest


def test_reconstructs_exact_weights_holdout_and_evaluation_recipe(
    tmp_path, monkeypatch
):
    module = _module()
    report = _report()
    report_path, checkpoint, data, manifest = _paths(tmp_path, report)
    fake = _FakeTrainBC()
    policy = SimpleNamespace(name="policy")
    monkeypatch.setattr(module, "_load_train_bc", lambda: fake)
    monkeypatch.setattr(module, "_load_policy", lambda *args: policy)

    result = module.run_probe(
        report_path=report_path,
        checkpoint_path=checkpoint,
        data_path=data,
        validation_manifest_path=manifest,
        device="cpu",
        batch_size=64,
    )

    assert fake._MASK_HIDDEN_INFO_PLAYER_TOKENS is True
    assert fake.calls["policy_weights"] == {
        "teacher_weights": {"teacher": 2.0},
        "phase_weights": {"main": 3.0},
        "forced_action_weight": 0.1,
        "winner_sample_weight": 1.2,
        "loser_sample_weight": 0.3,
        "vp_margin_weight": 0.4,
        "vps_to_win": 10,
        "per_game_policy_weight": True,
        "per_game_policy_weight_mode": "sqrt",
    }
    assert fake.calls["value_weights"] == {
        "phase_weights": {"main": 4.0},
        "forced_row_value_weight": 0.7,
        "per_game_value_weight": True,
        "per_game_value_weight_mode": "equal",
    }
    args, kwargs = fake.calls["evaluate"]
    assert args[0] is policy
    assert np.array_equal(args[2], [1, 3])
    assert np.array_equal(args[3], [1, 2, 3, 4, 5])
    assert np.array_equal(args[4], [5, 4, 3, 2, 1])
    assert args[5] == 64
    assert args[6:10] == (0.7, 0.9, "policy", 0.5)
    assert args[10:14] == (1.0, 0.25, 0.0, 0.0)
    assert kwargs["truncated_vp_margin_value_weight"] == 0.25
    assert kwargs["moe_balance_loss_weight"] == 0.01
    assert result["teacher_gap"] == {
        "active_policy_teacher_gap_rows": 2,
        "active_policy_kl_target_model_mean": 0.2,
        "active_policy_kl_target_prior_mean": 0.5,
        "active_policy_teacher_gap_closure": 0.6,
    }
    assert result["legacy_prior_kl"]["prior_kl_ratio"] == 0.75
    assert result["inputs"]["checkpoint"]["sha256"].startswith("sha256:")
    assert result["inputs"]["training_report"]["sha256"].startswith("sha256:")


def test_refuses_wrong_memmap_fingerprint(tmp_path, monkeypatch):
    module = _module()
    paths = _paths(tmp_path, _report())
    fake = _FakeTrainBC()
    fake._training_data_fingerprint = lambda *_: "sha256:wrong"
    monkeypatch.setattr(module, "_load_train_bc", lambda: fake)
    with pytest.raises(SystemExit, match="fingerprint differs"):
        module.run_probe(
            report_path=paths[0],
            checkpoint_path=paths[1],
            data_path=paths[2],
            validation_manifest_path=paths[3],
            device="cpu",
        )


def test_refuses_incomplete_report_instead_of_guessing_recipe(tmp_path, monkeypatch):
    module = _module()
    report = _report()
    del report["loser_sample_weight"]
    paths = _paths(tmp_path, report)
    monkeypatch.setattr(module, "_load_train_bc", lambda: _FakeTrainBC())
    with pytest.raises(SystemExit, match="loser_sample_weight"):
        module.run_probe(
            report_path=paths[0],
            checkpoint_path=paths[1],
            data_path=paths[2],
            validation_manifest_path=paths[3],
            device="cpu",
        )


def test_refuses_validation_manifest_byte_drift(tmp_path, monkeypatch):
    module = _module()
    report = _report()
    report["input_validation_game_seed_manifest_sha256"] = "sha256:not-the-file"
    paths = _paths(tmp_path, report)
    monkeypatch.setattr(module, "_load_train_bc", lambda: _FakeTrainBC())
    with pytest.raises(SystemExit, match="manifest bytes differ"):
        module.run_probe(
            report_path=paths[0],
            checkpoint_path=paths[1],
            data_path=paths[2],
            validation_manifest_path=paths[3],
            device="cpu",
        )
