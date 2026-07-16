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
        "public_award_feature_contract": "authoritative_v1",
        "public_card_count_features": False,
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
        self.corpus_loads = 0
        self.evaluate_calls = []
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

    def load_teacher_data_memmap(self, path):
        self.corpus_loads += 1
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
        self.evaluate_calls.append((args, kwargs))
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
    assert fake.calls["corpus"] == data
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


def test_functional_drift_uses_only_active_multi_action_rows():
    torch = pytest.importorskip("torch")
    module = _module()
    parent_logits = torch.tensor(
        [[2.0, 0.0, 1_000.0], [1.0, 1_000.0, 1_000.0], [0.0, 1.0, 2.0]]
    )
    candidate_logits = torch.tensor(
        [[0.0, 2.0, -1_000.0], [2.0, -1_000.0, -1_000.0], [2.0, 1.0, 0.0]]
    )
    parts = module._functional_drift_batch(  # noqa: SLF001
        parent_logits,
        candidate_logits,
        torch.tensor([0.1, 9.0, 9.0]),
        torch.tensor([0.4, -9.0, -9.0]),
        legal_mask=torch.tensor(
            [[True, True, False], [True, False, False], [True, True, True]]
        ),
        eligible=torch.tensor([True, True, False]),
    )

    assert parts["rows"] == 1.0
    assert parts["top1_flip_sum"] == 1.0
    assert parts["value_abs_delta_sum"] == pytest.approx(0.3)
    assert parts["value_squared_delta_sum"] == pytest.approx(0.09)
    assert np.isfinite(parts["parent_candidate_kl_sum"])
    assert np.isfinite(parts["candidate_parent_kl_sum"])


def test_single_checkpoint_parent_mode_uses_report_bound_parent(tmp_path, monkeypatch):
    module = _module()
    report_path, candidate, data, manifest = _paths(tmp_path, _report())
    parent = tmp_path / "legacy-parent.pt"
    parent.write_bytes(b"parent")
    report = _report()
    report["init_checkpoint_sha256"] = module._sha256(parent)  # noqa: SLF001
    report_path.write_text(json.dumps(report), encoding="utf-8")
    fake = _FakeTrainBC()
    monkeypatch.setattr(module, "_load_train_bc", lambda: fake)
    monkeypatch.setattr(
        module,
        "_load_policy",
        lambda *_: SimpleNamespace(
            config=SimpleNamespace(public_card_count_features=False),
            public_award_feature_contract="authoritative_v1",
        ),
    )
    monkeypatch.setattr(
        module,
        "_functional_drift",
        lambda **_: {"schema_version": "checkpoint-functional-dose-fingerprint-v1"},
    )

    result = module.run_probe(
        report_path=report_path,
        checkpoint_path=candidate,
        parent_checkpoint_path=parent,
        data_path=data,
        validation_manifest_path=manifest,
        device="cpu",
    )

    assert result["schema_version"] == "posthoc-checkpoint-teacher-gap/v1"
    assert set(result["inputs"]["parent_checkpoint"]) == {"path", "sha256"}
    assert result["functional_dose_fingerprint"]["schema_version"] == (
        "checkpoint-functional-dose-fingerprint-v1"
    )
    assert result["parent_target_kl_mean"] == pytest.approx(0.2)
    assert result["paired_parent_teacher_gap"]["absolute_teacher_gap_closure"] == (
        pytest.approx(0.0)
    )
    assert (
        result["legacy_stored_generation_prior_teacher_gap"]["selection_authority"]
        is False
    )


def test_single_checkpoint_parent_mode_refuses_unbound_parent(tmp_path, monkeypatch):
    module = _module()
    report_path, candidate, data, manifest = _paths(tmp_path, _report())
    parent = tmp_path / "unbound-parent.pt"
    parent.write_bytes(b"parent")
    monkeypatch.setattr(module, "_load_train_bc", lambda: _FakeTrainBC())

    with pytest.raises(SystemExit, match="report-authenticated learner parent"):
        module.run_probe(
            report_path=report_path,
            checkpoint_path=candidate,
            parent_checkpoint_path=parent,
            data_path=data,
            validation_manifest_path=manifest,
            device="cpu",
        )


def test_batch_loads_corpus_and_parent_once_and_compares_all_candidates(
    tmp_path, monkeypatch
):
    module = _module()
    report = _report()
    report_path, _checkpoint, data, manifest = _paths(tmp_path, report)
    parent = tmp_path / "parent.pt"
    step64 = tmp_path / "step64.pt"
    step128 = tmp_path / "step128.pt"
    parent.write_bytes(b"parent")
    step64.write_bytes(b"step64")
    step128.write_bytes(b"step128")
    report["init_checkpoint_sha256"] = module._sha256(parent)  # noqa: SLF001
    report_path.write_text(json.dumps(report), encoding="utf-8")

    fake = _FakeTrainBC()
    loaded = []

    def load_policy(_arch, path, _device):
        loaded.append(path.name)
        return SimpleNamespace(
            name=path.stem,
            config=SimpleNamespace(public_card_count_features=False),
            public_award_feature_contract="authoritative_v1",
        )

    def functional_drift(**kwargs):
        dose = 0.128 if kwargs["candidate_policy"].name == "step128" else 0.064
        return {
            "schema_version": "checkpoint-functional-dose-fingerprint-v1",
            "eligible_rows": 2,
            "surface": "validation_policy_active_multi_action_rows",
            "kl_parent_candidate_mean": dose,
            "kl_candidate_parent_mean": dose + 0.01,
            "top1_flip_rate": dose + 0.02,
            "parent_policy_entropy_mean": 1.0,
            "candidate_policy_entropy_mean": 1.0 - dose,
            "policy_entropy_delta": -dose,
            "value_mean_absolute_delta": dose + 0.03,
            "value_root_mean_square_delta": dose + 0.04,
        }

    monkeypatch.setattr(module, "_load_train_bc", lambda: fake)
    monkeypatch.setattr(module, "_load_policy", load_policy)
    monkeypatch.setattr(module, "_functional_drift", functional_drift)
    result = module.run_batch_probe(
        report_path=report_path,
        checkpoints=[("step64", step64), ("step128", step128)],
        parent_checkpoint_path=parent,
        data_path=data,
        validation_manifest_path=manifest,
        device="cpu",
        batch_size=64,
    )

    assert fake.corpus_loads == 1
    # The exact report-bound parent is evaluated once on the shared holdout,
    # then each candidate is evaluated on that identical row surface.
    assert len(fake.evaluate_calls) == 3
    assert loaded == ["parent.pt", "step64.pt", "step128.pt"]
    assert result["checkpoint_order"] == ["step64", "step128"]
    assert (
        result["shared_holdout"]["parent_checkpoint"]["sha256"]
        == report["init_checkpoint_sha256"]
    )
    assert result["shared_holdout"]["input_surface"] == {
        "public_award_feature_contract": "authoritative_v1",
        "public_card_count_features": False,
        "mask_hidden_info": True,
    }
    assert result["shared_holdout"]["comparison_identity_sha256"].startswith("sha256:")
    assert result["shared_holdout"]["parent_target_kl_mean"] == pytest.approx(0.2)
    assert set(result["checkpoints"]) == {"step64", "step128"}
    assert (
        result["checkpoints"]["step64"]["parent_checkpoint_sha256"]
        == report["init_checkpoint_sha256"]
    )
    delta = result["dose_comparison"]["metrics"]["kl_parent_candidate_mean"]
    assert delta["step128_minus_step64"] == pytest.approx(0.064)


def _teacher_gap_metrics(*, target_model: float, target_prior: float) -> dict:
    return {
        "active_policy_teacher_gap_rows": 11,
        "active_policy_kl_target_model_mean": target_model,
        "active_policy_kl_target_prior_mean": target_prior,
        "active_policy_teacher_gap_closure": 1.0 - target_model / target_prior,
    }


def test_fresh_parent_gap_is_zero_for_parent_bytes_despite_different_stored_prior():
    module = _module()
    # The generation-time prior is much closer to the target than the raw
    # learner parent.  That must not fabricate improvement when the candidate
    # is byte/function identical to the parent.
    parent = _teacher_gap_metrics(target_model=0.4, target_prior=0.1)
    result = module._paired_parent_teacher_gap(  # noqa: SLF001
        candidate=dict(parent), parent=parent
    )

    assert result["schema_version"] == module.PAIRED_PARENT_GAP_SCHEMA
    assert result["parent_active_policy_kl_target_model_mean"] == pytest.approx(0.4)
    assert result["stored_generation_prior"][
        "active_policy_kl_target_prior_mean"
    ] == pytest.approx(0.1)
    assert result["absolute_teacher_gap_closure"] == pytest.approx(0.0)
    assert result["relative_teacher_gap_closure"] == pytest.approx(0.0)
    assert result["improved_over_exact_parent"] is False


def test_fresh_parent_gap_is_positive_when_candidate_moves_toward_teacher():
    module = _module()
    parent = _teacher_gap_metrics(target_model=0.4, target_prior=0.1)
    candidate = _teacher_gap_metrics(target_model=0.2, target_prior=0.1)
    result = module._paired_parent_teacher_gap(  # noqa: SLF001
        candidate=candidate, parent=parent
    )

    assert result["absolute_teacher_gap_closure"] == pytest.approx(0.2)
    assert result["relative_teacher_gap_closure"] == pytest.approx(0.5)
    assert result["improved_over_exact_parent"] is True


def test_batch_refuses_parent_not_bound_by_training_report(tmp_path, monkeypatch):
    module = _module()
    report_path, _checkpoint, data, manifest = _paths(tmp_path, _report())
    parent = tmp_path / "parent.pt"
    candidate = tmp_path / "candidate2.pt"
    parent.write_bytes(b"parent")
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(module, "_load_train_bc", lambda: _FakeTrainBC())

    with pytest.raises(SystemExit, match="report-authenticated learner parent"):
        module.run_batch_probe(
            report_path=report_path,
            checkpoints=[("candidate", candidate)],
            parent_checkpoint_path=parent,
            data_path=data,
            validation_manifest_path=manifest,
            device="cpu",
        )


def test_batch_refuses_parent_candidate_public_card_schema_mismatch(
    tmp_path, monkeypatch
):
    module = _module()
    report = _report()
    # Older authenticated reports may not have recorded this field.  Even for
    # them, a functional-dose comparison must never cross input schemas.
    report.pop("public_card_count_features")
    report_path, _checkpoint, data, manifest = _paths(tmp_path, report)
    parent = tmp_path / "parent.pt"
    candidate = tmp_path / "candidate2.pt"
    parent.write_bytes(b"parent")
    candidate.write_bytes(b"candidate")
    report["init_checkpoint_sha256"] = module._sha256(parent)  # noqa: SLF001
    report_path.write_text(json.dumps(report), encoding="utf-8")

    fake = _FakeTrainBC()

    def load_policy(_arch, path, _device):
        return SimpleNamespace(
            config=SimpleNamespace(
                public_card_count_features=path.name == "candidate2.pt"
            ),
            public_award_feature_contract="authoritative_v1",
        )

    monkeypatch.setattr(module, "_load_train_bc", lambda: fake)
    monkeypatch.setattr(module, "_load_policy", load_policy)

    with pytest.raises(SystemExit, match="different public input schemas"):
        module.run_batch_probe(
            report_path=report_path,
            checkpoints=[("candidate", candidate)],
            parent_checkpoint_path=parent,
            data_path=data,
            validation_manifest_path=manifest,
            device="cpu",
        )


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


def test_modern_report_binds_emitted_holdout_not_input_sentinel(tmp_path, monkeypatch):
    module = _module()
    report = _report()
    paths = _paths(tmp_path, report)
    report["validation_game_seed_manifest"] = str(paths[3])
    report["input_validation_game_seed_manifest_sha256"] = "sha256:upstream-sentinel"
    seeds = np.asarray([101, 103], dtype=np.int64)
    emitted = {
        "schema_version": "train-validation-game-seeds-v1",
        "data": str(paths[2]),
        "data_fingerprint": "sha256:data",
        "validation_fraction": 0.05,
        "validation_seed": 17,
        "validation_max_samples": 0,
        "validation_game_seed_ranges": [],
        "validation_game_seed_count": 2,
        "validation_game_seed_set_sha256": "sha256:seeds",
        "game_seeds": [101, 103],
    }
    paths[3].write_text(json.dumps(emitted), encoding="utf-8")
    paths[0].write_text(json.dumps(report), encoding="utf-8")
    fake = _FakeTrainBC()
    fake._game_seed_set_sha256 = lambda value: (
        "sha256:seeds" if np.array_equal(value, seeds) else "sha256:wrong"
    )
    fake._canonical_json_sha256 = lambda value: "sha256:manifest-semantic"
    monkeypatch.setattr(module, "_load_train_bc", lambda: fake)
    monkeypatch.setattr(module, "_load_policy", lambda *args: SimpleNamespace())

    result = module.run_probe(
        report_path=paths[0],
        checkpoint_path=paths[1],
        data_path=paths[2],
        validation_manifest_path=paths[3],
        device="cpu",
    )
    assert result["teacher_gap"]["active_policy_teacher_gap_rows"] == 2

    other = tmp_path / "other-validation.json"
    other.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="path differs from emitted"):
        module.run_probe(
            report_path=paths[0],
            checkpoint_path=paths[1],
            data_path=paths[2],
            validation_manifest_path=other,
            device="cpu",
        )
