from __future__ import annotations

import importlib.util
import hashlib
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


def test_report_parent_binding_prefers_effective_step_zero_reference() -> None:
    module = _module()
    effective_sha = "sha256:" + "a" * 64

    assert module._report_parent_binding(  # noqa: SLF001
        {
            "init_checkpoint_sha256": "sha256:" + "b" * 64,
            "effective_initialization_reference": {
                "schema_version": "train-bc-effective-initialization-reference-v1",
                "optimizer_step": 0,
                "same_training_trajectory": True,
                "checkpoint_sha256": effective_sha,
            },
        }
    ) == (
        "effective_initialization_reference.checkpoint_sha256",
        effective_sha,
    )


def test_report_parent_binding_rejects_malformed_effective_reference() -> None:
    module = _module()
    with pytest.raises(SystemExit, match="effective initialization reference"):
        module._report_parent_binding(  # noqa: SLF001
            {
                "init_checkpoint_sha256": "sha256:" + "b" * 64,
                "effective_initialization_reference": {
                    "schema_version": "train-bc-effective-initialization-reference-v1",
                    "optimizer_step": 1,
                    "same_training_trajectory": True,
                    "checkpoint_sha256": "sha256:" + "a" * 64,
                },
            }
        )


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
        "target_reliability_confidence_weighting": True,
        "target_reliability_confidence_floor": 0.4,
        "forced_row_value_weight": 0.7,
        "forced_row_value_action_type_weights": {"END_TURN": 0.2},
        "per_game_value_weight": True,
        "per_game_value_weight_mode": "equal",
        "winner_sample_weight": 1.2,
        "loser_sample_weight": 0.3,
        "vp_margin_weight": 0.4,
        "vps_to_win": 10,
        "track": "2p_no_trade",
        "graph_history_features": False,
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
        "policy_kl_anchor_direction": "reverse",
        "value_uncertainty_loss_weight": 0.0,
        "aux_subgoal_loss_weight": 0.0,
        "belief_resource_loss_weight": 0.2,
        "moe_balance_loss_weight": 0.01,
        "value_categorical_loss_weight": 0.0,
        "resolved_categorical_value_loss_weight": 0.0,
        "value_hlgauss_sigma_ratio": 0.75,
        "value_target_lambda": 1.0,
        "value_root_blend_regime": {
            "mode": "phase_gated",
            "phases": ["PLAY_TURN"],
        },
        "scalar_value_loss_contract": {
            "schema_version": "scalar-value-loss-readout-v1",
            "readout": "deployed_tanh",
            "scale": 1.25,
            "formula": "tanh(raw * scale)",
        },
        "policy_distillation_scope": None,
        "value_training_scope": None,
    }


class _FakeTrainBC:
    POLICY_TARGET_BLEND_LEGACY_V1 = "legacy"

    def __init__(self):
        self.calls = {}
        self.corpus_loads = 0
        self.evaluate_calls = []
        self._MASK_HIDDEN_INFO_PLAYER_TOKENS = False
        self.policy_scope_calls = []
        self.value_scope_calls = []
        self.action_catalog = object()

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

    def _apply_authenticated_policy_distillation_scope(self, data, weights):
        self.policy_scope_calls.append(data)
        return weights

    def build_value_sample_weights(self, data, **kwargs):
        self.calls["value_weights"] = kwargs
        return np.asarray([5, 4, 3, 2, 1], dtype=np.float32)

    def _apply_authenticated_value_training_scope(self, data, weights):
        self.value_scope_calls.append(data)
        return weights

    def parse_track(self, track, **kwargs):
        self.calls["track"] = (track, kwargs)
        return SimpleNamespace()

    def _action_catalog_for_env_config(self, env_config):
        self.calls["action_catalog_env"] = env_config
        return self.action_catalog

    def _action_catalog_type_projection(self, action_catalog, configured_weights):
        assert action_catalog is self.action_catalog
        assert configured_weights == {}
        return np.ones(3, dtype=np.float64), (
            "ROLL",
            "END_TURN",
            "BUILD_ROAD",
        )

    def evaluate_bc_batches(self, *args, **kwargs):
        self.calls["evaluate"] = (args, kwargs)
        self.evaluate_calls.append((args, kwargs))
        return {
            "samples": 2,
            "loss": 0.6,
            "policy_loss": 0.2,
            "active_policy_teacher_gap_rows": 2,
            "active_policy_kl_target_model_mean": 0.2,
            "active_policy_kl_target_prior_mean": 0.5,
            "active_policy_teacher_gap_closure": 0.6,
            "prior_kl_rows": 4,
            "prior_kl_model_prior_mean": 0.3,
            "prior_kl_target_prior_mean": 0.4,
            "prior_kl_ratio": 0.75,
            "primary_value_loss": 0.4,
            "primary_value_loss_kind": "scalar_mse",
            "scalar_value_mse_diagnostic": 0.4,
            "value_loss": 0.4,
            "loss_denominators": {"value_loss": 2.0},
        }


class _AuxCorpus(dict):
    def __init__(self) -> None:
        super().__init__(
            action_taken=np.arange(5),
            game_seed=np.arange(5),
            stage_c_policy_sampling_weight=np.ones(5, dtype=np.float64),
            policy_weight_multiplier=np.asarray(
                [1, 2, 3, 4, 5], dtype=np.float32
            ),
        )
        self.meta = {
            "stage_c_policy_overlay": {
                "sampling_distribution": {
                    "schema_version": "a1-stage-c-policy-sampling-distribution-v2",
                    "column": "stage_c_policy_sampling_weight",
                    "arm": "STRATEGIC_BALANCED",
                }
            }
        }
        self.policy_aux_phase_scope_authenticated = False


class _AuxFakeTrainBC(_FakeTrainBC):
    class _IndexedValidationWeights:
        def __init__(self, rows, weights):
            self.rows = np.asarray(rows, dtype=np.int64)
            self.weights = np.asarray(weights, dtype=np.float64)

        def __getitem__(self, rows):
            requested = np.asarray(rows, dtype=np.int64)
            positions = {
                int(row): index for index, row in enumerate(self.rows.tolist())
            }
            return np.asarray(
                [self.weights[positions[int(row)]] for row in requested],
                dtype=np.float64,
            )

    def __init__(self):
        super().__init__()
        self.corpus = _AuxCorpus()

    def load_teacher_data_memmap(self, path):
        self.corpus_loads += 1
        self.calls["corpus"] = path
        return self.corpus

    @staticmethod
    def _array_content_sha256(array):
        digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        return f"sha256:{digest}"

    @staticmethod
    def _stage_c_policy_aux_base_measure(data, indices):
        return (
            np.asarray(
                data["stage_c_policy_sampling_weight"][indices], dtype=np.float64
            ),
            "stage_c_strategic_balanced",
        )

    @staticmethod
    def _conditioned_policy_aux_sampling_weights(base, multiplier):
        conditioned = np.where(np.asarray(multiplier) > 0.0, base, 0.0)
        return conditioned / conditioned.sum()

    @staticmethod
    def _policy_aux_validation_objective_weights(q, weights):
        return np.asarray(q, dtype=np.float64) * np.asarray(
            weights, dtype=np.float64
        )

    @staticmethod
    def _combine_policy_aux_validation_metrics(
        base, aux, *, policy_loss_weight, policy_aux_loss_weight, **_
    ):
        combined = dict(base)
        combined["policy_loss"] = float(base["policy_loss"]) + float(
            policy_aux_loss_weight
        ) * float(aux["policy_loss"])
        combined["loss"] = float(base["loss"]) + float(
            policy_loss_weight
        ) * float(policy_aux_loss_weight) * float(aux["policy_loss"])
        return combined

    def evaluate_bc_batches(self, *args, **kwargs):
        self.calls["evaluate"] = (args, kwargs)
        self.evaluate_calls.append((args, kwargs))
        auxiliary = isinstance(args[3], self._IndexedValidationWeights)
        model_kl = 0.8 if auxiliary else 0.2
        prior_kl = 1.0 if auxiliary else 0.5
        return {
            "samples": 2,
            "loss": model_kl + 0.4,
            "policy_loss": model_kl,
            "active_policy_teacher_gap_rows": 2,
            "active_policy_kl_target_model_mean": model_kl,
            "active_policy_kl_target_prior_mean": prior_kl,
            "active_policy_teacher_gap_closure": 1.0 - model_kl / prior_kl,
            "prior_kl_rows": 4,
            "prior_kl_model_prior_mean": 0.3,
            "prior_kl_target_prior_mean": 0.4,
            "prior_kl_ratio": 0.75,
            "primary_value_loss": 0.4,
            "primary_value_loss_kind": "scalar_mse",
            "scalar_value_mse_diagnostic": 0.4,
            "value_loss": 0.4,
            "loss_denominators": {
                "policy_loss": 3.0 if auxiliary else 6.0,
                "value_loss": 2.0,
            },
        }

    def sampler_report(self) -> dict:
        train = np.asarray([0, 2, 4], dtype=np.int64)
        validation = np.asarray([1, 3], dtype=np.int64)
        preconditioning = np.ones(3, dtype=np.float64)
        train_q = np.full(3, 1.0 / 3.0, dtype=np.float64)
        validation_q = np.full(2, 0.5, dtype=np.float64)

        def binding(values, *, mass=False):
            result = {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "content_sha256": self._array_content_sha256(values),
            }
            if mass:
                result["mass"] = float(values.sum())
            return result

        assert train.tolist() == [0, 2, 4]
        assert validation.tolist() == [1, 3]
        return {
            "schema_version": "train-policy-aux-sampling-v1",
            "enabled": True,
            "base_measure": "stage_c_strategic_balanced",
            "exact_per_game_policy_surprise_weighting": False,
            "conditioned_on_positive_policy_loss_weight": True,
            "authenticated_phase_allocation_applied": False,
            "preconditioning_weights": binding(preconditioning),
            "final_sampling_weights": binding(train_q),
            "validation_sampling_weights": binding(validation_q, mass=True),
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


def _parent_prepared(
    module, tmp_path: Path, *, award_training: object
) -> tuple[dict, Path, _FakeTrainBC]:
    parent = tmp_path / "legacy-parent.pt"
    parent.write_bytes(b"parent")
    report = _report()
    report["init_checkpoint_sha256"] = module._sha256(parent)  # noqa: SLF001
    if award_training is not None:
        report["public_award_feature_training"] = award_training
    fake = _FakeTrainBC()
    return (
        {
            "report": report,
            "train_bc": fake,
            "data": object(),
            "device": "cpu",
            "award_contract": "authoritative_v1",
        },
        parent,
        fake,
    )


def _legacy_parent_policy() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(public_card_count_features=False),
        public_award_feature_contract="legacy_zero_v0",
    )


def test_parent_reconstructs_authenticated_public_award_transition(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    award_training = {
        "initializer_contract": "legacy_zero_v0",
        "effective_contract": "authoritative_v1",
        "mixed_corpus_acknowledged": False,
        "legacy_column_zero_initialized": True,
    }
    prepared, parent, fake = _parent_prepared(
        module, tmp_path, award_training=award_training
    )
    policy = _legacy_parent_policy()

    def configure(received_policy, data, args):
        assert received_policy is policy
        assert data is prepared["data"]
        assert args.public_award_feature_contract == "authoritative_v1"
        assert args.allow_mixed_public_award_feature_contracts is False
        received_policy.public_award_feature_contract = "authoritative_v1"
        return {
            "initializer_contract": "legacy_zero_v0",
            "effective_contract": "authoritative_v1",
            "mixed_corpus_acknowledged": False,
            "legacy_column_zero_initialized": True,
        }

    fake._configure_public_award_feature_training = configure
    monkeypatch.setattr(module, "_load_policy", lambda *_args: policy)

    loaded, binding, surface = module._load_parent(  # noqa: SLF001
        prepared, parent, require_report_binding=True
    )

    assert loaded is policy
    assert binding["sha256"] == prepared["report"]["init_checkpoint_sha256"]
    assert binding["reconstructed_public_award_transition"] == {
        "initializer_contract": "legacy_zero_v0",
        "effective_contract": "authoritative_v1",
        "mixed_corpus_acknowledged": False,
        "legacy_column_zero_initialized": True,
    }
    assert surface == {
        "public_award_feature_contract": "authoritative_v1",
        "public_card_count_features": False,
        "mask_hidden_info": True,
    }


@pytest.mark.parametrize(
    "award_training",
    [
        None,
        {
            "initializer_contract": "authoritative_v1",
            "effective_contract": "authoritative_v1",
            "mixed_corpus_acknowledged": False,
            "legacy_column_zero_initialized": True,
        },
    ],
)
def test_parent_refuses_missing_or_mismatched_public_award_transition(
    tmp_path: Path, monkeypatch, award_training: object
) -> None:
    module = _module()
    prepared, parent, _fake = _parent_prepared(
        module, tmp_path, award_training=award_training
    )
    monkeypatch.setattr(module, "_load_policy", lambda *_args: _legacy_parent_policy())

    with pytest.raises(SystemExit, match="transition is not authenticated"):
        module._load_parent(  # noqa: SLF001
            prepared, parent, require_report_binding=True
        )


@pytest.mark.parametrize("malformed", ["false", 0, None])
def test_parent_refuses_non_boolean_mixed_corpus_acknowledgement(
    tmp_path: Path, monkeypatch, malformed: object
) -> None:
    module = _module()
    prepared, parent, _fake = _parent_prepared(
        module,
        tmp_path,
        award_training={
            "initializer_contract": "legacy_zero_v0",
            "effective_contract": "authoritative_v1",
            "mixed_corpus_acknowledged": malformed,
            "legacy_column_zero_initialized": True,
        },
    )
    monkeypatch.setattr(module, "_load_policy", lambda *_args: _legacy_parent_policy())

    with pytest.raises(SystemExit, match="mixed_corpus_acknowledged must be boolean"):
        module._load_parent(  # noqa: SLF001
            prepared, parent, require_report_binding=True
        )


def test_aux_enabled_report_reconstructs_exact_q_times_w_objective(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    fake = _AuxFakeTrainBC()
    report = _report()
    report["policy_aux_active_batch_size"] = 64
    report["policy_aux_loss_weight"] = 0.25
    report["policy_surprise_weight"] = 0.0
    report["per_game_policy_surprise_weighting"] = False
    report["policy_aux_phase_sampling_weights"] = None
    report["policy_aux_sampling"] = fake.sampler_report()
    report_path, checkpoint, data, manifest = _paths(tmp_path, report)
    monkeypatch.setattr(module, "_load_train_bc", lambda: fake)
    monkeypatch.setattr(module, "_load_policy", lambda *_: SimpleNamespace())

    result = module.run_probe(
        report_path=report_path,
        checkpoint_path=checkpoint,
        data_path=data,
        validation_manifest_path=manifest,
        device="cpu",
    )

    assert len(fake.evaluate_calls) == 2
    aux_weights = fake.evaluate_calls[1][0][3]
    assert aux_weights[np.asarray([1, 3], dtype=np.int64)] == pytest.approx(
        [1.0, 2.0]
    )
    assert result["policy_teacher_gap_objective"] == {
        "schema_version": module.POLICY_TEACHER_GAP_OBJECTIVE_SCHEMA,
        "selection_authority": True,
        "objective_matched": True,
        "formula": "base_plus_coefficient_times_aux_policy_teacher_kl",
        "policy_aux_enabled": True,
        "policy_aux_active_batch_size": 64,
        "policy_aux_loss_weight": 0.25,
        "policy_aux_measure": "conditioned_sampling_x_policy_weight",
    }
    assert result["teacher_gap"]["active_policy_kl_target_model_mean"] == (
        pytest.approx(0.4)
    )
    assert result["teacher_gap"]["active_policy_kl_target_prior_mean"] == (
        pytest.approx(0.75)
    )
    assert result["teacher_gap"]["active_policy_teacher_gap_closure"] == (
        pytest.approx(1.0 - 0.4 / 0.75)
    )
    measure = result["shared_holdout"]["objective_reconstruction"][
        "policy_aux_validation_measure"
    ]
    assert measure["sampling_weight_mass"] == pytest.approx(1.0)
    assert measure["objective_weight_mass"] == pytest.approx(3.0)


def test_aux_enabled_report_refuses_validation_sampling_hash_drift(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    fake = _AuxFakeTrainBC()
    report = _report()
    report.update(
        {
            "policy_aux_active_batch_size": 64,
            "policy_aux_loss_weight": 0.25,
            "policy_surprise_weight": 0.0,
            "per_game_policy_surprise_weighting": False,
            "policy_aux_phase_sampling_weights": None,
            "policy_aux_sampling": fake.sampler_report(),
        }
    )
    report["policy_aux_sampling"]["validation_sampling_weights"][
        "content_sha256"
    ] = "sha256:" + "0" * 64
    report_path, checkpoint, data, manifest = _paths(tmp_path, report)
    monkeypatch.setattr(module, "_load_train_bc", lambda: fake)

    with pytest.raises(
        SystemExit, match="validation sampling weights differs from training report"
    ):
        module.run_probe(
            report_path=report_path,
            checkpoint_path=checkpoint,
            data_path=data,
            validation_manifest_path=manifest,
            device="cpu",
        )

def test_reconstructs_exact_weights_holdout_and_evaluation_recipe(
    tmp_path, monkeypatch
):
    module = _module()
    report = _report()
    report["policy_distillation_scope"] = {
        "schema_version": "component-policy-distillation-scope-v1",
        "component_ids": ["current"],
    }
    report["value_training_scope"] = {
        "schema_version": "component-value-training-scope-v1",
        "component_ids": ["current"],
    }
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

    assert result["policy_teacher_gap_objective"] == {
        "schema_version": module.POLICY_TEACHER_GAP_OBJECTIVE_SCHEMA,
        "selection_authority": True,
        "objective_matched": True,
        "formula": "base_policy_teacher_kl",
        "policy_aux_enabled": False,
        "policy_aux_active_batch_size": 0,
        "policy_aux_loss_weight": 0.0,
        "policy_aux_measure": "disabled",
    }
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
        "target_reliability_confidence_weighting": True,
        "target_reliability_confidence_floor": 0.4,
    }
    assert fake.policy_scope_calls == [fake.calls["split"][0]]
    assert fake.calls["value_weights"] == {
        "phase_weights": {"main": 4.0},
        "forced_row_value_weight": 0.7,
        "forced_row_value_action_type_weights": {"END_TURN": 0.2},
        "action_catalog": fake.action_catalog,
        "per_game_value_weight": True,
        "per_game_value_weight_mode": "equal",
    }
    assert fake.value_scope_calls == [fake.calls["split"][0]]
    assert fake.calls["track"] == (
        "2p_no_trade",
        {"vps_to_win": 10, "use_graph_history_features": False},
    )
    args, kwargs = fake.calls["evaluate"]
    assert args[0] is policy
    assert np.array_equal(args[2], [1, 3])
    assert np.array_equal(args[3], [1, 2, 3, 4, 5])
    assert np.array_equal(args[4], [5, 4, 3, 2, 1])
    assert args[5] == 64
    assert args[6:10] == (0.7, 0.9, "policy", 0.5)
    assert args[10:14] == (1.0, 0.25, 0.0, 0.0)
    assert kwargs["truncated_vp_margin_value_weight"] == 0.25
    assert kwargs["policy_kl_anchor_direction"] == "reverse"
    assert kwargs["belief_resource_loss_weight"] == 0.2
    assert kwargs["moe_balance_loss_weight"] == 0.01
    assert kwargs["value_root_blend_phases"] == ("PLAY_TURN",)
    assert kwargs["value_root_blend_global_compat"] is False
    assert kwargs["scalar_value_objective"] == "mse"
    assert kwargs["scalar_value_loss_readout"] == "deployed_tanh"
    assert kwargs["scalar_value_loss_scale"] == pytest.approx(1.25)
    assert kwargs["value_validation_action_types_by_id"] == (
        "ROLL",
        "END_TURN",
        "BUILD_ROAD",
    )
    objective = result["shared_holdout"]["objective_reconstruction"]
    assert objective["target_reliability_confidence_weighting"] is True
    assert objective["forced_row_value_action_type_weights"] == {"END_TURN": 0.2}
    assert objective["policy_kl_anchor_direction"] == "reverse"
    assert objective["value_target_lambda"] == pytest.approx(1.0)
    assert objective["scalar_value_loss_contract"] == {
        "objective": "mse",
        "readout": "deployed_tanh",
        "scale": 1.25,
    }
    assert result["teacher_gap"] == {
        "active_policy_teacher_gap_rows": 2,
        "active_policy_kl_target_model_mean": 0.2,
        "active_policy_kl_target_prior_mean": 0.5,
        "active_policy_teacher_gap_closure": 0.6,
    }
    assert result["legacy_prior_kl"]["prior_kl_ratio"] == 0.75
    assert result["value_quality"]["value"] == pytest.approx(0.4)
    assert result["inputs"]["checkpoint"]["sha256"].startswith("sha256:")
    assert result["inputs"]["training_report"]["sha256"].startswith("sha256:")


def test_binary_value_contract_and_quality_projection_are_preserved() -> None:
    module = _module()
    report = {
        "scalar_value_objective": "binary_win_bce",
        "scalar_value_loss_contract": {
            "schema_version": "scalar-value-objective-v2",
            "objective": "binary_win_bce",
            "readout": "deployed_tanh",
            "scale": 1.25,
            "target_formula": "(z + 1) / 2",
            "logit_formula": "2 * scale * raw",
            "deployed_value_formula": "tanh(raw * scale)",
            "matches_scalar_mcts_when_value_squash_tanh": True,
        }
    }
    assert module._scalar_value_loss_spec(report) == (
        "binary_win_bce",
        "deployed_tanh",
        1.25,
    )
    projection = module._value_quality_projection(
        {
            "primary_value_loss": 0.7,
            "scalar_value_mse_diagnostic": 0.4,
            "value_loss": 0.7,
            "primary_value_loss_kind": "binary_win_bce",
            "loss_denominators": {"value_loss": 8.0},
        }
    )
    assert projection["metric_kind"] == "binary_win_bce"
    assert projection["value"] == pytest.approx(0.7)
    assert projection["scalar_value_mse_diagnostic"] == pytest.approx(0.4)

    report["scalar_value_loss_contract"]["target_formula"] = "z"
    with pytest.raises(SystemExit, match="malformed"):
        module._scalar_value_loss_spec(report)


def test_authenticated_scope_makes_excluded_replay_weights_inert(tmp_path, monkeypatch):
    module = _module()
    report = _report()
    report["policy_distillation_scope"] = {
        "schema_version": "component-policy-distillation-scope-v1",
        "component_ids": ["current"],
    }
    report["value_training_scope"] = {
        "schema_version": "component-value-training-scope-v1",
        "component_ids": ["current"],
    }
    report_path, _checkpoint, data_path, manifest = _paths(tmp_path, report)
    fake = _FakeTrainBC()
    corpus = SimpleNamespace(
        action_taken=np.arange(5),
        game_seed=np.arange(5),
        component_ids=("current", "replay"),
        policy_distillation_scope_authenticated=True,
        policy_distillation_component_indices=(0,),
        value_training_scope_authenticated=True,
        value_training_component_indices=(0,),
    )
    fake.load_teacher_data_memmap = lambda path: corpus
    fake.split_train_validation_indices = lambda data, **kwargs: {
        "train": np.asarray([0, 2, 4]),
        "validation": np.asarray([1, 3]),
    }
    fake._apply_authenticated_policy_distillation_scope = lambda data, weights: (
        np.asarray([weights[0], weights[1], weights[2], 0, 0])
    )
    fake._apply_authenticated_value_training_scope = lambda data, weights: np.asarray(
        [weights[0], weights[1], weights[2], 0, 0]
    )
    monkeypatch.setattr(module, "_load_train_bc", lambda: fake)

    first = module._prepare_probe(  # noqa: SLF001
        report_path=report_path,
        data_path=data_path,
        validation_manifest_path=manifest,
        device="cpu",
    )
    fake.build_sample_weights = lambda *args, **kwargs: np.asarray(
        [1, 2, 3, 1_000_000, 2_000_000], dtype=np.float32
    )
    fake.build_value_sample_weights = lambda *args, **kwargs: np.asarray(
        [5, 4, 3, 1_000_000, 2_000_000], dtype=np.float32
    )
    second = module._prepare_probe(  # noqa: SLF001
        report_path=report_path,
        data_path=data_path,
        validation_manifest_path=manifest,
        device="cpu",
    )

    assert np.array_equal(first["policy_weights"], second["policy_weights"])
    assert np.array_equal(first["value_weights"], second["value_weights"])
    assert first["policy_weights"][3:].tolist() == [0, 0]
    assert first["value_weights"][3:].tolist() == [0, 0]
    assert (
        first["shared_holdout"]["identity_sha256"]
        == second["shared_holdout"]["identity_sha256"]
    )


def test_authenticated_scope_rejects_training_report_component_drift():
    module = _module()
    report = _report()
    report["policy_distillation_scope"] = {"component_ids": ["replay"]}
    report["value_training_scope"] = {"component_ids": ["current"]}
    corpus = SimpleNamespace(
        component_ids=("current", "replay"),
        policy_distillation_scope_authenticated=True,
        policy_distillation_component_indices=(0,),
        value_training_scope_authenticated=True,
        value_training_component_indices=(0,),
    )

    with pytest.raises(SystemExit, match="differs from the loaded corpus"):
        module._scope_identity(corpus, report)  # noqa: SLF001


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


def test_combined_aux_teacher_gap_can_reverse_base_only_ranking() -> None:
    module = _module()

    def metrics(model_kl: float) -> dict:
        return {
            "active_policy_teacher_gap_rows": 10,
            "active_policy_kl_target_model_mean": model_kl,
            "active_policy_kl_target_prior_mean": 1.0,
        }

    parent = module._combine_policy_teacher_gap_metrics(  # noqa: SLF001
        metrics(1.0),
        metrics(1.0),
        coefficient=0.25,
    )
    base_favorite = module._combine_policy_teacher_gap_metrics(  # noqa: SLF001
        metrics(0.8),
        metrics(2.0),
        coefficient=0.25,
    )
    combined_favorite = module._combine_policy_teacher_gap_metrics(  # noqa: SLF001
        metrics(0.9),
        metrics(0.8),
        coefficient=0.25,
    )

    # Base-only KL ranks 0.8 ahead of 0.9.
    assert 0.8 < 0.9
    # The trained objective reverses that ranking after the independently
    # normalized AUX(q*w) term is included.
    assert (
        base_favorite["active_policy_kl_target_model_mean"]
        > combined_favorite["active_policy_kl_target_model_mean"]
    )
    assert (
        parent["active_policy_kl_target_model_mean"]
        - base_favorite["active_policy_kl_target_model_mean"]
        < 0.0
    )
    assert (
        parent["active_policy_kl_target_model_mean"]
        - combined_favorite["active_policy_kl_target_model_mean"]
        > 0.0
    )


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
    assert result["paired_parent_value_quality"]["candidate_minus_parent"] == (
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
