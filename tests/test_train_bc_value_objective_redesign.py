"""Focused contracts for the matched MSE-vs-HL-Gauss value tournament."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import train_bc
from tools.train_bc import (
    _hl_gauss_value_targets,
    _resolve_value_objective_weights,
    _train_xdim_batch,
    _value_training_metadata,
    _weighted_mean_loss,
    evaluate_bc_batches,
    main as train_main,
)
from tools.reanalyze_lite import (
    materialize_search_root_values,
    resolve_root_value_materialization,
)
from catan_zero.rl.pipeline_configs import TrainConfig, config_from_payload


def _args(
    mode: str,
    *,
    primary: float = 0.25,
    categorical: float = 0.0,
    scalar_aux: float = 0.0,
    target_lambda: float = 1.0,
    sigma_ratio: float = 0.75,
) -> SimpleNamespace:
    return SimpleNamespace(
        value_head_type=mode,
        value_loss_weight=primary,
        value_categorical_loss_weight=categorical,
        hlgauss_scalar_aux_loss_weight=scalar_aux,
        value_target_lambda=target_lambda,
        value_hlgauss_sigma_ratio=sigma_ratio,
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (_args("mse"), (0.25, 0.0)),
        (_args("scalar"), (0.25, 0.0)),
        (_args("mse", primary=0.0), (0.0, 0.0)),
        (_args("hlgauss"), (0.0, 0.25)),
        (_args("hlgauss", categorical=0.4), (0.0, 0.4)),
        (_args("hlgauss", categorical=0.4, scalar_aux=0.1), (0.1, 0.4)),
    ],
)
def test_resolve_value_objective_weights_matrix(args: SimpleNamespace, expected) -> None:
    assert _resolve_value_objective_weights(args) == pytest.approx(expected)


@pytest.mark.parametrize(
    "args",
    [
        _args("mse", categorical=0.1),
        _args("mse", scalar_aux=0.1),
        _args("hlgauss", primary=0.0, categorical=0.0),
        _args("mse", primary=-0.1),
        _args("hlgauss", categorical=-0.1),
        _args("hlgauss", scalar_aux=-0.1),
        _args("hlgauss", target_lambda=-0.01),
        _args("hlgauss", target_lambda=1.01),
        _args("hlgauss", sigma_ratio=0.0),
        _args("hlgauss", sigma_ratio=-0.1),
        _args("unknown"),
    ],
)
def test_resolve_value_objective_weights_rejects_contradictions(args: SimpleNamespace) -> None:
    with pytest.raises(SystemExit):
        _resolve_value_objective_weights(args)


def test_zero_epoch_run_is_rejected_before_loading_data(tmp_path) -> None:
    with pytest.raises(SystemExit, match="epochs must be >= 1"):
        train_main(
            [
                "--data",
                str(tmp_path / "missing"),
                "--checkpoint",
                str(tmp_path / "out.pt"),
                "--report",
                str(tmp_path / "report.json"),
                "--epochs",
                "0",
                "--skip-guards",
            ]
        )


def test_weighted_objective_uses_ddp_global_denominator(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def fake_all_reduce(denominator, op=None):
        del op
        denominator.fill_(4.0)  # local mass 1 + remote mass 3

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    value = torch.tensor([2.0], requires_grad=True)
    loss = _weighted_mean_loss(value, torch.tensor([1.0]))
    loss.backward()

    # Each rank scales by world_size/global_mass; DDP's later gradient average
    # then yields the true global weighted mean instead of a mean of rank means.
    assert loss.item() == pytest.approx(1.0)
    assert value.grad.item() == pytest.approx(0.5)


def test_fixed_population_denominator_preserves_importance_at_batch_one() -> None:
    torch = pytest.importorskip("torch")

    low = _weighted_mean_loss(
        torch.tensor([2.0]),
        torch.tensor([0.25]),
        fixed_weight_mean=1.0,
    )
    high = _weighted_mean_loss(
        torch.tensor([2.0]),
        torch.tensor([1.75]),
        fixed_weight_mean=1.0,
    )
    self_normalized_low = _weighted_mean_loss(
        torch.tensor([2.0]), torch.tensor([0.25])
    )
    self_normalized_high = _weighted_mean_loss(
        torch.tensor([2.0]), torch.tensor([1.75])
    )

    assert low.item() == pytest.approx(0.5)
    assert high.item() == pytest.approx(3.5)
    assert self_normalized_low.item() == pytest.approx(2.0)
    assert self_normalized_high.item() == pytest.approx(2.0)


def test_weighted_objective_fails_closed_on_collective_error(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def failed_collective(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("collective failed")

    monkeypatch.setattr(dist, "all_reduce", failed_collective)
    with pytest.raises(RuntimeError, match="collective failed"):
        _weighted_mean_loss(
            torch.tensor([2.0], requires_grad=True), torch.tensor([1.0])
        )


def test_scalar_aux_round_trips_and_changes_typed_config_hash() -> None:
    baseline = TrainConfig(value_head_type="hlgauss")
    hybrid = TrainConfig(
        value_head_type="hlgauss",
        hlgauss_scalar_aux_loss_weight=0.125,
    )

    rebuilt = config_from_payload(json.loads(hybrid.canonical_json()))

    assert rebuilt == hybrid
    assert rebuilt.hlgauss_scalar_aux_loss_weight == pytest.approx(0.125)
    assert hybrid.config_hash() != baseline.config_hash()


def test_value_training_metadata_marks_only_optimized_readouts() -> None:
    mse = _value_training_metadata(
        _args("mse"),
        scalar_weight=0.25,
        categorical_weight=0.0,
        categorical_bins=0,
        optimizer_steps=12,
        completed_epochs=2,
        scalar_training_weight_sum=100.0,
        categorical_training_weight_sum=0.0,
    )
    hl = _value_training_metadata(
        _args("hlgauss"),
        scalar_weight=0.0,
        categorical_weight=0.25,
        categorical_bins=33,
        optimizer_steps=24,
        completed_epochs=3,
        scalar_training_weight_sum=0.0,
        categorical_training_weight_sum=100.0,
    )

    assert mse["primary_readout"] == "scalar"
    assert mse["trained_value_readouts"] == ["scalar"]
    assert hl["primary_readout"] == "categorical"
    assert hl["trained_value_readouts"] == ["categorical"]
    assert hl["hlgauss_bins"] == 33
    assert hl["optimizer_steps"] == 24


def test_value_training_metadata_never_attests_a_zero_step_or_zero_mass_head() -> None:
    zero_step = _value_training_metadata(
        _args("hlgauss"),
        scalar_weight=0.0,
        categorical_weight=0.25,
        categorical_bins=33,
        optimizer_steps=0,
        completed_epochs=1,
        scalar_training_weight_sum=0.0,
        categorical_training_weight_sum=100.0,
    )
    zero_mass = _value_training_metadata(
        _args("hlgauss"),
        scalar_weight=0.0,
        categorical_weight=0.25,
        categorical_bins=33,
        optimizer_steps=1,
        completed_epochs=1,
        scalar_training_weight_sum=0.0,
        categorical_training_weight_sum=0.0,
    )

    assert zero_step["trained_value_readouts"] == []
    assert zero_mass["trained_value_readouts"] == []


def test_value_training_metadata_attests_real_mid_epoch_updates() -> None:
    midpoint = _value_training_metadata(
        _args("mse"),
        scalar_weight=0.25,
        categorical_weight=0.0,
        categorical_bins=0,
        optimizer_steps=64,
        completed_epochs=0,
        scalar_training_weight_sum=262_144.0,
        categorical_training_weight_sum=0.0,
    )

    assert midpoint["optimizer_steps"] == 64
    assert midpoint["completed_epochs"] == 0
    assert midpoint["trained_value_readouts"] == ["scalar"]


def test_report_records_requested_and_resolved_value_objective_weights() -> None:
    train_bc_path = Path(__file__).resolve().parents[1] / "tools" / "train_bc.py"
    tree = ast.parse(train_bc_path.read_text(encoding="utf-8"))
    main = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    report = next(
        node.value
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "report"
        and isinstance(node.value, ast.Dict)
    )
    keys = {
        node.value
        for node in report.keys
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert {
        "value_head_type",
        "value_loss_weight",
        "resolved_scalar_value_loss_weight",
        "scalar_value_loss_contract",
        "value_categorical_loss_weight",
        "resolved_categorical_value_loss_weight",
        "hlgauss_scalar_aux_loss_weight",
    } <= keys


def _validation_fixture():
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        value_categorical_bins = 5

        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

    class TinyXDimPolicy:
        device = torch.device("cpu")
        policy_type = "test_xdim"

        def __init__(self) -> None:
            self.model = TinyModel()
            self.categorical_bias = torch.zeros(6, dtype=torch.float32)

        def forward_legal_np(self, obs, legal_action_ids, legal_action_context, *, return_q):
            del obs, legal_action_context, return_q
            rows, width = legal_action_ids.shape
            # Tie outputs to a parameter so this behaves like a real model tensor,
            # while keeping exact expected losses easy to calculate.
            zero = self.model.marker * torch.ones((rows,), dtype=torch.float32)
            categorical = zero[:, None] + self.categorical_bias[None, :]
            return {
                "logits": zero[:, None].expand(rows, width),
                "value": zero,
                # Five win/loss atoms plus the model's truncation class.
                "value_categorical_logits": categorical,
            }

    data = {
        "obs": np.zeros((4, 2), dtype=np.float32),
        "legal_action_ids": np.asarray(
            [[10, 11, -1], [10, 11, -1], [10, 11, -1], [10, 11, -1]],
            dtype=np.int16,
        ),
        "legal_action_context": np.zeros((4, 3, 1), dtype=np.float32),
        "action_taken": np.asarray([10, 11, 10, 11], dtype=np.int16),
        "winner": np.asarray(["BLUE", "RED", "BLUE", "RED"]),
        "player": np.asarray(["BLUE", "BLUE", "RED", "RED"]),
        "truncated": np.zeros(4, dtype=np.bool_),
        "phase": np.asarray(["main"] * 4),
        "teacher_name": np.asarray(["teacher"] * 4),
    }
    return TinyXDimPolicy(), data


def _evaluate(
    policy,
    data,
    *,
    scalar_weight: float,
    categorical_weight: float,
    target_lambda: float = 1.0,
    value_weights: np.ndarray | None = None,
    batch_size: int = 2,
    truncation_weight: float = 0.25,
    data_loader_workers: int = 0,
    data_loader_prefetch: int = 2,
    scalar_value_objective: str = "mse",
    scalar_value_loss_readout: str = "raw",
    scalar_value_loss_scale: float = 1.0,
) -> dict:
    n = len(data["action_taken"])
    if value_weights is None:
        value_weights = np.ones(n, dtype=np.float32)
    return evaluate_bc_batches(
        policy,
        data,
        np.arange(n, dtype=np.int64),
        np.ones(n, dtype=np.float32),
        value_weights,
        batch_size,
        1.0,
        0.0,
        "policy",
        0.5,
        0.0,  # isolate the selected value objective from policy CE
        scalar_weight,
        0.0,
        0.0,
        (),
        10,
        "none",
        1.0,
        5.0,
        0.0,
        {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
        "none",
        truncated_vp_margin_value_weight=truncation_weight,
        value_categorical_loss_weight=categorical_weight,
        value_target_lambda=target_lambda,
        # These tests exercise the superseded corpus-global operator. Production
        # CLI runs now have to request it explicitly.
        value_root_blend_global_compat=(target_lambda != 1.0),
        data_loader_workers=data_loader_workers,
        data_loader_prefetch=data_loader_prefetch,
        scalar_value_objective=scalar_value_objective,
        scalar_value_loss_readout=scalar_value_loss_readout,
        scalar_value_loss_scale=scalar_value_loss_scale,
    )


def test_xdim_validation_memmap_prefetch_is_metric_identical() -> None:
    """Threaded validation changes scheduling, never rows, order, or metrics."""
    policy, data = _validation_fixture()
    corpus = object.__new__(train_bc.MemmapCorpus)
    corpus._eager = data
    corpus._lazy = {}
    corpus.row_count = len(data["action_taken"])

    synchronous = _evaluate(
        policy,
        corpus,
        scalar_weight=0.25,
        categorical_weight=0.0,
        batch_size=2,
        data_loader_workers=0,
    )
    prefetched = _evaluate(
        policy,
        corpus,
        scalar_weight=0.25,
        categorical_weight=0.0,
        batch_size=2,
        data_loader_workers=2,
        data_loader_prefetch=2,
    )

    assert prefetched == synchronous


def test_xdim_validation_default_mse_objective_and_telemetry() -> None:
    policy, data = _validation_fixture()
    policy.model.train(True)

    metrics = _evaluate(policy, data, scalar_weight=0.25, categorical_weight=0.0)

    assert metrics["samples"] == 4
    assert metrics["value_loss"] == pytest.approx(1.0)
    assert metrics["value_categorical_loss"] == pytest.approx(0.0)
    assert metrics["primary_value_loss"] == pytest.approx(metrics["value_loss"])
    assert metrics["primary_value_loss_kind"] == "scalar_mse"
    assert metrics["scalar_value_mse_diagnostic"] == pytest.approx(
        metrics["value_loss"]
    )
    assert metrics["loss"] == pytest.approx(0.25)
    assert metrics["component_reconstructed_loss"] == pytest.approx(metrics["loss"])
    assert policy.model.training is True  # validation restores the caller's mode


def test_xdim_validation_can_score_the_deployed_scalar_search_readout() -> None:
    torch = pytest.importorskip("torch")
    policy, data = _validation_fixture()
    with torch.no_grad():
        policy.model.marker.fill_(2.0)

    raw = _evaluate(
        policy,
        data,
        scalar_weight=1.0,
        categorical_weight=0.0,
        scalar_value_loss_readout="raw",
    )
    deployed = _evaluate(
        policy,
        data,
        scalar_weight=1.0,
        categorical_weight=0.0,
        scalar_value_loss_readout="deployed_tanh",
        scalar_value_loss_scale=1.0,
    )

    bounded = float(np.tanh(2.0))
    expected = ((bounded - 1.0) ** 2 + (bounded + 1.0) ** 2) / 2.0
    assert raw["value_loss"] == pytest.approx(5.0)
    assert deployed["value_loss"] == pytest.approx(expected)
    assert deployed["loss"] == pytest.approx(expected)


def test_binary_win_bce_contract_is_explicit_and_requires_deployed_readout() -> None:
    args = SimpleNamespace(
        scalar_value_objective="binary_win_bce",
        scalar_value_loss_readout="deployed_tanh",
        scalar_value_loss_scale=1.5,
    )
    assert train_bc._scalar_value_loss_contract(args) == {
        "schema_version": "scalar-value-objective-v2",
        "objective": "binary_win_bce",
        "readout": "deployed_tanh",
        "scale": 1.5,
        "target_formula": "(z + 1) / 2",
        "logit_formula": "2 * scale * raw",
        "deployed_value_formula": "tanh(raw * scale)",
        "matches_scalar_mcts_when_value_squash_tanh": True,
    }
    args.scalar_value_loss_readout = "raw"
    with pytest.raises(SystemExit, match="requires.*deployed_tanh"):
        train_bc._scalar_value_loss_contract(args)


def test_scalar_mse_contract_remains_byte_compatible() -> None:
    assert train_bc._scalar_value_loss_contract(
        SimpleNamespace(
            scalar_value_objective="mse",
            scalar_value_loss_readout="deployed_tanh",
            scalar_value_loss_scale=1.0,
        )
    ) == {
        "schema_version": "scalar-value-loss-readout-v1",
        "readout": "deployed_tanh",
        "scale": 1.0,
        "formula": "tanh(raw * scale)",
        "matches_scalar_mcts_when_value_squash_tanh": True,
    }


def test_binary_win_bce_uses_logistic_equivalent_of_deployed_tanh() -> None:
    torch = pytest.importorskip("torch")
    raw = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    targets = torch.tensor([-1.0, 0.0, 1.0])

    objective, mse, prediction = train_bc._scalar_value_objective_errors(
        raw,
        targets,
        objective="binary_win_bce",
        readout="deployed_tanh",
        scale=1.25,
    )

    expected_logits = raw * 2.5
    expected_targets = (targets + 1.0) * 0.5
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        expected_logits,
        expected_targets,
        reduction="none",
    )
    assert torch.allclose(objective, expected)
    assert torch.allclose(prediction, torch.tanh(raw * 1.25))
    assert torch.allclose(mse, (prediction - targets) ** 2)
    objective.mean().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_binary_win_bce_preserves_gradient_on_confidently_wrong_value() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor([-1.0])
    raw_mse = torch.tensor([4.0], requires_grad=True)
    mse, _, _ = train_bc._scalar_value_objective_errors(
        raw_mse,
        target,
        objective="mse",
        readout="deployed_tanh",
        scale=1.0,
    )
    mse.mean().backward()
    mse_gradient = abs(float(raw_mse.grad.item()))

    raw_bce = torch.tensor([4.0], requires_grad=True)
    bce, _, _ = train_bc._scalar_value_objective_errors(
        raw_bce,
        target,
        objective="binary_win_bce",
        readout="deployed_tanh",
        scale=1.0,
    )
    bce.mean().backward()
    bce_gradient = abs(float(raw_bce.grad.item()))

    assert bce_gradient > 100.0 * mse_gradient


def test_xdim_binary_win_bce_keeps_mse_as_a_separate_diagnostic() -> None:
    torch = pytest.importorskip("torch")
    policy, data = _validation_fixture()
    with torch.no_grad():
        policy.model.marker.fill_(2.0)

    metrics = _evaluate(
        policy,
        data,
        scalar_weight=1.0,
        categorical_weight=0.0,
        scalar_value_objective="binary_win_bce",
        scalar_value_loss_readout="deployed_tanh",
        scalar_value_loss_scale=1.0,
    )

    # Two wins and two losses at one shared logit produce the balanced binary
    # CE below. The deployed MSE is intentionally different and independently
    # aggregated so reports cannot call BCE "scalar MSE".
    expected_bce = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.full((4,), 4.0),
            torch.tensor([1.0, 0.0, 0.0, 1.0]),
        )
        .detach()
        .item()
    )
    bounded = float(np.tanh(2.0))
    expected_mse = ((bounded - 1.0) ** 2 + (bounded + 1.0) ** 2) / 2.0
    assert metrics["value_loss"] == pytest.approx(expected_bce)
    assert metrics["scalar_value_mse_diagnostic"] == pytest.approx(
        expected_mse
    )
    assert metrics["primary_value_loss_kind"] == "binary_win_bce"
    assert metrics["loss"] == pytest.approx(expected_bce)
    assert metrics["loss_denominators"][
        "scalar_value_mse_diagnostic"
    ] == pytest.approx(4.0)


def test_xdim_validation_hlgauss_is_categorical_primary_not_double_weighted() -> None:
    policy, data = _validation_fixture()

    metrics = _evaluate(policy, data, scalar_weight=0.0, categorical_weight=0.25)

    # Uniform six-class logits give CE=log(6). Scalar MSE remains useful
    # telemetry, but contributes zero to the matched categorical-primary arm.
    expected_ce = float(np.log(6.0))
    assert metrics["value_loss"] == pytest.approx(1.0)
    assert metrics["value_categorical_loss"] == pytest.approx(expected_ce)
    assert metrics["primary_value_loss"] == pytest.approx(expected_ce)
    assert metrics["primary_value_loss_kind"] == "hlgauss_ce"
    assert metrics["loss"] == pytest.approx(0.25 * expected_ce)
    assert metrics["component_reconstructed_loss"] == pytest.approx(metrics["loss"])
    assert metrics["loss_denominators"]["value_categorical_loss"] == pytest.approx(4.0)


def test_hlgauss_weighted_telemetry_is_partition_invariant_and_splits_truncation() -> None:
    torch = pytest.importorskip("torch")
    policy, data = _validation_fixture()
    data["truncated"] = np.asarray([False, True, False, True], dtype=np.bool_)
    policy.categorical_bias = torch.tensor(
        [-1.5, -0.5, 0.0, 0.75, 1.5, -2.0], dtype=torch.float32
    )
    value_weights = np.asarray([1.0, 3.0, 0.5, 2.0], dtype=np.float32)

    by_one = _evaluate(
        policy,
        data,
        scalar_weight=0.0,
        categorical_weight=1.0,
        value_weights=value_weights,
        batch_size=1,
    )
    uneven = _evaluate(
        policy,
        data,
        scalar_weight=0.0,
        categorical_weight=1.0,
        value_weights=value_weights,
        batch_size=3,
    )

    for key in (
        "value_categorical_loss",
        "value_categorical_clean_loss",
        "value_categorical_truncated_loss",
        "primary_value_loss",
    ):
        assert uneven[key] == pytest.approx(by_one[key])
    expected_clean_mass = float(value_weights[[0, 2]].sum())
    expected_truncated_mass = float(value_weights[[1, 3]].sum()) * 0.25
    assert uneven["loss_denominators"]["value_categorical_loss"] == pytest.approx(
        expected_clean_mass + expected_truncated_mass
    )
    assert uneven["loss_denominators"][
        "value_categorical_clean_loss"
    ] == pytest.approx(expected_clean_mass)
    assert uneven["loss_denominators"][
        "value_categorical_truncated_loss"
    ] == pytest.approx(expected_truncated_mass)
    assert uneven["value_categorical_clean_loss"] != pytest.approx(
        uneven["value_categorical_truncated_loss"]
    )


def test_xdim_validation_applies_same_root_value_lambda_convention_to_both_arms() -> None:
    torch = pytest.importorskip("torch")
    policy, data = _validation_fixture()
    # lambda=0 means pure stored search value under the canonical
    # lambda*z + (1-lambda)*V_search convention.
    data["root_value"] = np.full(4, 0.2, dtype=np.float32)
    data["root_value_mask"] = np.ones(4, dtype=np.bool_)
    policy.categorical_bias = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0, -3.0], dtype=torch.float32
    )

    mse_z = _evaluate(
        policy, data, scalar_weight=1.0, categorical_weight=0.0, target_lambda=1.0
    )
    mse_search = _evaluate(
        policy, data, scalar_weight=1.0, categorical_weight=0.0, target_lambda=0.0
    )
    cat_z = _evaluate(
        policy, data, scalar_weight=0.0, categorical_weight=1.0, target_lambda=1.0
    )
    cat_search = _evaluate(
        policy, data, scalar_weight=0.0, categorical_weight=1.0, target_lambda=0.0
    )

    assert mse_z["value_loss"] == pytest.approx(1.0)
    assert mse_search["value_loss"] == pytest.approx(0.2**2)
    assert cat_z["value_categorical_loss"] != pytest.approx(
        cat_search["value_categorical_loss"]
    )
    assert mse_search["component_reconstructed_loss"] == pytest.approx(mse_search["loss"])
    assert cat_search["component_reconstructed_loss"] == pytest.approx(cat_search["loss"])


def test_reanalysis_feeds_identical_bounded_search_targets_to_mse_and_hlgauss() -> None:
    torch = pytest.importorskip("torch")
    policy, data = _validation_fixture()
    raw = torch.tensor([4.0, -4.0, 0.25, -0.25], dtype=torch.float32)
    spec = resolve_root_value_materialization(
        value_readout="scalar", value_squash="tanh", value_scale=1.5
    )
    bounded = materialize_search_root_values({"value": raw}, spec)
    assert np.all((-1.0 <= bounded) & (bounded <= 1.0))
    data["root_value"] = bounded
    data["root_value_mask"] = np.ones(4, dtype=np.bool_)
    policy.categorical_bias = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0, -3.0], dtype=torch.float32
    )

    mse = _evaluate(
        policy, data, scalar_weight=1.0, categorical_weight=0.0, target_lambda=0.0
    )
    hl = _evaluate(
        policy, data, scalar_weight=0.0, categorical_weight=1.0, target_lambda=0.0
    )

    assert mse["value_loss"] == pytest.approx(float(np.mean(bounded**2)))
    expected_hl_targets = _hl_gauss_value_targets(
        torch.as_tensor(bounded),
        5,
        truncated=torch.zeros(4, dtype=torch.bool),
        add_truncation_class=True,
    )
    log_probs = torch.nn.functional.log_softmax(policy.categorical_bias, dim=-1)
    expected_hl = float((-(expected_hl_targets * log_probs).sum(dim=-1)).mean())
    assert hl["value_categorical_loss"] == pytest.approx(expected_hl)


@pytest.mark.parametrize(
    ("scalar_weight", "categorical_weight"),
    [(0.25, 0.0), (0.0, 0.25)],
)
def test_xdim_train_and_validation_compute_the_same_value_objective(
    scalar_weight: float,
    categorical_weight: float,
) -> None:
    torch = pytest.importorskip("torch")
    policy, data = _validation_fixture()
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
    n = len(data["action_taken"])
    batch = np.arange(n, dtype=np.int64)
    ones = np.ones(n, dtype=np.float32)

    trained = _train_xdim_batch(
        policy,
        optimizer,
        data,
        batch,
        ones,
        ones,
        1.0,
        0.0,
        "policy",
        0.5,
        0.0,
        scalar_weight,
        0.0,
        0.0,
        (),
        10,
        "none",
        1.0,
        5.0,
        0.0,
        diagnostics=False,
        value_categorical_loss_weight=categorical_weight,
        value_target_lambda=1.0,
    )
    validated = _evaluate(
        policy,
        data,
        scalar_weight=scalar_weight,
        categorical_weight=categorical_weight,
    )

    assert validated["loss"] == pytest.approx(trained["loss"])
    assert validated["value_loss"] == pytest.approx(trained["value_loss"])
    assert validated["value_categorical_loss"] == pytest.approx(
        trained["value_categorical_loss"]
    )


def test_xdim_train_and_validation_share_deployed_tanh_value_semantics() -> None:
    torch = pytest.importorskip("torch")
    policy, data = _validation_fixture()
    with torch.no_grad():
        policy.model.marker.fill_(2.0)
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
    batch = np.arange(len(data["action_taken"]), dtype=np.int64)
    weights = np.ones(len(batch), dtype=np.float32)

    trained = _train_xdim_batch(
        policy,
        optimizer,
        data,
        batch,
        weights,
        weights,
        1.0,
        0.0,
        "policy",
        0.5,
        0.0,
        1.0,
        0.0,
        0.0,
        (),
        10,
        "none",
        1.0,
        5.0,
        0.0,
        diagnostics=False,
        scalar_value_loss_readout="deployed_tanh",
        scalar_value_loss_scale=1.0,
    )
    validated = _evaluate(
        policy,
        data,
        scalar_weight=1.0,
        categorical_weight=0.0,
        scalar_value_loss_readout="deployed_tanh",
    )

    assert trained["value_loss"] == pytest.approx(validated["value_loss"])
    assert trained["loss"] == pytest.approx(validated["loss"])


def test_xdim_train_and_validation_share_binary_win_bce_semantics() -> None:
    torch = pytest.importorskip("torch")
    policy, data = _validation_fixture()
    with torch.no_grad():
        policy.model.marker.fill_(1.25)
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
    batch = np.arange(len(data["action_taken"]), dtype=np.int64)
    weights = np.ones(len(batch), dtype=np.float32)

    trained = _train_xdim_batch(
        policy,
        optimizer,
        data,
        batch,
        weights,
        weights,
        1.0,
        0.0,
        "policy",
        0.5,
        0.0,
        1.0,
        0.0,
        0.0,
        (),
        10,
        "none",
        1.0,
        5.0,
        0.0,
        diagnostics=False,
        scalar_value_objective="binary_win_bce",
        scalar_value_loss_readout="deployed_tanh",
        scalar_value_loss_scale=1.0,
    )
    validated = _evaluate(
        policy,
        data,
        scalar_weight=1.0,
        categorical_weight=0.0,
        scalar_value_objective="binary_win_bce",
        scalar_value_loss_readout="deployed_tanh",
    )

    assert trained["value_loss"] == pytest.approx(validated["value_loss"])
    assert trained["scalar_value_mse_diagnostic"] == pytest.approx(
        validated["scalar_value_mse_diagnostic"]
    )
    assert trained["primary_value_loss_kind"] == "binary_win_bce"
    assert validated["primary_value_loss_kind"] == "binary_win_bce"
