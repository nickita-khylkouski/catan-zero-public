from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import a1_b200_stage_c_learner_campaign as stage_c_campaign
from tools import train_bc


def _ddp_cancelling_objective_gradient_worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
) -> None:
    import torch
    import torch.distributed as dist

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(1, 1, bias=False)]
            )

        def forward(self, value):
            return self.blocks[0](value)

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    try:
        model = torch.nn.parallel.DistributedDataParallel(TinyModel())
        weight = model.module.blocks[0].weight
        sign = 1.0 if rank == 0 else -1.0
        policy_base = sign * weight.sum()
        policy_aux = 2.0 * sign * weight.sum()
        value = 3.0 * sign * weight.sum()
        result = train_bc._objective_gradient_interference(
            SimpleNamespace(model=model),
            policy_objective=policy_base + policy_aux,
            policy_aux_objective=policy_aux,
            value_objective=value,
        )
        Path(out_dir, f"rank-{rank}.json").write_text(
            json.dumps(result, sort_keys=True),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def test_optimizer_observability_reuses_default_off_diagnostics_cadence() -> None:
    parser = train_bc.build_parser()
    assert parser.get_default("train_diagnostics_every_batches") == 0


def test_objective_gradient_interference_short_dose_gets_first_step_baseline() -> None:
    assert train_bc._objective_gradient_interference_due(
        cadence_batches=64,
        batch_number=1,
        baseline_observed=False,
        accum_do_step=True,
    )
    assert not train_bc._objective_gradient_interference_due(
        cadence_batches=64,
        batch_number=1,
        baseline_observed=False,
        accum_do_step=False,
    )


def test_objective_gradient_interference_keeps_independent_batch_cadence() -> None:
    assert train_bc._objective_gradient_interference_due(
        cadence_batches=7,
        batch_number=7,
        baseline_observed=True,
        accum_do_step=True,
    )
    assert not train_bc._objective_gradient_interference_due(
        cadence_batches=7,
        batch_number=8,
        baseline_observed=True,
        accum_do_step=True,
    )
    assert not train_bc._objective_gradient_interference_due(
        cadence_batches=0,
        batch_number=1,
        baseline_observed=False,
        accum_do_step=True,
    )


def test_objective_gradient_observation_does_not_require_module_diagnostics() -> None:
    observed = train_bc._objective_gradient_observation_for_step(
        {
            "objective_gradient_interference": {
                "available": True,
                "trunk_gradient_cosine": -0.25,
            }
        },
        global_step=6,
        optimizer_step_applied=True,
    )

    assert observed == {
        "available": True,
        "trunk_gradient_cosine": -0.25,
        "optimizer_step": 7,
    }


def test_max_grad_norm_cli_default_preserves_historical_threshold() -> None:
    parser = train_bc.build_parser()
    required = [
        "--data",
        "data",
        "--checkpoint",
        "model.pt",
        "--report",
        "report.json",
    ]
    assert parser.get_default("max_grad_norm") == pytest.approx(1.0)
    assert parser.parse_args(
        required + ["--max-grad-norm", "2.0"]
    ).max_grad_norm == pytest.approx(2.0)
    assert parser.parse_args(
        required + ["--max-grad-norm", "0"]
    ).max_grad_norm == pytest.approx(0.0)


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_max_grad_norm_rejects_negative_or_nonfinite_values(value: float) -> None:
    with pytest.raises(SystemExit, match="use 0 to disable"):
        train_bc._validate_max_grad_norm(value)


def test_max_grad_norm_two_and_explicit_off_have_distinct_semantics() -> None:
    torch = pytest.importorskip("torch")

    def policy_with_grad() -> SimpleNamespace:
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.tensor([[3.0]])
        return SimpleNamespace(model=model)

    clipped_policy = policy_with_grad()
    pre_clip = train_bc._clip_grad_norm(clipped_policy, 2.0)
    clipped = train_bc._optimizer_clip_observability(
        pre_clip, max_grad_norm=2.0
    )
    assert pre_clip.item() == pytest.approx(3.0)
    assert clipped_policy.model.weight.grad.item() == pytest.approx(2.0)
    assert clipped["pre_clip_total_grad_norm"] == pytest.approx(3.0)
    assert clipped["post_clip_total_grad_norm"] == pytest.approx(2.0)
    assert clipped["global_grad_clip_coefficient"] == pytest.approx(2.0 / 3.0)
    assert clipped["max_grad_norm"] == pytest.approx(2.0)
    assert clipped["gradient_clipping_enabled"] is True
    assert (
        clipped["gradient_clip_scope"]
        == "all_trainable_parameters_one_global_norm"
    )
    assert clipped["gradient_clip_preserves_combined_direction"] is True
    assert clipped["clipped"] is True

    off_policy = policy_with_grad()
    pre_clip = train_bc._clip_grad_norm(off_policy, 0.0)
    off = train_bc._optimizer_clip_observability(pre_clip, max_grad_norm=0.0)
    assert pre_clip.item() == pytest.approx(3.0)
    assert off_policy.model.weight.grad.item() == pytest.approx(3.0)
    assert off["post_clip_total_grad_norm"] == pytest.approx(3.0)
    assert off["global_grad_clip_coefficient"] == pytest.approx(1.0)
    assert off["gradient_clipping_enabled"] is False
    assert off["clipped"] is False


def test_optimizer_observability_reports_preclip_norm_clip_and_module_updates() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Linear(2, 2, bias=False)
            self.value_head = torch.nn.Linear(2, 1, bias=False)

    model = TinyModel()
    policy = SimpleNamespace(model=model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)

    expected_total = sum(parameter.numel() * 4.0 for parameter in model.parameters()) ** 0.5
    state = train_bc._capture_optimizer_observability(policy)
    pre_clip = train_bc._clip_grad_norm(policy, 1.0)
    optimizer.step()
    observed = train_bc._finish_optimizer_observability(
        policy,
        state,
        pre_clip_total_grad_norm=pre_clip,
        max_grad_norm=1.0,
    )

    assert observed["pre_clip_total_grad_norm"] == pytest.approx(expected_total)
    assert observed["post_clip_total_grad_norm"] == pytest.approx(1.0)
    assert observed["global_grad_clip_coefficient"] == pytest.approx(
        1.0 / expected_total
    )
    assert observed["max_grad_norm"] == pytest.approx(1.0)
    assert observed["clipped"] is True
    assert observed["module_norm_scope"] == "global_replicated"
    assert set(observed["module_pre_clip_grad_norms"]) == {"trunk", "value_head"}
    assert observed["module_pre_clip_grad_norms"]["trunk"] == pytest.approx(4.0)
    assert observed["module_pre_clip_grad_norms"]["value_head"] == pytest.approx(
        2.0 * (2.0**0.5)
    )
    assert observed["module_parameter_delta_norms"]["trunk"] > 0.0
    assert observed["module_parameter_delta_norms"]["value_head"] > 0.0
    assert observed["module_parameter_counts"] == {"trunk": 4, "value_head": 2}
    assert observed["module_parameter_update_rms"]["trunk"] == pytest.approx(
        observed["module_parameter_delta_norms"]["trunk"] / 2.0
    )
    assert observed["module_relative_parameter_delta"]["trunk"] > 0.0


def test_optimizer_observability_name_normalizes_ddp_and_fsdp_prefixes() -> None:
    normalize = train_bc._optimizer_observability_module_name
    assert normalize("module.blocks.0.weight") == "blocks"
    assert normalize("_fsdp_wrapped_module.value_head.weight") == "value_head"
    assert normalize("module._fsdp_wrapped_module.action_bias") == "action_bias"


def test_objective_gradient_interference_measures_shared_trunk_not_heads() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 1, bias=False)])
            self.policy_head = torch.nn.Linear(1, 1, bias=False)
            self.value_head = torch.nn.Linear(1, 1, bias=False)

    model = TinyModel()
    with torch.no_grad():
        model.blocks[0].weight.copy_(torch.tensor([[1.0, 1.0]]))
        model.policy_head.weight.fill_(1.0)
        model.value_head.weight.fill_(1.0)
    policy = SimpleNamespace(model=model)
    shared = model.blocks[0](torch.tensor([[1.0, -2.0]]))
    policy_objective = model.policy_head(shared).sum()
    value_objective = -2.0 * model.value_head(shared).sum()

    observed = train_bc._objective_gradient_interference(
        policy,
        policy_objective=policy_objective,
        value_objective=value_objective,
    )

    assert observed["available"] is True
    assert observed["scope"] == "single_process_microbatch"
    assert observed["value_lr_mult_scales_shared_trunk"] is False
    assert observed["policy_trunk_grad_norm"] == pytest.approx(5.0**0.5)
    assert observed["value_trunk_grad_norm"] == pytest.approx(2.0 * 5.0**0.5)
    assert observed["value_to_policy_grad_norm_ratio"] == pytest.approx(2.0)
    assert observed["trunk_gradient_cosine"] == pytest.approx(-1.0)
    assert observed["opposing_coordinate_fraction"] == pytest.approx(1.0)
    assert observed["combined_trunk_grad_norm"] == pytest.approx(5.0**0.5)
    assert observed["modules"]["blocks.0"]["cosine"] == pytest.approx(-1.0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_objective_gradient_interference_includes_shared_action_representation() -> None:
    """Action-aware value gradients must not disappear from interference telemetry."""
    torch = pytest.importorskip("torch")

    class TinyActionAwareModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.legal_action_value_residual_enabled = True
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(1, 1, bias=False)]
            )
            self.action_encoder = torch.nn.Linear(1, 1, bias=False)
            self.policy_head = torch.nn.Linear(1, 1, bias=False)
            self.value_head = torch.nn.Linear(1, 1, bias=False)

    model = TinyActionAwareModel()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
    state = model.blocks[0](torch.ones(1, 1))
    action = model.action_encoder(torch.full((1, 1), 2.0))
    shared = state + action
    policy_objective = model.policy_head(shared).sum()
    value_objective = -model.value_head(shared).sum()

    observed = train_bc._objective_gradient_interference(
        SimpleNamespace(model=model),
        policy_objective=policy_objective,
        value_objective=value_objective,
    )

    expected = 5.0**0.5
    assert observed["policy_trunk_grad_norm"] == pytest.approx(expected)
    assert observed["value_trunk_grad_norm"] == pytest.approx(expected)
    assert observed["trunk_gradient_cosine"] == pytest.approx(-1.0)
    assert set(observed["modules"]) == {"action_encoder", "blocks.0"}
    assert all(parameter.grad is None for parameter in model.parameters())


def test_objective_gradient_interference_includes_initial_road_residual() -> None:
    """The repaired road context is shared once value consumes legal actions."""
    torch = pytest.importorskip("torch")

    class TinyRoadAwareModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.legal_action_value_residual_enabled = True
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(1, 1, bias=False)]
            )
            self.v6_initial_road_residual = torch.nn.Linear(1, 1, bias=False)
            self.policy_head = torch.nn.Linear(1, 1, bias=False)
            self.value_head = torch.nn.Linear(1, 1, bias=False)

    model = TinyRoadAwareModel()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
    shared = model.blocks[0](torch.ones(1, 1))
    shared = shared + model.v6_initial_road_residual(
        torch.full((1, 1), 2.0)
    )
    policy_objective = model.policy_head(shared).sum()
    value_objective = -model.value_head(shared).sum()

    observed = train_bc._objective_gradient_interference(
        SimpleNamespace(model=model),
        policy_objective=policy_objective,
        value_objective=value_objective,
    )

    expected = 5.0**0.5
    assert observed["policy_trunk_grad_norm"] == pytest.approx(expected)
    assert observed["value_trunk_grad_norm"] == pytest.approx(expected)
    assert observed["trunk_gradient_cosine"] == pytest.approx(-1.0)
    assert set(observed["modules"]) == {
        "blocks.0",
        "v6_initial_road_residual",
    }
    assert all(parameter.grad is None for parameter in model.parameters())


def test_objective_gradient_interference_is_explicit_when_objective_inactive() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])

    model = TinyModel()
    active = model.blocks[0](torch.ones(1, 1)).sum()
    observed = train_bc._objective_gradient_interference(
        SimpleNamespace(model=model),
        policy_objective=active,
        value_objective=torch.zeros(()),
    )
    assert observed == {
        "available": False,
        "reason": "inactive_policy_or_value_objective",
    }


def test_objective_gradient_interference_separates_active_policy_aux_branch() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 1, bias=False)])

    model = TinyModel()
    with torch.no_grad():
        model.blocks[0].weight.copy_(torch.tensor([[1.0, 1.0]]))
    shared = model.blocks[0](torch.tensor([[1.0, -2.0]]))
    base = shared.sum()
    aux = 2.0 * shared.sum()
    value = -shared.sum()
    belief = 3.0 * shared.sum()

    observed = train_bc._objective_gradient_interference(
        SimpleNamespace(model=model),
        policy_objective=base + aux,
        policy_aux_objective=aux,
        value_objective=value,
        additional_objectives={"belief_resource": belief},
    )

    root5 = 5.0**0.5
    assert observed["policy_base_trunk_grad_norm"] == pytest.approx(root5)
    assert observed["policy_aux_trunk_grad_norm"] == pytest.approx(2.0 * root5)
    assert observed["policy_trunk_grad_norm"] == pytest.approx(3.0 * root5)
    assert observed["policy_aux_to_base_grad_norm_ratio"] == pytest.approx(2.0)
    assert observed["policy_base_aux_gradient_cosine"] == pytest.approx(1.0)
    assert observed["objective_trunk_grad_l2"] == pytest.approx(
        {
            "policy": 3.0 * root5,
            "policy_base": root5,
            "active_policy": 2.0 * root5,
            "value": root5,
            "belief_resource": 3.0 * root5,
        }
    )
    assert observed["modules"]["blocks.0"]["policy_aux_grad_norm"] == pytest.approx(
        2.0 * root5
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_objective_gradient_interference_rejects_rank_local_signal_that_cancels_globally(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    import torch.multiprocessing as mp

    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _ddp_cancelling_objective_gradient_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    results = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert results[0] == results[1]
    observed = results[0]
    assert observed["scope"] == "global_ddp_microbatch"
    assert observed["aggregation"] == (
        "manual_all_reduce_then_world_average_of_ddp_scaled_gradients"
    )
    assert observed["world_size"] == 2
    assert observed["policy_objective"] == pytest.approx(0.0)
    assert observed["value_objective"] == pytest.approx(0.0)
    assert observed["policy_base_trunk_grad_norm"] == pytest.approx(0.0)
    assert observed["policy_aux_trunk_grad_norm"] == pytest.approx(0.0)
    assert observed["policy_trunk_grad_norm"] == pytest.approx(0.0)
    assert observed["value_trunk_grad_norm"] == pytest.approx(0.0)

    report = {
        "objective_gradient_interference_every_batches": (
            stage_c_campaign.OBJECTIVE_GRADIENT_CADENCE_BATCHES
        ),
        "objective_gradient_interference": {
            "cadence_batches": (
                stage_c_campaign.OBJECTIVE_GRADIENT_CADENCE_BATCHES
            ),
            "observations": [
                {
                    **observed,
                    "optimizer_step": (
                        stage_c_campaign.OBJECTIVE_GRADIENT_CADENCE_BATCHES
                    ),
                },
                {
                    **observed,
                    "optimizer_step": stage_c_campaign.MAX_STEPS,
                },
            ],
        },
    }
    with pytest.raises(
        stage_c_campaign.CampaignError,
        match="policy-base/AUX/value",
    ):
        stage_c_campaign._verify_completed_objective_gradient_signal(report)


def test_checkpoint_dose_telemetry_binds_exposure_and_feature_paths() -> None:
    metric = {
        "samples": 64,
        "policy_base_active_rows": 20,
        "policy_aux_active_rows": 12,
        "policy_base_effective_weight_sum": 18.0,
        "policy_aux_effective_weight_sum": 9.0,
        "policy_aux_loss_coefficient": 0.25,
        "value_active_rows": 60,
        "policy_kl_anchor_eligible_rows": 16,
        "loss_denominators": {
            "policy_loss": 27.0,
            "policy_base_loss": 18.0,
            "policy_aux_loss": 9.0,
            "value_loss": 51.0,
        },
        "objective_gradient_interference": {
            "observations": [
                {
                    "available": True,
                    "optimizer_step": 8,
                    "objective_trunk_grad_l2": {"policy": 0.5, "value": 0.25},
                }
            ]
        },
        "module_optimizer_observability": {
            "observed_steps": 1,
            "norm_scope": "global_replicated",
            "modules": {
                "public_card_count_residual": {
                    "mean_pre_clip_grad_norm": 0.4,
                    "max_pre_clip_grad_norm": 0.4,
                    "mean_parameter_delta_norm": 0.01,
                    "mean_parameter_update_rms": 0.001,
                    "mean_relative_parameter_delta": 0.02,
                    "parameter_count": 8,
                },
                "event_encoder": {
                    "mean_pre_clip_grad_norm": 0.3,
                    "max_pre_clip_grad_norm": 0.3,
                    "mean_parameter_delta_norm": 0.02,
                    "mean_parameter_update_rms": 0.002,
                    "mean_relative_parameter_delta": 0.03,
                    "parameter_count": 16,
                },
                "meaningful_history_sequence": {
                    "mean_pre_clip_grad_norm": 0.2,
                    "max_pre_clip_grad_norm": 0.2,
                    "mean_parameter_delta_norm": 0.015,
                    "mean_parameter_update_rms": 0.0015,
                    "mean_relative_parameter_delta": 0.025,
                    "parameter_count": 24,
                },
                "meaningful_history_target_proj": {
                    "mean_pre_clip_grad_norm": 0.1,
                    "max_pre_clip_grad_norm": 0.1,
                    "mean_parameter_delta_norm": 0.005,
                    "mean_parameter_update_rms": 0.0005,
                    "mean_relative_parameter_delta": 0.01,
                    "parameter_count": 12,
                },
            },
        },
    }

    dose = train_bc._checkpoint_dose_telemetry(
        [metric],
        optimizer_step=8,
        optimizer_observed_steps=8,
        optimizer_clipped_steps=2,
        optimizer_zero_objective_steps=0,
        optimizer_pre_clip_grad_norm_sum=4.0,
        optimizer_pre_clip_grad_norm_max=0.8,
        objective_gradient_cadence_batches=8,
        train_diagnostic_cadence_batches=8,
        public_card_enabled=True,
        meaningful_history_enabled=True,
        max_grad_norm=1.0,
    )

    assert dose["schema_version"] == train_bc.CHECKPOINT_DOSE_TELEMETRY_SCHEMA
    assert dose["active_rows"]["policy_total"] == 32
    assert dose["policy_effective_weight_sums"]["total"] == pytest.approx(27.0)
    assert dose["policy_stream_objective"] == {
        "schema_version": "train-policy-stream-objective-v1",
        "formula": "base_mean + aux_coefficient * aux_mean",
        "normalization": "independent_weighted_means",
        "base_coefficient": 1.0,
        "aux_enabled": True,
        "aux_coefficient": 0.25,
        "base_denominator": 18.0,
        "aux_denominator": 9.0,
    }
    assert dose["optimizer"]["clipped_fraction"] == pytest.approx(0.25)
    assert dose["optimizer"]["max_grad_norm"] == pytest.approx(1.0)
    assert dose["optimizer"]["gradient_clipping_enabled"] is True
    assert (
        dose["optimizer"]["gradient_clip_scope"]
        == "all_trainable_parameters_one_global_norm"
    )
    assert (
        dose["optimizer"]["minimum_observed_global_grad_clip_coefficient"]
        == pytest.approx(1.0)
    )
    assert dose["optimizer"]["cross_objective_coupling"] == (
        "policy_value_and_private_tower_gradients_share_the_same_per_step_"
        "global_clip_coefficient"
    )
    assert dose["policy_objective_dose"]["interpretation"] == {
        "schema_version": "policy-dose-clipping-interpretation-v1",
        "raw_exposure_authority": "training_row_draws_and_active_rows",
        "coefficient_weighted_measure": "pre_clip_objective_mixture_proxy",
        "coefficient_weighted_measure_is_realized_update_amplitude": False,
        "aux_coefficient_effect": (
            "changes_pre_clip_objective_mixture_and_gradient_direction"
        ),
        "realized_update_amplitude_status": "partially_clip_limited",
        "realized_update_amplitude_semantics": (
            "global_grad_clip_limited_on_a_subset_of_observed_steps"
        ),
        "clip_scope": "all_trainable_parameters_one_global_norm",
        "clip_transform": (
            "one_uniform_scalar_preserves_combined_gradient_direction"
        ),
        "cross_objective_coupling": (
            "a_policy_dominated_global_norm_scales_value_and_private_tower_"
            "gradients_by_the_same_clip_coefficient"
        ),
        "optimizer_observed_steps": 8,
        "optimizer_clipped_steps": 2,
        "clipped_fraction": pytest.approx(0.25),
    }
    assert dose["shared_trunk_objective_gradients"]["observed_steps"] == 1
    assert (
        dose["feature_path_gradients"]["public_card"]["status"]
        == "observed_nonzero"
    )
    assert (
        dose["feature_path_gradients"]["meaningful_history"]["status"]
        == "observed_nonzero"
    )
    assert dose["feature_path_gradients"]["public_card"][
        "nonzero_signal_modules"
    ] == ["public_card_count_residual"]
    assert dose["feature_path_gradients"]["public_card"][
        "zero_signal_modules"
    ] == []
    assert dose["feature_path_gradients"]["meaningful_history"][
        "nonzero_signal_modules"
    ] == [
        "event_encoder",
        "meaningful_history_sequence",
        "meaningful_history_target_proj",
    ]
    assert (
        dose["feature_path_gradients"]["public_card"][
            "independent_loss_objective"
        ]
        is False
    )


def test_checkpoint_dose_telemetry_reports_v7_input_paths() -> None:
    def row(value: float) -> dict[str, float | int]:
        return {
            "mean_pre_clip_grad_norm": value,
            "max_pre_clip_grad_norm": value,
            "mean_parameter_delta_norm": value,
            "mean_parameter_update_rms": value,
            "mean_relative_parameter_delta": value,
            "parameter_count": 8,
        }

    metric = {
        "module_optimizer_observability": {
            "schema_version": "module-optimizer-observability-v1",
            "observed_steps": 1,
            "cadence_batches": 1,
            "norm_scope": "pre_clip",
            "modules": {
                "v6_exact_resource_residual": row(0.1),
                "v6_initial_road_residual": row(0.2),
            },
        }
    }
    dose = train_bc._checkpoint_dose_telemetry(
        [metric],
        optimizer_step=1,
        optimizer_observed_steps=1,
        optimizer_clipped_steps=0,
        optimizer_zero_objective_steps=0,
        optimizer_pre_clip_grad_norm_sum=0.3,
        optimizer_pre_clip_grad_norm_max=0.3,
        objective_gradient_cadence_batches=0,
        train_diagnostic_cadence_batches=1,
        public_card_enabled=False,
        meaningful_history_enabled=False,
        v7_compatibility_inputs_enabled=True,
    )

    path = dose["feature_path_gradients"]["v7_compatibility_inputs"]
    assert path["status"] == "observed_nonzero"
    assert path["nonzero_signal_modules"] == [
        "v6_exact_resource_residual",
        "v6_initial_road_residual",
    ]


def test_checkpoint_dose_telemetry_distinguishes_zero_signal_modules() -> None:
    zero_signal = {
        "mean_pre_clip_grad_norm": 0.0,
        "max_pre_clip_grad_norm": 0.0,
        "mean_parameter_delta_norm": 0.0,
        "mean_parameter_update_rms": 0.0,
        "mean_relative_parameter_delta": 0.0,
        "parameter_count": 8,
    }
    nonzero_signal = {
        **zero_signal,
        "mean_pre_clip_grad_norm": 0.25,
        "max_pre_clip_grad_norm": 0.25,
        "mean_parameter_delta_norm": 0.01,
        "mean_parameter_update_rms": 0.001,
    }
    metric = {
        "samples": 1,
        "module_optimizer_observability": {
            "observed_steps": 1,
            "norm_scope": "global_replicated",
            "modules": {
                "public_card_count_residual": dict(zero_signal),
                "event_encoder": dict(zero_signal),
                "meaningful_history_residual_gate": dict(nonzero_signal),
                "meaningful_history_sequence": dict(zero_signal),
                "meaningful_history_target_proj": dict(zero_signal),
            },
        },
    }

    dose = train_bc._checkpoint_dose_telemetry(
        [metric],
        optimizer_step=1,
        optimizer_observed_steps=1,
        optimizer_clipped_steps=0,
        optimizer_zero_objective_steps=0,
        optimizer_pre_clip_grad_norm_sum=0.25,
        optimizer_pre_clip_grad_norm_max=0.25,
        objective_gradient_cadence_batches=1,
        train_diagnostic_cadence_batches=1,
        public_card_enabled=True,
        meaningful_history_enabled=True,
    )

    public_card = dose["feature_path_gradients"]["public_card"]
    assert public_card["status"] == "observed_zero"
    assert public_card["nonzero_signal_modules"] == []
    assert public_card["zero_signal_modules"] == [
        "public_card_count_residual"
    ]

    history = dose["feature_path_gradients"]["meaningful_history"]
    assert history["status"] == "observed_partial"
    assert history["nonzero_signal_modules"] == [
        "meaningful_history_residual_gate"
    ]
    assert history["zero_signal_modules"] == [
        "event_encoder",
        "meaningful_history_sequence",
        "meaningful_history_target_proj",
    ]


def test_checkpoint_dose_refuses_policy_aux_coefficient_drift() -> None:
    metrics = [
        {
            "samples": 1,
            "policy_base_active_rows": 0,
            "policy_aux_active_rows": 1,
            "policy_base_effective_weight_sum": 0.0,
            "policy_aux_effective_weight_sum": 1.0,
            "policy_aux_loss_coefficient": coefficient,
            "loss_denominators": {
                "policy_base_loss": 0.0,
                "policy_aux_loss": 1.0,
            },
        }
        for coefficient in (0.25, 0.5)
    ]

    with pytest.raises(RuntimeError, match="changed"):
        train_bc._checkpoint_dose_telemetry(
            metrics,
            optimizer_step=2,
            optimizer_observed_steps=2,
            optimizer_clipped_steps=0,
            optimizer_zero_objective_steps=0,
            optimizer_pre_clip_grad_norm_sum=1.0,
            optimizer_pre_clip_grad_norm_max=0.5,
            objective_gradient_cadence_batches=1,
            train_diagnostic_cadence_batches=1,
            public_card_enabled=False,
            meaningful_history_enabled=False,
        )


def test_checkpoint_dose_allows_zero_base_mass_but_requires_aux_mass() -> None:
    metric = {
        "samples": 1,
        "policy_base_active_rows": 0,
        "policy_aux_active_rows": 1,
        "policy_base_effective_weight_sum": 0.0,
        "policy_aux_effective_weight_sum": 1.0,
        "policy_aux_loss_coefficient": 0.25,
        "loss_denominators": {
            "policy_base_loss": 0.0,
            "policy_aux_loss": 1.0,
        },
    }

    dose = train_bc._checkpoint_dose_telemetry(
        [metric],
        optimizer_step=1,
        optimizer_observed_steps=1,
        optimizer_clipped_steps=0,
        optimizer_zero_objective_steps=0,
        optimizer_pre_clip_grad_norm_sum=0.5,
        optimizer_pre_clip_grad_norm_max=0.5,
        objective_gradient_cadence_batches=1,
        train_diagnostic_cadence_batches=1,
        public_card_enabled=False,
        meaningful_history_enabled=False,
    )

    assert dose["policy_stream_objective"]["base_denominator"] == 0.0
    assert dose["policy_stream_objective"]["aux_denominator"] == 1.0

    metric["policy_aux_effective_weight_sum"] = 0.0
    metric["loss_denominators"]["policy_aux_loss"] = 0.0
    with pytest.raises(RuntimeError, match="positive AUX objective mass"):
        train_bc._checkpoint_dose_telemetry(
            [metric],
            optimizer_step=1,
            optimizer_observed_steps=1,
            optimizer_clipped_steps=0,
            optimizer_zero_objective_steps=0,
            optimizer_pre_clip_grad_norm_sum=0.5,
            optimizer_pre_clip_grad_norm_max=0.5,
            objective_gradient_cadence_batches=1,
            train_diagnostic_cadence_batches=1,
            public_card_enabled=False,
            meaningful_history_enabled=False,
        )
