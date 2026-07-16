from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tools import train_bc


def test_policy_lr_area_hits_exact_boundary_and_then_stops() -> None:
    assert train_bc._policy_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=0.01,
        consumed_lr_area=0.0,
        target_lr_area=0.025,
    ) == pytest.approx(1.0)
    assert train_bc._policy_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=0.01,
        consumed_lr_area=0.02,
        target_lr_area=0.025,
    ) == pytest.approx(0.5)
    assert train_bc._policy_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=0.01,
        consumed_lr_area=0.025,
        target_lr_area=0.025,
    ) == 0.0


def test_policy_lr_area_accounts_for_independent_aux_objective() -> None:
    assert train_bc._policy_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=0.01,
        consumed_lr_area=0.0,
        target_lr_area=0.01,
        objective_multiplier=2.0,
    ) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("base_mass", "aux_mass", "expected_weight"),
    [
        (1.0, 0.0, 0.25),
        (0.0, 1.0, 1.0),
    ],
)
def test_policy_lr_area_boundary_uses_realized_streams(
    base_mass: float,
    aux_mass: float,
    expected_weight: float,
) -> None:
    assert train_bc._policy_microbatch_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=0.01,
        consumed_lr_area=0.0,
        target_lr_area=0.0025,
        pending_group_lr_area_weight=0.0,
        globally_base_objective_mass=base_mass,
        globally_aux_objective_mass=aux_mass,
        policy_aux_loss_weight=0.25,
        accumulation_group_size=1,
    ) == pytest.approx(expected_weight)


def test_policy_lr_area_boundary_accounts_for_pending_accumulation_dose() -> None:
    # The first microbatch has already contributed 0.5 full-dose equivalents
    # to this two-microbatch optimizer group. The boundary coefficient for the
    # second base-policy microbatch must spend only the remaining 0.001 LR-area.
    assert train_bc._policy_microbatch_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=0.01,
        consumed_lr_area=0.0,
        target_lr_area=0.006,
        pending_group_lr_area_weight=0.5,
        globally_base_objective_mass=1.0,
        globally_aux_objective_mass=0.0,
        policy_aux_loss_weight=1.0,
        accumulation_group_size=2,
    ) == pytest.approx(0.2)


def test_sparse_fixed_denominator_policy_batch_has_exact_lr_area_ledger() -> None:
    base_mass, aux_mass = train_bc._global_policy_stream_objective_masses(  # noqa: SLF001
        local_base_effective_weight_sum=1.0,
        local_base_fixed_denominator=4.0,
        local_aux_active_rows=0,
        ddp={"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
    )
    coefficient = train_bc._policy_microbatch_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=0.01,
        consumed_lr_area=0.0,
        target_lr_area=0.0025,
        pending_group_lr_area_weight=0.0,
        globally_base_objective_mass=base_mass,
        globally_aux_objective_mass=aux_mass,
        policy_aux_loss_weight=0.25,
        accumulation_group_size=1,
    )
    lr_area_weight, objective_fraction = (
        train_bc._realized_policy_microbatch_dose(  # noqa: SLF001
            policy_loss_weight=coefficient,
            policy_objective_fraction=coefficient,
            globally_base_objective_mass=base_mass,
            globally_aux_objective_mass=aux_mass,
            policy_aux_loss_weight=0.25,
            accumulation_group_size=1,
        )
    )

    assert base_mass == pytest.approx(0.25)
    assert coefficient == pytest.approx(1.0)
    assert lr_area_weight == pytest.approx(0.25)
    assert objective_fraction == pytest.approx(0.25)
    assert 0.01 * lr_area_weight == pytest.approx(0.0025)


def test_policy_dose_base_weights_follow_synchronous_global_batch_indices() -> None:
    corpus_weights = np.asarray([0.0, 2.0, 0.0, 5.0], dtype=np.float32)
    batch = np.asarray([3, 1], dtype=np.int64)

    selected = train_bc._base_policy_weights_for_training_batch(  # noqa: SLF001
        corpus_weights,
        batch,
    )

    np.testing.assert_array_equal(selected, np.asarray([5.0, 2.0]))


def test_policy_dose_base_weights_exclude_prefetched_auxiliary_tail() -> None:
    materialized_base_and_aux_weights = np.asarray(
        [2.0, 0.0, 17.0, 19.0],
        dtype=np.float32,
    )
    local_base_batch = np.asarray([0, 1], dtype=np.int64)

    selected = train_bc._base_policy_weights_for_training_batch(  # noqa: SLF001
        materialized_base_and_aux_weights,
        local_base_batch,
    )

    np.testing.assert_array_equal(selected, np.asarray([2.0, 0.0]))


def test_zero_policy_dose_preserves_historical_constant_weight() -> None:
    assert train_bc._policy_weight_for_lr_area(  # noqa: SLF001
        0.75,
        scheduled_base_lr=0.0,
        consumed_lr_area=0.0,
        target_lr_area=0.0,
    ) == pytest.approx(0.75)


def test_post_policy_value_routing_is_dormant_until_positive_dose_exhausts() -> None:
    dormant = train_bc._post_policy_dose_value_trunk_routing(  # noqa: SLF001
        base_scale=0.25,
        post_scale=0.0,
        target_lr_area=0.0,
        realized_policy_loss_weight=0.0,
    )
    active = train_bc._post_policy_dose_value_trunk_routing(  # noqa: SLF001
        base_scale=0.25,
        post_scale=0.0,
        target_lr_area=0.01,
        realized_policy_loss_weight=1.0,
    )
    exhausted = train_bc._post_policy_dose_value_trunk_routing(  # noqa: SLF001
        base_scale=0.25,
        post_scale=0.0,
        target_lr_area=0.01,
        realized_policy_loss_weight=0.0,
    )

    assert dormant["phase"] == "dormant_no_positive_policy_dose"
    assert dormant["effective_value_trunk_grad_scale"] == pytest.approx(0.25)
    assert active["effective_value_trunk_grad_scale"] == pytest.approx(0.25)
    assert exhausted["phase"] == "post_policy_dose"
    assert exhausted["effective_value_trunk_grad_scale"] == 0.0
    assert exhausted["shared_policy_representation_frozen"] is True


def test_post_policy_value_routing_rejects_invalid_scale() -> None:
    with pytest.raises(SystemExit, match="post-policy-dose.*\\[0, 1\\]"):
        train_bc._post_policy_dose_value_trunk_routing(  # noqa: SLF001
            base_scale=0.25,
            post_scale=1.1,
            target_lr_area=0.01,
            realized_policy_loss_weight=0.0,
        )


def test_policy_objective_fraction_preserves_fractional_boundary() -> None:
    assert train_bc._policy_objective_fraction(  # noqa: SLF001
        0.25, 1.0
    ) == pytest.approx(0.25)
    assert train_bc._policy_objective_fraction(  # noqa: SLF001
        0.0, 1.0
    ) == 0.0
    with pytest.raises(ValueError, match="exceeds"):
        train_bc._policy_objective_fraction(1.01, 1.0)  # noqa: SLF001


@pytest.mark.parametrize(
    (
        "base_presence",
        "aux_presence",
        "aux_weight",
        "group_size",
        "expected_weight",
        "expected_fraction",
    ),
    [
        ([True, False, False, False], [False] * 4, 1.0, 4, 0.25, 0.25),
        ([True] * 4, [False] * 4, 1.0, 4, 1.0, 1.0),
        ([False] * 2, [True] * 2, 0.25, 2, 0.25, 0.25),
        ([True] * 2, [True] * 2, 1.0, 2, 2.0, 2.0),
        ([False] * 4, [False] * 4, 1.0, 4, 0.0, 0.0),
    ],
)
def test_policy_group_dose_follows_realized_active_microbatches(
    base_presence,
    aux_presence,
    aux_weight,
    group_size,
    expected_weight,
    expected_fraction,
) -> None:
    weight = 0.0
    fraction = 0.0
    for base_active, aux_active in zip(
        base_presence, aux_presence, strict=True
    ):
        micro_weight, micro_fraction = (
            train_bc._realized_policy_microbatch_dose(  # noqa: SLF001
                policy_loss_weight=1.0,
                policy_objective_fraction=1.0,
                globally_base_objective_mass=float(base_active),
                globally_aux_objective_mass=float(aux_active),
                policy_aux_loss_weight=aux_weight,
                accumulation_group_size=group_size,
            )
        )
        weight += micro_weight
        fraction += micro_fraction

    assert weight == pytest.approx(expected_weight)
    assert fraction == pytest.approx(expected_fraction)


def test_global_policy_presence_uses_active_rows_not_fixed_denominator(
    monkeypatch,
) -> None:
    pytest.importorskip("torch")
    import torch.distributed as dist

    def remote_rank_has_policy(tensor, op):
        assert op == dist.ReduceOp.MAX
        tensor.fill_(1)

    monkeypatch.setattr(dist, "all_reduce", remote_rank_has_policy)
    assert train_bc._global_policy_objective_presence(  # noqa: SLF001
        local_base_active_rows=0,
        local_aux_active_rows=0,
        ddp={"enabled": True, "world_size": 2, "rank": 0, "local_rank": 0},
    )
    # A positive coverage/fixed denominator is deliberately absent from this
    # API: zero realized rows must never spend policy dose.
    assert not train_bc._global_policy_objective_presence(  # noqa: SLF001
        local_base_active_rows=0,
        local_aux_active_rows=0,
        ddp={"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
    )


def test_policy_stream_presence_keeps_base_and_aux_separate() -> None:
    assert train_bc._global_policy_stream_presence(  # noqa: SLF001
        local_base_active_rows=0,
        local_aux_active_rows=7,
        ddp={"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
    ) == (False, True)


def test_value_only_group_does_not_trigger_post_policy_freeze() -> None:
    consumed = 0.0
    weight, _fraction = train_bc._realized_policy_microbatch_dose(  # noqa: SLF001
        policy_loss_weight=1.0,
        policy_objective_fraction=1.0,
        globally_base_objective_mass=0.0,
        globally_aux_objective_mass=0.0,
        policy_aux_loss_weight=1.0,
        accumulation_group_size=1,
    )
    consumed += 0.01 * weight
    still_open = train_bc._policy_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=0.01,
        consumed_lr_area=consumed,
        target_lr_area=0.01,
    )
    routing = train_bc._post_policy_dose_value_trunk_routing(  # noqa: SLF001
        base_scale=0.25,
        post_scale=0.0,
        target_lr_area=0.01,
        realized_policy_loss_weight=still_open,
    )

    assert consumed == 0.0
    assert still_open == pytest.approx(1.0)
    assert routing["phase"] == "pre_or_boundary_policy_dose"
    assert routing["shared_policy_representation_frozen"] is False


def test_policy_dose_requires_matching_global_batch_topology() -> None:
    assert train_bc._validate_policy_dose_topology(  # noqa: SLF001
        target_lr_area=0.01,
        reference_global_batch_size=512,
        local_batch_size=64,
        grad_accum_steps=1,
        world_size=8,
    ) == 512
    with pytest.raises(SystemExit, match="cannot cross optimizer topology"):
        train_bc._validate_policy_dose_topology(  # noqa: SLF001
            target_lr_area=0.01,
            reference_global_batch_size=4096,
            local_batch_size=64,
            grad_accum_steps=1,
            world_size=8,
        )


def test_positive_policy_dose_requires_explicit_reference_topology() -> None:
    with pytest.raises(SystemExit, match="requires.*reference-global-batch-size"):
        train_bc._validate_policy_dose_topology(  # noqa: SLF001
            target_lr_area=0.01,
            reference_global_batch_size=0,
            local_batch_size=64,
            grad_accum_steps=1,
            world_size=8,
        )


def test_uncapped_training_can_request_an_early_checkpoint_frontier() -> None:
    assert train_bc._parse_checkpoint_steps(  # noqa: SLF001
        "8,16,32,64,128",
        max_steps=0,
    ) == (8, 16, 32, 64, 128)


def test_policy_only_gradient_suppression_keeps_shared_value_paths() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_bias = torch.nn.Linear(3, 1)
            self.edge_policy_mlp = torch.nn.Linear(3, 1)
            self.logit_scale = torch.nn.Parameter(torch.ones(()))
            self.state_norm = torch.nn.LayerNorm(3)
            self.value_head = torch.nn.Linear(3, 1)
            self.value_tower_split_layers = 1

    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    for parameter in model.action_bias.parameters():
        optimizer.state[parameter]["stale_momentum"] = torch.ones_like(parameter)
    optimizer.state[model.logit_scale]["stale_momentum"] = torch.ones_like(
        model.logit_scale
    )
    suppressed = train_bc._suppress_inactive_policy_only_gradients(  # noqa: SLF001
        SimpleNamespace(model=model),
        optimizer,
    )

    assert "logit_scale" in suppressed
    assert all(parameter.grad is None for parameter in model.action_bias.parameters())
    assert all(
        parameter.grad is None for parameter in model.edge_policy_mlp.parameters()
    )
    assert all(parameter.grad is None for parameter in model.state_norm.parameters())
    assert all(parameter.grad is not None for parameter in model.value_head.parameters())
    assert all(parameter not in optimizer.state for parameter in model.action_bias.parameters())
    assert model.logit_scale not in optimizer.state


def test_post_policy_freeze_clears_shared_adamw_state_but_keeps_value_tower() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hex_encoder = torch.nn.Linear(3, 3)
            self.action_encoder = torch.nn.Linear(3, 3)
            self.action_bias = torch.nn.Linear(3, 1)
            self.value_blocks = torch.nn.ModuleList([torch.nn.Linear(3, 3)])
            self.value_head = torch.nn.Linear(3, 1)

    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.01)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    shared_before = {
        name: parameter.detach().clone()
        for name, parameter in model.hex_encoder.named_parameters()
    }
    value_before = {
        name: parameter.detach().clone()
        for name, parameter in model.value_head.named_parameters()
    }

    suppressed = train_bc._suppress_shared_policy_representation_gradients(  # noqa: SLF001
        SimpleNamespace(model=model),
        optimizer,
    )
    optimizer.step()

    assert any(name.startswith("hex_encoder.") for name in suppressed)
    assert all(
        parameter.grad is None for parameter in model.hex_encoder.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.action_encoder.parameters()
    )
    assert all(
        parameter.grad is not None for parameter in model.value_blocks.parameters()
    )
    assert all(
        parameter.grad is not None for parameter in model.value_head.parameters()
    )
    assert all(
        parameter not in optimizer.state
        for parameter in model.hex_encoder.parameters()
    )
    assert all(
        parameter in optimizer.state for parameter in model.value_head.parameters()
    )
    assert all(
        torch.equal(parameter, shared_before[name])
        for name, parameter in model.hex_encoder.named_parameters()
    )
    assert any(
        not torch.equal(parameter, value_before[name])
        for name, parameter in model.value_head.named_parameters()
    )


def test_train_config_hash_binds_policy_dose_and_post_value_route() -> None:
    from catan_zero.rl.pipeline_configs import TrainConfig

    baseline = TrainConfig()
    treatment = replace(
        baseline,
        policy_dose_lr_area=0.01,
        policy_dose_reference_global_batch_size=512,
        post_policy_dose_value_trunk_grad_scale=0.0,
    )
    assert baseline.full_config_hash() != treatment.full_config_hash()


def test_policy_signal_attestation_uses_scheduled_objective_mass() -> None:
    report = train_bc._policy_training_signal_attestation(  # noqa: SLF001
        [
            {
                "samples": 100,
                "policy_base_active_rows": 40,
                "policy_aux_active_rows": 0,
                "policy_objective_active_rows": 10,
                "policy_objective_equivalent_active_rows": 7.5,
                "policy_objective_effective_weight_sum": 7.5,
                "policy_objective_equivalent_effective_weight_sum": 7.5,
                "policy_objective_optimizer_updates": 4,
                "policy_objective_equivalent_optimizer_updates": 3.5,
                "loss_denominators": {"policy_loss": 40.0},
            }
        ],
        policy_loss_weight=1.0,
        optimizer_steps=4,
        train_value_only=False,
    )

    assert report["policy_active_rows"] == 10
    assert report["policy_equivalent_active_rows"] == pytest.approx(7.5)
    assert report["policy_effective_weight_sum"] == pytest.approx(7.5)
    assert report["policy_optimizer_updates"] == 4
    assert report["policy_equivalent_optimizer_updates"] == pytest.approx(3.5)
    assert report["trained_policy_objective"] is True


def test_policy_signal_attestation_recovers_full_dose_equivalent_weight() -> None:
    report = train_bc._policy_training_signal_attestation(  # noqa: SLF001
        [
            {
                "samples": 8,
                "policy_base_active_rows": 8,
                "policy_objective_active_rows": 8,
                "policy_objective_equivalent_active_rows": 4.0,
                "policy_objective_effective_weight_sum": 2.0,
                "policy_objective_optimizer_updates": 1,
                "policy_objective_equivalent_optimizer_updates": 0.5,
            }
        ],
        policy_loss_weight=0.5,
        optimizer_steps=1,
        train_value_only=False,
    )

    assert report["policy_effective_weight_sum"] == pytest.approx(2.0)
    assert report["policy_equivalent_effective_weight_sum"] == pytest.approx(
        4.0
    )


def test_fractional_policy_strata_report_full_dose_equivalent_rows() -> None:
    data = {
        "legal_action_ids": np.asarray([[1, 2], [1, 2]], dtype=np.int16),
        "phase": np.asarray(["opening", "main"]),
    }
    dose = train_bc._training_strata_dose_for_batch(  # noqa: SLF001
        data,
        np.arange(2, dtype=np.int64),
        policy_weights=np.ones(2, dtype=np.float32),
        value_weights=np.ones(2, dtype=np.float32),
        value_active_mask=np.ones(2, dtype=np.bool_),
        policy_objective_fraction=0.25,
    )
    report = train_bc._nest_training_strata_dose(  # noqa: SLF001
        train_bc._flatten_training_strata_dose(dose)  # noqa: SLF001
    )

    assert report["policy_active_row_draws"] == 2
    assert report["policy_objective_active_row_draws"] == 2
    assert report["policy_objective_equivalent_row_draws"] == pytest.approx(
        0.5
    )
    assert report["dimensions"]["phase"]["opening"][
        "policy_objective_equivalent_rows"
    ] == pytest.approx(0.25)


def test_component_dose_counts_only_rows_passed_to_objective() -> None:
    class Composite(dict):
        component_ids = ("fresh", "replay")

        @staticmethod
        def component_indices_for_rows(rows):
            return np.asarray(rows, dtype=np.int64) % 2

    data = Composite(
        phase=np.asarray(["opening", "opening", "main", "main"])
    )
    component, phase = train_bc._policy_component_dose_for_batch(  # noqa: SLF001
        data,
        np.asarray([0, 3], dtype=np.int64),
        suffix="base",
        phase_names=("opening", "main"),
    )

    assert component == {"fresh.base": 1.0, "replay.base": 1.0}
    assert phase["fresh\0opening\0base"] == 1.0
    assert phase["replay\0main\0base"] == 1.0
