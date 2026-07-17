from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catan_zero.rl._catanatron import import_catanatron_module
from tools import train_bc
from tools.train_bc import (
    _bounded_count_fraction,
    ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS,
    ENTITY_GRAPH_VALUE_ONLY_FREEZE_GROUPS,
    _apply_lr_warmup,
    _lr_warmup_multiplier,
    _resolve_entity_graph_freeze_groups,
    _set_entity_graph_modules_trainable,
    _train_entity_batch,
    load_teacher_data,
)


def test_explicit_validation_range_help_requires_whole_game_holdout() -> None:
    action = train_bc.build_parser()._option_string_actions[  # noqa: SLF001
        "--validation-game-seed-ranges"
    ]
    assert "--validation-max-samples 0" in str(action.help)
    assert "still applies" not in str(action.help)


def test_bounded_count_fraction_rejects_mixed_ddp_scopes() -> None:
    assert _bounded_count_fraction(5, 10, label="coverage") == pytest.approx(0.5)
    assert _bounded_count_fraction(0, 0, label="coverage") == 0.0
    with pytest.raises(RuntimeError, match="incompatible count scopes"):
        _bounded_count_fraction(80, 10, label="coverage")


# --------------------------------------------------------------------------- lr warmup


def test_lr_warmup_multiplier_ramps_linearly_then_holds() -> None:
    assert _lr_warmup_multiplier(0, 10) == pytest.approx(0.1)
    assert _lr_warmup_multiplier(4, 10) == pytest.approx(0.5)
    assert _lr_warmup_multiplier(9, 10) == pytest.approx(1.0)
    assert _lr_warmup_multiplier(20, 10) == pytest.approx(1.0)


def test_lr_warmup_multiplier_disabled_when_warmup_steps_not_positive() -> None:
    assert _lr_warmup_multiplier(0, 0) == 1.0
    assert _lr_warmup_multiplier(0, -5) == 1.0


def test_apply_lr_warmup_sets_every_param_group() -> None:
    import torch

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

    multiplier = _apply_lr_warmup(optimizer, base_lr=2e-4, step=0, warmup_steps=4)

    assert multiplier == pytest.approx(0.25)
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(2e-4 * 0.25)

    multiplier = _apply_lr_warmup(optimizer, base_lr=2e-4, step=10, warmup_steps=4)
    assert multiplier == pytest.approx(1.0)
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(2e-4)


# --------------------------------------------------------------------------- freeze utility


def _make_entity_policy(**overrides):
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    return EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
        **overrides,
    )


def test_freeze_module_groups_cover_expected_submodules() -> None:
    assert set(ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS) == {
        "trunk",
        "action_encoder",
        "policy_head",
        "value_heads",
        "target_gather",
        "edge_policy",
        "action_cross",
        "static_action_residual",
        "public_card_residual",
        "meaningful_history_gate",
        "v7_resource_residual",
        "v7_initial_road_residual",
    }
    assert "value_head" in ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["value_heads"]
    assert "public_rule_state_residual" in ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["trunk"]
    assert (
        "value_categorical_head" in ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["value_heads"]
    )
    assert "final_vp_head" in ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["value_heads"]
    assert ENTITY_GRAPH_VALUE_ONLY_FREEZE_GROUPS == {
        "trunk",
        "action_encoder",
        "policy_head",
        "target_gather",
        "edge_policy",
        "action_cross",
        "static_action_residual",
        "v7_initial_road_residual",
    }


def test_set_entity_graph_modules_trainable_freezes_and_restores() -> None:
    policy = _make_entity_policy()

    touched = _set_entity_graph_modules_trainable(
        policy.model, ["trunk", "action_encoder", "policy_head"], trainable=False
    )
    assert set(touched) == {
        "hex_encoder",
        "vertex_encoder",
        "edge_encoder",
        "player_encoder",
        "global_encoder",
        "event_encoder",
        "type_embedding",
        "cls_token",
        "blocks",
        "state_norm",
        "action_encoder",
        "action_bias",
        "logit_scale",
    }
    for name in touched:
        attr = getattr(policy.model, name)
        params = attr.parameters() if hasattr(attr, "parameters") else (attr,)
        assert all(not p.requires_grad for p in params)
    # Value heads remain trainable unless their dedicated opt-in group is named.
    assert all(p.requires_grad for p in policy.model.value_head.parameters())
    assert all(p.requires_grad for p in policy.model.final_vp_head.parameters())

    _set_entity_graph_modules_trainable(policy.model, ["trunk"], trainable=True)
    assert all(p.requires_grad for p in policy.model.hex_encoder.parameters())
    # action_encoder/policy_head were not re-enabled -- still frozen.
    assert all(not p.requires_grad for p in policy.model.action_encoder.parameters())


def test_value_only_freezes_complete_v7_policy_surface() -> None:
    from catan_zero.rl.entity_feature_adapter import RUST_ENTITY_ADAPTER_V6

    policy = _make_entity_policy(
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
        v6_compatibility_preserving_inputs=True,
        action_cross_attention_layers=1,
    )
    _set_entity_graph_modules_trainable(
        policy.model, ENTITY_GRAPH_VALUE_ONLY_FREEZE_GROUPS, trainable=False
    )
    for module_name in (
        "v6_exact_resource_residual",
        "v6_initial_road_residual",
    ):
        assert all(
            not parameter.requires_grad
            for parameter in getattr(policy.model, module_name).parameters()
        )


def test_freeze_groups_are_explicit_not_data_driven() -> None:
    assert _resolve_entity_graph_freeze_groups(
        freeze_modules="", train_value_only=False
    ) == set()
    assert _resolve_entity_graph_freeze_groups(
        freeze_modules="public_card_residual,meaningful_history_gate",
        train_value_only=False,
    ) == {"public_card_residual", "meaningful_history_gate"}
    assert _resolve_entity_graph_freeze_groups(
        freeze_modules="", train_value_only=True
    ) == set(ENTITY_GRAPH_VALUE_ONLY_FREEZE_GROUPS)


def test_value_heads_group_freezes_scalar_categorical_and_auxiliary_readouts() -> None:
    from dataclasses import replace

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    policy = _make_entity_policy()
    policy = EntityGraphPolicy(
        replace(
            policy.config,
            value_categorical_bins=9,
            value_uncertainty_head=True,
            value_attention_pool=True,
        ),
        policy.static_action_features.detach().cpu().numpy(),
        device="cpu",
    )

    touched = _set_entity_graph_modules_trainable(
        policy.model, ["value_heads"], trainable=False
    )

    assert set(touched) == {
        "value_head",
        "value_categorical_head",
        "final_vp_head",
        "value_uncertainty_head",
        "value_probe",
        "value_probe_norm_q",
        "value_probe_norm_kv",
        "value_probe_attn",
        "value_pool_head",
    }
    for name in touched:
        attr = getattr(policy.model, name)
        params = attr.parameters() if hasattr(attr, "parameters") else (attr,)
        assert all(not parameter.requires_grad for parameter in params)

    # The policy path remains trainable for an action-local warmup.
    assert all(
        parameter.requires_grad
        for parameter in policy.model.action_encoder.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in policy.model.blocks.parameters()
    )


def test_action_local_groups_freeze_independently() -> None:
    from dataclasses import replace

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    base = _make_entity_policy()
    policy = EntityGraphPolicy(
        replace(
            base.config,
            action_target_gather=True,
            edge_policy_head=True,
            action_cross_attention_layers=1,
        ),
        base.static_action_features.detach().cpu().numpy(),
        device="cpu",
    )

    touched = _set_entity_graph_modules_trainable(
        policy.model, ["target_gather"], trainable=False
    )
    assert touched == ["target_gather_proj"]
    assert all(
        not parameter.requires_grad
        for parameter in policy.model.target_gather_proj.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in policy.model.edge_policy_mlp.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in policy.model.action_cross_blocks.parameters()
    )

    touched = _set_entity_graph_modules_trainable(
        policy.model, ["edge_policy", "action_cross"], trainable=False
    )
    assert set(touched) == {"edge_policy_mlp", "action_cross_blocks"}
    assert all(
        not parameter.requires_grad
        for parameter in policy.model.edge_policy_mlp.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in policy.model.action_cross_blocks.parameters()
    )


def test_set_entity_graph_modules_trainable_rejects_unknown_group() -> None:
    policy = _make_entity_policy()
    with pytest.raises(SystemExit):
        _set_entity_graph_modules_trainable(
            policy.model, ["not_a_real_group"], trainable=False
        )


def test_value_only_smoke_freezes_complete_upgraded_policy_path() -> None:
    """A value-only repair must also freeze every optional policy adapter."""
    from dataclasses import replace

    import torch
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    base = _make_entity_policy()
    policy = EntityGraphPolicy(
        replace(
            base.config,
            action_target_gather=True,
            edge_policy_head=True,
            action_cross_attention_layers=1,
            static_action_residual=True,
            legal_action_value_residual=True,
        ),
        base.static_action_features.detach().cpu().numpy(),
        device="cpu",
    )
    _set_entity_graph_modules_trainable(
        policy.model, ENTITY_GRAPH_VALUE_ONLY_FREEZE_GROUPS, trainable=False
    )

    samples = _collect_real_samples(6)
    entity_batch = _pad_entity_batch(policy, samples)
    legal_action_ids, legal_action_context = _pad_legal_action_arrays(policy, samples)

    optimizer = torch.optim.AdamW(
        [p for p in policy.model.parameters() if p.requires_grad],
        lr=1e-2,
        weight_decay=0.1,
    )

    frozen_param = next(policy.model.hex_encoder.parameters())
    frozen_before = frozen_param.detach().clone()
    policy_adapter_params = {
        name: parameter
        for name, parameter in policy.model.named_parameters()
        if name.startswith(
            (
                "target_gather_proj.",
                "edge_policy_mlp.",
                "action_cross_blocks.",
                "static_action_residual_proj.",
            )
        )
    }
    assert policy_adapter_params
    assert all(not parameter.requires_grad for parameter in policy_adapter_params.values())
    assert all(
        parameter.requires_grad
        for parameter in policy.model.legal_action_value_static_proj.parameters()
    )
    policy_adapter_before = {
        name: parameter.detach().clone()
        for name, parameter in policy_adapter_params.items()
    }
    value_param = next(policy.model.value_head.parameters())
    value_before = value_param.detach().clone()

    value_targets = torch.as_tensor(
        np.random.default_rng(0).normal(size=len(samples)).astype(np.float32),
        device=policy.device,
    )

    for _ in range(10):
        outputs = policy.forward_legal_np(
            entity_batch, legal_action_ids, legal_action_context, return_q=False
        )
        # Match the production combined-objective graph: the policy loss can be
        # present with a zero coefficient. If a policy adapter were still in
        # the optimizer, that exact-zero gradient would permit AdamW decay.
        value_loss = torch.nn.functional.mse_loss(
            outputs["value"], value_targets
        ) + 0.0 * outputs["logits"].sum()
        optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        assert frozen_param.grad is None
        assert all(
            parameter.grad is None
            for parameter in policy_adapter_params.values()
        )
        assert value_param.grad is not None
        assert bool((value_param.grad.abs().sum() > 0).item())
        optimizer.step()

    torch.testing.assert_close(frozen_param.detach(), frozen_before)
    for name, parameter in policy_adapter_params.items():
        torch.testing.assert_close(parameter.detach(), policy_adapter_before[name])
    assert not torch.allclose(value_param.detach(), value_before)


# --------------------------------------------------------------------------- policy-loss-weight
# end-to-end (via a real, on-disk DAgger-format shard, matching the production data path).


def _collect_real_samples(n: int):
    import_catanatron_module("catanatron")
    from catan_zero.rl.action_features import build_action_context_feature_table
    from catan_zero.rl.entity_token_features import build_entity_token_features
    from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
    from catan_zero.rl.self_play import StepSample, _phase_from_info, make_env_config

    config = make_env_config(vps_to_win=3)
    env = ColonistMultiAgentEnv(config)
    samples = []
    try:
        observations, info = env.reset(seed=9)
        for decision_index in range(n):
            player = str(info["current_player"])
            observation = np.asarray(observations[player], dtype=np.float64)
            valid_actions = tuple(int(a) for a in info["valid_actions"])
            entity_features = {
                key: value
                for key, value in build_entity_token_features(env, player).items()
                if key != "schema"
            }
            samples.append(
                StepSample(
                    observation=observation.copy(),
                    valid_actions=valid_actions,
                    action=int(valid_actions[0]),
                    player=player,
                    action_context_features=build_action_context_feature_table(
                        env, info
                    ),
                    entity_features=entity_features,
                    phase=_phase_from_info(info),
                    decision_index=decision_index,
                )
            )
            observations, _rewards, terminated, truncated, info = env.step(
                int(valid_actions[0])
            )
            if terminated or truncated:
                observations, info = env.reset(seed=9 + decision_index + 1)
    finally:
        env.close()
    return samples


def _pad_entity_batch(policy, samples):
    from catan_zero.rl.torch_ppo import _entity_graph_batch

    batch, _legal_ids, _legal_ctx = _entity_graph_batch(samples, policy)
    return batch


def _pad_legal_action_arrays(policy, samples):
    max_legal = max(len(s.valid_actions) for s in samples)
    context_size = int(policy.context_action_feature_size)
    legal_action_ids = np.full((len(samples), max_legal), -1, dtype=np.int64)
    legal_action_context = np.zeros(
        (len(samples), max_legal, context_size), dtype=np.float32
    )
    for row, sample in enumerate(samples):
        n = len(sample.valid_actions)
        legal_action_ids[row, :n] = np.asarray(sample.valid_actions, dtype=np.int64)
        ctx = np.asarray(sample.action_context_features, dtype=np.float32)[
            list(sample.valid_actions), :
        ]
        legal_action_context[row, :n, : ctx.shape[1]] = ctx[:, :context_size]
    return legal_action_ids, legal_action_context


def _write_and_load_shard(tmp_path: Path, samples):
    from tools.generate_dagger_data import DaggerEntityShardWriter, _row_from_sample

    out = tmp_path / "shard"
    out.mkdir()
    writer = DaggerEntityShardWriter(out, 1000, "npz")
    for sample in samples:
        row = _row_from_sample(
            sample,
            teacher="test_teacher",
            entity_features=sample.entity_features,
            game_seed=0,
            winner="BLUE",
            terminated=True,
            truncated=False,
            final_public_vps={"BLUE": 10, "RED": 4, "WHITE": 0, "ORANGE": 0},
            final_actual_vps={"BLUE": 10, "RED": 4, "WHITE": 0, "ORANGE": 0},
            policy_weight_multiplier=1.0,
            value_weight_multiplier=1.0,
        )
        row["has_final_public_vps"] = True
        row["has_final_actual_vps"] = True
        writer.add_row(row)
    writer.close()
    return load_teacher_data(out)


def test_policy_loss_weight_scales_the_policy_term_in_train_entity_batch(
    tmp_path,
) -> None:
    import torch

    samples = _collect_real_samples(6)
    data = _write_and_load_shard(tmp_path, samples)
    n = len(data["action_taken"])
    batch = np.arange(n)
    policy_weights = np.ones(n, dtype=np.float32)
    value_weights = np.ones(n, dtype=np.float32)

    def run(policy_loss_weight: float, value_loss_weight: float):
        policy = _make_entity_policy()
        optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-3)
        return _train_entity_batch(
            policy,
            optimizer,
            data,
            batch,
            policy_weights,
            value_weights,
            soft_target_temperature=1.0,
            soft_target_weight=0.0,
            soft_target_source="scores",
            soft_target_min_legal_coverage=0.0,
            policy_loss_weight=policy_loss_weight,
            value_loss_weight=value_loss_weight,
            final_vp_loss_weight=0.0,
            q_loss_weight=0.0,
            q_skip_teacher_prefixes=(),
            vps_to_win=10,
            advantage_policy_weighting="none",
            advantage_temperature=1.0,
            advantage_weight_cap=5.0,
            advantage_weight_floor=0.05,
            amp="none",
            diagnostics=False,
        )

    value_only_metrics = run(policy_loss_weight=0.0, value_loss_weight=1.0)
    policy_only_metrics = run(policy_loss_weight=1.0, value_loss_weight=0.0)

    # policy_loss/value_loss (the raw, UNweighted components) are always reported regardless
    # of the scalar weights -- only the combined "loss" used for backprop changes.
    assert value_only_metrics["policy_loss"] > 0.0
    assert value_only_metrics["loss"] == pytest.approx(
        value_only_metrics["value_loss"], rel=1e-4
    )
    assert policy_only_metrics["loss"] == pytest.approx(
        policy_only_metrics["policy_loss"], rel=1e-4
    )


def test_train_diagnostics_do_not_implicitly_run_two_extra_gradient_traversals(
    tmp_path, monkeypatch
) -> None:
    import torch

    data = _write_and_load_shard(tmp_path, _collect_real_samples(3))
    batch = np.arange(len(data["action_taken"]))
    weights = np.ones(len(batch), dtype=np.float32)
    calls = 0
    aux_objectives = []

    def interference(*_args, **kwargs):
        nonlocal calls
        calls += 1
        aux_objectives.append(kwargs.get("policy_aux_objective"))
        return {"available": True, "sentinel": True}

    monkeypatch.setattr(train_bc, "_objective_gradient_interference", interference)

    def run(
        *, measure: bool, auxiliary: bool = False, diagnostics: bool = True
    ) -> dict:
        policy = _make_entity_policy()
        optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
        return _train_entity_batch(
            policy,
            optimizer,
            data,
            batch,
            weights,
            weights,
            soft_target_temperature=1.0,
            soft_target_weight=0.0,
            soft_target_source="scores",
            soft_target_min_legal_coverage=0.0,
            policy_loss_weight=1.0,
            value_loss_weight=1.0,
            final_vp_loss_weight=0.0,
            q_loss_weight=0.0,
            q_skip_teacher_prefixes=(),
            vps_to_win=10,
            advantage_policy_weighting="none",
            advantage_temperature=1.0,
            advantage_weight_cap=5.0,
            advantage_weight_floor=0.05,
            amp="none",
            diagnostics=diagnostics,
            measure_objective_gradient_interference=measure,
            **(
                {
                    "policy_aux_data": data,
                    "policy_aux_batch": batch,
                    "policy_aux_sample_weights": weights,
                }
                if auxiliary
                else {}
            ),
        )

    ordinary = run(measure=False)
    assert calls == 0
    assert ordinary["optimizer_observability"][
        "objective_gradient_interference"
    ] is None

    explicit = run(measure=True)
    assert calls == 1
    assert explicit["optimizer_observability"][
        "objective_gradient_interference"
    ] == {"available": True, "sentinel": True}
    explicit_without_module_diagnostics = run(measure=True, diagnostics=False)
    assert calls == 2
    observability = explicit_without_module_diagnostics["optimizer_observability"]
    assert "module_pre_clip_grad_norms" not in observability
    assert observability["objective_gradient_interference"] == {
        "available": True,
        "sentinel": True,
    }
    explicit_aux = run(measure=True, auxiliary=True)
    assert calls == 3
    assert aux_objectives[0] is None
    assert aux_objectives[2] is not None
    assert bool(aux_objectives[2].requires_grad)
    assert explicit_aux["optimizer_observability"][
        "objective_gradient_interference"
    ] == {"available": True, "sentinel": True}


def test_zero_weight_final_vp_head_is_not_executed_by_entity_training(tmp_path) -> None:
    import torch

    data = _write_and_load_shard(tmp_path, _collect_real_samples(3))
    batch = np.arange(len(data["action_taken"]))
    weights = np.ones(len(batch), dtype=np.float32)
    policy = _make_entity_policy()
    calls = 0

    def count_call(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    hook = policy.model.final_vp_head.register_forward_hook(count_call)
    try:
        optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
        metrics = _train_entity_batch(
            policy,
            optimizer,
            data,
            batch,
            weights,
            weights,
            soft_target_temperature=1.0,
            soft_target_weight=0.0,
            soft_target_source="scores",
            soft_target_min_legal_coverage=0.0,
            policy_loss_weight=1.0,
            value_loss_weight=1.0,
            final_vp_loss_weight=0.0,
            q_loss_weight=0.0,
            q_skip_teacher_prefixes=(),
            vps_to_win=10,
            advantage_policy_weighting="none",
            advantage_temperature=1.0,
            advantage_weight_cap=5.0,
            advantage_weight_floor=0.05,
            amp="none",
            diagnostics=False,
        )
    finally:
        hook.remove()

    assert calls == 0
    assert metrics["final_vp_loss"] == 0.0
    assert metrics["final_vp_loss_weight_sum"] == 0.0


def test_all_zero_objective_mass_does_not_advance_adamw_or_decay_parameters(
    tmp_path,
) -> None:
    import copy
    import torch

    data = _write_and_load_shard(tmp_path, _collect_real_samples(3))
    batch = np.arange(len(data["action_taken"]))
    zero = np.zeros(len(batch), dtype=np.float32)
    policy = _make_entity_policy()
    before = copy.deepcopy(policy.model.state_dict())
    optimizer = torch.optim.AdamW(
        policy.model.parameters(), lr=1e-3, weight_decay=0.1
    )

    metrics = _train_entity_batch(
        policy,
        optimizer,
        data,
        batch,
        zero,
        zero,
        soft_target_temperature=1.0,
        soft_target_weight=0.0,
        soft_target_source="scores",
        soft_target_min_legal_coverage=0.0,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        final_vp_loss_weight=0.0,
        q_loss_weight=0.0,
        q_skip_teacher_prefixes=(),
        vps_to_win=10,
        advantage_policy_weighting="none",
        advantage_temperature=1.0,
        advantage_weight_cap=5.0,
        advantage_weight_floor=0.05,
        amp="none",
        diagnostics=False,
    )

    assert metrics["optimizer_step_applied"] is False
    assert (
        metrics["optimizer_observability"]["zero_objective_step_skipped"] is True
    )
    assert optimizer.state == {}
    after = policy.model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_zero_objective_last_microbatch_applies_accumulated_nonzero_gradient(
    tmp_path,
) -> None:
    import copy
    import torch

    data = _write_and_load_shard(tmp_path, _collect_real_samples(3))
    batch = np.arange(len(data["action_taken"]))
    active = np.ones(len(batch), dtype=np.float32)
    zero = np.zeros(len(batch), dtype=np.float32)
    policy = _make_entity_policy()
    before = copy.deepcopy(policy.model.state_dict())
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=1e-3)
    common = {
        "soft_target_temperature": 1.0,
        "soft_target_weight": 0.0,
        "soft_target_source": "scores",
        "soft_target_min_legal_coverage": 0.0,
        "policy_loss_weight": 1.0,
        "value_loss_weight": 1.0,
        "final_vp_loss_weight": 0.0,
        "q_loss_weight": 0.0,
        "q_skip_teacher_prefixes": (),
        "vps_to_win": 10,
        "advantage_policy_weighting": "none",
        "advantage_temperature": 1.0,
        "advantage_weight_cap": 5.0,
        "advantage_weight_floor": 0.05,
        "amp": "none",
        "diagnostics": False,
        "grad_accum_steps": 2,
    }

    first = _train_entity_batch(
        policy,
        optimizer,
        data,
        batch,
        active,
        active,
        accum_do_zero_grad=True,
        accum_do_step=False,
        **common,
    )
    second = _train_entity_batch(
        policy,
        optimizer,
        data,
        batch,
        zero,
        zero,
        accum_do_zero_grad=False,
        accum_do_step=True,
        **common,
    )

    assert first["optimizer_step_applied"] is False
    assert second["loss"] == 0.0
    assert second["optimizer_step_applied"] is True
    assert (
        second["optimizer_observability"]["zero_objective_step_skipped"] is False
    )
    after = policy.model.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_policy_dose_boundary_preserves_pending_accumulated_policy_gradient(
    tmp_path,
) -> None:
    import torch

    data = _write_and_load_shard(tmp_path, _collect_real_samples(3))
    batch = np.arange(len(data["action_taken"]))
    active = np.ones(len(batch), dtype=np.float32)
    zero = np.zeros(len(batch), dtype=np.float32)
    policy = _make_entity_policy()
    parameter = policy.model.action_bias.weight
    before = parameter.detach().clone()
    learning_rate = 1.0e-3
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=learning_rate)
    common = {
        "soft_target_temperature": 1.0,
        "soft_target_weight": 0.0,
        "soft_target_source": "scores",
        "soft_target_min_legal_coverage": 0.0,
        "value_loss_weight": 0.0,
        "final_vp_loss_weight": 0.0,
        "q_loss_weight": 0.0,
        "q_skip_teacher_prefixes": (),
        "vps_to_win": 10,
        "advantage_policy_weighting": "none",
        "advantage_temperature": 1.0,
        "advantage_weight_cap": 5.0,
        "advantage_weight_floor": 0.05,
        "amp": "none",
        "diagnostics": False,
        "grad_accum_steps": 2,
    }

    first = _train_entity_batch(
        policy,
        optimizer,
        data,
        batch,
        active,
        zero,
        policy_loss_weight=1.0,
        accum_do_zero_grad=True,
        accum_do_step=False,
        **common,
    )
    pending_gradient = parameter.grad.detach().clone()
    assert bool((pending_gradient.abs().sum() > 0.0).item())
    pending_weight, _ = train_bc._realized_policy_microbatch_dose(  # noqa: SLF001
        policy_loss_weight=1.0,
        policy_objective_fraction=1.0,
        globally_base_objective_mass=1.0,
        globally_aux_objective_mass=0.0,
        policy_aux_loss_weight=1.0,
        accumulation_group_size=2,
    )
    target_lr_area = learning_rate * pending_weight
    boundary_weight = train_bc._policy_microbatch_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=learning_rate,
        consumed_lr_area=0.0,
        target_lr_area=target_lr_area,
        pending_group_lr_area_weight=pending_weight,
        globally_base_objective_mass=1.0,
        globally_aux_objective_mass=0.0,
        policy_aux_loss_weight=1.0,
        accumulation_group_size=2,
    )
    routing = train_bc._post_policy_dose_value_trunk_routing(  # noqa: SLF001
        base_scale=0.25,
        post_scale=0.0,
        target_lr_area=target_lr_area,
        realized_policy_loss_weight=boundary_weight,
        pending_policy_lr_area_weight=pending_weight,
    )
    second = _train_entity_batch(
        policy,
        optimizer,
        data,
        batch,
        zero,
        zero,
        policy_loss_weight=boundary_weight,
        value_trunk_grad_scale=float(
            routing["effective_value_trunk_grad_scale"]
        ),
        preserve_accumulated_policy_gradients=True,
        accum_do_zero_grad=False,
        accum_do_step=True,
        **common,
    )

    assert first["optimizer_step_applied"] is False
    assert boundary_weight == 0.0
    assert routing["phase"] == "pre_or_boundary_policy_dose"
    assert routing["shared_policy_representation_frozen"] is False
    assert second["optimizer_step_applied"] is True
    torch.testing.assert_close(parameter.grad, pending_gradient)
    torch.testing.assert_close(
        parameter.detach(),
        before - learning_rate * pending_gradient,
    )
    assert learning_rate * pending_weight == pytest.approx(target_lr_area)


def test_nonzero_loss_with_exact_zero_gradient_preserves_optimizer_semantics() -> None:
    from types import SimpleNamespace

    import torch
    from tools.train_bc import _step_optimizer_fail_closed

    model = torch.nn.Linear(2, 1, bias=False)
    policy = SimpleNamespace(model=model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.5)
    before = model.weight.detach().clone()
    loss = model.weight.sum() * 0.0 + 1.0
    loss.backward()

    grad_norm, applied, skipped = _step_optimizer_fail_closed(
        policy,
        optimizer,
        loss=loss,
        max_grad_norm=1.0,
    )

    assert float(grad_norm) == 0.0
    assert applied is True
    assert skipped is False
    # SGD weight decay is part of a valid optimizer step even at a stationary
    # point; the fail-closed guard must not erase that semantic.
    assert not torch.equal(before, model.weight.detach())


def test_ddp_zero_gradient_uses_global_objective_presence_before_skipping(
    monkeypatch,
) -> None:
    """A locally empty sparse rank must step when any peer has an objective.

    Before the fix, the empty rank skipped while its peer applied Adam/AdamW at
    an exact stationary point, immediately desynchronizing optimizer state and
    decoupled weight decay across DDP replicas.
    """
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    from tools.train_bc import _step_optimizer_fail_closed

    model = torch.nn.Linear(2, 1, bias=False)
    policy = SimpleNamespace(model=model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.5)
    before = model.weight.detach().clone()
    local_zero_loss = model.weight.sum() * 0.0
    local_zero_loss.backward()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def peer_has_objective(flag, op=None):
        assert op in {dist.ReduceOp.MAX, dist.ReduceOp.MIN}
        if op == dist.ReduceOp.MAX:
            flag.fill_(1)

    monkeypatch.setattr(dist, "all_reduce", peer_has_objective)

    grad_norm, applied, skipped = _step_optimizer_fail_closed(
        policy,
        optimizer,
        loss=local_zero_loss,
        max_grad_norm=1.0,
    )

    assert float(grad_norm) == 0.0
    assert applied is True
    assert skipped is False
    assert not torch.equal(before, model.weight.detach())


def test_ddp_globally_empty_zero_gradient_still_skips(monkeypatch) -> None:
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    from tools.train_bc import _step_optimizer_fail_closed

    model = torch.nn.Linear(2, 1, bias=False)
    policy = SimpleNamespace(model=model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.5)
    before = model.weight.detach().clone()
    loss = model.weight.sum() * 0.0
    loss.backward()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", lambda flag, op=None: None)

    _, applied, skipped = _step_optimizer_fail_closed(
        policy,
        optimizer,
        loss=loss,
        max_grad_norm=1.0,
    )

    assert applied is False
    assert skipped is True
    assert torch.equal(before, model.weight.detach())


def test_nonfinite_adam_state_aborts_after_finite_gradient() -> None:
    from types import SimpleNamespace

    import torch
    from tools.train_bc import _step_optimizer_fail_closed

    model = torch.nn.Linear(1, 1, bias=False)
    policy = SimpleNamespace(model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer.state[parameter]["exp_avg"].fill_(float("inf"))
    loss = model(torch.ones(1, 1)).square().mean()
    loss.backward()

    with pytest.raises(FloatingPointError, match="model or optimizer state"):
        _step_optimizer_fail_closed(
            policy,
            optimizer,
            loss=loss,
            max_grad_norm=1.0,
        )


def test_nonfinite_gradient_norm_aborts_before_optimizer_step(
    tmp_path, monkeypatch
) -> None:
    import torch
    from tools import train_bc

    data = _write_and_load_shard(tmp_path, _collect_real_samples(3))
    batch = np.arange(len(data["action_taken"]))
    weights = np.ones(len(batch), dtype=np.float32)
    policy = _make_entity_policy()
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=1e-3)
    steps = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal steps
        steps += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    monkeypatch.setattr(
        train_bc,
        "_clip_grad_norm",
        lambda *_args, **_kwargs: torch.tensor(float("inf")),
    )

    with pytest.raises(FloatingPointError, match="non-finite BC gradient norm"):
        _train_entity_batch(
            policy,
            optimizer,
            data,
            batch,
            weights,
            weights,
            soft_target_temperature=1.0,
            soft_target_weight=0.0,
            soft_target_source="scores",
            soft_target_min_legal_coverage=0.0,
            policy_loss_weight=1.0,
            value_loss_weight=1.0,
            final_vp_loss_weight=0.0,
            q_loss_weight=0.0,
            q_skip_teacher_prefixes=(),
            vps_to_win=10,
            advantage_policy_weighting="none",
            advantage_temperature=1.0,
            advantage_weight_cap=5.0,
            advantage_weight_floor=0.05,
            amp="none",
            diagnostics=False,
        )
    assert steps == 0


def test_train_entity_reports_soft_targets_conditioned_on_policy_active_rows(
    tmp_path, monkeypatch
) -> None:
    import torch
    from tools import train_bc

    samples = _collect_real_samples(6)
    data = _write_and_load_shard(tmp_path, samples)
    n = len(data["action_taken"])
    batch = np.arange(n)
    legal = np.asarray(data["legal_action_ids"][batch])
    multi_action = np.sum(legal >= 0, axis=1) > 1
    assert np.count_nonzero(multi_action) >= 2

    # Every multi-action row has a valid soft distribution, while only one of
    # those rows is admitted to policy CE. This reproduces the production
    # distinction between stored targets and policy_weight_multiplier.
    def uniform_soft_targets(data_arg, batch_arg, device, *_args):
        legal_arg = np.asarray(data_arg["legal_action_ids"][batch_arg])
        support_np = legal_arg >= 0
        counts = np.maximum(support_np.sum(axis=1, keepdims=True), 1)
        targets_np = support_np.astype(np.float32) / counts
        has_soft_np = support_np.sum(axis=1) > 1
        return (
            torch.as_tensor(targets_np, device=device),
            torch.as_tensor(has_soft_np, dtype=torch.bool, device=device),
            torch.as_tensor(support_np, dtype=torch.bool, device=device),
        )

    monkeypatch.setattr(train_bc, "_soft_targets_legal", uniform_soft_targets)
    policy_weights = np.zeros(n, dtype=np.float32)
    active_row = int(np.flatnonzero(multi_action)[0])
    policy_weights[active_row] = 1.0
    policy = _make_entity_policy()
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)

    metrics = _train_entity_batch(
        policy,
        optimizer,
        data,
        batch,
        policy_weights,
        np.ones(n, dtype=np.float32),
        soft_target_temperature=1.0,
        soft_target_weight=1.0,
        soft_target_source="policy",
        soft_target_min_legal_coverage=0.0,
        policy_loss_weight=1.0,
        value_loss_weight=0.0,
        final_vp_loss_weight=0.0,
        q_loss_weight=0.0,
        q_skip_teacher_prefixes=(),
        vps_to_win=10,
        advantage_policy_weighting="none",
        advantage_temperature=1.0,
        advantage_weight_cap=5.0,
        advantage_weight_floor=0.05,
        amp="none",
        diagnostics=False,
    )

    assert metrics["soft_distillation_rows"] == int(np.count_nonzero(multi_action))
    assert metrics["soft_distillation_active_rows"] == 1
    assert metrics["active_count"] == 1


def test_policy_aux_batch_combines_parts_and_adds_no_value_gradient(tmp_path) -> None:
    import copy
    import torch

    samples = _collect_real_samples(6)
    data = _write_and_load_shard(tmp_path, samples)
    n = len(data["action_taken"])
    batch = np.arange(n)
    weights = np.ones(n, dtype=np.float32)
    template = _make_entity_policy()
    initial = copy.deepcopy(template.model.state_dict())

    def run(*, auxiliary: bool, aux_weight_scale: float = 1.0):
        policy = _make_entity_policy()
        policy.model.load_state_dict(initial)
        policy.model.eval()  # make the duplicated-forward equality exact (no dropout)
        optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
        metrics = _train_entity_batch(
            policy,
            optimizer,
            data,
            batch,
            weights,
            weights,
            soft_target_temperature=1.0,
            soft_target_weight=0.0,
            soft_target_source="scores",
            soft_target_min_legal_coverage=0.0,
            policy_loss_weight=1.0,
            value_loss_weight=0.0,
            final_vp_loss_weight=0.0,
            q_loss_weight=0.0,
            q_skip_teacher_prefixes=(),
            vps_to_win=10,
            advantage_policy_weighting="none",
            advantage_temperature=1.0,
            advantage_weight_cap=5.0,
            advantage_weight_floor=0.05,
            amp="none",
            diagnostics=False,
            **(
                {
                    "policy_aux_data": data,
                    "policy_aux_batch": batch,
                    "policy_aux_sample_weights": weights * aux_weight_scale,
                }
                if auxiliary
                else {}
            ),
        )
        return policy, metrics

    _control_policy, control = run(auxiliary=False)
    aux_policy, auxiliary = run(auxiliary=True)
    _scaled_policy, scaled_auxiliary = run(auxiliary=True, aux_weight_scale=17.0)
    # Duplicating the same policy rows in the active-policy stream must add one
    # complete auxiliary dose without shrinking the original base objective.
    assert auxiliary["policy_loss"] == pytest.approx(
        2.0 * control["policy_loss"], rel=1e-6
    )
    assert auxiliary["policy_loss_weighted_sum"] == pytest.approx(
        2.0 * control["policy_loss_weighted_sum"], rel=1e-6
    )
    assert auxiliary["policy_loss_weight_sum"] == pytest.approx(
        control["policy_loss_weight_sum"], rel=1e-6
    )
    assert auxiliary["policy_base_loss_weight_sum"] == pytest.approx(
        control["policy_loss_weight_sum"], rel=1e-6
    )
    assert auxiliary["policy_aux_loss_weight_sum"] == pytest.approx(
        control["policy_loss_weight_sum"], rel=1e-6
    )
    # Corpus-normalized row weights and world-size-dependent mass must not
    # redefine the AUX coefficient. Only the explicit loss weight may do that.
    assert scaled_auxiliary["policy_loss"] == pytest.approx(
        auxiliary["policy_loss"], rel=1e-6
    )
    assert scaled_auxiliary["policy_aux_loss_weight_sum"] == pytest.approx(
        17.0 * auxiliary["policy_aux_loss_weight_sum"], rel=1e-6
    )
    # Value telemetry/dose is base-only and value-head parameters receive no
    # gradient from the policy-only auxiliary forward.
    assert auxiliary["value_loss_weight_sum"] == pytest.approx(
        control["value_loss_weight_sum"]
    )
    assert auxiliary["policy_aux_active_count"] == n
    value_grads = [p.grad for p in aux_policy.model.value_head.parameters()]
    assert all(
        grad is None or torch.count_nonzero(grad).item() == 0 for grad in value_grads
    )
    policy_grads = [p.grad for p in aux_policy.model.action_encoder.parameters()]
    assert any(
        grad is not None and torch.count_nonzero(grad).item() > 0
        for grad in policy_grads
    )


def test_policy_stream_metrics_preserve_unequal_base_aux_sufficient_stats(
    tmp_path, monkeypatch
) -> None:
    import copy
    import torch
    from tools import train_bc

    data = _write_and_load_shard(tmp_path, _collect_real_samples(24))
    legal = np.asarray(data["legal_action_ids"])
    wide_rows = np.flatnonzero(np.sum(legal >= 0, axis=1) >= 4)
    assert len(wide_rows) >= 4
    aux_batch = wide_rows[:4].astype(np.int64)
    base_batch = np.asarray(
        [row for row in range(len(legal)) if row not in set(aux_batch)][:2],
        dtype=np.int64,
    )
    assert len(base_batch) == 2

    # Make every audited stratum deliberately stream-specific. Soft-target
    # coverage also differs (1/2 base rows versus 3/4 AUX rows).
    data["phase"] = np.asarray(
        ["UNUSED"] * len(legal), dtype=object
    )
    data["teacher_name"] = np.asarray(
        ["unused_teacher"] * len(legal), dtype=object
    )
    data["stream_marker"] = np.asarray(
        ["unused"] * len(legal), dtype=object
    )
    data["soft_marker"] = np.zeros(len(legal), dtype=np.bool_)
    data["phase"][base_batch] = "BASE_PHASE"
    data["phase"][aux_batch] = "AUX_PHASE"
    data["teacher_name"][base_batch] = "base_teacher"
    data["teacher_name"][aux_batch] = "aux_teacher"
    data["stream_marker"][base_batch] = "base"
    data["stream_marker"][aux_batch] = "aux"
    data["soft_marker"][base_batch[:1]] = True
    data["soft_marker"][aux_batch[:3]] = True

    original_forward = train_bc._forward_legal_np_for_batch

    def controlled_forward(policy, data_arg, batch_arg, legal_ids, **kwargs):
        outputs = original_forward(
            policy, data_arg, batch_arg, legal_ids, **kwargs
        )
        logits = outputs["logits"]
        target_columns = train_bc._target_columns(
            legal_ids,
            np.asarray(data_arg["action_taken"])[batch_arg].astype(np.int64),
        )
        legal_mask = torch.as_tensor(
            np.asarray(legal_ids) >= 0, dtype=torch.bool, device=logits.device
        )
        forced = torch.where(
            legal_mask,
            torch.zeros_like(logits),
            torch.full_like(logits, -1.0e9),
        )
        targets = torch.as_tensor(
            target_columns, dtype=torch.long, device=logits.device
        )
        markers = np.asarray(data_arg["stream_marker"])[batch_arg]
        for row, marker in enumerate(markers):
            target = int(targets[row].item())
            if marker == "base":
                forced[row, target] = 100.0
            elif marker == "aux":
                forced[row, target] = -100.0
                legal_columns = np.flatnonzero(np.asarray(legal_ids[row]) >= 0)
                wrong = int(next(value for value in legal_columns if value != target))
                forced[row, wrong] = 100.0
        outputs["logits"] = logits * 0.0 + forced
        return outputs

    def controlled_soft_targets(data_arg, batch_arg, device, *_args):
        legal_arg = np.asarray(data_arg["legal_action_ids"])[batch_arg]
        support = legal_arg >= 0
        enabled = np.asarray(data_arg["soft_marker"])[batch_arg]
        targets = np.zeros_like(legal_arg, dtype=np.float32)
        counts = np.maximum(support.sum(axis=1, keepdims=True), 1)
        targets[enabled] = (
            support[enabled].astype(np.float32) / counts[enabled]
        )
        return (
            torch.as_tensor(targets, device=device),
            torch.as_tensor(enabled, dtype=torch.bool, device=device),
            torch.as_tensor(support, dtype=torch.bool, device=device),
        )

    monkeypatch.setattr(
        train_bc, "_forward_legal_np_for_batch", controlled_forward
    )
    monkeypatch.setattr(train_bc, "_soft_targets_legal", controlled_soft_targets)
    template = _make_entity_policy()
    initial = copy.deepcopy(template.model.state_dict())
    weights = np.zeros(len(legal), dtype=np.float32)
    weights[base_batch] = np.asarray([1.0, 3.0], dtype=np.float32)
    weights[aux_batch] = np.asarray([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

    def run(base_weights: np.ndarray):
        policy = _make_entity_policy()
        policy.model.load_state_dict(initial)
        policy.model.eval()
        optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
        return _train_entity_batch(
            policy,
            optimizer,
            data,
            base_batch,
            base_weights,
            np.ones(len(legal), dtype=np.float32),
            soft_target_temperature=1.0,
            soft_target_weight=1.0,
            soft_target_source="policy",
            soft_target_min_legal_coverage=0.0,
            policy_loss_weight=1.0,
            value_loss_weight=0.0,
            final_vp_loss_weight=0.0,
            q_loss_weight=0.0,
            q_skip_teacher_prefixes=(),
            vps_to_win=10,
            advantage_policy_weighting="none",
            advantage_temperature=1.0,
            advantage_weight_cap=5.0,
            advantage_weight_floor=0.05,
            amp="none",
            diagnostics=True,
            policy_aux_data=data,
            policy_aux_batch=aux_batch,
            policy_aux_sample_weights=weights,
        )

    metrics = run(weights)
    assert metrics["policy_base_row_count"] == 2
    assert metrics["policy_aux_row_count"] == 4
    assert metrics["policy_total_row_count"] == 6
    assert metrics["policy_base_active_count"] == 2
    assert metrics["policy_aux_active_count"] == 4
    assert metrics["policy_total_active_count"] == 6
    assert metrics["policy_base_correct_count"] == 2
    assert metrics["policy_aux_correct_count"] == 0
    assert metrics["policy_total_correct_count"] == 2
    assert metrics["policy_base_top3_correct_count"] == 2
    assert metrics["policy_aux_top3_correct_count"] == 0
    assert metrics["policy_total_top3_correct_count"] == 2
    assert metrics["policy_base_accuracy"] == 1.0
    assert metrics["policy_aux_accuracy"] == 0.0
    assert metrics["accuracy"] == pytest.approx(2.0 / 6.0)
    assert metrics["policy_total_accuracy"] == metrics["accuracy"]
    assert metrics["policy_base_top3_accuracy"] == 1.0
    assert metrics["policy_aux_top3_accuracy"] == 0.0
    assert metrics["policy_total_top3_accuracy"] == metrics["top3_accuracy"]
    assert metrics["soft_distillation_base_rows"] == 1
    assert metrics["soft_distillation_aux_rows"] == 3
    assert metrics["soft_distillation_total_rows"] == 4
    assert metrics["soft_distillation_rows"] == 4
    assert metrics["soft_distillation_base_active_rows"] == 1
    assert metrics["soft_distillation_aux_active_rows"] == 3
    assert metrics["soft_distillation_active_rows"] == 4
    assert metrics["policy_base_phase_stats"]["BASE_PHASE"]["count"] == 2
    assert metrics["policy_aux_phase_stats"]["AUX_PHASE"]["count"] == 4
    assert metrics["phase_stats"]["BASE_PHASE"]["count"] == 2
    assert metrics["phase_stats"]["AUX_PHASE"]["count"] == 4
    assert metrics["policy_total_phase_stats"] == metrics["phase_stats"]
    assert metrics["policy_base_teacher_stats"]["base_teacher"]["count"] == 2
    assert metrics["policy_aux_teacher_stats"]["aux_teacher"]["count"] == 4
    assert metrics["policy_total_teacher_stats"] == metrics["teacher_stats"]

    zero_base_weights = weights.copy()
    zero_base_weights[base_batch] = 0.0
    zero_base = run(zero_base_weights)
    assert zero_base["policy_base_active_count"] == 0
    assert zero_base["policy_base_correct_count"] == 0
    assert zero_base["policy_base_accuracy"] == 0.0
    assert zero_base["policy_aux_active_count"] == 4
    assert zero_base["policy_total_active_count"] == 4
    assert zero_base["accuracy"] == zero_base["policy_aux_accuracy"] == 0.0
    assert zero_base["policy_base_phase_stats"] == {}
    assert zero_base["policy_aux_phase_stats"]["AUX_PHASE"]["count"] == 4


def test_entity_main_builds_base_only_policy_report_with_unforced_phases(
    tmp_path, monkeypatch, capsys
) -> None:
    import json

    from catan_zero.rl.self_play import make_env_config
    from tools import train_bc

    data = _write_and_load_shard(tmp_path, _collect_real_samples(6))
    data["phase"] = np.asarray(["OPEN"] * 6, dtype=object)
    data["phase"][0] = "FORCED"
    data["teacher_name"] = np.asarray(["entity_teacher"] * 6, dtype=object)
    data["policy_weight_multiplier"] = np.ones(6, dtype=np.float32)
    data["policy_weight_multiplier"][-1] = 0.0
    winner = str(np.asarray(data["player"]).astype(str)[0])
    data["winner"] = np.asarray([winner] * 6, dtype=object)
    if "terminated" in data:
        data["terminated"] = np.ones(6, dtype=np.bool_)
    if "truncated" in data:
        data["truncated"] = np.zeros(6, dtype=np.bool_)
    forced_action = int(data["action_taken"][0])
    data["legal_action_ids"][0] = -1
    data["legal_action_ids"][0, 0] = forced_action
    # This test exercises policy reporting, but the production learner also
    # admits value labels before training. Keep the synthetic terminal winner
    # seated in its single game instead of inheriting a truncated sample prefix
    # whose winner belongs to an unobserved player.
    winner = str(np.asarray(data["player"]).astype(str)[0])
    data["winner"] = np.asarray([winner] * 6, dtype=object)
    if "terminated" in data:
        data["terminated"] = np.ones(6, dtype=np.bool_)
    if "truncated" in data:
        data["truncated"] = np.zeros(6, dtype=np.bool_)

    monkeypatch.setattr(
        train_bc, "load_teacher_data", lambda _path, **_kwargs: data
    )
    monkeypatch.setattr(
        train_bc,
        "_env_config_for_teacher_data",
        lambda _args, _data, _ddp: make_env_config(vps_to_win=3),
    )
    report_path = tmp_path / "entity-report.json"
    train_bc.main(
        [
            "--data",
            str(tmp_path / "shard"),
            "--data-format",
            "npz",
            "--checkpoint",
            str(tmp_path / "entity.pt"),
            "--report",
            str(report_path),
            "--arch",
            "entity_graph",
            "--device",
            "cpu",
            "--hidden-size",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--validation-fraction",
            "0",
            "--host-lock-file",
            str(tmp_path / "train.lock"),
            "--train-diagnostics-every-batches",
            "1",
            "--skip-guards",
        ]
    )
    capsys.readouterr()
    metric = json.loads(report_path.read_text(encoding="utf-8"))["metrics"][0]
    assert metric["samples"] == metric["policy_base_row_count"] == 6
    assert metric["policy_aux_row_count"] == 0
    assert metric["policy_total_row_count"] == 6
    assert metric["policy_base_loss"] == pytest.approx(metric["policy_loss"])
    assert metric["policy_aux_loss"] == 0.0
    # The default policy objective now excludes one-legal-action plumbing.
    # Four OPEN rows remain active: one row is explicitly weight-zero and the
    # FORCED row receives the default forced_action_weight=0.
    assert metric["policy_base_accuracy_active_count"] == 4
    assert metric["policy_aux_accuracy_active_count"] == 0
    assert metric["policy_total_accuracy_active_count"] == 4
    assert metric["policy_total_accuracy"] == metric["accuracy"]
    assert metric["policy_total_phase_accuracy"] == metric["phase_accuracy"]
    assert "FORCED" not in metric["policy_base_phase_accuracy"]
    assert metric["policy_base_phase_accuracy"]["OPEN"]["count"] == 4
    unforced = metric["policy_base_phase_accuracy_excluding_forced"]
    assert unforced == metric["policy_total_phase_accuracy_excluding_forced"]
    assert unforced == metric["phase_accuracy_excluding_forced"]
    assert set(unforced) == {"OPEN"}
    assert unforced["OPEN"]["count"] == 4


def test_exact_max_steps_continues_past_configured_epoch_limit(
    tmp_path, monkeypatch, capsys
) -> None:
    import json

    from catan_zero.rl.self_play import make_env_config
    from tools import train_bc

    data = _write_and_load_shard(tmp_path, _collect_real_samples(6))
    winner = str(np.asarray(data["player"]).astype(str)[0])
    data["winner"] = np.asarray([winner] * 6, dtype=object)
    if "terminated" in data:
        data["terminated"] = np.ones(6, dtype=np.bool_)
    if "truncated" in data:
        data["truncated"] = np.zeros(6, dtype=np.bool_)
    monkeypatch.setattr(
        train_bc, "load_teacher_data", lambda _path, **_kwargs: data
    )
    monkeypatch.setattr(
        train_bc,
        "_env_config_for_teacher_data",
        lambda _args, _data, _ddp: make_env_config(vps_to_win=3),
    )
    report_path = tmp_path / "exact-dose-report.json"
    train_bc.main(
        [
            "--data",
            str(tmp_path / "shard"),
            "--data-format",
            "npz",
            "--checkpoint",
            str(tmp_path / "exact-dose.pt"),
            "--report",
            str(report_path),
            "--arch",
            "entity_graph",
            "--device",
            "cpu",
            "--hidden-size",
            "16",
            "--epochs",
            "1",
            "--max-steps",
            "3",
            "--exact-max-steps",
            "--batch-size",
            "4",
            "--validation-fraction",
            "0",
            "--host-lock-file",
            str(tmp_path / "train.lock"),
            "--skip-guards",
        ]
    )
    capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["epochs"] == 1
    assert report["effective_epoch_limit"] == 3
    assert report["exact_max_steps"] is True
    assert report["steps_completed"] == 3
    assert report["value_training"]["optimizer_steps"] == 3
    assert len(report["metrics"]) == 2


def test_unforced_phase_ddp_reducer_merges_unequal_rank_counts(monkeypatch) -> None:
    import torch.distributed as dist
    from tools import train_bc

    local = {"OPEN": {"count": 1, "top1": 1, "top3": 1}}
    remote = {
        "OPEN": {"count": 3, "top1": 1, "top3": 2},
        "ROBBER": {"count": 2, "top1": 0, "top3": 1},
    }

    def fake_all_gather_object(output, value):
        assert value == local
        output[:] = [local, remote]

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)
    reduced = train_bc._reduce_nested_count_stats(  # noqa: SLF001
        local,
        {"enabled": True, "world_size": 2, "rank": 0, "local_rank": 0},
    )
    assert reduced == {
        "OPEN": {"count": 4, "top1": 2, "top3": 3},
        "ROBBER": {"count": 2, "top1": 0, "top3": 1},
    }
    finalized = train_bc._finalize_phase_stats(reduced)  # noqa: SLF001
    assert finalized["OPEN"]["top1_accuracy"] == pytest.approx(0.5)
    assert finalized["ROBBER"]["top3_accuracy"] == pytest.approx(0.5)


def test_policy_target_distribution_metrics_follow_soft_teacher_and_opening_index() -> None:
    import torch
    from tools import train_bc

    logits = torch.tensor(
        [
            [3.0, 2.0, -4.0],
            [3.0, 2.0, 1.0],
            [0.0, 4.0, 2.0],
        ]
    )
    targets = torch.tensor(
        [
            [0.40, 0.60, 0.00],
            [0.05, 0.15, 0.80],
            [0.10, 0.70, 0.20],
        ]
    )
    has_soft = torch.tensor([True, True, True])
    active = torch.tensor([True, True, False])
    support = torch.ones_like(targets, dtype=torch.bool)
    data = {
        "phase": np.asarray(
            [
                "BUILD_INITIAL_SETTLEMENT",
                "BUILD_INITIAL_SETTLEMENT",
                "PLAY_TURN",
            ]
        ),
        "decision_index": np.asarray([0, 2, 9], dtype=np.int32),
    }

    sufficient = train_bc._policy_target_distribution_stats(  # noqa: SLF001
        data,
        np.arange(3, dtype=np.int64),
        logits,
        targets,
        has_soft,
        active,
        support,
    )
    metrics = train_bc._finalize_policy_target_distribution_stats(  # noqa: SLF001
        sufficient
    )

    assert metrics["weighting"] == "uniform_soft_target_policy_active_rows"
    assert metrics["objective_weighted"] is False
    assert metrics["overall"]["rows"] == 2
    assert metrics["overall"]["teacher_argmax_top1_accuracy"] == 0.0
    assert metrics["overall"]["teacher_argmax_top3_accuracy"] == 1.0
    assert metrics["overall"]["model_top1_target_mass"] == pytest.approx(
        (0.40 + 0.05) / 2.0
    )
    assert metrics["overall"]["model_top3_target_mass"] == pytest.approx(1.0)
    assert metrics["overall"]["kl_target_model"] == pytest.approx(
        metrics["overall"]["excess_cross_entropy_above_target_entropy"]
    )
    assert metrics["phase"]["BUILD_INITIAL_SETTLEMENT"]["rows"] == 2
    assert metrics["opening_decision_index"]["0"]["rows"] == 1
    assert metrics["opening_decision_index"]["2"]["rows"] == 1
    assert "9" not in metrics["opening_decision_index"]


def test_policy_target_distribution_ddp_reducer_sums_sufficient_stats(
    monkeypatch,
) -> None:
    import torch.distributed as dist
    from tools import train_bc

    local = {
        "schema_version": "policy-target-distribution-sufficient-stats-v1",
        "overall": {
            **train_bc._empty_policy_target_metric_parts(),  # noqa: SLF001
            "rows": 1.0,
            "teacher_argmax_top1_correct": 1.0,
            "cross_entropy_sum": 0.5,
            "target_entropy_sum": 0.2,
            "kl_target_model_sum": 0.3,
            "excess_cross_entropy_sum": 0.3,
        },
        "phase": {},
        "opening_decision_index": {},
    }
    remote = {
        "schema_version": "policy-target-distribution-sufficient-stats-v1",
        "overall": {
            **train_bc._empty_policy_target_metric_parts(),  # noqa: SLF001
            "rows": 3.0,
            "teacher_argmax_top1_correct": 1.0,
            "teacher_argmax_top3_correct": 2.0,
            "cross_entropy_sum": 2.5,
            "target_entropy_sum": 1.0,
            "kl_target_model_sum": 1.5,
            "excess_cross_entropy_sum": 1.5,
        },
        "phase": {},
        "opening_decision_index": {},
    }

    def fake_all_gather_object(output, value):
        assert value == local
        output[:] = [local, remote]

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)
    reduced = train_bc._reduce_policy_target_distribution_stats(  # noqa: SLF001
        local,
        {"enabled": True, "world_size": 2, "rank": 0, "local_rank": 0},
    )
    metrics = train_bc._finalize_policy_target_distribution_stats(  # noqa: SLF001
        reduced
    )["overall"]
    assert metrics["rows"] == 4
    assert metrics["teacher_argmax_top1_accuracy"] == pytest.approx(0.5)
    assert metrics["teacher_argmax_top3_accuracy"] == pytest.approx(0.5)
    assert metrics["soft_target_cross_entropy"] == pytest.approx(0.75)
    assert metrics["target_entropy"] == pytest.approx(0.30)
    assert metrics["kl_target_model"] == pytest.approx(0.45)


def test_objective_validation_aggregates_teacher_metrics_by_eligible_density() -> None:
    from tools import train_bc

    def report(*, samples: int, rows: float, hits: float) -> dict:
        parts = train_bc._empty_policy_target_metric_parts()  # noqa: SLF001
        parts.update(
            {
                "rows": rows,
                "teacher_argmax_top1_correct": hits,
                "teacher_argmax_top3_correct": rows,
                "cross_entropy_sum": rows,
                "target_entropy_sum": rows * 0.25,
                "kl_target_model_sum": rows * 0.75,
                "excess_cross_entropy_sum": rows * 0.75,
                "model_top1_target_mass_sum": rows * 0.5,
                "model_top3_target_mass_sum": rows * 0.9,
            }
        )
        return {
            "samples": samples,
            "loss": 0.0,
            "policy_loss": 0.0,
            "loss_denominators": {},
            "policy_target_distribution_sufficient_statistics": {
                "schema_version": (
                    "policy-target-distribution-sufficient-stats-v1"
                ),
                "overall": parts,
                "phase": {},
                "opening_decision_index": {},
            },
        }

    metrics, _ = train_bc._objective_measure_validation_aggregate(  # noqa: SLF001
        [
            report(samples=10, rows=5.0, hits=2.0),
            report(samples=20, rows=10.0, hits=8.0),
        ],
        np.asarray([0.5, 0.5]),
    )
    target = metrics["policy_target_distribution_metrics"]["overall"]
    assert target["row_probability"] == pytest.approx(0.5)
    assert target["teacher_argmax_top1_accuracy"] == pytest.approx(0.6)
    assert target["teacher_argmax_top3_accuracy"] == pytest.approx(1.0)
    assert target["soft_target_cross_entropy"] == pytest.approx(1.0)
