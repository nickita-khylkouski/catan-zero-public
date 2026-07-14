"""CAT-61 Stage-1: value-error/uncertainty head + stop-gradient + evaluator emission.

Covers the training-facing half of the ticket: the head is built on the #63
aux-head scaffolding, reads a STOP-GRADIENT copy of the trunk state (so its loss
never distorts value/trunk learning), trains without NaNs, and is surfaced to the
searcher only through the opt-in EntityGraphRustEvaluatorConfig.emit_uncertainty.
None of this needs the rust engine, so these run everywhere.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

# Reuse the real-decision-state batch builder from the action-attention test
# (same-directory import; pytest prepends the test dir to sys.path).
from test_entity_token_policy_action_attention import (
    _base_config,
    _real_entity_batch,
    _to_torch,
)


def _net_with_head():
    from catan_zero.rl.entity_token_policy import EntityGraphNet

    return EntityGraphNet(dataclasses.replace(_base_config(), value_uncertainty_head=True))


def test_head_off_by_default_no_output_and_identical_params():
    from catan_zero.rl.entity_token_policy import EntityGraphNet

    off = EntityGraphNet(_base_config())
    on = _net_with_head()
    # The head adds parameters only when enabled...
    off_keys = set(off.state_dict().keys())
    on_keys = set(on.state_dict().keys())
    assert off_keys.issubset(on_keys)
    new_keys = on_keys - off_keys
    assert new_keys and all(k.startswith("value_uncertainty_head.") for k in new_keys)

    off.eval()
    batch = _to_torch(_real_entity_batch())
    outputs = off(batch)
    assert "value_uncertainty" not in outputs  # default forward output unchanged


def test_head_emits_nonnegative_output_shaped_like_value():
    net = _net_with_head()
    net.eval()
    batch = _to_torch(_real_entity_batch())
    outputs = net(batch)
    assert "value_uncertainty" in outputs
    unc = outputs["value_uncertainty"]
    assert unc.shape == outputs["value"].shape
    # softplus at the emit site -> strictly non-negative predicted error.
    assert float(unc.detach().min()) >= 0.0


def test_uncertainty_head_is_stop_gradient_from_trunk_and_value_head():
    """Back-propagating ONLY the uncertainty output must not touch the trunk or
    the value head (the detach in forward), while it MUST reach the head."""
    import torch

    net = _net_with_head()
    net.train()
    batch = _to_torch(_real_entity_batch())
    outputs = net(batch)
    net.zero_grad(set_to_none=True)
    outputs["value_uncertainty"].sum().backward()

    def _has_grad(module) -> bool:
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    # The head itself learns...
    assert _has_grad(net.value_uncertainty_head)
    # ...but nothing upstream does: not the value head, not the trunk blocks.
    assert not _has_grad(net.value_head)
    assert not _has_grad(net.blocks)
    assert net.cls_token.grad is None or torch.all(net.cls_token.grad == 0)


def test_value_loss_still_trains_the_value_head():
    """Control for the stop-gradient test: the VALUE output does reach the value
    head, so the detach is specific to the uncertainty branch (not a dead net)."""
    import torch

    net = _net_with_head()
    net.train()
    batch = _to_torch(_real_entity_batch())
    outputs = net(batch)
    net.zero_grad(set_to_none=True)
    outputs["value"].sum().backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in net.value_head.parameters()
    )


def test_head_trains_without_nan_and_leaves_value_head_untouched():
    """Optimize ONLY the head against a realized-error target; loss stays finite
    and decreases, predictions track the target, and the value head's weights do
    not move (end-to-end confirmation of the stop-gradient)."""
    import torch

    net = _net_with_head()
    net.train()
    batch = _to_torch(_real_entity_batch())

    with torch.no_grad():
        value = net(batch)["value"]
    # Synthetic realized outcomes -> target error (z - v)^2, exactly the
    # train_bc.py target (which also detaches v).
    rng = np.random.default_rng(0)
    z = torch.as_tensor(rng.uniform(-1.0, 1.0, size=value.shape).astype(np.float32))
    target = (z - value) ** 2

    value_head_before = [p.detach().clone() for p in net.value_head.parameters()]
    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    losses = []
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        pred = net(batch)["value_uncertainty"]
        loss = torch.nn.functional.smooth_l1_loss(pred, target)
        assert torch.isfinite(loss), "uncertainty head produced a non-finite loss"
        loss.backward()
        opt.step()
        losses.append(float(loss))

    assert losses[-1] < losses[0]  # the head actually learned
    # value head weights are unchanged: the head's gradients never reached it,
    # even though it shared the optimizer.
    for before, after in zip(value_head_before, net.value_head.parameters()):
        assert torch.allclose(before, after)


# --- evaluator emission (opt-in, default OFF) --------------------------------
def _bare_evaluator(emit: bool):
    """An EntityGraphRustEvaluator with only .config set (no policy needed for
    the pure packing helpers)."""
    from catan_zero.search.neural_rust_mcts import (
        EntityGraphRustEvaluator,
        EntityGraphRustEvaluatorConfig,
    )

    ev = EntityGraphRustEvaluator.__new__(EntityGraphRustEvaluator)
    ev.config = EntityGraphRustEvaluatorConfig(emit_uncertainty=emit)
    return ev


def test_eval_result_shape_gated_by_emit_flag():
    off = _bare_evaluator(False)
    on = _bare_evaluator(True)
    priors = {1: 0.7, 2: 0.3}
    assert off._eval_result(priors, 0.5, 0.9) == (priors, 0.5)  # 2-tuple, unchanged
    packed = on._eval_result(priors, 0.5, 0.9)
    assert packed == (priors, 0.5, 0.9)
    # negative uncertainty is clamped to 0.0
    assert on._eval_result(priors, 0.5, -3.0)[2] == 0.0


def test_cache_entry_copies_priors_and_matches_shape():
    off = _bare_evaluator(False)
    on = _bare_evaluator(True)
    priors = {1: 0.7, 2: 0.3}
    off_entry = off._cache_entry(priors, 0.5, 0.9)
    assert off_entry == (priors, 0.5) and off_entry[0] is not priors  # copied
    on_entry = on._cache_entry(priors, 0.5, 0.9)
    assert on_entry == (priors, 0.5, 0.9) and on_entry[0] is not priors


def test_uncertainty_from_outputs_absent_and_present():
    import torch

    from catan_zero.search.neural_rust_mcts import _uncertainty_from_outputs

    assert _uncertainty_from_outputs({"value": torch.zeros(3)}, 0) == 0.0
    outputs = {"value_uncertainty": torch.tensor([0.1, 0.4, 0.9])}
    assert _uncertainty_from_outputs(outputs, 2) == pytest.approx(0.9)


def test_evaluator_rejects_checkpoint_with_deleted_uncertainty_weights(tmp_path):
    """A config-only warm start may reconstruct the optional module, but an
    uncertainty consumer must not mistake its random parameters for trained
    checkpoint state.  The default/off path remains load-compatible."""
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.search.neural_rust_mcts import (
        EntityGraphRustEvaluator,
        EntityGraphRustEvaluatorConfig,
    )

    config = dataclasses.replace(_base_config(), value_uncertainty_head=True)
    policy = EntityGraphPolicy(
        config,
        np.zeros(
            (int(config.action_size), int(config.static_action_feature_size)),
            dtype=np.float32,
        ),
        device="cpu",
    )
    checkpoint = tmp_path / "missing-uncertainty-head.pt"
    policy.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    removed = [
        key
        for key in tuple(payload["model"])
        if key.startswith("value_uncertainty_head.")
    ]
    assert removed
    for key in removed:
        del payload["model"][key]
    torch.save(payload, checkpoint)

    loaded = EntityGraphPolicy.load(
        checkpoint,
        device="cpu",
        allow_missing_optional_parameters=True,
    )
    assert loaded.model.value_uncertainty_head is not None
    assert set(removed) <= set(loaded._checkpoint_missing_state_keys)

    # Backward compatibility: the optional branch is inert when not consumed.
    EntityGraphRustEvaluator(
        loaded,
        config=EntityGraphRustEvaluatorConfig(emit_uncertainty=False),
    )
    with pytest.raises(ValueError, match="trained weights are absent"):
        EntityGraphRustEvaluator(
            loaded,
            config=EntityGraphRustEvaluatorConfig(emit_uncertainty=True),
        )
