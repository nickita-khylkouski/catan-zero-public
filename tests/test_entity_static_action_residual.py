"""Contracts for the function-preserving static-action dead-input repair."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
from catan_zero.rl.entity_token_features import (
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
)
from catan_zero.rl.entity_token_policy import (
    PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
    STATIC_ACTION_RESIDUAL_FEATURE_SIZE,
    STATIC_ACTION_RESIDUAL_SLICE,
    EntityGraphConfig,
    EntityGraphNet,
    EntityGraphPolicy,
)
from catan_zero.rl.hex_symmetry import HexSymmetry, N_SYMMETRIES


def _config(**overrides) -> EntityGraphConfig:
    values = dict(
        action_size=607,
        static_action_feature_size=45,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=32,
        state_layers=1,
        attention_heads=4,
        dropout=0.0,
    )
    values.update(overrides)
    return EntityGraphConfig(**values)


def _torch_batch(batch_size: int = 2, action_width: int = 6) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260713)
    batch: dict[str, torch.Tensor] = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", 0, EVENT_FEATURE_SIZE),
    ):
        batch[f"{name}_tokens"] = torch.randn(
            batch_size, count, width, generator=generator
        )
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(
                batch_size, count, dtype=torch.bool
            )
    batch["legal_action_tokens"] = torch.randn(
        batch_size,
        action_width,
        LEGAL_ACTION_FEATURE_SIZE,
        generator=generator,
    )
    batch["legal_action_context"] = torch.randn(
        batch_size,
        action_width,
        CONTEXT_ACTION_FEATURE_SIZE,
        generator=generator,
    )
    batch["legal_action_static_features"] = torch.randn(
        batch_size,
        action_width,
        STATIC_ACTION_RESIDUAL_FEATURE_SIZE,
        generator=generator,
    )
    return batch


def _numpy_entity(batch_size: int = 1, action_width: int = 2) -> dict[str, np.ndarray]:
    torch_batch = _torch_batch(batch_size=batch_size, action_width=action_width)
    result = {
        key: value.numpy()
        for key, value in torch_batch.items()
        if key not in {"legal_action_context", "legal_action_static_features"}
    }
    result["legal_action_mask"] = np.ones(
        (batch_size, action_width), dtype=np.bool_
    )
    return result


def _symmetry_entity(action_id: int) -> dict[str, np.ndarray]:
    """Minimal, shape-real entity batch consumed by HexSymmetry."""
    entity = _numpy_entity(batch_size=1, action_width=1)
    entity["legal_action_tokens"][:] = 0.0
    entity["hex_vertex_ids"] = np.full((1, 19, 6), -1, dtype=np.int64)
    entity["hex_edge_ids"] = np.full((1, 19, 6), -1, dtype=np.int64)
    entity["edge_vertex_ids"] = np.full((1, 72, 2), -1, dtype=np.int64)
    entity["event_target_ids"] = np.empty((1, 0, 4), dtype=np.int64)
    entity["legal_action_mask"][:] = True
    # The scalar itself is overwritten by the symmetry implementation; the
    # authoritative identity is supplied separately as an integer below.
    entity["legal_action_tokens"][0, 0, 1] = action_id / 607.0
    return entity


def _adversarial_symmetry() -> HexSymmetry:
    def identities(width: int) -> np.ndarray:
        return np.broadcast_to(
            np.arange(width, dtype=np.int64), (N_SYMMETRIES, width)
        ).copy()

    pi_act = identities(332)
    # Make every non-identity orientation visibly remap catalog id 0 while
    # retaining the production table dimensions and row-alignment contract.
    for orientation in range(1, N_SYMMETRIES):
        swap = orientation % 11 + 1
        pi_act[orientation, 0], pi_act[orientation, swap] = swap, 0
    return HexSymmetry(
        fwd_hex=identities(19),
        inv_hex=identities(19),
        fwd_vertex=identities(54),
        inv_vertex=identities(54),
        fwd_edge=identities(72),
        inv_edge=identities(72),
        pi_act=pi_act,
        canonical_hex_coord=np.zeros((19, 3), dtype=np.float32),
        op_names=tuple(str(index) for index in range(N_SYMMETRIES)),
    )


def test_flag_off_has_no_new_parameters_and_strict_loads() -> None:
    torch.manual_seed(7)
    legacy = EntityGraphNet(_config())
    explicit_off = EntityGraphNet(_config(static_action_residual=False))

    assert set(legacy.state_dict()) == set(explicit_off.state_dict())
    explicit_off.load_state_dict(legacy.state_dict(), strict=True)
    assert not any(
        name.startswith("static_action_residual_proj.")
        for name, _ in legacy.named_parameters()
    )


def test_zero_output_upgrade_is_bit_exact_then_becomes_live() -> None:
    torch.manual_seed(11)
    legacy = EntityGraphNet(_config()).eval()
    treatment = EntityGraphNet(_config(static_action_residual=True)).eval()
    missing, unexpected = treatment.load_state_dict(legacy.state_dict(), strict=False)

    assert not unexpected
    assert set(missing) == {
        "static_action_residual_proj.bias",
        "static_action_residual_proj.weight",
    }
    batch = _torch_batch()
    with torch.no_grad():
        control = legacy(batch, return_q=True)
        upgraded_at_init = treatment(batch, return_q=True)
    for key in control:
        assert torch.equal(control[key], upgraded_at_init[key]), key

    with torch.no_grad():
        treatment.static_action_residual_proj.weight.normal_(std=0.1)
    changed_static = {key: value.clone() for key, value in batch.items()}
    changed_static["legal_action_static_features"] = (
        batch["legal_action_static_features"] * 7.0 + 3.0
    )
    with torch.no_grad():
        left = treatment(batch)["logits"]
        right = treatment(changed_static)["logits"]
    assert not torch.equal(left, right)


def test_only_residual_tensors_train_and_receive_gradient() -> None:
    torch.manual_seed(19)
    model = EntityGraphNet(_config(static_action_residual=True)).train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.static_action_residual_proj.parameters():
        parameter.requires_grad_(True)

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == {
        "static_action_residual_proj.bias",
        "static_action_residual_proj.weight",
    }

    batch = _torch_batch(batch_size=3, action_width=7)
    labels = torch.tensor([0, 3, 6])
    loss = torch.nn.functional.cross_entropy(model(batch)["logits"], labels)
    loss.backward()
    output_weight = model.static_action_residual_proj.weight
    assert output_weight.grad is not None
    assert torch.count_nonzero(output_weight.grad).item() > 0
    assert model.static_action_residual_proj.bias.grad is not None


class _CaptureModel:
    def __init__(self) -> None:
        self.static_batches: list[np.ndarray] = []

    def __call__(self, batch, **_kwargs):
        static = batch["legal_action_static_features"]
        self.static_batches.append(static.detach().cpu().numpy())
        batch_size, action_width = static.shape[:2]
        return {
            "logits": torch.zeros(batch_size, action_width),
            "value": torch.zeros(batch_size),
        }


def _capture_policy(static_table: np.ndarray) -> tuple[EntityGraphPolicy, _CaptureModel]:
    policy = object.__new__(EntityGraphPolicy)
    policy.config = _config(static_action_residual=True)
    policy.action_size = 607
    policy.device = "cpu"
    policy.public_award_feature_contract = (
        PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
    )
    policy.static_action_features = torch.as_tensor(static_table, dtype=torch.float32)
    capture = _CaptureModel()
    policy.model = capture
    return policy, capture


def test_train_time_d6_indexes_mapped_catalog_identity(monkeypatch) -> None:
    import catan_zero.rl.entity_token_policy as module

    monkeypatch.setattr(module, "_assert_entity_batch_shapes", lambda *_args: None)
    symmetry = _adversarial_symmetry()
    action_id = next(
        index
        for index in range(symmetry.pi_act.shape[1])
        if len(set(int(v) for v in symmetry.pi_act[:, index])) > 1
    )
    orientations = np.asarray([1, 5], dtype=np.int64)
    legal_ids = np.full((2, 1), action_id, dtype=np.int64)
    entity = {
        key: np.repeat(value, 2, axis=0)
        for key, value in _symmetry_entity(action_id).items()
    }
    transformed = symmetry.permute_entity_batch(
        entity,
        orientations,
        legal_action_ids=legal_ids,
        action_size=607,
    )
    expected_ids = symmetry.pi_act[orientations, action_id]

    table = np.zeros((607, 45), dtype=np.float32)
    table[:, STATIC_ACTION_RESIDUAL_SLICE.start] = np.arange(607)
    policy, capture = _capture_policy(table)
    context = np.zeros((2, 1, CONTEXT_ACTION_FEATURE_SIZE), dtype=np.float32)
    policy.forward_legal_np(transformed, legal_ids, context)

    np.testing.assert_array_equal(
        capture.static_batches[-1][:, 0, 0], expected_ids.astype(np.float32)
    )
    # Row alignment/masking remains tied to the original legal ids.
    np.testing.assert_array_equal(legal_ids[:, 0], action_id)


def test_twelve_orientation_average_uses_twelve_mapped_catalog_ids(monkeypatch) -> None:
    import catan_zero.rl.entity_token_policy as module

    monkeypatch.setattr(module, "_assert_entity_batch_shapes", lambda *_args: None)
    symmetry = _adversarial_symmetry()
    action_id = next(
        index
        for index in range(symmetry.pi_act.shape[1])
        if len(set(int(v) for v in symmetry.pi_act[:, index])) > 1
    )
    legal_ids = np.asarray([[action_id]], dtype=np.int64)
    context = np.zeros((1, 1, CONTEXT_ACTION_FEATURE_SIZE), dtype=np.float32)
    table = np.zeros((607, 45), dtype=np.float32)
    table[:, STATIC_ACTION_RESIDUAL_SLICE.start] = np.arange(607)
    policy, capture = _capture_policy(table)

    def forward_fn(entity_n, legal_n, context_n, return_q):
        outputs = policy.forward_legal_np(
            entity_n, legal_n, context_n, return_q=return_q
        )
        return {
            "logits": outputs["logits"].numpy(),
            "value": outputs["value"].numpy(),
        }

    result = symmetry.average_forward(
        _symmetry_entity(action_id),
        legal_ids,
        context,
        forward_fn,
        action_size=607,
    )

    assert result["logits_per_orientation"].shape[0] == N_SYMMETRIES
    expected_ids = symmetry.pi_act[:, action_id].astype(np.float32)
    np.testing.assert_array_equal(capture.static_batches[-1][:, 0, 0], expected_ids)


def test_static_residual_rejects_too_narrow_checkpoint_table() -> None:
    with pytest.raises(ValueError, match="at least 41"):
        EntityGraphNet(
            dataclasses.replace(
                _config(),
                static_action_feature_size=40,
                static_action_residual=True,
            )
        )


def test_function_preserving_receipt_accepts_only_two_zero_tensors(tmp_path) -> None:
    from tools import a1_function_preserving_upgrade as upgrade

    source = tmp_path / "source.pt"
    initializer = tmp_path / "static-residual.pt"
    table = np.random.default_rng(31).normal(size=(607, 45)).astype(np.float32)
    base = EntityGraphPolicy(_config(), table, seed=37, device="cpu")
    base.save(source, mask_hidden_info=True)
    treatment = EntityGraphPolicy(
        _config(static_action_residual=True), table, seed=41, device="cpu"
    )
    missing, unexpected = treatment.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    assert not unexpected
    assert set(missing) == {
        "static_action_residual_proj.bias",
        "static_action_residual_proj.weight",
    }
    treatment.save(initializer, mask_hidden_info=True)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix("sha256:"),  # noqa: SLF001
        "flags": {"static_action_residual": True},
        "initialization_seed": 41,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, initializer)

    evidence = upgrade.inspect_upgrade(
        source,
        initializer,
        module=upgrade.MODULE_STATIC_ACTION_RESIDUAL,
    )
    assert evidence["shared_parameters_bit_identical"] is True
    assert evidence["new_parameter_initialization"] == {
        "static_action_residual_proj.bias": "zeros",
        "static_action_residual_proj.weight": "zeros",
    }
