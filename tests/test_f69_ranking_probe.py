from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from catan_zero.rl.entity_feature_adapter import (
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V5,
)
from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
)
from tools import f69_ranking_probe as probe


class _Game:
    @staticmethod
    def current_color() -> str:
        return "BLUE"

    @staticmethod
    def playable_action_indices(_colors, _action_catalog):
        return [10, 20]


class _Tensor:
    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray([[0.25, -0.25]], dtype=np.float32)


def _policy():
    return SimpleNamespace(
        action_size=607,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V5,
        trained_with_masked_hidden_info=True,
        public_award_feature_contract="authoritative_v1",
        config=SimpleNamespace(
            meaningful_public_history=True,
            meaningful_public_history_schema=MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
            event_history_limit=17,
        ),
        forward_legal_np=lambda *_args, **_kwargs: {
            "logits": _Tensor(),
            "q_values": _Tensor(),
        },
    )


def test_root_outputs_use_checkpoint_bound_feature_contract(monkeypatch) -> None:
    observed = {}

    monkeypatch.setattr(
        probe,
        "rust_policy_action_ids",
        lambda *_args, **_kwargs: (101, 202),
    )

    def entity(*_args, **kwargs):
        observed["entity"] = kwargs
        return {"tokens": np.zeros((1, 1, 1), dtype=np.float32)}

    def context(*_args, **kwargs):
        observed["context"] = kwargs
        return np.zeros((1, 2, 1), dtype=np.float32)

    monkeypatch.setattr(probe, "rust_game_to_entity_batch", entity)
    monkeypatch.setattr(probe, "rust_action_context_batch", context)

    out = probe._root_outputs(_policy(), _Game(), context_fill=-3.5)

    assert out["logits"].shape == (2,)
    assert observed["entity"]["public_observation"] is True
    assert observed["entity"]["meaningful_public_history"] is True
    assert observed["entity"]["history_limit"] == 17
    assert (
        observed["entity"]["meaningful_public_history_schema"]
        == MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
    )
    assert (
        observed["entity"]["entity_feature_adapter_version"]
        == RUST_ENTITY_ADAPTER_V5
    )
    assert observed["context"]["fill"] == -3.5
    assert observed["context"]["public_observation"] is True
    assert (
        observed["context"]["entity_feature_adapter_version"]
        == RUST_ENTITY_ADAPTER_V5
    )


def test_feature_contract_caps_meaningful_history_to_checkpoint_schema() -> None:
    policy = _policy()
    policy.config.event_history_limit = 999

    contract = probe._feature_contract(policy, context_fill=-1.25)

    assert contract == {
        "entity_feature_adapter_version": RUST_ENTITY_ADAPTER_V5,
        "public_observation": True,
        "meaningful_public_history": True,
        "meaningful_public_history_schema": MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
        "event_history_limit": 64,
        "action_context_fill": -1.25,
        "public_award_feature_contract": "authoritative_v1",
    }


def test_function_preserving_upgrade_keeps_checkpoint_feature_bindings(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class Config:
        action_target_gather: bool = False

    class Static:
        def detach(self):
            return self

        def cpu(self):
            return self

        @staticmethod
        def numpy():
            return np.zeros((2, 3), dtype=np.float32)

    class Model:
        def __init__(self):
            self.loaded = None
            self.evaluating = False
            self.weight = object()

        def state_dict(self):
            return {"shared.weight": self.weight}

        def load_state_dict(self, state, strict):
            self.loaded = (state, strict)
            return [], []

        def eval(self):
            self.evaluating = True

    observed = {}

    class FakePolicy:
        def __init__(
            self,
            config,
            static,
            *,
            device,
            entity_feature_adapter_version,
        ):
            observed.update(
                {
                    "config": config,
                    "static": static,
                    "device": device,
                    "adapter": entity_feature_adapter_version,
                }
            )
            self.config = config
            self.model = Model()
            self.trained_with_masked_hidden_info = False

    base = SimpleNamespace(
        config=Config(),
        static_action_features=Static(),
        device="cuda:0",
        model=Model(),
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V2,
        entity_feature_adapter_binding_source="checkpoint_metadata",
        trained_with_masked_hidden_info=True,
        public_award_feature_contract="authoritative_v1",
    )
    monkeypatch.setattr(probe, "EntityGraphPolicy", FakePolicy)

    upgraded = probe._upgraded_policy_from(
        base, {"action_target_gather": True}
    )

    assert observed["adapter"] == RUST_ENTITY_ADAPTER_V2
    assert observed["device"] == "cuda:0"
    assert observed["config"].action_target_gather is True
    assert upgraded.trained_with_masked_hidden_info is True
    assert upgraded.public_award_feature_contract == "authoritative_v1"
    assert upgraded.entity_feature_adapter_binding_source == "checkpoint_metadata"
    assert upgraded.model.loaded == (
        {"shared.weight": base.model.weight},
        False,
    )
    assert upgraded.model.evaluating is True
