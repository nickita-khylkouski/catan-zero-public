from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from catan_zero.rl.entity_feature_adapter import RUST_ENTITY_ADAPTER_V4
from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V1,
)
from tools import f74_symmetry_eval as symmetry_eval


def test_root_entity_uses_checkpoint_adapter_and_history(monkeypatch) -> None:
    policy = SimpleNamespace(
        action_size=607,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V4,
        config=SimpleNamespace(
            meaningful_public_history=True,
            meaningful_public_history_schema=MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V1,
            event_history_limit=23,
        ),
    )

    class Game:
        @staticmethod
        def current_color():
            return "BLUE"

        @staticmethod
        def playable_action_indices(_colors, _catalog):
            return [7, 8]

    observed = {}
    monkeypatch.setattr(
        symmetry_eval,
        "rust_policy_action_ids",
        lambda *_args, **_kwargs: (70, 80),
    )

    def entity(*_args, **kwargs):
        observed["entity"] = kwargs
        return {"tokens": np.zeros((1, 1, 1), dtype=np.float32)}

    def context(*_args, **kwargs):
        observed["context"] = kwargs
        return np.zeros((1, 2, 1), dtype=np.float32)

    monkeypatch.setattr(symmetry_eval, "rust_game_to_entity_batch", entity)
    monkeypatch.setattr(symmetry_eval, "rust_action_context_batch", context)

    root = symmetry_eval._root_entity(
        policy,
        Game(),
        public_observation=True,
        context_fill=-2.0,
    )

    assert root["legal_ids"].tolist() == [[70, 80]]
    assert observed["entity"]["public_observation"] is True
    assert observed["entity"]["meaningful_public_history"] is True
    assert observed["entity"]["history_limit"] == 23
    assert (
        observed["entity"]["meaningful_public_history_schema"]
        == MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V1
    )
    assert (
        observed["entity"]["entity_feature_adapter_version"]
        == RUST_ENTITY_ADAPTER_V4
    )
    assert observed["context"]["fill"] == -2.0
    assert observed["context"]["public_observation"] is True
    assert (
        observed["context"]["entity_feature_adapter_version"]
        == RUST_ENTITY_ADAPTER_V4
    )
