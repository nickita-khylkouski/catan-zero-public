"""Checkpoint-to-Rust-search semantic binding for entity features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from catan_zero.rl.entity_feature_adapter import (
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
    EntityFeatureAdapterContractError,
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V6,
    checkpoint_entity_feature_adapter_metadata,
)
from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy
from catan_zero.rl.meaningful_history import MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
from catan_zero.search.eval_server import (
    RemoteEvalClient,
    _require_implemented_entity_feature_adapter,
)
from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_bc  # type: ignore  # noqa: E402


def _tiny_policy() -> EntityGraphPolicy:
    config = EntityGraphConfig(
        action_size=8,
        static_action_feature_size=4,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    return EntityGraphPolicy(
        config,
        np.zeros((8, 4), dtype=np.float32),
        device="cpu",
    )


def _raw(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def test_direct_and_distributed_saves_write_versioned_adapter_metadata(
    tmp_path: Path,
) -> None:
    policy = _tiny_policy()
    expected = checkpoint_entity_feature_adapter_metadata(
        CURRENT_RUST_ENTITY_ADAPTER_VERSION
    )

    direct = tmp_path / "direct.pt"
    policy.save(direct)
    assert _raw(direct)["entity_feature_adapter"] == expected
    loaded = EntityGraphPolicy.load(direct, device="cpu")
    assert (
        loaded.entity_feature_adapter_version
        == CURRENT_RUST_ENTITY_ADAPTER_VERSION
    )
    assert loaded.entity_feature_adapter_binding_source == "checkpoint_metadata"

    distributed = tmp_path / "distributed.pt"
    train_bc._write_entity_checkpoint(
        policy,
        str(distributed),
        policy.model.state_dict(),
        False,
    )
    assert _raw(distributed)["entity_feature_adapter"] == expected


def test_missing_legacy_metadata_maps_explicitly_to_v2_and_resaves_canonically(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.pt"
    resaved = tmp_path / "legacy-resaved.pt"
    payload_path = tmp_path / "source.pt"
    _tiny_policy().save(payload_path)
    payload = _raw(payload_path)
    payload.pop("entity_feature_adapter")
    torch.save(payload, legacy)

    loaded = EntityGraphPolicy.load(legacy, device="cpu")
    assert loaded.entity_feature_adapter_version == RUST_ENTITY_ADAPTER_V2
    assert (
        loaded.entity_feature_adapter_binding_source
        == "legacy_missing_metadata_explicit_v2_mapping"
    )
    evaluator = EntityGraphRustEvaluator(loaded)
    assert evaluator.config.entity_feature_adapter_version == RUST_ENTITY_ADAPTER_V2
    with pytest.raises(ValueError, match="adapter/checkpoint mismatch"):
        EntityGraphRustEvaluator(
            loaded,
            config=EntityGraphRustEvaluatorConfig(
                entity_feature_adapter_version=CURRENT_RUST_ENTITY_ADAPTER_VERSION
            ),
        )
    assert (
        _require_implemented_entity_feature_adapter(
            loaded.entity_feature_adapter_version,
            context="legacy checkpoint regression",
        )
        == RUST_ENTITY_ADAPTER_V2
    )

    loaded.save(resaved)
    assert _raw(resaved)["entity_feature_adapter"] == {
        "schema_version": ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
        "version": RUST_ENTITY_ADAPTER_V2,
    }


def test_v6_checkpoint_roundtrip_and_evaluator_binding_are_exact(
    tmp_path: Path,
) -> None:
    config = EntityGraphConfig(
        action_size=8,
        static_action_feature_size=4,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        meaningful_public_history=True,
        meaningful_public_history_schema=MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
        event_history_limit=64,
        public_rule_state_features=True,
    )
    policy = EntityGraphPolicy(
        config,
        np.zeros((8, 4), dtype=np.float32),
        device="cpu",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6,
    )
    path = tmp_path / "v6.pt"
    policy.save(path)

    assert _raw(path)["entity_feature_adapter"] == (
        checkpoint_entity_feature_adapter_metadata(RUST_ENTITY_ADAPTER_V6)
    )
    loaded = EntityGraphPolicy.load(path, device="cpu")
    assert loaded.entity_feature_adapter_version == RUST_ENTITY_ADAPTER_V6
    evaluator = EntityGraphRustEvaluator(loaded)
    assert evaluator.config.entity_feature_adapter_version == RUST_ENTITY_ADAPTER_V6
    with pytest.raises(ValueError, match="adapter/checkpoint mismatch"):
        EntityGraphRustEvaluator(
            loaded,
            config=EntityGraphRustEvaluatorConfig(
                entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V2
            ),
        )


@pytest.mark.parametrize(
    "malformed",
    [
        CURRENT_RUST_ENTITY_ADAPTER_VERSION,
        {"version": CURRENT_RUST_ENTITY_ADAPTER_VERSION},
        {
            "schema_version": ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
            "version": "future-incompatible-v9",
        },
    ],
)
def test_present_malformed_or_unknown_metadata_is_rejected(
    tmp_path: Path,
    malformed: object,
) -> None:
    source = tmp_path / "source.pt"
    invalid = tmp_path / "invalid.pt"
    _tiny_policy().save(source)
    payload = _raw(source)
    payload["entity_feature_adapter"] = malformed
    torch.save(payload, invalid)

    with pytest.raises(EntityFeatureAdapterContractError):
        EntityGraphPolicy.load(invalid, device="cpu")


def test_remote_eval_client_rejects_unknown_server_adapter() -> None:
    with pytest.raises(EntityFeatureAdapterContractError, match="unknown"):
        RemoteEvalClient(
            object(),
            object(),
            0,
            action_size=8,
            trained_with_masked_hidden_info=False,
            entity_feature_adapter="future-incompatible-v9",
        )
