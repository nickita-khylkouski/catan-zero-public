from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import train_bc  # type: ignore  # noqa: E402
from ema_average_checkpoints import ema_average_checkpoints  # type: ignore  # noqa: E402
from interpolate_checkpoints import interpolate_checkpoints  # type: ignore  # noqa: E402
from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
    ENTITY_FEATURE_ADAPTER_SPECS,
    LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION,
    RUST_ENTITY_ADAPTER_V2,
)
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
    RUST_ENTITY_ADAPTER_VERSION,
)


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


def _raw_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def test_new_checkpoint_roundtrips_explicit_adapter_contract(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    _tiny_policy().save(path)

    raw = _raw_checkpoint(path)
    assert raw["entity_feature_adapter"] == {
        "schema_version": ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
        "version": CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    }
    loaded = EntityGraphPolicy.load(path, device="cpu")
    assert loaded.entity_feature_adapter_version == CURRENT_RUST_ENTITY_ADAPTER_VERSION
    assert loaded.entity_feature_adapter_binding_source == "checkpoint_metadata"
    EntityGraphRustEvaluator(loaded, config=EntityGraphRustEvaluatorConfig())


def test_missing_metadata_maps_to_pinned_legacy_v2_not_runtime_inference(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    legacy = tmp_path / "legacy-f7-shape.pt"
    _tiny_policy().save(source)
    raw = _raw_checkpoint(source)
    raw.pop("entity_feature_adapter")
    torch.save(raw, legacy)

    loaded = EntityGraphPolicy.load(legacy, device="cpu")
    assert LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION == RUST_ENTITY_ADAPTER_V2
    assert loaded.entity_feature_adapter_version == RUST_ENTITY_ADAPTER_V2
    assert (
        loaded.entity_feature_adapter_binding_source
        == "legacy_missing_metadata_explicit_v2_mapping"
    )
    # The deployed pre-metadata lineage continues to run on its exact legacy
    # adapter; this is not an alias to a guessed config/tensor-shape default.
    EntityGraphRustEvaluator(loaded, config=EntityGraphRustEvaluatorConfig())


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"schema_version": "future-schema", "version": RUST_ENTITY_ADAPTER_V2},
        {
            "schema_version": ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
            "version": "future-corrected-longest-road",
        },
    ],
)
def test_present_but_invalid_metadata_never_falls_back_to_legacy(
    tmp_path: Path,
    metadata: object,
) -> None:
    source = tmp_path / "source.pt"
    broken = tmp_path / "broken.pt"
    _tiny_policy().save(source)
    raw = _raw_checkpoint(source)
    raw["entity_feature_adapter"] = metadata
    torch.save(raw, broken)

    with pytest.raises(ValueError, match="adapter|schema|metadata"):
        EntityGraphPolicy.load(broken, device="cpu", strict_metadata=False)


def test_evaluator_rejects_unproven_in_memory_adapter_binding() -> None:
    policy = _tiny_policy()
    policy.entity_feature_adapter_version = "future-corrected-longest-road"
    with pytest.raises(ValueError, match="unknown entity feature adapter"):
        EntityGraphRustEvaluator(policy, config=EntityGraphRustEvaluatorConfig())


def test_training_ddp_fsdp_writer_stamps_same_contract(tmp_path: Path) -> None:
    policy = _tiny_policy()
    path = tmp_path / "distributed.pt"
    train_bc._write_entity_checkpoint(  # noqa: SLF001
        policy,
        str(path),
        policy.model.state_dict(),
        False,
    )
    assert _raw_checkpoint(path)["entity_feature_adapter"] == {
        "schema_version": ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
        "version": RUST_ENTITY_ADAPTER_VERSION,
    }


def test_checkpoint_transforms_normalize_legacy_v2_and_preserve_new_metadata(
    tmp_path: Path,
) -> None:
    new = tmp_path / "new.pt"
    legacy = tmp_path / "legacy.pt"
    _tiny_policy().save(new)
    raw = _raw_checkpoint(new)
    raw.pop("entity_feature_adapter")
    torch.save(raw, legacy)

    averaged = ema_average_checkpoints(
        checkpoints=[legacy, new],
        decay=0.5,
    )
    assert averaged["entity_feature_adapter"]["version"] == RUST_ENTITY_ADAPTER_V2

    [blended] = interpolate_checkpoints(
        base=new,
        candidate=legacy,
        alphas=(0.5,),
        output_template=str(tmp_path / "blend.pt"),
    )
    assert _raw_checkpoint(blended)["entity_feature_adapter"]["version"] == (
        RUST_ENTITY_ADAPTER_V2
    )


def test_v2_semantics_pin_known_omissions_until_a_retrained_v3_exists() -> None:
    spec = ENTITY_FEATURE_ADAPTER_SPECS[RUST_ENTITY_ADAPTER_V2]
    assert spec.player_has_longest_road == "constant_false"
    assert spec.trade_action_type_one_hot == "legacy_case_sensitive_miss"
    assert spec.trade_prompt_one_hot == "legacy_prompt_name_miss"
    assert spec.trade_panel == "offers_remaining_zero_current_offer_none"
    assert spec.context_trade_totals == "legacy_maritime_list_cardinality"
    assert spec.event_history == "empty"
