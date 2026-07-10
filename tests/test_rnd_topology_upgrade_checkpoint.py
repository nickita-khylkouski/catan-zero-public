from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)
from tools.rnd_topology_upgrade_checkpoint import (  # noqa: E402
    SCHEMA_VERSION,
    upgrade_checkpoint,
)


def _checkpoint(path: Path) -> EntityGraphPolicy:
    config = EntityGraphConfig(
        action_size=16,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=2,
        attention_heads=2,
        dropout=0.0,
    )
    policy = EntityGraphPolicy(
        config,
        np.zeros((16, LEGAL_ACTION_FEATURE_SIZE), dtype=np.float32),
        seed=3,
        device="cpu",
    )
    policy.save(path, mask_hidden_info=True, soft_target_source="policy")
    return policy


def test_upgrade_preserves_incumbent_tensors_and_adds_v2(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "upgraded.pt"
    incumbent = _checkpoint(source)

    report = upgrade_checkpoint(
        source,
        output,
        layers="1,2",
        kind="local_attention_v2",
        width=8,
        bases=2,
        heads=2,
    )
    upgraded = EntityGraphPolicy.load(output, device="cpu")

    assert report["schema_version"] == SCHEMA_VERSION
    assert upgraded.config.topology_adapter_layers == "1,2"
    assert upgraded.config.topology_adapter_kind == "local_attention_v2"
    assert upgraded.trained_with_masked_hidden_info
    assert report["parameter_count"] > sum(
        parameter.numel() for parameter in incumbent.model.parameters()
    )
    for name, tensor in incumbent.model.state_dict().items():
        assert torch.equal(upgraded.model.state_dict()[name], tensor)
    assert torch.count_nonzero(upgraded.model.topology_adapters["1"].up.weight) == 0
    assert torch.count_nonzero(upgraded.model.topology_adapters["2"].up.weight) == 0

    raw = torch.load(output, map_location="cpu", weights_only=False)
    provenance = raw["topology_adapter_upgrade"]
    assert provenance["source_checkpoint_sha256"] == report["source_checkpoint_sha256"]
    assert provenance["missing_adapter_tensors"]


def test_upgrade_refuses_in_place_and_double_upgrade(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "upgraded.pt"
    _checkpoint(source)
    with pytest.raises(ValueError, match="differ"):
        upgrade_checkpoint(
            source,
            source,
            layers="1",
            kind="basis_mean_v1",
            width=8,
            bases=2,
            heads=2,
        )
    upgrade_checkpoint(
        source,
        output,
        layers="1",
        kind="basis_mean_v1",
        width=8,
        bases=2,
        heads=2,
    )
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        upgrade_checkpoint(
            source,
            output,
            layers="1",
            kind="basis_mean_v1",
            width=8,
            bases=2,
            heads=2,
        )
    with pytest.raises(ValueError, match="already contains"):
        upgrade_checkpoint(
            output,
            tmp_path / "twice.pt",
            layers="1",
            kind="basis_mean_v1",
            width=8,
            bases=2,
            heads=2,
        )


def test_upgrade_cli_report_is_json_serializable(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "upgraded.pt"
    _checkpoint(source)
    report = upgrade_checkpoint(
        source,
        output,
        layers="1",
        kind="basis_mean_v1",
        width=8,
        bases=2,
        heads=2,
    )
    json.dumps(report, sort_keys=True)


def test_normal_load_rejects_partially_missing_adapter_state(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "upgraded.pt"
    broken = tmp_path / "broken.pt"
    _checkpoint(source)
    upgrade_checkpoint(
        source,
        output,
        layers="1,2",
        kind="local_attention_v2",
        width=8,
        bases=2,
        heads=2,
    )
    raw = torch.load(output, map_location="cpu", weights_only=False)
    removed = next(
        key for key in raw["model"] if key.startswith("topology_adapters.")
    )
    del raw["model"][removed]
    torch.save(raw, broken)

    with pytest.raises(RuntimeError, match="checkpoint state mismatch"):
        EntityGraphPolicy.load(broken, device="cpu")
