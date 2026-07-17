from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from tools import a1_function_preserving_upgrade as upgrade
from tools import f69_upgrade_checkpoint_config as upgrade_tool
from tools import a1_lineage_dose as lineage
from tools import a1_one_dose_train as one_dose
from tools import a1_promotion_transaction as promotion


def test_upgrade_tools_bind_project_imports_to_their_checkout() -> None:
    module = sys.modules[upgrade_tool.EntityGraphPolicy.__module__]
    module_path = Path(str(module.__file__)).resolve(strict=True)

    assert upgrade.REPO_SRC in module_path.parents
    assert upgrade_tool._REPO_SRC in module_path.parents  # noqa: SLF001


def test_upgrade_receipt_authenticates_recomputed_runtime_stamp(
    tmp_path: Path,
) -> None:
    from catan_zero.rl.checkpoint_runtime_semantics import (
        ENTITY_GRAPH_FORWARD_SEMANTICS_KEY,
        current_entity_graph_forward_semantics,
    )
    import catan_zero.rl.entity_token_policy as entity_token_policy

    source, initializer = _checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw[ENTITY_GRAPH_FORWARD_SEMANTICS_KEY] = (
        current_entity_graph_forward_semantics(
            Path(entity_token_policy.__file__)
        )
    )
    torch.save(raw, initializer)

    evidence = upgrade.inspect_upgrade(source, initializer)
    assert evidence["shared_parameters_bit_identical"] is True

    raw[ENTITY_GRAPH_FORWARD_SEMANTICS_KEY] = {
        "schema_version": "entity-graph-forward-semantics-v3",
        "semantic_sha256": "sha256:forged",
    }
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="authenticated current"):
        upgrade.inspect_upgrade(source, initializer)


def test_upgrade_receipt_replays_unchanged_historical_runtime_stamp(
    tmp_path: Path,
) -> None:
    from catan_zero.rl.checkpoint_runtime_semantics import (
        ENTITY_GRAPH_FORWARD_SEMANTICS_KEY,
    )

    source, initializer = _checkpoints(tmp_path)
    historical = {
        "schema_version": "entity-graph-forward-semantics-v3",
        "semantic_token_sha256": "sha256:" + "1" * 64,
    }
    for path in (source, initializer):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        raw[ENTITY_GRAPH_FORWARD_SEMANTICS_KEY] = historical
        torch.save(raw, path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"]["source_checkpoint_sha256"] = upgrade._sha(  # noqa: SLF001
        source
    ).removeprefix("sha256:")
    torch.save(raw, initializer)

    evidence = upgrade.inspect_upgrade(source, initializer)

    assert evidence["shared_parameters_bit_identical"] is True


def _checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "champion.pt"
    upgraded = tmp_path / "champion-gather.pt"
    base_config = {
        "state_trunk": "transformer",
        "action_size": 567,
        "static_action_feature_size": 1,
    }
    base_model = {
        "encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "policy.weight": torch.ones(2, 2),
    }
    torch.save(
        {"config": {"fields": base_config}, "model": base_model, "epoch": 7},
        source,
    )
    model = dict(base_model)
    model.update(
        {
            "target_gather_proj.0.bias": torch.zeros(3),
            "target_gather_proj.0.weight": torch.ones(3),
            "target_gather_proj.1.bias": torch.zeros(3),
            "target_gather_proj.1.weight": torch.zeros(3, 3),
        }
    )
    torch.save(
        {
            "config": {"fields": {**base_config, "action_target_gather": True}},
            "model": model,
            "epoch": 7,
            "upgrade_provenance": {
                "schema_version": "entity-graph-upgrade-v1",
                "source_checkpoint_sha256": upgrade._sha(source).removeprefix(  # noqa: SLF001
                    "sha256:"
                ),
                "flags": {"action_target_gather": True},
                "initialization_seed": 1,
                "trained_value_readouts_added": [],
                "forward_max_diff": 0.0,
                "forward_tolerance": 0.0,
                "forward_identical_at_init": True,
            },
        },
        upgraded,
    )
    return source, upgraded


def _issued(tmp_path: Path) -> tuple[Path, dict]:
    source, initializer = _checkpoints(tmp_path)
    receipt = tmp_path / "upgrade.receipt.json"
    payload = upgrade.issue_receipt(source, initializer, receipt)
    return receipt, payload


def _value_tower_split_checkpoints(
    tmp_path: Path, *, source_overrides: dict[str, object] | None = None
) -> tuple[Path, Path]:
    source = tmp_path / "value-tower-source.pt"
    upgraded = tmp_path / "value-tower-split.pt"
    spec = upgrade.ALLOWLIST[upgrade.MODULE_VALUE_TOWER_SPLIT_1]
    source_config = {
        "action_size": 567,
        "static_action_feature_size": 1,
        "state_trunk": "transformer",
        "state_layers": 6,
        "value_tower_split_layers": 0,
        "latent_deliberation_steps": 0,
        **(source_overrides or {}),
    }
    source_model = {"shared.weight": torch.arange(4, dtype=torch.float32)}
    for target_name, initializer in spec["new_parameter_initialization"].items():
        source_name = initializer.removeprefix("source_clone:")
        source_model.setdefault(
            source_name,
            torch.full((2,), float(len(source_model)), dtype=torch.float32),
        )
    torch.save(
        {"config": {"fields": source_config}, "model": source_model, "epoch": 7},
        source,
    )
    upgraded_model = dict(source_model)
    for target_name, initializer in spec["new_parameter_initialization"].items():
        upgraded_model[target_name] = source_model[
            initializer.removeprefix("source_clone:")
        ].clone()
    torch.save(
        {
            "config": {
                "fields": {**source_config, "value_tower_split_layers": 1}
            },
            "model": upgraded_model,
            "epoch": 7,
            "upgrade_provenance": {
                "schema_version": "entity-graph-upgrade-v1",
                "source_checkpoint_sha256": upgrade._sha(source).removeprefix(  # noqa: SLF001
                    "sha256:"
                ),
                "flags": {"value_tower_split_layers": 1},
                "initialization_seed": 1,
                "trained_value_readouts_added": [],
                "forward_max_diff": 0.0,
                "forward_tolerance": float(torch.finfo(torch.float32).eps),
                "forward_identical_at_init": True,
            },
        },
        upgraded,
    )
    return source, upgraded


def _topology_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    source, _gather = _checkpoints(tmp_path)
    upgraded = tmp_path / "champion-topology-gather.pt"
    raw = torch.load(source, map_location="cpu", weights_only=False)
    model = dict(raw["model"])
    width = 3
    spec = upgrade.ALLOWLIST[upgrade.MODULE_TOPOLOGY_TARGET_GATHER]
    for name, kind in spec["new_parameter_initialization"].items():
        if name.endswith(".weight") and (
            "norm." not in name and "target_gather_proj.0" not in name
        ):
            shape = (width, width)
        else:
            shape = (width,)
        if kind == "ones":
            tensor = torch.ones(shape)
        elif kind == "zeros":
            tensor = torch.zeros(shape)
        elif kind == "identity":
            tensor = torch.eye(width)
        else:  # pragma: no cover - the allowlist itself is closed above
            raise AssertionError(kind)
        model[name] = tensor
    flags = {
        "action_target_gather": True,
        "topology_residual_adapter": True,
    }
    raw["model"] = model
    raw["config"] = {"fields": {**raw["config"]["fields"], **flags}}
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix("sha256:"),  # noqa: SLF001
        "flags": flags,
        "initialization_seed": 1,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, upgraded)
    return source, upgraded


def _belief_checkpoints(tmp_path: Path, *, seed: int = 73) -> tuple[Path, Path]:
    """Build the real additive belief-head upgrade used by the learner."""
    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    source = tmp_path / "champion-real.pt"
    output = tmp_path / "champion-belief.pt"
    base = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=11,
        device="cpu",
    )
    base.save(source, mask_hidden_info=True)
    values = {
        field.name: getattr(base.config, field.name)
        for field in dataclasses.fields(EntityGraphConfig)
        if hasattr(base.config, field.name)
    }
    values["belief_resource_head"] = True
    belief = EntityGraphPolicy(
        EntityGraphConfig(**values),
        base.static_action_features.detach().cpu().numpy(),
        seed=seed,
        device="cpu",
    )
    missing, unexpected = belief.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    assert not unexpected
    assert set(missing) == set(
        upgrade.ALLOWLIST[upgrade.MODULE_BELIEF_RESOURCE_HEAD][
            "new_parameter_initialization"
        ]
    )
    belief.save(output, mask_hidden_info=True)
    raw = torch.load(output, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix("sha256:"),  # noqa: SLF001
        "flags": {"belief_resource_head": True},
        "initialization_seed": seed,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, output)
    return source, output


def _structured_action_value_checkpoints(
    tmp_path: Path, *, seed: int = 83
) -> tuple[Path, Path]:
    from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy

    source = tmp_path / "champion-structured-source.pt"
    output = tmp_path / "champion-structured-value.pt"
    config = EntityGraphConfig(
        action_size=16,
        static_action_feature_size=45,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    static = np.zeros((config.action_size, config.static_action_feature_size))
    base = EntityGraphPolicy(config, static, seed=11, device="cpu")
    base.save(source, mask_hidden_info=True)
    treatment = EntityGraphPolicy(
        dataclasses.replace(
            config,
            static_action_residual=True,
            legal_action_value_residual=True,
        ),
        static,
        seed=seed,
        device="cpu",
    )
    missing, unexpected = treatment.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    assert not unexpected
    spec = upgrade.ALLOWLIST[upgrade.MODULE_STRUCTURED_ACTION_VALUE]
    assert set(missing) == set(spec["new_parameter_initialization"])
    treatment.save(output, mask_hidden_info=True)
    raw = torch.load(output, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix("sha256:"),  # noqa: SLF001
        "flags": dict(spec["flags"]),
        "initialization_seed": seed,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, output)
    return source, output


def _canonical_v3_checkpoints(
    tmp_path: Path, *, seed: int = 89
) -> tuple[Path, Path]:
    from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy

    source = tmp_path / "champion-canonical-v3-source.pt"
    output = tmp_path / "champion-canonical-v3.pt"
    config = EntityGraphConfig(
        action_size=16,
        static_action_feature_size=45,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    static = np.zeros((config.action_size, config.static_action_feature_size))
    base = EntityGraphPolicy(config, static, seed=13, device="cpu")
    base.save(source, mask_hidden_info=True)
    spec = upgrade.ALLOWLIST[
        upgrade.MODULE_STRUCTURED_ACTION_VALUE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V3
    ]
    treatment = EntityGraphPolicy(
        dataclasses.replace(config, **spec["config_delta"]),
        static,
        seed=seed,
        device="cpu",
    )
    missing, unexpected = treatment.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    assert not unexpected
    assert set(missing) == set(spec["new_parameter_initialization"])
    treatment.save(output, mask_hidden_info=True)
    raw = torch.load(output, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix("sha256:"),  # noqa: SLF001
        "flags": dict(spec["flags"]),
        "initialization_seed": seed,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, output)
    return source, output


def _canonical_v4_checkpoints(
    tmp_path: Path, *, seed: int = 97
) -> tuple[Path, Path]:
    from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_feature_adapter import RUST_ENTITY_ADAPTER_V4
    from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy

    source = tmp_path / "champion-canonical-v4-source.pt"
    output = tmp_path / "champion-canonical-v4.pt"
    config = EntityGraphConfig(
        action_size=16,
        static_action_feature_size=45,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    static = np.zeros((config.action_size, config.static_action_feature_size))
    base = EntityGraphPolicy(config, static, seed=13, device="cpu")
    base.save(source, mask_hidden_info=True)
    spec = upgrade.ALLOWLIST[
        upgrade.MODULE_STRUCTURED_ACTION_VALUE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_RULE_STATE_V4
    ]
    treatment = EntityGraphPolicy(
        dataclasses.replace(config, **spec["config_delta"]),
        static,
        seed=seed,
        device="cpu",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V4,
    )
    missing, unexpected = treatment.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    assert not unexpected
    assert set(missing) == set(spec["new_parameter_initialization"])
    treatment.save(output, mask_hidden_info=True)
    raw = torch.load(output, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix("sha256:"),  # noqa: SLF001
        "flags": dict(spec["flags"]),
        "initialization_seed": seed,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, output)
    return source, output


def _aux_checkpoints(tmp_path: Path, *, seed: int = 79) -> tuple[Path, Path]:
    """Build the exact shared auxiliary initializer for both matched arms."""

    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    source = tmp_path / "champion-aux-source.pt"
    output = tmp_path / "champion-aux.pt"
    base = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=11,
        device="cpu",
    )
    base.save(source, mask_hidden_info=True)
    values = {
        field.name: getattr(base.config, field.name)
        for field in dataclasses.fields(EntityGraphConfig)
        if hasattr(base.config, field.name)
    }
    values["aux_subgoal_heads"] = True
    aux = EntityGraphPolicy(
        EntityGraphConfig(**values),
        base.static_action_features.detach().cpu().numpy(),
        seed=seed,
        device="cpu",
    )
    missing, unexpected = aux.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    assert not unexpected
    assert set(missing) == set(
        upgrade.ALLOWLIST[upgrade.MODULE_AUX_SUBGOAL_HEADS][
            "new_parameter_initialization"
        ]
    )
    aux.save(output, mask_hidden_info=True)
    raw = torch.load(output, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix("sha256:"),  # noqa: SLF001
        "flags": {"aux_subgoal_heads": True},
        "initialization_seed": seed,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, output)
    return source, output


def _tamper_and_rehash(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = upgrade._digest(value)  # noqa: SLF001
    path.write_text(json.dumps(value), encoding="utf-8")


def test_receipt_replays_exact_allowlisted_zero_diff_upgrade(tmp_path: Path) -> None:
    receipt, payload = _issued(tmp_path)
    verified = upgrade.verify_receipt(receipt)
    assert verified["receipt_sha256"] == payload["receipt_sha256"]
    assert verified["module"] == upgrade.MODULE_TARGET_GATHER
    assert verified["forward_max_diff"] == 0.0
    assert verified["new_parameters"] == sorted(
        upgrade.ALLOWLIST[upgrade.MODULE_TARGET_GATHER][
            "new_parameter_initialization"
        ]
    )
    with pytest.raises(upgrade.UpgradeError, match="overwrite"):
        upgrade.issue_receipt(
            Path(payload["source"]["path"]),
            Path(payload["upgraded_initializer"]["path"]),
            receipt,
        )


def test_receipt_replays_structured_action_value_upgrade(tmp_path: Path) -> None:
    source, initializer = _structured_action_value_checkpoints(tmp_path)
    receipt = tmp_path / "structured-action-value.receipt.json"

    issued = upgrade.issue_receipt(
        source,
        initializer,
        receipt,
        module=upgrade.MODULE_STRUCTURED_ACTION_VALUE,
    )
    verified = upgrade.verify_receipt(receipt)

    assert verified["receipt_sha256"] == issued["receipt_sha256"]
    assert verified["module"] == upgrade.MODULE_STRUCTURED_ACTION_VALUE
    assert verified["flags"] == {
        "static_action_residual": True,
        "legal_action_value_residual": True,
    }
    assert verified["forward_max_diff"] == 0.0
    assert verified["new_parameters"] == sorted(
        upgrade.ALLOWLIST[upgrade.MODULE_STRUCTURED_ACTION_VALUE][
            "new_parameter_initialization"
        ]
    )


def test_canonical_v3_upgrade_combines_all_zero_output_repairs() -> None:
    spec = upgrade.ALLOWLIST[
        upgrade.MODULE_STRUCTURED_ACTION_VALUE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V3
    ]
    assert spec["flags"] == {
        "static_action_residual": True,
        "legal_action_value_residual": True,
        "public_card_count_features": True,
        "public_card_count_residual_bias": False,
        "meaningful_public_history": True,
        "meaningful_public_history_schema": (
            upgrade.MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
        ),
        "event_history_limit": upgrade.MEANINGFUL_PUBLIC_HISTORY_LIMIT,
    }
    assert set(spec["new_parameter_initialization"]) == {
        "legal_action_value_residual_proj.weight",
        "legal_action_value_static_proj.weight",
        "static_action_residual_proj.bias",
        "static_action_residual_proj.weight",
        "public_card_count_residual.weight",
        "meaningful_history_residual_gate",
    }
    assert set(spec["new_parameter_initialization"].values()) == {"zeros"}


def test_canonical_v3_receipt_replays_exact_combined_initializer(
    tmp_path: Path,
) -> None:
    source, initializer = _canonical_v3_checkpoints(tmp_path)
    receipt = tmp_path / "canonical-v3.receipt.json"
    issued = upgrade.issue_receipt(
        source,
        initializer,
        receipt,
        module=(
            upgrade.MODULE_STRUCTURED_ACTION_VALUE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V3
        ),
    )
    verified = upgrade.verify_receipt(receipt)
    assert verified["receipt_sha256"] == issued["receipt_sha256"]
    assert verified["forward_max_diff"] == 0.0
    assert verified["shared_parameters_bit_identical"] is True


def test_canonical_v4_receipt_replays_actor_rule_state_adapter_transition(
    tmp_path: Path,
) -> None:
    source, initializer = _canonical_v4_checkpoints(tmp_path)
    receipt = tmp_path / "canonical-v4.receipt.json"
    issued = upgrade.issue_receipt(
        source,
        initializer,
        receipt,
        module=(
            upgrade.MODULE_STRUCTURED_ACTION_VALUE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_RULE_STATE_V4
        ),
    )
    verified = upgrade.verify_receipt(receipt)
    spec = upgrade.ALLOWLIST[
        upgrade.MODULE_STRUCTURED_ACTION_VALUE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_RULE_STATE_V4
    ]

    assert verified["receipt_sha256"] == issued["receipt_sha256"]
    assert verified["forward_max_diff"] == 0.0
    assert verified["shared_parameters_bit_identical"] is True
    assert verified["flags"] == spec["flags"]
    assert verified["entity_feature_adapter_version"] == (
        "rust_entity_adapter_v4_actor_public_rule_state"
    )
    assert "public_rule_state_residual.weight" in verified["new_parameters"]


def test_bias_free_public_card_v2_upgrade_has_only_zero_weight_delta(
    tmp_path: Path,
) -> None:
    source, _gather = _checkpoints(tmp_path)
    initializer = tmp_path / "champion-public-card-v2.pt"
    raw = torch.load(source, map_location="cpu", weights_only=False)
    raw["model"] = {
        **raw["model"],
        "public_card_count_residual.weight": torch.zeros(3, 11),
    }
    flags = {
        "public_card_count_features": True,
        "public_card_count_residual_bias": False,
    }
    raw["config"] = {"fields": {**raw["config"]["fields"], **flags}}
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix(  # noqa: SLF001
            "sha256:"
        ),
        "flags": flags,
        "initialization_seed": 1,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, initializer)

    evidence = upgrade.inspect_upgrade(
        source,
        initializer,
        module=upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES_V2,
    )
    assert evidence["new_parameters"] == [
        "public_card_count_residual.weight"
    ]
    assert evidence["new_parameter_initialization"] == {
        "public_card_count_residual.weight": "zeros"
    }
    assert evidence["flags"] == flags


def test_default_true_card_bias_is_omitted_only_from_legacy_receipt_digest_view():
    legacy = {"state_layers": 6, "public_card_count_residual_bias": True}
    bias_free = {"state_layers": 6, "public_card_count_residual_bias": False}

    assert upgrade._effective_config_receipt_view(legacy) == {  # noqa: SLF001
        "state_layers": 6
    }
    assert upgrade._effective_config_receipt_view(bias_free) == bias_free  # noqa: SLF001


def test_legacy_full_action_cross_topology_keeps_existing_receipt_digest_view():
    legacy = {"state_layers": 6}
    reconstructed = {
        "state_layers": 6,
        "action_cross_attention_bottleneck": 0,
    }
    budgeted_v7 = {
        "state_layers": 6,
        "action_cross_attention_bottleneck": 80,
    }

    assert upgrade._effective_config_receipt_view(reconstructed) == legacy  # noqa: SLF001
    assert upgrade._effective_config_receipt_view(budgeted_v7) == budgeted_v7  # noqa: SLF001


def test_receipt_replays_combined_topology_target_gather_upgrade(tmp_path: Path) -> None:
    source, initializer = _topology_checkpoints(tmp_path)
    evidence = upgrade.inspect_upgrade(
        source,
        initializer,
        module=upgrade.MODULE_TOPOLOGY_TARGET_GATHER,
    )
    assert evidence["flags"] == {
        "action_target_gather": True,
        "topology_residual_adapter": True,
    }
    assert evidence["new_parameter_initialization"][
        "topology_residual_adapter.source_projection.weight"
    ] == "identity"


def test_value_tower_split_receipt_binds_exact_six_layer_source(tmp_path: Path) -> None:
    source, initializer = _value_tower_split_checkpoints(tmp_path)

    evidence = upgrade.inspect_upgrade(
        source,
        initializer,
        module=upgrade.MODULE_VALUE_TOWER_SPLIT_1,
    )

    assert evidence["flags"] == {"value_tower_split_layers": 1}
    assert evidence["forward_tolerance"] == float(torch.finfo(torch.float32).eps)
    assert set(evidence["new_parameters"]) == set(
        upgrade.ALLOWLIST[upgrade.MODULE_VALUE_TOWER_SPLIT_1][
            "new_parameter_initialization"
        ]
    )


def test_value_tower_split_receipt_accepts_one_fp32_epsilon(tmp_path: Path) -> None:
    source, initializer = _value_tower_split_checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"]["forward_max_diff"] = float(
        torch.finfo(torch.float32).eps
    )
    torch.save(raw, initializer)

    evidence = upgrade.inspect_upgrade(
        source,
        initializer,
        module=upgrade.MODULE_VALUE_TOWER_SPLIT_1,
    )

    assert evidence["forward_max_diff"] == float(torch.finfo(torch.float32).eps)


def test_value_tower_split_receipt_rejects_more_than_one_fp32_epsilon(
    tmp_path: Path,
) -> None:
    source, initializer = _value_tower_split_checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"]["forward_max_diff"] = 2.0 * float(
        torch.finfo(torch.float32).eps
    )
    torch.save(raw, initializer)

    with pytest.raises(upgrade.UpgradeError, match="typed forward tolerance"):
        upgrade.inspect_upgrade(
            source,
            initializer,
            module=upgrade.MODULE_VALUE_TOWER_SPLIT_1,
        )


def test_current_v5_split1_is_one_direct_reviewed_upgrade_edge() -> None:
    composite = upgrade.ALLOWLIST[
        upgrade.MODULE_CURRENT_V5_VALUE_TOWER_SPLIT_1
    ]
    current = upgrade.ALLOWLIST[upgrade.MODULE_CURRENT_V5]
    split = upgrade.ALLOWLIST[upgrade.MODULE_VALUE_TOWER_SPLIT_1]

    assert composite["flags"] == {**current["flags"], **split["flags"]}
    assert composite["config_delta"] == {
        **current["config_delta"],
        **split["config_delta"],
    }
    assert composite["new_parameter_initialization"] == {
        **current["new_parameter_initialization"],
        **split["new_parameter_initialization"],
    }
    assert composite["source_config_requirements"] == split[
        "source_config_requirements"
    ]
    assert upgrade_tool._parse_flags("current_v5_split1") == composite["flags"]  # noqa: SLF001
    assert upgrade_tool._parse_flags(  # noqa: SLF001
        "gather,static,legal_action_value_residual,value_set_statistics,"
        "card_count_v2,history_v2,history_target_gather,public_rule_state,"
        "value_split:1"
    ) == composite["flags"]


def test_topology_aware_current_v5_split1_is_a_new_reviewed_upgrade_edge() -> None:
    historical = upgrade.ALLOWLIST[
        upgrade.MODULE_CURRENT_V5_VALUE_TOWER_SPLIT_1
    ]
    selected = upgrade.ALLOWLIST[
        upgrade.MODULE_CURRENT_V5_TOPOLOGY_VALUE_TOWER_SPLIT_1
    ]

    assert selected["flags"] == {
        **historical["flags"],
        "topology_residual_adapter": True,
    }
    assert selected["config_delta"] == {
        **historical["config_delta"],
        "topology_residual_adapter": True,
    }
    topology_initializers = {
        name: initializer
        for name, initializer in selected["new_parameter_initialization"].items()
        if name.startswith("topology_residual_adapter.")
    }
    assert topology_initializers == {
        "topology_residual_adapter.message_norm.bias": "zeros",
        "topology_residual_adapter.message_norm.weight": "ones",
        "topology_residual_adapter.output_projection.bias": "zeros",
        "topology_residual_adapter.output_projection.weight": "zeros",
        "topology_residual_adapter.source_norm.bias": "zeros",
        "topology_residual_adapter.source_norm.weight": "ones",
        "topology_residual_adapter.source_projection.bias": "zeros",
        "topology_residual_adapter.source_projection.weight": "identity",
    }
    assert upgrade_tool._parse_flags(  # noqa: SLF001
        "current_v5_topology_split1"
    ) == selected["flags"]


def test_adapter_v6_is_refused_as_a_function_preserving_upgrade() -> None:
    assert all("v6" not in module for module in upgrade.ALLOWLIST)
    with pytest.raises(SystemExit, match="not function preserving"):
        upgrade_tool._parse_flags("current_v6_topology_split1")  # noqa: SLF001


def test_b12_topology_upgrade_adds_only_zero_output_topology_parameters() -> None:
    selected = upgrade.ALLOWLIST[
        upgrade.MODULE_CURRENT_V5_SPLIT1_TOPOLOGY_ONLY
    ]
    current = upgrade.ALLOWLIST[upgrade.MODULE_CURRENT_V5]

    assert selected["flags"] == {"topology_residual_adapter": True}
    assert selected["config_delta"] == {"topology_residual_adapter": True}
    assert selected["source_config_requirements"] == {
        **current["flags"],
        "state_trunk": "transformer",
        "state_layers": 6,
        "value_tower_split_layers": 1,
        "latent_deliberation_steps": 0,
        "topology_residual_adapter": False,
    }
    assert set(selected["new_parameter_initialization"]) == {
        "topology_residual_adapter.message_norm.bias",
        "topology_residual_adapter.message_norm.weight",
        "topology_residual_adapter.output_projection.bias",
        "topology_residual_adapter.output_projection.weight",
        "topology_residual_adapter.source_norm.bias",
        "topology_residual_adapter.source_norm.weight",
        "topology_residual_adapter.source_projection.bias",
        "topology_residual_adapter.source_projection.weight",
    }


@pytest.mark.parametrize(
    "source_overrides",
    [
        {"state_layers": 8},
        {"state_trunk": "rrt"},
        {"value_tower_split_layers": 1},
        {"latent_deliberation_steps": 1},
    ],
)
def test_value_tower_split_receipt_rejects_wrong_source_topology(
    tmp_path: Path, source_overrides: dict[str, object]
) -> None:
    source, initializer = _value_tower_split_checkpoints(
        tmp_path, source_overrides=source_overrides
    )

    with pytest.raises(upgrade.UpgradeError, match="module preconditions"):
        upgrade.inspect_upgrade(
            source,
            initializer,
            module=upgrade.MODULE_VALUE_TOWER_SPLIT_1,
        )


@pytest.mark.parametrize("which", ("source", "upgraded"))
def test_receipt_refuses_config_fields_unknown_to_its_checkout(
    tmp_path: Path, which: str
) -> None:
    source, initializer = _checkpoints(tmp_path)
    path = source if which == "source" else initializer
    raw = torch.load(path, map_location="cpu", weights_only=False)
    raw["config"]["fields"]["future_topology_adapter"] = True
    torch.save(raw, path)
    if which == "source":
        upgraded = torch.load(initializer, map_location="cpu", weights_only=False)
        upgraded["upgrade_provenance"]["source_checkpoint_sha256"] = upgrade._sha(  # noqa: SLF001
            source
        ).removeprefix("sha256:")
        torch.save(upgraded, initializer)

    with pytest.raises(upgrade.UpgradeError, match="fields unknown to this checkout"):
        upgrade.inspect_upgrade(source, initializer)


def test_receipt_replays_seeded_belief_head_upgrade(tmp_path: Path) -> None:
    source, initializer = _belief_checkpoints(tmp_path)
    receipt = tmp_path / "belief-upgrade.receipt.json"
    payload = upgrade.issue_receipt(
        source,
        initializer,
        receipt,
        module=upgrade.MODULE_BELIEF_RESOURCE_HEAD,
    )
    verified = upgrade.verify_receipt(receipt)
    expected_seeded = {
        name
        for name, kind in upgrade.ALLOWLIST[upgrade.MODULE_BELIEF_RESOURCE_HEAD][
            "new_parameter_initialization"
        ].items()
        if kind == "seeded_torch_default"
    }
    assert payload["initialization_seed"] == 73
    assert set(verified["seeded_parameter_sha256"]) == expected_seeded
    assert verified["shared_parameters_bit_identical"] is True


def test_receipt_replays_shared_seeded_aux_head_upgrade(tmp_path: Path) -> None:
    source, initializer = _aux_checkpoints(tmp_path)
    receipt = tmp_path / "aux-upgrade.receipt.json"
    payload = upgrade.issue_receipt(
        source,
        initializer,
        receipt,
        module=upgrade.MODULE_AUX_SUBGOAL_HEADS,
    )
    verified = upgrade.verify_receipt(receipt)
    assert verified["module"] == upgrade.MODULE_AUX_SUBGOAL_HEADS
    assert verified["upgraded_initializer"] == payload["upgraded_initializer"]
    assert len(verified["new_parameters"]) == 20
    assert set(verified["seeded_parameter_sha256"]) == set(
        verified["new_parameters"]
    )


def test_aux_receipt_rejects_substituted_initializer_bytes(tmp_path: Path) -> None:
    source, initializer = _aux_checkpoints(tmp_path)
    receipt = tmp_path / "aux-upgrade.receipt.json"
    upgrade.issue_receipt(
        source,
        initializer,
        receipt,
        module=upgrade.MODULE_AUX_SUBGOAL_HEADS,
    )
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["model"]["aux_vp_in_n_head.3.bias"][0] += 0.01
    torch.save(raw, initializer)
    with pytest.raises(
        upgrade.UpgradeError,
        match="deterministic seeded_torch_default|does not replay exactly",
    ):
        upgrade.verify_receipt(receipt)


def test_receipt_accepts_belief_head_from_real_seeded_upgrader(
    tmp_path: Path, monkeypatch
) -> None:
    source, _ = _belief_checkpoints(tmp_path)
    initializer = tmp_path / "belief-real-upgrader.pt"
    monkeypatch.setattr(upgrade_tool, "_verify_forward_identical", lambda *_: 0.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "f69_upgrade_checkpoint_config.py",
            "--in-checkpoint",
            str(source),
            "--out-checkpoint",
            str(initializer),
            "--flags",
            "belief",
            "--seed",
            "73",
            "--device",
            "cpu",
        ],
    )
    upgrade_tool.main()
    evidence = upgrade.inspect_upgrade(
        source,
        initializer,
        module=upgrade.MODULE_BELIEF_RESOURCE_HEAD,
    )
    assert evidence["initialization_seed"] == 73


def test_belief_receipt_rejects_wrong_seed_or_tampered_random_head(tmp_path: Path) -> None:
    source, initializer = _belief_checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"]["initialization_seed"] = 72
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="deterministic seeded_torch_default"):
        upgrade.inspect_upgrade(
            source,
            initializer,
            module=upgrade.MODULE_BELIEF_RESOURCE_HEAD,
        )

    source, initializer = _belief_checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["model"]["belief_resource_head.1.weight"][0, 0] += 0.01
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="deterministic seeded_torch_default"):
        upgrade.inspect_upgrade(
            source,
            initializer,
            module=upgrade.MODULE_BELIEF_RESOURCE_HEAD,
        )


def test_receipt_digest_normalizes_numpy_config_scalars(tmp_path: Path) -> None:
    source, initializer = _checkpoints(tmp_path)
    for path in (source, initializer):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        raw["config"]["fields"]["action_size"] = np.int64(567)
        torch.save(raw, path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"]["source_checkpoint_sha256"] = upgrade._sha(  # noqa: SLF001
        source
    ).removeprefix("sha256:")
    torch.save(raw, initializer)

    receipt = tmp_path / "numpy-config.receipt.json"
    payload = upgrade.issue_receipt(source, initializer, receipt)
    assert upgrade.verify_receipt(receipt)["receipt_sha256"] == payload["receipt_sha256"]
    assert upgrade._digest({"value": np.int64(7)}) == upgrade._digest(  # noqa: SLF001
        {"value": 7}
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["source"].__setitem__("sha256", "sha256:" + "1" * 64),
        lambda value: value["upgraded_initializer"].__setitem__(
            "sha256", "sha256:" + "2" * 64
        ),
        lambda value: value.__setitem__("flags", {"action_target_gather": False}),
        lambda value: value.__setitem__("forward_max_diff", 1e-12),
        lambda value: value["new_parameters"].append("attacker.weight"),
    ],
    ids=("source", "initializer", "flags", "nonzero-diff", "new-key"),
)
def test_semantically_rehashed_receipt_tampering_is_rejected(
    tmp_path: Path, mutate
) -> None:
    receipt, _ = _issued(tmp_path)
    _tamper_and_rehash(receipt, mutate)
    with pytest.raises(upgrade.UpgradeError, match="does not replay exactly"):
        upgrade.verify_receipt(receipt)


def test_checkpoint_parameter_or_metadata_drift_is_rejected(tmp_path: Path) -> None:
    source, initializer = _checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["model"]["target_gather_proj.1.weight"][0, 0] = 0.01
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="deterministic zeros"):
        upgrade.inspect_upgrade(source, initializer)

    source, initializer = _checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["epoch"] = 8
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="metadata/provenance changed"):
        upgrade.inspect_upgrade(source, initializer)

    source, initializer = _checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["model"]["encoder.weight"] = raw["model"]["encoder.weight"].double()
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="shared checkpoint parameters changed"):
        upgrade.inspect_upgrade(source, initializer)

    source, initializer = _checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["mask_hidden_info"] = True
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="metadata/provenance changed"):
        upgrade.inspect_upgrade(source, initializer)


def test_one_dose_binds_upgraded_init_and_typed_lineage(tmp_path: Path) -> None:
    receipt, payload = _issued(tmp_path)
    verified = {
        "producer": payload["source"],
        "contract_sha256": "sha256:" + "c" * 64,
        "recipe": {
            "resume_optimizer": False,
            "batch_size": 512,
            "grad_accum_steps": 1,
            "world_size": 8,
            "global_batch_size": 4096,
            "max_steps": 1024,
        },
        "training_row_count": 4_194_304,
    }
    bound = one_dose.bind_function_preserving_upgrade(verified, receipt)
    dose = one_dose._direct_lineage_dose(bound)  # noqa: SLF001
    assert dose["declared_producer_sha256"] == payload["source"]["sha256"]
    assert dose["init_checkpoint_sha256"] == payload["upgraded_initializer"]["sha256"]
    assert dose["function_preserving_upgrade"]["receipt_sha256"] == upgrade._sha(  # noqa: SLF001
        receipt
    )
    assert lineage.validate_lineage_dose(dose) == dose


def test_exact_parent_remains_default_and_untyped_delta_is_refused() -> None:
    producer = "sha256:" + "a" * 64
    other = "sha256:" + "b" * 64
    exact = lineage.direct_lineage_dose(
        declared_producer_sha256=producer,
        init_checkpoint_sha256=producer,
        current_sampled_rows=10,
        current_optimizer_steps=1,
    )
    assert exact["function_preserving_upgrade"] is None
    with pytest.raises(lineage.LineageDoseError, match="untyped checkpoint chaining"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=producer,
            init_checkpoint_sha256=other,
            current_sampled_rows=10,
            current_optimizer_steps=1,
        )


def test_promotion_report_accepts_only_replayed_upgrade_lineage(tmp_path: Path) -> None:
    receipt, payload = _issued(tmp_path)
    verified = {
        "producer": payload["source"],
        "contract_sha256": "sha256:" + "c" * 64,
        "recipe": {
            "resume_optimizer": False,
            "batch_size": 512,
            "grad_accum_steps": 1,
            "world_size": 8,
            "global_batch_size": 4096,
            "max_steps": 1024,
        },
        "training_row_count": 4_194_304,
    }
    bound = one_dose.bind_function_preserving_upgrade(verified, receipt)
    dose = one_dose._direct_lineage_dose(bound)  # noqa: SLF001
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"trained candidate")
    recipe = {"epochs": 1, "max_steps": 1024, "symmetry_augment": False}
    contract = {
        "contract_sha256": verified["contract_sha256"],
        "science": {
            "learner_training_recipe": recipe,
            "learner_training_recipe_sha256": promotion._digest_value(recipe),  # noqa: SLF001
        },
        "checkpoints": [{"role": "producer", "sha256": payload["source"]["sha256"]}],
    }
    report_value = {
        "a1_contract_sha256": verified["contract_sha256"],
        "a1_learner_training_recipe_sha256": promotion._digest_value(recipe),  # noqa: SLF001
        "a1_bound_learner_training_recipe": recipe,
        "arch": "entity_graph",
        "mask_hidden_info": True,
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "symmetry_augment": False,
        "checkpoint": str(candidate),
        "init_checkpoint": payload["upgraded_initializer"]["path"],
        "init_checkpoint_sha256": payload["upgraded_initializer"]["sha256"],
        "a1_lineage_dose": dose,
        "steps_completed": 1024,
        "epochs": 1,
        "max_steps": 1024,
    }
    report = tmp_path / "report.json"
    report.write_text(json.dumps(report_value), encoding="utf-8")
    assert promotion._verify_training_report(  # noqa: SLF001
        report,
        contract=contract,
        contract_sha256=verified["contract_sha256"],
        candidate_path=candidate,
        candidate_sha256=promotion._sha256(candidate),  # noqa: SLF001
    ) == report_value

    report_value["a1_lineage_dose"]["function_preserving_upgrade"][
        "receipt_sha256"
    ] = "sha256:" + "9" * 64
    report.write_text(json.dumps(report_value), encoding="utf-8")
    with pytest.raises(promotion.PromotionError, match="does not bind producer/init"):
        promotion._verify_training_report(  # noqa: SLF001
            report,
            contract=contract,
            contract_sha256=verified["contract_sha256"],
            candidate_path=candidate,
            candidate_sha256=promotion._sha256(candidate),  # noqa: SLF001
        )
