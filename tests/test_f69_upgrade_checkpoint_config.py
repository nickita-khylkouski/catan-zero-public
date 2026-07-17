"""Regression: tools/f69_upgrade_checkpoint_config.py must upgrade a config
pickled BEFORE a field existed.

The seed checkpoint's EntityGraphConfig predates both the f69 flags and f67's
value_uncertainty_head, so the original `dataclasses.replace(base.config, ...)`
raised AttributeError (replace reads every current field off the stale object).
`_build_upgraded_config` copies the fields that exist, fills the rest from the
dataclass defaults, then applies the flag overrides.
"""

from __future__ import annotations

import copy
import dataclasses
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import f69_upgrade_checkpoint_config as upgrade_tool  # noqa: E402
from catan_zero.rl.entity_token_policy import EntityGraphConfig  # noqa: E402

# Fields absent from a seed config pickled before these landed.
_LATER_FIELDS = (
    "value_uncertainty_head",
    "action_target_gather",
    "action_cross_attention_layers",
    "value_attention_pool",
    "topology_residual_adapter",
)
_OVERRIDES = {
    "action_target_gather": True,
    "action_cross_attention_layers": 2,
    "value_attention_pool": True,
}


def _stale_config():
    """A real EntityGraphConfig instance with `_LATER_FIELDS` slots UNSET --
    exactly how a frozen+slots dataclass pickled before those fields existed
    deserialises under the newer class definition (only the pickled slots are
    set; accessing an unset slot raises AttributeError). Using a real instance
    (not a SimpleNamespace) is what makes the replace() failure the genuine
    AttributeError, not a spurious TypeError."""
    full = EntityGraphConfig(action_size=607, static_action_feature_size=1)
    stale = object.__new__(EntityGraphConfig)
    for f in fields(EntityGraphConfig):
        if f.name not in _LATER_FIELDS:
            object.__setattr__(stale, f.name, getattr(full, f.name))
    return stale


def test_replace_on_stale_config_reproduces_the_bug():
    """Documents the failure the fix addresses."""
    stale = _stale_config()
    assert not hasattr(stale, "value_uncertainty_head")
    with pytest.raises(AttributeError):
        dataclasses.replace(stale, **_OVERRIDES)


def test_build_upgraded_config_tolerates_missing_field():
    stale = _stale_config()
    upgraded = upgrade_tool._build_upgraded_config(stale, _OVERRIDES)

    assert isinstance(upgraded, EntityGraphConfig)
    # f69 overrides applied
    assert upgraded.action_target_gather is True
    assert upgraded.action_cross_attention_layers == 2
    assert upgraded.value_attention_pool is True
    # a field the stale pickle lacked is filled from the current default
    assert upgraded.value_uncertainty_head is False
    # pre-existing fields copied through unchanged
    assert upgraded.action_size == 607
    assert upgraded.hidden_size == 640
    assert upgraded.state_layers == 6


def test_v7_input_route_has_only_explicit_information_migration_constructor():
    overrides = upgrade_tool._parse_flags(  # noqa: SLF001
        "v5_to_v7_input_compatibility_migration"
    )

    assert overrides["v6_compatibility_preserving_inputs"] is True
    assert overrides["action_cross_attention_layers"] == 1
    assert overrides["topology_residual_adapter"] is True
    with pytest.raises(SystemExit, match="unsafe V6-trained"):
        upgrade_tool._parse_flags(  # noqa: SLF001
            "current_v7_compatibility_action_cross1"
        )


def test_v7_input_migration_constructs_complete_strict_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    from catan_zero.rl.entity_feature_adapter import (
        RUST_ENTITY_ADAPTER_V5,
        RUST_ENTITY_ADAPTER_V6,
    )
    from catan_zero.rl.checkpoint_runtime_semantics import (
        ENTITY_GRAPH_FORWARD_SEMANTICS_KEY,
        current_entity_graph_forward_semantics,
    )
    import catan_zero.rl.entity_token_policy as entity_token_policy
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config
    from tools import a1_information_contract_migration as migration_receipt

    legacy = tmp_path / "legacy-v5.pt"
    source = tmp_path / "current-v5.pt"
    migrated = tmp_path / "v7.pt"
    base = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=11,
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V5,
    )
    base.save(legacy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "f69_upgrade_checkpoint_config.py",
            "--in-checkpoint",
            str(legacy),
            "--out-checkpoint",
            str(source),
            "--flags",
            "current_v5_split1",
            "--seed",
            "67",
            "--device",
            "cpu",
            "--no-verify",
        ],
    )
    upgrade_tool.main()
    # Exact legacy/V5 parent bytes may predate runtime-semantics metadata.
    # The migration must authenticate its newly constructed output rather than
    # merely copying whatever the source happened to contain.
    source_raw = torch.load(source, map_location="cpu", weights_only=False)
    source_raw.pop(ENTITY_GRAPH_FORWARD_SEMANTICS_KEY)
    torch.save(source_raw, source)
    # The repository test environment may intentionally carry an older native
    # wheel; native anchor generation is covered by the sealed H100 gate.  This
    # unit test isolates checkpoint construction and strict serialization.
    monkeypatch.setattr(
        upgrade_tool,
        "_migration_anchor_evidence",
        lambda *_args, **_kwargs: {
            "schema_version": "adapter-v7-compatibility-step0-anchor-evidence-v1",
            "source_adapter": RUST_ENTITY_ADAPTER_V5,
            "target_adapter": RUST_ENTITY_ADAPTER_V6,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "f69_upgrade_checkpoint_config.py",
            "--in-checkpoint",
            str(source),
            "--out-checkpoint",
            str(migrated),
            "--flags",
            "v5_to_v7_input_compatibility_migration",
            "--seed",
            "73",
            "--device",
            "cpu",
        ],
    )

    upgrade_tool.main()

    source_raw = torch.load(source, map_location="cpu", weights_only=False)
    migrated_raw = torch.load(migrated, map_location="cpu", weights_only=False)
    expected_semantics = current_entity_graph_forward_semantics(
        Path(entity_token_policy.__file__)
    )
    assert (
        migrated_raw[ENTITY_GRAPH_FORWARD_SEMANTICS_KEY] == expected_semantics
    )
    policy = EntityGraphPolicy.load(
        migrated, device="cpu", enforce_runtime_semantics=True
    )
    replay = migration_receipt._verify_v7_input_routing_delta(  # noqa: SLF001
        source_raw,
        migrated_raw,
        initialization_seed=73,
    )
    assert replay["shared_parameters_bit_identical"] is True
    assert policy.config.v6_compatibility_preserving_inputs is True
    assert policy._checkpoint_missing_state_keys == ()  # noqa: SLF001
    assert (
        migrated_raw["information_contract_migration_provenance"]["migration"]
        == "v5_to_v7_input_compatibility"
    )
    assert (
        torch.count_nonzero(
            migrated_raw["model"]["v6_exact_resource_residual.weight"]
        ).item()
        == 0
    )
    assert (
        torch.count_nonzero(
            migrated_raw["model"]["v6_initial_road_residual.weight"]
        ).item()
        == 0
    )
    for name, tensor in source_raw["model"].items():
        assert torch.equal(tensor, migrated_raw["model"][name]), name

    forged_raw = copy.deepcopy(migrated_raw)
    forged_raw[ENTITY_GRAPH_FORWARD_SEMANTICS_KEY] = {
        "schema_version": "entity-graph-forward-semantics-v3",
        "semantic_sha256": "sha256:forged",
    }
    with pytest.raises(
        migration_receipt.MigrationError,
        match="authenticated current",
    ):
        migration_receipt._verify_v7_input_routing_delta(  # noqa: SLF001
            source_raw,
            forged_raw,
            initialization_seed=73,
        )

    truncated = tmp_path / "v7-truncated.pt"
    truncated_raw = copy.deepcopy(migrated_raw)
    truncated_raw["model"].pop("v6_initial_road_residual.weight")
    torch.save(truncated_raw, truncated)
    with pytest.raises(RuntimeError, match="checkpoint state mismatch"):
        EntityGraphPolicy.load(truncated, device="cpu")


def test_direct_v8_migration_preserves_v7_route_and_adds_exact_public_resources():
    overrides = upgrade_tool._parse_flags(  # noqa: SLF001
        "v5_to_v8_public_resource_compatibility_migration"
    )

    assert overrides["v6_compatibility_preserving_inputs"] is True
    assert overrides["action_cross_attention_layers"] == 1
    assert overrides["action_cross_attention_bottleneck"] == 80
    assert overrides["public_card_exact_resource_residual"] is True


def test_current_v8_migration_builds_the_complete_v2_to_v6_route():
    overrides = upgrade_tool._parse_flags(  # noqa: SLF001
        "current_v8_information_migration_topology_split1"
    )

    assert overrides["topology_residual_adapter"] is True
    assert overrides["v6_compatibility_preserving_inputs"] is True
    assert overrides["action_cross_attention_layers"] == 1
    assert overrides["public_card_exact_resource_residual"] is True


def test_build_upgraded_config_preserves_a_full_config():
    """A current (non-stale) config round-trips with only the overrides changed."""
    base = EntityGraphConfig(
        action_size=607, static_action_feature_size=1, hidden_size=512
    )
    upgraded = upgrade_tool._build_upgraded_config(base, _OVERRIDES)
    assert upgraded.hidden_size == 512
    assert upgraded.action_cross_attention_layers == 2
    assert dataclasses.replace(base, **_OVERRIDES) == upgraded


def test_topology_upgrade_flag_is_explicit_and_default_off():
    base = EntityGraphConfig(action_size=607, static_action_feature_size=1)
    assert base.topology_residual_adapter is False
    overrides = upgrade_tool._parse_flags("gather,topology")
    assert overrides == {
        "action_target_gather": True,
        "topology_residual_adapter": True,
    }
    upgraded = upgrade_tool._build_upgraded_config(base, overrides)
    assert upgraded.action_target_gather is True
    assert upgraded.topology_residual_adapter is True


def test_public_card_count_upgrade_flag_is_explicit_and_default_off():
    base = EntityGraphConfig(action_size=607, static_action_feature_size=1)
    assert base.public_card_count_features is False
    overrides = upgrade_tool._parse_flags("card_count")
    assert overrides == {"public_card_count_features": True}
    upgraded = upgrade_tool._build_upgraded_config(base, overrides)
    assert upgraded.public_card_count_features is True
    assert upgraded.public_card_count_residual_bias is True


def test_public_rule_state_upgrade_binds_v4_schema_and_parameter_allowlist():
    base = EntityGraphConfig(action_size=607, static_action_feature_size=1)
    assert base.public_rule_state_features is False

    overrides = upgrade_tool._parse_flags("public_rule_state")
    assert overrides == {
        "public_rule_state_features": True,
        "public_rule_state_feature_schema": "actor_public_rule_state_2p_v1",
    }
    upgraded = upgrade_tool._build_upgraded_config(base, overrides)
    assert upgraded.public_rule_state_features is True
    assert upgraded.public_rule_state_feature_schema == "actor_public_rule_state_2p_v1"
    assert "public_rule_state_residual." in upgrade_tool.NEW_PARAM_PREFIXES


def test_action_cross_upgrade_parameters_are_admitted_by_checkpoint_builder():
    # The cross block is an exact-identity warm-start module.  If this prefix
    # is absent, ``--flags cross:1`` constructs the model but rejects its own
    # new parameters before a checkpoint can be issued.
    assert "action_cross_blocks." in upgrade_tool.NEW_PARAM_PREFIXES


def test_v7_migration_token_binds_compatibility_route_and_live_decoder():
    overrides = upgrade_tool._parse_flags(
        "v5_to_v7_input_compatibility_migration"
    )
    assert overrides["v6_compatibility_preserving_inputs"] is True
    assert overrides["action_cross_attention_layers"] == 1
    assert overrides["topology_residual_adapter"] is True


def test_structured_action_value_upgrade_enables_both_zero_diff_paths():
    base = EntityGraphConfig(action_size=607, static_action_feature_size=45)
    assert base.static_action_residual is False
    assert base.legal_action_value_residual is False
    overrides = upgrade_tool._parse_flags("structured_action_value")
    assert overrides == {
        "static_action_residual": True,
        "legal_action_value_residual": True,
    }
    upgraded = upgrade_tool._build_upgraded_config(base, overrides)
    assert upgraded.static_action_residual is True
    assert upgraded.legal_action_value_residual is True


def test_value_tower_split_upgrade_is_explicit_and_default_off():
    base = EntityGraphConfig(action_size=607, static_action_feature_size=45)
    assert base.value_tower_split_layers == 0
    overrides = upgrade_tool._parse_flags("value_split:2")
    assert overrides == {"value_tower_split_layers": 2}
    upgraded = upgrade_tool._build_upgraded_config(base, overrides)
    assert upgraded.value_tower_split_layers == 2


def test_forward_tolerance_is_module_owned_and_split_only():
    assert upgrade_tool._forward_tolerance({}) == 0.0  # noqa: SLF001
    assert (  # noqa: SLF001
        upgrade_tool._forward_tolerance({"value_tower_split_layers": 1})
        == float(np.finfo(np.float32).eps)
    )


def test_canonical_v3_flag_bundle_enables_all_structured_input_repairs():
    overrides = upgrade_tool._parse_flags(  # noqa: SLF001
        "structured_action_value,card_count_v2,meaningful_history"
    )
    assert overrides["static_action_residual"] is True
    assert overrides["legal_action_value_residual"] is True
    assert overrides["public_card_count_features"] is True
    assert overrides["public_card_count_residual_bias"] is False
    assert overrides["meaningful_public_history"] is True
    assert overrides["event_history_limit"] == 32


def test_bias_free_public_card_count_upgrade_is_explicit_v2():
    base = EntityGraphConfig(action_size=607, static_action_feature_size=1)
    overrides = upgrade_tool._parse_flags("card_count_v2")
    assert overrides == {
        "public_card_count_features": True,
        "public_card_count_residual_bias": False,
    }
    upgraded = upgrade_tool._build_upgraded_config(base, overrides)
    assert upgraded.public_card_count_features is True
    assert upgraded.public_card_count_residual_bias is False


def test_meaningful_history_upgrade_is_bounded_and_explicit():
    base = EntityGraphConfig(action_size=607, static_action_feature_size=1)
    assert base.meaningful_public_history is False
    overrides = upgrade_tool._parse_flags("meaningful_history")
    assert overrides == {
        "meaningful_public_history": True,
        "meaningful_public_history_schema": "meaningful_public_history_2p_no_trade_v1",
        "event_history_limit": 32,
    }
    upgraded = upgrade_tool._build_upgraded_config(base, overrides)
    assert upgraded.meaningful_public_history is True
    assert upgraded.event_history_limit == 32


def test_preserve_source_top_level_keys_restores_mask_hidden_info(tmp_path):
    """CAT-80 regression: upgrading a masked checkpoint must NOT drop top-level
    provenance keys. EntityGraphPolicy.save() rebuilds the checkpoint from the
    fresh upgraded policy, resetting mask_hidden_info True->False (mislabeling a
    masked net as omniscient). _preserve_source_top_level_keys restores every
    source top-level key except the intentionally-mutated model+config."""
    import torch
    from catan_zero.rl.checkpoint_runtime_semantics import (
        ENTITY_GRAPH_FORWARD_SEMANTICS_KEY,
        current_entity_graph_forward_semantics,
    )
    import catan_zero.rl.entity_token_policy as entity_token_policy

    in_ckpt = tmp_path / "in.pt"
    out_ckpt = tmp_path / "out.pt"
    torch.save(
        {
            "policy_type": "entity_graph",
            "mask_hidden_info": True,
            "action_mask_version": "colonist-multiagent-v1",
            "static_action_features_sha256": "abc",
            "config": {"flags": "OLD"},
            "model": {"w": torch.zeros(2)},
        },
        in_ckpt,
    )
    # What EntityGraphPolicy.save() would have written: model+config mutated,
    # mask_hidden_info silently reset to the fresh-policy default False (the bug).
    torch.save(
        {
            "policy_type": "entity_graph",
            "mask_hidden_info": False,
            "action_mask_version": "colonist-multiagent-v1",
            "static_action_features_sha256": "abc",
            "config": {"flags": "NEW"},
            "model": {"w": torch.ones(2)},
            # A config-only cat-head upgrade must not manufacture evidence that
            # the new random readout was optimized.
            "trained_value_readouts": ["categorical"],
            "value_training": {"primary_readout": "categorical"},
            ENTITY_GRAPH_FORWARD_SEMANTICS_KEY: (
                current_entity_graph_forward_semantics(
                    Path(entity_token_policy.__file__)
                )
            ),
        },
        out_ckpt,
    )

    preserved = upgrade_tool._preserve_source_top_level_keys(
        str(in_ckpt), str(out_ckpt)
    )

    merged = torch.load(out_ckpt, map_location="cpu", weights_only=False)
    # provenance restored from source
    assert merged["mask_hidden_info"] is True
    assert "mask_hidden_info" in preserved
    # intentionally-mutated keys keep the UPGRADED values
    assert merged["config"] == {"flags": "NEW"}
    assert merged["model"]["w"].tolist() == [1.0, 1.0]
    assert "model" not in preserved and "config" not in preserved
    assert "trained_value_readouts" not in merged
    assert "value_training" not in merged
    assert merged[ENTITY_GRAPH_FORWARD_SEMANTICS_KEY] == (
        current_entity_graph_forward_semantics(Path(entity_token_policy.__file__))
    )


def test_preserve_source_top_level_keys_rejects_forged_runtime_stamp(tmp_path):
    import torch

    from catan_zero.rl.checkpoint_runtime_semantics import (
        ENTITY_GRAPH_FORWARD_SEMANTICS_KEY,
    )

    in_ckpt = tmp_path / "in.pt"
    out_ckpt = tmp_path / "out.pt"
    torch.save({"config": {}, "model": {}}, in_ckpt)
    torch.save(
        {
            "config": {},
            "model": {},
            ENTITY_GRAPH_FORWARD_SEMANTICS_KEY: {
                "schema_version": "entity-graph-forward-semantics-v3",
                "semantic_sha256": "sha256:forged",
            },
        },
        out_ckpt,
    )

    with pytest.raises(RuntimeError, match="authenticated current"):
        upgrade_tool._preserve_source_top_level_keys(  # noqa: SLF001
            str(in_ckpt), str(out_ckpt)
        )


def test_upgrade_seed_is_deterministic_and_durably_attested(
    tmp_path, monkeypatch
) -> None:
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    source = tmp_path / "source.pt"
    out_a = tmp_path / "a.pt"
    out_b = tmp_path / "b.pt"
    out_c = tmp_path / "c.pt"
    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    policy.save(source)

    for output in (out_a, out_b):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "f69_upgrade_checkpoint_config.py",
                "--in-checkpoint",
                str(source),
                "--out-checkpoint",
                str(output),
                "--flags",
                "catbins:9",
                "--seed",
                "73",
                "--device",
                "cpu",
                "--no-verify",
            ],
        )
        upgrade_tool.main()

    raw_a = torch.load(out_a, map_location="cpu", weights_only=False)
    raw_b = torch.load(out_b, map_location="cpu", weights_only=False)
    cat_keys = sorted(
        key for key in raw_a["model"] if key.startswith("value_categorical_head.")
    )
    assert cat_keys
    assert all(
        torch.equal(raw_a["model"][key], raw_b["model"][key]) for key in cat_keys
    )
    assert raw_a["upgrade_provenance"]["initialization_seed"] == 73
    assert raw_a["upgrade_provenance"]["trained_value_readouts_added"] == []
    assert (
        raw_a["upgrade_provenance"]["source_checkpoint_sha256"]
        == raw_b["upgrade_provenance"]["source_checkpoint_sha256"]
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "f69_upgrade_checkpoint_config.py",
            "--in-checkpoint",
            str(source),
            "--out-checkpoint",
            str(out_c),
            "--flags",
            "catbins:9",
            "--seed",
            "74",
            "--device",
            "cpu",
            "--no-verify",
        ],
    )
    upgrade_tool.main()
    raw_c = torch.load(out_c, map_location="cpu", weights_only=False)
    assert any(
        not torch.equal(raw_a["model"][key], raw_c["model"][key]) for key in cat_keys
    )


def test_combined_topology_gather_upgrade_verifies_exact_real_root(
    tmp_path, monkeypatch
) -> None:
    import torch

    pytest.importorskip("catanatron_rs")

    # Some developer environments retain an older importable wheel that
    # predates the native MCTS snapshot/copy surface.  That environment cannot
    # construct the real-root parity fixture, and is equivalent to the binding
    # being unavailable for this test (the unit/synthetic parity tests still
    # exercise the upgrade below).  Do not misreport a stale wheel as an
    # architecture-upgrade failure.
    from catan_zero.search.rust_mcts import _require_rust_module

    try:
        _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    source = tmp_path / "source.pt"
    output = tmp_path / "topology-gather.pt"
    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    policy.save(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "f69_upgrade_checkpoint_config.py",
            "--in-checkpoint",
            str(source),
            "--out-checkpoint",
            str(output),
            "--flags",
            "gather,topology",
            "--seed",
            "73",
            "--device",
            "cpu",
        ],
    )
    upgrade_tool.main()

    raw = torch.load(output, map_location="cpu", weights_only=False)
    assert raw["upgrade_provenance"]["flags"] == {
        "action_target_gather": True,
        "topology_residual_adapter": True,
    }
    assert raw["upgrade_provenance"]["forward_max_diff"] == 0.0
    assert raw["upgrade_provenance"]["forward_identical_at_init"] is True


def test_topology_only_upgrade_preserves_an_existing_learned_value_tower(
    tmp_path, monkeypatch
) -> None:
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    source = tmp_path / "split-source.pt"
    output = tmp_path / "split-plus-topology.pt"
    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=2,
        attention_heads=2,
        value_tower_split_layers=1,
        seed=19,
    )
    with torch.no_grad():
        for name, parameter in policy.model.named_parameters():
            if name.startswith(("value_blocks.", "value_state_norm.")):
                parameter.add_(0.125)
    policy.save(source)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "f69_upgrade_checkpoint_config.py",
            "--in-checkpoint",
            str(source),
            "--out-checkpoint",
            str(output),
            "--flags",
            "topology",
            "--seed",
            "73",
            "--device",
            "cpu",
            "--no-verify",
        ],
    )
    upgrade_tool.main()

    before = torch.load(source, map_location="cpu", weights_only=False)["model"]
    after = torch.load(output, map_location="cpu", weights_only=False)["model"]
    value_names = [
        name
        for name in before
        if name.startswith(("value_blocks.", "value_state_norm."))
    ]
    assert value_names
    assert all(torch.equal(before[name], after[name]) for name in value_names)
