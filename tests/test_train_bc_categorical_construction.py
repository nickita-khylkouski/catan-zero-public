from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.pipeline_configs import TrainConfig
from catan_zero.rl.config_cli import apply_config_file
from catan_zero.rl.self_play import make_env_config
from tools.train_bc import (
    _game_seed_set_sha256,
    _resolve_effective_meaningful_public_history,
    _resolve_effective_value_categorical_bins,
    _resolve_value_phase_weights,
    _training_data_fingerprint,
    _warm_start_grow,
    build_parser,
)


def _args(
    *,
    arch: str = "entity_graph",
    bins: int | None = None,
    init_checkpoint: str = "",
    grow_from_checkpoint: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        arch=arch,
        value_categorical_bins=bins,
        init_checkpoint=init_checkpoint,
        grow_from_checkpoint=grow_from_checkpoint,
    )


def _small_policy(*, bins: int, layers: int = 1) -> EntityGraphPolicy:
    return EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=layers,
        attention_heads=2,
        value_categorical_bins=bins,
        seed=3,
        device="cpu",
    )


def test_cli_default_is_inherit_sentinel_and_fresh_create_builds_requested_head() -> None:
    parser = build_parser()
    assert parser.get_default("value_categorical_bins") is None

    assert _resolve_effective_value_categorical_bins(_args()) == 0
    assert _resolve_effective_value_categorical_bins(_args(bins=9)) == 9

    policy = _small_policy(bins=9)
    assert policy.config.value_categorical_bins == 9
    assert policy.model.value_categorical_bins == 9
    assert policy.model.value_categorical_head[-1].out_features == 10


def test_typed_config_supplies_categorical_bins_before_effective_resolution(
    tmp_path,
) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(TrainConfig(value_categorical_bins=9).canonical_payload()),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--data",
            "corpus",
            "--checkpoint",
            "candidate.pt",
            "--report",
            "report.json",
            "--arch",
            "entity_graph",
            "--config",
            str(config_path),
        ]
    )
    apply_config_file(args, parser)

    assert args.value_categorical_bins == 9
    assert _resolve_effective_value_categorical_bins(args) == 9


@pytest.mark.parametrize("bins", [-3, -1, 1])
def test_invalid_categorical_bin_counts_fail_before_model_construction(bins: int) -> None:
    with pytest.raises(SystemExit, match="0 .* or >=2"):
        _resolve_effective_value_categorical_bins(_args(bins=bins))


def test_non_entity_arch_rejects_categorical_head_construction() -> None:
    with pytest.raises(SystemExit, match="only supported for --arch entity_graph"):
        _resolve_effective_value_categorical_bins(_args(arch="xdim_graph", bins=9))


def test_checkpoint_resume_inherits_bins_and_rejects_architecture_override(tmp_path) -> None:
    checkpoint = tmp_path / "cat9.pt"
    _small_policy(bins=9).save(checkpoint)

    assert _resolve_effective_value_categorical_bins(
        _args(init_checkpoint=str(checkpoint))
    ) == 9
    assert _resolve_effective_value_categorical_bins(
        _args(init_checkpoint=str(checkpoint), bins=9)
    ) == 9
    with pytest.raises(SystemExit, match="does not match --init-checkpoint"):
        _resolve_effective_value_categorical_bins(
            _args(init_checkpoint=str(checkpoint), bins=17)
        )


def test_grow_inherits_and_copies_compatible_categorical_head(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    source = _small_policy(bins=9, layers=1)
    with torch.no_grad():
        for index, parameter in enumerate(source.model.value_categorical_head.parameters()):
            parameter.fill_(0.125 * (index + 1))
    checkpoint = tmp_path / "source.pt"
    source.save(checkpoint)

    assert _resolve_effective_value_categorical_bins(
        _args(grow_from_checkpoint=str(checkpoint))
    ) == 9
    assert _resolve_effective_value_categorical_bins(
        _args(grow_from_checkpoint=str(checkpoint), bins=17)
    ) == 17

    # Same width, one extra trunk block: shared heads are shape-compatible and
    # must warm-start exactly, while only the new block remains fresh.
    target = _small_policy(bins=9, layers=2)
    report = _warm_start_grow(target, str(checkpoint), device="cpu")
    source_state = source.model.value_categorical_head.state_dict()
    target_state = target.model.value_categorical_head.state_dict()
    assert source_state.keys() == target_state.keys()
    for key in source_state:
        assert torch.equal(target_state[key], source_state[key])
    assert not any(
        name.startswith("value_categorical_head.")
        for name in report["missing_examples"] + report["shape_mismatch_examples"]
    )


def test_train_config_hash_captures_data_init_grow_and_holdout_identity() -> None:
    base = TrainConfig()
    variants = (
        TrainConfig(data="/corpora/gen5-a"),
        TrainConfig(data_fingerprint="sha256:data"),
        TrainConfig(init_checkpoint="/models/gen3.pt"),
        TrainConfig(init_checkpoint_sha256="sha256:init"),
        TrainConfig(grow_from_checkpoint="/models/gen3-cat.pt"),
        TrainConfig(grow_from_checkpoint_sha256="sha256:grow"),
        TrainConfig(resume_optimizer=False),
        TrainConfig(validation_fraction=0.10),
        TrainConfig(validation_max_samples=1234),
        TrainConfig(validation_game_seed_ranges="9000:9999"),
        TrainConfig(allow_missing_game_seed_validation_split=True),
        TrainConfig(value_categorical_bins=33),
        TrainConfig(action_module_lr_mult=0.3),
        TrainConfig(amp="bf16"),
        TrainConfig(fused_optimizer=True),
        TrainConfig(grad_accum_steps=2),
        TrainConfig(ddp_shard_data=True),
        TrainConfig(fsdp=True),
        TrainConfig(graph_history_features=True),
        TrainConfig(teacher_weights="mcts=2.0"),
        TrainConfig(phase_weights="robber=3.0"),
        TrainConfig(value_phase_weights="robber=8.0"),
        TrainConfig(q_skip_teacher_prefixes=""),
        TrainConfig(value_root_blend_phases="PLAY_TURN"),
        TrainConfig(value_root_blend_global_compat=True),
    )
    assert len({base.config_hash(), *(variant.config_hash() for variant in variants)}) == (
        1 + len(variants)
    )


def test_value_phase_weights_can_explicitly_opt_out_of_policy_phase_repair() -> None:
    inherited, inherited_source = _resolve_value_phase_weights(
        "",
        policy_phase_weights={"PLAY_TURN": 4.0},
    )
    explicit_none, none_source = _resolve_value_phase_weights(
        "none",
        policy_phase_weights={"PLAY_TURN": 4.0},
    )
    explicit_map, map_source = _resolve_value_phase_weights(
        "MOVE_ROBBER=2.0",
        policy_phase_weights={"PLAY_TURN": 4.0},
    )

    assert inherited == {"PLAY_TURN": 4.0}
    assert inherited_source == "policy_phase_weights"
    assert explicit_none == {}
    assert none_source == "explicit_none"
    assert explicit_map == {"MOVE_ROBBER": 2.0}
    assert map_source == "explicit_map"


def test_history_v2_resolver_uses_adapter_owned_cap_and_target_gather() -> None:
    args = SimpleNamespace(
        arch="entity_graph",
        meaningful_public_history=True,
        meaningful_public_history_pooling="ordered_attention_v2",
        meaningful_public_history_target_gather=True,
        event_history_limit=64,
        entity_feature_adapter_version=(
            "rust_entity_adapter_v5_meaningful_history_v2"
        ),
        public_rule_state_features=True,
        init_checkpoint="",
        grow_from_checkpoint="",
    )

    assert _resolve_effective_meaningful_public_history(args) == (
        True,
        64,
        "ordered_attention_v2",
        True,
    )


def test_history_v2_resolver_rejects_legacy_cap() -> None:
    args = SimpleNamespace(
        arch="entity_graph",
        meaningful_public_history=True,
        meaningful_public_history_pooling="ordered_attention_v2",
        meaningful_public_history_target_gather=True,
        event_history_limit=32,
        entity_feature_adapter_version=(
            "rust_entity_adapter_v5_meaningful_history_v2"
        ),
        public_rule_state_features=True,
        init_checkpoint="",
        grow_from_checkpoint="",
    )

    with pytest.raises(SystemExit, match="expected=64"):
        _resolve_effective_meaningful_public_history(args)


def test_memmap_fingerprint_and_validation_seed_hash_are_content_stable(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_meta.json").write_text(
        '{"row_count": 12}\n', encoding="utf-8"
    )

    fingerprint = _training_data_fingerprint(str(corpus), "memmap")
    assert fingerprint.startswith("sha256:")
    assert fingerprint == _training_data_fingerprint(str(corpus), "memmap")
    assert _game_seed_set_sha256(np.asarray([9, 3, 9, 5])) == _game_seed_set_sha256(
        np.asarray([5, 9, 3])
    )
