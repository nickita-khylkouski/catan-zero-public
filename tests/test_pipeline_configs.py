"""Tests for the typed pipeline configs + config-hash registry (task CAT-66).

Covers the four verification requirements from the ticket:
  * round-trip  -- config -> canonical payload -> config is identity.
  * hash stability -- identical fields always hash identically.
  * drift detection -- any single differing field changes the hash.
  * argv-equivalence -- typed-config defaults equal the CLI argparse defaults
    (the anti-CLI-default-override guarantee) and ``from_namespace`` over an
    argv namespace reproduces a direct construction.
Plus the registry (idempotent append / lookup), the cross-pipeline masking
consistency check, the additive CLI helpers, and the ticket's smoke test:
a deliberately-introduced CLI-vs-dataclass mismatch is reflected in the hash.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
from pathlib import Path

import pytest

from catan_zero.rl import config_cli, config_registry
from catan_zero.rl.pipeline_configs import (
    CONFIG_CLASSES,
    EvalConfig,
    GateConfig,
    GenerateConfig,
    PipelineConfig,
    TrainConfig,
    check_masking_consistency,
    config_from_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"

ALL_CONFIGS = (TrainConfig, GenerateConfig, EvalConfig, GateConfig)


def test_train_action_cross_field_remains_append_only() -> None:
    assert [field.name for field in dataclasses.fields(TrainConfig)[-2:]] == [
        "action_cross_attention_layers",
        "action_cross_attention_bottleneck",
    ]


# --------------------------------------------------------------------------- #
# Round-trip.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", ALL_CONFIGS)
def test_payload_round_trip_is_identity(cls: type[PipelineConfig]) -> None:
    cfg = cls()
    rebuilt = config_from_payload(cfg.canonical_payload())
    assert rebuilt == cfg
    # A JSON string round-trip (as the registry stores it) is also lossless.
    rebuilt_json = config_from_payload(json.loads(cfg.canonical_json()))
    assert rebuilt_json == cfg


def test_round_trip_preserves_non_default_values() -> None:
    cfg = TrainConfig(
        mask_hidden_info=True, value_loss_weight=1.0, seed=99, optimizer="adamw"
    )
    rebuilt = config_from_payload(json.loads(cfg.canonical_json()))
    assert rebuilt == cfg
    assert rebuilt.mask_hidden_info is True
    assert rebuilt.value_loss_weight == 1.0


def test_policy_aux_active_batch_size_changes_typed_train_hash() -> None:
    baseline = TrainConfig()
    policy_aux = TrainConfig(policy_aux_active_batch_size=128)

    assert baseline.policy_aux_active_batch_size == 0
    assert policy_aux.config_hash() != baseline.config_hash()
    assert config_from_payload(policy_aux.canonical_payload()) == policy_aux


def test_policy_signal_admission_floor_changes_typed_train_hash() -> None:
    historical = TrainConfig(
        minimum_policy_effective_rows_per_global_batch=0.0
    )
    fail_closed = TrainConfig(
        minimum_policy_effective_rows_per_global_batch=32.0
    )

    assert historical.config_hash() != fail_closed.config_hash()
    assert config_from_payload(fail_closed.canonical_payload()) == fail_closed


def test_forced_scalar_value_mass_ceiling_changes_typed_train_hash() -> None:
    historical = TrainConfig()
    fail_closed = TrainConfig(maximum_nominal_forced_scalar_value_mass_fraction=0.4)

    assert historical.maximum_nominal_forced_scalar_value_mass_fraction is None
    assert historical.config_hash() != fail_closed.config_hash()
    assert config_from_payload(fail_closed.canonical_payload()) == fail_closed


def test_exact_step_dose_changes_typed_train_hash() -> None:
    epoch_bounded = TrainConfig(max_steps=128, exact_max_steps=False)
    exact_dose = TrainConfig(max_steps=128, exact_max_steps=True)

    assert exact_dose.config_hash() != epoch_bounded.config_hash()
    assert exact_dose.canonical_payload()["fields"]["exact_max_steps"] is True
    assert config_from_payload(exact_dose.canonical_payload()) == exact_dose


def test_per_game_policy_surprise_changes_typed_train_hash() -> None:
    baseline = TrainConfig()
    exact_surprise = TrainConfig(per_game_policy_surprise_weighting=True)

    assert baseline.per_game_policy_surprise_weighting is False
    assert exact_surprise.config_hash() != baseline.config_hash()
    assert config_from_payload(exact_surprise.canonical_payload()) == exact_surprise


def test_value_outcome_balance_changes_typed_train_hash() -> None:
    baseline = TrainConfig()
    balanced = TrainConfig(
        value_player_outcome_balance_mode="sampler_balanced_v1"
    )

    assert baseline.value_player_outcome_balance_mode == "none"
    assert balanced.config_hash() != baseline.config_hash()
    assert config_from_payload(balanced.canonical_payload()) == balanced


def test_current_canonical_train_recipe_does_not_inherit_policy_phase_weights() -> None:
    from tools.train import _load_recipe
    from tools.train_bc import _coverage_unsupported_objectives

    recipe = (
        Path(__file__).resolve().parents[1]
        / "configs/training/a1_current_35m_b200.schema1.json"
    )
    config, engine = _load_recipe(recipe)

    assert config.phase_weights == (
        "PLAY_TURN=4.0,MOVE_ROBBER=3.0,"
        "BUILD_INITIAL_ROAD=2.0,DISCARD=1.5"
    )
    assert config.value_phase_weights == "none"
    assert config.value_player_outcome_balance_mode == "none"
    assert engine["base_sampler"] == "coverage_importance_v1"
    assert config.moe_routed_experts == 0
    assert config.moe_balance_loss_weight == pytest.approx(0.0)
    assert _coverage_unsupported_objectives(
        config,
        categorical_value_loss_weight=config.value_categorical_loss_weight,
    ) == ()


def test_coverage_rejects_balance_loss_when_sparse_moe_is_active() -> None:
    from tools.train_bc import _coverage_unsupported_objectives

    config = dataclasses.replace(
        TrainConfig(),
        moe_routed_experts=4,
        moe_balance_loss_weight=0.01,
    )

    assert _coverage_unsupported_objectives(
        config,
        categorical_value_loss_weight=config.value_categorical_loss_weight,
    ) == ("moe_balance_loss_weight",)


def test_config_from_payload_rejects_unknown_pipeline() -> None:
    with pytest.raises(ValueError, match="unknown pipeline"):
        config_from_payload({"pipeline": "nonsense", "fields": {}})


def test_config_from_payload_drops_unknown_fields_and_fills_missing() -> None:
    payload = {
        "pipeline": "gate",
        "schema_version": 1,
        "fields": {"elo1": 30.0, "bogus_field": 7},
    }
    cfg = config_from_payload(payload)
    assert isinstance(cfg, GateConfig)
    assert cfg.elo1 == 30.0
    assert cfg.alpha == 0.05  # missing -> current default


# --------------------------------------------------------------------------- #
# Hash stability + drift detection.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", ALL_CONFIGS)
def test_identical_configs_hash_identically(cls: type[PipelineConfig]) -> None:
    assert cls().config_hash() == cls().config_hash()
    assert cls().full_config_hash().startswith("sha256:")
    assert (
        cls().config_hash()
        == "sha256:" + cls().full_config_hash().split(":", 1)[1][:16]
    )


@pytest.mark.parametrize("cls", ALL_CONFIGS)
def test_every_single_field_change_changes_the_hash(cls: type[PipelineConfig]) -> None:
    base = cls()
    base_hash = base.config_hash()
    for field in dataclasses.fields(cls):
        mutated = _mutate_one_field(base, field)
        assert mutated.config_hash() != base_hash, (
            f"changing {cls.__name__}.{field.name} did not change the config hash "
            "(the hash must be sensitive to every science-critical field)"
        )


def test_different_pipelines_with_same_field_names_do_not_collide() -> None:
    # generate and eval both carry n_full/c_scale/public_observation; the
    # ``pipeline`` marker keeps their hashes distinct even at equal values.
    assert GenerateConfig().config_hash() != EvalConfig().config_hash()


def test_generate_native_hot_loop_changes_science_hash() -> None:
    reference = GenerateConfig()
    native = GenerateConfig(native_mcts_hot_loop=True)

    assert reference.native_mcts_hot_loop is False
    assert native.config_hash() != reference.config_hash()


def test_boundary_value_particles_are_bound_in_eval_identity() -> None:
    historical = EvalConfig(boundary_value_particles=1)
    experimental = EvalConfig(boundary_value_particles=2)

    assert historical.boundary_value_particles == 1
    assert experimental.config_hash() != historical.config_hash()


def test_initial_road_d1_scope_changes_generation_config_hash() -> None:
    global_d1 = GenerateConfig(rescale_noise_floor_c=8.0)
    road_only_d1 = GenerateConfig(
        rescale_noise_floor_c=8.0,
        rescale_noise_floor_initial_road_only=True,
    )

    assert global_d1.config_hash() != road_only_d1.config_hash()
    assert road_only_d1.field_values()[
        "rescale_noise_floor_initial_road_only"
    ] is True


def test_hash_is_field_order_independent() -> None:
    # sort_keys in canonical_json means declaration order never affects the hash.
    payload = TrainConfig(seed=5).canonical_payload()
    reordered = {
        "fields": payload["fields"],
        "schema_version": payload["schema_version"],
        "pipeline": payload["pipeline"],
    }
    assert json.dumps(payload, sort_keys=True) == json.dumps(reordered, sort_keys=True)


# --------------------------------------------------------------------------- #
# argv-equivalence: typed-config defaults == CLI argparse defaults.
# --------------------------------------------------------------------------- #


def _parse_argparse_defaults(path: Path) -> dict[str, object]:
    """dest (de-dashed long flag) -> literal default, for every add_argument."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out: dict[str, object] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        dest = None
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ):
                dest = arg.value[2:].replace("-", "_")
                break
        if dest is None:
            continue
        default: object = None
        has_default = False
        is_store_true = False
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    default = ast.literal_eval(kw.value)
                    has_default = True
                except (ValueError, SyntaxError):
                    default = _NONLITERAL
                    has_default = True
            if kw.arg == "action":
                action_val = getattr(kw.value, "value", None)
                if action_val == "store_true":
                    is_store_true = True
        if is_store_true and not has_default:
            default = False
            has_default = True
        if has_default:
            out[dest] = default
    return out


_NONLITERAL = object()


def _assert_typed_defaults_match_cli(
    cls: type[PipelineConfig], tool: str, *, skip: set[str] = frozenset()
) -> None:
    cli = _parse_argparse_defaults(TOOLS_DIR / tool)
    typed = cls()
    mismatches: list[str] = []
    for field in dataclasses.fields(cls):
        dest = "format" if field.name == "fmt" else field.name
        if dest in skip or dest not in cli:
            continue
        cli_default = cli[dest]
        if cli_default is _NONLITERAL:
            continue
        typed_default = getattr(typed, field.name)
        if not _same(cli_default, typed_default):
            mismatches.append(
                f"{cls.__name__}.{field.name}={typed_default!r} != {tool} --{dest} default={cli_default!r}"
            )
    assert not mismatches, (
        "typed-config default diverges from CLI default:\n" + "\n".join(mismatches)
    )


def _same(a: object, b: object) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return a == b


def test_train_config_defaults_match_train_bc_cli() -> None:
    _assert_typed_defaults_match_cli(TrainConfig, "train_bc.py")


def test_generate_config_defaults_match_generate_cli() -> None:
    _assert_typed_defaults_match_cli(GenerateConfig, "generate_gumbel_selfplay_data.py")


def test_eval_config_defaults_match_h2h_cli() -> None:
    # Identity/mode fields are not CLI flags shared by the raw-h2h tool.
    # elo0/elo1: after CAT-7 the h2h CLI defaults these to None (sentinel) and
    # resolves them to the effective values (EvalConfig's own defaults) in its
    # gate-config resolution block, so the raw argparse default intentionally no
    # longer equals the dataclass default. The effective defaults still match.
    _assert_typed_defaults_match_cli(
        EvalConfig,
        "gumbel_search_vs_raw_h2h.py",
        skip={
            "mode",
            "candidate",
            "baseline",
            "baseline_bot",
            "map_kind",
            "elo0",
            "elo1",
        },
    )


def test_gate_config_defaults_match_sprt_gate_cli() -> None:
    _assert_typed_defaults_match_cli(
        GateConfig,
        "sprt_gate.py",
        # elo0/elo1/alpha/beta: after CAT-7 the sprt_gate.py CLI defaults these to
        # None (sentinel) and resolves them to the effective values -- GateConfig's
        # own defaults (elo0=0/elo1=5/alpha=beta=0.05) -- in the gate-config
        # resolution block, so the RAW argparse default intentionally no longer
        # equals the dataclass default. The effective defaults still match.
        skip={
            "test_kind",
            "generation_public_observation",
            "elo0",
            "elo1",
            "alpha",
            "beta",
        },
    )


def test_from_namespace_reproduces_direct_construction() -> None:
    ns = argparse.Namespace(
        arch="entity_graph",
        mask_hidden_info=True,
        seed=7,
        value_loss_weight=1.0,
        optimizer="adamw",
        validation_contract_file_sha256="sha256:contract",
        validation_game_seed_set_sha256="sha256:validation",
        training_excluded_game_seed_set_sha256="sha256:excluded",
    )
    got = TrainConfig.from_namespace(ns)
    # Only the attrs present on the namespace override defaults.
    assert got == TrainConfig(
        arch="entity_graph",
        mask_hidden_info=True,
        seed=7,
        value_loss_weight=1.0,
        optimizer="adamw",
        validation_contract_file_sha256="sha256:contract",
        validation_game_seed_set_sha256="sha256:validation",
        training_excluded_game_seed_set_sha256="sha256:excluded",
    )


def test_generate_from_namespace_maps_format_to_fmt() -> None:
    ns = argparse.Namespace(
        **{
            "format": "npz_zst",
            "n_full": 128,
            "public_observation": True,
            "symmetry_averaged_eval": True,
            "symmetry_averaged_eval_threshold": 20,
            "n_full_wide": 256,
            "n_full_wide_threshold": 40,
            "wide_roots_always_full": True,
            "wide_candidates_threshold": 20,
        }
    )
    cfg = GenerateConfig.from_namespace(ns)
    assert cfg.fmt == "npz_zst"
    assert cfg.n_full == 128
    assert cfg.public_observation is True
    assert cfg.symmetry_averaged_eval is True
    assert cfg.symmetry_averaged_eval_threshold == 20
    assert cfg.n_full_wide == 256
    assert cfg.n_full_wide_threshold == 40
    assert cfg.wide_roots_always_full is True
    assert cfg.wide_candidates_threshold == 20


# --------------------------------------------------------------------------- #
# Cross-pipeline masking consistency.
# --------------------------------------------------------------------------- #


def test_masking_consistency_flags_mismatch() -> None:
    gen = GenerateConfig(public_observation=True)
    gate = GateConfig(generation_public_observation=False)
    problems = check_masking_consistency(gen, gate)
    assert problems
    assert "masking regime mismatch" in problems[0]


def test_masking_consistency_passes_when_aligned() -> None:
    gen = GenerateConfig(public_observation=True)
    ev = EvalConfig(public_observation=True)
    gate = GateConfig(generation_public_observation=True)
    assert check_masking_consistency(gen, ev, gate) == []


def test_masking_consistency_ignores_unknown_echo() -> None:
    # A gate with no echoed regime cannot contradict anything.
    gen = GenerateConfig(public_observation=True)
    gate = GateConfig(generation_public_observation=None)
    assert check_masking_consistency(gen, gate) == []


# --------------------------------------------------------------------------- #
# Registry.
# --------------------------------------------------------------------------- #


def test_registry_register_and_lookup(tmp_path: Path) -> None:
    reg = tmp_path / "reg.jsonl"
    cfg = TrainConfig(seed=42, mask_hidden_info=True)
    h = config_registry.register(cfg, purpose="unit-test", path=reg)
    assert h == cfg.config_hash()
    record = config_registry.lookup(h, path=reg)
    assert record is not None
    assert record["pipeline"] == "train"
    assert record["purpose"] == "unit-test"
    assert config_from_payload(record["config"]) == cfg


def test_registry_register_is_idempotent(tmp_path: Path) -> None:
    reg = tmp_path / "reg.jsonl"
    cfg = GenerateConfig(base_seed=5)
    config_registry.register(cfg, purpose="a", path=reg)
    config_registry.register(cfg, purpose="b", path=reg)  # same hash -> no new line
    lines = reg.read_text().strip().splitlines()
    assert len(lines) == 1
    assert config_registry.lookup(cfg.config_hash(), path=reg)["purpose"] == "a"


def test_registry_distinct_configs_append(tmp_path: Path) -> None:
    reg = tmp_path / "reg.jsonl"
    config_registry.register(GateConfig(elo1=5.0), path=reg)
    config_registry.register(GateConfig(elo1=30.0), path=reg)
    assert len(reg.read_text().strip().splitlines()) == 2


def test_registry_env_var_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg = tmp_path / "env_reg.jsonl"
    monkeypatch.setenv(config_registry.REGISTRY_ENV_VAR, str(reg))
    assert config_registry.default_registry_path() == reg
    cfg = EvalConfig(mode="cross_net", candidate="c.pt", baseline="b.pt")
    config_registry.register(cfg, purpose="via-env")
    assert reg.exists()
    assert config_registry.lookup(cfg.config_hash()) is not None


def test_registry_tolerates_partial_trailing_line(tmp_path: Path) -> None:
    reg = tmp_path / "reg.jsonl"
    cfg = TrainConfig(seed=1)
    config_registry.register(cfg, path=reg)
    with reg.open("a", encoding="utf-8") as fh:
        fh.write('{"config_hash": "sha256:trunc')  # crash mid-append
    assert config_registry.lookup(cfg.config_hash(), path=reg) is not None


# --------------------------------------------------------------------------- #
# Additive CLI helper.
# --------------------------------------------------------------------------- #


def _mini_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--mask-hidden-info", action="store_true")
    config_cli.add_config_flags(parser, default_purpose="test")
    return parser


def test_add_config_flags_is_noop_when_unused() -> None:
    parser = _mini_parser()
    args = parser.parse_args(["--seed", "9"])
    # Adding the flags does not perturb parsing of pre-existing flags.
    assert args.seed == 9
    assert args.value_loss_weight == 0.25
    assert args.mask_hidden_info is False
    assert args.config is None
    assert args.dump_config is None
    assert args.print_config_hash is False


def test_resolve_config_noop_run_registers_but_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config_registry.REGISTRY_ENV_VAR, str(tmp_path / "reg.jsonl"))
    parser = _mini_parser()
    args = parser.parse_args(["--seed", "3"])
    before = dict(vars(args))
    cfg = config_cli.resolve_config(args, TrainConfig.from_namespace, parser=parser)
    assert isinstance(cfg, TrainConfig)
    assert cfg.seed == 3
    # No CAT-66 flag set -> the namespace is untouched (argv-only == byte-identical).
    assert dict(vars(args)) == before


def test_dump_config_writes_and_registers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg = tmp_path / "reg.jsonl"
    monkeypatch.setenv(config_registry.REGISTRY_ENV_VAR, str(reg))
    dump = tmp_path / "cfg.json"
    parser = _mini_parser()
    args = parser.parse_args(
        ["--seed", "11", "--mask-hidden-info", "--dump-config", str(dump)]
    )
    cfg = config_cli.resolve_config(args, TrainConfig.from_namespace, parser=parser)
    assert dump.exists()
    loaded = config_cli.load_config(dump)
    assert loaded == cfg
    assert loaded.seed == 11 and loaded.mask_hidden_info is True
    assert config_registry.lookup(cfg.config_hash(), path=reg) is not None


def test_config_file_fills_only_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config_registry.REGISTRY_ENV_VAR, str(tmp_path / "reg.jsonl"))
    # Write a config file that sets seed=77 and value_loss_weight=1.0.
    src = TrainConfig(seed=77, value_loss_weight=1.0)
    cfg_file = tmp_path / "in.json"
    cfg_file.write_text(json.dumps(src.canonical_payload()))

    parser = _mini_parser()
    # Caller explicitly passes --seed 5 (must win) but leaves value-loss-weight default.
    args = parser.parse_args(["--seed", "5", "--config", str(cfg_file)])
    resolved = config_cli.resolve_config(
        args, TrainConfig.from_namespace, parser=parser
    )
    assert resolved.seed == 5  # explicit argv wins over file
    assert resolved.value_loss_weight == 1.0  # default -> filled from file


def test_explicit_cli_value_equal_to_default_still_wins_over_config(
    tmp_path: Path,
) -> None:
    """Presence in argv, not inequality from the default, defines explicit."""
    src = TrainConfig(seed=77)
    cfg_file = tmp_path / "in.json"
    cfg_file.write_text(json.dumps(src.canonical_payload()))

    parser = _mini_parser()
    argv = ["--seed", "1", "--config", str(cfg_file)]
    args = parser.parse_args(argv)
    resolved = config_cli.resolve_config(
        args, TrainConfig.from_namespace, parser=parser, argv=argv
    )

    assert resolved.seed == 1


def test_config_file_rejects_wrong_pipeline_before_applying_fields(tmp_path: Path) -> None:
    cfg_file = tmp_path / "eval.json"
    cfg_file.write_text(json.dumps(EvalConfig(mode="cross_net").canonical_payload()))
    parser = _mini_parser()
    argv = ["--config", str(cfg_file)]
    args = parser.parse_args(argv)

    with pytest.raises(SystemExit):
        config_cli.apply_config_file(
            args,
            parser,
            argv=argv,
            expected_pipeline=TrainConfig.PIPELINE,
        )
    assert args.seed == parser.get_default("seed")


def test_config_file_rejects_stale_schema_before_applying_fields(tmp_path: Path) -> None:
    payload = TrainConfig(seed=77).canonical_payload()
    payload["schema_version"] -= 1
    cfg_file = tmp_path / "stale.json"
    cfg_file.write_text(json.dumps(payload))
    parser = _mini_parser()
    argv = ["--config", str(cfg_file)]
    args = parser.parse_args(argv)

    with pytest.raises(SystemExit):
        config_cli.apply_config_file(
            args,
            parser,
            argv=argv,
            expected_pipeline=TrainConfig.PIPELINE,
        )
    assert args.seed == parser.get_default("seed")


def test_config_file_rejects_invalid_argparse_type(tmp_path: Path) -> None:
    payload = TrainConfig(seed=77).canonical_payload()
    payload["fields"]["seed"] = "not-an-integer"
    cfg_file = tmp_path / "invalid.json"
    cfg_file.write_text(json.dumps(payload))
    parser = _mini_parser()
    argv = ["--config", str(cfg_file)]
    args = parser.parse_args(argv)

    with pytest.raises(SystemExit):
        config_cli.apply_config_file(
            args,
            parser,
            argv=argv,
            expected_pipeline=TrainConfig.PIPELINE,
        )
    assert args.seed == parser.get_default("seed")


def test_generate_hash_binds_producer_checkpoint_bytes() -> None:
    first = GenerateConfig(
        checkpoint="champion.pt", producer_checkpoint_sha256="sha256:" + "a" * 64
    )
    second = dataclasses.replace(first, producer_checkpoint_sha256="sha256:" + "b" * 64)
    assert first.config_hash() != second.config_hash()


def test_generate_hash_binds_gpu_pipeline_sharing_topology() -> None:
    assert GenerateConfig(fleet_pipelines_per_gpu=1).config_hash() != GenerateConfig(
        fleet_pipelines_per_gpu=2
    ).config_hash()


def test_generate_hash_binds_learner_row_adapter_independently_of_teacher() -> None:
    tied = GenerateConfig(learner_entity_feature_adapter_version=None)
    advanced = GenerateConfig(
        learner_entity_feature_adapter_version=(
            "rust_entity_adapter_v3_structured_action_resources"
        )
    )

    assert tied.config_hash() != advanced.config_hash()


# --------------------------------------------------------------------------- #
# Ticket smoke test: a CLI-vs-dataclass mismatch is reflected in the hash.
# --------------------------------------------------------------------------- #


def test_smoke_cli_default_override_trap_is_caught_by_hash() -> None:
    """Simulate the c_scale 1.0-vs-0.1 incident: a launcher that silently used
    the wrong c_scale produces a *different* config hash than the correct run,
    so the two are never mistaken for the same regime (the class of bug that
    caused 7+ incidents can no longer hide)."""
    correct = GenerateConfig(c_scale=0.1)
    stale_launcher = GenerateConfig(c_scale=1.0)  # the pre-F1 default that leaked
    assert correct.config_hash() != stale_launcher.config_hash()
    # And the hash reflects the value actually used, not the dataclass default.
    assert (
        config_from_payload(json.loads(stale_launcher.canonical_json())).c_scale == 1.0
    )


def test_all_pipelines_registered_in_class_map() -> None:
    assert set(CONFIG_CLASSES) == {"train", "generate", "eval", "gate"}


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


def _mutate_one_field(cfg: PipelineConfig, field: dataclasses.Field) -> PipelineConfig:
    """Return a copy of ``cfg`` with ``field`` changed to a guaranteed-different value."""
    current = getattr(cfg, field.name)
    if isinstance(current, bool):
        new = not current
    elif isinstance(current, int):
        new = current + 1
    elif isinstance(current, float):
        new = current + 1.0
    elif isinstance(current, str):
        new = current + "_x"
    elif current is None:
        new = "sentinel_non_none"
    else:
        new = ("__mutated__",)
    return dataclasses.replace(cfg, **{field.name: new})
