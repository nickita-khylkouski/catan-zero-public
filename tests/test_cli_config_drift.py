"""Structural guard against config-default drift between the search/self-play
config dataclasses and the tools/ CLIs that construct them.

Two traps have fired in generation launchers, both because each CLI re-declared
its own copy of a default:

  * ``c_scale`` -- a launcher kept the pre-F1 ``c_scale=1.0`` (raw-Q sharpening)
    while the corrected sigma default is ``0.1`` (mctx's ``value_scale``). This
    silently produced near-one-hot training targets.
  * the coupled ``max_decisions`` / ``temperature_move_fraction`` pair -- the
    move-fraction is a fraction OF the cap, so raising the cap without
    re-deriving the fraction silently changes the number of temperature moves.

This test makes the dataclass the single source of truth: every CLI flag (or
Modal ``@app.local_entrypoint`` parameter) whose de-dashed name matches a field
of the dataclass it feeds MUST carry that field's default. Any divergence fails
here with a message naming the CLI, the flag, and both values -- so the trap is
structurally impossible to reintroduce silently.

Defaults are read statically (``ast``) so the test neither imports the heavy CLI
modules nor executes argparse; the dataclass defaults are the live source of
truth, imported directly.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from catan_zero.rl.gumbel_self_play import GumbelSelfPlayConfig
from catan_zero.rl.entity_feature_adapter import RUST_ENTITY_ADAPTER_V5
from catan_zero.rl.raw_selfplay import RawSelfPlayConfig
from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig
from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"

_MISSING = object()

# Some config fields deliberately use ``None`` as an inherit/defer sentinel.
# In particular, raw self-play has two different feature identities:
#
# * ``RawSelfPlayConfig.entity_feature_adapter_version`` is the schema used for
#   newly stored learner rows.
# * ``EntityGraphRustEvaluatorConfig.entity_feature_adapter_version=None``
#   binds behavior inference to the loaded checkpoint's exact adapter.
#
# They share a Python field name but must not be forced to share one concrete
# default.  Keep this exception explicit so unrelated concrete disagreements
# still fail closed.
_INHERITED_SHARED_DEFAULT_FIELDS = frozenset({"entity_feature_adapter_version"})

# Each CLI is compared only against the dataclass(es) it actually constructs, so
# an identically named flag that feeds a different config (e.g. raw's scalar
# ``--temperature`` vs the search's ``temperature`` field) is never cross-checked.
# EntityGraphRustEvaluatorConfig is included wherever a CLI builds the neural
# evaluator, so the evaluator's own flags (--value-scale, --prior-temperature,
# --value-squash, --public-observation) are guarded against default drift too.
ARGPARSE_CLIS: dict[str, tuple[type, ...]] = {
    "generate_gumbel_selfplay_data.py": (
        GumbelSelfPlayConfig,
        GumbelChanceMCTSConfig,
        EntityGraphRustEvaluatorConfig,
    ),
    "gumbel_search_vs_raw_h2h.py": (GumbelChanceMCTSConfig, EntityGraphRustEvaluatorConfig),
    "generate_raw_selfplay_data.py": (RawSelfPlayConfig, EntityGraphRustEvaluatorConfig),
}

# A CLI can intentionally choose a stricter current operating regime than a
# reusable dataclass's conservative compatibility default. Keep those choices
# named here instead of weakening drift detection globally:
#
# * generic evaluators default to authoritative observation so legacy
#   omniscient checkpoints remain usable;
# * the current raw-data CLI defaults to a masked checkpoint and public
#   behavior, and stores meaningful-history learner rows.
ARGPARSE_CLI_DEFAULT_OVERRIDES: dict[str, dict[str, object]] = {
    "generate_raw_selfplay_data.py": {
        "public_observation": True,
        "meaningful_public_history": True,
    },
}

# Modal factory exposes its parameters as @app.local_entrypoint function kwargs,
# not argparse; the defaults live in the function signatures.
ENTRYPOINT_CLIS: dict[str, dict] = {
    "modal_gumbel_factory.py": {
        "funcs": ("launch_gumbel_pilot", "launch_gumbel_gen"),
        "configs": (GumbelSelfPlayConfig, GumbelChanceMCTSConfig),
    },
}


def _dataclass_defaults(cls: type) -> dict[str, object]:
    out: dict[str, object] = {}
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            out[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            out[field.name] = field.default_factory()
    return out


def _merged_defaults(configs: tuple[type, ...]) -> dict[str, object]:
    """Union of the associated dataclass defaults, asserting shared fields agree."""
    merged: dict[str, object] = {}
    for cls in configs:
        for name, value in _dataclass_defaults(cls).items():
            if name in merged and merged[name] != value:
                inherited = name in _INHERITED_SHARED_DEFAULT_FIELDS and (
                    merged[name] is None or value is None
                )
                if inherited:
                    if merged[name] is None:
                        merged[name] = value
                    continue
                raise AssertionError(
                    f"associated configs disagree on shared field {name!r}: "
                    f"{merged[name]!r} vs {value!r} (from {cls.__name__})"
                )
            merged[name] = value
    return merged


def _flag_to_field(flag: str) -> str | None:
    """'--temperature-move-fraction' -> 'temperature_move_fraction'."""
    if not flag.startswith("--"):
        return None
    return flag[2:].replace("-", "_")


def _literal(node: ast.AST) -> object:
    if isinstance(node, ast.Name) and node.id == "RUST_ENTITY_ADAPTER_V5":
        return RUST_ENTITY_ADAPTER_V5
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return _MISSING  # non-literal default (a Name/expr) -- not our concern


def _parse_argparse_defaults(path: Path) -> dict[str, object]:
    """Map de-dashed long-flag name -> literal default for every add_argument."""
    tree = ast.parse(path.read_text(), filename=str(path))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        field = None
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                field = _flag_to_field(arg.value)
                if field is not None:
                    break
        if field is None:
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                value = _literal(kw.value)
                if value is not _MISSING:
                    defaults[field] = value
    return defaults


def _parse_entrypoint_defaults(path: Path, func_names: tuple[str, ...]) -> dict[str, dict[str, object]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = set(func_names)
    out: dict[str, dict[str, object]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in wanted:
            continue
        params = node.args.args
        defaults = node.args.defaults  # align to the tail of params
        offset = len(params) - len(defaults)
        collected: dict[str, object] = {}
        for i, default_node in enumerate(defaults):
            value = _literal(default_node)
            if value is not _MISSING:
                collected[params[offset + i].arg] = value
        out[node.name] = collected
    return out


def _same(a: object, b: object) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return a == b


def _mismatches(source: str, cli_defaults: dict[str, object], truth: dict[str, object]) -> list[str]:
    problems: list[str] = []
    for field, cli_value in cli_defaults.items():
        if cli_value is None:
            continue  # deprecated / omit-if-None sentinel: nothing to drift
        if field not in truth:
            continue  # flag does not feed a dataclass field we track
        if not _same(cli_value, truth[field]):
            problems.append(
                f"{source}: --{field.replace('_', '-')} default={cli_value!r} "
                f"diverges from dataclass default={truth[field]!r}"
            )
    return problems


def test_argparse_cli_defaults_match_dataclasses() -> None:
    problems: list[str] = []
    for filename, configs in ARGPARSE_CLIS.items():
        path = TOOLS_DIR / filename
        assert path.exists(), f"expected CLI missing: {path}"
        truth = _merged_defaults(configs)
        truth.update(ARGPARSE_CLI_DEFAULT_OVERRIDES.get(filename, {}))
        cli_defaults = _parse_argparse_defaults(path)
        problems.extend(_mismatches(filename, cli_defaults, truth))
    assert not problems, "config-default drift detected:\n" + "\n".join(problems)


def test_inherited_shared_default_does_not_conflict_with_concrete_row_schema() -> None:
    @dataclasses.dataclass
    class LearnerRows:
        entity_feature_adapter_version: str = "learner-v3"

    @dataclasses.dataclass
    class CheckpointOwnedEvaluator:
        entity_feature_adapter_version: str | None = None

    assert _merged_defaults(
        (LearnerRows, CheckpointOwnedEvaluator)
    )["entity_feature_adapter_version"] == "learner-v3"
    assert _merged_defaults(
        (CheckpointOwnedEvaluator, LearnerRows)
    )["entity_feature_adapter_version"] == "learner-v3"


def test_two_concrete_shared_adapter_defaults_still_fail_closed() -> None:
    @dataclasses.dataclass
    class AdapterV3:
        entity_feature_adapter_version: str = "v3"

    @dataclasses.dataclass
    class AdapterV4:
        entity_feature_adapter_version: str = "v4"

    with pytest.raises(AssertionError, match="associated configs disagree"):
        _merged_defaults((AdapterV3, AdapterV4))


def test_raw_selfplay_keeps_teacher_and_stored_row_adapter_roles_separate() -> None:
    assert EntityGraphRustEvaluatorConfig().entity_feature_adapter_version is None
    assert RawSelfPlayConfig().entity_feature_adapter_version is not None
    defaults = _parse_argparse_defaults(TOOLS_DIR / "generate_raw_selfplay_data.py")
    assert "entity_feature_adapter_version" not in defaults
    assert (
        defaults["learner_entity_feature_adapter_version"]
        == RUST_ENTITY_ADAPTER_V5
    )
    assert defaults["public_observation"] is True
    assert defaults["meaningful_public_history"] is True


def test_modal_entrypoint_defaults_match_dataclasses() -> None:
    problems: list[str] = []
    for filename, spec in ENTRYPOINT_CLIS.items():
        path = TOOLS_DIR / filename
        assert path.exists(), f"expected CLI missing: {path}"
        truth = _merged_defaults(spec["configs"])
        per_func = _parse_entrypoint_defaults(path, spec["funcs"])
        for func in spec["funcs"]:
            assert func in per_func, f"{filename}: entrypoint {func} not found"
            problems.extend(_mismatches(f"{filename}::{func}", per_func[func], truth))
    assert not problems, "config-default drift detected:\n" + "\n".join(problems)


def test_temperature_decisions_coupling_holds() -> None:
    """The move-fraction trap: the absolute --temperature-decisions default and
    the --max-decisions default must reproduce the dataclass move-fraction, so
    the two stay coupled by construction rather than by hand-recomputation."""
    defaults = _parse_argparse_defaults(TOOLS_DIR / "generate_gumbel_selfplay_data.py")
    assert "temperature_decisions" in defaults, "expected an absolute --temperature-decisions flag"
    assert "max_decisions" in defaults
    derived = float(defaults["temperature_decisions"]) / float(defaults["max_decisions"])
    truth = GumbelSelfPlayConfig().temperature_move_fraction
    assert math.isclose(derived, truth, rel_tol=0.0, abs_tol=1e-9), (
        f"temperature_decisions/max_decisions = {defaults['temperature_decisions']}/"
        f"{defaults['max_decisions']} = {derived} != GumbelSelfPlayConfig."
        f"temperature_move_fraction = {truth}"
    )


def test_raw_selfplay_uses_absolute_temperature_decisions() -> None:
    """generate_raw_selfplay_data.py must keep the absolute (cap-invariant) form."""
    defaults = _parse_argparse_defaults(TOOLS_DIR / "generate_raw_selfplay_data.py")
    assert defaults.get("temperature_decisions") == RawSelfPlayConfig().temperature_decisions
    assert "temperature_move_fraction" not in defaults, (
        "raw self-play must not reintroduce the fraction-of-cap coupling"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
