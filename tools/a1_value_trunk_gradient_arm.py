#!/usr/bin/env python3
"""Seal the independent f7 scalar-value-to-trunk gradient diagnostic.

The source is the verified production TEMP learner.  The treatment keeps its
initializer, corpus, target temperatures, RNG, optimizer, 4.19M-row dose, and
8x512/global-4096 DDP geometry.  The only optimization delta is
``value_trunk_grad_scale: 1.0 -> 0.0``: scalar value predictions and value-head
updates remain unchanged while the scalar value objective cannot update the
shared trunk.  This tool deliberately prepares and verifies a plan; it never
launches compute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_production_temperature_replication as temperature  # noqa: E402
from tools import train_bc  # noqa: E402


SCHEMA = "a1-value-trunk-gradient-arm-plan-v1"
SOURCE_SCALE = 1.0
TREATMENT_SCALE = 0.0
ABLATION_ID = "scalar-value-stop-trunk-gradient"
SOURCE_FILES = (
    "tools/a1_value_trunk_gradient_arm.py",
    "tools/a1_production_temperature_replication.py",
    "tools/a1_production_l1_rerun.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
    "src/catan_zero/rl/pipeline_configs.py",
)
AUTH_FLAGS = (
    "--a1-learner-ablation-id",
    "--a1-effective-learner-recipe-json",
    "--a1-effective-learner-recipe-sha256",
    "--a1-ablation-code-binding-json",
    "--a1-ablation-code-tree-sha256",
    "--a1-reviewed-lock-file-sha256",
)
OUTPUT_NAMES = (
    "candidate.pt",
    "candidate.pt.optimizer.pt",
    "candidate.pt.training-progress.json",
    "train.report.json",
)
PREDICTED_FALSIFIER = {
    "hypothesis": (
        "the scalar-MSE gradient into the shared trunk is a material cause of "
        "one-dose TEMP learner strength loss"
    ),
    "mechanism_must_hold": {
        "initial_forward_max_abs_diff": 0.0,
        "value_head_parameter_gradient_ratio": 1.0,
        "scalar_value_trunk_gradient_ratio": 0.0,
        "policy_trunk_gradient_ratio": 1.0,
    },
    "strength_test": {
        "baseline": "matched production TEMP candidate",
        "operator": "exact paired n128+D6 candidate-vs-baseline evaluator",
        "complete_pairs": 600,
        "games": 1200,
        "supported_only_if": "superiority_pentanomial_sprt_decision_H1",
        "falsified_if": "superiority_pentanomial_sprt_decision_H0",
        "otherwise": "inconclusive_do_not_claim_mechanism",
    },
}
MANIFEST_FIELDS = {
    "schema_version",
    "diagnostic_only",
    "promotion_eligible",
    "launch_authorized",
    "source_temperature_manifest",
    "source_temperature_manifest_sha256",
    "f7_parent",
    "source_descriptor",
    "validation_sentinel",
    "selected_dose",
    "component_bindings",
    "stored_policy_component_temperatures",
    "event_history_training_contract",
    "repo_binding",
    "only_declared_optimization_delta",
    "gradient_routing_contract",
    "matched_contract",
    "predicted_falsifier",
    "command",
    "command_sha256",
    "output_root",
    "manifest_sha256",
}


class ValueTrunkArmError(RuntimeError):
    """The requested plan is not the exact scalar-value trunk diagnostic."""


def _fail(message: str) -> None:
    raise ValueTrunkArmError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _is_nested(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _trainer_index(command: Sequence[str]) -> int:
    positions = [
        index for index, item in enumerate(command) if Path(item).name == "train_bc.py"
    ]
    if len(positions) != 1:
        _fail("source TEMP command must name exactly one train_bc.py")
    return positions[0]


def _set_unique(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    equals = [
        index for index, item in enumerate(command) if item.startswith(flag + "=")
    ]
    if len(positions) + len(equals) != 1:
        _fail(f"source TEMP command must contain exactly one {flag}")
    if equals:
        command[equals[0]] = f"{flag}={value}"
        return
    position = positions[0]
    if position + 1 >= len(command) or command[position + 1].startswith("--"):
        _fail(f"source TEMP command has no value for {flag}")
    command[position + 1] = value


def _repo_binding(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    try:
        commit = temperature.base._assert_bound_checkout(repo)  # noqa: SLF001
        for relative in SOURCE_FILES:
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", relative),
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError, temperature.base.L1Error) as error:
        raise ValueTrunkArmError(
            "value-trunk arm sources must be clean tracked public-main bytes"
        ) from error
    return {
        "repository_root": str(repo),
        "public_main_commit": commit,
        "files": {
            relative: temperature.base._ref(repo / relative)  # noqa: SLF001
            for relative in SOURCE_FILES
        },
    }


def _code_binding(repo_binding: Mapping[str, Any]) -> dict[str, Any]:
    files = repo_binding["files"]
    records = [
        {
            "kind": "learner_code",
            "relative_path": relative,
            "path": files[relative]["path"],
            "sha256": files[relative]["sha256"],
        }
        for relative in SOURCE_FILES
    ]
    binding: dict[str, Any] = {
        "schema_version": "a1-value-trunk-ablation-code-binding-v1",
        "records": records,
    }
    binding["code_tree_sha256"] = temperature.base._digest(binding)  # noqa: SLF001
    return binding


def _source_effective_recipe(source_command: Sequence[str]) -> dict[str, Any]:
    parser = train_bc.build_parser()
    try:
        args = parser.parse_args(
            list(source_command)[_trainer_index(source_command) + 1 :]
        )
    except SystemExit as error:
        raise ValueTrunkArmError(
            "cannot parse the verified TEMP training command"
        ) from error
    source = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        args,
        {"world_size": 8, "rank": 0, "local_rank": 0, "enabled": True},
    )
    if "value_trunk_grad_scale" in source:
        _fail("source TEMP unexpectedly has non-default value-trunk routing")
    source["per_game_value_weight_mode"] = str(args.per_game_value_weight_mode)
    return source


def _derive_command(
    source: Sequence[str],
    *,
    trainer: Path,
    output_root: Path,
    repo_binding: Mapping[str, Any],
    reviewed_source_sha256: str,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    if any(flag in source for flag in AUTH_FLAGS):
        _fail("source TEMP already carries a learner-ablation authorization")
    command = list(source)
    command[_trainer_index(command)] = str(trainer.resolve(strict=True))
    _set_unique(command, "--checkpoint", str(output_root / "candidate.pt"))
    _set_unique(command, "--report", str(output_root / "train.report.json"))
    command.extend(("--value-trunk-grad-scale", str(TREATMENT_SCALE)))

    source_recipe = _source_effective_recipe(source)
    effective = dict(source_recipe)
    effective["value_trunk_grad_scale"] = TREATMENT_SCALE
    code_binding = _code_binding(repo_binding)
    command.extend(
        (
            "--a1-learner-ablation-id",
            ABLATION_ID,
            "--a1-effective-learner-recipe-json",
            _canonical(effective),
            "--a1-effective-learner-recipe-sha256",
            temperature.base._digest(effective),  # noqa: SLF001
            "--a1-ablation-code-binding-json",
            _canonical(code_binding),
            "--a1-ablation-code-tree-sha256",
            code_binding["code_tree_sha256"],
            "--a1-reviewed-lock-file-sha256",
            reviewed_source_sha256,
        )
    )
    return command, source_recipe, effective


def _fresh_root(output_root: Path, source_root: Path) -> Path:
    root = output_root.expanduser().resolve(strict=False)
    if _is_nested(root, source_root.expanduser().resolve(strict=False)):
        _fail("value-trunk output root must be independent of TEMP")
    existing = [str(root / name) for name in OUTPUT_NAMES if (root / name).exists()]
    if existing:
        _fail(f"value-trunk output identity is not fresh: {existing}")
    return root


def prepare(
    *,
    source_temperature_manifest: Path,
    repo: Path,
    output_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    try:
        source = temperature.verify(source_temperature_manifest)
    except temperature.TemperatureReplicationError as error:
        raise ValueTrunkArmError(str(error)) from error
    source_manifest = source["manifest"]
    temperature._validate_recipe(  # noqa: SLF001
        source["command"],
        descriptor=source_manifest["source_descriptor"]["path"],
        sentinel=source_manifest["validation_sentinel"]["path"],
        f7=source_manifest["f7_parent"]["path"],
    )
    dose = source_manifest["selected_dose"]
    if (
        dose.get("world_size") != 8
        or dose.get("per_rank_batch_size") != 512
        or dose.get("global_samples") != 4_194_304
        or dose.get("optimizer") != "fresh_adam"
    ):
        _fail("source TEMP is not the selected independent 8x512 one-dose geometry")
    binding = _repo_binding(repo)
    root = _fresh_root(output_root, source["output_root"])
    manifest_path = manifest_path.expanduser().resolve(strict=False)
    if manifest_path.exists():
        _fail(f"refusing existing value-trunk plan: {manifest_path}")
    command, source_recipe, effective_recipe = _derive_command(
        source["command"],
        trainer=Path(binding["files"]["tools/train_bc.py"]["path"]),
        output_root=root,
        repo_binding=binding,
        reviewed_source_sha256=source["manifest_ref"]["sha256"],
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "source_temperature_manifest": source["manifest_ref"],
        "source_temperature_manifest_sha256": source_manifest["manifest_sha256"],
        "f7_parent": source_manifest["f7_parent"],
        "source_descriptor": source_manifest["source_descriptor"],
        "validation_sentinel": source_manifest["validation_sentinel"],
        "selected_dose": dose,
        "component_bindings": source_manifest["component_bindings"],
        "stored_policy_component_temperatures": source_manifest[
            "stored_policy_component_temperatures"
        ],
        "event_history_training_contract": source_manifest[
            "event_history_training_contract"
        ],
        "repo_binding": binding,
        "only_declared_optimization_delta": {
            "value_trunk_grad_scale": {
                "source": SOURCE_SCALE,
                "treatment": TREATMENT_SCALE,
            }
        },
        "gradient_routing_contract": {
            "forward_value_identity": True,
            "value_head_parameter_gradient_scale": 1.0,
            "shared_state_upstream_gradient_scale": TREATMENT_SCALE,
            "policy_gradient_scale": 1.0,
            "ddp": "rank_local_scale_before_standard_allreduce",
        },
        "matched_contract": {
            "source_effective_recipe": source_recipe,
            "treatment_effective_recipe": effective_recipe,
            "exact_f7_initializer": True,
            "fresh_adam": True,
            "candidate_chaining": False,
            "global_samples": 4_194_304,
            "world_size": 8,
            "per_rank_batch_size": 512,
            "global_batch_size": 4096,
            "all_other_objectives_exact": True,
        },
        "predicted_falsifier": PREDICTED_FALSIFIER,
        "command": command,
        "command_sha256": temperature.base._digest(command),  # noqa: SLF001
        "output_root": str(root),
    }
    manifest["manifest_sha256"] = temperature.base._digest(manifest)  # noqa: SLF001
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temperature.base._write_exclusive(manifest_path, manifest, mode=0o444)  # noqa: SLF001
    except temperature.base.L1Error as error:
        raise ValueTrunkArmError(str(error)) from error
    return manifest


def verify(manifest_path: Path, *, require_fresh_output: bool = True) -> dict[str, Any]:
    try:
        manifest_ref = temperature.base._ref(manifest_path)  # noqa: SLF001
        manifest = temperature.base._load(Path(manifest_ref["path"]))  # noqa: SLF001
    except temperature.base.L1Error as error:
        raise ValueTrunkArmError(str(error)) from error
    unhashed = dict(manifest)
    stated = unhashed.pop("manifest_sha256", None)
    if stated != temperature.base._digest(unhashed):  # noqa: SLF001
        _fail("value-trunk plan semantic digest drift")
    if set(manifest) != MANIFEST_FIELDS:
        _fail("value-trunk plan fields differ from schema")
    if not (
        manifest.get("schema_version") == SCHEMA
        and manifest.get("diagnostic_only") is True
        and manifest.get("promotion_eligible") is False
        and manifest.get("launch_authorized") is False
    ):
        _fail("value-trunk plan authorization boundary drift")
    source_ref = manifest.get("source_temperature_manifest")
    temperature.base._verify_ref(source_ref, "source_temperature_manifest")  # noqa: SLF001
    source = temperature.verify(Path(source_ref["path"]))
    if manifest.get("source_temperature_manifest_sha256") != source["manifest"].get(
        "manifest_sha256"
    ):
        _fail("source TEMP semantic identity drift")
    binding = manifest.get("repo_binding")
    if not isinstance(binding, Mapping) or set(binding) != {
        "repository_root",
        "public_main_commit",
        "files",
    }:
        _fail("value-trunk repository binding is malformed")
    repo = Path(str(binding["repository_root"])).resolve(strict=True)
    try:
        temperature.base._assert_bound_checkout(  # noqa: SLF001
            repo, str(binding["public_main_commit"])
        )
    except temperature.base.L1Error as error:
        raise ValueTrunkArmError(str(error)) from error
    files = binding.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_FILES):
        _fail("value-trunk source binding is incomplete")
    for relative, ref in files.items():
        path = temperature.base._verify_ref(ref, f"source.{relative}")  # noqa: SLF001
        if path != (repo / relative).resolve(strict=True):
            _fail(f"value-trunk source path escaped checkout: {relative}")
    exact_copies = (
        "f7_parent",
        "source_descriptor",
        "validation_sentinel",
        "selected_dose",
        "component_bindings",
        "stored_policy_component_temperatures",
        "event_history_training_contract",
    )
    drift = [
        key for key in exact_copies if manifest.get(key) != source["manifest"].get(key)
    ]
    if drift:
        _fail(f"value-trunk plan changed TEMP causal contract fields: {drift}")
    if manifest.get("only_declared_optimization_delta") != {
        "value_trunk_grad_scale": {"source": 1.0, "treatment": 0.0}
    }:
        _fail("value-trunk optimization delta drift")
    if manifest.get("predicted_falsifier") != PREDICTED_FALSIFIER:
        _fail("value-trunk predicted falsifier drift")
    root = Path(str(manifest.get("output_root", ""))).resolve(strict=False)
    if _is_nested(root, source["output_root"]):
        _fail("value-trunk output root is not independent of TEMP")
    expected, source_recipe, effective_recipe = _derive_command(
        source["command"],
        trainer=Path(files["tools/train_bc.py"]["path"]),
        output_root=root,
        repo_binding=binding,
        reviewed_source_sha256=source["manifest_ref"]["sha256"],
    )
    command = manifest.get("command")
    if command != expected:
        _fail("value-trunk command is not the exact one-axis TEMP derivation")
    if manifest.get("command_sha256") != temperature.base._digest(command):  # noqa: SLF001
        _fail("value-trunk command digest drift")
    matched = manifest.get("matched_contract")
    if not isinstance(matched, Mapping) or (
        matched.get("source_effective_recipe") != source_recipe
        or matched.get("treatment_effective_recipe") != effective_recipe
        or matched.get("world_size") != 8
        or matched.get("per_rank_batch_size") != 512
        or matched.get("global_batch_size") != 4096
        or matched.get("global_samples") != 4_194_304
    ):
        _fail("value-trunk matched contract drift")
    if require_fresh_output:
        _fresh_root(root, source["output_root"])
    return {
        "manifest": manifest,
        "manifest_ref": manifest_ref,
        "source": source,
        "repo": repo,
        "command": command,
        "output_root": root,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-temperature-manifest", required=True, type=Path)
    prep.add_argument("--repo", default=REPO_ROOT, type=Path)
    prep.add_argument("--output-root", required=True, type=Path)
    prep.add_argument("--manifest", required=True, type=Path)
    check = sub.add_parser("verify")
    check.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "prepare":
        result = prepare(
            source_temperature_manifest=args.source_temperature_manifest,
            repo=args.repo,
            output_root=args.output_root,
            manifest_path=args.manifest,
        )
    else:
        result = verify(args.manifest)["manifest"]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
