#!/usr/bin/env python3
"""Decode, but never launch, the historical full-dose belief-resource arm.

The arm is a one-axis diagnostic derived from a verified production TEMP
replication manifest.  It changes only the initializer (through a reviewed,
function-preserving belief-head upgrade receipt), the two belief CLI flags,
and independent output paths.  All data, temperature, dose, optimizer, RNG,
masking, and objective settings are inherited byte-for-byte from TEMP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = (REPO_ROOT / "src").resolve(strict=True)
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != REPO_SRC]
sys.path.insert(0, str(REPO_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from tools import a1_function_preserving_upgrade as upgrade  # noqa: E402
from tools import a1_production_temperature_replication as temperature  # noqa: E402


SCHEMA = "a1-belief-resource-calibrated-arm-v2-obsolete-dose"
EXECUTOR_RELATIVE_PATH = "tools/a1_belief_resource_arm_execute.py"
LOSS_WEIGHT = 0.01
OBSOLETE_REASON = (
    "legacy belief-resource arm is bound to 4,194,304 rows / 1024 steps; "
    "commission the fresh head from selected-dose TEMP+geometry evidence before execution"
)
UPGRADE_EVIDENCE_FIELDS = (
    "module",
    "source",
    "upgraded_initializer",
    "flags",
    "initialization_seed",
    "forward_max_diff",
    "forward_identical_at_init",
    "shared_parameters_bit_identical",
    "shared_parameter_count",
    "new_parameters",
    "new_parameter_initialization",
    "effective_source_config_sha256",
    "effective_upgraded_config_sha256",
    "seeded_parameter_sha256",
)
SOURCE_FILES = (
    "tools/a1_belief_resource_arm.py",
    EXECUTOR_RELATIVE_PATH,
    "tools/a1_function_preserving_upgrade.py",
    "tools/a1_production_temperature_replication.py",
    "tools/a1_production_l1_rerun.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/belief_aux_targets.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
)
OUTPUT_NAMES = (
    "candidate.pt",
    "candidate.pt.optimizer.pt",
    "candidate.pt.training-progress.json",
    "train.report.json",
    "diagnostic-execution.claim.json",
    "diagnostic-execution.receipt.json",
    "stdout.log",
    "stderr.log",
)
MANIFEST_FIELDS = {
    "schema_version",
    "diagnostic_only",
    "promotion_eligible",
    "launch_authorized",
    "obsolete_reason",
    "diagnostic_execution_authorized",
    "launch_interface_present",
    "diagnostic_executor",
    "source_temperature_manifest",
    "source_temperature_manifest_sha256",
    "f7_parent",
    "initialization_treatment",
    "function_preserving_upgrade_receipt",
    "function_preserving_upgrade",
    "effective_treatment_architecture",
    "source_descriptor",
    "validation_sentinel",
    "component_bindings",
    "stored_policy_component_temperatures",
    "event_history_training_contract",
    "selected_dose",
    "runtime_python",
    "execution_preconditions",
    "source_binding",
    "only_declared_optimization_delta",
    "matched_contract",
    "command",
    "command_sha256",
    "output_root",
    "manifest_sha256",
}


class BeliefArmError(RuntimeError):
    """The requested learner is not the exact one-axis belief arm."""


def _fail(message: str) -> None:
    raise BeliefArmError(message)


def _is_nested(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _set_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    equals = [index for index, item in enumerate(command) if item.startswith(flag + "=")]
    if len(positions) + len(equals) != 1:
        _fail(f"source TEMP command must contain exactly one {flag}")
    if equals:
        command[equals[0]] = f"{flag}={value}"
    else:
        index = positions[0]
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            _fail(f"source TEMP command has no value for {flag}")
        command[index + 1] = value


def _append_unique_flag(command: list[str], flag: str, value: str | None = None) -> None:
    if flag in command or any(item.startswith(flag + "=") for item in command):
        _fail(f"source TEMP command already contains treatment flag {flag}")
    command.append(flag)
    if value is not None:
        command.append(value)


def _trainer_index(command: Sequence[str]) -> int:
    positions = [
        index for index, item in enumerate(command) if Path(item).name == "train_bc.py"
    ]
    if len(positions) != 1:
        _fail("source TEMP command must name exactly one train_bc.py")
    return positions[0]


def _derive_command(
    source: Sequence[str],
    *,
    trainer: Path,
    initializer: Path,
    output_root: Path,
) -> list[str]:
    command = list(source)
    command[_trainer_index(command)] = str(trainer.resolve(strict=True))
    _set_option(command, "--init-checkpoint", str(initializer.resolve(strict=True)))
    _set_option(command, "--checkpoint", str(output_root / "candidate.pt"))
    _set_option(command, "--report", str(output_root / "train.report.json"))
    _append_unique_flag(command, "--belief-resource-head")
    _append_unique_flag(command, "--belief-resource-loss-weight", str(LOSS_WEIGHT))
    return command


def _source_binding(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    try:
        commit = temperature.base._assert_bound_checkout(repo)  # noqa: SLF001
        for relative in SOURCE_FILES:
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", relative),
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError, temperature.base.L1Error) as error:
        raise BeliefArmError(
            "belief arm sources must be clean tracked public-main bytes"
        ) from error
    files = {
        relative: temperature.base._ref(repo / relative)  # noqa: SLF001
        for relative in SOURCE_FILES
    }
    return {
        "repository_root": str(repo),
        "public_main_commit": commit,
        "files": files,
        "files_sha256": temperature.base._digest(files),  # noqa: SLF001
    }


def _effective_config(checkpoint: Path) -> dict[str, Any]:
    import dataclasses
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphConfig

    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:
        raise BeliefArmError(f"cannot load upgraded initializer: {error}") from error
    if not isinstance(raw, Mapping):
        _fail("upgraded initializer root is not a mapping")
    config = raw.get("config")
    if dataclasses.is_dataclass(config):
        values = {
            field.name: getattr(config, field.name)
            for field in dataclasses.fields(config)
            if hasattr(config, field.name)
        }
    elif isinstance(config, Mapping):
        fields = config.get("fields", config)
        if not isinstance(fields, Mapping):
            _fail("upgraded initializer config is malformed")
        values = dict(fields)
    else:
        _fail("upgraded initializer has no entity-graph config")
    known = {field.name for field in dataclasses.fields(EntityGraphConfig)}
    effective = dataclasses.asdict(
        EntityGraphConfig(**{key: value for key, value in values.items() if key in known})
    )
    return _json_config(effective)


def _json_config(value: Any) -> Any:
    """Normalize checkpoint config scalars without weakening manifest hashing."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_config(item) for item in value]
    # PyTorch checkpoint metadata can preserve NumPy scalar classes. Their
    # ``item`` value is the exact native scalar represented by the checkpoint.
    if type(value).__module__.startswith("numpy") and callable(
        getattr(value, "item", None)
    ):
        return _json_config(value.item())
    _fail(f"belief initializer config contains non-JSON value: {type(value)!r}")


def _validate_upgrade_contract(
    receipt: Mapping[str, Any], *, f7_ref: Mapping[str, str]
) -> dict[str, Any]:
    if not (
        receipt.get("schema_version") == upgrade.SCHEMA
        and receipt.get("module") == upgrade.MODULE_BELIEF_RESOURCE_HEAD
        and receipt.get("source") == dict(f7_ref)
        and receipt.get("flags") == {"belief_resource_head": True}
        and receipt.get("forward_max_diff") == 0.0
        and receipt.get("forward_identical_at_init") is True
        and receipt.get("shared_parameters_bit_identical") is True
    ):
        _fail("belief initializer is not the reviewed exact f7 upgrade receipt")
    upgraded = receipt.get("upgraded_initializer")
    if not isinstance(upgraded, Mapping):
        _fail("belief upgrade receipt has no upgraded initializer")
    config = _effective_config(Path(str(upgraded.get("path", ""))))
    forbidden = {
        "action_target_gather": False,
        "topology_residual_adapter": False,
        "edge_policy_head": False,
        "aux_subgoal_heads": False,
        "action_cross_attention_layers": 0,
    }
    drift = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in forbidden.items()
        if config.get(key) != expected
    }
    if config.get("belief_resource_head") is not True or drift:
        _fail(f"belief initializer contains an unrelated architecture delta: {drift}")
    return config


def _fresh_output_root(output_root: Path, source_root: Path) -> Path:
    root = output_root.expanduser().resolve(strict=False)
    source = source_root.expanduser().resolve(strict=False)
    if _is_nested(root, source):
        _fail("belief output root must be independent of the TEMP output root")
    existing = [str(root / name) for name in OUTPUT_NAMES if (root / name).exists()]
    if existing:
        _fail(f"belief output identity is not fresh: {existing}")
    return root


def prepare(
    *,
    source_temperature_manifest: Path,
    upgrade_receipt: Path,
    repo: Path,
    output_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    try:
        source = temperature.verify(source_temperature_manifest)
        receipt = upgrade.verify_receipt(upgrade_receipt)
    except (temperature.TemperatureReplicationError, upgrade.UpgradeError) as error:
        raise BeliefArmError(str(error)) from error
    source_manifest = source["manifest"]
    source_command = source["command"]
    temperature._validate_recipe(  # noqa: SLF001
        source_command,
        descriptor=source_manifest["source_descriptor"]["path"],
        sentinel=source_manifest["validation_sentinel"]["path"],
        f7=source_manifest["f7_parent"]["path"],
    )
    config = _validate_upgrade_contract(receipt, f7_ref=source_manifest["f7_parent"])
    binding = _source_binding(repo)
    root = _fresh_output_root(output_root, source["output_root"])
    manifest_path = manifest_path.expanduser().resolve(strict=False)
    if manifest_path.exists():
        _fail(f"refusing existing belief manifest: {manifest_path}")
    command = _derive_command(
        source_command,
        trainer=Path(binding["files"]["tools/train_bc.py"]["path"]),
        initializer=Path(receipt["upgraded_initializer"]["path"]),
        output_root=root,
    )
    executor = binding["files"][EXECUTOR_RELATIVE_PATH]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "obsolete_reason": OBSOLETE_REASON,
        "diagnostic_execution_authorized": False,
        "launch_interface_present": "none: historical full-dose decoder only",
        "diagnostic_executor": executor,
        "source_temperature_manifest": source["manifest_ref"],
        "source_temperature_manifest_sha256": source_manifest["manifest_sha256"],
        "f7_parent": source_manifest["f7_parent"],
        "initialization_treatment": receipt["upgraded_initializer"],
        "function_preserving_upgrade_receipt": receipt["receipt"],
        "function_preserving_upgrade": {
            key: receipt[key] for key in UPGRADE_EVIDENCE_FIELDS
        },
        "effective_treatment_architecture": config,
        "source_descriptor": source_manifest["source_descriptor"],
        "validation_sentinel": source_manifest["validation_sentinel"],
        "component_bindings": source_manifest["component_bindings"],
        "stored_policy_component_temperatures": source_manifest[
            "stored_policy_component_temperatures"
        ],
        "event_history_training_contract": source_manifest[
            "event_history_training_contract"
        ],
        "selected_dose": source_manifest["selected_dose"],
        "runtime_python": source_manifest["runtime_python"],
        "execution_preconditions": source_manifest["execution_preconditions"],
        "source_binding": binding,
        "only_declared_optimization_delta": {
            "belief_resource_head": True,
            "belief_resource_loss_weight": LOSS_WEIGHT,
        },
        "matched_contract": {
            "exact_temperature_recipe": True,
            "independent_f7_initialization": True,
            "fresh_optimizer": True,
            "candidate_chaining": False,
            "topology_or_gather": False,
            "policy_kl_anchor_weight": 0.0,
            "global_samples": 4_194_304,
            "training_rng_rank_offset": True,
        },
        "command": command,
        "command_sha256": temperature.base._digest(command),  # noqa: SLF001
        "output_root": str(root),
    }
    manifest["manifest_sha256"] = temperature.base._digest(manifest)  # noqa: SLF001
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temperature.base._write_exclusive(manifest_path, manifest, mode=0o444)  # noqa: SLF001
    except temperature.base.L1Error as error:
        raise BeliefArmError(str(error)) from error
    return manifest


def verify(manifest_path: Path, *, require_fresh_output: bool = True) -> dict[str, Any]:
    try:
        manifest_ref = temperature.base._ref(manifest_path)  # noqa: SLF001
        manifest = temperature.base._load(Path(manifest_ref["path"]))  # noqa: SLF001
    except temperature.base.L1Error as error:
        raise BeliefArmError(str(error)) from error
    unhashed = dict(manifest)
    stated = unhashed.pop("manifest_sha256", None)
    if stated != temperature.base._digest(unhashed):  # noqa: SLF001
        _fail("belief manifest semantic digest drift")
    if set(manifest) != MANIFEST_FIELDS:
        _fail("belief manifest fields differ from schema")
    if not (
        manifest.get("schema_version") == SCHEMA
        and manifest.get("diagnostic_only") is True
        and manifest.get("promotion_eligible") is False
        and manifest.get("launch_authorized") is False
        and manifest.get("obsolete_reason") == OBSOLETE_REASON
        and manifest.get("diagnostic_execution_authorized") is False
        and manifest.get("launch_interface_present")
        == "none: historical full-dose decoder only"
    ):
        _fail("belief manifest authorization boundary drift")
    source_ref = manifest.get("source_temperature_manifest")
    temperature.base._verify_ref(source_ref, "source_temperature_manifest")  # noqa: SLF001
    source = temperature.verify(Path(source_ref["path"]))
    if manifest.get("source_temperature_manifest_sha256") != source["manifest"][
        "manifest_sha256"
    ]:
        _fail("source TEMP manifest semantic identity drift")
    receipt_ref = manifest.get("function_preserving_upgrade_receipt")
    temperature.base._verify_ref(receipt_ref, "upgrade_receipt")  # noqa: SLF001
    try:
        receipt = upgrade.verify_receipt(Path(receipt_ref["path"]))
    except upgrade.UpgradeError as error:
        raise BeliefArmError(str(error)) from error
    config = _validate_upgrade_contract(receipt, f7_ref=source["manifest"]["f7_parent"])
    if manifest.get("effective_treatment_architecture") != config:
        _fail("effective belief architecture drift")
    embedded = manifest.get("function_preserving_upgrade")
    expected_embedded = {key: receipt[key] for key in UPGRADE_EVIDENCE_FIELDS}
    if embedded != expected_embedded:
        _fail("embedded function-preserving evidence drift")
    binding = manifest.get("source_binding")
    if not isinstance(binding, Mapping):
        _fail("belief source binding is malformed")
    repo = Path(str(binding.get("repository_root", ""))).resolve(strict=True)
    try:
        temperature.base._assert_bound_checkout(  # noqa: SLF001
            repo, str(binding.get("public_main_commit", ""))
        )
    except temperature.base.L1Error as error:
        raise BeliefArmError(str(error)) from error
    files = binding.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_FILES):
        _fail("belief source binding is incomplete")
    if binding.get("files_sha256") != temperature.base._digest(files):  # noqa: SLF001
        _fail("belief source file-map digest drift")
    for relative, ref in files.items():
        path = temperature.base._verify_ref(ref, f"source.{relative}")  # noqa: SLF001
        if path != (repo / relative).resolve(strict=True):
            _fail(f"belief source path escaped checkout: {relative}")
    if manifest.get("diagnostic_executor") != files[EXECUTOR_RELATIVE_PATH]:
        _fail("belief executor is not authenticated by source binding")
    source_manifest = source["manifest"]
    exact_copies = (
        "f7_parent",
        "source_descriptor",
        "validation_sentinel",
        "component_bindings",
        "stored_policy_component_temperatures",
        "event_history_training_contract",
        "selected_dose",
        "runtime_python",
        "execution_preconditions",
    )
    drift = [key for key in exact_copies if manifest.get(key) != source_manifest.get(key)]
    if drift:
        _fail(f"belief arm changed TEMP causal contract fields: {drift}")
    if manifest.get("initialization_treatment") != receipt["upgraded_initializer"]:
        _fail("belief initializer/receipt binding drift")
    if manifest.get("only_declared_optimization_delta") != {
        "belief_resource_head": True,
        "belief_resource_loss_weight": LOSS_WEIGHT,
    }:
        _fail("belief optimization-delta declaration drift")
    root = Path(str(manifest.get("output_root", ""))).resolve(strict=False)
    if _is_nested(root, source["output_root"]):
        _fail("belief output root is not independent of TEMP")
    command = manifest.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        _fail("belief command is malformed")
    if manifest.get("command_sha256") != temperature.base._digest(command):  # noqa: SLF001
        _fail("belief command digest drift")
    expected = _derive_command(
        source["command"],
        trainer=Path(files["tools/train_bc.py"]["path"]),
        initializer=Path(receipt["upgraded_initializer"]["path"]),
        output_root=root,
    )
    if command != expected:
        _fail("belief command is not the exact one-axis TEMP derivation")
    matched = manifest.get("matched_contract")
    if matched != {
        "exact_temperature_recipe": True,
        "independent_f7_initialization": True,
        "fresh_optimizer": True,
        "candidate_chaining": False,
        "topology_or_gather": False,
        "policy_kl_anchor_weight": 0.0,
        "global_samples": 4_194_304,
        "training_rng_rank_offset": True,
    }:
        _fail("belief matched-contract declaration drift")
    if temperature.base._option(command, "--policy-kl-anchor-weight") != "0.0":  # noqa: SLF001
        _fail("belief arm must not enable a replay/KL anchor")
    temperature._validate_recipe(  # noqa: SLF001
        source["command"],
        descriptor=source_manifest["source_descriptor"]["path"],
        sentinel=source_manifest["validation_sentinel"]["path"],
        f7=source_manifest["f7_parent"]["path"],
    )
    if require_fresh_output:
        _fresh_output_root(root, source["output_root"])
    return {
        "manifest": manifest,
        "manifest_ref": manifest_ref,
        "repo": repo,
        "command": command,
        "output_root": root,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-temperature-manifest", required=True, type=Path)
    parser.add_argument("--upgrade-receipt", required=True, type=Path)
    parser.add_argument("--repo", default=REPO_ROOT, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = prepare(
        source_temperature_manifest=args.source_temperature_manifest,
        upgrade_receipt=args.upgrade_receipt,
        repo=args.repo,
        output_root=args.output_root,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "prepared": str(args.manifest.resolve()),
                "launched": False,
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
