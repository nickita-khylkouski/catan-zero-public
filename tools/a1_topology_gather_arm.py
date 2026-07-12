#!/usr/bin/env python3
"""Prepare (never launch) the function-preserving topology-gather K3 arm.

This arm is derived from a sealed corrected-anchor-K3 manifest.  Its training
command is identical except for the initialization/output identities and the
diagnostic ablation label.  The initialization must be an exact, gather-only
upgrade of the same f7 bytes and the bound corpora must prove that topology
targets are present and valid.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_corrected_policy_arm as corrected  # noqa: E402


SCHEMA = "a1-topology-gather-arm-manifest-v1"
SOURCE_SCHEMA = corrected.SCHEMA
EXECUTOR_RELATIVE_PATH = "tools/a1_topology_gather_arm_execute.py"
SOURCE_FILES = (
    "tools/a1_topology_gather_arm.py",
    EXECUTOR_RELATIVE_PATH,
    "tools/a1_corrected_policy_arm_execute.py",
    "tools/a1_corrected_policy_arm.py",
    "tools/f69_upgrade_checkpoint_config.py",
    "tools/audit_memmap_architecture_targets.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/entity_token_policy.py",
)
EXPECTED_NEW_PARAMETERS = (
    "target_gather_proj.0.bias",
    "target_gather_proj.0.weight",
    "target_gather_proj.1.bias",
    "target_gather_proj.1.weight",
)


class ArmError(RuntimeError):
    """The requested experiment is not the one-axis topology arm."""


def _load_source(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload, ref = corrected._load_json(path)  # noqa: SLF001
    stated = payload.get("manifest_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload.get("schema_version") != SOURCE_SCHEMA or stated != corrected._digest(  # noqa: SLF001
        unhashed
    ):
        raise ArmError("source corrected K3 manifest schema or digest is invalid")
    if not (
        payload.get("diagnostic_only") is True
        and payload.get("promotion_eligible") is False
        and payload.get("launch_authorized") is False
        and payload.get("diagnostic_execution_authorized") is True
        and payload.get("launch_interface_present")
        == "tools/a1_corrected_policy_arm_execute.py --go"
    ):
        raise ArmError("source corrected K3 manifest lacks its exact diagnostic executor")
    recipe = payload.get("recipe")
    command = payload.get("command")
    if (
        not isinstance(recipe, dict)
        or payload.get("recipe_sha256") != corrected._digest(recipe)  # noqa: SLF001
        or not isinstance(command, list)
        or not all(isinstance(item, str) for item in command)
        or payload.get("command_sha256") != corrected._digest(command)  # noqa: SLF001
    ):
        raise ArmError("source corrected K3 recipe/command binding is invalid")
    required_recipe = {
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "steps": 1024,
        "base_value_row_dose": 4_194_304,
        "policy_aux_active_batch_size_per_rank": 128,
        "policy_aux_active_row_dose": 1_048_576,
        "replay_supervised_policy": False,
        "replay_supervised_value": False,
        "replay_forward_kl_weight": 0.006,
        "soft_target_weight": 1.0,
        "fresh_optimizer": True,
        "independent_f7_initialization": True,
    }
    if any(recipe.get(key) != value for key, value in required_recipe.items()):
        raise ArmError("source is not the exact corrected anchor-only K3 recipe")
    initialization = payload.get("initialization")
    descriptor = payload.get("descriptor")
    sentinel = payload.get("validation_sentinel")
    if not all(isinstance(value, dict) for value in (initialization, descriptor, sentinel)):
        raise ArmError("source K3 omits initialization, descriptor, or sentinel identity")
    for identity in (initialization, descriptor, sentinel):
        if corrected._file_ref(Path(identity.get("path", ""))) != identity:  # noqa: SLF001
            raise ArmError("source K3 bound artifact bytes drifted")
    if corrected._option(command, "--init-checkpoint") != initialization["path"]:  # noqa: SLF001
        raise ArmError("source command/checkpoint identity mismatch")
    if corrected._option(command, "--data") != descriptor["path"]:  # noqa: SLF001
        raise ArmError("source command/descriptor identity mismatch")
    if corrected._option(command, "--validation-game-sentinel-manifest") != sentinel["path"]:  # noqa: SLF001
        raise ArmError("source command/validation sentinel identity mismatch")
    if "--validation-game-seed-manifest" in command:
        raise ArmError("source K3 command mixes validation controls")
    return payload, ref


def _torch_load(path: Path) -> Mapping[str, Any]:
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ArmError(f"checkpoint is not a mapping: {path}")
    return raw


def _config_fields(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        fields = raw.get("fields", raw)
        if isinstance(fields, Mapping):
            return dict(fields)
    if dataclasses.is_dataclass(raw):
        return {
            field.name: getattr(raw, field.name)
            for field in dataclasses.fields(raw)
            if hasattr(raw, field.name)
        }
    raise ArmError("checkpoint config cannot be normalized")


def _effective_config(raw: Any) -> dict[str, Any]:
    from catan_zero.rl.entity_token_policy import EntityGraphConfig

    values = _config_fields(raw)
    known = {field.name for field in dataclasses.fields(EntityGraphConfig)}
    try:
        return dataclasses.asdict(EntityGraphConfig(**{
            key: value for key, value in values.items() if key in known
        }))
    except (TypeError, ValueError) as error:
        raise ArmError(f"checkpoint config cannot instantiate current schema: {error}") from error


def _equal_artifact_value(left: Any, right: Any) -> bool:
    import numpy as np
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping) and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_equal_artifact_value(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right) and len(left) == len(right)
            and all(_equal_artifact_value(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def _validate_upgrade(source: Path, upgraded: Path) -> dict[str, Any]:
    """Prove the checkpoint delta is exactly the inert gather branch."""
    import torch

    source_ref = corrected._file_ref(source)  # noqa: SLF001
    upgraded_ref = corrected._file_ref(upgraded)  # noqa: SLF001
    before = _torch_load(Path(source_ref["path"]))
    after = _torch_load(Path(upgraded_ref["path"]))
    provenance = after.get("upgrade_provenance")
    if not isinstance(provenance, Mapping) or not (
        provenance.get("schema_version") == "entity-graph-upgrade-v1"
        and provenance.get("source_checkpoint_sha256")
        == source_ref["sha256"].removeprefix("sha256:")
        and provenance.get("flags") == {"action_target_gather": True}
        and provenance.get("forward_max_diff") == 0.0
        and provenance.get("forward_identical_at_init") is True
        and provenance.get("trained_value_readouts_added") == []
    ):
        raise ArmError("gather checkpoint lacks exact function-preserving provenance")
    source_config = _effective_config(before.get("config"))
    treatment_config = _effective_config(after.get("config"))
    expected_treatment_config = dict(source_config)
    expected_treatment_config["action_target_gather"] = True
    if treatment_config != expected_treatment_config or not (
        treatment_config.get("state_trunk", "transformer") == "transformer"
        and treatment_config.get("action_target_gather") is True
        and int(treatment_config.get("action_cross_attention_layers", 0)) == 0
        and treatment_config.get("edge_policy_head", False) is False
        and treatment_config.get("value_attention_pool", False) is False
        and treatment_config.get("relational_block_pattern", "") == ""
        and int(treatment_config.get("relational_ff_size", 0)) == 0
    ):
        raise ArmError("upgraded checkpoint effective config delta is not gather-only")
    provenance_drift = [
        key for key in before
        if key not in {"model", "config", "upgrade_provenance"}
        and (key not in after or not _equal_artifact_value(before[key], after[key]))
    ]
    if provenance_drift:
        raise ArmError(f"gather upgrade changed/dropped source provenance: {provenance_drift}")
    before_model, after_model = before.get("model"), after.get("model")
    if not isinstance(before_model, Mapping) or not isinstance(after_model, Mapping):
        raise ArmError("checkpoint model state is malformed")
    removed = sorted(set(before_model) - set(after_model))
    added = sorted(set(after_model) - set(before_model))
    if removed or tuple(added) != EXPECTED_NEW_PARAMETERS:
        raise ArmError(f"gather parameter identity drift: added={added} removed={removed}")
    changed = [
        name for name in before_model
        if not torch.equal(before_model[name], after_model[name])
    ]
    if changed:
        raise ArmError(f"shared f7 parameters changed during gather upgrade: {changed[:8]}")
    expected_init = {
        "target_gather_proj.0.weight": "ones",
        "target_gather_proj.0.bias": "zeros",
        "target_gather_proj.1.weight": "zeros",
        "target_gather_proj.1.bias": "zeros",
    }
    for name, kind in expected_init.items():
        tensor = after_model[name]
        reference = torch.ones_like(tensor) if kind == "ones" else torch.zeros_like(tensor)
        if not torch.equal(tensor, reference):
            raise ArmError(f"new gather parameter is not deterministic {kind}: {name}")
    return {
        "utility": "tools/f69_upgrade_checkpoint_config.py --flags gather",
        "source": source_ref,
        "upgraded": upgraded_ref,
        "flags": {"action_target_gather": True},
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
        "shared_parameter_count": len(before_model),
        "shared_parameters_bit_identical": True,
        "new_parameters": added,
        "new_parameter_initialization": expected_init,
    }


def _validate_coverage(path: Path, descriptor_path: Path) -> dict[str, Any]:
    audit, audit_ref = corrected._load_json(path)  # noqa: SLF001
    if audit.get("schema_version") != "memmap-architecture-target-audit-bundle-v1":
        raise ArmError("architecture target audit schema drift")
    descriptor, _ = corrected._preflight_descriptor(descriptor_path)  # noqa: SLF001
    components = descriptor["components"]
    supervised_ids = set(descriptor.get("policy_distillation_component_ids", ())) | set(
        descriptor.get("value_training_component_ids", ())
    )
    component_by_id = {row.get("component_id"): row for row in components}
    if not supervised_ids or not supervised_ids <= set(component_by_id):
        raise ArmError("K3 descriptor has an invalid supervised component scope")
    # Topology targets are consumed only by supervised policy/value rows.  The
    # replay component in K3 is deliberately KL-anchor-only, so requiring it in
    # the architecture audit both confuses the causal contract and rejects the
    # real two-corpus audit.  Bind the audited set by resolved corpus identity;
    # audit ordering is not semantically meaningful.
    expected_dirs = {
        str(Path(component_by_id[component_id]["corpus_dir"]).resolve())
        for component_id in supervised_ids
    }
    rows = audit.get("audits")
    audited_dirs = [row.get("corpus_dir") for row in rows] if isinstance(rows, list) else []
    if (
        len(audited_dirs) != len(expected_dirs)
        or len(set(audited_dirs)) != len(audited_dirs)
        or set(audited_dirs) != expected_dirs
    ):
        raise ArmError("coverage audit does not bind exactly the supervised K3 corpora")
    coverage = []
    for row in rows:
        legal = row.get("legal_action_targets", {})
        graph = row.get("graph_incidence", {})
        viability = row.get("viability", {})
        if not (
            viability.get("action_target_gather") is True
            and graph.get("out_of_range_ids") == 0
            and legal.get("invalid_legal_action_ids") == 0
            and legal.get("out_of_range_target_rows") == 0
            and legal.get("search_active_rows_with_any_target", 0) > 0
            and legal.get("actions_with_any_target", 0) > 0
        ):
            raise ArmError("K3 corpus lacks valid, learnable topology target coverage")
        coverage.append({
            "corpus_dir": row["corpus_dir"],
            "actions": legal.get("actions"),
            "actions_with_any_target": legal.get("actions_with_any_target"),
            "target_coverage": legal.get("target_coverage"),
            "rows_with_any_target": legal.get("rows_with_any_target"),
            "row_target_coverage": legal.get("row_target_coverage"),
            "search_active_rows_with_any_target": legal["search_active_rows_with_any_target"],
            "chosen_actions_with_any_target": legal.get("chosen_actions_with_any_target"),
        })
    return {"artifact": audit_ref, "components": coverage,
            "coverage_sha256": corrected._digest(coverage)}  # noqa: SLF001


def _source_binding(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    try:
        commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=repo, text=True).strip()
        subprocess.run(("git", "diff", "--quiet", "HEAD", "--", *SOURCE_FILES), cwd=repo, check=True)
        for relative in SOURCE_FILES:
            subprocess.run(("git", "ls-files", "--error-unmatch", relative), cwd=repo,
                           check=True, stdout=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArmError("topology arm sources must be clean tracked canonical bytes") from error
    files = {relative: corrected._file_ref(repo / relative) for relative in SOURCE_FILES}  # noqa: SLF001
    return {"repository_root": str(repo), "git_commit": commit, "files": files,
            "files_sha256": corrected._digest(files)}  # noqa: SLF001


def _derive_command(source: Sequence[str], *, upgraded: Path, output_root: Path) -> tuple[list[str], dict[str, Any]]:
    command = list(source)
    changes: dict[str, Any] = {}
    updates = {
        "--init-checkpoint": str(upgraded),
        "--checkpoint": str(output_root / "candidate.pt"),
        "--report": str(output_root / "train.report.json"),
    }
    for flag, value in updates.items():
        old = corrected._option(command, flag)  # noqa: SLF001
        corrected._set_option(command, flag, value)  # noqa: SLF001
        changes[flag] = {"source": old, "treatment": value}
    return command, changes


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    repo = args.repo.expanduser().resolve(strict=True)
    source, source_ref = _load_source(args.source_manifest)
    output_root = args.output_root.expanduser().resolve()
    for name in ("candidate.pt", "candidate.pt.optimizer.pt", "train.report.json"):
        if (output_root / name).exists():
            raise ArmError(f"refusing existing topology-arm output: {output_root / name}")
    source_init = Path(source["initialization"]["path"])
    upgraded = args.gather_checkpoint.expanduser().resolve(strict=True)
    upgrade = _validate_upgrade(source_init, upgraded)
    coverage = _validate_coverage(args.architecture_audit, Path(source["descriptor"]["path"]))
    command, changes = _derive_command(source["command"], upgraded=upgraded, output_root=output_root)
    # Everything that affects optimization remains byte-identical because the
    # derived argv changes only checkpoint/output paths.  Real corrected-K3
    # receipts are plain diagnostic torchrun commands and intentionally carry
    # no hidden A1 ablation metadata flags; the sealed manifest recipe/hash is
    # the authoritative objective identity.
    source_binding = _source_binding(repo)
    executor_ref = source_binding.get("files", {}).get(EXECUTOR_RELATIVE_PATH)
    if not isinstance(executor_ref, dict):
        raise ArmError("source binding does not authenticate the topology executor")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "diagnostic_execution_authorized": True,
        "launch_interface_present": f"{EXECUTOR_RELATIVE_PATH} --go",
        "diagnostic_executor": executor_ref,
        "source_corrected_k3_manifest": source_ref,
        "source_corrected_k3_manifest_sha256": source["manifest_sha256"],
        "source_recipe": source["recipe"],
        "source_recipe_sha256": source["recipe_sha256"],
        "descriptor": source["descriptor"],
        "validation_sentinel": source["validation_sentinel"],
        "validation_sentinel_selection_sha256": source[
            "validation_sentinel_selection_sha256"
        ],
        "initialization_source": source["initialization"],
        "initialization_treatment": upgrade["upgraded"],
        "function_preserving_upgrade": upgrade,
        "corpus_topology_target_coverage": coverage,
        "source_binding": source_binding,
        "only_declared_optimization_delta": "action_target_gather=true",
        "matched_contract": {
            "recipe_sha256": source["recipe_sha256"],
            "descriptor": source["descriptor"],
            "validation_sentinel": source["validation_sentinel"],
            "dose_sampler_objective_operator_unchanged": True,
            "optimizer_state_reused": False,
            "step0_network_outputs_bit_identical": True,
        },
        "allowlisted_command_changes": changes,
        "command": command,
        "command_sha256": corrected._digest(command),  # noqa: SLF001
        "executor_compatibility": {
            "executor": f"{EXECUTOR_RELATIVE_PATH} --go",
            "receipt_schema": "a1-topology-gather-arm-execution-receipt-v1",
            "compatible_now": True,
            "idle_topology": "exactly_8_visible_B200s",
            "one_shot": True,
        },
    }
    manifest["manifest_sha256"] = corrected._digest(manifest)  # noqa: SLF001
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "topology-gather-arm.manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ArmError(f"prepared manifest drift: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    return manifest, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--gather-checkpoint", required=True, type=Path)
    parser.add_argument("--architecture-audit", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo", default=REPO_ROOT, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    manifest, path = prepare(build_parser().parse_args(argv))
    print(json.dumps({"prepared": str(path), "launched": False,
                      "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
