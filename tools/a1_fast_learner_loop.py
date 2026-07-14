#!/usr/bin/env python3
"""Run the selected short-dose learner or one deliberate treatment on 8 B200s.

This is deliberately a research runner, not another authorization framework.  It
starts from a *successful* one-dose manifest, changes the requested learner axis,
keeps the initializer and every other command-line option fixed, and executes the
full eight-rank job.  Scope treatments receive a descriptor and validation
sentinel rebound to the same held-out games.

The production-relevant baseline is ``selected_dose_no_symmetry``: 128 fresh
Adam steps from the supplied champion, local batch 512 on eight ranks (524,288
row draws total), and no D6 training augmentation. Matched gameplay isolated
this short controlled dose as the gain; D6 augmentation was neutral and a
65,536-draw gather-only arm did not improve its parent. ``temp_control`` remains
as a legacy receipt-replay name; new runs should use the explicit selected-dose
names.

The runner also reads the trained checkpoint's serialized ``config.fields``.
Older ad-hoc completion scripts read the wrapper dictionary as if it were the
config itself and consequently reported enabled auxiliary heads as disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import train_bc  # noqa: E402


ARMS = {
    "selected_dose",
    "selected_dose_no_symmetry",
    "temp_control",
    "current_policy_scope",
    "current_policy_double_dose",
    "current_value_scope",
    "current_both_scope",
    "pure_search_target",
    "current_policy_pure_search",
    "deployed_tanh_value",
    "pure_search_deployed_tanh",
    "pure_search_deployed_tanh_double_dose",
    "current_policy_deployed_tanh",
}
CURRENT_COMPONENTS = ["n128_current", "n256_current"]


class ResearchRunError(RuntimeError):
    """The requested run cannot be derived from the successful source job."""


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResearchRunError(f"expected JSON object: {path}")
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ResearchRunError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _json_value(value: Any) -> Any:
    """Normalize checkpoint metadata scalars without losing their value."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if type(value).__module__.startswith("numpy") and callable(
        getattr(value, "item", None)
    ):
        return _json_value(value.item())
    raise ResearchRunError(f"checkpoint config contains non-JSON value {type(value)!r}")


def _option(command: Sequence[str], flag: str) -> str:
    positions = [index for index, item in enumerate(command) if item == flag]
    equals = [
        item.split("=", 1)[1]
        for item in command
        if item.startswith(flag + "=")
    ]
    if len(positions) + len(equals) != 1:
        raise ResearchRunError(f"source command must contain exactly one {flag}")
    if equals:
        return equals[0]
    index = positions[0]
    if index + 1 >= len(command):
        raise ResearchRunError(f"source command has no value for {flag}")
    return command[index + 1]


def _set_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    equals = [
        index for index, item in enumerate(command) if item.startswith(flag + "=")
    ]
    if len(positions) + len(equals) != 1:
        raise ResearchRunError(f"source command must contain exactly one {flag}")
    if equals:
        command[equals[0]] = f"{flag}={value}"
        return
    index = positions[0]
    if index + 1 >= len(command):
        raise ResearchRunError(f"source command has no value for {flag}")
    command[index + 1] = value


def _set_boolean_option(command: list[str], flag: str, *, enabled: bool) -> None:
    """Normalize one BooleanOptionalAction without contradictory flags."""

    negative = "--no-" + flag.removeprefix("--")
    command[:] = [item for item in command if item not in {flag, negative}]
    command.append(flag if enabled else negative)


def _preflight_descriptor(path: Path) -> dict[str, Any]:
    try:
        return train_bc._preflight_memmap_composite_descriptor(path)  # noqa: SLF001
    except SystemExit as error:
        raise ResearchRunError(f"descriptor preflight failed: {error}") from error


def _derive_descriptor_and_sentinel(
    *,
    source_descriptor_path: Path,
    source_sentinel_path: Path,
    output_root: Path,
    arm: str,
) -> tuple[Path, Path, dict[str, Any]]:
    source_payload = _load_json(source_descriptor_path)
    source_meta = _preflight_descriptor(source_descriptor_path)
    component_ids = list(source_meta.get("component_ids", ()))
    if component_ids != ["n128_current", "n256_current", "gen3_replay"]:
        raise ResearchRunError(
            f"unexpected source component order: {component_ids!r}"
        )
    descriptor = dict(source_payload)
    if arm in {
        "current_policy_scope",
        "current_policy_double_dose",
        "current_policy_pure_search",
        "current_policy_deployed_tanh",
    }:
        descriptor["policy_distillation_component_ids"] = CURRENT_COMPONENTS
    elif arm == "current_value_scope":
        descriptor["value_training_component_ids"] = CURRENT_COMPONENTS
    elif arm == "current_both_scope":
        descriptor["policy_distillation_component_ids"] = CURRENT_COMPONENTS
        descriptor["value_training_component_ids"] = CURRENT_COMPONENTS
    else:
        raise ResearchRunError(f"arm {arm!r} does not need a derived descriptor")

    descriptor_path = output_root / "memmap_composite.json"
    _write_new_json(descriptor_path, descriptor)
    treatment_meta = _preflight_descriptor(descriptor_path)

    stable_fields = (
        "component_ids",
        "component_game_sampling_ratios",
        "policy_kl_anchor_component_ids",
        "stored_policy_component_temperatures",
        "learner_recipe_overrides",
    )
    drift = {
        key: {"source": source_meta.get(key), "treatment": treatment_meta.get(key)}
        for key in stable_fields
        if source_meta.get(key) != treatment_meta.get(key)
    }
    if drift:
        raise ResearchRunError(f"scope treatment changed matched data: {drift}")

    sentinel = _load_json(source_sentinel_path)
    if not (
        sentinel.get("source_composite_descriptor_file_sha256")
        == source_meta.get("descriptor_file_sha256")
        and sentinel.get("source_composite_descriptor_fingerprint")
        == source_meta.get("descriptor_fingerprint")
    ):
        raise ResearchRunError("source sentinel is not bound to source descriptor")
    sentinel["source_composite_descriptor_file_sha256"] = treatment_meta[
        "descriptor_file_sha256"
    ]
    sentinel["source_composite_descriptor_fingerprint"] = treatment_meta[
        "descriptor_fingerprint"
    ]
    sentinel_path = output_root / "validation.sentinel.json"
    _write_new_json(sentinel_path, sentinel)
    return descriptor_path, sentinel_path, treatment_meta


def _checkpoint_config_fields(path: Path) -> dict[str, Any]:
    """Return effective fields from both legacy and typed checkpoint configs."""

    import dataclasses
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ResearchRunError("checkpoint root is not a mapping")
    config = raw.get("config")
    if dataclasses.is_dataclass(config):
        return _json_value({
            field.name: getattr(config, field.name)
            for field in dataclasses.fields(config)
            if hasattr(config, field.name)
        })
    if isinstance(config, Mapping):
        fields = config.get("fields", config)
        if isinstance(fields, Mapping):
            return _json_value(dict(fields))
    raise ResearchRunError("checkpoint has no readable model config")


def derive_run(
    *,
    source_manifest_path: Path,
    output_root: Path,
    arm: str,
    trainer: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    if arm not in ARMS:
        raise ResearchRunError(f"unsupported arm {arm!r}; choose from {sorted(ARMS)}")
    source = _load_json(source_manifest_path)
    command = source.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ResearchRunError("source manifest has no command array")
    command = list(command)
    if trainer is not None:
        trainer = trainer.expanduser().resolve(strict=True)
        matches = [
            index
            for index, item in enumerate(command)
            if item.endswith("/tools/train_bc.py")
        ]
        if len(matches) != 1:
            raise ResearchRunError("source command does not have one trainer path")
        command[matches[0]] = str(trainer)

    # These are the scientific invariants that prevent the two historical
    # failures: candidate chaining and unequal optimizer dose.
    if _option(command, "--max-steps") != "128":
        raise ResearchRunError("source is not the selected 128-step dose")
    if _option(command, "--batch-size") != "512":
        raise ResearchRunError("source is not the selected local batch 512")
    if _option(command, "--nproc-per-node") != "8":
        raise ResearchRunError("source is not the selected eight-rank B200 recipe")
    if _option(command, "--lr") != "3e-05" or _option(command, "--lr-schedule") != "flat":
        raise ResearchRunError("source is not the selected flat 3e-5 LR recipe")
    if "--no-resume-optimizer" not in command:
        raise ResearchRunError("selected dose must start with a fresh optimizer")
    if _option(command, "--init-checkpoint") == _option(command, "--checkpoint"):
        raise ResearchRunError("source command chains its output as initializer")

    output_root = output_root.expanduser().resolve()
    checkpoint = output_root / "candidate.pt"
    report = output_root / "train.report.json"
    if any(path.exists() for path in (checkpoint, report)):
        raise ResearchRunError(f"run output is not fresh: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_descriptor_path = Path(_option(command, "--data")).resolve(strict=True)
    source_sentinel_path = Path(
        _option(command, "--validation-game-sentinel-manifest")
    ).resolve(strict=True)
    treatment_meta: dict[str, Any] | None = None
    if arm in {
        "current_policy_scope",
        "current_policy_double_dose",
        "current_value_scope",
        "current_both_scope",
        "current_policy_pure_search",
        "current_policy_deployed_tanh",
    }:
        descriptor, sentinel, treatment_meta = _derive_descriptor_and_sentinel(
            source_descriptor_path=source_descriptor_path,
            source_sentinel_path=source_sentinel_path,
            output_root=output_root,
            arm=arm,
        )
        _set_option(command, "--data", str(descriptor))
        _set_option(command, "--validation-game-sentinel-manifest", str(sentinel))

    if arm in {
        "pure_search_target",
        "current_policy_pure_search",
        "pure_search_deployed_tanh",
        "pure_search_deployed_tanh_double_dose",
    }:
        _set_option(command, "--soft-target-weight", "1.0")
    if arm in {
        "deployed_tanh_value",
        "pure_search_deployed_tanh",
        "pure_search_deployed_tanh_double_dose",
        "current_policy_deployed_tanh",
    }:
        if "--scalar-value-loss-transform" in command:
            _set_option(command, "--scalar-value-loss-transform", "deployed_tanh")
        else:
            command.extend(("--scalar-value-loss-transform", "deployed_tanh"))
    if arm == "pure_search_deployed_tanh_double_dose":
        # One clean point on the dose-response curve: reload the same f7
        # initializer and double samples without candidate chaining.
        _set_option(command, "--max-steps", "256")
    if arm == "current_policy_double_dose":
        _set_option(command, "--max-steps", "256")

    # D6 was causally neutral at this dose (149 wins versus 150 for the
    # no-symmetry arm on identical games), so the reusable baseline avoids its
    # transform cost. ``selected_dose`` preserves the source command exactly.
    if arm == "selected_dose_no_symmetry":
        _set_boolean_option(command, "--symmetry-augment", enabled=False)
        _set_boolean_option(command, "--symmetry-augment-events", enabled=False)

    _set_option(command, "--checkpoint", str(checkpoint))
    _set_option(command, "--report", str(report))
    manifest = {
        "schema_version": "a1-fast-learner-run-v1",
        "arm": arm,
        "created_at_unix_ns": time.time_ns(),
        "source_manifest": {
            "path": str(source_manifest_path.resolve(strict=True)),
            "sha256": _file_sha(source_manifest_path),
        },
        "initializer": {
            "path": _option(command, "--init-checkpoint"),
            "sha256": _file_sha(Path(_option(command, "--init-checkpoint"))),
        },
        "output_root": str(output_root),
        "command": command,
        "command_sha256": _canonical_sha(command),
        "trainer": next(
            item for item in command if item.endswith("/tools/train_bc.py")
        ),
        "descriptor_meta": treatment_meta,
        "selected_dose_contract": {
            "world_size": int(_option(command, "--nproc-per-node")),
            "local_batch_size": int(_option(command, "--batch-size")),
            "optimizer_steps": int(_option(command, "--max-steps")),
            "global_row_draws": (
                int(_option(command, "--nproc-per-node"))
                * int(_option(command, "--batch-size"))
                * int(_option(command, "--max-steps"))
            ),
            "fresh_optimizer": "--no-resume-optimizer" in command,
            "symmetry_augment": "--symmetry-augment" in command,
        },
        "status": "prepared",
    }
    return command, manifest


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument(
        "--trainer",
        type=Path,
        help="Optional trainer checkout for a treatment implemented after the source run",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    command, manifest = derive_run(
        source_manifest_path=args.source_manifest,
        output_root=args.output_root,
        arm=args.arm,
        trainer=args.trainer,
    )
    manifest_path = args.output_root.expanduser().resolve() / "run.manifest.json"
    _write_new_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "command": command}, indent=2))
    if not args.execute:
        return 0

    # systemd-run inherits a conservative RLIMIT_NOFILE even on hosts whose
    # interactive learner shell has already been tuned.  DataLoader workers and
    # eight DDP ranks need the production limit; set it here so the research
    # command behaves the same from SSH, systemd, or a scheduler.
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    desired_limit = min(max(soft_limit, 65_536), hard_limit)
    if desired_limit > soft_limit:
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired_limit, hard_limit))

    started = time.monotonic()
    completed = subprocess.run(command, check=False)
    result: dict[str, Any] = {
        "schema_version": "a1-fast-learner-result-v1",
        "arm": args.arm,
        "returncode": int(completed.returncode),
        "elapsed_sec": time.monotonic() - started,
        "manifest_sha256": _file_sha(manifest_path),
    }
    checkpoint = args.output_root.expanduser().resolve() / "candidate.pt"
    report = args.output_root.expanduser().resolve() / "train.report.json"
    if completed.returncode == 0 and checkpoint.is_file() and report.is_file():
        result["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": _file_sha(checkpoint),
            "effective_config": _checkpoint_config_fields(checkpoint),
        }
        result["report"] = {"path": str(report), "sha256": _file_sha(report)}
    _write_new_json(args.output_root.expanduser().resolve() / "run.result.json", result)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(run())
