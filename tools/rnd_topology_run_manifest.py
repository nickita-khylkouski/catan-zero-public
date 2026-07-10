#!/usr/bin/env python3
"""Build an authenticated run manifest for the topology learning gate.

This tool does not evaluate or modify a checkpoint.  It binds one completed
``train_bc`` run to the frozen experiment registration using the exact
``catan-zero-topology-run/v1`` object consumed by
``rnd_topology_holdout_export.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


RUN_SCHEMA = "catan-zero-topology-run/v1"


class ManifestError(ValueError):
    """An input cannot be authenticated as the requested registered run."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _existing_file(path: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ManifestError(f"{name} is not a file: {resolved}")
    return resolved


def _load_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{name} must contain a JSON object")
    return value


def _report_path(value: Any, *, report: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"training report {field} must be a non-empty path")
    path = Path(value).expanduser()
    return (report.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _prefixed_sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ManifestError(f"{field} must be a sha256:-prefixed lowercase digest")
    return value[7:]


def _validate_experiment(
    experiment: Mapping[str, Any], *, arm: str, training_seed: int
) -> None:
    if experiment.get("config_sha256_scope") != "canonical_json_without_config_sha256":
        raise ManifestError("experiment config has an unsupported self-hash scope")
    semantic = dict(experiment)
    declared = semantic.pop("config_sha256", None)
    if declared != _canonical_sha(semantic):
        raise ManifestError("experiment config self-hash is invalid")

    arms = experiment.get("arms")
    if not isinstance(arms, list):
        raise ManifestError("experiment config arms must be a list")
    matches = [item for item in arms if isinstance(item, dict) and item.get("arm_id") == arm]
    if len(matches) != 1:
        raise ManifestError(f"experiment config must contain exactly one arm {arm!r}")

    gate = experiment.get("learning_gate")
    seeds = gate.get("seeds") if isinstance(gate, dict) else None
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ManifestError("experiment learning_gate.seeds must be unique integers")
    if training_seed not in seeds:
        raise ManifestError(
            f"training seed {training_seed} is not registered for arm {arm!r}"
        )


def _file_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "file_sha256": _sha256_file(path)}


def _publish_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish complete bytes without ever replacing an existing destination."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ManifestError(f"refusing to overwrite {destination}")
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ManifestError(f"refusing to overwrite {destination}") from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_run_manifest(
    *,
    arm: str,
    training_seed: int,
    training_manifest: Path,
    training_report: Path,
    experiment_config: Path,
    checkpoint: Path,
    optimizer_sidecar: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate, bind, and exclusively publish one topology-run manifest."""

    if not isinstance(arm, str) or not arm.strip() or arm != arm.strip():
        raise ManifestError("arm must be a non-empty canonical string")
    if type(training_seed) is not int:
        raise ManifestError("training_seed must be an integer")
    if output.expanduser().resolve().exists():
        raise ManifestError(f"refusing to overwrite {output.expanduser().resolve()}")

    training_manifest = _existing_file(training_manifest, name="training manifest")
    training_report = _existing_file(training_report, name="training report")
    experiment_config = _existing_file(experiment_config, name="experiment config")
    checkpoint = _existing_file(checkpoint, name="checkpoint")
    optimizer_sidecar = _existing_file(optimizer_sidecar, name="optimizer sidecar")

    experiment = _load_object(experiment_config, name="experiment config")
    _validate_experiment(experiment, arm=arm, training_seed=training_seed)
    selected_games = _load_object(training_manifest, name="training manifest")
    if selected_games.get("schema_version") != "a1-selected-training-games-v1":
        raise ManifestError("training manifest has an unsupported schema")
    gate = experiment["learning_gate"]
    training_sha = _sha256_file(training_manifest)
    if gate.get("training_manifest_sha256") != training_sha:
        raise ManifestError("training manifest differs from the experiment registration")

    report = _load_object(training_report, name="training report")
    if report.get("seed") != training_seed:
        raise ManifestError("training report seed differs from the requested registered seed")
    if _report_path(report.get("checkpoint"), report=training_report, field="checkpoint") != checkpoint:
        raise ManifestError("training report checkpoint path differs from the requested checkpoint")
    checkpoint_sha = _sha256_file(checkpoint)
    if _prefixed_sha(report.get("checkpoint_sha256"), field="training report checkpoint_sha256") != checkpoint_sha:
        raise ManifestError("training report checkpoint digest differs from checkpoint bytes")

    canonical_sidecar = Path(str(checkpoint) + ".optimizer.pt").resolve()
    if optimizer_sidecar != canonical_sidecar:
        raise ManifestError("optimizer sidecar is not the checkpoint's canonical sidecar")
    if _report_path(report.get("optimizer_sidecar"), report=training_report, field="optimizer_sidecar") != optimizer_sidecar:
        raise ManifestError("training report optimizer-sidecar path differs from the requested sidecar")
    sidecar_sha = _sha256_file(optimizer_sidecar)
    if _prefixed_sha(
        report.get("optimizer_sidecar_sha256"),
        field="training report optimizer_sidecar_sha256",
    ) != sidecar_sha:
        raise ManifestError("training report optimizer-sidecar digest differs from sidecar bytes")

    payload = {
        "schema_version": RUN_SCHEMA,
        "arm": arm,
        "training_seed": training_seed,
        "training_manifest_sha256": training_sha,
        "training_report": _file_ref(training_report),
        "experiment_config": _file_ref(experiment_config),
        "checkpoint": {"path": str(checkpoint), "file_sha256": checkpoint_sha},
        "optimizer_sidecar": {
            "path": str(optimizer_sidecar),
            "file_sha256": sidecar_sha,
        },
    }
    _publish_json_exclusive(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--optimizer-sidecar", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        payload = build_run_manifest(
            arm=args.arm,
            training_seed=args.training_seed,
            training_manifest=args.training_manifest,
            training_report=args.training_report,
            experiment_config=args.experiment_config,
            checkpoint=args.checkpoint,
            optimizer_sidecar=args.optimizer_sidecar,
            output=args.output,
        )
    except ManifestError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
