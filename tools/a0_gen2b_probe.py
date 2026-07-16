#!/usr/bin/env python3
"""Seal and execute the exact A0 gen2B scalar-vs-HL reuse probe.

This is intentionally narrower than a generic experiment launcher.  It exists
to make the roadmap's highest-leverage mechanism test hard to accidentally
change:

* known gen1/report/corpus-meta SHA-256 anchors must match;
* every file in the memmap corpus is hashed into an immutable inventory;
* the historical report is the recipe authority and its validation split is
  reconstructed into explicit game-seed ranges;
* the exact row order for all three epochs is hashed;
* the categorical init is built deterministically and attested;
* both arms start fresh Adam state and run on distinct one-GPU visibility
  islands (``CUDA_VISIBLE_DEVICES=0`` and ``=1`` by default).

The normal workflow on the B200 checkout is::

    /home/ubuntu/catan-zero/.venv/bin/python tools/a0_gen2b_probe.py seal \
      --manifest configs/experiments/a0_gen2b_hlgauss.json \
      --repo-root /home/ubuntu/catan-zero-rnd \
      --artifact-root /home/ubuntu/catan-zero \
      --python /home/ubuntu/catan-zero/.venv/bin/python
    /home/ubuntu/catan-zero/.venv/bin/python tools/a0_gen2b_probe.py run \
      --lock /home/ubuntu/catan-zero/runs/rl_program_20260709/a0_gen2b_hlgauss/a0.lock.json \
      --repo-root /home/ubuntu/catan-zero-rnd

``seal`` is CPU-only. ``run`` refuses a changed input, code tree, Python/Torch
environment, busy GPU, existing output, or same physical GPU for both arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_REPO_FROM_TOOL = Path(__file__).resolve().parents[1]
_LOCAL_SRC = _REPO_FROM_TOOL / "src"
if str(_LOCAL_SRC) not in sys.path:
    # Checkpoint configs are pickled under catan_zero.*. Prefer the exact local
    # source tree being hashed/locked over any older installed wheel.
    sys.path.insert(0, str(_LOCAL_SRC))


SCHEMA = "a0-gen2b-probe-manifest-v1"
LOCK_SCHEMA = "a0-gen2b-probe-lock-v1"
VALIDATION_SCHEMA = "a0-gen2b-validation-lock-v1"
RESULT_SCHEMA = "a0-gen2b-probe-result-v1"
_REPORT_SENTINEL = "$report"
_CHUNK = 8 * 1024 * 1024

_BOOLEAN_OPTIONAL = {
    "fused_optimizer",
    "symmetry_augment",
    "symmetry_augment_events",
}
_STORE_TRUE = {
    "mask_hidden_info",
    "graph_history_features",
    "value_uncertainty_head",
    "aux_subgoal_heads",
    "edge_policy_head",
    "train_value_only",
    "per_game_policy_weight",
    "per_game_value_weight",
    "allow_teacher_score_q_loss",
    "allow_legacy_action_mask_upgrade",
    "trust_curated_data_quality",
    "require_strict_35m_teacher",
    "require_production_35m_teacher",
    "require_35m_model",
}
_MAP_FIELDS = {"teacher_weights", "phase_weights", "value_phase_weights"}
_LIST_FIELDS = {"q_skip_teacher_prefixes"}


class ContractError(SystemExit):
    """A fail-closed A0 contract violation."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _report_path_matches(report_value: Any, expected_relative: str) -> bool:
    """Accept historical absolute paths while pinning their repo-relative tail."""

    if not isinstance(report_value, str) or not report_value:
        return False
    actual_parts = Path(report_value).parts
    expected_parts = Path(expected_relative).parts
    return len(actual_parts) >= len(expected_parts) and tuple(
        actual_parts[-len(expected_parts) :]
    ) == tuple(expected_parts)


def _require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ContractError(f"missing {label}: {path}")
    actual = _sha256(path)
    if actual != str(expected).removeprefix("sha256:"):
        raise ContractError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual} ({path})"
        )
    return actual


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA:
        raise ContractError(
            f"manifest schema must be {SCHEMA!r}, got {manifest.get('schema_version')!r}"
        )
    if (
        manifest.get("diagnostic_only") is not True
        or manifest.get("promotion_eligible") is not False
    ):
        raise ContractError(
            "historical A0 loser-weight replay must declare "
            "diagnostic_only=true and promotion_eligible=false"
        )
    if "loser_sample_weight=0.3" not in str(manifest.get("obsolete_recipe_note", "")):
        raise ContractError(
            "historical A0 manifest must label loser_sample_weight=0.3 obsolete"
        )
    inputs = manifest.get("inputs")
    recipe = manifest.get("recipe")
    upgrade = manifest.get("upgrade")
    gpus = manifest.get("physical_gpus")
    if not all(isinstance(v, Mapping) for v in (inputs, recipe, upgrade, gpus)):
        raise ContractError(
            "manifest inputs/recipe/upgrade/physical_gpus must be objects"
        )
    if int(upgrade.get("bins", 0)) != 33 or int(upgrade.get("seed", -1)) != 1:
        raise ContractError(
            "A0 requires deterministic catbins:33 with initialization seed 1"
        )
    if int(recipe.get("epochs", 0)) != 3:
        raise ContractError("A0 requires exactly three epochs")
    if bool(recipe.get("mask_hidden_info")) is not True:
        raise ContractError("A0 requires masked/public-observation training")
    if bool(recipe.get("symmetry_augment")):
        raise ContractError("A0 exact replication must not add symmetry augmentation")
    if int(recipe.get("seed", -1)) != 1:
        raise ContractError("A0 training seed must be 1")
    scalar_gpu = str(gpus.get("scalar", ""))
    hl_gpu = str(gpus.get("hlgauss", ""))
    if not scalar_gpu or not hl_gpu or scalar_gpu == hl_gpu:
        raise ContractError("scalar and HL arms require two distinct physical GPU ids")
    if "," in scalar_gpu or "," in hl_gpu:
        raise ContractError("each A0 arm must see exactly one physical GPU")
    required = manifest.get("required_historical_report_fields")
    if not isinstance(required, list) or not required:
        raise ContractError(
            "required_historical_report_fields must be a non-empty list"
        )
    unknown_report_sentinels = {
        key
        for key, value in recipe.items()
        if value == _REPORT_SENTINEL and key not in set(required)
    }
    if unknown_report_sentinels:
        raise ContractError(
            "$report fields must be required historical fields: "
            f"{sorted(unknown_report_sentinels)}"
        )


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-12
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return dict(left) == dict(right)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return list(left) == list(right)
    return left == right


def _resolve_recipe(
    manifest: Mapping[str, Any], historical_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve report-derived fields and reject recipe/report disagreement."""

    recipe = dict(manifest["recipe"])
    required = [str(key) for key in manifest["required_historical_report_fields"]]
    missing = [key for key in required if key not in historical_report]
    if missing:
        raise ContractError(
            "historical report is missing recipe fields required for an exact A0: "
            f"{missing}"
        )
    for key, value in list(recipe.items()):
        if value == _REPORT_SENTINEL:
            recipe[key] = historical_report[key]
            continue
        # Newer fields may be absent from the old report; in that case their
        # neutral value is explicit in the manifest. If an old report DOES carry
        # a field, it is authoritative and must agree -- including lambda,
        # weighting and sampling knobs that could otherwise invalidate A0.
        if key in historical_report:
            if not _same_value(value, historical_report[key]):
                raise ContractError(
                    f"historical recipe drift for {key}: manifest={value!r}, "
                    f"report={historical_report[key]!r}"
                )
    expected_rows = int(manifest["inputs"]["expected_rows"])
    if int(historical_report["samples"]) != expected_rows:
        raise ContractError(
            f"historical report samples={historical_report['samples']} != {expected_rows}"
        )
    for key in ("validation_fraction", "validation_max_samples", "validation_seed"):
        if key not in recipe:
            raise ContractError(f"resolved recipe has no {key}")
    if not 0.0 < float(recipe["validation_fraction"]) < 0.9:
        raise ContractError("historical validation_fraction must be in (0, 0.9)")
    if int(recipe["validation_max_samples"]) < 0:
        raise ContractError("historical validation_max_samples must be >= 0")
    return recipe


def _corpus_inventory(corpus_dir: Path) -> tuple[list[dict[str, Any]], str]:
    if not corpus_dir.is_dir():
        raise ContractError(f"missing memmap corpus directory: {corpus_dir}")
    entries: list[dict[str, Any]] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"corpus inventory refuses symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContractError(f"corpus inventory refuses non-regular file: {path}")
        entries.append(
            {
                "path": path.relative_to(corpus_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not entries:
        raise ContractError(f"empty corpus directory: {corpus_dir}")
    return entries, _digest_json(entries)


def _load_game_seeds(corpus_dir: Path, meta: Mapping[str, Any]) -> np.ndarray:
    columns = meta.get("columns")
    if not isinstance(columns, Mapping) or "game_seed" not in columns:
        raise ContractError(
            "A0 corpus has no game_seed column; exact game split is impossible"
        )
    schema = columns["game_seed"]
    if not isinstance(schema, Mapping) or schema.get("kind") != "fixed":
        raise ContractError("A0 game_seed must be a fixed memmap column")
    inner = tuple(int(value) for value in schema.get("inner_shape", []))
    if inner not in ((), (1,)):
        raise ContractError(f"unexpected game_seed inner_shape={inner}")
    row_count = int(meta["row_count"])
    path = corpus_dir / "game_seed.dat"
    try:
        seeds = np.memmap(
            path,
            dtype=np.dtype(str(schema["dtype"])),
            mode="r",
            shape=(row_count, *inner),
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ContractError(f"cannot map {path}: {exc}") from exc
    return np.asarray(seeds, dtype=np.int64).reshape(row_count)


def _int64_set_sha(values: np.ndarray) -> str:
    canonical = np.sort(np.unique(np.asarray(values, dtype=np.int64))).astype(
        "<i8", copy=False
    )
    return f"sha256:{hashlib.sha256(canonical.tobytes()).hexdigest()}"


def _compress_ranges(values: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous])
        start = previous = value
    ranges.append([start, previous])
    return ranges


def _ranges_cli(ranges: Sequence[Sequence[int]]) -> str:
    return ",".join(f"{int(start)}:{int(end)}" for start, end in ranges)


def _validation_and_order_contract(
    seeds: np.ndarray,
    *,
    validation_fraction: float,
    validation_seed: int,
    validation_max_samples: int,
    train_seed: int,
    epochs: int,
) -> dict[str, Any]:
    """Reproduce train_bc's game split and exact single-rank epoch order."""

    n = int(seeds.size)
    unique, counts = np.unique(seeds, return_counts=True)
    if n == 0 or unique.size <= 1:
        raise ContractError("A0 requires a non-empty, non-degenerate game_seed column")
    rng = np.random.default_rng(int(validation_seed))
    shuffled = rng.permutation(unique)
    count_by_seed = {int(seed): int(count) for seed, count in zip(unique, counts)}
    target_rows = max(1, int(round(n * float(validation_fraction))))
    selected: list[int] = []
    selected_rows = 0
    for seed in shuffled:
        selected.append(int(seed))
        selected_rows += count_by_seed[int(seed)]
        if selected_rows >= target_rows:
            break
    selected_array = np.asarray(selected, dtype=np.int64)
    validation_mask = np.isin(seeds, selected_array)
    train_indices = np.flatnonzero(~validation_mask).astype(np.int64, copy=False)
    validation_indices = np.flatnonzero(validation_mask).astype(np.int64, copy=False)
    if validation_max_samples > 0 and validation_indices.size > validation_max_samples:
        cap_rng = np.random.default_rng(int(validation_seed) + 1)
        validation_indices = np.sort(
            cap_rng.choice(
                validation_indices,
                size=int(validation_max_samples),
                replace=False,
            )
        )
    held_out = np.sort(np.unique(seeds[validation_indices])).astype(np.int64)
    order_digest = hashlib.sha256()
    order_rng = np.random.default_rng(int(train_seed))
    for epoch in range(int(epochs)):
        positions = order_rng.permutation(train_indices.size)
        row_order = np.asarray(train_indices[positions], dtype="<i8")
        order_digest.update(int(epoch + 1).to_bytes(4, "little"))
        order_digest.update(row_order.tobytes())
    selected_ranges = _compress_ranges(selected)
    return {
        "schema_version": VALIDATION_SCHEMA,
        "row_count": n,
        "unique_game_seed_count": int(unique.size),
        "validation_fraction": float(validation_fraction),
        "validation_seed": int(validation_seed),
        "validation_max_samples": int(validation_max_samples),
        "selected_game_seed_count_before_row_cap": len(selected),
        "selected_game_seed_set_sha256_before_row_cap": _int64_set_sha(selected_array),
        "selected_game_seed_ranges": selected_ranges,
        "selected_game_seed_ranges_cli": _ranges_cli(selected_ranges),
        "validation_rows": int(validation_indices.size),
        "validation_game_seed_count_after_row_cap": int(held_out.size),
        "validation_game_seed_set_sha256": _int64_set_sha(held_out),
        "validation_game_seeds": [int(seed) for seed in held_out],
        "train_rows": int(train_indices.size),
        "train_row_set_sha256": f"sha256:{hashlib.sha256(np.asarray(train_indices, dtype='<i8').tobytes()).hexdigest()}",
        "train_seed": int(train_seed),
        "epochs": int(epochs),
        "epoch_row_order_sha256": f"sha256:{order_digest.hexdigest()}",
    }


def _extract_value_trace(report: Mapping[str, Any]) -> list[float]:
    metrics = report.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ContractError("historical report has no per-epoch metrics")
    trace: list[float] = []
    for epoch in metrics:
        if not isinstance(epoch, Mapping) or not isinstance(
            epoch.get("validation"), Mapping
        ):
            raise ContractError(
                "historical report lacks validation metrics for an epoch"
            )
        validation = epoch["validation"]
        if "value_loss" not in validation:
            raise ContractError("historical validation metrics have no value_loss")
        trace.append(float(validation["value_loss"]))
    return trace


def _validate_historical_trace(
    actual: Sequence[float], expected: Sequence[float], tolerance: float = 0.002
) -> None:
    if len(actual) != len(expected):
        raise ContractError(
            f"historical value trace has {len(actual)} epochs, expected {len(expected)}"
        )
    for index, (observed, target) in enumerate(zip(actual, expected), start=1):
        if abs(float(observed) - float(target)) > float(tolerance):
            raise ContractError(
                f"historical value trace epoch {index}: {observed:.6f} not within "
                f"{tolerance} of {target:.6f}"
            )


def _code_inventory(repo_root: Path) -> tuple[list[dict[str, Any]], str]:
    paths = list((repo_root / "src" / "catan_zero").rglob("*.py"))
    paths.extend(
        [
            repo_root / "tools" / "train_bc.py",
            repo_root / "tools" / "f69_upgrade_checkpoint_config.py",
            repo_root / "tools" / "a0_gen2b_probe.py",
            repo_root / "configs" / "guards" / "train_bc.json",
        ]
    )
    entries = []
    for path in sorted(set(paths)):
        if not path.is_file():
            raise ContractError(f"missing code input: {path}")
        entries.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return entries, _digest_json(entries)


def _python_environment(python: Path, repo_root: Path) -> dict[str, Any]:
    command = [
        str(python),
        "-c",
        (
            "import json, numpy, platform, torch; "
            "print(json.dumps({'python':platform.python_version(),"
            "'numpy':numpy.__version__,'torch':torch.__version__,"
            "'torch_cuda':torch.version.cuda},sort_keys=True))"
        ),
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"cannot inspect training Python {python}: {exc}") from exc
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"invalid environment probe output: {proc.stdout!r}"
        ) from exc
    # Preserve the venv entry point rather than resolving its symlink to the
    # system interpreter. Executing /usr/bin/python3.11 after resolving a venv
    # link loses that venv's site-packages (torch/numpy) and makes a valid B200
    # environment look broken.
    result["executable"] = _preserved_executable_path(python)
    return result


def _preserved_executable_path(python: Path) -> str:
    """Return an absolute invocation path without dereferencing venv symlinks."""
    return str(python.expanduser().absolute())


def _validate_upgrade_checkpoint(
    checkpoint: Path,
    *,
    source_sha256: str,
    bins: int,
    seed: int,
) -> None:
    try:
        import torch

        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - torch emits many load exception types
        raise ContractError(
            f"cannot inspect upgraded checkpoint {checkpoint}: {exc}"
        ) from exc
    provenance = raw.get("upgrade_provenance") if isinstance(raw, Mapping) else None
    if not isinstance(provenance, Mapping):
        raise ContractError("categorical init has no upgrade_provenance")
    expected_flags = {"value_categorical_bins": int(bins)}
    expected = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": source_sha256,
        "flags": expected_flags,
        "initialization_seed": int(seed),
        "trained_value_readouts_added": [],
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ContractError(
                f"categorical init provenance mismatch for {key}: "
                f"expected {value!r}, got {provenance.get(key)!r}"
            )
    # Current EntityGraphPolicy.save() persists configs in the durable,
    # name-keyed form; historical checkpoints contain the pickled dataclass.
    # Validate both through the repository's canonical compatibility adapter.
    from catan_zero.rl.config_serialization import config_attr_view

    config = config_attr_view(raw.get("config"))
    config_bins = (
        getattr(config, "value_categorical_bins", None) if config is not None else None
    )
    if int(config_bins or 0) != int(bins):
        raise ContractError(
            f"categorical init config has bins={config_bins}, expected {bins}"
        )
    trained = set(raw.get("trained_value_readouts") or [])
    if "categorical" in trained:
        raise ContractError(
            "config-only categorical init falsely claims trained readout"
        )


def _run_upgrade(
    repo_root: Path,
    python: Path,
    source: Path,
    output: Path,
    *,
    bins: int,
    seed: int,
    replace: bool,
) -> tuple[dict[str, Any], str]:
    if output.exists() and not replace:
        raise ContractError(
            f"categorical init already exists: {output}; pass --replace-upgrade to rebuild"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temp.exists():
        temp.unlink()
    command = [
        str(python),
        "tools/f69_upgrade_checkpoint_config.py",
        "--in-checkpoint",
        str(source),
        "--out-checkpoint",
        str(temp),
        "--flags",
        f"catbins:{bins}",
        "--seed",
        str(seed),
        "--device",
        "cpu",
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        )
        evidence = json.loads(proc.stdout)
        if evidence.get("forward_max_diff") != 0.0:
            raise ContractError(
                f"categorical init is not behavior preserving: {evidence.get('forward_max_diff')}"
            )
        os.replace(temp, output)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        stderr = getattr(exc, "stderr", "")
        raise ContractError(
            f"categorical init upgrade failed: {exc}\n{stderr}"
        ) from exc
    finally:
        if temp.exists():
            temp.unlink()
    output_sha = _sha256(output)
    _validate_upgrade_checkpoint(
        output,
        source_sha256=_sha256(source),
        bins=bins,
        seed=seed,
    )
    return evidence, output_sha


def _cli_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return ",".join(f"{key}={value[key]}" for key in sorted(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(str(item) for item in value)
    if isinstance(value, bool):
        raise TypeError("boolean CLI values require action-aware handling")
    return str(value)


def _recipe_cli(recipe: Mapping[str, Any]) -> list[str]:
    args: list[str] = []
    # The lock is serialized with sorted JSON keys.  Build argv in that same
    # canonical order so a freshly loaded lock reconstructs byte-identical arm
    # contracts instead of depending on the manifest's insertion order.
    for key in sorted(recipe):
        value = recipe[key]
        flag = f"--{key.replace('_', '-')}"
        if key in _BOOLEAN_OPTIONAL:
            args.append(flag if bool(value) else f"--no-{key.replace('_', '-')}")
        elif key in _STORE_TRUE:
            if bool(value):
                args.append(flag)
        elif key in _MAP_FIELDS or key in _LIST_FIELDS:
            rendered = _cli_value(value)
            if rendered:
                args.extend([flag, rendered])
        elif value != "":
            args.extend([flag, _cli_value(value)])
    return args


def _arm_contract(
    *,
    recipe: Mapping[str, Any],
    validation: Mapping[str, Any],
    data: str,
    init_checkpoint: str,
    checkpoint: str,
    report: str,
    objective: str,
    bins: int,
) -> dict[str, Any]:
    if objective not in {"mse", "hlgauss"}:
        raise ContractError(f"unknown A0 objective {objective!r}")
    argv = [
        "tools/train_bc.py",
        "--data",
        data,
        "--init-checkpoint",
        init_checkpoint,
        "--checkpoint",
        checkpoint,
        "--report",
        report,
        *_recipe_cli(recipe),
        "--validation-game-seed-ranges",
        str(validation["selected_game_seed_ranges_cli"]),
        "--value-head-type",
        objective,
        "--value-categorical-loss-weight",
        "0",
        "--hlgauss-scalar-aux-loss-weight",
        "0",
        "--value-hlgauss-sigma-ratio",
        "0.75",
        "--no-resume-optimizer",
        "--save-each-epoch",
        "--allow-concurrent-bc",
        "--device",
        "cuda:0",
    ]
    # This sealed A0 tool is explicitly diagnostic-only. Preserve replay of its
    # historical 0.3 recipe under train_bc's production fail-closed contract by
    # carrying the required acknowledgement in the reconstructed command. The
    # learner weights themselves remain byte-semantically unchanged.
    if float(recipe.get("loser_sample_weight", 1.0)) < 1.0:
        argv.append(
            "--acknowledge-diagnostic-outcome-conditioned-policy-distillation"
        )
    if objective == "hlgauss":
        argv.extend(["--value-categorical-bins", str(int(bins))])
    return {
        "objective": objective,
        "argv": argv,
        "argv_sha256": _digest_json(argv),
        "init_checkpoint": init_checkpoint,
        "checkpoint": checkpoint,
        "report": report,
    }


def _build_arm_contracts(
    manifest: Mapping[str, Any],
    recipe: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    def artifact_path(value: str) -> str:
        path = Path(value)
        if artifact_root is not None and not path.is_absolute():
            return str((artifact_root / path).resolve())
        return str(path)

    output_dir = artifact_path(str(manifest["output_dir"]))
    data = artifact_path(str(manifest["inputs"]["corpus_dir"]))
    source = artifact_path(str(manifest["inputs"]["source_checkpoint"]))
    upgraded = artifact_path(str(manifest["upgrade"]["checkpoint"]))
    bins = int(manifest["upgrade"]["bins"])
    scalar = _arm_contract(
        recipe=recipe,
        validation=validation,
        data=data,
        init_checkpoint=source,
        checkpoint=f"{output_dir}/scalar/checkpoint.pt",
        report=f"{output_dir}/scalar/report.json",
        objective="mse",
        bins=bins,
    )
    hl = _arm_contract(
        recipe=recipe,
        validation=validation,
        data=data,
        init_checkpoint=upgraded,
        checkpoint=f"{output_dir}/hlgauss33/checkpoint.pt",
        report=f"{output_dir}/hlgauss33/report.json",
        objective="hlgauss",
        bins=bins,
    )
    # The common contract is stronger than diffing shell text: every shared
    # science field, held-out set and exact row order has one digest used by
    # both arms. Only the init/objective/output/device fields above may differ.
    common = {
        "recipe": recipe,
        "data": data,
        "validation_game_seed_set_sha256": validation[
            "validation_game_seed_set_sha256"
        ],
        "selected_game_seed_set_sha256": validation[
            "selected_game_seed_set_sha256_before_row_cap"
        ],
        "epoch_row_order_sha256": validation["epoch_row_order_sha256"],
        "fresh_optimizer": True,
        "save_each_epoch": True,
    }
    return {
        "matched_common": common,
        "matched_common_sha256": _digest_json(common),
        "scalar": scalar,
        "hlgauss33": hl,
    }


def _manifest_relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def seal(args: argparse.Namespace) -> Path:
    repo_root = Path(args.repo_root).resolve()
    artifact_root = Path(args.artifact_root or args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_json(manifest_path)
    _validate_manifest(manifest)
    inputs = manifest["inputs"]
    source = _resolve(artifact_root, inputs["source_checkpoint"])
    historical_report_path = _resolve(artifact_root, inputs["historical_report"])
    corpus_dir = _resolve(artifact_root, inputs["corpus_dir"])
    meta_path = corpus_dir / "corpus_meta.json"
    source_sha = _require_sha(
        source, inputs["source_checkpoint_sha256"], "gen1 source checkpoint"
    )
    report_sha = _require_sha(
        historical_report_path,
        inputs["historical_report_sha256"],
        "gen2B historical report",
    )
    meta_sha = _require_sha(
        meta_path, inputs["corpus_meta_sha256"], "gen2 memmap metadata"
    )
    historical_report = _load_json(historical_report_path)
    # The historical gen2B report predates recording ``data`` in report.json.
    # Its corpus is instead anchored by the checked-in expected row count plus
    # the full-byte corpus inventory/meta digest below.  If a report *does*
    # name data, it must agree; absence is recorded by the lock rather than
    # fabricated.  init_checkpoint was recorded historically and remains
    # mandatory because it is the mechanism test's starting function.
    report_data = historical_report.get("data")
    if report_data not in (None, "") and not _report_path_matches(
        report_data, inputs["corpus_dir"]
    ):
        raise ContractError(
            f"historical report data={report_data!r} does not identify locked "
            f"{inputs['corpus_dir']!r}"
        )
    report_init = historical_report.get("init_checkpoint")
    if not _report_path_matches(report_init, inputs["source_checkpoint"]):
        raise ContractError(
            f"historical report init_checkpoint={report_init!r} does not identify "
            f"locked {inputs['source_checkpoint']!r}"
        )
    if historical_report.get("validation_game_seed_ranges") not in (None, "", []):
        raise ContractError(
            "historical gen2B used explicit validation_game_seed_ranges; the A0 "
            "manifest must encode those exact ranges instead of reconstructing the "
            "fractional split"
        )
    recipe = _resolve_recipe(manifest, historical_report)
    historical_trace = _extract_value_trace(historical_report)
    _validate_historical_trace(
        historical_trace, [float(value) for value in manifest["historical_value_trace"]]
    )
    meta = _load_json(meta_path)
    if meta.get("schema") != "memmap_corpus_v1":
        raise ContractError(f"unsupported corpus schema {meta.get('schema')!r}")
    expected_rows = int(inputs["expected_rows"])
    if int(meta.get("row_count", -1)) != expected_rows:
        raise ContractError(
            f"corpus row_count={meta.get('row_count')} != expected {expected_rows}"
        )
    corpus_inventory, corpus_tree_sha = _corpus_inventory(corpus_dir)
    seeds = _load_game_seeds(corpus_dir, meta)
    validation = _validation_and_order_contract(
        seeds,
        validation_fraction=float(recipe["validation_fraction"]),
        validation_seed=int(recipe["validation_seed"]),
        validation_max_samples=int(recipe["validation_max_samples"]),
        train_seed=int(recipe["seed"]),
        epochs=int(recipe["epochs"]),
    )
    for key, computed_key in (
        ("train_samples", "train_rows"),
        ("validation_samples", "validation_rows"),
    ):
        if key not in historical_report:
            raise ContractError(
                f"historical report has no {key}; exact split reconstruction cannot "
                "be proven"
            )
        if int(historical_report[key]) != int(validation[computed_key]):
            raise ContractError(
                f"historical split cannot be reconstructed: report {key}="
                f"{historical_report[key]} but computed {computed_key}="
                f"{validation[computed_key]}"
            )
    if historical_report.get("validation_game_seed_set_sha256") not in (None, ""):
        if (
            historical_report["validation_game_seed_set_sha256"]
            != validation["validation_game_seed_set_sha256"]
        ):
            raise ContractError("historical validation game-seed digest does not match")

    python = Path(args.python).expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    if not python.is_file():
        raise ContractError(f"training Python does not exist: {python}")
    environment = _python_environment(python, repo_root)
    code_inventory, code_tree_sha = _code_inventory(repo_root)
    output_dir = _resolve(artifact_root, str(manifest["output_dir"]))
    lock_path = Path(args.lock).resolve() if args.lock else output_dir / "a0.lock.json"
    if lock_path.exists() and not args.force:
        raise ContractError(
            f"lock already exists: {lock_path}; pass --force to replace"
        )
    upgrade_path = _resolve(artifact_root, str(manifest["upgrade"]["checkpoint"]))
    upgrade_evidence, upgrade_sha = _run_upgrade(
        repo_root,
        python,
        source,
        upgrade_path,
        bins=int(manifest["upgrade"]["bins"]),
        seed=int(manifest["upgrade"]["seed"]),
        replace=bool(args.replace_upgrade),
    )
    arm_contracts = _build_arm_contracts(
        manifest, recipe, validation, artifact_root=artifact_root
    )
    seed_contract = {
        "training_seed": int(recipe["seed"]),
        "validation_seed": int(recipe["validation_seed"]),
        "categorical_init_seed": int(manifest["upgrade"]["seed"]),
        "selected_game_seed_set_sha256": validation[
            "selected_game_seed_set_sha256_before_row_cap"
        ],
        "validation_game_seed_set_sha256": validation[
            "validation_game_seed_set_sha256"
        ],
        "epoch_row_order_sha256": validation["epoch_row_order_sha256"],
    }
    input_contract = {
        "source_checkpoint_sha256": source_sha,
        "historical_report_sha256": report_sha,
        "corpus_meta_sha256": meta_sha,
        "corpus_tree_sha256": corpus_tree_sha,
        "categorical_init_sha256": upgrade_sha,
        "code_tree_sha256": code_tree_sha,
        "environment_sha256": _digest_json(environment),
    }
    lock = {
        "schema_version": LOCK_SCHEMA,
        "experiment_id": manifest["experiment_id"],
        "sealed_at_unix": int(time.time()),
        "repo_root_at_seal": str(repo_root),
        "artifact_root_at_seal": str(artifact_root),
        "manifest_path": _manifest_relative_path(manifest_path, repo_root),
        "manifest_sha256": _sha256(manifest_path),
        # Keep the venv entry point itself.  _manifest_relative_path resolves
        # symlinks so repository inputs have canonical paths, but doing that
        # to a venv's ``bin/python`` records the system interpreter and drops
        # the venv site-packages when the lock is verified or executed.
        "python": _preserved_executable_path(python),
        "environment": environment,
        "environment_sha256": _digest_json(environment),
        "inputs": dict(inputs),
        "input_contract": input_contract,
        "input_contract_sha256": _digest_json(input_contract),
        "corpus_inventory": corpus_inventory,
        "corpus_tree_sha256": corpus_tree_sha,
        "code_inventory": code_inventory,
        "code_tree_sha256": code_tree_sha,
        "historical_value_trace": historical_trace,
        "resolved_recipe": recipe,
        "recipe_sha256": _digest_json(recipe),
        "validation": validation,
        "seed_contract": seed_contract,
        "seed_contract_sha256": _digest_json(seed_contract),
        "upgrade": {
            **dict(manifest["upgrade"]),
            "checkpoint_sha256": upgrade_sha,
            "evidence": upgrade_evidence,
        },
        "physical_gpus": dict(manifest["physical_gpus"]),
        "arm_contracts": arm_contracts,
    }
    _write_json_atomic(lock_path, lock)
    _write_json_atomic(output_dir / "validation.lock.json", validation)
    print(
        json.dumps(
            {
                "lock": str(lock_path),
                "input_contract_sha256": lock["input_contract_sha256"],
                "recipe_sha256": lock["recipe_sha256"],
                "seed_contract_sha256": lock["seed_contract_sha256"],
                "matched_common_sha256": arm_contracts["matched_common_sha256"],
                "commands": {
                    name: shlex.join([str(python), *arm_contracts[name]["argv"]])
                    for name in ("scalar", "hlgauss33")
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return lock_path


def _load_and_verify_lock(lock_path: Path, repo_root: Path) -> dict[str, Any]:
    lock = _load_json(lock_path)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ContractError(
            f"unsupported A0 lock schema {lock.get('schema_version')!r}"
        )
    manifest_path = _resolve(repo_root, str(lock["manifest_path"]))
    _require_sha(manifest_path, str(lock["manifest_sha256"]), "A0 manifest")
    manifest = _load_json(manifest_path)
    _validate_manifest(manifest)
    artifact_root = Path(lock.get("artifact_root_at_seal") or repo_root).resolve()
    corpus_dir = _resolve(artifact_root, str(lock["inputs"]["corpus_dir"]))
    corpus_inventory, corpus_digest = _corpus_inventory(corpus_dir)
    if (
        corpus_inventory != lock["corpus_inventory"]
        or corpus_digest != lock["corpus_tree_sha256"]
    ):
        raise ContractError("corpus tree inventory/digest drift")
    code_inventory, code_digest = _code_inventory(repo_root)
    if (
        code_inventory != lock["code_inventory"]
        or code_digest != lock["code_tree_sha256"]
    ):
        raise ContractError("code tree inventory/digest drift")
    inputs = lock["inputs"]
    source = _resolve(artifact_root, str(inputs["source_checkpoint"]))
    report = _resolve(artifact_root, str(inputs["historical_report"]))
    meta = corpus_dir / "corpus_meta.json"
    _require_sha(source, str(inputs["source_checkpoint_sha256"]), "gen1 source")
    _require_sha(report, str(inputs["historical_report_sha256"]), "gen2B report")
    _require_sha(meta, str(inputs["corpus_meta_sha256"]), "gen2 corpus metadata")
    upgrade_path = _resolve(artifact_root, str(lock["upgrade"]["checkpoint"]))
    _require_sha(
        upgrade_path,
        str(lock["upgrade"]["checkpoint_sha256"]),
        "categorical init",
    )
    _validate_upgrade_checkpoint(
        upgrade_path,
        source_sha256=str(inputs["source_checkpoint_sha256"]),
        bins=int(lock["upgrade"]["bins"]),
        seed=int(lock["upgrade"]["seed"]),
    )
    python = _resolve(repo_root, str(lock["python"]))
    environment = _python_environment(python, repo_root)
    if environment != lock["environment"]:
        raise ContractError(
            f"training environment drift: expected {lock['environment']}, got {environment}"
        )
    for name, value in (
        ("recipe", lock["resolved_recipe"]),
        ("seed_contract", lock["seed_contract"]),
        ("input_contract", lock["input_contract"]),
    ):
        expected = lock[f"{name}_sha256"]
        if _digest_json(value) != expected:
            raise ContractError(f"{name} self-digest mismatch inside lock")
    rebuilt_arms = _build_arm_contracts(
        manifest,
        lock["resolved_recipe"],
        lock["validation"],
        artifact_root=artifact_root,
    )
    if rebuilt_arms != lock["arm_contracts"]:
        raise ContractError("arm command/common contract drift inside lock")
    if dict(manifest["physical_gpus"]) != lock["physical_gpus"]:
        raise ContractError("physical GPU assignment drift inside lock")
    return lock


def _gpu_is_busy(gpu: str) -> bool:
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"cannot verify GPU {gpu} availability: {exc}") from exc
    return bool(proc.stdout.strip())


def _ensure_fd_limit(minimum: int = 65536) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < minimum:
        if hard < minimum:
            raise ContractError(f"RLIMIT_NOFILE hard limit {hard} < required {minimum}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (minimum, hard))


def _decision_validation_metrics(metric: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve the population used for a science-facing validation decision.

    ``train_bc`` retains ``validation`` as a row-concatenated compatibility
    statistic.  Authenticated composites also emit
    ``validation_objective_matched`` for the actual component -> game -> row
    training measure.  A decision must never silently prefer the compatibility
    field when the exact population is present.
    """

    if "validation_objective_matched" in metric:
        matched = metric.get("validation_objective_matched")
        metrics = matched.get("metrics") if isinstance(matched, Mapping) else None
        if (
            not isinstance(matched, Mapping)
            or matched.get("schema_version") != "composite-validation-measure-v2"
            or matched.get("objective_matched") is not True
            or not isinstance(metrics, Mapping)
        ):
            raise ContractError(
                "completed report has malformed objective-matched validation"
            )
        return metrics
    validation = metric.get("validation")
    if not isinstance(validation, Mapping):
        raise ContractError("completed report lacks validation metrics")
    return validation


def _report_trace(report: Mapping[str, Any], field: str) -> list[float]:
    result: list[float] = []
    for metric in report.get("metrics", []):
        if not isinstance(metric, Mapping):
            raise ContractError("completed report has a malformed epoch metric")
        validation = _decision_validation_metrics(metric)
        if field not in validation:
            raise ContractError(f"completed report lacks decision validation.{field}")
        result.append(float(validation[field]))
    if len(result) != 3:
        raise ContractError(f"completed report has {len(result)} epochs, expected 3")
    return result


def _postflight(lock: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    for name in ("scalar", "hlgauss33"):
        contract = lock["arm_contracts"][name]
        report_path = _resolve(repo_root, str(contract["report"]))
        checkpoint_path = _resolve(repo_root, str(contract["checkpoint"]))
        report = _load_json(report_path)
        for key, expected in lock.get("resolved_recipe", {}).items():
            if key in report and not _same_value(report[key], expected):
                raise ContractError(
                    f"{name}: completed report recipe drift for {key}: "
                    f"expected {expected!r}, got {report[key]!r}"
                )
        if report.get("resume_optimizer") is not False:
            raise ContractError(
                f"{name}: report does not attest resume_optimizer=false"
            )
        if report.get("optimizer_restored") is not False:
            raise ContractError(f"{name}: optimizer state was restored")
        if (
            report.get("validation_game_seed_set_sha256")
            != lock["validation"]["validation_game_seed_set_sha256"]
        ):
            raise ContractError(f"{name}: held-out game-seed set drift")
        if int(report.get("validation_game_seed_count", -1)) != int(
            lock["validation"]["validation_game_seed_count_after_row_cap"]
        ):
            raise ContractError(f"{name}: held-out game-seed count drift")
        if int(report.get("steps_completed", 0)) <= 0:
            raise ContractError(f"{name}: no optimizer steps completed")
        expected_init_sha = (
            str(lock["inputs"]["source_checkpoint_sha256"])
            if name == "scalar"
            else str(lock["upgrade"]["checkpoint_sha256"])
        )
        if str(report.get("init_checkpoint_sha256", "")).removeprefix(
            "sha256:"
        ) != expected_init_sha.removeprefix("sha256:"):
            raise ContractError(f"{name}: init-checkpoint SHA-256 drift")
        expected_readout = "scalar" if name == "scalar" else "categorical"
        value_training = report.get("value_training")
        if not isinstance(value_training, Mapping):
            raise ContractError(f"{name}: report has no value_training provenance")
        if value_training.get("schema_version") != "value-training-v1":
            raise ContractError(f"{name}: unsupported value_training provenance")
        if value_training.get("primary_readout") != expected_readout:
            raise ContractError(f"{name}: wrong primary value readout provenance")
        trained = set(value_training.get("trained_value_readouts") or [])
        if expected_readout not in trained:
            raise ContractError(f"{name}: primary value readout was not trained")
        if name == "hlgauss33" and "scalar" in trained:
            raise ContractError("hlgauss33: scalar auxiliary unexpectedly trained")
        if not checkpoint_path.is_file():
            raise ContractError(f"{name}: missing final checkpoint {checkpoint_path}")
        artifact_hashes[name] = {
            "report_sha256": f"sha256:{_sha256(report_path)}",
            "checkpoint_sha256": f"sha256:{_sha256(checkpoint_path)}",
        }
        reports[name] = report
    scalar_trace = _report_trace(reports["scalar"], "primary_value_loss")
    expected = [float(value) for value in lock["historical_value_trace"]]
    scalar_reproduces = (
        scalar_trace[1] > scalar_trace[0]
        and scalar_trace[2] >= scalar_trace[1]
        and all(
            abs(value - target) <= 0.03 for value, target in zip(scalar_trace, expected)
        )
    )
    hl_trace = _report_trace(reports["hlgauss33"], "primary_value_loss")
    hl_training_stable = (
        hl_trace[2] <= hl_trace[0]
        and hl_trace[1] <= hl_trace[0] * 1.01
        and hl_trace[2] <= hl_trace[1] * 1.01
    )
    verdict = {
        "schema_version": RESULT_SCHEMA,
        "scalar_primary_validation_trace": scalar_trace,
        "historical_scalar_validation_trace": expected,
        "scalar_reproduces_historical_failure": scalar_reproduces,
        "hl_primary_validation_trace": hl_trace,
        "hl_training_stable": hl_training_stable,
        "a0_interpretable": scalar_reproduces,
        "a0_training_loss_gate_pass": scalar_reproduces and hl_training_stable,
        "artifacts": artifact_hashes,
        "note": (
            "Calibration, phase/width slices and H2H remain separate binding gates; "
            "this result covers only the matched training-loss mechanism."
        ),
    }
    return verdict


def run(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    lock_path = Path(args.lock).resolve()
    lock = _load_and_verify_lock(lock_path, repo_root)
    gpus = lock["physical_gpus"]
    if str(gpus["scalar"]) == str(gpus["hlgauss"]):
        raise ContractError("lock assigns both arms to the same physical GPU")
    _ensure_fd_limit()
    if not args.allow_busy_gpus:
        for name in ("scalar", "hlgauss"):
            gpu = str(gpus[name])
            if _gpu_is_busy(gpu):
                raise ContractError(
                    f"GPU {gpu} ({name}) already has a compute process; refusing to stack A0"
                )
    python = _resolve(repo_root, str(lock["python"]))
    arm_specs = (("scalar", "scalar"), ("hlgauss33", "hlgauss"))
    for name, _gpu_key in arm_specs:
        contract = lock["arm_contracts"][name]
        for output_key in ("checkpoint", "report"):
            output = _resolve(repo_root, str(contract[output_key]))
            if output.exists():
                raise ContractError(
                    f"{name} output already exists: {output}; A0 never resumes/overwrites"
                )
        log = _resolve(repo_root, str(contract["report"])).with_name("train.log")
        if log.exists():
            raise ContractError(f"{name} log already exists: {log}")

    processes: dict[str, tuple[subprocess.Popen[str], Any]] = {}
    try:
        for name, gpu_key in arm_specs:
            contract = lock["arm_contracts"][name]
            for output_key in ("checkpoint", "report"):
                output = _resolve(repo_root, str(contract[output_key]))
                output.parent.mkdir(parents=True, exist_ok=True)
            log = _resolve(repo_root, str(contract["report"])).with_name("train.log")
            handle = open(log, "w", encoding="utf-8")
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpus[gpu_key]),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONPATH": str(repo_root / "src"),
                "PYTHONHASHSEED": "0",
            }
            try:
                process = subprocess.Popen(
                    [str(python), *contract["argv"]],
                    cwd=repo_root,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
            except Exception:
                handle.close()
                raise
            processes[name] = (process, handle)
        failed: tuple[str, int] | None = None
        while processes:
            for name, (process, handle) in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del processes[name]
                if code != 0 and failed is None:
                    failed = (name, code)
                    for other, (other_process, _) in processes.items():
                        print(
                            f"terminating {other} because {name} failed",
                            file=sys.stderr,
                        )
                        other_process.terminate()
            if failed is not None:
                break
            time.sleep(1.0)
        if failed is not None:
            raise ContractError(f"A0 arm {failed[0]} exited {failed[1]}")
    finally:
        for process, handle in processes.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
            handle.close()
    verdict = _postflight(lock, repo_root)
    result_path = lock_path.with_name("a0.result.json")
    _write_json_atomic(result_path, verdict)
    print(json.dumps({"result": str(result_path), **verdict}, indent=2, sort_keys=True))
    if not verdict["a0_training_loss_gate_pass"]:
        raise ContractError(
            "A0 training-loss gate did not pass; do not interpret/promote the HL arm"
        )


def verify(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    lock = _load_and_verify_lock(Path(args.lock).resolve(), repo_root)
    payload: dict[str, Any] = {
        "verified": True,
        "input_contract_sha256": lock["input_contract_sha256"],
        "recipe_sha256": lock["recipe_sha256"],
        "seed_contract_sha256": lock["seed_contract_sha256"],
        "matched_common_sha256": lock["arm_contracts"]["matched_common_sha256"],
    }
    if args.results:
        verdict = _postflight(lock, repo_root)
        payload["result"] = verdict
        if not verdict["a0_training_loss_gate_pass"]:
            raise ContractError(
                "completed A0 does not pass the training-loss mechanism gate"
            )
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal_parser = sub.add_parser("seal", help="CPU-only preflight, upgrade, and lock")
    seal_parser.add_argument("--manifest", required=True)
    seal_parser.add_argument("--repo-root", required=True)
    seal_parser.add_argument(
        "--artifact-root",
        default="",
        help="Root containing runs/ artifacts; defaults to --repo-root.",
    )
    seal_parser.add_argument("--python", required=True)
    seal_parser.add_argument("--lock", default="")
    seal_parser.add_argument("--force", action="store_true")
    seal_parser.add_argument("--replace-upgrade", action="store_true")
    seal_parser.set_defaults(func=seal)
    run_parser = sub.add_parser("run", help="verify lock and launch both one-GPU arms")
    run_parser.add_argument("--lock", required=True)
    run_parser.add_argument("--repo-root", required=True)
    run_parser.add_argument(
        "--allow-busy-gpus",
        action="store_true",
        help="Explicit emergency override; normally any existing compute PID blocks launch.",
    )
    run_parser.set_defaults(func=run)
    verify_parser = sub.add_parser(
        "verify", help="rehash the lock; optionally verify reports"
    )
    verify_parser.add_argument("--lock", required=True)
    verify_parser.add_argument("--repo-root", required=True)
    verify_parser.add_argument("--results", action="store_true")
    verify_parser.set_defaults(func=verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
