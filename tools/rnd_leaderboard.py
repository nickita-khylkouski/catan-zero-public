#!/usr/bin/env python3
"""Fail-closed R&D experiment ledger and leaderboard.

This tool deliberately does not run a search or construct a model.  It is the
measurement boundary between experimental runners and a comparison report.  A
runner emits one JSON result bundle per arm; this tool validates provenance,
paired seed/seat coverage, and measured work before it will rank anything.

The format is documented in ``docs/RND_LEADERBOARD_HARNESS.md`` and represented
by the templates under ``configs/rnd``.  The implementation is stdlib-only so
the same validation can run on a laptop, a generation worker, or in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "catan-zero-rnd-leaderboard/v1"
COUNTER_FIELDS = (
    "nominal_visits",
    "scheduled_visits",
    "logical_leaves",
    "orientation_rows",
    "evaluator_calls",
    "wall_time_sec",
)
INTEGER_COUNTER_FIELDS = frozenset(COUNTER_FIELDS[:-1])
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PUBLIC_INFORMATION_REGIMES = frozenset(
    {"public_conservation_pimc", "public_observation_policy"}
)
ALL_INFORMATION_REGIMES = PUBLIC_INFORMATION_REGIMES | {
    "authoritative_hidden_state"
}
SEAT_COLORS = ("RED", "BLUE")


class ValidationError(ValueError):
    """The experiment bundle cannot support a valid comparison."""


@dataclass(frozen=True)
class ArmSummary:
    arm_id: str
    comparison_role: str
    architecture_id: str
    parameter_count: int
    search_id: str
    architecture_config_sha256: str
    search_config_sha256: str
    checkpoint_sha256: str
    git_commit: str
    code_dirty: bool
    patch_sha256: str | None
    games: int
    pairs: int
    wins: int
    draws: int
    losses: int
    mean_score: float
    counters: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "arm_id": self.arm_id,
            "comparison_role": self.comparison_role,
            "architecture_id": self.architecture_id,
            "parameter_count": self.parameter_count,
            "search_id": self.search_id,
            "architecture_config_sha256": self.architecture_config_sha256,
            "search_config_sha256": self.search_config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "code": {
                "git_commit": self.git_commit,
                "dirty": self.code_dirty,
                "patch_sha256": self.patch_sha256,
            },
            "games": self.games,
            "pairs": self.pairs,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "mean_score": self.mean_score,
            "work_totals": dict(self.counters),
            "work_per_game": {
                key: value / self.games for key, value in self.counters.items()
            },
        }
        return row


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{context} must be a JSON object")
    return value


def _require_sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{context} must be a JSON array")
    return value


def _require_string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _require_sha256(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _require_string(mapping, key, context).lower()
    if not SHA256_RE.fullmatch(value):
        raise ValidationError(f"{context}.{key} must be a lowercase SHA-256 digest")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, context: str, *, minimum: int = 0) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{context}.{key} must be an integer >= {minimum}")
    return value


def _require_number(
    mapping: Mapping[str, Any], key: str, context: str, *, minimum: float = 0.0
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{context}.{key} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValidationError(f"{context}.{key} must be a finite number >= {minimum}")
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read JSON {source}: {exc}") from exc
    return dict(_require_mapping(value, str(source)))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_registry(campaign: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if campaign.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(f"campaign.schema_version must be {SCHEMA_VERSION!r}")
    _require_string(campaign, "campaign_id", "campaign")
    information_requirement = _require_string(
        campaign, "required_information_regime", "campaign"
    )
    if information_requirement not in ("public_only", "allow_authoritative"):
        raise ValidationError(
            "campaign.required_information_regime must be public_only or allow_authoritative"
        )
    if not isinstance(campaign.get("require_same_training_manifest"), bool):
        raise ValidationError(
            "campaign.require_same_training_manifest must be boolean"
        )
    frozen_fields = campaign.get("frozen_candidate_fields", [])
    if not isinstance(frozen_fields, list) or any(
        field
        not in {
            "architecture_id",
            "architecture_config_sha256",
            "search_id",
            "search_config_sha256",
            "checkpoint_sha256",
        }
        for field in frozen_fields
    ):
        raise ValidationError(
            "campaign.frozen_candidate_fields must be an array containing only "
            "architecture_id/architecture_config_sha256/search_id/"
            "search_config_sha256/checkpoint_sha256"
        )
    if len(set(frozen_fields)) != len(frozen_fields):
        raise ValidationError("campaign.frozen_candidate_fields must not contain duplicates")
    registry = _require_sequence(campaign.get("arms"), "campaign.arms")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw_arm in enumerate(registry):
        context = f"campaign.arms[{index}]"
        arm = _require_mapping(raw_arm, context)
        arm_id = _require_string(arm, "arm_id", context)
        if arm_id in result:
            raise ValidationError(f"duplicate campaign arm_id {arm_id!r}")
        source_status = _require_string(arm, "source_status", context)
        adapter_status = _require_string(arm, "measurement_adapter_status", context)
        runnable = arm.get("runnable")
        if not isinstance(runnable, bool):
            raise ValidationError(f"{context}.runnable must be boolean")
        if runnable and (source_status != "implemented" or adapter_status != "implemented"):
            raise ValidationError(
                f"{context} cannot be runnable unless source and measurement adapter are implemented"
            )
        _require_string(arm, "architecture_id", context)
        _require_string(arm, "search_id", context)
        comparison_role = str(arm.get("comparison_role", "ranked"))
        if comparison_role not in ("ranked", "control"):
            raise ValidationError(f"{context}.comparison_role must be ranked or control")
        if "expected_parameter_count" in arm:
            _require_int(arm, "expected_parameter_count", context, minimum=1)
        result[arm_id] = arm
    if not result:
        raise ValidationError("campaign.arms must not be empty")

    required_raw = _require_sequence(campaign.get("required_arm_ids"), "campaign.required_arm_ids")
    required = [str(value) for value in required_raw]
    if not required or any(not value.strip() for value in required):
        raise ValidationError("campaign.required_arm_ids must contain non-empty strings")
    if len(set(required)) != len(required):
        raise ValidationError("campaign.required_arm_ids must not contain duplicates")
    unknown_required = sorted(set(required) - set(result))
    if unknown_required:
        raise ValidationError(
            f"campaign.required_arm_ids references unknown arms: {', '.join(unknown_required)}"
        )

    contracts = _require_mapping(campaign.get("budget_contracts"), "campaign.budget_contracts")
    for regime, mandatory_metric in (
        ("equal_work", "logical_leaves"),
        ("equal_time", "wall_time_sec"),
    ):
        contract_context = f"campaign.budget_contracts.{regime}"
        contract = _require_mapping(contracts.get(regime), contract_context)
        raw_metrics = _require_sequence(contract.get("match_metrics"), f"{contract_context}.match_metrics")
        metrics = []
        for index, metric in enumerate(raw_metrics):
            if not isinstance(metric, str) or not metric:
                raise ValidationError(f"{contract_context}.match_metrics[{index}] must be a string")
            metrics.append(metric)
        if not metrics:
            raise ValidationError(f"{contract_context}.match_metrics must not be empty")
        unknown_metrics = sorted(set(metrics) - set(COUNTER_FIELDS))
        if unknown_metrics:
            raise ValidationError(
                f"{contract_context} references unknown counters: {', '.join(unknown_metrics)}"
            )
        if mandatory_metric not in metrics:
            raise ValidationError(f"{contract_context} must match on {mandatory_metric}")
        _require_number(contract, "absolute_tolerance", contract_context)
        _require_number(contract, "relative_tolerance", contract_context)
    return result


def _validate_counter_bundle(raw: Any, context: str) -> dict[str, float]:
    counters = _require_mapping(raw, context)
    missing = [key for key in COUNTER_FIELDS if key not in counters]
    if missing:
        raise ValidationError(
            f"{context} is missing required measured counters: {', '.join(missing)}"
        )
    unknown = sorted(set(counters) - set(COUNTER_FIELDS))
    if unknown:
        raise ValidationError(f"{context} has unknown counters: {', '.join(unknown)}")
    validated: dict[str, float] = {}
    for key in COUNTER_FIELDS:
        if key in INTEGER_COUNTER_FIELDS:
            validated[key] = float(_require_int(counters, key, context))
        else:
            validated[key] = _require_number(counters, key, context)
    return validated


def _validate_provenance(
    run: Mapping[str, Any],
    context: str,
    registry_arm: Mapping[str, Any],
    *,
    verify_local_checkpoints: bool,
) -> tuple[str, int, str, str, str, str, str, bool, str | None]:
    architecture = _require_mapping(run.get("architecture"), f"{context}.architecture")
    architecture_id = _require_string(architecture, "architecture_id", f"{context}.architecture")
    if architecture_id != registry_arm["architecture_id"]:
        raise ValidationError(
            f"{context}.architecture_id {architecture_id!r} does not match campaign registry"
        )
    parameter_count = _require_int(
        architecture, "parameter_count", f"{context}.architecture", minimum=1
    )
    expected_parameter_count = registry_arm.get("expected_parameter_count")
    if expected_parameter_count is not None and parameter_count != int(expected_parameter_count):
        raise ValidationError(
            f"{context}.architecture.parameter_count={parameter_count} does not match "
            f"registered expected_parameter_count={expected_parameter_count}"
        )
    architecture_config_sha = _require_sha256(
        architecture, "config_sha256", f"{context}.architecture"
    )

    search = _require_mapping(run.get("search"), f"{context}.search")
    search_id = _require_string(search, "search_id", f"{context}.search")
    if search_id != registry_arm["search_id"]:
        raise ValidationError(f"{context}.search_id {search_id!r} does not match campaign registry")
    search_config_sha = _require_sha256(search, "config_sha256", f"{context}.search")

    checkpoint = _require_mapping(run.get("checkpoint"), f"{context}.checkpoint")
    checkpoint_path = _require_string(checkpoint, "path", f"{context}.checkpoint")
    checkpoint_sha256 = _require_sha256(checkpoint, "sha256", f"{context}.checkpoint")
    if verify_local_checkpoints:
        local_path = Path(checkpoint_path)
        if not local_path.is_file():
            raise ValidationError(f"{context}.checkpoint.path does not exist: {local_path}")
        measured = sha256_file(local_path)
        if measured != checkpoint_sha256:
            raise ValidationError(
                f"{context}.checkpoint SHA mismatch: recorded={checkpoint_sha256}, measured={measured}"
            )

    code = _require_mapping(run.get("code"), f"{context}.code")
    git_commit = _require_string(code, "git_commit", f"{context}.code").lower()
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValidationError(f"{context}.code.git_commit must be a 40-character commit hash")
    dirty = code.get("dirty")
    if not isinstance(dirty, bool):
        raise ValidationError(f"{context}.code.dirty must be boolean")
    patch_sha: str | None = None
    if dirty:
        patch_sha = _require_sha256(code, "patch_sha256", f"{context}.code")
        if patch_sha == "0" * 64:
            raise ValidationError(f"{context}.code.patch_sha256 cannot be the zero digest when dirty")

    return (
        architecture_id,
        parameter_count,
        search_id,
        architecture_config_sha,
        search_config_sha,
        checkpoint_sha256,
        git_commit,
        dirty,
        patch_sha,
    )


def _validate_reference(
    run: Mapping[str, Any],
    context: str,
    *,
    verify_local_checkpoints: bool,
) -> dict[str, Any]:
    reference = _require_mapping(run.get("reference"), f"{context}.reference")
    reference_id = _require_string(reference, "reference_id", f"{context}.reference")
    architecture = _require_mapping(
        reference.get("architecture"), f"{context}.reference.architecture"
    )
    architecture_id = _require_string(
        architecture, "architecture_id", f"{context}.reference.architecture"
    )
    parameter_count = _require_int(
        architecture, "parameter_count", f"{context}.reference.architecture", minimum=1
    )
    architecture_config_sha = _require_sha256(
        architecture, "config_sha256", f"{context}.reference.architecture"
    )
    search = _require_mapping(reference.get("search"), f"{context}.reference.search")
    search_id = _require_string(search, "search_id", f"{context}.reference.search")
    search_config_sha = _require_sha256(
        search, "config_sha256", f"{context}.reference.search"
    )
    checkpoint = _require_mapping(
        reference.get("checkpoint"), f"{context}.reference.checkpoint"
    )
    checkpoint_path = _require_string(
        checkpoint, "path", f"{context}.reference.checkpoint"
    )
    checkpoint_sha = _require_sha256(
        checkpoint, "sha256", f"{context}.reference.checkpoint"
    )
    if verify_local_checkpoints:
        local_path = Path(checkpoint_path)
        if not local_path.is_file():
            raise ValidationError(f"{context}.reference.checkpoint.path does not exist: {local_path}")
        measured = sha256_file(local_path)
        if measured != checkpoint_sha:
            raise ValidationError(
                f"{context}.reference checkpoint SHA mismatch: "
                f"recorded={checkpoint_sha}, measured={measured}"
            )
    return {
        "reference_id": reference_id,
        "architecture": {
            "architecture_id": architecture_id,
            "parameter_count": parameter_count,
            "config_sha256": architecture_config_sha,
        },
        "search": {"search_id": search_id, "config_sha256": search_config_sha},
        "checkpoint": {"path": checkpoint_path, "sha256": checkpoint_sha},
    }


def _validate_pairing(
    games: Sequence[Mapping[str, Any]],
    context: str,
    *,
    information_requirement: str,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    game_ids: set[str] = set()
    for index, game in enumerate(games):
        game_context = f"{context}.games[{index}]"
        game_id = _require_string(game, "game_id", game_context)
        if game_id in game_ids:
            raise ValidationError(f"{context} has duplicate game_id {game_id!r}")
        game_ids.add(game_id)
        pair_id = _require_string(game, "pair_id", game_context)
        _require_int(game, "seed", game_context)
        if _require_string(game, "track", game_context) != "2p_no_trade":
            raise ValidationError(f"{game_context}.track must be '2p_no_trade'")
        candidate_regime = _require_string(game, "information_regime", game_context)
        reference_regime = _require_string(
            game, "reference_information_regime", game_context
        )
        for role, regime in (
            ("candidate", candidate_regime),
            ("reference", reference_regime),
        ):
            if regime not in ALL_INFORMATION_REGIMES:
                raise ValidationError(
                    f"{game_context} {role} has unknown information regime {regime!r}"
                )
            if information_requirement == "public_only" and regime not in PUBLIC_INFORMATION_REGIMES:
                raise ValidationError(
                    f"{game_context} {role} regime {regime!r} violates public_only campaign"
                )
        seats = _require_mapping(game.get("seat_assignment"), f"{game_context}.seat_assignment")
        if set(seats) != {"candidate", "reference"}:
            raise ValidationError(
                f"{game_context}.seat_assignment must contain exactly candidate and reference"
            )
        candidate_seat = _require_int(seats, "candidate", f"{game_context}.seat_assignment")
        reference_seat = _require_int(seats, "reference", f"{game_context}.seat_assignment")
        if {candidate_seat, reference_seat} != {0, 1}:
            raise ValidationError(
                f"{game_context} candidate and reference seats must be exactly 0 and 1"
            )
        completed = game.get("completed")
        if completed is not True:
            raise ValidationError(f"{game_context}.completed must be true; truncated games are not ranked")
        winner = _require_string(game, "winner", game_context)
        if winner not in SEAT_COLORS:
            raise ValidationError(
                f"{game_context}.winner must be one of {SEAT_COLORS!r}"
            )
        score = _require_number(game, "candidate_score", game_context)
        expected_score = 1.0 if winner == SEAT_COLORS[candidate_seat] else 0.0
        if score != expected_score:
            raise ValidationError(
                f"{game_context}.candidate_score={score:g} contradicts winner={winner!r} "
                f"and candidate seat={candidate_seat}; expected {expected_score:g}"
            )
        _validate_counter_bundle(game.get("counters"), f"{game_context}.counters")
        grouped[pair_id].append(game)

    if not grouped:
        raise ValidationError(f"{context}.games must not be empty")
    for pair_id, pair_games in grouped.items():
        pair_context = f"{context}.pair[{pair_id}]"
        if len(pair_games) != 2:
            raise ValidationError(f"{pair_context} must contain exactly two seat-swapped games")
        first, second = pair_games
        if first["seed"] != second["seed"]:
            raise ValidationError(f"{pair_context} games must use the same seed")
        first_seats = first["seat_assignment"]
        second_seats = second["seat_assignment"]
        if not (
            first_seats["candidate"] == second_seats["reference"]
            and first_seats["reference"] == second_seats["candidate"]
        ):
            raise ValidationError(f"{pair_context} games are not exact candidate/reference seat swaps")
    return dict(grouped)


def _validate_seed_manifest(bundle: Mapping[str, Any], context: str) -> dict[str, Any]:
    manifest = _require_mapping(bundle.get("seed_manifest"), f"{context}.seed_manifest")
    path = _require_string(manifest, "path", f"{context}.seed_manifest")
    digest = _require_sha256(manifest, "sha256", f"{context}.seed_manifest")
    schema_version = _require_string(
        manifest, "schema_version", f"{context}.seed_manifest"
    )
    if schema_version != "catan-zero-rnd-paired-seeds/v1":
        raise ValidationError(
            f"{context}.seed_manifest.schema_version must be "
            "'catan-zero-rnd-paired-seeds/v1'"
        )
    track = _require_string(manifest, "track", f"{context}.seed_manifest")
    if track != "2p_no_trade":
        raise ValidationError(f"{context}.seed_manifest.track must be '2p_no_trade'")
    seed_count = _require_int(
        manifest, "seed_count", f"{context}.seed_manifest", minimum=1
    )
    source = Path(path)
    if not source.is_file():
        raise ValidationError(
            f"{context}.seed_manifest.path does not exist: {source}"
        )
    measured_digest = sha256_file(source)
    if measured_digest != digest:
        raise ValidationError(
            f"{context}.seed_manifest SHA mismatch: "
            f"recorded={digest}, measured={measured_digest}"
        )
    payload = _load_json(source)
    if payload.get("schema_version") != schema_version:
        raise ValidationError(
            f"{context}.seed_manifest file schema_version does not match bundle metadata"
        )
    if payload.get("track") != track:
        raise ValidationError(
            f"{context}.seed_manifest file track does not match bundle metadata"
        )
    raw_seeds = _require_sequence(
        payload.get("seeds"), f"{context}.seed_manifest file seeds"
    )
    seeds: list[int] = []
    for index, seed in enumerate(raw_seeds):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValidationError(
                f"{context}.seed_manifest file seeds[{index}] must be an integer >= 0"
            )
        seeds.append(seed)
    if len(seeds) != seed_count:
        raise ValidationError(
            f"{context}.seed_manifest seed_count={seed_count} does not match "
            f"the file's {len(seeds)} seeds"
        )
    if len(set(seeds)) != len(seeds):
        raise ValidationError(f"{context}.seed_manifest file seeds must be unique")
    return {
        "path": path,
        "sha256": digest,
        "schema_version": schema_version,
        "track": track,
        "seed_count": seed_count,
        "seeds": tuple(seeds),
    }


def _validate_training_manifest(bundle: Mapping[str, Any], context: str) -> dict[str, Any]:
    manifest = _require_mapping(
        bundle.get("training_manifest"), f"{context}.training_manifest"
    )
    path = _require_string(manifest, "path", f"{context}.training_manifest")
    digest = _require_sha256(manifest, "sha256", f"{context}.training_manifest")
    schema_version = _require_string(
        manifest, "schema_version", f"{context}.training_manifest"
    )
    if schema_version != "catan-zero-rnd-training-manifest/v1":
        raise ValidationError(
            f"{context}.training_manifest.schema_version must be "
            "'catan-zero-rnd-training-manifest/v1'"
        )
    source = Path(path)
    if not source.is_file():
        raise ValidationError(
            f"{context}.training_manifest.path does not exist: {source}"
        )
    measured_digest = sha256_file(source)
    if measured_digest != digest:
        raise ValidationError(
            f"{context}.training_manifest SHA mismatch: "
            f"recorded={digest}, measured={measured_digest}"
        )
    payload = _load_json(source)
    if payload.get("schema_version") != schema_version:
        raise ValidationError(
            f"{context}.training_manifest file schema_version does not match bundle metadata"
        )
    return {"path": path, "sha256": digest, "schema_version": schema_version}


def _validate_native_engine(bundle: Mapping[str, Any], context: str) -> dict[str, Any]:
    engine = _require_mapping(bundle.get("native_engine"), f"{context}.native_engine")
    engine_id = _require_string(engine, "engine_id", f"{context}.native_engine")
    if engine_id != "catanatron_rs":
        raise ValidationError(f"{context}.native_engine.engine_id must be 'catanatron_rs'")
    version = _require_string(engine, "version", f"{context}.native_engine")
    path = _require_string(engine, "path", f"{context}.native_engine")
    digest = _require_sha256(engine, "sha256", f"{context}.native_engine")
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"{context}.native_engine.path does not exist: {source}")
    measured = sha256_file(source)
    if measured != digest:
        raise ValidationError(
            f"{context}.native_engine SHA mismatch: recorded={digest}, measured={measured}"
        )
    return {"engine_id": engine_id, "version": version, "path": path, "sha256": digest}


def _validate_hardware(bundle: Mapping[str, Any], context: str) -> dict[str, Any]:
    hardware = _require_mapping(bundle.get("hardware"), f"{context}.hardware")
    device = _require_string(hardware, "device", f"{context}.hardware")
    device_type = _require_string(hardware, "device_type", f"{context}.hardware")
    if device_type not in {"cpu", "cuda"}:
        raise ValidationError(f"{context}.hardware.device_type must be cpu or cuda")
    host = _require_sha256(hardware, "host_fingerprint", f"{context}.hardware")
    machine = _require_string(hardware, "machine", f"{context}.hardware")
    model = _require_string(hardware, "accelerator_model", f"{context}.hardware")
    uuid = hardware.get("accelerator_uuid")
    total_memory = hardware.get("total_memory_bytes")
    capability = hardware.get("compute_capability")
    if device_type == "cuda":
        if not isinstance(uuid, str) or not uuid.strip():
            raise ValidationError(f"{context}.hardware.accelerator_uuid is required for CUDA")
        if isinstance(total_memory, bool) or not isinstance(total_memory, int) or total_memory <= 0:
            raise ValidationError(f"{context}.hardware.total_memory_bytes must be positive for CUDA")
        if not isinstance(capability, str) or not capability.strip():
            raise ValidationError(f"{context}.hardware.compute_capability is required for CUDA")
    elif any(value is not None for value in (uuid, total_memory, capability)):
        raise ValidationError(f"{context}.hardware CPU accelerator details must be null")
    return {
        "device": device,
        "device_type": device_type,
        "host_fingerprint": host,
        "machine": machine,
        "accelerator_model": model,
        "accelerator_uuid": uuid,
        "total_memory_bytes": total_memory,
        "compute_capability": capability,
    }


def _pair_counter_totals(pair_games: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        key: sum(float(game["counters"][key]) for game in pair_games)
        for key in COUNTER_FIELDS
    }


def _validate_pair_alignment(
    pairs_by_arm: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> None:
    arm_ids = sorted(pairs_by_arm)
    reference_arm = arm_ids[0]
    reference_pairs = set(pairs_by_arm[reference_arm])
    reference_seed_and_seats = {
        pair_id: (
            int(pairs_by_arm[reference_arm][pair_id][0]["seed"]),
            sorted(
                (
                    int(game["seat_assignment"]["candidate"]),
                    int(game["seat_assignment"]["reference"]),
                )
                for game in pairs_by_arm[reference_arm][pair_id]
            ),
        )
        for pair_id in reference_pairs
    }
    for arm_id in arm_ids[1:]:
        if set(pairs_by_arm[arm_id]) != reference_pairs:
            raise ValidationError(
                f"arm {arm_id!r} does not cover exactly the same pair_ids as {reference_arm!r}"
            )
        for pair_id in reference_pairs:
            seed = int(pairs_by_arm[arm_id][pair_id][0]["seed"])
            seats = sorted(
                (
                    int(game["seat_assignment"]["candidate"]),
                    int(game["seat_assignment"]["reference"]),
                )
                for game in pairs_by_arm[arm_id][pair_id]
            )
            expected_seed, expected_seats = reference_seed_and_seats[pair_id]
            if seed != expected_seed:
                raise ValidationError(
                    f"pair {pair_id!r} seed mismatch across arms: {expected_seed} vs {seed}"
                )
            if seats != expected_seats:
                raise ValidationError(
                    f"pair {pair_id!r} seat schedule mismatch across arms: "
                    f"{expected_seats} vs {seats}"
                )


def _validate_budget_contract(
    campaign: Mapping[str, Any],
    regime: str,
    pairs_by_arm: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    contracts = _require_mapping(campaign.get("budget_contracts"), "campaign.budget_contracts")
    contract = _require_mapping(contracts.get(regime), f"campaign.budget_contracts.{regime}")
    metrics = list(_require_sequence(contract.get("match_metrics"), f"budget_contracts.{regime}.match_metrics"))
    if not metrics:
        raise ValidationError(f"budget_contracts.{regime}.match_metrics must not be empty")
    unknown = sorted(set(metrics) - set(COUNTER_FIELDS))
    if unknown:
        raise ValidationError(f"budget contract references unknown counters: {', '.join(unknown)}")
    absolute_tolerance = _require_number(
        contract, "absolute_tolerance", f"budget_contracts.{regime}"
    )
    relative_tolerance = _require_number(
        contract, "relative_tolerance", f"budget_contracts.{regime}"
    )

    _validate_pair_alignment(pairs_by_arm)
    arm_ids = sorted(pairs_by_arm)
    reference_pairs = set(pairs_by_arm[arm_ids[0]])
    reference_seeds = {
        pair_id: int(pairs_by_arm[arm_ids[0]][pair_id][0]["seed"])
        for pair_id in reference_pairs
    }
    reference_seat_schedules = {
        pair_id: sorted(
            (
                int(game["seat_assignment"]["candidate"]),
                int(game["seat_assignment"]["reference"]),
            )
            for game in pairs_by_arm[arm_ids[0]][pair_id]
        )
        for pair_id in reference_pairs
    }
    for arm_id in arm_ids[1:]:
        if set(pairs_by_arm[arm_id]) != reference_pairs:
            raise ValidationError(
                f"arm {arm_id!r} does not cover exactly the same pair_ids as {arm_ids[0]!r}"
            )
        for pair_id in reference_pairs:
            seed = int(pairs_by_arm[arm_id][pair_id][0]["seed"])
            if seed != reference_seeds[pair_id]:
                raise ValidationError(
                    f"pair {pair_id!r} seed mismatch across arms: {reference_seeds[pair_id]} vs {seed}"
                )
            seat_schedule = sorted(
                (
                    int(game["seat_assignment"]["candidate"]),
                    int(game["seat_assignment"]["reference"]),
                )
                for game in pairs_by_arm[arm_id][pair_id]
            )
            if seat_schedule != reference_seat_schedules[pair_id]:
                raise ValidationError(
                    f"pair {pair_id!r} seat schedule mismatch across arms: "
                    f"{reference_seat_schedules[pair_id]} vs {seat_schedule}"
                )

    comparisons: list[dict[str, Any]] = []
    for pair_id in sorted(reference_pairs):
        totals = {
            arm_id: _pair_counter_totals(pairs_by_arm[arm_id][pair_id])
            for arm_id in arm_ids
        }
        for metric in metrics:
            values = {arm_id: arm_totals[metric] for arm_id, arm_totals in totals.items()}
            low = min(values.values())
            high = max(values.values())
            allowed = max(absolute_tolerance, relative_tolerance * max(abs(low), abs(high)))
            if high - low > allowed + 1e-12:
                raise ValidationError(
                    f"{regime} budget mismatch for pair={pair_id!r}, metric={metric}: "
                    f"range={high - low:.9g} exceeds tolerance={allowed:.9g}; values={values}"
                )
            comparisons.append(
                {
                    "pair_id": pair_id,
                    "seed": reference_seeds[pair_id],
                    "metric": metric,
                    "values": values,
                    "range": high - low,
                    "allowed_tolerance": allowed,
                }
            )
    return {
        "regime": regime,
        "match_metrics": metrics,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "status": "pass",
        "comparisons": comparisons,
    }


def validate_and_aggregate(
    campaign: Mapping[str, Any],
    result_bundles: Sequence[Mapping[str, Any]],
    *,
    verify_local_checkpoints: bool = False,
) -> dict[str, Any]:
    registry = _validate_registry(campaign)
    campaign_id = str(campaign["campaign_id"])
    information_requirement = str(campaign["required_information_regime"])
    if not result_bundles:
        raise ValidationError("at least one result bundle is required")

    expected_regime: str | None = None
    seen_arm_ids: set[str] = set()
    summaries: list[ArmSummary] = []
    pairs_by_arm: dict[str, Mapping[str, Sequence[Mapping[str, Any]]]] = {}
    run_ids: list[str] = []
    expected_reference: dict[str, Any] | None = None
    expected_seed_manifest: dict[str, Any] | None = None
    expected_training_manifest: dict[str, Any] | None = None
    expected_native_engine: dict[str, Any] | None = None
    expected_hardware_class: tuple[Any, ...] | None = None
    hardware_by_arm: dict[str, dict[str, Any]] = {}

    for bundle_index, raw_bundle in enumerate(result_bundles):
        context = f"results[{bundle_index}]"
        bundle = _require_mapping(raw_bundle, context)
        if bundle.get("schema_version") != SCHEMA_VERSION:
            raise ValidationError(f"{context}.schema_version must be {SCHEMA_VERSION!r}")
        if _require_string(bundle, "campaign_id", context) != campaign_id:
            raise ValidationError(f"{context}.campaign_id does not match campaign")
        if _require_string(bundle, "track", context) != "2p_no_trade":
            raise ValidationError(f"{context}.track must be '2p_no_trade'")
        if (
            _require_string(bundle, "required_information_regime", context)
            != information_requirement
        ):
            raise ValidationError(
                f"{context}.required_information_regime does not match campaign"
            )
        seed_manifest = _validate_seed_manifest(bundle, context)
        if expected_seed_manifest is None:
            expected_seed_manifest = seed_manifest
        elif seed_manifest != expected_seed_manifest:
            raise ValidationError(
                f"{context}.seed_manifest does not exactly match the other arms"
            )
        training_manifest = _validate_training_manifest(bundle, context)
        if expected_training_manifest is None:
            expected_training_manifest = training_manifest
        elif (
            bool(campaign["require_same_training_manifest"])
            and training_manifest != expected_training_manifest
        ):
            raise ValidationError(
                f"{context}.training_manifest does not exactly match the other arms"
            )
        regime = _require_string(bundle, "budget_regime", context)
        if regime not in ("equal_work", "equal_time"):
            raise ValidationError(f"{context}.budget_regime must be equal_work or equal_time")
        if expected_regime is None:
            expected_regime = regime
        elif regime != expected_regime:
            raise ValidationError("all result bundles must use the same budget_regime")
        native_engine = _validate_native_engine(bundle, context)
        if expected_native_engine is None:
            expected_native_engine = native_engine
        elif native_engine != expected_native_engine:
            raise ValidationError(
                f"{context}.native_engine does not exactly match the other arms"
            )
        hardware = _validate_hardware(bundle, context)
        required_model = campaign.get("required_accelerator_model")
        if required_model is not None:
            if not isinstance(required_model, str) or not required_model.strip():
                raise ValidationError(
                    "campaign.required_accelerator_model must be a non-empty string when set"
                )
            if required_model not in str(hardware["accelerator_model"]):
                raise ValidationError(
                    f"{context}.hardware.accelerator_model={hardware['accelerator_model']!r} "
                    f"does not satisfy required {required_model!r}"
                )
        hardware_class = (
            hardware["device_type"],
            hardware["accelerator_model"],
            hardware["total_memory_bytes"],
            hardware["compute_capability"],
        )
        if regime == "equal_time":
            if expected_hardware_class is None:
                expected_hardware_class = hardware_class
            elif hardware_class != expected_hardware_class:
                raise ValidationError(
                    f"{context}.hardware class does not match the other equal_time arms"
                )

        run_id = _require_string(bundle, "run_id", context)
        if run_id in run_ids:
            raise ValidationError(f"duplicate run_id {run_id!r}")
        run_ids.append(run_id)
        arm_id = _require_string(bundle, "arm_id", context)
        if arm_id in seen_arm_ids:
            raise ValidationError(f"duplicate result bundle for arm_id {arm_id!r}")
        seen_arm_ids.add(arm_id)
        hardware_by_arm[arm_id] = hardware
        registry_arm = registry.get(arm_id)
        if registry_arm is None:
            raise ValidationError(f"{context}.arm_id {arm_id!r} is not registered")
        if registry_arm.get("runnable") is not True:
            raise ValidationError(
                f"{context}.arm_id {arm_id!r} is not runnable; implement and attest its adapter first"
            )

        (
            architecture_id,
            params,
            search_id,
            architecture_config_sha,
            search_config_sha,
            checkpoint_sha,
            git_commit,
            code_dirty,
            patch_sha,
        ) = _validate_provenance(
            bundle,
            context,
            registry_arm,
            verify_local_checkpoints=verify_local_checkpoints,
        )
        reference = _validate_reference(
            bundle,
            context,
            verify_local_checkpoints=verify_local_checkpoints,
        )
        if expected_reference is None:
            expected_reference = reference
        elif reference != expected_reference:
            raise ValidationError(
                f"{context}.reference does not exactly match the reference used by other arms"
            )
        raw_games = _require_sequence(bundle.get("games"), f"{context}.games")
        games = [_require_mapping(game, f"{context}.games[{i}]") for i, game in enumerate(raw_games)]
        paired = _validate_pairing(
            games,
            context,
            information_requirement=information_requirement,
        )
        if len(paired) != int(seed_manifest["seed_count"]):
            raise ValidationError(
                f"{context} has {len(paired)} paired seeds but seed_manifest "
                f"declares {seed_manifest['seed_count']}"
            )
        observed_seeds = [
            int(pair_games[0]["seed"])
            for _pair_id, pair_games in sorted(paired.items())
        ]
        if observed_seeds != list(seed_manifest["seeds"]):
            raise ValidationError(
                f"{context} paired game seeds do not exactly match the ordered "
                f"seed_manifest seeds: games={observed_seeds}, "
                f"manifest={list(seed_manifest['seeds'])}"
            )
        pairs_by_arm[arm_id] = paired

        scores = [float(game["candidate_score"]) for game in games]
        counters = {
            key: sum(float(game["counters"][key]) for game in games)
            for key in COUNTER_FIELDS
        }
        summaries.append(
            ArmSummary(
                arm_id=arm_id,
                comparison_role=str(registry_arm.get("comparison_role", "ranked")),
                architecture_id=architecture_id,
                parameter_count=params,
                search_id=search_id,
                architecture_config_sha256=architecture_config_sha,
                search_config_sha256=search_config_sha,
                checkpoint_sha256=checkpoint_sha,
                git_commit=git_commit,
                code_dirty=code_dirty,
                patch_sha256=patch_sha,
                games=len(games),
                pairs=len(paired),
                wins=sum(score == 1.0 for score in scores),
                draws=sum(score == 0.5 for score in scores),
                losses=sum(score == 0.0 for score in scores),
                mean_score=sum(scores) / len(scores),
                counters=counters,
            )
        )

    required_arm_ids = set(
        str(value)
        for value in _require_sequence(campaign.get("required_arm_ids"), "campaign.required_arm_ids")
    )
    missing_arms = sorted(required_arm_ids - seen_arm_ids)
    extra_arms = sorted(seen_arm_ids - required_arm_ids)
    if missing_arms or extra_arms:
        raise ValidationError(
            f"result arm set must exactly match required_arm_ids; missing={missing_arms}, extra={extra_arms}"
        )

    for field_name in campaign.get("frozen_candidate_fields", []):
        values = {getattr(summary, str(field_name)) for summary in summaries}
        if len(values) != 1:
            raise ValidationError(
                f"campaign requires {field_name} frozen across arms, got {sorted(values)}"
            )

    assert expected_regime is not None
    assert expected_reference is not None
    assert expected_seed_manifest is not None
    assert expected_training_manifest is not None
    assert expected_native_engine is not None
    _validate_pair_alignment(pairs_by_arm)
    ranked_pair_sets = {
        summary.arm_id: pairs_by_arm[summary.arm_id]
        for summary in summaries
        if summary.comparison_role == "ranked"
    }
    if len(ranked_pair_sets) < 2:
        raise ValidationError("a leaderboard needs at least two arms with comparison_role=ranked")
    budget_validation = _validate_budget_contract(campaign, expected_regime, ranked_pair_sets)
    ranked = sorted(
        (summary for summary in summaries if summary.comparison_role == "ranked"),
        key=lambda row: (-row.mean_score, row.arm_id),
    )
    controls = sorted(
        (summary for summary in summaries if summary.comparison_role == "control"),
        key=lambda row: row.arm_id,
    )
    reference_pair_set = next(iter(pairs_by_arm.values()))
    pair_ids = sorted(reference_pair_set.keys())
    pairing_schedule = [
        {
            "pair_id": pair_id,
            "seed": int(reference_pair_set[pair_id][0]["seed"]),
            "seat_assignments": sorted(
                (
                    {
                        "candidate": int(game["seat_assignment"]["candidate"]),
                        "reference": int(game["seat_assignment"]["reference"]),
                    }
                    for game in reference_pair_set[pair_id]
                ),
                key=lambda seats: (seats["candidate"], seats["reference"]),
            ),
        }
        for pair_id in pair_ids
    ]
    seeds = [int(pair["seed"]) for pair in pairing_schedule]
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "budget_regime": expected_regime,
        "valid": True,
        "ranking_scope": "descriptive_only_paired_games; use a predeclared gate for claims",
        "run_ids": sorted(run_ids),
        "reference": expected_reference,
        "required_information_regime": information_requirement,
        "seed_manifest": expected_seed_manifest,
        "training_manifest": expected_training_manifest,
        "native_engine": expected_native_engine,
        "hardware_by_arm": {arm: hardware_by_arm[arm] for arm in sorted(hardware_by_arm)},
        "require_same_training_manifest": bool(
            campaign["require_same_training_manifest"]
        ),
        "paired_seed_count": len(seeds),
        "paired_seeds": seeds,
        "pairing_schedule": pairing_schedule,
        "budget_validation": budget_validation,
        "leaderboard": [row.to_dict() for row in ranked],
        "controls": [row.to_dict() for row in controls],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = list(report["leaderboard"])
    controls = list(report.get("controls", []))
    lines = [
        f"# R&D leaderboard: {report['campaign_id']}",
        "",
        f"Budget regime: `{report['budget_regime']}`  ",
        f"Budget contract: **{str(report['budget_validation']['status']).upper()}**  ",
        f"Paired seeds: **{report['paired_seed_count']}**  ",
        "Ranking is descriptive; it is not a promotion or significance claim.",
        "",
        "| Rank | Arm | Architecture | Params | Search | W-D-L | Score | Logical leaves | Orientation rows | Eval calls | Wall time (s) |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        work = row["work_totals"]
        lines.append(
            "| {rank} | `{arm}` | `{arch}` | {params:,} | `{search}` | {wins}-{draws}-{losses} "
            "| {score:.4f} | {leaves:,.0f} | {orientations:,.0f} | {calls:,.0f} | {wall:.3f} |".format(
                rank=rank,
                arm=row["arm_id"],
                arch=row["architecture_id"],
                params=int(row["parameter_count"]),
                search=row["search_id"],
                wins=row["wins"],
                draws=row["draws"],
                losses=row["losses"],
                score=float(row["mean_score"]),
                leaves=float(work["logical_leaves"]),
                orientations=float(work["orientation_rows"]),
                calls=float(work["evaluator_calls"]),
                wall=float(work["wall_time_sec"]),
            )
        )
    if controls:
        lines.extend(
            [
                "",
                "## Non-compute-matched controls",
                "",
                "| Arm | Architecture | Search | W-D-L | Score | Logical leaves | Wall time (s) |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in controls:
            work = row["work_totals"]
            lines.append(
                f"| `{row['arm_id']}` | `{row['architecture_id']}` | `{row['search_id']}` | "
                f"{row['wins']}-{row['draws']}-{row['losses']} | {row['mean_score']:.4f} | "
                f"{work['logical_leaves']:,.0f} | {work['wall_time_sec']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "| Arm | Git commit | Architecture config | Search config | Checkpoint | Nominal visits | Scheduled visits |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in rows + controls:
        work = row["work_totals"]
        lines.append(
            f"| `{row['arm_id']}` | `{row['code']['git_commit']}` | "
            f"`{row['architecture_config_sha256']}` | `{row['search_config_sha256']}` | "
            f"`{row['checkpoint_sha256']}` | "
            f"{work['nominal_visits']:,.0f} | {work['scheduled_visits']:,.0f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate an arm/campaign registry")
    validate.add_argument("--campaign", required=True)

    aggregate = subparsers.add_parser(
        "aggregate", help="validate result bundles and write JSON plus Markdown"
    )
    aggregate.add_argument("--campaign", required=True)
    aggregate.add_argument("--result", action="append", required=True, help="repeat per arm")
    aggregate.add_argument("--out-json", required=True)
    aggregate.add_argument("--out-md", required=True)
    aggregate.add_argument(
        "--verify-local-checkpoints",
        action="store_true",
        help="hash each recorded checkpoint path and require an exact match",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        campaign = _load_json(args.campaign)
        if args.command == "validate-config":
            registry = _validate_registry(campaign)
            print(json.dumps({"valid": True, "campaign_id": campaign["campaign_id"], "arms": len(registry)}))
            return 0

        bundles = [_load_json(path) for path in args.result]
        report = validate_and_aggregate(
            campaign,
            bundles,
            verify_local_checkpoints=bool(args.verify_local_checkpoints),
        )
        out_json = Path(args.out_json)
        out_md = Path(args.out_md)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out_md.write_text(render_markdown(report), encoding="utf-8")
        print(json.dumps({"valid": True, "out_json": str(out_json), "out_md": str(out_md)}))
        return 0
    except ValidationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
