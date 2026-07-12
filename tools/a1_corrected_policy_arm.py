#!/usr/bin/env python3
"""Prepare, but never launch, the next one-dose A1 learner recipe.

The production-next control preserves the operator that actually won: the
4/7 n128, 8/35 n256, 1/5 predecessor-replay mixture supervises policy and value
on all three components, uses 0.9 soft targets, loser weight 1, no auxiliary
policy stream, and no KL anchor.  Replay removal and pure 1.0 search targets are
useful *separate* ablations, but combining them in the default would destroy the
causal control.  The emitted manifest is preparation only; this module
deliberately has no launch mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import train_bc  # noqa: E402


SCHEMA = "a1-next-production-learner-manifest-v3"
SUPERVISION_CONTRACT_SCHEMA = "a1-next-learner-supervision-contract-v2"
CURRENT_TEACHER_COMPONENT_IDS = ("n128_current", "n256_current")
REPLAY_COMPONENT_ID = "gen3_replay"
REPLAY_ANCHOR_WEIGHT = 0.0
WINNING_COMPONENT_RATIOS = (4.0 / 7.0, 8.0 / 35.0, 1.0 / 5.0)
GLOBAL_ROW_DOSE = 4_194_304
# The expectation is derived from authenticated training-only rows for every
# concrete descriptor.  This tolerance is wider than six binomial sigmas for
# the historical winning mix while still rejecting ~0.8% dose drift.
POLICY_BASE_ACTIVE_ROW_TOLERANCE = 4_100
EXPECTED_POLICY_AUX_ACTIVE_ROWS = 0
EVENT_HISTORY_COMMAND_CONTRACT_SCHEMA = "a1-event-history-command-contract-v1"
EVENT_HISTORY_ACK_FLAG = (
    "--acknowledge-empty-event-history-payload-inventory-sha256"
)
EVENT_HISTORY_CROP_FLAG = "--crop-authenticated-empty-event-history"
LINEAGE_ROLES = (
    "parent_failed_claim",
    "parent_failed_receipt",
    "retry_contract",
    "retry_receipt",
)
LINEAGE_DIGEST_FIELDS = {
    "parent_failed_claim": "state_sha256",
    "parent_failed_receipt": "receipt_sha256",
    "retry_contract": "retry_contract_sha256",
    "retry_receipt": "receipt_sha256",
}
SOURCE_FILES = (
    "tools/a1_corrected_policy_arm.py",
    "tools/a1_corrected_policy_arm_execute.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/entity_token_policy.py",
)


class ArmError(RuntimeError):
    """The requested arm is not the exact corrected one-dose experiment."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_ref(path: Path) -> dict[str, str]:
    lexical = path.expanduser()
    if lexical.is_symlink():
        raise ArmError(f"bound artifact must be a regular non-symlink file: {lexical}")
    path = lexical.resolve(strict=True)
    if not path.is_file():
        raise ArmError(f"bound artifact must be a regular non-symlink file: {path}")
    return {"path": str(path), "sha256": _file_sha(path)}


def _load_json(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    ref = _file_ref(path)
    try:
        value = json.loads(Path(ref["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArmError(f"cannot parse bound JSON {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("schema_version"), str):
        raise ArmError(f"bound JSON has no schema identity: {path}")
    return value, ref


def _option(command: Sequence[str], flag: str) -> str:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ArmError(f"source command must contain exactly one valued {flag}")
    value = str(command[positions[0] + 1])
    if value.startswith("--"):
        raise ArmError(f"source command has valueless {flag}")
    return value


def _set_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) > 1:
        raise ArmError(f"source command repeats {flag}")
    if positions:
        index = positions[0]
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise ArmError(f"source command has valueless {flag}")
        command[index + 1] = value
    else:
        command.extend((flag, value))


def _event_history_training_contract(
    descriptor_meta: Mapping[str, Any],
) -> dict[str, Any]:
    components = descriptor_meta.get("components")
    if not isinstance(components, list) or not components:
        raise ArmError("descriptor has no authenticated event-history components")
    bindings = []
    for component in components:
        if not isinstance(component, Mapping):
            raise ArmError("descriptor event-history component is malformed")
        component_id = component.get("component_id")
        inventory = component.get("payload_inventory_sha256")
        if (
            not isinstance(component_id, str)
            or not isinstance(inventory, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", inventory) is None
        ):
            raise ArmError("descriptor component lacks an authenticated payload inventory")
        bindings.append(
            {"component_id": component_id, "payload_inventory_sha256": inventory}
        )
    return {
        "schema": EVENT_HISTORY_COMMAND_CONTRACT_SCHEMA,
        "empty_payload_inventory_acknowledgements": bindings,
        "crop_authenticated_empty_event_history": True,
    }


def _bind_event_history_training_command(
    command: list[str], descriptor_meta: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _event_history_training_contract(descriptor_meta)
    expected = [
        row["payload_inventory_sha256"]
        for row in contract["empty_payload_inventory_acknowledgements"]
    ]
    positions = [index for index, value in enumerate(command) if value == EVENT_HISTORY_ACK_FLAG]
    had_crop = EVENT_HISTORY_CROP_FLAG in command
    if positions:
        observed = [
            command[index + 1]
            for index in positions
            if index + 1 < len(command) and not command[index + 1].startswith("--")
        ]
        if (
            len(observed) != len(positions)
            or len(set(observed)) != len(observed)
            or any(re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None for value in observed)
        ):
            raise ArmError("source command event-history acknowledgements drift")
        # The next recipe removes the replay component, so an authenticated
        # three-component source command legitimately carries one obsolete ACK.
        # Replace the complete valued set; never leave an extra authorization
        # that is no longer represented by the derived descriptor.
        if observed != expected:
            for index in reversed(positions):
                del command[index : index + 2]
            for inventory in expected:
                command.extend((EVENT_HISTORY_ACK_FLAG, inventory))
    else:
        for inventory in expected:
            command.extend((EVENT_HISTORY_ACK_FLAG, inventory))
    if EVENT_HISTORY_CROP_FLAG not in command:
        command.append(EVENT_HISTORY_CROP_FLAG)
    elif command.count(EVENT_HISTORY_CROP_FLAG) != 1:
        raise ArmError("source command repeats authenticated event-history crop flag")
    return contract, {
        "event_history_acknowledgements": {
            "source": "absent" if not positions else observed,
            "treatment": expected,
        },
        EVENT_HISTORY_CROP_FLAG: {
            "source": "present" if had_crop else "absent",
            "treatment": "present",
        },
    }


def _load_source_receipt(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload, ref = _load_json(path)
    stated = payload.get("receipt_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if stated != _digest(unhashed):
        raise ArmError("source launch receipt schema or semantic digest is invalid")
    if payload.get("diagnostic_only") is not True or payload.get("promotion_eligible") is not False:
        raise ArmError("source launch receipt must be diagnostic-only/non-promotable")
    command = payload.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ArmError("source launch receipt has no replayable command")
    if payload.get("command_sha256") != _digest(command):
        raise ArmError("source launch command digest drift")
    return payload, ref


def _preflight_descriptor(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = path.expanduser().resolve(strict=True)
    try:
        verified = train_bc._preflight_memmap_composite_descriptor(path)  # noqa: SLF001
    except SystemExit as error:
        raise ArmError(f"corrected descriptor preflight failed: {error}") from error
    return verified, _file_ref(path)


def _component_training_policy_activity(component: Mapping[str, Any]) -> dict[str, Any]:
    """Measure policy-active mass on authenticated non-validation rows."""

    component_id = component.get("component_id")
    corpus_dir = Path(str(component.get("corpus_dir", ""))).expanduser().resolve(strict=True)
    corpus_meta = component.get("corpus_meta")
    if not isinstance(component_id, str) or not isinstance(corpus_meta, Mapping):
        raise ArmError("descriptor component lacks authenticated corpus metadata")
    row_count = int(corpus_meta.get("row_count", 0))
    if row_count <= 0:
        raise ArmError(f"component {component_id} has no rows")
    validation_path = Path(
        str(component.get("validation_manifest", ""))
    ).expanduser().resolve(strict=True)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation_seeds = validation.get("game_seeds")
    if not isinstance(validation_seeds, list) or not all(
        isinstance(seed, int) for seed in validation_seeds
    ):
        raise ArmError(f"component {component_id} validation seeds are malformed")
    game_seeds = np.memmap(
        corpus_dir / "game_seed.dat", dtype=np.dtype("<i8"), mode="r", shape=(row_count,)
    )
    policy_weights = np.memmap(
        corpus_dir / "policy_weight_multiplier.dat",
        dtype=np.dtype("<f4"), mode="r", shape=(row_count,),
    )
    held_out = np.asarray(validation_seeds, dtype=np.int64)
    training_rows = 0
    active_rows = 0
    per_game_rows: dict[int, int] = {}
    per_game_active: dict[int, int] = {}
    chunk_rows = 1 << 22
    for start in range(0, row_count, chunk_rows):
        stop = min(start + chunk_rows, row_count)
        is_validation = np.isin(game_seeds[start:stop], held_out)
        training = ~is_validation
        training_rows += int(np.count_nonzero(training))
        active = training & (policy_weights[start:stop] > 0.0)
        active_rows += int(np.count_nonzero(active))
        training_seeds = np.asarray(game_seeds[start:stop][training], dtype=np.int64)
        unique, inverse = np.unique(training_seeds, return_inverse=True)
        row_counts = np.bincount(inverse)
        active_counts = np.bincount(
            inverse, weights=np.asarray(active[training], dtype=np.int64)
        ).astype(np.int64)
        for seed, rows, active_count in zip(
            unique.tolist(), row_counts.tolist(), active_counts.tolist(), strict=True
        ):
            per_game_rows[seed] = per_game_rows.get(seed, 0) + int(rows)
            per_game_active[seed] = per_game_active.get(seed, 0) + int(active_count)
    expected_training_rows = (
        corpus_meta.get("a1_post_wave_audit", {}).get("training_row_count")
        if isinstance(corpus_meta.get("a1_post_wave_audit"), Mapping)
        else None
    )
    if expected_training_rows is not None and training_rows != int(expected_training_rows):
        raise ArmError(
            f"component {component_id} training-row count differs from authenticated audit"
        )
    if training_rows <= 0 or active_rows <= 0:
        raise ArmError(f"component {component_id} has no training policy-active mass")
    expected_training_games = (
        corpus_meta.get("selected_game_seed_manifest", {}).get("training_game_count")
        if isinstance(corpus_meta.get("selected_game_seed_manifest"), Mapping)
        else None
    )
    if expected_training_games is not None and len(per_game_rows) != int(expected_training_games):
        raise ArmError(
            f"component {component_id} training-game count differs from authenticated audit"
        )
    game_uniform_fraction = float(
        np.mean(
            [
                per_game_active.get(seed, 0) / rows
                for seed, rows in per_game_rows.items()
            ]
        )
    )
    return {
        "component_id": component_id,
        "training_rows": training_rows,
        "training_policy_active_rows": active_rows,
        "training_game_count": len(per_game_rows),
        "raw_row_policy_active_fraction": active_rows / training_rows,
        "game_uniform_policy_active_fraction": game_uniform_fraction,
        "validation_manifest_sha256": component.get("validation_manifest_sha256"),
        "payload_inventory_sha256": component.get("payload_inventory_sha256"),
    }


def _derive_policy_active_dose(
    descriptor_meta: Mapping[str, Any], *, global_row_dose: int = GLOBAL_ROW_DOSE
) -> dict[str, Any]:
    component_ids = descriptor_meta.get("component_ids")
    ratios = descriptor_meta.get("component_game_sampling_ratios")
    components = descriptor_meta.get("components")
    if (
        not isinstance(component_ids, list)
        or not isinstance(ratios, list)
        or not isinstance(components, list)
        or len(component_ids) != len(ratios)
        or len(component_ids) != len(components)
        or not np.isclose(sum(float(value) for value in ratios), 1.0)
    ):
        raise ArmError("cannot derive active dose from malformed component mixture")
    stats = [_component_training_policy_activity(component) for component in components]
    if [row["component_id"] for row in stats] != component_ids:
        raise ArmError("active-dose component ordering drift")
    available_training_rows = sum(int(row["training_rows"]) for row in stats)
    if available_training_rows < int(global_row_dose):
        raise ArmError(
            "authenticated training corpus is smaller than the sealed one-dose draw; "
            "epochs=1 would stop before max_steps"
        )
    expected_fraction = sum(
        float(ratio) * float(row["game_uniform_policy_active_fraction"])
        for ratio, row in zip(ratios, stats, strict=True)
    )
    expected_rows = int(round(global_row_dose * expected_fraction))
    tolerance = POLICY_BASE_ACTIVE_ROW_TOLERANCE
    return {
        "derivation": "authenticated_game_uniform_activity_weighted_by_component_sampling_ratio",
        "component_statistics": stats,
        "component_sampling_ratios": [float(value) for value in ratios],
        "global_row_dose": int(global_row_dose),
        "available_training_rows": available_training_rows,
        "expected_active_fraction": expected_fraction,
        "reference_base_active_rows": expected_rows,
        "base_active_rows_tolerance": tolerance,
        "min_base_active_rows": expected_rows - tolerance,
        "max_base_active_rows": expected_rows + tolerance,
        "expected_aux_active_rows": EXPECTED_POLICY_AUX_ACTIVE_ROWS,
        "accounting": "realized_policy_active_rows_not_global_samples",
    }


def _bind_teacher_lineage(
    descriptor_meta: Mapping[str, Any], *, parent_checkpoint_sha256: str,
    expected_predecessor_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind current data via the canonical next-turn lineage semantics."""

    from tools import a1_flywheel_turn as flywheel

    try:
        return flywheel.bind_teacher_lineage(
            descriptor_meta,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            expected_predecessor_sha256=expected_predecessor_sha256,
        )
    except flywheel.FlywheelTurnError as error:
        raise ArmError(str(error)) from error


def _build_corrected_descriptor(
    source_path: Path, output_path: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    source, source_ref = _preflight_descriptor(source_path)
    component_ids = source.get("component_ids")
    ratios = source.get("component_game_sampling_ratios")
    if (
        source.get("schema_version") != "memmap_composite_v2"
        or not isinstance(component_ids, list)
        or len(component_ids) < 2
        or len(set(component_ids)) != len(component_ids)
        or not all(component in CURRENT_TEACHER_COMPONENT_IDS for component in component_ids[:-1])
        or component_ids[-1] not in {REPLAY_COMPONENT_ID, "predecessor_replay"}
        or not isinstance(ratios, list)
        or len(ratios) != len(component_ids)
        or not np.isclose(sum(float(value) for value in ratios[:-1]), 0.8)
        or not np.isclose(float(ratios[-1]), 0.2)
    ):
        raise ArmError(
            "source descriptor must bind 80% current teachers then 20% predecessor replay"
        )
    overrides = dict(source["learner_recipe_overrides"])
    overrides["policy_kl_anchor_weight"] = REPLAY_ANCHOR_WEIGHT
    overrides["policy_kl_anchor_direction"] = "forward"
    overrides["loser_sample_weight"] = 1.0
    components = []
    for source_component, ratio in zip(source["components"], ratios, strict=True):
        components.append(
            {
                key: source_component[key]
                for key in (
                    "corpus_dir", "corpus_meta_sha256", "payload_inventory_sha256",
                    "validation_manifest", "validation_manifest_sha256", "component_id",
                )
            }
            | {"game_sampling_ratio": ratio}
        )
    payload = {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "components": components,
        "learner_recipe_overrides": overrides,
        "learner_recipe_overrides_sha256": _digest(overrides),
        "policy_kl_anchor_component_ids": list(component_ids),
        "policy_distillation_component_ids": list(component_ids),
        "value_training_component_ids": list(component_ids),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != encoded:
            raise ArmError(f"prepared descriptor drift: {output_path}")
    else:
        temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, output_path)
    verified, ref = _preflight_descriptor(output_path)
    if (
        verified.get("schema_version") != "memmap_composite_v2"
        or verified.get("policy_distillation_scope_explicit") is not True
        or verified.get("policy_distillation_component_ids") != component_ids
        or verified.get("component_ids") != component_ids
        or verified.get("component_game_sampling_ratios") != list(ratios)
        or verified.get("policy_kl_anchor_component_ids") != component_ids
        or verified.get("value_training_component_ids") != component_ids
    ):
        raise ArmError("derived descriptor must preserve the winning 80:20 operator mix")
    verified_overrides = verified.get("learner_recipe_overrides")
    if not isinstance(verified_overrides, dict) or (
        float(verified_overrides.get("policy_kl_anchor_weight", -1.0))
        != REPLAY_ANCHOR_WEIGHT
        or verified_overrides.get("policy_kl_anchor_direction") != "forward"
        or float(verified_overrides.get("loser_sample_weight", -1.0)) != 1.0
    ):
        raise ArmError("derived descriptor must bind loser=1 and disabled replay anchor")
    return verified, ref, source_ref


def _build_corrected_sentinel(
    *, source_receipt: dict[str, Any], source_descriptor: dict[str, Any],
    descriptor: Path, descriptor_meta: dict[str, Any], output_path: Path,
    python: str, repo: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str], dict[str, Any]]:
    raw_source = source_receipt.get("sentinel")
    expected_source_sha = source_receipt.get("sentinel_sha256")
    if not isinstance(raw_source, str) or not isinstance(expected_source_sha, str):
        raise ArmError("source receipt does not bind a validation sentinel")
    source_payload, source_ref = _load_json(Path(raw_source))
    if source_ref["sha256"] != expected_source_sha:
        raise ArmError("source receipt validation sentinel bytes drifted")
    if (
        source_payload.get("schema_version") != "train-validation-game-sentinel-v1"
        or source_payload.get("source_composite_descriptor_file_sha256")
        != source_descriptor["descriptor_file_sha256"]
        or source_payload.get("source_composite_descriptor_fingerprint")
        != source_descriptor["descriptor_fingerprint"]
    ):
        raise ArmError("source validation sentinel is not bound to source descriptor")
    source_games = source_payload.get("game_seeds")
    if (
        not isinstance(source_games, list)
        or not source_games
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in source_games)
    ):
        raise ArmError("source validation sentinel has no authenticated selected games")
    component_ids = list(descriptor_meta.get("component_ids", ()))
    component_ratios = list(descriptor_meta.get("component_game_sampling_ratios", ()))
    if (
        len(component_ids) < 2
        or len(component_ids) != len(component_ratios)
        or not np.isclose(sum(float(value) for value in component_ratios), 1.0)
    ):
        raise ArmError("next validation sentinel component mixture is malformed")
    ratio_contract = {
        component_id: float(ratio)
        for component_id, ratio in zip(component_ids, component_ratios, strict=True)
    }
    if not output_path.exists():
        command = [
            python, str(repo / "tools/derive_validation_game_sentinel.py"),
            "--composite", str(descriptor), "--out", str(output_path),
            "--target-rows", str(source_payload["target_row_count"]),
            "--selection-seed", str(source_payload["selection_seed"]),
            "--validation-fraction", "0.05", "--validation-seed", "17",
            "--exclude-selected-games-from", str(source_ref["path"]),
            "--component-target-ratios-json", _canonical(ratio_contract).decode(),
        ]
        try:
            subprocess.run(command, cwd=repo, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise ArmError(f"cannot derive corrected validation sentinel: {error}") from error
    corrected, corrected_ref = _load_json(output_path)
    if (
        corrected.get("schema_version") != "train-validation-game-sentinel-v1"
        or corrected.get("source_composite_descriptor_file_sha256")
        != descriptor_meta["descriptor_file_sha256"]
        or corrected.get("source_composite_descriptor_fingerprint")
        != descriptor_meta["descriptor_fingerprint"]
        or corrected.get("selection_seed") != source_payload.get("selection_seed")
        or corrected.get("target_row_count") != source_payload.get("target_row_count")
    ):
        raise ArmError("next validation sentinel derivation contract drifted")
    if (
        not isinstance(corrected.get("selected_game_seed_set_sha256"), str)
        or not isinstance(corrected.get("excluded_game_seed_set_sha256"), str)
        or int(corrected.get("selected_game_seed_count", 0)) <= 0
        or int(corrected.get("selected_row_count", 0)) <= 0
    ):
        raise ArmError("next validation sentinel contains no authenticated selection")
    corrected_games = corrected.get("game_seeds")
    if not isinstance(corrected_games, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in corrected_games
    ):
        raise ArmError("next validation sentinel selected games are malformed")
    overlap = set(source_games).intersection(corrected_games)
    if overlap:
        raise ArmError(
            "next validation sentinel reuses games already consumed for model selection"
        )
    selected = set(corrected_games)
    component_rows: list[dict[str, Any]] = []
    assigned: set[int] = set()
    total_rows = 0
    assigned_target_rows = 0
    for component_index, component in enumerate(descriptor_meta.get("components", ())):
        component_id = component.get("component_id")
        if component_id not in ratio_contract:
            raise ArmError("next validation sentinel component identity drift")
        validation_payload, _ = _load_json(Path(str(component["validation_manifest"])))
        validation_games = validation_payload.get("game_seeds")
        if not isinstance(validation_games, list):
            raise ArmError(f"component {component_id} validation games are malformed")
        component_selected = selected.intersection(validation_games)
        if assigned.intersection(component_selected):
            raise ArmError("next validation sentinel games overlap components")
        assigned.update(component_selected)
        corpus_meta = component.get("corpus_meta")
        row_count = int(corpus_meta.get("row_count", 0)) if isinstance(corpus_meta, Mapping) else 0
        if row_count <= 0:
            raise ArmError(f"component {component_id} has no authenticated rows")
        corpus_dir = Path(str(component["corpus_dir"])).expanduser().resolve(strict=True)
        game_seeds = np.memmap(
            corpus_dir / "game_seed.dat", dtype=np.dtype("<i8"), mode="r", shape=(row_count,)
        )
        validation_array = np.asarray(sorted(set(validation_games)), dtype=np.int64)
        validation_row_counts: dict[int, int] = {}
        for start in range(0, row_count, 1 << 22):
            stop = min(start + (1 << 22), row_count)
            heldout = np.asarray(game_seeds[start:stop][
                np.isin(game_seeds[start:stop], validation_array)
            ], dtype=np.int64)
            unique, counts = np.unique(heldout, return_counts=True)
            for seed, count in zip(unique.tolist(), counts.tolist(), strict=True):
                validation_row_counts[seed] = validation_row_counts.get(seed, 0) + int(count)
        selected_rows = sum(validation_row_counts.get(seed, 0) for seed in component_selected)
        max_game_rows = max(validation_row_counts.values(), default=0)
        if component_index + 1 == len(component_ids):
            target_component_rows = int(corrected["target_row_count"]) - assigned_target_rows
        else:
            target_component_rows = int(round(
                int(corrected["target_row_count"]) * ratio_contract[component_id]
            ))
            assigned_target_rows += target_component_rows
        if max_game_rows <= 0 or abs(selected_rows - target_component_rows) > max_game_rows:
            raise ArmError(
                f"component {component_id} validation rows are not near its sealed target"
            )
        total_rows += selected_rows
        component_rows.append({
            "component_id": component_id,
            "target_row_ratio": ratio_contract[component_id],
            "target_row_count": target_component_rows,
            "selected_game_count": len(component_selected),
            "selected_row_count": selected_rows,
            "max_whole_game_row_count": max_game_rows,
        })
    if assigned != selected or total_rows != int(corrected["selected_row_count"]):
        raise ArmError("next validation sentinel component row accounting drift")
    predecessor = component_ids[-1]
    predecessor_row = component_rows[-1]
    if (
        predecessor not in {REPLAY_COMPONENT_ID, "predecessor_replay"}
        or predecessor_row["selected_game_count"] <= 0
        or predecessor_row["selected_row_count"] <= 0
        or not np.isclose(predecessor_row["target_row_ratio"], 0.2)
    ):
        raise ArmError("next validation sentinel lacks fresh predecessor coverage")
    independence = {
        "schema_version": "a1-validation-independence-contract-v1",
        "source_selected_game_seed_set_sha256": source_payload[
            "selected_game_seed_set_sha256"
        ],
        "fresh_selected_game_seed_set_sha256": corrected[
            "selected_game_seed_set_sha256"
        ],
        "selection_overlap_game_count": 0,
        "selection_scope": "fresh_whole_games_stratified_to_winning_operator",
        "component_rows": component_rows,
        "predecessor_component_id": predecessor,
        "predecessor_target_row_ratio": 0.2,
        "complete_component_holdouts_remain_training_excluded": True,
    }
    independence["contract_sha256"] = _digest(independence)
    return corrected, corrected_ref, source_ref, independence


def _lineage(entries: Sequence[str]) -> dict[str, Any]:
    parsed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        role, separator, raw_path = entry.partition("=")
        if not separator or role not in LINEAGE_ROLES or role in parsed:
            raise ArmError("lineage entries must uniquely bind ROLE=PATH for all required roles")
        payload, ref = _load_json(Path(raw_path))
        digest_field = LINEAGE_DIGEST_FIELDS[role]
        stated = payload.get(digest_field)
        unhashed = {key: value for key, value in payload.items() if key != digest_field}
        if stated != _digest(unhashed):
            raise ArmError(f"{role} semantic digest is invalid")
        parsed[role] = {"file": ref, "schema_version": payload["schema_version"]}
    if tuple(role for role in LINEAGE_ROLES if role not in parsed):
        missing = [role for role in LINEAGE_ROLES if role not in parsed]
        raise ArmError(f"failed-retry lineage is incomplete: {missing}")
    ordered = [{"role": role, **parsed[role]} for role in LINEAGE_ROLES]
    return {"artifacts": ordered, "lineage_sha256": _digest(ordered)}


def _source_binding(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repo, text=True
        ).strip()
        subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", *SOURCE_FILES),
            cwd=repo,
            check=True,
        )
        for relative in SOURCE_FILES:
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", relative),
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArmError("corrected arm code must be clean, tracked canonical Git bytes") from error
    files = {relative: _file_ref(repo / relative) for relative in SOURCE_FILES}
    return {"repository_root": str(repo), "git_commit": commit, "files": files,
            "files_sha256": _digest(files)}


def _rebind_a1_metadata(command: list[str], repo: Path) -> dict[str, Any]:
    """Rebind effective recipe and the prior reviewed runtime closure."""
    required = (
        "--a1-learner-ablation-id",
        "--a1-effective-learner-recipe-json",
        "--a1-effective-learner-recipe-sha256",
        "--a1-ablation-code-binding-json",
        "--a1-ablation-code-tree-sha256",
        "--a1-reviewed-lock-file-sha256",
    )
    present = [flag in command for flag in required]
    if not any(present):
        return {
            "mode": "plain-authenticated-composite-diagnostic",
            "effective_recipe": None,
            "code_binding": _source_binding(repo),
        }
    if not all(present):
        raise ArmError("source command lacks sealed A1 effective-recipe/code metadata")
    try:
        effective = json.loads(_option(command, "--a1-effective-learner-recipe-json"))
        prior_binding = json.loads(_option(command, "--a1-ablation-code-binding-json"))
    except json.JSONDecodeError as error:
        raise ArmError(f"source A1 metadata is invalid JSON: {error}") from error
    if not isinstance(effective, dict) or not isinstance(prior_binding, dict):
        raise ArmError("source A1 metadata is not object-valued")
    recipe_updates: dict[str, Any] = {
        "batch_size": 512, "grad_accum_steps": 1, "global_batch_size": 4096,
        "world_size": 8, "max_steps": 1024, "epochs": 1,
        "loser_sample_weight": 1.0, "winner_sample_weight": 1.0,
        "forced_action_weight": 0.0, "forced_row_value_weight": 1.0,
        "policy_loss_weight": 1.0, "soft_target_source": "policy",
        "soft_target_weight": 0.9, "soft_target_temperature": 0.7,
        "soft_target_min_legal_coverage": 0.5,
        "policy_aux_active_batch_size": 0,
        "policy_kl_anchor_weight": REPLAY_ANCHOR_WEIGHT,
        "value_loss_weight": 0.25, "value_lr_mult": 0.3,
        "value_target_lambda": 1.0, "lr": 3e-5,
        "lr_warmup_steps": 100, "lr_schedule": "flat",
    }
    for key in set(recipe_updates) - {"policy_aux_active_batch_size"}:
        if key not in effective:
            raise ArmError(f"source effective recipe omits required field {key}")
    effective.update(recipe_updates)
    records = prior_binding.get("records")
    if not isinstance(records, list) or not records:
        raise ArmError("source A1 code binding has no reviewed runtime closure")
    rebound = []
    relative_paths = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("relative_path"), str):
            raise ArmError("source A1 code-binding record is malformed")
        relative = record["relative_path"]
        if relative == "tools/a1_shared_trunk_gradient_probe.py":
            raise ArmError("training runtime must not include the untracked gradient probe")
        path = (repo / relative).resolve(strict=True)
        relative_paths.append(relative)
        rebound.append(
            {"kind": str(record.get("kind", "runtime_code")),
             "relative_path": relative, "path": str(path), "sha256": _file_sha(path)}
        )
    try:
        subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", *relative_paths),
            cwd=repo, check=True,
        )
        for relative in relative_paths:
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", relative), cwd=repo,
                check=True, stdout=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArmError("reviewed runtime closure is not clean and tracked") from error
    binding = {
        "schema_version": "a1-learner-ablation-code-binding-v1",
        "repository_root": str(repo),
        "records": rebound,
    }
    code_sha = _digest(binding)
    binding["code_tree_sha256"] = code_sha
    _set_option(command, "--a1-learner-ablation-id", "next-winning-operator-control")
    _set_option(command, "--a1-effective-learner-recipe-json", _canonical(effective).decode())
    _set_option(command, "--a1-effective-learner-recipe-sha256", _digest(effective))
    _set_option(command, "--a1-ablation-code-binding-json", _canonical(binding).decode())
    _set_option(command, "--a1-ablation-code-tree-sha256", code_sha)
    return {"effective_recipe": effective, "code_binding": binding}


def _derive_command(
    source: Sequence[str], *, repo: Path, descriptor: Path, sentinel: Path,
    parent: Path, output_root: Path,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    command = list(source)
    if "torch.distributed.run" not in command or not any(
        item in {"--nproc-per-node=8", "--nproc_per_node=8"} for item in command
    ):
        raise ArmError("source command must use exactly eight torchrun ranks")
    trainer_positions = [
        index for index, item in enumerate(command) if Path(item).name == "train_bc.py"
    ]
    if len(trainer_positions) != 1:
        raise ArmError("source command must name exactly one train_bc.py")
    command[trainer_positions[0]] = str(repo / "tools/train_bc.py")
    required_flags = ("--no-resume-optimizer", "--mask-hidden-info")
    if any(flag not in command for flag in required_flags):
        raise ArmError(f"source command is missing a required safety flag: {required_flags}")
    updates = {
        "--data": str(descriptor),
        "--validation-game-sentinel-manifest": str(sentinel),
        "--init-checkpoint": str(parent),
        "--checkpoint": str(output_root / "candidate.pt"),
        "--report": str(output_root / "train.report.json"),
        "--batch-size": "512",
        "--grad-accum-steps": "1",
        "--max-steps": "1024",
        "--epochs": "1",
        "--loser-sample-weight": "1.0",
        "--winner-sample-weight": "1.0",
        "--forced-action-weight": "0.0",
        "--forced-row-value-weight": "1.0",
        "--policy-loss-weight": "1.0",
        "--soft-target-source": "policy",
        "--soft-target-weight": "0.9",
        "--soft-target-temperature": "0.7",
        "--soft-target-min-legal-coverage": "0.5",
        "--policy-aux-active-batch-size": "0",
        "--policy-kl-anchor-direction": "forward",
        "--policy-kl-anchor-weight": str(REPLAY_ANCHOR_WEIGHT),
        "--value-loss-weight": "0.25",
        "--value-lr-mult": "0.3",
        "--value-target-lambda": "1.0",
        "--lr": "3e-5",
        "--lr-warmup-steps": "100",
        "--lr-schedule": "flat",
    }
    before = {flag: _option(command, flag) if flag in command else "<absent>" for flag in updates}
    for flag, value in updates.items():
        _set_option(command, flag, value)
    if "--validation-game-seed-manifest" in command:
        raise ArmError("source command mixes seed-manifest and sentinel validation controls")
    changes = {
        flag: {"source": before[flag], "corrected": value}
        for flag, value in updates.items()
        if before[flag] != value
    }
    return command, changes


def _next_supervision_contract(
    descriptor_meta: Mapping[str, Any], command: Sequence[str],
    policy_active_dose: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the exact winning supervision operator."""

    component_ids = list(descriptor_meta.get("component_ids", ()))
    ratios = list(descriptor_meta.get("component_game_sampling_ratios", ()))
    if (
        len(component_ids) < 2
        or len(set(component_ids)) != len(component_ids)
        or not all(component in CURRENT_TEACHER_COMPONENT_IDS for component in component_ids[:-1])
        or component_ids[-1] not in {REPLAY_COMPONENT_ID, "predecessor_replay"}
        or not np.isclose(sum(ratios[:-1]), 0.8)
        or not np.isclose(ratios[-1], 0.2)
    ):
        raise ArmError("next learner winning component identity/ratio contract drift")
    if (
        descriptor_meta.get("policy_distillation_scope_explicit") is not True
        or descriptor_meta.get("value_training_scope_explicit") is not True
        or descriptor_meta.get("policy_distillation_component_ids") != component_ids
        or descriptor_meta.get("value_training_component_ids") != component_ids
        or descriptor_meta.get("policy_kl_anchor_component_ids") != component_ids
    ):
        raise ArmError("next learner must supervise the exact winning component mix")
    exact_options = {
        "--soft-target-source": "policy",
        "--soft-target-weight": "0.9",
        "--policy-aux-active-batch-size": "0",
        "--policy-kl-anchor-direction": "forward",
        "--policy-kl-anchor-weight": str(REPLAY_ANCHOR_WEIGHT),
        "--winner-sample-weight": "1.0",
        "--loser-sample-weight": "1.0",
    }
    observed = {flag: _option(command, flag) for flag in exact_options}
    if observed != exact_options:
        raise ArmError(f"next learner command supervision drift: {observed}")
    contract = {
        "schema_version": SUPERVISION_CONTRACT_SCHEMA,
        "component_ids": component_ids,
        "component_roles": [
            *[
                {"component_id": component, "role": "current_teacher"}
                for component in component_ids[:-1]
            ],
            {"component_id": component_ids[-1], "role": "immediate_predecessor_replay"},
        ],
        "component_game_sampling_ratios": ratios,
        "policy_distillation_component_ids": component_ids,
        "value_training_component_ids": component_ids,
        "replay_component_id": component_ids[-1],
        "replay_sampling_ratio": ratios[-1],
        "replay_objective": "supervised_policy_and_value_exact_winning_operator",
        "replay_forward_kl_weight": 0.0,
        "soft_target_source": "policy",
        "soft_target_weight": 0.9,
        "policy_aux_active_batch_size_per_rank": 0,
        "policy_active_row_dose": dict(policy_active_dose),
        "outcome_conditioned_policy_weighting": False,
        "expected_train_report_provenance": {
            "policy_distillation_scope.component_ids": component_ids,
            "value_training_scope.component_ids": component_ids,
            "memmap_composite.policy_kl_anchor_component_ids": component_ids,
            "soft_target_weight": 0.9,
        },
    }
    contract["contract_sha256"] = _digest(contract)
    return contract


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    repo = args.repo.expanduser().resolve(strict=True)
    source, source_ref = _load_source_receipt(args.source_receipt)
    source_descriptor = args.source_descriptor.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    descriptor = output_root / "corrected-anchor-memmap-composite.json"
    descriptor_meta, descriptor_ref, source_descriptor_ref = _build_corrected_descriptor(
        source_descriptor, descriptor
    )
    policy_active_dose = _derive_policy_active_dose(descriptor_meta)
    parent_checkpoint = getattr(args, "parent_checkpoint", None)
    expected_parent_sha256 = getattr(args, "expected_parent_sha256", None)
    legacy_checkpoint = getattr(args, "f7_checkpoint", None)
    legacy_sha256 = getattr(args, "expected_f7_sha256", None)
    new_mode = parent_checkpoint is not None or expected_parent_sha256 is not None
    legacy_mode = legacy_checkpoint is not None or legacy_sha256 is not None
    if new_mode == legacy_mode:
        raise ArmError(
            "select exactly one parent interface: current --parent-* or legacy --f7-*"
        )
    expected_predecessor_sha256: str | None = None
    if new_mode:
        if parent_checkpoint is None or expected_parent_sha256 is None:
            raise ArmError("current parent checkpoint and digest must be supplied together")
        handoff_path = getattr(args, "post_promotion_handoff", None)
        if handoff_path is None:
            raise ArmError("current parent requires --post-promotion-handoff")
        from tools import a1_flywheel_turn as flywheel

        handoff_path = Path(handoff_path).expanduser().resolve(strict=True)
        try:
            (
                handoff_payload,
                promotion_receipt,
                handoff_ref,
                _promotion_receipt_ref,
            ) = flywheel.verify_current_parent_handoff(handoff_path)
        except flywheel.FlywheelTurnError as error:
            raise ArmError(f"current parent handoff replay failed: {error}") from error
        dethroned = promotion_receipt.get("champion")
        if not isinstance(dethroned, Mapping) or not isinstance(
            dethroned.get("sha256"), str
        ):
            raise ArmError("promotion receipt does not bind the dethroned champion")
        expected_predecessor_sha256 = str(dethroned["sha256"])
        parent_lineage = {
            "mode": "post_promotion_current_parent",
            "handoff": handoff_ref,
            "handoff_sha256": handoff_payload["handoff_sha256"],
            "registry_version": handoff_payload["registry_after"]["version"],
            "dethroned_champion_sha256": expected_predecessor_sha256,
        }
    else:
        if legacy_checkpoint is None or legacy_sha256 is None:
            raise ArmError("legacy f7 checkpoint and digest must be supplied together")
        parent_checkpoint = legacy_checkpoint
        expected_parent_sha256 = legacy_sha256
        parent_lineage = {"mode": "historical_f7_cli_compatibility"}
    parent_ref = _file_ref(Path(parent_checkpoint))
    if parent_ref["sha256"] != expected_parent_sha256:
        raise ArmError("initialization checkpoint differs from the expected parent bytes")
    if new_mode and handoff_payload["producer_identity"]["checkpoint"] != parent_ref:
        raise ArmError("current learner parent differs from promoted generator identity")
    teacher_lineage = _bind_teacher_lineage(
        descriptor_meta,
        parent_checkpoint_sha256=parent_ref["sha256"],
        expected_predecessor_sha256=expected_predecessor_sha256,
    )
    source_identities = {
        "parent_checkpoint_sha256": parent_ref["sha256"],
        "descriptor_sha256": source_descriptor_ref["sha256"],
    }
    for field, expected in source_identities.items():
        if source.get(field) != expected:
            raise ArmError(f"source receipt does not reuse exact {field} identity")
    lineage = _lineage(args.failed_lineage_artifact)
    source_binding = _source_binding(repo)
    source_descriptor_meta = _preflight_descriptor(source_descriptor)[0]
    (
        sentinel_payload,
        sentinel_ref,
        source_sentinel_ref,
        validation_independence_contract,
    ) = _build_corrected_sentinel(
        source_receipt=source,
        source_descriptor=source_descriptor_meta,
        descriptor=descriptor,
        descriptor_meta=descriptor_meta,
        output_path=output_root / "validation.sentinel.json",
        python=str(source["command"][0]),
        repo=repo,
    )
    for name in ("candidate.pt", "candidate.pt.optimizer.pt", "train.report.json"):
        if (output_root / name).exists():
            raise ArmError(f"refusing existing corrected-arm output: {output_root / name}")
    command, changes = _derive_command(
        source["command"], repo=repo, descriptor=descriptor,
        sentinel=Path(sentinel_ref["path"]), parent=Path(parent_ref["path"]),
        output_root=output_root,
    )
    event_history_contract, event_history_changes = (
        _bind_event_history_training_command(command, descriptor_meta)
    )
    changes.update(event_history_changes)
    a1_metadata = _rebind_a1_metadata(command, repo)
    supervision_contract = _next_supervision_contract(
        descriptor_meta, command, policy_active_dose
    )
    component_ids = list(descriptor_meta["component_ids"])
    component_ratios = list(descriptor_meta["component_game_sampling_ratios"])
    expected_policy_base_active_rows = int(
        policy_active_dose["reference_base_active_rows"]
    )
    recipe = {
        "world_size": 8, "local_batch_size": 512, "global_batch_size": 4096,
        "steps": 1024, "base_value_row_dose": GLOBAL_ROW_DOSE,
        "policy_aux_active_batch_size_per_rank": 0,
        "policy_aux_active_row_dose": 0,
        "expected_policy_base_active_rows": expected_policy_base_active_rows,
        "policy_base_active_row_tolerance": POLICY_BASE_ACTIVE_ROW_TOLERANCE,
        "expected_policy_aux_active_rows": EXPECTED_POLICY_AUX_ACTIVE_ROWS,
        "policy_distillation_component_ids": component_ids,
        "value_training_component_ids": component_ids,
        "component_game_sampling_ratios": component_ratios,
        "current_supervised_base_row_dose": GLOBAL_ROW_DOSE * 0.8,
        "replay_supervised_base_row_dose": GLOBAL_ROW_DOSE * 0.2,
        "replay_anchor_base_row_dose": 0.0,
        "replay_supervised_policy": True,
        "replay_supervised_value": True,
        "replay_forward_kl_weight": REPLAY_ANCHOR_WEIGHT,
        "soft_target_weight": 0.9, "fresh_optimizer": True,
        "independent_parent_initialization": True,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "diagnostic_execution_authorized": True,
        "launch_interface_present": "tools/a1_corrected_policy_arm_execute.py --go",
        "source_receipt": source_ref,
        "failed_retry_lineage": lineage,
        "source_descriptor": source_descriptor_ref,
        "descriptor": descriptor_ref,
        "descriptor_fingerprint": descriptor_meta["descriptor_fingerprint"],
        "source_validation_sentinel": source_sentinel_ref,
        "validation_sentinel": sentinel_ref,
        "validation_sentinel_selection_sha256": sentinel_payload[
            "selected_game_seed_set_sha256"
        ],
        "validation_independence_contract": validation_independence_contract,
        "initialization": parent_ref,
        "evaluation_baseline": parent_ref,
        "parent_lineage": parent_lineage,
        "teacher_lineage": teacher_lineage,
        "source_binding": source_binding,
        "a1_runtime_metadata": a1_metadata,
        "event_history_training_contract": event_history_contract,
        "supervision_contract": supervision_contract,
        "recipe": recipe,
        "recipe_sha256": _digest(recipe),
        "causal_interpretation": {
            "exact_winning_operator_control": True,
            "bundled_optimization_not_parent_replication": False,
            "current_teacher_prior_is_initializer": True,
            "teacher_gap_closure_is_search_improvement_over_initializer": True,
            "reason": (
                "preserve the realized r3 supervision operator before changing one "
                "causal axis at a time"
            ),
        },
        "diagnostic_ablations": [
            {
                "arm_id": "CURRENT_ONLY_REPLAY_ABLATION",
                "single_delta": (
                    "remove predecessor replay and renormalize authenticated current "
                    "teacher weights to sum to one"
                ),
                "soft_target_weight": 0.9,
                "production_default": False,
            },
            {
                "arm_id": "PURE_SEARCH_TARGET_ABLATION",
                "single_delta": "soft_target_weight 0.9 -> 1.0",
                "component_game_sampling_ratios": component_ratios,
                "production_default": False,
            },
        ],
        "allowlisted_command_changes": changes,
        "command": command,
        "command_sha256": _digest(command),
    }
    manifest["manifest_sha256"] = _digest(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "corrected-policy-arm.manifest.json"
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
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-descriptor", required=True, type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--expected-parent-sha256")
    parser.add_argument("--post-promotion-handoff", type=Path)
    # Historical interface remains parseable solely to replay issued manifests.
    parser.add_argument("--f7-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-f7-sha256", help=argparse.SUPPRESS)
    parser.add_argument(
        "--failed-lineage-artifact", action="append", default=[],
        help="Repeat ROLE=PATH for parent claim/receipt and retry contract/receipt.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo", default=REPO_ROOT, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    manifest, path = prepare(build_parser().parse_args(argv))
    print(json.dumps({"prepared": str(path), "launched": False,
                      "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
