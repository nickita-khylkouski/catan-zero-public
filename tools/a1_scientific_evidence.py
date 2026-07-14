#!/usr/bin/env python3
"""Produce measured evidence for the central A1 learner transaction.

The coordinator is an append-only authority/verifier, not a data scanner.  This
module is the sole producer for facts that must be measured from a live learner,
authenticated composite rows, or a checkpoint.  Receipts bind this file's own
digest so hand-authored JSON cannot impersonate a measurement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import train_bc  # noqa: E402
from tools.fleet import a1_production_executor as production_executor  # noqa: E402


SCHEMA_RUNTIME = "a1-b200-learner-runtime-admission-v1"
SCHEMA_SAMPLE = "a1-authenticated-sample-evidence-v2"
SCHEMA_ROUTING = "a1-mixed-component-routing-authority-v3"
SCHEMA_INITIALIZER_ZERO = "a1-initializer-slot12-zero-evidence-v1"
SCHEMA_TRAINED_DELTA = "a1-trained-model-slot12-delta-evidence-v1"
SCHEMA_PUBLIC_AWARD_TRANSITION = (
    "a1-public-award-initializer-transition-evidence-v1"
)
SAMPLER_ALGORITHM = (
    "numpy-pcg64-component-uniform-game-uniform-row-replacement-v1"
)
CHUNK_ROWS = 16_384
WORLD_SIZE = 8
SHORT_SAMPLE_DOSE = 524_288
B200_LEARNER_HOST_ID = "b200-learner"
B200_LEARNER_HOSTNAME = "149-118-65-110"
B200_LEARNER_MACHINE_ID = "e71d46177526e026a826ec4afcd39d70"
B200_LEARNER_GPU_UUIDS = (
    "GPU-c444a2e6-e5e4-0974-4144-9807e6f7d68a",
    "GPU-a6f349ce-a4fb-3291-d268-d4950107751f",
    "GPU-7998a0f4-43e2-c0f7-bf7c-b2999fdd63c8",
    "GPU-80857176-d21e-4b06-ae6c-9821f4395c52",
    "GPU-1971a2ec-2a2d-1120-2f4a-72a154703794",
    "GPU-82c0cf1c-c4e2-67ff-c0fa-dcd89cc3967f",
    "GPU-7ea809bd-52f9-9422-d69c-5bb446186cbe",
    "GPU-7c0a25f9-9c0a-fa76-99ec-1a1fca4d8bea",
)
COMPONENT_IDS = (
    "current_producer",
    "recent_history",
    "hard_negative",
    "historical_replay",
)
class EvidenceError(RuntimeError):
    """Refusal to emit evidence that was not measured exactly."""


def verify_recovery_component_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact sealed descriptor map, never a renamed projection."""

    try:
        result = train_bc._validate_flywheel_category_semantics(  # noqa: SLF001
            dict(value), where="scientific recovery composite"
        )
    except SystemExit as error:
        raise EvidenceError(str(error)) from error
    recent = result["recent_history"]
    if (
        recent.get("semantic") != "recovery_reference"
        or recent.get("relation") != "safety_reference_unproven_predecessor"
        or recent.get("causal_parent_proven") is not False
        or recent.get("promotion_proof_recreated") is not False
    ):
        raise EvidenceError(
            "scientific composite recent_history is not the sealed recovery safety reference"
        )
    if (
        result["current_producer"]["checkpoint"]["sha256"]
        == recent["checkpoint"]["sha256"]
    ):
        raise EvidenceError("recovery and safety-reference checkpoints are identical")
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"evidence is not canonical JSON: {error}") from error


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def canonical_row_identity(
    *,
    payload_member_sha256: str,
    row_offset: int,
    component_id: str,
    prior_policy_present: bool,
    legal_action_count: int,
) -> str:
    if component_id not in COMPONENT_IDS:
        raise EvidenceError("sampled row component drift")
    return _digest(
        {
            "schema_version": "a1-p1-kl-row-identity-v1",
            "payload_member_sha256": payload_member_sha256,
            "row_offset": row_offset,
            "component_id": component_id,
            "prior_policy_present": prior_policy_present,
            "legal_action_count": legal_action_count,
        }
    )


def _ordered_identity_update(
    digest: Any, *, index: int, row_identity_sha256: str
) -> None:
    digest.update(str(index).encode("ascii"))
    digest.update(b"\0")
    digest.update(row_identity_sha256.encode("ascii"))
    digest.update(b"\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def origin_tool_sha256() -> str:
    return _file_sha256(Path(__file__).resolve(strict=True))


def _sealed(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if "state_sha256" in value:
        raise EvidenceError("caller may not prepopulate state_sha256")
    value["state_sha256"] = _digest(value)
    return value


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.urandom(8).hex()}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_immutable(temporary, path)


def _publish_immutable(temporary: Path, destination: Path) -> None:
    """Publish without overwrite; exact existing bytes are idempotent."""

    try:
        os.link(temporary, destination)
        os.chmod(destination, 0o444)
    except FileExistsError:
        metadata = destination.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise EvidenceError(
                f"authoritative evidence destination is not a regular file: {destination}"
            )
        if _file_sha256(temporary) != _file_sha256(destination):
            raise EvidenceError(
                f"authoritative evidence already exists with different bytes: {destination}"
            )
        os.chmod(destination, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _load_sealed(
    path: Path, *, where: str, expected_origin_tool_sha256: str
) -> dict[str, Any]:
    _require_sha(expected_origin_tool_sha256, where=f"{where} expected producer")
    path = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{where} must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot load {where}: {error}") from error
    if not isinstance(payload, dict):
        raise EvidenceError(f"{where} is not an object")
    unsigned = dict(payload)
    stated = unsigned.pop("state_sha256", None)
    if stated != _digest(unsigned):
        raise EvidenceError(f"{where} state digest drift")
    if payload.get("origin_tool_sha256") != expected_origin_tool_sha256:
        raise EvidenceError(f"{where} origin tool digest drift")
    return payload


def _require_exact_keys(
    value: Any, expected: set[str], *, where: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{where} is not an object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise EvidenceError(f"{where} keys drift: missing={missing}, extra={extra}")
    return value


def _require_sha(value: Any, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
    ):
        raise EvidenceError(f"{where} is not a sha256 identity")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise EvidenceError(f"{where} is not a sha256 identity") from error
    return value


def _run(argv: list[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            argv,
            cwd=None if cwd is None else str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError) and error.stderr
            else str(error)
        )
        raise EvidenceError(f"probe failed ({argv[0]}): {detail}") from error
    return result.stdout.strip()


def _checkout_tree_sha256(repo_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
    ).stdout
    if status:
        raise EvidenceError(
            "scientific runtime admission requires a completely clean checkout"
        )
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    tree = _run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root)
    raw = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
    ).stdout
    paths = [Path(value.decode("utf-8")) for value in raw.split(b"\0") if value]
    records = []
    for relative in sorted(paths):
        path = (repo_root / relative).resolve(strict=True)
        if not path.is_file():
            raise EvidenceError(f"tracked checkout member is not a file: {relative}")
        records.append(
            {"path": relative.as_posix(), "sha256": _file_sha256(path)}
        )
    return _digest(
        {
            "schema_version": "a1-clean-checkout-identity-v1",
            "commit": commit,
            "tree": tree,
            "tracked_files": records,
        }
    )


def _local_runtime_report(repo_root: Path) -> dict[str, Any]:
    hostname = socket.gethostname()
    machine_id = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
    rows = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    parsed: list[tuple[int, str, str, str]] = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 4:
            raise EvidenceError("nvidia-smi learner inventory shape drift")
        parsed.append((int(fields[0]), fields[1], fields[2], fields[3]))
    parsed.sort()
    if [row[0] for row in parsed] != list(range(WORLD_SIZE)):
        raise EvidenceError("learner does not expose exact GPU indices 0..7")
    import catanatron_rs
    import torch

    host_key_candidates = sorted(Path("/etc/ssh").glob("ssh_host_*_key.pub"))
    if not host_key_candidates:
        raise EvidenceError("learner has no readable SSH host public key")
    nofile_soft, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return {
        "host_id": B200_LEARNER_HOST_ID,
        "hostname": hostname,
        "machine_id": machine_id,
        "ssh_host_key_sha256": _file_sha256(host_key_candidates[0]),
        "checkout_tree_sha256": _checkout_tree_sha256(repo_root),
        "tool_sha256": origin_tool_sha256(),
        "gpu_indices": [row[0] for row in parsed],
        "gpu_names": [row[1] for row in parsed],
        "gpu_uuids": [row[2] for row in parsed],
        "pci_bus_ids": [row[3] for row in parsed],
        "python": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "catanatron_rs_version": str(catanatron_rs.__version__),
        "native_wheel_sha256": production_executor._native_wheel_release_identity()[
            "sha256"
        ],
        "native_mcts_capabilities": sorted(
            production_executor.NATIVE_REQUIRED_CAPABILITIES
        ),
        "nofile_soft": int(nofile_soft),
        "nofile_hard": int(nofile_hard),
    }


def build_runtime_admission_receipt(
    *, repo_root: Path = _REPO_ROOT,
) -> dict[str, Any]:
    measured = _local_runtime_report(repo_root)
    if measured.get("tool_sha256") != origin_tool_sha256():
        raise EvidenceError("runtime report does not bind this producer")
    return _sealed(
        {
            "schema_version": SCHEMA_RUNTIME,
            "status": "complete",
            "hosts": {B200_LEARNER_HOST_ID: measured},
            "origin_tool_sha256": origin_tool_sha256(),
        }
    )


def _load_composite(descriptor: Path):
    descriptor = descriptor.expanduser().resolve(strict=True)
    authenticated = train_bc._preflight_memmap_composite_descriptor(descriptor)  # noqa: SLF001
    if (
        authenticated.get("schema_version") != "memmap_composite_v2"
        or authenticated.get("diagnostic_only") is not False
        or authenticated.get("promotion_eligible") is not True
    ):
        raise EvidenceError("scientific evidence requires the production v2 composite")
    data = train_bc.load_teacher_data_memmap(
        descriptor, composite_meta=authenticated
    )
    if tuple(data.component_ids) != COMPONENT_IDS:
        raise EvidenceError("production composite component order drift")
    semantics = authenticated.get("category_semantics")
    source_authority = authenticated.get("source_authority_ref")
    checked_semantics = verify_recovery_component_semantics(semantics or {})
    if (
        authenticated.get("category_semantics_sha256") != _digest(checked_semantics)
        or not isinstance(source_authority, dict)
        or set(source_authority) != {"path", "file_sha256", "authority_sha256"}
    ):
        raise EvidenceError(
            "scientific composite lacks exact recovery semantics/source authority"
        )
    for field in ("file_sha256", "authority_sha256"):
        _require_sha(source_authority[field], where=f"source authority {field}")
    return descriptor, authenticated, data


def _assert_composite_stable(
    descriptor: Path, authenticated: Mapping[str, Any]
) -> None:
    replay = train_bc._preflight_memmap_composite_descriptor(descriptor)  # noqa: SLF001
    if replay != authenticated:
        raise EvidenceError("composite descriptor or payload inventory changed during scan")


def build_mixed_routing_receipt(descriptor: Path) -> dict[str, Any]:
    descriptor, authenticated, data = _load_composite(descriptor)
    component_counts: dict[str, int] = {}
    legacy_nonzero = 0
    legacy_digest = hashlib.sha256()
    authoritative_digest = hashlib.sha256()
    ordered_digest = hashlib.sha256()
    for component, (component_id, corpus) in enumerate(
        zip(data.component_ids, data.corpora, strict=True)
    ):
        rows = int(corpus.row_count)
        component_counts[component_id] = rows
        positive = 0
        for start in range(0, rows, CHUNK_ROWS):
            stop = min(rows, start + CHUNK_ROWS)
            tokens = np.asarray(corpus["player_tokens"][start:stop])
            if tokens.ndim != 3 or tokens.shape[-1] <= 12:
                raise EvidenceError(f"{component_id} player-token shape drift")
            slot = np.asarray(tokens[..., 12])
            if not np.isfinite(slot).all() or not np.isin(slot, (0, 1)).all():
                raise EvidenceError(f"{component_id} slot12 is not finite binary")
            nonzero = int(np.count_nonzero(slot))
            positive += nonzero
            chunk_identity = _canonical_bytes(
                {
                    "component_index": component,
                    "component_id": component_id,
                    "row_start": start,
                    "row_stop": stop,
                    "shape": list(slot.shape),
                    "dtype": slot.dtype.str,
                }
            ) + b"\0" + np.ascontiguousarray(slot).tobytes()
            ordered_digest.update(chunk_identity)
            if component_id == "historical_replay":
                legacy_digest.update(chunk_identity)
            else:
                authoritative_digest.update(chunk_identity)
        if component_id == "historical_replay":
            legacy_nonzero += positive
        elif positive == 0:
            raise EvidenceError(
                f"authoritative component {component_id} has no positive slot12 support"
            )
    if legacy_nonzero != 0:
        raise EvidenceError("legacy replay contains nonzero public-award slot12")
    _assert_composite_stable(descriptor, authenticated)
    routes = {
        "current_producer": "authoritative_v1",
        "recent_history": "authoritative_v1",
        "hard_negative": "authoritative_v1",
        "historical_replay": "legacy_zero_v0",
    }
    return _sealed(
        {
            "schema_version": SCHEMA_ROUTING,
            "status": "complete",
            "descriptor_sha256": authenticated["descriptor_file_sha256"],
            "payload_inventory_sha256": authenticated[
                "payload_inventory_sha256"
            ],
            "category_semantics": authenticated["category_semantics"],
            "category_semantics_sha256": authenticated[
                "category_semantics_sha256"
            ],
            "source_authority": authenticated["source_authority_ref"],
            "component_ids": list(data.component_ids),
            "component_routes": routes,
            "component_row_counts": component_counts,
            "legacy_slot12_nonzero_count": 0,
            "legacy_slot12_all_zero": True,
            "legacy_slot12_evidence_sha256": "sha256:"
            + legacy_digest.hexdigest(),
            "authoritative_slot12_evidence_sha256": "sha256:"
            + authoritative_digest.hexdigest(),
            "ordered_row_routing_evidence_sha256": "sha256:"
            + ordered_digest.hexdigest(),
            "per_row_component_authenticated": True,
            "mixed_authoritative_transition_approved": True,
            "model_slot12_zero_initialization_required": True,
            "origin_tool_sha256": origin_tool_sha256(),
        }
    )


def _training_indices(data) -> np.ndarray:
    split = train_bc.split_train_validation_indices(
        data,
        validation_fraction=0.05,
        validation_seed=17,
        validation_max_samples=0,
    )
    return np.asarray(split["train"], dtype=np.int64)


def _row_set_sha256(row_identities: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for identity in sorted(set(row_identities)):
        digest.update(identity.encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _load_prior_rows(
    path: Path | None,
) -> tuple[set[str], dict[str, set[str]], str | None, str | None]:
    all_rows: set[str] = set()
    by_component = {component: set() for component in COMPONENT_IDS}
    if path is None:
        return all_rows, by_component, None, None
    resolved = path.expanduser().resolve(strict=True)
    before_sha256 = _file_sha256(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvidenceError(
                    f"prior sample row {line_number} is invalid JSON"
                ) from error
            identity = row.get("row_identity_sha256")
            component = row.get("component_id")
            if (
                not isinstance(identity, str)
                or component not in by_component
            ):
                raise EvidenceError(f"prior sample row {line_number} shape drift")
            all_rows.add(identity)
            by_component[component].add(identity)
    after_sha256 = _file_sha256(resolved)
    if before_sha256 != after_sha256:
        raise EvidenceError("prior sample row bytes changed during scan")
    return all_rows, by_component, after_sha256, _row_set_sha256(all_rows)


def _expected_unique_overlap(
    probabilities: np.ndarray, draws: int
) -> float:
    """Expected shared unique rows for two iid equal-dose replacement samples."""

    values = np.asarray(probabilities, dtype=np.float64)
    q = -np.expm1(float(draws) * np.log1p(-values))
    return float(np.square(q).sum(dtype=np.float64))


def build_sample_evidence(
    descriptor: Path,
    *,
    sampler_seed: int,
    sample_dose: int,
    rows_path: Path,
    prior_rows_path: Path | None = None,
) -> dict[str, Any]:
    descriptor, authenticated, data = _load_composite(descriptor)
    if isinstance(sampler_seed, bool) or sampler_seed < 0:
        raise EvidenceError("sampler seed must be a non-negative integer")
    if sample_dose != SHORT_SAMPLE_DOSE:
        raise EvidenceError("central learner sample dose must be exactly 524288")
    train_indices = _training_indices(data)
    weights = train_bc._composite_game_sampling_weights(  # noqa: SLF001
        data, train_indices
    )
    if weights is None:
        raise EvidenceError("production composite lacks game-uniform sampler weights")
    rng = np.random.default_rng(sampler_seed)
    positions = rng.choice(
        len(train_indices),
        size=sample_dose,
        replace=True,
        p=weights / float(weights.sum()),
    )
    sampled_rows = train_indices[np.asarray(positions, dtype=np.int64)]
    component_indices = np.asarray(
        data.component_indices_for_rows(sampled_rows), dtype=np.int64
    )
    offsets = np.asarray(data.component_offsets, dtype=np.int64)
    component_inventory = {
        component["component_id"]: component["payload_inventory_sha256"]
        for component in authenticated["components"]
    }
    order_digest = hashlib.sha256()
    evidence_digest = hashlib.sha256()
    eligible_digest = hashlib.sha256()
    identities: list[str] = []
    identities_by_component = {
        component: [] for component in COMPONENT_IDS
    }
    eligible = 0
    rows_path = rows_path.expanduser().resolve(strict=False)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = rows_path.with_name(
        f".{rows_path.name}.tmp.{os.getpid()}.{os.urandom(8).hex()}"
    )
    descriptor_fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor_fd, "wb", closefd=True) as handle:
        for start in range(0, sample_dose, CHUNK_ROWS):
            stop = min(sample_dose, start + CHUNK_ROWS)
            global_rows = sampled_rows[start:stop]
            components = component_indices[start:stop]
            legal = np.asarray(data["legal_action_ids"][global_rows])
            prior = np.asarray(data["prior_policy"][global_rows], dtype=np.float32)
            valid = legal >= 0
            legal_counts = valid.sum(axis=1)
            has_prior = (prior * valid).sum(axis=1) > 1.0e-6
            for local_index in range(stop - start):
                draw_index = start + local_index
                component_index = int(components[local_index])
                component_id = data.component_ids[component_index]
                row_offset = int(global_rows[local_index] - offsets[component_index])
                row = {
                    "row_identity_sha256": canonical_row_identity(
                        payload_member_sha256=component_inventory[component_id],
                        row_offset=row_offset,
                        component_id=component_id,
                        prior_policy_present=bool(has_prior[local_index]),
                        legal_action_count=int(legal_counts[local_index]),
                    ),
                    "payload_member_sha256": component_inventory[component_id],
                    "row_offset": row_offset,
                    "component_id": component_id,
                    "prior_policy_present": bool(has_prior[local_index]),
                    "legal_action_count": int(legal_counts[local_index]),
                }
                encoded = _canonical_bytes(row) + b"\n"
                handle.write(encoded)
                identity = row["row_identity_sha256"]
                identities.append(identity)
                identities_by_component[component_id].append(identity)
                _ordered_identity_update(
                    order_digest,
                    index=draw_index,
                    row_identity_sha256=identity,
                )
                evidence_line = _canonical_bytes(
                    {"draw_index": draw_index, **row}
                ) + b"\n"
                evidence_digest.update(evidence_line)
                if (
                    component_id == "historical_replay"
                    and row["prior_policy_present"]
                    and row["legal_action_count"] > 1
                ):
                    eligible += 1
                    eligible_digest.update(evidence_line)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_immutable(temporary, rows_path)
    prior, prior_by_component, prior_file_sha256, prior_row_set_sha256 = (
        _load_prior_rows(prior_rows_path)
    )
    if prior_rows_path is not None and not prior:
        raise EvidenceError("FINAL prior sample evidence is empty")
    unique = set(identities)
    overlap = unique & prior
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities = probabilities / float(probabilities.sum())
    expected_overlap = _expected_unique_overlap(probabilities, sample_dose)
    # McDiarmid bound for a function of the two independent m-draw streams;
    # changing one draw alters unique-set overlap by at most two.
    alpha = 1.0e-9
    excess_bound = math.sqrt(4.0 * sample_dose * math.log(2.0 / alpha))
    per_component: dict[str, dict[str, Any]] = {}
    train_components = np.asarray(
        data.component_indices_for_rows(train_indices), dtype=np.int64
    )
    for index, component_id in enumerate(data.component_ids):
        current_unique = set(identities_by_component[component_id])
        component_probabilities = probabilities[train_components == index]
        expected = _expected_unique_overlap(component_probabilities, sample_dose)
        per_component[component_id] = {
            "draw_count": len(identities_by_component[component_id]),
            "unique_row_count": len(current_unique),
            "prior_unique_row_count": len(prior_by_component[component_id]),
            "observed_unique_overlap_count": len(
                current_unique & prior_by_component[component_id]
            ),
            "analytic_expected_unique_overlap_decimal": format(expected, ".12f"),
        }
    sampler_identity = {
        "schema_version": "a1-authenticated-sampler-identity-v1",
        "algorithm": SAMPLER_ALGORITHM,
        "descriptor_sha256": authenticated["descriptor_file_sha256"],
        "payload_inventory_sha256": authenticated["payload_inventory_sha256"],
        "category_semantics_sha256": authenticated[
            "category_semantics_sha256"
        ],
        "source_authority_sha256": authenticated["source_authority_ref"][
            "authority_sha256"
        ],
        "sampler_seed": sampler_seed,
        "sample_dose": sample_dose,
    }
    _assert_composite_stable(descriptor, authenticated)
    result = _sealed(
        {
            "schema_version": SCHEMA_SAMPLE,
            "status": "complete",
            "sample_dose": sample_dose,
            "sampler_seed": sampler_seed,
            "sampler_algorithm": SAMPLER_ALGORITHM,
            "sampler_identity_sha256": _digest(sampler_identity),
            "sample_order_sha256": "sha256:" + order_digest.hexdigest(),
            "row_set_sha256": _row_set_sha256(identities),
            "unique_row_count": len(unique),
            "prior_rows_file_sha256": prior_file_sha256,
            "prior_row_set_sha256": prior_row_set_sha256,
            "prior_unique_row_count": len(prior),
            "observed_unique_overlap_count": len(overlap),
            "analytic_expected_unique_overlap_decimal": format(
                expected_overlap, ".12f"
            ),
            "overlap_excess_bound_decimal": format(excess_bound, ".12f"),
            "overlap_alpha_decimal": "0.000000001",
            "overlap_within_independent_bound": (
                not prior
                or abs(len(overlap) - expected_overlap) <= excess_bound
            ),
            "component_overlap": per_component,
            "kl_eligible_rows": eligible,
            "kl_eligible_mass_decimal": format(
                eligible / sample_dose, ".12f"
            ).rstrip("0").rstrip("."),
            "kl_ordered_evidence_sha256": "sha256:"
            + evidence_digest.hexdigest(),
            "kl_eligible_evidence_sha256": "sha256:"
            + eligible_digest.hexdigest(),
            "descriptor_sha256": authenticated["descriptor_file_sha256"],
            "payload_inventory_sha256": authenticated[
                "payload_inventory_sha256"
            ],
            "category_semantics": authenticated["category_semantics"],
            "category_semantics_sha256": authenticated[
                "category_semantics_sha256"
            ],
            "source_authority": authenticated["source_authority_ref"],
            "rows_file_sha256": _file_sha256(rows_path),
            "origin_tool_sha256": origin_tool_sha256(),
            "replay_verified": True,
        }
    )
    if prior and not result["overlap_within_independent_bound"]:
        raise EvidenceError("sample overlap exceeds independent-replacement bound")
    return result


def _load_slot12_column(checkpoint: Path):
    import torch

    checkpoint = checkpoint.expanduser().resolve(strict=True)
    checkpoint_sha256 = _file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise EvidenceError("checkpoint has no model state")
    public_award_contract = payload.get("public_award_feature_contract")
    if public_award_contract != "authoritative_v1":
        raise EvidenceError(
            "slot12 evidence requires checkpoint metadata "
            "public_award_feature_contract=authoritative_v1"
        )
    matches = {
        name: tensor
        for name, tensor in state.items()
        if name.endswith("player_encoder.0.weight")
    }
    if len(matches) != 1:
        raise EvidenceError("checkpoint has no unique player encoder input weight")
    name, weight = next(iter(matches.items()))
    if weight.ndim != 2 or weight.shape[1] <= 12:
        raise EvidenceError("player encoder cannot expose slot12 input column")
    column = weight[:, 12].detach().cpu().contiguous()
    parameter_identity = {
        "name": name,
        "dtype": str(column.dtype),
        "shape": list(column.shape),
        "input_column_index": 12,
        "input_feature": "public_award_status",
    }
    if not bool(torch.isfinite(column).all().item()):
        raise EvidenceError("model slot12 input column is non-finite")
    if _file_sha256(checkpoint) != checkpoint_sha256:
        raise EvidenceError("checkpoint bytes changed during slot12 scan")
    return checkpoint_sha256, parameter_identity, column, public_award_contract


def _slot12_column_sha256(parameter_identity: Mapping[str, Any], column) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_bytes(parameter_identity))
    digest.update(b"\0")
    digest.update(column.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _tensor_exact_sha256(name: str, tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    header = {
        "name": name,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }
    digest = hashlib.sha256(_canonical_bytes(header) + b"\0")
    digest.update(value.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _checkpoint_value_equal(left: Any, right: Any) -> bool:
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        return bool(
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and left.layout == right.layout
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return bool(
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_checkpoint_value_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return bool(
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _checkpoint_value_equal(a, b)
                for a, b in zip(left, right, strict=True)
            )
        )
    return bool(type(left) is type(right) and left == right)


def _public_award_transition_evidence(
    source_checkpoint: Path, transitioned_checkpoint: Path
) -> dict[str, Any]:
    """Replay the only legal legacy->authoritative initializer mutation."""

    import torch

    source = source_checkpoint.expanduser().resolve(strict=True)
    transitioned = transitioned_checkpoint.expanduser().resolve(strict=True)
    source_sha = _file_sha256(source)
    transitioned_sha = _file_sha256(transitioned)
    before = torch.load(source, map_location="cpu", weights_only=False)
    after = torch.load(transitioned, map_location="cpu", weights_only=False)
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise EvidenceError("public-award transition checkpoint root is malformed")
    source_contract = str(
        before.get(
            "public_award_feature_contract",
            train_bc.PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
        )
    )
    if source_contract != train_bc.PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO:
        raise EvidenceError(
            "public-award transition source must be legacy_zero_v0"
        )
    if (
        after.get("public_award_feature_contract")
        != train_bc.PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
    ):
        raise EvidenceError(
            "public-award transition output must declare authoritative_v1"
        )
    before_model = before.get("model")
    after_model = after.get("model")
    if not isinstance(before_model, Mapping) or not isinstance(after_model, Mapping):
        raise EvidenceError("public-award transition model state is malformed")
    if set(before_model) != set(after_model):
        raise EvidenceError("public-award transition changed model parameter keys")
    matches = [name for name in before_model if name.endswith("player_encoder.0.weight")]
    if len(matches) != 1:
        raise EvidenceError(
            "public-award transition has no unique player encoder input weight"
        )
    target_name = matches[0]
    source_weight = before_model[target_name]
    output_weight = after_model[target_name]
    if (
        not torch.is_tensor(source_weight)
        or not torch.is_tensor(output_weight)
        or source_weight.dtype != output_weight.dtype
        or source_weight.layout != output_weight.layout
        or tuple(source_weight.shape) != tuple(output_weight.shape)
        or source_weight.ndim != 2
        or source_weight.shape[1] <= train_bc.PLAYER_LONGEST_ROAD_SLOT
        or not bool(torch.isfinite(source_weight).all().item())
        or not bool(torch.isfinite(output_weight).all().item())
    ):
        raise EvidenceError("public-award transition player encoder shape/type drift")
    column = train_bc.PLAYER_LONGEST_ROAD_SLOT
    if bool(torch.count_nonzero(output_weight[:, column]).item()):
        raise EvidenceError("public-award transition output column is not exact zero")
    if not torch.equal(source_weight[:, :column], output_weight[:, :column]) or not torch.equal(
        source_weight[:, column + 1 :], output_weight[:, column + 1 :]
    ):
        raise EvidenceError(
            "public-award transition changed player encoder outside slot12"
        )
    unchanged_records: list[dict[str, Any]] = []
    for name in sorted(before_model):
        if name == target_name:
            continue
        left, right = before_model[name], after_model[name]
        if (
            not torch.is_tensor(left)
            or not torch.is_tensor(right)
            or left.dtype != right.dtype
            or left.layout != right.layout
            or tuple(left.shape) != tuple(right.shape)
            or not torch.equal(left, right)
        ):
            raise EvidenceError(
                f"public-award transition changed inherited parameter {name!r}"
            )
        unchanged_records.append(
            {"name": name, "tensor_sha256": _tensor_exact_sha256(name, left)}
        )
    before_metadata = {
        key: value
        for key, value in before.items()
        if key not in {"model", "public_award_feature_contract"}
    }
    after_metadata = {
        key: value
        for key, value in after.items()
        if key
        not in {
            "model",
            "public_award_feature_contract",
            "public_award_initializer_transition",
        }
    }
    if not _checkpoint_value_equal(before_metadata, after_metadata):
        raise EvidenceError("public-award transition changed unrelated metadata")
    expected_provenance = {
        "schema_version": "a1-public-award-initializer-transition-v1",
        "source_checkpoint_sha256": source_sha,
        "source_contract": train_bc.PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
        "target_contract": train_bc.PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
        "parameter_name": target_name,
        "input_column_index": column,
        "operation": "exact_zero_before_optimizer",
    }
    if after.get("public_award_initializer_transition") != expected_provenance:
        raise EvidenceError("public-award transition provenance drift")
    identity = {
        "name": target_name,
        "dtype": str(source_weight.dtype),
        "shape": list(source_weight.shape),
        "input_column_index": column,
        "input_feature": "public_award_status",
    }
    source_column = source_weight[:, column].detach().cpu().contiguous()
    output_column = output_weight[:, column].detach().cpu().contiguous()
    return _sealed(
        {
            "schema_version": SCHEMA_PUBLIC_AWARD_TRANSITION,
            "status": "complete",
            "source_checkpoint_sha256": source_sha,
            "transitioned_checkpoint_sha256": transitioned_sha,
            "source_public_award_feature_contract": source_contract,
            "transitioned_public_award_feature_contract": (
                train_bc.PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
            ),
            "changed_parameter_name": target_name,
            "changed_input_column_index": column,
            "source_slot12_column_sha256": _slot12_column_sha256(
                identity, source_column
            ),
            "transitioned_slot12_column_sha256": _slot12_column_sha256(
                identity, output_column
            ),
            "transitioned_slot12_max_abs_decimal": "0",
            "unchanged_parameter_count": len(unchanged_records),
            "unchanged_parameter_identity_sha256": _digest(unchanged_records),
            "unchanged_parameters_bit_identical": True,
            "unrelated_metadata_bit_identical": True,
            "legacy_zero_input_function_preserving": True,
            "optimizer_steps": 0,
            "origin_tool_sha256": origin_tool_sha256(),
        }
    )


def build_public_award_transition_initializer(
    source_checkpoint: Path, transitioned_checkpoint: Path
) -> dict[str, Any]:
    """Create and attest one immutable pre-optimizer transition checkpoint."""

    import torch

    source = source_checkpoint.expanduser().resolve(strict=True)
    destination = transitioned_checkpoint.expanduser().resolve(strict=False)
    if source.is_symlink() or not source.is_file():
        raise EvidenceError("public-award transition source must be a regular file")
    raw = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("model"), Mapping):
        raise EvidenceError("public-award transition source checkpoint is malformed")
    source_contract = str(
        raw.get(
            "public_award_feature_contract",
            train_bc.PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
        )
    )
    if source_contract != train_bc.PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO:
        raise EvidenceError(
            "public-award transition source must be legacy_zero_v0"
        )
    matches = [
        name for name in raw["model"] if name.endswith("player_encoder.0.weight")
    ]
    if len(matches) != 1:
        raise EvidenceError(
            "public-award transition source has no unique player encoder input weight"
        )
    target_name = matches[0]
    source_weight = raw["model"][target_name]
    if (
        not torch.is_tensor(source_weight)
        or source_weight.ndim != 2
        or source_weight.shape[1] <= train_bc.PLAYER_LONGEST_ROAD_SLOT
        or not bool(torch.isfinite(source_weight).all().item())
    ):
        raise EvidenceError("public-award transition source column is invalid")
    payload = copy.deepcopy(dict(raw))
    model = dict(payload["model"])
    transitioned_weight = source_weight.detach().clone()
    transitioned_weight[:, train_bc.PLAYER_LONGEST_ROAD_SLOT].zero_()
    model[target_name] = transitioned_weight
    payload["model"] = model
    payload["public_award_feature_contract"] = (
        train_bc.PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
    )
    payload["public_award_initializer_transition"] = {
        "schema_version": "a1-public-award-initializer-transition-v1",
        "source_checkpoint_sha256": _file_sha256(source),
        "source_contract": train_bc.PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
        "target_contract": train_bc.PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
        "parameter_name": target_name,
        "input_column_index": train_bc.PLAYER_LONGEST_ROAD_SLOT,
        "operation": "exact_zero_before_optimizer",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{os.urandom(8).hex()}"
    )
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _publish_immutable(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _public_award_transition_evidence(source, destination)


def build_initializer_slot12_zero_receipt(checkpoint: Path) -> dict[str, Any]:
    checkpoint_sha256, parameter_identity, column, contract = _load_slot12_column(
        checkpoint
    )
    max_abs = float(column.abs().max().item()) if column.numel() else 0.0
    evidence = _slot12_column_sha256(parameter_identity, column)
    if max_abs != 0.0:
        raise EvidenceError("initializer slot12 input column is not exactly zero")
    return _sealed(
        {
            "schema_version": SCHEMA_INITIALIZER_ZERO,
            "status": "complete",
            "measurement_phase": "pre_optimizer",
            "initializer_checkpoint_sha256": checkpoint_sha256,
            "public_award_feature_contract": contract,
            "model_slot12_parameter_set_sha256": _digest(parameter_identity),
            "model_slot12_parameter_count": int(column.numel()),
            "initializer_slot12_max_abs_decimal": "0",
            "initializer_slot12_column_sha256": evidence,
            "initializer_slot12_zero_evidence_sha256": evidence,
            "origin_tool_sha256": origin_tool_sha256(),
        }
    )


def _canonical_float(value: float) -> str:
    if not math.isfinite(float(value)):
        raise EvidenceError("slot12 measurement is non-finite")
    return format(float(value), ".17g")


def build_trained_slot12_delta_receipt(
    initializer_checkpoint: Path, candidate_checkpoint: Path
) -> dict[str, Any]:
    import torch

    initializer_sha, initializer_identity, before, initializer_contract = (
        _load_slot12_column(initializer_checkpoint)
    )
    candidate_sha, candidate_identity, after, candidate_contract = (
        _load_slot12_column(candidate_checkpoint)
    )
    if initializer_identity != candidate_identity:
        raise EvidenceError("trained slot12 parameter identity changed")
    before64 = before.to(dtype=torch.float64)
    after64 = after.to(dtype=torch.float64)
    delta = after64 - before64
    before_max = float(before64.abs().max().item()) if before64.numel() else 0.0
    if before_max != 0.0:
        raise EvidenceError("trained delta initializer was not exactly zero")
    after_max = float(after64.abs().max().item()) if after64.numel() else 0.0
    delta_max = float(delta.abs().max().item()) if delta.numel() else 0.0
    delta_l2 = float(torch.linalg.vector_norm(delta).item()) if delta.numel() else 0.0
    nonzero = int(torch.count_nonzero(after64).item())
    evidence = hashlib.sha256()
    evidence.update(_canonical_bytes(initializer_identity))
    evidence.update(b"\0")
    evidence.update(before.numpy().tobytes())
    evidence.update(b"\0")
    evidence.update(after.numpy().tobytes())
    initializer_column_sha = _slot12_column_sha256(initializer_identity, before)
    candidate_column_sha = _slot12_column_sha256(candidate_identity, after)
    return _sealed(
        {
            "schema_version": SCHEMA_TRAINED_DELTA,
            "status": "complete",
            "measurement_phase": "post_optimizer",
            "initializer_checkpoint_sha256": initializer_sha,
            "candidate_checkpoint_sha256": candidate_sha,
            "initializer_public_award_feature_contract": initializer_contract,
            "candidate_public_award_feature_contract": candidate_contract,
            "model_slot12_parameter_set_sha256": _digest(initializer_identity),
            "model_slot12_parameter_count": int(after.numel()),
            "initializer_slot12_max_abs_decimal": "0",
            "initializer_slot12_column_sha256": initializer_column_sha,
            "candidate_slot12_column_sha256": candidate_column_sha,
            "candidate_slot12_max_abs_decimal": _canonical_float(after_max),
            "slot12_delta_max_abs_decimal": _canonical_float(delta_max),
            "slot12_delta_l2_decimal": _canonical_float(delta_l2),
            "candidate_slot12_nonzero_count": nonzero,
            "candidate_slot12_finite": True,
            "learned_signal_observed": nonzero > 0,
            "slot12_delta_evidence_sha256": "sha256:" + evidence.hexdigest(),
            "origin_tool_sha256": origin_tool_sha256(),
        }
    )


def verify_runtime_admission_receipt(
    path: Path, *, expected_origin_tool_sha256: str
) -> dict[str, Any]:
    _require_sha(expected_origin_tool_sha256, where="expected evidence producer")
    receipt = _load_sealed(
        path,
        where="B200 runtime admission receipt",
        expected_origin_tool_sha256=expected_origin_tool_sha256,
    )
    _require_exact_keys(
        receipt,
        {"schema_version", "status", "hosts", "origin_tool_sha256", "state_sha256"},
        where="B200 runtime admission receipt",
    )
    if receipt["schema_version"] != SCHEMA_RUNTIME or receipt["status"] != "complete":
        raise EvidenceError("B200 runtime admission schema/status drift")
    hosts = receipt["hosts"]
    if not isinstance(hosts, dict) or set(hosts) != {B200_LEARNER_HOST_ID}:
        raise EvidenceError("runtime admission does not describe the sole learner")
    report = _require_exact_keys(
        hosts[B200_LEARNER_HOST_ID],
        {
            "host_id",
            "hostname",
            "machine_id",
            "ssh_host_key_sha256",
            "checkout_tree_sha256",
            "tool_sha256",
            "gpu_indices",
            "gpu_names",
            "gpu_uuids",
            "pci_bus_ids",
            "python",
            "torch_version",
            "torch_cuda_version",
            "catanatron_rs_version",
            "native_wheel_sha256",
            "native_mcts_capabilities",
            "nofile_soft",
            "nofile_hard",
        },
        where="B200 runtime report",
    )
    release = production_executor._native_wheel_release_identity()
    runtime = production_executor.PRODUCTION_RUNTIME
    capabilities = report["native_mcts_capabilities"]
    if (
        report["host_id"] != B200_LEARNER_HOST_ID
        or report["hostname"] != B200_LEARNER_HOSTNAME
        or report["machine_id"] != B200_LEARNER_MACHINE_ID
        or report["tool_sha256"] != expected_origin_tool_sha256
        or report["gpu_indices"] != list(range(WORLD_SIZE))
        or report["gpu_uuids"] != list(B200_LEARNER_GPU_UUIDS)
        or not isinstance(report["gpu_names"], list)
        or len(report["gpu_names"]) != WORLD_SIZE
        or any("B200" not in str(name).upper() for name in report["gpu_names"])
        or not isinstance(report["pci_bus_ids"], list)
        or len(report["pci_bus_ids"]) != WORLD_SIZE
        or len(set(report["pci_bus_ids"])) != WORLD_SIZE
        or report["python"] != runtime["python_version"]
        or report["torch_version"] != runtime["torch_version"]
        or report["torch_cuda_version"] != runtime["torch_cuda_version"]
        or report["catanatron_rs_version"] != release["version"]
        or report["native_wheel_sha256"] != release["sha256"]
        or not isinstance(capabilities, list)
        or not set(production_executor.NATIVE_REQUIRED_CAPABILITIES)
        <= set(capabilities)
        or type(report["nofile_soft"]) is not int
        or type(report["nofile_hard"]) is not int
        or report["nofile_soft"] < 65_536
        or report["nofile_hard"] < report["nofile_soft"]
    ):
        raise EvidenceError("B200 runtime report semantics drift")
    for field in (
        "ssh_host_key_sha256",
        "checkout_tree_sha256",
        "tool_sha256",
        "native_wheel_sha256",
    ):
        _require_sha(report[field], where=f"B200 runtime {field}")
    return receipt


def verify_mixed_routing_receipt(
    path: Path, *, descriptor: Path, expected_origin_tool_sha256: str
) -> dict[str, Any]:
    receipt = _load_sealed(
        path,
        where="mixed routing receipt",
        expected_origin_tool_sha256=expected_origin_tool_sha256,
    )
    replay = build_mixed_routing_receipt(descriptor)
    if receipt != replay:
        raise EvidenceError("mixed routing receipt failed payload replay")
    return receipt


def verify_sample_evidence(
    path: Path,
    *,
    descriptor: Path,
    rows_path: Path,
    prior_rows_path: Path | None = None,
    expected_origin_tool_sha256: str,
) -> dict[str, Any]:
    receipt = _load_sealed(
        path,
        where="sample evidence receipt",
        expected_origin_tool_sha256=expected_origin_tool_sha256,
    )
    if receipt.get("schema_version") != SCHEMA_SAMPLE:
        raise EvidenceError("sample evidence schema drift")
    if _file_sha256(rows_path.expanduser().resolve(strict=True)) != receipt.get(
        "rows_file_sha256"
    ):
        raise EvidenceError("sample evidence row bytes drift")
    with tempfile.TemporaryDirectory(prefix="a1-sample-replay-") as temporary:
        replay_rows = Path(temporary) / "rows.jsonl"
        replay = build_sample_evidence(
            descriptor,
            sampler_seed=int(receipt["sampler_seed"]),
            sample_dose=int(receipt["sample_dose"]),
            rows_path=replay_rows,
            prior_rows_path=prior_rows_path,
        )
    if receipt != replay:
        raise EvidenceError("sample evidence failed exact sampler/payload replay")
    return receipt


def verify_initializer_slot12_zero_receipt(
    path: Path, *, checkpoint: Path, expected_origin_tool_sha256: str
) -> dict[str, Any]:
    receipt = _load_sealed(
        path,
        where="initializer slot12 zero receipt",
        expected_origin_tool_sha256=expected_origin_tool_sha256,
    )
    replay = build_initializer_slot12_zero_receipt(checkpoint)
    if receipt != replay:
        raise EvidenceError("initializer slot12 zero receipt failed tensor replay")
    return receipt


def verify_public_award_transition_receipt(
    path: Path,
    *,
    source_checkpoint: Path,
    transitioned_checkpoint: Path,
    expected_origin_tool_sha256: str,
) -> dict[str, Any]:
    receipt = _load_sealed(
        path,
        where="public-award initializer transition receipt",
        expected_origin_tool_sha256=expected_origin_tool_sha256,
    )
    replay = _public_award_transition_evidence(
        source_checkpoint, transitioned_checkpoint
    )
    if receipt != replay:
        raise EvidenceError(
            "public-award initializer transition receipt failed checkpoint replay"
        )
    return receipt


def verify_trained_slot12_delta_receipt(
    path: Path,
    *,
    initializer_checkpoint: Path,
    candidate_checkpoint: Path,
    expected_origin_tool_sha256: str,
) -> dict[str, Any]:
    receipt = _load_sealed(
        path,
        where="trained slot12 delta receipt",
        expected_origin_tool_sha256=expected_origin_tool_sha256,
    )
    replay = build_trained_slot12_delta_receipt(
        initializer_checkpoint, candidate_checkpoint
    )
    if receipt != replay:
        raise EvidenceError("trained slot12 delta receipt failed tensor replay")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runtime = sub.add_parser("runtime-admission")
    runtime.add_argument("--output", type=Path, required=True)
    routing = sub.add_parser("mixed-routing")
    routing.add_argument("--descriptor", type=Path, required=True)
    routing.add_argument("--output", type=Path, required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("--descriptor", type=Path, required=True)
    sample.add_argument("--sampler-seed", type=int, required=True)
    sample.add_argument("--sample-dose", type=int, default=524288)
    sample.add_argument("--rows-output", type=Path, required=True)
    sample.add_argument("--prior-rows", type=Path, default=None)
    sample.add_argument("--output", type=Path, required=True)
    initializer = sub.add_parser("initializer-slot12-zero")
    initializer.add_argument("--checkpoint", type=Path, required=True)
    initializer.add_argument("--output", type=Path, required=True)
    transition = sub.add_parser("public-award-initializer-transition")
    transition.add_argument("--source-checkpoint", type=Path, required=True)
    transition.add_argument("--transitioned-checkpoint", type=Path, required=True)
    transition.add_argument("--output", type=Path, required=True)
    trained = sub.add_parser("trained-slot12-delta")
    trained.add_argument("--initializer-checkpoint", type=Path, required=True)
    trained.add_argument("--candidate-checkpoint", type=Path, required=True)
    trained.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "runtime-admission":
            receipt = build_runtime_admission_receipt()
        elif args.command == "mixed-routing":
            receipt = build_mixed_routing_receipt(args.descriptor)
        elif args.command == "sample":
            receipt = build_sample_evidence(
                args.descriptor,
                sampler_seed=args.sampler_seed,
                sample_dose=args.sample_dose,
                rows_path=args.rows_output,
                prior_rows_path=args.prior_rows,
            )
        elif args.command == "initializer-slot12-zero":
            receipt = build_initializer_slot12_zero_receipt(args.checkpoint)
        elif args.command == "public-award-initializer-transition":
            receipt = build_public_award_transition_initializer(
                args.source_checkpoint, args.transitioned_checkpoint
            )
        else:
            receipt = build_trained_slot12_delta_receipt(
                args.initializer_checkpoint, args.candidate_checkpoint
            )
        _atomic_write(args.output.expanduser().resolve(strict=False), receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except (EvidenceError, OSError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
