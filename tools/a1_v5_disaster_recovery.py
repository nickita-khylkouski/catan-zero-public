#!/usr/bin/env python3
"""One-shot administrative recovery of the exact surviving A1 v5 producer.

This is deliberately *not* a promotion replayer.  The original v5 promotion
receipt, registry, and current-pointer bytes are gone.  One exact surviving
handoff and the exact checkpoint it names remain.  This tool can establish a
fresh, quarantined registry lineage from those bytes while preserving the
evidence loss as a first-class invariant:

* the recovered checkpoint is generator-only;
* f7 remains the public/tournament safety reference;
* f7 is not called a proven causal parent without a surviving binding;
* the missing promotion artifacts are recorded as absent and unreplayed; and
* the recovery receipt is never a promotion or training receipt.

Dry-run is read-only.  ``--go`` publishes a prepared journal, registry,
pointer, and committed receipt in that order.  A crash may resume only the
exact prepared journal in the same one-shot namespace.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_pre_wave_contract as pre_wave  # noqa: E402
from tools import production_runtime_contract as runtime_contract  # noqa: E402


RECOVERY_SCHEMA = "a1-v5-disaster-recovery-bootstrap-v1"
JOURNAL_SCHEMA = "a1-v5-disaster-recovery-journal-v1"
RECOVERY_MODE = "administrative_disaster_recovery"
RECOVERY_RELATION = "safety_reference_unproven_predecessor"
RECOVERED_GENERATOR_ROLE = "generator_champion"
SAFETY_ROLES = ("public_champion", "tournament_bot")
RECOVERY_NAMESPACE_BASENAME = "a1-v5-disaster-recovery"
RECOVERY_LOCK_BASENAME = ".a1-v5-disaster-recovery.lock"

EXPECTED_HANDOFF_FINGERPRINT = dict(pre_wave.HISTORICAL_V5_HANDOFF_FINGERPRINT)
EXPECTED_F7_SHA256 = (
    "sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4"
)
EXPECTED_F7_VERSION = 4
EXPECTED_V5_VERSION = 5
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")


class RecoveryError(RuntimeError):
    """The exact v5 recovery boundary cannot be established honestly."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RecoveryError(f"value is not canonical JSON: {error}") from error


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _md5_bytes(value: bytes) -> str:
    # Registry compatibility identifier, never a security identity.
    return hashlib.md5(value).hexdigest()  # noqa: S324


def _json_runtime_value(value: Any, *, where: str) -> Any:
    """Normalize runtime/config scalars without weakening canonical JSON.

    Historical checkpoints may carry NumPy scalar dimensions (notably
    ``np.int64(action_size)``).  They are semantically ordinary JSON numbers,
    but the stdlib encoder does not recognize them.  Convert only scalar
    ``item()`` values and recursively reject everything else; tensors, arrays,
    and arbitrary objects must never leak into the recovery receipt.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_runtime_value(child, where=f"{where}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_runtime_value(child, where=f"{where}[{index}]")
            for index, child in enumerate(value)
        ]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError) as error:
            raise RecoveryError(f"{where} is not a scalar runtime value") from error
        if scalar is value:
            raise RecoveryError(f"{where} scalar conversion did not make progress")
        return _json_runtime_value(scalar, where=where)
    raise RecoveryError(
        f"{where} has unsupported runtime value type {type(value).__name__}"
    )


def _require_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise RecoveryError(f"{where} must be a canonical sha256 digest")
    return value


def _stable_read(path: Path, *, where: str) -> tuple[bytes, tuple[int, ...]]:
    """Read one regular file without following its final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RecoveryError(f"cannot open {where} without symlink following: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RecoveryError(f"{where} is not a regular file")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RecoveryError(f"{where} changed while it was read")
    try:
        live = path.stat(follow_symlinks=False)
    except OSError as error:
        raise RecoveryError(f"cannot restat {where}: {error}") from error
    if identity != (
        live.st_dev,
        live.st_ino,
        live.st_size,
        live.st_mtime_ns,
        live.st_ctime_ns,
    ):
        raise RecoveryError(f"{where} was atomically replaced while it was read")
    return b"".join(chunks), identity


def _stable_json(path: Path, *, where: str) -> tuple[dict[str, Any], bytes]:
    raw, _ = _stable_read(path, where=where)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryError(f"cannot parse {where}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryError(f"{where} must be a JSON object")
    return value, raw


def _canonical_existing(path: Path, *, where: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise RecoveryError(f"cannot resolve {where}: {error}") from error
    if resolved != lexical or not resolved.is_file() or resolved.is_symlink():
        raise RecoveryError(f"{where} must be a canonical regular file: {lexical}")
    return resolved


def _canonical_missing(path: Path, *, where: str) -> Path:
    """Require a missing lexical path whose existing ancestry is canonical."""

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if lexical.exists() or lexical.is_symlink():
        raise RecoveryError(
            f"{where} unexpectedly exists; restore/replay normal promotion evidence instead"
        )
    ancestor = lexical.parent
    while not ancestor.exists() and not ancestor.is_symlink():
        if ancestor == ancestor.parent:
            raise RecoveryError(f"{where} has no existing parent ancestry")
        ancestor = ancestor.parent
    if ancestor.is_symlink() or ancestor.resolve(strict=True) != ancestor:
        raise RecoveryError(f"{where} ancestry is not canonical")
    return lexical


def _missing_claim_path(raw: Any, *, where: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RecoveryError(f"{where} path is malformed")
    return _canonical_missing(Path(raw), where=where)


def _validate_producer_identity(
    identity: Any, *, checkpoint_path: Path, checkpoint_sha256: str
) -> dict[str, Any]:
    if not isinstance(identity, dict) or set(identity) != {
        "schema_version",
        "checkpoint",
        "search_config",
        "agent_identity_sha256",
    }:
        raise RecoveryError("surviving producer identity shape drift")
    unsigned = dict(identity)
    stated = unsigned.pop("agent_identity_sha256")
    if stated != _digest(unsigned):
        raise RecoveryError("surviving producer identity semantic digest mismatch")
    checkpoint = identity["checkpoint"]
    if checkpoint != {"path": str(checkpoint_path), "sha256": checkpoint_sha256}:
        raise RecoveryError("surviving producer identity binds different checkpoint bytes")
    if not isinstance(identity["search_config"], dict) or not identity["search_config"]:
        raise RecoveryError("surviving producer identity has no typed search config")
    return dict(identity)


def _runtime_smoke(checkpoint: Path) -> dict[str, Any]:
    """Actually load the exact checkpoint under the sealed CPU runtime."""

    import torch

    contract_path = runtime_contract.DEFAULT_CONTRACT.resolve(strict=True)
    expected = runtime_contract.load_runtime_contract(contract_path)
    if (
        platform.python_version() != expected["python_version"]
        or torch.__version__ != expected["torch_version"]
        or str(torch.version.cuda) != expected["torch_cuda_version"]
    ):
        raise RecoveryError(
            "checkpoint smoke must run under the exact production Python/Torch/CUDA runtime"
        )
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - exact trusted checkpoint must load.
        raise RecoveryError(f"cannot load recovered checkpoint: {error}") from error
    if not isinstance(raw, Mapping) or raw.get("mask_hidden_info") is not True:
        raise RecoveryError("recovered checkpoint lacks mask_hidden_info=true")
    model = raw.get("model")
    if not isinstance(model, Mapping) or not model:
        raise RecoveryError("recovered checkpoint model state is missing")
    if any(not torch.is_tensor(tensor) for tensor in model.values()):
        raise RecoveryError("recovered checkpoint model state contains non-tensors")
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    try:
        policy = EntityGraphPolicy.load(str(checkpoint), device="cpu")
    except Exception as error:  # noqa: BLE001 - loadability is the smoke invariant.
        raise RecoveryError(f"current policy runtime cannot load recovered checkpoint: {error}") from error
    config = policy.config
    config_value = _json_runtime_value(
        (
            {
                field.name: getattr(config, field.name)
                for field in dataclasses.fields(config)
            }
            if dataclasses.is_dataclass(config)
            else repr(config)
        ),
        where="policy config",
    )
    parameter_signature = [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(model.items())
    ]
    module_path = Path(sys.modules[EntityGraphPolicy.__module__].__file__).resolve(strict=True)
    if REPO_ROOT / "src" not in module_path.parents:
        raise RecoveryError("checkpoint smoke imported policy code outside this checkout")
    return {
        "schema_version": "a1-v5-recovery-runtime-smoke-v1",
        "runtime_contract": {
            "path": str(contract_path),
            "sha256": _sha256_bytes(_stable_read(contract_path, where="runtime contract")[0]),
            "python_version": expected["python_version"],
            "torch_version": expected["torch_version"],
            "torch_cuda_version": expected["torch_cuda_version"],
        },
        "loader": {
            "module": EntityGraphPolicy.__module__,
            "path": str(module_path),
            "sha256": _sha256_bytes(_stable_read(module_path, where="policy loader")[0]),
        },
        "mask_hidden_info": True,
        "parameter_count": sum(int(tensor.numel()) for tensor in model.values()),
        "parameter_signature_sha256": _digest(parameter_signature),
        "config_sha256": _digest(config_value),
        "load_complete": True,
    }


def _tool_identity() -> dict[str, Any]:
    tracked = (
        "tools/a1_v5_disaster_recovery.py",
        "tools/a1_pre_wave_contract.py",
        "configs/runtime/a1_production_runtime.json",
    )
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True
        ).strip()
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise RecoveryError("recovery source commit is malformed")
        subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", *tracked),
            cwd=REPO_ROOT,
            check=True,
        )
        for relative in tracked:
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", relative),
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RecoveryError("recovery sources must be clean, tracked canonical bytes") from error
    files = {
        relative: _sha256_bytes(
            _stable_read(REPO_ROOT / relative, where=f"recovery source {relative}")[0]
        )
        for relative in tracked
    }
    return {"git_commit": commit, "files": files, "files_sha256": _digest(files)}


def _verify_recorded_tool_identity(recorded: Any) -> None:
    """Replay the exact recovery sources from their recorded ancestor commit.

    A committed recovery receipt must remain consumable after ordinary
    fail-closed hardening.  Requiring the *current* HEAD string forever made a
    one-shot receipt unusable after any descendant commit, even when every
    recorded source blob remained available.  This verifier accepts no source
    substitution: the recorded commit must be an ancestor of HEAD and every
    recorded blob must still hash exactly to the receipt's identity.
    """

    if not isinstance(recorded, dict) or set(recorded) != {
        "git_commit",
        "files",
        "files_sha256",
    }:
        raise RecoveryError("recorded recovery source identity is malformed")
    commit = recorded["git_commit"]
    files = recorded["files"]
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(files, dict)
        or not files
        or recorded["files_sha256"] != _digest(files)
    ):
        raise RecoveryError("recorded recovery source identity digest is malformed")
    try:
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for relative, expected in files.items():
            if (
                not isinstance(relative, str)
                or relative.startswith("/")
                or ".." in Path(relative).parts
                or not isinstance(expected, str)
                or _SHA_RE.fullmatch(expected) is None
            ):
                raise RecoveryError("recorded recovery source entry is malformed")
            blob = subprocess.check_output(
                ("git", "show", f"{commit}:{relative}"), cwd=REPO_ROOT
            )
            if _sha256_bytes(blob) != expected:
                raise RecoveryError(
                    f"recorded recovery source blob drift at {commit}:{relative}"
                )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RecoveryError(
            "recorded recovery source commit is unavailable or not an ancestor"
        ) from error


def _fresh_namespace(namespace: Path) -> tuple[Path, dict[str, str]]:
    lexical = Path(os.path.abspath(os.fspath(namespace.expanduser())))
    if lexical.name != RECOVERY_NAMESPACE_BASENAME:
        raise RecoveryError(
            f"recovery namespace basename must be {RECOVERY_NAMESPACE_BASENAME!r}"
        )
    _canonical_missing(lexical, where="recovery namespace")
    destinations = {
        "namespace": str(lexical),
        "registry": str(lexical / "private/champion_registry.json"),
        "current_pointer": str(lexical / "private/CURRENT_CHAMPION"),
        "receipt": str(
            lexical / "private/receipts/a1-v5-disaster-recovery.receipt.json"
        ),
        "prepared_journal": str(
            lexical / "private/receipts/a1-v5-disaster-recovery.prepared.json"
        ),
    }
    return lexical, destinations


def _handoff_evidence(
    handoff_path: Path,
) -> tuple[dict[str, Any], dict[Path, bytes], dict[str, Any]]:
    handoff, handoff_bytes = _stable_json(handoff_path, where="surviving v5 handoff")
    declared = handoff.get("handoff_sha256")
    unsigned = dict(handoff)
    unsigned.pop("handoff_sha256", None)
    if declared != _digest(unsigned):
        raise RecoveryError("surviving handoff semantic digest mismatch")
    if handoff.get("schema_version") != "a1-post-promotion-producer-handoff-v1":
        raise RecoveryError("surviving handoff schema drift")
    receipt = handoff.get("promotion_receipt")
    registry = handoff.get("registry_after")
    pointer = handoff.get("current_champion")
    identity = handoff.get("producer_identity")
    if not all(isinstance(value, dict) for value in (receipt, registry, pointer, identity)):
        raise RecoveryError("surviving handoff lineage shape drift")
    checkpoint_ref = identity.get("checkpoint")
    if not isinstance(checkpoint_ref, dict) or set(checkpoint_ref) != {"path", "sha256"}:
        raise RecoveryError("surviving handoff checkpoint reference is malformed")
    checkpoint_path = _canonical_existing(
        Path(str(checkpoint_ref["path"])), where="recovered v5 checkpoint"
    )
    checkpoint_bytes, _ = _stable_read(checkpoint_path, where="recovered v5 checkpoint")
    checkpoint_sha = _sha256_bytes(checkpoint_bytes)
    producer_identity = _validate_producer_identity(
        identity, checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha
    )
    actual_fingerprint = {
        "checkpoint_sha256": checkpoint_sha,
        "handoff_file_sha256": _sha256_bytes(handoff_bytes),
        "handoff_sha256": declared,
        "producer_identity_sha256": producer_identity["agent_identity_sha256"],
        "promotion_receipt_file_sha256": receipt.get("sha256"),
        "promotion_receipt_sha256": receipt.get("receipt_sha256"),
        "registry_version": registry.get("version"),
    }
    if actual_fingerprint != EXPECTED_HANDOFF_FINGERPRINT:
        raise RecoveryError("surviving handoff is not the sole allowlisted v5 artifact")
    if (
        registry.get("role") != RECOVERED_GENERATOR_ROLE
        or registry.get("checkpoint") != checkpoint_ref
        or registry.get("version") != EXPECTED_V5_VERSION
    ):
        raise RecoveryError("surviving handoff generator role/checkpoint/version drift")
    try:
        expected_pointer = base64.b64decode(pointer["bytes_base64"], validate=True)
    except (KeyError, ValueError) as error:
        raise RecoveryError("surviving handoff pointer bytes are malformed") from error
    if expected_pointer != (str(checkpoint_path) + "\n").encode("utf-8"):
        raise RecoveryError("surviving handoff pointer claim names different bytes")
    lost_receipt_path = _missing_claim_path(
        receipt.get("path"), where="lost original promotion receipt"
    )
    lost_registry_path = _missing_claim_path(
        registry.get("path"), where="lost original registry"
    )
    lost_pointer_path = _missing_claim_path(
        pointer.get("path"), where="lost original current pointer"
    )
    lost_claims = {
        "promotion_receipt": {
            "claimed_path": str(lost_receipt_path),
            "claimed_file_sha256": receipt.get("sha256"),
            "claimed_semantic_sha256": receipt.get("receipt_sha256"),
            "claimed_transaction_id": receipt.get("transaction_id"),
            "present": False,
            "replayed": False,
        },
        "registry_after": {
            "claimed_path": str(lost_registry_path),
            "claimed_file_sha256": registry.get("sha256"),
            "claimed_version": registry.get("version"),
            "present": False,
            "replayed": False,
        },
        "current_pointer": {
            "claimed_path": str(lost_pointer_path),
            "claimed_file_sha256": pointer.get("sha256"),
            "present": False,
            "replayed": False,
        },
    }
    for group in lost_claims.values():
        for key, value in group.items():
            if key.endswith("sha256"):
                _require_sha(value, f"lost claim {key}")
    evidence = {
        "surviving_handoff": {
            "path": str(handoff_path),
            "sha256": actual_fingerprint["handoff_file_sha256"],
            "handoff_sha256": declared,
            "schema_version": handoff["schema_version"],
        },
        "producer_identity": producer_identity,
        "lost_claims_from_surviving_handoff": lost_claims,
        "recovered_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "md5": _md5_bytes(checkpoint_bytes),
            "historical_generation_version_claim": EXPECTED_V5_VERSION,
        },
    }
    return evidence, {handoff_path: handoff_bytes, checkpoint_path: checkpoint_bytes}, handoff


def _safety_reference(path: Path) -> tuple[dict[str, Any], bytes]:
    canonical = _canonical_existing(path, where="f7 safety reference")
    raw, _ = _stable_read(canonical, where="f7 safety reference")
    sha = _sha256_bytes(raw)
    if sha != EXPECTED_F7_SHA256:
        raise RecoveryError("safety reference is not exact authenticated f7 bytes")
    return (
        {
            "path": str(canonical),
            "sha256": sha,
            "md5": _md5_bytes(raw),
            "version": EXPECTED_F7_VERSION,
            "relationship": RECOVERY_RELATION,
            "causal_parent_proven": False,
        },
        raw,
    )


def _revalidate_snapshot(snapshot: Mapping[Path, bytes]) -> None:
    for path, expected in snapshot.items():
        actual, _ = _stable_read(path, where=f"recovery snapshot {path}")
        if actual != expected:
            raise RecoveryError(f"recovery input changed during verification: {path}")


def build_plan(
    *,
    handoff_path: Path,
    safety_reference_path: Path,
    namespace: Path,
    runtime_smoke_fn: Callable[[Path], dict[str, Any]] = _runtime_smoke,
    tool_identity_fn: Callable[[], dict[str, Any]] = _tool_identity,
) -> dict[str, Any]:
    """Build the deterministic read-only recovery plan."""

    handoff_path = _canonical_existing(handoff_path, where="surviving v5 handoff")
    _, destinations = _fresh_namespace(namespace)
    evidence, snapshot, _ = _handoff_evidence(handoff_path)
    safety, safety_bytes = _safety_reference(safety_reference_path)
    snapshot[Path(safety["path"])] = safety_bytes
    smoke = runtime_smoke_fn(Path(evidence["recovered_checkpoint"]["path"]))
    if not isinstance(smoke, dict) or smoke.get("load_complete") is not True:
        raise RecoveryError("recovered checkpoint runtime smoke did not complete")
    source_identity = tool_identity_fn()
    if not isinstance(source_identity, dict) or not source_identity:
        raise RecoveryError("recovery source identity is missing")
    _revalidate_snapshot(snapshot)
    lineage_id = _digest(
        {
            "schema_version": RECOVERY_SCHEMA,
            "handoff_sha256": evidence["surviving_handoff"]["sha256"],
            "checkpoint_sha256": evidence["recovered_checkpoint"]["sha256"],
            "safety_reference_sha256": safety["sha256"],
        }
    )
    timestamp_ns = Path(evidence["surviving_handoff"]["path"]).stat().st_mtime_ns
    plan: dict[str, Any] = {
        "schema_version": RECOVERY_SCHEMA,
        "mode": "dry-run",
        "recovery_kind": RECOVERY_MODE,
        "lineage": {
            "lineage_id": lineage_id,
            "name": "a1-v5-recovered-evidence-loss",
            "promotion_proof_recreated": False,
            "historical_generation_version_claim": EXPECTED_V5_VERSION,
            "verified_promotion_count": None,
        },
        "evidence_status": {
            "promotion_evidence_lost": True,
            "checkpoint_and_search_identity_recovered": True,
            "causal_parent_proven": False,
        },
        **evidence,
        "safety_reference": safety,
        "runtime_smoke": smoke,
        "role_policy": {
            "recovered_generator_role": RECOVERED_GENERATOR_ROLE,
            "recovered_generator_only": True,
            "safety_reference_roles": list(SAFETY_ROLES),
            "current_pointer_names_recovered_generator": True,
        },
        "promotion_policy": {
            "recovery_receipt_is_promotion_proof": False,
            "recovery_receipt_is_training_proof": False,
            "ordinary_promotion_mode_forbidden": True,
            "fresh_full_gate_required": True,
            "strict_h1_baseline_sha256": evidence["recovered_checkpoint"]["sha256"],
            "conjunctive_f7_non_regression_sha256": safety["sha256"],
            "dual_baseline_required": True,
            "auto_promotion": False,
        },
        "wave_policy": {
            "lineage_mode": "recovery_reference",
            "recent_history_claim_forbidden": True,
            "history_component_relation": RECOVERY_RELATION,
            "old_search_decision_hashes_reused": False,
            "new_operator_binding_required": True,
        },
        "source_identity": source_identity,
        "recovery_unix_ns": timestamp_ns,
        "recovery_timestamp": timestamp_ns / 1_000_000_000,
        "destinations": destinations,
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


def _verify_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(plan)
    if value.get("schema_version") != RECOVERY_SCHEMA or value.get("mode") != "dry-run":
        raise RecoveryError("recovery plan schema/mode drift")
    stated = value.pop("plan_sha256", None)
    if stated != _digest(value):
        raise RecoveryError("recovery plan semantic digest mismatch")
    plan_value = dict(plan)
    if (
        plan_value.get("recovery_kind") != RECOVERY_MODE
        or plan_value.get("evidence_status", {}).get("promotion_evidence_lost") is not True
        or plan_value.get("lineage", {}).get("promotion_proof_recreated") is not False
        or plan_value.get("safety_reference", {}).get("relationship") != RECOVERY_RELATION
        or plan_value.get("promotion_policy", {}).get("dual_baseline_required") is not True
    ):
        raise RecoveryError("recovery plan weakened evidence-loss/dual-baseline policy")
    return plan_value


def _verify_plan_sources(
    plan: Mapping[str, Any],
    *,
    runtime_smoke_fn: Callable[[Path], dict[str, Any]] = _runtime_smoke,
    tool_identity_fn: Callable[[], dict[str, Any]] = _tool_identity,
) -> None:
    expected = build_plan(
        handoff_path=Path(plan["surviving_handoff"]["path"]),
        safety_reference_path=Path(plan["safety_reference"]["path"]),
        namespace=Path(plan["destinations"]["namespace"]),
        runtime_smoke_fn=runtime_smoke_fn,
        tool_identity_fn=tool_identity_fn,
    )
    if expected != plan:
        raise RecoveryError("recovery sources no longer rebuild the prepared plan")


def _render_registry(plan: Mapping[str, Any]) -> bytes:
    timestamp = float(plan["recovery_timestamp"])
    recovered = plan["recovered_checkpoint"]
    safety = plan["safety_reference"]
    common = {
        "recovery_schema": RECOVERY_SCHEMA,
        "recovery_plan_sha256": plan["plan_sha256"],
        "recovery_lineage_id": plan["lineage"]["lineage_id"],
        "promotion_proof_recreated": False,
        "recovery_receipt": plan["destinations"]["receipt"],
    }
    deployed_identity = _digest(
        {
            "schema_version": "a1-deployed-agent-search-config-v1",
            "checkpoint": {
                "path": recovered["path"],
                "sha256": recovered["sha256"],
            },
            "search_config": plan["producer_identity"]["search_config"],
        }
    )
    generator = {
        "role": RECOVERED_GENERATOR_ROLE,
        "checkpoint_path": recovered["path"],
        "md5": recovered["md5"],
        "version": EXPECTED_V5_VERSION,
        "updated_at": timestamp,
        "provenance": {
            **common,
            "evidence_status": "checkpoint_and_search_identity_only",
            "agent_identity_sha256": plan["producer_identity"]["agent_identity_sha256"],
            "search_config": plan["producer_identity"]["search_config"],
            # These are the exact field names used by the canonical fleet
            # evaluator.  This is a fresh deployed-operator binding derived
            # from the surviving handoff, not a claim that the lost promotion
            # receipt was recreated.
            "a1_candidate_agent_identity_sha256": deployed_identity,
            "a1_candidate_search_config": plan["producer_identity"]["search_config"],
        },
    }
    roles = {RECOVERED_GENERATOR_ROLE: generator}
    transitions = [
        {
            "ts": timestamp,
            "kind": "disaster_recovery_set_role",
            "role": RECOVERED_GENERATOR_ROLE,
            "reason": "exact v5 generator recovery with explicit promotion-evidence loss",
            "from_pointer": None,
            "to_pointer": generator,
        }
    ]
    for role in SAFETY_ROLES:
        pointer = {
            "role": role,
            "checkpoint_path": safety["path"],
            "md5": safety["md5"],
            "version": EXPECTED_F7_VERSION,
            "updated_at": timestamp,
            "provenance": {
                **common,
                "relationship": RECOVERY_RELATION,
                "causal_parent_proven": False,
                "role_reason": "retain authenticated f7 as public safety baseline",
            },
        }
        roles[role] = pointer
        transitions.append(
            {
                "ts": timestamp,
                "kind": "disaster_recovery_set_role",
                "role": role,
                "reason": "f7 safety role retained; v5 historical promotion proof unavailable",
                "from_pointer": None,
                "to_pointer": pointer,
            }
        )
    pool_entry = {
        "checkpoint_path": safety["path"],
        "md5": safety["md5"],
        "version": EXPECTED_F7_VERSION,
        "added_at": timestamp,
        "status": "active",
        "provenance": {**common, "relationship": RECOVERY_RELATION},
    }
    transitions.append(
        {
            "ts": timestamp,
            "kind": "pool_append",
            "role": "opponent_pool",
            "reason": "f7 recovery safety reference",
            "from_pointer": None,
            "to_pointer": pool_entry,
        }
    )
    state = {
        "roles": roles,
        "opponent_pool": [pool_entry],
        "transitions": transitions,
        # This is a new recovery lineage.  Version=5 is not promotion_count=5.
        "promotion_counts": {},
    }
    return json.dumps(state, indent=2, sort_keys=True).encode("utf-8")


def _pointer_bytes(plan: Mapping[str, Any]) -> bytes:
    return (str(plan["recovered_checkpoint"]["path"]) + "\n").encode("utf-8")


def _journal_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": JOURNAL_SCHEMA,
        "status": "prepared",
        "plan": dict(plan),
        "registry_sha256": _sha256_bytes(_render_registry(plan)),
        "current_pointer_sha256": _sha256_bytes(_pointer_bytes(plan)),
        "prepared_unix_ns": int(plan["recovery_unix_ns"]),
    }
    value["journal_sha256"] = _digest(value)
    return value


def _verify_journal(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != JOURNAL_SCHEMA:
        raise RecoveryError("prepared recovery journal schema drift")
    unsigned = dict(value)
    stated = unsigned.pop("journal_sha256", None)
    if stated != _digest(unsigned):
        raise RecoveryError("prepared recovery journal digest mismatch")
    plan = _verify_plan(value.get("plan", {}))
    if value != _journal_payload(plan):
        raise RecoveryError("prepared recovery journal does not deterministically replay")
    return value


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _publication_lock(namespace: Path):
    """Hold the one canonical recovery lock across every publication step.

    The lock deliberately lives beside the namespace: the namespace does not
    exist before the first commit, while its canonical parent does.  Keeping
    the descriptor open until the final receipt has replayed makes both first
    publication and exact crash-resume single-writer transactions.
    """

    parent = namespace.parent
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise RecoveryError("recovery namespace parent is not canonical")
    lock_path = parent / RECOVERY_LOCK_BASENAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RecoveryError(
            f"cannot open canonical recovery publication lock: {error}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RecoveryError("recovery publication lock is not a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise RecoveryError("recovery publication lock mode must be exactly 0600")
        live = lock_path.stat(follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) != (live.st_dev, live.st_ino):
            raise RecoveryError("recovery publication lock was replaced while opened")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RecoveryError("recovery publication lock is already held") from error
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            f"pid={os.getpid()} namespace={namespace}\n".encode("utf-8"),
        )
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve(strict=True) != path.parent:
        raise RecoveryError(f"recovery destination parent is not canonical: {path.parent}")
    temporary = path.with_name(f".{path.name}.publish.{os.getpid()}.{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_exact(path: Path, payload: bytes, *, mode: int, where: str) -> None:
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or _stable_read(path, where=where)[0] != payload
            or path.stat().st_mode & 0o777 != mode
        ):
            raise RecoveryError(f"existing {where} differs from prepared bytes")
        return
    try:
        _write_exclusive(path, payload, mode=mode)
    except FileExistsError:
        if (
            path.is_symlink()
            or not path.is_file()
            or _stable_read(path, where=where)[0] != payload
            or path.stat().st_mode & 0o777 != mode
        ):
            raise RecoveryError(f"racing {where} differs from prepared bytes")


def commit(
    plan: Mapping[str, Any],
    *,
    runtime_smoke_fn: Callable[[Path], dict[str, Any]] = _runtime_smoke,
    tool_identity_fn: Callable[[], dict[str, Any]] = _tool_identity,
) -> dict[str, Any]:
    """Publish or exactly resume one prepared recovery transaction."""

    plan = _verify_plan(plan)
    namespace = Path(plan["destinations"]["namespace"])
    with _publication_lock(namespace):
        if namespace.exists():
            journal_path = Path(plan["destinations"]["prepared_journal"])
            if not journal_path.is_file():
                raise RecoveryError(
                    "existing recovery namespace has no exact prepared journal"
                )
        else:
            try:
                namespace.mkdir(mode=0o700)
            except FileExistsError:
                raise RecoveryError(
                    "recovery namespace raced despite the canonical lock"
                ) from None
            if namespace.is_symlink() or namespace.resolve(strict=True) != namespace:
                raise RecoveryError("recovery namespace creation was redirected")
        # build_plan requires a fresh namespace, so source replay is performed
        # directly here and compared to the prepared immutable identities.
        evidence, snapshot, _ = _handoff_evidence(
            Path(plan["surviving_handoff"]["path"])
        )
        safety, safety_bytes = _safety_reference(
            Path(plan["safety_reference"]["path"])
        )
        snapshot[Path(safety["path"])] = safety_bytes
        if (
            evidence["surviving_handoff"] != plan["surviving_handoff"]
            or evidence["producer_identity"] != plan["producer_identity"]
            or evidence["lost_claims_from_surviving_handoff"]
            != plan["lost_claims_from_surviving_handoff"]
            or evidence["recovered_checkpoint"] != plan["recovered_checkpoint"]
            or safety != plan["safety_reference"]
            or runtime_smoke_fn(Path(plan["recovered_checkpoint"]["path"]))
            != plan["runtime_smoke"]
            or tool_identity_fn() != plan["source_identity"]
        ):
            raise RecoveryError(
                "recovery input/runtime/source identity drifted after planning"
            )
        _revalidate_snapshot(snapshot)
        journal = _journal_payload(plan)
        journal_path = Path(plan["destinations"]["prepared_journal"])
        journal_bytes = (
            json.dumps(journal, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        _publish_exact(
            journal_path, journal_bytes, mode=0o444, where="prepared journal"
        )
        loaded_journal, _ = _stable_json(journal_path, where="prepared journal")
        _verify_journal(loaded_journal)
        registry_path = Path(plan["destinations"]["registry"])
        pointer_path = Path(plan["destinations"]["current_pointer"])
        receipt_path = Path(plan["destinations"]["receipt"])
        registry_bytes = _render_registry(plan)
        pointer_bytes = _pointer_bytes(plan)
        _publish_exact(
            registry_path, registry_bytes, mode=0o600, where="recovery registry"
        )
        _publish_exact(
            pointer_path, pointer_bytes, mode=0o600, where="recovery pointer"
        )
        receipt: dict[str, Any] = {
            **plan,
            "mode": "committed",
            "prepared_journal": {
                "path": str(journal_path),
                "sha256": _sha256_bytes(journal_bytes),
                "journal_sha256": journal["journal_sha256"],
            },
            "registry": {
                "path": str(registry_path),
                "sha256": journal["registry_sha256"],
            },
            "current_pointer": {
                "path": str(pointer_path),
                "sha256": journal["current_pointer_sha256"],
            },
            "committed_unix_ns": journal["prepared_unix_ns"],
        }
        receipt["recovery_receipt_sha256"] = _digest(receipt)
        receipt_bytes = (
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        _publish_exact(
            receipt_path, receipt_bytes, mode=0o444, where="recovery receipt"
        )
        return verify_committed_receipt(
            receipt_path,
            runtime_smoke_fn=runtime_smoke_fn,
            tool_identity_fn=tool_identity_fn,
        )["receipt"]


def verify_committed_receipt(
    receipt_path: Path,
    *,
    runtime_smoke_fn: Callable[[Path], dict[str, Any]] = _runtime_smoke,
    tool_identity_fn: Callable[[], dict[str, Any]] = _tool_identity,
) -> dict[str, Any]:
    """Canonical verifier API for coordinator, learner, and wave consumers.

    Returns a compact authority projection plus the fully replayed receipt.
    It intentionally does not return a promotion receipt or post-promotion
    handoff shape, so standard promotion consumers continue to reject it.
    """

    receipt_path = _canonical_existing(receipt_path, where="v5 recovery receipt")
    receipt, receipt_bytes = _stable_json(receipt_path, where="v5 recovery receipt")
    if receipt.get("schema_version") != RECOVERY_SCHEMA or receipt.get("mode") != "committed":
        raise RecoveryError("v5 recovery receipt schema/mode drift")
    unsigned = dict(receipt)
    stated = unsigned.pop("recovery_receipt_sha256", None)
    if stated != _digest(unsigned):
        raise RecoveryError("v5 recovery receipt semantic digest mismatch")
    plan = dict(receipt)
    for field in (
        "prepared_journal",
        "registry",
        "current_pointer",
        "committed_unix_ns",
        "recovery_receipt_sha256",
    ):
        plan.pop(field, None)
    plan["mode"] = "dry-run"
    _verify_plan(plan)
    journal_path = _canonical_existing(
        Path(receipt["prepared_journal"]["path"]), where="recovery prepared journal"
    )
    journal, journal_bytes = _stable_json(journal_path, where="recovery prepared journal")
    _verify_journal(journal)
    if (
        receipt["prepared_journal"]
        != {
            "path": str(journal_path),
            "sha256": _sha256_bytes(journal_bytes),
            "journal_sha256": journal["journal_sha256"],
        }
        or journal["plan"] != plan
    ):
        raise RecoveryError("recovery receipt/journal binding drift")
    registry_path = _canonical_existing(
        Path(receipt["registry"]["path"]), where="recovery registry"
    )
    pointer_path = _canonical_existing(
        Path(receipt["current_pointer"]["path"]), where="recovery current pointer"
    )
    registry_bytes, _ = _stable_read(registry_path, where="recovery registry")
    pointer_bytes, _ = _stable_read(pointer_path, where="recovery current pointer")
    if (
        registry_bytes != _render_registry(plan)
        or _sha256_bytes(registry_bytes) != receipt["registry"]["sha256"]
        or pointer_bytes != _pointer_bytes(plan)
        or _sha256_bytes(pointer_bytes) != receipt["current_pointer"]["sha256"]
    ):
        raise RecoveryError("recovery registry/current-pointer replay drift")
    # Exact source replay without requiring a fresh namespace.
    evidence, snapshot, _ = _handoff_evidence(Path(plan["surviving_handoff"]["path"]))
    safety, safety_bytes = _safety_reference(Path(plan["safety_reference"]["path"]))
    snapshot[Path(safety["path"])] = safety_bytes
    current_source_identity = tool_identity_fn()
    source_identity_replayed = current_source_identity == plan["source_identity"]
    if not source_identity_replayed and tool_identity_fn is _tool_identity:
        _verify_recorded_tool_identity(plan["source_identity"])
        source_identity_replayed = True
    if (
        evidence["surviving_handoff"] != plan["surviving_handoff"]
        or evidence["producer_identity"] != plan["producer_identity"]
        or evidence["lost_claims_from_surviving_handoff"]
        != plan["lost_claims_from_surviving_handoff"]
        or evidence["recovered_checkpoint"] != plan["recovered_checkpoint"]
        or safety != plan["safety_reference"]
        or runtime_smoke_fn(Path(plan["recovered_checkpoint"]["path"]))
        != plan["runtime_smoke"]
        or not source_identity_replayed
    ):
        raise RecoveryError("recovery receipt no longer replays exact source/runtime identity")
    _revalidate_snapshot(snapshot)
    authority = {
        "schema_version": "a1-v5-disaster-recovery-authority-v1",
        "recovery_receipt": {
            "path": str(receipt_path),
            "sha256": _sha256_bytes(receipt_bytes),
            "recovery_receipt_sha256": stated,
        },
        "recovery_lineage_id": plan["lineage"]["lineage_id"],
        "recovered_generator": dict(plan["recovered_checkpoint"]),
        RECOVERY_RELATION: dict(plan["safety_reference"]),
        "producer_identity": dict(plan["producer_identity"]),
        "promotion_proof_recreated": False,
        "dual_baseline_fresh_gate_required": True,
        "promotion_eligible": False,
        "training_proof": False,
        "wave_lineage_mode": "recovery_reference",
    }
    authority["authority_sha256"] = _digest(authority)
    return {"authority": authority, "receipt": receipt}


def _destinations_for_existing_namespace(namespace: Path) -> dict[str, str]:
    lexical = Path(os.path.abspath(os.fspath(namespace.expanduser())))
    if lexical.name != RECOVERY_NAMESPACE_BASENAME:
        raise RecoveryError(
            f"recovery namespace basename must be {RECOVERY_NAMESPACE_BASENAME!r}"
        )
    return {
        "namespace": str(lexical),
        "registry": str(lexical / "private/champion_registry.json"),
        "current_pointer": str(lexical / "private/CURRENT_CHAMPION"),
        "receipt": str(
            lexical / "private/receipts/a1-v5-disaster-recovery.receipt.json"
        ),
        "prepared_journal": str(
            lexical / "private/receipts/a1-v5-disaster-recovery.prepared.json"
        ),
    }


def resume_plan(
    *, namespace: Path, handoff_path: Path, safety_reference_path: Path
) -> dict[str, Any]:
    """Resume only the exact journal and exact source arguments."""

    destinations = _destinations_for_existing_namespace(namespace)
    journal_path = _canonical_existing(
        Path(destinations["prepared_journal"]), where="prepared recovery journal"
    )
    journal, _ = _stable_json(journal_path, where="prepared recovery journal")
    journal = _verify_journal(journal)
    plan = journal["plan"]
    if plan.get("destinations") != destinations:
        raise RecoveryError("resume namespace differs from prepared recovery")
    if (
        Path(plan["surviving_handoff"]["path"]).resolve(strict=True)
        != handoff_path.expanduser().resolve(strict=True)
        or Path(plan["safety_reference"]["path"]).resolve(strict=True)
        != safety_reference_path.expanduser().resolve(strict=True)
    ):
        raise RecoveryError("resume inputs differ from prepared recovery")
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surviving-handoff", required=True)
    parser.add_argument("--safety-reference", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    namespace = Path(args.namespace)
    try:
        destinations = _destinations_for_existing_namespace(namespace)
        if args.go and Path(destinations["prepared_journal"]).is_file():
            plan = resume_plan(
                namespace=namespace,
                handoff_path=Path(args.surviving_handoff),
                safety_reference_path=Path(args.safety_reference),
            )
        else:
            plan = build_plan(
                handoff_path=Path(args.surviving_handoff),
                safety_reference_path=Path(args.safety_reference),
                namespace=namespace,
            )
        result = commit(plan) if args.go else plan
    except (RecoveryError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
