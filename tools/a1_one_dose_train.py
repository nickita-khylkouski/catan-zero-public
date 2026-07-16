#!/usr/bin/env python3
"""Execute the sealed A1 one-dose learner transaction, fail closed.

This is the only production A1 training entry point.  It consumes a sealed
``a1-pre-wave-contract-lock-v2`` plus the audited memmap/validation sidecar,
replays their byte and seed bindings, and then constructs the exact B200
``train_bc`` invocation bound by the lock. The historical one-GPU topology is
replayable; current production may select one exact 8-GPU B200 DDP topology
with the same global batch and optimizer/sample dose. The default is a
read-only dry run; ``--go`` is required to probe every selected B200 and start
training.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack, contextmanager
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import resource
import re
import stat
import subprocess
import sys
import tempfile
import time
import socket
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from tools import a1_pre_wave_contract as a1_contract  # noqa: E402
from tools import a1_current_science_contract as current_science  # noqa: E402
from tools import a1_build_post_wave_composite as composite_builder  # noqa: E402
from tools import a1_frozen_lock_verifier as frozen_lock_verifier  # noqa: E402
from tools import train_bc  # noqa: E402
from tools import a1_lineage_dose as lineage  # noqa: E402
from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402
from tools import a1_ddp_epoch_canary as ddp_canary  # noqa: E402
from tools import a1_aux_pair_coordinator as aux_coordinator  # noqa: E402
from tools import a1_stage_c_final_replication as stage_c_final  # noqa: E402
from tools import a1_scientific_evidence as scientific_evidence  # noqa: E402
from tools import audit_entity_graph_information_surface as information_surface  # noqa: E402


RECEIPT_SCHEMA = "a1-one-dose-training-receipt-v3"
CLAIM_SCHEMA = "a1-one-dose-training-claim-v3"
RETRY_RECEIPT_SCHEMA = "a1-one-dose-training-receipt-v4"
RETRY_CLAIM_SCHEMA = "a1-one-dose-training-claim-v4"
PLAN_SCHEMA = "a1-one-dose-training-plan-v2"
REPORT_EXECUTION_BINDING_SCHEMA = "a1-one-dose-execution-binding-v1"
REPORT_EXECUTION_BINDING_FIELD = "a1_one_dose_execution_binding"
REPORT_INPUT_BINDING_SCHEMA = "a1-one-dose-input-binding-v1"
REPORT_INPUT_BINDING_FIELD = "a1_one_dose_input_binding"
REPORT_LEARNER_LINEAGE_PARENT_FIELD = "a1_learner_lineage_parent"
LEARNER_LINEAGE_PARENT_SCHEMA = "a1-learner-lineage-parent-v1"
INDEPENDENT_PARENT_AUTHORITY_SCHEMA = (
    "a1-independent-diagnostic-learner-parent-authority-v1"
)
TRAINING_TRANSACTION_SCHEMA = "a1-one-dose-training-transaction-v1"
PRODUCTION_TRAINER_AUTHORITY_SCHEMA = "a1-production-trainer-authority-v1"
PRODUCTION_TRAINER_CODE_SURFACE = tuple(
    sorted(
        train_bc.A1_REQUIRED_LEARNER_CODE_SUFFIXES
        | {
            "tools/a1_one_dose_train.py",
            "tools/a1_build_post_wave_composite.py",
            "tools/audit_entity_graph_information_surface.py",
            "tools/mixed_memmap_corpus.py",
            "tools/policy_target_reanalysis_contract.py",
        }
        # The child records every imported catan_zero module.  Bind the full
        # package before taking the durable dose claim so lazy imports cannot
        # slip different neural/loss bytes into the run after planning.
        | {
            path.relative_to(_REPO_ROOT).as_posix()
            for path in (_REPO_ROOT / "src" / "catan_zero").rglob("*.py")
        }
    )
)
RETRY_CONTRACT_SCHEMA = "a1-one-dose-learner-retry-contract-v1"
RETRY_IDENTITY_SCHEMA = "a1-one-dose-learner-retry-identity-v1"
RETRY_REPAIR_KIND = "entity_graph_graph_layers_default_4_to_checkpoint_6"
PRODUCTION_PREFLIGHT_RETRY_REPAIR_KIND = (
    "production_ddp_preflight_metadata_set_json_serialization"
)
# This pair is intentionally byte-specific.  The first trainer consumed the
# original claim after rank 0 returned Python ``set`` objects in production
# source-authority metadata and then failed while publishing the JSON DDP
# preflight.  The second trainer converts those two values to sorted lists.
# A future train_bc edit cannot silently reuse this one-off repair authority.
BUGGY_PRODUCTION_PREFLIGHT_TRAINER_SHA256 = (
    "sha256:ad22fe0a7ea6cf05816d600e96360d501d6aca7ca52d77ed62dfc4cf43f25386"
)
FIXED_PRODUCTION_PREFLIGHT_TRAINER_SHA256 = (
    "sha256:6fa6bc031aba2f8949d36d0865ea4497450ae760a5cdf227840a9d922a1a2ec9"
)
PRODUCTION_PREFLIGHT_TRANSPORT_RETRY_REPAIR_KIND = (
    "production_ddp_preflight_tcpstore_chunked_metadata_transport"
)
# The first typed retry reached the production preflight but tried to publish
# its >8 MiB authenticated metadata as one TCPStore value.  These byte
# identities make the second (and final) zero-step repair specific to that
# failed trainer and the chunked, digest-bound transport implementation.
BUGGY_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256 = (
    FIXED_PRODUCTION_PREFLIGHT_TRAINER_SHA256
)
FIXED_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256 = (
    "sha256:533abdc210a86de7e1accce7531d008216599bd758f9147b4574ad11e8c5a6fd"
)
ABLATION_RECEIPT_SCHEMA = "a1-learner-ablation-training-receipt-v1"
ABLATION_CLAIM_SCHEMA = "a1-learner-ablation-training-claim-v1"
UPGRADE_RECEIPT_SCHEMA = "a1-architecture-upgrade-training-receipt-v1"
UPGRADE_CLAIM_SCHEMA = "a1-architecture-upgrade-training-claim-v1"
CENTRAL_RECEIPT_SCHEMA = "a1-central-learner-training-receipt-v1"
CENTRAL_CLAIM_SCHEMA = "a1-central-learner-training-claim-v1"
CLAIM_DIRECTORY = ".a1-one-dose-training-claims"
MIN_NOFILE = 65_536
MAX_IDLE_GPU_MEMORY_MIB = 64
MAX_DDP_CANARY_AGE_NS = 60 * 60 * 1_000_000_000
DATA_LOADER_WORKERS = 2
DATA_LOADER_PREFETCH = 2
LEGACY_SINGLE_GPU_TOPOLOGY = "legacy-single-gpu"
B200_8GPU_DDP_TOPOLOGY = "b200-8gpu-ddp"
B200_8GPU_DDP_GPUS = tuple(range(8))
TRAINING_TOPOLOGIES: dict[str, dict[str, Any]] = {
    LEGACY_SINGLE_GPU_TOPOLOGY: {
        "world_size": 1,
        "local_batch_size": 4096,
        "grad_accum_steps": 1,
        "global_batch_size": 4096,
    },
    B200_8GPU_DDP_TOPOLOGY: {
        "world_size": 8,
        "local_batch_size": 512,
        "grad_accum_steps": 1,
        "global_batch_size": 4096,
    },
}
EVENT_HISTORY_ACK_FLAG = "--acknowledge-empty-event-history-payload-inventory-sha256"
EVENT_HISTORY_CROP_FLAG = "--crop-authenticated-empty-event-history"
TRUSTED_A1_LOCK_FILE_SHA256 = (
    "sha256:8301c7547e1745812c69ca04934424755c7116eb5e221688abc58c1bcb7a3122"
)
TRUSTED_A1_LOCK_PATH = Path(
    "/home/ubuntu/catan-zero/runs/rl_program_20260710/"
    "a1_infoset_n128_v133/contract.lock.json"
)
TRUSTED_A1_VERIFIER_PATH = Path(
    "/home/ubuntu/catan-zero-v1/tools/a1_pre_wave_contract.py"
)
TRUSTED_A1_VERIFIER_SHA256 = (
    "sha256:45594de3835242904a7c3257c5ff644531c4a3c70a447880b20b3b1a23d8c9cc"
)
# The current v5 production composite is sealed to this recovery lock and its
# immutable verifier.  This is a second explicit trust anchor, not a mutable
# replacement for the historical 20260710 anchor above.
V5_PRODUCTION_A1_LOCK_FILE_SHA256 = (
    "sha256:ce9553336a4b5e25311e4aad40e307e713652d6d11a779d830ce8ea28b05dee2"
)
V5_PRODUCTION_A1_LOCK_PATH = Path(
    "/home/ubuntu/catan-zero-production/private/"
    "a1-v5-recovery-n128-p4-64000games-64gpu-20260714-r2/lock.json"
)
V5_PRODUCTION_A1_VERIFIER_PATH = Path(
    "/home/ubuntu/catan-zero-wave-5ba993a/tools/a1_pre_wave_contract.py"
)
V5_PRODUCTION_A1_VERIFIER_SHA256 = (
    "sha256:ab5d4ef8d4a3f82ecacb6c94ff613e24041ec9d1d4e2722ae6c65a19220f101c"
)
V5_PRODUCTION_CONTRACT_ID = "a1-v5-recovery-n128-p4-64000games-64gpu-20260714-r2"
V5_PRODUCTION_CONTRACT_SHA256 = (
    "sha256:2becf946235fb55dff606b90906acee6c6933eba21d507d49a258690a371891a"
)
V5_PRODUCTION_PARENT_SHA256 = (
    "sha256:6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c"
)
SEALED_A1_MODEL_CLI: dict[str, str] = {
    "--hidden-size": "640",
    "--graph-layers": "6",
    "--attention-heads": "8",
    "--graph-dropout": "0.05",
    "--entity-state-trunk": "transformer",
    "--relational-block-pattern": "",
    "--relational-ff-size": "0",
    "--relational-bases": "4",
    "--relational-action-cross-layers": "1",
    "--latent-deliberation-steps": "0",
    "--latent-deliberation-slots": "8",
    "--moe-routed-experts": "0",
    "--moe-top-k": "2",
    "--moe-expert-ff-size": "0",
}
SEALED_A1_MODEL_REPORT: dict[str, int | float] = {
    "hidden_size": 640,
    "graph_layers": 6,
    "attention_heads": 8,
    "graph_dropout": 0.05,
}
ACCEPTABLE_COMPUTE_MODES = frozenset({"DEFAULT", "EXCLUSIVE_PROCESS"})
MPS_EXECUTABLES = frozenset({"nvidia-cuda-mps-control", "nvidia-cuda-mps-server"})
CHILD_ENVIRONMENT_KEYS = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "TMPDIR",
        "TZ",
    }
)

# Ablations may change learner optimization/loss semantics only. Corpus,
# validation, architecture, topology, checkpoint, masking, and audit bindings
# remain sealed by the original contract and are never accepted as overrides.
A1_LEARNER_ABLATION_FIELDS = frozenset(
    {
        "epochs",
        "max_steps",
        "lr",
        "lr_warmup_steps",
        "lr_schedule",
        "value_lr_mult",
        "public_card_lr_mult",
        "trunk_lr_mult",
        "value_trunk_grad_scale",
        "policy_loss_weight",
        "policy_aux_active_batch_size",
        "soft_target_source",
        "soft_target_weight",
        "soft_target_temperature",
        "soft_target_min_legal_coverage",
        "value_loss_weight",
        "final_vp_loss_weight",
        "aux_subgoal_loss_weight",
        "policy_kl_anchor_weight",
        "policy_kl_anchor_direction",
        "policy_kl_target",
        "policy_kl_dual_lr",
        "policy_kl_max_weight",
        "policy_surprise_weight",
        "per_game_policy_surprise_weighting",
        "advantage_policy_weighting",
        "per_game_policy_weight",
        "per_game_policy_weight_mode",
        "per_game_value_weight",
        "per_game_value_weight_mode",
        "vp_margin_weight",
        "truncated_vp_margin_value_weight",
        "forced_action_weight",
        "forced_row_value_weight",
        "forced_row_value_action_type_weights",
        "winner_sample_weight",
        "loser_sample_weight",
    }
)

LEGACY_AUX_REGULARIZATION_MODULE = architecture_upgrade.MODULE_AUX_SUBGOAL_HEADS
AUX_REGULARIZATION_MODULE = architecture_upgrade.MODULE_AUX_SUBGOAL_POINTER_HEADS
AUX_CONTROL_ARM = "AUX0"
AUX_TREATMENT_ARM = "AUXT"
AUX_SELECTED_SAMPLE_DOSE = 524_288
AUX_SELECTED_OPTIMIZER_STEPS = 128
AUX_WARMUP_OPTIMIZER_STEPS = 128
AUX_WARMUP_TRAINABLE_PREFIXES = (
    "aux_longest_road_head",
    "aux_largest_army_head",
    "aux_vp_in_n_head",
    "aux_next_settlement_pointer_head",
    "aux_robber_target_head",
)
AUX_SUBGOAL_HEAD_MODULES = frozenset(
    {
        "aux_longest_road_head",
        "aux_largest_army_head",
        "aux_vp_in_n_head",
        "aux_next_settlement_pointer_head",
        "aux_robber_target_head",
    }
)
AUX_SUBGOAL_TARGET_FIELDS = frozenset(
    {
        "aux_longest_road",
        "aux_largest_army",
        "aux_vp_in_n",
        "aux_next_settlement",
        "aux_robber_target",
    }
)
P1_CENTRAL_ARMS = frozenset({"K0", "K3", "K10"})
CENTRAL_LEARNER_BINDING_SCHEMA = "a1-central-learner-binding-v3"
LOCK_VERIFICATION_CACHE_SCHEMA = "a1-lock-verification-cache-v1"
LOCK_VERIFICATION_CACHE_VALIDATOR_VERSION = 1
LOCK_VERIFICATION_CACHE_ENV = "A1_ONE_DOSE_LOCK_VERIFICATION_CACHE_DIR"


class ExecutorError(RuntimeError):
    """A fail-closed A1 executor refusal."""


def _reviewed_lock_trust_anchors() -> dict[str, dict[str, Any]]:
    """Return explicit replay authorities for issued production locks.

    The historical constants remain separate so existing replay tests and
    issued receipts retain their original trust identity.
    """

    return {
        TRUSTED_A1_LOCK_FILE_SHA256: {
            "lock_path": TRUSTED_A1_LOCK_PATH,
            "verifier_path": TRUSTED_A1_VERIFIER_PATH,
            "verifier_sha256": TRUSTED_A1_VERIFIER_SHA256,
            "contract_id": None,
            "contract_sha256": None,
            "producer_sha256": None,
        },
        V5_PRODUCTION_A1_LOCK_FILE_SHA256: {
            "lock_path": V5_PRODUCTION_A1_LOCK_PATH,
            "verifier_path": V5_PRODUCTION_A1_VERIFIER_PATH,
            "verifier_sha256": V5_PRODUCTION_A1_VERIFIER_SHA256,
            "contract_id": V5_PRODUCTION_CONTRACT_ID,
            "contract_sha256": V5_PRODUCTION_CONTRACT_SHA256,
            "producer_sha256": V5_PRODUCTION_PARENT_SHA256,
        },
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _stable_canonical_regular_file(
    value: str | Path, *, where: str
) -> tuple[Path, str]:
    """Resolve one exact artifact once and reject symlink/TOCTOU ambiguity."""

    lexical = Path(value).expanduser()
    try:
        resolved = lexical.resolve(strict=True)
        lexical_absolute = lexical.absolute()
        before = resolved.stat()
    except OSError as error:
        raise ExecutorError(f"cannot inspect {where}: {error}") from error
    if (
        str(resolved) != str(value)
        or lexical_absolute != resolved
        or lexical.is_symlink()
        or not stat.S_ISREG(before.st_mode)
    ):
        raise ExecutorError(f"{where} must be one canonical regular-file path")
    digest = _file_sha256(resolved)
    after = resolved.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise ExecutorError(f"{where} changed while it was authenticated")
    return resolved, digest


def _current_production_trainer_authority() -> dict[str, Any]:
    """Authenticate the reviewed trainer in this executor's checkout.

    A frozen repository supplied on the CLI is authority only for replaying the
    historical generation lock.  Post-wave composites intentionally train with
    the reviewed learner in the current immutable checkout, so the executable
    trainer gets its own independent path-and-byte authority.
    """

    expected = (_REPO_ROOT / "tools" / "train_bc.py").resolve(strict=True)
    imported = Path(train_bc.__file__).resolve(strict=True)
    if imported != expected:
        raise ExecutorError(
            "production trainer import does not come from the current executor checkout"
        )
    trainer, trainer_sha256 = _stable_canonical_regular_file(
        expected, where="current production trainer"
    )
    code_surface = []
    for relative_path in PRODUCTION_TRAINER_CODE_SURFACE:
        code_file, code_sha256 = _stable_canonical_regular_file(
            _REPO_ROOT / relative_path,
            where=f"production trainer dependency {relative_path}",
        )
        code_surface.append(
            {
                "relative_path": relative_path,
                "path": str(code_file),
                "sha256": code_sha256,
            }
        )
    authority: dict[str, Any] = {
        "schema_version": PRODUCTION_TRAINER_AUTHORITY_SCHEMA,
        "repository_root": str(_REPO_ROOT.resolve(strict=True)),
        "relative_path": "tools/train_bc.py",
        "path": str(trainer),
        "sha256": trainer_sha256,
        "code_surface": code_surface,
        "code_surface_sha256": _value_sha256(code_surface),
    }
    authority["authority_sha256"] = _value_sha256(authority)
    return authority


def _require_current_production_trainer_authority(
    verified: Mapping[str, Any], *, command: Sequence[str] | None = None
) -> dict[str, Any] | None:
    """Reject production trainer path/byte drift before a dose is claimed."""

    if verified.get("data_kind") != "production_composite_v2":
        return None
    expected = _current_production_trainer_authority()
    if verified.get("trainer_authority") != expected:
        raise ExecutorError(
            "production trainer authority drifted from the current reviewed trainer"
        )
    if command is not None:
        trainer_tokens = [
            token
            for token in command
            if isinstance(token, str) and Path(token).name == "train_bc.py"
        ]
        if trainer_tokens != [expected["path"]]:
            raise ExecutorError(
                "production command does not execute the bound current trainer"
            )
    return copy.deepcopy(expected)


def _repo_relative_code_path(path: str) -> Path:
    parts = Path(path).parts
    for marker in ("configs", "native", "src", "tools", "vendor"):
        if marker in parts:
            return Path(*parts[parts.index(marker) :])
    raise ExecutorError(f"cannot map sealed code path into current repository: {path}")


def _current_ablation_code_binding(lock: dict[str, Any]) -> dict[str, Any]:
    """Rebind the sealed learner/runtime inventory to this reviewed checkout."""

    provenance = lock.get("provenance")
    if not isinstance(provenance, dict):
        raise ExecutorError("sealed A1 contract has no code provenance")
    records_by_relative_path: dict[str, dict[str, str]] = {}
    for section, kind in (
        ("learner_code", "learner_code"),
        ("runtime_code_tree", "runtime_code"),
    ):
        source = provenance.get(section)
        if not isinstance(source, list) or not source:
            raise ExecutorError(f"sealed A1 contract has no {section} inventory")
        for old in source:
            if not isinstance(old, dict) or not isinstance(old.get("path"), str):
                raise ExecutorError(f"sealed A1 {section} record is malformed")
            relative = _repo_relative_code_path(old["path"])
            current = (_REPO_ROOT / relative).resolve(strict=True)
            relative_key = relative.as_posix()
            record = {
                "kind": kind,
                "relative_path": relative_key,
                "path": str(current),
                "sha256": _file_sha256(current),
            }
            prior = records_by_relative_path.get(relative_key)
            # Learner-code semantics dominate when the same file also appears
            # in the transitive runtime inventory. The bytes/path must agree;
            # otherwise the sealed inventories are internally contradictory.
            if prior is not None and (
                prior["path"] != record["path"] or prior["sha256"] != record["sha256"]
            ):
                raise ExecutorError(
                    f"conflicting A1 code provenance for {relative_key}"
                )
            if prior is None or kind == "learner_code":
                records_by_relative_path[relative_key] = record
    # These files are causal AUX runtime authority even if an older sealed lock
    # predates one of them.  Bind them explicitly so changing the receipt
    # allowlist, DDP canary, trainer, or model head implementation changes the
    # portable matched-pair identity.
    required_runtime_paths = (
        "src/catan_zero/rl/entity_token_policy.py",
        "tools/a1_ddp_epoch_canary.py",
        "tools/a1_function_preserving_upgrade.py",
        "tools/a1_one_dose_train.py",
        "tools/a1_stage_c_final_replication.py",
        "tools/train_bc.py",
    )
    for relative_key in required_runtime_paths:
        current = (_REPO_ROOT / relative_key).resolve(strict=True)
        records_by_relative_path[relative_key] = {
            "kind": "learner_code",
            "relative_path": relative_key,
            "path": str(current),
            "sha256": _file_sha256(current),
        }
    records = list(records_by_relative_path.values())
    records.sort(key=lambda row: (row["kind"], row["relative_path"]))
    binding = {
        "schema_version": "a1-learner-ablation-code-binding-v1",
        "repository_root": str(_REPO_ROOT.resolve(strict=True)),
        "records": records,
    }
    binding["code_tree_sha256"] = _value_sha256(binding)
    return binding


def _verify_completed_ablation_code_binding(
    binding: Mapping[str, Any], *, lock: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate the immutable checkout used by a completed ablation.

    Completion replay cannot substitute the verifier's newer checkout for the
    code bytes that actually rendered and ran the learner.  This verifier
    rebuilds the required relative-path inventory from the sealed lock, then
    proves every recorded byte beneath the receipt-bound historical checkout.
    It is deliberately not exposed by the launch CLI.
    """

    if not isinstance(binding, dict) or set(binding) != {
        "schema_version",
        "repository_root",
        "records",
        "code_tree_sha256",
    }:
        raise ExecutorError("completed ablation code binding shape drift")
    unsigned = dict(binding)
    stated = unsigned.pop("code_tree_sha256", None)
    if binding.get(
        "schema_version"
    ) != "a1-learner-ablation-code-binding-v1" or stated != _value_sha256(unsigned):
        raise ExecutorError("completed ablation code binding digest drift")
    root_ref = binding.get("repository_root")
    if not isinstance(root_ref, str) or not Path(root_ref).is_absolute():
        raise ExecutorError("completed ablation repository root is not absolute")
    root_lexical = Path(root_ref).expanduser()
    if root_lexical.is_symlink() or not root_lexical.is_dir():
        raise ExecutorError("completed ablation repository root is unavailable")
    try:
        root = root_lexical.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot resolve completed ablation repository root: {error}"
        ) from error
    if str(root) != root_ref:
        raise ExecutorError("completed ablation repository root is not canonical")

    provenance = lock.get("provenance")
    if not isinstance(provenance, dict):
        raise ExecutorError("sealed A1 contract has no code provenance")
    expected_kinds: dict[str, str] = {}
    for section, kind in (
        ("learner_code", "learner_code"),
        ("runtime_code_tree", "runtime_code"),
    ):
        source = provenance.get(section)
        if not isinstance(source, list) or not source:
            raise ExecutorError(f"sealed A1 contract has no {section} inventory")
        for old in source:
            if not isinstance(old, dict) or not isinstance(old.get("path"), str):
                raise ExecutorError(f"sealed A1 {section} record is malformed")
            relative = _repo_relative_code_path(old["path"]).as_posix()
            if relative not in expected_kinds or kind == "learner_code":
                expected_kinds[relative] = kind
    for relative in (
        "src/catan_zero/rl/entity_token_policy.py",
        "tools/a1_ddp_epoch_canary.py",
        "tools/a1_function_preserving_upgrade.py",
        "tools/a1_one_dose_train.py",
        "tools/train_bc.py",
    ):
        expected_kinds[relative] = "learner_code"

    raw_records = binding.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ExecutorError("completed ablation code binding has no records")
    observed: dict[str, dict[str, str]] = {}
    for raw in raw_records:
        if not isinstance(raw, dict) or set(raw) != {
            "kind",
            "relative_path",
            "path",
            "sha256",
        }:
            raise ExecutorError("completed ablation code record shape drift")
        kind = raw.get("kind")
        relative_ref = raw.get("relative_path")
        path_ref = raw.get("path")
        digest = raw.get("sha256")
        if (
            kind not in {"learner_code", "runtime_code"}
            or not isinstance(relative_ref, str)
            or not relative_ref
            or Path(relative_ref).is_absolute()
            or ".." in Path(relative_ref).parts
            or not isinstance(path_ref, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest)) is None
            or relative_ref in observed
        ):
            raise ExecutorError("completed ablation code record is invalid")
        expected_path = root / relative_ref
        lexical_path = Path(path_ref).expanduser()
        if lexical_path.is_symlink() or not lexical_path.is_file():
            raise ExecutorError(
                f"completed ablation code file is unavailable: {relative_ref}"
            )
        try:
            actual_path = lexical_path.resolve(strict=True)
        except OSError as error:
            raise ExecutorError(
                f"cannot resolve completed ablation code file {relative_ref}: {error}"
            ) from error
        if (
            actual_path != expected_path
            or path_ref != str(expected_path)
            or _file_sha256(actual_path) != digest
        ):
            raise ExecutorError(
                f"completed ablation code bytes/path drift: {relative_ref}"
            )
        observed[relative_ref] = dict(raw)
    if {
        relative: record["kind"] for relative, record in observed.items()
    } != expected_kinds:
        raise ExecutorError(
            "completed ablation code inventory differs from sealed lock"
        )
    return copy.deepcopy(binding)


def _portable_ablation_code_identity(
    binding: dict[str, Any],
) -> dict[str, Any]:
    """Project reviewed trainer bytes without host-local repository paths."""

    raw_records = binding.get("records")
    if not isinstance(raw_records, list):
        raise ExecutorError("learner ablation code binding has no records")
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ExecutorError("learner ablation code record is malformed")
        kind = raw.get("kind")
        relative_path = raw.get("relative_path")
        sha256 = raw.get("sha256")
        if (
            kind not in {"learner_code", "runtime_code"}
            or not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(sha256)) is None
        ):
            raise ExecutorError("learner ablation portable code record is invalid")
        key = (str(kind), relative_path)
        if key in seen:
            raise ExecutorError("learner ablation portable code record is duplicated")
        seen.add(key)
        records.append(
            {
                "kind": str(kind),
                "relative_path": relative_path,
                "sha256": str(sha256),
            }
        )
    records.sort(key=lambda row: (row["kind"], row["relative_path"]))
    identity: dict[str, Any] = {
        "schema_version": "a1-portable-learner-code-identity-v1",
        "records": records,
    }
    identity["code_sha256"] = _value_sha256(identity)
    return identity


def _portable_upgrade_identity(upgrade: dict[str, Any]) -> dict[str, Any]:
    """Project function-preserving initializer evidence across host paths."""

    try:
        identity = {
            "schema_version": "a1-portable-function-upgrade-identity-v1",
            "module": str(upgrade["module"]),
            "source_checkpoint_sha256": str(upgrade["source"]["sha256"]),
            "initializer_sha256": str(upgrade["upgraded_initializer"]["sha256"]),
            "flags": copy.deepcopy(upgrade["flags"]),
            "initialization_seed": upgrade["initialization_seed"],
            "forward_max_diff": upgrade["forward_max_diff"],
            "forward_identical_at_init": upgrade["forward_identical_at_init"],
            "shared_parameters_bit_identical": upgrade[
                "shared_parameters_bit_identical"
            ],
            "shared_parameter_count": upgrade["shared_parameter_count"],
            "new_parameters": list(upgrade["new_parameters"]),
            "new_parameter_initialization": copy.deepcopy(
                upgrade["new_parameter_initialization"]
            ),
            "effective_source_config_sha256": str(
                upgrade["effective_source_config_sha256"]
            ),
            "effective_upgraded_config_sha256": str(
                upgrade["effective_upgraded_config_sha256"]
            ),
            "seeded_parameter_sha256": copy.deepcopy(
                upgrade.get("seeded_parameter_sha256", {})
            ),
        }
    except (KeyError, TypeError) as error:
        raise ExecutorError(
            "function-preserving upgrade identity is incomplete"
        ) from error
    digest_values = [
        identity["source_checkpoint_sha256"],
        identity["initializer_sha256"],
        identity["effective_source_config_sha256"],
        identity["effective_upgraded_config_sha256"],
        *identity["seeded_parameter_sha256"].values(),
    ]
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
        for value in digest_values
    ):
        raise ExecutorError("function-preserving upgrade digest is invalid")
    identity["identity_sha256"] = _value_sha256(identity)
    return identity


def _lexical_python_executable(path: Path) -> Path:
    """Validate an interpreter without resolving away its virtual environment.

    ``venv/bin/python`` is normally a symlink to the base interpreter.  Invoking
    the resolved target bypasses the adjacent ``pyvenv.cfg`` and silently drops
    the venv's Torch/dependencies.  Preserve the absolute lexical path for argv
    while still proving that its target is a real executable file.
    """

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        target = lexical.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(f"cannot resolve learner python: {error}") from error
    if (
        not lexical.is_file()
        or not target.is_file()
        or not os.access(lexical, os.X_OK)
        or not os.access(target, os.X_OK)
    ):
        raise ExecutorError(f"python is not executable: {lexical}")
    return lexical


def _fsync_directory(path: Path) -> None:
    """Durably publish directory-entry changes or fail closed."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(path: Path) -> None:
    """Create ``path`` and sync every newly-created directory entry."""

    path = Path(path)
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ExecutorError(f"expected directory, found non-directory: {path}")
    for created in reversed(missing):
        _fsync_directory(created)
        _fsync_directory(created.parent)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _with_digest(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = _value_sha256(result)
    return result


def _selected_gpus(verified: dict[str, Any], *, fallback_gpu: int) -> tuple[int, ...]:
    topology = verified.get("training_topology")
    raw = [fallback_gpu] if topology is None else topology.get("physical_gpus")
    if (
        not isinstance(raw, list)
        or not raw
        or any(
            isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in raw
        )
        or len(set(raw)) != len(raw)
    ):
        raise ExecutorError("training topology has invalid physical GPU ownership")
    gpus = tuple(raw)
    world_size = int(verified.get("recipe", {}).get("world_size", len(gpus)))
    if len(gpus) != world_size:
        raise ExecutorError(
            "training topology GPU count differs from learner world size"
        )
    return gpus


def _child_environment(
    gpu: int | Sequence[int], *, repository_root: Path | None = None
) -> dict[str, str]:
    """Return the complete, secret-free environment for the learner child.

    Do not start from ``os.environ``: an operator shell may contain distributed,
    CUDA, Python, proxy, credential, or preload variables that silently change
    the one-dose process.  HOME comes from the operating-system account record,
    not the ambient HOME variable, and every other entry is an explicit value.
    """

    raw_gpus = (
        [gpu] if isinstance(gpu, int) and not isinstance(gpu, bool) else list(gpu)
    )
    if (
        not raw_gpus
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_gpus
        )
        or len(set(raw_gpus)) != len(raw_gpus)
    ):
        raise ExecutorError(
            "child environment GPUs must be unique non-negative integers"
        )
    try:
        account_home = pwd.getpwuid(os.getuid()).pw_dir
    except (KeyError, OSError) as error:
        raise ExecutorError("cannot resolve the learner account home") from error
    repo_root = (
        _REPO_ROOT.resolve(strict=True)
        if repository_root is None
        else repository_root.expanduser().resolve(strict=True)
    )
    if repo_root.is_symlink() or not repo_root.is_dir():
        raise ExecutorError("learner repository root must be a regular directory")
    environment = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, raw_gpus)),
        "HOME": str(account_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{repo_root / 'src'}:{repo_root}",
        "TMPDIR": "/tmp",
        "TZ": "UTC",
    }
    if set(environment) != CHILD_ENVIRONMENT_KEYS or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ExecutorError("learner child environment allowlist drift")
    return environment


def _execution_binding(
    *, command: list[str], environment: dict[str, str]
) -> dict[str, Any]:
    if set(environment) != CHILD_ENVIRONMENT_KEYS or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ExecutorError("cannot bind a non-allowlisted learner environment")
    return {
        "schema_version": REPORT_EXECUTION_BINDING_SCHEMA,
        "command_sha256": _value_sha256(command),
        "environment": dict(environment),
        "environment_sha256": _value_sha256(environment),
    }


def _input_binding(verified: dict[str, Any]) -> dict[str, Any]:
    """Bind the executor's authenticated data, split, and topology authority."""

    recipe = verified["recipe"]
    topology = verified.get("training_topology")
    if topology is None:
        topology = {
            "schema_version": "a1-one-dose-training-topology-v1",
            "name": LEGACY_SINGLE_GPU_TOPOLOGY,
            "world_size": int(recipe["world_size"]),
            "physical_gpus": [0],
            "local_batch_size": int(recipe["batch_size"]),
            "grad_accum_steps": int(recipe["grad_accum_steps"]),
            "global_batch_size": int(recipe["global_batch_size"]),
            "dose_preserving": True,
        }
    payload: dict[str, Any] = {
        "schema_version": REPORT_INPUT_BINDING_SCHEMA,
        "contract_sha256": verified["contract_sha256"],
        "data": str(verified["data_path"]),
        "data_kind": verified.get("data_kind", "a1_memmap_v1"),
        "data_fingerprint": verified["data_fingerprint"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "corpus_row_count": int(verified["corpus_row_count"]),
        "training_row_count": int(verified["training_row_count"]),
        "validation_row_count": int(verified["validation_row_count"]),
        "sealed_learner_recipe_sha256": _value_sha256(
            verified.get("bound_recipe", recipe)
        ),
        "effective_learner_recipe_sha256": _value_sha256(recipe),
        "training_topology": topology,
        "ddp_canary": verified.get("ddp_canary"),
        "aux_subgoal_preclaim_contract": verified.get("aux_subgoal_preclaim_contract"),
        "aux_pair_executor_authority_sha256": (
            verified.get("aux_pair_executor_authority", {}).get("authority_sha256")
            if isinstance(verified.get("aux_pair_executor_authority"), dict)
            else None
        ),
        "p1_arm_executor_authority_sha256": (
            verified.get("p1_arm_executor_authority", {}).get("authority_sha256")
            if isinstance(verified.get("p1_arm_executor_authority"), dict)
            else None
        ),
        "final_replication_executor_authority_sha256": (
            verified.get("final_replication_executor_authority", {}).get(
                "authority_sha256"
            )
            if isinstance(verified.get("final_replication_executor_authority"), dict)
            else None
        ),
        "stage_c_final_replication_authority_sha256": (
            verified.get("stage_c_final_replication_authority", {}).get(
                "authority_sha256"
            )
            if isinstance(verified.get("stage_c_final_replication_authority"), dict)
            else None
        ),
        "stage_c_final_replication_binding": (
            copy.deepcopy(verified["stage_c_final_replication_binding"])
            if isinstance(verified.get("stage_c_final_replication_binding"), dict)
            else None
        ),
        "central_published_executor_authority": (
            copy.deepcopy(verified["central_published_executor_authority"])
            if isinstance(verified.get("central_published_executor_authority"), dict)
            else None
        ),
        "diagnostic_comparison_source": (
            copy.deepcopy(verified["diagnostic_comparison_source"])
            if isinstance(verified.get("diagnostic_comparison_source"), dict)
            else None
        ),
        "learner_lineage_parent": (
            copy.deepcopy(verified["learner_lineage_parent"])
            if isinstance(verified.get("learner_lineage_parent"), dict)
            else None
        ),
    }
    if verified.get("data_kind") == "production_composite_v2":
        trainer_authority = _require_current_production_trainer_authority(verified)
        assert trainer_authority is not None
        payload.update(
            {
                "trainer_authority": trainer_authority,
                "event_history_training_contract": copy.deepcopy(
                    verified["event_history_training_contract"]
                ),
                "event_history_component_authority": copy.deepcopy(
                    verified["event_history_component_authority"]
                ),
                "production_mix_contract_sha256": _value_sha256(
                    verified["production_mix_contract"]
                ),
                "production_sampling_receipt_sha256": verified[
                    "production_sampling_receipt_sha256"
                ],
                "validation_split_receipt": verified["validation_split_receipt"],
                "validation_split_receipt_sha256": verified[
                    "validation_split_receipt_sha256"
                ],
                "composite_build_receipt": verified["composite_build_receipt"],
                "source_authority": verified.get("source_authority_ref"),
                "category_semantics": verified.get("category_semantics"),
                "category_semantics_sha256": verified.get("category_semantics_sha256"),
            }
        )
        if "p1_training_descriptor_authority" in verified:
            payload["p1_training_descriptor_authority"] = copy.deepcopy(
                verified["p1_training_descriptor_authority"]
            )
        if "diagnostic_training_descriptor_authority" in verified:
            payload["diagnostic_training_descriptor_authority"] = copy.deepcopy(
                verified["diagnostic_training_descriptor_authority"]
            )
    elif verified.get("data_kind") == "coherent_direct_memmap_v1":
        payload["coherent_corpus_admission"] = copy.deepcopy(
            verified["coherent_direct_corpus_binding"]["corpus_admission"]
        )
        payload["coherent_direct_corpus_binding_sha256"] = verified[
            "coherent_direct_corpus_binding"
        ]["binding_sha256"]
    else:
        payload.update(
            {
                "validation_manifest": str(verified["validation_path"]),
                "validation_manifest_file_sha256": verified["validation_file_sha256"],
                "selected_game_seed_set_sha256": verified[
                    "selected_game_seed_set_sha256"
                ],
                "training_game_seed_set_sha256": verified[
                    "training_game_seed_set_sha256"
                ],
                "validation_game_seed_set_sha256": verified[
                    "validation_game_seed_set_sha256"
                ],
            }
        )
    lock_verifier_authority = verified.get("lock_verifier_authority")
    if lock_verifier_authority is not None:
        payload["lock_verifier_authority"] = copy.deepcopy(lock_verifier_authority)
    payload["binding_sha256"] = _value_sha256(payload)
    return payload


def _training_transaction_sha256(
    *, command: Sequence[str], input_binding: Mapping[str, Any]
) -> str:
    """Bind child argv and frozen-input authority into one launch identity."""

    return _value_sha256(
        {
            "schema_version": TRAINING_TRANSACTION_SCHEMA,
            "command_sha256": _value_sha256(list(command)),
            "input_binding_sha256": input_binding.get("binding_sha256"),
        }
    )


def _validate_execution_binding(binding: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "command_sha256",
        "environment",
        "environment_sha256",
    }
    environment = binding.get("environment")
    if (
        set(binding) != expected_keys
        or binding.get("schema_version") != REPORT_EXECUTION_BINDING_SCHEMA
        or not isinstance(binding.get("command_sha256"), str)
        or not isinstance(environment, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        )
        or set(environment) != CHILD_ENVIRONMENT_KEYS
        or binding.get("environment_sha256") != _value_sha256(environment)
    ):
        raise ExecutorError("A1 execution binding is invalid")


def _producer(lock: dict[str, Any]) -> dict[str, Any]:
    matches = [
        record
        for record in lock.get("checkpoints", [])
        if isinstance(record, dict) and record.get("role") == "producer"
    ]
    if len(matches) != 1:
        raise ExecutorError("sealed A1 contract must bind exactly one producer")
    return matches[0]


def _lock_verification_cache_root() -> Path:
    override = os.environ.get(LOCK_VERIFICATION_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "catan-zero" / "a1-lock-verification"


def _verification_path_identity(path: Path) -> tuple[dict[str, Any], bool]:
    """Describe one verification input without trusting pathname metadata alone."""

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        info = lexical.lstat()
    except FileNotFoundError:
        return ({"path": str(lexical), "kind": "missing"}, True)
    except OSError as error:
        raise ExecutorError(
            f"cannot inspect lock-verification input {lexical}: {error}"
        ) from error
    common = {
        "path": str(lexical),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size_bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
        "mode": int(stat.S_IMODE(info.st_mode)),
        "uid": int(info.st_uid),
    }
    if stat.S_ISREG(info.st_mode):
        return ({**common, "kind": "regular_file"}, True)
    if stat.S_ISDIR(info.st_mode):
        return ({**common, "kind": "directory"}, True)
    # The historical verifier may accept a symlink in an archival path, but a
    # reusable receipt must never turn that mutable indirection into authority.
    kind = "symlink" if stat.S_ISLNK(info.st_mode) else "special"
    return ({**common, "kind": kind}, False)


def _referenced_verification_paths(
    lock_path: Path,
    raw_lock: Mapping[str, Any],
    *,
    verifier_path: Path,
) -> tuple[list[dict[str, Any]], bool]:
    """Snapshot the lock's direct and JSON-linked verification inputs.

    The sealed lock is already authenticated before this traversal.  We follow
    only explicit ``path``/``*_path`` fields and only parse small JSON files;
    checkpoint and corpus payloads are statted, never reread by the cache.
    """

    pending_values: list[tuple[Any, Path]] = [(raw_lock, lock_path.parent)]
    paths: dict[str, Path] = {}
    json_frontier: list[Path] = []
    json_frontier_index = 0

    def discover_path(path: Path) -> None:
        """Record and enqueue one lexical path exactly once."""

        lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
        path_key = str(lexical)
        if path_key in paths:
            return
        paths[path_key] = lexical
        if len(paths) > 16_384:
            raise ExecutorError("reviewed lock references too many verification paths")
        if lexical.suffix.lower() == ".json":
            json_frontier.append(lexical)

    explicit = (lock_path, verifier_path, Path(__file__).resolve(strict=True))
    for path in explicit:
        discover_path(path)
    while pending_values or json_frontier_index < len(json_frontier):
        if pending_values:
            value, relative_to = pending_values.pop()
            if isinstance(value, Mapping):
                for key, child in value.items():
                    if isinstance(child, str) and (
                        key == "path" or key.endswith("_path")
                    ):
                        candidate = Path(child).expanduser()
                        if not candidate.is_absolute():
                            candidate = relative_to / candidate
                        discover_path(candidate)
                    else:
                        pending_values.append((child, relative_to))
            elif isinstance(value, (list, tuple)):
                pending_values.extend((child, relative_to) for child in value)

        # Drain only the newly discovered JSON frontier. Each linked document
        # is inspected and parsed at most once; nested values can enqueue more
        # paths on the next value-walk iteration without rescanning old paths.
        while json_frontier_index < len(json_frontier):
            path = json_frontier[json_frontier_index]
            json_frontier_index += 1
            try:
                before = path.lstat()
            except (FileNotFoundError, NotADirectoryError):
                continue
            except OSError as error:
                raise ExecutorError(
                    f"cannot inspect linked lock-verification JSON {path}: {error}"
                ) from error
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_size > 16 * 1024 * 1024
            ):
                continue
            try:
                linked = json.loads(path.read_text(encoding="utf-8"))
                after = path.lstat()
            except (OSError, UnicodeError, json.JSONDecodeError):
                # The full verifier owns the semantic error.  It is enough for
                # the cache to retain the file identity and decline recursion.
                continue
            before_id = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_id = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if before_id != after_id:
                raise ExecutorError(
                    f"linked lock-verification JSON changed while read: {path}"
                )
            pending_values.append((linked, path.parent))

    identities: list[dict[str, Any]] = []
    cache_eligible = True
    for path in sorted(paths.values(), key=lambda item: str(item)):
        identity, eligible = _verification_path_identity(path)
        identities.append(identity)
        cache_eligible = cache_eligible and eligible
    return identities, cache_eligible


def _sealed_verifier_code_binding(
    raw_lock: Mapping[str, Any], *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    provenance = raw_lock.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ExecutorError("reviewed A1 lock has no code provenance")
    sections: dict[str, Any] = {}
    for section in ("generator_code", "learner_code", "runtime_code_tree"):
        records = provenance.get(section)
        if section == "generator_code" and records is None:
            continue
        if not isinstance(records, list) or not records:
            raise ExecutorError(f"reviewed A1 lock has no {section} records")
        records_sha256 = _value_sha256(records)
        declared = provenance.get(f"{section}_sha256")
        if declared is not None and declared != records_sha256:
            raise ExecutorError(f"reviewed A1 {section} declared digest drift")
        sections[section] = {
            "records_sha256": records_sha256,
            "declared_sha256": declared,
        }
    binding: dict[str, Any] = {
        "verifier_path": str(Path(authority["verifier_path"])),
        "verifier_sha256": str(authority["verifier_sha256"]),
        "sections": sections,
    }
    binding["code_tree_binding_sha256"] = _value_sha256(binding)
    return binding


def _lock_verification_cache_binding(
    *,
    lock_path: Path,
    lock_file_sha256: str,
    semantic_lock_sha256: str,
    reviewed_lock_file_sha256: str,
    authority: Mapping[str, Any],
    code_binding: Mapping[str, Any],
    filesystem_identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    portable_authority = {
        str(key): str(value) if isinstance(value, Path) else value
        for key, value in authority.items()
    }
    binding: dict[str, Any] = {
        "schema_version": LOCK_VERIFICATION_CACHE_SCHEMA,
        "validator_version": LOCK_VERIFICATION_CACHE_VALIDATOR_VERSION,
        "require_all_job_claims": True,
        "lock_path": str(lock_path),
        "lock_file_sha256": lock_file_sha256,
        "semantic_lock_sha256": semantic_lock_sha256,
        "reviewed_lock_file_sha256": reviewed_lock_file_sha256,
        "trust_anchor_sha256": _value_sha256(portable_authority),
        "sealed_verifier_code_binding": dict(code_binding),
        "verification_implementation_sha256": _file_sha256(Path(__file__)),
        "filesystem_identities": [dict(row) for row in filesystem_identities],
        "filesystem_identities_sha256": _value_sha256(
            [dict(row) for row in filesystem_identities]
        ),
    }
    binding["binding_sha256"] = _value_sha256(binding)
    return binding


def _lock_verification_cache_path(binding: Mapping[str, Any]) -> Path:
    digest = str(binding["binding_sha256"]).split(":", 1)[1]
    return _lock_verification_cache_root() / f"{digest}.json"


def _load_lock_verification_cache(
    binding: Mapping[str, Any], *, cache_eligible: bool
) -> dict[str, Any] | None:
    if not cache_eligible:
        return None
    path = _lock_verification_cache_path(binding)
    try:
        directory_info = path.parent.lstat()
        cache_info = path.lstat()
        if (
            not stat.S_ISDIR(directory_info.st_mode)
            or directory_info.st_uid != os.geteuid()
            or stat.S_IMODE(directory_info.st_mode) & 0o077
            or not stat.S_ISREG(cache_info.st_mode)
            or stat.S_ISLNK(cache_info.st_mode)
            or cache_info.st_uid != os.geteuid()
            or stat.S_IMODE(cache_info.st_mode) & 0o077
        ):
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "binding",
        "verified_lock",
        "verified_lock_sha256",
        "receipt_sha256",
    }:
        return None
    unsigned = dict(payload)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    verified = payload.get("verified_lock")
    if (
        payload.get("schema_version") != LOCK_VERIFICATION_CACHE_SCHEMA
        or payload.get("binding") != binding
        or not isinstance(verified, dict)
        or payload.get("verified_lock_sha256") != _value_sha256(verified)
        or receipt_sha256 != _value_sha256(unsigned)
        or verified.get("contract_sha256") != binding.get("semantic_lock_sha256")
    ):
        return None
    return copy.deepcopy(verified)


def _publish_lock_verification_cache(
    binding: Mapping[str, Any], verified: Mapping[str, Any], *, cache_eligible: bool
) -> None:
    """Best-effort atomic publication; cache failure never weakens verification."""

    if not cache_eligible:
        return
    path = _lock_verification_cache_path(binding)
    temporary: str | None = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory_info = path.parent.lstat()
        if (
            not stat.S_ISDIR(directory_info.st_mode)
            or directory_info.st_uid != os.geteuid()
            or stat.S_IMODE(directory_info.st_mode) & 0o077
        ):
            return
        unsigned: dict[str, Any] = {
            "schema_version": LOCK_VERIFICATION_CACHE_SCHEMA,
            "binding": dict(binding),
            "verified_lock": copy.deepcopy(dict(verified)),
            "verified_lock_sha256": _value_sha256(dict(verified)),
        }
        payload = {**unsigned, "receipt_sha256": _value_sha256(unsigned)}
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    except (OSError, TypeError, ValueError):
        return
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _verify_lock_with_sealed_runtime(
    lock_path: Path, *, reviewed_lock_file_sha256: str | None = None
) -> dict[str, Any]:
    """Replay lock reconstruction with the exact repository path it sealed.

    `build_lock` intentionally records absolute code paths, so importing its
    verifier from a new ablation checkout makes an otherwise-identical lock
    reconstruct with the new root and fail. Run only this immutable-input
    verification in the original hash-bound runtime; the derived trainer is
    separately bound by `_current_ablation_code_binding`.
    """

    if reviewed_lock_file_sha256 is None:
        # Historical/default execution remains byte-for-byte on the original
        # in-process verifier path. Only the explicitly reviewed ablation path
        # may select the old absolute-root verifier from lock provenance.
        return a1_contract.verify_lock(lock_path, require_all_job_claims=True)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", reviewed_lock_file_sha256):
        raise ExecutorError("reviewed A1 lock file sha256 is malformed")
    authority = _reviewed_lock_trust_anchors().get(reviewed_lock_file_sha256)
    if authority is None:
        raise ExecutorError(
            "reviewed A1 lock digest is not an issued production trust anchor"
        )
    trusted_lock_path = Path(authority["lock_path"])
    if lock_path.resolve(strict=True) != trusted_lock_path.resolve(strict=True):
        raise ExecutorError("A1 lock path does not match its production trust anchor")
    actual_lock_sha = _file_sha256(lock_path)
    if actual_lock_sha != reviewed_lock_file_sha256:
        raise ExecutorError(
            "A1 lock bytes do not match the explicitly reviewed digest: "
            f"reviewed={reviewed_lock_file_sha256} actual={actual_lock_sha}"
        )
    # Parsing/selecting paths happens only after the raw bytes are authenticated.
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot read reviewed A1 lock: {error}") from error
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        raise ExecutorError("reviewed A1 lock has no code provenance")
    semantic_lock_sha256 = raw.get("contract_sha256")
    unhashed_lock = dict(raw)
    unhashed_lock.pop("contract_sha256", None)
    semantic_lock_valid = bool(
        isinstance(semantic_lock_sha256, str)
        and semantic_lock_sha256 == _value_sha256(unhashed_lock)
    )
    expected_contract_id = authority.get("contract_id")
    expected_contract_sha = authority.get("contract_sha256")
    expected_producer_sha = authority.get("producer_sha256")
    if expected_contract_id is not None and (
        raw.get("contract_id") != expected_contract_id
        or raw.get("contract_sha256") != expected_contract_sha
        or _producer(raw).get("sha256") != expected_producer_sha
        or raw.get("promotion_handoff", {}).get("producer_checkpoint", {}).get("sha256")
        != expected_producer_sha
    ):
        raise ExecutorError(
            "reviewed production lock differs from its sealed handoff/contract identity"
        )
    runtime = provenance.get("runtime_code_tree")
    matches = [
        record
        for record in runtime or []
        if isinstance(record, dict)
        and str(record.get("path", "")) == str(authority["verifier_path"])
    ]
    if len(matches) != 1 or matches[0].get("sha256") != authority["verifier_sha256"]:
        raise ExecutorError("reviewed A1 lock does not bind its sealed verifier")
    verifier = Path(authority["verifier_path"]).expanduser().resolve(strict=True)
    canonical_lock_path = lock_path.resolve(strict=True)
    identities_before: list[dict[str, Any]] = []
    cache_eligible = False
    cache_binding: dict[str, Any] | None = None
    if semantic_lock_valid:
        code_binding = _sealed_verifier_code_binding(raw, authority=authority)
        identities_before, cache_eligible = _referenced_verification_paths(
            canonical_lock_path,
            raw,
            verifier_path=verifier,
        )
        assert isinstance(semantic_lock_sha256, str)
        cache_binding = _lock_verification_cache_binding(
            lock_path=canonical_lock_path,
            lock_file_sha256=actual_lock_sha,
            semantic_lock_sha256=semantic_lock_sha256,
            reviewed_lock_file_sha256=reviewed_lock_file_sha256,
            authority=authority,
            code_binding=code_binding,
            filesystem_identities=identities_before,
        )
        cached = _load_lock_verification_cache(
            cache_binding, cache_eligible=cache_eligible
        )
        if cached is not None:
            identities_after, still_eligible = _referenced_verification_paths(
                canonical_lock_path,
                raw,
                verifier_path=verifier,
            )
            if still_eligible and identities_after == identities_before:
                print(
                    json.dumps(
                        {
                            "progress": "a1_lock_verification_cache",
                            "status": "hit",
                            "binding_sha256": cache_binding["binding_sha256"],
                            "filesystem_identity_count": len(identities_before),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return cached
    # Authenticate every dependency the sealed verifier may import before
    # starting its interpreter. The raw lock is already pinned/authenticated,
    # so these paths/digests are trusted declarations rather than attacker
    # supplied selectors.
    for section in ("learner_code", "runtime_code_tree"):
        records = provenance.get(section)
        if not isinstance(records, list) or not records:
            raise ExecutorError(f"reviewed A1 lock has no {section} records")
        for record in records:
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
            ):
                raise ExecutorError(f"reviewed A1 {section} record is malformed")
            dependency = Path(record["path"]).expanduser().resolve(strict=True)
            actual_dependency_sha = _file_sha256(dependency)
            if actual_dependency_sha != record["sha256"]:
                raise ExecutorError(
                    "sealed A1 verifier dependency drift before import: "
                    f"{dependency} expected={record['sha256']} "
                    f"actual={actual_dependency_sha}"
                )
    actual_verifier_sha = _file_sha256(verifier)
    if actual_verifier_sha != authority["verifier_sha256"]:
        raise ExecutorError(
            "pinned sealed verifier digest mismatch: "
            f"expected={authority['verifier_sha256']} actual={actual_verifier_sha}"
        )
    if not semantic_lock_valid:
        raise ExecutorError("reviewed A1 lock semantic digest drift")
    sealed_root = verifier.parents[1]
    script = (
        "import json,sys; from pathlib import Path; "
        "from tools.a1_pre_wave_contract import verify_lock; "
        "print(json.dumps(verify_lock(Path(sys.argv[1]), require_all_job_claims=True),"
        "sort_keys=True,separators=(',',':')))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{sealed_root / 'src'}:{sealed_root}"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(lock_path)],
            cwd=str(sealed_root),
            env=environment,
            check=True,
            text=True,
            capture_output=True,
        )
        verified = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        detail = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError) and error.stderr
            else str(error)
        )
        raise ExecutorError(f"sealed A1 lock verifier refused: {detail}") from error
    if not isinstance(verified, dict) or verified.get("contract_sha256") != raw.get(
        "contract_sha256"
    ):
        raise ExecutorError("sealed A1 verifier returned a different lock identity")
    assert cache_binding is not None
    identities_after, still_eligible = _referenced_verification_paths(
        canonical_lock_path,
        raw,
        verifier_path=verifier,
    )
    if identities_after != identities_before:
        raise ExecutorError(
            "lock-verification filesystem inputs changed during full verification"
        )
    _publish_lock_verification_cache(
        cache_binding,
        verified,
        cache_eligible=cache_eligible and still_eligible,
    )
    print(
        json.dumps(
            {
                "progress": "a1_lock_verification_cache",
                "status": "miss",
                "binding_sha256": cache_binding["binding_sha256"],
                "filesystem_identity_count": len(identities_before),
                "cache_eligible": cache_eligible and still_eligible,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return verified


def _verify_training_lock(
    lock_path: Path,
    *,
    reviewed_lock_file_sha256: str | None = None,
    frozen_repo: Path | None = None,
    frozen_verifier_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Select exactly one authenticated lock-verification authority."""

    if bool(frozen_repo) != bool(frozen_verifier_sha256):
        raise ExecutorError(
            "--frozen-repo and --frozen-verifier-sha256 are required together"
        )
    if frozen_repo is None:
        return (
            _verify_lock_with_sealed_runtime(
                lock_path,
                reviewed_lock_file_sha256=reviewed_lock_file_sha256,
            ),
            None,
        )
    assert frozen_verifier_sha256 is not None
    try:
        lock, authority = frozen_lock_verifier.verify_frozen_lock(
            lock_path,
            frozen_repo=frozen_repo,
            expected_verifier_sha256=frozen_verifier_sha256,
        )
    except frozen_lock_verifier.FrozenVerifierError as error:
        raise ExecutorError(f"frozen A1 lock verifier refused: {error}") from error
    # A post-wave ablation needs two independent authorities: the historical
    # verifier that can reconstruct the path-bound production lock, and the
    # reviewed raw lock digest that binds the derived learner recipe.  These
    # are complementary, not competing, trust claims.  Require both views to
    # agree on the exact bytes before exposing the reviewed digest downstream.
    if reviewed_lock_file_sha256 is not None:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", reviewed_lock_file_sha256):
            raise ExecutorError("reviewed A1 lock file sha256 is malformed")
        actual_lock_sha256 = _file_sha256(lock_path)
        if (
            actual_lock_sha256 != reviewed_lock_file_sha256
            or authority.get("lock_file_sha256") != reviewed_lock_file_sha256
        ):
            raise ExecutorError(
                "frozen verifier and reviewed ablation disagree on A1 lock bytes"
            )
    return lock, authority


def _require_a1_science(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    science = lock.get("science")
    if not isinstance(science, dict):
        raise ExecutorError("sealed A1 contract has no science section")
    search = science.get("search_operator")
    if not isinstance(search, dict) or int(search.get("n_full", -1)) != 128:
        raise ExecutorError(
            "current A1 operator decision requires global n_full=128; "
            "n64 and global n196/n256 are not authorized"
        )
    if current_science.is_coherent_search(search):
        try:
            current_science.require_current_operator(
                search_value=search,
                evaluator_value=science.get("evaluator"),
                generation_value=lock.get("generation"),
                learner_recipe_value=science.get("learner_training_recipe"),
                target_regime=lock.get("post_wave_acceptance", {}).get(
                    "require_target_information_regime"
                ),
                require_adopted=True,
            )
        except current_science.ScienceContractError as error:
            raise ExecutorError(str(error)) from error
    recipe = science.get("learner_training_recipe")
    if recipe not in (
        a1_contract.EXPECTED_LEARNER_TRAINING_RECIPE,
        a1_contract.CURRENT_LEARNER_TRAINING_RECIPE,
        a1_contract.COHERENT_PUBLIC_LEARNER_TRAINING_RECIPE,
    ):
        raise ExecutorError(
            "sealed A1 learner recipe differs from the exact one-dose recipe"
        )
    objective = science.get("learner_value_objective")
    if objective != {
        "objective": "mse",
        "value_readout": "scalar",
        "value_categorical_bins": None,
        "hlgauss_sigma_ratio": None,
    }:
        raise ExecutorError("current A1 one-dose executor requires scalar MSE/readout")
    if (
        recipe["world_size"] != 1
        or recipe["global_batch_size"] != 4096
        or recipe["optimizer"] != "adam"
        or recipe["resume_optimizer"] is not False
        or recipe["fused_optimizer"] is not False
        or recipe["value_lr_mult"] != 0.3
    ):
        raise ExecutorError(
            "sealed A1 topology/optimizer invariants are not one-B200 fresh Adam"
        )
    return recipe, objective


def _verify_coherent_direct_training_inputs(
    *,
    admission_path: Path,
    lock: dict[str, Any],
    lock_path: Path,
    lock_verifier_authority: dict[str, Any] | None,
    reviewed_lock_file_sha256: str | None,
    recipe: dict[str, Any],
    objective: dict[str, Any],
    data_path: Path,
    validation_path: Path | None,
) -> dict[str, Any]:
    """Authenticate the diagnostic coherent corpus without legacy 12k claims."""

    if validation_path is None:
        raise ExecutorError("coherent direct corpus requires a whole-game holdout")
    try:
        admission_path = admission_path.expanduser().resolve(strict=True)
        payload = json.loads(admission_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(
            f"cannot read coherent corpus admission: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ExecutorError("coherent corpus admission must be a JSON object")
    final_admission: dict[str, Any] | None = None
    if payload.get("schema_version") == stage_c_final.FINAL_CORPUS_ADMISSION_SCHEMA:
        try:
            verified_final = stage_c_final.verify_final_corpus_admission(
                admission_path
            )
        except stage_c_final.FinalReplicationError as error:
            raise ExecutorError(
                f"Stage-C final corpus admission refused: {error}"
            ) from error
        final_admission = copy.deepcopy(verified_final)
        coherent_payload = final_admission.pop("_coherent_admission", None)
        if not isinstance(coherent_payload, dict):
            raise ExecutorError(
                "Stage-C final admission lost its low-level coherent corpus"
            )
        payload = coherent_payload
    unsigned = dict(payload)
    stated = unsigned.pop("admission_sha256", None)
    if (
        payload.get("schema_version") != "a1-coherent-n128-corpus-admission-v1"
        or payload.get("status") != "admitted_for_diagnostic_policy_distillation"
        or payload.get("diagnostic_only") is not True
        or payload.get("promotion_eligible") is not False
        or stated != _value_sha256(unsigned)
    ):
        raise ExecutorError("coherent corpus admission schema/digest/role drift")
    corpus = payload.get("corpus")
    policy = payload.get("policy_distillation_contract")
    contract = payload.get("contract")
    if not all(isinstance(value, dict) for value in (corpus, policy, contract)):
        raise ExecutorError("coherent corpus admission sections are malformed")
    producer = _producer(lock)
    if (
        Path(str(corpus.get("data_path", ""))).expanduser().resolve(strict=True)
        != data_path
        or Path(str(corpus.get("validation_manifest", {}).get("path", "")))
        .expanduser()
        .resolve(strict=True)
        != validation_path
        or corpus.get("producer_checkpoint_sha256") != producer.get("sha256")
        or corpus.get("selected_games") != 8_192
        or not isinstance(corpus.get("selected_game_seed_set_sha256"), str)
        or corpus.get("target_information_regime") != "public_belief_single_tree_v1"
        or corpus.get("search_evidence_schema") != "gumbel_root_search_evidence_v1"
        or corpus.get("incompatible_policy_active_rows") != 0
        or policy.get("coherent_public_n128_only") is not True
        or policy.get("legacy_pimc_rows_allowed") is not False
    ):
        raise ExecutorError("coherent corpus admission identity drift")

    meta_path = data_path / "corpus_meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot read coherent corpus metadata: {error}") from error
    if (
        not isinstance(meta, dict)
        or _file_sha256(meta_path) != corpus.get("corpus_meta_file_sha256")
        or meta.get("payload_inventory_sha256")
        != corpus.get("payload_inventory_sha256")
    ):
        raise ExecutorError("coherent corpus metadata differs from admission")
    try:
        train_bc._validate_memmap_payload_inventory(data_path, meta)  # noqa: SLF001
        validation = train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
            validation_path,
            validation_fraction=0.05,
            validation_seed=17,
            validation_max_samples=0,
            validation_game_seed_ranges=[],
        )
        data = train_bc.load_teacher_data_memmap(data_path)
    except (OSError, SystemExit, ValueError) as error:
        raise ExecutorError(
            f"coherent corpus bytes/holdout refused: {error}"
        ) from error
    if (
        validation["a1_contract_sha256"] != contract.get("contract_sha256")
        or validation["file_sha256"] != corpus["validation_manifest"]["file_sha256"]
    ):
        raise ExecutorError("coherent holdout binds a different target contract")
    observed = np.asarray(data["game_seed"], dtype=np.int64).reshape(-1)
    if observed.size == 0:
        raise ExecutorError("coherent corpus has no game rows")
    run_starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(observed[1:] != observed[:-1]) + 1,
        )
    )
    run_values = observed[run_starts]
    selected = np.sort(np.unique(observed))
    if (
        np.unique(run_values).size != run_values.size
        or selected.size != int(corpus["selected_games"])
        or train_bc._game_seed_set_sha256(selected)  # noqa: SLF001
        != corpus["selected_game_seed_set_sha256"]
    ):
        raise ExecutorError(
            "coherent corpus differs from its explicit selected seed set"
        )
    validation_seeds = np.asarray(validation["game_seeds"], dtype=np.int64)
    if not np.isin(validation_seeds, selected).all():
        raise ExecutorError("coherent holdout includes a seed outside the corpus")
    validation_row_count = int(np.count_nonzero(np.isin(observed, validation_seeds)))
    if validation_row_count != int(validation["validation_row_count"]):
        raise ExecutorError("coherent holdout row count differs from actual memmap")
    training_seeds = selected[~np.isin(selected, validation_seeds)]
    training_row_count = int(observed.size) - validation_row_count
    selected_sha = train_bc._game_seed_set_sha256(selected)  # noqa: SLF001
    training_sha = train_bc._game_seed_set_sha256(training_seeds)  # noqa: SLF001
    corpus_binding = {
        "path": str(data_path),
        "corpus_meta_file_sha256": _file_sha256(meta_path),
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "selected_game_count": int(selected.size),
        "seed_start": int(corpus["seed_start"]),
        "seed_end": int(corpus["seed_end"]),
        "selected_game_seed_set_sha256": selected_sha,
        "training_game_count": int(training_seeds.size),
        "training_game_seed_set_sha256": training_sha,
    }
    validation_binding = {
        "path": str(validation_path),
        "file_sha256": validation["file_sha256"],
        "game_count": int(np.asarray(validation["game_seeds"], dtype=np.int64).size),
        "game_seed_set_sha256": validation["validation_game_seed_set_sha256"],
        "row_count": validation_row_count,
    }
    direct_binding: dict[str, Any] = {
        "schema_version": train_bc.COHERENT_DIRECT_CORPUS_BINDING_SCHEMA,
        "diagnostic_only": final_admission is None,
        "promotion_eligible": False,
        "promotion_eligible_after_full_gate": final_admission is not None,
        "full_gate_required": final_admission is not None,
        "corpus_admission": {
            "path": str(admission_path),
            "file_sha256": _file_sha256(admission_path),
            "admission_sha256": (
                final_admission["admission_sha256"]
                if final_admission is not None
                else stated
            ),
        },
        "target_contract_sha256": contract["contract_sha256"],
        "producer_checkpoint_sha256": producer["sha256"],
        # Filled only after the independently authorized f7 architecture
        # upgrade is replayed.  Keeping the slot in the first binding makes the
        # later transition an explicit digest-changing operation instead of an
        # untyped side channel.
        "learner_initializer": None,
        "corpus": corpus_binding,
        "validation": validation_binding,
        "learner": {
            "training_recipe": copy.deepcopy(recipe),
            "training_recipe_sha256": _value_sha256(recipe),
            "value_objective": copy.deepcopy(objective),
            "topology": {
                "name": B200_8GPU_DDP_TOPOLOGY,
                "world_size": 8,
                "local_batch_size": 512,
                "grad_accum_steps": 1,
                "global_batch_size": 4096,
            },
        },
    }
    direct_binding["binding_sha256"] = _value_sha256(direct_binding)
    return {
        "lock": lock,
        "lock_path": lock_path,
        "lock_file_sha256": _file_sha256(lock_path),
        "reviewed_lock_file_sha256": reviewed_lock_file_sha256,
        "lock_verifier_authority": lock_verifier_authority,
        "contract_sha256": contract["contract_sha256"],
        "learner_lock_contract_sha256": lock["contract_sha256"],
        "recipe": recipe,
        "objective": objective,
        "producer": producer,
        "data_kind": "coherent_direct_memmap_v1",
        "data_path": data_path,
        "corpus_meta_file_sha256": _file_sha256(meta_path),
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "data_fingerprint": train_bc._training_data_fingerprint(  # noqa: SLF001
            str(data_path), "memmap"
        ),
        "corpus_row_count": int(observed.size),
        "training_row_count": training_row_count,
        "validation_row_count": validation_row_count,
        "selected_game_seed_set_sha256": selected_sha,
        "training_game_seed_set_sha256": training_sha,
        "validation_path": validation_path,
        "validation_file_sha256": validation["file_sha256"],
        "validation_game_seed_set_sha256": validation[
            "validation_game_seed_set_sha256"
        ],
        "coherent_corpus_admission": copy.deepcopy(
            final_admission if final_admission is not None else payload
        ),
        "stage_c_final_corpus_admission": copy.deepcopy(final_admission),
        "coherent_direct_corpus_binding": direct_binding,
    }


def verify_training_inputs(
    *,
    lock_path: Path,
    data_path: Path,
    validation_path: Path | None,
    composite_build_receipt: Path | None = None,
    reviewed_lock_file_sha256: str | None = None,
    frozen_repo: Path | None = None,
    frozen_verifier_sha256: str | None = None,
    coherent_corpus_admission: Path | None = None,
) -> dict[str, Any]:
    """Replay the sealed lock and complete audit→memmap→holdout chain."""

    try:
        lock_path = lock_path.expanduser().resolve(strict=True)
        data_path = data_path.expanduser().resolve(strict=True)
        validation_path = (
            None
            if validation_path is None
            else validation_path.expanduser().resolve(strict=True)
        )
        composite_build_receipt = (
            None
            if composite_build_receipt is None
            else composite_build_receipt.expanduser().resolve(strict=True)
        )
        coherent_corpus_admission = (
            None
            if coherent_corpus_admission is None
            else coherent_corpus_admission.expanduser().resolve(strict=True)
        )
    except OSError as error:
        raise ExecutorError(f"cannot resolve A1 training input: {error}") from error
    if not (data_path.is_dir() or data_path.is_file()):
        raise ExecutorError(
            f"A1 data path is not a corpus directory/descriptor: {data_path}"
        )

    try:
        lock, lock_verifier_authority = _verify_training_lock(
            lock_path,
            reviewed_lock_file_sha256=reviewed_lock_file_sha256,
            frozen_repo=frozen_repo,
            frozen_verifier_sha256=frozen_verifier_sha256,
        )
        recipe, objective = _require_a1_science(lock)
        if coherent_corpus_admission is not None:
            if data_path.is_file() or composite_build_receipt is not None:
                raise ExecutorError(
                    "coherent direct-corpus admission requires a memmap directory"
                )
            return _verify_coherent_direct_training_inputs(
                admission_path=coherent_corpus_admission,
                lock=lock,
                lock_path=lock_path,
                lock_verifier_authority=lock_verifier_authority,
                reviewed_lock_file_sha256=reviewed_lock_file_sha256,
                recipe=recipe,
                objective=objective,
                data_path=data_path,
                validation_path=validation_path,
            )
        meta = train_bc._preflight_a1_memmap_metadata(  # noqa: SLF001
            data_path, validation_manifest_path=validation_path
        )
        if meta is None:
            raise ExecutorError("data is not an audited A1 memmap corpus")
        if data_path.is_file():
            return _verify_production_composite_inputs(
                lock=lock,
                lock_path=lock_path,
                reviewed_lock_file_sha256=reviewed_lock_file_sha256,
                recipe=recipe,
                objective=objective,
                producer=_producer(lock),
                data_path=data_path,
                meta=meta,
                validation_path=validation_path,
                build_receipt_path=composite_build_receipt,
                lock_verifier_authority=lock_verifier_authority,
            )
        if composite_build_receipt is not None:
            raise ExecutorError(
                "--composite-build-receipt is valid only for a production composite"
            )
        if validation_path is None:
            raise ExecutorError(
                "ordinary A1 memmap input requires a validation manifest"
            )
        validation = train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
            validation_path,
            validation_fraction=0.05,
            validation_seed=17,
            validation_max_samples=0,
            validation_game_seed_ranges=[],
        )
        train_bc._validate_a1_validation_manifest_corpus_binding(  # noqa: SLF001
            meta, validation
        )
        corpus = train_bc.load_teacher_data_memmap(data_path)
        bound = train_bc._validate_a1_corpus_artifacts_and_seeds(  # noqa: SLF001
            meta,
            validation,
            np.asarray(corpus["game_seed"], dtype=np.int64),
        )
    except (a1_contract.ContractError, SystemExit, OSError, ValueError) as error:
        raise ExecutorError(
            f"A1 training-input verification failed: {error}"
        ) from error

    contract_sha = str(lock["contract_sha256"])
    if validation["a1_contract_sha256"] != contract_sha:
        raise ExecutorError("validation sidecar binds a different A1 contract")
    if bound["learner_training_recipe"] != recipe:
        raise ExecutorError("memmap audit binds a different learner recipe")
    if bound["learner_value_objective"] != objective:
        raise ExecutorError("memmap audit binds a different learner objective")
    producer = _producer(lock)
    if bound["producer_checkpoint_sha256"] != producer.get("sha256"):
        raise ExecutorError("memmap audit producer differs from the sealed producer")

    meta_path = data_path / "corpus_meta.json"
    corpus_row_count = int(meta["row_count"])
    validation_row_count = int(validation["validation_row_count"])
    training_row_count = corpus_row_count - validation_row_count
    if training_row_count <= 0:
        raise ExecutorError("audited A1 corpus has no training rows")
    return {
        "lock": lock,
        "lock_path": lock_path,
        "lock_file_sha256": _file_sha256(lock_path),
        "reviewed_lock_file_sha256": reviewed_lock_file_sha256,
        "lock_verifier_authority": lock_verifier_authority,
        "contract_sha256": contract_sha,
        "recipe": recipe,
        "objective": objective,
        "producer": producer,
        "data_path": data_path,
        "corpus_meta_file_sha256": _file_sha256(meta_path),
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "data_fingerprint": train_bc._training_data_fingerprint(  # noqa: SLF001
            str(data_path), "memmap"
        ),
        "corpus_row_count": corpus_row_count,
        "training_row_count": training_row_count,
        "validation_row_count": validation_row_count,
        "selected_game_seed_set_sha256": bound["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": bound["training_game_seed_set_sha256"],
        "validation_path": validation_path,
        "validation_file_sha256": validation["file_sha256"],
        "validation_game_seed_set_sha256": validation[
            "validation_game_seed_set_sha256"
        ],
    }


def _component_game_identity_sha256(
    component_ids: Sequence[str], component_seed_sets: Sequence[np.ndarray]
) -> str:
    records = [
        {"component_id": component_id, "game_seed": int(seed)}
        for component_id, seeds in zip(component_ids, component_seed_sets, strict=True)
        for seed in sorted(set(map(int, np.asarray(seeds, dtype=np.int64).tolist())))
    ]
    return _value_sha256(records)


def _validate_production_composite_build_receipt(
    *,
    path: Path,
    lock: dict[str, Any],
    lock_path: Path,
    data_path: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Bind the learner to the builder's final atomic commit record."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot load composite build receipt: {error}") from error
    expected = {
        "schema_version",
        "contract",
        "selected_game_manifest",
        "post_wave_audit",
        "historical_component_reference",
        "source_bindings",
        "source_bindings_sha256",
        "source_authority",
        "descriptor",
        "sampling_receipt",
        "verified_descriptor_fingerprint",
        "receipt_sha256",
    }
    receipt_schema = (
        payload.get("schema_version") if isinstance(payload, dict) else None
    )
    if receipt_schema == "a1-post-wave-composite-build-v2":
        expected.add("fresh_target_activation")
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or receipt_schema
        not in {
            "a1-post-wave-composite-build-v1",
            "a1-post-wave-composite-build-v2",
        }
    ):
        raise ExecutorError("composite build receipt fields/schema drift")
    unhashed = dict(payload)
    stated = unhashed.pop("receipt_sha256", None)
    if stated != _value_sha256(unhashed):
        raise ExecutorError("composite build receipt semantic digest drift")
    contract_ref = payload["contract"]
    descriptor_ref = payload["descriptor"]
    authority_ref = payload["source_authority"]
    if (
        not isinstance(contract_ref, dict)
        or set(contract_ref) != {"path", "file_sha256", "contract_sha256"}
        or not Path(str(contract_ref["path"])).is_absolute()
        or contract_ref["file_sha256"] != _file_sha256(lock_path)
        or contract_ref["contract_sha256"] != lock["contract_sha256"]
    ):
        raise ExecutorError("composite build receipt binds a different A1 lock")
    if (
        not isinstance(descriptor_ref, dict)
        or set(descriptor_ref) != {"path", "file_sha256", "fingerprint"}
        or descriptor_ref["path"] != str(data_path)
        or descriptor_ref["file_sha256"] != meta.get("descriptor_file_sha256")
        or descriptor_ref["fingerprint"] != meta.get("descriptor_fingerprint")
        or payload["verified_descriptor_fingerprint"]
        != meta.get("descriptor_fingerprint")
    ):
        raise ExecutorError("composite build receipt descriptor binding drift")
    expected_authority = meta.get("source_authority_ref")
    if (
        not isinstance(authority_ref, dict)
        or authority_ref != expected_authority
        or not isinstance(meta.get("source_authority"), dict)
    ):
        raise ExecutorError("composite build receipt source-authority binding drift")
    current_contract = meta["source_authority"].get("current_contract")
    if (
        not isinstance(current_contract, dict)
        or current_contract.get("file_sha256") != _file_sha256(lock_path)
        or current_contract.get("contract_sha256") != lock["contract_sha256"]
    ):
        raise ExecutorError("composite source authority binds a different A1 lock")
    bindings = payload["source_bindings"]
    if (
        not isinstance(bindings, list)
        or payload["source_bindings_sha256"] != _value_sha256(bindings)
        or bindings != meta["source_authority"].get("fresh_source_bindings")
        or payload["sampling_receipt"]
        != meta.get("production_mix_contract", {}).get("sampling_receipt")
    ):
        raise ExecutorError("composite build receipt input/sampling binding drift")
    if receipt_schema == "a1-post-wave-composite-build-v2":
        activation = payload.get("fresh_target_activation")
        authority_activation = meta["source_authority"].get("fresh_target_activation")
        audit_ref = payload.get("post_wave_audit")
        if (
            not isinstance(activation, dict)
            or activation != authority_activation
            or activation.get("passed") is not True
            or not isinstance(audit_ref, dict)
            or audit_ref.get("target_activation_sha256")
            != activation.get("target_activation_sha256")
        ):
            raise ExecutorError(
                "composite build receipt target-activation binding drift"
            )
    return {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "receipt_sha256": stated,
    }


def _verify_production_composite_inputs(
    *,
    lock: dict[str, Any],
    lock_path: Path,
    reviewed_lock_file_sha256: str | None,
    recipe: dict[str, Any],
    objective: dict[str, Any],
    producer: dict[str, Any],
    data_path: Path,
    meta: dict[str, Any],
    validation_path: Path | None,
    build_receipt_path: Path | None,
    lock_verifier_authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the promotion-eligible 64/12/4/20 replay descriptor.

    The flywheel receipt describes the pre-split physical corpus. This executor
    independently replays train_bc's deterministic component-aware whole-game
    split and binds the realized train/validation mass before constructing an
    optimizer command.
    """

    if validation_path is not None:
        raise ExecutorError(
            "production composite binds its deterministic component-aware split; "
            "do not pass a separate validation manifest"
        )
    if build_receipt_path is None:
        raise ExecutorError(
            "production composite requires its atomic --composite-build-receipt"
        )
    expected_ids = [
        "current_producer",
        "recent_history",
        "hard_negative",
        "historical_replay",
    ]
    expected_ratios = {
        "current_producer": 0.64,
        "recent_history": 0.12,
        "hard_negative": 0.04,
        "historical_replay": 0.20,
    }
    raw_components = meta.get("components")
    if not isinstance(raw_components, list) or len(raw_components) != len(expected_ids):
        raise ExecutorError(
            "production composite lacks authenticated component metadata for "
            "event-history admission"
        )
    component_metadata: dict[str, dict[str, Any]] = {}
    component_payload_scans: dict[str, dict[str, Any]] = {}
    event_history_component_authority: list[dict[str, str]] = []
    for expected_id, component in zip(expected_ids, raw_components, strict=True):
        nested = component.get("corpus_meta") if isinstance(component, dict) else None
        inventory_sha256 = (
            component.get("payload_inventory_sha256")
            if isinstance(component, dict)
            else None
        )
        if (
            not isinstance(component, dict)
            or component.get("component_id") != expected_id
            or not isinstance(nested, dict)
            or not isinstance(inventory_sha256, str)
            or nested.get("payload_inventory_sha256") != inventory_sha256
        ):
            raise ExecutorError(
                "production component event-history authority/order drifted: "
                f"expected={expected_id!r}"
            )
        component_metadata[expected_id] = nested
        payload_scan = nested.get("event_history_payload_scan")
        if payload_scan is not None:
            if not isinstance(payload_scan, dict):
                raise ExecutorError(
                    f"production component {expected_id!r} has malformed "
                    "event-history payload scan"
                )
            component_payload_scans[expected_id] = payload_scan
        event_history_component_authority.append(
            {
                "component_id": expected_id,
                "payload_inventory_sha256": inventory_sha256,
            }
        )
    event_history_acknowledgements: list[str] = []
    try:
        for component_id, component in component_metadata.items():
            corpus_audit = information_surface.audit_memmap_metadata(
                component,
                payload_scan=component_payload_scans.get(component_id),
            )
            if corpus_audit.get("event_history_trainable") is not True:
                event_history_acknowledgements.append(
                    str(component["payload_inventory_sha256"])
                )
    except information_surface.InformationSurfaceError as error:
        raise ExecutorError(
            f"production component event-history audit refused: {error}"
        ) from error
    event_history_acknowledgements.sort()
    try:
        event_history_training_contract = (
            information_surface.build_a1_training_event_history_contract(
                component_metadata,
                graph_history_features=True,
                event_history_consumer_enabled=True,
                empty_payload_inventory_acknowledgements=(
                    event_history_acknowledgements
                ),
                component_payload_scans=component_payload_scans,
            )
        )
    except (KeyError, information_surface.InformationSurfaceError) as error:
        raise ExecutorError(
            f"production composite event-history admission refused: {error}"
        ) from error
    event_history_trainable = event_history_training_contract.get(
        "training_event_history_trainable"
    )
    event_history_usable = event_history_training_contract.get(
        "event_history_end_to_end_usable"
    )
    event_history_status = event_history_training_contract.get("status")
    if not (
        (
            event_history_trainable is True
            and event_history_usable is True
            and event_history_status
            in {
                "verified_nonzero",
                "partially_trainable_with_empty_components_acknowledged",
            }
        )
        or (
            event_history_trainable is False
            and event_history_usable is False
            and event_history_status == "empty_payloads_acknowledged"
        )
    ):
        raise ExecutorError(
            "production composite event-history contract is neither the legacy "
            "authenticated-empty surface nor trainable meaningful public history"
        )
    contract = meta.get("production_mix_contract")
    learner_overrides = meta.get("learner_recipe_overrides")
    if (
        meta.get("schema_version") != "memmap_composite_v2"
        or meta.get("diagnostic_only") is not False
        or meta.get("promotion_eligible") is not True
        or not isinstance(contract, dict)
        or contract.get("schema_version") != "flywheel-replay-composite-v2"
        or meta.get("component_ids") != expected_ids
        or contract.get("fresh_component_ids") != expected_ids[:3]
        or contract.get("replay_component_ids") != expected_ids[3:]
        or contract.get("fresh_source_game_ratios")
        != {"current_producer": 0.8, "recent_history": 0.15, "hard_negative": 0.05}
        or contract.get("effective_component_sampling_ratios") != expected_ratios
        or meta.get("component_game_sampling_ratios")
        != [expected_ratios[value] for value in expected_ids]
        or meta.get("stored_policy_component_temperatures")
        != composite_builder.STORED_POLICY_COMPONENT_TEMPERATURES
        or learner_overrides != composite_builder.LEARNER_RECIPE_OVERRIDES
        or meta.get("policy_kl_anchor_component_ids") != []
        or meta.get("policy_distillation_component_ids") != expected_ids
        or meta.get("value_training_component_ids") != expected_ids
        or not isinstance(meta.get("entity_feature_adapter_component_versions"), dict)
        or set(meta["entity_feature_adapter_component_versions"]) != set(expected_ids)
        or len(set(meta["entity_feature_adapter_component_versions"].values())) != 1
        or not math.isclose(
            float(contract.get("realized_replay_ratio", -1.0)),
            0.20,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ExecutorError("production composite is not exact 64/12/4/20 replay")
    if contract.get("initializer_checkpoint_sha256") != producer.get("sha256"):
        raise ExecutorError(
            "production composite initializer differs from sealed producer"
        )
    # The generation lock predates the post-wave composite, so it binds the
    # base learner recipe with K=0.  The byte-authenticated descriptor preserves
    # that known TEMP control while binding the exact component scopes. Preserve
    # the sealed recipe separately while rendering every authenticated override;
    # no caller-provided delta is admitted here.
    bound_recipe = dict(recipe)
    recipe = dict(recipe)
    recipe.update(learner_overrides)
    build_receipt = _validate_production_composite_build_receipt(
        path=build_receipt_path,
        lock=lock,
        lock_path=lock_path,
        data_path=data_path,
        meta=meta,
    )
    receipt = contract.get("sampling_receipt")
    if (
        not isinstance(receipt, dict)
        or contract.get("sampling_receipt_sha256") != _value_sha256(receipt)
        or receipt.get("effective_component_sampling_ratios") != expected_ratios
    ):
        raise ExecutorError("production composite sampling receipt is invalid")
    components = meta.get("components")
    if not isinstance(components, list) or len(components) != 4:
        raise ExecutorError("production composite must have exactly four components")
    if [component.get("source_category") for component in components] != expected_ids:
        raise ExecutorError("production composite source categories/order drift")
    component_dirs = [Path(str(component["corpus_dir"])) for component in components]
    if len(set(component_dirs)) != 4:
        raise ExecutorError("production composite component corpora are not distinct")

    corpus = train_bc.load_teacher_data_memmap(data_path, composite_meta=meta)
    split = train_bc.split_train_validation_indices(
        corpus,
        validation_fraction=0.05,
        validation_seed=17,
        validation_max_samples=0,
    )
    train_indices = np.asarray(split["train"], dtype=np.int64)
    validation_indices = np.asarray(split["validation"], dtype=np.int64)
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ExecutorError("production composite whole-game split is empty")
    offsets = list(map(int, corpus.component_offsets))
    component_split_records: list[dict[str, Any]] = []
    selected_seed_sets: list[np.ndarray] = []
    training_seed_sets: list[np.ndarray] = []
    validation_seed_sets: list[np.ndarray] = []
    globally_seen_game_seeds: set[int] = set()
    for index, (component_id, component) in enumerate(
        zip(expected_ids, corpus.corpora, strict=True)
    ):
        start, stop = offsets[index], offsets[index + 1]
        seeds = np.asarray(component["game_seed"], dtype=np.int64)
        local_train = (
            train_indices[(train_indices >= start) & (train_indices < stop)] - start
        )
        local_validation = (
            validation_indices[
                (validation_indices >= start) & (validation_indices < stop)
            ]
            - start
        )
        all_games = np.unique(seeds)
        component_game_seeds = set(map(int, all_games))
        overlapping_game_seeds = globally_seen_game_seeds.intersection(
            component_game_seeds
        )
        if overlapping_game_seeds:
            raise ExecutorError(
                "production composite reuses game seeds across components: "
                f"component={component_id} examples={sorted(overlapping_game_seeds)[:8]}"
            )
        globally_seen_game_seeds.update(component_game_seeds)
        train_games = np.unique(seeds[local_train])
        validation_games = np.unique(seeds[local_validation])
        if set(map(int, train_games)).intersection(map(int, validation_games)) or len(
            train_games
        ) + len(validation_games) != len(all_games):
            raise ExecutorError(
                f"component {component_id} validation is not a whole-game partition"
            )
        selected_seed_sets.append(all_games)
        training_seed_sets.append(train_games)
        validation_seed_sets.append(validation_games)
        component_split_records.append(
            {
                "component_id": component_id,
                "game_sampling_ratio": expected_ratios[component_id],
                "selected_game_count": int(len(all_games)),
                "training_game_count": int(len(train_games)),
                "validation_game_count": int(len(validation_games)),
                "row_count": int(stop - start),
                "training_row_count": int(local_train.size),
                "validation_row_count": int(local_validation.size),
                "selected_game_identity_sha256": _component_game_identity_sha256(
                    [component_id], [all_games]
                ),
                "training_game_identity_sha256": _component_game_identity_sha256(
                    [component_id], [train_games]
                ),
                "validation_game_identity_sha256": _component_game_identity_sha256(
                    [component_id], [validation_games]
                ),
            }
        )
    split_receipt = {
        "schema_version": "a1-production-composite-whole-game-split-v1",
        "validation_fraction": 0.05,
        "validation_seed": 17,
        "validation_max_samples": 0,
        "component_sampling_ratios": expected_ratios,
        "components": component_split_records,
        "aggregate": {
            "selected_game_count": sum(
                record["selected_game_count"] for record in component_split_records
            ),
            "training_game_count": sum(
                record["training_game_count"] for record in component_split_records
            ),
            "validation_game_count": sum(
                record["validation_game_count"] for record in component_split_records
            ),
            "row_count": int(len(corpus)),
            "training_row_count": int(train_indices.size),
            "validation_row_count": int(validation_indices.size),
        },
    }
    split_receipt_sha256 = _value_sha256(split_receipt)
    trainer_validation_game_seed_set_sha256 = train_bc._game_seed_set_sha256(  # noqa: SLF001
        np.concatenate(validation_seed_sets)
    )
    descriptor_sha = str(meta["descriptor_file_sha256"])
    return {
        "lock": lock,
        "lock_path": lock_path,
        "lock_file_sha256": _file_sha256(lock_path),
        "reviewed_lock_file_sha256": reviewed_lock_file_sha256,
        "lock_verifier_authority": copy.deepcopy(lock_verifier_authority),
        "contract_sha256": str(lock["contract_sha256"]),
        "bound_recipe": bound_recipe,
        "recipe": recipe,
        "objective": objective,
        "producer": producer,
        "data_path": data_path,
        "data_kind": "production_composite_v2",
        "trainer_authority": _current_production_trainer_authority(),
        "event_history_training_contract": event_history_training_contract,
        "event_history_component_authority": event_history_component_authority,
        "corpus_meta_file_sha256": descriptor_sha,
        "descriptor_fingerprint": meta["descriptor_fingerprint"],
        "learner_recipe_overrides": meta["learner_recipe_overrides"],
        "learner_recipe_overrides_sha256": meta["learner_recipe_overrides_sha256"],
        "entity_feature_adapter_component_versions": dict(
            meta["entity_feature_adapter_component_versions"]
        ),
        "aux_subgoal_target_contract_sha256": meta[
            "aux_subgoal_target_contract_sha256"
        ],
        "public_award_feature_transition_contract_sha256": meta[
            "public_award_feature_transition_contract_sha256"
        ],
        "source_authority_semantic_sha256": meta.get(
            "source_authority_semantic_sha256"
        ),
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "data_fingerprint": train_bc._training_data_fingerprint(  # noqa: SLF001
            str(data_path), "memmap"
        ),
        "corpus_row_count": int(len(corpus)),
        "training_row_count": int(train_indices.size),
        "validation_row_count": int(validation_indices.size),
        "selected_game_seed_set_sha256": _component_game_identity_sha256(
            expected_ids, selected_seed_sets
        ),
        "training_game_seed_set_sha256": _component_game_identity_sha256(
            expected_ids, training_seed_sets
        ),
        "validation_path": data_path,
        "validation_file_sha256": descriptor_sha,
        "validation_game_seed_set_sha256": _component_game_identity_sha256(
            expected_ids, validation_seed_sets
        ),
        "trainer_validation_game_seed_set_sha256": (
            trainer_validation_game_seed_set_sha256
        ),
        "production_mix_contract": contract,
        "source_authority": meta.get("source_authority"),
        "source_authority_ref": meta.get("source_authority_ref"),
        "category_semantics": meta.get("category_semantics"),
        "category_semantics_sha256": meta.get("category_semantics_sha256"),
        "production_sampling_receipt_sha256": contract["sampling_receipt_sha256"],
        "validation_split_receipt": split_receipt,
        "validation_split_receipt_sha256": split_receipt_sha256,
        "composite_build_receipt": build_receipt,
    }


def bind_training_topology(
    verified: dict[str, Any], *, topology: str, gpu: int
) -> dict[str, Any]:
    """Bind one authorized physical topology without changing the dose.

    DDP changes only local batch and world size. The global batch, number of
    epochs, max steps, optimizer freshness, learning rate, and data identity
    remain exactly those of the sealed one-dose recipe.
    """

    spec = TRAINING_TOPOLOGIES.get(topology)
    if spec is None:
        raise ExecutorError(f"unsupported training topology {topology!r}")
    matched_aux = verified.get("learner_ablation", {}).get("matched_aux_regularization")
    central_p1 = verified.get("learner_ablation", {}).get("central_p1")
    central_learner = verified.get("central_learner_binding")
    stage_c_final_binding = verified.get("stage_c_final_replication_binding")
    if (
        matched_aux is not None
        or central_p1 is not None
        or central_learner is not None
        or stage_c_final_binding is not None
    ) and topology != B200_8GPU_DDP_TOPOLOGY:
        raise ExecutorError(
            "central P1/AUX or Stage-C FINAL requires exact b200-8gpu-ddp topology"
        )
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ExecutorError("training topology GPU must be a non-negative integer")
    if topology == B200_8GPU_DDP_TOPOLOGY:
        if gpu != 0:
            raise ExecutorError("8-GPU B200 DDP topology must own physical GPUs 0-7")
        gpus = B200_8GPU_DDP_GPUS
    else:
        gpus = (gpu,)
    bound = dict(verified.get("bound_recipe", verified["recipe"]))
    effective = dict(verified["recipe"])
    if (
        int(bound["global_batch_size"]) != 4096
        or int(bound["world_size"]) != 1
        or int(bound["batch_size"]) != 4096
        or int(bound["grad_accum_steps"]) != 1
    ):
        raise ExecutorError("sealed recipe is not the exact legacy 4096-global dose")
    effective.update(
        {
            "world_size": int(spec["world_size"]),
            "batch_size": int(spec["local_batch_size"]),
            "grad_accum_steps": int(spec["grad_accum_steps"]),
            "global_batch_size": int(spec["global_batch_size"]),
        }
    )
    realized_global = (
        int(effective["batch_size"])
        * int(effective["grad_accum_steps"])
        * int(effective["world_size"])
    )
    if realized_global != int(bound["global_batch_size"]):
        raise ExecutorError("training topology changes the sealed global batch/dose")
    result = dict(verified)
    result.update(
        {
            "bound_recipe": bound,
            "recipe": effective,
            "training_topology": {
                "schema_version": "a1-one-dose-training-topology-v1",
                "name": topology,
                "world_size": int(effective["world_size"]),
                "physical_gpus": list(gpus),
                "local_batch_size": int(effective["batch_size"]),
                "grad_accum_steps": int(effective["grad_accum_steps"]),
                "global_batch_size": realized_global,
                "dose_preserving": True,
            },
        }
    )
    learner_ablation = result.get("learner_ablation")
    if isinstance(learner_ablation, dict):
        learner_ablation = copy.deepcopy(learner_ablation)
        topology_bound_recipe = dict(learner_ablation["bound_recipe"])
        topology_bound_recipe.update(
            {
                "world_size": int(spec["world_size"]),
                "batch_size": int(spec["local_batch_size"]),
                "grad_accum_steps": int(spec["grad_accum_steps"]),
                "global_batch_size": int(spec["global_batch_size"]),
            }
        )
        learner_ablation["bound_recipe"] = topology_bound_recipe
        learner_ablation["bound_recipe_sha256"] = _value_sha256(topology_bound_recipe)
        learner_ablation["effective_recipe"] = dict(effective)
        learner_ablation["effective_recipe_sha256"] = _value_sha256(effective)
        matched_aux = learner_ablation.get("matched_aux_regularization")
        if isinstance(matched_aux, dict):
            shared = dict(matched_aux["shared_identity"])
            shared["training_topology"] = dict(result["training_topology"])
            matched_aux["shared_identity"] = shared
            matched_aux["shared_identity_sha256"] = _value_sha256(shared)
            learner_ablation["matched_aux_regularization"] = matched_aux
        result["learner_ablation"] = learner_ablation
        result["claim_identity_sha256"] = _value_sha256(
            {
                "schema_version": "a1-learner-ablation-claim-identity-v3",
                "contract_sha256": result["contract_sha256"],
                "function_preserving_upgrade": (
                    None
                    if result.get("function_preserving_upgrade") is None
                    else {
                        "module": result["function_preserving_upgrade"]["module"],
                        "receipt_sha256": result["function_preserving_upgrade"][
                            "receipt"
                        ]["sha256"],
                        "receipt_digest": result["function_preserving_upgrade"][
                            "receipt_sha256"
                        ],
                        "initializer_sha256": result["function_preserving_upgrade"][
                            "upgraded_initializer"
                        ]["sha256"],
                    }
                ),
                "ablation": learner_ablation,
                "training_topology": result["training_topology"],
            }
        )
    central_binding = result.get("central_learner_binding")
    if isinstance(central_binding, dict):
        central_binding = copy.deepcopy(central_binding)
        central_binding["effective_recipe"] = dict(effective)
        central_binding["effective_recipe_sha256"] = _value_sha256(effective)
        result["central_learner_binding"] = central_binding
    return result


def _verify_ddp_canary_receipt(
    receipt_path: Path,
    *,
    reference_time_ns: int,
    completed_repository_root: Path | None = None,
) -> dict[str, Any]:
    """Verify one host-local canary at an explicit trusted point in time.

    This private primitive serves two callers with different trust boundaries:
    new launches pass the current wall clock, while completed-receipt replay
    passes the already-authenticated terminal claim start time.  Keeping the
    timestamp out of the public launch binder prevents callers from backdating
    a stale canary to acquire a new dose claim.
    """

    try:
        path = receipt_path.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot load 8-GPU canary receipt: {error}") from error
    if not isinstance(payload, dict):
        raise ExecutorError("8-GPU canary receipt is not an object")
    stated = payload.get("receipt_sha256")
    unhashed = dict(payload)
    unhashed.pop("receipt_sha256", None)
    if stated != _value_sha256(unhashed):
        raise ExecutorError("8-GPU canary receipt semantic digest drift")
    created_unix_ns = payload.get("created_unix_ns")
    now_ns = reference_time_ns
    if (
        payload.get("schema_version") != ddp_canary.SCHEMA
        or payload.get("passed") is not True
        or payload.get("diagnostic_only") is not True
        or payload.get("promotion_eligible") is not False
        or payload.get("hostname") != socket.gethostname()
        or isinstance(created_unix_ns, bool)
        or not isinstance(created_unix_ns, int)
        or isinstance(now_ns, bool)
        or not isinstance(now_ns, int)
        or created_unix_ns > now_ns
        or now_ns - created_unix_ns > MAX_DDP_CANARY_AGE_NS
        or payload.get("world_size") != 8
        or payload.get("local_batch_size") != 512
        or payload.get("global_batch_size") != 4096
        or payload.get("ddp_shard_data") is not False
        or payload.get("training_rng_rank_offset") is not True
        or not isinstance(payload.get("training_rng_contracts"), list)
        or len(payload["training_rng_contracts"]) != 8
        or any(
            not isinstance(contract, dict)
            for contract in payload["training_rng_contracts"]
        )
        or [
            contract.get("effective_torch_seed")
            for contract in payload["training_rng_contracts"]
        ]
        != [ddp_canary.SEED + rank for rank in range(8)]
        or any(
            contract.get("rank_offset_enabled") is not True
            for contract in payload["training_rng_contracts"]
        )
        or not isinstance(payload.get("dropout_probe_sha256_by_rank"), list)
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            for value in payload.get("dropout_probe_sha256_by_rank", [])
        )
        or len(set(payload["dropout_probe_sha256_by_rank"])) != 8
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(payload.get("global_draw_sha256", ""))
        )
        is None
        or not isinstance(payload.get("rank_slice_sha256"), list)
        or len(payload["rank_slice_sha256"]) != 8
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            for value in payload["rank_slice_sha256"]
        )
        or payload.get("distributed_backend") != "nccl"
        or not isinstance(payload.get("cuda_collective"), dict)
        or payload["cuda_collective"].get("operation") != "all_reduce_sum"
        or payload["cuda_collective"].get("expected") != 36.0
        or payload["cuda_collective"].get("actual_by_rank") != [36.0] * 8
        or payload["cuda_collective"].get("passed") is not True
        or payload.get("padded_global_draws")
        != payload.get("local_draws_per_rank", -1) * 8
        or not isinstance(payload.get("gpu_names"), list)
        or len(payload["gpu_names"]) != 8
        or any("B200" not in str(name).upper() for name in payload["gpu_names"])
        or not isinstance(payload.get("gpu_identities"), list)
        or len(payload["gpu_identities"]) != 8
        or any(not isinstance(record, dict) for record in payload["gpu_identities"])
        or [record.get("physical_index") for record in payload["gpu_identities"]]
        != list(range(8))
        or len({record.get("uuid") for record in payload["gpu_identities"]}) != 8
        or len({record.get("pci_bus_id") for record in payload["gpu_identities"]}) != 8
        or [record.get("name") for record in payload["gpu_identities"]]
        != payload["gpu_names"]
    ):
        raise ExecutorError("8-GPU canary did not prove the exact B200 DDP topology")
    expected_root = (
        None
        if completed_repository_root is None
        else completed_repository_root.expanduser().resolve(strict=True)
    )
    expected_files = (
        {
            "tool": Path(ddp_canary.__file__).resolve(),
            "train_bc": Path(train_bc.__file__).resolve(),
        }
        if expected_root is None
        else {
            "tool": expected_root / "tools" / "a1_ddp_epoch_canary.py",
            "train_bc": expected_root / "tools" / "train_bc.py",
        }
    )
    for field, expected_path in expected_files.items():
        if expected_path.is_symlink() or not expected_path.is_file():
            raise ExecutorError(f"8-GPU canary {field} historical code is unavailable")
        expected_path = expected_path.resolve(strict=True)
        record = payload.get(field)
        if (
            not isinstance(record, dict)
            or record.get("path") != str(expected_path)
            or record.get("sha256") != _file_sha256(expected_path)
        ):
            raise ExecutorError(f"8-GPU canary {field} implementation drift")
    runtime_identity = payload.get("runtime_identity")
    python_identity = (
        runtime_identity.get("python") if isinstance(runtime_identity, dict) else None
    )
    if (
        not isinstance(runtime_identity, dict)
        or runtime_identity.get("schema_version")
        != "a1-b200-learner-runtime-identity-v1"
        or not isinstance(python_identity, dict)
        or not isinstance(python_identity.get("implementation"), str)
        or not python_identity.get("implementation")
        or not isinstance(python_identity.get("version"), str)
        or not python_identity.get("version")
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(python_identity.get("executable_sha256", "")),
        )
        is None
        or not isinstance(runtime_identity.get("torch_version"), str)
        or not runtime_identity.get("torch_version")
        or not isinstance(runtime_identity.get("torch_cuda_version"), str)
        or not runtime_identity.get("torch_cuda_version")
        or isinstance(runtime_identity.get("cudnn_version"), bool)
        or not isinstance(runtime_identity.get("cudnn_version"), int)
        or int(runtime_identity["cudnn_version"]) <= 0
        or not isinstance(runtime_identity.get("numpy_version"), str)
        or not runtime_identity.get("numpy_version")
        or not isinstance(runtime_identity.get("nvidia_driver_version"), str)
        or not runtime_identity.get("nvidia_driver_version")
    ):
        raise ExecutorError("8-GPU canary learner runtime identity drift")
    canary_semantics = {
        "schema_version": "a1-ddp-canary-semantic-identity-v1",
        "world_size": payload["world_size"],
        "local_batch_size": payload["local_batch_size"],
        "global_batch_size": payload["global_batch_size"],
        "ddp_shard_data": payload["ddp_shard_data"],
        "training_rng_rank_offset": payload["training_rng_rank_offset"],
        "effective_torch_seeds": [
            contract["effective_torch_seed"]
            for contract in payload["training_rng_contracts"]
        ],
        "dropout_probe_sha256_by_rank": payload["dropout_probe_sha256_by_rank"],
        "global_draw_sha256": payload["global_draw_sha256"],
        "rank_slice_sha256": payload["rank_slice_sha256"],
        "distributed_backend": payload["distributed_backend"],
        "cuda_collective": {
            "operation": payload["cuda_collective"]["operation"],
            "expected": payload["cuda_collective"]["expected"],
            "actual_by_rank": payload["cuda_collective"]["actual_by_rank"],
            "passed": payload["cuda_collective"]["passed"],
        },
        "padded_global_draws": payload["padded_global_draws"],
        "local_draws_per_rank": payload["local_draws_per_rank"],
        # Paths are host-local evidence.  Code bytes are the portable semantic
        # authority that two independent B200 hosts must share.
        "tool_sha256": payload["tool"]["sha256"],
        "train_bc_sha256": payload["train_bc"]["sha256"],
        "runtime_identity": runtime_identity,
    }
    return {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "receipt_sha256": stated,
        "global_draw_sha256": payload["global_draw_sha256"],
        "rank_slice_sha256": payload["rank_slice_sha256"],
        "semantic_identity": canary_semantics,
        "semantic_identity_sha256": _value_sha256(canary_semantics),
    }


def _bind_verified_ddp_canary(
    verified: dict[str, Any], canary: dict[str, Any]
) -> dict[str, Any]:
    result = dict(verified)
    result["ddp_canary"] = canary
    learner_ablation = result.get("learner_ablation")
    if isinstance(learner_ablation, dict) and isinstance(
        learner_ablation.get("matched_aux_regularization"), dict
    ):
        learner_ablation = copy.deepcopy(learner_ablation)
        matched_aux = learner_ablation["matched_aux_regularization"]
        shared = dict(matched_aux["shared_identity"])
        # The exact receipt remains in this arm's input/claim evidence.  The
        # matched scientific identity deliberately excludes hostname, path,
        # GPU UUID, timestamp, and receipt-file bytes so honest arms can run in
        # parallel on different 8xB200 hosts.
        canary_semantics = canary["semantic_identity"]
        shared["ddp_canary_semantics"] = canary_semantics
        shared["ddp_canary_semantics_sha256"] = _value_sha256(canary_semantics)
        matched_aux["shared_identity"] = shared
        matched_aux["shared_identity_sha256"] = _value_sha256(shared)
        learner_ablation["matched_aux_regularization"] = matched_aux
        result["learner_ablation"] = learner_ablation
        upgrade = result["function_preserving_upgrade"]
        result["claim_identity_sha256"] = _value_sha256(
            {
                "schema_version": "a1-learner-ablation-claim-identity-v4",
                "contract_sha256": result["contract_sha256"],
                "function_preserving_upgrade": {
                    "module": upgrade["module"],
                    "receipt_sha256": upgrade["receipt"]["sha256"],
                    "receipt_digest": upgrade["receipt_sha256"],
                    "initializer_sha256": upgrade["upgraded_initializer"]["sha256"],
                },
                "ablation": learner_ablation,
                "training_topology": result["training_topology"],
                "ddp_canary": result["ddp_canary"],
            }
        )
        return bind_aux_subgoal_preclaim_contract(result)
    return result


def bind_ddp_canary(
    verified: dict[str, Any], receipt_path: Path | None
) -> dict[str, Any]:
    """Bind a fresh canary for a new launch using wall-clock freshness only."""

    topology = verified.get("training_topology", {})
    if topology.get("name") != B200_8GPU_DDP_TOPOLOGY:
        if receipt_path is not None:
            raise ExecutorError("DDP canary receipt is valid only for 8-GPU topology")
        return verified
    if receipt_path is None:
        raise ExecutorError("8-GPU production topology requires --ddp-canary-receipt")
    canary = _verify_ddp_canary_receipt(receipt_path, reference_time_ns=time.time_ns())
    return _bind_verified_ddp_canary(verified, canary)


def bind_aux_subgoal_preclaim_contract(
    verified: dict[str, Any],
) -> dict[str, Any]:
    """Replay AUXT's exact trainer contract before consuming its one-dose claim.

    ``train_bc`` still validates the same contract inside every child rank.  This
    outer replay exists because a child-side refusal happens after the durable
    one-dose claim is created.  Missing/version-0/zero-coverage labels are input
    defects, not optimizer attempts, and therefore must fail before that claim.
    """

    matched = verified.get("learner_ablation", {}).get("matched_aux_regularization")
    if not isinstance(matched, dict):
        return verified
    weight = float(matched.get("aux_subgoal_loss_weight", -1.0))
    pair_authority = verified.get("aux_pair_executor_authority")
    if not isinstance(pair_authority, dict):
        raise ExecutorError(
            "corrected matched AUX admission requires central pair authority"
        )
    treatment_weight = pair_authority.get("selected_aux_coefficient")
    if (
        isinstance(treatment_weight, bool)
        or not isinstance(treatment_weight, (int, float))
        or not math.isfinite(float(treatment_weight))
        or not 0.001 <= float(treatment_weight) <= 0.05
    ):
        raise ExecutorError("central AUX treatment coefficient is invalid")
    prior = verified.get("aux_subgoal_preclaim_contract")
    try:
        data_path = Path(verified["data_path"])
        if verified.get("data_kind") == "production_composite_v2":
            meta = train_bc._preflight_a1_memmap_metadata(  # noqa: SLF001
                data_path, validation_manifest_path=None
            )
            if not isinstance(meta, dict):
                raise SystemExit("production composite metadata is unavailable")
            data = train_bc.load_teacher_data_memmap(data_path, composite_meta=meta)
            split = train_bc.split_train_validation_indices(
                data,
                validation_fraction=0.05,
                validation_seed=17,
                validation_max_samples=0,
            )
        else:
            data = train_bc.load_teacher_data_memmap(data_path)
            validation = train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
                verified["validation_path"],
                validation_fraction=0.05,
                validation_seed=17,
                validation_max_samples=0,
                validation_game_seed_ranges=[],
            )
            split = train_bc.split_train_validation_indices(
                data,
                validation_fraction=0.05,
                validation_seed=17,
                validation_max_samples=0,
                validation_game_seeds=np.asarray(
                    validation["game_seeds"], dtype=np.int64
                ),
            )
        train_indices = np.asarray(split["train"], dtype=np.int64)
        if int(train_indices.size) != int(verified["training_row_count"]):
            raise SystemExit(
                "AUX pre-claim split differs from authenticated training rows"
            )
        # Both AUX0 and AUXT must pass the selected treatment-grade admission.
        # Otherwise AUX0 could consume its claim while AUXT refused.
        admission = train_bc._validate_aux_subgoal_training_contract(  # noqa: SLF001
            data, train_indices, loss_weight=float(treatment_weight)
        )
    except (KeyError, OSError, SystemExit, TypeError, ValueError) as error:
        raise ExecutorError(
            f"AUX pre-claim training-contract refusal: {error}"
        ) from error
    contract = {
        "schema_version": "a1-aux-subgoal-preclaim-contract-v2",
        "arm_loss_weight": weight,
        "admission_loss_weight": float(treatment_weight),
        "treatment_grade_admission": admission,
        "treatment_grade_admission_sha256": _value_sha256(admission),
    }
    if prior is not None and prior != contract:
        raise ExecutorError("AUX pre-claim contract drift after corpus replay")
    result = dict(verified)
    result["aux_subgoal_preclaim_contract"] = contract
    learner_ablation = copy.deepcopy(result["learner_ablation"])
    matched = learner_ablation["matched_aux_regularization"]
    shared = dict(matched["shared_identity"])
    shared["aux_subgoal_treatment_admission"] = admission
    shared["aux_subgoal_treatment_admission_sha256"] = _value_sha256(admission)
    matched["shared_identity"] = shared
    matched["shared_identity_sha256"] = _value_sha256(shared)
    learner_ablation["matched_aux_regularization"] = matched
    result["learner_ablation"] = learner_ablation
    result["claim_identity_sha256"] = _expected_final_aux_claim_identity(result)
    return result


def _expected_final_aux_claim_identity(verified: dict[str, Any]) -> str:
    """Derive one relocation/canary-refresh-stable pointer-arm identity."""

    upgrade = verified.get("function_preserving_upgrade")
    learner_ablation = verified.get("learner_ablation")
    preclaim = verified.get("aux_subgoal_preclaim_contract")
    if (
        not isinstance(upgrade, dict)
        or not isinstance(learner_ablation, dict)
        or not isinstance(learner_ablation.get("matched_aux_regularization"), dict)
        or not isinstance(preclaim, dict)
    ):
        raise ExecutorError("matched AUX final identity inputs are incomplete")
    matched = learner_ablation["matched_aux_regularization"]
    shared = matched.get("shared_identity")
    topology = verified.get("training_topology")
    if not isinstance(shared, dict) or not isinstance(topology, dict):
        raise ExecutorError("matched AUX portable identity inputs are incomplete")
    required_shared_digests = (
        "portable_science_identity_sha256",
        "p1_selection_authority_sha256",
        "effective_p1_recipe_sha256",
        "composite_authority_sha256",
        "exact_current_parent_sha256",
        "portable_code_identity_sha256",
        "portable_upgrade_identity_sha256",
        "pointer_upgrade_identity_sha256",
        "warmup_terminal_sha256",
        "gradient_geometry_terminal_sha256",
        "selector_rule_sha256",
        "initializer_sha256",
        "pair_contract_state_sha256",
        "ddp_canary_semantics_sha256",
        "aux_subgoal_treatment_admission_sha256",
    )
    if any(
        re.fullmatch(r"sha256:[0-9a-f]{64}", str(shared.get(field, ""))) is None
        for field in required_shared_digests
    ):
        raise ExecutorError("matched AUX portable identity digest is incomplete")
    portable_science = {
        "pair_id": shared["pair_id"],
        "portable_science_identity_sha256": shared["portable_science_identity_sha256"],
        "p1_selection_authority_sha256": shared["p1_selection_authority_sha256"],
        "effective_recipe_sha256": learner_ablation["effective_recipe_sha256"],
        "effective_p1_recipe_sha256": shared["effective_p1_recipe_sha256"],
        "composite_authority_sha256": shared["composite_authority_sha256"],
        "exact_current_parent_sha256": shared["exact_current_parent_sha256"],
        "portable_code_identity_sha256": shared["portable_code_identity_sha256"],
        "portable_upgrade_identity_sha256": shared["portable_upgrade_identity_sha256"],
        "pointer_upgrade_identity_sha256": shared["pointer_upgrade_identity_sha256"],
        "warmup_terminal_sha256": shared["warmup_terminal_sha256"],
        "gradient_geometry_terminal_sha256": shared[
            "gradient_geometry_terminal_sha256"
        ],
        "selector_rule_sha256": shared["selector_rule_sha256"],
        "selected_aux_coefficient_decimal": shared["selected_aux_coefficient_decimal"],
        "initializer_sha256": shared["initializer_sha256"],
        "pair_contract_state_sha256": shared["pair_contract_state_sha256"],
        "aux_pair_authority_sha256": matched["aux_pair_authority_sha256"],
        "training_topology": topology,
        "ddp_canary_semantics_sha256": shared["ddp_canary_semantics_sha256"],
        "aux_subgoal_treatment_admission_sha256": shared[
            "aux_subgoal_treatment_admission_sha256"
        ],
    }
    return _value_sha256(
        {
            "schema_version": "a1-aux-pointer-arm-claim-identity-v1",
            "contract_sha256": verified["contract_sha256"],
            "arm_id": matched["arm_id"],
            "arm_loss_weight": matched["aux_subgoal_loss_weight"],
            "portable_science": portable_science,
        }
    )


def bind_learner_ablation(
    verified: dict[str, Any],
    *,
    ablation_id: str,
    overrides_json: str,
    reviewed_code_tree_sha256: str,
    diagnostic_dose_curve: bool = False,
    diagnostic_checkpoint_steps: str = "",
    _authenticated_completed_code_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive a diagnostic learner recipe without weakening the sealed inputs."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", ablation_id):
        raise ExecutorError(
            "--ablation-id must be a nonempty 1-80 character safe identifier"
        )

    def _reject_json_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        overrides = json.loads(overrides_json, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ExecutorError(f"invalid --recipe-overrides-json: {error}") from error
    if not isinstance(overrides, dict) or not overrides:
        raise ExecutorError("--recipe-overrides-json must be a nonempty JSON object")
    if diagnostic_checkpoint_steps and not diagnostic_dose_curve:
        raise ExecutorError(
            "--diagnostic-checkpoint-steps requires --diagnostic-dose-curve"
        )
    forbidden = set(overrides) - A1_LEARNER_ABLATION_FIELDS
    if forbidden:
        raise ExecutorError(
            "A1 learner ablation may not change sealed topology/input fields: "
            f"{sorted(forbidden)}"
        )
    upgrade = verified.get("function_preserving_upgrade")
    if (
        isinstance(upgrade, dict)
        and upgrade.get("module") == LEGACY_AUX_REGULARIZATION_MODULE
        and "aux_subgoal_loss_weight" in overrides
    ):
        raise ExecutorError(
            "settlement_aux_target_aliasing: legacy CLS absolute-vertex "
            "auxiliary arms are retired; use the pointer commissioning contract"
        )
    aux_upgrade = bool(
        isinstance(upgrade, dict) and upgrade.get("module") == AUX_REGULARIZATION_MODULE
    )
    if aux_upgrade:
        raise ExecutorError(
            "corrected pointer AUX arms require central warmup/geometry/pair "
            "authority; generic ablation ids and caller-chosen coefficients are "
            "not admissible"
        )
    generic_ablation_upgrade = bool(
        isinstance(upgrade, dict)
        and upgrade.get("module")
        in {
            architecture_upgrade.MODULE_TARGET_GATHER,
            architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES,
            architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES_V2,
            architecture_upgrade.MODULE_MEANINGFUL_PUBLIC_HISTORY,
            architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY,
            architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2,
            architecture_upgrade.MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY,
            architecture_upgrade.MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY_FROM_V1,
        }
    )
    # These reviewed modules are exact zero-output initializers. The generic
    # ablation receipt binds the initializer identity and full effective loss
    # recipe; arbitrary architecture deltas remain inadmissible.
    if upgrade is not None and not generic_ablation_upgrade:
        raise ExecutorError(
            "a generic learner ablation may be combined only with a reviewed "
            "function-preserving action-target-gather, public-card, or "
            "meaningful-history upgrade"
        )
    if "aux_subgoal_loss_weight" in overrides and not aux_upgrade:
        raise ExecutorError(
            "aux_subgoal_loss_weight ablation requires the receipt-backed "
            "shared aux-head initializer"
        )
    if "public_card_lr_mult" in overrides and not (
        isinstance(upgrade, dict)
        and upgrade.get("module")
        in {
            architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES,
            architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES_V2,
            architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY,
            architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2,
        }
    ):
        raise ExecutorError(
            "public_card_lr_mult requires the receipt-backed public-card "
            "function-preserving initializer"
        )
    bound = dict(verified["recipe"])
    effective = dict(bound)
    # Added after the original sealed A1 lock. Default 1.0 is the exact
    # historical behavior; only a receipt-backed public-card initializer may
    # request a different value below.
    effective["public_card_lr_mult"] = 1.0
    effective["per_game_policy_surprise_weighting"] = False
    # Historical A1 omitted this typed knob because per-game weighting was
    # locked off. Bind the then-current train_bc default explicitly in every
    # derived recipe so enabling the existing CAT-60 path can never silently
    # mean equal when an operator intended sqrt (or vice versa).
    effective["per_game_value_weight_mode"] = "equal"
    if "value_trunk_grad_scale" in overrides:
        effective["value_trunk_grad_scale"] = 1.0
    for key, value in overrides.items():
        if key == "forced_row_value_action_type_weights":
            if type(value) is not str:
                raise ExecutorError(
                    "A1 learner ablation forced_row_value_action_type_weights "
                    "must preserve JSON type str"
                )
            try:
                parsed_type_weights = (
                    train_bc._parse_forced_row_value_action_type_weights(value)
                )
            except SystemExit as error:
                raise ExecutorError(str(error)) from error
            if not parsed_type_weights:
                raise ExecutorError(
                    "A1 learner ablation forced action-type map must be nonempty"
                )
            effective[key] = train_bc._canonical_forced_row_value_action_type_weights(
                parsed_type_weights
            )
            continue
        if key == "per_game_value_weight_mode":
            if value not in {"equal", "sqrt"}:
                raise ExecutorError(
                    "per_game_value_weight_mode must be 'equal' or 'sqrt'"
                )
            effective[key] = value
            continue
        expected_type = (
            int
            if key == "policy_aux_active_batch_size"
            else float
            if key
            in {
                "value_trunk_grad_scale",
                "public_card_lr_mult",
                "trunk_lr_mult",
                "policy_kl_target",
                "policy_kl_dual_lr",
                "policy_kl_max_weight",
            }
            else bool
            if key == "per_game_policy_surprise_weighting"
            else str
            if key == "policy_kl_anchor_direction"
            else type(bound[key])
        )
        if type(value) is not expected_type:
            raise ExecutorError(
                f"A1 learner ablation {key!r} must preserve JSON type "
                f"{expected_type.__name__}, got {type(value).__name__}"
            )
        effective[key] = value
    numeric_domains: dict[str, tuple[float | None, float | None, bool]] = {
        "epochs": (1.0, None, True),
        "max_steps": (1.0, None, True),
        "lr": (0.0, None, False),
        "lr_warmup_steps": (0.0, None, True),
        "value_lr_mult": (0.0, None, False),
        "public_card_lr_mult": (0.0, None, False),
        "trunk_lr_mult": (0.0, 1.0, False),
        "value_trunk_grad_scale": (0.0, 1.0, True),
        "policy_loss_weight": (0.0, None, True),
        "policy_aux_active_batch_size": (0.0, None, True),
        "soft_target_weight": (0.0, 1.0, True),
        "soft_target_temperature": (0.0, None, False),
        "soft_target_min_legal_coverage": (0.0, 1.0, True),
        "value_loss_weight": (0.0, None, True),
        "final_vp_loss_weight": (0.0, None, True),
        "aux_subgoal_loss_weight": (0.0, 0.05, True),
        "policy_kl_anchor_weight": (0.0, None, True),
        "policy_kl_target": (0.0, None, True),
        "policy_kl_dual_lr": (0.0, None, False),
        "policy_kl_max_weight": (0.0, None, False),
        "policy_surprise_weight": (0.0, None, True),
        "vp_margin_weight": (0.0, None, True),
        "truncated_vp_margin_value_weight": (0.0, None, True),
        "forced_action_weight": (0.0, None, True),
        "forced_row_value_weight": (0.0, None, True),
        "winner_sample_weight": (0.0, None, True),
        "loser_sample_weight": (0.0, None, True),
    }
    enum_domains = {
        "lr_schedule": {"flat", "cosine", "linear"},
        "soft_target_source": {"prefer_policy", "prefer_scores", "policy", "scores"},
        "advantage_policy_weighting": {"none", "outcome_value"},
        "per_game_policy_weight_mode": {"equal", "sqrt"},
        "per_game_value_weight_mode": {"equal", "sqrt"},
        "policy_kl_anchor_direction": {"forward"},
    }
    for key, value in overrides.items():
        if key in numeric_domains:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ExecutorError(f"A1 learner ablation {key} must be finite")
            minimum, maximum, minimum_inclusive = numeric_domains[key]
            if minimum is not None and (
                numeric < minimum if minimum_inclusive else numeric <= minimum
            ):
                relation = ">=" if minimum_inclusive else ">"
                raise ExecutorError(
                    f"A1 learner ablation {key} must be {relation} {minimum}"
                )
            if maximum is not None and numeric > maximum:
                raise ExecutorError(f"A1 learner ablation {key} must be <= {maximum}")
        if key in enum_domains and value not in enum_domains[key]:
            raise ExecutorError(
                f"A1 learner ablation {key} must be one of {sorted(enum_domains[key])}"
            )
    if (
        effective["per_game_value_weight_mode"] == "sqrt"
        and not effective["per_game_value_weight"]
    ):
        raise ExecutorError(
            "per_game_value_weight_mode=sqrt requires per_game_value_weight=true"
        )
    if (
        "soft_target_temperature" in overrides
        and effective["soft_target_source"] == "policy"
    ):
        raise ExecutorError(
            "soft_target_temperature is inert for soft_target_source=policy; "
            "do not encode a fake ablation drift"
        )
    adaptive_fields = {
        "policy_kl_target",
        "policy_kl_dual_lr",
        "policy_kl_max_weight",
        "policy_kl_anchor_direction",
    }
    requested_adaptive_fields = adaptive_fields & set(overrides)
    if requested_adaptive_fields:
        if "policy_kl_target" not in requested_adaptive_fields:
            raise ExecutorError("adaptive policy-KL options require policy_kl_target")
        missing = adaptive_fields - requested_adaptive_fields
        if missing:
            raise ExecutorError(
                "adaptive policy-KL ablation must bind its complete controller: "
                f"missing {sorted(missing)}"
            )
        if effective["policy_kl_anchor_direction"] != "forward":
            raise ExecutorError(
                "adaptive policy-KL ablation requires forward parent KL"
            )
        if float(effective["policy_kl_anchor_weight"]) > float(
            effective["policy_kl_max_weight"]
        ):
            raise ExecutorError("adaptive policy-KL initial weight exceeds its maximum")
    active_objective_mass = sum(
        float(effective[key])
        for key in (
            "policy_loss_weight",
            "value_loss_weight",
            "final_vp_loss_weight",
            "policy_kl_anchor_weight",
        )
    )
    if active_objective_mass <= 0.0:
        raise ExecutorError(
            "A1 learner ablation disables every active training objective"
        )
    try:
        checkpoint_steps = train_bc._parse_checkpoint_steps(  # noqa: SLF001
            diagnostic_checkpoint_steps,
            max_steps=int(effective["max_steps"]),
        )
    except SystemExit as error:
        raise ExecutorError(str(error)) from error
    aux_arm: str | None = None
    drift = {
        key: {"contract": bound[key], "effective": effective[key]}
        for key in sorted(set(overrides) & set(bound))
        if bound[key] != effective[key]
    }
    if effective["per_game_value_weight_mode"] != "equal":
        drift["per_game_value_weight_mode"] = {
            "contract": "equal (implicit train_bc default; weighting locked off)",
            "effective": effective["per_game_value_weight_mode"],
        }
    if effective.get("policy_aux_active_batch_size", 0) != 0:
        drift["policy_aux_active_batch_size"] = {
            "contract": 0,
            "effective": effective["policy_aux_active_batch_size"],
        }
    if effective.get("value_trunk_grad_scale", 1.0) != 1.0:
        drift["value_trunk_grad_scale"] = {
            "contract": 1.0,
            "effective": effective["value_trunk_grad_scale"],
        }
    if effective.get("public_card_lr_mult", 1.0) != 1.0:
        drift["public_card_lr_mult"] = {
            "contract": 1.0,
            "effective": effective["public_card_lr_mult"],
        }
    if effective.get("trunk_lr_mult", 1.0) != 1.0:
        drift["trunk_lr_mult"] = {
            "contract": float(bound.get("trunk_lr_mult", 1.0)),
            "effective": effective["trunk_lr_mult"],
        }
    if effective.get("per_game_policy_surprise_weighting", False):
        drift["per_game_policy_surprise_weighting"] = {
            "contract": False,
            "effective": True,
        }
    if effective.get("forced_row_value_action_type_weights", ""):
        drift["forced_row_value_action_type_weights"] = {
            "contract": "disabled (implicit historical default)",
            "effective": effective["forced_row_value_action_type_weights"],
        }
    if effective.get("policy_kl_target") is not None:
        for key in (
            "policy_kl_target",
            "policy_kl_dual_lr",
            "policy_kl_max_weight",
            "policy_kl_anchor_direction",
        ):
            drift[key] = {
                "contract": bound.get(key, "disabled (implicit historical default)"),
                "effective": effective[key],
            }
    if not drift and aux_arm != "AUX0":
        raise ExecutorError("A1 learner ablation is a no-op")
    code_binding = (
        _current_ablation_code_binding(verified["lock"])
        if _authenticated_completed_code_binding is None
        else _verify_completed_ablation_code_binding(
            _authenticated_completed_code_binding, lock=verified["lock"]
        )
    )
    portable_code_identity = _portable_ablation_code_identity(code_binding)
    reviewed_lock_sha = verified.get("reviewed_lock_file_sha256")
    if reviewed_lock_sha != verified.get("lock_file_sha256"):
        raise ExecutorError("A1 ablation does not bind the reviewed raw lock bytes")
    if reviewed_code_tree_sha256 != code_binding["code_tree_sha256"]:
        raise ExecutorError(
            "current ablation code tree does not match the explicitly reviewed digest: "
            f"reviewed={reviewed_code_tree_sha256!r} "
            f"current={code_binding['code_tree_sha256']!r}"
        )
    result = dict(verified)
    result["bound_recipe"] = bound
    result["recipe"] = effective
    matched_aux_regularization = None
    if aux_arm is not None:
        assert isinstance(upgrade, dict)
        receipt = upgrade["receipt"]
        initializer = upgrade["upgraded_initializer"]
        portable_upgrade_identity = _portable_upgrade_identity(upgrade)
        shared_identity = {
            "schema_version": "a1-aux-regularization-shared-identity-v2",
            "contract_sha256": verified["contract_sha256"],
            "bound_recipe_sha256": _value_sha256(bound),
            "producer_checkpoint_sha256": verified["producer"]["sha256"],
            "payload_inventory_sha256": verified.get("payload_inventory_sha256"),
            "data_fingerprint": verified.get("data_fingerprint"),
            "training_game_seed_set_sha256": verified.get(
                "training_game_seed_set_sha256"
            ),
            "validation_game_seed_set_sha256": verified.get(
                "validation_game_seed_set_sha256"
            ),
            "upgrade_module": upgrade["module"],
            "upgrade_receipt_file_sha256": receipt["sha256"],
            "upgrade_receipt_digest": upgrade["receipt_sha256"],
            "initializer_sha256": initializer["sha256"],
            "portable_code_identity": portable_code_identity,
            "portable_code_identity_sha256": portable_code_identity["code_sha256"],
            "portable_upgrade_identity": portable_upgrade_identity,
            "portable_upgrade_identity_sha256": portable_upgrade_identity[
                "identity_sha256"
            ],
        }
        matched_aux_regularization = {
            "schema_version": "a1-matched-aux-regularization-arm-v1",
            "arm_id": aux_arm,
            "aux_subgoal_loss_weight": float(effective["aux_subgoal_loss_weight"]),
            "upgrade_module": upgrade["module"],
            "upgrade_receipt": receipt["path"],
            "upgrade_receipt_file_sha256": receipt["sha256"],
            "upgrade_receipt_digest": upgrade["receipt_sha256"],
            "initializer": initializer["path"],
            "initializer_sha256": initializer["sha256"],
            "shared_identity": shared_identity,
            "shared_identity_sha256": _value_sha256(shared_identity),
        }
    result["learner_ablation"] = {
        "schema_version": "a1-learner-ablation-v1",
        "ablation_id": ablation_id,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "promotion_block_reason": "requires_normal_evidence_packaging_after_ablation",
        "bound_recipe": bound,
        "bound_recipe_sha256": _value_sha256(bound),
        "effective_recipe": effective,
        "effective_recipe_sha256": _value_sha256(effective),
        "recipe_drift": drift,
        "recipe_drift_sha256": _value_sha256(drift),
        "code_binding": code_binding,
        "code_tree_sha256": code_binding["code_tree_sha256"],
        "reviewed_lock_file_sha256": reviewed_lock_sha,
        "reporting_contract": {
            "diagnostic_dose_curve": bool(diagnostic_dose_curve),
            "checkpoint_steps": list(checkpoint_steps),
            "train_diagnostics_every_batches": (16 if diagnostic_dose_curve else 0),
            # Two observations (steps 64 and 128) are enough to compare the
            # four exposure arms without turning objective attribution into a
            # material fraction of the dose. Each observation reuses the real
            # forward graph and never mutates Parameter.grad or optimizer state.
            "objective_gradient_interference_every_batches": (
                64 if diagnostic_dose_curve else 0
            ),
            "optimizer_trajectory_unchanged": True,
        },
        **(
            {"matched_aux_regularization": matched_aux_regularization}
            if matched_aux_regularization is not None
            else {}
        ),
    }
    if matched_aux_regularization is not None:
        # Pair admission is part of arm issuance, not a late child-side check.
        # Both arms therefore refuse malformed labels before either can acquire
        # an irreversible dose claim.
        result = bind_aux_subgoal_preclaim_contract(result)
    result["claim_identity_sha256"] = _value_sha256(
        {
            "schema_version": "a1-learner-ablation-claim-identity-v3",
            "contract_sha256": verified["contract_sha256"],
            "function_preserving_upgrade": (
                None
                if upgrade is None
                else {
                    "module": upgrade["module"],
                    "receipt_sha256": upgrade["receipt"]["sha256"],
                    "receipt_digest": upgrade["receipt_sha256"],
                    "initializer_sha256": upgrade["upgraded_initializer"]["sha256"],
                }
            ),
            "diagnostic_comparison_source": result.get("diagnostic_comparison_source"),
            "learner_lineage_parent": result.get("learner_lineage_parent"),
            "ablation": result["learner_ablation"],
        }
    )
    return result


FRESH_POLICY_DISTILLATION_COMPONENT_IDS = (
    "current_producer",
    "recent_history",
    "hard_negative",
)
FRESH_VALUE_TRAINING_COMPONENT_IDS = FRESH_POLICY_DISTILLATION_COMPONENT_IDS
ALL_POST_WAVE_COMPONENT_IDS = (
    *FRESH_POLICY_DISTILLATION_COMPONENT_IDS,
    "historical_replay",
)
DIAGNOSTIC_TRAINING_DESCRIPTOR_SCHEMA = "a1-diagnostic-training-descriptor-authority-v1"


def bind_diagnostic_training_descriptor(
    verified: dict[str, Any],
    *,
    descriptor_path: Path,
    fresh_policy_distillation_only: bool = False,
    fresh_value_training_only: bool = False,
) -> dict[str, Any]:
    """Derive one byte-bound diagnostic descriptor from the production input.

    ``train_bc`` authenticates several argv values through the composite
    descriptor. A recipe ablation therefore cannot merely change argv: the
    descriptor must carry the same reviewed delta. This helper derives that
    input without mutating the published production descriptor. It also owns
    the fresh-policy-only treatment, whose value scope deliberately remains
    all-component.
    """

    if verified.get("data_kind") != "production_composite_v2":
        if fresh_policy_distillation_only or fresh_value_training_only:
            raise ExecutorError(
                "fresh policy/value scope requires a post-wave composite"
            )
        return verified
    learner_ablation = verified.get("learner_ablation")
    if not isinstance(learner_ablation, dict) or isinstance(
        verified.get("central_learner_binding"), dict
    ):
        if fresh_policy_distillation_only or fresh_value_training_only:
            raise ExecutorError(
                "fresh policy/value scope requires a generic diagnostic ablation"
            )
        return verified

    base_path, base_file_sha256 = _stable_canonical_regular_file(
        verified["data_path"], where="base production training descriptor"
    )
    try:
        base = json.loads(base_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(
            f"cannot read base production descriptor: {error}"
        ) from error
    if (
        not isinstance(base, dict)
        or base_file_sha256 != verified["corpus_meta_file_sha256"]
        or _value_sha256(base) != verified["descriptor_fingerprint"]
        or tuple(base.get("policy_distillation_component_ids", ()))
        != ALL_POST_WAVE_COMPONENT_IDS
        or tuple(base.get("value_training_component_ids", ()))
        != ALL_POST_WAVE_COMPONENT_IDS
    ):
        raise ExecutorError("base production descriptor identity/scope drifted")

    base_overrides = base.get("learner_recipe_overrides")
    if not isinstance(base_overrides, dict) or not base_overrides:
        raise ExecutorError(
            "base production descriptor has no learner override authority"
        )
    effective_recipe = verified["recipe"]
    derived_overrides: dict[str, Any] = {}
    for key in base_overrides:
        if key not in effective_recipe:
            raise ExecutorError(
                f"effective learner recipe lost descriptor-authorized field {key!r}"
            )
        derived_overrides[key] = effective_recipe[key]
    typed_forced_key = "forced_row_value_action_type_weights"
    if effective_recipe.get(typed_forced_key):
        # This additive diagnostic field intentionally does not exist in the
        # immutable production descriptor. Its canonical value is carried by
        # the derived descriptor and replayed by train_bc before optimizer use.
        derived_overrides[typed_forced_key] = effective_recipe[typed_forced_key]
    for key, disabled_value in (
        ("public_card_lr_mult", 1.0),
        ("trunk_lr_mult", 1.0),
        ("per_game_policy_surprise_weighting", False),
    ):
        if effective_recipe.get(key, disabled_value) != disabled_value:
            # These are coupled to the receipt-backed public-card initializer
            # and exact per-game sampler by the generic ablation authority.
            derived_overrides[key] = effective_recipe[key]
    if effective_recipe.get("policy_kl_target") is not None:
        for key in (
            "policy_kl_anchor_weight",
            "policy_kl_anchor_direction",
            "policy_kl_target",
            "policy_kl_dual_lr",
            "policy_kl_max_weight",
        ):
            derived_overrides[key] = effective_recipe[key]
    reporting_contract = learner_ablation.get("reporting_contract")
    lr_dose_campaign = bool(
        isinstance(reporting_contract, dict)
        and reporting_contract.get("diagnostic_dose_curve") is True
    )
    if lr_dose_campaign:
        # This is the narrow, diagnostic-only LR/dose campaign surface.  These
        # optimizer fields are absent from the immutable production composite,
        # so copy their exact effective values into the derived descriptor for
        # an independent train_bc replay instead of relying on argv alone.
        for key in (
            "epochs",
            "max_steps",
            "lr",
            "lr_warmup_steps",
            "lr_schedule",
        ):
            derived_overrides[key] = effective_recipe[key]
        policy_aux_batch = int(effective_recipe.get("policy_aux_active_batch_size", 0))
        if policy_aux_batch > 0:
            derived_overrides["policy_aux_active_batch_size"] = policy_aux_batch

    derived = copy.deepcopy(base)
    derived["learner_recipe_overrides"] = derived_overrides
    derived["learner_recipe_overrides_sha256"] = _value_sha256(derived_overrides)
    if fresh_policy_distillation_only or fresh_value_training_only:
        if (
            derived_overrides.get("per_game_policy_weight") is not False
            or derived_overrides.get("per_game_policy_weight_mode") != "equal"
        ):
            raise ExecutorError(
                "fresh policy/value scopes require the selected per-game-policy-"
                "weight-off/equal diagnostic baseline"
            )
    if fresh_policy_distillation_only:
        derived["policy_distillation_component_ids"] = list(
            FRESH_POLICY_DISTILLATION_COMPONENT_IDS
        )
    if fresh_value_training_only:
        derived["value_training_component_ids"] = list(
            FRESH_VALUE_TRAINING_COMPONENT_IDS
        )

    semantic_delta: dict[str, Any] = {}
    if derived_overrides != base_overrides:
        semantic_delta["learner_recipe_overrides"] = {
            "base": copy.deepcopy(base_overrides),
            "effective": copy.deepcopy(derived_overrides),
        }
    if (
        derived["policy_distillation_component_ids"]
        != base["policy_distillation_component_ids"]
    ):
        semantic_delta["policy_distillation_component_ids"] = {
            "base": copy.deepcopy(base["policy_distillation_component_ids"]),
            "effective": copy.deepcopy(derived["policy_distillation_component_ids"]),
        }
    if derived["value_training_component_ids"] != base["value_training_component_ids"]:
        semantic_delta["value_training_component_ids"] = {
            "base": copy.deepcopy(base["value_training_component_ids"]),
            "effective": copy.deepcopy(derived["value_training_component_ids"]),
        }
    if not semantic_delta:
        return verified

    # The trainer independently replays the unchanged production descriptor
    # and accepts only these two reviewed diagnostic deltas. Marking the
    # derived bytes diagnostic prevents an optimization treatment from
    # masquerading as the promotion-eligible production recipe.
    derived["diagnostic_only"] = True
    derived["promotion_eligible"] = False
    diagnostic_derivation_authority = {
        "schema_version": train_bc.FLYWHEEL_DIAGNOSTIC_DERIVATION_SCHEMA,
        "base_descriptor": {
            "path": str(base_path),
            "file_sha256": base_file_sha256,
            "fingerprint": verified["descriptor_fingerprint"],
        },
        "semantic_delta": copy.deepcopy(semantic_delta),
        "semantic_delta_sha256": _value_sha256(semantic_delta),
        "diagnostic_only": True,
        "promotion_eligible": False,
    }
    derived["diagnostic_derivation_authority"] = diagnostic_derivation_authority

    lexical = descriptor_path.expanduser().absolute()
    if (
        lexical == base_path
        or lexical.parent != lexical.parent.resolve(strict=False)
        or lexical.is_symlink()
    ):
        raise ExecutorError(
            "diagnostic training descriptor path must be distinct, lexical, "
            "absolute, and non-symlink"
        )
    derived_bytes = (
        json.dumps(derived, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    derived_file_sha256 = "sha256:" + hashlib.sha256(derived_bytes).hexdigest()
    derived_fingerprint = _value_sha256(derived)
    authority = {
        "schema_version": DIAGNOSTIC_TRAINING_DESCRIPTOR_SCHEMA,
        "base_descriptor": {
            "path": str(base_path),
            "file_sha256": base_file_sha256,
            "fingerprint": verified["descriptor_fingerprint"],
        },
        "derived_descriptor": {
            "path": str(lexical),
            "file_sha256": derived_file_sha256,
            "fingerprint": derived_fingerprint,
        },
        "semantic_delta": semantic_delta,
        "semantic_delta_sha256": _value_sha256(semantic_delta),
        "diagnostic_derivation_authority": diagnostic_derivation_authority,
        "learner_recipe_overrides": copy.deepcopy(derived_overrides),
        "learner_recipe_overrides_sha256": _value_sha256(derived_overrides),
        "policy_distillation_component_ids": copy.deepcopy(
            derived["policy_distillation_component_ids"]
        ),
        "value_training_component_ids": copy.deepcopy(
            derived["value_training_component_ids"]
        ),
        "diagnostic_only": True,
        "promotion_eligible": False,
    }
    authority["authority_sha256"] = _value_sha256(authority)

    result = dict(verified)
    result.update(
        {
            "data_path": lexical,
            "corpus_meta_file_sha256": derived_file_sha256,
            "descriptor_fingerprint": derived_fingerprint,
            "data_fingerprint": derived_fingerprint,
            "validation_path": lexical,
            "validation_file_sha256": derived_file_sha256,
            "learner_recipe_overrides": copy.deepcopy(derived_overrides),
            "learner_recipe_overrides_sha256": _value_sha256(derived_overrides),
            "diagnostic_training_descriptor_authority": authority,
            "_diagnostic_training_descriptor_bytes": derived_bytes,
        }
    )
    bound_ablation = copy.deepcopy(learner_ablation)
    bound_ablation["training_descriptor_authority"] = authority
    result["learner_ablation"] = bound_ablation
    result["claim_identity_sha256"] = _value_sha256(
        {
            "schema_version": "a1-learner-ablation-claim-identity-v3",
            "contract_sha256": verified["contract_sha256"],
            "function_preserving_upgrade": verified.get("function_preserving_upgrade"),
            "ablation": bound_ablation,
        }
    )
    return result


def _materialize_diagnostic_training_descriptor(verified: Mapping[str, Any]) -> None:
    """Publish the planned derived descriptor once, without overwrite."""

    authority = verified.get("diagnostic_training_descriptor_authority")
    raw_bytes = verified.get("_diagnostic_training_descriptor_bytes")
    if authority is None and raw_bytes is None:
        return
    if not isinstance(authority, dict) or not isinstance(raw_bytes, bytes):
        raise ExecutorError(
            "diagnostic training descriptor materialization is incomplete"
        )
    target = Path(authority["derived_descriptor"]["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if (
            target.is_symlink()
            or not target.is_file()
            or _file_sha256(target) != authority["derived_descriptor"]["file_sha256"]
            or stat.S_IMODE(target.stat().st_mode) != 0o444
        ):
            raise ExecutorError("existing diagnostic training descriptor differs")
        return
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if (
                target.is_symlink()
                or not target.is_file()
                or _file_sha256(target)
                != authority["derived_descriptor"]["file_sha256"]
            ):
                raise ExecutorError("raced diagnostic training descriptor differs")
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _central_learner_binding(
    *,
    stage: str,
    central_authority_schema: str,
    central_authority_sha256: str,
    selected_aux_decision: str | None,
    effective_recipe: dict[str, Any],
    immutable_contract_recipe: dict[str, Any],
    sample_receipt: dict[str, Any],
    initializer_sha256: str,
    code_binding: dict[str, Any],
    reviewed_lock_file_sha256: str,
    published_executor_authority: dict[str, Any],
) -> dict[str, Any]:
    if stage not in {"P1", AUX_CONTROL_ARM, AUX_TREATMENT_ARM, "FINAL"}:
        raise ExecutorError(f"unsupported central learner stage {stage!r}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", central_authority_sha256):
        raise ExecutorError("central learner authority digest is malformed")
    required_sample = {
        "state_sha256",
        "descriptor_sha256",
        "payload_inventory_sha256",
        "category_semantics",
        "category_semantics_sha256",
        "source_authority",
        "sampler_identity_sha256",
        "sample_order_sha256",
        "row_set_sha256",
        "unique_row_count",
        "rows_file_sha256",
        "sample_dose",
        "sampler_seed",
        "prior_rows_file_sha256",
        "prior_row_set_sha256",
        "kl_eligible_rows",
        "kl_eligible_mass_decimal",
        "kl_ordered_evidence_sha256",
        "kl_eligible_evidence_sha256",
    }
    published = published_executor_authority
    published_authority = (
        published.get("authority") if isinstance(published, dict) else None
    )
    if (
        not isinstance(sample_receipt, dict)
        or not required_sample <= set(sample_receipt)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", initializer_sha256)
        or not isinstance(published_authority, dict)
        or published.get("schema_version")
        != aux_coordinator.PUBLISHED_EXECUTOR_AUTHORITY_SCHEMA
        or published_authority.get("schema_version") != central_authority_schema
        or published_authority.get("authority_sha256") != central_authority_sha256
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(published.get("file_sha256")))
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(published_authority.get("state_sha256")),
        )
    ):
        raise ExecutorError(
            "central learner sample/initializer/published authority is incomplete"
        )
    sample_binding = {
        "schema_version": "a1-central-sample-binding-v2",
        "sample_receipt_state_sha256": sample_receipt["state_sha256"],
        "descriptor_sha256": sample_receipt["descriptor_sha256"],
        "payload_inventory_sha256": sample_receipt["payload_inventory_sha256"],
        "category_semantics": copy.deepcopy(sample_receipt["category_semantics"]),
        "category_semantics_sha256": sample_receipt["category_semantics_sha256"],
        "source_authority": copy.deepcopy(sample_receipt["source_authority"]),
        "sampler_identity_sha256": sample_receipt["sampler_identity_sha256"],
        "sample_order_sha256": sample_receipt["sample_order_sha256"],
        "row_set_sha256": sample_receipt["row_set_sha256"],
        "unique_row_count": sample_receipt["unique_row_count"],
        "rows_file_sha256": sample_receipt["rows_file_sha256"],
        "sample_dose": sample_receipt["sample_dose"],
        "sampler_seed": sample_receipt["sampler_seed"],
        "prior_rows_file_sha256": sample_receipt["prior_rows_file_sha256"],
        "prior_row_set_sha256": sample_receipt["prior_row_set_sha256"],
        "kl_eligible_rows": sample_receipt["kl_eligible_rows"],
        "kl_eligible_mass_decimal": sample_receipt["kl_eligible_mass_decimal"],
        "kl_ordered_evidence_sha256": sample_receipt["kl_ordered_evidence_sha256"],
        "kl_eligible_evidence_sha256": sample_receipt["kl_eligible_evidence_sha256"],
    }
    return {
        "schema_version": CENTRAL_LEARNER_BINDING_SCHEMA,
        "stage": stage,
        "central_authority_schema": central_authority_schema,
        "central_authority_sha256": central_authority_sha256,
        "executor_authority_path": published["path"],
        "executor_authority_file_sha256": published["file_sha256"],
        "executor_authority_state_sha256": published_authority["state_sha256"],
        "selected_aux_decision": selected_aux_decision,
        "diagnostic_only": stage != "FINAL",
        # FINAL is only eligible to enter the external gate.  The learner may
        # never self-assert the later coordinator promotion transition.
        "promotion_eligible": False,
        "eligible_for_full_gate": stage == "FINAL",
        "full_gate_required": stage == "FINAL",
        "immutable_contract_recipe": copy.deepcopy(immutable_contract_recipe),
        "immutable_contract_recipe_sha256": _value_sha256(immutable_contract_recipe),
        "effective_recipe": copy.deepcopy(effective_recipe),
        "effective_recipe_sha256": _value_sha256(effective_recipe),
        "initializer_sha256": initializer_sha256,
        "sample_binding": sample_binding,
        "code_binding": copy.deepcopy(code_binding),
        "code_tree_sha256": code_binding["code_tree_sha256"],
        "reviewed_lock_file_sha256": reviewed_lock_file_sha256,
    }


def _published_executor_authority(
    *, root: Path, experiment_id: str, filename: str, expected: dict[str, Any]
) -> dict[str, Any]:
    """Replay the immutable coordinator artifact handed to the child learner."""

    root_path = root.expanduser().resolve(strict=True)
    authority_path = (
        root_path / experiment_id.removeprefix("sha256:") / filename
    ).resolve(strict=True)
    try:
        published = aux_coordinator.verify_published_executor_authority(authority_path)
    except aux_coordinator.CoordinatorError as error:
        raise ExecutorError(
            f"published executor authority replay refused: {error}"
        ) from error
    if published.get("authority") != expected:
        raise ExecutorError(
            "published executor authority differs from coordinator loader"
        )
    return published


def _bind_p1_training_descriptor(
    verified: dict[str, Any],
    *,
    authority: Mapping[str, Any],
    published_executor_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the centrally issued trainer-visible KL scope for one P1 arm."""

    arm_id = str(authority.get("arm_id"))
    descriptor_authority = authority.get("training_descriptor_authority")
    try:
        descriptor_authority = aux_coordinator._verify_p1_training_descriptor_authority(  # noqa: SLF001
            descriptor_authority,
            arm_id=arm_id,
            composite=authority["composite"],
            eligibility=authority["kl_eligibility_authority"],
        )
    except (KeyError, TypeError, aux_coordinator.CoordinatorError) as error:
        raise ExecutorError(
            f"P1 training descriptor authority refused: {error}"
        ) from error

    if descriptor_authority["kind"] == "base":
        descriptor_path = Path(verified["data_path"]).expanduser().resolve(strict=True)
    else:
        published_path = (
            Path(str(published_executor_authority.get("path", "")))
            .expanduser()
            .resolve(strict=True)
        )
        descriptor_path = (
            published_path.parent / str(descriptor_authority["filename"])
        ).resolve(strict=True)
        if descriptor_path.parent != published_path.parent:
            raise ExecutorError("P1 derived descriptor escaped the central transaction")
        try:
            aux_coordinator._verify_p1_derived_training_descriptor_file(  # noqa: SLF001
                published_path.parent,
                arm_id=arm_id,
                authority=descriptor_authority,
            )
        except (OSError, aux_coordinator.CoordinatorError) as error:
            raise ExecutorError(
                f"P1 derived training descriptor refused: {error}"
            ) from error
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot read P1 training descriptor: {error}") from error
    if (
        not isinstance(descriptor, dict)
        or _file_sha256(descriptor_path)
        != descriptor_authority["descriptor_file_sha256"]
        or _value_sha256(descriptor) != descriptor_authority["descriptor_fingerprint"]
        or descriptor.get("policy_kl_anchor_component_ids")
        != descriptor_authority["policy_kl_anchor_component_ids"]
    ):
        raise ExecutorError(
            "P1 trainer-visible descriptor differs from central authority"
        )
    if (
        descriptor_authority["base_descriptor_sha256"]
        != verified["corpus_meta_file_sha256"]
    ):
        raise ExecutorError("P1 derived descriptor does not descend from verified data")

    result = dict(verified)
    result["p1_training_descriptor_authority"] = copy.deepcopy(descriptor_authority)
    if descriptor_authority["kind"] != "base":
        result.update(
            {
                "data_path": descriptor_path,
                "corpus_meta_file_sha256": descriptor_authority[
                    "descriptor_file_sha256"
                ],
                "descriptor_fingerprint": descriptor_authority[
                    "descriptor_fingerprint"
                ],
                "data_fingerprint": train_bc._training_data_fingerprint(  # noqa: SLF001
                    str(descriptor_path), "memmap"
                ),
                "validation_path": descriptor_path,
                "validation_file_sha256": descriptor_authority[
                    "descriptor_file_sha256"
                ],
            }
        )
    return result


def bind_p1_arm(
    verified: dict[str, Any],
    *,
    authority: dict[str, Any],
    published_executor_authority: dict[str, Any],
    reviewed_code_tree_sha256: str,
) -> dict[str, Any]:
    """Bind one centrally claimed, diagnostic-only K0/K3/K10 learner arm."""

    if verified.get("learner_ablation") is not None:
        raise ExecutorError("central P1 authority cannot wrap another ablation")
    if verified.get("function_preserving_upgrade") is not None:
        raise ExecutorError(
            "P1 arms must reload the exact current parent independently"
        )
    expected_keys = {
        "schema_version",
        "sweep_id",
        "arm_id",
        "sweep_state_sha256",
        "arm_claim",
        "arm",
        "current_parent_authority",
        "composite",
        "kl_eligibility_authority",
        "training_descriptor_authority",
        "p1_sample_evidence_receipt",
        "recovery_authority",
        "recovery_component_semantics",
        "native_runtime_authority",
        "native_learner_admission_receipt",
        "portable_code_identity_sha256",
        "portable_runtime_identity_sha256",
        "allocation",
        "authority_sha256",
        "state_sha256",
    }
    if set(authority) != expected_keys:
        raise ExecutorError("central P1 executor authority field set drift")
    try:
        aux_coordinator._verify_sealed(authority, "P1 executor authority")  # noqa: SLF001
    except aux_coordinator.CoordinatorError as error:
        raise ExecutorError(f"central P1 sealed authority refused: {error}") from error
    unsigned = dict(authority)
    unsigned.pop("state_sha256", None)
    stated_authority_sha = unsigned.pop("authority_sha256", None)
    if authority.get(
        "schema_version"
    ) != "a1-p1-arm-executor-authority-v1" or stated_authority_sha != _value_sha256(
        unsigned
    ):
        raise ExecutorError("central P1 executor authority digest/schema drift")
    arm_id = authority.get("arm_id")
    arm = authority.get("arm")
    claim = authority.get("arm_claim")
    if (
        arm_id not in P1_CENTRAL_ARMS
        or not isinstance(arm, dict)
        or not isinstance(claim, dict)
        or arm.get("arm_id") != arm_id
        or claim.get("arm_id") != arm_id
        or claim.get("sweep_id") != authority.get("sweep_id")
        or claim.get("arm") != arm
        or claim.get("allocation") != authority.get("allocation")
        or claim.get("prior_authority_sha256") != authority.get("sweep_state_sha256")
    ):
        raise ExecutorError("central P1 arm/claim/allocation chain drift")
    try:
        recovery = authority["recovery_authority"]
        parent = aux_coordinator.verify_current_parent_authority(
            authority["current_parent_authority"],
            recovery_authority=recovery,
        )
        composite = aux_coordinator._verify_composite(authority["composite"])
        aux_coordinator.verify_allocation(authority["allocation"])
    except aux_coordinator.CoordinatorError as error:
        raise ExecutorError(f"central P1 authority replay refused: {error}") from error
    try:
        exact_recovery_semantics = aux_coordinator.recovery_component_semantics(
            recovery, composite["category_semantics"]
        )
    except (KeyError, TypeError, aux_coordinator.CoordinatorError) as error:
        raise ExecutorError(
            f"P1 composite recovery semantics are incomplete: {error}"
        ) from error
    if (
        authority["recovery_component_semantics"] != exact_recovery_semantics
        or verified.get("producer", {}).get("sha256") != parent["checkpoint_sha256"]
        or verified.get("data_kind") != "production_composite_v2"
    ):
        raise ExecutorError(
            "P1 must independently initialize from the exact current promoted parent "
            "and exact production composite"
        )
    data_checks = {
        "descriptor_sha256": verified.get("corpus_meta_file_sha256"),
        "data_fingerprint": verified.get("data_fingerprint"),
        "payload_inventory_sha256": verified.get("payload_inventory_sha256"),
        "production_sampling_receipt_sha256": verified.get(
            "production_sampling_receipt_sha256"
        ),
        "validation_split_receipt_sha256": verified.get(
            "validation_split_receipt_sha256"
        ),
        "training_game_seed_set_sha256": verified.get("training_game_seed_set_sha256"),
        "validation_game_seed_set_sha256": verified.get(
            "validation_game_seed_set_sha256"
        ),
    }
    data_drift = {
        key: {"authority": composite.get(key), "verified": value}
        for key, value in data_checks.items()
        if composite.get(key) != value
    }
    if data_drift or composite.get("complete_game_inputs") is not True:
        raise ExecutorError(
            f"central P1 composite differs from locally replayed bytes/split: {data_drift}"
        )
    code_binding = _current_ablation_code_binding(verified["lock"])
    if (
        reviewed_code_tree_sha256 != code_binding["code_tree_sha256"]
        or authority["portable_code_identity_sha256"] != reviewed_code_tree_sha256
    ):
        raise ExecutorError("central P1 code tree differs from reviewed digest")
    recipe = arm.get("effective_recipe")
    if not isinstance(recipe, dict) or arm.get(
        "effective_recipe_sha256"
    ) != _value_sha256(recipe):
        raise ExecutorError("central P1 effective recipe digest drift")
    # The authenticated corpus was originally issued under the historical v2
    # lock.  The central coordinator deliberately selects the reviewed v3
    # learner recipe.  Comparing the issued arm to ``verified['recipe']`` would
    # either reject the real authority or tempt the executor to reconstruct a
    # synthetic v2 arm (the exact integration bug this binder closes).
    base = aux_coordinator.canonical_p1_final_lock_authority()["base_recipe"]
    base = copy.deepcopy(base)
    base["policy_kl_anchor_weight"] = arm.get("policy_kl_anchor_weight")
    expected_recipe = copy.deepcopy(base)
    expected_recipe.update(
        {
            "world_size": 8,
            "batch_size": 512,
            "global_batch_size": 4096,
            "grad_accum_steps": 1,
        }
    )
    if recipe != expected_recipe:
        raise ExecutorError("central P1 lost exact FP32 8x512/128-step fresh dose")
    immutable_recipe = dict(verified["recipe"])
    drift = {
        key: {"contract": immutable_recipe.get(key), "effective": recipe.get(key)}
        for key in sorted(set(immutable_recipe) | set(recipe))
        if immutable_recipe.get(key) != recipe.get(key)
    }
    learner_ablation = {
        "schema_version": "a1-learner-ablation-v1",
        "ablation_id": f"a1-p1-central-{str(arm_id).lower()}",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "promotion_block_reason": "requires_independent_final_replication",
        "bound_recipe": immutable_recipe,
        "bound_recipe_sha256": _value_sha256(immutable_recipe),
        "effective_recipe": copy.deepcopy(recipe),
        "effective_recipe_sha256": _value_sha256(recipe),
        "recipe_drift": drift,
        "recipe_drift_sha256": _value_sha256(drift),
        "code_binding": code_binding,
        "code_tree_sha256": code_binding["code_tree_sha256"],
        "reviewed_lock_file_sha256": verified["reviewed_lock_file_sha256"],
        "central_p1": {
            "schema_version": "a1-central-p1-arm-binding-v1",
            "sweep_id": authority["sweep_id"],
            "arm_id": arm_id,
            "authority_sha256": stated_authority_sha,
            "policy_kl_anchor_weight_decimal": arm["policy_kl_anchor_weight_decimal"],
            "exact_current_parent_sha256": parent["checkpoint_sha256"],
            "sampler_seed": recipe["sampler_seed"],
            "sampler_identity_sha256": authority["kl_eligibility_authority"][
                "sampler_identity_sha256"
            ],
            "sample_order_sha256": authority["kl_eligibility_authority"][
                "sample_order_sha256"
            ],
            "training_descriptor_authority": copy.deepcopy(
                authority["training_descriptor_authority"]
            ),
        },
    }
    result = _bind_p1_training_descriptor(
        verified,
        authority=authority,
        published_executor_authority=published_executor_authority,
    )
    result["bound_recipe"] = base
    result["recipe"] = copy.deepcopy(recipe)
    result["p1_arm_executor_authority"] = copy.deepcopy(authority)
    result["learner_ablation"] = learner_ablation
    result["central_learner_binding"] = _central_learner_binding(
        stage="P1",
        central_authority_schema=str(authority["schema_version"]),
        central_authority_sha256=str(stated_authority_sha),
        selected_aux_decision=None,
        effective_recipe=recipe,
        immutable_contract_recipe=immutable_recipe,
        sample_receipt=authority["p1_sample_evidence_receipt"],
        initializer_sha256=parent["checkpoint_sha256"],
        code_binding=code_binding,
        reviewed_lock_file_sha256=str(verified["reviewed_lock_file_sha256"]),
        published_executor_authority=published_executor_authority,
    )
    result["central_published_executor_authority"] = copy.deepcopy(
        published_executor_authority
    )
    result["claim_identity_sha256"] = _value_sha256(
        {
            "schema_version": "a1-central-p1-arm-claim-identity-v1",
            "contract_sha256": result["contract_sha256"],
            "authority_sha256": stated_authority_sha,
            "ablation": learner_ablation,
        }
    )
    return result


def bind_aux_pair_arm(
    verified: dict[str, Any],
    *,
    authority: dict[str, Any],
    published_executor_authority: dict[str, Any],
    warmed_initializer: Path,
    reviewed_code_tree_sha256: str,
) -> dict[str, Any]:
    """Bind one centrally issued corrected AUX0/AUXT arm.

    This is intentionally separate from :func:`bind_learner_ablation`.
    Operator-chosen labels or coefficients cannot represent a commissioned
    pointer experiment: the selected P1 recipe, one shared warmed checkpoint,
    gradient selector, host allocation, and arm claim all come from the central
    append-only coordinator.
    """

    if verified.get("learner_ablation") is not None:
        raise ExecutorError("central AUX authority cannot wrap another ablation")
    upgrade = verified.get("function_preserving_upgrade")
    if (
        not isinstance(upgrade, dict)
        or upgrade.get("module") != AUX_REGULARIZATION_MODULE
    ):
        raise ExecutorError("central AUX requires the corrected pointer upgrade")
    try:
        aux_coordinator._verify_sealed(authority, "AUX executor authority")  # noqa: SLF001
    except aux_coordinator.CoordinatorError as error:
        raise ExecutorError(f"central AUX sealed authority refused: {error}") from error
    unsigned_authority = dict(authority)
    unsigned_authority.pop("state_sha256", None)
    stated_authority_sha = unsigned_authority.pop("authority_sha256", None)
    if (
        authority.get("schema_version") != aux_coordinator.EXECUTOR_AUTHORITY_SCHEMA
        or stated_authority_sha != aux_coordinator._digest(unsigned_authority)  # noqa: SLF001
    ):
        raise ExecutorError("central AUX executor authority digest/schema drift")
    try:
        pair = aux_coordinator._verify_sealed(  # noqa: SLF001
            authority.get("aux_pair_contract"), "AUX pair contract"
        )
        arm_claim = aux_coordinator._verify_sealed(  # noqa: SLF001
            authority.get("arm_claim"), "AUX arm claim"
        )
        p1_authority = aux_coordinator.verify_p1_recipe_data_authority(
            pair["portable_science_identity"]["p1_recipe_data_authority"]
        )
        recovery = p1_authority["recovery_authority"]
        parent_authority = aux_coordinator.verify_current_parent_authority(
            pair["portable_science_identity"]["current_parent_authority"],
            recovery_authority=recovery,
        )
        transition_authority = aux_coordinator.verify_public_award_transition_authority(
            pair["portable_science_identity"]["public_award_transition_authority"],
            expected_parent=parent_authority,
        )
        pointer_authority = aux_coordinator.verify_pointer_upgrade_authority(
            pair["portable_science_identity"]["pointer_upgrade_authority"],
            expected_parent_sha256=transition_authority["transitioned_checkpoint"][
                "sha256"
            ],
        )
        selector_rule = aux_coordinator.verify_selector_rule(
            pair["portable_science_identity"]["selector_rule"]
        )
    except (KeyError, TypeError, aux_coordinator.CoordinatorError) as error:
        raise ExecutorError(f"central AUX authority replay refused: {error}") from error
    arm = authority.get("arm")
    if not isinstance(arm, dict):
        raise ExecutorError("central AUX authority has no arm")
    arm_id = arm.get("arm_id")
    if arm_id not in {AUX_CONTROL_ARM, AUX_TREATMENT_ARM}:
        raise ExecutorError("central AUX permits only canonical AUX0/AUXT arms")
    if (
        pair.get("schema_version") != aux_coordinator.PAIR_SCHEMA
        or pair.get("arms", {}).get(arm_id) != arm
        or arm_claim.get("stage") != arm_id
        or arm_claim.get("prior_authority_sha256") != pair.get("state_sha256")
        or authority.get("authority_sha256") != stated_authority_sha
    ):
        raise ExecutorError("central AUX arm/pair/claim chain drift")
    selected_decimal = pair.get("selected_aux_coefficient_decimal")
    selected = pair.get("selected_aux_coefficient")
    expected_weight = 0.0 if arm_id == AUX_CONTROL_ARM else selected
    if (
        authority.get("selected_aux_coefficient_decimal") != selected_decimal
        or authority.get("selected_aux_coefficient") != selected
        or isinstance(expected_weight, bool)
        or not isinstance(expected_weight, (int, float))
        or not math.isfinite(float(expected_weight))
        or arm.get("aux_subgoal_loss_weight") != float(expected_weight)
        or (arm_id == AUX_TREATMENT_ARM and not 0.001 <= float(expected_weight) <= 0.05)
    ):
        raise ExecutorError("central AUX selected coefficient/arm drift")
    joint = pair.get("joint")
    if not isinstance(joint, dict):
        raise ExecutorError("central AUX pair lacks joint recipe")
    exact_joint = {
        "sample_dose": AUX_SELECTED_SAMPLE_DOSE,
        "optimizer_steps": AUX_SELECTED_OPTIMIZER_STEPS,
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "grad_accum_steps": 1,
        "amp": "none",
        "fresh_adam": True,
        "resume_optimizer": False,
        "warmup_optimizer_sidecar_discarded": True,
    }
    joint_drift = {
        key: {"expected": value, "actual": joint.get(key)}
        for key, value in exact_joint.items()
        if joint.get(key) != value
    }
    selected_recipe = joint.get("effective_recipe")
    if (
        joint_drift
        or not isinstance(selected_recipe, dict)
        or joint.get("effective_recipe_sha256") != _value_sha256(selected_recipe)
        or selected_recipe != p1_authority["effective_recipe"]
        or selected_recipe.get("amp") != "none"
        or selected_recipe.get("max_steps") != AUX_SELECTED_OPTIMIZER_STEPS
        or selected_recipe.get("truncated_vp_margin_value_weight") != 0.25
    ):
        raise ExecutorError(
            f"central AUX joint selected recipe/dose drift: {joint_drift}"
        )
    base_recipe = aux_coordinator.canonical_p1_final_lock_authority()["base_recipe"]
    base_recipe = copy.deepcopy(base_recipe)
    base_recipe["policy_kl_anchor_weight"] = selected_recipe.get(
        "policy_kl_anchor_weight"
    )
    expected_selected_recipe = copy.deepcopy(base_recipe)
    expected_selected_recipe.update(
        {
            "world_size": 8,
            "batch_size": 512,
            "global_batch_size": 4096,
            "grad_accum_steps": 1,
        }
    )
    if selected_recipe != expected_selected_recipe:
        raise ExecutorError(
            "central P1 selection differs from the exact final-v3 selected recipe"
        )
    composite = p1_authority["composite"]
    if verified.get("data_kind") != "production_composite_v2":
        raise ExecutorError("corrected AUX requires the typed 64/12/4/20 composite")
    data_checks = {
        "descriptor_sha256": verified.get("corpus_meta_file_sha256"),
        "data_fingerprint": verified.get("data_fingerprint"),
        "payload_inventory_sha256": verified.get("payload_inventory_sha256"),
        "production_sampling_receipt_sha256": verified.get(
            "production_sampling_receipt_sha256"
        ),
        "validation_split_receipt_sha256": verified.get(
            "validation_split_receipt_sha256"
        ),
        "training_game_seed_set_sha256": verified.get("training_game_seed_set_sha256"),
        "validation_game_seed_set_sha256": verified.get(
            "validation_game_seed_set_sha256"
        ),
    }
    data_drift = {
        key: {"authority": composite.get(key), "verified": value}
        for key, value in data_checks.items()
        if composite.get(key) != value
    }
    if data_drift or composite.get("complete_game_inputs") is not True:
        raise ExecutorError(
            f"central AUX composite differs from locally replayed bytes/split: {data_drift}"
        )
    if (
        parent_authority["checkpoint_sha256"] != verified["producer"]["sha256"]
        or pair.get("exact_current_parent_sha256")
        != parent_authority["checkpoint_sha256"]
        or transition_authority["source_checkpoint"]["sha256"]
        != verified["producer"]["sha256"]
        or transition_authority["source_checkpoint"]["path"]
        != verified["producer"]["path"]
        or verified.get("public_award_transition_source_evidence")
        != transition_authority["receipt"]["evidence"]
        or pointer_authority["source_checkpoint_sha256"]
        != transition_authority["transitioned_checkpoint"]["sha256"]
        or upgrade["source"]["sha256"]
        != transition_authority["transitioned_checkpoint"]["sha256"]
        or upgrade["source"]["path"]
        != transition_authority["transitioned_checkpoint"]["path"]
        or pointer_authority["upgraded_initializer_sha256"]
        != upgrade["upgraded_initializer"]["sha256"]
        or pointer_authority["receipt_sha256"] != upgrade["receipt_sha256"]
        or pointer_authority["new_parameter_set_sha256"]
        != _value_sha256(upgrade["new_parameters"])
    ):
        raise ExecutorError(
            "central AUX current-parent/pointer-upgrade authority drift"
        )
    lexical_warmed = warmed_initializer.expanduser()
    if lexical_warmed.is_symlink() or not lexical_warmed.is_file():
        raise ExecutorError("shared warmed AUX initializer must be a regular file")
    warmed = lexical_warmed.resolve(strict=True)
    warmed_sha = _file_sha256(warmed)
    if warmed_sha != joint.get("initializer_sha256"):
        raise ExecutorError("shared warmed AUX initializer bytes drift")
    code_binding = _current_ablation_code_binding(verified["lock"])
    if reviewed_code_tree_sha256 != code_binding["code_tree_sha256"]:
        raise ExecutorError("central AUX code tree differs from reviewed digest")
    if verified.get("reviewed_lock_file_sha256") != verified.get("lock_file_sha256"):
        raise ExecutorError("central AUX does not bind reviewed final lock bytes")
    effective = dict(selected_recipe)
    effective["aux_subgoal_loss_weight"] = float(expected_weight)
    portable_code_identity = _portable_ablation_code_identity(code_binding)
    portable_upgrade_identity = _portable_upgrade_identity(upgrade)
    shared_identity = {
        "schema_version": "a1-aux-pointer-shared-identity-v1",
        "pair_id": pair["pair_id"],
        "portable_science_identity_sha256": pair["portable_science_identity_sha256"],
        "p1_selection_authority_sha256": pair["p1_selection_authority_sha256"],
        "effective_p1_recipe_sha256": joint["effective_recipe_sha256"],
        "composite_authority_sha256": _value_sha256(composite),
        "exact_current_parent_sha256": pair["exact_current_parent_sha256"],
        "upgrade_module": AUX_REGULARIZATION_MODULE,
        "upgrade_receipt_file_sha256": upgrade["receipt"]["sha256"],
        "upgrade_receipt_digest": upgrade["receipt_sha256"],
        "portable_code_identity": portable_code_identity,
        "portable_code_identity_sha256": portable_code_identity["code_sha256"],
        "portable_upgrade_identity": portable_upgrade_identity,
        "portable_upgrade_identity_sha256": portable_upgrade_identity[
            "identity_sha256"
        ],
        "pointer_upgrade_identity_sha256": pair["pointer_upgrade_identity_sha256"],
        "warmup_terminal_sha256": pair["warmup_terminal_sha256"],
        "gradient_geometry_terminal_sha256": pair["gradient_geometry_terminal_sha256"],
        "selector_rule": selector_rule,
        "selector_rule_sha256": pair["selector_rule_sha256"],
        "selected_aux_coefficient_decimal": selected_decimal,
        "initializer_sha256": warmed_sha,
        "pair_contract_state_sha256": pair["state_sha256"],
    }
    matched = {
        "schema_version": "a1-matched-aux-pointer-arm-v1",
        "arm_id": arm_id,
        "aux_subgoal_loss_weight": float(expected_weight),
        "selected_aux_coefficient_decimal": selected_decimal,
        "upgrade_module": AUX_REGULARIZATION_MODULE,
        "upgrade_receipt": upgrade["receipt"]["path"],
        "upgrade_receipt_file_sha256": upgrade["receipt"]["sha256"],
        "upgrade_receipt_digest": upgrade["receipt_sha256"],
        "upgrade_initializer": upgrade["upgraded_initializer"]["path"],
        "upgrade_initializer_sha256": upgrade["upgraded_initializer"]["sha256"],
        "initializer": str(warmed),
        "initializer_sha256": warmed_sha,
        "aux_pair_authority_sha256": stated_authority_sha,
        "shared_identity": shared_identity,
        "shared_identity_sha256": _value_sha256(shared_identity),
    }
    drift = (
        {}
        if float(expected_weight) == 0.0
        else {
            "aux_subgoal_loss_weight": {
                "contract": float(selected_recipe.get("aux_subgoal_loss_weight", 0.0)),
                "effective": float(expected_weight),
            }
        }
    )
    result = dict(verified)
    result["bound_recipe"] = base_recipe
    result["recipe"] = effective
    result["architecture_initializer"] = {
        "path": str(warmed),
        "sha256": warmed_sha,
    }
    result["initializer_transition_chain"] = [
        _public_award_initializer_transition_record(transition_authority),
        _pointer_initializer_transition_record(upgrade),
        _load_warmup_transition_record(
            published_executor_authority=published_executor_authority,
            expected_terminal_sha256=str(pair["warmup_terminal_sha256"]),
        ),
    ]
    result["aux_pair_executor_authority"] = copy.deepcopy(authority)
    result["learner_ablation"] = {
        "schema_version": "a1-learner-ablation-v1",
        "ablation_id": f"a1-aux-pointer-v1-{arm_id.lower()}",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "promotion_block_reason": "matched_aux_commissioning_only",
        "bound_recipe": base_recipe,
        "bound_recipe_sha256": _value_sha256(base_recipe),
        "effective_recipe": effective,
        "effective_recipe_sha256": _value_sha256(effective),
        "recipe_drift": drift,
        "recipe_drift_sha256": _value_sha256(drift),
        "code_binding": code_binding,
        "code_tree_sha256": code_binding["code_tree_sha256"],
        "reviewed_lock_file_sha256": verified["reviewed_lock_file_sha256"],
        "matched_aux_regularization": matched,
    }
    result["central_learner_binding"] = _central_learner_binding(
        stage=arm_id,
        central_authority_schema=str(authority["schema_version"]),
        central_authority_sha256=str(stated_authority_sha),
        selected_aux_decision=arm_id,
        effective_recipe=effective,
        immutable_contract_recipe=dict(verified["recipe"]),
        sample_receipt=p1_authority["p1_sample_evidence_receipt"],
        initializer_sha256=warmed_sha,
        code_binding=code_binding,
        reviewed_lock_file_sha256=str(verified["reviewed_lock_file_sha256"]),
        published_executor_authority=published_executor_authority,
    )
    result["central_published_executor_authority"] = copy.deepcopy(
        published_executor_authority
    )
    return result


def bind_final_replication(
    verified: dict[str, Any],
    *,
    authority: dict[str, Any],
    published_executor_authority: dict[str, Any],
    experiment: dict[str, Any],
    final_warmed_initializer: Path | None,
    reviewed_code_tree_sha256: str,
) -> dict[str, Any]:
    """Bind the sole promotion-eligible post-selection replication.

    P1 and AUX checkpoints are diagnostic selectors. FINAL therefore reloads
    the exact current parent, uses the independently authenticated 424243
    sample order, and may reproduce the pointer commissioning only when AUXT
    won. No diagnostic checkpoint is accepted as an initializer.
    """

    expected_loader_keys = {
        "schema_version",
        "final_replication_authority",
        "final_claim",
        "allocation",
        "authority_sha256",
        "state_sha256",
    }
    try:
        aux_coordinator._verify_sealed(authority, "FINAL executor authority")  # noqa: SLF001
    except aux_coordinator.CoordinatorError as error:
        raise ExecutorError(f"FINAL sealed authority refused: {error}") from error
    unsigned = dict(authority)
    unsigned.pop("state_sha256", None)
    stated_authority_sha = unsigned.pop("authority_sha256", None)
    if (
        set(authority) != expected_loader_keys
        or authority.get("schema_version")
        != aux_coordinator.FINAL_EXECUTOR_AUTHORITY_SCHEMA
        or stated_authority_sha != aux_coordinator._digest(unsigned)  # noqa: SLF001
    ):
        raise ExecutorError("FINAL executor authority digest/schema drift")
    try:
        final = aux_coordinator._verify_sealed(  # noqa: SLF001
            authority["final_replication_authority"], "FINAL authority"
        )
        claim = aux_coordinator._verify_sealed(  # noqa: SLF001
            authority["final_claim"], "FINAL claim"
        )
        aux_coordinator.verify_allocation(authority["allocation"])
        p1 = aux_coordinator.verify_p1_recipe_data_authority(
            final["diagnostic_p1_selection_authority"]
        )
        parent = aux_coordinator.verify_current_parent_authority(
            final["initializer_authority"]["exact_current_parent_authority"],
            recovery_authority=p1["recovery_authority"],
        )
        transition_authority = aux_coordinator.verify_public_award_transition_authority(
            final["initializer_authority"]["public_award_transition_authority"],
            expected_parent=parent,
        )
        loaded_experiment = aux_coordinator._verify_sealed(  # noqa: SLF001
            experiment, "FINAL experiment authority"
        )
    except (KeyError, TypeError, aux_coordinator.CoordinatorError) as error:
        raise ExecutorError(f"FINAL authority replay refused: {error}") from error
    if (
        final.get("schema_version") != aux_coordinator.FINAL_REPLICATION_SCHEMA
        or claim.get("schema_version") != "a1-final-replication-claim-v1"
        or claim.get("final_replication_id") != final.get("final_replication_id")
        or claim.get("prior_authority_sha256") != final.get("state_sha256")
        or claim.get("allocation") != authority.get("allocation")
        or authority.get("allocation") != final.get("allocation")
        or final.get("diagnostic_only") is not False
        or final.get("promotion_eligible_after_full_gate") is not True
        or final.get("full_gate_required") is not True
        or final.get("auto_promotion") is not False
    ):
        raise ExecutorError("FINAL claim/scientific-role chain drift")
    if not isinstance(loaded_experiment, dict):
        raise ExecutorError("FINAL experiment authority is missing")
    portable_science = loaded_experiment.get("portable_science_identity")
    if not isinstance(portable_science, dict):
        raise ExecutorError("FINAL experiment lacks portable science identity")
    expected_code_sha = portable_science.get("portable_code_identity_sha256")
    code_binding = _current_ablation_code_binding(verified["lock"])
    if (
        reviewed_code_tree_sha256 != code_binding["code_tree_sha256"]
        or expected_code_sha != reviewed_code_tree_sha256
        or verified.get("reviewed_lock_file_sha256") != verified.get("lock_file_sha256")
    ):
        raise ExecutorError("FINAL code/lock differs from reviewed experiment bytes")
    if (
        portable_science.get("public_award_transition_authority")
        != transition_authority
    ):
        raise ExecutorError("FINAL public-award transition differs from experiment")
    if (
        verified.get("producer", {}).get("sha256") != parent["checkpoint_sha256"]
        or verified.get("data_kind") != "production_composite_v2"
    ):
        raise ExecutorError(
            "FINAL must independently reload the exact current promoted parent/composite"
        )
    composite = p1["composite"]
    data_checks = {
        "descriptor_sha256": verified.get("corpus_meta_file_sha256"),
        "data_fingerprint": verified.get("data_fingerprint"),
        "payload_inventory_sha256": verified.get("payload_inventory_sha256"),
        "production_sampling_receipt_sha256": verified.get(
            "production_sampling_receipt_sha256"
        ),
        "validation_split_receipt_sha256": verified.get(
            "validation_split_receipt_sha256"
        ),
        "training_game_seed_set_sha256": verified.get("training_game_seed_set_sha256"),
        "validation_game_seed_set_sha256": verified.get(
            "validation_game_seed_set_sha256"
        ),
    }
    data_drift = {
        key: {"authority": composite.get(key), "verified": value}
        for key, value in data_checks.items()
        if composite.get(key) != value
    }
    if data_drift or composite.get("complete_game_inputs") is not True:
        raise ExecutorError(f"FINAL composite bytes/split drift: {data_drift}")
    training = final.get("training")
    sampling = final.get("sampling_receipt")
    routing = final.get("component_routing_receipt")
    if (
        not isinstance(training, dict)
        or not isinstance(sampling, dict)
        or not isinstance(routing, dict)
        or training
        != {
            "sample_dose": 524288,
            "optimizer_steps": 128,
            "world_size": 8,
            "local_batch_size": 512,
            "global_batch_size": 4096,
            "grad_accum_steps": 1,
            "amp": "none",
            "optimizer": "fresh_adam",
            "resume_optimizer": False,
            "sampler_seed": aux_coordinator.FINAL_SAMPLER_SEED,
            "sample_order_sha256": sampling.get("sample_order_sha256"),
        }
        or sampling.get("sampler_seed") != aux_coordinator.FINAL_SAMPLER_SEED
        or sampling.get("overlap_within_independent_bound") is not True
        or sampling.get("prior_rows_file_sha256")
        != p1["p1_sample_evidence_receipt"]["rows_file_sha256"]
        or sampling.get("prior_row_set_sha256")
        != p1["p1_sample_evidence_receipt"]["row_set_sha256"]
        or sampling.get("prior_unique_row_count")
        != p1["p1_sample_evidence_receipt"]["unique_row_count"]
        or sampling.get("replay_verified") is not True
        or routing.get("mixed_authoritative_transition_approved") is not True
        or routing.get("per_row_component_authenticated") is not True
        or routing.get("legacy_slot12_all_zero") is not True
        or routing.get("model_slot12_zero_initialization_required") is not True
    ):
        raise ExecutorError("FINAL dose/sampling/component-routing authority drift")
    selected_aux = final.get("selected_aux_decision")
    if selected_aux not in {AUX_CONTROL_ARM, AUX_TREATMENT_ARM}:
        raise ExecutorError("FINAL selected AUX decision is invalid")
    full_recipe = final.get("effective_recipe")
    if (
        not isinstance(full_recipe, dict)
        or final.get("effective_recipe_sha256") != aux_coordinator._digest(full_recipe)  # noqa: SLF001
    ):
        raise ExecutorError("FINAL effective recipe digest drift")
    selected_p1 = p1["effective_recipe"]
    base = aux_coordinator.canonical_p1_final_lock_authority()["base_recipe"]
    base = copy.deepcopy(base)
    base["policy_kl_anchor_weight"] = selected_p1.get("policy_kl_anchor_weight")
    expected_p1 = copy.deepcopy(base)
    expected_p1.update(
        {
            "world_size": 8,
            "batch_size": 512,
            "global_batch_size": 4096,
            "grad_accum_steps": 1,
        }
    )
    if selected_p1 != expected_p1:
        raise ExecutorError("FINAL selected P1 recipe is not exact final-v3")
    use_treatment = selected_aux == AUX_TREATMENT_ARM
    expected_full = copy.deepcopy(expected_p1)
    expected_full.update(
        {
            "sampler_seed": aux_coordinator.FINAL_SAMPLER_SEED,
            "aux_subgoal_heads": use_treatment,
            "aux_settlement_pointer_head": use_treatment,
            "aux_subgoal_loss_weight": (
                final.get("selected_aux_coefficient") if use_treatment else 0.0
            ),
        }
    )
    if full_recipe != expected_full:
        raise ExecutorError("FINAL recipe differs from selected P1/AUX decisions")
    initializer_authority = final["initializer_authority"]
    if use_treatment:
        upgrade = verified.get("function_preserving_upgrade")
        pointer = initializer_authority.get("pointer_upgrade_authority")
        reference = initializer_authority.get("reference_warmup_terminal")
        if (
            not isinstance(upgrade, dict)
            or upgrade.get("module") != AUX_REGULARIZATION_MODULE
            or not isinstance(pointer, dict)
            or pointer.get("source_checkpoint_sha256")
            != transition_authority["transitioned_checkpoint"]["sha256"]
            or upgrade.get("source", {}).get("sha256")
            != transition_authority["transitioned_checkpoint"]["sha256"]
            or upgrade.get("source", {}).get("path")
            != transition_authority["transitioned_checkpoint"]["path"]
            or verified.get("public_award_transition_source_evidence")
            != transition_authority["receipt"]["evidence"]
            or pointer.get("upgraded_initializer_sha256")
            != upgrade["upgraded_initializer"]["sha256"]
            or pointer.get("receipt_sha256") != upgrade["receipt_sha256"]
            or not isinstance(reference, dict)
            or initializer_authority.get("base_parent_lineage_reloaded") is not True
            or initializer_authority.get("warmup_initializer_role")
            != "shared_immutable_architecture_initializer"
            or initializer_authority.get("exact_reference_warmup_bytes_reused")
            is not True
            or final_warmed_initializer is None
        ):
            raise ExecutorError(
                "FINAL AUXT pointer upgrade/warmup replay is incomplete"
            )
        warmed = final_warmed_initializer.expanduser()
        if warmed.is_symlink() or not warmed.is_file():
            raise ExecutorError("FINAL warmed initializer must be a regular file")
        warmed = warmed.resolve(strict=True)
        expected_warmed_sha = reference.get("result", {}).get(
            "warmed_checkpoint_sha256"
        )
        if _file_sha256(warmed) != expected_warmed_sha:
            raise ExecutorError("FINAL deterministic warmed initializer bytes drift")
        architecture_initializer = {
            "path": str(warmed),
            "sha256": expected_warmed_sha,
        }
    else:
        if (
            verified.get("function_preserving_upgrade") is not None
            or final_warmed_initializer is not None
            or initializer_authority.get("pointer_upgrade_authority") is not None
            or initializer_authority.get("warmup_recipe") is not None
            or initializer_authority.get("reference_warmup_terminal") is not None
            or initializer_authority.get("warmup_initializer_role") is not None
            or initializer_authority.get("exact_reference_warmup_bytes_reused")
            is not False
        ):
            raise ExecutorError(
                "FINAL AUX0 must skip pointer commissioning and warmed bytes"
            )
        architecture_initializer = copy.deepcopy(
            transition_authority["transitioned_checkpoint"]
        )
    learner_recipe = {
        key: value
        for key, value in full_recipe.items()
        if key not in {"aux_subgoal_heads", "aux_settlement_pointer_head"}
    }
    base["sampler_seed"] = aux_coordinator.FINAL_SAMPLER_SEED
    base["aux_subgoal_loss_weight"] = learner_recipe["aux_subgoal_loss_weight"]
    result = dict(verified)
    result["bound_recipe"] = base
    result["recipe"] = learner_recipe
    result["architecture_initializer"] = architecture_initializer
    initializer_transition_chain = [
        _public_award_initializer_transition_record(transition_authority)
    ]
    if use_treatment:
        initializer_transition_chain.extend(
            [
                _pointer_initializer_transition_record(upgrade),
                _load_warmup_transition_record(
                    published_executor_authority=published_executor_authority,
                    expected_terminal_sha256=str(reference["state_sha256"]),
                ),
            ]
        )
    result["initializer_transition_chain"] = initializer_transition_chain
    result["final_replication_executor_authority"] = copy.deepcopy(authority)
    result["final_replication_binding"] = {
        "schema_version": "a1-final-replication-training-binding-v1",
        "final_replication_id": final["final_replication_id"],
        "authority_sha256": stated_authority_sha,
        "authority_effective_recipe_sha256": final["effective_recipe_sha256"],
        "sampler_identity_sha256": sampling["sampler_identity_sha256"],
        "sample_order_sha256": sampling["sample_order_sha256"],
        "row_set_sha256": sampling["row_set_sha256"],
        "component_routing_state_sha256": final["component_routing_state_sha256"],
        "selected_aux_decision": selected_aux,
        "selected_aux_coefficient_decimal": final["selected_aux_coefficient_decimal"],
        "diagnostic_checkpoint_loaded": False,
        "promotion_eligible": False,
        "eligible_for_full_gate": True,
        "full_gate_required": True,
    }
    result["central_learner_binding"] = _central_learner_binding(
        stage="FINAL",
        central_authority_schema=str(authority["schema_version"]),
        central_authority_sha256=str(stated_authority_sha),
        selected_aux_decision=selected_aux,
        effective_recipe=learner_recipe,
        immutable_contract_recipe=dict(verified["recipe"]),
        sample_receipt=sampling,
        initializer_sha256=architecture_initializer["sha256"],
        code_binding=code_binding,
        reviewed_lock_file_sha256=str(verified["reviewed_lock_file_sha256"]),
        published_executor_authority=published_executor_authority,
    )
    result["central_published_executor_authority"] = copy.deepcopy(
        published_executor_authority
    )
    result["claim_identity_sha256"] = _value_sha256(result["final_replication_binding"])
    return result


def bind_stage_c_final_replication(
    verified: dict[str, Any],
    *,
    authority_path: Path,
    arm_name: str,
    reviewed_code_tree_sha256: str,
) -> dict[str, Any]:
    """Bind a fresh current-parent replay of the externally selected Stage-C dose.

    The diagnostic checkpoint selects only the optimizer-step count.  The
    initializer and coherent teacher are both the exact authoritative current
    parent named by the sealed final authority; candidate chaining is refused.
    """

    try:
        authority = stage_c_final.verify_final_authority(authority_path)
    except stage_c_final.FinalReplicationError as error:
        raise ExecutorError(f"Stage-C final authority refused: {error}") from error
    final_corpus = verified.get("stage_c_final_corpus_admission")
    upgrade = verified.get("function_preserving_upgrade")
    if not isinstance(final_corpus, dict) or not isinstance(upgrade, dict):
        raise ExecutorError(
            "Stage-C final requires its production corpus and zero-diff initializer"
        )
    authority_corpus = authority["final_corpus_admission"]
    corpus_admission_path = Path(str(authority_corpus["path"])).resolve(strict=True)
    verified_admission_path = Path(
        str(verified["coherent_direct_corpus_binding"]["corpus_admission"]["path"])
    ).resolve(strict=True)
    if (
        corpus_admission_path != verified_admission_path
        or _file_sha256(corpus_admission_path)
        != authority_corpus["file_sha256"]
        or final_corpus.get("admission_sha256")
        != authority_corpus["admission_sha256"]
    ):
        raise ExecutorError("Stage-C final authority binds different corpus bytes")
    initializer = authority["initializer"]
    if (
        verified.get("producer") != initializer.get("exact_parent")
        or upgrade.get("source") != initializer.get("exact_parent")
        or upgrade.get("upgraded_initializer")
        != initializer.get("upgraded_initializer")
        or upgrade.get("receipt_sha256")
        != initializer.get("upgrade_receipt_sha256")
        or initializer.get("fresh_adam") is not True
        or initializer.get("resume_optimizer") is not False
        or initializer.get("candidate_chaining") is not False
    ):
        raise ExecutorError("Stage-C final current-parent initializer drifted")
    lock_ref = authority["reviewed_code"]["lock"]
    if (
        verified.get("reviewed_lock_file_sha256")
        != verified.get("lock_file_sha256")
        or verified.get("lock_file_sha256") != lock_ref.get("file_sha256")
        or Path(str(verified["lock_path"])).resolve(strict=True)
        != Path(str(lock_ref["path"])).resolve(strict=True)
    ):
        raise ExecutorError("Stage-C final lock differs from reviewed authority")
    code_binding = _current_ablation_code_binding(verified["lock"])
    if (
        reviewed_code_tree_sha256 != code_binding["code_tree_sha256"]
        or authority["reviewed_code"]["code_tree_sha256"]
        != reviewed_code_tree_sha256
    ):
        raise ExecutorError("Stage-C final code tree differs from reviewed authority")
    arms = authority["training"].get("matched_arms")
    if not isinstance(arms, dict) or arm_name not in arms:
        raise ExecutorError(f"unknown Stage-C final matched arm: {arm_name}")
    selected_arm = arms[arm_name]
    selected_recipe = selected_arm.get("recipe")
    if not isinstance(selected_recipe, dict):
        raise ExecutorError("Stage-C final selected recipe is malformed")
    allowed = A1_LEARNER_ABLATION_FIELDS | {"sampler_seed"}
    if set(selected_recipe) - allowed:
        raise ExecutorError(
            "Stage-C final recipe changes a non-learner field: "
            f"{sorted(set(selected_recipe) - allowed)}"
        )
    bound = dict(verified["recipe"])
    effective = dict(bound)
    effective.update(
        {
            "public_card_lr_mult": 1.0,
            "per_game_policy_surprise_weighting": False,
            "per_game_value_weight_mode": "equal",
        }
    )
    effective.update(copy.deepcopy(selected_recipe))
    selected_step = int(authority["diagnostic_selection"]["selected_step"])
    max_steps = int(authority["training"]["max_optimizer_steps"])
    checkpoint_steps = list(authority["training"]["checkpoint_steps"])
    if (
        int(effective.get("max_steps", -1)) != max_steps
        or checkpoint_steps != [8, 12, 16, 32]
        or checkpoint_steps[-1] != max_steps
        or selected_arm.get("recipe_sha256") != _value_sha256(selected_recipe)
        or effective.get("optimizer") != "adam"
        or effective.get("resume_optimizer") is not False
        or effective.get("fused_optimizer") is not False
        or float(effective.get("value_target_lambda", -1.0)) != 1.0
        or final_corpus.get("search_value_evidence", {}).get(
            "naive_root_blend_authorized"
        )
        is not False
        or final_corpus.get("search_value_evidence", {}).get(
            "terminal_target_remains_authoritative"
        )
        is not True
    ):
        raise ExecutorError("Stage-C final fresh-Adam selected-dose recipe drifted")
    binding = {
        "schema_version": "a1-stage-c-final-matched-training-binding-v2",
        "authority": {
            "path": str(authority_path.resolve(strict=True)),
            "file_sha256": _file_sha256(authority_path.resolve(strict=True)),
            "authority_sha256": authority["authority_sha256"],
        },
        "current_parent_checkpoint_sha256": initializer["exact_parent"]["sha256"],
        "initializer_checkpoint_sha256": initializer["upgraded_initializer"][
            "sha256"
        ],
        "selected_diagnostic_step": selected_step,
        "selected_diagnostic_checkpoint_sha256": authority[
            "diagnostic_selection"
        ]["selected_diagnostic_checkpoint_sha256"],
        "selected_diagnostic_checkpoint_loaded": False,
        "matched_arm": arm_name,
        "matched_arm_role": selected_arm["role"],
        "matched_sample_order": True,
        "terminal_value_target_only": True,
        "same_trajectory_checkpoint_steps": checkpoint_steps,
        "external_adjudication_sha256": authority["external_adjudication"][
            "adjudication_sha256"
        ],
        "independent_root_manifest_sha256": final_corpus["root_manifest"][
            "root_manifest_sha256"
        ],
        "fresh_independent_target_bytes": True,
        "diagnostic_only": False,
        "promotion_eligible": False,
        "eligible_for_full_gate": True,
        "full_gate_required": True,
        "auto_promotion": False,
    }
    result = dict(verified)
    result["bound_recipe"] = bound
    result["recipe"] = effective
    result["stage_c_final_replication_authority"] = copy.deepcopy(authority)
    result["stage_c_final_replication_binding"] = binding
    result["stage_c_final_code_binding"] = code_binding
    result["claim_identity_sha256"] = _value_sha256(binding)
    return result


def _build_direct_train_command(
    verified: dict[str, Any],
    *,
    python: Path,
    checkpoint: Path,
    report: Path,
) -> list[str]:
    """Render the direct ``python train_bc.py`` argv without a launcher.

    Topology owners must wrap this exactly once with
    :func:`_topologize_train_command`. Keeping the semantic argv separate from
    the launcher prevents downstream sealed executors from accidentally
    nesting torchrun around an already-distributed command.
    """

    recipe = verified["recipe"]
    initializer = verified.get("architecture_initializer", verified["producer"])
    trainer_path = _REPO_ROOT / "tools" / "train_bc.py"
    if verified.get("data_kind") == "production_composite_v2":
        trainer_authority = _require_current_production_trainer_authority(verified)
        assert trainer_authority is not None
        trainer_path = Path(trainer_authority["path"])
    elif isinstance(verified.get("learner_ablation"), dict):
        code_binding = verified["learner_ablation"].get("code_binding")
        records = (
            code_binding.get("records") if isinstance(code_binding, dict) else None
        )
        candidates = [
            Path(str(record["path"]))
            for record in records or ()
            if isinstance(record, dict)
            and record.get("relative_path") == "tools/train_bc.py"
        ]
        if len(candidates) != 1:
            raise ExecutorError(
                "learner ablation must bind exactly one train_bc entrypoint"
            )
        trainer_path = candidates[0].expanduser().resolve(strict=True)
    elif (
        verified.get("learner_ablation") is None
        and verified.get("central_learner_binding") is None
    ):
        candidates = [
            Path(str(record.get("path")))
            for record in (
                verified.get("lock", {}).get("provenance", {}).get("learner_code", [])
            )
            if isinstance(record, dict)
            and str(record.get("path", "")).endswith("/tools/train_bc.py")
        ]
        if candidates:
            if len(candidates) != 1:
                raise ExecutorError(
                    "sealed A1 contract binds multiple train_bc entrypoints"
                )
            trainer_path = candidates[0].expanduser().resolve(strict=True)
    command = [str(python), str(trainer_path)]
    command.extend(["--arch", "entity_graph"])
    for flag, value in SEALED_A1_MODEL_CLI.items():
        command.extend((flag, value))
    if (
        verified.get("function_preserving_upgrade", {}).get("module")
        == AUX_REGULARIZATION_MODULE
    ):
        # Both AUX0 and AUXT request the same already-present architecture.
        # The checkpoint bytes, not this CLI switch, remain the construction
        # authority; train_bc reports requested and effective architecture.
        command.append("--aux-subgoal-heads")
        command.append("--aux-settlement-pointer-head")
    upgrade_module = verified.get("function_preserving_upgrade", {}).get("module")
    if upgrade_module in {
        architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES,
        architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES_V2,
        architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY,
        architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2,
    }:
        # Make the reviewed architecture delta visible in argv as well as the
        # initializer receipt. train_bc independently verifies that this flag
        # agrees with the checkpoint-owned architecture.
        command.append("--public-card-count-features")
    if upgrade_module in {
        architecture_upgrade.MODULE_MEANINGFUL_PUBLIC_HISTORY,
        architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY,
        architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2,
        architecture_upgrade.MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY,
        architecture_upgrade.MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY_FROM_V1,
    }:
        command.extend(["--meaningful-public-history", "--event-history-limit", "32"])
    if upgrade_module in {
        architecture_upgrade.MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY,
        architecture_upgrade.MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY_FROM_V1,
    }:
        command.extend(
            [
                "--meaningful-public-history-pooling",
                "ordered_attention_v2",
            ]
        )
    command.extend(
        [
            "--data",
            str(verified["data_path"]),
            "--data-format",
            "memmap",
            "--data-loader-workers",
            str(DATA_LOADER_WORKERS),
            "--data-loader-prefetch",
            str(DATA_LOADER_PREFETCH),
            "--device",
            "cuda",
            "--track",
            str(recipe["track"]),
            "--vps-to-win",
            str(recipe["vps_to_win"]),
            "--graph-history-features",
            "--seed",
            str(recipe["seed"]),
            "--epochs",
            str(recipe["epochs"]),
            "--max-steps",
            str(recipe["max_steps"]),
            "--batch-size",
            str(recipe["batch_size"]),
            "--grad-accum-steps",
            str(recipe["grad_accum_steps"]),
            "--optimizer",
            str(recipe["optimizer"]),
            "--no-resume-optimizer",
            "--lr",
            str(recipe["lr"]),
            "--lr-warmup-steps",
            str(recipe["lr_warmup_steps"]),
            "--lr-schedule",
            str(recipe["lr_schedule"]),
            "--weight-decay",
            str(recipe["weight_decay"]),
            "--no-fused-optimizer",
            "--value-lr-mult",
            str(recipe["value_lr_mult"]),
            "--action-module-lr-mult",
            str(recipe["action_module_lr_mult"]),
            "--public-card-lr-mult",
            str(recipe.get("public_card_lr_mult", 1.0)),
            "--trunk-lr-mult",
            str(recipe.get("trunk_lr_mult", 1.0)),
            "--policy-loss-weight",
            str(recipe["policy_loss_weight"]),
            "--soft-target-source",
            str(recipe["soft_target_source"]),
            "--soft-target-weight",
            str(recipe["soft_target_weight"]),
            "--soft-target-temperature",
            str(recipe["soft_target_temperature"]),
            "--soft-target-min-legal-coverage",
            str(recipe["soft_target_min_legal_coverage"]),
            "--value-loss-weight",
            str(recipe["value_loss_weight"]),
            "--value-target-lambda",
            str(recipe["value_target_lambda"]),
            "--value-head-type",
            "mse",
            "--value-categorical-bins",
            "0",
            "--value-categorical-loss-weight",
            str(recipe["value_categorical_loss_weight"]),
            "--hlgauss-scalar-aux-loss-weight",
            str(recipe["hlgauss_scalar_aux_loss_weight"]),
            "--final-vp-loss-weight",
            str(recipe["final_vp_loss_weight"]),
            "--q-loss-weight",
            str(recipe["q_loss_weight"]),
            "--policy-kl-anchor-weight",
            str(recipe["policy_kl_anchor_weight"]),
            "--policy-kl-anchor-direction",
            str(recipe.get("policy_kl_anchor_direction", "forward")),
            "--value-uncertainty-loss-weight",
            str(recipe["value_uncertainty_loss_weight"]),
            "--aux-subgoal-loss-weight",
            str(recipe["aux_subgoal_loss_weight"]),
            "--freeze-modules",
            str(recipe["freeze_modules"]),
            "--policy-surprise-weight",
            str(recipe["policy_surprise_weight"]),
            "--advantage-policy-weighting",
            str(recipe["advantage_policy_weighting"]),
            "--vp-margin-weight",
            str(recipe["vp_margin_weight"]),
            "--truncated-vp-margin-value-weight",
            str(recipe["truncated_vp_margin_value_weight"]),
            "--amp",
            str(recipe["amp"]),
            "--mask-hidden-info",
            "--no-symmetry-augment",
            "--forced-action-weight",
            str(recipe["forced_action_weight"]),
            "--forced-row-value-weight",
            str(recipe["forced_row_value_weight"]),
            "--winner-sample-weight",
            str(recipe["winner_sample_weight"]),
            "--loser-sample-weight",
            str(recipe["loser_sample_weight"]),
            "--teacher-weights",
            str(recipe["teacher_weights"]),
            "--phase-weights",
            str(recipe["phase_weights"]),
            "--value-phase-weights",
            str(recipe["value_phase_weights"]),
            "--validation-fraction",
            "0.05",
            "--validation-seed",
            "17",
            "--validation-max-samples",
            "0",
            "--init-checkpoint",
            str(initializer["path"]),
            "--checkpoint",
            str(checkpoint),
            "--report",
            str(report),
            "--require-35m-model",
            "--skip-teacher-quality-gate",
            "--trust-curated-data-quality",
        ]
    )
    if recipe.get("policy_kl_target") is not None:
        command.extend(
            [
                "--policy-kl-target",
                str(recipe["policy_kl_target"]),
                "--policy-kl-dual-lr",
                str(recipe["policy_kl_dual_lr"]),
                "--policy-kl-max-weight",
                str(recipe["policy_kl_max_weight"]),
            ]
        )
    if bool(recipe.get("per_game_policy_surprise_weighting", False)):
        command.append("--per-game-policy-surprise-weighting")
    if recipe.get("forced_row_value_action_type_weights"):
        command.extend(
            [
                "--forced-row-value-action-type-weights",
                str(recipe["forced_row_value_action_type_weights"]),
            ]
        )
    if "sampler_seed" in recipe:
        command.extend(["--sampler-seed", str(recipe["sampler_seed"])])
    max_steps = int(recipe.get("max_steps", 0))
    reporting = (verified.get("learner_ablation") or {}).get("reporting_contract", {})
    explicit_checkpoints = list(reporting.get("checkpoint_steps", ()))
    if not explicit_checkpoints:
        stage_c_final_binding = verified.get("stage_c_final_replication_binding")
        if isinstance(stage_c_final_binding, dict):
            explicit_checkpoints = [
                int(step)
                for step in stage_c_final_binding.get(
                    "same_trajectory_checkpoint_steps", ()
                )
                if int(step) < max_steps
            ]
    if explicit_checkpoints:
        # An explicit generic diagnostic schedule is topology/data-kind
        # independent.  The terminal max-step checkpoint remains the ordinary
        # output; only strictly earlier steps belong on --checkpoint-steps.
        command.extend(["--checkpoint-steps", ",".join(map(str, explicit_checkpoints))])
    elif (
        verified.get("data_kind") == "production_composite_v2"
        and verified.get("central_learner_binding") is None
    ):
        if verified.get("learner_ablation") is None and max_steps == 128:
            # Preserve the useful part of the production dose curve without
            # launching chained candidates.
            command.extend(["--checkpoint-steps", "64,96"])
        elif (
            verified.get("learner_ablation", {})
            .get("reporting_contract", {})
            .get("diagnostic_dose_curve")
            and max_steps >= 128
        ):
            # Diagnostic trajectories expose their within-run dose curve.  A
            # 128-step arm emits step 64 plus its terminal; the selected
            # 256-step replay emits 64/128 plus its terminal.  Every trajectory
            # still initializes independently from the sealed parent with fresh
            # Adam -- none consumes another candidate checkpoint.
            checkpoints = [64]
            step = 128
            while step < max_steps:
                checkpoints.append(step)
                step *= 2
            command.extend(["--checkpoint-steps", ",".join(map(str, checkpoints))])
        elif verified.get("learner_ablation") is not None and max_steps > 128:
            # Historical diagnostic behavior, retained for non-campaign
            # ablations that did not request the explicit 64-step frontier.
            checkpoints = []
            step = 128
            while step < max_steps:
                checkpoints.append(step)
                step *= 2
            command.extend(["--checkpoint-steps", ",".join(map(str, checkpoints))])
    if verified.get("data_kind") == "production_composite_v2":
        # The production 64/12/4/20 descriptor has three corrected components
        # plus homogeneous legacy replay. train_bc authenticates and routes
        # slot12 per component, zero-initializes the inherited input column, and
        # stamps the resulting checkpoint authoritative_v1. Omitting these
        # flags either refuses the mixed corpus or erases the corrected signal.
        command.extend(
            [
                "--public-award-feature-contract",
                "authoritative_v1",
                "--allow-mixed-public-award-feature-contracts",
            ]
        )
        event_history_contract = verified.get("event_history_training_contract")
        acknowledgements = (
            event_history_contract.get("empty_payload_inventory_acknowledgements")
            if isinstance(event_history_contract, dict)
            else None
        )
        trainable_history = event_history_contract.get(
            "training_event_history_trainable"
        )
        usable_history = event_history_contract.get("event_history_end_to_end_usable")
        if (
            not isinstance(acknowledgements, list)
            or any(not isinstance(value, str) for value in acknowledgements)
            or (trainable_history is True and usable_history is not True)
            or (trainable_history is False and usable_history is not False)
            or trainable_history not in {True, False}
        ):
            raise ExecutorError(
                "production composite lacks a coherent authenticated "
                "event-history contract"
            )
        for acknowledgement in acknowledgements:
            command.extend([EVENT_HISTORY_ACK_FLAG, acknowledgement])
        if trainable_history is False:
            if not acknowledgements:
                raise ExecutorError(
                    "empty production event history lacks payload acknowledgements"
                )
            command.append(EVENT_HISTORY_CROP_FLAG)
    if verified.get("data_kind") != "production_composite_v2":
        command.extend(
            [
                "--validation-game-seed-manifest",
                str(verified["validation_path"]),
            ]
        )
    coherent_binding = verified.get("coherent_direct_corpus_binding")
    if coherent_binding is not None:
        if verified.get("data_kind") != "coherent_direct_memmap_v1":
            raise ExecutorError("coherent corpus binding attached to wrong data kind")
        command.extend(
            [
                "--a1-coherent-corpus-binding-json",
                _canonical_bytes(coherent_binding).decode("ascii"),
            ]
        )
    if bool(recipe.get("training_rng_rank_offset", False)):
        command.append("--training-rng-rank-offset")
    if "value_trunk_grad_scale" in recipe:
        command.extend(
            ("--value-trunk-grad-scale", str(recipe["value_trunk_grad_scale"]))
        )
    if bool(recipe["per_game_value_weight"]):
        command.append("--per-game-value-weight")
        # Historical non-ablation recipes locked this switch off and therefore
        # had no typed mode.  If a future sealed production recipe enables it,
        # bind the requested operator instead of silently inheriting
        # train_bc's `equal` default.  Latest-main ablations already append the
        # mode inside their existing command-hash block below; keep that
        # ordering unchanged for reproducibility.
        if verified.get("learner_ablation") is None:
            command.extend(
                [
                    "--per-game-value-weight-mode",
                    str(recipe.get("per_game_value_weight_mode", "equal")),
                ]
            )
    if bool(recipe.get("per_game_policy_weight", False)):
        command.extend(
            [
                "--per-game-policy-weight",
                "--per-game-policy-weight-mode",
                str(recipe.get("per_game_policy_weight_mode", "equal")),
            ]
        )
    if "policy_aux_active_batch_size" in recipe:
        command.extend(
            [
                "--policy-aux-active-batch-size",
                str(recipe["policy_aux_active_batch_size"]),
            ]
        )
    central_binding = verified.get("central_learner_binding")
    learner_ablation = verified.get("learner_ablation")
    stage_c_final_binding = verified.get("stage_c_final_replication_binding")
    if isinstance(central_binding, dict):
        command.append("--allow-concurrent-bc")
        if verified.get("data_kind") not in {
            "production_composite_v2",
            "coherent_direct_memmap_v1",
        }:
            command.extend(
                [
                    EVENT_HISTORY_ACK_FLAG,
                    str(verified["payload_inventory_sha256"]),
                    EVENT_HISTORY_CROP_FLAG,
                ]
            )
        command.extend(
            [
                "--a1-central-learner-binding-json",
                _canonical_bytes(central_binding).decode("ascii"),
                "--a1-central-executor-authority",
                str(central_binding["executor_authority_path"]),
                "--a1-central-executor-authority-sha256",
                str(central_binding["executor_authority_file_sha256"]),
            ]
        )
        matched_aux = (learner_ablation or {}).get("matched_aux_regularization")
        if isinstance(matched_aux, dict):
            command.extend(
                [
                    "--a1-aux-regularization-binding-json",
                    _canonical_bytes(matched_aux).decode("ascii"),
                ]
            )
    elif learner_ablation is not None:
        command.append("--allow-concurrent-bc")
        if verified.get("data_kind") not in {
            "production_composite_v2",
            "coherent_direct_memmap_v1",
        }:
            command.extend(
                [
                    EVENT_HISTORY_ACK_FLAG,
                    str(verified["payload_inventory_sha256"]),
                    EVENT_HISTORY_CROP_FLAG,
                ]
            )
        command.extend(
            [
                # Module gradient/update attribution is reporting-only and
                # leaves the optimizer trajectory unchanged. Sample every 16th
                # batch so a short 128-step dose gets eight observations without
                # cloning the 35M parameter set on every update.
                *(
                    ["--train-diagnostics-every-batches", "16"]
                    if learner_ablation.get("reporting_contract", {}).get(
                        "diagnostic_dose_curve"
                    )
                    else []
                ),
                *(
                    [
                        "--objective-gradient-interference-every-batches",
                        str(
                            learner_ablation.get("reporting_contract", {}).get(
                                "objective_gradient_interference_every_batches",
                                64,
                            )
                        ),
                    ]
                    if int(
                        learner_ablation.get("reporting_contract", {}).get(
                            "objective_gradient_interference_every_batches",
                            (
                                64
                                if learner_ablation.get("reporting_contract", {}).get(
                                    "diagnostic_dose_curve"
                                )
                                else 0
                            ),
                        )
                    )
                    > 0
                    else []
                ),
                # Each executor child sees exactly one physical GPU through
                # CUDA_VISIBLE_DEVICES and owns a distinct durable ablation
                # claim/output set.  The generic host-wide BC lock would
                # otherwise serialize or reject independent diagnostic arms.
                # Never add this to the historical/default one-dose command.
                "--per-game-value-weight-mode",
                str(recipe.get("per_game_value_weight_mode", "equal")),
                "--a1-learner-ablation-id",
                str(learner_ablation["ablation_id"]),
                "--a1-effective-learner-recipe-json",
                _canonical_bytes(recipe).decode("ascii"),
                "--a1-effective-learner-recipe-sha256",
                _value_sha256(recipe),
                "--a1-ablation-code-binding-json",
                _canonical_bytes(learner_ablation["code_binding"]).decode("ascii"),
                "--a1-ablation-code-tree-sha256",
                str(learner_ablation["code_tree_sha256"]),
                "--a1-reviewed-lock-file-sha256",
                str(learner_ablation["reviewed_lock_file_sha256"]),
            ]
        )
        matched_aux = learner_ablation.get("matched_aux_regularization")
        if matched_aux is not None:
            command.extend(
                [
                    "--a1-aux-regularization-binding-json",
                    _canonical_bytes(matched_aux).decode("ascii"),
                ]
            )
    elif isinstance(stage_c_final_binding, dict):
        # The authority and coherent-corpus binding are executor-owned report
        # provenance.  The child receives only the exact optimizer flags and
        # permission for its eight local DDP ranks to share one host.
        command.append("--allow-concurrent-bc")
    if "--ddp-shard-data" in command:
        raise ExecutorError("sealed memmap learner may not shard corpus data by rank")
    return command


def _topologize_train_command(
    direct_command: Sequence[str], *, world_size: int
) -> list[str]:
    """Wrap one direct trainer argv in at most one local torchrun launcher."""

    command = list(direct_command)
    if (
        len(command) < 2
        or Path(command[1]).name != "train_bc.py"
        or "torch.distributed.run" in command
        or any(
            token.startswith("--nproc_per_node") or token.startswith("--nproc-per-node")
            for token in command
        )
    ):
        raise ExecutorError(
            "topology wrapper requires one unwrapped direct train_bc command"
        )
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
    ):
        raise ExecutorError("training launcher world_size must be a positive integer")
    if world_size == 1:
        return command
    return [
        command[0],
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        command[1],
        *command[2:],
    ]


def build_train_command(
    verified: dict[str, Any],
    *,
    python: Path,
    checkpoint: Path,
    report: Path,
) -> list[str]:
    """Render every effective learner field and its authorized topology."""

    matched = verified.get("learner_ablation", {}).get("matched_aux_regularization")
    if isinstance(matched, dict):
        topology = verified.get("training_topology")
        canary = verified.get("ddp_canary")
        preclaim = verified.get("aux_subgoal_preclaim_contract")
        runtime_identity = (
            matched.get("shared_identity", {})
            .get("ddp_canary_semantics", {})
            .get("runtime_identity")
        )
        python_identity = (
            runtime_identity.get("python")
            if isinstance(runtime_identity, dict)
            else None
        )
        try:
            learner_python_sha256 = _file_sha256(
                Path(python).expanduser().resolve(strict=True)
            )
        except OSError as error:
            raise ExecutorError(
                f"cannot authenticate matched AUX Python: {error}"
            ) from error
        if (
            not isinstance(preclaim, dict)
            or not isinstance(canary, dict)
            or not isinstance(topology, dict)
            or topology.get("name") != B200_8GPU_DDP_TOPOLOGY
            or verified.get("claim_identity_sha256")
            != _expected_final_aux_claim_identity(verified)
            or matched.get("shared_identity", {}).get(
                "aux_subgoal_treatment_admission_sha256"
            )
            != preclaim.get("treatment_grade_admission_sha256")
            or matched.get("shared_identity", {}).get("ddp_canary_semantics_sha256")
            != canary.get("semantic_identity_sha256")
            or not isinstance(python_identity, dict)
            or python_identity.get("executable_sha256") != learner_python_sha256
        ):
            raise ExecutorError(
                "matched AUX command requires final treatment admission, "
                "8xB200 canary/runtime, and final claim identity before rendering"
            )
    world_size = int(verified["recipe"]["world_size"])
    if world_size not in {1, 8}:
        raise ExecutorError(f"unsupported sealed learner world_size={world_size}")
    direct = _build_direct_train_command(
        verified,
        python=python,
        checkpoint=checkpoint,
        report=report,
    )
    command = _topologize_train_command(direct, world_size=world_size)
    _require_current_production_trainer_authority(verified, command=command)
    return command


def _verify_independent_parent_authority(
    path: Path,
    *,
    verified: Mapping[str, Any],
    upgrade: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a non-producer initializer for one diagnostic campaign.

    A corpus producer and a learner initializer are different scientific
    identities.  Production training keeps them identical.  This narrow
    authority lets a diagnostic campaign compare an independently selected
    parent without rewriting corpus provenance or pretending that checkpoint
    generated the data.
    """

    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ExecutorError(
            f"independent learner-parent authority must be a regular file: {resolved}"
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(
            f"cannot load independent learner-parent authority: {error}"
        ) from error
    expected_keys = {
        "schema_version",
        "diagnostic_only",
        "promotion_eligible",
        "corpus_binding",
        "learner_parent",
        "function_preserving_upgrade",
        "authority_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ExecutorError("independent learner-parent authority shape drift")
    unsigned = dict(payload)
    stated = unsigned.pop("authority_sha256", None)
    if (
        payload.get("schema_version") != INDEPENDENT_PARENT_AUTHORITY_SCHEMA
        or payload.get("diagnostic_only") is not True
        or payload.get("promotion_eligible") is not False
        or stated != _value_sha256(unsigned)
    ):
        raise ExecutorError("independent learner-parent authority digest/role drift")

    admission = payload.get("corpus_binding", {}).get("coherent_corpus_admission")
    if not isinstance(admission, dict) or set(admission) != {
        "path",
        "file_sha256",
        "admission_sha256",
    }:
        raise ExecutorError("independent parent authority lost corpus admission")
    admission_path = Path(str(admission["path"])).expanduser().resolve(strict=True)
    if (
        admission_path.is_symlink()
        or not admission_path.is_file()
        or _file_sha256(admission_path) != admission["file_sha256"]
    ):
        raise ExecutorError("independent parent corpus admission bytes drifted")

    data_path = Path(str(verified["data_path"])).expanduser().resolve(strict=True)
    expected_corpus = {
        "data_path": str(data_path),
        "corpus_meta_file_sha256": verified.get("corpus_meta_file_sha256"),
        "payload_inventory_sha256": verified.get("payload_inventory_sha256"),
        "data_fingerprint": verified.get("data_fingerprint"),
        "producer_checkpoint": copy.deepcopy(verified.get("producer")),
        "coherent_corpus_admission": admission,
    }
    expected_upgrade = {
        "module": upgrade.get("module"),
        "receipt_file_sha256": upgrade.get("receipt", {}).get("sha256"),
        "receipt_sha256": upgrade.get("receipt_sha256"),
        "upgraded_initializer": copy.deepcopy(upgrade.get("upgraded_initializer")),
    }
    if (
        payload.get("corpus_binding") != expected_corpus
        or payload.get("learner_parent") != upgrade.get("source")
        or payload.get("function_preserving_upgrade") != expected_upgrade
    ):
        raise ExecutorError(
            "independent learner-parent authority differs from verified corpus/upgrade"
        )
    return copy.deepcopy(payload)


def bind_function_preserving_upgrade(
    verified: dict[str, Any],
    receipt_path: Path,
    *,
    allow_public_award_transition_source: bool = False,
    allow_diagnostic_recent_history_source: bool = False,
    independent_parent_authority_path: Path | None = None,
    allow_stage_c_final_current_parent: bool = False,
) -> dict[str, Any]:
    """Select a non-byte-identical initializer only after replaying its receipt."""

    if verified.get("learner_ablation") is not None:
        raise ExecutorError(
            "promotion-eligible architecture upgrades cannot be learner ablations"
        )
    try:
        upgrade = architecture_upgrade.verify_receipt(receipt_path)
    except architecture_upgrade.UpgradeError as error:
        raise ExecutorError(f"architecture upgrade receipt refused: {error}") from error
    source_transition_evidence = None
    diagnostic_comparison_source = None
    independent_parent_authority = None
    source_matches_producer = upgrade["source"]["sha256"] == verified["producer"][
        "sha256"
    ] and Path(upgrade["source"]["path"]) == Path(verified["producer"]["path"])
    if source_matches_producer and independent_parent_authority_path is not None:
        raise ExecutorError(
            "independent learner-parent authority is invalid when initializer "
            "already equals the sealed corpus producer"
        )
    if not source_matches_producer:
        recent_history = verified.get("category_semantics", {}).get("recent_history")
        recent_checkpoint = (
            recent_history.get("checkpoint")
            if isinstance(recent_history, dict)
            else None
        )
        diagnostic_recent_history_matches = bool(
            allow_diagnostic_recent_history_source
            and verified.get("data_kind") == "production_composite_v2"
            and isinstance(recent_checkpoint, dict)
            and upgrade["source"]["sha256"] == recent_checkpoint.get("sha256")
            and Path(upgrade["source"]["path"])
            == Path(str(recent_checkpoint.get("path", "")))
            and upgrade.get("module")
            in {
                architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES,
                architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_FEATURES_V2,
            }
        )
        if independent_parent_authority_path is not None:
            independent_parent_authority = _verify_independent_parent_authority(
                independent_parent_authority_path,
                verified=verified,
                upgrade=upgrade,
            )
            diagnostic_comparison_source = {
                "schema_version": "a1-diagnostic-independent-parent-v1",
                "role": "independent_parent",
                "source": copy.deepcopy(upgrade["source"]),
                "sealed_producer": copy.deepcopy(verified["producer"]),
                "authority": {
                    "path": str(independent_parent_authority_path.resolve(strict=True)),
                    "sha256": _file_sha256(
                        independent_parent_authority_path.resolve(strict=True)
                    ),
                    "authority_sha256": independent_parent_authority[
                        "authority_sha256"
                    ],
                },
                "diagnostic_only": True,
                "promotion_eligible": False,
            }
        elif diagnostic_recent_history_matches:
            diagnostic_comparison_source = {
                "schema_version": "a1-diagnostic-recent-history-parent-v1",
                "role": "recent_history",
                "source": copy.deepcopy(upgrade["source"]),
                "sealed_producer": copy.deepcopy(verified["producer"]),
                "category_semantics_sha256": verified.get("category_semantics_sha256"),
                "diagnostic_only": True,
                "promotion_eligible": False,
            }
        elif not allow_public_award_transition_source:
            raise ExecutorError(
                "architecture upgrade source differs from sealed producer checkpoint"
            )
        elif diagnostic_comparison_source is None:
            try:
                source_transition_evidence = aux_coordinator.scientific_evidence._public_award_transition_evidence(  # noqa: SLF001
                    Path(verified["producer"]["path"]),
                    Path(upgrade["source"]["path"]),
                )
            except (
                OSError,
                ValueError,
                aux_coordinator.scientific_evidence.EvidenceError,
            ) as error:
                raise ExecutorError(
                    f"architecture upgrade transition source refused: {error}"
                ) from error
        if (
            diagnostic_comparison_source is None
            and source_transition_evidence is not None
        ):
            if (
                source_transition_evidence["source_checkpoint_sha256"]
                != verified["producer"]["sha256"]
                or source_transition_evidence["transitioned_checkpoint_sha256"]
                != upgrade["source"]["sha256"]
                or source_transition_evidence["optimizer_steps"] != 0
                or source_transition_evidence["legacy_zero_input_function_preserving"]
                is not True
            ):
                raise ExecutorError(
                    "architecture upgrade transition source lost exact parent lineage"
                )
    receipt = upgrade["receipt"]
    lineage_binding = {
        "schema_version": "a1-lineage-function-preserving-upgrade-v1",
        "module": upgrade["module"],
        "receipt": receipt["path"],
        "receipt_sha256": receipt["sha256"],
        "source_checkpoint_sha256": upgrade["source"]["sha256"],
        "upgraded_initializer_sha256": upgrade["upgraded_initializer"]["sha256"],
    }
    result = dict(verified)
    result["architecture_initializer"] = dict(upgrade["upgraded_initializer"])
    result["function_preserving_upgrade"] = upgrade
    result["function_preserving_upgrade_lineage"] = lineage_binding
    if source_transition_evidence is not None:
        result["public_award_transition_source_evidence"] = source_transition_evidence
    if diagnostic_comparison_source is not None:
        result["diagnostic_comparison_source"] = diagnostic_comparison_source
        # A production composite's data producer and the checkpoint used as an
        # independent diagnostic learner parent are different identities.  Do
        # not overload ``producer``: bind the permitted recent-history parent
        # explicitly so dose accounting can authenticate f7 while corpus
        # provenance continues to name the v5 producer.
        if independent_parent_authority is not None:
            result["independent_parent_authority"] = copy.deepcopy(
                independent_parent_authority
            )
            result["learner_lineage_parent"] = {
                "schema_version": LEARNER_LINEAGE_PARENT_SCHEMA,
                "role": "diagnostic_independent_parent",
                "checkpoint": copy.deepcopy(upgrade["source"]),
                "corpus_producer": copy.deepcopy(verified["producer"]),
                "independent_parent_authority_sha256": (
                    independent_parent_authority["authority_sha256"]
                ),
                "function_preserving_upgrade_receipt_sha256": receipt["sha256"],
                "diagnostic_only": True,
                "promotion_eligible": False,
            }
        else:
            result["learner_lineage_parent"] = {
                "schema_version": LEARNER_LINEAGE_PARENT_SCHEMA,
                "role": "diagnostic_recent_history",
                "checkpoint": copy.deepcopy(upgrade["source"]),
                "corpus_producer": copy.deepcopy(verified["producer"]),
                "category_semantics_sha256": verified.get("category_semantics_sha256"),
                "function_preserving_upgrade_receipt_sha256": receipt["sha256"],
                "diagnostic_only": True,
                "promotion_eligible": False,
            }
    coherent_binding = result.get("coherent_direct_corpus_binding")
    if coherent_binding is not None:
        if not isinstance(coherent_binding, dict):
            raise ExecutorError("coherent direct-corpus binding is malformed")
        is_stage_c_final = (
            result.get("stage_c_final_corpus_admission") is not None
        )
        if is_stage_c_final and not allow_stage_c_final_current_parent:
            raise ExecutorError(
                "Stage-C final current-parent initialization requires its sealed "
                "final-replication authority"
            )
        if is_stage_c_final and (
            not source_matches_producer or independent_parent_authority is not None
        ):
            raise ExecutorError(
                "Stage-C final must upgrade the exact current corpus producer"
            )
        if not is_stage_c_final and independent_parent_authority is None:
            raise ExecutorError(
                "coherent direct corpus requires an independently authorized "
                "learner parent before rendering training"
            )
        coherent_binding = copy.deepcopy(coherent_binding)
        initializer_binding = {
            "role": (
                "stage_c_final_exact_current_parent"
                if is_stage_c_final
                else "diagnostic_independent_parent"
            ),
            "parent_checkpoint_sha256": upgrade["source"]["sha256"],
            "initializer_checkpoint_sha256": upgrade["upgraded_initializer"]["sha256"],
            "upgrade_module": upgrade["module"],
            "upgrade_receipt_file_sha256": receipt["sha256"],
            "upgrade_receipt_sha256": upgrade["receipt_sha256"],
        }
        if independent_parent_authority is not None:
            initializer_binding["independent_parent_authority_sha256"] = (
                independent_parent_authority["authority_sha256"]
            )
        coherent_binding["learner_initializer"] = initializer_binding
        coherent_binding.pop("binding_sha256", None)
        coherent_binding["binding_sha256"] = _value_sha256(coherent_binding)
        result["coherent_direct_corpus_binding"] = coherent_binding
    result["claim_identity_sha256"] = _value_sha256(
        {
            "schema_version": "a1-architecture-upgrade-training-identity-v1",
            "contract_sha256": verified["contract_sha256"],
            "upgrade_receipt_sha256": receipt["sha256"],
            "upgrade_receipt_digest": upgrade["receipt_sha256"],
        }
    )
    return result


def _initializer_transition_record(
    *,
    kind: str,
    role: str,
    source_checkpoint_sha256: str,
    output_checkpoint_sha256: str,
    sampled_rows: int,
    optimizer_steps: int,
    optimizer_state_terminal: str,
    receipt_path: str,
    receipt_file_sha256: str,
    receipt_state_sha256: str,
) -> dict[str, Any]:
    """Build one normalized typed initializer transition for lineage accounting."""

    return {
        "schema_version": lineage.INITIALIZER_TRANSITION_SCHEMA,
        "kind": kind,
        "role": role,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "output_checkpoint_sha256": output_checkpoint_sha256,
        "sampled_rows": sampled_rows,
        "optimizer_steps": optimizer_steps,
        "optimizer_state_terminal": optimizer_state_terminal,
        "receipt_path": receipt_path,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_state_sha256": receipt_state_sha256,
        "inherited_parameters_bit_identical": True,
        "main_output_max_abs_diff_decimal": "0",
    }


def _public_award_initializer_transition_record(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = authority["receipt"]
    evidence = receipt["evidence"]
    return _initializer_transition_record(
        kind="public_award_zero_initialization",
        role="feature_schema_zero_initialization",
        source_checkpoint_sha256=authority["source_checkpoint"]["sha256"],
        output_checkpoint_sha256=authority["transitioned_checkpoint"]["sha256"],
        sampled_rows=0,
        optimizer_steps=0,
        optimizer_state_terminal="not_constructed",
        receipt_path=str(receipt["path"]),
        receipt_file_sha256=str(receipt["file_sha256"]),
        receipt_state_sha256=str(evidence["state_sha256"]),
    )


def _pointer_initializer_transition_record(
    upgrade: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = upgrade["receipt"]
    return _initializer_transition_record(
        kind="function_preserving_pointer_upgrade",
        role="architecture_zero_diff_upgrade",
        source_checkpoint_sha256=str(upgrade["source"]["sha256"]),
        output_checkpoint_sha256=str(upgrade["upgraded_initializer"]["sha256"]),
        sampled_rows=0,
        optimizer_steps=0,
        optimizer_state_terminal="not_constructed",
        receipt_path=str(receipt["path"]),
        receipt_file_sha256=str(receipt["sha256"]),
        receipt_state_sha256=str(upgrade["receipt_sha256"]),
    )


def _warmup_initializer_transition_record(
    terminal: Mapping[str, Any], *, path: Path, file_sha256: str
) -> dict[str, Any]:
    result = terminal["result"]
    if (
        terminal.get("schema_version") != "a1-aux-pointer-warmup-terminal-v1"
        or result.get("status") != "complete"
        or result.get("optimizer_sidecar_discarded_for_joint") is not True
        or result.get("inherited_parameters_bit_identical") is not True
        or result.get("main_output_max_diff") != 0.0
    ):
        raise ExecutorError("typed initializer lineage lost head-only warmup evidence")
    return _initializer_transition_record(
        kind="head_only_auxiliary_warmup",
        role="head_only_auxiliary_commissioning",
        source_checkpoint_sha256=str(result["input_initializer_sha256"]),
        output_checkpoint_sha256=str(result["warmed_checkpoint_sha256"]),
        sampled_rows=int(result["sampled_rows"]),
        optimizer_steps=int(result["optimizer_steps"]),
        optimizer_state_terminal="discarded_before_joint_training",
        receipt_path=str(path),
        receipt_file_sha256=file_sha256,
        receipt_state_sha256=str(terminal["state_sha256"]),
    )


def _load_warmup_transition_record(
    *, published_executor_authority: Mapping[str, Any], expected_terminal_sha256: str
) -> dict[str, Any]:
    authority_path = Path(str(published_executor_authority["path"]))
    terminal_path = authority_path.parent / "20-warmup-terminal.json"
    try:
        terminal, file_sha256, _identity = aux_coordinator._stable_read_immutable_json(  # noqa: SLF001
            terminal_path, where="typed initializer warmup terminal"
        )
    except (OSError, aux_coordinator.CoordinatorError) as error:
        raise ExecutorError(f"cannot replay typed warmup lineage: {error}") from error
    if terminal.get("state_sha256") != expected_terminal_sha256:
        raise ExecutorError("typed warmup terminal differs from central authority")
    return _warmup_initializer_transition_record(
        terminal, path=terminal_path, file_sha256=file_sha256
    )


def _active_mps_processes(proc_root: Path = Path("/proc")) -> list[str]:
    """Return live CUDA MPS control/server processes without invoking a shell."""

    found: list[str] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as error:
        raise ExecutorError(
            f"cannot inspect process table for CUDA MPS: {error}"
        ) from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            # Processes can exit while /proc is being traversed. A process whose
            # directory disappears during the scan no longer threatens the run.
            continue
        except PermissionError as error:
            raise ExecutorError(
                f"cannot prove process {entry.name} is not CUDA MPS: {error}"
            ) from error
        except OSError as error:
            raise ExecutorError(
                f"cannot inspect process {entry.name} for CUDA MPS: {error}"
            ) from error
        argv = [
            part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part
        ]
        if not argv:
            continue
        executable = Path(argv[0]).name
        if executable in MPS_EXECUTABLES:
            found.append(f"pid={entry.name} executable={executable}")
    return sorted(found)


def _probe_b200(
    gpu: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    mps_probe: Callable[[], list[str]] = _active_mps_processes,
) -> str:
    """Fail closed unless ``gpu`` is one idle, directly-owned B200."""

    try:
        result = runner(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=name,compute_mode,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutorError(
            f"cannot verify selected B200 GPU {gpu}: {error}"
        ) from error
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ExecutorError(
            f"selected GPU {gpu} query returned {len(rows)} rows: {rows}"
        )
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise ExecutorError(f"selected GPU {gpu} query is malformed: {rows[0]!r}")
    name, compute_mode_raw, memory_used_raw = fields
    if "B200" not in name.upper():
        raise ExecutorError(f"selected GPU {gpu} is not exactly one B200: {name!r}")
    compute_mode = compute_mode_raw.upper().replace("-", "_").replace(" ", "_")
    if compute_mode not in ACCEPTABLE_COMPUTE_MODES:
        raise ExecutorError(
            f"selected B200 GPU {gpu} has unsafe compute mode {compute_mode_raw!r}; "
            f"expected Default or Exclusive Process"
        )
    try:
        memory_used_mib = int(memory_used_raw)
    except ValueError as error:
        raise ExecutorError(
            f"selected B200 GPU {gpu} has unparseable memory usage {memory_used_raw!r}"
        ) from error
    if memory_used_mib < 0 or memory_used_mib > MAX_IDLE_GPU_MEMORY_MIB:
        raise ExecutorError(
            f"selected B200 GPU {gpu} is not idle: memory.used={memory_used_mib} MiB "
            f"(maximum idle allowance {MAX_IDLE_GPU_MEMORY_MIB} MiB)"
        )

    try:
        processes = runner(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        mps_processes = mps_probe()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutorError(
            f"cannot prove selected B200 GPU {gpu} is idle: {error}"
        ) from error
    compute_processes = [
        line.strip() for line in processes.stdout.splitlines() if line.strip()
    ]
    if compute_processes:
        raise ExecutorError(
            f"selected B200 GPU {gpu} has active compute process(es): {compute_processes}"
        )
    if mps_processes:
        raise ExecutorError(
            "CUDA MPS is active; the sealed one-B200 learner requires direct exclusive "
            f"ownership: {mps_processes}"
        )
    return name


def _raise_nofile_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, MIN_NOFILE), hard)
    if target < MIN_NOFILE:
        raise ExecutorError(f"hard RLIMIT_NOFILE={hard} is below required {MIN_NOFILE}")
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


def _gpu_lock_path(gpu: int, *, lock_root: Path = Path("/tmp")) -> Path:
    if int(gpu) < 0:
        raise ExecutorError("physical GPU lock index must be non-negative")
    return lock_root / f"catan_zero_a1_b200_gpu{int(gpu)}.lock"


@contextmanager
def _physical_gpu_lock(gpu: int, *, lock_root: Path = Path("/tmp")):
    """Own one physical B200 across probe, claim, child, and receipt.

    The lock is advisory but fail-closed: nonblocking `flock`, no symlinks,
    regular file owned by the executor uid, private mode only. Different GPU
    indices intentionally map to independent files so diagnostic arms can run
    concurrently without weakening same-device exclusion.
    """

    path = _gpu_lock_path(gpu, lock_root=lock_root)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as error:
        raise ExecutorError(f"cannot open physical GPU lock {path}: {error}") from error
    handle = os.fdopen(fd, "r+b", buffering=0)
    try:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ExecutorError(f"physical GPU lock is not a regular file: {path}")
        if info.st_uid != os.geteuid():
            raise ExecutorError(
                f"physical GPU lock is not owned by uid {os.geteuid()}: {path}"
            )
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise ExecutorError(
                f"physical GPU lock has unsafe mode {oct(stat.S_IMODE(info.st_mode))}: {path}"
            )
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise ExecutorError(
                    f"physical B200 GPU {gpu} is already reserved by another A1 executor"
                ) from error
            raise ExecutorError(
                f"cannot lock physical B200 GPU {gpu}: {error}"
            ) from error
        yield path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def _claim_path(verified: dict[str, Any]) -> Path:
    """Return the one stable claim path for the sealed contract identity.

    The sealed seed-ledger path is the shared, contract-bound anchor. A caller
    cannot obtain a second dose by choosing another receipt or copying the lock.
    """

    contract_sha = str(
        verified.get("claim_identity_sha256", verified.get("contract_sha256", ""))
    )
    prefix = "sha256:"
    digest = contract_sha.removeprefix(prefix)
    if (
        not contract_sha.startswith(prefix)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ExecutorError(f"invalid sealed contract identity: {contract_sha!r}")
    try:
        ledger_value = verified["lock"]["fleet"]["seed_ledger"]["path"]
        ledger = Path(str(ledger_value)).expanduser().resolve(strict=True)
    except (KeyError, TypeError, OSError) as error:
        raise ExecutorError(
            "sealed contract has no resolvable seed-ledger claim anchor"
        ) from error
    if not ledger.is_file():
        raise ExecutorError(f"sealed seed-ledger anchor is not a file: {ledger}")
    return ledger.parent / CLAIM_DIRECTORY / f"{digest}.json"


def _load_claim_state(
    claim: Path,
    *,
    contract_sha256: str,
    claim_identity_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(claim.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(
            f"A1 contract claim is unreadable/corrupt: {claim}"
        ) from error
    if not isinstance(payload, dict):
        raise ExecutorError(f"A1 contract claim is not an object: {claim}")
    stated_digest = payload.get("state_sha256")
    unhashed = dict(payload)
    unhashed.pop("state_sha256", None)
    if stated_digest != _value_sha256(unhashed):
        raise ExecutorError(f"A1 contract claim digest is invalid: {claim}")
    expected_schemas = (
        {
            RETRY_CLAIM_SCHEMA,
            ABLATION_CLAIM_SCHEMA,
            UPGRADE_CLAIM_SCHEMA,
            CENTRAL_CLAIM_SCHEMA,
        }
        if claim_identity_sha256 is not None
        and claim_identity_sha256 != contract_sha256
        else {CLAIM_SCHEMA}
    )
    if payload.get("schema_version") not in expected_schemas:
        raise ExecutorError(f"A1 contract claim schema is invalid: {claim}")
    if payload.get("contract_sha256") != contract_sha256:
        raise ExecutorError(f"A1 contract claim identity mismatch: {claim}")
    if (
        claim_identity_sha256 is not None
        and payload.get("claim_identity_sha256", payload.get("contract_sha256"))
        != claim_identity_sha256
    ):
        raise ExecutorError(f"A1 derived claim identity mismatch: {claim}")
    return payload


def _require_unconsumed_contract(verified: dict[str, Any]) -> None:
    claim = _claim_path(verified)
    if claim.exists():
        state = _load_claim_state(
            claim,
            contract_sha256=str(verified["contract_sha256"]),
            claim_identity_sha256=str(
                verified.get("claim_identity_sha256", verified["contract_sha256"])
            ),
        )
        raise ExecutorError(
            "sealed A1 dose already has a durable claim: "
            f"status={state.get('status')!r} path={claim}"
        )


def _claim_attempt(verified: dict[str, Any], payload: dict[str, Any]) -> Path:
    claim = _claim_path(verified)
    _mkdir_durable(claim.parent)
    try:
        fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise ExecutorError(f"A1 training claim already exists: {claim}") from error
    durable_payload = _with_digest(payload, "state_sha256")
    with os.fdopen(fd, "wb") as handle:
        handle.write(_canonical_bytes(durable_payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(claim.parent)
    return claim


def _write_terminal_claim(
    claim: Path,
    payload: dict[str, Any],
    *,
    contract_sha256: str,
    claim_identity_sha256: str | None = None,
) -> dict[str, Any]:
    current = _load_claim_state(
        claim,
        contract_sha256=contract_sha256,
        claim_identity_sha256=claim_identity_sha256,
    )
    if current.get("status") != "claimed":
        raise ExecutorError(
            f"A1 claim is already terminal: status={current.get('status')!r} path={claim}"
        )
    terminal = _with_digest(payload, "state_sha256")
    tmp = claim.with_name(f".{claim.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_bytes(terminal) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, claim)
        _fsync_directory(claim.parent)
    finally:
        tmp.unlink(missing_ok=True)
    return terminal


def _train_command_namespace(command: list[str]) -> argparse.Namespace:
    canonical = (_REPO_ROOT / "tools" / "train_bc.py").resolve(strict=False)
    trainer_indices = [
        index
        for index, value in enumerate(command)
        if Path(value).resolve(strict=False) == canonical
    ]
    if len(trainer_indices) != 1:
        raise ExecutorError(
            "retry proof command is not the canonical train_bc entry point"
        )
    trainer_index = trainer_indices[0]
    try:
        args = train_bc.build_parser().parse_args(command[trainer_index + 1 :])
    except SystemExit as error:
        raise ExecutorError(
            "retry proof command cannot be parsed by train_bc"
        ) from error
    # Replay train_bc's effective architecture, not the parser's sentinel.
    # The failed production argv omitted --hidden-size; train_bc resolves that
    # to 640 for entity_graph before running checkpoint compatibility checks.
    if args.hidden_size is None:
        args.hidden_size = (
            640
            if args.arch == "entity_graph"
            else 768
            if args.arch == "xdim_graph"
            else 512
        )
    return args


def _literal_option_values(command: list[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(command):
        if item == flag:
            if index + 1 >= len(command):
                raise ExecutorError(f"retry proof command has valueless {flag}")
            values.append(command[index + 1])
        elif item.startswith(flag + "="):
            values.append(item.split("=", 1)[1])
    return values


def _checkpoint_architecture_mismatches(args: argparse.Namespace) -> list[str]:
    if not args.init_checkpoint:
        raise ExecutorError("retry proof command has no --init-checkpoint")
    try:
        import torch

        checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ExecutorError(
            f"cannot replay init-checkpoint architecture preflight: {error}"
        ) from error
    if not isinstance(checkpoint, dict):
        raise ExecutorError("init checkpoint is not a policy-checkpoint object")
    return train_bc._checkpoint_config_mismatches(  # noqa: SLF001
        policy_type=checkpoint.get("policy_type"),
        config=checkpoint.get("config"),
        args=args,
    )


def _load_failed_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"parent failure receipt is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ExecutorError("parent failure receipt is not an object")
    stated = payload.get("receipt_sha256")
    unhashed = dict(payload)
    unhashed.pop("receipt_sha256", None)
    if stated != _value_sha256(unhashed):
        raise ExecutorError("parent failure receipt semantic digest is invalid")
    if payload.get("schema_version") != RECEIPT_SCHEMA:
        raise ExecutorError("parent failure receipt schema is invalid")
    return payload


def _load_failed_retry_receipt(path: Path) -> dict[str, Any]:
    """Load a terminal v4 retry receipt without weakening its own digest."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(
            f"parent retry failure receipt is unreadable: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise ExecutorError("parent retry failure receipt is not an object")
    stated = payload.get("receipt_sha256")
    unhashed = dict(payload)
    unhashed.pop("receipt_sha256", None)
    if stated != _value_sha256(unhashed):
        raise ExecutorError("parent retry failure receipt semantic digest is invalid")
    if payload.get("schema_version") != RETRY_RECEIPT_SCHEMA:
        raise ExecutorError("parent retry failure receipt schema is invalid")
    return payload


def _load_retry_contract_reference(
    reference: Any, *, where: str
) -> tuple[dict[str, Any], Path]:
    """Replay a v1 retry contract from an exact path/file/semantic reference."""

    if not isinstance(reference, dict) or set(reference) != {
        "path",
        "file_sha256",
        "retry_contract_sha256",
    }:
        raise ExecutorError(f"{where} retry-contract reference drift")
    path = Path(str(reference["path"])).expanduser()
    if path.is_symlink():
        raise ExecutorError(f"{where} retry contract must not be a symlink")
    try:
        path = path.resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"{where} retry contract is unreadable") from error
    if not isinstance(payload, dict):
        raise ExecutorError(f"{where} retry contract is not an object")
    unhashed = dict(payload)
    stated = unhashed.pop("retry_contract_sha256", None)
    if (
        payload.get("schema_version") != RETRY_CONTRACT_SCHEMA
        or _file_sha256(path) != reference["file_sha256"]
        or stated != _value_sha256(unhashed)
        or stated != reference["retry_contract_sha256"]
    ):
        raise ExecutorError(f"{where} retry contract bytes/digest drift")
    return payload, path


def _write_retry_contract_no_clobber(path: Path, payload: dict[str, Any]) -> None:
    _mkdir_durable(path.parent)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("xb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(tmp, path)
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise ExecutorError(
            f"refusing to overwrite learner retry contract: {path}"
        ) from error
    finally:
        tmp.unlink(missing_ok=True)


def _validated_recorded_production_trainer_authority(
    authority: Any, *, expected_trainer_sha256: str, where: str
) -> dict[str, Any]:
    """Replay a receipt-bound trainer authority without trusting its labels."""

    expected_keys = {
        "schema_version",
        "repository_root",
        "relative_path",
        "path",
        "sha256",
        "code_surface",
        "code_surface_sha256",
        "authority_sha256",
    }
    if not isinstance(authority, dict) or set(authority) != expected_keys:
        raise ExecutorError(f"{where} trainer authority fields drift")
    unhashed = dict(authority)
    stated_authority_sha256 = unhashed.pop("authority_sha256", None)
    code_surface = authority.get("code_surface")
    if (
        authority.get("schema_version") != PRODUCTION_TRAINER_AUTHORITY_SCHEMA
        or authority.get("relative_path") != "tools/train_bc.py"
        or authority.get("sha256") != expected_trainer_sha256
        or stated_authority_sha256 != _value_sha256(unhashed)
        or not isinstance(code_surface, list)
        or authority.get("code_surface_sha256") != _value_sha256(code_surface)
    ):
        raise ExecutorError(f"{where} trainer authority digest/identity drift")
    records: dict[str, str] = {}
    for record in code_surface:
        if (
            not isinstance(record, dict)
            or set(record) != {"relative_path", "path", "sha256"}
            or not isinstance(record.get("relative_path"), str)
            or not isinstance(record.get("path"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(record.get("sha256"))) is None
            or record["relative_path"] in records
        ):
            raise ExecutorError(f"{where} trainer code-surface record drift")
        records[record["relative_path"]] = record["sha256"]
    if records.get("tools/train_bc.py") != expected_trainer_sha256:
        raise ExecutorError(f"{where} train_bc code-surface identity drift")
    return copy.deepcopy(authority)


def _production_retry_output_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    """Enumerate every learner artifact the failed command could have emitted."""

    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=False)
    report = Path(args.report).expanduser().resolve(strict=False)
    try:
        checkpoint_steps = train_bc._parse_checkpoint_steps(  # noqa: SLF001
            str(args.checkpoint_steps), max_steps=int(args.max_steps)
        )
    except SystemExit as error:
        raise ExecutorError(
            "parent retry checkpoint-step contract is invalid"
        ) from error
    paths = [
        checkpoint,
        Path(str(checkpoint) + ".optimizer.pt"),
        Path(str(checkpoint) + ".training-progress.json"),
        report,
    ]
    paths.extend(
        train_bc._step_checkpoint_path(checkpoint, step).resolve(strict=False)  # noqa: SLF001
        for step in checkpoint_steps
    )
    return tuple(paths)


def _production_retry_ddp_canary_semantics(
    canary: Any, *, expected_trainer_sha256: str, where: str
) -> dict[str, Any]:
    """Project a fresh canary across the one intended train_bc byte repair."""

    if not isinstance(canary, dict):
        raise ExecutorError(f"{where} DDP canary is missing")
    semantic_identity = canary.get("semantic_identity")
    if (
        not isinstance(semantic_identity, dict)
        or canary.get("semantic_identity_sha256") != _value_sha256(semantic_identity)
        or semantic_identity.get("train_bc_sha256") != expected_trainer_sha256
    ):
        raise ExecutorError(f"{where} DDP canary authority drift")
    portable = copy.deepcopy(semantic_identity)
    portable.pop("train_bc_sha256")
    return portable


def _authorize_production_preflight_serialization_retry(
    *,
    verified: dict[str, Any],
    parent_claim: Path,
    retry_command: list[str],
    checkpoint: Path,
    report: Path,
    receipt: Path,
    retry_contract_path: Path,
    publish: bool,
) -> dict[str, Any]:
    """Authorize the one byte-typed production DDP preflight repair.

    The consumed parent used one exact buggy ``train_bc`` and produced no
    learner artifacts.  Its command, input authority, environment, terminal
    claim, and failed receipt are replayed here.  Only the repaired trainer
    bytes and fresh output namespace may differ.
    """

    current_authority = _require_current_production_trainer_authority(verified)
    assert current_authority is not None
    current_authority = _validated_recorded_production_trainer_authority(
        current_authority,
        expected_trainer_sha256=FIXED_PRODUCTION_PREFLIGHT_TRAINER_SHA256,
        where="fixed production preflight",
    )

    expected_parent_claim = _claim_path(verified).resolve(strict=False)
    parent_claim = parent_claim.expanduser()
    if parent_claim.is_symlink():
        raise ExecutorError("retry parent claim must not be a symlink")
    try:
        parent_claim = parent_claim.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(f"cannot resolve parent failed claim: {error}") from error
    if parent_claim != expected_parent_claim:
        raise ExecutorError(
            "retry parent claim is not the canonical sealed-contract claim"
        )
    parent = _load_claim_state(
        parent_claim, contract_sha256=str(verified["contract_sha256"])
    )
    if (
        parent.get("status") != "failed"
        or parent.get("outputs") is not None
        or parent.get("lineage_dose") is not None
        or parent.get("returncode") != 1
        or parent.get("failure") != "ExecutorError: train_bc exited nonzero: 1"
    ):
        raise ExecutorError(
            "production preflight retry requires the exact terminal zero-output "
            "nonzero parent"
        )
    parent_command = parent.get("command")
    if not isinstance(parent_command, list) or not all(
        isinstance(item, str) for item in parent_command
    ):
        raise ExecutorError("retry parent claim has no canonical command")
    if parent.get("command_sha256") != _value_sha256(parent_command):
        raise ExecutorError("retry parent command digest does not replay")
    parent_binding = parent.get("execution_binding")
    if not isinstance(parent_binding, dict):
        raise ExecutorError("retry parent claim has no execution binding")
    _validate_execution_binding(parent_binding)
    if parent_binding["command_sha256"] != parent["command_sha256"]:
        raise ExecutorError("retry parent execution binding disagrees with its command")
    parent_input_binding = parent.get("input_binding")
    if not isinstance(parent_input_binding, dict) or parent_input_binding.get(
        "binding_sha256"
    ) != _value_sha256(
        {
            key: value
            for key, value in parent_input_binding.items()
            if key != "binding_sha256"
        }
    ):
        raise ExecutorError("retry parent input binding is invalid")

    receipt_target = Path(str(parent.get("receipt_target", ""))).expanduser()
    if receipt_target.is_symlink():
        raise ExecutorError("parent failure receipt must not be a symlink")
    try:
        receipt_target = receipt_target.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot resolve parent failure receipt: {error}"
        ) from error
    parent_receipt = _load_failed_receipt(receipt_target)
    agreement_fields = (
        "contract_sha256",
        "command",
        "command_sha256",
        "execution_binding",
        "input_binding",
        "training_transaction_sha256",
        "trainer_authority",
        "returncode",
        "outputs",
        "lineage_dose",
        "failure",
    )
    if (
        parent_receipt.get("status") != "failed"
        or parent_receipt.get("claim") != str(parent_claim)
        or parent_receipt.get("claim_state_sha256") != parent.get("state_sha256")
        or any(parent_receipt.get(key) != parent.get(key) for key in agreement_fields)
    ):
        raise ExecutorError("parent failed claim and receipt do not agree")
    parent_authority = _validated_recorded_production_trainer_authority(
        parent_receipt.get("trainer_authority"),
        expected_trainer_sha256=BUGGY_PRODUCTION_PREFLIGHT_TRAINER_SHA256,
        where="buggy production preflight parent",
    )
    if parent_input_binding.get("trainer_authority") != parent_authority:
        raise ExecutorError(
            "parent input binding disagrees with buggy trainer authority"
        )

    parent_records = {
        record["relative_path"]: record["sha256"]
        for record in parent_authority["code_surface"]
    }
    current_records = {
        record["relative_path"]: record["sha256"]
        for record in current_authority["code_surface"]
    }
    allowed_code_repairs = {
        "tools/train_bc.py",
        "tools/a1_one_dose_train.py",
    }
    if set(parent_records) != set(current_records) or any(
        parent_records[path] != current_records[path]
        for path in parent_records
        if path not in allowed_code_repairs
    ):
        raise ExecutorError(
            "production retry changes code outside the typed preflight repair surface"
        )

    parent_args = _train_command_namespace(parent_command)
    if (
        parent_command.count("torch.distributed.run") != 1
        or parent_command.count("--nproc_per_node=8") != 1
        or int(verified["recipe"]["world_size"]) != 8
    ):
        raise ExecutorError("production preflight retry parent is not exact 8-rank DDP")
    parent_output_paths = _production_retry_output_paths(parent_args)
    for path in parent_output_paths:
        if path.exists() or path.is_symlink():
            raise ExecutorError(
                "cannot prove zero-output/zero-step production preflight failure; "
                f"artifact exists: {path}"
            )

    retry_args = _train_command_namespace(retry_command)
    allowed_argv_drift = {"checkpoint", "report"}
    parent_values = vars(parent_args)
    retry_values = vars(retry_args)
    semantic_drift = sorted(
        key
        for key in set(parent_values) | set(retry_values)
        if key not in allowed_argv_drift
        and parent_values.get(key) != retry_values.get(key)
    )
    if semantic_drift:
        raise ExecutorError(
            f"production preflight retry changes learner semantics: {semantic_drift}"
        )
    selected_gpus = _selected_gpus(verified, fallback_gpu=0)
    if parent_binding["environment"] != _child_environment(selected_gpus):
        raise ExecutorError("production preflight retry changes learner environment")
    current_input_binding = _input_binding(verified)
    ignored_input_fields = {
        "binding_sha256",
        "ddp_canary",
        "trainer_authority",
    }
    parent_semantics = {
        key: value
        for key, value in parent_input_binding.items()
        if key not in ignored_input_fields
    }
    current_semantics = {
        key: value
        for key, value in current_input_binding.items()
        if key not in ignored_input_fields
    }
    if parent_semantics != current_semantics:
        raise ExecutorError("production preflight retry changes authenticated inputs")
    parent_canary_semantics = _production_retry_ddp_canary_semantics(
        parent_input_binding.get("ddp_canary"),
        expected_trainer_sha256=BUGGY_PRODUCTION_PREFLIGHT_TRAINER_SHA256,
        where="buggy production preflight parent",
    )
    current_canary_semantics = _production_retry_ddp_canary_semantics(
        current_input_binding.get("ddp_canary"),
        expected_trainer_sha256=FIXED_PRODUCTION_PREFLIGHT_TRAINER_SHA256,
        where="fixed production preflight retry",
    )
    if parent_canary_semantics != current_canary_semantics:
        raise ExecutorError("production preflight retry changes DDP canary semantics")

    checkpoint = checkpoint.expanduser().resolve(strict=False)
    report = report.expanduser().resolve(strict=False)
    receipt = receipt.expanduser().resolve(strict=False)
    retry_contract_path = retry_contract_path.expanduser()
    if retry_contract_path.is_symlink():
        raise ExecutorError("retry contract path must not be a symlink")
    retry_contract_path = retry_contract_path.resolve(strict=False)
    if (
        Path(retry_args.checkpoint).expanduser().resolve(strict=False) != checkpoint
        or Path(retry_args.report).expanduser().resolve(strict=False) != report
    ):
        raise ExecutorError("retry command does not bind the requested fresh outputs")
    _require_fresh_outputs(checkpoint, report, receipt)
    if retry_contract_path in {
        checkpoint,
        Path(str(checkpoint) + ".optimizer.pt"),
        Path(str(checkpoint) + ".training-progress.json"),
        report,
        receipt,
        parent_claim,
        receipt_target,
    }:
        raise ExecutorError("retry contract path aliases a claim, receipt, or output")
    if set(_production_retry_output_paths(retry_args)) & set(parent_output_paths):
        raise ExecutorError("retry must use a completely fresh output set")

    parent_evidence = {
        "claim": str(parent_claim),
        "claim_file_sha256": _file_sha256(parent_claim),
        "claim_state_sha256": parent["state_sha256"],
        "receipt": str(receipt_target),
        "receipt_file_sha256": _file_sha256(receipt_target),
        "receipt_sha256": parent_receipt["receipt_sha256"],
        "command_sha256": parent["command_sha256"],
        "input_binding_sha256": parent_input_binding["binding_sha256"],
        "training_transaction_sha256": parent["training_transaction_sha256"],
        "returncode": parent["returncode"],
        "failure": parent["failure"],
    }
    retry_identity_evidence = {
        "schema_version": RETRY_IDENTITY_SCHEMA,
        "repair_kind": PRODUCTION_PREFLIGHT_RETRY_REPAIR_KIND,
        "parent_contract_sha256": verified["contract_sha256"],
        "parent": parent_evidence,
    }
    # The claim identity excludes output paths and the repaired checkout.  This
    # exact consumed parent can therefore mint only one O_EXCL retry claim.
    retry_identity = _value_sha256(retry_identity_evidence)
    preserved = {
        "parent_contract_sha256": verified["contract_sha256"],
        "lock_file_sha256": verified["lock_file_sha256"],
        "corpus_meta_file_sha256": verified["corpus_meta_file_sha256"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "data_fingerprint": verified["data_fingerprint"],
        "producer_checkpoint_sha256": verified["producer"]["sha256"],
        "effective_learner_recipe_sha256": _value_sha256(verified["recipe"]),
        "learner_value_objective_sha256": _value_sha256(verified["objective"]),
        "selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
        "validation_game_seed_set_sha256": verified["validation_game_seed_set_sha256"],
        "parent_input_semantics_sha256": _value_sha256(parent_semantics),
        "retry_input_semantics_sha256": _value_sha256(current_semantics),
        "parent_ddp_canary": copy.deepcopy(parent_input_binding["ddp_canary"]),
        "ddp_canary_semantics_without_trainer_sha256": _value_sha256(
            current_canary_semantics
        ),
    }
    retry_contract = {
        "schema_version": RETRY_CONTRACT_SCHEMA,
        "retry_identity": retry_identity_evidence,
        "retry_identity_sha256": retry_identity,
        "parent": {
            **parent_evidence,
            "trainer_authority": parent_authority,
            "pre_optimizer_proof": {
                "kind": PRODUCTION_PREFLIGHT_RETRY_REPAIR_KIND,
                "buggy_train_bc_sha256": (BUGGY_PRODUCTION_PREFLIGHT_TRAINER_SHA256),
                "failure_phase": "ddp_preflight_json_publish_before_model_optimizer",
                "optimizer_steps": 0,
                "outputs": None,
                "absent_output_paths": [str(path) for path in parent_output_paths],
            },
        },
        "preserved_bindings": preserved,
        "retry": {
            "command_sha256": _value_sha256(retry_command),
            "trainer_authority": current_authority,
            "fixed_train_bc_sha256": FIXED_PRODUCTION_PREFLIGHT_TRAINER_SHA256,
            "ddp_canary": copy.deepcopy(current_input_binding["ddp_canary"]),
            "allowed_argv_drift": sorted(allowed_argv_drift),
            "checkpoint": str(checkpoint),
            "optimizer_sidecar": str(Path(str(checkpoint) + ".optimizer.pt")),
            "progress_sidecar": str(Path(str(checkpoint) + ".training-progress.json")),
            "report": str(report),
            "receipt": str(receipt),
        },
    }
    retry_contract["retry_contract_sha256"] = _value_sha256(retry_contract)
    derived_claim_path = _claim_path(
        {**verified, "claim_identity_sha256": retry_identity}
    ).resolve(strict=False)
    if retry_contract_path == derived_claim_path:
        raise ExecutorError("retry contract path aliases its derived durable claim")
    if publish:
        if retry_contract_path.exists():
            try:
                existing = json.loads(retry_contract_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ExecutorError("existing retry contract is unreadable") from error
            if existing != retry_contract:
                raise ExecutorError(
                    "existing retry contract differs; refusing overwrite/edit"
                )
        else:
            _write_retry_contract_no_clobber(retry_contract_path, retry_contract)
    derived = dict(verified)
    derived.update(
        {
            "claim_identity_sha256": retry_identity,
            "retry_contract": retry_contract,
            "retry_contract_path": retry_contract_path,
            "retry_contract_file_sha256": (
                _file_sha256(retry_contract_path) if publish else None
            ),
        }
    )
    return derived


def _authorize_production_preflight_transport_retry(
    *,
    verified: dict[str, Any],
    parent_claim: Path,
    retry_command: list[str],
    checkpoint: Path,
    report: Path,
    receipt: Path,
    retry_contract_path: Path,
    publish: bool,
) -> dict[str, Any]:
    """Authorize one final retry for the TCPStore single-value overflow.

    The parent must be the *failed first typed retry*, not merely another
    failed production attempt.  This replays both v4 parent artifacts, the
    first retry contract, and that contract's original v3 causal parent before
    allowing only exact chunked-transport trainer bytes and fresh outputs.
    """

    current_authority = _require_current_production_trainer_authority(verified)
    assert current_authority is not None
    current_authority = _validated_recorded_production_trainer_authority(
        current_authority,
        expected_trainer_sha256=FIXED_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256,
        where="fixed production preflight transport",
    )

    parent_claim = parent_claim.expanduser()
    if parent_claim.is_symlink():
        raise ExecutorError("retry parent claim must not be a symlink")
    try:
        parent_claim = parent_claim.resolve(strict=True)
        parent_hint = json.loads(parent_claim.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(
            f"cannot load parent failed retry claim: {error}"
        ) from error
    if not isinstance(parent_hint, dict):
        raise ExecutorError("parent failed retry claim is not an object")
    parent_identity = parent_hint.get("claim_identity_sha256")
    if not isinstance(parent_identity, str):
        raise ExecutorError("parent failed retry claim has no derived identity")
    expected_parent_claim = _claim_path(
        {**verified, "claim_identity_sha256": parent_identity}
    ).resolve(strict=False)
    if parent_claim != expected_parent_claim:
        raise ExecutorError("transport retry parent is not its canonical derived claim")
    parent = _load_claim_state(
        parent_claim,
        contract_sha256=str(verified["contract_sha256"]),
        claim_identity_sha256=parent_identity,
    )
    if (
        parent.get("schema_version") != RETRY_CLAIM_SCHEMA
        or parent.get("status") != "failed"
        or parent.get("outputs") is not None
        or parent.get("lineage_dose") is not None
        or parent.get("returncode") != 1
        or parent.get("failure") != "ExecutorError: train_bc exited nonzero: 1"
    ):
        raise ExecutorError(
            "transport retry requires the exact failed zero-output v4 parent"
        )
    parent_command = parent.get("command")
    if not isinstance(parent_command, list) or not all(
        isinstance(item, str) for item in parent_command
    ):
        raise ExecutorError("transport retry parent has no canonical command")
    if parent.get("command_sha256") != _value_sha256(parent_command):
        raise ExecutorError("transport retry parent command digest does not replay")
    parent_binding = parent.get("execution_binding")
    if not isinstance(parent_binding, dict):
        raise ExecutorError("transport retry parent has no execution binding")
    _validate_execution_binding(parent_binding)
    if parent_binding["command_sha256"] != parent["command_sha256"]:
        raise ExecutorError("transport retry parent execution binding drift")
    parent_input_binding = parent.get("input_binding")
    if not isinstance(parent_input_binding, dict) or parent_input_binding.get(
        "binding_sha256"
    ) != _value_sha256(
        {
            key: value
            for key, value in parent_input_binding.items()
            if key != "binding_sha256"
        }
    ):
        raise ExecutorError("transport retry parent input binding is invalid")

    receipt_target = Path(str(parent.get("receipt_target", ""))).expanduser()
    if receipt_target.is_symlink():
        raise ExecutorError("parent retry failure receipt must not be a symlink")
    try:
        receipt_target = receipt_target.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(f"cannot resolve parent retry receipt: {error}") from error
    parent_receipt = _load_failed_retry_receipt(receipt_target)
    agreement_fields = (
        "contract_sha256",
        "claim_identity_sha256",
        "retry_contract",
        "command",
        "command_sha256",
        "execution_binding",
        "input_binding",
        "training_transaction_sha256",
        "trainer_authority",
        "returncode",
        "outputs",
        "lineage_dose",
        "failure",
    )
    if (
        parent_receipt.get("status") != "failed"
        or parent_receipt.get("claim") != str(parent_claim)
        or parent_receipt.get("claim_state_sha256") != parent.get("state_sha256")
        or any(parent_receipt.get(key) != parent.get(key) for key in agreement_fields)
    ):
        raise ExecutorError("parent failed retry claim and receipt do not agree")

    first_contract, first_contract_path = _load_retry_contract_reference(
        parent.get("retry_contract"), where="parent"
    )
    if parent_receipt.get("retry_contract") != parent.get("retry_contract"):
        raise ExecutorError("parent claim/receipt retry-contract references disagree")
    first_identity = first_contract.get("retry_identity")
    if (
        not isinstance(first_identity, dict)
        or set(first_identity)
        != {"schema_version", "repair_kind", "parent_contract_sha256", "parent"}
        or first_identity.get("schema_version") != RETRY_IDENTITY_SCHEMA
        or first_identity.get("repair_kind") != PRODUCTION_PREFLIGHT_RETRY_REPAIR_KIND
        or first_identity.get("parent_contract_sha256") != verified["contract_sha256"]
        or first_contract.get("retry_identity_sha256") != _value_sha256(first_identity)
        or first_contract.get("retry_identity_sha256") != parent_identity
    ):
        raise ExecutorError("parent is not the exact first production preflight retry")
    first_retry = first_contract.get("retry")
    first_parent = first_contract.get("parent")
    if not isinstance(first_retry, dict) or not isinstance(first_parent, dict):
        raise ExecutorError("first retry contract causal fields are invalid")
    parent_authority = _validated_recorded_production_trainer_authority(
        first_retry.get("trainer_authority"),
        expected_trainer_sha256=BUGGY_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256,
        where="single-value TCPStore parent",
    )
    if (
        first_retry.get("fixed_train_bc_sha256")
        != BUGGY_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256
        or parent.get("trainer_authority") != parent_authority
        or parent_input_binding.get("trainer_authority") != parent_authority
    ):
        raise ExecutorError("failed transport parent trainer authority drift")

    # Replay the original failed v3 claim/receipt named by the first contract.
    original_evidence = first_identity.get("parent")
    if not isinstance(original_evidence, dict):
        raise ExecutorError("first retry contract lacks original-parent evidence")
    original_claim_path = Path(str(original_evidence.get("claim", ""))).expanduser()
    original_receipt_path = Path(str(original_evidence.get("receipt", ""))).expanduser()
    if original_claim_path.is_symlink() or original_receipt_path.is_symlink():
        raise ExecutorError("first retry causal parent must not use symlinks")
    try:
        original_claim_path = original_claim_path.resolve(strict=True)
        original_receipt_path = original_receipt_path.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot replay first retry causal parent: {error}"
        ) from error
    if original_claim_path != _claim_path(verified).resolve(strict=False):
        raise ExecutorError("first retry does not descend from the sealed dose claim")
    original_claim = _load_claim_state(
        original_claim_path, contract_sha256=str(verified["contract_sha256"])
    )
    original_receipt = _load_failed_receipt(original_receipt_path)
    if (
        _file_sha256(original_claim_path) != original_evidence.get("claim_file_sha256")
        or original_claim.get("state_sha256")
        != original_evidence.get("claim_state_sha256")
        or _file_sha256(original_receipt_path)
        != original_evidence.get("receipt_file_sha256")
        or original_receipt.get("receipt_sha256")
        != original_evidence.get("receipt_sha256")
        or original_claim.get("receipt_target") != str(original_receipt_path)
        or original_receipt.get("claim") != str(original_claim_path)
        or original_receipt.get("claim_state_sha256")
        != original_claim.get("state_sha256")
        or original_claim.get("command_sha256")
        != original_evidence.get("command_sha256")
        or original_claim.get("returncode") != original_evidence.get("returncode")
        or original_claim.get("failure") != original_evidence.get("failure")
        or first_parent.get("claim_file_sha256")
        != original_evidence.get("claim_file_sha256")
        or first_parent.get("receipt_file_sha256")
        != original_evidence.get("receipt_file_sha256")
    ):
        raise ExecutorError("first retry original causal chain bytes/digests drift")

    parent_records = {
        record["relative_path"]: record["sha256"]
        for record in parent_authority["code_surface"]
    }
    current_records = {
        record["relative_path"]: record["sha256"]
        for record in current_authority["code_surface"]
    }
    allowed_code_repairs = {"tools/train_bc.py", "tools/a1_one_dose_train.py"}
    if set(parent_records) != set(current_records) or any(
        parent_records[path] != current_records[path]
        for path in parent_records
        if path not in allowed_code_repairs
    ):
        raise ExecutorError(
            "transport retry changes code outside the typed transport repair surface"
        )

    parent_args = _train_command_namespace(parent_command)
    if (
        parent_command.count("torch.distributed.run") != 1
        or parent_command.count("--nproc_per_node=8") != 1
        or int(verified["recipe"]["world_size"]) != 8
    ):
        raise ExecutorError("transport retry parent is not exact 8-rank DDP")
    parent_output_paths = _production_retry_output_paths(parent_args)
    for path in parent_output_paths:
        if path.exists() or path.is_symlink():
            raise ExecutorError(
                "cannot prove zero-output/zero-step transport failure; "
                f"artifact exists: {path}"
            )

    retry_args = _train_command_namespace(retry_command)
    allowed_argv_drift = {"checkpoint", "report"}
    parent_values = vars(parent_args)
    retry_values = vars(retry_args)
    semantic_drift = sorted(
        key
        for key in set(parent_values) | set(retry_values)
        if key not in allowed_argv_drift
        and parent_values.get(key) != retry_values.get(key)
    )
    if semantic_drift:
        raise ExecutorError(
            f"transport retry changes learner semantics: {semantic_drift}"
        )
    selected_gpus = _selected_gpus(verified, fallback_gpu=0)
    if parent_binding["environment"] != _child_environment(selected_gpus):
        raise ExecutorError("transport retry changes learner environment")
    current_input_binding = _input_binding(verified)
    ignored_input_fields = {"binding_sha256", "ddp_canary", "trainer_authority"}
    parent_semantics = {
        key: value
        for key, value in parent_input_binding.items()
        if key not in ignored_input_fields
    }
    current_semantics = {
        key: value
        for key, value in current_input_binding.items()
        if key not in ignored_input_fields
    }
    if parent_semantics != current_semantics:
        raise ExecutorError("transport retry changes authenticated inputs")
    parent_canary_semantics = _production_retry_ddp_canary_semantics(
        parent_input_binding.get("ddp_canary"),
        expected_trainer_sha256=BUGGY_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256,
        where="single-value TCPStore parent",
    )
    current_canary_semantics = _production_retry_ddp_canary_semantics(
        current_input_binding.get("ddp_canary"),
        expected_trainer_sha256=FIXED_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256,
        where="chunked TCPStore retry",
    )
    if parent_canary_semantics != current_canary_semantics:
        raise ExecutorError("transport retry changes DDP canary semantics")

    checkpoint = checkpoint.expanduser().resolve(strict=False)
    report = report.expanduser().resolve(strict=False)
    receipt = receipt.expanduser().resolve(strict=False)
    retry_contract_path = retry_contract_path.expanduser()
    if retry_contract_path.is_symlink():
        raise ExecutorError("retry contract path must not be a symlink")
    retry_contract_path = retry_contract_path.resolve(strict=False)
    if (
        Path(retry_args.checkpoint).expanduser().resolve(strict=False) != checkpoint
        or Path(retry_args.report).expanduser().resolve(strict=False) != report
    ):
        raise ExecutorError("transport retry command does not bind fresh outputs")
    _require_fresh_outputs(checkpoint, report, receipt)
    if retry_contract_path in {
        checkpoint,
        Path(str(checkpoint) + ".optimizer.pt"),
        Path(str(checkpoint) + ".training-progress.json"),
        report,
        receipt,
        parent_claim,
        receipt_target,
        first_contract_path,
    }:
        raise ExecutorError("transport retry contract aliases evidence or output")
    if set(_production_retry_output_paths(retry_args)) & set(parent_output_paths):
        raise ExecutorError("transport retry must use a completely fresh output set")

    parent_evidence = {
        "claim": str(parent_claim),
        "claim_file_sha256": _file_sha256(parent_claim),
        "claim_state_sha256": parent["state_sha256"],
        "receipt": str(receipt_target),
        "receipt_file_sha256": _file_sha256(receipt_target),
        "receipt_sha256": parent_receipt["receipt_sha256"],
        "retry_contract": str(first_contract_path),
        "retry_contract_file_sha256": _file_sha256(first_contract_path),
        "retry_contract_sha256": first_contract["retry_contract_sha256"],
        "retry_identity_sha256": first_contract["retry_identity_sha256"],
        "command_sha256": parent["command_sha256"],
        "input_binding_sha256": parent_input_binding["binding_sha256"],
        "training_transaction_sha256": parent["training_transaction_sha256"],
        "returncode": parent["returncode"],
        "failure": parent["failure"],
    }
    retry_identity_evidence = {
        "schema_version": RETRY_IDENTITY_SCHEMA,
        "repair_kind": PRODUCTION_PREFLIGHT_TRANSPORT_RETRY_REPAIR_KIND,
        "parent_contract_sha256": verified["contract_sha256"],
        "parent": parent_evidence,
    }
    retry_identity = _value_sha256(retry_identity_evidence)
    preserved = {
        "parent_contract_sha256": verified["contract_sha256"],
        "first_retry_identity_sha256": first_contract["retry_identity_sha256"],
        "first_retry_contract_sha256": first_contract["retry_contract_sha256"],
        "lock_file_sha256": verified["lock_file_sha256"],
        "corpus_meta_file_sha256": verified["corpus_meta_file_sha256"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "data_fingerprint": verified["data_fingerprint"],
        "producer_checkpoint_sha256": verified["producer"]["sha256"],
        "effective_learner_recipe_sha256": _value_sha256(verified["recipe"]),
        "learner_value_objective_sha256": _value_sha256(verified["objective"]),
        "selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
        "validation_game_seed_set_sha256": verified["validation_game_seed_set_sha256"],
        "parent_input_semantics_sha256": _value_sha256(parent_semantics),
        "retry_input_semantics_sha256": _value_sha256(current_semantics),
        "ddp_canary_semantics_without_trainer_sha256": _value_sha256(
            current_canary_semantics
        ),
    }
    retry_contract = {
        "schema_version": RETRY_CONTRACT_SCHEMA,
        "retry_identity": retry_identity_evidence,
        "retry_identity_sha256": retry_identity,
        "parent": {
            **parent_evidence,
            "trainer_authority": parent_authority,
            "causal_parent_retry_contract": copy.deepcopy(first_contract),
            "pre_optimizer_proof": {
                "kind": PRODUCTION_PREFLIGHT_TRANSPORT_RETRY_REPAIR_KIND,
                "buggy_train_bc_sha256": (
                    BUGGY_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256
                ),
                "failure_phase": "tcpstore_single_value_publish_before_model_optimizer",
                "optimizer_steps": 0,
                "outputs": None,
                "absent_output_paths": [str(path) for path in parent_output_paths],
            },
        },
        "preserved_bindings": preserved,
        "retry": {
            "command_sha256": _value_sha256(retry_command),
            "trainer_authority": current_authority,
            "fixed_train_bc_sha256": (
                FIXED_PRODUCTION_PREFLIGHT_TRANSPORT_TRAINER_SHA256
            ),
            "transport": {
                "schema_version": train_bc.A1_PREFLIGHT_STORE_PACKET_SCHEMA,
                "chunk_bytes": train_bc.A1_PREFLIGHT_STORE_CHUNK_BYTES,
                "publish_order": "chunks_then_authenticated_manifest",
            },
            "ddp_canary": copy.deepcopy(current_input_binding["ddp_canary"]),
            "allowed_argv_drift": sorted(allowed_argv_drift),
            "checkpoint": str(checkpoint),
            "optimizer_sidecar": str(Path(str(checkpoint) + ".optimizer.pt")),
            "progress_sidecar": str(Path(str(checkpoint) + ".training-progress.json")),
            "report": str(report),
            "receipt": str(receipt),
        },
    }
    retry_contract["retry_contract_sha256"] = _value_sha256(retry_contract)
    derived_claim_path = _claim_path(
        {**verified, "claim_identity_sha256": retry_identity}
    ).resolve(strict=False)
    if retry_contract_path == derived_claim_path:
        raise ExecutorError("retry contract path aliases its derived durable claim")
    if publish:
        if retry_contract_path.exists():
            try:
                existing = json.loads(retry_contract_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ExecutorError("existing retry contract is unreadable") from error
            if existing != retry_contract:
                raise ExecutorError(
                    "existing retry contract differs; refusing overwrite/edit"
                )
        else:
            _write_retry_contract_no_clobber(retry_contract_path, retry_contract)
    derived = dict(verified)
    derived.update(
        {
            "claim_identity_sha256": retry_identity,
            "retry_contract": retry_contract,
            "retry_contract_path": retry_contract_path,
            "retry_contract_file_sha256": (
                _file_sha256(retry_contract_path) if publish else None
            ),
        }
    )
    return derived


def authorize_failed_before_optimizer_retry(
    *,
    verified: dict[str, Any],
    parent_claim: Path,
    retry_command: list[str],
    checkpoint: Path,
    report: Path,
    receipt: Path,
    retry_contract_path: Path,
    publish: bool,
) -> dict[str, Any]:
    """Derive one immutable learner-only retry from a proven zero-step failure.

    The parent generation/data contract remains the training-data identity.  A
    separate digest identifies this retry authorization and therefore produces a
    fresh durable claim path.  The failed parent claim and receipt are only read
    and hashed; neither is edited, removed, or reused.
    """

    if verified.get("data_kind") == "production_composite_v2":
        try:
            parent_schema = json.loads(
                parent_claim.expanduser().read_text(encoding="utf-8")
            ).get("schema_version")
        except (AttributeError, OSError, UnicodeError, json.JSONDecodeError):
            parent_schema = None
        if parent_schema == RETRY_CLAIM_SCHEMA:
            return _authorize_production_preflight_transport_retry(
                verified=verified,
                parent_claim=parent_claim,
                retry_command=retry_command,
                checkpoint=checkpoint,
                report=report,
                receipt=receipt,
                retry_contract_path=retry_contract_path,
                publish=publish,
            )
        return _authorize_production_preflight_serialization_retry(
            verified=verified,
            parent_claim=parent_claim,
            retry_command=retry_command,
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            retry_contract_path=retry_contract_path,
            publish=publish,
        )

    live_bindings = {
        Path(verified["lock_path"]): verified["lock_file_sha256"],
        Path(verified["data_path"]) / "corpus_meta.json": verified[
            "corpus_meta_file_sha256"
        ],
        Path(verified["validation_path"]): verified["validation_file_sha256"],
        Path(verified["producer"]["path"]): verified["producer"]["sha256"],
    }
    for path, expected_sha256 in live_bindings.items():
        try:
            actual_sha256 = _file_sha256(path.resolve(strict=True))
        except OSError as error:
            raise ExecutorError(
                f"cannot replay sealed retry binding {path}: {error}"
            ) from error
        if actual_sha256 != expected_sha256:
            raise ExecutorError(f"sealed retry binding drift: {path}")

    expected_parent_claim = _claim_path(verified).resolve(strict=False)
    parent_claim = parent_claim.expanduser()
    if parent_claim.is_symlink():
        raise ExecutorError("retry parent claim must not be a symlink")
    try:
        parent_claim = parent_claim.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(f"cannot resolve parent failed claim: {error}") from error
    if parent_claim != expected_parent_claim:
        raise ExecutorError(
            "retry parent claim is not the canonical sealed-contract claim"
        )
    parent = _load_claim_state(
        parent_claim, contract_sha256=str(verified["contract_sha256"])
    )
    if parent.get("status") != "failed":
        raise ExecutorError("retry requires a terminal failed parent claim")
    if parent.get("outputs") is not None:
        raise ExecutorError("retry parent claim contains training outputs")
    if not isinstance(parent.get("returncode"), int) or parent["returncode"] == 0:
        raise ExecutorError("retry parent claim does not prove a nonzero child exit")
    if not isinstance(parent.get("failure"), str) or not parent["failure"]:
        raise ExecutorError("retry parent claim has no failure evidence")
    parent_command = parent.get("command")
    if not isinstance(parent_command, list) or not all(
        isinstance(item, str) for item in parent_command
    ):
        raise ExecutorError("retry parent claim has no canonical command")
    if parent.get("command_sha256") != _value_sha256(parent_command):
        raise ExecutorError("retry parent command digest does not replay")
    parent_binding = parent.get("execution_binding")
    if not isinstance(parent_binding, dict):
        raise ExecutorError("retry parent claim has no execution binding")
    _validate_execution_binding(parent_binding)
    if parent_binding["command_sha256"] != parent["command_sha256"]:
        raise ExecutorError("retry parent execution binding disagrees with its command")

    receipt_target = Path(str(parent.get("receipt_target", ""))).expanduser()
    if receipt_target.is_symlink():
        raise ExecutorError("parent failure receipt must not be a symlink")
    try:
        receipt_target = receipt_target.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot resolve parent failure receipt: {error}"
        ) from error
    parent_receipt = _load_failed_receipt(receipt_target)
    if (
        parent_receipt.get("status") != "failed"
        or parent_receipt.get("outputs") is not None
        or parent_receipt.get("claim") != str(parent_claim)
        or parent_receipt.get("claim_state_sha256") != parent.get("state_sha256")
        or parent_receipt.get("contract_sha256") != verified["contract_sha256"]
        or parent_receipt.get("command_sha256") != parent["command_sha256"]
        or parent_receipt.get("returncode") != parent["returncode"]
        or parent_receipt.get("failure") != parent["failure"]
        or parent_receipt.get("execution_binding") != parent_binding
    ):
        raise ExecutorError("parent failed claim and receipt do not agree")
    preserved_receipt_bindings = {
        "lock": str(verified["lock_path"]),
        "lock_file_sha256": verified["lock_file_sha256"],
        "corpus": str(verified["data_path"]),
        "corpus_meta_file_sha256": verified["corpus_meta_file_sha256"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "validation_manifest": str(verified["validation_path"]),
        "validation_manifest_file_sha256": verified["validation_file_sha256"],
        "producer_checkpoint_sha256": verified["producer"]["sha256"],
        "learner_training_recipe_sha256": _value_sha256(verified["recipe"]),
    }
    for key, expected in preserved_receipt_bindings.items():
        if parent_receipt.get(key) != expected or parent.get(key) != expected:
            raise ExecutorError(f"parent failure drifted from sealed binding {key}")

    if _literal_option_values(parent_command, "--graph-layers"):
        raise ExecutorError(
            "authorized parent must literally omit --graph-layers and use the "
            "historical train_bc default"
        )
    parent_args = _train_command_namespace(parent_command)
    parent_mismatches = _checkpoint_architecture_mismatches(parent_args)
    if parent_mismatches != ["graph_layers checkpoint=6 cli=4"]:
        raise ExecutorError(
            "parent failure is not the authorized pre-optimizer graph-layer mismatch"
        )
    parent_checkpoint = Path(parent_args.checkpoint).expanduser().resolve(strict=False)
    parent_report = Path(parent_args.report).expanduser().resolve(strict=False)
    parent_optimizer = Path(str(parent_checkpoint) + ".optimizer.pt")
    for path in (parent_checkpoint, parent_optimizer, parent_report):
        if path.exists():
            raise ExecutorError(
                f"cannot prove zero-output/zero-step parent failure; artifact exists: {path}"
            )

    if _literal_option_values(retry_command, "--graph-layers") != ["6"]:
        raise ExecutorError(
            "corrected retry still fails architecture authorization: must contain "
            "exactly one literal --graph-layers 6"
        )
    retry_args = _train_command_namespace(retry_command)
    retry_mismatches = _checkpoint_architecture_mismatches(retry_args)
    if retry_mismatches:
        raise ExecutorError(
            "corrected retry command still fails architecture preflight: "
            + "; ".join(retry_mismatches)
        )
    allowed_drift = {
        "graph_layers",
        "checkpoint",
        "report",
    }
    parent_values = vars(parent_args)
    retry_values = vars(retry_args)
    drift = sorted(
        key
        for key in set(parent_values) | set(retry_values)
        if key not in allowed_drift and parent_values.get(key) != retry_values.get(key)
    )
    if drift:
        raise ExecutorError(
            f"retry changes non-architecture learner semantics: {drift}"
        )
    if Path(retry_args.init_checkpoint).resolve(strict=False) != Path(
        parent_args.init_checkpoint
    ).resolve(strict=False):
        raise ExecutorError("retry changes the sealed producer checkpoint")
    checkpoint = checkpoint.expanduser().resolve(strict=False)
    report = report.expanduser().resolve(strict=False)
    receipt = receipt.expanduser().resolve(strict=False)
    retry_contract_path = retry_contract_path.expanduser()
    if retry_contract_path.is_symlink():
        raise ExecutorError("retry contract path must not be a symlink")
    retry_contract_path = retry_contract_path.resolve(strict=False)
    if (
        Path(retry_args.checkpoint).resolve(strict=False) != checkpoint
        or Path(retry_args.report).resolve(strict=False) != report
    ):
        raise ExecutorError("retry command does not bind the requested fresh outputs")
    _require_fresh_outputs(checkpoint, report, receipt)
    if retry_contract_path in {
        checkpoint,
        Path(str(checkpoint) + ".optimizer.pt"),
        report,
        receipt,
        parent_claim,
        receipt_target,
    }:
        raise ExecutorError("retry contract path aliases a claim, receipt, or output")
    if {checkpoint, Path(str(checkpoint) + ".optimizer.pt"), report} & {
        parent_checkpoint,
        parent_optimizer,
        parent_report,
    }:
        raise ExecutorError("retry must use a completely fresh output set")

    preserved = {
        "parent_contract_sha256": verified["contract_sha256"],
        "parent_lock": str(verified["lock_path"]),
        "parent_lock_file_sha256": verified["lock_file_sha256"],
        "corpus": str(verified["data_path"]),
        "corpus_meta_file_sha256": verified["corpus_meta_file_sha256"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "data_fingerprint": verified["data_fingerprint"],
        "producer_checkpoint_sha256": verified["producer"]["sha256"],
        "producer_checkpoint": str(verified["producer"]["path"]),
        "learner_training_recipe_sha256": _value_sha256(verified["recipe"]),
        "learner_value_objective_sha256": _value_sha256(verified["objective"]),
        "selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
        "validation_manifest_file_sha256": verified["validation_file_sha256"],
        "validation_manifest": str(verified["validation_path"]),
        "validation_game_seed_set_sha256": verified["validation_game_seed_set_sha256"],
    }
    parent_evidence = {
        "claim": str(parent_claim),
        "claim_file_sha256": _file_sha256(parent_claim),
        "claim_state_sha256": parent["state_sha256"],
        "receipt": str(receipt_target),
        "receipt_file_sha256": _file_sha256(receipt_target),
        "receipt_sha256": parent_receipt["receipt_sha256"],
        "command_sha256": parent["command_sha256"],
        "returncode": parent["returncode"],
        "failure": parent["failure"],
    }
    retry_identity_evidence = {
        "schema_version": RETRY_IDENTITY_SCHEMA,
        "repair_kind": RETRY_REPAIR_KIND,
        "parent_contract_sha256": verified["contract_sha256"],
        "parent": parent_evidence,
    }
    # This is deliberately independent of r2 argv and output paths.  Therefore
    # changing filenames cannot mint another retry claim; O_EXCL on the derived
    # claim path physically caps this repair at one attempt.
    retry_identity = _value_sha256(retry_identity_evidence)
    retry_contract = {
        "schema_version": RETRY_CONTRACT_SCHEMA,
        "retry_identity": retry_identity_evidence,
        "retry_identity_sha256": retry_identity,
        "parent": {
            **parent_evidence,
            "pre_optimizer_proof": {
                "kind": "replayed_init_checkpoint_architecture_preflight",
                "mismatches": parent_mismatches,
                "optimizer_steps": 0,
                "outputs": None,
            },
        },
        "preserved_bindings": preserved,
        "retry": {
            "command_sha256": _value_sha256(retry_command),
            "architecture_correction": {
                "graph_layers_before": int(parent_args.graph_layers),
                "graph_layers_after": int(retry_args.graph_layers),
            },
            "checkpoint": str(checkpoint),
            "optimizer_sidecar": str(Path(str(checkpoint) + ".optimizer.pt")),
            "report": str(report),
            "receipt": str(receipt),
        },
    }
    retry_contract["retry_contract_sha256"] = _value_sha256(retry_contract)
    derived_claim_path = _claim_path(
        {**verified, "claim_identity_sha256": retry_identity}
    ).resolve(strict=False)
    if retry_contract_path == derived_claim_path:
        raise ExecutorError("retry contract path aliases its derived durable claim")
    if publish:
        if retry_contract_path.exists():
            try:
                existing = json.loads(retry_contract_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ExecutorError("existing retry contract is unreadable") from error
            if existing != retry_contract:
                raise ExecutorError(
                    "existing retry contract differs; refusing overwrite/edit"
                )
        else:
            _write_retry_contract_no_clobber(retry_contract_path, retry_contract)
    derived = dict(verified)
    derived.update(
        {
            "claim_identity_sha256": retry_identity,
            "retry_contract": retry_contract,
            "retry_contract_path": retry_contract_path,
            "retry_contract_file_sha256": (
                _file_sha256(retry_contract_path) if publish else None
            ),
        }
    )
    return derived


def _write_receipt_no_clobber(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    _mkdir_durable(path.parent)
    payload = _with_digest(payload, "receipt_sha256")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("xb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o444)
        os.link(tmp, path)
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise ExecutorError(f"refusing to overwrite A1 receipt: {path}") from error
    finally:
        tmp.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    return payload


def _central_live_allocation(verified: dict[str, Any]) -> dict[str, Any] | None:
    """Return the exact centrally commissioned allocation, if this is a central run."""

    central = verified.get("central_learner_binding")
    if not isinstance(central, dict):
        return None
    stage = central.get("stage")
    if stage == "P1":
        authority = verified.get("p1_arm_executor_authority")
        allocation = (
            authority.get("allocation") if isinstance(authority, dict) else None
        )
    elif stage in {AUX_CONTROL_ARM, AUX_TREATMENT_ARM}:
        authority = verified.get("aux_pair_executor_authority")
        claim = authority.get("arm_claim") if isinstance(authority, dict) else None
        allocation = claim.get("allocation") if isinstance(claim, dict) else None
    elif stage == "FINAL":
        authority = verified.get("final_replication_executor_authority")
        allocation = (
            authority.get("allocation") if isinstance(authority, dict) else None
        )
    else:
        raise ExecutorError(f"unsupported central learner stage {stage!r}")
    if not isinstance(allocation, dict):
        raise ExecutorError("central learner authority has no exact allocation")
    try:
        return aux_coordinator.verify_allocation(allocation)
    except aux_coordinator.CoordinatorError as error:
        raise ExecutorError(f"central learner allocation refused: {error}") from error


def _verify_central_live_allocation(
    verified: dict[str, Any],
    *,
    selected_gpus: Sequence[int],
    runtime_probe: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Replay exact host/machine/GPU UUID/PCI identity immediately pre-claim."""

    allocation = _central_live_allocation(verified)
    if allocation is None:
        return None
    if list(selected_gpus) != allocation["physical_gpu_indices"]:
        raise ExecutorError("selected GPUs differ from central allocation")
    probe = runtime_probe or scientific_evidence._local_runtime_report  # noqa: SLF001
    try:
        report = probe(_REPO_ROOT)
        aux_coordinator._verify_allocation_matches_native_report(  # noqa: SLF001
            allocation, report
        )
    except (
        OSError,
        ValueError,
        aux_coordinator.CoordinatorError,
        scientific_evidence.EvidenceError,
    ) as error:
        raise ExecutorError(
            f"live central B200 allocation replay failed: {error}"
        ) from error
    return report


def _exact_objective_exposure(report_payload: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct exact current-dose objective exposure from trainer counters."""

    integer_fields = (
        "base_training_row_draws",
        "policy_aux_training_row_draws",
        "policy_base_active_rows",
        "policy_aux_active_rows",
        "policy_total_active_rows",
        "value_active_rows",
        "policy_kl_anchor_eligible_rows",
    )
    values: dict[str, int] = {}
    for field in integer_fields:
        value = report_payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExecutorError(
                f"A1 training report has invalid exact-dose counter {field!r}"
            )
        values[field] = value
    if (
        values["policy_aux_training_row_draws"] != values["policy_aux_active_rows"]
        or values["policy_total_active_rows"]
        != values["policy_base_active_rows"] + values["policy_aux_active_rows"]
        or report_payload.get("total_training_row_draws")
        != values["base_training_row_draws"] + values["policy_aux_training_row_draws"]
    ):
        raise ExecutorError("A1 training report objective-dose arithmetic drift")
    if values["value_active_rows"] > values["base_training_row_draws"]:
        raise ExecutorError("A1 exact value-active dose exceeds the base draw dose")
    if values["policy_kl_anchor_eligible_rows"] > values["base_training_row_draws"]:
        raise ExecutorError("A1 exact anchor-eligible dose exceeds the base draw dose")
    return {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": values["base_training_row_draws"],
        "policy_base_active_sampled_rows": values["policy_base_active_rows"],
        "policy_aux_active_sampled_rows": values["policy_aux_active_rows"],
        "policy_active_sampled_rows": values["policy_total_active_rows"],
        "value_active_sampled_rows": values["value_active_rows"],
        "anchor_eligible_sampled_rows": values["policy_kl_anchor_eligible_rows"],
    }


def _effective_global_batch_size(recipe: dict[str, Any]) -> int:
    realized = (
        int(recipe["batch_size"])
        * int(recipe["grad_accum_steps"])
        * int(recipe["world_size"])
    )
    if realized <= 0 or realized != int(recipe["global_batch_size"]):
        raise ExecutorError(
            "learner topology does not realize its declared global batch size"
        )
    return realized


def _expected_optimizer_steps(
    verified: dict[str, Any], *, recipe: dict[str, Any] | None = None
) -> int:
    effective = verified["recipe"] if recipe is None else recipe
    steps = math.ceil(
        int(verified["training_row_count"]) / _effective_global_batch_size(effective)
    )
    if int(effective["max_steps"]) > 0:
        steps = min(steps, int(effective["max_steps"]))
    return int(steps)


def _learner_lineage_parent_sha256(verified: Mapping[str, Any]) -> str:
    """Return the authenticated learner parent without changing corpus identity.

    Ordinary one-dose training starts from the sealed corpus producer.  The
    Diagnostic exceptions are accepted only when the explicit parent record,
    comparison-source authority, and typed function-preserving upgrade all
    name the same checkpoint.  Corpus provenance remains a separate identity.
    """

    producer = verified.get("producer")
    if not isinstance(producer, Mapping):
        raise ExecutorError("one-dose learner has no sealed corpus producer")
    parent = verified.get("learner_lineage_parent")
    diagnostic = verified.get("diagnostic_comparison_source")
    if parent is None:
        if diagnostic is not None:
            raise ExecutorError(
                "diagnostic comparison source lacks an explicit learner lineage parent"
            )
        return str(producer["sha256"])
    upgrade = verified.get("function_preserving_upgrade_lineage")
    checkpoint = parent.get("checkpoint") if isinstance(parent, Mapping) else None
    if (
        isinstance(parent, Mapping)
        and parent.get("role") == "diagnostic_independent_parent"
    ):
        authority = verified.get("independent_parent_authority")
        expected_independent = {
            "schema_version": LEARNER_LINEAGE_PARENT_SCHEMA,
            "role": "diagnostic_independent_parent",
            "checkpoint": copy.deepcopy(
                diagnostic.get("source") if isinstance(diagnostic, Mapping) else None
            ),
            "corpus_producer": copy.deepcopy(dict(producer)),
            "independent_parent_authority_sha256": (
                authority.get("authority_sha256")
                if isinstance(authority, Mapping)
                else None
            ),
            "function_preserving_upgrade_receipt_sha256": (
                upgrade.get("receipt_sha256") if isinstance(upgrade, Mapping) else None
            ),
            "diagnostic_only": True,
            "promotion_eligible": False,
        }
        if (
            not isinstance(diagnostic, Mapping)
            or diagnostic.get("role") != "independent_parent"
            or diagnostic.get("diagnostic_only") is not True
            or diagnostic.get("promotion_eligible") is not False
            or not isinstance(authority, Mapping)
            or authority.get("schema_version") != INDEPENDENT_PARENT_AUTHORITY_SCHEMA
            or not isinstance(upgrade, Mapping)
            or not isinstance(checkpoint, Mapping)
            or dict(parent) != expected_independent
            or checkpoint.get("sha256") != upgrade.get("source_checkpoint_sha256")
        ):
            raise ExecutorError(
                "diagnostic learner lineage parent lost independent-parent authority"
            )
        return str(checkpoint["sha256"])
    expected = {
        "schema_version": LEARNER_LINEAGE_PARENT_SCHEMA,
        "role": "diagnostic_recent_history",
        "checkpoint": copy.deepcopy(
            diagnostic.get("source") if isinstance(diagnostic, Mapping) else None
        ),
        "corpus_producer": copy.deepcopy(dict(producer)),
        "category_semantics_sha256": verified.get("category_semantics_sha256"),
        "function_preserving_upgrade_receipt_sha256": (
            upgrade.get("receipt_sha256") if isinstance(upgrade, Mapping) else None
        ),
        "diagnostic_only": True,
        "promotion_eligible": False,
    }
    if (
        verified.get("data_kind") != "production_composite_v2"
        or not isinstance(diagnostic, Mapping)
        or diagnostic.get("role") != "recent_history"
        or diagnostic.get("diagnostic_only") is not True
        or diagnostic.get("promotion_eligible") is not False
        or not isinstance(upgrade, Mapping)
        or not isinstance(checkpoint, Mapping)
        or dict(parent) != expected
        or checkpoint.get("sha256") != upgrade.get("source_checkpoint_sha256")
    ):
        raise ExecutorError(
            "diagnostic learner lineage parent lost exact recent-history authority"
        )
    return str(checkpoint["sha256"])


def _direct_lineage_dose(
    verified: dict[str, Any],
    *,
    report_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recipe = verified["recipe"]
    if recipe.get("resume_optimizer") is not False:
        raise ExecutorError(
            "canonical one-dose lineage requires a fresh optimizer per dose"
        )
    steps = _expected_optimizer_steps(verified, recipe=recipe)
    objective_exposure = None
    global_batch_size = _effective_global_batch_size(recipe)
    sampled_rows = (
        min(int(verified["training_row_count"]), steps * global_batch_size)
        if int(recipe["world_size"]) == 1
        else steps * global_batch_size
    )
    # Historical sealed reports predate exact objective counters.  Current
    # trainers emit the full counter set even when the policy-aux sampler is
    # disabled; bind that evidence whenever any of it is present and refuse a
    # partial counter surface.  Gating this on a recipe flag silently discarded
    # the ordinary production learner's exact 65k-policy/524k-value exposure.
    objective_counter_fields = {
        "base_training_row_draws",
        "policy_aux_training_row_draws",
        "policy_base_active_rows",
        "policy_aux_active_rows",
        "policy_total_active_rows",
        "value_active_rows",
        "policy_kl_anchor_eligible_rows",
        "total_training_row_draws",
    }
    present_objective_counter_fields = (
        set(report_payload).intersection(objective_counter_fields)
        if report_payload is not None
        else set()
    )
    exact_objective_specific_fields = objective_counter_fields - {
        "base_training_row_draws",
        "total_training_row_draws",
    }
    if present_objective_counter_fields.intersection(exact_objective_specific_fields):
        if present_objective_counter_fields != objective_counter_fields:
            missing = sorted(
                objective_counter_fields - present_objective_counter_fields
            )
            raise ExecutorError(
                "A1 training report has a partial exact objective-dose surface: "
                f"missing={missing}"
            )
        assert report_payload is not None
        objective_exposure = _exact_objective_exposure(report_payload)
        if int(objective_exposure["base_sampled_rows"]) != sampled_rows:
            raise ExecutorError(
                "A1 objective counters differ from the topology-derived base dose"
            )
        sampled_rows = int(objective_exposure["base_sampled_rows"])
    elif report_payload is not None:
        reported_draws = report_payload.get("base_training_row_draws")
        if reported_draws is not None:
            if (
                isinstance(reported_draws, bool)
                or not isinstance(reported_draws, int)
                or reported_draws != sampled_rows
            ):
                raise ExecutorError(
                    "A1 base sampler dose differs from the topology-derived dose: "
                    f"expected={sampled_rows} actual={reported_draws!r}"
                )
            sampled_rows = reported_draws
    try:
        return lineage.direct_lineage_dose(
            declared_producer_sha256=_learner_lineage_parent_sha256(verified),
            init_checkpoint_sha256=verified.get(
                "architecture_initializer", verified["producer"]
            )["sha256"],
            function_preserving_upgrade=verified.get(
                "function_preserving_upgrade_lineage"
            )
            if verified.get("initializer_transition_chain") is None
            else None,
            initializer_transition_chain=verified.get("initializer_transition_chain"),
            current_sampled_rows=sampled_rows,
            current_optimizer_steps=steps,
            objective_exposure=objective_exposure,
        )
    except lineage.LineageDoseError as error:
        raise ExecutorError(f"invalid one-dose learner lineage: {error}") from error


def _bind_training_report(
    report: Path,
    *,
    verified: dict[str, Any],
    execution_binding: dict[str, Any],
) -> None:
    """Atomically bind the trainer report to the exact executor environment.

    ``train_bc`` is part of the immutable A1 learner runtime inventory and must
    not be edited after the wave.  The transaction executor is deliberately not
    in that inventory, so it adds this operational binding after the child exits
    and before the report is verified, hashed, or receipted.
    """

    if report.is_symlink() or not report.is_file():
        raise ExecutorError(f"A1 training report is not a regular file: {report}")
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot parse A1 training report: {error}") from error
    if not isinstance(payload, dict):
        raise ExecutorError("A1 training report must be a JSON object")
    if (
        REPORT_EXECUTION_BINDING_FIELD in payload
        or REPORT_INPUT_BINDING_FIELD in payload
        or "a1_lineage_dose" in payload
        or REPORT_LEARNER_LINEAGE_PARENT_FIELD in payload
    ):
        raise ExecutorError("A1 training child pre-populated executor-owned provenance")
    _validate_execution_binding(execution_binding)
    payload[REPORT_EXECUTION_BINDING_FIELD] = execution_binding
    payload[REPORT_INPUT_BINDING_FIELD] = _input_binding(verified)
    payload[REPORT_LEARNER_LINEAGE_PARENT_FIELD] = copy.deepcopy(
        verified.get("learner_lineage_parent")
    )
    payload["a1_lineage_dose"] = _direct_lineage_dose(verified, report_payload=payload)
    learner_ablation = verified.get("learner_ablation")
    central_binding = verified.get("central_learner_binding")
    if isinstance(central_binding, dict):
        published = verified.get("central_published_executor_authority")
        realized = payload.get("a1_realized_central_sample_order")
        sample_binding = central_binding["sample_binding"]
        realized_fields = {
            "sample_dose": "sample_dose",
            "sample_order_sha256": "sample_order_sha256",
            "row_set_sha256": "row_set_sha256",
            "unique_row_count": "unique_row_count",
            "kl_eligible_rows": "kl_eligible_rows",
            "kl_eligible_mass_decimal": "kl_eligible_mass_decimal",
            "kl_ordered_evidence_sha256": "kl_ordered_evidence_sha256",
            "kl_eligible_evidence_sha256": "kl_eligible_evidence_sha256",
        }
        if (
            payload.get("a1_central_learner_binding") != central_binding
            or payload.get("a1_central_published_executor_authority") != published
            or not isinstance(realized, dict)
            or realized.get("schema_version") != "a1-realized-central-sample-order-v1"
            or realized.get("physical_row_identity") is not True
            or realized.get("validation_rows_excluded") is not True
            or any(
                realized.get(field) != sample_binding.get(binding_field)
                for field, binding_field in realized_fields.items()
            )
            or payload.get("a1_effective_learner_training_recipe")
            != central_binding["effective_recipe"]
            or payload.get("a1_effective_learner_training_recipe_sha256")
            != central_binding["effective_recipe_sha256"]
            or payload.get("diagnostic_only") is not central_binding["diagnostic_only"]
            or payload.get("promotion_eligible")
            is not central_binding["promotion_eligible"]
            or payload.get("eligible_for_full_gate")
            is not central_binding["eligible_for_full_gate"]
        ):
            raise ExecutorError(
                "central learner child did not prove its coordinator authority and "
                "realized physical sample order"
            )
        matched_aux = (learner_ablation or {}).get("matched_aux_regularization")
        if (
            matched_aux is not None
            and payload.get("a1_aux_regularization_binding") != matched_aux
        ):
            raise ExecutorError("central AUX child did not echo pointer-arm authority")
        if central_binding.get("stage") == "P1":
            descriptor_authority = verified.get("p1_training_descriptor_authority")
            expected_anchor_rows = (
                descriptor_authority.get("expected_policy_kl_anchor_eligible_rows")
                if isinstance(descriptor_authority, dict)
                else None
            )
            if payload.get("policy_kl_anchor_eligible_rows") != expected_anchor_rows:
                raise ExecutorError(
                    "P1 trainer-visible KL-anchor rows differ from issued descriptor: "
                    f"expected={expected_anchor_rows!r} "
                    f"actual={payload.get('policy_kl_anchor_eligible_rows')!r}"
                )
        payload["a1_realized_central_sample_evidence_sha256"] = _value_sha256(realized)
    elif (
        learner_ablation is not None
        and verified.get("data_kind") == "production_composite_v2"
    ):
        # Promotion composites intentionally carry no historical A1 sentinel,
        # so train_bc cannot reconstruct the outer lock's diagnostic ablation
        # object.  The one-dose executor owns that authenticated lock and adds
        # its exact binding transactionally after requiring the child to echo
        # the matched aux receipt/initializer authority it did validate.
        matched_aux = learner_ablation.get("matched_aux_regularization")
        if (
            (
                matched_aux is not None
                and payload.get("a1_aux_regularization_binding") != matched_aux
            )
            or (
                matched_aux is None
                and payload.get("a1_aux_regularization_binding") is not None
            )
            or payload.get("a1_learner_ablation") is not None
            or payload.get("a1_effective_learner_training_recipe") is not None
            or not isinstance(payload.get("value_training"), dict)
            or "learner_ablation" in payload["value_training"]
        ):
            raise ExecutorError(
                "production aux child did not echo the exact matched-arm authority"
            )
        payload["a1_effective_learner_training_recipe"] = verified["recipe"]
        payload["a1_effective_learner_training_recipe_sha256"] = _value_sha256(
            verified["recipe"]
        )
        payload["a1_learner_ablation"] = learner_ablation
        payload["value_training"]["learner_ablation"] = learner_ablation
        descriptor_authority = verified.get("diagnostic_training_descriptor_authority")
        if descriptor_authority is not None:
            payload["a1_diagnostic_training_descriptor_authority"] = (
                descriptor_authority
            )
            payload["value_training"]["diagnostic_training_descriptor_authority"] = (
                descriptor_authority
            )
        payload["diagnostic_only"] = True
        payload["promotion_eligible"] = False
    tmp = report.with_name(f".{report.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("xb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, report)
        _fsync_directory(report.parent)
    finally:
        tmp.unlink(missing_ok=True)
        _fsync_directory(report.parent)


def _require_finite_metric_tree(value: Any, *, where: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ExecutorError(f"{where} contains a non-finite metric")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_metric_tree(child, where=f"{where}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_metric_tree(child, where=f"{where}[{index}]")


def _verify_matched_aux_torch_artifacts(
    checkpoint: Path,
    optimizer: Path,
    *,
    expected_steps: int,
    recipe: dict[str, Any],
) -> None:
    """Prove matched AUX outputs are loadable pointer-model/fresh-Adam state."""

    try:
        import torch

        checkpoint_blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        optimizer_blob = torch.load(optimizer, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ExecutorError(
            f"cannot load matched AUX torch outputs: {error}"
        ) from error
    if not isinstance(checkpoint_blob, dict):
        raise ExecutorError("matched AUX checkpoint root is not a mapping")
    model_state = checkpoint_blob.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise ExecutorError("matched AUX checkpoint has no model state")
    config = checkpoint_blob.get("config")
    try:
        config_fields = architecture_upgrade._config(config)  # noqa: SLF001
    except architecture_upgrade.UpgradeError as error:
        raise ExecutorError(
            f"matched AUX checkpoint config is invalid: {error}"
        ) from error
    if (
        checkpoint_blob.get("policy_type") != "entity_graph"
        or config_fields.get("aux_subgoal_heads") is not True
        or config_fields.get("aux_settlement_pointer_head") is not True
        or any(name.startswith("aux_next_settlement_head.") for name in model_state)
        or not all(
            any(name.startswith(prefix + ".") for name in model_state)
            for prefix in AUX_WARMUP_TRAINABLE_PREFIXES
        )
    ):
        raise ExecutorError(
            "matched AUX checkpoint is not the corrected pointer architecture"
        )
    static = checkpoint_blob.get("static_action_features")
    if static is None:
        raise ExecutorError("matched AUX checkpoint lacks static action features")
    if hasattr(static, "detach"):
        static = static.detach().cpu().numpy()
    static_array = np.asarray(static, dtype=np.float32)
    if (
        checkpoint_blob.get("static_action_features_sha256")
        != train_bc._array_sha256(static_array)  # noqa: SLF001
    ):
        raise ExecutorError("matched AUX static action-feature bytes drift")
    try:
        from catan_zero.rl.config_serialization import config_from_dict
        from catan_zero.rl.entity_token_policy import (
            EntityGraphConfig,
            EntityGraphPolicy,
        )

        strict_config = config_from_dict(EntityGraphConfig, config)
        reconstructed = EntityGraphPolicy(
            strict_config, static_array, seed=0, device="cpu"
        )
        reconstructed.model.load_state_dict(model_state, strict=True)
    except Exception as error:
        raise ExecutorError(
            f"matched AUX checkpoint cannot strict-load as the declared model: {error}"
        ) from error
    if (
        not isinstance(optimizer_blob, dict)
        or optimizer_blob.get("format") != "plain"
        or not isinstance(optimizer_blob.get("optimizer"), dict)
    ):
        raise ExecutorError("matched AUX optimizer sidecar is not plain DDP Adam")
    optimizer_state = optimizer_blob["optimizer"]
    state = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    if (
        not isinstance(state, dict)
        or not state
        or not isinstance(groups, list)
        or not groups
    ):
        raise ExecutorError("matched AUX optimizer state is structurally empty")
    # Rebuild the exact arm-specific trainable surface.  AUX0 excludes all five
    # optional readouts from Adam; AUXT includes them.  Reconstructing the
    # optimizer's parameter groups makes a plausible-looking unrelated Adam
    # pickle insufficient evidence.
    try:
        freeze = train_bc._freeze_inactive_training_heads(  # noqa: SLF001
            reconstructed.model,
            final_vp_loss_weight=float(recipe.get("final_vp_loss_weight", 0.0)),
            value_uncertainty_loss_weight=float(
                recipe.get("value_uncertainty_loss_weight", 0.0)
            ),
            value_categorical_loss_weight=float(
                recipe.get("value_categorical_loss_weight", 0.0)
            ),
            aux_subgoal_loss_weight=float(recipe["aux_subgoal_loss_weight"]),
            belief_resource_loss_weight=float(
                recipe.get("belief_resource_loss_weight", 0.0)
            ),
        )
        expected_params = train_bc._build_optimizer_param_groups(  # noqa: SLF001
            reconstructed.model,
            base_lr=float(recipe["lr"]),
            value_lr_mult=float(recipe["value_lr_mult"]),
            action_module_lr_mult=float(recipe.get("action_module_lr_mult", 1.0)),
            public_card_lr_mult=float(recipe.get("public_card_lr_mult", 1.0)),
            trunk_lr_mult=float(recipe.get("trunk_lr_mult", 1.0)),
            architecture="entity_graph",
        )
        expected_optimizer = torch.optim.Adam(expected_params, lr=float(recipe["lr"]))
        expected_groups = expected_optimizer.state_dict()["param_groups"]
    except Exception as error:
        raise ExecutorError(
            f"cannot reconstruct matched AUX optimizer surface: {error}"
        ) from error
    expected_active = (
        set(AUX_SUBGOAL_HEAD_MODULES)
        if float(recipe["aux_subgoal_loss_weight"]) > 0.0
        else set()
    )
    if (
        set(freeze.get("active_optional_submodules", []))
        & set(AUX_SUBGOAL_HEAD_MODULES)
        != expected_active
    ):
        raise ExecutorError("matched AUX reconstructed optional-head surface drift")
    if len(groups) != len(expected_groups):
        raise ExecutorError("matched AUX Adam parameter-group count drift")
    expected_param_ids: set[int] = set()
    actual_param_ids: set[int] = set()
    expected_parameter_by_id: dict[int, Any] = {}
    exact_group_fields = (
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "capturable",
        "differentiable",
    )
    for actual, expected in zip(groups, expected_groups, strict=True):
        actual_params = actual.get("params")
        expected_params_ids = expected.get("params")
        if (
            not isinstance(actual_params, list)
            or not isinstance(expected_params_ids, list)
            or len(actual_params) != len(expected_params_ids)
            or any(
                actual.get(field) != expected.get(field) for field in exact_group_fields
            )
        ):
            raise ExecutorError("matched AUX Adam parameter group/hyperparameter drift")
        actual_param_ids.update(int(value) for value in actual_params)
        expected_param_ids.update(int(value) for value in expected_params_ids)
    for live_group, serialized_group in zip(
        expected_optimizer.param_groups, expected_groups, strict=True
    ):
        for parameter, parameter_id in zip(
            live_group["params"], serialized_group["params"], strict=True
        ):
            expected_parameter_by_id[int(parameter_id)] = parameter
    if (
        len(actual_param_ids) != sum(len(group["params"]) for group in groups)
        or actual_param_ids != expected_param_ids
        or set(state) != expected_param_ids
    ):
        raise ExecutorError("matched AUX Adam state does not match trainable surface")
    observed_steps: list[int] = []
    for parameter_id, entry in state.items():
        if not isinstance(entry, dict) or "step" not in entry:
            raise ExecutorError("matched AUX Adam state lacks a step counter")
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ExecutorError("matched AUX Adam state field set drift")
        parameter = expected_parameter_by_id.get(int(parameter_id))
        exp_avg = entry.get("exp_avg")
        exp_avg_sq = entry.get("exp_avg_sq")
        if (
            parameter is None
            or not torch.is_tensor(exp_avg)
            or not torch.is_tensor(exp_avg_sq)
            or tuple(exp_avg.shape) != tuple(parameter.shape)
            or tuple(exp_avg_sq.shape) != tuple(parameter.shape)
            or exp_avg.dtype != parameter.dtype
            or exp_avg_sq.dtype != parameter.dtype
            or exp_avg.device.type != "cpu"
            or exp_avg_sq.device.type != "cpu"
            or not bool(torch.isfinite(exp_avg).all())
            or not bool(torch.isfinite(exp_avg_sq).all())
            or bool((exp_avg_sq < 0).any())
        ):
            raise ExecutorError(
                "matched AUX Adam moments do not match the reconstructed parameter"
            )
        raw_step = entry["step"]
        if hasattr(raw_step, "item"):
            raw_step = raw_step.item()
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
            raise ExecutorError("matched AUX Adam step counter is not numeric")
        numeric_step = float(raw_step)
        if not numeric_step.is_integer():
            raise ExecutorError("matched AUX Adam step counter is fractional")
        observed_steps.append(int(numeric_step))
    if set(observed_steps) != {int(expected_steps)}:
        raise ExecutorError(
            "matched AUX Adam state does not prove the exact fresh-optimizer dose"
        )
    try:
        expected_optimizer.load_state_dict(optimizer_state)
    except Exception as error:
        raise ExecutorError(
            f"matched AUX Adam state cannot attach to reconstructed model: {error}"
        ) from error


def _strict_load_production_entity_checkpoint(checkpoint: Path, *, where: str) -> None:
    """Reconstruct an ordinary production checkpoint with an exact state load.

    ``torch.load`` proves only that pickle deserialization succeeds.  The
    completion boundary must also prove that the declared architecture and all
    enabled parameter tensors form a loadable inference model.  This catches
    missing, unexpected, or shape-drifted state before a terminal claim can be
    published.
    """

    try:
        from catan_zero.rl.entity_token_policy import EntityGraphPolicy

        policy = EntityGraphPolicy.load(
            checkpoint,
            device="cpu",
            strict_metadata=True,
            allow_missing_optional_parameters=False,
        )
    except Exception as error:
        raise ExecutorError(
            f"{where} cannot strict-load as the declared entity model: {error}"
        ) from error
    parameter_count = sum(parameter.numel() for parameter in policy.model.parameters())
    if not 30_000_000 <= parameter_count <= 40_000_000:
        raise ExecutorError(
            f"{where} strict-loaded an unexpected parameter count: {parameter_count}"
        )


def _require_production_event_history_surface(
    surface: Any,
    *,
    expected_contract: Mapping[str, Any],
    row_count: int,
    where: str,
) -> None:
    """Require the authenticated production history surface in an artifact."""

    if not isinstance(surface, dict) or any(
        surface.get(key) != value for key, value in expected_contract.items()
    ):
        raise ExecutorError(f"{where} event-history contract drifted")
    if expected_contract.get("training_event_history_trainable") is True:
        native = expected_contract.get("native_inference")
        expected_width = (
            native.get("history_limit") if isinstance(native, dict) else None
        )
        if (
            not isinstance(expected_width, int)
            or expected_width < 1
            or surface.get("training_event_tensor_width") != expected_width
            or "empty_event_mask_scan" in surface
            or "event_encoder_freeze" in surface
        ):
            raise ExecutorError(
                f"{where} did not preserve the authenticated meaningful-history "
                "axis and trainable event encoder"
            )
        return

    scan = surface.get("empty_event_mask_scan")
    freeze = surface.get("event_encoder_freeze")
    if (
        surface.get("training_event_tensor_width") != 0
        or not isinstance(scan, dict)
        or scan.get("schema") != "training-empty-event-mask-scan-v1"
        or scan.get("row_count") != row_count
        or scan.get("nonzero_event_mask_count") != 0
        or not isinstance(scan.get("scan_sha256"), str)
        or not isinstance(freeze, dict)
        or freeze.get("reason")
        != "authenticated empty event axis is cropped to width zero"
        or not isinstance(freeze.get("frozen_parameter_names"), list)
        or not freeze["frozen_parameter_names"]
        or freeze.get("frozen_parameter_tensors")
        != len(freeze["frozen_parameter_names"])
        or freeze.get("unexpected_frozen_parameter_tensors") != 0
        or freeze.get("optimizer_excluded_parameter_tensors")
        != freeze.get("frozen_parameter_tensors")
        or freeze.get("optimizer_excluded_parameters")
        != freeze.get("frozen_parameters")
    ):
        raise ExecutorError(
            f"{where} did not prove the empty event axis was scanned, cropped, "
            "and excluded from Adam"
        )


def _verify_production_validation_seed_manifest(
    path: Path,
    *,
    report_payload: Mapping[str, Any],
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the emitted calibration cohort to the authenticated train split."""

    manifest_path, file_sha256 = _stable_canonical_regular_file(
        path, where="production validation-seed manifest"
    )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(
            f"cannot parse production validation-seed manifest: {error}"
        ) from error
    expected_keys = {
        "schema_version",
        "data",
        "data_fingerprint",
        "validation_fraction",
        "validation_seed",
        "validation_max_samples",
        "validation_game_seed_ranges",
        "validation_game_seed_count",
        "validation_game_seed_set_sha256",
        "training_excluded_game_seed_count",
        "training_excluded_game_seed_set_sha256",
        "game_seeds",
    }
    seeds = payload.get("game_seeds") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema_version") != "train-validation-game-seeds-v1"
        or payload.get("data") != str(verified["data_path"])
        or payload.get("data_fingerprint") != verified["data_fingerprint"]
        or payload.get("validation_fraction") != 0.05
        or payload.get("validation_seed") != 17
        or payload.get("validation_max_samples") != 0
        or payload.get("validation_game_seed_ranges") != []
        or not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ExecutorError("production validation-seed manifest schema drifted")
    try:
        seed_array = np.asarray(seeds, dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ExecutorError(
            "production validation-seed manifest contains a non-int64 seed"
        ) from error
    expected_count = int(
        verified["validation_split_receipt"]["aggregate"]["validation_game_count"]
    )
    expected_digest = str(verified["trainer_validation_game_seed_set_sha256"])
    actual_digest = train_bc._game_seed_set_sha256(seed_array)  # noqa: SLF001
    if (
        not np.all(seed_array[1:] > seed_array[:-1])
        or len(seed_array) != expected_count
        or payload.get("validation_game_seed_count") != expected_count
        or payload.get("training_excluded_game_seed_count") != expected_count
        or actual_digest != expected_digest
        or payload.get("validation_game_seed_set_sha256") != expected_digest
        or payload.get("training_excluded_game_seed_set_sha256") != expected_digest
        or report_payload.get("validation_game_seed_manifest") != str(manifest_path)
        or report_payload.get("validation_game_seed_count") != expected_count
        or report_payload.get("validation_game_seed_set_sha256") != expected_digest
    ):
        raise ExecutorError(
            "production validation-seed manifest differs from the authenticated split"
        )
    return {
        "path": str(manifest_path),
        "file_sha256": file_sha256,
        "game_seed_count": expected_count,
        "game_seed_set_sha256": expected_digest,
    }


def _require_production_public_award_transition(
    award_training: Any,
    *,
    verified: Mapping[str, Any],
    where: str,
) -> None:
    """Replay the exact 64/12/4/20 legacy-to-authoritative transition."""

    transition = (
        award_training.get("mixed_authoritative_transition")
        if isinstance(award_training, dict)
        else None
    )
    if not isinstance(transition, dict):
        raise ExecutorError(f"{where} lacks a mixed public-award transition")
    unhashed = dict(transition)
    stated = unhashed.pop("transition_sha256", None)
    expected_ids = [
        "current_producer",
        "recent_history",
        "hard_negative",
        "historical_replay",
    ]
    expected_ratios = [0.64, 0.12, 0.04, 0.20]
    expected_contracts = [
        "authoritative_v1",
        "authoritative_v1",
        "authoritative_v1",
        "legacy_zero_v0",
    ]
    split_components = verified["validation_split_receipt"]["components"]
    expected_rows = [int(record["row_count"]) for record in split_components]
    audits = transition.get("component_audits")
    if (
        stated != _value_sha256(unhashed)
        or transition.get("schema_version") != "mixed-authoritative-transition-v1"
        or transition.get("routing_authority")
        != "authenticated_component_identity_not_feature_values"
        or transition.get("component_ids") != expected_ids
        or transition.get("component_sampling_ratios") != expected_ratios
        or transition.get("component_contracts") != expected_contracts
        or transition.get("corrected_corpus_rows") != sum(expected_rows[:3])
        or transition.get("legacy_corpus_rows") != expected_rows[3]
        or transition.get("corrected_sampler_mass") != 0.8
        or transition.get("legacy_sampler_mass") != 0.2
        or transition.get("legacy_rows_zero_slot12") is not True
        or transition.get("corrected_rows_pass_slot12") is not True
        or transition.get("checkpoint_contract") != "authoritative_v1"
        or not isinstance(audits, list)
        or len(audits) != 4
    ):
        raise ExecutorError(f"{where} public-award transition contract drifted")
    for index, (audit, component_id, contract, row_count) in enumerate(
        zip(audits, expected_ids, expected_contracts, expected_rows, strict=True)
    ):
        if not isinstance(audit, dict):
            raise ExecutorError(f"{where} public-award audit {index} is malformed")
        audit_unhashed = dict(audit)
        audit_digest = audit_unhashed.pop("audit_sha256", None)
        if (
            audit_digest != _value_sha256(audit_unhashed)
            or audit.get("component_id") != component_id
            or audit.get("contract") != contract
            or audit.get("rows") != row_count
            or (
                contract == "legacy_zero_v0"
                and (
                    audit.get("legacy_slot12_all_zero") is not True
                    or audit.get("nonzero_award_values") != 0
                )
            )
            or (
                contract == "authoritative_v1"
                and (
                    audit.get("corrected_positive_support") is not True
                    or not isinstance(audit.get("rows_with_award"), int)
                    or int(audit["rows_with_award"]) <= 0
                )
            )
        ):
            raise ExecutorError(
                f"{where} public-award component audit {component_id} drifted"
            )
    if (
        not isinstance(award_training, dict)
        or award_training.get("requested_contract") != "authoritative_v1"
        or award_training.get("initializer_contract") != "legacy_zero_v0"
        or award_training.get("effective_contract") != "authoritative_v1"
        or award_training.get("mixed_corpus_acknowledged") is not True
        or award_training.get("legacy_column_zero_initialized") is not True
        or award_training.get("diagnostic_only") is not False
        or award_training.get("promotion_eligible") is not True
    ):
        raise ExecutorError(f"{where} public-award training state drifted")


def _verify_production_checkout_runtime_binding(
    binding: Any, *, trainer_authority: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay every imported learner module recorded by the child trainer."""

    expected_keys = {
        "schema_version",
        "repo_root",
        "source_root",
        "trainer",
        "trainer_sha256",
        "modules",
        "binding_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != expected_keys:
        raise ExecutorError("production checkout runtime binding schema drifted")
    unhashed = dict(binding)
    stated = unhashed.pop("binding_sha256")
    repo_root = Path(str(binding["repo_root"]))
    source_root = Path(str(binding["source_root"]))
    trainer = Path(str(binding["trainer"]))
    if (
        binding.get("schema_version") != train_bc.CHECKOUT_RUNTIME_BINDING_SCHEMA
        or stated != _value_sha256(unhashed)
        or repo_root != Path(str(trainer_authority["repository_root"]))
        or source_root != repo_root / "src"
        or trainer != Path(str(trainer_authority["path"]))
        or binding.get("trainer_sha256") != trainer_authority.get("sha256")
    ):
        raise ExecutorError("production checkout runtime root/trainer drifted")
    modules = binding.get("modules")
    authority_surface = trainer_authority.get("code_surface")
    if not isinstance(authority_surface, list):
        raise ExecutorError("production trainer authority lacks its code surface")
    authority_by_relative_path: dict[str, str] = {}
    for record in authority_surface:
        if (
            not isinstance(record, dict)
            or set(record) != {"relative_path", "path", "sha256"}
            or not isinstance(record.get("relative_path"), str)
            or not isinstance(record.get("sha256"), str)
        ):
            raise ExecutorError("production trainer authority code-surface drifted")
        authority_by_relative_path[str(record["relative_path"])] = str(record["sha256"])
    required = {
        "catan_zero",
        "catan_zero.rl.optim_state",
        "catan_zero.rl.entity_token_policy",
    }
    if not isinstance(modules, dict) or not required.issubset(modules):
        raise ExecutorError(
            "production checkout runtime lacks required learner modules"
        )
    for module_name, record in modules.items():
        if (
            not isinstance(module_name, str)
            or not isinstance(record, dict)
            or set(record) != {"path", "sha256"}
        ):
            raise ExecutorError("production checkout runtime module schema drifted")
        try:
            module_path = Path(str(record["path"])).resolve(strict=True)
            module_path.relative_to(source_root.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise ExecutorError(
                f"production runtime module escaped current source root: {module_name}"
            ) from error
        try:
            relative_path = module_path.relative_to(repo_root.resolve(strict=True))
        except ValueError as error:
            raise ExecutorError(
                f"production runtime module escaped trainer repository: {module_name}"
            ) from error
        expected_sha256 = authority_by_relative_path.get(relative_path.as_posix())
        if (
            expected_sha256 is None
            or str(module_path) != record["path"]
            or _file_sha256(module_path) != record["sha256"]
            or record["sha256"] != expected_sha256
        ):
            raise ExecutorError(
                f"production runtime module was not pre-bound or its bytes drifted: "
                f"{module_name}"
            )
    return copy.deepcopy(binding)


def _verify_training_outputs(
    *,
    checkpoint: Path,
    report: Path,
    verified: dict[str, Any],
    execution_binding: dict[str, Any],
    command: list[str] | None = None,
) -> dict[str, Any]:
    _validate_execution_binding(execution_binding)
    optimizer = Path(str(checkpoint) + ".optimizer.pt")
    progress = Path(str(checkpoint) + ".training-progress.json")
    for path in (checkpoint, optimizer, progress, report):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ExecutorError(f"A1 training output is missing or empty: {path}")
    try:
        report_payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot parse A1 training report: {error}") from error
    recipe = verified["recipe"]
    bound_recipe = verified.get("bound_recipe", recipe)
    central_binding = verified.get("central_learner_binding")
    learner_ablation = verified.get("learner_ablation")
    matched_aux = (
        learner_ablation.get("matched_aux_regularization")
        if isinstance(learner_ablation, dict)
        else None
    )
    is_production_composite = verified.get("data_kind") == "production_composite_v2"
    checkout_runtime_binding = (
        _verify_production_checkout_runtime_binding(
            report_payload.get("checkout_runtime_binding"),
            trainer_authority=verified["trainer_authority"],
        )
        if is_production_composite
        else None
    )
    validation_seed_manifest = (
        _verify_production_validation_seed_manifest(
            report.with_suffix(".validation_seeds.json"),
            report_payload=report_payload,
            verified=verified,
        )
        if is_production_composite
        else None
    )
    expected_steps = _expected_optimizer_steps(verified, recipe=recipe)
    effective_global_batch_size = _effective_global_batch_size(recipe)
    lineage_dose = _direct_lineage_dose(verified, report_payload=report_payload)
    if matched_aux is not None and (
        expected_steps != AUX_SELECTED_OPTIMIZER_STEPS
        or lineage_dose.get("current_sampled_rows") != AUX_SELECTED_SAMPLE_DOSE
        or recipe.get("amp") != "none"
        or int(recipe.get("world_size", 0)) != 8
        or int(recipe.get("batch_size", 0)) != 512
        or int(recipe.get("grad_accum_steps", 0)) != 1
    ):
        raise ExecutorError(
            "matched AUX output verifier requires exact FP32 8x512/128-step dose"
        )
    resume_identity = report_payload.get("training_resume_recipe_identity")
    if not isinstance(resume_identity, dict) or report_payload.get(
        "training_resume_recipe_identity_sha256"
    ) != _value_sha256(resume_identity):
        raise ExecutorError(
            "A1 training report lacks an authenticated resume-recipe identity"
        )
    try:
        from catan_zero.rl.optim_state import (
            TrainingProgressError,
            load_training_progress,
        )

        progress_payload = load_training_progress(
            checkpoint, expected_recipe_identity=resume_identity
        )
    except TrainingProgressError as error:
        raise ExecutorError(
            f"A1 training progress commit marker refused: {error}"
        ) from error
    if (
        progress_payload.get("optimizer_step") != expected_steps
        or progress_payload.get("completed_epochs") != 1
        or resume_identity.get("world_size") != int(recipe["world_size"])
    ):
        raise ExecutorError(
            "A1 training progress does not prove the exact one-dose terminal"
        )
    if matched_aux is not None:
        expected_world_size = int(recipe["world_size"])
        rank_numpy_rng = progress_payload.get("rank_numpy_rng_states")
        rank_torch_rng = progress_payload.get("rank_torch_rng_states")
        if (
            not isinstance(rank_numpy_rng, list)
            or len(rank_numpy_rng) != expected_world_size
            or any(not isinstance(row, dict) for row in rank_numpy_rng)
            or not isinstance(rank_torch_rng, list)
            or len(rank_torch_rng) != expected_world_size
            or any(not isinstance(row, dict) for row in rank_torch_rng)
            or progress_payload.get("symmetry_rng_state") is not None
        ):
            raise ExecutorError(
                "matched AUX progress lacks exact per-rank sampler/dropout RNG "
                "or incorrectly records a symmetry RNG while augmentation is off"
            )
    if command is not None:
        if not command or not isinstance(command[0], str):
            raise ExecutorError("A1 output verification lacks a canonical command")
        if execution_binding.get("command_sha256") != _value_sha256(command):
            raise ExecutorError(
                "A1 training progress verification command binding drift"
            )
    expected = {
        "arch": "entity_graph",
        **SEALED_A1_MODEL_REPORT,
        "world_size": int(recipe["world_size"]),
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "fused_optimizer": False,
        "epochs": 1,
        "max_steps": int(recipe["max_steps"]),
        "batch_size": int(recipe["batch_size"]),
        "grad_accum_steps": int(recipe["grad_accum_steps"]),
        "effective_global_batch_size": effective_global_batch_size,
        "ddp_shard_data": False,
        "amp": recipe["amp"],
        "lr": float(recipe["lr"]),
        "weight_decay": float(recipe["weight_decay"]),
        "seed": int(recipe["seed"]),
        "training_rng_rank_offset": bool(recipe.get("training_rng_rank_offset", False)),
        "mask_hidden_info": True,
        "symmetry_augment": False,
        "data": str(verified["data_path"]),
        "data_format": "memmap",
        "data_fingerprint": verified["data_fingerprint"],
        "samples": int(verified["corpus_row_count"]),
        "global_samples": int(verified["corpus_row_count"]),
        "train_samples": int(verified["training_row_count"]),
        "validation_samples": int(verified["validation_row_count"]),
        "track": recipe["track"],
        "vps_to_win": int(recipe["vps_to_win"]),
        "checkpoint": str(checkpoint),
        "init_checkpoint": str(
            verified.get("architecture_initializer", verified["producer"])["path"]
        ),
        "init_checkpoint_sha256": verified.get(
            "architecture_initializer", verified["producer"]
        )["sha256"],
        REPORT_LEARNER_LINEAGE_PARENT_FIELD: copy.deepcopy(
            verified.get("learner_lineage_parent")
        ),
        "a1_lineage_dose": lineage_dose,
        "forced_action_weight": float(recipe["forced_action_weight"]),
        "forced_row_value_weight": float(recipe["forced_row_value_weight"]),
        "per_game_policy_weight": bool(recipe.get("per_game_policy_weight", False)),
        "per_game_policy_weight_mode": str(
            recipe.get("per_game_policy_weight_mode", "equal")
        ),
        "per_game_value_weight": bool(recipe["per_game_value_weight"]),
        "value_loss_weight": float(recipe["value_loss_weight"]),
        "truncated_vp_margin_value_weight": float(
            recipe["truncated_vp_margin_value_weight"]
        ),
        "require_35m_model": True,
        "steps_completed": expected_steps,
        "total_training_steps": expected_steps,
    }
    if isinstance(learner_ablation, dict):
        # These post-seal optimizer axes are required evidence for diagnostic
        # arms, while legacy production reports retain their historical shape.
        expected["trunk_lr_mult"] = float(recipe.get("trunk_lr_mult", 1.0))
        if recipe.get("policy_kl_target") is not None:
            expected.update(
                {
                    "policy_kl_anchor_weight": float(recipe["policy_kl_anchor_weight"]),
                    "policy_kl_anchor_direction": str(
                        recipe["policy_kl_anchor_direction"]
                    ),
                    "policy_kl_target": float(recipe["policy_kl_target"]),
                    "policy_kl_dual_lr": float(recipe["policy_kl_dual_lr"]),
                    "policy_kl_max_weight": float(recipe["policy_kl_max_weight"]),
                }
            )
    if recipe.get("forced_row_value_action_type_weights"):
        expected["forced_row_value_action_type_weights"] = (
            train_bc._parse_forced_row_value_action_type_weights(
                str(recipe["forced_row_value_action_type_weights"])
            )
        )
    if matched_aux is not None:
        expected.update(
            {
                "a1_aux_regularization_binding": matched_aux,
                "aux_subgoal_loss_weight": float(
                    matched_aux["aux_subgoal_loss_weight"]
                ),
                "aux_subgoal_heads": True,
                "requested_aux_subgoal_heads": True,
                "aux_settlement_pointer_head": True,
                "requested_aux_settlement_pointer_head": True,
                "ddp_find_unused_parameters": False,
            }
        )
    if is_production_composite:
        expected.update(
            {
                "a1_contract_sha256": None,
                "input_validation_game_seed_manifest": None,
                "input_validation_game_seed_manifest_sha256": None,
                "validation_game_seed_set_sha256": verified[
                    "trainer_validation_game_seed_set_sha256"
                ],
                "a1_selected_game_seed_set_sha256": None,
                "a1_training_game_seed_set_sha256": None,
                "a1_memmap_payload_inventory_sha256": None,
                "a1_learner_training_recipe_sha256": None,
            }
        )
    else:
        expected.update(
            {
                "a1_contract_sha256": verified["contract_sha256"],
                "input_validation_game_seed_manifest": str(verified["validation_path"]),
                "input_validation_game_seed_manifest_sha256": verified[
                    "validation_file_sha256"
                ],
                "validation_game_seed_set_sha256": verified[
                    "validation_game_seed_set_sha256"
                ],
                "a1_selected_game_seed_set_sha256": verified[
                    "selected_game_seed_set_sha256"
                ],
                "a1_training_game_seed_set_sha256": verified[
                    "training_game_seed_set_sha256"
                ],
                "a1_memmap_payload_inventory_sha256": verified[
                    "payload_inventory_sha256"
                ],
                "a1_learner_training_recipe_sha256": _value_sha256(bound_recipe),
            }
        )
    drift = {
        key: {"expected": value, "actual": report_payload.get(key)}
        for key, value in expected.items()
        if report_payload.get(key) != value
    }
    if drift:
        raise ExecutorError(f"A1 training report invariant drift: {drift}")
    checkpoint_step_values = (
        [] if command is None else _literal_option_values(command, "--checkpoint-steps")
    )
    if len(checkpoint_step_values) > 1:
        raise ExecutorError("A1 training command repeats --checkpoint-steps")
    expected_checkpoint_steps = (
        ()
        if not checkpoint_step_values
        else train_bc._parse_checkpoint_steps(  # noqa: SLF001
            checkpoint_step_values[0], max_steps=int(recipe["max_steps"])
        )
    )
    intermediate_records = report_payload.get("intermediate_checkpoints", [])
    if (
        report_payload.get("checkpoint_steps_requested", [])
        != list(expected_checkpoint_steps)
        or not isinstance(intermediate_records, list)
        or len(intermediate_records) != len(expected_checkpoint_steps)
    ):
        raise ExecutorError(
            "A1 training report lost its same-trajectory intermediate checkpoints"
        )
    verified_intermediate: list[dict[str, Any]] = []
    for step, record in zip(
        expected_checkpoint_steps, intermediate_records, strict=True
    ):
        expected_path = train_bc._step_checkpoint_path(checkpoint, step)  # noqa: SLF001
        expected_record = {
            "schema_version": "train-bc-intermediate-checkpoint-v1",
            "optimizer_step": step,
            "checkpoint": str(expected_path),
            "checkpoint_sha256": _file_sha256(expected_path)
            if expected_path.is_file()
            else "",
            "size_bytes": expected_path.stat().st_size
            if expected_path.is_file()
            else 0,
            "same_training_trajectory": True,
            "optimizer_sidecar": None,
        }
        if (
            record != expected_record
            or not expected_path.is_file()
            or expected_path.is_symlink()
            or Path(str(expected_path) + ".optimizer.pt").exists()
            or Path(str(expected_path) + ".training-progress.json").exists()
        ):
            raise ExecutorError(
                f"A1 intermediate checkpoint step {step} is missing or malformed"
            )
        try:
            import torch

            snapshot = torch.load(expected_path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise ExecutorError(
                f"cannot load A1 intermediate checkpoint step {step}: {error}"
            ) from error
        snapshot_value = (
            snapshot.get("value_training") if isinstance(snapshot, dict) else None
        )
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("policy_type") != "entity_graph"
            or not isinstance(snapshot.get("model"), dict)
            or not isinstance(snapshot_value, dict)
            or snapshot_value.get("optimizer_steps") != step
            or snapshot_value.get("completed_epochs") != 0
            or "scalar" not in snapshot_value.get("trained_value_readouts", [])
            or snapshot_value.get("intermediate_checkpoint")
            != {
                "schema_version": "train-bc-intermediate-checkpoint-v1",
                "optimizer_step": step,
                "same_training_trajectory": True,
                "optimizer_sidecar_intentionally_omitted": True,
            }
        ):
            raise ExecutorError(
                f"A1 intermediate checkpoint step {step} lost training provenance"
            )
        if is_production_composite:
            _require_production_event_history_surface(
                snapshot.get("training_information_surface"),
                expected_contract=verified["event_history_training_contract"],
                row_count=int(verified["corpus_row_count"]),
                where=f"production intermediate checkpoint step {step}",
            )
            if (
                snapshot.get("public_award_feature_contract") != "authoritative_v1"
                or snapshot_value.get("checkout_runtime_binding")
                != checkout_runtime_binding
            ):
                raise ExecutorError(
                    f"production intermediate checkpoint step {step} lost "
                    "award/runtime authority"
                )
            _strict_load_production_entity_checkpoint(
                expected_path,
                where=f"production intermediate checkpoint step {step}",
            )
        _fsync_file(expected_path)
        verified_intermediate.append(expected_record)
    if "policy_aux_active_batch_size" in recipe:
        exposure = lineage_dose["objective_exposure"]
        expected_draws = {
            "policy_aux_active_batch_size": int(recipe["policy_aux_active_batch_size"]),
            "base_training_row_draws": int(exposure["base_sampled_rows"]),
            "policy_aux_training_row_draws": int(
                exposure["policy_aux_active_sampled_rows"]
            ),
            "policy_base_active_rows": int(exposure["policy_base_active_sampled_rows"]),
            "policy_aux_active_rows": int(exposure["policy_aux_active_sampled_rows"]),
            "policy_total_active_rows": int(exposure["policy_active_sampled_rows"]),
        }
        draw_drift = {
            key: {"expected": value, "actual": report_payload.get(key)}
            for key, value in expected_draws.items()
            if report_payload.get(key) != value
        }
        if draw_drift:
            raise ExecutorError(f"A1 objective-dose report drift: {draw_drift}")
    if is_production_composite:
        if (
            report_payload.get("a1_bound_learner_training_recipe") is not None
            or report_payload.get("a1_bound_learner_value_objective") is not None
        ):
            raise ExecutorError(
                "flywheel composite incorrectly inherited an A1 validation identity"
            )
        composite = report_payload.get("memmap_composite")
        expected_composite = {
            "schema_version": "memmap_composite_v2",
            "descriptor_path": str(verified["data_path"]),
            "descriptor_file_sha256": verified["corpus_meta_file_sha256"],
            "descriptor_fingerprint": verified["descriptor_fingerprint"],
            "payload_inventory_sha256": verified["payload_inventory_sha256"],
            "learner_recipe_overrides": verified["learner_recipe_overrides"],
            "learner_recipe_overrides_sha256": verified[
                "learner_recipe_overrides_sha256"
            ],
            "aux_subgoal_target_contract_sha256": verified[
                "aux_subgoal_target_contract_sha256"
            ],
            "public_award_feature_transition_contract_sha256": verified[
                "public_award_feature_transition_contract_sha256"
            ],
            "source_authority_semantic_sha256": verified.get(
                "source_authority_semantic_sha256"
            ),
            "component_count": 4,
            "component_ids": [
                "current_producer",
                "recent_history",
                "hard_negative",
                "historical_replay",
            ],
            "component_game_sampling_ratios": [0.64, 0.12, 0.04, 0.20],
            "policy_distillation_component_ids": (
                verified.get("diagnostic_training_descriptor_authority", {}).get(
                    "policy_distillation_component_ids",
                    list(ALL_POST_WAVE_COMPONENT_IDS),
                )
            ),
            "policy_distillation_scope_explicit": True,
            "value_training_component_ids": (
                verified.get("diagnostic_training_descriptor_authority", {}).get(
                    "value_training_component_ids",
                    list(ALL_POST_WAVE_COMPONENT_IDS),
                )
            ),
            "value_training_scope_explicit": True,
            "diagnostic_derivation_authority": (
                verified.get("diagnostic_training_descriptor_authority", {}).get(
                    "diagnostic_derivation_authority"
                )
            ),
            "stored_policy_component_temperatures": (
                composite_builder.STORED_POLICY_COMPONENT_TEMPERATURES
            ),
            "entity_feature_adapter_component_versions": verified[
                "entity_feature_adapter_component_versions"
            ],
            "flywheel_replay_contract": verified["production_mix_contract"],
            "category_semantics": verified.get("category_semantics"),
            "category_semantics_sha256": verified.get("category_semantics_sha256"),
        }
        expected_diagnostic = (
            central_binding["diagnostic_only"]
            if isinstance(central_binding, dict)
            else learner_ablation is not None
        )
        expected_promotion = (
            central_binding["promotion_eligible"]
            if isinstance(central_binding, dict)
            else learner_ablation is None
        )
        if (
            not isinstance(composite, dict)
            or any(
                composite.get(key) != value for key, value in expected_composite.items()
            )
            or report_payload.get("diagnostic_only") != expected_diagnostic
            or report_payload.get("promotion_eligible") != expected_promotion
            or (
                isinstance(central_binding, dict)
                and report_payload.get("eligible_for_full_gate")
                is not central_binding["eligible_for_full_gate"]
            )
        ):
            raise ExecutorError(
                "training report does not bind the exact promotion replay composite"
            )
        descriptor_authority = verified.get("diagnostic_training_descriptor_authority")
        if descriptor_authority is not None:
            expected_policy_components = set(
                descriptor_authority["policy_distillation_component_ids"]
            )
            expected_value_components = set(
                descriptor_authority["value_training_component_ids"]
            )
            policy_scope = report_payload.get("policy_distillation_scope")
            policy_components = (
                policy_scope.get("components")
                if isinstance(policy_scope, dict)
                else None
            )
            component_dose = report_payload.get("policy_component_active_dose")
            value_scope = report_payload.get("value_training_scope")
            value_components = (
                value_scope.get("components") if isinstance(value_scope, dict) else None
            )
            value_component_dose = report_payload.get("value_component_active_dose")
            if (
                report_payload.get("a1_diagnostic_training_descriptor_authority")
                != descriptor_authority
                or report_payload.get("value_training", {}).get(
                    "diagnostic_training_descriptor_authority"
                )
                != descriptor_authority
                or not isinstance(policy_scope, dict)
                or policy_scope.get("schema_version")
                != "component-policy-distillation-scope-v1"
                or policy_scope.get("component_ids")
                != descriptor_authority["policy_distillation_component_ids"]
                or not isinstance(policy_components, dict)
                or set(policy_components) != set(ALL_POST_WAVE_COMPONENT_IDS)
                or not isinstance(component_dose, dict)
                or set(component_dose) != set(ALL_POST_WAVE_COMPONENT_IDS)
                or not isinstance(value_scope, dict)
                or value_scope.get("schema_version")
                != "component-value-training-scope-v1"
                or value_scope.get("component_ids")
                != descriptor_authority["value_training_component_ids"]
                or not isinstance(value_components, dict)
                or set(value_components) != set(ALL_POST_WAVE_COMPONENT_IDS)
                or not isinstance(value_component_dose, dict)
                or set(value_component_dose) != set(ALL_POST_WAVE_COMPONENT_IDS)
            ):
                raise ExecutorError(
                    "diagnostic descriptor scope/provenance differs from issued authority"
                )
            for component_id in ALL_POST_WAVE_COMPONENT_IDS:
                policy_enabled = component_id in expected_policy_components
                value_enabled = component_id in expected_value_components
                policy_record = policy_components[component_id]
                dose_record = component_dose[component_id]
                value_record = value_components[component_id]
                value_dose_record = value_component_dose[component_id]
                if (
                    not isinstance(policy_record, dict)
                    or policy_record.get("policy_distillation_enabled")
                    is not policy_enabled
                    or not isinstance(dose_record, dict)
                    or not isinstance(value_record, dict)
                    or value_record.get("value_training_enabled") is not value_enabled
                    or not isinstance(value_dose_record, dict)
                ):
                    raise ExecutorError(
                        f"diagnostic objective scope drifted for {component_id}"
                    )
                positive_rows = policy_record.get("positive_policy_rows")
                weight_sum = policy_record.get("policy_weight_sum")
                active_rows = dose_record.get("total_active_rows")
                if (
                    isinstance(positive_rows, bool)
                    or not isinstance(positive_rows, int)
                    or isinstance(weight_sum, bool)
                    or not isinstance(weight_sum, (int, float))
                    or not math.isfinite(float(weight_sum))
                    or isinstance(active_rows, bool)
                    or not isinstance(active_rows, int)
                    or (
                        policy_enabled
                        and not (
                            positive_rows > 0
                            and float(weight_sum) > 0.0
                            and active_rows > 0
                        )
                    )
                    or (
                        not policy_enabled
                        and not (
                            positive_rows == 0
                            and float(weight_sum) == 0.0
                            and active_rows == 0
                        )
                    )
                ):
                    raise ExecutorError(
                        f"diagnostic policy exposure drifted for {component_id}"
                    )
                positive_value_rows = value_record.get("positive_value_rows")
                value_weight_sum = value_record.get("value_weight_sum")
                value_active_rows = value_dose_record.get("active_rows")
                if (
                    isinstance(positive_value_rows, bool)
                    or not isinstance(positive_value_rows, int)
                    or isinstance(value_weight_sum, bool)
                    or not isinstance(value_weight_sum, (int, float))
                    or not math.isfinite(float(value_weight_sum))
                    or isinstance(value_active_rows, bool)
                    or not isinstance(value_active_rows, int)
                    or (
                        value_enabled
                        and not (
                            positive_value_rows > 0
                            and float(value_weight_sum) > 0.0
                            and value_active_rows > 0
                        )
                    )
                    or (
                        not value_enabled
                        and not (
                            positive_value_rows == 0
                            and float(value_weight_sum) == 0.0
                            and value_active_rows == 0
                        )
                    )
                ):
                    raise ExecutorError(
                        f"diagnostic value exposure drifted for {component_id}"
                    )
        if isinstance(central_binding, dict):
            realized = report_payload.get("a1_realized_central_sample_order")
            if (
                report_payload.get("a1_central_learner_binding") != central_binding
                or report_payload.get("a1_central_published_executor_authority")
                != verified.get("central_published_executor_authority")
                or not isinstance(realized, dict)
                or report_payload.get("a1_realized_central_sample_evidence_sha256")
                != _value_sha256(realized)
            ):
                raise ExecutorError(
                    "central training report lost published authority/realized order"
                )
    else:
        if report_payload.get("a1_bound_learner_training_recipe") != bound_recipe:
            raise ExecutorError(
                "A1 training report does not echo the exact sealed recipe"
            )
        if (
            report_payload.get("a1_bound_learner_value_objective")
            != verified["objective"]
        ):
            raise ExecutorError(
                "A1 training report does not echo the sealed value objective"
            )
    if report_payload.get(REPORT_EXECUTION_BINDING_FIELD) != execution_binding:
        raise ExecutorError(
            "A1 training report does not bind the exact child environment/command"
        )
    if report_payload.get(REPORT_INPUT_BINDING_FIELD) != _input_binding(verified):
        raise ExecutorError(
            "A1 training report does not bind the authenticated input/split/topology"
        )
    production_information_surface = report_payload.get("training_information_surface")
    if is_production_composite:
        award_training = report_payload.get("public_award_feature_training")
        if report_payload.get("public_award_feature_contract") != "authoritative_v1":
            raise ExecutorError(
                "production report did not prove the legacy-to-authoritative "
                "public-award transition"
            )
        _require_production_public_award_transition(
            award_training,
            verified=verified,
            where="production training report",
        )
        _require_production_event_history_surface(
            production_information_surface,
            expected_contract=verified["event_history_training_contract"],
            row_count=int(verified["corpus_row_count"]),
            where="production training report",
        )
        try:
            import torch

            terminal_checkpoint = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
        except Exception as error:
            raise ExecutorError(
                f"cannot load production terminal checkpoint: {error}"
            ) from error
        _require_production_event_history_surface(
            (
                terminal_checkpoint.get("training_information_surface")
                if isinstance(terminal_checkpoint, dict)
                else None
            ),
            expected_contract=verified["event_history_training_contract"],
            row_count=int(verified["corpus_row_count"]),
            where="production terminal checkpoint",
        )
        terminal_value_training = (
            terminal_checkpoint.get("value_training")
            if isinstance(terminal_checkpoint, dict)
            else None
        )
        if (
            terminal_checkpoint.get("public_award_feature_contract")
            != "authoritative_v1"
            or not isinstance(terminal_value_training, dict)
            or terminal_value_training.get("checkout_runtime_binding")
            != checkout_runtime_binding
        ):
            raise ExecutorError(
                "production terminal checkpoint lost award/runtime authority"
            )
        _strict_load_production_entity_checkpoint(
            checkpoint, where="production terminal checkpoint"
        )
    metrics = report_payload.get("metrics")
    if (
        not isinstance(metrics, list)
        or len(metrics) != 1
        or not isinstance(metrics[0], dict)
        or metrics[0].get("epoch") != 1
    ):
        raise ExecutorError(
            "A1 training report does not prove exactly one completed epoch"
        )
    for key in ("loss", "policy_loss", "value_loss"):
        value = metrics[0].get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ExecutorError(f"A1 training report has invalid epoch metric {key!r}")
    validation_metrics = metrics[0].get("validation")
    if not isinstance(validation_metrics, dict):
        raise ExecutorError("A1 training report has no one-epoch validation metrics")
    validation_loss = validation_metrics.get("loss")
    if (
        validation_metrics.get("samples") != int(verified["validation_row_count"])
        or isinstance(validation_loss, bool)
        or not isinstance(validation_loss, (int, float))
        or not math.isfinite(float(validation_loss))
    ):
        raise ExecutorError(
            "A1 training report has invalid validation coverage/metrics"
        )
    if matched_aux is not None:
        information_surface = report_payload.get("training_information_surface")
        freeze = (
            information_surface.get("inactive_training_head_freeze")
            if isinstance(information_surface, dict)
            else None
        )
        if not isinstance(freeze, dict):
            raise ExecutorError(
                "matched aux report lacks inactive-head/DDP optimizer evidence"
            )
        frozen = set(freeze.get("frozen_submodules", []))
        active = set(freeze.get("active_optional_submodules", []))
        aux_weight = float(matched_aux["aux_subgoal_loss_weight"])
        if aux_weight == 0.0:
            if not AUX_SUBGOAL_HEAD_MODULES.issubset(frozen) or (
                AUX_SUBGOAL_HEAD_MODULES & active
            ):
                raise ExecutorError(
                    "AUX0 did not freeze/exclude every shared auxiliary head"
                )
        elif aux_weight > 0.0:
            if not AUX_SUBGOAL_HEAD_MODULES.issubset(active) or (
                AUX_SUBGOAL_HEAD_MODULES & frozen
            ):
                raise ExecutorError("AUXT did not activate every shared auxiliary head")
        else:  # The binding validator should make this unreachable.
            raise ExecutorError("unsupported matched auxiliary loss weight")
        parts = metrics[0].get("aux_subgoal_loss_parts")
        if not isinstance(parts, dict) or set(parts) != AUX_SUBGOAL_TARGET_FIELDS:
            raise ExecutorError("matched aux report has incomplete per-head evidence")
        denominators: list[float] = []
        for field in sorted(AUX_SUBGOAL_TARGET_FIELDS):
            record = parts.get(field)
            if (
                not isinstance(record, dict)
                or set(record) != {"weighted_sum", "weight_sum"}
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in record.values()
                )
            ):
                raise ExecutorError(
                    f"matched aux per-head evidence is malformed: {field}"
                )
            denominators.append(float(record["weight_sum"]))
        if (aux_weight == 0.0 and any(value != 0.0 for value in denominators)) or (
            aux_weight > 0.0 and any(value <= 0.0 for value in denominators)
        ):
            raise ExecutorError(
                "matched aux per-head label exposure disagrees with its arm"
            )
    if is_production_composite:
        matched = metrics[0].get("validation_objective_matched")
        ratios = {
            "current_producer": 0.64,
            "recent_history": 0.12,
            "hard_negative": 0.04,
            "historical_replay": 0.20,
        }
        split_components = {
            record["component_id"]: record
            for record in verified["validation_split_receipt"]["components"]
        }
        matched_components = (
            matched.get("components") if isinstance(matched, dict) else None
        )
        matched_metrics = matched.get("metrics") if isinstance(matched, dict) else None
        if (
            not isinstance(matched, dict)
            or matched.get("schema_version") != "composite-validation-measure-v2"
            or matched.get("objective_matched") is not True
            or matched.get("samples") != int(verified["validation_row_count"])
            or matched.get("games")
            != int(
                verified["validation_split_receipt"]["aggregate"][
                    "validation_game_count"
                ]
            )
            or matched.get("component_sampling_ratios") != ratios
            or not isinstance(matched_components, dict)
            or set(matched_components) != set(ratios)
            or not isinstance(matched_metrics, dict)
            or any(
                key not in matched_metrics
                for key in ("loss", "policy_loss", "value_loss")
            )
        ):
            raise ExecutorError(
                "production acceptance requires objective-matched composite validation"
            )
        for component_id, ratio in ratios.items():
            report_component = matched_components[component_id]
            split_component = split_components[component_id]
            if (
                not isinstance(report_component, dict)
                or report_component.get("authenticated_sampling_ratio") != ratio
                or report_component.get("games")
                != split_component["validation_game_count"]
                or report_component.get("rows")
                != split_component["validation_row_count"]
                or not isinstance(report_component.get("metrics"), dict)
            ):
                raise ExecutorError(
                    "objective-matched validation component coverage differs from "
                    f"the authenticated split: {component_id}"
                )
        _require_finite_metric_tree(
            matched_metrics, where="validation_objective_matched.metrics"
        )
        _require_finite_metric_tree(
            matched_components,
            where="validation_objective_matched.components",
        )
    parameter_count = report_payload.get("parameter_count")
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or not 30_000_000 <= parameter_count <= 40_000_000
    ):
        raise ExecutorError("A1 training report does not prove the required 35M model")
    value_training = report_payload.get("value_training")
    expected_value_training: dict[str, Any] = {
        "primary_readout": "scalar",
        "optimizer_steps": expected_steps,
        "completed_epochs": 1,
    }
    if not is_production_composite:
        expected_value_training.update(
            {
                "a1_contract_sha256": verified["contract_sha256"],
                "a1_selected_game_seed_set_sha256": verified[
                    "selected_game_seed_set_sha256"
                ],
                "a1_training_game_seed_set_sha256": verified[
                    "training_game_seed_set_sha256"
                ],
                "a1_learner_training_recipe_sha256": _value_sha256(bound_recipe),
                "a1_memmap_payload_inventory_sha256": verified[
                    "payload_inventory_sha256"
                ],
            }
        )
    if not isinstance(value_training, dict) or any(
        value_training.get(key) != value
        for key, value in expected_value_training.items()
    ):
        raise ExecutorError("A1 training report value-training provenance drift")
    if learner_ablation is not None and central_binding is None:
        if (
            report_payload.get("a1_effective_learner_training_recipe") != recipe
            or report_payload.get("a1_effective_learner_training_recipe_sha256")
            != _value_sha256(recipe)
            or report_payload.get("a1_learner_ablation") != learner_ablation
            or report_payload.get("diagnostic_only") is not True
            or report_payload.get("promotion_eligible") is not False
            or value_training.get("learner_ablation") != learner_ablation
        ):
            raise ExecutorError(
                "A1 learner ablation provenance/diagnostic marker drift"
            )
    if "scalar" not in value_training.get("trained_value_readouts", []):
        raise ExecutorError(
            "A1 candidate does not attest a trained scalar value readout"
        )
    if matched_aux is not None:
        _verify_matched_aux_torch_artifacts(
            checkpoint,
            optimizer,
            expected_steps=expected_steps,
            recipe=recipe,
        )
    durable_outputs = [checkpoint, optimizer, progress, report]
    if validation_seed_manifest is not None:
        durable_outputs.append(Path(str(validation_seed_manifest["path"])))
    for path in durable_outputs:
        _fsync_file(path)
    for parent in {path.parent for path in durable_outputs}:
        _fsync_directory(parent)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "optimizer_sidecar": str(optimizer),
        "optimizer_sidecar_sha256": _file_sha256(optimizer),
        "training_progress": str(progress),
        "training_progress_sha256": _file_sha256(progress),
        "training_progress_payload_sha256": progress_payload.get("progress_sha256"),
        "report": str(report),
        "report_sha256": _file_sha256(report),
        **(
            {"intermediate_checkpoints": verified_intermediate}
            if expected_checkpoint_steps
            else {}
        ),
        "sample_receipt_state_sha256": (
            central_binding["sample_binding"]["sample_receipt_state_sha256"]
            if isinstance(central_binding, dict)
            else None
        ),
        "sample_order_sha256": (
            central_binding["sample_binding"]["sample_order_sha256"]
            if isinstance(central_binding, dict)
            else None
        ),
        "row_set_sha256": (
            central_binding["sample_binding"]["row_set_sha256"]
            if isinstance(central_binding, dict)
            else None
        ),
        "realized_sample_evidence_sha256": (
            report_payload.get("a1_realized_central_sample_evidence_sha256")
            if isinstance(central_binding, dict)
            else None
        ),
        "execution_binding_sha256": _value_sha256(execution_binding),
        "input_binding_sha256": _input_binding(verified)["binding_sha256"],
        "steps_completed": expected_steps,
        # With-replacement sampling and terminal batch padding mean the number
        # of distinct rows is not derivable from the row-draw dose.  Do not
        # manufacture a false uniqueness claim.
        "unique_training_rows": None,
        "base_sampler_draw_events": lineage_dose["current_sampled_rows"],
        "sampler_draw_events": lineage_dose["current_sampled_rows"],
        "sampled_rows": lineage_dose["current_sampled_rows"],
        "lineage_dose": lineage_dose,
        "corpus_row_count": int(verified["corpus_row_count"]),
        "training_row_count": int(verified["training_row_count"]),
        "validation_row_count": int(verified["validation_row_count"]),
        "production_sampling_receipt_sha256": verified.get(
            "production_sampling_receipt_sha256"
        ),
        "validation_split_receipt_sha256": verified.get(
            "validation_split_receipt_sha256"
        ),
        **(
            {
                "validation_seed_manifest": validation_seed_manifest["path"],
                "validation_seed_manifest_sha256": validation_seed_manifest[
                    "file_sha256"
                ],
                "validation_game_seed_count": validation_seed_manifest[
                    "game_seed_count"
                ],
                "validation_game_seed_set_sha256": validation_seed_manifest[
                    "game_seed_set_sha256"
                ],
            }
            if validation_seed_manifest is not None
            else {}
        ),
    }


def _generic_completed_ablation_overrides(
    learner_ablation: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover the effective generic override from terminal receipt semantics."""

    if (
        not isinstance(learner_ablation, dict)
        or learner_ablation.get("schema_version") != "a1-learner-ablation-v1"
        or isinstance(learner_ablation.get("matched_aux_regularization"), dict)
    ):
        raise ExecutorError("completed generic ablation identity is invalid")
    bound = learner_ablation.get("bound_recipe")
    effective = learner_ablation.get("effective_recipe")
    drift = learner_ablation.get("recipe_drift")
    if (
        not isinstance(bound, dict)
        or not isinstance(effective, dict)
        or not isinstance(drift, dict)
        or learner_ablation.get("bound_recipe_sha256") != _value_sha256(bound)
        or learner_ablation.get("effective_recipe_sha256") != _value_sha256(effective)
        or learner_ablation.get("recipe_drift_sha256") != _value_sha256(drift)
    ):
        raise ExecutorError("completed generic ablation recipe digest drift")
    overrides = {
        key: copy.deepcopy(effective[key])
        for key in sorted(A1_LEARNER_ABLATION_FIELDS)
        if key in effective and (key not in bound or effective[key] != bound[key])
    }
    if not overrides:
        raise ExecutorError("completed generic ablation has no effective treatment")
    return overrides


def _replay_completed_ablation_receipt_authority(
    payload: dict[str, Any], *, claim_path: Path, authenticated_started_unix_ns: int
) -> dict[str, Any]:
    """Replay a completed non-matched ablation from its immutable authorities."""

    input_binding = payload.get("input_binding")
    learner_ablation = payload.get("learner_ablation")
    if not isinstance(input_binding, dict) or not isinstance(learner_ablation, dict):
        raise ExecutorError("completed ablation lacks input/ablation authority")
    if isinstance(learner_ablation.get("matched_aux_regularization"), dict):
        raise ExecutorError(
            "matched AUX receipts require the matched-AUX receipt verifier"
        )
    lock_ref = payload.get("lock")
    corpus_ref = payload.get("corpus")
    reviewed_lock_sha256 = learner_ablation.get("reviewed_lock_file_sha256")
    if (
        not isinstance(lock_ref, str)
        or not isinstance(corpus_ref, str)
        or not isinstance(reviewed_lock_sha256, str)
        or reviewed_lock_sha256 != payload.get("lock_file_sha256")
    ):
        raise ExecutorError("completed ablation has no reviewed lock/data authority")
    lock_lexical = Path(lock_ref).expanduser()
    if lock_lexical.is_symlink() or not lock_lexical.is_file():
        raise ExecutorError("completed ablation reviewed lock must be a regular file")
    try:
        lock_path = lock_lexical.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot resolve completed ablation reviewed lock: {error}"
        ) from error
    if _file_sha256(lock_path) != reviewed_lock_sha256:
        raise ExecutorError("completed ablation reviewed lock bytes drift")

    data_kind = input_binding.get("data_kind")
    validation_ref = payload.get("validation_manifest")
    if not isinstance(validation_ref, str):
        raise ExecutorError("completed ablation has no validation manifest")
    validation_path = Path(validation_ref)
    coherent_admission = None
    composite_build_receipt = None
    if data_kind == "coherent_direct_memmap_v1":
        admission = input_binding.get("coherent_corpus_admission")
        admission_ref = admission.get("path") if isinstance(admission, dict) else None
        if not isinstance(admission_ref, str):
            raise ExecutorError("completed ablation has no coherent admission")
        coherent_admission = Path(admission_ref)
    elif data_kind == "production_composite_v2":
        raise ExecutorError(
            "production-composite ablations require their descriptor-specific verifier"
        )
    replayed = verify_training_inputs(
        lock_path=lock_path,
        data_path=Path(corpus_ref),
        validation_path=validation_path,
        composite_build_receipt=composite_build_receipt,
        reviewed_lock_file_sha256=reviewed_lock_sha256,
        coherent_corpus_admission=coherent_admission,
    )

    upgrade = payload.get("function_preserving_upgrade")
    upgrade_receipt = upgrade.get("receipt") if isinstance(upgrade, dict) else None
    upgrade_receipt_path = (
        upgrade_receipt.get("path") if isinstance(upgrade_receipt, dict) else None
    )
    if not isinstance(upgrade_receipt_path, str):
        raise ExecutorError("completed ablation lacks upgrade receipt authority")
    diagnostic_source = input_binding.get("diagnostic_comparison_source")
    independent_authority = (
        diagnostic_source.get("authority")
        if isinstance(diagnostic_source, dict)
        else None
    )
    independent_authority_path = (
        independent_authority.get("path")
        if isinstance(independent_authority, dict)
        else None
    )
    if not isinstance(independent_authority_path, str):
        raise ExecutorError("completed ablation lacks independent-parent authority")
    reconstructed = bind_function_preserving_upgrade(
        replayed,
        Path(upgrade_receipt_path),
        independent_parent_authority_path=Path(independent_authority_path),
    )
    code_binding = learner_ablation.get("code_binding")
    reporting = learner_ablation.get("reporting_contract")
    if not isinstance(code_binding, dict) or not isinstance(reporting, dict):
        raise ExecutorError("completed ablation lacks code/reporting authority")
    overrides = _generic_completed_ablation_overrides(learner_ablation)
    checkpoint_steps = reporting.get("checkpoint_steps")
    if not isinstance(checkpoint_steps, list) or any(
        isinstance(step, bool) or not isinstance(step, int) for step in checkpoint_steps
    ):
        raise ExecutorError("completed ablation checkpoint schedule is invalid")
    reconstructed = bind_learner_ablation(
        reconstructed,
        ablation_id=str(learner_ablation.get("ablation_id", "")),
        overrides_json=_canonical_bytes(overrides).decode("ascii"),
        reviewed_code_tree_sha256=str(learner_ablation.get("code_tree_sha256", "")),
        diagnostic_dose_curve=reporting.get("diagnostic_dose_curve") is True,
        diagnostic_checkpoint_steps=",".join(map(str, checkpoint_steps)),
        _authenticated_completed_code_binding=code_binding,
    )

    topology = payload.get("training_topology")
    gpu = payload.get("gpu")
    if (
        not isinstance(topology, dict)
        or not isinstance(topology.get("name"), str)
        or isinstance(gpu, bool)
        or not isinstance(gpu, int)
    ):
        raise ExecutorError("completed ablation topology authority is invalid")
    reconstructed = bind_training_topology(
        reconstructed, topology=str(topology["name"]), gpu=gpu
    )
    canary_ref = payload.get("ddp_canary")
    canary_path = canary_ref.get("path") if isinstance(canary_ref, dict) else None
    historical_root = Path(str(code_binding["repository_root"]))
    if not isinstance(canary_path, str):
        raise ExecutorError("completed ablation lacks canary/start authority")
    canary = _verify_ddp_canary_receipt(
        Path(canary_path),
        reference_time_ns=authenticated_started_unix_ns,
        completed_repository_root=historical_root,
    )
    reconstructed = _bind_verified_ddp_canary(reconstructed, canary)

    if (
        reconstructed.get("function_preserving_upgrade") != upgrade
        or reconstructed.get("learner_ablation") != learner_ablation
        or reconstructed.get("training_topology") != topology
        or reconstructed.get("ddp_canary") != canary_ref
        or _input_binding(reconstructed) != input_binding
        or reconstructed.get("claim_identity_sha256")
        != payload.get("claim_identity_sha256")
    ):
        raise ExecutorError("completed ablation canonical derived-state drift")
    if (
        payload.get("contract_sha256") != reconstructed["contract_sha256"]
        or payload.get("lock") != str(reconstructed["lock_path"])
        or payload.get("lock_file_sha256") != reconstructed["lock_file_sha256"]
        or payload.get("corpus") != str(reconstructed["data_path"])
        or payload.get("corpus_meta_file_sha256")
        != reconstructed["corpus_meta_file_sha256"]
        or payload.get("payload_inventory_sha256")
        != reconstructed["payload_inventory_sha256"]
        or payload.get("producer_checkpoint_sha256")
        != reconstructed["producer"]["sha256"]
        or payload.get("validation_manifest") != str(reconstructed["validation_path"])
        or payload.get("validation_manifest_file_sha256")
        != reconstructed["validation_file_sha256"]
        or payload.get("learner_lineage_parent")
        != reconstructed.get("learner_lineage_parent")
    ):
        raise ExecutorError("completed ablation drifted from replayed lock/input")
    expected_claim_path = _claim_path(reconstructed).resolve(strict=False)
    if claim_path != expected_claim_path:
        raise ExecutorError(
            "completed ablation claim is not the canonical derived path"
        )

    outputs = payload.get("outputs")
    command = payload.get("command")
    if not isinstance(outputs, dict) or not isinstance(command, list) or not command:
        raise ExecutorError("completed ablation lacks command/output authority")
    checkpoint = Path(str(outputs.get("checkpoint", "")))
    report = Path(str(outputs.get("report", "")))
    canonical_command = build_train_command(
        reconstructed,
        python=Path(str(command[0])),
        checkpoint=checkpoint,
        report=report,
    )
    environment = _child_environment(
        _selected_gpus(reconstructed, fallback_gpu=gpu),
        repository_root=historical_root,
    )
    execution_binding = _execution_binding(
        command=canonical_command, environment=environment
    )
    if (
        command != canonical_command
        or payload.get("command_sha256") != _value_sha256(canonical_command)
        or payload.get("execution_binding") != execution_binding
        or payload.get("training_transaction_sha256")
        != _training_transaction_sha256(
            command=canonical_command, input_binding=input_binding
        )
    ):
        raise ExecutorError("completed ablation command/environment replay drift")
    reverified_outputs = _verify_training_outputs(
        checkpoint=checkpoint,
        report=report,
        verified=reconstructed,
        execution_binding=execution_binding,
        command=canonical_command,
    )
    if reverified_outputs != outputs:
        raise ExecutorError("completed ablation canonical output verification drift")
    return reconstructed


def _load_authenticated_completed_ablation_receipt(path: Path) -> dict[str, Any]:
    """Authenticate one completed generic ablation without AUX pair authority."""

    lexical = path.expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise ExecutorError("completed ablation receipt must be a regular file")
    try:
        receipt_path = lexical.resolve(strict=True)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(
            f"cannot load completed ablation receipt: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ExecutorError("completed ablation receipt is not an object")
    unsigned = dict(payload)
    stated = unsigned.pop("receipt_sha256", None)
    if (
        stated != _value_sha256(unsigned)
        or payload.get("schema_version") != ABLATION_RECEIPT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("diagnostic_only") is not True
        or payload.get("promotion_eligible") is not False
    ):
        raise ExecutorError("completed ablation receipt schema/status/digest drift")
    learner_ablation = payload.get("learner_ablation")
    if not isinstance(learner_ablation, dict) or isinstance(
        learner_ablation.get("matched_aux_regularization"), dict
    ):
        raise ExecutorError("completed receipt is not a generic learner ablation")

    claim_ref = payload.get("claim")
    if not isinstance(claim_ref, str) or not claim_ref:
        raise ExecutorError("completed ablation receipt has no terminal claim")
    claim_lexical = Path(claim_ref).expanduser()
    if claim_lexical.is_symlink() or not claim_lexical.is_file():
        raise ExecutorError("completed ablation terminal claim must be a regular file")
    try:
        claim_path = claim_lexical.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot resolve completed ablation terminal claim: {error}"
        ) from error
    claim_stat = claim_path.stat()
    if (
        stat.S_IMODE(claim_stat.st_mode) != 0o444
        or claim_stat.st_uid != receipt_path.stat().st_uid
    ):
        raise ExecutorError("completed ablation terminal claim mode/owner drift")
    contract_sha256 = payload.get("contract_sha256")
    claim_identity_sha256 = payload.get("claim_identity_sha256")
    if not isinstance(contract_sha256, str) or not isinstance(
        claim_identity_sha256, str
    ):
        raise ExecutorError("completed ablation receipt has no claim identity")
    claim = _load_claim_state(
        claim_path,
        contract_sha256=contract_sha256,
        claim_identity_sha256=claim_identity_sha256,
    )
    target_ref = claim.get("receipt_target")
    if not isinstance(target_ref, str) or not target_ref:
        raise ExecutorError("completed ablation terminal claim has no receipt target")
    target_lexical = Path(target_ref).expanduser()
    if target_lexical.is_symlink():
        raise ExecutorError("completed ablation terminal receipt target is a symlink")
    try:
        target_path = target_lexical.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot resolve completed ablation terminal receipt target: {error}"
        ) from error
    if (
        claim.get("schema_version") != ABLATION_CLAIM_SCHEMA
        or claim.get("status") != "complete"
        or payload.get("claim_state_sha256") != claim.get("state_sha256")
        or target_path != receipt_path
    ):
        raise ExecutorError("completed ablation terminal claim identity/status drift")
    claim_projection = dict(claim)
    claim_projection.pop("state_sha256", None)
    claim_projection.pop("receipt_target", None)
    claim_projection["schema_version"] = ABLATION_RECEIPT_SCHEMA
    receipt_projection = dict(payload)
    receipt_projection.pop("receipt_sha256", None)
    receipt_projection.pop("claim", None)
    receipt_projection.pop("claim_state_sha256", None)
    if claim_projection != receipt_projection:
        raise ExecutorError("completed ablation receipt differs from terminal claim")
    authenticated_started_unix_ns = claim.get("started_unix_ns")
    if isinstance(authenticated_started_unix_ns, bool) or not isinstance(
        authenticated_started_unix_ns, int
    ):
        raise ExecutorError("completed ablation claim lacks a valid start time")
    _replay_completed_ablation_receipt_authority(
        payload,
        claim_path=claim_path,
        authenticated_started_unix_ns=authenticated_started_unix_ns,
    )

    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ExecutorError("completed ablation receipt has no output artifacts")
    artifact_fields = (
        ("checkpoint", "checkpoint_sha256"),
        ("optimizer_sidecar", "optimizer_sidecar_sha256"),
        ("training_progress", "training_progress_sha256"),
        ("report", "report_sha256"),
    )
    resolved_artifacts: list[Path] = []
    for path_field, digest_field in artifact_fields:
        artifact_ref = outputs.get(path_field)
        expected_digest = outputs.get(digest_field)
        if not isinstance(artifact_ref, str) or not isinstance(expected_digest, str):
            raise ExecutorError(f"completed ablation output lacks {path_field} binding")
        artifact_lexical = Path(artifact_ref).expanduser()
        if artifact_lexical.is_symlink() or not artifact_lexical.is_file():
            raise ExecutorError(
                f"completed ablation {path_field} must be a regular file"
            )
        try:
            artifact_path = artifact_lexical.resolve(strict=True)
            actual_digest = _file_sha256(artifact_path)
        except OSError as error:
            raise ExecutorError(
                f"cannot authenticate completed ablation {path_field}: {error}"
            ) from error
        if actual_digest != expected_digest:
            raise ExecutorError(f"completed ablation {path_field} byte drift")
        resolved_artifacts.append(artifact_path)
    if len(set(resolved_artifacts)) != len(resolved_artifacts):
        raise ExecutorError("completed ablation output artifact paths are not distinct")
    return payload


def _replay_completed_aux_receipt_authority(
    payload: dict[str, Any], *, claim_path: Path, authenticated_started_unix_ns: int
) -> dict[str, Any]:
    """Replay the pinned lock/input chain and canonical derived claim path."""

    input_binding = payload.get("input_binding")
    learner_ablation = payload.get("learner_ablation")
    matched = (
        learner_ablation.get("matched_aux_regularization")
        if isinstance(learner_ablation, dict)
        else None
    )
    if not isinstance(input_binding, dict) or not isinstance(matched, dict):
        raise ExecutorError("matched aux receipt lacks input/ablation authority")
    lock_ref = payload.get("lock")
    corpus_ref = payload.get("corpus")
    reviewed_lock_sha256 = learner_ablation.get("reviewed_lock_file_sha256")
    if (
        not isinstance(lock_ref, str)
        or not isinstance(corpus_ref, str)
        or not isinstance(reviewed_lock_sha256, str)
        or reviewed_lock_sha256 != payload.get("lock_file_sha256")
    ):
        raise ExecutorError("matched aux receipt has no reviewed lock/data authority")
    lock_lexical = Path(lock_ref).expanduser()
    if lock_lexical.is_symlink() or not lock_lexical.is_file():
        raise ExecutorError("matched aux reviewed lock must be a regular file")
    try:
        lock_path = lock_lexical.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot resolve matched aux reviewed lock: {error}"
        ) from error
    if _file_sha256(lock_path) != reviewed_lock_sha256:
        raise ExecutorError("matched aux reviewed lock bytes drift")

    data_kind = input_binding.get("data_kind")
    if data_kind == "production_composite_v2":
        validation_path = None
        build_ref = input_binding.get("composite_build_receipt")
        build_path_value = (
            build_ref.get("path") if isinstance(build_ref, dict) else None
        )
        if not isinstance(build_path_value, str):
            raise ExecutorError("matched aux composite has no build receipt")
        composite_build_receipt = Path(build_path_value)
    else:
        validation_value = input_binding.get("validation_manifest")
        if not isinstance(validation_value, str):
            raise ExecutorError("matched aux corpus has no validation manifest")
        validation_path = Path(validation_value)
        composite_build_receipt = None
    replayed = verify_training_inputs(
        lock_path=lock_path,
        data_path=Path(corpus_ref),
        validation_path=validation_path,
        composite_build_receipt=composite_build_receipt,
        reviewed_lock_file_sha256=reviewed_lock_sha256,
    )
    arm = matched.get("arm_id")
    weight = matched.get("aux_subgoal_loss_weight")
    pair_authority = payload.get("aux_pair_executor_authority")
    published_authority = input_binding.get("central_published_executor_authority")
    if not isinstance(pair_authority, dict) or not isinstance(
        published_authority, dict
    ):
        raise ExecutorError("matched aux receipt lacks central pair authority")
    selected_weight = pair_authority.get("selected_aux_coefficient")
    expected_weight = (
        0.0
        if arm == AUX_CONTROL_ARM
        else selected_weight
        if arm == AUX_TREATMENT_ARM
        else None
    )
    if expected_weight is None or weight != expected_weight:
        raise ExecutorError("matched aux receipt arm/weight drift")
    try:
        reconstructed = bind_function_preserving_upgrade(
            replayed,
            Path(str(matched["upgrade_receipt"])),
            allow_public_award_transition_source=True,
        )
        reconstructed = bind_aux_pair_arm(
            reconstructed,
            authority=pair_authority,
            published_executor_authority=published_authority,
            warmed_initializer=Path(str(matched["initializer"])),
            reviewed_code_tree_sha256=str(learner_ablation["code_tree_sha256"]),
        )
        reconstructed = bind_training_topology(
            reconstructed, topology=B200_8GPU_DDP_TOPOLOGY, gpu=0
        )
        canary_ref = payload.get("ddp_canary")
        canary_path_value = (
            canary_ref.get("path") if isinstance(canary_ref, dict) else None
        )
        if not isinstance(canary_path_value, str):
            raise ExecutorError("matched aux receipt lacks canary/start authority")
        canary = _verify_ddp_canary_receipt(
            Path(canary_path_value),
            reference_time_ns=authenticated_started_unix_ns,
        )
        reconstructed = _bind_verified_ddp_canary(reconstructed, canary)
        reconstructed = bind_aux_subgoal_preclaim_contract(reconstructed)
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutorError(f"matched aux canonical replay refused: {error}") from error
    if (
        reconstructed.get("function_preserving_upgrade")
        != payload.get("function_preserving_upgrade")
        or reconstructed.get("aux_pair_executor_authority") != pair_authority
        or reconstructed.get("learner_ablation") != learner_ablation
        or reconstructed.get("training_topology") != payload.get("training_topology")
        or reconstructed.get("ddp_canary") != payload.get("ddp_canary")
        or reconstructed.get("aux_subgoal_preclaim_contract")
        != input_binding.get("aux_subgoal_preclaim_contract")
        or _input_binding(reconstructed) != input_binding
        or reconstructed.get("claim_identity_sha256")
        != payload.get("claim_identity_sha256")
    ):
        raise ExecutorError("matched aux canonical derived-state drift")
    if (
        payload.get("contract_sha256") != reconstructed["contract_sha256"]
        or payload.get("lock") != str(reconstructed["lock_path"])
        or payload.get("lock_file_sha256") != reconstructed["lock_file_sha256"]
        or payload.get("corpus") != str(reconstructed["data_path"])
        or payload.get("corpus_meta_file_sha256")
        != reconstructed["corpus_meta_file_sha256"]
        or payload.get("payload_inventory_sha256")
        != reconstructed["payload_inventory_sha256"]
        or payload.get("producer_checkpoint_sha256")
        != reconstructed["producer"]["sha256"]
    ):
        raise ExecutorError("matched aux receipt drifted from replayed lock/input")
    expected_claim_path = _claim_path(reconstructed).resolve(strict=False)
    if claim_path != expected_claim_path:
        raise ExecutorError("matched aux claim is not the canonical lock-derived path")
    outputs = payload.get("outputs")
    command = payload.get("command")
    if not isinstance(outputs, dict) or not isinstance(command, list) or not command:
        raise ExecutorError("matched aux receipt lacks command/output authority")
    checkpoint = Path(str(outputs.get("checkpoint", "")))
    report = Path(str(outputs.get("report", "")))
    canonical_command = build_train_command(
        reconstructed,
        python=Path(str(command[0])),
        checkpoint=checkpoint,
        report=report,
    )
    environment = _child_environment(_selected_gpus(reconstructed, fallback_gpu=0))
    execution_binding = _execution_binding(
        command=canonical_command, environment=environment
    )
    if (
        command != canonical_command
        or payload.get("command_sha256") != _value_sha256(canonical_command)
        or payload.get("execution_binding") != execution_binding
    ):
        raise ExecutorError("matched aux command/environment replay drift")
    reverified_outputs = _verify_training_outputs(
        checkpoint=checkpoint,
        report=report,
        verified=reconstructed,
        execution_binding=execution_binding,
        command=canonical_command,
    )
    if reverified_outputs != outputs:
        raise ExecutorError("matched aux canonical output verification drift")
    return reconstructed


def _load_authenticated_completed_aux_receipt(path: Path) -> dict[str, Any]:
    """Replay a completed AUX receipt against its terminal claim and outputs."""

    lexical = path.expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise ExecutorError("matched aux receipt must be a regular file")
    try:
        receipt_path = lexical.resolve(strict=True)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot load matched aux receipt: {error}") from error
    if not isinstance(payload, dict):
        raise ExecutorError("matched aux receipt is not an object")
    unsigned = dict(payload)
    stated = unsigned.pop("receipt_sha256", None)
    if (
        stated != _value_sha256(unsigned)
        or payload.get("schema_version") != ABLATION_RECEIPT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("diagnostic_only") is not True
        or payload.get("promotion_eligible") is not False
    ):
        raise ExecutorError("matched aux receipt schema/status/digest drift")

    claim_ref = payload.get("claim")
    if not isinstance(claim_ref, str) or not claim_ref:
        raise ExecutorError("matched aux receipt has no terminal claim")
    claim_lexical = Path(claim_ref).expanduser()
    if claim_lexical.is_symlink() or not claim_lexical.is_file():
        raise ExecutorError("matched aux terminal claim must be a regular file")
    try:
        claim_path = claim_lexical.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot resolve matched aux terminal claim: {error}"
        ) from error
    claim_stat = claim_path.stat()
    if (
        stat.S_IMODE(claim_stat.st_mode) != 0o444
        or claim_stat.st_uid != receipt_path.stat().st_uid
    ):
        raise ExecutorError("matched aux terminal claim mode/owner drift")
    contract_sha256 = payload.get("contract_sha256")
    claim_identity_sha256 = payload.get("claim_identity_sha256")
    if not isinstance(contract_sha256, str) or not isinstance(
        claim_identity_sha256, str
    ):
        raise ExecutorError("matched aux receipt has no claim identity")
    claim = _load_claim_state(
        claim_path,
        contract_sha256=contract_sha256,
        claim_identity_sha256=claim_identity_sha256,
    )
    target_ref = claim.get("receipt_target")
    if not isinstance(target_ref, str) or not target_ref:
        raise ExecutorError("matched aux terminal claim has no receipt target")
    target_lexical = Path(target_ref).expanduser()
    if target_lexical.is_symlink():
        raise ExecutorError("matched aux terminal receipt target is a symlink")
    try:
        target_path = target_lexical.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot resolve matched aux terminal receipt target: {error}"
        ) from error
    if (
        claim.get("schema_version") != ABLATION_CLAIM_SCHEMA
        or claim.get("status") != "complete"
        or payload.get("claim_state_sha256") != claim.get("state_sha256")
        or target_path != receipt_path
    ):
        raise ExecutorError("matched aux terminal claim identity/status drift")

    # The claim is written first and the receipt merely adds its exact claim
    # reference.  Replaying this projection prevents a self-digested fabricated
    # receipt from masquerading as an executor-completed dose.
    claim_projection = dict(claim)
    claim_projection.pop("state_sha256", None)
    claim_projection.pop("receipt_target", None)
    claim_projection["schema_version"] = ABLATION_RECEIPT_SCHEMA
    receipt_projection = dict(payload)
    receipt_projection.pop("receipt_sha256", None)
    receipt_projection.pop("claim", None)
    receipt_projection.pop("claim_state_sha256", None)
    if claim_projection != receipt_projection:
        raise ExecutorError("matched aux receipt differs from its terminal claim")
    authenticated_started_unix_ns = claim.get("started_unix_ns")
    if isinstance(authenticated_started_unix_ns, bool) or not isinstance(
        authenticated_started_unix_ns, int
    ):
        raise ExecutorError("matched aux terminal claim lacks a valid start time")
    _replay_completed_aux_receipt_authority(
        payload,
        claim_path=claim_path,
        authenticated_started_unix_ns=authenticated_started_unix_ns,
    )

    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ExecutorError("matched aux receipt has no output artifacts")
    artifact_fields = (
        ("checkpoint", "checkpoint_sha256"),
        ("optimizer_sidecar", "optimizer_sidecar_sha256"),
        ("training_progress", "training_progress_sha256"),
        ("report", "report_sha256"),
    )
    resolved_artifacts: list[Path] = []
    for path_field, digest_field in artifact_fields:
        artifact_ref = outputs.get(path_field)
        expected_digest = outputs.get(digest_field)
        if not isinstance(artifact_ref, str) or not isinstance(expected_digest, str):
            raise ExecutorError(f"matched aux output lacks {path_field} binding")
        artifact_lexical = Path(artifact_ref).expanduser()
        if artifact_lexical.is_symlink() or not artifact_lexical.is_file():
            raise ExecutorError(f"matched aux {path_field} must be a regular file")
        try:
            artifact_path = artifact_lexical.resolve(strict=True)
            actual_digest = _file_sha256(artifact_path)
        except OSError as error:
            raise ExecutorError(
                f"cannot authenticate matched aux {path_field}: {error}"
            ) from error
        if actual_digest != expected_digest:
            raise ExecutorError(f"matched aux {path_field} byte drift")
        resolved_artifacts.append(artifact_path)
    if len(set(resolved_artifacts)) != len(resolved_artifacts):
        raise ExecutorError("matched aux output artifact paths are not distinct")
    return payload


def verify_matched_aux_receipt_pair(
    control_receipt: Path, treatment_receipt: Path
) -> dict[str, Any]:
    """Prove two authenticated completed receipts form one AUX0/AUXT pair."""

    control = _load_authenticated_completed_aux_receipt(control_receipt)
    treatment = _load_authenticated_completed_aux_receipt(treatment_receipt)
    by_arm: dict[str, dict[str, Any]] = {}
    for payload in (control, treatment):
        ablation = payload.get("learner_ablation")
        matched = (
            ablation.get("matched_aux_regularization")
            if isinstance(ablation, dict)
            else None
        )
        arm = matched.get("arm_id") if isinstance(matched, dict) else None
        if arm not in {AUX_CONTROL_ARM, AUX_TREATMENT_ARM} or arm in by_arm:
            raise ExecutorError("matched aux receipts do not contain one AUX0 and AUXT")
        by_arm[str(arm)] = payload
    if set(by_arm) != {AUX_CONTROL_ARM, AUX_TREATMENT_ARM}:
        raise ExecutorError("matched aux receipt pair is incomplete")
    control = by_arm[AUX_CONTROL_ARM]
    treatment = by_arm[AUX_TREATMENT_ARM]
    control_ablation = control["learner_ablation"]
    treatment_ablation = treatment["learner_ablation"]
    control_match = control_ablation["matched_aux_regularization"]
    treatment_match = treatment_ablation["matched_aux_regularization"]
    control_authority = control.get("aux_pair_executor_authority")
    treatment_authority = treatment.get("aux_pair_executor_authority")
    if not isinstance(control_authority, dict) or not isinstance(
        treatment_authority, dict
    ):
        raise ExecutorError("matched aux receipts lack central pair authority")
    control_pair = control_authority.get("aux_pair_contract")
    treatment_pair = treatment_authority.get("aux_pair_contract")
    selected_weight = control_authority.get("selected_aux_coefficient")
    if (
        control_match["aux_subgoal_loss_weight"] != 0.0
        or treatment_match["aux_subgoal_loss_weight"] != selected_weight
        or not isinstance(control_pair, dict)
        or control_pair != treatment_pair
        or control_authority.get("arm", {}).get("arm_id") != AUX_CONTROL_ARM
        or treatment_authority.get("arm", {}).get("arm_id") != AUX_TREATMENT_ARM
        or control_match["shared_identity"] != treatment_match["shared_identity"]
        or control_match["shared_identity_sha256"]
        != treatment_match["shared_identity_sha256"]
        or control.get("function_preserving_upgrade")
        != treatment.get("function_preserving_upgrade")
        or control.get("learner_training_recipe_sha256")
        != treatment.get("learner_training_recipe_sha256")
    ):
        raise ExecutorError("AUX0/AUXT initializer or shared scientific identity drift")
    control_recipe = control_ablation.get("effective_recipe")
    treatment_recipe = treatment_ablation.get("effective_recipe")
    if not isinstance(control_recipe, dict) or not isinstance(treatment_recipe, dict):
        raise ExecutorError("matched aux receipts lack effective recipes")
    recipe_delta = {
        key
        for key in set(control_recipe) | set(treatment_recipe)
        if control_recipe.get(key) != treatment_recipe.get(key)
    }
    if recipe_delta != {"aux_subgoal_loss_weight"}:
        raise ExecutorError(
            "matched aux effective recipes differ outside aux_subgoal_loss_weight"
        )
    control_input = dict(control.get("input_binding", {}))
    treatment_input = dict(treatment.get("input_binding", {}))
    control_input.pop("binding_sha256", None)
    treatment_input.pop("binding_sha256", None)
    control_input.pop("effective_learner_recipe_sha256", None)
    treatment_input.pop("effective_learner_recipe_sha256", None)
    control_canary = control_input.pop("ddp_canary", None)
    treatment_canary = treatment_input.pop("ddp_canary", None)
    control_aux_preflight = control_input.pop("aux_subgoal_preclaim_contract", None)
    treatment_aux_preflight = treatment_input.pop("aux_subgoal_preclaim_contract", None)
    control_semantics = (
        control_canary.get("semantic_identity")
        if isinstance(control_canary, dict)
        else None
    )
    treatment_semantics = (
        treatment_canary.get("semantic_identity")
        if isinstance(treatment_canary, dict)
        else None
    )
    shared_semantics = control_match["shared_identity"].get("ddp_canary_semantics")
    control_admission = (
        control_aux_preflight.get("treatment_grade_admission")
        if isinstance(control_aux_preflight, dict)
        else None
    )
    treatment_admission = (
        treatment_aux_preflight.get("treatment_grade_admission")
        if isinstance(treatment_aux_preflight, dict)
        else None
    )
    shared_admission = control_match["shared_identity"].get(
        "aux_subgoal_treatment_admission"
    )
    if (
        control_input != treatment_input
        or control_input.get("training_topology", {}).get("name")
        != B200_8GPU_DDP_TOPOLOGY
        or control_input.get("training_topology", {}).get("world_size") != 8
        or control_input.get("training_topology", {}).get("global_batch_size") != 4096
        or not isinstance(control_canary, dict)
        or not isinstance(treatment_canary, dict)
        or control_semantics != treatment_semantics
        or control_semantics != shared_semantics
        or control_canary.get("semantic_identity_sha256")
        != _value_sha256(control_semantics)
        or treatment_canary.get("semantic_identity_sha256")
        != _value_sha256(treatment_semantics)
        or control_match["shared_identity"].get("ddp_canary_semantics_sha256")
        != _value_sha256(shared_semantics)
        or not isinstance(control_aux_preflight, dict)
        or control_aux_preflight.get("arm_loss_weight") != 0.0
        or control_aux_preflight.get("admission_loss_weight") != selected_weight
        or not isinstance(treatment_aux_preflight, dict)
        or treatment_aux_preflight.get("arm_loss_weight") != selected_weight
        or treatment_aux_preflight.get("admission_loss_weight") != selected_weight
        or control_admission != treatment_admission
        or control_admission != shared_admission
        or control_aux_preflight.get("treatment_grade_admission_sha256")
        != _value_sha256(control_admission)
        or treatment_aux_preflight.get("treatment_grade_admission_sha256")
        != _value_sha256(treatment_admission)
        or control_match["shared_identity"].get(
            "aux_subgoal_treatment_admission_sha256"
        )
        != _value_sha256(shared_admission)
    ):
        raise ExecutorError("matched aux data/split/topology/DDP authority drift")
    if control.get("lineage_dose") != treatment.get("lineage_dose"):
        raise ExecutorError("matched aux sample/optimizer dose drift")
    outputs_control = control.get("outputs")
    outputs_treatment = treatment.get("outputs")
    dose_fields = (
        "steps_completed",
        "unique_training_rows",
        "base_sampler_draw_events",
        "sampler_draw_events",
        "sampled_rows",
        "training_progress_payload_sha256",
        "corpus_row_count",
        "training_row_count",
        "validation_row_count",
        "production_sampling_receipt_sha256",
        "validation_split_receipt_sha256",
    )
    if not isinstance(outputs_control, dict) or not isinstance(outputs_treatment, dict):
        raise ExecutorError("matched aux receipts lack output-dose evidence")
    if any(
        outputs_control.get(field) != outputs_treatment.get(field)
        for field in dose_fields
    ):
        raise ExecutorError("matched aux realized sampler/dose evidence drift")
    if (
        outputs_control.get("steps_completed") != AUX_SELECTED_OPTIMIZER_STEPS
        or outputs_control.get("sampled_rows") != AUX_SELECTED_SAMPLE_DOSE
        or outputs_control.get("unique_training_rows") is not None
    ):
        raise ExecutorError("matched aux outputs do not prove the selected short dose")
    for payload in (control, treatment):
        command = payload.get("command")
        if (
            not isinstance(command, list)
            or command.count("torch.distributed.run") != 1
            or command.count("--nproc_per_node=8") != 1
            or command.count("--no-resume-optimizer") != 1
            or "--resume-optimizer" in command
            or command.count("--aux-subgoal-heads") != 1
            or command.count("--aux-settlement-pointer-head") != 1
        ):
            raise ExecutorError("matched aux command topology/fresh-Adam drift")
    return {
        "schema_version": "a1-matched-aux-pointer-pair-v1",
        "passed": True,
        "shared_identity_sha256": control_match["shared_identity_sha256"],
        "initializer_sha256": control_match["initializer_sha256"],
        "world_size": 8,
        "global_batch_size": 4096,
        "sampled_rows": outputs_control["sampled_rows"],
        "optimizer_steps": outputs_control["steps_completed"],
    }


def _require_fresh_outputs(
    checkpoint: Path,
    report: Path,
    receipt: Path,
    *,
    claim: Path | None = None,
) -> None:
    paths = (
        checkpoint,
        Path(str(checkpoint) + ".optimizer.pt"),
        Path(str(checkpoint) + ".training-progress.json"),
        report,
        receipt,
    )
    if len(set(paths)) != len(paths):
        raise ExecutorError(
            "checkpoint, optimizer/progress sidecars, report, and receipt paths "
            "must be distinct"
        )
    if claim is not None and claim in paths:
        raise ExecutorError(
            "checkpoint, optimizer sidecar, report, and receipt must be distinct "
            f"from the sealed-contract claim path: {claim}"
        )
    for path in paths:
        if path.exists() or path.is_symlink():
            raise ExecutorError(f"refusing non-fresh A1 output path: {path}")


def _canonical_one_dose_output_namespace(
    *,
    checkpoint: Path,
    report: Path,
    receipt: Path,
    one_dose_claim: Path,
) -> dict[str, str]:
    """Canonicalize the fresh output namespace without following leaf symlinks."""

    def canonical(path: Path, where: str) -> Path:
        lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
        if lexical.is_symlink():
            raise ExecutorError(f"{where} output may not be a symlink")
        try:
            parent = lexical.parent.resolve(strict=False)
        except OSError as error:
            raise ExecutorError(
                f"{where} output parent is unavailable: {error}"
            ) from error
        if lexical.parent != parent:
            raise ExecutorError(f"{where} output parent may not be a symlink")
        return parent / lexical.name

    checkpoint_path = canonical(checkpoint, "checkpoint")
    report_path = canonical(report, "report")
    receipt_path = canonical(receipt, "receipt")
    claim_path = canonical(one_dose_claim, "one-dose claim")
    namespace = {
        "checkpoint": str(checkpoint_path),
        "optimizer_sidecar": str(
            canonical(Path(str(checkpoint_path) + ".optimizer.pt"), "optimizer sidecar")
        ),
        "training_progress": str(
            canonical(
                Path(str(checkpoint_path) + ".training-progress.json"),
                "training progress",
            )
        ),
        "report": str(report_path),
        "receipt": str(receipt_path),
        "one_dose_claim": str(claim_path),
    }
    if namespace["one_dose_claim"] in {
        value for key, value in namespace.items() if key != "one_dose_claim"
    }:
        raise ExecutorError("one-dose output aliases the sealed claim path")
    if len(set(namespace.values())) != len(namespace):
        raise ExecutorError("one-dose output namespace contains aliases")
    return namespace


def _require_final_matched_aux_command_binding(
    verified: dict[str, Any],
    command: list[str],
    *,
    checkpoint: Path,
    report: Path,
) -> None:
    """Refuse stale direct-library commands before the one-dose claim."""

    matched = verified.get("learner_ablation", {}).get("matched_aux_regularization")
    if not isinstance(matched, dict):
        return
    flag = "--a1-aux-regularization-binding-json"
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ExecutorError("matched AUX command lacks one final binding")
    try:
        command_binding = json.loads(command[positions[0] + 1])
    except json.JSONDecodeError as error:
        raise ExecutorError("matched AUX command binding is invalid JSON") from error
    if command_binding != matched:
        raise ExecutorError(
            "matched AUX command was built before final pre-claim/canary identity"
        )
    if not command or not isinstance(command[0], str):
        raise ExecutorError("matched AUX command has no Python executable")
    rebuilt = build_train_command(
        verified,
        python=Path(command[0]),
        checkpoint=checkpoint,
        report=report,
    )
    if command != rebuilt:
        raise ExecutorError("matched AUX command differs from canonical final argv")


def _one_dose_claim_and_receipt_schemas(
    verified: Mapping[str, Any],
) -> tuple[str, str]:
    if "retry_contract" in verified:
        return RETRY_CLAIM_SCHEMA, RETRY_RECEIPT_SCHEMA
    if isinstance(verified.get("central_learner_binding"), dict):
        return CENTRAL_CLAIM_SCHEMA, CENTRAL_RECEIPT_SCHEMA
    if verified.get("learner_ablation") is not None:
        return ABLATION_CLAIM_SCHEMA, ABLATION_RECEIPT_SCHEMA
    if verified.get("function_preserving_upgrade") is not None:
        return UPGRADE_CLAIM_SCHEMA, UPGRADE_RECEIPT_SCHEMA
    return CLAIM_SCHEMA, RECEIPT_SCHEMA


def _recover_terminal_complete_receipt(
    *,
    verified: dict[str, Any],
    command: list[str],
    checkpoint: Path,
    report: Path,
    receipt: Path,
    gpu: int,
    publish: bool,
) -> dict[str, Any] | None:
    """Recover only the receipt-publication gap after a proven completion.

    This never resumes or re-runs training.  Recovery is available only when
    the durable claim is already terminal-complete, the receipt is absent, and
    the current command/input/output replay reproduces every scientific
    binding and output digest recorded in that claim.
    """

    claim_path = _claim_path(verified)
    if not claim_path.exists() and not claim_path.is_symlink():
        return None
    if claim_path.is_symlink() or not claim_path.is_file():
        raise ExecutorError("terminal completion recovery claim must be a regular file")
    claim_stat = claim_path.stat()
    if stat.S_IMODE(claim_stat.st_mode) != 0o444:
        raise ExecutorError("terminal completion recovery claim mode drift")
    claim_identity = str(
        verified.get("claim_identity_sha256", verified["contract_sha256"])
    )
    claim = _load_claim_state(
        claim_path,
        contract_sha256=str(verified["contract_sha256"]),
        claim_identity_sha256=claim_identity,
    )
    if claim.get("status") != "complete":
        return None
    if receipt.exists() or receipt.is_symlink():
        return None

    expected_claim_schema, receipt_schema = _one_dose_claim_and_receipt_schemas(
        verified
    )
    output_namespace = _canonical_one_dose_output_namespace(
        checkpoint=checkpoint,
        report=report,
        receipt=receipt,
        one_dose_claim=claim_path,
    )
    # A different output namespace is a forbidden second dose, not a recovery
    # attempt.  Let the ordinary claim path produce its established refusal.
    if claim.get("receipt_target") != output_namespace["receipt"]:
        return None
    selected_gpus = _selected_gpus(verified, fallback_gpu=gpu)
    environment = _child_environment(selected_gpus)
    execution_binding = _execution_binding(command=command, environment=environment)
    input_binding = _input_binding(verified)
    transaction_sha256 = _training_transaction_sha256(
        command=command, input_binding=input_binding
    )
    outputs = claim.get("outputs")
    expected_static = {
        "schema_version": expected_claim_schema,
        "contract_sha256": verified["contract_sha256"],
        "status": "complete",
        "lock": str(verified["lock_path"]),
        "lock_file_sha256": verified["lock_file_sha256"],
        "corpus": str(verified["data_path"]),
        "corpus_meta_file_sha256": verified["corpus_meta_file_sha256"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "validation_manifest": str(verified["validation_path"]),
        "validation_manifest_file_sha256": verified["validation_file_sha256"],
        "producer_checkpoint_sha256": verified["producer"]["sha256"],
        "learner_lineage_parent": copy.deepcopy(verified.get("learner_lineage_parent")),
        "learner_training_recipe_sha256": _value_sha256(
            verified.get("bound_recipe", verified["recipe"])
        ),
        "command": command,
        "command_sha256": _value_sha256(command),
        "execution_binding": execution_binding,
        "input_binding": input_binding,
        "training_transaction_sha256": transaction_sha256,
        "trainer_authority": verified.get("trainer_authority"),
        "lock_verifier_authority": verified.get("lock_verifier_authority"),
        "world_size": int(verified["recipe"]["world_size"]),
        "gpu": gpu,
        "gpus": list(selected_gpus),
        "training_topology": verified.get("training_topology"),
        "ddp_canary": verified.get("ddp_canary"),
        "production_sampling_receipt_sha256": verified.get(
            "production_sampling_receipt_sha256"
        ),
        "validation_split_receipt_sha256": verified.get(
            "validation_split_receipt_sha256"
        ),
        "returncode": 0,
        "failure": None,
        "receipt_target": output_namespace["receipt"],
    }
    drift = {
        key: {"expected": value, "actual": claim.get(key)}
        for key, value in expected_static.items()
        if claim.get(key) != value
    }
    if drift:
        raise ExecutorError(f"terminal completion recovery binding drift: {drift}")
    if (
        isinstance(claim.get("started_unix_ns"), bool)
        or not isinstance(claim.get("started_unix_ns"), int)
        or isinstance(claim.get("finished_unix_ns"), bool)
        or not isinstance(claim.get("finished_unix_ns"), int)
        or int(claim["finished_unix_ns"]) < int(claim["started_unix_ns"])
        or not isinstance(claim.get("gpu_name"), str)
        or not isinstance(claim.get("gpu_names"), list)
        or len(claim["gpu_names"]) != len(selected_gpus)
        or not isinstance(outputs, dict)
    ):
        raise ExecutorError(
            "terminal completion recovery runtime/output evidence drift"
        )
    if (
        outputs.get("checkpoint") != output_namespace["checkpoint"]
        or outputs.get("optimizer_sidecar") != output_namespace["optimizer_sidecar"]
        or outputs.get("training_progress") != output_namespace["training_progress"]
        or outputs.get("report") != output_namespace["report"]
    ):
        raise ExecutorError("terminal completion recovery output namespace drift")

    if claim_identity != str(verified["contract_sha256"]):
        if claim.get("claim_identity_sha256") != claim_identity:
            raise ExecutorError("terminal completion recovery derived identity drift")
    if "retry_contract" in verified:
        expected_retry = {
            "path": str(verified["retry_contract_path"]),
            "file_sha256": verified["retry_contract_file_sha256"],
            "retry_contract_sha256": verified["retry_contract"][
                "retry_contract_sha256"
            ],
        }
        if claim.get("retry_contract") != expected_retry:
            raise ExecutorError("terminal completion recovery retry authority drift")

    reverified_outputs = _verify_training_outputs(
        checkpoint=checkpoint,
        report=report,
        verified=verified,
        execution_binding=execution_binding,
        command=command,
    )
    if reverified_outputs != outputs or claim.get("lineage_dose") != outputs.get(
        "lineage_dose"
    ):
        raise ExecutorError("terminal completion recovery output evidence drift")

    evidence = dict(claim)
    evidence.pop("state_sha256", None)
    evidence.pop("receipt_target", None)
    evidence["schema_version"] = receipt_schema
    evidence["claim"] = str(claim_path)
    evidence["claim_state_sha256"] = claim["state_sha256"]
    if not publish:
        return evidence
    return _write_receipt_no_clobber(receipt, evidence)


def _execute_locked(
    *,
    verified: dict[str, Any],
    command: list[str],
    checkpoint: Path,
    report: Path,
    receipt: Path,
    gpu: int,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    probe: Callable[[int], str] = _probe_b200,
    runtime_probe: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Claim, execute, verify, and atomically receipt exactly one A1 dose."""

    _require_current_production_trainer_authority(verified, command=command)

    matched_aux = verified.get("learner_ablation", {}).get("matched_aux_regularization")
    if isinstance(matched_aux, dict) and not isinstance(
        verified.get("aux_subgoal_preclaim_contract"), dict
    ):
        raise ExecutorError(
            "matched AUX admission must be bound before command construction"
        )
    # Re-run the canonical admission against live bytes, but never silently
    # change scientific identity after a caller has rendered the command.
    replayed = bind_aux_subgoal_preclaim_contract(verified)
    if (
        replayed.get("aux_subgoal_preclaim_contract")
        != verified.get("aux_subgoal_preclaim_contract")
        or replayed.get("learner_ablation") != verified.get("learner_ablation")
        or replayed.get("claim_identity_sha256")
        != verified.get("claim_identity_sha256")
    ):
        raise ExecutorError(
            "matched AUX verified state is not final; prepare it before command build"
        )
    _require_final_matched_aux_command_binding(
        verified, command, checkpoint=checkpoint, report=report
    )
    claim = _claim_path(verified)
    for parent in {checkpoint.parent, report.parent, receipt.parent, claim.parent}:
        _mkdir_durable(parent)
    # Descriptor-authorized ablations are immutable inputs, not mutable corpus
    # edits. Publish the exact planned bytes before hardware acquisition and
    # before the irreversible one-dose claim.
    _materialize_diagnostic_training_descriptor(verified)
    output_namespace = _canonical_one_dose_output_namespace(
        checkpoint=checkpoint,
        report=report,
        receipt=receipt,
        one_dose_claim=claim,
    )
    if (
        output_namespace["checkpoint"] != str(checkpoint)
        or output_namespace["report"] != str(report)
        or output_namespace["receipt"] != str(receipt)
        or output_namespace["one_dose_claim"] != str(claim)
    ):
        raise ExecutorError("one-dose output paths must be canonical absolute paths")
    _require_fresh_outputs(checkpoint, report, receipt, claim=claim)
    # Hardware refusal must precede the durable one-dose claim. Occupancy or an
    # MPS daemon is an operational precondition failure, not a consumed dose.
    selected_gpus = _selected_gpus(verified, fallback_gpu=gpu)
    gpu_names = [probe(selected_gpu) for selected_gpu in selected_gpus]
    _verify_central_live_allocation(
        verified, selected_gpus=selected_gpus, runtime_probe=runtime_probe
    )
    child_environment = _child_environment(selected_gpus)
    execution_binding = _execution_binding(
        command=command, environment=child_environment
    )
    input_binding = _input_binding(verified)
    training_transaction_sha256 = _training_transaction_sha256(
        command=command, input_binding=input_binding
    )
    started_ns = time.time_ns()
    claim_identity = str(
        verified.get("claim_identity_sha256", verified["contract_sha256"])
    )
    is_retry = "retry_contract" in verified
    is_ablation = verified.get("learner_ablation") is not None
    is_upgrade = verified.get("function_preserving_upgrade") is not None
    is_central = isinstance(verified.get("central_learner_binding"), dict)
    central_execution_commitment = None
    if is_central:
        try:
            central_execution_commitment = (
                aux_coordinator.commit_central_learner_execution(
                    published_executor_authority=verified[
                        "central_published_executor_authority"
                    ],
                    command=command,
                    environment=child_environment,
                    output_namespace=output_namespace,
                    central_binding=verified["central_learner_binding"],
                    input_binding=input_binding,
                    one_dose_claim_identity_sha256=claim_identity,
                    aux_regularization_binding=(
                        verified.get("learner_ablation", {}).get(
                            "matched_aux_regularization"
                        )
                        if is_central
                        else None
                    ),
                )
            )
        except (KeyError, aux_coordinator.CoordinatorError) as error:
            raise ExecutorError(
                f"central learner execution commitment refused: {error}"
            ) from error
    retry_reference = (
        {
            "path": str(verified["retry_contract_path"]),
            "file_sha256": verified["retry_contract_file_sha256"],
            "retry_contract_sha256": verified["retry_contract"][
                "retry_contract_sha256"
            ],
        }
        if is_retry
        else None
    )
    claim_payload = {
        "schema_version": (
            RETRY_CLAIM_SCHEMA
            if is_retry
            else CENTRAL_CLAIM_SCHEMA
            if is_central
            else ABLATION_CLAIM_SCHEMA
            if is_ablation
            else UPGRADE_CLAIM_SCHEMA
            if is_upgrade
            else CLAIM_SCHEMA
        ),
        "status": "claimed",
        "contract_sha256": verified["contract_sha256"],
        "command_sha256": _value_sha256(command),
        "execution_binding": execution_binding,
        "input_binding": input_binding,
        "training_transaction_sha256": training_transaction_sha256,
        "started_unix_ns": started_ns,
    }
    if is_retry or is_central or is_ablation or is_upgrade:
        claim_payload["claim_identity_sha256"] = claim_identity
    if is_retry:
        claim_payload.update(
            {
                "claim_identity_sha256": claim_identity,
                "retry_contract": retry_reference,
            }
        )
    if is_central:
        claim_payload["central_execution_commitment"] = central_execution_commitment
    claim = _claim_attempt(verified, claim_payload)
    status = "failed"
    returncode: int | None = None
    output_artifacts: dict[str, Any] | None = None
    failure: str | None = None
    try:
        _mkdir_durable(checkpoint.parent)
        _mkdir_durable(report.parent)
        _mkdir_durable(receipt.parent)
        result = runner(
            command,
            cwd=str(_REPO_ROOT),
            env=child_environment,
            check=False,
            preexec_fn=_raise_nofile_limit,
        )
        returncode = int(result.returncode)
        if returncode != 0:
            raise ExecutorError(f"train_bc exited nonzero: {returncode}")
        _bind_training_report(
            report,
            verified=verified,
            execution_binding=execution_binding,
        )
        output_artifacts = _verify_training_outputs(
            checkpoint=checkpoint,
            report=report,
            verified=verified,
            execution_binding=execution_binding,
            command=command,
        )
        immutable_outputs = [
            checkpoint,
            Path(str(checkpoint) + ".optimizer.pt"),
            Path(str(checkpoint) + ".training-progress.json"),
            report,
        ]
        immutable_outputs.extend(
            Path(str(record["checkpoint"]))
            for record in output_artifacts.get("intermediate_checkpoints", [])
        )
        if "validation_seed_manifest" in output_artifacts:
            immutable_outputs.append(
                Path(str(output_artifacts["validation_seed_manifest"]))
            )
        for output_path in immutable_outputs:
            os.chmod(output_path, 0o444)
        for parent in {checkpoint.parent, report.parent}:
            _fsync_directory(parent)
        status = "complete"
    except Exception as error:  # receipt every claimed attempt, then re-raise.
        failure = f"{type(error).__name__}: {error}"
    finished_ns = time.time_ns()
    evidence_payload = {
        "schema_version": (
            RETRY_RECEIPT_SCHEMA
            if is_retry
            else CENTRAL_RECEIPT_SCHEMA
            if is_central
            else ABLATION_RECEIPT_SCHEMA
            if is_ablation
            else UPGRADE_RECEIPT_SCHEMA
            if is_upgrade
            else RECEIPT_SCHEMA
        ),
        "status": status,
        "contract_sha256": verified["contract_sha256"],
        "lock": str(verified["lock_path"]),
        "lock_file_sha256": verified["lock_file_sha256"],
        "corpus": str(verified["data_path"]),
        "corpus_meta_file_sha256": verified["corpus_meta_file_sha256"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "validation_manifest": str(verified["validation_path"]),
        "validation_manifest_file_sha256": verified["validation_file_sha256"],
        "producer_checkpoint_sha256": verified["producer"]["sha256"],
        "learner_lineage_parent": copy.deepcopy(verified.get("learner_lineage_parent")),
        "learner_training_recipe_sha256": _value_sha256(
            verified.get("bound_recipe", verified["recipe"])
        ),
        "command": command,
        "command_sha256": _value_sha256(command),
        "execution_binding": execution_binding,
        "input_binding": input_binding,
        "training_transaction_sha256": training_transaction_sha256,
        "trainer_authority": verified.get("trainer_authority"),
        "lock_verifier_authority": verified.get("lock_verifier_authority"),
        "world_size": int(verified["recipe"]["world_size"]),
        "gpu": gpu,
        "gpus": list(selected_gpus),
        "gpu_name": gpu_names[0],
        "gpu_names": gpu_names,
        "training_topology": verified.get("training_topology"),
        "ddp_canary": verified.get("ddp_canary"),
        "production_sampling_receipt_sha256": verified.get(
            "production_sampling_receipt_sha256"
        ),
        "validation_split_receipt_sha256": verified.get(
            "validation_split_receipt_sha256"
        ),
        "started_unix_ns": started_ns,
        "finished_unix_ns": finished_ns,
        "returncode": returncode,
        "outputs": output_artifacts,
        "lineage_dose": (
            None if output_artifacts is None else output_artifacts["lineage_dose"]
        ),
        "failure": failure,
    }
    if is_retry:
        evidence_payload.update(
            {
                "claim_identity_sha256": claim_identity,
                "retry_contract": retry_reference,
            }
        )
    if is_central:
        evidence_payload.update(
            {
                "claim_identity_sha256": claim_identity,
                "central_learner_binding": verified["central_learner_binding"],
                "central_published_executor_authority": verified.get(
                    "central_published_executor_authority"
                ),
                "central_execution_commitment": central_execution_commitment,
            }
        )
    if is_ablation:
        evidence_payload.update(
            {
                "claim_identity_sha256": claim_identity,
                "learner_ablation": verified["learner_ablation"],
                "effective_learner_training_recipe_sha256": _value_sha256(
                    verified["recipe"]
                ),
                "diagnostic_only": True,
                "promotion_eligible": False,
                "aux_pair_executor_authority": verified.get(
                    "aux_pair_executor_authority"
                ),
            }
        )
    if is_upgrade:
        evidence_payload["claim_identity_sha256"] = claim_identity
        evidence_payload["function_preserving_upgrade"] = verified[
            "function_preserving_upgrade"
        ]
    terminal_claim_payload = dict(evidence_payload)
    terminal_claim_payload["schema_version"] = (
        RETRY_CLAIM_SCHEMA
        if is_retry
        else CENTRAL_CLAIM_SCHEMA
        if is_central
        else ABLATION_CLAIM_SCHEMA
        if is_ablation
        else UPGRADE_CLAIM_SCHEMA
        if is_upgrade
        else CLAIM_SCHEMA
    )
    terminal_claim_payload["receipt_target"] = str(receipt)
    terminal_claim = _write_terminal_claim(
        claim,
        terminal_claim_payload,
        contract_sha256=str(verified["contract_sha256"]),
        claim_identity_sha256=claim_identity,
    )
    evidence_payload["claim"] = str(claim)
    evidence_payload["claim_state_sha256"] = terminal_claim["state_sha256"]
    receipt_payload = _write_receipt_no_clobber(receipt, evidence_payload)
    if status != "complete":
        raise ExecutorError(failure or "A1 training failed")
    return receipt_payload


def execute(
    *,
    verified: dict[str, Any],
    command: list[str],
    checkpoint: Path,
    report: Path,
    receipt: Path,
    gpu: int,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    probe: Callable[[int], str] = _probe_b200,
) -> dict[str, Any]:
    """Execute one dose while owning every physical GPU in its topology."""

    recovered = _recover_terminal_complete_receipt(
        verified=verified,
        command=command,
        checkpoint=checkpoint,
        report=report,
        receipt=receipt,
        gpu=gpu,
        publish=True,
    )
    if recovered is not None:
        return recovered

    kwargs = {
        "verified": verified,
        "command": command,
        "checkpoint": checkpoint,
        "report": report,
        "receipt": receipt,
        "gpu": gpu,
        "runner": runner,
        "probe": probe,
    }
    selected_gpus = _selected_gpus(verified, fallback_gpu=gpu)
    with ExitStack() as locks:
        for selected_gpu in selected_gpus:
            locks.enter_context(_physical_gpu_lock(selected_gpu))
        return _execute_locked(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=None,
        help=(
            "required for an ordinary A1 memmap; forbidden for a promotion-eligible "
            "flywheel composite, which binds its own whole-game split"
        ),
    )
    parser.add_argument(
        "--composite-build-receipt",
        type=Path,
        default=None,
        help=(
            "required atomic builder receipt for a promotion-eligible flywheel "
            "composite; forbidden for ordinary A1 memmaps"
        ),
    )
    parser.add_argument(
        "--coherent-corpus-admission",
        type=Path,
        default=None,
        help=(
            "signed admission for one direct coherent-public memmap; diagnostic "
            "overlays require an ablation and independent FINAL slices require "
            "--stage-c-final-authority"
        ),
    )
    parser.add_argument(
        "--stage-c-final-authority",
        type=Path,
        default=None,
        help=(
            "sealed independent Stage-C current-parent replication authority; "
            "selected diagnostic checkpoint bytes are never loaded"
        ),
    )
    parser.add_argument(
        "--stage-c-final-arm",
        choices=stage_c_final.FINAL_ARM_NAMES,
        default=None,
        help="required matched control/treatment arm for --stage-c-final-authority",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--architecture-upgrade-receipt",
        type=Path,
        default=None,
        help=(
            "immutable allowlisted zero-diff initializer receipt; exact producer "
            "bytes remain the default"
        ),
    )
    parser.add_argument(
        "--independent-parent-authority",
        type=Path,
        default=None,
        help=(
            "diagnostic-only authority separating an independently selected "
            "learner parent from the checkpoint that produced the corpus"
        ),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="physical B200 index (8-GPU topology requires 0 and owns GPUs 0-7)",
    )
    parser.add_argument(
        "--topology",
        choices=sorted(TRAINING_TOPOLOGIES),
        default=LEGACY_SINGLE_GPU_TOPOLOGY,
        help="dose-preserving learner process topology",
    )
    parser.add_argument(
        "--ddp-canary-receipt",
        type=Path,
        default=None,
        help="required same-host NCCL/sampler canary receipt for 8-GPU DDP",
    )
    parser.add_argument(
        "--ablation-id",
        default="",
        help="nonempty diagnostic learner-ablation identity (requires both ablation flags)",
    )
    parser.add_argument(
        "--recipe-overrides-json",
        default="",
        help="nonempty JSON object overriding only allowlisted learner recipe fields",
    )
    parser.add_argument(
        "--ablation-code-tree-sha256",
        default="",
        help="explicitly reviewed digest printed by a prior refused/dry-run inspection",
    )
    parser.add_argument(
        "--reviewed-lock-file-sha256",
        default="",
        help="explicitly reviewed sha256 of the raw immutable lock bytes",
    )
    parser.add_argument(
        "--fresh-policy-distillation-only",
        action="store_true",
        help=(
            "diagnostic post-wave arm: distill policy only on current/recent/hard "
            "components while retaining all components for value training"
        ),
    )
    parser.add_argument(
        "--diagnostic-dose-curve",
        action="store_true",
        help=(
            "reporting-only learner campaign mode: emit the 64/128/terminal "
            "same-trajectory frontier and module optimizer telemetry; requires "
            "a generic diagnostic ablation"
        ),
    )
    parser.add_argument(
        "--diagnostic-checkpoint-steps",
        default="",
        help=(
            "optional strictly increasing intermediate optimizer steps for a "
            "generic diagnostic dose curve; terminal --max-steps remains the "
            "final checkpoint"
        ),
    )
    parser.add_argument(
        "--fresh-value-training-only",
        action="store_true",
        help=(
            "diagnostic post-wave arm: train value only on current/recent/hard "
            "components while leaving policy distillation scope unchanged"
        ),
    )
    parser.add_argument(
        "--frozen-repo",
        type=Path,
        default=None,
        help=(
            "exact historical checkout that sealed a path-bound lock; requires "
            "--frozen-verifier-sha256"
        ),
    )
    parser.add_argument(
        "--frozen-verifier-sha256",
        default="",
        help=("explicit SHA-256 of <frozen-repo>/tools/a1_pre_wave_contract.py"),
    )
    parser.add_argument(
        "--aux-coordinator-root",
        type=Path,
        default=None,
        help="central append-only AUX commissioning root",
    )
    parser.add_argument(
        "--aux-experiment-id",
        default="",
        help="centrally issued AUX experiment id",
    )
    parser.add_argument(
        "--aux-arm",
        choices=(AUX_CONTROL_ARM, AUX_TREATMENT_ARM),
        default=None,
        help="centrally claimed corrected pointer arm",
    )
    parser.add_argument(
        "--aux-observed-allocation-json",
        type=Path,
        default=None,
        help="exact current 8xB200 allocation identity",
    )
    parser.add_argument(
        "--aux-warmed-initializer",
        type=Path,
        default=None,
        help="shared head-only-warmed checkpoint issued by the AUX coordinator",
    )
    parser.add_argument(
        "--p1-coordinator-root",
        type=Path,
        default=None,
        help="central append-only P1 sweep root",
    )
    parser.add_argument(
        "--p1-sweep-id", default="", help="centrally issued P1 sweep id"
    )
    parser.add_argument(
        "--p1-arm",
        choices=tuple(sorted(P1_CENTRAL_ARMS)),
        default=None,
        help="centrally claimed diagnostic K0/K3/K10 arm",
    )
    parser.add_argument(
        "--p1-observed-allocation-json",
        type=Path,
        default=None,
        help="exact current sole-8xB200 allocation identity",
    )
    parser.add_argument(
        "--final-coordinator-root",
        type=Path,
        default=None,
        help="central append-only FINAL replication root",
    )
    parser.add_argument(
        "--final-experiment-id",
        default="",
        help="centrally issued AUX experiment whose selection FINAL replicates",
    )
    parser.add_argument(
        "--final-observed-allocation-json",
        type=Path,
        default=None,
        help="exact current sole-8xB200 allocation identity for FINAL",
    )
    parser.add_argument(
        "--final-warmed-initializer",
        type=Path,
        default=None,
        help="deterministically replayed pointer warmup output (AUXT FINAL only)",
    )
    parser.add_argument(
        "--retry-parent-claim",
        type=Path,
        default=None,
        help="terminal failed-before-optimizer parent claim authorizing one derived retry",
    )
    parser.add_argument(
        "--retry-contract",
        type=Path,
        default=None,
        help="immutable learner-only retry contract output (required with parent claim)",
    )
    parser.add_argument(
        "--go", action="store_true", help="execute locally; default is verified dry-run"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        python = _lexical_python_executable(args.python)
        if args.gpu < 0:
            raise ExecutorError("--gpu must be non-negative")
        ablation_values = (
            args.ablation_id,
            args.recipe_overrides_json,
            args.ablation_code_tree_sha256,
            args.reviewed_lock_file_sha256,
        )
        aux_values = (
            args.aux_coordinator_root,
            args.aux_experiment_id,
            args.aux_arm,
            args.aux_observed_allocation_json,
            args.aux_warmed_initializer,
        )
        aux_requested = any(value not in (None, "") for value in aux_values)
        p1_values = (
            args.p1_coordinator_root,
            args.p1_sweep_id,
            args.p1_arm,
            args.p1_observed_allocation_json,
        )
        p1_requested = any(value not in (None, "") for value in p1_values)
        final_required_values = (
            args.final_coordinator_root,
            args.final_experiment_id,
            args.final_observed_allocation_json,
        )
        final_requested = any(
            value not in (None, "")
            for value in (*final_required_values, args.final_warmed_initializer)
        )
        stage_c_final_requested = args.stage_c_final_authority is not None
        if stage_c_final_requested != (args.stage_c_final_arm is not None):
            raise ExecutorError(
                "--stage-c-final-authority and --stage-c-final-arm are required together"
            )
        if final_requested and not all(
            value not in (None, "") for value in final_required_values
        ):
            raise ExecutorError(
                "all FINAL coordinator/experiment/allocation flags must be supplied together"
            )
        if p1_requested and not all(value not in (None, "") for value in p1_values):
            raise ExecutorError(
                "all central P1 coordinator/sweep/arm/allocation flags must be supplied together"
            )
        if p1_requested and (
            aux_requested
            or final_requested
            or stage_c_final_requested
            or args.ablation_id
            or args.recipe_overrides_json
        ):
            raise ExecutorError(
                "central P1 arms are mutually exclusive with AUX/generic ablations"
            )
        if aux_requested and not all(value not in (None, "") for value in aux_values):
            raise ExecutorError(
                "all central AUX coordinator/experiment/arm/allocation/warmed-"
                "initializer flags must be supplied together"
            )
        if aux_requested and (
            final_requested
            or stage_c_final_requested
            or args.ablation_id
            or args.recipe_overrides_json
        ):
            raise ExecutorError(
                "central AUX arms cannot use generic ablation ids/overrides"
            )
        if final_requested and (
            stage_c_final_requested or args.ablation_id or args.recipe_overrides_json
        ):
            raise ExecutorError("FINAL cannot use generic ablation ids/overrides")
        if stage_c_final_requested and (
            args.ablation_id
            or args.recipe_overrides_json
            or args.independent_parent_authority is not None
            or args.coherent_corpus_admission is None
            or args.architecture_upgrade_receipt is None
            or args.topology != B200_8GPU_DDP_TOPOLOGY
            or args.gpu != 0
        ):
            raise ExecutorError(
                "Stage-C FINAL requires coherent admission, zero-diff upgrade, "
                "and physical b200-8gpu-ddp GPU0 ownership; generic/independent-"
                "parent diagnostics are forbidden"
            )
        if (
            aux_requested
            or p1_requested
            or final_requested
            or stage_c_final_requested
        ) and not (
            args.ablation_code_tree_sha256 and args.reviewed_lock_file_sha256
        ):
            raise ExecutorError(
                "central P1/AUX/FINAL or Stage-C FINAL requires reviewed code-tree "
                "and final-lock digests"
            )
        generic_ablation_requested = bool(
            args.ablation_id or args.recipe_overrides_json
        )
        if (
            args.coherent_corpus_admission is not None
            and not (generic_ablation_requested or stage_c_final_requested)
        ):
            raise ExecutorError(
                "--coherent-corpus-admission requires an explicit diagnostic ablation"
            )
        if args.diagnostic_dose_curve and not generic_ablation_requested:
            raise ExecutorError(
                "--diagnostic-dose-curve requires a generic diagnostic ablation"
            )
        if args.diagnostic_checkpoint_steps and not args.diagnostic_dose_curve:
            raise ExecutorError(
                "--diagnostic-checkpoint-steps requires --diagnostic-dose-curve"
            )
        if args.independent_parent_authority is not None and (
            not args.diagnostic_dose_curve
            or not generic_ablation_requested
            or args.architecture_upgrade_receipt is None
        ):
            raise ExecutorError(
                "--independent-parent-authority requires a generic diagnostic "
                "dose curve and --architecture-upgrade-receipt"
            )
        if args.fresh_policy_distillation_only and not generic_ablation_requested:
            raise ExecutorError(
                "--fresh-policy-distillation-only requires a generic diagnostic ablation"
            )
        if args.fresh_value_training_only and not generic_ablation_requested:
            raise ExecutorError(
                "--fresh-value-training-only requires a generic diagnostic ablation"
            )
        if generic_ablation_requested and not all(ablation_values):
            raise ExecutorError(
                "--ablation-id, --recipe-overrides-json, "
                "--ablation-code-tree-sha256, and --reviewed-lock-file-sha256 "
                "must be supplied together"
            )
        frozen_requested = bool(args.frozen_repo or args.frozen_verifier_sha256)
        if frozen_requested and not (
            args.frozen_repo is not None and args.frozen_verifier_sha256
        ):
            raise ExecutorError(
                "--frozen-repo and --frozen-verifier-sha256 are required together"
            )
        verified = verify_training_inputs(
            lock_path=args.lock,
            data_path=args.data,
            validation_path=args.validation_manifest,
            composite_build_receipt=args.composite_build_receipt,
            reviewed_lock_file_sha256=(
                args.reviewed_lock_file_sha256
                if generic_ablation_requested
                or aux_requested
                or p1_requested
                or final_requested
                or stage_c_final_requested
                else None
            ),
            frozen_repo=args.frozen_repo,
            frozen_verifier_sha256=(
                args.frozen_verifier_sha256 if frozen_requested else None
            ),
            coherent_corpus_admission=args.coherent_corpus_admission,
        )
        if args.architecture_upgrade_receipt is not None:
            verified = bind_function_preserving_upgrade(
                verified,
                args.architecture_upgrade_receipt,
                allow_public_award_transition_source=(aux_requested or final_requested),
                allow_diagnostic_recent_history_source=bool(args.diagnostic_dose_curve),
                independent_parent_authority_path=(args.independent_parent_authority),
                allow_stage_c_final_current_parent=stage_c_final_requested,
            )
        if stage_c_final_requested:
            assert args.stage_c_final_authority is not None
            verified = bind_stage_c_final_replication(
                verified,
                authority_path=args.stage_c_final_authority,
                arm_name=args.stage_c_final_arm,
                reviewed_code_tree_sha256=args.ablation_code_tree_sha256,
            )
        if generic_ablation_requested:
            verified = bind_learner_ablation(
                verified,
                ablation_id=args.ablation_id,
                overrides_json=args.recipe_overrides_json,
                reviewed_code_tree_sha256=args.ablation_code_tree_sha256,
                diagnostic_dose_curve=bool(args.diagnostic_dose_curve),
                diagnostic_checkpoint_steps=args.diagnostic_checkpoint_steps,
            )
            checkpoint_for_descriptor = Path(
                os.path.abspath(os.fspath(args.checkpoint.expanduser()))
            )
            verified = bind_diagnostic_training_descriptor(
                verified,
                descriptor_path=checkpoint_for_descriptor.with_name(
                    f"{checkpoint_for_descriptor.name}.training-descriptor.json"
                ),
                fresh_policy_distillation_only=(args.fresh_policy_distillation_only),
                fresh_value_training_only=args.fresh_value_training_only,
            )
        if p1_requested:
            assert args.p1_coordinator_root is not None
            assert args.p1_arm is not None
            assert args.p1_observed_allocation_json is not None
            allocation_path = args.p1_observed_allocation_json.expanduser()
            if allocation_path.is_symlink() or not allocation_path.is_file():
                raise ExecutorError("P1 observed allocation must be a regular file")
            try:
                observed_allocation = json.loads(
                    allocation_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ExecutorError(
                    f"cannot load P1 observed allocation: {error}"
                ) from error
            try:
                p1_authority = aux_coordinator.load_p1_arm_executor_authority(
                    args.p1_coordinator_root,
                    args.p1_sweep_id,
                    arm_id=args.p1_arm,
                    observed_allocation=observed_allocation,
                )
            except aux_coordinator.CoordinatorError as error:
                raise ExecutorError(f"central P1 authority refused: {error}") from error
            published_p1_authority = _published_executor_authority(
                root=args.p1_coordinator_root,
                experiment_id=args.p1_sweep_id,
                filename=f"p1-15-{args.p1_arm.lower()}-executor-authority.json",
                expected=p1_authority,
            )
            verified = bind_p1_arm(
                verified,
                authority=p1_authority,
                published_executor_authority=published_p1_authority,
                reviewed_code_tree_sha256=args.ablation_code_tree_sha256,
            )
        if aux_requested:
            assert args.aux_coordinator_root is not None
            assert args.aux_arm is not None
            assert args.aux_observed_allocation_json is not None
            assert args.aux_warmed_initializer is not None
            allocation_path = args.aux_observed_allocation_json.expanduser()
            if allocation_path.is_symlink() or not allocation_path.is_file():
                raise ExecutorError("AUX observed allocation must be a regular file")
            try:
                observed_allocation = json.loads(
                    allocation_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ExecutorError(
                    f"cannot load AUX observed allocation: {error}"
                ) from error
            try:
                aux_authority = aux_coordinator.load_aux_pair_executor_authority(
                    args.aux_coordinator_root,
                    args.aux_experiment_id,
                    arm_id=args.aux_arm,
                    observed_allocation=observed_allocation,
                )
            except aux_coordinator.CoordinatorError as error:
                raise ExecutorError(
                    f"central AUX authority refused: {error}"
                ) from error
            published_aux_authority = _published_executor_authority(
                root=args.aux_coordinator_root,
                experiment_id=args.aux_experiment_id,
                filename=f"65-{args.aux_arm.lower()}-executor-authority.json",
                expected=aux_authority,
            )
            verified = bind_aux_pair_arm(
                verified,
                authority=aux_authority,
                published_executor_authority=published_aux_authority,
                warmed_initializer=args.aux_warmed_initializer,
                reviewed_code_tree_sha256=args.ablation_code_tree_sha256,
            )
        if final_requested:
            assert args.final_coordinator_root is not None
            assert args.final_observed_allocation_json is not None
            allocation_path = args.final_observed_allocation_json.expanduser()
            if allocation_path.is_symlink() or not allocation_path.is_file():
                raise ExecutorError("FINAL observed allocation must be a regular file")
            try:
                observed_allocation = json.loads(
                    allocation_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ExecutorError(
                    f"cannot load FINAL observed allocation: {error}"
                ) from error
            try:
                final_authority = (
                    aux_coordinator.load_final_replication_executor_authority(
                        args.final_coordinator_root,
                        args.final_experiment_id,
                        observed_allocation=observed_allocation,
                    )
                )
                final_experiment = aux_coordinator.load_experiment(
                    args.final_coordinator_root, args.final_experiment_id
                )
            except aux_coordinator.CoordinatorError as error:
                raise ExecutorError(
                    f"central FINAL authority refused: {error}"
                ) from error
            published_final_authority = _published_executor_authority(
                root=args.final_coordinator_root,
                experiment_id=args.final_experiment_id,
                filename="93-final-executor-authority.json",
                expected=final_authority,
            )
            verified = bind_final_replication(
                verified,
                authority=final_authority,
                published_executor_authority=published_final_authority,
                experiment=final_experiment,
                final_warmed_initializer=args.final_warmed_initializer,
                reviewed_code_tree_sha256=args.ablation_code_tree_sha256,
            )
        verified = bind_training_topology(
            verified, topology=args.topology, gpu=args.gpu
        )
        verified = bind_ddp_canary(verified, args.ddp_canary_receipt)
        verified = bind_aux_subgoal_preclaim_contract(verified)
        checkpoint = Path(os.path.abspath(os.fspath(args.checkpoint.expanduser())))
        report = Path(os.path.abspath(os.fspath(args.report.expanduser())))
        receipt = Path(os.path.abspath(os.fspath(args.receipt.expanduser())))
        command = build_train_command(
            verified,
            python=python,
            checkpoint=checkpoint,
            report=report,
        )
        if (args.retry_parent_claim is None) != (args.retry_contract is None):
            raise ExecutorError(
                "--retry-parent-claim and --retry-contract must be supplied together"
            )
        if (
            verified.get("learner_ablation") is not None
            and args.retry_parent_claim is not None
        ):
            raise ExecutorError(
                "learner ablations cannot use the historical retry path"
            )
        if (
            isinstance(verified.get("central_learner_binding"), dict)
            and args.retry_parent_claim is not None
        ):
            raise ExecutorError(
                "central P1/AUX/FINAL uses completion-only crash recovery; retries are forbidden"
            )
        if (
            isinstance(verified.get("stage_c_final_replication_binding"), dict)
            and args.retry_parent_claim is not None
        ):
            raise ExecutorError(
                "Stage-C FINAL uses completion-only recovery; typed retries are forbidden"
            )
        if args.retry_parent_claim is not None:
            verified = authorize_failed_before_optimizer_retry(
                verified=verified,
                parent_claim=args.retry_parent_claim,
                retry_command=command,
                checkpoint=checkpoint,
                report=report,
                receipt=receipt,
                retry_contract_path=args.retry_contract,
                publish=bool(args.go),
            )
        claim = _claim_path(verified)
        recovery_evidence = _recover_terminal_complete_receipt(
            verified=verified,
            command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            gpu=args.gpu,
            publish=False,
        )
        if recovery_evidence is None:
            _require_fresh_outputs(checkpoint, report, receipt, claim=claim)
            _require_unconsumed_contract(verified)
        selected_gpus = _selected_gpus(verified, fallback_gpu=args.gpu)
        child_environment = _child_environment(selected_gpus)
        execution_binding = _execution_binding(
            command=command, environment=child_environment
        )
        input_binding = _input_binding(verified)
        transaction_sha256 = _training_transaction_sha256(
            command=command, input_binding=input_binding
        )
        plan = {
            "schema_version": PLAN_SCHEMA,
            "mode": (
                "recover-terminal-receipt"
                if recovery_evidence is not None and args.go
                else "recover-terminal-receipt-dry-run"
                if recovery_evidence is not None
                else "go"
                if args.go
                else "dry-run"
            ),
            "contract_sha256": verified["contract_sha256"],
            "claim_identity_sha256": verified.get(
                "claim_identity_sha256", verified["contract_sha256"]
            ),
            "retry_contract": (
                verified.get("retry_contract") if "retry_contract" in verified else None
            ),
            "global_n_full": 128,
            "world_size": int(verified["recipe"]["world_size"]),
            "gpu": args.gpu,
            "gpus": list(selected_gpus),
            "training_topology": verified.get("training_topology"),
            "ddp_canary": verified.get("ddp_canary"),
            "data_kind": verified.get("data_kind", "a1_memmap_v1"),
            "production_sampling_receipt_sha256": verified.get(
                "production_sampling_receipt_sha256"
            ),
            "validation_split_receipt_sha256": verified.get(
                "validation_split_receipt_sha256"
            ),
            "composite_build_receipt": verified.get("composite_build_receipt"),
            "source_authority": verified.get("source_authority_ref"),
            "event_history_training_contract": verified.get(
                "event_history_training_contract"
            ),
            "command": command,
            "command_sha256": _value_sha256(command),
            "input_binding": input_binding,
            "training_transaction_sha256": transaction_sha256,
            "trainer_authority": verified.get("trainer_authority"),
            "lock_verifier_authority": verified.get("lock_verifier_authority"),
            "execution_binding": execution_binding,
            "checkpoint": str(checkpoint),
            "report": str(report),
            "receipt": str(receipt),
            "function_preserving_upgrade": verified.get("function_preserving_upgrade"),
            "diagnostic_comparison_source": verified.get(
                "diagnostic_comparison_source"
            ),
            "learner_lineage_parent": verified.get("learner_lineage_parent"),
        }
        if verified.get("learner_ablation") is not None:
            plan.update(
                {
                    "learner_ablation": verified["learner_ablation"],
                    "diagnostic_only": True,
                    "promotion_eligible": False,
                }
            )
        print(json.dumps(plan, indent=2, sort_keys=True))
        if not args.go:
            return 0
        execute(
            verified=verified,
            command=command,
            checkpoint=checkpoint,
            report=report,
            receipt=receipt,
            gpu=args.gpu,
        )
        return 0
    except (ExecutorError, OSError) as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
