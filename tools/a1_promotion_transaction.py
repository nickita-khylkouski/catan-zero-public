#!/usr/bin/env python3
"""Fail-closed, recoverable A1 generator-champion promotion transaction.

This tool performs no evaluation and never deploys a checkpoint to the fleet.
It consumes a sealed A1 contract and a typed, passing promotion adjudication,
then updates only the authoritative ChampionRegistry and its local
CURRENT_CHAMPION pointer.  Both files are protected by one exclusive lock and
are replaced atomically one at a time.  Because POSIX cannot atomically replace
two unrelated paths, a durable prepared receipt and exact before-byte backups
make an interrupted two-file commit recoverable.

Promotion and recovery are dry-run by default.  ``--go`` is always required for
mutation.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_pre_wave_contract as a1_contract  # noqa: E402
from tools import a1_one_dose_train as one_dose  # noqa: E402
from tools import a1_learner_dose_contract as learner_dose  # noqa: E402
from tools import _a1_promotion_lock_state as promotion_lock_state  # noqa: E402
from tools.a1_external_panel_compare import (  # noqa: E402
    ExternalPanelComparisonError,
    compare_matched_external_panels,
)
from tools.champion_registry import ChampionRegistry  # noqa: E402
from tools.high_regret_suite_contract import (  # noqa: E402
    REPLAY_CONTRACT,
    SUITE_SCHEMA,
    bind_state_to_manifest,
    load_source_manifest,
    load_source_validation_binding,
    validate_replay_metadata,
    validate_replay_trajectories,
)
from tools.sprt_gate import evaluate_pentanomial_sprt, pair_scores_from_h2h_games  # noqa: E402


ADJUDICATION_SCHEMA = "a1-promotion-adjudication-v2"
PREVIOUS_ADJUDICATION_SCHEMA = "a1-promotion-adjudication-v1"
RECEIPT_SCHEMA = "a1-promotion-transaction-receipt-v3"
PREVIOUS_RECEIPT_SCHEMA = "a1-promotion-transaction-receipt-v2"
LEGACY_RECEIPT_SCHEMA = "a1-promotion-transaction-receipt-v1"
COHORT_EXCLUSIONS_SCHEMA = "a1-promotion-cohort-exclusions-v1"
EVIDENCE_SCHEMA = "a1-promotion-evidence-v1"
HIGH_REGRET_SCHEMA = "a1-high-regret-comparison-v1"
BUCKET_VETO_SCHEMA = "a1-bucket-veto-v1"
HIGH_REGRET_REPORT_SCHEMA = "a1-held-out-high-regret-report-v1"
HIGH_REGRET_ENGINE_IDENTITY_SCHEMA = "a1-high-regret-engine-identity-v1"
ARCHIVED_STATE_RECONSTRUCTION_SCHEMA = "a1-archived-state-reconstruction-v1"
HIGH_REGRET_SUITE_SCHEMA = SUITE_SCHEMA
BUCKET_GAME_REPORT_SCHEMA = "a1-bucket-game-report-v1"
FLEET_EVALUATION_POOL_SCHEMA = "a1-fleet-evaluation-pool-v1"
LEGACY_INCUMBENT_PROVENANCE_SCHEMA = "a1-legacy-incumbent-provenance-v1"
LEGACY_CONTRACT_ATTESTATION_SCHEMA = "a1-markerless-v2-promotion-attestation-v1"
# One immutable pre-promotion contract was sealed before promotion_handoff
# existed.  Compatibility is intentionally an exact identity allowlist, not a
# schema-wide bypass: new markerless v2 locks can never opt themselves in.
HISTORICAL_MARKERLESS_A1_CONTRACT = {
    "contract_id": "a1-infoset-n128-p4-12000games-20260710-r1",
    "contract_sha256": "sha256:c88cec355237f4526159650befb209ea3a8c2d095a32dd645fe04bd01d1c59c4",
    "lock_file_sha256": "sha256:8301c7547e1745812c69ca04934424755c7116eb5e221688abc58c1bcb7a3122",
    "source_draft_sha256": "sha256:ae4af7ba7df732137bca201198bdbef73a2500bebe42bc8cda118cfb082d10fe",
    "training_receipt_sha256": "sha256:3567ec5e8bd9716ec9ce738415a259f984e643e55735ecc594c7df46c0a4801f",
    "training_receipt_digest": "sha256:187288dabbe4ce981196db63a2e73946587877f11afadcdc7994eec2b89067b1",
}
MAX_CALIBRATION_RMSE_REGRESSION = 0.02
MAX_EXTERNAL_WIN_RATE_REGRESSION = 0.02
# Exact one-time compatibility for the historical pre-promotion lock.  These
# values are NOT loop defaults: after the first promotion the incumbent
# operator comes from the authoritative registry agent identity, and a new
# candidate operator comes from the newly sealed contract.
BOOTSTRAP_CANDIDATE_C_SCALE = 0.10
BOOTSTRAP_CHAMPION_C_SCALE = 0.03
ROLE_SEARCH_CONFIG_SCHEMA = "a1-deployed-agent-search-config-v1"
MIN_BUCKET_WIN_RATE = 0.45
MIN_BUCKET_GAMES = 8
REQUIRED_PROMOTION_BUCKETS = {
    "phase:opening",
    "phase:robber_dev",
    "phase:chance",
    "phase:build_trade",
    "opening",
    "41+",
    "blowout",
    "close",
}
REQUIRED_EVIDENCE_KINDS = {
    "mechanism_calibration",
    "internal_h2h",
    "external_panel",
    "high_regret",
    "bucket_veto",
}
REQUIRED_CHECKS = {
    "provenance",
    "mechanism_calibration",
    "internal_h2h",
    "external_panel",
    "high_regret",
    "bucket_veto",
}


class PromotionError(RuntimeError):
    """Raised when promotion evidence or transaction state fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _seal_receipt(value: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    sealed.pop("receipt_sha256", None)
    sealed["receipt_sha256"] = _digest_value(sealed)
    return sealed


def _verify_receipt_digest(value: dict[str, Any]) -> dict[str, Any]:
    declared = _validate_sha256(
        value.get("receipt_sha256"), where="receipt.receipt_sha256"
    )
    unhashed = dict(value)
    unhashed.pop("receipt_sha256", None)
    actual = _digest_value(unhashed)
    if declared != actual:
        raise PromotionError(
            f"recovery receipt semantic digest mismatch: {declared} != {actual}"
        )
    return value


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"{path} must contain a JSON object")
    return value


def _require_exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromotionError(f"{where} must be an object")
    actual = set(value)
    if actual != keys:
        raise PromotionError(
            f"{where} keys differ: missing={sorted(keys - actual)} "
            f"unexpected={sorted(actual - keys)}"
        )
    return dict(value)


def _absolute(path: Any, *, base: Path) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise PromotionError("artifact path must be a non-empty string")
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def _lexical_absolute(path: Path) -> Path:
    """Absolute path without resolving symlinks."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _canonical_existing_file(path: Path, *, where: str) -> Path:
    """Return one existing regular path, rejecting every symlink component."""
    lexical = _lexical_absolute(path)
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PromotionError(f"cannot resolve {where}: {error}") from error
    if lexical != resolved:
        raise PromotionError(f"{where} must not contain symlinks: {lexical}")
    if not resolved.is_file():
        raise PromotionError(f"{where} must be an existing regular file: {resolved}")
    return resolved


def _historical_checkpoint_path(
    raw: Any, *, report_path: Path, checkpoint: Path, where: str
) -> Path:
    """Resolve a historical checkpoint against one unique report ancestor."""

    if not isinstance(raw, str) or not raw.strip():
        raise PromotionError(f"{where} must be a path")
    declared = Path(raw).expanduser()
    if ".." in declared.parts:
        raise PromotionError(f"{where} contains traversal")

    if declared.is_absolute():
        resolved = _canonical_existing_file(declared, where=where)
        if resolved != checkpoint:
            raise PromotionError(f"{where} does not bind the incumbent checkpoint")
        return resolved

    clean_parts = tuple(part for part in declared.parts if part not in {"", "."})
    if len(clean_parts) < 2:
        raise PromotionError(f"{where} must be a multi-component relative path")

    matches: list[Path] = []
    for base in (report_path.parent, *report_path.parent.parents):
        candidate = base.joinpath(*clean_parts)
        lexical = _lexical_absolute(candidate)
        try:
            resolved = lexical.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise PromotionError(f"cannot resolve {where}: {error}") from error
        if lexical != resolved:
            raise PromotionError(f"{where} must not contain symlinks: {lexical}")
        if candidate.exists() or candidate.is_symlink():
            matches.append(_canonical_existing_file(candidate, where=where))
    if len(matches) != 1:
        qualifier = "ambiguous" if matches else "unresolvable"
        raise PromotionError(f"{where} is {qualifier} relative to report ancestors")
    if matches[0] != checkpoint:
        raise PromotionError(f"{where} does not bind the incumbent checkpoint")
    return matches[0]


def _canonical_new_file(path: Path, *, where: str) -> Path:
    """Return a canonical not-yet-existing path under a real directory."""
    lexical = _lexical_absolute(path)
    if lexical.exists() or lexical.is_symlink():
        raise PromotionError(f"{where} must be a fresh non-symlink path: {lexical}")
    try:
        resolved = lexical.resolve(strict=False)
    except OSError as error:
        raise PromotionError(f"cannot resolve {where}: {error}") from error
    if lexical != resolved:
        raise PromotionError(f"{where} path must not contain symlinks: {lexical}")
    return resolved


def _canonical_lock_path(registry_path: Path) -> Path:
    return registry_path.with_suffix(registry_path.suffix + ".a1.lock")


def _enforce_canonical_lock(registry_path: Path, requested: Path | None) -> Path:
    canonical = _canonical_lock_path(registry_path)
    if canonical.is_symlink():
        raise PromotionError(
            f"canonical promotion lock must not be a symlink: {canonical}"
        )
    if requested is None:
        return canonical
    lexical = _lexical_absolute(requested)
    try:
        resolved_parent = lexical.parent.resolve(strict=True)
    except OSError as error:
        raise PromotionError(
            f"cannot resolve promotion lock parent: {error}"
        ) from error
    normalized = resolved_parent / lexical.name
    if lexical.parent != resolved_parent or normalized != canonical:
        raise PromotionError(
            f"alternate promotion lock is forbidden; required canonical lock is {canonical}"
        )
    return canonical


def _validate_sha256(value: Any, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise PromotionError(f"{where} must be a full lowercase sha256: digest")
    return value


def _validate_file_ref(
    raw: Any, *, base: Path, where: str, extra_keys: set[str] | None = None
) -> tuple[Path, dict[str, Any]]:
    keys = {"path", "sha256"} | set(extra_keys or ())
    value = _require_exact_keys(raw, keys, where=where)
    path = _absolute(value["path"], base=base)
    if not path.is_file():
        raise PromotionError(f"{where} artifact is missing: {path}")
    declared = _validate_sha256(value["sha256"], where=f"{where}.sha256")
    actual = _sha256(path)
    if declared != actual:
        raise PromotionError(
            f"{where} artifact drift: declared {declared}, actual {actual} ({path})"
        )
    value["path"] = str(path)
    return path, value


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(
        path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def _write_new_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    # Contract verification may replay a post-promotion handoff while the
    # transaction already holds this exact registry lock.  ``flock`` is not
    # reentrant across separately opened file descriptions, so blindly opening
    # the lock again makes a transaction reject itself.  Reuse is deliberately
    # narrower than process-wide: only the same PID, thread, and absolute path
    # may nest.  Other threads and processes still contend on the kernel lock.
    absolute_path = Path(os.path.abspath(os.fspath(path)))
    key = (os.getpid(), threading.get_ident(), os.fspath(absolute_path))
    held = getattr(promotion_lock_state.LOCK_STATE, "held", None)
    if held is None:
        held = {}
        promotion_lock_state.LOCK_STATE.held = held
    inherited = held.get(key)
    if inherited is not None:
        opened_dev, opened_ino, depth = inherited
        named = absolute_path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino) != (opened_dev, opened_ino)
        ):
            raise PromotionError(f"promotion lock identity drifted: {absolute_path}")
        held[key] = (opened_dev, opened_ino, depth + 1)
        try:
            yield
            named = absolute_path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(named.st_mode)
                or (named.st_dev, named.st_ino) != (opened_dev, opened_ino)
            ):
                raise PromotionError(
                    f"promotion lock identity drifted: {absolute_path}"
                )
        finally:
            current = held.get(key)
            if current is not None:
                if current[2] <= 2:
                    held[key] = (opened_dev, opened_ino, 1)
                else:
                    held[key] = (opened_dev, opened_ino, current[2] - 1)
        return

    path = absolute_path
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    registered = False
    try:
        descriptor = os.open(
            path,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        created = True
    except FileExistsError:
        descriptor = os.open(
            path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise PromotionError(f"promotion lock is not a regular file: {path}")
        os.fchmod(descriptor, 0o600)
        if created:
            _fsync_dir(path.parent)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PromotionError(f"promotion lock is already held: {path}") from error
        held[key] = (opened.st_dev, opened.st_ino, 1)
        registered = True
        yield
        named = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise PromotionError(f"promotion lock identity drifted: {path}")
    finally:
        if registered:
            held.pop(key, None)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_current_pointer(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PromotionError(f"cannot read current pointer {path}: {error}") from error
    nonempty = [line.strip() for line in lines if line.strip()]
    if len(nonempty) != 1:
        raise PromotionError(
            f"current pointer {path} must contain exactly one non-empty checkpoint path"
        )
    return str(_absolute(nonempty[0], base=path.parent))


def _verify_symmetry_training_provenance(
    report: dict[str, Any], recipe: dict[str, Any], *, where: str
) -> None:
    """Bind learner D6 event semantics, with flag-off history compatibility.

    Historical reports always recorded ``symmetry_augment`` but predate the
    effective ``symmetry_augment_events`` field.  Missing event provenance is
    harmless only when augmentation was off.  Once augmentation is enabled,
    omitting either the recipe declaration or the report result would make two
    different training operators indistinguishable at promotion time.
    """

    expected_augment = recipe.get("symmetry_augment")
    actual_augment = report.get("symmetry_augment")
    if expected_augment is False and actual_augment is None:
        # The sealed historical production-L1 recipe was flag-off before the
        # report carried either symmetry field.
        actual_augment = False
    if not isinstance(expected_augment, bool) or not isinstance(actual_augment, bool):
        raise PromotionError(f"{where} has no boolean symmetry_augment binding")
    if actual_augment is not expected_augment:
        raise PromotionError(f"{where} symmetry_augment differs from its sealed recipe")

    if actual_augment:
        if not isinstance(recipe.get("symmetry_augment_events"), bool):
            raise PromotionError(
                f"{where} symmetry-on recipe omits symmetry_augment_events"
            )
        if not isinstance(report.get("symmetry_augment_events"), bool):
            raise PromotionError(
                f"{where} symmetry-on report omits symmetry_augment_events"
            )
        expected_events = bool(recipe["symmetry_augment_events"])
        if bool(report["symmetry_augment_events"]) is not expected_events:
            raise PromotionError(
                f"{where} symmetry_augment_events differs from its sealed recipe"
            )
        return

    # Backward compatibility is deliberately limited to flag-off reports.
    # New reports emit the effective False value; old ones may omit the field.
    if "symmetry_augment_events" in report and report["symmetry_augment_events"] is not False:
        raise PromotionError(
            f"{where} flag-off report has active symmetry_augment_events"
        )


def _verify_training_report(
    path: Path,
    *,
    contract: dict[str, Any],
    contract_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
    production_l1_completion: bool = False,
    production_target_gather_completion: bool = False,
    production_temperature_completion: bool = False,
    production_learner_dose: learner_dose.LearnerDose | None = None,
) -> dict[str, Any]:
    report = _load_json(path)
    if sum((production_l1_completion, production_target_gather_completion,
            production_temperature_completion)) > 1:
        raise PromotionError("training report cannot use multiple production schemas")
    if production_temperature_completion:
        dose = production_learner_dose or learner_dose.HISTORICAL_FULL_DOSE
        recipe = contract.get("science", {}).get("learner_training_recipe")
        if not isinstance(recipe, dict):
            recipe = {"symmetry_augment": False}
        _verify_symmetry_training_provenance(
            report, recipe, where="production temperature report"
        )
        required = {
            "arch": "entity_graph", "mask_hidden_info": True,
            "track": "2p_no_trade", "vps_to_win": 10, "world_size": 8,
            "batch_size": 512, "effective_global_batch_size": 4096,
            "epochs": 1,
            "max_steps": dose.optimizer_steps,
            "steps_completed": dose.optimizer_steps,
            "base_training_row_draws": dose.global_samples,
            "optimizer": "adam",
            "resume_optimizer": False, "optimizer_restored": False,
            "lr": 3e-5, "lr_schedule": "flat", "lr_warmup_steps": 100,
            "weight_decay": 0.0, "value_lr_mult": 0.3,
            "action_module_lr_mult": 1.0, "soft_target_weight": 0.9,
            "value_loss_weight": 0.25, "value_target_lambda": 1.0,
            "forced_action_weight": 0.0, "forced_row_value_weight": 1.0,
            "winner_sample_weight": 1.0, "loser_sample_weight": 1.0,
            "training_rng_rank_offset": True, "ddp_shard_data": False,
        }
        for key, expected in required.items():
            if report.get(key) != expected:
                raise PromotionError(
                    f"production temperature report drift at {key}: "
                    f"{report.get(key)!r} != {expected!r}"
                )
        if production_learner_dose is not None:
            try:
                learner_dose.assert_report(report, dose)
            except learner_dose.LearnerDoseError as error:
                raise PromotionError(str(error)) from error
        report_checkpoint = _absolute(report.get("checkpoint"), base=path.parent)
        if report_checkpoint != candidate_path or _sha256(report_checkpoint) != candidate_sha256:
            raise PromotionError("production temperature report candidate bytes drifted")
        return report
    if production_target_gather_completion:
        recipe = contract.get("science", {}).get("learner_training_recipe")
        if not isinstance(recipe, dict):
            recipe = {"symmetry_augment": False}
        _verify_symmetry_training_provenance(
            report, recipe, where="production target-gather report"
        )
        required = {
            "arch": "entity_graph",
            "mask_hidden_info": True,
            "track": "2p_no_trade",
            "vps_to_win": 10,
            "world_size": 4,
            "batch_size": 512,
            "effective_global_batch_size": 2048,
            "epochs": 1,
            "max_steps": 2048,
            "steps_completed": 2048,
            "training_row_draws": 4_194_304,
            "optimizer": "adam",
            "resume_optimizer": False,
            "optimizer_restored": False,
            "soft_target_weight": 0.9,
            "value_loss_weight": 0.25,
            "loser_sample_weight": 1.0,
            "policy_aux_active_batch_size": 0,
            "action_module_lr_mult": 4.0,
            "value_lr_mult": 1.0,
            "freeze_modules": "trunk,action_encoder,policy_head,value_heads",
            "require_only_trainable_prefixes": "target_gather_proj",
            "action_target_gather": True,
            "ddp_find_unused_parameters": True,
            "ddp_shard_data": False,
            "value_target_lambda": 1.0,
            "forced_action_weight": 0.0,
            "forced_row_value_weight": 1.0,
            "winner_sample_weight": 1.0,
            "lr_schedule": "flat",
            "lr_warmup_steps": 100,
            "weight_decay": 0.0,
            "seed": 1,
            "max_grad_norm": 1.0,
            "gradient_clipping_enabled": True,
        }
        for key, expected in required.items():
            if report.get(key) != expected:
                raise PromotionError(
                    f"production target-gather report drift at {key}: "
                    f"{report.get(key)!r} != {expected!r}"
                )
        report_checkpoint = _absolute(report.get("checkpoint"), base=path.parent)
        if (
            report_checkpoint != candidate_path
            or _sha256(report_checkpoint) != candidate_sha256
        ):
            raise PromotionError("production target-gather report candidate bytes drifted")
        return report
    if production_l1_completion:
        dose = production_learner_dose or learner_dose.HISTORICAL_FULL_DOSE
        recipe = contract.get("science", {}).get("learner_training_recipe")
        if not isinstance(recipe, dict):
            # Historical production-L1 manifests predate a nested learner
            # recipe, but their verifier fixes the exact flag-off recipe below.
            recipe = {"symmetry_augment": False}
        _verify_symmetry_training_provenance(
            report, recipe, where="production L1 report"
        )
        required = {
            "arch": "entity_graph",
            "mask_hidden_info": True,
            "track": "2p_no_trade",
            "vps_to_win": 10,
            "world_size": 8,
            "batch_size": 512,
            "epochs": 1,
            "max_steps": dose.optimizer_steps,
            "steps_completed": dose.optimizer_steps,
            "optimizer": "adam",
            "resume_optimizer": False,
            "optimizer_restored": False,
            "loser_sample_weight": 1.0,
            "soft_target_weight": 0.9,
            "policy_aux_active_batch_size": 0,
            "max_grad_norm": 1.0,
            "gradient_clipping_enabled": True,
        }
        for key, expected in required.items():
            if report.get(key) != expected:
                raise PromotionError(
                    f"production L1 report drift at {key}: "
                    f"{report.get(key)!r} != {expected!r}"
                )
        if production_learner_dose is not None:
            try:
                learner_dose.assert_report(report, dose)
            except learner_dose.LearnerDoseError as error:
                raise PromotionError(str(error)) from error
        report_checkpoint = _absolute(report.get("checkpoint"), base=path.parent)
        if report_checkpoint != candidate_path or _sha256(report_checkpoint) != candidate_sha256:
            raise PromotionError("production L1 report candidate bytes drifted")
        producers = [
            record for record in contract.get("checkpoints", [])
            if isinstance(record, dict) and record.get("role") == "producer"
        ]
        if len(producers) != 1 or report.get("init_checkpoint_sha256") != producers[0].get(
            "sha256"
        ):
            raise PromotionError("production L1 report parent differs from contract producer")
        init_path = _canonical_existing_file(
            Path(str(report.get("init_checkpoint"))), where="production L1 parent"
        )
        if _sha256(init_path) != report["init_checkpoint_sha256"]:
            raise PromotionError("production L1 parent bytes drifted")
        return report
    is_dual = report.get("a1_dual_arm_execution_binding") is not None
    if is_dual:
        recipe = report.get("a1_bound_learner_training_recipe")
        if not isinstance(recipe, dict):
            raise PromotionError("dual-arm training report has no bound learner recipe")
        recipe_sha = _digest_value(recipe)
        world_size = report.get("world_size")
        topology = {
            2: {"batch_size": 512, "grad_accum_steps": 4},
            8: {"batch_size": 512, "grad_accum_steps": 1},
        }.get(world_size)
        effective = (
            recipe
            if world_size == 8
            else report.get("a1_effective_learner_training_recipe")
        )
        if (
            topology is None
            or report.get("batch_size") != topology["batch_size"]
            or not isinstance(effective, dict)
            or effective.get("world_size") != world_size
            or effective.get("batch_size") != topology["batch_size"]
            or effective.get("grad_accum_steps") != topology["grad_accum_steps"]
            or effective.get("global_batch_size") != 4096
            or effective.get("ddp_shard_data") is not False
        ):
            raise PromotionError("dual-arm training report topology drift")
    else:
        recipe = contract["science"]["learner_training_recipe"]
        recipe_sha = contract["science"]["learner_training_recipe_sha256"]
    _verify_symmetry_training_provenance(
        report, recipe, where="candidate training report"
    )
    required = {
        "a1_contract_sha256": contract_sha256,
        "a1_learner_training_recipe_sha256": recipe_sha,
        "a1_bound_learner_training_recipe": recipe,
        "arch": "entity_graph",
        "mask_hidden_info": True,
        "track": "2p_no_trade",
        "vps_to_win": 10,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise PromotionError(
                f"candidate training report drift at {key}: "
                f"{report.get(key)!r} != {expected!r}"
            )
    report_checkpoint = _absolute(report.get("checkpoint"), base=path.parent)
    if report_checkpoint != candidate_path:
        raise PromotionError(
            "candidate training report checkpoint differs from the promoted candidate: "
            f"{report_checkpoint} != {candidate_path}"
        )
    if _sha256(report_checkpoint) != candidate_sha256:
        raise PromotionError(
            "candidate bytes drifted while validating its training report"
        )
    producers = [
        record
        for record in contract.get("checkpoints", [])
        if isinstance(record, dict) and record.get("role") == "producer"
    ]
    if len(producers) != 1:
        raise PromotionError("sealed A1 contract must bind exactly one producer")
    producer_sha = _validate_sha256(
        producers[0].get("sha256"), where="contract producer sha256"
    )
    init_sha = report.get("init_checkpoint_sha256")
    curriculum_parent = report.get("a1_curriculum_parent")
    architecture_upgrade_binding: dict[str, Any] | None = None
    if init_sha != producer_sha:
        lineage_value = report.get("a1_lineage_dose")
        try:
            lineage_value = one_dose.lineage.validate_lineage_dose(lineage_value)
        except one_dose.lineage.LineageDoseError as error:
            raise PromotionError(f"candidate initialization lineage refused: {error}") from error
        architecture_upgrade_binding = lineage_value.get(
            "function_preserving_upgrade"
        )
        if architecture_upgrade_binding is not None:
            receipt_path = _canonical_existing_file(
                Path(str(architecture_upgrade_binding["receipt"])),
                where="function-preserving architecture upgrade receipt",
            )
            try:
                upgrade = one_dose.architecture_upgrade.verify_receipt(receipt_path)
            except one_dose.architecture_upgrade.UpgradeError as error:
                raise PromotionError(
                    f"function-preserving architecture upgrade refused: {error}"
                ) from error
            expected_binding = {
                "schema_version": "a1-lineage-function-preserving-upgrade-v1",
                "module": upgrade["module"],
                "receipt": upgrade["receipt"]["path"],
                "receipt_sha256": upgrade["receipt"]["sha256"],
                "source_checkpoint_sha256": upgrade["source"]["sha256"],
                "upgraded_initializer_sha256": upgrade["upgraded_initializer"][
                    "sha256"
                ],
            }
            if (
                architecture_upgrade_binding != expected_binding
                or upgrade["source"]["sha256"] != producer_sha
                or upgrade["upgraded_initializer"]["sha256"] != init_sha
            ):
                raise PromotionError(
                    "function-preserving architecture upgrade does not bind producer/init"
                )
            init_path = _canonical_existing_file(
                Path(str(report.get("init_checkpoint"))),
                where="architecture-upgraded initializer",
            )
            if (
                init_path != Path(upgrade["upgraded_initializer"]["path"])
                or _sha256(init_path) != init_sha
            ):
                raise PromotionError("architecture-upgraded initializer bytes drifted")
        else:
            # A combined-196k candidate is the sole supported curriculum
            # exception. Its parent receipt is independently authenticated.
            parent_checkpoint = (
                curriculum_parent.get("parent_checkpoint")
                if isinstance(curriculum_parent, dict)
                else None
            )
            if (
                not is_dual
                or not isinstance(curriculum_parent, dict)
                or set(curriculum_parent)
                != {
                    "schema_version",
                    "receipt_path",
                    "receipt_sha256",
                    "parent_arm_id",
                    "parent_subset_id",
                    "parent_checkpoint",
                    "generation_producer_sha256",
                }
                or curriculum_parent.get("schema_version")
                != "a1-curriculum-parent-binding-v1"
                or curriculum_parent.get("parent_arm_id") != "n256"
                or curriculum_parent.get("parent_subset_id") != "full-56k"
                or curriculum_parent.get("generation_producer_sha256") != producer_sha
                or not isinstance(parent_checkpoint, dict)
                or set(parent_checkpoint) != {"path", "sha256"}
                or parent_checkpoint.get("sha256") != init_sha
            ):
                raise PromotionError(
                    "candidate training report init checkpoint differs from producer"
                )
            parent_path = _canonical_existing_file(
                Path(str(parent_checkpoint["path"])),
                where="curriculum parent checkpoint",
            )
            parent_receipt = _canonical_existing_file(
                Path(str(curriculum_parent["receipt_path"])),
                where="curriculum parent receipt",
            )
            if (
                _sha256(parent_path) != init_sha
                or _sha256(parent_receipt) != curriculum_parent.get("receipt_sha256")
            ):
                raise PromotionError("curriculum parent checkpoint/receipt bytes drifted")
    steps = report.get("steps_completed")
    epochs = report.get("epochs")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise PromotionError(
            "candidate training report has no completed optimizer steps"
        )
    if epochs != recipe.get("epochs"):
        raise PromotionError(
            "candidate training report epoch count differs from sealed recipe"
        )
    if report.get("max_steps") != recipe.get("max_steps"):
        raise PromotionError(
            "candidate training report max_steps differs from sealed recipe"
        )
    return report


def _verify_one_dose_training_receipt(
    path: Path,
    *,
    contract_lock: Path,
    contract: dict[str, Any],
    candidate_path: Path,
    candidate_sha256: str,
    training_report_path: Path,
    training_report_sha256: str,
    legacy_snapshot: _LegacyPromotionSnapshot | None = None,
) -> dict[str, Any]:
    """Prove the candidate came from the sealed, environment-bound A1 dose.

    A training report alone is reproducible text and is therefore insufficient
    promotion authority.  The v3 one-dose receipt binds the command, exact child
    environment, durable single-dose claim, candidate, optimizer sidecar, and
    executor-augmented report.  Direct promotion replays those bindings so it
    cannot bypass the resumable iteration orchestrator.
    """

    path = _canonical_existing_file(path, where="A1 one-dose training receipt")
    raw_schema = _load_json(path).get("schema_version")
    if raw_schema in {
        "a1-production-l1-rerun-completion-v1",
        "a1-production-l1-rerun-completion-v2",
    }:
        return _verify_production_l1_completion_receipt(
            path,
            contract=contract,
            candidate_path=candidate_path,
            candidate_sha256=candidate_sha256,
            training_report_path=training_report_path,
            training_report_sha256=training_report_sha256,
        )
    if raw_schema == "a1-production-target-gather-retrain-completion-v1":
        return _verify_production_target_gather_completion_receipt(
            path,
            contract=contract,
            candidate_path=candidate_path,
            candidate_sha256=candidate_sha256,
            training_report_path=training_report_path,
            training_report_sha256=training_report_sha256,
        )
    if raw_schema in {
        "a1-production-temperature-replication-completion-v1",
        "a1-production-temperature-replication-completion-v2",
    }:
        return _verify_production_temperature_completion_receipt(
            path,
            contract=contract,
            candidate_path=candidate_path,
            candidate_sha256=candidate_sha256,
            training_report_path=training_report_path,
            training_report_sha256=training_report_sha256,
        )
    if raw_schema == "a1-dual-arm-training-receipt-v1":
        from tools import a1_dual_arm_train as dual_train
        from tools import a1_dual_learner_contract as learner_contract

        try:
            value = dual_train.verify_receipt(path)
        except dual_train.DualTrainError as error:
            raise PromotionError(f"dual-arm training receipt refused: {error}") from error
        if value.get("contract_sha256") != contract.get("contract_sha256"):
            raise PromotionError("dual-arm receipt binds a different contract")
        outputs = value.get("outputs")
        inputs = value.get("inputs")
        if not isinstance(outputs, dict) or not isinstance(inputs, dict):
            raise PromotionError("dual-arm receipt input/output provenance is malformed")
        learner_lock_input = inputs.get("learner_lock")
        corpus_meta_input = inputs.get("corpus_meta")
        validation_input = inputs.get("validation")
        producer_input = inputs.get("producer")
        if not all(
            isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            for ref in (
                learner_lock_input,
                corpus_meta_input,
                validation_input,
                producer_input,
            )
        ):
            raise PromotionError("dual-arm receipt lacks exact reviewed input refs")
        try:
            verified_training = dual_train.verify_inputs(
                learner_lock=Path(str(learner_lock_input["path"])),
                reviewed_lock_file_sha256=str(learner_lock_input["sha256"]),
                data=Path(str(corpus_meta_input["path"])).parent,
                validation=Path(str(validation_input["path"])),
                producer_checkpoint=Path(str(producer_input["path"])),
            )
            value = dual_train.verify_receipt(path, verified=verified_training)
        except dual_train.DualTrainError as error:
            raise PromotionError(
                f"dual-arm full training replay refused: {error}"
            ) from error
        checkpoint_ref = outputs.get("checkpoint")
        report_ref = outputs.get("report")
        optimizer_ref = outputs.get("optimizer")
        if (
            checkpoint_ref
            != {"path": str(candidate_path), "sha256": candidate_sha256}
            or report_ref
            != {"path": str(training_report_path), "sha256": training_report_sha256}
            or not isinstance(optimizer_ref, dict)
            or _sha256(Path(str(optimizer_ref.get("path"))))
            != optimizer_ref.get("sha256")
        ):
            raise PromotionError("dual-arm receipt output bytes differ from candidate")
        producer_ref = inputs.get("producer")
        producers = [
            record
            for record in contract.get("checkpoints", [])
            if isinstance(record, dict) and record.get("role") == "producer"
        ]
        if (
            len(producers) != 1
            or not isinstance(producer_ref, dict)
            or producer_ref.get("sha256") != producers[0].get("sha256")
        ):
            raise PromotionError("dual-arm receipt producer differs from contract")
        audit_ref = inputs.get("audit")
        if not isinstance(audit_ref, dict):
            raise PromotionError("dual-arm receipt has no audit binding")
        audit_path = _canonical_existing_file(
            Path(str(audit_ref.get("path"))), where="dual-arm post-wave audit"
        )
        if _sha256(audit_path) != audit_ref.get("sha256"):
            raise PromotionError("dual-arm receipt audit bytes drifted")
        audit = _load_json(audit_path)
        if (
            audit.get("contract_sha256") != contract.get("contract_sha256")
            or _absolute(audit.get("contract_path"), base=audit_path.parent)
            != contract_lock
        ):
            raise PromotionError("dual-arm audit does not bind this contract lock")
        report = _load_json(training_report_path)
        if report.get("a1_dual_arm_execution_binding") != value.get(
            "execution_binding"
        ):
            raise PromotionError("dual-arm report/receipt execution binding mismatch")
        learner_lock_ref = inputs.get("learner_lock")
        if not isinstance(learner_lock_ref, dict) or set(learner_lock_ref) != {
            "path",
            "sha256",
        }:
            raise PromotionError("dual-arm receipt has no reviewed learner lock")
        try:
            learner_lock = learner_contract.verify_lock(
                Path(str(learner_lock_ref["path"])),
                reviewed_file_sha256=str(learner_lock_ref["sha256"]),
            )
        except learner_contract.LearnerContractError as error:
            raise PromotionError(f"dual learner lock refused: {error}") from error
        generation_ref = learner_lock.get("generation_arm_lock")
        if (
            not isinstance(generation_ref, dict)
            or generation_ref.get("path") != str(contract_lock)
            or generation_ref.get("sha256") != _sha256(contract_lock)
            or learner_lock.get("generation_contract_sha256")
            != contract.get("contract_sha256")
            or (learner_lock.get("arm_id"), learner_lock.get("subset_id"))
            != (value.get("arm_id"), value.get("subset_id"))
            or learner_lock.get("recipe")
            != report.get("a1_bound_learner_training_recipe")
            or learner_lock.get("objective")
            != report.get("a1_bound_learner_value_objective")
        ):
            raise PromotionError("dual learner lock does not authorize candidate report")
        return {
            "path": str(path),
            "sha256": _sha256(path),
            "receipt_sha256": value["receipt_sha256"],
            "claim": value["claim"]["path"],
            "claim_state_sha256": value["claim_completion"]["sha256"],
            "execution_binding_sha256": _digest_value(value["execution_binding"]),
        }
    if legacy_snapshot is None:
        value = _verify_receipt_digest(_load_json(path))
    else:
        if (
            path != legacy_snapshot.training_receipt.path
            or contract_lock != legacy_snapshot.contract_lock.path
        ):
            raise PromotionError("legacy snapshot paths differ from dose verification")
        value = _verify_receipt_digest(
            dict(legacy_snapshot.training_receipt.value)
        )
    expected_keys = {
        "schema_version",
        "status",
        "contract_sha256",
        "lock",
        "lock_file_sha256",
        "corpus",
        "corpus_meta_file_sha256",
        "payload_inventory_sha256",
        "validation_manifest",
        "validation_manifest_file_sha256",
        "producer_checkpoint_sha256",
        "learner_training_recipe_sha256",
        "command",
        "command_sha256",
        "execution_binding",
        "world_size",
        "gpu",
        "gpu_name",
        "started_unix_ns",
        "finished_unix_ns",
        "returncode",
        "outputs",
        "failure",
        "claim",
        "claim_state_sha256",
        "receipt_sha256",
    }
    receipt_schema = value.get("schema_version")
    is_retry = receipt_schema == one_dose.RETRY_RECEIPT_SCHEMA
    is_upgrade = receipt_schema == one_dose.UPGRADE_RECEIPT_SCHEMA
    if is_retry:
        expected_keys |= {"claim_identity_sha256", "retry_contract"}
    if "lineage_dose" in value:
        expected_keys.add("lineage_dose")
    if is_upgrade:
        expected_keys |= {"claim_identity_sha256", "function_preserving_upgrade"}
    value = _require_exact_keys(value, expected_keys, where="one-dose training receipt")
    if receipt_schema not in {
        one_dose.RECEIPT_SCHEMA,
        one_dose.RETRY_RECEIPT_SCHEMA,
        one_dose.UPGRADE_RECEIPT_SCHEMA,
    }:
        raise PromotionError(
            "one-dose receipt schema must be a supported direct or sealed-retry schema"
        )
    if (
        value["status"] != "complete"
        or value["returncode"] != 0
        or value["failure"] is not None
        or value["world_size"] != 1
    ):
        raise PromotionError(
            "one-dose training receipt is not a successful direct dose"
        )
    if value["contract_sha256"] != contract["contract_sha256"]:
        raise PromotionError("one-dose training receipt binds a different A1 contract")
    if _absolute(value["lock"], base=path.parent) != contract_lock:
        raise PromotionError(
            "one-dose training receipt binds a different contract lock"
        )
    contract_lock_sha256 = (
        _sha256(contract_lock)
        if legacy_snapshot is None
        else _sha256_bytes(legacy_snapshot.contract_lock.data)
    )
    if value["lock_file_sha256"] != contract_lock_sha256:
        raise PromotionError("one-dose training receipt contract-lock bytes drifted")
    retry_reference: dict[str, Any] | None = None
    if is_retry:
        retry_reference = _require_exact_keys(
            value["retry_contract"],
            {"path", "file_sha256", "retry_contract_sha256"},
            where="one-dose training receipt.retry_contract",
        )
        retry_contract_path = _canonical_existing_file(
            _absolute(retry_reference["path"], base=path.parent),
            where="one-dose learner retry contract",
        )
        if _sha256(retry_contract_path) != retry_reference["file_sha256"]:
            raise PromotionError("one-dose learner retry contract bytes drifted")
        retry_contract = _load_json(retry_contract_path)
        if retry_contract.get("schema_version") != one_dose.RETRY_CONTRACT_SCHEMA:
            raise PromotionError("one-dose learner retry contract schema is invalid")
        retry_unhashed = dict(retry_contract)
        stated_contract_sha = retry_unhashed.pop("retry_contract_sha256", None)
        if (
            stated_contract_sha != _digest_value(retry_unhashed)
            or stated_contract_sha != retry_reference["retry_contract_sha256"]
        ):
            raise PromotionError("one-dose learner retry contract digest mismatch")
        retry_identity = retry_contract.get("retry_identity")
        if (
            not isinstance(retry_identity, dict)
            or retry_identity.get("schema_version") != one_dose.RETRY_IDENTITY_SCHEMA
            or retry_identity.get("repair_kind") != one_dose.RETRY_REPAIR_KIND
            or retry_identity.get("parent_contract_sha256")
            != contract["contract_sha256"]
            or retry_contract.get("retry_identity_sha256")
            != _digest_value(retry_identity)
            or value["claim_identity_sha256"]
            != retry_contract.get("retry_identity_sha256")
        ):
            raise PromotionError("one-dose learner retry identity is invalid")
    upgrade_value: dict[str, Any] | None = None
    if is_upgrade:
        raw_upgrade = value.get("function_preserving_upgrade")
        if not isinstance(raw_upgrade, dict):
            raise PromotionError("architecture-upgrade receipt lacks upgrade evidence")
        receipt_ref = raw_upgrade.get("receipt")
        if not isinstance(receipt_ref, dict) or set(receipt_ref) != {"path", "sha256"}:
            raise PromotionError("architecture-upgrade receipt has malformed receipt ref")
        try:
            upgrade_value = one_dose.architecture_upgrade.verify_receipt(
                Path(str(receipt_ref["path"]))
            )
        except one_dose.architecture_upgrade.UpgradeError as error:
            raise PromotionError(f"architecture-upgrade receipt refused: {error}") from error
        if upgrade_value != raw_upgrade:
            raise PromotionError("training receipt architecture-upgrade evidence drifted")
        expected_identity = _digest_value(
            {
                "schema_version": "a1-architecture-upgrade-training-identity-v1",
                "contract_sha256": contract["contract_sha256"],
                "upgrade_receipt_sha256": receipt_ref["sha256"],
                "upgrade_receipt_digest": upgrade_value["receipt_sha256"],
            }
        )
        if value.get("claim_identity_sha256") != expected_identity:
            raise PromotionError("architecture-upgrade training identity drifted")
    gpu = value["gpu"]
    if (
        isinstance(gpu, bool)
        or not isinstance(gpu, int)
        or gpu < 0
        or not isinstance(value["gpu_name"], str)
        or "B200" not in value["gpu_name"].upper()
    ):
        raise PromotionError("one-dose receipt does not attest one physical B200")
    started = value["started_unix_ns"]
    finished = value["finished_unix_ns"]
    if (
        isinstance(started, bool)
        or not isinstance(started, int)
        or isinstance(finished, bool)
        or not isinstance(finished, int)
        or started <= 0
        or finished < started
    ):
        raise PromotionError("one-dose receipt timestamps are invalid")

    command = value["command"]
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise PromotionError("one-dose training receipt command is invalid")
    if value["command_sha256"] != _digest_value(command):
        raise PromotionError("one-dose training receipt command digest mismatch")
    execution_binding = value["execution_binding"]
    try:
        one_dose._validate_execution_binding(execution_binding)  # noqa: SLF001
    except (one_dose.ExecutorError, AttributeError, TypeError) as error:
        raise PromotionError(
            f"one-dose execution binding is invalid: {error}"
        ) from error
    if execution_binding["command_sha256"] != value["command_sha256"]:
        raise PromotionError("one-dose command and execution binding disagree")
    try:
        expected_environment = one_dose._child_environment(gpu)  # noqa: SLF001
    except one_dose.ExecutorError as error:
        raise PromotionError(
            f"cannot reconstruct one-dose child environment: {error}"
        ) from error
    if execution_binding["environment"] != expected_environment:
        raise PromotionError(
            "one-dose child environment differs from the exact allowlist"
        )

    output_keys = {
        "checkpoint",
        "checkpoint_sha256",
        "optimizer_sidecar",
        "optimizer_sidecar_sha256",
        "report",
        "report_sha256",
        "execution_binding_sha256",
        "steps_completed",
        "corpus_row_count",
        "training_row_count",
        "validation_row_count",
    }
    if isinstance(value.get("outputs"), dict) and "lineage_dose" in value["outputs"]:
        output_keys |= {"lineage_dose", "sampled_rows"}
    outputs = _require_exact_keys(
        value["outputs"],
        output_keys,
        where="one-dose training receipt.outputs",
    )
    output_checkpoint = _absolute(outputs["checkpoint"], base=path.parent)
    if (
        output_checkpoint != candidate_path
        or outputs["checkpoint_sha256"] != candidate_sha256
    ):
        raise PromotionError("one-dose receipt candidate differs from adjudication")
    if _sha256(output_checkpoint) != outputs["checkpoint_sha256"]:
        raise PromotionError("one-dose receipt candidate bytes drifted")
    output_report = _absolute(outputs["report"], base=path.parent)
    if (
        output_report != training_report_path
        or outputs["report_sha256"] != training_report_sha256
        or _sha256(output_report) != outputs["report_sha256"]
    ):
        raise PromotionError(
            "one-dose receipt training report differs from adjudication"
        )
    optimizer = _canonical_existing_file(
        _absolute(outputs["optimizer_sidecar"], base=path.parent),
        where="one-dose optimizer sidecar",
    )
    if _sha256(optimizer) != outputs["optimizer_sidecar_sha256"]:
        raise PromotionError("one-dose optimizer-sidecar bytes drifted")
    if outputs["execution_binding_sha256"] != _digest_value(execution_binding):
        raise PromotionError("one-dose output execution-binding digest mismatch")
    counts = {
        name: outputs[name]
        for name in (
            "steps_completed",
            "corpus_row_count",
            "training_row_count",
            "validation_row_count",
        )
    }
    if (
        any(
            isinstance(number, bool) or not isinstance(number, int) or number < 0
            for number in counts.values()
        )
        or counts["steps_completed"] <= 0
    ):
        raise PromotionError("one-dose output row/step counts are invalid")
    if (
        counts["training_row_count"] <= 0
        or counts["validation_row_count"] <= 0
        or counts["training_row_count"] + counts["validation_row_count"]
        != counts["corpus_row_count"]
    ):
        raise PromotionError(
            "one-dose output train/validation coverage is inconsistent"
        )

    report = _load_json(output_report)
    if report.get(one_dose.REPORT_EXECUTION_BINDING_FIELD) != execution_binding:
        raise PromotionError(
            "candidate training report does not bind the one-dose command/environment"
        )
    if is_upgrade:
        try:
            report_lineage = one_dose.lineage.validate_lineage_dose(
                report.get("a1_lineage_dose")
            )
        except one_dose.lineage.LineageDoseError as error:
            raise PromotionError(f"architecture-upgrade lineage refused: {error}") from error
        if (
            value.get("lineage_dose") != report_lineage
            or outputs.get("lineage_dose") != report_lineage
            or upgrade_value is None
            or report_lineage.get("function_preserving_upgrade", {}).get(
                "receipt_sha256"
            )
            != upgrade_value["receipt"]["sha256"]
        ):
            raise PromotionError("architecture-upgrade report/receipt lineage drifted")
    if value["learner_training_recipe_sha256"] != contract["science"].get(
        "learner_training_recipe_sha256"
    ):
        raise PromotionError("one-dose receipt learner recipe differs from contract")
    producers = [
        record
        for record in contract.get("checkpoints", [])
        if isinstance(record, dict) and record.get("role") == "producer"
    ]
    if len(producers) != 1 or value["producer_checkpoint_sha256"] != producers[0].get(
        "sha256"
    ):
        raise PromotionError("one-dose receipt producer differs from contract")

    claim_path = _canonical_existing_file(
        _absolute(value["claim"], base=path.parent), where="one-dose durable claim"
    )
    try:
        claim = one_dose._load_claim_state(  # noqa: SLF001
            claim_path,
            contract_sha256=contract["contract_sha256"],
            claim_identity_sha256=(
                value["claim_identity_sha256"] if (is_retry or is_upgrade) else None
            ),
        )
    except one_dose.ExecutorError as error:
        raise PromotionError(f"one-dose durable claim is invalid: {error}") from error
    if (
        claim.get("status") != "complete"
        or claim.get("receipt_target") != str(path)
        or claim.get("state_sha256") != value["claim_state_sha256"]
        or claim.get("command_sha256") != value["command_sha256"]
        or claim.get("execution_binding") != execution_binding
        or claim.get("outputs") != outputs
        or (is_retry and claim.get("retry_contract") != retry_reference)
        or (
            is_upgrade
            and claim.get("function_preserving_upgrade") != upgrade_value
        )
    ):
        raise PromotionError("one-dose receipt and durable claim disagree")
    return {
        "path": str(path),
        "sha256": (
            _sha256(path)
            if legacy_snapshot is None
            else _sha256_bytes(legacy_snapshot.training_receipt.data)
        ),
        "receipt_sha256": value["receipt_sha256"],
        "claim": str(claim_path),
        "claim_state_sha256": value["claim_state_sha256"],
        "execution_binding_sha256": outputs["execution_binding_sha256"],
    }


def _verify_production_l1_completion_receipt(
    path: Path,
    *,
    contract: dict[str, Any],
    candidate_path: Path,
    candidate_sha256: str,
    training_report_path: Path,
    training_report_sha256: str,
) -> dict[str, Any]:
    """Replay the sealed 8-rank production-L1 execution and durable sidecars."""

    from tools import a1_production_l1_rerun as production_l1

    value = _load_json(path)
    schema = value.get("schema_version")
    legacy = schema == production_l1.LEGACY_COMPLETION_SCHEMA
    if schema not in {
        production_l1.LEGACY_COMPLETION_SCHEMA,
        production_l1.COMPLETION_SCHEMA,
    }:
        raise PromotionError("production L1 completion schema is unsupported")
    expected = {
        "schema_version", "diagnostic_only", "production_eligible",
        "created_at_unix_ns", "manifest", "submission", "checkpoint",
        "report", "unit_state", "receipt_sha256",
    }
    if not legacy:
        expected.add("dose_contract")
    value = _require_exact_keys(value, expected, where="production L1 completion receipt")
    unhashed = dict(value)
    stated = unhashed.pop("receipt_sha256")
    if stated != _digest_value(unhashed):
        raise PromotionError("production L1 completion receipt digest mismatch")
    if (
        value["diagnostic_only"] is not False
        or value["production_eligible"] is not True
        or value["unit_state"]
        != {"ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"}
    ):
        raise PromotionError("production L1 completion is not a clean eligible run")
    manifest_path, manifest_ref = _validate_file_ref(
        value["manifest"], base=path.parent, where="production L1 manifest"
    )
    try:
        verified_manifest = production_l1.verify(manifest_path)
    except (production_l1.L1Error, OSError, KeyError, ValueError) as error:
        raise PromotionError(f"production L1 manifest replay refused: {error}") from error
    manifest = verified_manifest["manifest"]
    dose = verified_manifest.get("dose") or (
        learner_dose.HISTORICAL_FULL_DOSE
        if legacy
        else learner_dose.PARETO_SELECTED_DOSE
    )
    try:
        if legacy:
            learner_dose.assert_legacy_payload(manifest.get("selected_dose"), dose)
        else:
            learner_dose.assert_payload(value.get("dose_contract"), dose)
            learner_dose.assert_payload(manifest.get("selected_dose"), dose)
    except learner_dose.LearnerDoseError as error:
        raise PromotionError(str(error)) from error
    checkpoint_path, checkpoint_ref = _validate_file_ref(
        value["checkpoint"], base=path.parent, where="production L1 checkpoint"
    )
    report_path, report_ref = _validate_file_ref(
        value["report"], base=path.parent, where="production L1 report"
    )
    if (
        checkpoint_path != candidate_path
        or checkpoint_ref["sha256"] != candidate_sha256
        or report_path != training_report_path
        or report_ref["sha256"] != training_report_sha256
    ):
        raise PromotionError("production L1 completion outputs differ from adjudication")
    submission_path, submission_ref = _validate_file_ref(
        value["submission"], base=path.parent, where="production L1 submission"
    )
    submission = _load_json(submission_path)
    submission_unhashed = dict(submission)
    submission_digest = submission_unhashed.pop("receipt_sha256", None)
    if (
        submission.get("schema_version")
        != (
            production_l1.LEGACY_SUBMISSION_SCHEMA
            if legacy
            else production_l1.SUBMISSION_SCHEMA
        )
        or submission.get("diagnostic_only") is not False
        or submission.get("production_eligible") is not True
        or submission.get("manifest") != manifest_ref
        or submission.get("command_sha256") != manifest.get("command_sha256")
        or submission_digest != _digest_value(submission_unhashed)
    ):
        raise PromotionError("production L1 submission receipt drifted")
    claim_path, claim_ref = _validate_file_ref(
        submission.get("claim"), base=submission_path.parent,
        where="production L1 one-shot claim",
    )
    claim = _load_json(claim_path)
    claim_unhashed = dict(claim)
    claim_digest = claim_unhashed.pop("claim_sha256", None)
    if (
        claim.get("schema_version")
        != (production_l1.LEGACY_CLAIM_SCHEMA if legacy else production_l1.CLAIM_SCHEMA)
        or claim.get("manifest") != manifest_ref
        or claim.get("unit") != submission.get("unit")
        or claim_digest != _digest_value(claim_unhashed)
    ):
        raise PromotionError("production L1 one-shot claim drifted")

    report = _load_json(report_path)
    progress_path = Path(str(checkpoint_path) + ".training-progress.json")
    optimizer_path = Path(str(checkpoint_path) + ".optimizer.pt")
    progress = _load_json(
        _canonical_existing_file(progress_path, where="production L1 progress sidecar")
    )
    progress_unhashed = dict(progress)
    progress_digest = progress_unhashed.pop("progress_sha256", None)
    if progress_digest != _digest_value(progress_unhashed):
        raise PromotionError("production L1 progress sidecar semantic digest drift")
    progress_checkpoint = _validate_file_ref(
        progress.get("checkpoint"), base=progress_path.parent,
        where="production L1 progress checkpoint",
    )[1]
    progress_optimizer_path, progress_optimizer = _validate_file_ref(
        progress.get("optimizer"), base=progress_path.parent,
        where="production L1 optimizer sidecar",
    )
    if (
        progress_checkpoint != checkpoint_ref
        or progress_optimizer_path != optimizer_path
        or progress.get("optimizer_step") != dose.optimizer_steps
        or progress.get("completed_epochs") != 1
        or not isinstance(progress.get("rank_torch_rng_states"), list)
        or len(progress["rank_torch_rng_states"]) != 8
        or report.get("steps_completed") != dose.optimizer_steps
        or report.get("world_size") != dose.world_size
        or report.get("batch_size") != dose.per_rank_batch_size
    ):
        raise PromotionError("production L1 progress/dose topology drifted")
    if not legacy:
        try:
            learner_dose.assert_report(report, dose)
        except learner_dose.LearnerDoseError as error:
            raise PromotionError(str(error)) from error
    producers = [
        record for record in contract.get("checkpoints", [])
        if isinstance(record, dict) and record.get("role") == "producer"
    ]
    if (
        len(producers) != 1
        or manifest.get("f7_parent", {}).get("sha256") != producers[0].get("sha256")
        or report.get("init_checkpoint_sha256") != producers[0].get("sha256")
    ):
        raise PromotionError("production L1 receipt parent differs from contract producer")
    execution_binding = {
        "schema_version": "a1-production-l1-promotion-execution-binding-v1",
        "manifest": manifest_ref,
        "submission": submission_ref,
        "claim": claim_ref,
        "command_sha256": manifest["command_sha256"],
        "unit": submission["unit"],
        "dose_contract": dose.payload(),
        "world_size": dose.world_size,
        "optimizer": progress_optimizer,
        "progress": {"path": str(progress_path), "sha256": _sha256(progress_path)},
    }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "receipt_sha256": stated,
        "claim": str(claim_path),
        "claim_state_sha256": claim_digest,
        "execution_binding_sha256": _digest_value(execution_binding),
    }


def _verify_architecture_retrain_tensor_invariant(
    *, source: Path, candidate: Path, upgrade: dict[str, Any]
) -> None:
    """Prove an adapter-only retrain did not mutate inherited model tensors."""
    try:
        import torch

        source_raw = torch.load(source, map_location="cpu", weights_only=False)
        candidate_raw = torch.load(candidate, map_location="cpu", weights_only=False)
    except Exception as error:
        raise PromotionError(
            f"cannot replay architecture-retrain checkpoint tensors: {error}"
        ) from error
    source_model = source_raw.get("model") if isinstance(source_raw, dict) else None
    candidate_model = (
        candidate_raw.get("model") if isinstance(candidate_raw, dict) else None
    )
    if not isinstance(source_model, dict) or not isinstance(candidate_model, dict):
        raise PromotionError("architecture-retrain checkpoint model state is malformed")
    added = upgrade.get("new_parameters")
    if not isinstance(added, list) or not all(isinstance(key, str) for key in added):
        raise PromotionError("architecture-upgrade new-parameter evidence is malformed")
    expected_keys = set(source_model) | set(added)
    if set(candidate_model) != expected_keys or set(source_model) & set(added):
        raise PromotionError("architecture-retrain candidate parameter keys drifted")
    changed = [
        key
        for key, source_tensor in source_model.items()
        if not torch.equal(source_tensor, candidate_model[key])
    ]
    if changed:
        raise PromotionError(
            "architecture-retrain inherited shared tensors changed: "
            f"{changed[:8]}"
        )


def _verify_production_temperature_completion_receipt(
    path: Path,
    *,
    contract: dict[str, Any],
    candidate_path: Path,
    candidate_sha256: str,
    training_report_path: Path,
    training_report_sha256: str,
) -> dict[str, Any]:
    """Replay the sealed winning-TEMP production replication."""

    from tools import a1_production_temperature_replication as temperature

    value = _load_json(path)
    schema = value.get("schema_version")
    legacy = schema == temperature.LEGACY_COMPLETION_SCHEMA
    if schema not in {
        temperature.LEGACY_COMPLETION_SCHEMA,
        temperature.COMPLETION_SCHEMA,
    }:
        raise PromotionError("production temperature completion schema is unsupported")
    completion_keys = {
        "schema_version", "diagnostic_only", "production_eligible",
        "created_at_unix_ns", "manifest", "submission", "checkpoint",
        "report", "unit_state", "replication_contract", "receipt_sha256",
    }
    if not legacy:
        completion_keys.add("dose_contract")
    value = _require_exact_keys(
        value,
        completion_keys,
        where="production temperature completion receipt",
    )
    unhashed = dict(value)
    stated = unhashed.pop("receipt_sha256")
    if (
        stated != _digest_value(unhashed)
        or value["diagnostic_only"] is not False
        or value["production_eligible"] is not True
        or value["unit_state"]
        != {"ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"}
    ):
        raise PromotionError("production temperature completion is not eligible")
    manifest_path, manifest_ref = _validate_file_ref(
        value["manifest"], base=path.parent, where="production temperature manifest"
    )
    try:
        verified = temperature.verify(manifest_path)
    except (temperature.TemperatureReplicationError, OSError, KeyError, ValueError) as error:
        raise PromotionError(f"production temperature manifest replay refused: {error}") from error
    if verified["manifest_ref"] != manifest_ref:
        raise PromotionError("production temperature manifest reference drifted")
    dose = verified.get("dose") or (
        learner_dose.HISTORICAL_FULL_DOSE
        if legacy
        else learner_dose.PARETO_SELECTED_DOSE
    )
    expected_contract = {
        "initializer_sha256": temperature.F7_SHA256,
        "global_samples": dose.global_samples,
        "optimizer": "fresh_adam",
        "stored_policy_component_temperatures": temperature.COMPONENT_TEMPERATURES,
        "diagnostic_selection_artifact_relabelled": False,
    }
    try:
        if value.get("replication_contract") != expected_contract:
            raise PromotionError("production temperature replication contract drifted")
        if not legacy:
            learner_dose.assert_payload(value.get("dose_contract"), dose)
    except learner_dose.LearnerDoseError as error:
        raise PromotionError(str(error)) from error
    checkpoint_path, checkpoint_ref = _validate_file_ref(
        value["checkpoint"], base=path.parent, where="production temperature checkpoint"
    )
    report_path, report_ref = _validate_file_ref(
        value["report"], base=path.parent, where="production temperature report"
    )
    if (
        checkpoint_path != candidate_path
        or checkpoint_ref["sha256"] != candidate_sha256
        or candidate_sha256
        in {temperature.F7_SHA256, temperature.WINNING_DIAGNOSTIC_SHA256}
        or report_path != training_report_path
        or report_ref["sha256"] != training_report_sha256
    ):
        raise PromotionError("production temperature outputs differ from adjudication")
    submission_path, submission_ref = _validate_file_ref(
        value["submission"], base=path.parent, where="production temperature submission"
    )
    submission = _load_json(submission_path)
    expected_submission_keys = {
        "schema_version", "diagnostic_only", "production_eligible",
        "created_at_unix_ns", "manifest", "claim", "unit", "command_sha256",
        "systemd_command_sha256", "systemd_stdout", "receipt_sha256",
    }
    submission_unhashed = dict(submission)
    submission_digest = submission_unhashed.pop("receipt_sha256", None)
    manifest = verified["manifest"]
    if (
        set(submission) != expected_submission_keys
        or submission.get("schema_version")
        != (
            temperature.LEGACY_SUBMISSION_SCHEMA
            if legacy
            else temperature.SUBMISSION_SCHEMA
        )
        or submission.get("diagnostic_only") is not False
        or submission.get("production_eligible") is not True
        or submission.get("manifest") != manifest_ref
        or submission.get("command_sha256") != manifest.get("command_sha256")
        or submission_digest != _digest_value(submission_unhashed)
    ):
        raise PromotionError("production temperature submission receipt drifted")
    claim_path, claim_ref = _validate_file_ref(
        submission.get("claim"), base=submission_path.parent,
        where="production temperature one-shot claim",
    )
    claim = _load_json(claim_path)
    claim_unhashed = dict(claim)
    claim_digest = claim_unhashed.pop("claim_sha256", None)
    if (
        claim_path != verified["output_root"] / "execution.claim.json"
        or set(claim) != {
            "schema_version", "created_at_unix_ns", "manifest", "unit",
            "claim_sha256",
        }
        or claim.get("schema_version")
        != (temperature.LEGACY_CLAIM_SCHEMA if legacy else temperature.CLAIM_SCHEMA)
        or claim.get("manifest") != manifest_ref
        or claim.get("unit") != submission.get("unit")
        or claim_digest != _digest_value(claim_unhashed)
    ):
        raise PromotionError("production temperature one-shot claim drifted")
    expected_systemd = temperature._systemd_command(  # noqa: SLF001
        unit=str(submission["unit"]),
        repo=verified["repo"],
        root=verified["output_root"],
        command=verified["command"],
    )
    if submission.get("systemd_command_sha256") != _digest_value(expected_systemd):
        raise PromotionError("production temperature systemd command drifted")
    producers = [
        record for record in contract.get("checkpoints", [])
        if isinstance(record, dict) and record.get("role") == "producer"
    ]
    report = _load_json(report_path)
    try:
        temperature._verify_completed_report(  # noqa: SLF001
            report,
            verified=verified,
            checkpoint=checkpoint_ref,
        )
    except temperature.TemperatureReplicationError as error:
        raise PromotionError(
            f"production temperature report replay refused: {error}"
        ) from error
    if (
        len(producers) != 1
        or manifest.get("f7_parent", {}).get("sha256") != producers[0].get("sha256")
        or report.get("init_checkpoint_sha256") != producers[0].get("sha256")
    ):
        raise PromotionError("production temperature parent differs from contract producer")
    execution_binding = {
        "schema_version": "a1-production-temperature-promotion-execution-binding-v1",
        "manifest": manifest_ref, "submission": submission_ref, "claim": claim_ref,
        "command_sha256": manifest["command_sha256"], "unit": submission["unit"],
        "dose_contract": dose.payload(),
        "world_size": dose.world_size,
    }
    return {
        "path": str(path), "sha256": _sha256(path), "receipt_sha256": stated,
        "claim": str(claim_path), "claim_state_sha256": claim_digest,
        "execution_binding_sha256": _digest_value(execution_binding),
    }


def _verify_production_target_gather_completion_receipt(
    path: Path,
    *,
    contract: dict[str, Any],
    candidate_path: Path,
    candidate_sha256: str,
    training_report_path: Path,
    training_report_sha256: str,
) -> dict[str, Any]:
    """Replay the sealed four-rank, adapter-only production retrain."""
    from tools import a1_production_gather_retrain as gather

    value = _require_exact_keys(
        _load_json(path),
        {
            "schema_version", "diagnostic_only", "production_eligible",
            "created_at_unix_ns", "manifest", "submission", "checkpoint",
            "report", "operator_sha256", "progress", "optimizer",
            "model_delta", "unit_state", "receipt_sha256",
        },
        where="production target-gather completion receipt",
    )
    unhashed = dict(value)
    stated = unhashed.pop("receipt_sha256")
    if stated != _digest_value(unhashed):
        raise PromotionError("production target-gather completion digest mismatch")
    if (
        value["schema_version"] != gather.COMPLETION_SCHEMA
        or value["diagnostic_only"] is not False
        or value["production_eligible"] is not True
        or value["unit_state"]
        != {"ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"}
    ):
        raise PromotionError("production target-gather completion is not eligible")

    manifest_path, manifest_ref = _validate_file_ref(
        value["manifest"], base=path.parent,
        where="production target-gather manifest",
    )
    try:
        verified = gather.verify(manifest_path)
    except (gather.GatherRetrainError, OSError, KeyError, ValueError) as error:
        raise PromotionError(
            f"production target-gather manifest replay refused: {error}"
        ) from error
    manifest = verified["manifest"]
    if verified["manifest_ref"] != manifest_ref:
        raise PromotionError("production target-gather manifest reference drifted")
    operator = manifest.get("operator")
    policy_aux_active_batch_size = (
        operator.get("policy_aux_active_batch_size_per_rank", 0)
        if isinstance(operator, dict)
        else None
    )
    exact_operator = {
        "world_size": 4,
        "per_rank_batch_size": 512,
        "optimizer_steps": 2048,
        "global_base_draws": 4_194_304,
        "current_fraction": 0.8,
        "current_n128_fraction": 5.0 / 7.0,
        "current_n256_fraction": 2.0 / 7.0,
        "exact_predecessor_replay_fraction": 0.2,
        "soft_target_weight": 0.9,
        "value_loss_weight": 0.25,
        "loser_sample_weight": 1.0,
        "action_module_lr_mult": 4.0,
        "freeze_modules": ["trunk", "action_encoder", "policy_head", "value_heads"],
        "required_trainable_prefixes": ["target_gather_proj"],
        "fresh_optimizer": True,
        "ddp_find_unused_parameters": True,
    }
    if policy_aux_active_batch_size == 64:
        exact_operator.update(
            {
                "policy_aux_active_batch_size_per_rank": 64,
                "global_policy_aux_active_draws": 524_288,
            }
        )
    elif policy_aux_active_batch_size != 0:
        raise PromotionError("production target-gather auxiliary dose drifted")
    if (
        operator != exact_operator
        or value["operator_sha256"] != manifest.get("operator_sha256")
        or value["operator_sha256"] != _digest_value(exact_operator)
    ):
        raise PromotionError("production target-gather operator drifted")

    checkpoint_path, checkpoint_ref = _validate_file_ref(
        value["checkpoint"], base=path.parent,
        where="production target-gather checkpoint",
    )
    report_path, report_ref = _validate_file_ref(
        value["report"], base=path.parent,
        where="production target-gather report",
    )
    progress_path, progress_ref = _validate_file_ref(
        value["progress"], base=path.parent,
        where="production target-gather progress",
    )
    optimizer_path, optimizer_ref = _validate_file_ref(
        value["optimizer"], base=path.parent,
        where="production target-gather optimizer",
    )
    if (
        checkpoint_path != candidate_path
        or checkpoint_ref["sha256"] != candidate_sha256
        or report_path != training_report_path
        or report_ref["sha256"] != training_report_sha256
    ):
        raise PromotionError(
            "production target-gather completion outputs differ from adjudication"
        )

    submission_path, submission_ref = _validate_file_ref(
        value["submission"], base=path.parent,
        where="production target-gather submission",
    )
    submission = _require_exact_keys(
        _load_json(submission_path),
        {
            "schema_version", "diagnostic_only", "production_eligible",
            "created_at_unix_ns", "manifest", "claim", "unit",
            "command_sha256", "systemd_command_sha256", "systemd_stdout",
            "execution_binding", "execution_binding_sha256",
            "receipt_sha256",
        },
        where="production target-gather submission receipt",
    )
    submission_unhashed = dict(submission)
    submission_digest = submission_unhashed.pop("receipt_sha256")
    if (
        submission["schema_version"] != gather.SUBMISSION_SCHEMA
        or submission["diagnostic_only"] is not False
        or submission["production_eligible"] is not True
        or submission["manifest"] != manifest_ref
        or submission["command_sha256"] != manifest.get("command_sha256")
        or submission["systemd_command_sha256"]
        != _digest_value(gather._systemd_command(verified, submission["unit"]))  # noqa: SLF001
        or submission["execution_binding"]
        != gather._execution_binding(verified)  # noqa: SLF001
        or submission["execution_binding_sha256"]
        != _digest_value(gather._execution_binding(verified))  # noqa: SLF001
        or submission_digest != _digest_value(submission_unhashed)
    ):
        raise PromotionError("production target-gather submission drifted")
    claim_path, claim_ref = _validate_file_ref(
        submission["claim"], base=submission_path.parent,
        where="production target-gather claim",
    )
    claim = _require_exact_keys(
        _load_json(claim_path),
        {"schema_version", "created_at_unix_ns", "manifest", "unit", "claim_sha256"},
        where="production target-gather claim",
    )
    claim_unhashed = dict(claim)
    claim_digest = claim_unhashed.pop("claim_sha256")
    if (
        claim["schema_version"] != gather.CLAIM_SCHEMA
        or claim["manifest"] != manifest_ref
        or claim["unit"] != submission["unit"]
        or claim_digest != _digest_value(claim_unhashed)
    ):
        raise PromotionError("production target-gather claim drifted")

    progress = _load_json(progress_path)
    progress_unhashed = dict(progress)
    progress_digest = progress_unhashed.pop("progress_sha256", None)
    progress_checkpoint = _validate_file_ref(
        progress.get("checkpoint"), base=progress_path.parent,
        where="production target-gather progress checkpoint",
    )[1]
    progress_optimizer_path, progress_optimizer = _validate_file_ref(
        progress.get("optimizer"), base=progress_path.parent,
        where="production target-gather progress optimizer",
    )
    if (
        progress_digest != _digest_value(progress_unhashed)
        or progress_checkpoint != checkpoint_ref
        or progress_optimizer_path != optimizer_path
        or progress_optimizer != optimizer_ref
        or progress.get("optimizer_step") != 2048
        or progress.get("completed_epochs") != 1
        or not isinstance(progress.get("rank_torch_rng_states"), list)
        or len(progress["rank_torch_rng_states"]) != 4
    ):
        raise PromotionError("production target-gather progress/dose topology drifted")

    upgrade = manifest.get("function_preserving_upgrade")
    if not isinstance(upgrade, dict):
        raise PromotionError("production target-gather upgrade evidence is malformed")
    try:
        replayed_upgrade = one_dose.architecture_upgrade.verify_receipt(
            Path(str(upgrade.get("receipt", {}).get("path", "")))
        )
    except one_dose.architecture_upgrade.UpgradeError as error:
        raise PromotionError(
            f"production target-gather architecture upgrade refused: {error}"
        ) from error
    source_path, source_ref = _validate_file_ref(
        manifest.get("learner_source_incumbent"), base=manifest_path.parent,
        where="production target-gather learner source incumbent",
    )
    init_path, init_ref = _validate_file_ref(
        replayed_upgrade.get("upgraded_initializer"), base=manifest_path.parent,
        where="production target-gather upgraded initializer",
    )
    if (
        replayed_upgrade != upgrade
        or replayed_upgrade.get("source") != source_ref
        or manifest.get("corpus_producer") == source_ref
    ):
        raise PromotionError(
            "production target-gather learner source/upgrade binding drifted"
        )
    producers = [
        record for record in contract.get("checkpoints", [])
        if isinstance(record, dict) and record.get("role") == "producer"
    ]
    if (
        len(producers) != 1
        or manifest.get("corpus_producer", {}).get("sha256")
        != producers[0].get("sha256")
    ):
        raise PromotionError(
            "production target-gather corpus producer differs from contract producer"
        )
    report = _load_json(report_path)
    report_dose_matches = report.get("policy_aux_active_batch_size") == (
        policy_aux_active_batch_size
    )
    if policy_aux_active_batch_size == 64:
        report_dose_matches = (
            report_dose_matches
            and report.get("policy_aux_training_row_draws") == 524_288
            and report.get("total_training_row_draws") == 4_718_592
        )
    if (
        report.get("init_checkpoint") != str(init_path)
        or report.get("init_checkpoint_sha256") != init_ref["sha256"]
        or report.get("resume_optimizer") is not False
        or report.get("optimizer_restored") is not False
        or report.get("training_row_draws") != 4_194_304
        or not report_dose_matches
        or report.get("require_only_trainable_prefixes") != "target_gather_proj"
    ):
        raise PromotionError(
            "production target-gather report parent/optimizer/trainable surface drifted"
        )
    try:
        replayed_delta = gather._verify_adapter_only_model_delta(  # noqa: SLF001
            init_path, checkpoint_path
        )
    except gather.GatherRetrainError as error:
        raise PromotionError(
            f"production target-gather model delta refused: {error}"
        ) from error
    if value["model_delta"] != replayed_delta:
        raise PromotionError("production target-gather model-delta evidence drifted")
    _verify_architecture_retrain_tensor_invariant(
        source=source_path, candidate=checkpoint_path, upgrade=replayed_upgrade
    )
    execution_binding = {
        "schema_version": "a1-production-target-gather-promotion-execution-binding-v1",
        "manifest": manifest_ref,
        "submission": submission_ref,
        "claim": claim_ref,
        "command_sha256": manifest["command_sha256"],
        "unit": submission["unit"],
        "world_size": 4,
        "learner_source_incumbent": source_ref,
        "corpus_producer": manifest["corpus_producer"],
        "architecture_upgrade_receipt": upgrade["receipt"],
        "optimizer": optimizer_ref,
        "progress": progress_ref,
    }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "receipt_sha256": stated,
        "claim": str(claim_path),
        "claim_state_sha256": claim_digest,
        "execution_binding_sha256": _digest_value(execution_binding),
        "evaluation_parent_sha256": source_ref["sha256"],
    }


@dataclasses.dataclass(frozen=True)
class _JsonSnapshot:
    path: Path
    data: bytes
    value: dict[str, Any]
    identity: tuple[int, int, int, int, int]


@dataclasses.dataclass(frozen=True)
class _LegacyPromotionSnapshot:
    contract_lock: _JsonSnapshot
    source_draft: _JsonSnapshot
    training_receipt: _JsonSnapshot
    attestation: _JsonSnapshot | None = None


def _build_legacy_contract_attestation_snapshot(
    contract_lock: Path, training_receipt: Path
) -> tuple[dict[str, Any], _LegacyPromotionSnapshot]:
    contract_lock = _canonical_existing_file(
        contract_lock, where="historical markerless A1 contract lock"
    )
    training_receipt = _canonical_existing_file(
        training_receipt, where="historical A1 training receipt"
    )
    lock_snapshot = _stable_json_snapshot(
        contract_lock, where="historical markerless A1 contract lock"
    )
    lock_bytes, lock = lock_snapshot.data, lock_snapshot.value
    if lock.get("schema_version") != a1_contract.LEGACY_LOCK_SCHEMA:
        raise PromotionError("legacy promotion attestation requires a v2 lock")
    if "promotion_handoff" in lock:
        raise PromotionError("legacy promotion attestation requires a markerless lock")
    unhashed = dict(lock)
    declared_contract_sha = unhashed.pop("contract_sha256", None)
    if declared_contract_sha != _digest_value(unhashed):
        raise PromotionError("markerless v2 contract semantic digest mismatch")
    source_draft = lock.get("source_draft")
    if not isinstance(source_draft, dict) or set(source_draft) != {"path", "sha256"}:
        raise PromotionError("markerless v2 contract has no exact source_draft record")
    source_path = _canonical_existing_file(
        Path(str(source_draft["path"])), where="historical contract source draft"
    )
    source_snapshot = _stable_json_snapshot(
        source_path, where="historical contract source draft"
    )
    source_bytes, source_payload = source_snapshot.data, source_snapshot.value
    observed = {
        "contract_id": lock.get("contract_id"),
        "contract_sha256": declared_contract_sha,
        "lock_file_sha256": _sha256_bytes(lock_bytes),
        "source_draft_sha256": source_draft.get("sha256"),
    }
    expected_contract = {
        key: HISTORICAL_MARKERLESS_A1_CONTRACT[key] for key in observed
    }
    if observed != expected_contract:
        raise PromotionError(
            "markerless v2 contract is not the allowlisted historical A1 contract"
        )
    if (
        _sha256_bytes(source_bytes) != source_draft["sha256"]
        or source_payload.get("schema_version") != a1_contract.LEGACY_DRAFT_SCHEMA
        or source_payload.get("contract_id") != lock["contract_id"]
        or "promotion_handoff" in source_payload
    ):
        raise PromotionError("historical markerless source draft binding drift")

    receipt_snapshot = _stable_json_snapshot(
        training_receipt, where="historical A1 training receipt"
    )
    receipt_bytes, receipt_value = receipt_snapshot.data, receipt_snapshot.value
    receipt = _verify_receipt_digest(receipt_value)
    if (
        receipt.get("schema_version")
        not in {one_dose.RECEIPT_SCHEMA, one_dose.RETRY_RECEIPT_SCHEMA}
        or receipt.get("status") != "complete"
        or receipt.get("returncode") != 0
        or receipt.get("contract_sha256") != declared_contract_sha
        or receipt.get("lock") != str(contract_lock)
        or receipt.get("lock_file_sha256") != observed["lock_file_sha256"]
        or _sha256_bytes(receipt_bytes)
        != HISTORICAL_MARKERLESS_A1_CONTRACT["training_receipt_sha256"]
        or receipt.get("receipt_sha256")
        != HISTORICAL_MARKERLESS_A1_CONTRACT["training_receipt_digest"]
    ):
        raise PromotionError(
            "training receipt does not bind the exact historical contract lock"
        )
    attestation: dict[str, Any] = {
        "schema_version": LEGACY_CONTRACT_ATTESTATION_SCHEMA,
        "purpose": "promotion_only_historical_pre_promotion_contract",
        "contract_lock": {
            "path": str(contract_lock),
            "sha256": observed["lock_file_sha256"],
            "contract_sha256": declared_contract_sha,
            "contract_id": lock["contract_id"],
        },
        "source_draft": {
            "path": str(source_path),
            "sha256": source_draft["sha256"],
        },
        "training_receipt": {
            "path": str(training_receipt),
            "sha256": _sha256_bytes(receipt_bytes),
            "receipt_sha256": receipt["receipt_sha256"],
            "schema_version": receipt["schema_version"],
        },
    }
    attestation["attestation_sha256"] = _digest_value(attestation)
    return attestation, _LegacyPromotionSnapshot(
        contract_lock=lock_snapshot,
        source_draft=source_snapshot,
        training_receipt=receipt_snapshot,
    )


def build_legacy_contract_attestation(
    contract_lock: Path, training_receipt: Path
) -> dict[str, Any]:
    """Bind the sole allowlisted markerless v2 lock to its completed dose.

    This does not make the lock valid for generation, rendering, post-wave
    audit, or producer handoff.  It is consumed only by this promotion module.
    """

    value, _snapshot = _build_legacy_contract_attestation_snapshot(
        contract_lock, training_receipt
    )
    return value


def _stable_json_snapshot(path: Path, *, where: str) -> _JsonSnapshot:
    """Read one canonical regular file once and bind bytes to its live pathname."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PromotionError(f"cannot open {where}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PromotionError(f"{where} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
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
            raise PromotionError(f"{where} changed while being read")
        named = path.stat(follow_symlinks=False)
        if identity != (
            named.st_dev,
            named.st_ino,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
        ) or not stat.S_ISREG(named.st_mode):
            raise PromotionError(f"{where} pathname changed while being read")
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"cannot parse {where}: {error}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"{where} must contain a JSON object")
    return _JsonSnapshot(path=path, data=data, value=value, identity=identity)


def _revalidate_snapshot_path(snapshot: _JsonSnapshot, *, where: str) -> None:
    try:
        named = snapshot.path.stat(follow_symlinks=False)
    except OSError as error:
        raise PromotionError(f"cannot revalidate {where}: {error}") from error
    actual = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
        named.st_ctime_ns,
    )
    if actual != snapshot.identity or not stat.S_ISREG(named.st_mode):
        raise PromotionError(f"{where} pathname changed after snapshot")


def _revalidate_legacy_snapshot(snapshot: _LegacyPromotionSnapshot) -> None:
    for where, item in (
        ("historical contract lock", snapshot.contract_lock),
        ("historical source draft", snapshot.source_draft),
        ("historical training receipt", snapshot.training_receipt),
        ("legacy contract attestation", snapshot.attestation),
    ):
        if item is not None:
            _revalidate_snapshot_path(item, where=where)


def _verify_legacy_contract_attestation(
    path: Path, *, contract_lock: Path
) -> tuple[dict[str, Any], _LegacyPromotionSnapshot]:
    path = _canonical_existing_file(path, where="legacy contract attestation")
    attestation_snapshot = _stable_json_snapshot(
        path, where="legacy contract attestation"
    )
    attestation_value = attestation_snapshot.value
    value = _require_exact_keys(
        attestation_value,
        {
            "schema_version",
            "purpose",
            "contract_lock",
            "source_draft",
            "training_receipt",
            "attestation_sha256",
        },
        where="legacy contract attestation",
    )
    if (
        value["schema_version"] != LEGACY_CONTRACT_ATTESTATION_SCHEMA
        or value["purpose"] != "promotion_only_historical_pre_promotion_contract"
    ):
        raise PromotionError("legacy contract attestation schema/purpose drift")
    declared = _validate_sha256(
        value["attestation_sha256"], where="legacy attestation digest"
    )
    unhashed = dict(value)
    unhashed.pop("attestation_sha256")
    if declared != _digest_value(unhashed):
        raise PromotionError("legacy contract attestation semantic digest mismatch")
    contract_ref = _require_exact_keys(
        value["contract_lock"],
        {"path", "sha256", "contract_sha256", "contract_id"},
        where="legacy attestation contract_lock",
    )
    if contract_ref["path"] != str(contract_lock):
        raise PromotionError("legacy attestation binds a different contract path")
    receipt_ref = _require_exact_keys(
        value["training_receipt"],
        {"path", "sha256", "receipt_sha256", "schema_version"},
        where="legacy attestation training_receipt",
    )
    receipt_path = _canonical_existing_file(
        Path(str(receipt_ref["path"])), where="attested training receipt"
    )
    rebuilt, inputs = _build_legacy_contract_attestation_snapshot(
        contract_lock, receipt_path
    )
    if rebuilt != value:
        raise PromotionError("legacy contract attestation does not replay exactly")
    snapshot = dataclasses.replace(inputs, attestation=attestation_snapshot)
    _revalidate_legacy_snapshot(snapshot)
    lock = snapshot.contract_lock.value
    lock_bytes = snapshot.contract_lock.data
    if (
        _sha256_bytes(lock_bytes) != contract_ref["sha256"]
        or lock.get("contract_sha256") != contract_ref["contract_sha256"]
        or lock.get("contract_id") != contract_ref["contract_id"]
    ):
        raise PromotionError("legacy contract snapshot differs from attestation")
    return value, snapshot


def _verify_contract_with_snapshot(
    path: Path,
    *,
    verify_lock_fn: Callable[..., dict[str, Any]] = a1_contract.verify_lock,
    legacy_contract_attestation: Path | None = None,
    expected_training_receipt: Path | None = None,
) -> tuple[dict[str, Any], _LegacyPromotionSnapshot | None]:
    legacy_snapshot: _LegacyPromotionSnapshot | None = None
    if legacy_contract_attestation is None:
        try:
            lock = verify_lock_fn(path, require_all_job_claims=True)
        except Exception as error:
            raise PromotionError(
                f"sealed A1 contract verification failed: {error}"
            ) from error
    else:
        legacy_value, legacy_snapshot = _verify_legacy_contract_attestation(
            legacy_contract_attestation, contract_lock=path
        )
        lock = legacy_snapshot.contract_lock.value
        if expected_training_receipt is not None and legacy_value[
            "training_receipt"
        ]["path"] != str(expected_training_receipt):
            raise PromotionError(
                "legacy contract attestation binds a different training receipt"
            )
    search = lock.get("science", {}).get("search_operator", {})
    if search.get("n_full") != 128:
        raise PromotionError(
            f"current A1 promotion requires global n_full=128, got {search.get('n_full')!r}"
        )
    if (
        search.get("n_full_wide") is not None
        or search.get("wide_roots_always_full") is not False
    ):
        raise PromotionError(
            "current A1 promotion is global n128 only; adaptive/global alternate "
            "budgets are forbidden"
        )
    contract_sha = lock.get("contract_sha256")
    _validate_sha256(contract_sha, where="contract.contract_sha256")
    return lock, legacy_snapshot


def _verify_contract(
    path: Path,
    *,
    verify_lock_fn: Callable[..., dict[str, Any]] = a1_contract.verify_lock,
    legacy_contract_attestation: Path | None = None,
    expected_training_receipt: Path | None = None,
) -> dict[str, Any]:
    lock, _snapshot = _verify_contract_with_snapshot(
        path,
        verify_lock_fn=verify_lock_fn,
        legacy_contract_attestation=legacy_contract_attestation,
        expected_training_receipt=expected_training_receipt,
    )
    return lock


def _verify_bound_checkpoint(
    raw: Any, *, expected_path: Path, expected_sha256: str, where: str, base: Path
) -> None:
    value = _require_exact_keys(raw, {"path", "sha256"}, where=where)
    path = _absolute(value["path"], base=base)
    sha256 = _validate_sha256(value["sha256"], where=f"{where}.sha256")
    if path != expected_path or sha256 != expected_sha256:
        raise PromotionError(f"{where} does not bind the adjudicated checkpoint")


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PromotionError(f"{where} must be a positive integer")
    return value


def _finite_number(value: Any, *, where: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionError(f"{where} must be numeric")
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise PromotionError(f"{where} must be finite")
    if minimum is not None and number < minimum:
        raise PromotionError(f"{where} must be >= {minimum}")
    return number


def _verify_fleet_pool_provenance(
    payload: dict[str, Any],
    *,
    kind: str,
    checkpoint_refs: dict[str, tuple[Path, str]],
    effective_config: dict[str, Any],
    where: str,
) -> None:
    """Replay the provenance wrapper emitted by ``a1_evaluation_pool``.

    A pooled fleet report is not one evaluator invocation, so it intentionally
    has no synthetic ``typed_config``.  It instead retains every hashed shard,
    its exact contiguous seed interval, and one science-only effective config.
    Promotion accepts that representation only after replaying all bindings.
    """

    merge = payload.get("fleet_merge")
    if not isinstance(merge, dict):
        raise PromotionError(f"{where}.fleet_merge must be an object")
    required = {
        "schema_version",
        "kind",
        "sources",
        "seed_intervals",
        "effective_search_config_sha256",
        *checkpoint_refs,
    }
    if kind == "internal_h2h":
        required.add("shard_config_hashes")
    value = _require_exact_keys(merge, required, where=f"{where}.fleet_merge")
    if value["schema_version"] != FLEET_EVALUATION_POOL_SCHEMA or value["kind"] != kind:
        raise PromotionError(f"{where} has an unexpected fleet-pool schema/kind")
    for role, (path, sha256) in checkpoint_refs.items():
        _verify_bound_checkpoint(
            value[role],
            expected_path=path,
            expected_sha256=sha256,
            where=f"{where}.fleet_merge.{role}",
            base=path.parent,
        )
    if value["effective_search_config_sha256"] != _digest_value(effective_config):
        raise PromotionError(f"{where} pooled effective-search config digest mismatch")
    sources = value["sources"]
    intervals = value["seed_intervals"]
    if (
        not isinstance(sources, list)
        or not sources
        or not isinstance(intervals, list)
        or len(intervals) != len(sources)
    ):
        raise PromotionError(f"{where} has incomplete fleet shard provenance")
    source_paths: set[str] = set()
    for index, source in enumerate(sources):
        path, _ref = _validate_file_ref(
            source, base=Path.cwd(), where=f"{where}.fleet_merge.sources[{index}]"
        )
        if str(path) in source_paths:
            raise PromotionError(f"{where} repeats a fleet source report")
        source_paths.add(str(path))
    cursor: int | None = None
    interval_paths: set[str] = set()
    for index, raw in enumerate(intervals):
        interval = _require_exact_keys(
            raw,
            {"base_seed", "end_seed", "path"},
            where=f"{where}.fleet_merge.seed_intervals[{index}]",
        )
        lo = interval["base_seed"]
        hi = interval["end_seed"]
        if (
            isinstance(lo, bool)
            or not isinstance(lo, int)
            or isinstance(hi, bool)
            or not isinstance(hi, int)
            or hi <= lo
        ):
            raise PromotionError(f"{where} has an invalid fleet seed interval")
        path = str(_absolute(interval["path"], base=Path.cwd()))
        if path not in source_paths or path in interval_paths:
            raise PromotionError(f"{where} seed interval does not bind one source")
        if cursor is not None and lo != cursor:
            raise PromotionError(f"{where} fleet seed intervals are not contiguous")
        cursor = hi
        interval_paths.add(path)
    if interval_paths != source_paths:
        raise PromotionError(f"{where} fleet seed intervals omit a source")
    if kind == "internal_h2h":
        hashes = value["shard_config_hashes"]
        if not isinstance(hashes, list) or len(hashes) != len(sources):
            raise PromotionError(f"{where} lacks per-shard typed-config hashes")
        hash_paths: set[str] = set()
        for index, raw in enumerate(hashes):
            row = _require_exact_keys(
                raw,
                {"path", "config_hash", "full_config_hash"},
                where=f"{where}.fleet_merge.shard_config_hashes[{index}]",
            )
            path = str(_absolute(row["path"], base=Path.cwd()))
            if path not in source_paths or path in hash_paths:
                raise PromotionError(f"{where} shard config hash path mismatch")
            _validate_sha256(row["full_config_hash"], where="full_config_hash")
            short = row["config_hash"]
            if (
                not isinstance(short, str)
                or not short.startswith("sha256:")
                or len(short) != 23
            ):
                raise PromotionError(f"{where} has an invalid short config hash")
            hash_paths.add(path)
        if hash_paths != source_paths:
            raise PromotionError(f"{where} shard config hashes omit a source")


def _verify_calibration_source(
    payload: dict[str, Any],
    *,
    source_path: Path,
    checkpoint: Path,
    expected_readout: str,
    expected_value_scale: float,
    expected_value_squash: str,
    where: str,
    contract: dict[str, Any] | None = None,
    allow_legacy_incumbent: bool = False,
) -> tuple[float, dict[str, Any]]:
    if payload.get("schema_version") != "phase-sliced-value-calibration-v2":
        raise PromotionError(f"{where} is not phase-sliced-value-calibration-v2")
    if _absolute(payload.get("checkpoint"), base=checkpoint.parent) != checkpoint:
        raise PromotionError(f"{where} checkpoint differs from its evidence role")
    if payload.get("value_readout") != expected_readout:
        raise PromotionError(f"{where} value readout differs from the sealed objective")
    provenance = payload.get("readout_provenance")
    if not isinstance(provenance, dict):
        raise PromotionError(f"{where}.readout_provenance must be an object")
    if provenance.get("requested_readout") != expected_readout:
        raise PromotionError(f"{where} requested readout drift")
    trained = provenance.get("trained_value_readouts")
    if not isinstance(trained, list) or expected_readout not in trained:
        raise PromotionError(f"{where} does not prove the selected readout was trained")
    optimizer_steps = provenance.get("optimizer_steps")
    completed_epochs = provenance.get("completed_epochs")
    if isinstance(optimizer_steps, int) and optimizer_steps > 0:
        _positive_int(optimizer_steps, where=f"{where}.optimizer_steps")
        _positive_int(completed_epochs, where=f"{where}.completed_epochs")
        if payload.get("legacy_incumbent_provenance") is not None:
            raise PromotionError(
                f"{where} may not attach a legacy bridge to native provenance"
            )
    else:
        if not allow_legacy_incumbent or contract is None:
            raise PromotionError(f"{where}.optimizer_steps must be a positive integer")
        if expected_readout != "scalar":
            raise PromotionError(f"{where} legacy provenance is scalar-only")
        bridge = _require_exact_keys(
            payload.get("legacy_incumbent_provenance"),
            {
                "schema_version",
                "contract_sha256",
                "checkpoint_sha256",
                "historical_training_report",
            },
            where=f"{where}.legacy_incumbent_provenance",
        )
        if bridge["schema_version"] != LEGACY_INCUMBENT_PROVENANCE_SCHEMA:
            raise PromotionError(f"{where} has an unexpected legacy bridge schema")
        if bridge["contract_sha256"] != contract.get("contract_sha256"):
            raise PromotionError(f"{where} legacy bridge binds a different contract")
        checkpoint_sha256 = _sha256(checkpoint)
        if bridge["checkpoint_sha256"] != checkpoint_sha256:
            raise PromotionError(f"{where} legacy bridge checkpoint hash mismatch")
        producers = [
            item
            for item in contract.get("checkpoints", [])
            if isinstance(item, dict) and item.get("role") == "producer"
        ]
        if len(producers) != 1:
            raise PromotionError(f"{where} contract has no unique producer checkpoint")
        producer = producers[0]
        producer_path = _absolute(producer.get("path"), base=source_path.parent)
        if producer_path != checkpoint or producer.get("sha256") != checkpoint_sha256:
            raise PromotionError(
                f"{where} legacy bridge is not for the contract-bound incumbent"
            )
        report_path, _report_ref = _validate_file_ref(
            bridge["historical_training_report"],
            base=source_path.parent,
            where=f"{where}.historical_training_report",
        )
        historical = _load_json(report_path)
        _historical_checkpoint_path(
            historical.get("checkpoint"),
            report_path=report_path,
            checkpoint=checkpoint,
            where=f"{where} historical report checkpoint",
        )
        if (
            historical.get("checkpoint_sha256") is not None
            and historical["checkpoint_sha256"] != checkpoint_sha256
        ):
            raise PromotionError(f"{where} historical report checkpoint hash mismatch")
        _positive_int(
            historical.get("steps_completed"),
            where=f"{where}.historical_training_report.steps_completed",
        )
        _positive_int(
            historical.get("epochs"),
            where=f"{where}.historical_training_report.epochs",
        )
        if optimizer_steps is not None or completed_epochs is not None:
            raise PromotionError(
                f"{where} legacy calibration must retain null native step provenance"
            )
    selection = payload.get("row_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("held_out_filter_applied") is not True
    ):
        raise PromotionError(f"{where} is not computed on a held-out row selection")
    cohort_keys = {
        "mode",
        "validation_fraction",
        "validation_seed",
        "validation_game_seed_ranges",
        "seed_manifest_sha256",
        "configured_game_seed_count",
        "observed_game_seed_count",
        "observed_row_count",
    }
    if not cohort_keys.issubset(selection):
        raise PromotionError(f"{where}.row_selection lacks immutable cohort fields")
    if selection.get("mode") != "validation_seed_manifest":
        raise PromotionError(f"{where} must use a validation-seed manifest")
    seed_manifest_sha = selection.get("seed_manifest_sha256")
    seed_digest = (
        seed_manifest_sha.removeprefix("sha256:")
        if isinstance(seed_manifest_sha, str)
        else ""
    )
    if len(seed_digest) != 64 or any(
        character not in "0123456789abcdef" for character in seed_digest
    ):
        raise PromotionError(f"{where} has no full validation-manifest SHA-256")
    global_metrics = payload.get("global")
    if not isinstance(global_metrics, dict):
        raise PromotionError(f"{where}.global must be an object")
    _positive_int(global_metrics.get("n"), where=f"{where}.global.n")
    # Training reports score the raw scalar readout, but the playing operator
    # consumes the sealed post-readout transform (historically tanh).  Using
    # the top-level raw RMSE here can therefore bless a checkpoint whose
    # calibration regresses after the exact transform used by MCTS.  The
    # calibration artifact already computes all transform views; bind and use
    # the deployed one instead of silently validating a different function.
    diagnostics = payload.get("deployed_readout_diagnostics")
    if not isinstance(diagnostics, dict):
        raise PromotionError(f"{where} has no deployed readout diagnostics")
    expected_scale = _finite_number(
        expected_value_scale,
        where=f"{where}.expected_value_scale",
        minimum=0.0,
    )
    if float(diagnostics.get("value_scale", float("nan"))) != expected_scale:
        raise PromotionError(f"{where} deployed value scale differs from the sealed operator")
    expected_transform = (
        f"scalar_{expected_value_squash}"
        if expected_readout == "scalar"
        else "scalar_clip"
    )
    if diagnostics.get("configured_value_squash") != expected_value_squash:
        raise PromotionError(f"{where} deployed value squash differs from the sealed operator")
    if diagnostics.get("configured_effective_transform") != expected_transform:
        raise PromotionError(f"{where} deployed value transform differs from the sealed operator")
    views = diagnostics.get("views")
    if not isinstance(views, dict) or not isinstance(views.get(expected_transform), dict):
        raise PromotionError(f"{where} lacks the sealed deployed readout view")
    deployed_global = views[expected_transform].get("global")
    if not isinstance(deployed_global, dict):
        raise PromotionError(f"{where} deployed readout global metrics are missing")
    if deployed_global.get("n") != global_metrics.get("n"):
        raise PromotionError(f"{where} raw/deployed calibration cohort size differs")
    rmse = _finite_number(
        deployed_global.get("value_rmse"),
        where=f"{where}.deployed_readout_diagnostics.views.{expected_transform}.global.value_rmse",
        minimum=0.0,
    )
    shard_dir = payload.get("shard_dir")
    if not isinstance(shard_dir, str) or not shard_dir:
        raise PromotionError(f"{where} has no source shard_dir")
    cohort = {
        "shard_dir": str(_absolute(shard_dir, base=checkpoint.parent)),
        "row_selection": {key: selection[key] for key in sorted(cohort_keys)},
        "global_n": global_metrics["n"],
    }
    return rmse, cohort


def _require_sealed_semantics(
    actual: dict[str, Any], expected: dict[str, Any], *, where: str
) -> None:
    """Fail closed unless every sealed semantic is explicitly attested.

    Canonical JSON comparison deliberately distinguishes booleans from integers
    (``False`` must not satisfy a sealed ``0``) and also catches missing keys.
    Extra scheduling/provenance fields are allowed; only science semantics bind.
    """

    for key, expected_value in expected.items():
        if key not in actual:
            raise PromotionError(f"{where} omits sealed A1 semantic {key!r}")
        actual_value = actual[key]
        both_numbers = (
            isinstance(actual_value, (int, float))
            and not isinstance(actual_value, bool)
            and isinstance(expected_value, (int, float))
            and not isinstance(expected_value, bool)
        )
        values_match = (
            float(actual_value) == float(expected_value)
            if both_numbers
            else _canonical_bytes(actual_value) == _canonical_bytes(expected_value)
        )
        if not values_match:
            raise PromotionError(
                f"{where} sealed A1 semantic drift: {key}={actual_value!r}, "
                f"expected {expected_value!r}"
            )


def _is_bootstrap_transition_contract(contract: dict[str, Any]) -> bool:
    return (
        contract.get("contract_id")
        == HISTORICAL_MARKERLESS_A1_CONTRACT["contract_id"]
        and contract.get("contract_sha256")
        == HISTORICAL_MARKERLESS_A1_CONTRACT["contract_sha256"]
    )


def _candidate_search_config(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the candidate operator explicitly bound by this contract.

    Only the one allowlisted markerless contract predates an explicit candidate
    identity and receives the historical .10 bootstrap compatibility value.
    Every subsequent cycle uses the contract's effective c_scale verbatim.
    """
    config = _sealed_evaluation_semantics(contract)
    if _is_bootstrap_transition_contract(contract):
        config["c_scale"] = BOOTSTRAP_CANDIDATE_C_SCALE
    return config


def _contract_value_readout(contract: dict[str, Any]) -> str:
    """Return the value readout bound by either supported A1 lock shape.

    The historical bootstrap lock embeds the readout in its learner objective.
    Post-promotion generation-arm locks intentionally carry only the deployed
    scalar ``science.value_readout``.  Promotion of a child of that deployed
    producer must accept the latter without silently inventing a learner
    objective.  If both representations are present they must agree.
    """

    science = contract.get("science")
    if not isinstance(science, dict):
        raise PromotionError("sealed A1 contract has no science object")
    direct = science.get("value_readout")
    learner = science.get("learner_value_objective")
    nested = learner.get("value_readout") if isinstance(learner, dict) else None
    if direct is not None and nested is not None and direct != nested:
        raise PromotionError("sealed A1 contract value_readout representations disagree")
    value = nested if nested is not None else direct
    if value not in {"scalar", "categorical"}:
        raise PromotionError(
            f"sealed A1 contract has unsupported value_readout {value!r}"
        )
    return str(value)


def _incumbent_search_config(
    contract: dict[str, Any],
    *,
    registry: ChampionRegistry,
    champion_path: Path,
    champion_sha256: str,
) -> dict[str, Any]:
    """Recover the incumbent operator from its authoritative registry identity."""
    pointer = registry.get_role("generator_champion")
    if pointer is None:
        raise PromotionError("authoritative registry has no generator_champion")
    if Path(pointer.checkpoint_path).expanduser().resolve() != champion_path:
        raise PromotionError("registry incumbent differs from adjudicated champion")
    provenance = pointer.provenance if isinstance(pointer.provenance, dict) else {}
    raw = provenance.get("a1_candidate_search_config")
    if raw is None:
        if not _is_bootstrap_transition_contract(contract):
            raise PromotionError(
                "registry incumbent has no bound agent search identity; the .03 "
                "bootstrap is permitted only for the exact historical first transition"
            )
        config = _sealed_evaluation_semantics(contract)
        config["c_scale"] = BOOTSTRAP_CHAMPION_C_SCALE
        return config
    sealed = _sealed_evaluation_semantics(contract)
    actual = _require_exact_keys(
        raw, set(sealed), where="registry generator_champion search_config"
    )
    _finite_number(
        actual["c_scale"],
        where="registry generator_champion search_config.c_scale",
        minimum=0.0,
    )
    if float(actual["c_scale"]) <= 0.0:
        raise PromotionError(
            "registry generator_champion search_config.c_scale must be positive"
        )
    # The incumbent's c_scale is part of the promoted agent identity and may
    # intentionally differ from the next candidate contract.  Every other
    # search semantic remains contract-sealed so the comparison changes only
    # the explicitly supported operator axis.
    for key, expected in sealed.items():
        if key == "c_scale":
            continue
        if _canonical_bytes(actual[key]) != _canonical_bytes(expected):
            raise PromotionError(
                "registry generator_champion search_config sealed A1 semantic "
                f"drift: {key}"
            )
    identity_sha = provenance.get("a1_candidate_agent_identity_sha256")
    _validate_sha256(
        identity_sha, where="registry incumbent agent identity sha256"
    )
    _validate_sha256(champion_sha256, where="incumbent checkpoint sha256")
    expected_identity_sha = _agent_identity(
        {"path": str(champion_path), "sha256": champion_sha256}, actual
    )["agent_identity_sha256"]
    if identity_sha != expected_identity_sha:
        raise PromotionError(
            "registry incumbent agent identity does not bind its checkpoint and "
            "search_config"
        )
    return actual


def _normalize_search_runtime_binding(
    raw: Any, *, expected_search_config: dict[str, Any], where: str
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PromotionError(f"{where} must be an object")
    actual = dict(raw)
    runtime_keys = {"native_mcts_hot_loop", "mcts_implementation"}
    present_runtime = runtime_keys & set(actual)
    native_runtime_bound = False
    if present_runtime:
        if present_runtime != runtime_keys:
            raise PromotionError(f"{where} has an incomplete native MCTS runtime binding")
        if (
            actual.pop("native_mcts_hot_loop") is not True
            or actual.pop("mcts_implementation") != "rust_native_hot_loop_v1"
        ):
            raise PromotionError(f"{where} has an unsupported native MCTS runtime binding")
        native_runtime_bound = True
    if (
        actual.get("evaluator_rust_featurize") is True
        and expected_search_config.get("evaluator_rust_featurize") is False
    ):
        if not native_runtime_bound:
            raise PromotionError(
                f"{where} sealed A1 semantic drift: Rust featurization lacks "
                "the sealed native runtime binding"
            )
        # The Rust feature path is parity-gated execution machinery.  Keep the
        # deployed agent identity at the contract's semantic value while the
        # report and engine identity retain proof of the accelerated runtime.
        actual["evaluator_rust_featurize"] = False
    return actual


def _verify_role_search_config(
    raw: Any,
    *,
    expected_search_config: dict[str, Any],
    where: str,
) -> dict[str, Any]:
    """Validate a complete, typed, role-specific deployed search operator.

    This is intentionally an exact-key comparator after removing the one
    fully specified, parity-gated Rust execution binding.  Those runtime
    fields select an implementation of the same operator and therefore do not
    become part of the promoted agent identity.  No partial, alternate, or
    unknown field is accepted.
    """

    actual = _normalize_search_runtime_binding(
        raw, expected_search_config=expected_search_config, where=where
    )
    actual = _require_exact_keys(actual, set(expected_search_config), where=where)
    _require_sealed_semantics(actual, expected_search_config, where=where)
    return actual


def _verify_role_search_pair(
    candidate_raw: Any,
    champion_raw: Any,
    *,
    candidate_search_config: dict[str, Any],
    champion_search_config: dict[str, Any],
    where: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove two reports differ only by the approved role c_scale."""

    candidate = _verify_role_search_config(
        candidate_raw,
        expected_search_config=candidate_search_config,
        where=f"{where}.candidate",
    )
    champion = _verify_role_search_config(
        champion_raw,
        expected_search_config=champion_search_config,
        where=f"{where}.champion",
    )
    for key in sorted(set(candidate) | set(champion)):
        if key == "c_scale":
            continue
        if _canonical_bytes(candidate[key]) != _canonical_bytes(champion[key]):
            raise PromotionError(f"{where} role search drift outside c_scale: {key}")
    return candidate, champion


def _agent_identity(
    checkpoint: dict[str, Any], search_config: dict[str, Any]
) -> dict[str, Any]:
    identity = {
        "schema_version": ROLE_SEARCH_CONFIG_SCHEMA,
        "checkpoint": {
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
        },
        "search_config": search_config,
    }
    identity["agent_identity_sha256"] = _digest_value(identity)
    return identity


def _verify_agent_identity(
    raw: Any,
    *,
    expected_search_config: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    base: Path,
    where: str,
) -> dict[str, Any]:
    value = _require_exact_keys(
        raw,
        {
            "schema_version",
            "checkpoint",
            "search_config",
            "agent_identity_sha256",
        },
        where=where,
    )
    if value["schema_version"] != ROLE_SEARCH_CONFIG_SCHEMA:
        raise PromotionError(f"{where} schema is not supported")
    declared = _validate_sha256(
        value["agent_identity_sha256"], where=f"{where}.agent_identity_sha256"
    )
    unhashed = dict(value)
    unhashed.pop("agent_identity_sha256")
    if declared != _digest_value(unhashed):
        raise PromotionError(f"{where} semantic digest mismatch")
    _verify_bound_checkpoint(
        value["checkpoint"],
        expected_path=checkpoint_path,
        expected_sha256=checkpoint_sha256,
        where=f"{where}.checkpoint",
        base=base,
    )
    if value["checkpoint"]["path"] != str(checkpoint_path):
        raise PromotionError(f"{where}.checkpoint.path must be canonical and absolute")
    actual_search = _require_exact_keys(
        value["search_config"], set(expected_search_config), where=f"{where}.search_config"
    )
    _require_sealed_semantics(
        actual_search, expected_search_config, where=f"{where}.search_config"
    )
    return value


def _verify_evaluation_baseline_binding(
    raw: Any,
    *,
    champion_path: Path,
    champion_sha256: str,
    champion_search_config: dict[str, Any],
    base: Path,
    where: str,
) -> dict[str, Any]:
    value = _require_exact_keys(
        raw,
        {
            "schema_version",
            "comparison_mode",
            "promotion_eligible",
            "historical_comparison_reason",
            "candidate_parent",
            "baseline",
            "registry",
            "authoritative_incumbent",
        },
        where=where,
    )
    if (
        value["schema_version"] != "a1-evaluation-baseline-binding-v1"
        or value["comparison_mode"] != "promotion_parent"
        or value["promotion_eligible"] is not True
        or value["historical_comparison_reason"] is not None
    ):
        raise PromotionError(f"{where} is not a promotion-parent evaluation binding")
    for label in ("candidate_parent", "baseline"):
        _verify_bound_checkpoint(
            value[label],
            expected_path=champion_path,
            expected_sha256=champion_sha256,
            where=f"{where}.{label}",
            base=base,
        )
    _validate_file_ref(value["registry"], base=base, where=f"{where}.registry")
    incumbent = _require_exact_keys(
        value["authoritative_incumbent"],
        {
            "path",
            "sha256",
            "version",
            "agent_identity_sha256",
            "search_config",
        },
        where=f"{where}.authoritative_incumbent",
    )
    _verify_bound_checkpoint(
        {"path": incumbent["path"], "sha256": incumbent["sha256"]},
        expected_path=champion_path,
        expected_sha256=champion_sha256,
        where=f"{where}.authoritative_incumbent",
        base=base,
    )
    _verify_role_search_config(
        incumbent["search_config"],
        expected_search_config=champion_search_config,
        where=f"{where}.authoritative_incumbent.search_config",
    )
    expected_identity = _agent_identity(
        {"path": str(champion_path), "sha256": champion_sha256},
        champion_search_config,
    )["agent_identity_sha256"]
    if incumbent["agent_identity_sha256"] != expected_identity:
        raise PromotionError(
            f"{where} incumbent identity does not bind the adjudicated champion"
        )
    return value


def _sealed_evaluation_semantics(contract: dict[str, Any]) -> dict[str, Any]:
    """Project the immutable A1 lock into deterministic evaluation semantics.

    Evaluation deliberately forces a full n128 search on every decision, while
    all other search and evaluator knobs inherit the sealed production operator.
    This projection is shared by internal H2H and the external neutral harness so
    two reports cannot agree with each other while jointly drifting from A1.
    """

    try:
        science = contract["science"]
        search = science["effective_search_config"]
        evaluator = science["evaluator"]
        max_decisions = contract["generation"]["max_decisions"]
    except (KeyError, TypeError) as error:
        raise PromotionError(
            "sealed A1 contract lacks complete effective search/evaluator semantics"
        ) from error
    if not isinstance(search, dict) or not isinstance(evaluator, dict):
        raise PromotionError(
            "sealed A1 contract search/evaluator semantics must be objects"
        )

    def search_value(name: str) -> Any:
        if name not in search:
            raise PromotionError(
                f"sealed A1 contract omits effective_search_config.{name}"
            )
        return search[name]

    def evaluator_value(name: str) -> Any:
        if name not in evaluator:
            raise PromotionError(f"sealed A1 contract omits evaluator.{name}")
        return evaluator[name]

    n_full = search_value("n_full")
    if n_full != 128:
        raise PromotionError("sealed A1 promotion evaluation requires n_full=128")
    semantics = {
        "public_observation": evaluator_value("public_observation"),
        "belief_chance_spectra": search_value("belief_chance_spectra"),
        "information_set_search": search_value("information_set_search"),
        "determinization_particles": search_value("determinization_particles"),
        "determinization_min_simulations": search_value(
            "determinization_min_simulations"
        ),
        "n_full": n_full,
        "n_fast": n_full,
        "p_full": 1.0,
        "force_full_every_decision": True,
        "n_full_wide": search_value("n_full_wide"),
        "n_full_wide_threshold": search_value("n_full_wide_threshold"),
        "wide_roots_always_full": search_value("wide_roots_always_full"),
        "raw_policy_above_width": search_value("raw_policy_above_width"),
        "max_depth": search_value("max_depth"),
        "max_decisions": max_decisions,
        "temperature": 0.0,
        "c_visit": search_value("c_visit"),
        "c_scale": search_value("c_scale"),
        "rescale_noise_floor_c": search_value("rescale_noise_floor_c"),
        "sigma_eval": search_value("sigma_eval"),
        "max_root_candidates": search_value("max_root_candidates"),
        "max_root_candidates_wide": search_value("max_root_candidates_wide"),
        "wide_candidates_threshold": search_value("wide_candidates_threshold"),
        "symmetry_averaged_eval": search_value("symmetry_averaged_eval"),
        "symmetry_averaged_eval_threshold": search_value(
            "symmetry_averaged_eval_threshold"
        ),
        "correct_rust_chance_spectra": search_value("correct_rust_chance_spectra"),
        "lazy_interior_chance": search_value("lazy_interior_chance"),
        "prior_temperature": evaluator_value("prior_temperature"),
        "value_scale": evaluator_value("value_scale"),
        "value_squash": evaluator_value("value_squash"),
        "value_readout": evaluator_value("value_readout"),
        "play_sh_winner": search_value("play_sh_winner"),
        "exact_budget_sh": search_value("exact_budget_sh"),
        "exact_budget_sh_min_n": search_value("exact_budget_sh_min_n"),
        "root_wave_batching": search_value("root_wave_batching"),
        "use_batch_api": search_value("use_batch_api"),
        "policy_target_min_visits": search_value("policy_target_min_visits"),
        "uncertainty_backup_weighting": search_value("uncertainty_backup_weighting"),
        "uncertainty_backup_a": search_value("uncertainty_backup_a"),
        "uncertainty_backup_exp": search_value("uncertainty_backup_exp"),
        "uncertainty_backup_cap": search_value("uncertainty_backup_cap"),
        "variance_aware_q": search_value("variance_aware_q"),
        "variance_aware_k": search_value("variance_aware_k"),
        "variance_aware_closed_form_js": search_value("variance_aware_closed_form_js"),
        "evaluator_context_fill": evaluator_value("context_fill"),
        "evaluator_cache_size": evaluator_value("cache_size"),
        "evaluator_rust_featurize": evaluator_value("rust_featurize"),
        "evaluator_emit_uncertainty": evaluator_value("emit_uncertainty"),
    }
    return semantics


def _verify_internal_h2h_source(
    payload: dict[str, Any],
    *,
    candidate: Path,
    champion: Path,
    where: str,
    sealed_semantics: dict[str, Any],
    candidate_search_config: dict[str, Any],
    champion_search_config: dict[str, Any],
    required_n_full: int = 128,
) -> None:
    _verify_evaluation_baseline_binding(
        payload.get("evaluation_binding"),
        champion_path=champion,
        champion_sha256=_sha256(champion),
        champion_search_config=champion_search_config,
        base=champion.parent,
        where=f"{where}.evaluation_binding",
    )
    if (
        _absolute(payload.get("candidate_checkpoint"), base=candidate.parent)
        != candidate
    ):
        raise PromotionError(f"{where} candidate checkpoint drift")
    if _absolute(payload.get("baseline_checkpoint"), base=champion.parent) != champion:
        raise PromotionError(f"{where} incumbent checkpoint drift")
    pooled = isinstance(payload.get("fleet_merge"), dict)
    if pooled:
        if payload.get("candidate_checkpoint_sha256") != _sha256(candidate):
            raise PromotionError(f"{where} candidate checkpoint SHA-256 drift")
        if payload.get("baseline_checkpoint_sha256") != _sha256(champion):
            raise PromotionError(f"{where} incumbent checkpoint SHA-256 drift")
        fields = payload.get("effective_search_config")
        if not isinstance(fields, dict):
            raise PromotionError(f"{where} has no pooled effective search config")
        _verify_fleet_pool_provenance(
            payload,
            kind="internal_h2h",
            checkpoint_refs={
                "candidate": (candidate, _sha256(candidate)),
                "champion": (champion, _sha256(champion)),
            },
            effective_config=fields,
            where=where,
        )
    else:
        typed_config = payload.get("typed_config")
        if not isinstance(typed_config, dict):
            raise PromotionError(f"{where} has no typed evaluation config")
        canonical_config = _canonical_bytes(typed_config)
        config_digest = hashlib.sha256(canonical_config).hexdigest()
        if payload.get("full_config_hash") != "sha256:" + config_digest:
            raise PromotionError(f"{where} full config hash does not replay")
        if payload.get("config_hash") != "sha256:" + config_digest[:16]:
            raise PromotionError(f"{where} short config hash does not replay")
        fields = typed_config.get("fields")
        if typed_config.get("pipeline") != "eval" or not isinstance(fields, dict):
            raise PromotionError(f"{where} typed config is not an eval config")
        if fields.get("mode") != "cross_net":
            raise PromotionError(f"{where} typed config is not cross-net")
        if (
            _absolute(fields.get("candidate"), base=candidate.parent) != candidate
            or _absolute(fields.get("baseline"), base=champion.parent) != champion
        ):
            raise PromotionError(f"{where} typed config checkpoint identity drift")
    expected_fields = dict(sealed_semantics)
    expected_fields.update(
        {
            "candidate_c_scale": candidate_search_config["c_scale"],
            "baseline_c_scale": champion_search_config["c_scale"],
            "candidate_n_full": sealed_semantics["n_full"],
            "baseline_n_full": sealed_semantics["n_full"],
            "candidate_n_full_wide": sealed_semantics["n_full_wide"],
            "baseline_n_full_wide": sealed_semantics["n_full_wide"],
            "candidate_n_full_wide_threshold": sealed_semantics[
                "n_full_wide_threshold"
            ],
            "baseline_n_full_wide_threshold": sealed_semantics["n_full_wide_threshold"],
            "candidate_value_readout": sealed_semantics["value_readout"],
            "baseline_value_readout": sealed_semantics["value_readout"],
            "candidate_value_squash": sealed_semantics["value_squash"],
            "baseline_value_squash": sealed_semantics["value_squash"],
        }
    )
    for key in ("n_full", "n_fast", "candidate_n_full", "baseline_n_full"):
        expected_fields[key] = int(required_n_full)
    config_where = "pooled effective config" if pooled else "typed config"
    runtime_fields = dict(fields)
    # Historical cross-net reports had only the shared value_squash field.
    # Normalize that exact legacy shape, while binding new role-specific fields
    # so a diagnostic clip-vs-tanh operator cannot be laundered into promotion.
    runtime_fields.setdefault(
        "candidate_value_squash", runtime_fields.get("value_squash")
    )
    runtime_fields.setdefault(
        "baseline_value_squash", runtime_fields.get("value_squash")
    )
    if (
        pooled
        and runtime_fields.get("native_mcts_hot_loop") is True
        and "mcts_implementation" not in runtime_fields
    ):
        if (
            payload.get("native_mcts_hot_loop") is not True
            or payload.get("mcts_implementation") != "rust_native_hot_loop_v1"
        ):
            raise PromotionError(
                f"{where} pooled native runtime lacks its top-level implementation binding"
            )
        runtime_fields["mcts_implementation"] = payload["mcts_implementation"]
    normalized_fields = _normalize_search_runtime_binding(
        runtime_fields,
        expected_search_config=expected_fields,
        where=f"{where} {config_where}",
    )
    _require_sealed_semantics(
        normalized_fields, expected_fields, where=f"{where} {config_where}"
    )
    if fields.get("map_kind") != "BASE":
        raise PromotionError(
            f"{where} internal H2H must explicitly attest randomized BASE maps; "
            f"got {fields.get('map_kind')!r}"
        )
    candidate_evaluation_config = dict(candidate_search_config)
    candidate_evaluation_config["n_full"] = int(required_n_full)
    candidate_evaluation_config["n_fast"] = int(required_n_full)
    champion_evaluation_config = dict(champion_search_config)
    champion_evaluation_config["n_full"] = int(required_n_full)
    champion_evaluation_config["n_fast"] = int(required_n_full)
    _verify_role_search_pair(
        {
            **{key: normalized_fields[key] for key in candidate_search_config},
            "c_scale": normalized_fields["candidate_c_scale"],
        },
        {
            **{key: normalized_fields[key] for key in champion_search_config},
            "c_scale": normalized_fields["baseline_c_scale"],
        },
        candidate_search_config=candidate_evaluation_config,
        champion_search_config=champion_evaluation_config,
        where=f"{where} deployed search",
    )
    if fields.get("public_observation") is not True:
        raise PromotionError(f"{where} typed config is not public-observation")
    expected_information_recipe = {
        "information_set_search": True,
        "determinization_particles": 4,
        "determinization_min_simulations": 32,
    }
    for key, expected in expected_information_recipe.items():
        if fields.get(key) != expected:
            raise PromotionError(
                f"{where} typed config has unsafe information-set recipe: "
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )
    if (
        fields.get("candidate_n_full") != int(required_n_full)
        or fields.get("baseline_n_full") != int(required_n_full)
    ):
        raise PromotionError(
            f"{where} typed config is not global n{int(required_n_full)}"
        )
    for key in (
        "n_full_wide",
        "candidate_n_full_wide",
        "baseline_n_full_wide",
        "n_full_wide_threshold",
        "candidate_n_full_wide_threshold",
        "baseline_n_full_wide_threshold",
    ):
        if fields.get(key) is not None:
            raise PromotionError(
                f"{where} typed config enables forbidden wide budget {key}"
            )
    if payload.get("verdict") != "H1":
        raise PromotionError(f"{where} verdict is not H1")
    if (
        payload.get("candidate_value_readout") != "scalar"
        or payload.get("baseline_value_readout") != "scalar"
    ):
        raise PromotionError(f"{where} must use scalar readouts for both roles")
    if payload.get("public_observation") is not True:
        raise PromotionError(f"{where} must use public observations")
    for key, expected in expected_information_recipe.items():
        if payload.get(key) != expected:
            raise PromotionError(
                f"{where} does not attest the sealed information-set recipe: "
                f"{key}={payload.get(key)!r}"
            )
    budgets = payload.get("search_budgets_by_role")
    expected_budget = {
        "n_full": int(required_n_full),
        "n_full_wide": None,
        "n_full_wide_threshold": None,
    }
    if (
        not isinstance(budgets, dict)
        or budgets.get("candidate") != expected_budget
        or budgets.get("baseline") != expected_budget
    ):
        raise PromotionError(f"{where} does not use the sealed global n128 budget")
    sprt = payload.get("pentanomial_sprt")
    if not isinstance(sprt, dict) or sprt.get("decision") != "H1":
        raise PromotionError(f"{where} pentanomial verdict is not H1")
    complete_pairs = _positive_int(
        payload.get("complete_pairs"), where=f"{where}.complete_pairs"
    )
    if complete_pairs < 200:
        raise PromotionError(f"{where} has fewer than 200 complete pairs")
    if payload.get("errors") != []:
        raise PromotionError(f"{where} contains evaluation errors")
    if int(payload.get("games_truncated", -1)) != 0:
        raise PromotionError(f"{where} contains truncated games")
    games = payload.get("games")
    if not isinstance(games, list) or len(games) != int(
        payload.get("games_played", -1)
    ):
        raise PromotionError(f"{where} does not retain its complete game evidence")
    if len(games) != int(payload.get("games_with_winner", -1)):
        raise PromotionError(f"{where} has incomplete winner records")
    pair_scores, diagnostics = pair_scores_from_h2h_games(games)
    replayed = evaluate_pentanomial_sprt(
        pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    if replayed["decision"] != "H1" or replayed != sprt:
        raise PromotionError(f"{where} pentanomial evidence does not replay exactly")
    if diagnostics != payload.get("pair_diagnostics"):
        raise PromotionError(f"{where} pair diagnostics do not replay exactly")


def _verify_internal_h2h_cohort(payload: dict[str, Any], *, where: str) -> None:
    """Bind the paired seed cohort, not merely its aggregate SPRT verdict."""

    games = payload.get("games")
    if not isinstance(games, list):
        raise PromotionError(f"{where} games must be a list")
    by_pair: dict[int, dict[str, Any]] = {}
    identities: set[tuple[int, str]] = set()
    seed_to_pair: dict[int, int] = {}
    orientation_encoding: str | None = None
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise PromotionError(f"{where}.games[{index}] must be an object")
        pair_id = game.get("pair_id")
        game_seed = game.get("game_seed")
        orientation = game.get("orientation")
        if (
            isinstance(pair_id, bool)
            or not isinstance(pair_id, int)
            or isinstance(game_seed, bool)
            or not isinstance(game_seed, int)
            or orientation
            not in {
                "candidate_first",
                "candidate_second",
                "candidate_red",
                "candidate_blue",
            }
        ):
            raise PromotionError(f"{where}.games[{index}] has invalid cohort identity")
        game_encoding = (
            "color"
            if orientation in {"candidate_red", "candidate_blue"}
            else "legacy"
        )
        if orientation_encoding is not None and game_encoding != orientation_encoding:
            raise PromotionError(f"{where} mixes orientation encodings")
        orientation_encoding = game_encoding
        identity = (pair_id, orientation)
        if identity in identities:
            raise PromotionError(f"{where} repeats a paired orientation")
        identities.add(identity)
        prior = by_pair.setdefault(pair_id, {"seed": game_seed, "orientations": set()})
        if prior["seed"] != game_seed:
            raise PromotionError(f"{where} pair uses different seeds by orientation")
        prior["orientations"].add(orientation)
        owner = seed_to_pair.setdefault(game_seed, pair_id)
        if owner != pair_id:
            raise PromotionError(f"{where} reuses one game seed across multiple pairs")
    required_orientations = (
        {"candidate_red", "candidate_blue"}
        if orientation_encoding == "color"
        else {"candidate_first", "candidate_second"}
    )
    if any(row["orientations"] != required_orientations for row in by_pair.values()):
        raise PromotionError(f"{where} does not cover both orientations per seed")
    if len(by_pair) != int(payload.get("complete_pairs", -1)):
        raise PromotionError(f"{where} complete-pair count differs from its cohort")


def _verify_n64_confirmation(
    raw: Any,
    *,
    base: Path,
    contract: dict[str, Any],
    candidate: dict[str, Any],
    champion: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ref = _require_exact_keys(raw, {"path", "sha256"}, where="nth_confirmation")
    path, verified = _validate_file_ref(ref, base=base, where="nth_confirmation")
    payload = _load_json(path)
    _verify_internal_h2h_source(
        payload,
        candidate=Path(candidate["path"]),
        champion=Path(champion["path"]),
        where="nth_confirmation",
        sealed_semantics=_sealed_evaluation_semantics(contract),
        candidate_search_config=candidate["search_config"],
        champion_search_config=champion["search_config"],
        required_n_full=64,
    )
    _verify_internal_h2h_cohort(payload, where="nth_confirmation")
    intervals = _contiguous_seed_intervals(
        _explicit_game_seeds(payload, where="nth_confirmation"),
        kind="internal_h2h_n64_confirmation",
        where="nth_confirmation",
    )
    return verified, intervals


def _verify_external_panel_source(
    payload: dict[str, Any],
    *,
    checkpoint: Path,
    checkpoint_md5: str,
    where: str,
    sealed_semantics: dict[str, Any],
    deployed_search_config: dict[str, Any],
    role: str | None = None,
) -> tuple[float, dict[str, Any]]:
    del role  # Compatibility-only label; exact config is supplied explicitly.
    if payload.get("stratum") != "neutral-harness":
        raise PromotionError(f"{where} is not a neutral-harness panel")
    if payload.get("harness") != "catanatron_native_engine":
        raise PromotionError(f"{where} uses an unexpected referee harness")
    if payload.get("mode") != "search" or payload.get("public_observation") is not True:
        raise PromotionError(f"{where} must use public-observation search")
    expected_information_recipe = {
        "information_set_search": True,
        "determinization_particles": 4,
        "determinization_min_simulations": 32,
    }
    for key, expected in expected_information_recipe.items():
        if payload.get(key) != expected:
            raise PromotionError(
                f"{where} does not attest the sealed information-set recipe: "
                f"{key}={payload.get(key)!r}"
            )
    if payload.get("candidate_value_readout") != "scalar":
        raise PromotionError(f"{where} must use the sealed scalar readout")
    trained = payload.get("trained_value_readouts")
    if not isinstance(trained, list) or "scalar" not in trained:
        raise PromotionError(f"{where} does not prove scalar value training")
    if payload.get("n_full") != 128 or payload.get("n_full_wide") is not None:
        raise PromotionError(f"{where} does not use the sealed global n128 budget")
    if (
        _absolute(payload.get("candidate_checkpoint"), base=checkpoint.parent)
        != checkpoint
    ):
        raise PromotionError(f"{where} candidate checkpoint drift")
    if payload.get("candidate_checkpoint_md5") != checkpoint_md5:
        raise PromotionError(f"{where} candidate checkpoint MD5 drift")
    pooled = isinstance(payload.get("fleet_merge"), dict)
    if pooled and payload.get("candidate_checkpoint_sha256") != _sha256(checkpoint):
        raise PromotionError(f"{where} candidate checkpoint SHA-256 drift")
    sprt = payload.get("pentanomial_sprt")
    if not isinstance(sprt, dict):
        raise PromotionError(f"{where} has no external-panel SPRT report")
    complete_pairs = _positive_int(
        payload.get("complete_pairs"), where=f"{where}.complete_pairs"
    )
    if complete_pairs < 500:
        raise PromotionError(f"{where} has fewer than 500 complete pairs")
    if payload.get("errors") != [] or payload.get("worker_errors") != []:
        raise PromotionError(f"{where} contains evaluation errors")
    if int(payload.get("games_engine_divergence", -1)) != 0:
        raise PromotionError(f"{where} contains engine divergence")
    rate = _finite_number(
        payload.get("candidate_win_rate"),
        where=f"{where}.candidate_win_rate",
        minimum=0.0,
    )
    if rate > 1.0:
        raise PromotionError(f"{where}.candidate_win_rate must be <= 1")
    search_config = payload.get("search_config")
    if not isinstance(search_config, dict) or not search_config:
        raise PromotionError(f"{where} has no resolved search_config")
    verified_search_config = _verify_role_search_config(
        search_config,
        expected_search_config=deployed_search_config,
        where=f"{where}.search_config",
    )
    if verified_search_config != deployed_search_config:
        raise PromotionError(f"{where} search config differs from agent identity")
    if pooled:
        effective = payload.get("effective_search_config")
        if effective != search_config:
            raise PromotionError(
                f"{where} pooled effective config differs from search_config"
            )
        _verify_fleet_pool_provenance(
            payload,
            kind="external_panel",
            checkpoint_refs={"checkpoint": (checkpoint, _sha256(checkpoint))},
            effective_config=search_config,
            where=where,
        )
    for key, expected in expected_information_recipe.items():
        if search_config.get(key) != expected:
            raise PromotionError(
                f"{where}.search_config has unsafe {key}={search_config.get(key)!r}"
            )
    games = payload.get("games")
    if not isinstance(games, list) or not games:
        raise PromotionError(f"{where} has no retained paired-game cohort")
    cohort_rows: list[tuple[int, int, str]] = []
    outcomes: list[bool] = []
    orientations_by_pair: dict[int, set[str]] = {}
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise PromotionError(f"{where}.games[{index}] is not an object")
        pair_id = game.get("pair_id")
        game_seed = game.get("game_seed")
        orientation = game.get("orientation")
        outcome = game.get("candidate_won")
        if (
            isinstance(pair_id, bool)
            or not isinstance(pair_id, int)
            or pair_id < 0
            or isinstance(game_seed, bool)
            or not isinstance(game_seed, int)
            or game_seed < 0
            or orientation not in {"candidate_first", "candidate_second"}
            or not isinstance(outcome, bool)
        ):
            raise PromotionError(
                f"{where}.games[{index}] lacks a complete cohort outcome"
            )
        row = (pair_id, game_seed, orientation)
        cohort_rows.append(row)
        outcomes.append(outcome)
        orientations_by_pair.setdefault(pair_id, set()).add(orientation)
    if len(set(cohort_rows)) != len(cohort_rows):
        raise PromotionError(f"{where} contains duplicate paired-game cohort rows")
    if len(orientations_by_pair) != complete_pairs or any(
        orientations != {"candidate_first", "candidate_second"}
        for orientations in orientations_by_pair.values()
    ):
        raise PromotionError(f"{where} does not retain two orientations per pair")
    if len(games) != int(payload.get("games_played", -1)):
        raise PromotionError(f"{where} retained games differ from games_played")
    if (
        len(outcomes) != int(payload.get("games_with_winner", -1))
        or int(payload.get("games_truncated", -1)) != 0
    ):
        raise PromotionError(f"{where} contains incomplete external-panel games")
    wins = sum(outcomes)
    if (
        payload.get("candidate_wins") != wins
        or payload.get("baseline_wins") != len(outcomes) - wins
        or rate != wins / len(outcomes)
    ):
        raise PromotionError(f"{where} win-rate summary does not replay from raw games")
    normalized_games = [{**game, "search_won": game["candidate_won"]} for game in games]
    pair_scores, pair_diagnostics = pair_scores_from_h2h_games(normalized_games)
    if pair_diagnostics.get("incomplete_pairs") != 0 or pair_diagnostics != payload.get(
        "pair_diagnostics"
    ):
        raise PromotionError(f"{where} paired outcomes do not replay from raw games")
    threshold_fields = ("elo0", "elo1", "alpha", "beta")
    try:
        replayed_sprt = evaluate_pentanomial_sprt(
            pair_scores,
            **{field: float(sprt[field]) for field in threshold_fields},
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PromotionError(f"{where} has malformed external-panel SPRT") from error
    if replayed_sprt != sprt or payload.get("verdict") != sprt.get("decision"):
        raise PromotionError(f"{where} external-panel SPRT does not replay")
    cohort_fields = (
        "stratum",
        "harness",
        "baseline_bot",
        "mode",
        "public_observation",
        "information_set_search",
        "determinization_particles",
        "determinization_min_simulations",
        "candidate_value_readout",
        "trained_value_readouts",
        "n_full",
        "n_full_wide",
        "map_kind",
        "gate_config",
        "pairs_requested",
        "games_requested",
    )
    cohort = {
        "cohort_config": {key: payload.get(key) for key in cohort_fields},
        "cohort_rows": sorted(cohort_rows),
        "pooled": pooled,
    }
    if pooled:
        cohort["fleet_seed_intervals"] = [
            (interval["base_seed"], interval["end_seed"])
            for interval in payload["fleet_merge"]["seed_intervals"]
        ]
    return rate, cohort


def _verify_high_regret_source(
    payload: dict[str, Any],
    *,
    candidate: Path,
    candidate_sha256: str,
    champion: Path,
    champion_sha256: str,
    where: str,
    sealed_semantics: dict[str, Any],
    candidate_search_config: dict[str, Any],
    champion_search_config: dict[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "suite",
        "held_out",
        "candidate",
        "champion",
        "passed",
        "verdict",
        "complete_pairs",
        "errors",
        "report",
        "suite_manifest",
        "pentanomial_sprt",
        "pair_diagnostics",
    }
    value = _require_exact_keys(payload, expected_keys, where=where)
    if (
        value["schema_version"] != HIGH_REGRET_SCHEMA
        or value["suite"] != "held_out_high_regret"
    ):
        raise PromotionError(f"{where} has an unexpected high-regret schema/suite")
    if value["held_out"] is not True or value["passed"] is not True:
        raise PromotionError(f"{where} is not a passing held-out high-regret result")
    if value["verdict"] != "H1":
        raise PromotionError(f"{where} high-regret verdict is not passing")
    _verify_bound_checkpoint(
        value["candidate"],
        expected_path=candidate,
        expected_sha256=candidate_sha256,
        where=f"{where}.candidate",
        base=candidate.parent,
    )
    _verify_bound_checkpoint(
        value["champion"],
        expected_path=champion,
        expected_sha256=champion_sha256,
        where=f"{where}.champion",
        base=champion.parent,
    )
    _positive_int(value["complete_pairs"], where=f"{where}.complete_pairs")
    if value["errors"] != []:
        raise PromotionError(f"{where} contains high-regret evaluation errors")
    report_path, _report_ref = _validate_file_ref(
        value["report"], base=candidate.parent, where=f"{where}.report"
    )
    suite_path, suite_ref = _validate_file_ref(
        value["suite_manifest"], base=candidate.parent, where=f"{where}.suite_manifest"
    )
    report = _require_exact_keys(
        _load_json(report_path),
        {
            "schema_version",
            "suite",
            "held_out",
            "suite_manifest",
            "candidate",
            "champion",
            "errors",
            "games",
            "pentanomial_sprt",
            "pair_diagnostics",
            "evaluation_config",
            "planned_engine_identity",
            "engine_identity",
            "archived_state_reconstruction",
        },
        where=f"{where}.report payload",
    )
    if (
        report["schema_version"] != HIGH_REGRET_REPORT_SCHEMA
        or report["suite"] != "held_out_high_regret"
        or report["held_out"] is not True
        or report["errors"] != []
    ):
        raise PromotionError(f"{where}.report is not a clean held-out high-regret run")
    _verify_bound_checkpoint(
        report["candidate"],
        expected_path=candidate,
        expected_sha256=candidate_sha256,
        where=f"{where}.report.candidate",
        base=report_path.parent,
    )
    _verify_bound_checkpoint(
        report["champion"],
        expected_path=champion,
        expected_sha256=champion_sha256,
        where=f"{where}.report.champion",
        base=report_path.parent,
    )
    report_suite_path, report_suite_ref = _validate_file_ref(
        report["suite_manifest"],
        base=report_path.parent,
        where=f"{where}.report.suite_manifest",
    )
    if (
        report_suite_path != suite_path
        or report_suite_ref["sha256"] != suite_ref["sha256"]
    ):
        raise PromotionError(f"{where} and its report bind different held-out suites")
    evaluation_config = report["evaluation_config"]
    if not isinstance(evaluation_config, dict):
        raise PromotionError(f"{where}.report has no evaluation_config")
    normalized_evaluation_config = dict(evaluation_config)
    if (
        normalized_evaluation_config.get("evaluator_rust_featurize") is True
        and sealed_semantics.get("evaluator_rust_featurize") is False
    ):
        if normalized_evaluation_config.get("native_mcts_hot_loop") is not True:
            raise PromotionError(
                f"{where}.report.evaluation_config sealed A1 semantic drift: "
                "Rust featurization lacks the bound native MCTS runtime"
            )
        # Rust featurization is a parity-gated implementation of the same
        # feature contract. The exact evaluator/replay/wheel/runtime hashes
        # below still bind the implementation that produced this report.
        normalized_evaluation_config["evaluator_rust_featurize"] = False
    _require_sealed_semantics(
        normalized_evaluation_config,
        {
            **sealed_semantics,
            "c_scale": candidate_search_config["c_scale"],
            "candidate_c_scale": candidate_search_config["c_scale"],
            "baseline_c_scale": champion_search_config["c_scale"],
            "candidate_n_full": sealed_semantics["n_full"],
            "baseline_n_full": sealed_semantics["n_full"],
            "candidate_n_full_wide": sealed_semantics["n_full_wide"],
            "baseline_n_full_wide": sealed_semantics["n_full_wide"],
            "candidate_n_full_wide_threshold": sealed_semantics[
                "n_full_wide_threshold"
            ],
            "baseline_n_full_wide_threshold": sealed_semantics[
                "n_full_wide_threshold"
            ],
            "candidate_value_readout": sealed_semantics["value_readout"],
            "baseline_value_readout": sealed_semantics["value_readout"],
        },
        where=f"{where}.report.evaluation_config",
    )
    _verify_role_search_pair(
        {
            **candidate_search_config,
            "c_scale": evaluation_config["candidate_c_scale"],
        },
        {
            **champion_search_config,
            "c_scale": evaluation_config["baseline_c_scale"],
        },
        candidate_search_config=candidate_search_config,
        champion_search_config=champion_search_config,
        where=f"{where}.report deployed search",
    )
    planned_engine = _require_exact_keys(
        report["planned_engine_identity"],
        {
            "schema_version", "repo_commit", "native_wheel_sha256",
            "evaluator_sha256", "replay_sha256",
        },
        where=f"{where}.report.planned_engine_identity",
    )
    actual_engine = _require_exact_keys(
        report["engine_identity"],
        {
            "schema_version", "repo_commit", "native_wheel_sha256",
            "evaluator_sha256", "replay_sha256", "native_runtime_sha256",
        },
        where=f"{where}.report.engine_identity",
    )
    if planned_engine.get("schema_version") != HIGH_REGRET_ENGINE_IDENTITY_SCHEMA:
        raise PromotionError(f"{where}.report has an unknown engine identity schema")
    for key, expected in planned_engine.items():
        if actual_engine.get(key) != expected:
            raise PromotionError(f"{where}.report runtime engine identity drift at {key}")
    if not isinstance(planned_engine["repo_commit"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", planned_engine["repo_commit"]
    ):
        raise PromotionError(f"{where}.report engine repo commit is invalid")
    for key in (
        "native_wheel_sha256", "evaluator_sha256", "replay_sha256",
        "native_runtime_sha256",
    ):
        _validate_sha256(actual_engine[key], where=f"{where}.report.engine_identity.{key}")
    reconstruction = _require_exact_keys(
        report["archived_state_reconstruction"],
        {
            "schema_version", "constructor", "map_kind", "action_prefix",
            "chance_stream", "replay_contract",
        },
        where=f"{where}.report.archived_state_reconstruction",
    )
    if reconstruction != {
        "schema_version": ARCHIVED_STATE_RECONSTRUCTION_SCHEMA,
        "constructor": "catanatron_rs.Game.simple",
        "map_kind": "BASE",
        "action_prefix": "[0,target_decision)",
        "chance_stream": "random.Random(game_seed ^ 0xA17E)",
        "replay_contract": REPLAY_CONTRACT,
    }:
        raise PromotionError(f"{where}.report archived-state reconstruction drift")
    suite = _require_exact_keys(
        _load_json(suite_path),
        {
            "schema_version",
            "suite",
            "held_out",
            "source_manifest",
            "validation_seed_manifest",
            "selection",
            "states",
            "suite_sha256",
        },
        where=f"{where}.suite_manifest payload",
    )
    suite_digest = _validate_sha256(
        suite["suite_sha256"], where=f"{where}.suite_manifest.suite_sha256"
    )
    unhashed_suite = dict(suite)
    unhashed_suite.pop("suite_sha256")
    if suite_digest != _digest_value(unhashed_suite):
        raise PromotionError(f"{where} held-out suite semantic digest mismatch")
    if (
        suite["schema_version"] != HIGH_REGRET_SUITE_SCHEMA
        or suite["suite"] != "held_out_high_regret"
        or suite["held_out"] is not True
    ):
        raise PromotionError(f"{where} suite manifest is not a held-out suite")
    source_manifest_path, _source_manifest_ref = _validate_file_ref(
        suite["source_manifest"],
        base=suite_path.parent,
        where=f"{where}.suite_manifest.source_manifest",
    )
    validation_binding = _require_exact_keys(
        suite["validation_seed_manifest"],
        {
            "path",
            "sha256",
            "schema_version",
            "game_seed_count",
            "game_seed_set_sha256",
        },
        where=f"{where}.suite_manifest.validation_seed_manifest",
    )
    selection = suite["selection"]
    states = suite["states"]
    if (
        not isinstance(selection, dict)
        or selection.get("algorithm")
        != "trainer-validation-stratified-regret-unique-game-v3"
        or not isinstance(states, list)
        or not states
        or selection.get("selected_pairs") != len(states)
    ):
        raise PromotionError(f"{where} held-out suite selection is malformed")
    try:
        validate_replay_metadata(selection, states)
        shard_paths, manifest_identities = load_source_manifest(source_manifest_path)
        held_out_seeds, source_validation_binding = load_source_validation_binding(
            source_manifest_path
        )
    except ValueError as error:
        raise PromotionError(f"{where} {error}") from error
    if validation_binding != source_validation_binding:
        raise PromotionError(
            f"{where} suite and source manifest bind different training holdouts"
        )
    expected_strata = {
        "phase:opening",
        "phase:robber_dev",
        "phase:chance",
        "phase:build_trade",
        "41+",
    }
    selected_by_stratum = selection.get("selected_by_stratum")
    stratum_min_pairs = selection.get("stratum_min_pairs")
    if (
        selection.get("selection_scope")
        != "full_authenticated_training_validation_manifest"
        or selection.get("holdout_fraction") != 1.0
        or selection.get("selected_pairs") != 240
        or selection.get("holdout_seed") != 17
        or isinstance(stratum_min_pairs, bool)
        or not isinstance(stratum_min_pairs, int)
        or stratum_min_pairs != 24
        or not isinstance(selected_by_stratum, dict)
        or set(selected_by_stratum) != expected_strata
        or any(value != stratum_min_pairs for value in selected_by_stratum.values())
    ):
        raise PromotionError(
            f"{where} held-out suite violates the fixed stratified policy"
        )
    state_by_pair: dict[int, tuple[int, int]] = {}
    selected_game_seeds: set[int] = set()
    actual_strata = {label: 0 for label in expected_strata}
    inventory_cache: dict[Path, tuple[str, int]] = {}
    source_row_cache: dict[Path, tuple[Any, Any, int]] = {}
    bound_states: list[dict[str, Any]] = []
    for index, raw_state in enumerate(states):
        try:
            state = bind_state_to_manifest(
                raw_state,
                suite_base=suite_path.parent,
                manifest_path=source_manifest_path,
                shard_paths=shard_paths,
                identities=manifest_identities,
                inventory_cache=inventory_cache,
                source_row_cache=source_row_cache,
            )
        except ValueError as error:
            raise PromotionError(f"{where}.suite.states[{index}] {error}") from error
        pair_id = state.get("pair_id")
        game_seed = state.get("game_seed")
        decision_index = state.get("decision_index")
        legal_count = state.get("legal_count")
        if (
            isinstance(pair_id, bool)
            or not isinstance(pair_id, int)
            or pair_id < 0
            or isinstance(game_seed, bool)
            or not isinstance(game_seed, int)
            or game_seed not in held_out_seeds
            or game_seed in selected_game_seeds
            or isinstance(decision_index, bool)
            or not isinstance(decision_index, int)
            or decision_index < 0
            or isinstance(legal_count, bool)
            or not isinstance(legal_count, int)
            or legal_count < 0
            or pair_id in state_by_pair
        ):
            raise PromotionError(f"{where}.suite.states[{index}] has invalid identity")
        state_by_pair[pair_id] = (game_seed, decision_index)
        selected_game_seeds.add(game_seed)
        bound_states.append(state)
        phase = str(state.get("phase", "")).upper()
        if "BUILD_INITIAL_SETTLEMENT" in phase or "BUILD_INITIAL_ROAD" in phase:
            phase_stratum = "opening"
        elif "ROBBER" in phase or "KNIGHT" in phase or "DEVELOPMENT_CARD" in phase:
            phase_stratum = "robber_dev"
        elif "DISCARD" in phase or "ROLL" in phase:
            phase_stratum = "chance"
        else:
            phase_stratum = "build_trade"
        actual_strata[f"phase:{phase_stratum}"] += 1
        if legal_count >= 41:
            actual_strata["41+"] += 1
    if any(actual_strata[label] < stratum_min_pairs for label in expected_strata):
        raise PromotionError(f"{where} held-out suite lacks required stratum coverage")
    if (
        selection.get("eligible_unique_games")
        != len({identity[2] for identity in manifest_identities})
        or selection.get("replay_complete_unique_games", 0) < len(states)
        or selection.get("selected_unique_games") != len(states)
        or len(selected_game_seeds) != len(states)
    ):
        raise PromotionError(f"{where} held-out suite is not source-game independent")
    try:
        validate_replay_trajectories(bound_states)
    except ValueError as error:
        raise PromotionError(f"{where} {error}") from error
    games = report["games"]
    if not isinstance(games, list) or not games:
        raise PromotionError(f"{where}.report has no raw paired games")
    identities: set[tuple[int, str]] = set()
    orientations_by_pair: dict[int, set[str]] = {}
    truncated_by_pair: dict[int, bool] = {}
    orientation_encoding: str | None = None
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise PromotionError(f"{where}.report.games[{index}] is malformed")
        pair_id = game.get("pair_id")
        orientation = game.get("orientation")
        if (
            isinstance(pair_id, bool)
            or not isinstance(pair_id, int)
            or pair_id < 0
            or orientation
            not in {
                "candidate_first",
                "candidate_second",
                "candidate_red",
                "candidate_blue",
            }
        ):
            raise PromotionError(f"{where}.report.games[{index}] lacks pair identity")
        identity = (pair_id, orientation)
        game_encoding = "color" if orientation in {"candidate_red", "candidate_blue"} else "legacy"
        if orientation_encoding is not None and game_encoding != orientation_encoding:
            raise PromotionError(f"{where}.report mixes orientation encodings")
        orientation_encoding = game_encoding
        if game_encoding == "color":
            expected_colors = (
                ("RED", "BLUE")
                if orientation == "candidate_red"
                else ("BLUE", "RED")
            )
            if (game.get("candidate_color"), game.get("baseline_color")) != expected_colors:
                raise PromotionError(
                    f"{where}.report.games[{index}] orientation/color mismatch"
                )
        else:
            colors = (game.get("candidate_color"), game.get("baseline_color"))
            if (colors[0] is None) != (colors[1] is None):
                raise PromotionError(
                    f"{where}.report.games[{index}] has incomplete legacy colors"
                )
            expected_colors = (
                ("RED", "BLUE")
                if orientation == "candidate_first"
                else ("BLUE", "RED")
            )
            if colors[0] is not None and colors != expected_colors:
                raise PromotionError(
                    f"{where}.report.games[{index}] legacy orientation/color mismatch"
                )
        truncated = game.get("truncated")
        outcome = game.get("candidate_won")
        if identity in identities or not isinstance(truncated, bool):
            raise PromotionError(
                f"{where}.report.games[{index}] is duplicate or incomplete"
            )
        if (truncated and outcome is not None) or (
            not truncated and not isinstance(outcome, bool)
        ):
            raise PromotionError(
                f"{where}.report.games[{index}] has inconsistent truncation outcome"
            )
        identities.add(identity)
        if state_by_pair.get(pair_id) != (
            game.get("archived_game_seed"),
            game.get("archived_decision_index"),
        ):
            raise PromotionError(
                f"{where}.report.games[{index}] is not from its held-out suite state"
            )
        orientations_by_pair.setdefault(pair_id, set()).add(orientation)
        truncated_by_pair[pair_id] = truncated_by_pair.get(pair_id, False) or truncated
    if set(orientations_by_pair) != set(state_by_pair) or any(
        orientations
        != (
            {"candidate_red", "candidate_blue"}
            if orientation_encoding == "color"
            else {"candidate_first", "candidate_second"}
        )
        for orientations in orientations_by_pair.values()
    ):
        raise PromotionError(f"{where}.report does not cover every suite pair twice")
    normalized_games = [{**game, "search_won": game["candidate_won"]} for game in games]
    pair_scores, diagnostics = pair_scores_from_h2h_games(normalized_games)
    replayed = evaluate_pentanomial_sprt(
        pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    complete_pairs = (
        diagnostics["ww_pairs"] + diagnostics["split_pairs"] + diagnostics["ll_pairs"]
    )
    truncated_pairs = sum(truncated_by_pair.values())
    if (
        diagnostics["incomplete_pairs"] != truncated_pairs
        or complete_pairs + truncated_pairs != len(state_by_pair)
        or diagnostics != report["pair_diagnostics"]
        or replayed != report["pentanomial_sprt"]
        or diagnostics != value["pair_diagnostics"]
        or replayed != value["pentanomial_sprt"]
        or complete_pairs != value["complete_pairs"]
        or replayed["decision"] != value["verdict"]
        or replayed["decision"] != "H1"
    ):
        raise PromotionError(f"{where} high-regret paired statistics do not replay")


def _verify_bucket_veto_source(
    payload: dict[str, Any],
    *,
    candidate: Path,
    candidate_sha256: str,
    champion: Path,
    champion_sha256: str,
    where: str,
) -> None:
    expected_keys = {
        "schema_version",
        "candidate",
        "champion",
        "veto",
        "veto_buckets",
        "per_bucket",
        "report",
    }
    value = _require_exact_keys(payload, expected_keys, where=where)
    if value["schema_version"] != BUCKET_VETO_SCHEMA:
        raise PromotionError(f"{where} has an unexpected bucket-veto schema")
    _verify_bound_checkpoint(
        value["candidate"],
        expected_path=candidate,
        expected_sha256=candidate_sha256,
        where=f"{where}.candidate",
        base=candidate.parent,
    )
    _verify_bound_checkpoint(
        value["champion"],
        expected_path=champion,
        expected_sha256=champion_sha256,
        where=f"{where}.champion",
        base=champion.parent,
    )
    if value["veto"] is not False or value["veto_buckets"] != []:
        raise PromotionError(f"{where} vetoes promotion")
    report_path, _report_ref = _validate_file_ref(
        value["report"], base=candidate.parent, where=f"{where}.report"
    )
    report = _require_exact_keys(
        _load_json(report_path),
        {"schema_version", "candidate", "champion", "errors", "games"},
        where=f"{where}.report payload",
    )
    if report["schema_version"] != BUCKET_GAME_REPORT_SCHEMA or report["errors"] != []:
        raise PromotionError(f"{where}.report is not a clean bucket-game report")
    _verify_bound_checkpoint(
        report["candidate"],
        expected_path=candidate,
        expected_sha256=candidate_sha256,
        where=f"{where}.report.candidate",
        base=report_path.parent,
    )
    _verify_bound_checkpoint(
        report["champion"],
        expected_path=champion,
        expected_sha256=champion_sha256,
        where=f"{where}.report.champion",
        base=report_path.parent,
    )
    raw_games = report["games"]
    if not isinstance(raw_games, list) or not raw_games:
        raise PromotionError(f"{where}.report has no bucket-labelled games")
    counts: dict[str, list[int]] = {}
    identities: set[tuple[int, str]] = set()
    orientations_by_pair: dict[int, set[str]] = {}
    orientation_encoding: str | None = None
    for index, game in enumerate(raw_games):
        if not isinstance(game, dict):
            raise PromotionError(f"{where}.report.games[{index}] is malformed")
        pair_id = game.get("pair_id")
        orientation = game.get("orientation")
        if (
            isinstance(pair_id, bool)
            or not isinstance(pair_id, int)
            or pair_id < 0
            or orientation
            not in {
                "candidate_first",
                "candidate_second",
                "candidate_red",
                "candidate_blue",
            }
        ):
            raise PromotionError(f"{where}.report.games[{index}] lacks pair identity")
        identity = (pair_id, orientation)
        game_encoding = "color" if orientation in {"candidate_red", "candidate_blue"} else "legacy"
        if orientation_encoding is not None and game_encoding != orientation_encoding:
            raise PromotionError(f"{where}.report mixes orientation encodings")
        orientation_encoding = game_encoding
        if game_encoding == "color":
            expected_colors = (
                ("RED", "BLUE")
                if orientation == "candidate_red"
                else ("BLUE", "RED")
            )
            if (game.get("candidate_color"), game.get("baseline_color")) != expected_colors:
                raise PromotionError(
                    f"{where}.report.games[{index}] orientation/color mismatch"
                )
        elif game.get("candidate_color") is not None or game.get("baseline_color") is not None:
            expected_colors = (
                ("RED", "BLUE")
                if orientation == "candidate_first"
                else ("BLUE", "RED")
            )
            if (game.get("candidate_color"), game.get("baseline_color")) != expected_colors:
                raise PromotionError(
                    f"{where}.report.games[{index}] legacy orientation/color mismatch"
                )
        outcome = game.get("candidate_won")
        labels = game.get("buckets")
        if identity in identities or not isinstance(outcome, bool):
            raise PromotionError(
                f"{where}.report.games[{index}] is duplicate or incomplete"
            )
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) and label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise PromotionError(f"{where}.report.games[{index}] has invalid buckets")
        identities.add(identity)
        orientations_by_pair.setdefault(pair_id, set()).add(orientation)
        for label in labels:
            bucket_counts = counts.setdefault(label, [0, 0])
            bucket_counts[0 if outcome else 1] += 1
    if any(
        orientations
        != (
            {"candidate_red", "candidate_blue"}
            if orientation_encoding == "color"
            else {"candidate_first", "candidate_second"}
        )
        for orientations in orientations_by_pair.values()
    ):
        raise PromotionError(f"{where}.report contains an incomplete bucket pair")
    replayed_buckets: dict[str, dict[str, Any]] = {}
    replayed_veto: list[str] = []
    for label, (wins, losses) in sorted(counts.items()):
        count = wins + losses
        winrate = wins / count
        status = (
            "insufficient_data"
            if count < MIN_BUCKET_GAMES
            else "pass"
            if winrate >= MIN_BUCKET_WIN_RATE
            else "fail"
        )
        replayed_buckets[label] = {"status": status, "n": count, "winrate": winrate}
        if status == "fail":
            replayed_veto.append(label)
    if (
        value["per_bucket"] != replayed_buckets
        or value["veto_buckets"] != replayed_veto
        or value["veto"] is not bool(replayed_veto)
    ):
        raise PromotionError(f"{where} bucket outcomes do not replay from raw games")
    if set(replayed_buckets) != REQUIRED_PROMOTION_BUCKETS:
        raise PromotionError(
            f"{where} bucket coverage mismatch: "
            f"missing={sorted(REQUIRED_PROMOTION_BUCKETS - set(replayed_buckets))} "
            f"unexpected={sorted(set(replayed_buckets) - REQUIRED_PROMOTION_BUCKETS)}"
        )
    buckets = value["per_bucket"]
    if not isinstance(buckets, dict) or not buckets:
        raise PromotionError(f"{where}.per_bucket must be a non-empty object")
    for name, result in buckets.items():
        if not isinstance(name, str) or not isinstance(result, dict):
            raise PromotionError(f"{where}.per_bucket is malformed")
        if result.get("status") != "pass":
            raise PromotionError(f"{where} bucket {name!r} is not a pass")
        count = _positive_int(result.get("n"), where=f"{where}.per_bucket[{name}].n")
        if count < MIN_BUCKET_GAMES:
            raise PromotionError(f"{where} bucket {name!r} has insufficient data")
        winrate = _finite_number(
            result.get("winrate"),
            where=f"{where}.per_bucket[{name}].winrate",
            minimum=0.0,
        )
        if winrate < MIN_BUCKET_WIN_RATE:
            raise PromotionError(
                f"{where} bucket {name!r} regresses by more than the fixed 5% limit"
            )


def _verify_promotion_evidence(
    path: Path,
    *,
    kind: str,
    contract: dict[str, Any],
    expected_readout: str = "scalar",
    candidate: dict[str, Any],
    champion: dict[str, Any],
) -> dict[str, Any]:
    contract_sha256 = contract["contract_sha256"]
    sealed_semantics = _sealed_evaluation_semantics(contract)
    value = _load_json(path)
    expected_keys = {
        "schema_version",
        "kind",
        "passed",
        "verdict",
        "contract_sha256",
        "candidate",
        "champion",
        "sources",
        "result",
        "evidence_sha256",
    }
    value = _require_exact_keys(value, expected_keys, where=f"{kind} evidence")
    declared = _validate_sha256(
        value["evidence_sha256"], where=f"{kind} evidence.evidence_sha256"
    )
    unhashed = dict(value)
    unhashed.pop("evidence_sha256")
    if declared != _digest_value(unhashed):
        raise PromotionError(f"{kind} evidence semantic digest mismatch")
    if value["schema_version"] != EVIDENCE_SCHEMA or value["kind"] != kind:
        raise PromotionError(f"{kind} evidence schema/kind mismatch")
    if value["passed"] is not True:
        raise PromotionError(f"{kind} evidence is not passing")
    if value["contract_sha256"] != contract_sha256:
        raise PromotionError(f"{kind} evidence binds a different A1 contract")
    candidate_path = Path(candidate["path"])
    champion_path = Path(champion["path"])
    _verify_bound_checkpoint(
        value["candidate"],
        expected_path=candidate_path,
        expected_sha256=candidate["sha256"],
        where=f"{kind} evidence.candidate",
        base=path.parent,
    )
    _verify_bound_checkpoint(
        value["champion"],
        expected_path=champion_path,
        expected_sha256=champion["sha256"],
        where=f"{kind} evidence.champion",
        base=path.parent,
    )
    sources = value["sources"]
    if not isinstance(sources, list) or not sources:
        raise PromotionError(f"{kind} evidence.sources must be non-empty")
    source_by_role: dict[str, tuple[Path, dict[str, Any]]] = {}
    for index, raw in enumerate(sources):
        item = _require_exact_keys(
            raw, {"role", "path", "sha256"}, where=f"{kind} evidence.sources[{index}]"
        )
        role = item["role"]
        if not isinstance(role, str) or role in source_by_role:
            raise PromotionError(
                f"{kind} evidence source role is invalid or duplicated"
            )
        source_path, _verified = _validate_file_ref(
            {"path": item["path"], "sha256": item["sha256"]},
            base=path.parent,
            where=f"{kind} evidence source {role}",
        )
        source_by_role[role] = (source_path, _load_json(source_path))
    result = value["result"]
    if not isinstance(result, dict):
        raise PromotionError(f"{kind} evidence.result must be an object")
    if kind == "mechanism_calibration":
        if set(source_by_role) != {"candidate_calibration", "champion_calibration"}:
            raise PromotionError("mechanism calibration source roles mismatch")
        result = _require_exact_keys(
            result,
            {"value_readout", "max_rmse_regression"},
            where="mechanism calibration evidence.result",
        )
        if result["value_readout"] != expected_readout:
            raise PromotionError("mechanism calibration value_readout drift")
        candidate_rmse, candidate_cohort = _verify_calibration_source(
            source_by_role["candidate_calibration"][1],
            source_path=source_by_role["candidate_calibration"][0],
            checkpoint=candidate_path,
            expected_readout=expected_readout,
            expected_value_scale=sealed_semantics["value_scale"],
            expected_value_squash=sealed_semantics["value_squash"],
            where="candidate calibration",
        )
        champion_rmse, champion_cohort = _verify_calibration_source(
            source_by_role["champion_calibration"][1],
            source_path=source_by_role["champion_calibration"][0],
            checkpoint=champion_path,
            expected_readout=expected_readout,
            expected_value_scale=sealed_semantics["value_scale"],
            expected_value_squash=sealed_semantics["value_squash"],
            where="champion calibration",
            contract=contract,
            allow_legacy_incumbent=True,
        )
        if candidate_cohort != champion_cohort:
            raise PromotionError(
                "candidate and champion calibration reports use different cohorts"
            )
        max_regression = _finite_number(
            result["max_rmse_regression"],
            where="mechanism calibration max_rmse_regression",
            minimum=0.0,
        )
        if max_regression != MAX_CALIBRATION_RMSE_REGRESSION:
            raise PromotionError(
                "mechanism calibration regression limit differs from the fixed policy"
            )
        if candidate_rmse > champion_rmse + max_regression:
            raise PromotionError(
                "candidate calibration exceeds the allowed RMSE regression"
            )
        if value["verdict"] != "pass":
            raise PromotionError("mechanism calibration verdict is not pass")
    elif kind == "internal_h2h":
        if set(source_by_role) != {"internal_h2h"}:
            raise PromotionError("internal H2H source roles mismatch")
        _verify_internal_h2h_source(
            source_by_role["internal_h2h"][1],
            candidate=candidate_path,
            champion=champion_path,
            where="internal H2H",
            sealed_semantics=sealed_semantics,
            candidate_search_config=candidate["search_config"],
            champion_search_config=champion["search_config"],
        )
        if value["verdict"] != "H1" or result:
            raise PromotionError("internal H2H envelope verdict/result drift")
    elif kind == "external_panel":
        if set(source_by_role) != {"candidate_panel", "champion_panel"}:
            raise PromotionError("external panel source roles mismatch")
        candidate_panel = source_by_role["candidate_panel"][1]
        champion_panel = source_by_role["champion_panel"][1]
        for panel_name, panel in (
            ("candidate external panel", candidate_panel),
            ("champion external panel", champion_panel),
        ):
            _verify_evaluation_baseline_binding(
                panel.get("evaluation_binding"),
                champion_path=champion_path,
                champion_sha256=champion["sha256"],
                champion_search_config=champion["search_config"],
                base=champion_path.parent,
                where=f"{panel_name}.evaluation_binding",
            )
            planned_engine = _require_exact_keys(
                panel.get("planned_engine_identity"),
                {
                    "schema_version",
                    "repo_commit",
                    "native_wheel_sha256",
                    "python_referee_sha256",
                },
                where=f"{panel_name}.planned_engine_identity",
            )
            actual_engine = _require_exact_keys(
                panel.get("engine_identity"),
                {
                    "schema_version",
                    "repo_commit",
                    "native_wheel_sha256",
                    "native_runtime_sha256",
                    "python_referee_sha256",
                },
                where=f"{panel_name}.engine_identity",
            )
            for key, expected in planned_engine.items():
                if actual_engine.get(key) != expected:
                    raise PromotionError(
                        f"{panel_name} runtime engine identity drift at {key}"
                    )
            _validate_sha256(
                actual_engine["native_runtime_sha256"],
                where=f"{panel_name}.engine_identity.native_runtime_sha256",
            )
        if (
            candidate_panel.get("baseline_bot") != "catanatron_value"
            or champion_panel.get("baseline_bot") != "catanatron_value"
        ):
            raise PromotionError("external panels must use catanatron_value")
        candidate_rate, candidate_cohort = _verify_external_panel_source(
            candidate_panel,
            checkpoint=candidate_path,
            checkpoint_md5=candidate["md5"],
            where="candidate external panel",
            sealed_semantics=sealed_semantics,
            deployed_search_config=candidate["search_config"],
        )
        champion_rate, champion_cohort = _verify_external_panel_source(
            champion_panel,
            checkpoint=champion_path,
            checkpoint_md5=champion["md5"],
            where="champion external panel",
            sealed_semantics=sealed_semantics,
            deployed_search_config=champion["search_config"],
        )
        _verify_role_search_pair(
            candidate_panel["search_config"],
            champion_panel["search_config"],
            candidate_search_config=candidate["search_config"],
            champion_search_config=champion["search_config"],
            where="external panel deployed agents",
        )
        if candidate_cohort != champion_cohort:
            raise PromotionError(
                "candidate and champion external panels use different cohorts/configs"
            )
        result = _require_exact_keys(
            result,
            {"max_win_rate_regression"},
            where="external panel evidence.result",
        )
        max_regression = _finite_number(
            result["max_win_rate_regression"],
            where="external panel max_win_rate_regression",
            minimum=0.0,
        )
        if max_regression != MAX_EXTERNAL_WIN_RATE_REGRESSION:
            raise PromotionError(
                "external panel regression limit differs from the fixed policy"
            )
        try:
            paired_external = compare_matched_external_panels(
                candidate_panel,
                champion_panel,
                noninferiority_margin=max_regression,
            )
        except ExternalPanelComparisonError as error:
            raise PromotionError(
                f"external panels cannot form a paired noninferiority test: {error}"
            ) from error
        if (
            paired_external["candidate_win_rate"] != candidate_rate
            or paired_external["champion_win_rate"] != champion_rate
        ):
            raise PromotionError("external paired comparison does not replay summary rates")
        if not paired_external["noninferiority"]["passed"]:
            raise PromotionError(
                "candidate external noninferiority is unresolved at alpha=0.05: "
                f"delta={paired_external['candidate_minus_champion']:.6f}, "
                f"one-sided lower={paired_external['noninferiority']['one_sided_95_lower']:.6f}, "
                f"required >= {-max_regression:.6f}"
            )
        if value["verdict"] != "pass":
            raise PromotionError("external panel envelope verdict is not pass")
    elif kind == "high_regret":
        if set(source_by_role) != {"high_regret"} or result:
            raise PromotionError("high-regret source roles/result mismatch")
        _verify_high_regret_source(
            source_by_role["high_regret"][1],
            candidate=candidate_path,
            candidate_sha256=candidate["sha256"],
            champion=champion_path,
            champion_sha256=champion["sha256"],
            where="high-regret comparison",
            sealed_semantics=sealed_semantics,
            candidate_search_config=candidate["search_config"],
            champion_search_config=champion["search_config"],
        )
        if value["verdict"] != "pass":
            raise PromotionError("high-regret envelope verdict is not pass")
    elif kind == "bucket_veto":
        if set(source_by_role) != {"bucket_veto"} or result:
            raise PromotionError("bucket-veto source roles/result mismatch")
        _verify_bucket_veto_source(
            source_by_role["bucket_veto"][1],
            candidate=candidate_path,
            candidate_sha256=candidate["sha256"],
            champion=champion_path,
            champion_sha256=champion["sha256"],
            where="bucket-veto result",
        )
        if value["verdict"] != "pass":
            raise PromotionError("bucket-veto envelope verdict is not pass")
    else:  # pragma: no cover - caller constrains the set.
        raise PromotionError(f"unsupported promotion evidence kind {kind}")
    return value


def _contiguous_seed_intervals(
    seeds: set[int], *, kind: str, where: str
) -> list[dict[str, Any]]:
    if not seeds:
        raise PromotionError(f"{where} has no explicit game seeds")
    ordered = sorted(seeds)
    intervals: list[dict[str, Any]] = []
    start = previous = ordered[0]
    for seed in ordered[1:]:
        if seed == previous + 1:
            previous = seed
            continue
        intervals.append({"kind": kind, "base_seed": start, "end_seed": previous + 1})
        start = previous = seed
    intervals.append({"kind": kind, "base_seed": start, "end_seed": previous + 1})
    return intervals


def _explicit_game_seeds(payload: dict[str, Any], *, where: str) -> set[int]:
    games = payload.get("games")
    if not isinstance(games, list) or not games:
        raise PromotionError(f"{where} has no retained games for cohort isolation")
    seeds: set[int] = set()
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise PromotionError(f"{where}.games[{index}] is not an object")
        seed = game.get("game_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise PromotionError(
                f"{where}.games[{index}] has no non-negative integer game_seed; "
                "promotion cohorts must retain exact seed identity"
            )
        seeds.add(seed)
    return seeds


def _promotion_cohort_intervals(
    evidence_path: Path, evidence: dict[str, Any], *, kind: str
) -> list[dict[str, Any]]:
    """Recover the selection-sensitive final cohort from verified raw evidence."""
    sources = {
        str(item["role"]): _absolute(item["path"], base=evidence_path.parent)
        for item in evidence["sources"]
    }
    if kind == "internal_h2h":
        payload = _load_json(sources["internal_h2h"])
        return _contiguous_seed_intervals(
            _explicit_game_seeds(payload, where="internal H2H"),
            kind=kind,
            where="internal H2H",
        )
    if kind == "external_panel":
        candidate_payload = _load_json(sources["candidate_panel"])
        champion_payload = _load_json(sources["champion_panel"])
        candidate_seeds = _explicit_game_seeds(
            candidate_payload, where="candidate external panel"
        )
        champion_seeds = _explicit_game_seeds(
            champion_payload, where="champion external panel"
        )
        if candidate_seeds != champion_seeds:
            raise PromotionError(
                "candidate and champion external panels have different game-seed cohorts"
            )
        return _contiguous_seed_intervals(
            candidate_seeds,
            kind=kind,
            where="external panel",
        )
    return []


def _seed_interval(
    raw: Any, *, where: str, kind: str | None = None
) -> dict[str, Any]:
    value = _require_exact_keys(
        raw, {"base_seed", "end_seed"}, where=where
    )
    base_seed = value["base_seed"]
    end_seed = value["end_seed"]
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or base_seed < 0
        or isinstance(end_seed, bool)
        or not isinstance(end_seed, int)
        or end_seed <= base_seed
    ):
        raise PromotionError(f"{where} must be a non-empty non-negative half-open range")
    result: dict[str, Any] = {"base_seed": base_seed, "end_seed": end_seed}
    if kind is not None:
        result["kind"] = kind
    return result


def _verify_cohort_exclusions(
    path: Path,
    *,
    contract_sha256: str,
    candidate_sha256: str,
    final_intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind prior selection cohorts and prove the final cohort is fresh."""
    value = _require_exact_keys(
        _load_json(path),
        {
            "schema_version",
            "contract_sha256",
            "candidate_sha256",
            "cohorts",
            "manifest_sha256",
        },
        where="promotion cohort-exclusions manifest",
    )
    if value["schema_version"] != COHORT_EXCLUSIONS_SCHEMA:
        raise PromotionError(
            f"cohort-exclusions schema must be {COHORT_EXCLUSIONS_SCHEMA!r}"
        )
    declared = _validate_sha256(
        value["manifest_sha256"], where="cohort exclusions.manifest_sha256"
    )
    unhashed = dict(value)
    unhashed.pop("manifest_sha256")
    if declared != _digest_value(unhashed):
        raise PromotionError("cohort-exclusions semantic digest mismatch")
    if value["contract_sha256"] != contract_sha256:
        raise PromotionError("cohort exclusions bind a different A1 contract")
    if value["candidate_sha256"] != candidate_sha256:
        raise PromotionError("cohort exclusions bind a different candidate")
    cohorts = value["cohorts"]
    if not isinstance(cohorts, list) or not cohorts:
        raise PromotionError(
            "cohort exclusions must bind at least one prior diagnostic/adjudication cohort"
        )
    excluded: list[dict[str, Any]] = []
    labels: set[str] = set()
    bound_sources: list[dict[str, str]] = []
    for index, raw in enumerate(cohorts):
        cohort = _require_exact_keys(
            raw,
            {"label", "kind", "source", "seed_intervals"},
            where=f"cohort exclusions.cohorts[{index}]",
        )
        label = cohort["label"]
        kind = cohort["kind"]
        if (
            not isinstance(label, str)
            or not label.strip()
            or label in labels
            or not isinstance(kind, str)
            or not kind.strip()
        ):
            raise PromotionError("cohort exclusion label/kind is invalid or duplicated")
        labels.add(label)
        _source_path, source_ref = _validate_file_ref(
            cohort["source"],
            base=path.parent,
            where=f"cohort exclusions.cohorts[{index}].source",
        )
        bound_sources.append({"label": label, **source_ref})
        intervals = cohort["seed_intervals"]
        if not isinstance(intervals, list) or not intervals:
            raise PromotionError(
                f"cohort exclusions.cohorts[{index}].seed_intervals must be non-empty"
            )
        for interval_index, interval in enumerate(intervals):
            excluded.append(
                {
                    "label": label,
                    **_seed_interval(
                        interval,
                        where=(
                            f"cohort exclusions.cohorts[{index}]."
                            f"seed_intervals[{interval_index}]"
                        ),
                        kind=kind,
                    ),
                }
            )
    overlaps: list[dict[str, Any]] = []
    for final in final_intervals:
        for prior in excluded:
            overlap_start = max(final["base_seed"], prior["base_seed"])
            overlap_end = min(final["end_seed"], prior["end_seed"])
            if overlap_start < overlap_end:
                overlaps.append(
                    {
                        "final_kind": final["kind"],
                        "prior_kind": prior["kind"],
                        "prior_label": prior["label"],
                        "base_seed": overlap_start,
                        "end_seed": overlap_end,
                    }
                )
    if overlaps:
        raise PromotionError(
            "final promotion cohort overlaps a prior diagnostic/adjudication cohort: "
            f"{overlaps[:5]}"
        )
    return {
        "manifest": {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "manifest_sha256": declared,
        },
        "bound_sources": bound_sources,
        "excluded_seed_intervals": excluded,
        "final_seed_intervals": final_intervals,
        "overlap_count": 0,
    }


def _verify_adjudication(
    path: Path,
    *,
    contract: dict[str, Any],
    contract_lock: Path,
    training_receipt: Path,
    registry: ChampionRegistry,
    current_pointer: Path,
    legacy_snapshot: _LegacyPromotionSnapshot | None = None,
) -> dict[str, Any]:
    raw = _load_json(path)
    common_keys = {
        "schema_version",
        "passed",
        "decision",
        "contract_sha256",
        "candidate",
        "champion",
        "checks",
        "nth_confirmation_required",
        "evidence",
        "adjudication_sha256",
    }
    schema_version = raw.get("schema_version")
    if schema_version == ADJUDICATION_SCHEMA:
        expected_keys = common_keys | {"nth_confirmation"}
    elif schema_version == PREVIOUS_ADJUDICATION_SCHEMA:
        expected_keys = common_keys | {"nth_confirmation_passed"}
    else:
        raise PromotionError(
            "adjudication schema must be "
            f"{ADJUDICATION_SCHEMA!r} (or historical {PREVIOUS_ADJUDICATION_SCHEMA!r})"
        )
    value = _require_exact_keys(raw, expected_keys, where="adjudication")
    legacy_v1 = schema_version == PREVIOUS_ADJUDICATION_SCHEMA
    declared_digest = _validate_sha256(
        value["adjudication_sha256"], where="adjudication.adjudication_sha256"
    )
    unhashed = dict(value)
    unhashed.pop("adjudication_sha256")
    if declared_digest != _digest_value(unhashed):
        raise PromotionError("adjudication semantic digest mismatch")
    if value["passed"] is not True or value["decision"] != "promote":
        raise PromotionError("adjudication is not a typed passing promote decision")
    contract_sha = contract["contract_sha256"]
    if value["contract_sha256"] != contract_sha:
        raise PromotionError("adjudication binds a different sealed A1 contract")

    base = path.parent
    candidate_raw = _require_exact_keys(
        value["candidate"],
        {"path", "sha256", "version", "training_report", "agent_identity"},
        where="candidate",
    )
    candidate_path, candidate_ref = _validate_file_ref(
        {"path": candidate_raw["path"], "sha256": candidate_raw["sha256"]},
        base=base,
        where="candidate",
    )
    training_path, training_ref = _validate_file_ref(
        candidate_raw["training_report"], base=base, where="candidate.training_report"
    )
    training_receipt_schema = _load_json(training_receipt).get("schema_version")
    production_l1_completion = (
        training_receipt_schema
        in {
            "a1-production-l1-rerun-completion-v1",
            "a1-production-l1-rerun-completion-v2",
        }
    )
    production_target_gather_completion = (
        training_receipt_schema
        == "a1-production-target-gather-retrain-completion-v1"
    )
    production_temperature_completion = (
        training_receipt_schema
        in {
            "a1-production-temperature-replication-completion-v1",
            "a1-production-temperature-replication-completion-v2",
        }
    )
    production_learner_dose = (
        learner_dose.PARETO_SELECTED_DOSE
        if training_receipt_schema
        in {
            "a1-production-l1-rerun-completion-v2",
            "a1-production-temperature-replication-completion-v2",
        }
        else None
    )
    training_report_payload = _verify_training_report(
        training_path,
        contract=contract,
        contract_sha256=contract_sha,
        candidate_path=candidate_path,
        candidate_sha256=candidate_ref["sha256"],
        production_l1_completion=production_l1_completion,
        production_target_gather_completion=production_target_gather_completion,
        production_temperature_completion=production_temperature_completion,
        production_learner_dose=production_learner_dose,
    )
    training_receipt_ref = _verify_one_dose_training_receipt(
        training_receipt,
        contract_lock=contract_lock,
        contract=contract,
        candidate_path=candidate_path,
        candidate_sha256=candidate_ref["sha256"],
        training_report_path=training_path,
        training_report_sha256=training_ref["sha256"],
        legacy_snapshot=legacy_snapshot,
    )
    champion_raw = _require_exact_keys(
        value["champion"],
        {"path", "sha256", "version", "agent_identity"},
        where="champion",
    )
    champion_path, champion_ref = _validate_file_ref(
        {"path": champion_raw["path"], "sha256": champion_raw["sha256"]},
        base=base,
        where="champion",
    )
    curriculum_parent = training_report_payload.get("a1_curriculum_parent")
    lineage_value = training_report_payload.get("a1_lineage_dose")
    function_upgrade = (
        lineage_value.get("function_preserving_upgrade")
        if isinstance(lineage_value, dict)
        else None
    )
    evaluation_parent_sha = training_receipt_ref.get("evaluation_parent_sha256")
    if evaluation_parent_sha is None:
        evaluation_parent_sha = (
            curriculum_parent.get("generation_producer_sha256")
            if isinstance(curriculum_parent, dict)
            else function_upgrade.get("source_checkpoint_sha256")
            if isinstance(function_upgrade, dict)
            else training_report_payload.get("init_checkpoint_sha256")
        )
    if evaluation_parent_sha != champion_ref["sha256"]:
        raise PromotionError(
            "candidate training parent/init checkpoint differs from adjudicated "
            "promotion baseline"
        )
    # Ordinary candidates also bind the exact init path. Curriculum candidates
    # legitimately warm-start from an intermediate dose, while their sealed
    # generation producer remains the promotion parent checked above.
    init_path = training_report_payload.get("init_checkpoint")
    if curriculum_parent is None and function_upgrade is None and init_path is not None:
        if _absolute(init_path, base=training_path.parent) != champion_path:
            raise PromotionError(
                "candidate training init path differs from adjudicated promotion baseline"
            )
    if (
        candidate_path == champion_path
        or candidate_ref["sha256"] == champion_ref["sha256"]
    ):
        raise PromotionError(
            "candidate and incumbent champion must have distinct bytes"
        )
    for label, raw_version in (
        ("candidate.version", candidate_raw["version"]),
        ("champion.version", champion_raw["version"]),
    ):
        if (
            isinstance(raw_version, bool)
            or not isinstance(raw_version, int)
            or raw_version < 0
        ):
            raise PromotionError(f"{label} must be a non-negative integer")
    if candidate_raw["version"] != champion_raw["version"] + 1:
        raise PromotionError("candidate version must be exactly incumbent version + 1")

    incumbent = registry.get_role("generator_champion")
    if incumbent is None:
        raise PromotionError("authoritative registry has no generator_champion")
    if str(Path(incumbent.checkpoint_path).expanduser().resolve()) != str(
        champion_path
    ):
        raise PromotionError(
            "adjudicated champion path differs from registry generator_champion"
        )
    if incumbent.md5 != _md5(champion_path):
        raise PromotionError(
            "registry generator_champion md5 differs from incumbent bytes"
        )
    if incumbent.version != champion_raw["version"]:
        raise PromotionError("adjudicated champion version differs from registry")
    if _read_current_pointer(current_pointer) != str(champion_path):
        raise PromotionError(
            "CURRENT_CHAMPION pointer differs from adjudicated incumbent"
        )

    candidate_binding = {**candidate_ref, "md5": _md5(candidate_path)}
    champion_binding = {**champion_ref, "md5": _md5(champion_path)}
    candidate_search_config = _candidate_search_config(contract)
    champion_search_config = _incumbent_search_config(
        contract,
        registry=registry,
        champion_path=champion_path,
        champion_sha256=champion_ref["sha256"],
    )
    candidate_identity = _verify_agent_identity(
        candidate_raw["agent_identity"],
        expected_search_config=candidate_search_config,
        checkpoint_path=candidate_path,
        checkpoint_sha256=candidate_ref["sha256"],
        base=base,
        where="candidate.agent_identity",
    )
    champion_identity = _verify_agent_identity(
        champion_raw["agent_identity"],
        expected_search_config=champion_search_config,
        checkpoint_path=champion_path,
        checkpoint_sha256=champion_ref["sha256"],
        base=base,
        where="champion.agent_identity",
    )
    _verify_role_search_pair(
        candidate_identity["search_config"],
        champion_identity["search_config"],
        candidate_search_config=candidate_search_config,
        champion_search_config=champion_search_config,
        where="adjudication deployed agents",
    )
    candidate_binding["agent_identity"] = candidate_identity
    candidate_binding["search_config"] = candidate_identity["search_config"]
    champion_binding["agent_identity"] = champion_identity
    champion_binding["search_config"] = champion_identity["search_config"]

    checks = _require_exact_keys(value["checks"], REQUIRED_CHECKS, where="checks")
    failed_checks = sorted(
        name for name, passed in checks.items() if passed is not True
    )
    if failed_checks:
        raise PromotionError(f"adjudication has non-passing checks: {failed_checks}")
    next_count = registry.promotion_count("generator_champion") + 1
    nth_required = next_count % 3 == 0
    if value["nth_confirmation_required"] is not nth_required:
        raise PromotionError(
            "adjudication every-third confirmation requirement disagrees with registry count"
        )
    nth_confirmation = None
    nth_confirmation_intervals: list[dict[str, Any]] = []
    if legacy_v1:
        if nth_required:
            raise PromotionError(
                "historical v1 adjudication cannot authorize an every-third "
                "promotion without immutable n64 evidence"
            )
        if value["nth_confirmation_passed"] not in {False, None}:
            raise PromotionError(
                "historical v1 non-third adjudication cannot assert a naked "
                "n64 confirmation boolean"
            )
    elif nth_required:
        if value["nth_confirmation"] is None:
            raise PromotionError(
                "required every-third n64 confirmation lacks immutable evidence"
            )
        nth_confirmation, nth_confirmation_intervals = _verify_n64_confirmation(
            value["nth_confirmation"],
            base=base,
            contract=contract,
            candidate=candidate_binding,
            champion=champion_binding,
        )
    elif value["nth_confirmation"] is not None:
        raise PromotionError("non-required nth confirmation must be null")

    evidence = value["evidence"]
    if not isinstance(evidence, list):
        raise PromotionError("adjudication.evidence must be a list")
    evidence_by_kind: dict[str, dict[str, Any]] = {}
    final_cohort_intervals: list[dict[str, Any]] = list(nth_confirmation_intervals)
    for index, record in enumerate(evidence):
        item = _require_exact_keys(
            record, {"kind", "path", "sha256"}, where=f"evidence[{index}]"
        )
        kind = item["kind"]
        if not isinstance(kind, str) or kind in evidence_by_kind:
            raise PromotionError(f"evidence[{index}].kind is invalid or duplicated")
        evidence_path, verified = _validate_file_ref(
            {"path": item["path"], "sha256": item["sha256"]},
            base=base,
            where=f"evidence[{index}]",
        )
        evidence_value = _verify_promotion_evidence(
            evidence_path,
            kind=kind,
            contract=contract,
            expected_readout=_contract_value_readout(contract),
            candidate=candidate_binding,
            champion=champion_binding,
        )
        final_cohort_intervals.extend(
            _promotion_cohort_intervals(
                evidence_path, evidence_value, kind=kind
            )
        )
        evidence_by_kind[kind] = {"kind": kind, **verified}
    missing_evidence = REQUIRED_EVIDENCE_KINDS - set(evidence_by_kind)
    unexpected_evidence = set(evidence_by_kind) - REQUIRED_EVIDENCE_KINDS
    if missing_evidence or unexpected_evidence:
        raise PromotionError(
            f"adjudication evidence kinds differ: missing={sorted(missing_evidence)} "
            f"unexpected={sorted(unexpected_evidence)}"
        )

    return {
        "candidate": {
            **candidate_ref,
            "version": candidate_raw["version"],
            "md5": candidate_binding["md5"],
            "training_report": training_ref,
            "agent_identity": candidate_identity,
        },
        "training_receipt": training_receipt_ref,
        "champion": {
            **champion_ref,
            "version": champion_raw["version"],
            "md5": champion_binding["md5"],
            "agent_identity": champion_identity,
        },
        "evidence": [evidence_by_kind[kind] for kind in sorted(evidence_by_kind)],
        "adjudication_sha256": declared_digest,
        "next_promotion_count": next_count,
        "nth_confirmation_required": nth_required,
        "nth_confirmation": nth_confirmation,
        "final_cohort_intervals": final_cohort_intervals,
    }


def _stage_registry(
    registry_path: Path,
    *,
    verified: dict[str, Any],
    contract_sha256: str,
    adjudication_path: Path,
    receipt_path: Path,
    reason: str,
) -> tuple[bytes, int]:
    stage = registry_path.parent / f".{registry_path.name}.{uuid.uuid4().hex}.stage"
    _write_new_bytes(stage, registry_path.read_bytes())
    try:
        registry = ChampionRegistry.load(stage)
        champion = verified["champion"]
        candidate = verified["candidate"]
        provenance = {
            "a1_contract_sha256": contract_sha256,
            "a1_promotion_adjudication": str(adjudication_path),
            "a1_promotion_adjudication_sha256": verified["adjudication_sha256"],
            "a1_one_dose_training_receipt": verified["training_receipt"]["path"],
            "a1_one_dose_training_receipt_sha256": verified["training_receipt"][
                "sha256"
            ],
            "a1_one_dose_execution_binding_sha256": verified["training_receipt"][
                "execution_binding_sha256"
            ],
            "a1_promotion_receipt": str(receipt_path),
            "a1_candidate_agent_identity_sha256": candidate["agent_identity"][
                "agent_identity_sha256"
            ],
            "a1_candidate_search_config": candidate["agent_identity"]["search_config"],
            "a1_champion_agent_identity_sha256": champion["agent_identity"][
                "agent_identity_sha256"
            ],
            "a1_champion_search_config": champion["agent_identity"]["search_config"],
            "fleet_ckpt_updated": False,
        }
        registry.append_pool(
            champion["path"],
            expected_md5=champion["md5"],
            version=champion["version"],
            provenance=provenance,
            status="active",
            reason="dethroned A1 generator champion",
        )
        registry.set_role(
            "generator_champion",
            candidate["path"],
            expected_md5=candidate["md5"],
            version=candidate["version"],
            provenance=provenance,
            reason=reason,
        )
        count = registry.record_promotion("generator_champion")
        if count != verified["next_promotion_count"]:
            raise PromotionError("staged registry promotion count drift")
        registry.save()
        return stage.read_bytes(), count
    finally:
        stage.unlink(missing_ok=True)
        stage.with_suffix(stage.suffix + ".tmp").unlink(missing_ok=True)


def _backup_paths(receipt_path: Path) -> tuple[Path, Path]:
    return (
        receipt_path.with_name(receipt_path.name + ".registry.before"),
        receipt_path.with_name(receipt_path.name + ".current.before"),
    )


def prepare_promotion(
    *,
    registry_path: Path,
    current_pointer: Path,
    contract_lock: Path,
    adjudication_path: Path,
    training_receipt: Path,
    cohort_exclusions: Path,
    receipt_path: Path,
    reason: str,
    legacy_contract_attestation: Path | None = None,
    verify_lock_fn: Callable[..., dict[str, Any]] = a1_contract.verify_lock,
) -> dict[str, Any]:
    registry_path = _canonical_existing_file(
        registry_path, where="authoritative registry"
    )
    current_pointer = _canonical_existing_file(
        current_pointer, where="CURRENT_CHAMPION pointer"
    )
    contract_lock = _canonical_existing_file(contract_lock, where="A1 contract lock")
    adjudication_path = _canonical_existing_file(
        adjudication_path, where="promotion adjudication"
    )
    training_receipt = _canonical_existing_file(
        training_receipt, where="A1 one-dose training receipt"
    )
    cohort_exclusions = _canonical_existing_file(
        cohort_exclusions, where="promotion cohort-exclusions manifest"
    )
    receipt_path = _canonical_new_file(receipt_path, where="promotion receipt")
    if registry_path.stat().st_size == 0:
        raise PromotionError(
            "authoritative registry must be an existing non-empty file"
        )
    contract, legacy_snapshot = _verify_contract_with_snapshot(
        contract_lock,
        verify_lock_fn=verify_lock_fn,
        legacy_contract_attestation=legacy_contract_attestation,
        expected_training_receipt=training_receipt,
    )
    registry = ChampionRegistry.load(registry_path)
    verified = _verify_adjudication(
        adjudication_path,
        contract=contract,
        contract_lock=contract_lock,
        training_receipt=training_receipt,
        registry=registry,
        current_pointer=current_pointer,
        legacy_snapshot=legacy_snapshot,
    )
    cohort_disjointness = _verify_cohort_exclusions(
        cohort_exclusions,
        contract_sha256=contract["contract_sha256"],
        candidate_sha256=verified["candidate"]["sha256"],
        final_intervals=verified["final_cohort_intervals"],
    )
    registry_before = registry_path.read_bytes()
    current_before = current_pointer.read_bytes()
    registry_after, promotion_count = _stage_registry(
        registry_path,
        verified=verified,
        contract_sha256=contract["contract_sha256"],
        adjudication_path=adjudication_path.resolve(),
        receipt_path=receipt_path.resolve(),
        reason=reason,
    )
    current_after = (verified["candidate"]["path"] + "\n").encode("utf-8")
    legacy_attestation_ref: dict[str, str] | None = None
    if legacy_snapshot is not None:
        _revalidate_legacy_snapshot(legacy_snapshot)
        if legacy_snapshot.attestation is None:  # pragma: no cover - internal invariant.
            raise PromotionError("legacy promotion snapshot has no attestation bytes")
        legacy_attestation_ref = {
            "path": str(legacy_snapshot.attestation.path),
            "sha256": _sha256_bytes(legacy_snapshot.attestation.data),
        }
    transaction_id = uuid.uuid4().hex
    return {
        "schema_version": RECEIPT_SCHEMA,
        "transaction_id": transaction_id,
        "status": "dry_run",
        "created_at": time.time(),
        "registry": {
            "path": str(registry_path.resolve()),
            "before_sha256": _sha256_bytes(registry_before),
            "after_sha256": _sha256_bytes(registry_after),
        },
        "current_pointer": {
            "path": str(current_pointer.resolve()),
            "before_sha256": _sha256_bytes(current_before),
            "after_sha256": _sha256_bytes(current_after),
        },
        "contract": {
            "path": str(contract_lock.resolve()),
            "contract_sha256": contract["contract_sha256"],
            "n_full": 128,
            "n_full_wide": None,
            "legacy_contract_attestation": legacy_attestation_ref,
        },
        "adjudication": {
            "path": str(adjudication_path.resolve()),
            "adjudication_sha256": verified["adjudication_sha256"],
        },
        "training_receipt": verified["training_receipt"],
        "promotion_cohort_disjointness": cohort_disjointness,
        "candidate": verified["candidate"],
        "champion": verified["champion"],
        "evidence": verified["evidence"],
        "promotion_count": promotion_count,
        "nth_confirmation_required": verified["nth_confirmation_required"],
        "reason": reason,
        "fleet_ckpt_updated": False,
        "rollback": {},
        "_bytes": {
            "registry_before": registry_before,
            "registry_after": registry_after,
            "current_before": current_before,
            "current_after": current_after,
        },
        "_legacy_snapshot": legacy_snapshot,
    }


def _public_receipt(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in plan.items()
        if key not in {"_bytes", "_legacy_snapshot"}
    }


def execute_promotion(
    *,
    registry_path: Path,
    current_pointer: Path,
    contract_lock: Path,
    adjudication_path: Path,
    training_receipt: Path,
    cohort_exclusions: Path,
    receipt_path: Path,
    reason: str,
    legacy_contract_attestation: Path | None = None,
    lock_path: Path | None = None,
    go: bool = False,
    verify_lock_fn: Callable[..., dict[str, Any]] = a1_contract.verify_lock,
) -> dict[str, Any]:
    registry_path = _canonical_existing_file(
        registry_path, where="authoritative registry"
    )
    current_pointer = _canonical_existing_file(
        current_pointer, where="CURRENT_CHAMPION pointer"
    )
    contract_lock = _canonical_existing_file(contract_lock, where="A1 contract lock")
    adjudication_path = _canonical_existing_file(
        adjudication_path, where="promotion adjudication"
    )
    training_receipt = _canonical_existing_file(
        training_receipt, where="A1 one-dose training receipt"
    )
    cohort_exclusions = _canonical_existing_file(
        cohort_exclusions, where="promotion cohort-exclusions manifest"
    )
    if legacy_contract_attestation is not None:
        legacy_contract_attestation = _canonical_existing_file(
            legacy_contract_attestation, where="legacy contract attestation"
        )
    receipt_path = _canonical_new_file(receipt_path, where="promotion receipt")
    lock_path = _enforce_canonical_lock(registry_path, lock_path)
    with _exclusive_lock(lock_path):
        plan = prepare_promotion(
            registry_path=registry_path,
            current_pointer=current_pointer,
            contract_lock=contract_lock,
            adjudication_path=adjudication_path,
            training_receipt=training_receipt,
            cohort_exclusions=cohort_exclusions,
            receipt_path=receipt_path,
            reason=reason,
            legacy_contract_attestation=legacy_contract_attestation,
            verify_lock_fn=verify_lock_fn,
        )
        if not go:
            return _public_receipt(plan)

        payload = plan["_bytes"]
        mutation_snapshot = plan.get("_legacy_snapshot")
        if mutation_snapshot is not None:
            if not isinstance(mutation_snapshot, _LegacyPromotionSnapshot):
                raise PromotionError("legacy mutation snapshot is malformed")
            _revalidate_legacy_snapshot(mutation_snapshot)
        registry_backup, current_backup = _backup_paths(receipt_path)
        if registry_backup.exists() or current_backup.exists():
            raise PromotionError("rollback backup path already exists")
        _write_new_bytes(registry_backup, payload["registry_before"])
        try:
            _write_new_bytes(current_backup, payload["current_before"])
        except BaseException:
            registry_backup.unlink(missing_ok=True)
            raise
        receipt = _public_receipt(plan)
        receipt["status"] = "prepared"
        receipt["lock_path"] = str(lock_path)
        receipt["rollback"] = {
            "registry_backup": str(registry_backup.resolve()),
            "registry_backup_sha256": _sha256(registry_backup),
            "current_backup": str(current_backup.resolve()),
            "current_backup_sha256": _sha256(current_backup),
        }
        receipt = _seal_receipt(receipt)
        _write_new_bytes(
            receipt_path,
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        try:
            _atomic_write_bytes(registry_path, payload["registry_after"])
            _atomic_write_bytes(current_pointer, payload["current_after"])
            if _sha256(registry_path) != receipt["registry"]["after_sha256"]:
                raise PromotionError("registry post-commit hash mismatch")
            if _sha256(current_pointer) != receipt["current_pointer"]["after_sha256"]:
                raise PromotionError("current pointer post-commit hash mismatch")
            receipt["status"] = "committed"
            receipt["committed_at"] = time.time()
            receipt = _seal_receipt(receipt)
            _atomic_write_json(receipt_path, receipt)
            return receipt
        except BaseException as error:
            rollback_errors: list[str] = []
            for path, before in (
                (registry_path, payload["registry_before"]),
                (current_pointer, payload["current_before"]),
            ):
                try:
                    _atomic_write_bytes(path, before)
                except BaseException as rollback_error:
                    rollback_errors.append(f"{path}: {rollback_error}")
            receipt["status"] = "rollback_failed" if rollback_errors else "rolled_back"
            receipt["error"] = str(error)
            receipt["rollback_errors"] = rollback_errors
            receipt = _seal_receipt(receipt)
            _atomic_write_json(receipt_path, receipt)
            if rollback_errors:
                raise PromotionError(
                    f"promotion failed and rollback was incomplete: {rollback_errors}"
                ) from error
            raise PromotionError(
                "promotion failed; original registry/pointer restored"
            ) from error


def _load_recovery_receipt(
    receipt_path: Path,
) -> tuple[dict[str, Any], Path, Path, Path, Path, Path]:
    receipt_path = _canonical_existing_file(receipt_path, where="promotion receipt")
    receipt = _verify_receipt_digest(_load_json(receipt_path))
    receipt_schema = receipt.get("schema_version")
    if receipt_schema not in {
        RECEIPT_SCHEMA,
        PREVIOUS_RECEIPT_SCHEMA,
        LEGACY_RECEIPT_SCHEMA,
    }:
        raise PromotionError(
            f"receipt schema must be {RECEIPT_SCHEMA!r}, previous "
            f"{PREVIOUS_RECEIPT_SCHEMA!r}, or legacy {LEGACY_RECEIPT_SCHEMA!r}"
        )
    status = receipt.get("status")
    if status not in {"prepared", "committed", "rollback_failed"}:
        raise PromotionError(f"receipt status {status!r} is not recoverable")
    base_keys = {
        "schema_version",
        "transaction_id",
        "status",
        "created_at",
        "registry",
        "current_pointer",
        "contract",
        "adjudication",
        "candidate",
        "champion",
        "evidence",
        "promotion_count",
        "nth_confirmation_required",
        "reason",
        "fleet_ckpt_updated",
        "rollback",
        "lock_path",
        "receipt_sha256",
    }
    if receipt_schema in {RECEIPT_SCHEMA, PREVIOUS_RECEIPT_SCHEMA}:
        base_keys.add("training_receipt")
    if receipt_schema == RECEIPT_SCHEMA:
        base_keys.add("promotion_cohort_disjointness")
    status_keys = {
        "prepared": set(),
        "committed": {"committed_at"},
        "rollback_failed": {"error", "rollback_errors"},
    }[str(status)]
    _require_exact_keys(receipt, base_keys | status_keys, where="recovery receipt")
    registry_state = _require_exact_keys(
        receipt["registry"],
        {"path", "before_sha256", "after_sha256"},
        where="receipt.registry",
    )
    current_state = _require_exact_keys(
        receipt["current_pointer"],
        {"path", "before_sha256", "after_sha256"},
        where="receipt.current_pointer",
    )
    rollback = _require_exact_keys(
        receipt["rollback"],
        {
            "registry_backup",
            "registry_backup_sha256",
            "current_backup",
            "current_backup_sha256",
        },
        where="receipt.rollback",
    )
    for where, state in (
        ("receipt.registry", registry_state),
        ("receipt.current_pointer", current_state),
    ):
        _validate_sha256(state["before_sha256"], where=f"{where}.before_sha256")
        _validate_sha256(state["after_sha256"], where=f"{where}.after_sha256")
    _validate_sha256(
        rollback["registry_backup_sha256"],
        where="receipt.rollback.registry_backup_sha256",
    )
    _validate_sha256(
        rollback["current_backup_sha256"],
        where="receipt.rollback.current_backup_sha256",
    )
    registry_path = _canonical_existing_file(
        Path(str(registry_state["path"])), where="receipt registry"
    )
    current_pointer = _canonical_existing_file(
        Path(str(current_state["path"])), where="receipt current pointer"
    )
    if str(registry_path) != registry_state["path"]:
        raise PromotionError("receipt registry path is not canonical")
    if str(current_pointer) != current_state["path"]:
        raise PromotionError("receipt current-pointer path is not canonical")
    canonical_lock = _canonical_lock_path(registry_path)
    if receipt["lock_path"] != str(canonical_lock):
        raise PromotionError("receipt binds a non-canonical promotion lock")
    registry_backup = _canonical_existing_file(
        Path(str(rollback["registry_backup"])), where="registry rollback backup"
    )
    current_backup = _canonical_existing_file(
        Path(str(rollback["current_backup"])), where="current-pointer rollback backup"
    )
    expected_registry_backup, expected_current_backup = _backup_paths(receipt_path)
    if (
        registry_backup != expected_registry_backup
        or current_backup != expected_current_backup
    ):
        raise PromotionError("receipt rollback backup paths are not transaction-local")
    return (
        receipt,
        receipt_path,
        registry_path,
        current_pointer,
        registry_backup,
        current_backup,
    )


def recover_transaction(
    *, receipt_path: Path, go: bool = False, lock_path: Path | None = None
) -> dict[str, Any]:
    (
        receipt,
        receipt_path,
        registry_path,
        current_pointer,
        registry_backup,
        current_backup,
    ) = _load_recovery_receipt(receipt_path)
    lock_path = _enforce_canonical_lock(registry_path, lock_path)
    with _exclusive_lock(lock_path):
        if _sha256(registry_backup) != receipt["rollback"]["registry_backup_sha256"]:
            raise PromotionError("registry rollback backup hash mismatch")
        if _sha256(current_backup) != receipt["rollback"]["current_backup_sha256"]:
            raise PromotionError("current-pointer rollback backup hash mismatch")
        for label, path, state in (
            ("registry", registry_path, receipt["registry"]),
            ("current pointer", current_pointer, receipt["current_pointer"]),
        ):
            actual = _sha256(path)
            if actual not in {state["before_sha256"], state["after_sha256"]}:
                raise PromotionError(
                    f"{label} contains unknown bytes; refusing receipt recovery: {actual}"
                )
        result = {
            "schema_version": RECEIPT_SCHEMA,
            "transaction_id": receipt["transaction_id"],
            "status": "recovery_dry_run" if not go else "recovered",
            "registry": str(registry_path),
            "current_pointer": str(current_pointer),
            "receipt": str(receipt_path.resolve()),
        }
        if not go:
            return result
        original_registry = registry_path.read_bytes()
        original_current = current_pointer.read_bytes()
        try:
            _atomic_write_bytes(registry_path, registry_backup.read_bytes())
            _atomic_write_bytes(current_pointer, current_backup.read_bytes())
            if _sha256(registry_path) != receipt["registry"]["before_sha256"]:
                raise PromotionError("registry recovery verification failed")
            if _sha256(current_pointer) != receipt["current_pointer"]["before_sha256"]:
                raise PromotionError("current-pointer recovery verification failed")
        except BaseException as error:
            rollback_errors: list[str] = []
            for path, original in (
                (registry_path, original_registry),
                (current_pointer, original_current),
            ):
                try:
                    _atomic_write_bytes(path, original)
                except BaseException as rollback_error:
                    rollback_errors.append(f"{path}: {rollback_error}")
            if rollback_errors:
                raise PromotionError(
                    f"recovery failed and compensating rollback was incomplete: {rollback_errors}"
                ) from error
            raise PromotionError(
                "recovery failed; pre-recovery registry/pointer restored"
            ) from error
        receipt["status"] = "recovered"
        receipt["recovered_at"] = time.time()
        receipt = _seal_receipt(receipt)
        _atomic_write_json(receipt_path, receipt)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote = subparsers.add_parser(
        "promote", help="preflight or commit one A1 promotion"
    )
    promote.add_argument("--registry", required=True, type=Path)
    promote.add_argument("--current-pointer", required=True, type=Path)
    promote.add_argument("--contract-lock", required=True, type=Path)
    promote.add_argument("--adjudication", required=True, type=Path)
    promote.add_argument("--training-receipt", required=True, type=Path)
    promote.add_argument("--cohort-exclusions", required=True, type=Path)
    promote.add_argument("--legacy-contract-attestation", type=Path, default=None)
    promote.add_argument("--receipt", required=True, type=Path)
    promote.add_argument("--reason", required=True)
    promote.add_argument("--lock-file", type=Path, default=None)
    promote.add_argument("--go", action="store_true", help="commit; default is dry-run")

    recover = subparsers.add_parser(
        "recover", help="restore exact before bytes from a receipt"
    )
    recover.add_argument("--receipt", required=True, type=Path)
    recover.add_argument("--lock-file", type=Path, default=None)
    recover.add_argument(
        "--go", action="store_true", help="restore; default is dry-run"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "promote":
            result = execute_promotion(
                registry_path=args.registry,
                current_pointer=args.current_pointer,
                contract_lock=args.contract_lock,
                adjudication_path=args.adjudication,
                training_receipt=args.training_receipt,
                cohort_exclusions=args.cohort_exclusions,
                legacy_contract_attestation=args.legacy_contract_attestation,
                receipt_path=args.receipt,
                reason=args.reason,
                lock_path=args.lock_file,
                go=bool(args.go),
            )
        else:
            result = recover_transaction(
                receipt_path=args.receipt,
                lock_path=args.lock_file,
                go=bool(args.go),
            )
    except PromotionError as error:
        print(f"REFUSING: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
