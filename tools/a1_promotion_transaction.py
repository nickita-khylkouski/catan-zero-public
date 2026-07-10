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
import fcntl
import hashlib
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_pre_wave_contract as a1_contract  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402
from tools.sprt_gate import evaluate_pentanomial_sprt, pair_scores_from_h2h_games  # noqa: E402


ADJUDICATION_SCHEMA = "a1-promotion-adjudication-v1"
RECEIPT_SCHEMA = "a1-promotion-transaction-receipt-v1"
EVIDENCE_SCHEMA = "a1-promotion-evidence-v1"
HIGH_REGRET_SCHEMA = "a1-high-regret-comparison-v1"
BUCKET_VETO_SCHEMA = "a1-bucket-veto-v1"
MAX_CALIBRATION_RMSE_REGRESSION = 0.02
MAX_EXTERNAL_WIN_RATE_REGRESSION = 0.02
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
    except OSError as error:
        raise PromotionError(f"cannot resolve {where}: {error}") from error
    if lexical != resolved:
        raise PromotionError(f"{where} must not contain symlinks: {lexical}")
    if not resolved.is_file():
        raise PromotionError(f"{where} must be an existing regular file: {resolved}")
    return resolved


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
        raise PromotionError(f"canonical promotion lock must not be a symlink: {canonical}")
    if requested is None:
        return canonical
    lexical = _lexical_absolute(requested)
    try:
        resolved_parent = lexical.parent.resolve(strict=True)
    except OSError as error:
        raise PromotionError(f"cannot resolve promotion lock parent: {error}") from error
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
    descriptor = os.open(path, os.O_RDONLY)
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
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PromotionError(f"promotion lock is already held: {path}") from error
        yield
    finally:
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


def _verify_training_report(
    path: Path,
    *,
    contract: dict[str, Any],
    contract_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
) -> dict[str, Any]:
    report = _load_json(path)
    recipe = contract["science"]["learner_training_recipe"]
    recipe_sha = contract["science"]["learner_training_recipe_sha256"]
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
        raise PromotionError("candidate bytes drifted while validating its training report")
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
    if report.get("init_checkpoint_sha256") != producer_sha:
        raise PromotionError("candidate training report init checkpoint differs from producer")
    steps = report.get("steps_completed")
    epochs = report.get("epochs")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise PromotionError("candidate training report has no completed optimizer steps")
    if epochs != recipe.get("epochs"):
        raise PromotionError("candidate training report epoch count differs from sealed recipe")
    if report.get("max_steps") != recipe.get("max_steps"):
        raise PromotionError("candidate training report max_steps differs from sealed recipe")
    return report


def _verify_contract(
    path: Path,
    *,
    verify_lock_fn: Callable[..., dict[str, Any]] = a1_contract.verify_lock,
) -> dict[str, Any]:
    try:
        lock = verify_lock_fn(path, require_all_job_claims=True)
    except Exception as error:
        raise PromotionError(f"sealed A1 contract verification failed: {error}") from error
    search = lock.get("science", {}).get("search_operator", {})
    if search.get("n_full") != 128:
        raise PromotionError(
            f"current A1 promotion requires global n_full=128, got {search.get('n_full')!r}"
        )
    if search.get("n_full_wide") is not None or search.get("wide_roots_always_full") is not False:
        raise PromotionError(
            "current A1 promotion is global n128 only; adaptive/global alternate "
            "budgets are forbidden"
        )
    contract_sha = lock.get("contract_sha256")
    _validate_sha256(contract_sha, where="contract.contract_sha256")
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


def _verify_calibration_source(
    payload: dict[str, Any], *, checkpoint: Path, expected_readout: str, where: str
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
    _positive_int(provenance.get("optimizer_steps"), where=f"{where}.optimizer_steps")
    _positive_int(provenance.get("completed_epochs"), where=f"{where}.completed_epochs")
    selection = payload.get("row_selection")
    if not isinstance(selection, dict) or selection.get("held_out_filter_applied") is not True:
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
    rmse = _finite_number(
        global_metrics.get("value_rmse"), where=f"{where}.global.value_rmse", minimum=0.0
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


def _verify_internal_h2h_source(
    payload: dict[str, Any], *, candidate: Path, champion: Path, where: str
) -> None:
    if _absolute(payload.get("candidate_checkpoint"), base=candidate.parent) != candidate:
        raise PromotionError(f"{where} candidate checkpoint drift")
    if _absolute(payload.get("baseline_checkpoint"), base=champion.parent) != champion:
        raise PromotionError(f"{where} incumbent checkpoint drift")
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
    if _absolute(fields.get("candidate"), base=candidate.parent) != candidate or _absolute(
        fields.get("baseline"), base=champion.parent
    ) != champion:
        raise PromotionError(f"{where} typed config checkpoint identity drift")
    if fields.get("public_observation") is not True:
        raise PromotionError(f"{where} typed config is not public-observation")
    if fields.get("candidate_n_full") != 128 or fields.get("baseline_n_full") != 128:
        raise PromotionError(f"{where} typed config is not global n128")
    for key in (
        "n_full_wide",
        "candidate_n_full_wide",
        "baseline_n_full_wide",
        "n_full_wide_threshold",
        "candidate_n_full_wide_threshold",
        "baseline_n_full_wide_threshold",
    ):
        if fields.get(key) is not None:
            raise PromotionError(f"{where} typed config enables forbidden wide budget {key}")
    if payload.get("verdict") != "H1":
        raise PromotionError(f"{where} verdict is not H1")
    if payload.get("candidate_value_readout") != "scalar" or payload.get(
        "baseline_value_readout"
    ) != "scalar":
        raise PromotionError(f"{where} must use scalar readouts for both roles")
    if payload.get("public_observation") is not True:
        raise PromotionError(f"{where} must use public observations")
    budgets = payload.get("search_budgets_by_role")
    expected_budget = {
        "n_full": 128,
        "n_full_wide": None,
        "n_full_wide_threshold": None,
    }
    if not isinstance(budgets, dict) or budgets.get("candidate") != expected_budget or budgets.get(
        "baseline"
    ) != expected_budget:
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
    if not isinstance(games, list) or len(games) != int(payload.get("games_played", -1)):
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


def _verify_external_panel_source(
    payload: dict[str, Any], *, checkpoint: Path, checkpoint_md5: str, where: str
) -> tuple[float, dict[str, Any]]:
    if payload.get("stratum") != "neutral-harness":
        raise PromotionError(f"{where} is not a neutral-harness panel")
    if payload.get("harness") != "catanatron_native_engine":
        raise PromotionError(f"{where} uses an unexpected referee harness")
    if payload.get("mode") != "search" or payload.get("public_observation") is not True:
        raise PromotionError(f"{where} must use public-observation search")
    if payload.get("candidate_value_readout") != "scalar":
        raise PromotionError(f"{where} must use the sealed scalar readout")
    trained = payload.get("trained_value_readouts")
    if not isinstance(trained, list) or "scalar" not in trained:
        raise PromotionError(f"{where} does not prove scalar value training")
    if payload.get("n_full") != 128 or payload.get("n_full_wide") is not None:
        raise PromotionError(f"{where} does not use the sealed global n128 budget")
    if _absolute(payload.get("candidate_checkpoint"), base=checkpoint.parent) != checkpoint:
        raise PromotionError(f"{where} candidate checkpoint drift")
    if payload.get("candidate_checkpoint_md5") != checkpoint_md5:
        raise PromotionError(f"{where} candidate checkpoint MD5 drift")
    if payload.get("verdict") == "H0":
        raise PromotionError(f"{where} records a binding external regression")
    sprt = payload.get("pentanomial_sprt")
    if not isinstance(sprt, dict) or sprt.get("decision") not in {"H1", "continue"}:
        raise PromotionError(f"{where} has an invalid external-panel verdict")
    _positive_int(payload.get("complete_pairs"), where=f"{where}.complete_pairs")
    if payload.get("errors") != [] or payload.get("worker_errors") != []:
        raise PromotionError(f"{where} contains evaluation errors")
    if int(payload.get("games_engine_divergence", -1)) != 0:
        raise PromotionError(f"{where} contains engine divergence")
    rate = _finite_number(
        payload.get("candidate_win_rate"), where=f"{where}.candidate_win_rate", minimum=0.0
    )
    search_config = payload.get("search_config")
    if not isinstance(search_config, dict) or not search_config:
        raise PromotionError(f"{where} has no resolved search_config")
    games = payload.get("games")
    if not isinstance(games, list) or not games:
        raise PromotionError(f"{where} has no retained paired-game cohort")
    cohort_rows: list[tuple[int, int, str]] = []
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise PromotionError(f"{where}.games[{index}] is not an object")
        try:
            row = (
                int(game["pair_id"]),
                int(game["game_seed"]),
                str(game["orientation"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PromotionError(f"{where}.games[{index}] lacks cohort identity") from error
        cohort_rows.append(row)
    if len(set(cohort_rows)) != len(cohort_rows):
        raise PromotionError(f"{where} contains duplicate paired-game cohort rows")
    if len(games) != int(payload.get("games_played", -1)):
        raise PromotionError(f"{where} retained games differ from games_played")
    cohort = {
        "baseline_bot": payload["baseline_bot"],
        "map_kind": payload.get("map_kind"),
        "search_config": search_config,
        "gate_config": payload.get("gate_config"),
        "pairs_requested": payload.get("pairs_requested"),
        "games_requested": payload.get("games_requested"),
        "cohort_rows": sorted(cohort_rows),
    }
    return rate, cohort


def _verify_high_regret_source(
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
        "suite",
        "held_out",
        "candidate",
        "champion",
        "passed",
        "verdict",
        "complete_pairs",
        "errors",
    }
    value = _require_exact_keys(payload, expected_keys, where=where)
    if value["schema_version"] != HIGH_REGRET_SCHEMA or value["suite"] != "held_out_high_regret":
        raise PromotionError(f"{where} has an unexpected high-regret schema/suite")
    if value["held_out"] is not True or value["passed"] is not True:
        raise PromotionError(f"{where} is not a passing held-out high-regret result")
    if value["verdict"] not in {"H1", "noninferior"}:
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
    buckets = value["per_bucket"]
    if not isinstance(buckets, dict) or not buckets:
        raise PromotionError(f"{where}.per_bucket must be a non-empty object")
    for name, result in buckets.items():
        if not isinstance(name, str) or not isinstance(result, dict):
            raise PromotionError(f"{where}.per_bucket is malformed")
        if result.get("status") != "pass":
            raise PromotionError(f"{where} bucket {name!r} is not a pass")
        count = _positive_int(
            result.get("n"), where=f"{where}.per_bucket[{name}].n"
        )
        if count < 8:
            raise PromotionError(f"{where} bucket {name!r} has insufficient data")
        _finite_number(
            result.get("winrate"), where=f"{where}.per_bucket[{name}].winrate", minimum=0.0
        )


def _verify_promotion_evidence(
    path: Path,
    *,
    kind: str,
    contract_sha256: str,
    expected_readout: str = "scalar",
    candidate: dict[str, Any],
    champion: dict[str, Any],
) -> dict[str, Any]:
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
            raise PromotionError(f"{kind} evidence source role is invalid or duplicated")
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
            checkpoint=candidate_path,
            expected_readout=expected_readout,
            where="candidate calibration",
        )
        champion_rmse, champion_cohort = _verify_calibration_source(
            source_by_role["champion_calibration"][1],
            checkpoint=champion_path,
            expected_readout=expected_readout,
            where="champion calibration",
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
            raise PromotionError("candidate calibration exceeds the allowed RMSE regression")
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
        )
        if value["verdict"] != "H1" or result:
            raise PromotionError("internal H2H envelope verdict/result drift")
    elif kind == "external_panel":
        if set(source_by_role) != {"candidate_panel", "champion_panel"}:
            raise PromotionError("external panel source roles mismatch")
        candidate_panel = source_by_role["candidate_panel"][1]
        champion_panel = source_by_role["champion_panel"][1]
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
        )
        champion_rate, champion_cohort = _verify_external_panel_source(
            champion_panel,
            checkpoint=champion_path,
            checkpoint_md5=champion["md5"],
            where="champion external panel",
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
        if candidate_rate + max_regression < champion_rate:
            raise PromotionError("candidate external panel exceeds the allowed regression")
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


def _verify_adjudication(
    path: Path,
    *,
    contract: dict[str, Any],
    registry: ChampionRegistry,
    current_pointer: Path,
) -> dict[str, Any]:
    raw = _load_json(path)
    expected_keys = {
        "schema_version",
        "passed",
        "decision",
        "contract_sha256",
        "candidate",
        "champion",
        "checks",
        "nth_confirmation_required",
        "nth_confirmation_passed",
        "evidence",
        "adjudication_sha256",
    }
    value = _require_exact_keys(raw, expected_keys, where="adjudication")
    if value["schema_version"] != ADJUDICATION_SCHEMA:
        raise PromotionError(f"adjudication schema must be {ADJUDICATION_SCHEMA!r}")
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
        value["candidate"], {"path", "sha256", "version", "training_report"}, where="candidate"
    )
    candidate_path, candidate_ref = _validate_file_ref(
        {"path": candidate_raw["path"], "sha256": candidate_raw["sha256"]},
        base=base,
        where="candidate",
    )
    training_path, training_ref = _validate_file_ref(
        candidate_raw["training_report"], base=base, where="candidate.training_report"
    )
    _verify_training_report(
        training_path,
        contract=contract,
        contract_sha256=contract_sha,
        candidate_path=candidate_path,
        candidate_sha256=candidate_ref["sha256"],
    )
    champion_raw = _require_exact_keys(
        value["champion"], {"path", "sha256", "version"}, where="champion"
    )
    champion_path, champion_ref = _validate_file_ref(
        {"path": champion_raw["path"], "sha256": champion_raw["sha256"]},
        base=base,
        where="champion",
    )
    if candidate_path == champion_path or candidate_ref["sha256"] == champion_ref["sha256"]:
        raise PromotionError("candidate and incumbent champion must have distinct bytes")
    for label, raw_version in (
        ("candidate.version", candidate_raw["version"]),
        ("champion.version", champion_raw["version"]),
    ):
        if isinstance(raw_version, bool) or not isinstance(raw_version, int) or raw_version < 0:
            raise PromotionError(f"{label} must be a non-negative integer")
    if candidate_raw["version"] != champion_raw["version"] + 1:
        raise PromotionError("candidate version must be exactly incumbent version + 1")

    incumbent = registry.get_role("generator_champion")
    if incumbent is None:
        raise PromotionError("authoritative registry has no generator_champion")
    if str(Path(incumbent.checkpoint_path).expanduser().resolve()) != str(champion_path):
        raise PromotionError("adjudicated champion path differs from registry generator_champion")
    if incumbent.md5 != _md5(champion_path):
        raise PromotionError("registry generator_champion md5 differs from incumbent bytes")
    if incumbent.version != champion_raw["version"]:
        raise PromotionError("adjudicated champion version differs from registry")
    if _read_current_pointer(current_pointer) != str(champion_path):
        raise PromotionError("CURRENT_CHAMPION pointer differs from adjudicated incumbent")

    candidate_binding = {**candidate_ref, "md5": _md5(candidate_path)}
    champion_binding = {**champion_ref, "md5": _md5(champion_path)}

    checks = _require_exact_keys(value["checks"], REQUIRED_CHECKS, where="checks")
    failed_checks = sorted(name for name, passed in checks.items() if passed is not True)
    if failed_checks:
        raise PromotionError(f"adjudication has non-passing checks: {failed_checks}")
    next_count = registry.promotion_count("generator_champion") + 1
    nth_required = next_count % 3 == 0
    if value["nth_confirmation_required"] is not nth_required:
        raise PromotionError(
            "adjudication every-third confirmation requirement disagrees with registry count"
        )
    if nth_required and value["nth_confirmation_passed"] is not True:
        raise PromotionError("required every-third n64 confirmation did not pass")
    if not nth_required and value["nth_confirmation_passed"] not in {False, None}:
        raise PromotionError("non-required nth confirmation must be false or null")

    evidence = value["evidence"]
    if not isinstance(evidence, list):
        raise PromotionError("adjudication.evidence must be a list")
    evidence_by_kind: dict[str, dict[str, Any]] = {}
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
        _verify_promotion_evidence(
            evidence_path,
            kind=kind,
            contract_sha256=contract_sha,
            expected_readout=str(
                contract["science"]["learner_value_objective"]["value_readout"]
            ),
            candidate=candidate_binding,
            champion=champion_binding,
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
        },
        "champion": {
            **champion_ref,
            "version": champion_raw["version"],
            "md5": champion_binding["md5"],
        },
        "evidence": [evidence_by_kind[kind] for kind in sorted(evidence_by_kind)],
        "adjudication_sha256": declared_digest,
        "next_promotion_count": next_count,
        "nth_confirmation_required": nth_required,
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
            "a1_promotion_receipt": str(receipt_path),
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
    receipt_path: Path,
    reason: str,
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
    receipt_path = _canonical_new_file(receipt_path, where="promotion receipt")
    if registry_path.stat().st_size == 0:
        raise PromotionError("authoritative registry must be an existing non-empty file")
    contract = _verify_contract(contract_lock, verify_lock_fn=verify_lock_fn)
    registry = ChampionRegistry.load(registry_path)
    verified = _verify_adjudication(
        adjudication_path,
        contract=contract,
        registry=registry,
        current_pointer=current_pointer,
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
        },
        "adjudication": {
            "path": str(adjudication_path.resolve()),
            "adjudication_sha256": verified["adjudication_sha256"],
        },
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
    }


def _public_receipt(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "_bytes"}


def execute_promotion(
    *,
    registry_path: Path,
    current_pointer: Path,
    contract_lock: Path,
    adjudication_path: Path,
    receipt_path: Path,
    reason: str,
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
    receipt_path = _canonical_new_file(receipt_path, where="promotion receipt")
    lock_path = _enforce_canonical_lock(registry_path, lock_path)
    with _exclusive_lock(lock_path):
        plan = prepare_promotion(
            registry_path=registry_path,
            current_pointer=current_pointer,
            contract_lock=contract_lock,
            adjudication_path=adjudication_path,
            receipt_path=receipt_path,
            reason=reason,
            verify_lock_fn=verify_lock_fn,
        )
        if not go:
            return _public_receipt(plan)

        payload = plan["_bytes"]
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
            raise PromotionError("promotion failed; original registry/pointer restored") from error


def _load_recovery_receipt(
    receipt_path: Path,
) -> tuple[dict[str, Any], Path, Path, Path, Path, Path]:
    receipt_path = _canonical_existing_file(receipt_path, where="promotion receipt")
    receipt = _verify_receipt_digest(_load_json(receipt_path))
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise PromotionError(f"receipt schema must be {RECEIPT_SCHEMA!r}")
    status = receipt.get("status")
    if status not in {"prepared", "committed", "rollback_failed"}:
        raise PromotionError(
            f"receipt status {status!r} is not recoverable"
        )
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
    status_keys = {
        "prepared": set(),
        "committed": {"committed_at"},
        "rollback_failed": {"error", "rollback_errors"},
    }[str(status)]
    _require_exact_keys(receipt, base_keys | status_keys, where="recovery receipt")
    registry_state = _require_exact_keys(
        receipt["registry"], {"path", "before_sha256", "after_sha256"}, where="receipt.registry"
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
    if registry_backup != expected_registry_backup or current_backup != expected_current_backup:
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
    promote = subparsers.add_parser("promote", help="preflight or commit one A1 promotion")
    promote.add_argument("--registry", required=True, type=Path)
    promote.add_argument("--current-pointer", required=True, type=Path)
    promote.add_argument("--contract-lock", required=True, type=Path)
    promote.add_argument("--adjudication", required=True, type=Path)
    promote.add_argument("--receipt", required=True, type=Path)
    promote.add_argument("--reason", required=True)
    promote.add_argument("--lock-file", type=Path, default=None)
    promote.add_argument("--go", action="store_true", help="commit; default is dry-run")

    recover = subparsers.add_parser("recover", help="restore exact before bytes from a receipt")
    recover.add_argument("--receipt", required=True, type=Path)
    recover.add_argument("--lock-file", type=Path, default=None)
    recover.add_argument("--go", action="store_true", help="restore; default is dry-run")
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
