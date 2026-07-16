#!/usr/bin/env python3
"""Prometheus exporter for Catan-Zero generation progress.

DCGM and node-exporter remain authoritative for GPU and host telemetry. This
small exporter adds the application layer they cannot see: generator process
health, durable game/row/simulation/shard progress, failures/truncations,
typed config hash, seed range, output-disk capacity, and progress staleness.
It reads existing config/manifest/progress files and ``/proc`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import socket
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

METRIC_PREFIX = "catan_fleet_"
GPU_DIR_RE = re.compile(r"^gpu(?P<gpu>[0-9]+)(?:_pipeline(?P<pipeline>[01]))?$")
A1_GPU_DIR_RE = re.compile(
    r"^(?P<alias>[A-Za-z0-9][A-Za-z0-9-]*)_gpu(?P<gpu>[0-9]+)__"
    r"(?P<category>current_producer|recent_history|hard_negative)$"
)

# Exact sealed A1 production recipe. Monitoring deliberately duplicates these
# scalar expectations instead of treating a config hash as sufficient: an
# operator must be able to see which safety-critical knob drifted while the
# wave is live. Keep this table synchronized with the sealed A1 contract and
# generate_gumbel_selfplay_data guard.
PUBLIC_TARGET_INFORMATION_REGIME = "public_conservation_pimc_v1"
COHERENT_PUBLIC_TARGET_INFORMATION_REGIME = "public_belief_single_tree_v1"
UNKNOWN_TARGET_INFORMATION_REGIME = "unknown"
EXPECTED_LEGACY_A1_RECIPE: dict[str, Any] = {
    "public_observation": True,
    "information_set_search": True,
    "determinization_particles": 4,
    "determinization_min_simulations": 32,
    "n_full": 128,
    "n_fast": 16,
    "p_full": 0.25,
    "symmetry_averaged_eval": True,
    "symmetry_averaged_eval_threshold": 20,
    "c_scale": 0.03,
    "c_visit": 50.0,
    "max_depth": 80,
    "lazy_interior_chance": True,
    "belief_chance_spectra": False,
    "target_information_regime": PUBLIC_TARGET_INFORMATION_REGIME,
}
EXPECTED_COHERENT_PUBLIC_A1_RECIPE: dict[str, Any] = {
    "public_observation": True,
    "information_set_search": False,
    "determinization_particles": 1,
    "determinization_min_simulations": 32,
    "n_full": 128,
    "n_fast": 16,
    "p_full": 0.25,
    "symmetry_averaged_eval": True,
    "symmetry_averaged_eval_threshold": 20,
    "c_scale": 0.1,
    "c_visit": 50.0,
    "max_depth": 80,
    "lazy_interior_chance": True,
    "belief_chance_spectra": False,
    "sigma_eval": 0.79,
    "coherent_public_belief_search": True,
    "correct_rust_chance_spectra": True,
    "native_mcts_hot_loop": True,
    "forced_root_target_mode": "trajectory_only",
    "record_automatic_transitions": False,
    "meaningful_public_history": True,
    "event_history_limit": 32,
    "rust_featurize": True,
    "preserve_search_evidence": True,
    "target_information_regime": COHERENT_PUBLIC_TARGET_INFORMATION_REGIME,
}


def _is_post_promotion_attestation(
    attestation: Mapping[str, Any] | None,
) -> bool:
    """Recognize the lineage-bound job sidecar emitted after promotion.

    Historical v2 jobs and pre-promotion v3 jobs do not bind a promoted
    checkpoint's deployed search identity. Current post-promotion jobs do,
    for every source category. Requiring both the v3 schema and a canonical
    digest avoids classifying a partially written or hand-authored sidecar as
    current merely because it contains a similarly named field.
    """

    if attestation is None or attestation.get("schema_version") != (
        "a1-generation-job-attestation-v3"
    ):
        return False
    identity = attestation.get("producer_checkpoint_search_identity_sha256")
    return isinstance(identity, str) and re.fullmatch(
        r"sha256:[0-9a-f]{64}", identity
    ) is not None


def _expected_recipe(
    *,
    run: str,
    category: str,
    post_promotion: bool = False,
    coherent_public: bool = False,
) -> dict[str, Any]:
    """Return the exact safe recipe for historical or current A1 output.

    The immutable pre-promotion dual-arm wave nests outputs below
    ``.../<campaign>/<arm>/<job>`` and used c_scale=.10 for current-producer
    jobs but .03 for its history and hard-negative jobs. A post-promotion job
    is distinguished by its lineage-bound v3 attestation; all three categories
    search the retained producer seat with the deployed v5 c_scale=.10.
    Legacy runs keep the pre-wave n128/.03 expectation.
    """

    if coherent_public:
        return dict(EXPECTED_COHERENT_PUBLIC_A1_RECIPE)

    expected = dict(EXPECTED_LEGACY_A1_RECIPE)
    if run in {"n128", "n256"} and category in {
        "current_producer",
        "recent_history",
        "hard_negative",
    }:
        expected["n_full"] = int(run.removeprefix("n"))
        expected["c_scale"] = (
            0.1 if post_promotion or category == "current_producer" else 0.03
        )
    elif post_promotion and category in {
        "current_producer",
        "recent_history",
        "hard_negative",
    }:
        expected["c_scale"] = 0.1
    return expected


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _config_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()[:16]


def _flag_value(argv: Sequence[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def _flag_bool(argv: Sequence[str], positive: str, negative: str) -> bool | None:
    """Return the last explicit argparse boolean flag, if either is present."""
    resolved: bool | None = None
    for value in argv:
        if value == positive:
            resolved = True
        elif value == negative:
            resolved = False
    return resolved


def discover_generators(proc_root: Path = Path("/proc")) -> dict[str, set[int]]:
    """Map resolved ``--out-dir`` paths to live top-level generator PIDs."""
    found: dict[str, set[int]] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if not any(
            value.endswith("generate_gumbel_selfplay_data.py") for value in argv
        ):
            continue
        out_dir = _flag_value(argv, "--out-dir")
        if not out_dir:
            continue
        found.setdefault(str(Path(out_dir).expanduser().resolve()), set()).add(
            int(entry.name)
        )
    return found


def discover_generator_argv(
    proc_root: Path = Path("/proc"),
) -> dict[str, tuple[str, ...]]:
    """Map each live generator output to one deterministic, inspectable argv."""
    found: dict[str, tuple[int, tuple[str, ...]]] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = tuple(
            part.decode("utf-8", "replace") for part in raw.split(b"\0") if part
        )
        if not any(
            value.endswith("generate_gumbel_selfplay_data.py") for value in argv
        ):
            continue
        out_dir = _flag_value(argv, "--out-dir")
        if not out_dir:
            continue
        resolved = str(Path(out_dir).expanduser().resolve())
        pid = int(entry.name)
        prior = found.get(resolved)
        if prior is None or pid < prior[0]:
            found[resolved] = (pid, argv)
    return {path: record[1] for path, record in found.items()}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _boolean(value: Any, default: bool = False) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _recipe_value(
    key: str,
    *,
    manifest: Mapping[str, Any] | None,
    fields: Mapping[str, Any],
    argv: Sequence[str],
    flag: str,
) -> Any:
    """Read one effective recipe value from durable then live provenance."""
    if manifest is not None:
        if key in manifest and manifest[key] is not None:
            return manifest[key]
        cli_args = manifest.get("cli_args")
        if (
            isinstance(cli_args, Mapping)
            and key in cli_args
            and cli_args[key] is not None
        ):
            return cli_args[key]
    if key in fields and fields[key] is not None:
        return fields[key]
    return _flag_value(argv, flag)


def _recipe_bool_value(
    key: str,
    *,
    manifest: Mapping[str, Any] | None,
    fields: Mapping[str, Any],
    argv: Sequence[str],
    positive: str,
    negative: str,
) -> bool:
    if manifest is not None:
        if key in manifest and manifest[key] is not None:
            return _boolean(manifest[key])
        cli_args = manifest.get("cli_args")
        if (
            isinstance(cli_args, Mapping)
            and key in cli_args
            and cli_args[key] is not None
        ):
            return _boolean(cli_args[key])
    if key in fields and fields[key] is not None:
        return _boolean(fields[key])
    explicit = _flag_bool(argv, positive, negative)
    return bool(explicit) if explicit is not None else False


@dataclass(frozen=True)
class RunSnapshot:
    host: str
    alias: str
    gpu: str
    pipeline: str
    category: str
    run: str
    role: str
    config_hash: str
    public_observation: bool
    information_set_search: bool
    determinization_particles: int
    determinization_min_simulations: int
    n_full: int
    n_fast: int
    p_full: float
    symmetry_averaged_eval: bool
    symmetry_averaged_eval_threshold: int
    c_scale: float
    c_visit: float
    max_depth: int
    lazy_interior_chance: bool
    belief_chance_spectra: bool
    sigma_eval: float
    coherent_public_belief_search: bool
    correct_rust_chance_spectra: bool
    native_mcts_hot_loop: bool
    forced_root_target_mode: str
    record_automatic_transitions: bool
    meaningful_public_history: bool
    event_history_limit: int
    rust_featurize: bool
    preserve_search_evidence: bool
    target_information_regime: str
    target_information_regime_attested: bool
    recipe_safe: bool
    seed_start: int
    seed_end: int
    games_requested: int
    games_completed: int
    rows: int
    simulations: int
    shards: int
    failures: int
    truncations: int
    process_count: int
    complete: bool
    stale_seconds: float
    healthy: bool
    output_dir: Path


def _aggregate_progress(gpu_dir: Path) -> tuple[dict[str, float], float]:
    totals = {
        "games_requested": 0.0,
        "games_completed": 0.0,
        "rows": 0.0,
        "simulations": 0.0,
        "shards": 0.0,
        "failures": 0.0,
        "truncations": 0.0,
    }
    newest = 0.0
    for path in sorted(gpu_dir.glob("worker_*/progress.json")):
        payload = _load_json(path)
        if payload is None:
            continue
        newest = max(newest, path.stat().st_mtime)
        totals["games_requested"] += _number(payload.get("games_requested"))
        # games_completed_local is the durable, shard-confirmed counter.
        totals["games_completed"] += _number(payload.get("games_completed_local"))
        totals["rows"] += _number(payload.get("rows"))
        totals["simulations"] += _number(payload.get("simulations_used_total"))
        totals["shards"] += _number(payload.get("shard_count_confirmed"))
        totals["failures"] += _number(payload.get("games_failed"))
        totals["truncations"] += _number(payload.get("games_truncated"))
    return totals, newest


def snapshot_run(
    gpu_dir: Path,
    *,
    host: str,
    processes: Mapping[str, set[int]],
    generator_argv: Mapping[str, Sequence[str]] | None = None,
    now: float,
    stale_after_seconds: float,
) -> RunSnapshot | None:
    legacy_match = GPU_DIR_RE.fullmatch(gpu_dir.name)
    a1_match = A1_GPU_DIR_RE.fullmatch(gpu_dir.name)
    if legacy_match is None and a1_match is None:
        return None
    match = a1_match or legacy_match
    assert match is not None
    config_path = gpu_dir / "config.json"
    manifest_path = gpu_dir / "manifest.json"
    a1_contract_path = gpu_dir / "a1_contract.json"
    config = _load_json(config_path) or {}
    fields = config.get("fields") if isinstance(config.get("fields"), dict) else {}
    manifest = _load_json(manifest_path)
    a1_contract = _load_json(a1_contract_path)
    progress, progress_mtime = _aggregate_progress(gpu_dir)
    if (
        not config
        and manifest is None
        and a1_contract is None
        and progress_mtime == 0.0
    ):
        return None

    if manifest is not None:
        values = {
            "games_requested": _number(manifest.get("games_requested")),
            "games_completed": _number(manifest.get("games_completed")),
            "rows": _number(manifest.get("rows")),
            "simulations": _number(manifest.get("simulations_used_total")),
            "shards": float(len(manifest.get("shards", [])))
            if isinstance(manifest.get("shards"), list)
            else 0.0,
            "failures": _number(manifest.get("games_failed")),
            "truncations": _number(manifest.get("games_truncated")),
        }
    else:
        values = progress

    mtimes = [value for value in (progress_mtime,) if value > 0]
    for path in (config_path, manifest_path, a1_contract_path):
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
    newest = max(mtimes, default=0.0)
    age = max(0.0, now - newest) if newest else float("inf")
    resolved = str(gpu_dir.resolve())
    pids = processes.get(resolved, set())
    argv = tuple((generator_argv or {}).get(resolved, ()))
    attested_attempts = _number(
        a1_contract.get("attempts") if a1_contract is not None else None
    )
    games_requested = int(
        values["games_requested"] or _number(fields.get("games")) or attested_attempts
    )
    games_completed = int(values["games_completed"])
    errors = manifest.get("errors", []) if manifest is not None else []
    complete = bool(
        manifest is not None
        and games_requested > 0
        and games_completed == games_requested
        and values["failures"] == 0
        and not errors
        and manifest.get("fatal_execution_error") in (None, {})
    )
    healthy = bool(
        (len(pids) > 0 and age <= stale_after_seconds and values["failures"] == 0)
        or complete
    )
    config_hash = (
        str(manifest.get("config_hash"))
        if manifest is not None and manifest.get("config_hash")
        else _config_hash(config)
        if config
        else str(a1_contract.get("effective_search_config_sha256"))
        if a1_contract is not None
        and isinstance(a1_contract.get("effective_search_config_sha256"), str)
        else "pending"
    )
    seed_start = int(
        _number(
            manifest.get("base_seed")
            if manifest is not None
            else fields.get("base_seed")
        )
    )
    if not seed_start:
        seed_start = int(_number(fields.get("base_seed")))
    if not seed_start and a1_contract is not None:
        seed_start = int(_number(a1_contract.get("base_seed")))
    run = gpu_dir.parent.name
    recipe_source = {
        key: _recipe_value(
            key,
            manifest=manifest,
            fields=fields,
            argv=argv,
            flag="--" + key.replace("_", "-"),
        )
        for key in (
            "determinization_particles",
            "determinization_min_simulations",
            "n_full",
            "n_fast",
            "p_full",
            "symmetry_averaged_eval_threshold",
            "c_scale",
            "c_visit",
            "max_depth",
            "sigma_eval",
            "event_history_limit",
        )
    }
    public_observation = _recipe_bool_value(
        "public_observation",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--public-observation",
        negative="--no-public-observation",
    )
    information_set_search = _recipe_bool_value(
        "information_set_search",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--information-set-search",
        negative="--no-information-set-search",
    )
    symmetry_averaged_eval = _recipe_bool_value(
        "symmetry_averaged_eval",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--symmetry-averaged-eval",
        negative="--no-symmetry-averaged-eval",
    )
    lazy_interior_chance = _recipe_bool_value(
        "lazy_interior_chance",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--lazy-interior-chance",
        negative="--no-lazy-interior-chance",
    )
    belief_chance_spectra = _recipe_bool_value(
        "belief_chance_spectra",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--belief-chance-spectra",
        negative="--no-belief-chance-spectra",
    )
    coherent_public_belief_search = _recipe_bool_value(
        "coherent_public_belief_search",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--coherent-public-belief-search",
        negative="--no-coherent-public-belief-search",
    )
    correct_rust_chance_spectra = _recipe_bool_value(
        "correct_rust_chance_spectra",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--correct-rust-chance-spectra",
        negative="--no-correct-rust-chance-spectra",
    )
    native_mcts_hot_loop = _recipe_bool_value(
        "native_mcts_hot_loop",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--native-mcts-hot-loop",
        negative="--no-native-mcts-hot-loop",
    )
    record_automatic_transitions = _recipe_bool_value(
        "record_automatic_transitions",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--record-automatic-transitions",
        negative="--no-record-automatic-transitions",
    )
    meaningful_public_history = _recipe_bool_value(
        "meaningful_public_history",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--meaningful-public-history",
        negative="--no-meaningful-public-history",
    )
    rust_featurize = _recipe_bool_value(
        "rust_featurize",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--rust-featurize",
        negative="--no-rust-featurize",
    )
    preserve_search_evidence = _recipe_bool_value(
        "preserve_search_evidence",
        manifest=manifest,
        fields=fields,
        argv=argv,
        positive="--preserve-search-evidence",
        negative="--no-preserve-search-evidence",
    )
    determinization_particles = int(_number(recipe_source["determinization_particles"]))
    determinization_min_simulations = int(
        _number(recipe_source["determinization_min_simulations"])
    )
    n_full = int(_number(recipe_source["n_full"]))
    n_fast = int(_number(recipe_source["n_fast"]))
    p_full = _number(recipe_source["p_full"])
    symmetry_averaged_eval_threshold = int(
        _number(recipe_source["symmetry_averaged_eval_threshold"])
    )
    c_scale = _number(recipe_source["c_scale"])
    c_visit = _number(recipe_source["c_visit"])
    max_depth = int(_number(recipe_source["max_depth"]))
    sigma_eval = _number(recipe_source["sigma_eval"])
    event_history_limit = int(_number(recipe_source["event_history_limit"]))
    forced_root_target_mode = str(
        _recipe_value(
            "forced_root_target_mode",
            manifest=manifest,
            fields=fields,
            argv=argv,
            flag="--forced-root-target-mode",
        )
        or ""
    )
    raw_target_regime = (
        manifest.get("target_information_regime") if manifest is not None else None
    )
    target_information_regime_attested = isinstance(raw_target_regime, str)
    if not target_information_regime_attested:
        configured_regime = fields.get("target_information_regime")
        raw_target_regime = (
            configured_regime if isinstance(configured_regime, str) else None
        )
    # The generator deterministically stamps this public regime only after its
    # native determinization capability check succeeds. While a lane is live
    # and has not written its terminal manifest yet, expose that effective
    # regime from the exact launch setting but keep the separate attested bit 0.
    target_information_regime = (
        str(raw_target_regime)
        if raw_target_regime is not None
        else COHERENT_PUBLIC_TARGET_INFORMATION_REGIME
        if coherent_public_belief_search
        else PUBLIC_TARGET_INFORMATION_REGIME
        if information_set_search
        else UNKNOWN_TARGET_INFORMATION_REGIME
    )
    effective_recipe = {
        "public_observation": public_observation,
        "information_set_search": information_set_search,
        "determinization_particles": determinization_particles,
        "determinization_min_simulations": determinization_min_simulations,
        "n_full": n_full,
        "n_fast": n_fast,
        "p_full": p_full,
        "symmetry_averaged_eval": symmetry_averaged_eval,
        "symmetry_averaged_eval_threshold": symmetry_averaged_eval_threshold,
        "c_scale": c_scale,
        "c_visit": c_visit,
        "max_depth": max_depth,
        "lazy_interior_chance": lazy_interior_chance,
        "belief_chance_spectra": belief_chance_spectra,
        "target_information_regime": target_information_regime,
    }
    coherent_recipe = bool(coherent_public_belief_search) or (
        target_information_regime == COHERENT_PUBLIC_TARGET_INFORMATION_REGIME
    )
    if coherent_recipe:
        effective_recipe.update(
            {
                "sigma_eval": sigma_eval,
                "coherent_public_belief_search": coherent_public_belief_search,
                "correct_rust_chance_spectra": correct_rust_chance_spectra,
                "native_mcts_hot_loop": native_mcts_hot_loop,
                "forced_root_target_mode": forced_root_target_mode,
                "record_automatic_transitions": record_automatic_transitions,
                "meaningful_public_history": meaningful_public_history,
                "event_history_limit": event_history_limit,
                "rust_featurize": rust_featurize,
                "preserve_search_evidence": preserve_search_evidence,
            }
        )
    recipe_safe = effective_recipe == _expected_recipe(
        run=run,
        category=(a1_match.group("category") if a1_match is not None else "legacy"),
        post_promotion=_is_post_promotion_attestation(a1_contract),
        coherent_public=coherent_recipe,
    )
    seed_end = seed_start + games_requested
    if a1_contract is not None:
        attested_seed_end = int(_number(a1_contract.get("seed_end")))
        if attested_seed_end > seed_start:
            seed_end = attested_seed_end
    role = "teacher" if n_full >= 128 else "volume"
    return RunSnapshot(
        host=host,
        alias=a1_match.group("alias") if a1_match is not None else host,
        gpu=match.group("gpu"),
        pipeline=(legacy_match.group("pipeline") if legacy_match is not None else None)
        or "0",
        category=a1_match.group("category") if a1_match is not None else "legacy",
        run=run,
        role=role,
        config_hash=config_hash,
        public_observation=public_observation,
        information_set_search=information_set_search,
        determinization_particles=determinization_particles,
        determinization_min_simulations=determinization_min_simulations,
        n_full=n_full,
        n_fast=n_fast,
        p_full=p_full,
        symmetry_averaged_eval=symmetry_averaged_eval,
        symmetry_averaged_eval_threshold=symmetry_averaged_eval_threshold,
        c_scale=c_scale,
        c_visit=c_visit,
        max_depth=max_depth,
        lazy_interior_chance=lazy_interior_chance,
        belief_chance_spectra=belief_chance_spectra,
        sigma_eval=sigma_eval,
        coherent_public_belief_search=coherent_public_belief_search,
        correct_rust_chance_spectra=correct_rust_chance_spectra,
        native_mcts_hot_loop=native_mcts_hot_loop,
        forced_root_target_mode=forced_root_target_mode,
        record_automatic_transitions=record_automatic_transitions,
        meaningful_public_history=meaningful_public_history,
        event_history_limit=event_history_limit,
        rust_featurize=rust_featurize,
        preserve_search_evidence=preserve_search_evidence,
        target_information_regime=target_information_regime,
        target_information_regime_attested=target_information_regime_attested,
        recipe_safe=recipe_safe,
        seed_start=seed_start,
        seed_end=seed_end,
        games_requested=games_requested,
        games_completed=games_completed,
        rows=int(values["rows"]),
        simulations=int(values["simulations"]),
        shards=int(values["shards"]),
        failures=int(values["failures"]),
        truncations=int(values["truncations"]),
        process_count=len(pids),
        complete=complete,
        stale_seconds=age,
        healthy=healthy,
        output_dir=gpu_dir,
    )


def collect_snapshots(
    roots: Iterable[Path],
    *,
    host: str,
    processes: Mapping[str, set[int]],
    generator_argv: Mapping[str, Sequence[str]] | None = None,
    now: float,
    stale_after_seconds: float,
    max_run_age_seconds: float,
) -> list[RunSnapshot]:
    snapshots: list[RunSnapshot] = []
    for root in roots:
        expanded = root.expanduser()
        candidates = {
            *expanded.glob("*/gpu*"),
            *expanded.glob("*/*_gpu*__*"),
            # Sealed dual-arm production layout:
            #   <root>/<campaign>/<arm>/<arm>_gpuNN__<category>
            *expanded.glob("*/*/*_gpu*__*"),
        }
        for gpu_dir in sorted(candidates):
            snapshot = snapshot_run(
                gpu_dir,
                host=host,
                processes=processes,
                generator_argv=generator_argv,
                now=now,
                stale_after_seconds=stale_after_seconds,
            )
            if snapshot is None:
                continue
            if (
                snapshot.process_count == 0
                and snapshot.stale_seconds > max_run_age_seconds
            ):
                continue
            snapshots.append(snapshot)
    # Avoid unbounded label cardinality: expose one current run per physical
    # GPU/pipeline slot. Sealed A1 runs its three categories sequentially on a
    # lane, so the active category wins the same slot arbitration as legacy.
    by_slot: dict[tuple[str, str], RunSnapshot] = {}
    for snapshot in snapshots:
        slot = (snapshot.gpu, snapshot.pipeline)
        prior = by_slot.get(slot)
        score = (snapshot.process_count > 0, -snapshot.stale_seconds)
        prior_score = (
            (prior.process_count > 0, -prior.stale_seconds)
            if prior is not None
            else None
        )
        if prior_score is None or score > prior_score:
            by_slot[slot] = snapshot
    return [
        by_slot[slot]
        for slot in sorted(by_slot, key=lambda item: tuple(map(int, item)))
    ]


def _escape_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _sample(name: str, value: float | int, labels: Mapping[str, Any]) -> str:
    rendered = ",".join(
        f'{key}="{_escape_label(labels[key])}"' for key in sorted(labels)
    )
    return f"{METRIC_PREFIX}{name}{{{rendered}}} {value}"


def render_metrics(
    snapshots: Sequence[RunSnapshot],
    *,
    host: str,
    roots: Sequence[Path],
    now: float,
    scrape_success: bool = True,
) -> str:
    lines = [
        "# HELP catan_fleet_exporter_up Exporter collection succeeded.",
        "# TYPE catan_fleet_exporter_up gauge",
        _sample("exporter_up", int(scrape_success), {"host": host}),
        "# HELP catan_fleet_exporter_scrape_timestamp_seconds Unix scrape time.",
        "# TYPE catan_fleet_exporter_scrape_timestamp_seconds gauge",
        _sample("exporter_scrape_timestamp_seconds", now, {"host": host}),
    ]
    metric_fields = {
        "generator_processes": "process_count",
        "generator_healthy": "healthy",
        "generator_complete": "complete",
        "generator_progress_age_seconds": "stale_seconds",
        "generator_games_requested": "games_requested",
        "generator_games_completed": "games_completed",
        "generator_rows": "rows",
        "generator_simulations": "simulations",
        "generator_shards": "shards",
        "generator_failures": "failures",
        "generator_truncations": "truncations",
        "generator_seed_start": "seed_start",
        "generator_seed_end": "seed_end",
        "generator_public_observation": "public_observation",
        "generator_information_set_search": "information_set_search",
        "generator_determinization_particles": "determinization_particles",
        "generator_determinization_min_simulations": "determinization_min_simulations",
        "generator_n_full": "n_full",
        "generator_n_fast": "n_fast",
        "generator_p_full": "p_full",
        "generator_symmetry_averaged_eval": "symmetry_averaged_eval",
        "generator_symmetry_averaged_eval_threshold": "symmetry_averaged_eval_threshold",
        "generator_c_scale": "c_scale",
        "generator_c_visit": "c_visit",
        "generator_max_depth": "max_depth",
        "generator_lazy_interior_chance": "lazy_interior_chance",
        "generator_belief_chance_spectra": "belief_chance_spectra",
        "generator_sigma_eval": "sigma_eval",
        "generator_coherent_public_belief_search": "coherent_public_belief_search",
        "generator_correct_rust_chance_spectra": "correct_rust_chance_spectra",
        "generator_native_mcts_hot_loop": "native_mcts_hot_loop",
        "generator_record_automatic_transitions": "record_automatic_transitions",
        "generator_meaningful_public_history": "meaningful_public_history",
        "generator_event_history_limit": "event_history_limit",
        "generator_rust_featurize": "rust_featurize",
        "generator_preserve_search_evidence": "preserve_search_evidence",
        "generator_target_information_regime_attested": "target_information_regime_attested",
        "generator_recipe_safe": "recipe_safe",
    }
    for snapshot in snapshots:
        labels = {
            "host": snapshot.host,
            "alias": snapshot.alias,
            "gpu": snapshot.gpu,
            "pipeline": snapshot.pipeline,
            "category": snapshot.category,
            "run": snapshot.run,
            "role": snapshot.role,
            "config_hash": snapshot.config_hash,
        }
        info_labels = {
            **labels,
            "n_full": snapshot.n_full,
            "n_fast": snapshot.n_fast,
            "p_full": snapshot.p_full,
            "public_observation": str(snapshot.public_observation).lower(),
            "information_set_search": str(snapshot.information_set_search).lower(),
            "determinization_particles": snapshot.determinization_particles,
            "determinization_min_simulations": snapshot.determinization_min_simulations,
            "symmetry_averaged_eval": str(snapshot.symmetry_averaged_eval).lower(),
            "symmetry_averaged_eval_threshold": snapshot.symmetry_averaged_eval_threshold,
            "c_scale": snapshot.c_scale,
            "c_visit": snapshot.c_visit,
            "max_depth": snapshot.max_depth,
            "lazy_interior_chance": str(snapshot.lazy_interior_chance).lower(),
            "belief_chance_spectra": str(snapshot.belief_chance_spectra).lower(),
            "sigma_eval": snapshot.sigma_eval,
            "coherent_public_belief_search": str(
                snapshot.coherent_public_belief_search
            ).lower(),
            "correct_rust_chance_spectra": str(
                snapshot.correct_rust_chance_spectra
            ).lower(),
            "native_mcts_hot_loop": str(snapshot.native_mcts_hot_loop).lower(),
            "forced_root_target_mode": snapshot.forced_root_target_mode,
            "record_automatic_transitions": str(
                snapshot.record_automatic_transitions
            ).lower(),
            "meaningful_public_history": str(
                snapshot.meaningful_public_history
            ).lower(),
            "event_history_limit": snapshot.event_history_limit,
            "rust_featurize": str(snapshot.rust_featurize).lower(),
            "preserve_search_evidence": str(
                snapshot.preserve_search_evidence
            ).lower(),
            "target_information_regime": snapshot.target_information_regime,
            "seed_range": f"[{snapshot.seed_start},{snapshot.seed_end})",
        }
        lines.append(_sample("generator_info", 1, info_labels))
        for metric, field in metric_fields.items():
            value = getattr(snapshot, field)
            if isinstance(value, bool):
                value = int(value)
            lines.append(_sample(metric, value, labels))
        lines.append(
            _sample("generator_active", int(snapshot.process_count > 0), labels)
        )
    lines.append(
        _sample(
            "generator_lanes_active_total",
            sum(snapshot.process_count > 0 for snapshot in snapshots),
            {"host": host},
        )
    )
    lines.append(
        _sample(
            "generator_lanes_recipe_safe_total",
            sum(
                snapshot.process_count > 0 and snapshot.recipe_safe
                for snapshot in snapshots
            ),
            {"host": host},
        )
    )
    for root in roots:
        resolved = root.expanduser().resolve()
        try:
            usage = shutil.disk_usage(
                resolved if resolved.exists() else resolved.parent
            )
        except OSError:
            continue
        labels = {"host": host, "path": str(resolved)}
        lines.append(_sample("output_disk_free_bytes", usage.free, labels))
        lines.append(_sample("output_disk_total_bytes", usage.total, labels))
    return "\n".join(lines) + "\n"


def collect_metrics(args: argparse.Namespace) -> str:
    now = time.time()
    processes = discover_generators(Path(args.proc_root))
    generator_argv = discover_generator_argv(Path(args.proc_root))
    roots = [Path(value) for value in args.run_root]
    snapshots = collect_snapshots(
        roots,
        host=args.host_label,
        processes=processes,
        generator_argv=generator_argv,
        now=now,
        stale_after_seconds=float(args.stale_after_seconds),
        max_run_age_seconds=float(args.max_run_age_seconds),
    )
    return render_metrics(snapshots, host=args.host_label, roots=roots, now=now)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9500)
    parser.add_argument("--host-label", default=socket.gethostname())
    parser.add_argument("--run-root", action="append", default=None)
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--stale-after-seconds", type=float, default=300.0)
    parser.add_argument("--max-run-age-seconds", type=float, default=86400.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_root is None:
        args.run_root = [str(Path.home() / "gen_out")]
    if args.port <= 0 or args.port > 65535:
        parser.error("--port must be in 1..65535")
    if args.stale_after_seconds <= 0 or args.max_run_age_seconds <= 0:
        parser.error("staleness windows must be positive")
    if args.once:
        print(collect_metrics(args), end="")
        return 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path not in {"/", "/metrics"}:
                self.send_error(404)
                return
            try:
                body = collect_metrics(args).encode("utf-8")
                status = 200
            except Exception as error:  # exporter must remain scrapeable on bad input
                body = render_metrics(
                    [],
                    host=args.host_label,
                    roots=[],
                    now=time.time(),
                    scrape_success=False,
                ).encode("utf-8")
                body += f"# collection error: {type(error).__name__}\n".encode()
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((args.listen, args.port), Handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
