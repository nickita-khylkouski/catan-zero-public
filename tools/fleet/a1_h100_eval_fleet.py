#!/usr/bin/env python3
"""Deterministic B200-origin controller for the heterogeneous A1 H100 eval fleet.

The controller deliberately uses plain SSH as the production transport.  It
does not require a scheduler daemon on the eight H100 hosts, and every mutable
operation is gated by ``--go``.  A Ray cluster specification can be rendered
for a later daemon-managed backend without installing or starting Ray.

The unit of capacity is a physical GPU, never a host.  Six four-GPU hosts and
two eight-GPU hosts therefore yield forty equal evaluator lanes.  Internal H2H
uses one shard per lane.  External candidate/incumbent panels pair adjacent
lanes and assign both sides the exact same seed interval.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_evaluation_pool as evaluation_pool  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402
from tools.prelaunch_guard import VAL_ONLY_SEED_RANGE  # noqa: E402


MANIFEST_SCHEMA = "a1-h100-eval-fleet-manifest-v1"
PLAN_SCHEMA = "a1-h100-eval-fleet-plan-v1"
RAY_SCHEMA = "a1-h100-eval-ray-cluster-v1"
EXPECTED_SHAPES = {
    "c1": 4,
    "c2": 4,
    "c3": 4,
    "c4": 4,
    "c5": 4,
    "c6": 4,
    "h100-8a": 8,
    "h100-8b": 8,
}
CANARY_ALIASES = {"c1", "h100-8a"}
DEFAULT_WORKERS_PER_GPU = 16
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_ADDRESS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.:-]*$")

SCIENCE_CONFIG: dict[str, Any] = {
    # Deliberate two-stratum evaluation: broad randomized BASE maps for direct
    # candidate-vs-incumbent strength; fixed TOURNAMENT for the cross-engine
    # Python referee, whose map parity is certified only there.
    "internal_map_kind": "BASE",
    "external_map_kind": "TOURNAMENT",
    "n_full": 128,
    "c_scale": 0.03,
    "c_visit": 50.0,
    "sigma_eval": 0.98,
    "rescale_noise_floor_c": 0.0,
    "lazy_interior_chance": True,
    "correct_rust_chance_spectra": True,
    "public_observation": True,
    "information_set_search": True,
    "belief_chance_spectra": False,
    "determinization_particles": 4,
    "determinization_min_simulations": 32,
    "symmetry_averaged_eval": True,
    "symmetry_averaged_eval_threshold": 20,
    "evaluator_rust_featurize": True,
    "native_mcts_hot_loop": True,
    "value_readout": "scalar",
    "value_squash": "tanh",
    "max_depth": 80,
    "max_decisions": 600,
    "max_root_candidates": 16,
    "max_root_candidates_wide": 54,
    "wide_candidates_threshold": 24,
    "gate_config": "flywheel",
    "external_vps_to_win": 10,
    "external_max_player_trade_offers_per_turn": 0,
}


class FleetError(RuntimeError):
    """The requested fleet operation could not be proved safe."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _run_id_from_plan_fields(plan: dict[str, Any]) -> str:
    run_key = {
        "manifest_hash": plan["manifest_hash"],
        "repo_commit": plan["repo_commit"],
        "tool_hashes": plan["tool_hashes"],
        "candidate_sha256": plan["candidate"]["sha256"],
        "champion_sha256": plan["champion"]["sha256"],
        "evaluation_binding": plan["evaluation_binding"],
        "engine_identity": plan["engine_identity"],
        "science_hash": plan["science_config_hash"],
        "internal_pairs": plan["pair_claims"]["internal"]["pairs"],
        "external_pairs": plan["pair_claims"]["external_matched"]["pairs"],
        "internal_base_seed": plan["pair_claims"]["internal"]["base_seed"],
        "external_base_seed": plan["pair_claims"]["external_matched"][
            "base_seed"
        ],
        "iteration_id": plan["iteration_id"],
        "seed_cohort_id": plan.get("seed_cohort_id"),
        "scope": plan["scope"],
        "workers_per_gpu": plan["workers_per_gpu"],
    }
    # Legacy v1 plans predate role-specific search calibration.  Do not add a
    # synthetic field while replaying their immutable run IDs.
    if "role_search_config" in plan:
        run_key["role_search_config"] = plan["role_search_config"]
    # Optional host subsets were added after the original immutable fleet
    # plans.  Bind them into new run IDs without changing legacy replay.
    if "host_aliases" in plan:
        run_key["host_aliases"] = plan["host_aliases"]
    return "a1-eval-" + hashlib.sha256(_canonical(run_key)).hexdigest()[:16]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_ref(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _evaluation_binding(
    *,
    candidate_parent: Path,
    baseline: Path,
    registry: ChampionRegistry,
    comparison_mode: str,
    historical_comparison_reason: str | None,
    champion_c_scale: float,
) -> dict[str, Any]:
    """Bind the causal parent, comparison baseline, and registry incumbent.

    Promotion-grade evaluation is deliberately strict: the candidate's parent,
    the internal baseline, and the authoritative generator champion must be the
    same checkpoint.  A different historical baseline is diagnostic-only and
    requires an explicit typed mode plus a nonempty reason.
    """
    if comparison_mode not in {"promotion_parent", "historical_comparison"}:
        raise FleetError(f"unknown evaluation comparison mode {comparison_mode!r}")
    parent_ref = _checkpoint_ref(candidate_parent)
    baseline_ref = _checkpoint_ref(baseline)
    pointer = registry.get_role("generator_champion")
    if pointer is None:
        raise FleetError("authoritative registry has no generator_champion")
    incumbent_path = Path(pointer.checkpoint_path).expanduser().resolve(strict=True)
    if pointer.md5 != _md5(incumbent_path):
        raise FleetError("registry generator_champion MD5 differs from its bytes")
    search_config = pointer.provenance.get("a1_candidate_search_config")
    identity_sha = pointer.provenance.get("a1_candidate_agent_identity_sha256")
    if not isinstance(search_config, dict) or not search_config:
        raise FleetError("registry incumbent has no bound search operator identity")
    if not isinstance(identity_sha, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", identity_sha
    ):
        raise FleetError("registry incumbent has no valid agent identity digest")
    incumbent_ref = _checkpoint_ref(incumbent_path)
    expected_identity = _digest(
        {
            "schema_version": "a1-deployed-agent-search-config-v1",
            "checkpoint": incumbent_ref,
            "search_config": search_config,
        }
    )
    if identity_sha != expected_identity:
        raise FleetError(
            "registry incumbent identity does not bind its checkpoint and search config"
        )
    if comparison_mode == "promotion_parent":
        if historical_comparison_reason is not None:
            raise FleetError("promotion-parent evaluation cannot carry a historical reason")
        if parent_ref != baseline_ref:
            raise FleetError(
                "promotion baseline differs from candidate parent/init checkpoint"
            )
        if baseline_ref != incumbent_ref:
            raise FleetError(
                "promotion baseline differs from authoritative registry incumbent"
            )
        try:
            incumbent_c_scale = float(search_config["c_scale"])
        except (KeyError, TypeError, ValueError) as error:
            raise FleetError("registry incumbent has no valid c_scale") from error
        if incumbent_c_scale != float(champion_c_scale):
            raise FleetError(
                "explicit champion c_scale differs from registry incumbent identity"
            )
        promotion_eligible = True
    else:
        if not isinstance(historical_comparison_reason, str) or not historical_comparison_reason.strip():
            raise FleetError(
                "historical comparison requires an explicit nonempty reason"
            )
        promotion_eligible = False
    return {
        "schema_version": "a1-evaluation-baseline-binding-v1",
        "comparison_mode": comparison_mode,
        "promotion_eligible": promotion_eligible,
        "historical_comparison_reason": historical_comparison_reason,
        "candidate_parent": parent_ref,
        "baseline": baseline_ref,
        "registry": {
            "path": str(registry.path.expanduser().resolve(strict=True)),
            "sha256": _sha256(registry.path.expanduser().resolve(strict=True)),
        },
        "authoritative_incumbent": {
            **incumbent_ref,
            "version": pointer.version,
            "agent_identity_sha256": identity_sha,
            "search_config": search_config,
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FleetError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise FleetError(f"{path} must contain one JSON object")
    return value


def _absolute(value: Any, *, field: str) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise FleetError(f"{field} must be an absolute path")
    if any(character in str(path) for character in ("\n", "\r", "\0")):
        raise FleetError(f"{field} contains an invalid character")
    return str(path)


def load_manifest(
    path: Path, *, expected_shapes: dict[str, int] | None = None
) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise FleetError(f"unsupported fleet manifest schema in {path}")
    for field in (
        "ssh_user",
        "ssh_key",
        "remote_repo",
        "remote_python",
        "remote_root",
        "validation_ledger",
    ):
        if field not in manifest:
            raise FleetError(f"fleet manifest is missing {field}")
    if not SAFE_NAME.fullmatch(str(manifest["ssh_user"])):
        raise FleetError("ssh_user is not a safe POSIX account name")
    manifest["ssh_key"] = str(Path(str(manifest["ssh_key"])).expanduser())
    for field in ("remote_repo", "remote_python", "remote_root", "validation_ledger"):
        manifest[field] = _absolute(manifest[field], field=field)
    checking = str(manifest.get("strict_host_key_checking", "accept-new"))
    if checking not in {"yes", "accept-new"}:
        raise FleetError("strict_host_key_checking must be 'yes' or 'accept-new'")
    manifest["strict_host_key_checking"] = checking
    raw_hosts = manifest.get("hosts")
    if not isinstance(raw_hosts, list):
        raise FleetError("fleet manifest hosts must be a list")
    hosts: list[dict[str, Any]] = []
    for raw in raw_hosts:
        if not isinstance(raw, dict):
            raise FleetError("each fleet host must be an object")
        alias = str(raw.get("alias", ""))
        address = str(raw.get("address", ""))
        if not SAFE_NAME.fullmatch(alias):
            raise FleetError(f"invalid fleet alias {alias!r}")
        if not SAFE_ADDRESS.fullmatch(address):
            raise FleetError(f"invalid fleet address for {alias}")
        try:
            gpu_count = int(raw["gpu_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise FleetError(f"invalid gpu_count for {alias}") from error
        hosts.append({"alias": alias, "address": address, "gpu_count": gpu_count})
    actual = {host["alias"]: host["gpu_count"] for host in hosts}
    if len(actual) != len(hosts):
        raise FleetError("fleet manifest contains duplicate aliases")
    approved_shapes = EXPECTED_SHAPES if expected_shapes is None else expected_shapes
    if actual != approved_shapes:
        raise FleetError(
            "A1 eval manifest differs from its exact approved fleet shape: "
            f"expected {approved_shapes}, got {actual}"
        )
    manifest["hosts"] = sorted(hosts, key=lambda item: item["alias"])
    manifest["manifest_hash"] = _digest(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    return manifest


def gpu_slots(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    slots = [
        {
            "alias": host["alias"],
            "address": host["address"],
            "gpu": gpu,
            "slot_id": f"{host['alias']}-g{gpu}",
        }
        for host in manifest["hosts"]
        for gpu in range(int(host["gpu_count"]))
    ]
    expected_slots = sum(int(host["gpu_count"]) for host in manifest["hosts"])
    if len(slots) != expected_slots:
        raise FleetError(
            f"A1 eval manifest resolved {len(slots)} GPUs, expected {expected_slots}"
        )
    return slots


def _split_ranges(total: int, lanes: int, base_seed: int) -> list[tuple[int, int]]:
    if total <= 0:
        raise FleetError("pair count must be positive")
    if lanes <= 0:
        raise FleetError("lane count must be positive")
    quotient, remainder = divmod(total, lanes)
    cursor = int(base_seed)
    result = []
    for lane in range(lanes):
        count = quotient + (1 if lane < remainder else 0)
        result.append((cursor, count))
        cursor += count
    if cursor != int(base_seed) + total:
        raise AssertionError("pair-range split did not conserve the interval")
    return result


def _science_args(*, c_scale: float | None = 0.03) -> list[str]:
    args = [
        "--n-full",
        "128",
        "--c-visit",
        "50.0",
        "--sigma-eval",
        "0.98",
        "--rescale-noise-floor-c",
        "0.0",
        "--lazy-interior-chance",
        "--correct-rust-chance-spectra",
        "--public-observation",
        "--information-set-search",
        "--no-belief-chance-spectra",
        "--determinization-particles",
        "4",
        "--determinization-min-simulations",
        "32",
        "--symmetry-averaged-eval",
        "--symmetry-averaged-eval-threshold",
        "20",
        "--evaluator-rust-featurize",
        "--native-mcts-hot-loop",
        "--value-readout",
        "scalar",
        "--value-squash",
        "tanh",
        "--max-depth",
        "80",
        "--max-decisions",
        "600",
        "--max-root-candidates",
        "16",
        "--max-root-candidates-wide",
        "54",
        "--wide-candidates-threshold",
        "24",
        "--gate-config",
        "flywheel",
    ]
    if c_scale is not None:
        args[2:2] = ["--c-scale", str(float(c_scale))]
    return args


def _role_search_config(
    *, candidate_c_scale: float, champion_c_scale: float
) -> dict[str, dict[str, float]]:
    values = {
        "candidate": float(candidate_c_scale),
        "champion": float(champion_c_scale),
    }
    for role, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise FleetError(f"{role}_c_scale must be finite and positive")
    return {role: {"c_scale": value} for role, value in values.items()}


def _plan_role_search_config(plan: dict[str, Any]) -> dict[str, dict[str, float]]:
    raw = plan.get("role_search_config")
    if raw is None:
        return _role_search_config(candidate_c_scale=0.03, champion_c_scale=0.03)
    if not isinstance(raw, dict):
        raise FleetError("evaluation plan role_search_config is malformed")
    try:
        config = _role_search_config(
            candidate_c_scale=raw["candidate"]["c_scale"],
            champion_c_scale=raw["champion"]["c_scale"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FleetError("evaluation plan role_search_config is malformed") from error
    if raw != config:
        raise FleetError("evaluation plan role_search_config is not canonical")
    return config


def _science_hash(plan: dict[str, Any]) -> str:
    if "role_search_config" not in plan:
        return _digest(SCIENCE_CONFIG)
    return _digest(
        {
            "science_config": SCIENCE_CONFIG,
            "role_search_config": _plan_role_search_config(plan),
        }
    )


def _internal_argv(
    *,
    python: str,
    candidate: str,
    champion: str,
    pairs: int,
    seed: int,
    workers: int,
    out: str,
    candidate_c_scale: float | None = None,
    champion_c_scale: float | None = None,
) -> list[str]:
    argv = [
        python,
        "tools/gumbel_search_cross_net_h2h.py",
        "--candidate",
        candidate,
        "--baseline",
        champion,
        "--pairs",
        str(pairs),
        "--base-seed",
        str(seed),
        "--workers",
        str(workers),
        "--threads-per-worker",
        "1",
        "--device",
        "cuda",
        "--map-kind",
        str(SCIENCE_CONFIG["internal_map_kind"]),
        *_science_args(
            c_scale=(
                0.03
                if candidate_c_scale is None and champion_c_scale is None
                else None
            )
        ),
        "--out",
        out,
    ]
    if candidate_c_scale is not None or champion_c_scale is not None:
        if candidate_c_scale is None or champion_c_scale is None:
            raise FleetError("both internal role c_scale values must be explicit")
        out_index = argv.index("--out")
        argv[out_index:out_index] = [
            "--candidate-c-scale",
            str(float(candidate_c_scale)),
            "--baseline-c-scale",
            str(float(champion_c_scale)),
        ]
    return argv


def _external_argv(
    *,
    python: str,
    checkpoint: str,
    pairs: int,
    seed: int,
    workers: int,
    artifact_dir: str,
    out: str,
    c_scale: float = 0.03,
    engine_identity: dict[str, str],
) -> list[str]:
    return [
        python,
        "tools/catanatron_neutral_harness_match.py",
        "--checkpoint",
        checkpoint,
        "--opponent",
        "catanatron_value",
        "--mode",
        "search",
        "--engine-repo-commit",
        engine_identity["repo_commit"],
        "--native-wheel-sha256",
        engine_identity["native_wheel_sha256"],
        "--python-referee-sha256",
        engine_identity["python_referee_sha256"],
        "--vps-to-win",
        "10",
        "--max-player-trade-offers-per-turn",
        "0",
        "--pairs",
        str(pairs),
        "--base-seed",
        str(seed),
        "--workers",
        str(workers),
        "--threads-per-worker",
        "1",
        "--device",
        "cuda",
        *_science_args(c_scale=c_scale),
        "--artifact-dir",
        artifact_dir,
        "--resume",
        "--out",
        out,
    ]


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _tool_hashes(repo_root: Path) -> dict[str, str]:
    names = (
        "tools/gumbel_search_cross_net_h2h.py",
        "tools/catanatron_neutral_harness_match.py",
        "tools/fleet/launch_detached.sh",
    )
    return {name: _sha256(repo_root / name) for name in names}


def _engine_identity(repo_root: Path, repo_commit: str) -> dict[str, str]:
    inventory = repo_root / "native/catanatron-rs/WHEEL_SHA256SUMS"
    rows = [line.split() for line in inventory.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 2 or not re.fullmatch(r"[0-9a-f]{64}", rows[0][0]):
        raise FleetError("native wheel checksum inventory is not one sealed wheel")
    referee_root = repo_root / "vendor/catanatron/catanatron"
    referee_files = sorted(path for path in referee_root.rglob("*.py") if path.is_file())
    if not referee_files:
        raise FleetError("vendored Python referee source tree is empty")
    referee_hasher = hashlib.sha256()
    for path in referee_files:
        relative = path.relative_to(referee_root).as_posix().encode("utf-8")
        referee_hasher.update(len(relative).to_bytes(8, "big"))
        referee_hasher.update(relative)
        payload = path.read_bytes()
        referee_hasher.update(len(payload).to_bytes(8, "big"))
        referee_hasher.update(payload)
    return {
        "schema_version": "a1-neutral-engine-identity-v1",
        "repo_commit": repo_commit,
        "native_wheel_sha256": "sha256:" + rows[0][0],
        "python_referee_sha256": "sha256:" + referee_hasher.hexdigest(),
    }


def build_plan(
    manifest: dict[str, Any],
    *,
    candidate: Path,
    champion: Path,
    candidate_parent: Path,
    registry: ChampionRegistry,
    internal_pairs: int,
    external_pairs: int,
    internal_base_seed: int,
    external_base_seed: int,
    workers_per_gpu: int = DEFAULT_WORKERS_PER_GPU,
    iteration_id: str = "a1",
    seed_cohort_id: str | None = None,
    scope: str = "full",
    host_aliases: Sequence[str] | None = None,
    repo_root: Path = _REPO_ROOT,
    repo_commit: str | None = None,
    tool_hashes: dict[str, str] | None = None,
    candidate_c_scale: float = 0.03,
    champion_c_scale: float = 0.03,
    comparison_mode: str = "promotion_parent",
    historical_comparison_reason: str | None = None,
) -> dict[str, Any]:
    candidate = candidate.expanduser().resolve(strict=True)
    champion = champion.expanduser().resolve(strict=True)
    candidate_sha = _sha256(candidate)
    champion_sha = _sha256(champion)
    if candidate_sha == champion_sha:
        raise FleetError("candidate and champion checkpoint bytes are identical")
    if workers_per_gpu <= 0:
        raise FleetError("workers_per_gpu must be positive")
    if scope not in {"canary", "full"}:
        raise FleetError("scope must be 'canary' or 'full'")
    all_slots = gpu_slots(manifest)
    slots = (
        [slot for slot in all_slots if slot["alias"] in CANARY_ALIASES]
        if scope == "canary"
        else all_slots
    )
    selected_aliases: list[str] | None = None
    if host_aliases is not None:
        requested = set(host_aliases)
        known = {str(host["alias"]) for host in manifest["hosts"]}
        unknown = sorted(requested - known)
        if unknown:
            raise FleetError(f"unknown evaluation host aliases: {unknown}")
        if not requested:
            raise FleetError("evaluation host subset must not be empty")
        slots = [slot for slot in slots if slot["alias"] in requested]
        selected_aliases = [
            str(host["alias"])
            for host in manifest["hosts"]
            if host["alias"] in requested
        ]
        if not slots or {slot["alias"] for slot in slots} != requested:
            raise FleetError("evaluation host subset is incompatible with plan scope")
    if internal_pairs < len(slots) or external_pairs < len(slots) // 2:
        raise FleetError(
            "production fleet plan requires at least one pair per internal GPU "
            "and per matched external cohort"
        )
    if not SAFE_NAME.fullmatch(iteration_id):
        raise FleetError("iteration_id must be a safe nonempty identifier")
    if seed_cohort_id is not None and not SAFE_NAME.fullmatch(seed_cohort_id):
        raise FleetError("seed_cohort_id must be a safe nonempty identifier")
    root = str(manifest["remote_root"])
    # All fleet nodes stage the bytes at the B200 source's exact absolute path.
    # This keeps evaluator typed-config paths aligned with the training receipt
    # and adjudication contract; hashes still prove the paths' contents.
    remote_candidate = str(candidate)
    remote_champion = str(champion)
    seed_intervals = [
        (internal_base_seed, internal_base_seed + internal_pairs, "internal"),
        (external_base_seed, external_base_seed + external_pairs, "external"),
    ]
    val_lo, val_hi = VAL_ONLY_SEED_RANGE
    for lo, hi, purpose in seed_intervals:
        if not (val_lo <= lo < hi <= val_hi):
            raise FleetError(
                f"{purpose} seed interval [{lo}, {hi}) is outside the dedicated "
                f"VAL-only band [{val_lo}, {val_hi})"
            )
    if not (
        seed_intervals[0][1] <= seed_intervals[1][0]
        or seed_intervals[1][1] <= seed_intervals[0][0]
    ):
        raise FleetError("internal and external validation seed intervals overlap")
    role_search_config = _role_search_config(
        candidate_c_scale=candidate_c_scale,
        champion_c_scale=champion_c_scale,
    )
    evaluation_binding = _evaluation_binding(
        candidate_parent=candidate_parent,
        baseline=champion,
        registry=registry,
        comparison_mode=comparison_mode,
        historical_comparison_reason=historical_comparison_reason,
        champion_c_scale=role_search_config["champion"]["c_scale"],
    )
    science_hash = _digest(
        {
            "science_config": SCIENCE_CONFIG,
            "role_search_config": role_search_config,
        }
    )
    resolved_repo_commit = repo_commit or _git_commit(repo_root)
    resolved_tool_hashes = tool_hashes or _tool_hashes(repo_root)
    engine_identity = _engine_identity(repo_root, resolved_repo_commit)
    run_identity = {
        "manifest_hash": manifest["manifest_hash"],
        "repo_commit": resolved_repo_commit,
        "tool_hashes": resolved_tool_hashes,
        "candidate": {"sha256": candidate_sha},
        "champion": {"sha256": champion_sha},
        "evaluation_binding": evaluation_binding,
        "engine_identity": engine_identity,
        "science_config_hash": science_hash,
        "role_search_config": role_search_config,
        "pair_claims": {
            "internal": {
                "base_seed": internal_base_seed,
                "pairs": internal_pairs,
            },
            "external_matched": {
                "base_seed": external_base_seed,
                "pairs": external_pairs,
            },
        },
        "iteration_id": iteration_id,
        "seed_cohort_id": seed_cohort_id,
        "scope": scope,
        "workers_per_gpu": workers_per_gpu,
    }
    if selected_aliases is not None:
        run_identity["host_aliases"] = selected_aliases
    run_id = _run_id_from_plan_fields(run_identity)
    run_root = f"{root}/runs/{run_id}"
    jobs: list[dict[str, Any]] = []
    for slot, (seed, count) in zip(
        slots,
        _split_ranges(internal_pairs, len(slots), internal_base_seed),
        strict=True,
    ):
        if count == 0:
            continue
        job_id = f"internal-{slot['slot_id']}"
        job_dir = f"{run_root}/internal/{job_id}"
        argv = _internal_argv(
            python=manifest["remote_python"],
            candidate=remote_candidate,
            champion=remote_champion,
            pairs=count,
            seed=seed,
            workers=workers_per_gpu,
            out=f"{job_dir}/report.json",
            candidate_c_scale=role_search_config["candidate"]["c_scale"],
            champion_c_scale=role_search_config["champion"]["c_scale"],
        )
        jobs.append(
            {
                **slot,
                "job_id": job_id,
                "phase": "internal",
                "role": "h2h",
                "base_seed": seed,
                "pairs": count,
                "job_dir": job_dir,
                "report": f"{job_dir}/report.json",
                "argv": argv,
                "command_hash": _digest(argv),
            }
        )
    external_slot_pairs = list(zip(slots[0::2], slots[1::2], strict=True))
    for cohort, ((seed, count), (candidate_slot, incumbent_slot)) in enumerate(
        zip(
            _split_ranges(external_pairs, len(external_slot_pairs), external_base_seed),
            external_slot_pairs,
            strict=True,
        )
    ):
        if count == 0:
            continue
        cohort_id = f"cohort-{cohort:02d}"
        for role, slot, checkpoint in (
            ("candidate", candidate_slot, remote_candidate),
            ("champion", incumbent_slot, remote_champion),
        ):
            job_id = f"external-{role}-{slot['slot_id']}"
            job_dir = f"{run_root}/external/{job_id}"
            argv = _external_argv(
                python=manifest["remote_python"],
                checkpoint=checkpoint,
                pairs=count,
                seed=seed,
                workers=workers_per_gpu,
                artifact_dir=f"{job_dir}/games",
                out=f"{job_dir}/report.json",
                c_scale=role_search_config[role]["c_scale"],
                engine_identity=engine_identity,
            )
            jobs.append(
                {
                    **slot,
                    "job_id": job_id,
                    "phase": "external",
                    "role": role,
                    "cohort_id": cohort_id,
                    "base_seed": seed,
                    "pairs": count,
                    "job_dir": job_dir,
                    "report": f"{job_dir}/report.json",
                    "argv": argv,
                    "command_hash": _digest(argv),
                }
            )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "run_id": run_id,
        "iteration_id": iteration_id,
        "seed_cohort_id": seed_cohort_id,
        "scope": scope,
        "manifest_hash": manifest["manifest_hash"],
        "repo_commit": resolved_repo_commit,
        "tool_hashes": resolved_tool_hashes,
        "science_config": SCIENCE_CONFIG,
        "science_config_hash": science_hash,
        "role_search_config": role_search_config,
        "evaluation_binding": evaluation_binding,
        "engine_identity": engine_identity,
        "workers_per_gpu": workers_per_gpu,
        "candidate": {
            "source": str(candidate),
            "remote": remote_candidate,
            "sha256": candidate_sha,
        },
        "champion": {
            "source": str(champion),
            "remote": remote_champion,
            "sha256": champion_sha,
        },
        "pair_claims": {
            "internal": {"base_seed": internal_base_seed, "pairs": internal_pairs},
            "external_matched": {
                "base_seed": external_base_seed,
                "pairs": external_pairs,
            },
        },
        "jobs": jobs,
    }
    if selected_aliases is not None:
        plan["host_aliases"] = selected_aliases
    plan["plan_hash"] = _digest(plan)
    return plan


def write_new_readonly(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_new_readonly_or_identical(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _canonical(_read_json(path)) != _canonical(value):
            raise FleetError(f"existing collected artifact differs: {path}")
        return
    write_new_readonly(path, value)


def _verify_plan_evaluation_binding(plan: dict[str, Any]) -> None:
    raw = plan.get("evaluation_binding")
    if not isinstance(raw, dict):
        raise FleetError("evaluation plan has no typed baseline binding")
    registry_ref = raw.get("registry")
    parent_ref = raw.get("candidate_parent")
    if not isinstance(registry_ref, dict) or not isinstance(parent_ref, dict):
        raise FleetError("evaluation plan baseline binding is malformed")
    registry_path = Path(str(registry_ref.get("path"))).expanduser().resolve(strict=True)
    if _sha256(registry_path) != registry_ref.get("sha256"):
        raise FleetError("evaluation registry bytes drifted after planning")
    expected = _evaluation_binding(
        candidate_parent=Path(str(parent_ref.get("path"))),
        baseline=Path(str(plan["champion"]["source"])),
        registry=ChampionRegistry.load(registry_path),
        comparison_mode=str(raw.get("comparison_mode")),
        historical_comparison_reason=raw.get("historical_comparison_reason"),
        champion_c_scale=_plan_role_search_config(plan)["champion"]["c_scale"],
    )
    if _canonical(raw) != _canonical(expected):
        raise FleetError("evaluation plan baseline binding does not replay")


def load_plan(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    plan = _read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise FleetError("unsupported A1 H100 eval plan schema")
    declared = plan.get("plan_hash")
    replay = _digest({key: value for key, value in plan.items() if key != "plan_hash"})
    if declared != replay:
        raise FleetError("evaluation plan hash does not replay")
    if plan.get("manifest_hash") != manifest.get("manifest_hash"):
        raise FleetError(
            "evaluation plan was built for a different private fleet manifest"
        )
    if plan.get("science_config") != SCIENCE_CONFIG:
        raise FleetError("evaluation plan science config drift")
    _plan_role_search_config(plan)
    _verify_plan_evaluation_binding(plan)
    if plan.get("engine_identity") != _engine_identity(
        _REPO_ROOT, str(plan.get("repo_commit"))
    ):
        raise FleetError("evaluation plan engine identity does not replay")
    if plan.get("science_config_hash") != _science_hash(plan):
        raise FleetError("evaluation plan science config drift")
    if plan.get("run_id") != _run_id_from_plan_fields(plan):
        raise FleetError("evaluation plan run identity does not replay")
    if not re.fullmatch(r"[0-9a-f]{40}", str(plan.get("repo_commit", ""))):
        raise FleetError("evaluation plan has no full Git commit")
    cohort = plan.get("seed_cohort_id")
    if cohort is not None and not SAFE_NAME.fullmatch(str(cohort)):
        raise FleetError("evaluation plan has an invalid seed_cohort_id")
    expected_tools = {
        "tools/gumbel_search_cross_net_h2h.py",
        "tools/catanatron_neutral_harness_match.py",
        "tools/fleet/launch_detached.sh",
    }
    if set(plan.get("tool_hashes", {})) != expected_tools or any(
        not re.fullmatch(r"sha256:[0-9a-f]{64}", str(value))
        for value in plan.get("tool_hashes", {}).values()
    ):
        raise FleetError("evaluation plan tool hashes are incomplete or malformed")
    for role in ("candidate", "champion"):
        source = Path(str(plan[role]["source"])).expanduser().resolve(strict=True)
        if _sha256(source) != plan[role]["sha256"]:
            raise FleetError(f"{role} checkpoint bytes drifted after planning")
        if plan[role].get("remote") != str(source):
            raise FleetError(
                f"{role} remote path must equal the immutable B200 source path"
            )
    _validate_planned_jobs(plan, manifest)
    return plan


def _validate_planned_jobs(plan: dict[str, Any], manifest: dict[str, Any]) -> None:
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise FleetError("evaluation plan jobs must be a list")
    all_slots = gpu_slots(manifest)
    scope = plan.get("scope")
    if scope not in {"canary", "full"}:
        raise FleetError("evaluation plan has an invalid scope")
    slots = (
        [slot for slot in all_slots if slot["alias"] in CANARY_ALIASES]
        if scope == "canary"
        else all_slots
    )
    if "host_aliases" in plan:
        raw_aliases = plan["host_aliases"]
        if (
            not isinstance(raw_aliases, list)
            or not raw_aliases
            or any(not isinstance(alias, str) for alias in raw_aliases)
            or len(set(raw_aliases)) != len(raw_aliases)
        ):
            raise FleetError("evaluation plan has an invalid host subset")
        manifest_order = [
            str(host["alias"])
            for host in manifest["hosts"]
            if host["alias"] in set(raw_aliases)
        ]
        if manifest_order != raw_aliases:
            raise FleetError("evaluation plan host subset is unknown or out of order")
        slots = [slot for slot in slots if slot["alias"] in set(raw_aliases)]
        if not slots or {slot["alias"] for slot in slots} != set(raw_aliases):
            raise FleetError("evaluation plan host subset is incompatible with scope")
    valid_slots = {(slot["alias"], slot["gpu"]): slot for slot in slots}
    run_root = f"{str(manifest['remote_root']).rstrip('/')}/runs/{plan['run_id']}"
    job_ids: set[str] = set()
    by_phase: dict[str, list[dict[str, Any]]] = {"internal": [], "external": []}
    role_search_config = _plan_role_search_config(plan)
    legacy_shared_search = "role_search_config" not in plan
    for job in jobs:
        if not isinstance(job, dict) or job.get("phase") not in by_phase:
            raise FleetError("evaluation plan contains an invalid job")
        identity = (job.get("alias"), job.get("gpu"))
        if (
            identity not in valid_slots
            or job.get("slot_id") != valid_slots[identity]["slot_id"]
        ):
            raise FleetError(f"evaluation job has an invalid GPU slot: {identity}")
        job_id = str(job.get("job_id", ""))
        if job_id in job_ids or not SAFE_NAME.fullmatch(job_id):
            raise FleetError(f"duplicate or invalid evaluation job id {job_id!r}")
        job_ids.add(job_id)
        expected_job_dir = f"{run_root}/{job['phase']}/{job_id}"
        if job.get("job_dir") != expected_job_dir or job.get(
            "report"
        ) != f"{expected_job_dir}/report.json":
            raise FleetError(f"evaluation job path escapes its sealed run: {job_id}")
        if job.get("command_hash") != _digest(job.get("argv")):
            raise FleetError(f"evaluation job command hash drift: {job_id}")
        by_phase[job["phase"]].append(job)
    if len(by_phase["internal"]) != len(slots) or len(by_phase["external"]) != len(
        slots
    ):
        raise FleetError(
            f"sealed A1 {scope} plan must allocate {len(slots)} jobs per phase"
        )
    if {(job["alias"], job["gpu"]) for job in by_phase["internal"]} != set(
        valid_slots
    ) or {(job["alias"], job["gpu"]) for job in by_phase["external"]} != set(
        valid_slots
    ):
        raise FleetError("each evaluation phase must allocate every physical H100 once")
    claims = plan.get("pair_claims", {})
    expected_internal = _split_ranges(
        int(claims["internal"]["pairs"]),
        len(slots),
        int(claims["internal"]["base_seed"]),
    )
    internal_by_slot = {(job["alias"], job["gpu"]): job for job in by_phase["internal"]}
    for slot, (seed, pairs) in zip(slots, expected_internal, strict=True):
        job = internal_by_slot[(slot["alias"], slot["gpu"])]
        expected = _internal_argv(
            python=manifest["remote_python"],
            candidate=plan["candidate"]["remote"],
            champion=plan["champion"]["remote"],
            pairs=pairs,
            seed=seed,
            workers=int(plan["workers_per_gpu"]),
            out=job["report"],
            candidate_c_scale=(
                None
                if legacy_shared_search
                else role_search_config["candidate"]["c_scale"]
            ),
            champion_c_scale=(
                None
                if legacy_shared_search
                else role_search_config["champion"]["c_scale"]
            ),
        )
        if job["pairs"] != pairs or job["base_seed"] != seed or job["argv"] != expected:
            raise FleetError(f"internal shard contract drift: {job['job_id']}")
    slot_pairs = list(zip(slots[0::2], slots[1::2], strict=True))
    expected_external = _split_ranges(
        int(claims["external_matched"]["pairs"]),
        len(slot_pairs),
        int(claims["external_matched"]["base_seed"]),
    )
    external_by_slot = {(job["alias"], job["gpu"]): job for job in by_phase["external"]}
    for cohort, ((seed, pairs), (candidate_slot, champion_slot)) in enumerate(
        zip(expected_external, slot_pairs, strict=True)
    ):
        for role, slot in (("candidate", candidate_slot), ("champion", champion_slot)):
            job = external_by_slot[(slot["alias"], slot["gpu"])]
            expected = _external_argv(
                python=manifest["remote_python"],
                checkpoint=plan[role]["remote"],
                pairs=pairs,
                seed=seed,
                workers=int(plan["workers_per_gpu"]),
                artifact_dir=f"{job['job_dir']}/games",
                    out=job["report"],
                    c_scale=role_search_config[role]["c_scale"],
                    engine_identity=plan["engine_identity"],
                )
            if (
                job["role"] != role
                or job["cohort_id"] != f"cohort-{cohort:02d}"
                or job["pairs"] != pairs
                or job["base_seed"] != seed
                or job["argv"] != expected
            ):
                raise FleetError(f"external matched-cohort drift: {job['job_id']}")


def _claim_payload(plan: dict[str, Any]) -> dict[str, Any]:
    claims = plan.get("pair_claims")
    if not isinstance(claims, dict):
        raise FleetError("plan has no validation pair claims")
    intervals = []
    for purpose in ("internal", "external_matched"):
        try:
            base = int(claims[purpose]["base_seed"])
            pairs = int(claims[purpose]["pairs"])
        except (KeyError, TypeError, ValueError) as error:
            raise FleetError(
                f"plan has an invalid {purpose} validation claim"
            ) from error
        if pairs <= 0:
            raise FleetError(f"plan {purpose} claim must be positive")
        intervals.append(
            {"purpose": purpose, "base_seed": base, "end_seed": base + pairs}
        )
    return {
        "schema_version": "a1-val-only-eval-claim-v2",
        "plan_hash": plan["plan_hash"],
        "run_id": plan["run_id"],
        "iteration_id": plan["iteration_id"],
        "seed_cohort_id": plan.get("seed_cohort_id"),
        "science_config_hash": plan["science_config_hash"],
        "status": "claimed",
        "intervals": intervals,
    }


def _claims_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["base_seed"]) < int(right["end_seed"]) and int(
        right["base_seed"]
    ) < int(left["end_seed"])


def _shared_claim_is_exact(
    wanted_payload: dict[str, Any],
    prior_payload: dict[str, Any],
    wanted: dict[str, Any],
    occupied: dict[str, Any],
) -> bool:
    """Allow only explicit common-random-number cohorts to share exact ranges."""
    cohort = wanted_payload.get("seed_cohort_id")
    return bool(
        cohort
        and cohort == prior_payload.get("seed_cohort_id")
        and wanted.get("purpose") == occupied.get("purpose")
        and int(wanted["base_seed"]) == int(occupied["base_seed"])
        and int(wanted["end_seed"]) == int(occupied["end_seed"])
    )


def claim_validation_ranges(manifest: dict[str, Any], plan: dict[str, Any]) -> str:
    """Atomically claim both VAL-only intervals or adopt this plan's exact claim."""
    ledger = Path(manifest["validation_ledger"])
    claims_dir = Path(str(ledger) + ".claims")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    claims_dir.mkdir(parents=True, exist_ok=True)
    payload = _claim_payload(plan)
    claim_path = claims_dir / f"{plan['plan_hash'][7:]}.json"
    lock_path = Path(str(ledger) + ".lock")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with os.fdopen(lock_fd, "r+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if claim_path.exists():
                prior = _read_json(claim_path)
                if _canonical(prior) != _canonical(payload):
                    raise FleetError(
                        "existing validation claim does not match this plan"
                    )
                journal_rows = []
                if ledger.exists():
                    for line in ledger.read_text(encoding="utf-8").splitlines():
                        try:
                            journal_rows.append(json.loads(line))
                        except json.JSONDecodeError as error:
                            raise FleetError(
                                f"invalid validation ledger row: {error}"
                            ) from error
                if not any(
                    row.get("event") == "claim"
                    and row.get("plan_hash") == plan["plan_hash"]
                    for row in journal_rows
                ):
                    with ledger.open("a", encoding="utf-8") as journal:
                        journal.write(
                            json.dumps(
                                {"event": "claim", "recovered": True, **payload},
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        journal.flush()
                        os.fsync(journal.fileno())
                return "adopted"
            for path in sorted(claims_dir.glob("*.json")):
                prior = _read_json(path)
                if prior.get("schema_version") not in {
                    "a1-val-only-eval-claim-v1",
                    payload["schema_version"],
                }:
                    raise FleetError(f"unknown validation claim schema in {path}")
                for wanted in payload["intervals"]:
                    for occupied in prior.get("intervals", []):
                        if _claims_overlap(wanted, occupied) and not _shared_claim_is_exact(
                            payload, prior, wanted, occupied
                        ):
                            raise FleetError(
                                "VAL-only seed overlap with prior claim "
                                f"{prior.get('plan_hash')}: {wanted} vs {occupied}"
                            )
            descriptor = os.open(
                claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                claim_path.unlink(missing_ok=True)
                raise
            with ledger.open("a", encoding="utf-8") as journal:
                journal.write(
                    json.dumps({"event": "claim", **payload}, sort_keys=True) + "\n"
                )
                journal.flush()
                os.fsync(journal.fileno())
            return "claimed"
    finally:
        # fdopen owns and closes lock_fd on normal and exceptional paths after it
        # has been entered.  If fdopen itself failed, close the still-open fd.
        try:
            os.close(lock_fd)
        except OSError:
            pass


def record_validation_status(
    manifest: dict[str, Any], plan: dict[str, Any], *, status: str
) -> None:
    if not SAFE_NAME.fullmatch(status):
        raise FleetError("validation status must be a safe identifier")
    if claim_validation_ranges(manifest, plan) not in {"claimed", "adopted"}:
        raise AssertionError("claim adoption returned an unknown state")
    ledger = Path(manifest["validation_ledger"])
    lock_path = Path(str(ledger) + ".lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with ledger.open("a", encoding="utf-8") as journal:
            journal.write(
                json.dumps(
                    {
                        "event": "status",
                        "plan_hash": plan["plan_hash"],
                        "run_id": plan["run_id"],
                        "iteration_id": plan["iteration_id"],
                        "status": status,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            journal.flush()
            os.fsync(journal.fileno())


def _ssh_base(manifest: dict[str, Any], host: dict[str, Any]) -> list[str]:
    return [
        "ssh",
        "-i",
        manifest["ssh_key"],
        "-o",
        "BatchMode=yes",
        "-o",
        f"StrictHostKeyChecking={manifest['strict_host_key_checking']}",
        "-o",
        "ConnectTimeout=15",
        f"{manifest['ssh_user']}@{host['address']}",
    ]


def _scp_base(manifest: dict[str, Any]) -> list[str]:
    return [
        "scp",
        "-i",
        manifest["ssh_key"],
        "-o",
        "BatchMode=yes",
        "-o",
        f"StrictHostKeyChecking={manifest['strict_host_key_checking']}",
        "-o",
        "ConnectTimeout=15",
    ]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    argv = list(command)
    attempts = 3 if argv and Path(argv[0]).name in {"ssh", "scp"} else 1
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(attempts):
        try:
            return subprocess.run(
                argv,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as error:
            last_error = error
            executable = Path(argv[0]).name if argv else ""
            retryable = executable == "scp" or error.returncode == 255
            if retryable and attempt + 1 < attempts:
                time.sleep(1 << attempt)
            else:
                break
    assert last_error is not None
    raise last_error


def _host_by_alias(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {host["alias"]: host for host in manifest["hosts"]}


def _preflight_command(
    manifest: dict[str, Any], plan: dict[str, Any], host: dict[str, Any]
) -> str:
    repo = shlex.quote(manifest["remote_repo"])
    pythonpath = manifest["remote_repo"] + "/src:" + manifest["remote_repo"]
    import_probe = (
        "from pathlib import Path; import catan_zero; "
        "assert Path(catan_zero.__file__).resolve().is_relative_to("
        f"Path({manifest['remote_repo']!r}) / 'src')"
    )
    lines = [
        "set -euo pipefail",
        f"cd {repo}",
        f'test "$(git rev-parse HEAD)" = {shlex.quote(plan["repo_commit"])}',
        "git diff --quiet --exit-code -- .",
        "git diff --cached --quiet --exit-code -- .",
        (
            "grep -Fxq "
            + shlex.quote(
                plan["engine_identity"]["native_wheel_sha256"].removeprefix(
                    "sha256:"
                )
                + "  catanatron_rs-0.1.5-cp311-cp311-manylinux_2_34_x86_64.whl"
            )
            + " native/catanatron-rs/WHEEL_SHA256SUMS"
        ),
        f'test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq {host["gpu_count"]}',
        "test \"$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -vc 'H100')\" -eq 0",
        # A healthy idle fleet keeps one MPS server attached to every GPU.
        # Reject every other compute process while allowing that daemon.
        'test "$(nvidia-smi --query-compute-apps=process_name '
        "--format=csv,noheader,nounits 2>/dev/null "
        "| grep -Evc '(^|/)nvidia-cuda-mps-server$' || true)\" -eq 0",
        'test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits '
        "| awk '$1 > 128 {n++} END {print n+0}')\" -eq 0",
        f"test -x {shlex.quote(manifest['remote_python'])}",
        f"PYTHONPATH={shlex.quote(pythonpath)} "
        f"{shlex.quote(manifest['remote_python'])} -c "
        f"{shlex.quote(import_probe)}",
    ]
    for tool, expected in sorted(plan["tool_hashes"].items()):
        lines.append(
            f"test \"sha256:$(sha256sum {shlex.quote(tool)} | cut -d' ' -f1)\" = {shlex.quote(expected)}"
        )
    for role in ("candidate", "champion"):
        lines.append(
            f"test \"sha256:$(sha256sum {shlex.quote(plan[role]['remote'])} | cut -d' ' -f1)\" = {shlex.quote(plan[role]['sha256'])}"
        )
    return "\n".join(lines)


def _launch_job_command(manifest: dict[str, Any], job: dict[str, Any]) -> str:
    job_dir = str(job["job_dir"])
    report = str(job["report"])
    log = f"{job_dir}/run.log"
    command = " ".join(shlex.quote(part) for part in job["argv"])
    inner = (
        "set +e; "
        f"cd {shlex.quote(manifest['remote_repo'])}; "
        f"env CUDA_VISIBLE_DEVICES={int(job['gpu'])} "
        "CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_host "
        "CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_host "
        f"PYTHONPATH={shlex.quote(manifest['remote_repo'] + '/src:' + manifest['remote_repo'])} "
        f"PYTHONUNBUFFERED=1 {command}; rc=$?; "
        f"printf '%s\\n' \"$rc\" > {shlex.quote(job_dir + '/.rc.tmp')}; "
        f"mv -f {shlex.quote(job_dir + '/.rc.tmp')} {shlex.quote(job_dir + '/.rc')}; "
        f'if [ "$rc" -eq 0 ] && [ -s {shlex.quote(report)} ]; then '
        f"rm -f {shlex.quote(job_dir + '/.failed')}; touch {shlex.quote(job_dir + '/.done')}; "
        f"else rm -f {shlex.quote(job_dir + '/.done')}; touch {shlex.quote(job_dir + '/.failed')}; fi; "
        'exit "$rc"'
    )
    detached = [
        manifest["remote_repo"].rstrip("/") + "/tools/fleet/launch_detached.sh",
        job_dir,
        log,
        "60",
        "--",
        "bash",
        "-lc",
        inner,
    ]
    protected_paths = [
        report,
        log,
        *(f"{job_dir}/{name}" for name in (".done", ".failed", ".pid", ".rc")),
    ]
    phase_root = str(Path(job_dir).parent)
    run_root = str(Path(phase_root).parent)
    remote_root = str(manifest["remote_root"]).rstrip("/")
    protected_roots = [remote_root, f"{remote_root}/runs", run_root, phase_root]
    create_roots: list[str] = []
    for path in protected_roots:
        quoted = shlex.quote(path)
        create_roots.extend(
            [
                f"test ! -L {quoted}",
                f"if [ ! -e {quoted} ]; then mkdir {quoted}; fi",
                f"test -d {quoted}",
                f"test \"$(readlink -f {quoted})\" = {quoted}",
            ]
        )
    quoted_job_dir = shlex.quote(job_dir)
    return "\n".join(
        [
            "set -euo pipefail",
            *create_roots,
            f"test ! -L {quoted_job_dir}",
            f"if [ ! -e {quoted_job_dir} ]; then mkdir {quoted_job_dir}; fi",
            f"test -d {quoted_job_dir}",
            f"test \"$(readlink -f {quoted_job_dir})\" = {quoted_job_dir}",
            *(f"test ! -L {shlex.quote(path)}" for path in protected_paths),
            f"if [ -f {shlex.quote(job_dir + '/.done')} ]; then echo {shlex.quote(job['job_id'] + ':done')};",
            f'elif [ -s {shlex.quote(job_dir + "/.pid")} ] && kill -0 "$(cat {shlex.quote(job_dir + "/.pid")})" 2>/dev/null; then echo {shlex.quote(job["job_id"] + ":active")};',
            "else",
            f"rm -f {shlex.quote(job_dir + '/.failed')} {shlex.quote(job_dir + '/.rc')};",
            " ".join(shlex.quote(part) for part in detached),
            "fi",
        ]
    )


def _stage_local(plan: dict[str, Any]) -> None:
    for role in ("candidate", "champion"):
        source = Path(plan[role]["source"])
        target = Path(plan[role]["remote"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and _sha256(target) == plan[role]["sha256"]:
            continue
        temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
        shutil.copyfile(source, temporary)
        if _sha256(temporary) != plan[role]["sha256"]:
            temporary.unlink(missing_ok=True)
            raise FleetError(f"local stage hash mismatch for {role}")
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(temporary, target)


def _prepare_remote_host(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    host: dict[str, Any],
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> None:
    checkpoint_dirs = sorted(
        {str(Path(plan[role]["remote"]).parent) for role in ("candidate", "champion")}
    )
    runner(
        [
            *_ssh_base(manifest, host),
            "mkdir -p " + " ".join(shlex.quote(path) for path in checkpoint_dirs),
        ]
    )
    target = f"{manifest['ssh_user']}@{host['address']}"
    for role in ("candidate", "champion"):
        remote = plan[role]["remote"]
        probe = (
            f"test -s {shlex.quote(remote)} && "
            f"test \"sha256:$(sha256sum {shlex.quote(remote)} | cut -d' ' -f1)\" = "
            f"{shlex.quote(plan[role]['sha256'])}"
        )
        try:
            runner([*_ssh_base(manifest, host), probe])
            continue
        except subprocess.CalledProcessError:
            pass
        temporary = remote + f".tmp.{os.getpid()}"
        runner([*_scp_base(manifest), plan[role]["source"], f"{target}:{temporary}"])
        finalize = (
            f"test \"sha256:$(sha256sum {shlex.quote(temporary)} | cut -d' ' -f1)\" = "
            f"{shlex.quote(plan[role]['sha256'])} && chmod 0444 {shlex.quote(temporary)} "
            f"&& mv -f {shlex.quote(temporary)} {shlex.quote(remote)}"
        )
        runner([*_ssh_base(manifest, host), finalize])
    runner([*_ssh_base(manifest, host), _preflight_command(manifest, plan, host)])


def _jobs(plan: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    return [job for job in plan["jobs"] if job["phase"] == phase]


def _parallel(
    rows: Iterable[Any], function: Callable[[Any], Any], workers: int = 8
) -> list[Any]:
    values = list(rows)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(values) or 1))) as pool:
        futures = {pool.submit(function, value): value for value in values}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def dry_run_commands(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    phase: str,
    *,
    selected_job_ids: set[str] | None = None,
) -> dict[str, Any]:
    hosts = _host_by_alias(manifest)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in _jobs(plan, phase):
        if selected_job_ids is not None and job["job_id"] not in selected_job_ids:
            continue
        grouped.setdefault(job["alias"], []).append(job)
    rows = []
    for alias in sorted(grouped):
        host = hosts[alias]
        command = "\n".join(
            [_preflight_command(manifest, plan, host)]
            + [_launch_job_command(manifest, job) for job in grouped[alias]]
        )
        rows.append(
            {
                "alias": alias,
                "target": f"{manifest['ssh_user']}@{host['address']}",
                "gpus": host["gpu_count"],
                "jobs": len(grouped[alias]),
                "ssh_command": [*_ssh_base(manifest, host), command],
            }
        )
    return {
        "dry_run": True,
        "phase": phase,
        "plan_hash": plan["plan_hash"],
        "hosts": rows,
    }


def launch_phase(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    phase: str,
    *,
    selected_job_ids: set[str] | None = None,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> dict[str, Any]:
    _require_b200_origin(runner=runner)
    claim_state = claim_validation_ranges(manifest, plan)
    _stage_local(plan)
    hosts = _host_by_alias(manifest)
    jobs = [
        job
        for job in _jobs(plan, phase)
        if selected_job_ids is None or job["job_id"] in selected_job_ids
    ]
    aliases = sorted({job["alias"] for job in jobs})
    _parallel(
        [hosts[alias] for alias in aliases],
        lambda host: _prepare_remote_host(manifest, plan, host, runner=runner),
    )
    grouped: dict[str, list[dict[str, Any]]] = {alias: [] for alias in aliases}
    for job in jobs:
        grouped[job["alias"]].append(job)

    def launch(alias: str) -> dict[str, Any]:
        host = hosts[alias]
        command = "set -euo pipefail\n" + "\n".join(
            _launch_job_command(manifest, job) for job in grouped[alias]
        )
        result = runner([*_ssh_base(manifest, host), command])
        return {"alias": alias, "jobs": len(grouped[alias]), "stdout": result.stdout}

    rows = _parallel(aliases, launch)
    return {
        "phase": phase,
        "plan_hash": plan["plan_hash"],
        "validation_claim": claim_state,
        "launched": sorted(rows, key=lambda row: row["alias"]),
    }


def _require_b200_origin(
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> None:
    try:
        result = runner(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ]
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FleetError("--go must run on the B200 control host") from error
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names or any("B200" not in name.upper() for name in names):
        raise FleetError(
            f"--go must run on a B200-only control host, got {names or ['no GPUs']}"
        )


def _status_command(jobs: Sequence[dict[str, Any]]) -> str:
    lines = ["set -u"]
    for job in jobs:
        directory = shlex.quote(str(job["job_dir"]))
        job_id = shlex.quote(str(job["job_id"]))
        lines.extend(
            [
                f"d={directory}; state=missing; pid='';",
                'if { [ -e "$d" ] && [ "$(readlink -f "$d")" != "$d" ]; } '
                '|| [ -L "$d" ] || [ -L "$d/.done" ] || [ -L "$d/.failed" ] '
                '|| [ -L "$d/.pid" ] || [ -L "$d/.rc" ] '
                '|| [ -L "$d/report.json" ] || [ -L "$d/run.log" ]; then state=unsafe; '
                'elif [ -f "$d/.done" ]; then state=done; '
                'elif [ -f "$d/.failed" ]; then state=failed; '
                'elif [ -s "$d/.pid" ]; then pid=$(cat "$d/.pid"); '
                'if kill -0 "$pid" 2>/dev/null; then state=active; else state=stale; fi; fi;',
                f'printf \'%s\\t%s\\t%s\\n\' {job_id} "$state" "$pid";',
            ]
        )
    return "\n".join(lines)


def parse_status(stdout: str, *, alias: str) -> list[dict[str, Any]]:
    rows = []
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            raise FleetError(f"invalid status line from {alias}: {line!r}")
        rows.append(
            {
                "job_id": parts[0],
                "state": parts[1],
                "pid": parts[2] or None,
                "alias": alias,
            }
        )
    return rows


def status_phase(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    phase: str,
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> dict[str, Any]:
    hosts = _host_by_alias(manifest)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in _jobs(plan, phase):
        grouped.setdefault(job["alias"], []).append(job)

    def poll(alias: str) -> list[dict[str, Any]]:
        result = runner(
            [*_ssh_base(manifest, hosts[alias]), _status_command(grouped[alias])]
        )
        return parse_status(result.stdout, alias=alias)

    rows = [row for group in _parallel(sorted(grouped), poll) for row in group]
    counts = {
        state: sum(row["state"] == state for row in rows)
        for state in ("done", "active", "failed", "stale", "missing", "unsafe")
    }
    return {
        "phase": phase,
        "plan_hash": plan["plan_hash"],
        "counts": counts,
        "jobs": sorted(rows, key=lambda row: row["job_id"]),
    }


def jobs_to_resume(
    plan: dict[str, Any], status: dict[str, Any], phase: str
) -> set[str]:
    expected = {job["job_id"] for job in _jobs(plan, phase)}
    states = {row["job_id"]: row["state"] for row in status["jobs"]}
    if set(states) != expected:
        raise FleetError(
            "status response does not cover every planned job exactly once"
        )
    valid_states = {"done", "active", "failed", "stale", "missing"}
    if any(state not in valid_states for state in states.values()):
        raise FleetError("status contains an unsafe or unknown remote job state")
    return {
        job_id
        for job_id, state in states.items()
        if state in {"failed", "stale", "missing"}
    }


def _fetch_report(
    manifest: dict[str, Any],
    job: dict[str, Any],
    destination: Path,
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> Path:
    host = _host_by_alias(manifest)[job["alias"]]
    digest_result = runner(
        [
            *_ssh_base(manifest, host),
            f"test ! -L {shlex.quote(job['job_dir'])} && "
            f"test \"$(readlink -f {shlex.quote(job['job_dir'])})\" = "
            f"{shlex.quote(job['job_dir'])} && "
            f"test ! -L {shlex.quote(job['job_dir'] + '/.done')} && "
            f"test ! -L {shlex.quote(job['report'])} && "
            f"test \"$(readlink -f {shlex.quote(job['report'])})\" = "
            f"{shlex.quote(job['report'])} && "
            f"test -f {shlex.quote(job['job_dir'] + '/.done')} && "
            f"sha256sum {shlex.quote(job['report'])} | cut -d' ' -f1",
        ]
    )
    expected = digest_result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise FleetError(f"invalid report digest from {job['job_id']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _sha256(destination) == "sha256:" + expected:
        return destination
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    target = f"{manifest['ssh_user']}@{host['address']}:{job['report']}"
    runner([*_scp_base(manifest), target, str(temporary)])
    if _sha256(temporary) != "sha256:" + expected:
        temporary.unlink(missing_ok=True)
        raise FleetError(f"report transfer hash mismatch for {job['job_id']}")
    os.replace(temporary, destination)
    return destination


def collect_phase(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    phase: str,
    output_dir: Path,
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> dict[str, Any]:
    _require_b200_origin(runner=runner)
    status = status_phase(manifest, plan, phase, runner=runner)
    if status["counts"]["done"] != len(_jobs(plan, phase)):
        raise FleetError(f"cannot collect incomplete {phase} phase: {status['counts']}")
    report_dir = output_dir.expanduser() / plan["run_id"] / "shards" / phase
    reports = _parallel(
        _jobs(plan, phase),
        lambda job: _fetch_report(
            manifest, job, report_dir / f"{job['job_id']}.json", runner=runner
        ),
    )
    pooled_dir = output_dir.expanduser() / plan["run_id"] / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    if phase == "internal":
        result = evaluation_pool.pool_internal(
            reports,
            candidate=Path(plan["candidate"]["remote"]),
            champion=Path(plan["champion"]["remote"]),
        )
        result["evaluation_binding"] = plan["evaluation_binding"]
        result["planned_engine_identity"] = plan["engine_identity"]
        destination = pooled_dir / "internal.json"
        write_new_readonly_or_identical(destination, result)
        outputs = {"internal": str(destination)}
    else:
        by_role: dict[str, list[Path]] = {"candidate": [], "champion": []}
        path_by_id = {path.stem: path for path in reports}
        for job in _jobs(plan, phase):
            by_role[job["role"]].append(path_by_id[job["job_id"]])
        outputs = {}
        for role in ("candidate", "champion"):
            result = evaluation_pool.pool_neutral(
                by_role[role], checkpoint=Path(plan[role]["remote"])
            )
            result["evaluation_binding"] = plan["evaluation_binding"]
            result["planned_engine_identity"] = plan["engine_identity"]
            destination = pooled_dir / f"external-{role}.json"
            write_new_readonly_or_identical(destination, result)
            outputs[role] = str(destination)
    record_validation_status(manifest, plan, status=f"{phase}_collected")
    return {"phase": phase, "plan_hash": plan["plan_hash"], "outputs": outputs}


def ray_cluster_spec(manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Describe, but do not install/start, a Ray deployment for this plan."""
    head_address = manifest.get("ray_head_address")
    if not head_address or not SAFE_ADDRESS.fullmatch(str(head_address)):
        raise FleetError(
            "manifest must define a safe ray_head_address to render Ray config"
        )
    workers = [
        {
            "alias": host["alias"],
            "ssh_target": f"{manifest['ssh_user']}@{host['address']}",
            "num_gpus": host["gpu_count"],
            "resources": {"H100": host["gpu_count"]},
            "start_argv": [
                "ray",
                "start",
                f"--address={head_address}:6379",
                f"--num-gpus={host['gpu_count']}",
                "--resources="
                + json.dumps({"H100": host["gpu_count"]}, sort_keys=True),
            ],
        }
        for host in manifest["hosts"]
    ]
    physical_gpu_slots = sum(int(host["gpu_count"]) for host in manifest["hosts"])
    return {
        "schema_version": RAY_SCHEMA,
        "plan_hash": plan["plan_hash"],
        "installation_performed": False,
        "head": {
            "address": head_address,
            "num_gpus": 0,
            "start_argv": ["ray", "start", "--head", "--port=6379", "--num-gpus=0"],
        },
        "workers": workers,
        "scheduler_contract": {
            "actor_resources": {"num_gpus": 1, "resources": {"H100": 1}},
            "physical_gpu_slots": physical_gpu_slots,
            "max_concurrent_actors": physical_gpu_slots,
            "job_commands_are_plan_argv": True,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--candidate", type=Path, required=True)
    plan.add_argument("--champion", type=Path, required=True)
    plan.add_argument(
        "--candidate-parent",
        type=Path,
        required=True,
        help="Authenticated parent/init checkpoint from the candidate training receipt.",
    )
    plan.add_argument("--registry", type=Path, required=True)
    plan.add_argument(
        "--comparison-mode",
        choices=("promotion_parent", "historical_comparison"),
        default="promotion_parent",
    )
    plan.add_argument("--historical-comparison-reason")
    plan.add_argument("--internal-pairs", type=int, default=600)
    plan.add_argument("--external-pairs", type=int, default=500)
    plan.add_argument("--internal-base-seed", type=int, required=True)
    plan.add_argument("--external-base-seed", type=int, required=True)
    plan.add_argument(
        "--workers-per-gpu", type=int, default=DEFAULT_WORKERS_PER_GPU
    )
    plan.add_argument("--iteration-id", required=True)
    plan.add_argument(
        "--seed-cohort-id",
        help=(
            "Explicit common-random-number cohort. Plans with the same ID, "
            "purpose, and exact interval may intentionally reuse VAL seeds for "
            "matched checkpoint or search-config comparisons; partial overlap "
            "remains forbidden."
        ),
    )
    plan.add_argument("--scope", choices=("canary", "full"), default="full")
    plan.add_argument(
        "--host-aliases",
        help=(
            "Optional comma-separated approved host subset. The private manifest "
            "still validates the complete fleet; only sealed jobs are restricted."
        ),
    )
    plan.add_argument(
        "--candidate-c-scale",
        type=float,
        required=True,
        help="Contract-bound candidate agent c_scale (never inferred by role).",
    )
    plan.add_argument(
        "--champion-c-scale",
        type=float,
        required=True,
        help="Registry-bound incumbent agent c_scale (never inferred by role).",
    )
    plan.add_argument("--out", type=Path, required=True)
    for name in ("launch", "resume"):
        operation = commands.add_parser(name)
        operation.add_argument("--plan", type=Path, required=True)
        operation.add_argument(
            "--phase", choices=("internal", "external"), required=True
        )
        mode = operation.add_mutually_exclusive_group(required=True)
        mode.add_argument("--dry-run", action="store_true")
        mode.add_argument("--go", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--plan", type=Path, required=True)
    status.add_argument("--phase", choices=("internal", "external"), required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--phase", choices=("internal", "external"), required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    ray = commands.add_parser("ray-config")
    ray.add_argument("--plan", type=Path, required=True)
    ray.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "plan":
            value = build_plan(
                manifest,
                candidate=args.candidate,
                champion=args.champion,
                candidate_parent=args.candidate_parent,
                registry=ChampionRegistry.load(args.registry),
                internal_pairs=args.internal_pairs,
                external_pairs=args.external_pairs,
                internal_base_seed=args.internal_base_seed,
                external_base_seed=args.external_base_seed,
                workers_per_gpu=args.workers_per_gpu,
                iteration_id=args.iteration_id,
                seed_cohort_id=args.seed_cohort_id,
                scope=args.scope,
                host_aliases=(
                    [alias for alias in args.host_aliases.split(",") if alias]
                    if args.host_aliases is not None
                    else None
                ),
                candidate_c_scale=args.candidate_c_scale,
                champion_c_scale=args.champion_c_scale,
                comparison_mode=args.comparison_mode,
                historical_comparison_reason=args.historical_comparison_reason,
            )
            write_new_readonly(args.out, value)
            result = {
                "plan": str(args.out.resolve()),
                "plan_hash": value["plan_hash"],
                "jobs": len(value["jobs"]),
            }
        else:
            plan = load_plan(args.plan, manifest)
            if args.command == "launch":
                result = (
                    dry_run_commands(manifest, plan, args.phase)
                    if args.dry_run
                    else launch_phase(manifest, plan, args.phase)
                )
            elif args.command == "resume":
                status = status_phase(manifest, plan, args.phase)
                resumable = jobs_to_resume(plan, status, args.phase)
                if args.dry_run:
                    result = {
                        **dry_run_commands(
                            manifest,
                            plan,
                            args.phase,
                            selected_job_ids=resumable,
                        ),
                        "resumable_job_ids": sorted(resumable),
                    }
                else:
                    result = launch_phase(
                        manifest, plan, args.phase, selected_job_ids=resumable
                    )
            elif args.command == "status":
                result = status_phase(manifest, plan, args.phase)
            elif args.command == "collect":
                result = collect_phase(manifest, plan, args.phase, args.output_dir)
            elif args.command == "ray-config":
                result = ray_cluster_spec(manifest, plan)
                write_new_readonly(args.out, result)
                result = {
                    "ray_config": str(args.out.resolve()),
                    "plan_hash": plan["plan_hash"],
                }
            else:  # pragma: no cover - argparse requires a known command.
                raise AssertionError(args.command)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        FleetError,
        evaluation_pool.PoolError,
        OSError,
        subprocess.CalledProcessError,
        KeyError,
        ValueError,
    ) as error:
        print(f"A1 H100 evaluation fleet refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
