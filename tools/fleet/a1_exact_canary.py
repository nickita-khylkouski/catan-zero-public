#!/usr/bin/env python3
"""Fail-closed static canary for the exact A1 n128 production render.

The validator consumes the immutable render plus the executor's public dry-run
plan.  It does not launch work.  Its purpose is to prove that both a 4-GPU and
an 8-GPU host shape are represented by independent, per-GPU lanes using the
one authorized search recipe and the host-managed MPS client environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_pre_wave_contract as contract  # noqa: E402


class CanaryError(RuntimeError):
    """The rendered cohort is not the exact production canary contract."""


CATEGORY_ORDER = ("current_producer", "recent_history", "hard_negative")
EXPECTED_MPS_ENVIRONMENT = {
    "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps_pipe_host",
    "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps_log_host",
}
EXPECTED_VALUE_FLAGS = {
    "--n-full": "128",
    "--n-fast": "16",
    "--p-full": "0.25",
    "--c-scale": "0.1",
    "--c-visit": "50.0",
    "--max-depth": "80",
    "--workers": "16",
    "--device": "cuda",
    "--symmetry-averaged-eval-threshold": "20",
    "--rescale-noise-floor-c": "0.0",
    "--determinization-particles": "4",
    "--determinization-min-simulations": "32",
}
REQUIRED_SWITCHES = {
    "--symmetry-averaged-eval",
    "--public-observation",
    "--information-set-search",
    "--lazy-interior-chance",
    "--no-belief-chance-spectra",
    "--no-wide-roots-always-full",
    "--no-eval-server",
    "--seed-claim",
    "--resume",
}
FORBIDDEN_FLAGS = {
    "--n-full-wide",
    "--n-full-wide-threshold",
    "--wide-roots-always-full",
    "--eval-server",
    "--skip-guards",
    "--no-seed-claim",
    "--no-public-observation",
    "--no-information-set-search",
    "--belief-chance-spectra",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CanaryError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise CanaryError(f"{path} must contain a JSON object")
    return value


def _flag_map(argv: Any, *, job_id: str) -> tuple[dict[str, str], set[str]]:
    if not isinstance(argv, list) or not argv or not all(
        isinstance(item, str) for item in argv
    ):
        raise CanaryError(f"{job_id}: argv must be a non-empty string list")
    values: dict[str, str] = {}
    switches: set[str] = set()
    index = 1
    while index < len(argv):
        item = argv[index]
        if not item.startswith("--"):
            raise CanaryError(f"{job_id}: orphan argv token {item!r}")
        if item in values or item in switches:
            raise CanaryError(f"{job_id}: duplicate flag {item}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            values[item] = argv[index + 1]
            index += 2
        else:
            switches.add(item)
            index += 1
    return values, switches


def _verify_digest(payload: dict[str, Any], field: str, *, where: str) -> None:
    unhashed = dict(payload)
    digest = unhashed.pop(field, None)
    if digest != contract._digest_value(unhashed):
        raise CanaryError(f"{where} semantic digest mismatch")


def _parse_shapes(raw: Sequence[str]) -> dict[str, int]:
    shapes: dict[str, int] = {}
    for item in raw:
        alias, separator, count = item.partition("=")
        if not separator or not alias or alias in shapes:
            raise CanaryError(f"invalid/duplicate host shape {item!r}; use ALIAS=4 or ALIAS=8")
        try:
            value = int(count)
        except ValueError as error:
            raise CanaryError(f"invalid GPU count in host shape {item!r}") from error
        if value not in (4, 8):
            raise CanaryError(f"host shape {alias!r} must be exactly 4 or 8 GPUs")
        shapes[alias] = value
    if sorted(shapes.values()) != [4, 8]:
        raise CanaryError("canary must name exactly one 4-GPU host and one 8-GPU host")
    return shapes


def _bound_lock(
    rendered: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any] | None:
    """Replay the sealed topology when validating a real executor plan.

    Small unit fixtures predate the path fields and are validated from their
    internally authenticated render/plan shape. Production render and dry-run
    payloads carry both paths; those must resolve to one verified v2/v3 lock.
    """

    render_path = rendered.get("contract_path")
    plan_path = plan.get("lock")
    if render_path is None and plan_path is None:
        return None
    if not isinstance(render_path, str) or not isinstance(plan_path, str):
        raise CanaryError("production render/plan must both bind the sealed lock path")
    try:
        rendered_lock_path = Path(render_path).expanduser().resolve(strict=True)
        planned_lock_path = Path(plan_path).expanduser().resolve(strict=True)
    except OSError as error:
        raise CanaryError(f"cannot resolve sealed canary lock: {error}") from error
    if rendered_lock_path != planned_lock_path:
        raise CanaryError("executor plan and render bind different lock paths")
    try:
        lock = contract.verify_lock(rendered_lock_path)
    except Exception as error:
        raise CanaryError(f"sealed canary lock replay failed: {error}") from error
    if lock.get("contract_sha256") != rendered.get("contract_sha256"):
        raise CanaryError("sealed canary lock identity differs from render/plan")
    return lock


def validate_exact_canary(
    rendered: dict[str, Any], plan: dict[str, Any], host_shapes: dict[str, int]
) -> dict[str, Any]:
    """Validate exact recipe, pinning, MPS binding, topology, outputs, and seeds."""

    if rendered.get("schema_version") != contract.RENDER_SCHEMA:
        raise CanaryError(f"render schema must be {contract.RENDER_SCHEMA}")
    _verify_digest(rendered, "render_sha256", where="render")
    _verify_digest(plan, "plan_sha256", where="executor plan")
    if plan.get("contract_sha256") != rendered.get("contract_sha256"):
        raise CanaryError("executor plan binds a different contract")
    if plan.get("render_sha256") != rendered.get("render_sha256"):
        raise CanaryError("executor plan binds a different render")
    if plan.get("client_environment") != EXPECTED_MPS_ENVIRONMENT:
        raise CanaryError(
            "executor MPS client environment must exactly bind the managed host pipe/log"
        )

    bound_lock = _bound_lock(rendered, plan)
    commands = rendered.get("commands")
    declared_lanes = plan.get("lane_count")
    declared_jobs = plan.get("job_count")
    if (
        not isinstance(commands, list)
        or not commands
        or isinstance(declared_lanes, bool)
        or not isinstance(declared_lanes, int)
        or declared_lanes <= 0
        or isinstance(declared_jobs, bool)
        or not isinstance(declared_jobs, int)
        or declared_jobs <= 0
        or len(commands) != declared_jobs
        or declared_jobs != declared_lanes * len(CATEGORY_ORDER)
    ):
        raise CanaryError(
            "A1 render/plan must bind a positive lane count and exactly three jobs per lane"
        )
    bound_jobs: dict[str, dict[str, Any]] | None = None
    if bound_lock is not None:
        try:
            topology = contract._sealed_game_contract_shape(bound_lock)  # noqa: SLF001
        except contract.ContractError as error:
            raise CanaryError(f"sealed canary topology is invalid: {error}") from error
        if (
            topology["worker_count"] != declared_lanes
            or topology["job_count"] != declared_jobs
        ):
            raise CanaryError(
                "executor topology differs from the sealed v2/v3 game contract"
            )
        bound_jobs = {
            str(job["job_id"]): job for job in bound_lock["fleet"]["jobs"]
        }
        if len(bound_jobs) != declared_jobs:
            raise CanaryError("sealed canary lock job inventory is not exact")

    commands_by_job: dict[str, dict[str, Any]] = {}
    outputs: set[str] = set()
    seed_ranges: list[tuple[int, int, str]] = []
    lanes: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for command in commands:
        if not isinstance(command, dict):
            raise CanaryError("render command must be an object")
        job_id = str(command.get("job_id", ""))
        if not job_id or job_id in commands_by_job:
            raise CanaryError(f"blank/duplicate job id {job_id!r}")
        if bound_jobs is not None and job_id not in bound_jobs:
            raise CanaryError(f"render contains unsealed job id {job_id!r}")
        commands_by_job[job_id] = command
        argv = command.get("argv")
        if command.get("argv_sha256") != contract._digest_value(argv):
            raise CanaryError(f"{job_id}: argv digest mismatch")
        values, switches = _flag_map(argv, job_id=job_id)
        for flag, expected in EXPECTED_VALUE_FLAGS.items():
            if values.get(flag) != expected:
                raise CanaryError(f"{job_id}: {flag} must be exactly {expected}")
        missing = REQUIRED_SWITCHES - switches
        forbidden = FORBIDDEN_FLAGS & (set(values) | switches)
        if missing:
            raise CanaryError(f"{job_id}: missing required switches {sorted(missing)}")
        if forbidden:
            raise CanaryError(f"{job_id}: forbidden flags {sorted(forbidden)}")

        alias = str(command.get("host_alias", ""))
        gpu = command.get("gpu")
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
            raise CanaryError(f"{job_id}: physical GPU must be a non-negative integer")
        environment = command.get("environment")
        if not isinstance(environment, dict) or environment.get(
            "CUDA_VISIBLE_DEVICES"
        ) != str(gpu):
            raise CanaryError(f"{job_id}: CUDA_VISIBLE_DEVICES does not pin its physical GPU")

        out_dir = values.get("--out-dir")
        if not out_dir or out_dir in outputs:
            raise CanaryError(f"{job_id}: output directory is blank or shared")
        outputs.add(out_dir)
        try:
            start = int(values["--base-seed"])
            attempts = int(values["--games"])
        except (KeyError, ValueError) as error:
            raise CanaryError(f"{job_id}: invalid base seed/games") from error
        if start < 0 or attempts <= 0:
            raise CanaryError(f"{job_id}: invalid seed interval")
        seed_ranges.append((start, start + attempts, job_id))
        lanes.setdefault((alias, gpu), []).append(command)

    if bound_jobs is not None and set(commands_by_job) != set(bound_jobs):
        raise CanaryError("render does not exactly cover the sealed job inventory")

    seed_ranges.sort()
    for previous, current in zip(seed_ranges, seed_ranges[1:]):
        if current[0] < previous[1]:
            raise CanaryError(
                f"seed overlap: {previous[2]} [{previous[0]},{previous[1]}) and "
                f"{current[2]} [{current[0]},{current[1]})"
            )

    plan_lanes = plan.get("lanes")
    if not isinstance(plan_lanes, list) or len(plan_lanes) != declared_lanes:
        raise CanaryError(
            f"executor plan lanes must be an exact {declared_lanes}-item list"
        )
    plan_by_placement: dict[tuple[str, int], dict[str, Any]] = {}
    for lane in plan_lanes:
        if not isinstance(lane, dict):
            raise CanaryError("executor lane must be an object")
        placement = (str(lane.get("host_alias", "")), lane.get("gpu"))
        if placement in plan_by_placement:
            raise CanaryError(f"executor repeats physical lane {placement}")
        plan_by_placement[placement] = lane
    if set(plan_by_placement) != set(lanes):
        raise CanaryError("executor lanes do not exactly cover rendered physical lanes")
    if len(lanes) != declared_lanes:
        raise CanaryError(
            f"render must contain exactly {declared_lanes} independent physical lanes"
        )
    if bound_jobs is not None:
        sealed_placements = {
            (str(job["host_alias"]), int(job["gpu"])) for job in bound_jobs.values()
        }
        if set(lanes) != sealed_placements:
            raise CanaryError("render physical lanes differ from the sealed lock")

    for placement, lane_commands in lanes.items():
        ordered = sorted(lane_commands, key=lambda item: CATEGORY_ORDER.index(item["category"]))
        if tuple(item.get("category") for item in ordered) != CATEGORY_ORDER:
            raise CanaryError(f"lane {placement} does not contain the three exact categories")
        job_ids = [str(item["job_id"]) for item in ordered]
        if plan_by_placement[placement].get("jobs") != job_ids:
            raise CanaryError(f"lane {placement} executor job order drift")
        for index, item in enumerate(ordered):
            dependency = [] if index == 0 else [job_ids[index - 1]]
            if item.get("must_run_after") != dependency:
                raise CanaryError(f"lane {placement} category dependency drift")

    cohorts: dict[str, Any] = {}
    for alias, gpu_count in host_shapes.items():
        actual = sorted(gpu for host, gpu in lanes if host == alias)
        expected = list(range(gpu_count))
        if actual != expected:
            raise CanaryError(
                f"host {alias!r} must expose exact GPU lanes {expected}, got {actual}"
            )
        cohorts[alias] = {
            "gpu_count": gpu_count,
            "gpus": actual,
            "lane_count": gpu_count,
            "job_count": gpu_count * len(CATEGORY_ORDER),
        }

    return {
        "status": "pass",
        "contract_sha256": rendered["contract_sha256"],
        "render_sha256": rendered["render_sha256"],
        "recipe": {
            "n_full": 128,
            "n_fast": 16,
            "p_full": 0.25,
            "public_observation": True,
            "information_set_search": True,
            "determinization_particles": 4,
            "determinization_min_simulations": 32,
            "symmetry_averaged_eval": True,
            "symmetry_averaged_eval_threshold": 20,
            "adaptive_wide_budget": False,
            "workers_per_gpu": 16,
        },
        "mps_client_environment": EXPECTED_MPS_ENVIRONMENT,
        "lane_count": len(lanes),
        "job_count": len(commands_by_job),
        "unique_output_count": len(outputs),
        "disjoint_seed_range_count": len(seed_ranges),
        "cohorts": cohorts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--host-shape",
        required=True,
        action="append",
        metavar="ALIAS=GPU_COUNT",
        help="repeat exactly twice: one 4-GPU host and one 8-GPU host",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_exact_canary(
            _load(args.render), _load(args.plan), _parse_shapes(args.host_shape)
        )
    except CanaryError as error:
        print(f"REFUSING: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
