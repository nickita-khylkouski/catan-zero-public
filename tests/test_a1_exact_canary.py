from __future__ import annotations

import copy

import pytest

from tools import a1_pre_wave_contract as contract
from tools.fleet import a1_exact_canary as canary
from tools.fleet import a1_lane_supervisor as supervisor
from tools.fleet import a1_production_executor as executor


def _command(alias: str, gpu: int, category: str, seed: int) -> dict:
    worker = f"{alias}_gpu{gpu}"
    job = f"{worker}__{category}"
    previous = {
        "current_producer": None,
        "recent_history": f"{worker}__current_producer",
        "hard_negative": f"{worker}__recent_history",
    }[category]
    argv = [
        "tools/generate_gumbel_selfplay_data.py",
        "--out-dir", f"/wave/{job}",
        "--games", "10",
        "--workers", "16",
        "--checkpoint", "/models/champion.pt",
        "--device", "cuda",
        "--n-full", "128",
        "--n-fast", "16",
        "--p-full", "0.25",
        "--c-visit", "50.0",
        "--c-scale", "0.1",
        "--max-depth", "80",
        "--rescale-noise-floor-c", "0.0",
        "--sigma-eval", "0.98",
        "--base-seed", str(seed),
        "--symmetry-averaged-eval",
        "--symmetry-averaged-eval-threshold", "20",
        "--public-observation",
        "--information-set-search",
        "--determinization-particles", "4",
        "--determinization-min-simulations", "32",
        "--lazy-interior-chance",
        "--no-belief-chance-spectra",
        "--no-wide-roots-always-full",
        "--no-eval-server",
        "--seed-claim",
        "--resume",
    ]
    return {
        "job_id": job,
        "worker_id": worker,
        "host_alias": alias,
        "gpu": gpu,
        "category": category,
        "environment": {"CUDA_VISIBLE_DEVICES": str(gpu)},
        "argv": argv,
        "argv_sha256": contract._digest_value(argv),
        "must_run_after": [] if previous is None else [previous],
    }


def _fixture(*, current_v3: bool = False) -> tuple[dict, dict]:
    commands = []
    seed = 300_000_000_000
    shapes = {
        **{f"c{i}": 4 for i in range(1, 9 if current_v3 else 7)},
        **{
            f"h100-8{letter}": 8
            for letter in ("a", "b", "c", "d")[: 4 if current_v3 else 2]
        },
    }
    for alias, count in shapes.items():
        for gpu in range(count):
            for category in canary.CATEGORY_ORDER:
                commands.append(_command(alias, gpu, category, seed))
                seed += 10
    lane_count = sum(shapes.values())
    assert lane_count == (64 if current_v3 else 40)
    assert len(commands) == lane_count * len(canary.CATEGORY_ORDER)
    rendered = {
        "schema_version": contract.RENDER_SCHEMA,
        "contract_sha256": "sha256:" + "a" * 64,
        "commands": commands,
    }
    rendered["render_sha256"] = contract._digest_value(rendered)
    lanes = []
    for alias, count in shapes.items():
        for gpu in range(count):
            lane_commands = [
                command
                for command in commands
                if command["host_alias"] == alias and command["gpu"] == gpu
            ]
            lanes.append(
                {
                    "worker_id": lane_commands[0]["worker_id"],
                    "host_alias": alias,
                    "gpu": gpu,
                    "jobs": [command["job_id"] for command in lane_commands],
                }
            )
    plan = {
        "contract_sha256": rendered["contract_sha256"],
        "render_sha256": rendered["render_sha256"],
        "client_environment": dict(canary.EXPECTED_MPS_ENVIRONMENT),
        "lane_count": lane_count,
        "job_count": len(commands),
        "lanes": lanes,
    }
    plan["plan_sha256"] = contract._digest_value(plan)
    return rendered, plan


def _rehash(rendered: dict, plan: dict) -> None:
    rendered.pop("render_sha256", None)
    rendered["render_sha256"] = contract._digest_value(rendered)
    plan["render_sha256"] = rendered["render_sha256"]
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = contract._digest_value(plan)


def test_exact_canary_proves_four_and_eight_gpu_shapes() -> None:
    rendered, plan = _fixture()
    report = canary.validate_exact_canary(
        rendered, plan, {"c1": 4, "h100-8a": 8}
    )

    assert report["status"] == "pass"
    assert report["recipe"] == {
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
    }
    assert report["cohorts"]["c1"]["job_count"] == 12
    assert report["cohorts"]["h100-8a"]["job_count"] == 24
    assert report["lane_count"] == 40
    assert report["job_count"] == 120
    assert report["unique_output_count"] == 120
    assert report["disjoint_seed_range_count"] == 120


def test_exact_canary_accepts_current_v3_64_lane_192_job_topology() -> None:
    rendered, plan = _fixture(current_v3=True)
    report = canary.validate_exact_canary(
        rendered, plan, {"c1": 4, "h100-8a": 8}
    )

    assert report["lane_count"] == 64
    assert report["job_count"] == 192
    assert report["unique_output_count"] == 192
    assert report["disjoint_seed_range_count"] == 192


def test_canary_mps_contract_matches_executor_and_lane_supervisor() -> None:
    assert executor.CLIENT_ENVIRONMENT == canary.EXPECTED_MPS_ENVIRONMENT
    assert supervisor.CLIENT_ENVIRONMENT == canary.EXPECTED_MPS_ENVIRONMENT


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda command: command["argv"].__setitem__(command["argv"].index("128"), "64"), "--n-full"),
        (lambda command: command["argv"].extend(["--n-full-wide", "256"]), "forbidden"),
        (lambda command: command["argv"].remove("--symmetry-averaged-eval"), "missing required"),
        (lambda command: command["argv"].remove("--information-set-search"), "missing required"),
        (lambda command: command["environment"].__setitem__("CUDA_VISIBLE_DEVICES", "7"), "CUDA_VISIBLE"),
    ],
)
def test_exact_canary_rejects_recipe_or_placement_drift(mutation, message: str) -> None:
    rendered, plan = _fixture()
    mutation(rendered["commands"][0])
    rendered["commands"][0]["argv_sha256"] = contract._digest_value(
        rendered["commands"][0]["argv"]
    )
    _rehash(rendered, plan)

    with pytest.raises(canary.CanaryError, match=message):
        canary.validate_exact_canary(rendered, plan, {"c1": 4, "h100-8a": 8})


def test_exact_canary_rejects_missing_mps_binding() -> None:
    rendered, plan = _fixture()
    plan["client_environment"].pop("CUDA_MPS_PIPE_DIRECTORY")
    plan.pop("plan_sha256")
    plan["plan_sha256"] = contract._digest_value(plan)

    with pytest.raises(canary.CanaryError, match="MPS client environment"):
        canary.validate_exact_canary(rendered, plan, {"c1": 4, "h100-8a": 8})


def test_exact_canary_rejects_shared_output_or_seed() -> None:
    rendered, plan = _fixture()
    first, second = rendered["commands"][:2]
    first_values, _ = canary._flag_map(first["argv"], job_id=first["job_id"])
    second["argv"][second["argv"].index("--out-dir") + 1] = first_values["--out-dir"]
    second["argv_sha256"] = contract._digest_value(second["argv"])
    _rehash(rendered, plan)

    with pytest.raises(canary.CanaryError, match="output directory"):
        canary.validate_exact_canary(rendered, plan, {"c1": 4, "h100-8a": 8})


def test_exact_canary_rejects_shape_or_executor_lane_drift() -> None:
    rendered, plan = _fixture()
    broken = copy.deepcopy(plan)
    broken["lanes"][0]["jobs"].reverse()
    broken.pop("plan_sha256")
    broken["plan_sha256"] = contract._digest_value(broken)
    with pytest.raises(canary.CanaryError, match="job order drift"):
        canary.validate_exact_canary(rendered, broken, {"c1": 4, "h100-8a": 8})

    with pytest.raises(canary.CanaryError, match="exact GPU lanes"):
        canary.validate_exact_canary(rendered, plan, {"c1": 8, "h100-8a": 4})


def test_exact_canary_rejects_declared_topology_count_drift() -> None:
    rendered, plan = _fixture(current_v3=True)
    plan["lane_count"] = 40
    plan["job_count"] = 120
    plan.pop("plan_sha256")
    plan["plan_sha256"] = contract._digest_value(plan)

    with pytest.raises(canary.CanaryError, match="three jobs per lane"):
        canary.validate_exact_canary(rendered, plan, {"c1": 4, "h100-8a": 8})
