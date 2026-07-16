from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import modal_gumbel_factory_launch_guard as guard  # noqa: E402


FACTORY = TOOLS / "modal_gumbel_factory_gpu.py"
GENERATION_CONFIG = (
    REPO / "configs/generation/coherent_public_n128.schema19.json"
)
PRODUCTION_RUNTIME = REPO / "configs/runtime/a1_production_runtime.json"


def _binding(
    tmp_path: Path,
    *,
    wheel_bytes: bytes = b"legacy-wheel",
    containers: int = 44,
    games_per_container: int = 500,
) -> dict:
    wheel = (
        tmp_path
        / "catanatron_rs-0.1.2-cp311-cp311-manylinux_2_34_x86_64.whl"
    )
    wheel.write_bytes(wheel_bytes)
    return guard.build_launch_binding(
        factory_source=FACTORY,
        wheel_path=wheel,
        accelerator="L4",
        canonical_generation_config=GENERATION_CONFIG,
        production_runtime_config=PRODUCTION_RUNTIME,
        launch_science={
            "checkpoint_rel": "checkpoints/v3a_masked/checkpoint.pt",
            "device": "cuda",
            "public_observation": True,
            "n_full": 64,
            "n_fast": 16,
            "p_full": 0.25,
            "c_visit": 50.0,
            "c_scale": 0.03,
            "max_decisions": 600,
            "max_depth": 80,
            "prior_temperature": 1.0,
            "value_scale": 1.0,
            "track": "2p_no_trade",
            "vps_to_win": 10,
            "obs_width": 806,
            "shard_size": 2048,
            "fmt": "npz_zst",
        },
        containers=containers,
        games_per_container=games_per_container,
    )


def test_guard_exposes_stale_runtime_science_and_mass_launch(tmp_path: Path) -> None:
    binding = _binding(tmp_path)

    assert binding["legacy_runtime"]["accelerator"] == "L4"
    assert (
        binding["legacy_runtime"]["native_wheel_filename"]
        != binding["current_authority"]["native_wheel_filename"]
    )
    assert binding["science_drift_from_current_canonical"]["n_full"] == {
        "factory": 64,
        "canonical": 128,
    }
    assert binding["science_drift_from_current_canonical"]["c_scale"] == {
        "factory": 0.03,
        "canonical": 0.1,
    }
    assert "coherent_public_belief_search" in binding[
        "missing_critical_current_science_fields"
    ]
    assert "native_mcts_hot_loop" in binding[
        "missing_critical_current_science_fields"
    ]
    assert binding["launch_size"]["games_target"] == 22_000


def test_guard_requires_exact_binding_acknowledgement(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    required = guard.required_acknowledgement(binding)

    with pytest.raises(ValueError, match="not the current canonical generation path"):
        guard.require_acknowledgement(binding, "")
    with pytest.raises(ValueError, match="not the current canonical generation path"):
        guard.require_acknowledgement(binding, required + "-wrong")

    assert guard.require_acknowledgement(binding, required) == required


def test_binding_changes_with_wheel_bytes_or_launch_size(tmp_path: Path) -> None:
    baseline = _binding(tmp_path, wheel_bytes=b"wheel-a", containers=44)
    changed_wheel = _binding(tmp_path, wheel_bytes=b"wheel-b", containers=44)
    changed_size = _binding(tmp_path, wheel_bytes=b"wheel-a", containers=43)

    assert baseline["binding_sha256"] != changed_wheel["binding_sha256"]
    assert baseline["binding_sha256"] != changed_size["binding_sha256"]


def test_mass_launcher_has_no_implicit_container_or_game_count_defaults() -> None:
    module = ast.parse(FACTORY.read_text(encoding="utf-8"))
    launch = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "launch_gpu_gen"
    )
    positional = launch.args.args
    defaults = [None] * (len(positional) - len(launch.args.defaults)) + list(
        launch.args.defaults
    )
    defaults_by_name = {
        argument.arg: default for argument, default in zip(positional, defaults)
    }

    assert defaults_by_name["containers"] is None
    assert defaults_by_name["games_per_container"] is None


def test_factory_binds_acknowledgement_into_payload_resume_and_manifest() -> None:
    source = FACTORY.read_text(encoding="utf-8")

    assert source.count("_bind_and_acknowledge_payloads(") == 3
    assert 'payload["factory_launch_binding"] = binding' in source
    assert '"factory_launch_binding": payload["factory_launch_binding"]' in source
    assert "science_payload = {" in source
