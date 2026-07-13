from __future__ import annotations

import copy
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from tools.fleet import a1_h100_eval_fleet as fleet
from tools.champion_registry import ChampionRegistry


def test_future_plan_default_uses_validated_16_worker_packing() -> None:
    assert fleet.DEFAULT_WORKERS_PER_GPU == 16
    assert (
        inspect.signature(fleet.build_plan).parameters["workers_per_gpu"].default
        == fleet.DEFAULT_WORKERS_PER_GPU
    )
    args = fleet._parser().parse_args(
        [
            "--manifest",
            "fleet.json",
            "plan",
            "--candidate",
            "candidate.pt",
            "--champion",
            "champion.pt",
            "--candidate-parent",
            "champion.pt",
            "--registry",
            "registry.json",
            "--internal-base-seed",
            "1",
            "--external-base-seed",
            "2",
                "--iteration-id",
                "packing-default",
                "--candidate-c-scale",
                "0.10",
                "--champion-c-scale",
                "0.03",
                "--out",
            "plan.json",
        ]
    )
    assert args.workers_per_gpu == fleet.DEFAULT_WORKERS_PER_GPU


def test_remote_transport_retries_transient_failure_but_local_commands_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if "false" in argv or (argv and argv[0] == "git"):
            raise subprocess.CalledProcessError(1, argv, stderr="command refused")
        if len(calls) < 3:
            raise subprocess.CalledProcessError(255, argv, stderr="connection reset")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(fleet.subprocess, "run", fake_run)
    monkeypatch.setattr(fleet.time, "sleep", lambda _seconds: None)
    assert fleet._run(["ssh", "host", "true"]).stdout == "ok"  # noqa: SLF001
    assert len(calls) == 3

    calls.clear()
    with pytest.raises(subprocess.CalledProcessError):
        fleet._run(["ssh", "host", "false"])  # noqa: SLF001
    assert len(calls) == 1

    calls.clear()
    with pytest.raises(subprocess.CalledProcessError):
        fleet._run(["git", "status"])  # noqa: SLF001
    assert len(calls) == 1


def _manifest_file(
    tmp_path: Path, *, shapes: dict[str, int] | None = None
) -> Path:
    selected_shapes = fleet.EXPECTED_SHAPES if shapes is None else shapes
    hosts = [
        {
            "alias": alias,
            "address": fleet.FULL_EXPECTED_HOSTS[alias][0],
            "gpu_count": count,
        }
        for alias, count in selected_shapes.items()
    ]
    value = {
        "schema_version": fleet.MANIFEST_SCHEMA,
        "ssh_user": "ubuntu",
        "ssh_key": str(tmp_path / "id_ed25519"),
        "strict_host_key_checking": "accept-new",
        "remote_repo": "/home/ubuntu/catan-zero-v1",
        "remote_python": "/home/ubuntu/catan-zero-v1/.venv/bin/python",
        "remote_root": "/home/ubuntu/a1-evaluation",
        "validation_ledger": str(tmp_path / "VAL_ONLY_EVAL_LEDGER.jsonl"),
        "ray_head_address": "10.0.0.2",
        "hosts": hosts,
    }
    path = tmp_path / "fleet.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _plan(
    tmp_path: Path, *, shapes: dict[str, int] | None = None
) -> tuple[dict, dict]:
    manifest = fleet.load_manifest(_manifest_file(tmp_path, shapes=shapes))
    candidate = tmp_path / "candidate.pt"
    champion = tmp_path / "champion.pt"
    candidate.write_bytes(b"candidate")
    champion.write_bytes(b"champion")
    registry = _registry(tmp_path, champion)
    plan = fleet.build_plan(
        manifest,
        candidate=candidate,
        champion=champion,
        candidate_parent=champion,
        registry=registry,
        internal_pairs=600,
        external_pairs=500,
        internal_base_seed=6_190_000_000,
        external_base_seed=6_191_000_000,
        workers_per_gpu=8,
        repo_commit="a" * 40,
        tool_hashes={
            "tools/gumbel_search_cross_net_h2h.py": "sha256:" + "1" * 64,
            "tools/catanatron_neutral_harness_match.py": "sha256:" + "2" * 64,
            "tools/fleet/launch_detached.sh": "sha256:" + "3" * 64,
        },
    )
    return manifest, plan


def test_remote_preflight_refuses_untracked_runtime_code(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    host = manifest["hosts"][0]

    command = fleet._preflight_command(manifest, plan, host)  # noqa: SLF001

    assert 'test -z "$(git status --porcelain=v1 --untracked-files=all)"' in command


def _registry(tmp_path: Path, champion: Path, *, c_scale: float = 0.03) -> ChampionRegistry:
    registry = ChampionRegistry(tmp_path / "champion_registry.json")
    search_config = {"c_scale": c_scale}
    checkpoint = fleet._checkpoint_ref(champion)  # noqa: SLF001
    identity = fleet._digest(  # noqa: SLF001
        {
            "schema_version": "a1-deployed-agent-search-config-v1",
            "checkpoint": checkpoint,
            "search_config": search_config,
        }
    )
    registry.set_role(
        "generator_champion",
        champion,
        version=4,
        provenance={
            "a1_candidate_agent_identity_sha256": identity,
            "a1_candidate_search_config": search_config,
        },
        reason="test",
    )
    registry.save()
    return registry


def _binding_kwargs(plan: dict) -> dict:
    binding = plan["evaluation_binding"]
    return {
        "candidate_parent": Path(binding["candidate_parent"]["path"]),
        "registry": ChampionRegistry.load(binding["registry"]["path"]),
    }


def test_manifest_requires_an_exact_approved_fleet_shape(
    tmp_path: Path,
) -> None:
    path = _manifest_file(tmp_path)
    value = json.loads(path.read_text())
    value["hosts"][-1]["gpu_count"] = 4
    path.write_text(json.dumps(value))
    with pytest.raises(fleet.FleetError, match="exact approved fleet mapping"):
        fleet.load_manifest(path)


def test_legacy_40_gpu_manifest_remains_hash_identical(tmp_path: Path) -> None:
    # The default loader's backward-compatible topology dispatch must not alter
    # the normalized payload or manifest hash of any existing sealed plan.
    path = _manifest_file(tmp_path)
    default = fleet.load_manifest(path)
    explicit_legacy = fleet.load_manifest(
        path, expected_shapes=fleet.EXPECTED_SHAPES
    )
    assert default == explicit_legacy
    assert len(fleet.gpu_slots(default)) == 40


def test_expanded_48_gpu_manifest_allocates_every_pair_exactly_once(
    tmp_path: Path,
) -> None:
    manifest, plan = _plan(tmp_path, shapes=fleet.EXPANDED_EXPECTED_SHAPES)
    assert [host["alias"] for host in manifest["hosts"]] == [
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
        "c7",
        "c8",
        "h100-8a",
        "h100-8b",
    ]
    internal = [job for job in plan["jobs"] if job["phase"] == "internal"]
    assert len(internal) == 48
    assert sum(job["pairs"] for job in internal) == 600
    assert [job["pairs"] for job in internal].count(13) == 24
    assert [job["pairs"] for job in internal].count(12) == 24
    intervals = sorted(
        (job["base_seed"], job["base_seed"] + job["pairs"])
        for job in internal
    )
    assert intervals[0][0] == 6_190_000_000
    assert intervals[-1][1] == 6_190_000_600
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))


def test_full_64_gpu_manifest_allocates_every_pair_exactly_once(
    tmp_path: Path,
) -> None:
    manifest, plan = _plan(tmp_path, shapes=fleet.FULL_EXPECTED_SHAPES)
    assert [host["alias"] for host in manifest["hosts"]] == [
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
        "c7",
        "c8",
        "h100-8a",
        "h100-8b",
        "h100-8c",
        "h100-8d",
    ]
    internal = [job for job in plan["jobs"] if job["phase"] == "internal"]
    assert len(internal) == 64
    assert sum(job["pairs"] for job in internal) == 600
    assert [job["pairs"] for job in internal].count(10) == 24
    assert [job["pairs"] for job in internal].count(9) == 40
    intervals = sorted(
        (job["base_seed"], job["base_seed"] + job["pairs"])
        for job in internal
    )
    assert intervals[0][0] == 6_190_000_000
    assert intervals[-1][1] == 6_190_000_600
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))


def test_manifest_rejects_address_substitution(tmp_path: Path) -> None:
    value = json.loads(_manifest_file(tmp_path).read_text())
    value["hosts"][0]["address"] = "203.0.113.99"
    path = tmp_path / "substituted-address.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(fleet.FleetError, match="exact approved fleet mapping"):
        fleet.load_manifest(path)


@pytest.mark.parametrize(
    "bad_aliases",
    [
        ("c7",),
        ("h100-8c", "h100-8d"),
        ("c7", "c8", "h100-8c"),
    ],
)
def test_partial_or_forbidden_expanded_topology_is_rejected(
    tmp_path: Path, bad_aliases: tuple[str, ...]
) -> None:
    value = json.loads(_manifest_file(tmp_path).read_text())
    for index, alias in enumerate(bad_aliases):
        value["hosts"].append(
            {"alias": alias, "address": f"10.9.0.{index + 1}", "gpu_count": 4}
        )
    path = tmp_path / "bad-expanded.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(fleet.FleetError, match="exact approved fleet mapping"):
        fleet.load_manifest(path)


def test_internal_plan_weights_by_physical_gpu_and_conserves_seed_interval(
    tmp_path: Path,
) -> None:
    _manifest, plan = _plan(tmp_path)
    jobs = [job for job in plan["jobs"] if job["phase"] == "internal"]
    assert len(jobs) == 40
    assert sum(job["pairs"] for job in jobs) == 600
    assert {job["pairs"] for job in jobs} == {15}
    by_host = {
        alias: sum(job["pairs"] for job in jobs if job["alias"] == alias)
        for alias in fleet.EXPECTED_SHAPES
    }
    assert by_host["c1"] == 60
    assert by_host["h100-8a"] == 120
    intervals = sorted(
        (job["base_seed"], job["base_seed"] + job["pairs"]) for job in jobs
    )
    assert intervals[0][0] == 6_190_000_000
    assert intervals[-1][1] == 6_190_000_600
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))


def test_full_plan_can_seal_an_approved_host_subset(tmp_path: Path) -> None:
    manifest, template = _plan(tmp_path)
    aliases = ["c2", "c3", "c4", "c5", "c6", "h100-8b"]
    plan = fleet.build_plan(
        manifest,
        candidate=Path(template["candidate"]["source"]),
        champion=Path(template["champion"]["source"]),
        **_binding_kwargs(template),
        internal_pairs=280,
        external_pairs=14,
        internal_base_seed=6_195_600_000,
        external_base_seed=6_195_601_000,
        workers_per_gpu=16,
        scope="full",
        host_aliases=aliases,
        repo_commit="a" * 40,
        tool_hashes=template["tool_hashes"],
    )

    assert plan["host_aliases"] == aliases
    internal = [job for job in plan["jobs"] if job["phase"] == "internal"]
    assert len(internal) == 28
    assert sum(job["pairs"] for job in internal) == 280
    assert {job["pairs"] for job in internal} == {10}
    assert {job["alias"] for job in internal} == set(aliases)
    fleet._validate_planned_jobs(plan, manifest)  # noqa: SLF001

    drifted = copy.deepcopy(plan)
    drifted["host_aliases"] = list(reversed(aliases))
    with pytest.raises(fleet.FleetError, match="unknown or out of order"):
        fleet._validate_planned_jobs(drifted, manifest)  # noqa: SLF001


def test_external_plan_uses_matched_candidate_champion_cohorts(tmp_path: Path) -> None:
    _manifest, plan = _plan(tmp_path)
    jobs = [job for job in plan["jobs"] if job["phase"] == "external"]
    assert len(jobs) == 40
    cohorts: dict[str, list[dict]] = {}
    for job in jobs:
        cohorts.setdefault(job["cohort_id"], []).append(job)
    assert len(cohorts) == 20
    for cohort in cohorts.values():
        assert {job["role"] for job in cohort} == {"candidate", "champion"}
        assert len({(job["base_seed"], job["pairs"]) for job in cohort}) == 1
        assert {job["pairs"] for job in cohort} == {25}
        assert len({job["slot_id"] for job in cohort}) == 2
        for job in cohort:
            argv = job["argv"]
            assert argv[argv.index("--engine-repo-commit") + 1] == plan["repo_commit"]
            assert (
                argv[argv.index("--native-wheel-sha256") + 1]
                == plan["engine_identity"]["native_wheel_sha256"]
            )
            assert (
                argv[argv.index("--python-referee-sha256") + 1]
                == plan["engine_identity"]["python_referee_sha256"]
            )
    by_role = {
        role: sorted(
            (job["base_seed"], job["base_seed"] + job["pairs"])
            for job in jobs
            if job["role"] == role
        )
        for role in ("candidate", "champion")
    }
    assert by_role["candidate"] == by_role["champion"]
    assert by_role["candidate"][0][0] == 6_191_000_000
    assert by_role["candidate"][-1][1] == 6_191_000_500


def test_n256_initialized_from_f7_cannot_default_to_gen3_baseline(
    tmp_path: Path,
) -> None:
    manifest = fleet.load_manifest(_manifest_file(tmp_path))
    candidate = tmp_path / "n256-candidate.pt"
    f7_parent = tmp_path / "f7.pt"
    gen3 = tmp_path / "gen3.pt"
    candidate.write_bytes(b"n256")
    f7_parent.write_bytes(b"f7")
    gen3.write_bytes(b"gen3")
    registry = _registry(tmp_path, f7_parent, c_scale=0.10)
    common = {
        "candidate": candidate,
        "champion": gen3,
        "candidate_parent": f7_parent,
        "registry": registry,
        "internal_pairs": 600,
        "external_pairs": 500,
        "internal_base_seed": 6_190_000_000,
        "external_base_seed": 6_191_000_000,
        "candidate_c_scale": 0.10,
        "champion_c_scale": 0.03,
        "repo_commit": "a" * 40,
        "tool_hashes": {},
    }
    with pytest.raises(
        fleet.FleetError,
        match="promotion baseline differs from candidate parent/init checkpoint",
    ):
        fleet.build_plan(manifest, **common)

    historical = fleet.build_plan(
        manifest,
        **common,
        comparison_mode="historical_comparison",
        historical_comparison_reason="diagnose n256 against archived gen3 only",
    )
    assert historical["evaluation_binding"]["promotion_eligible"] is False
    assert historical["evaluation_binding"]["candidate_parent"]["sha256"] == fleet._sha256(  # noqa: SLF001
        f7_parent
    )
    assert historical["role_search_config"] == {
        "candidate": {"c_scale": 0.10, "value_squash": "tanh"},
        "champion": {"c_scale": 0.03, "value_squash": "tanh"},
    }


def test_canary_scope_uses_every_gpu_on_one_four_and_one_eight_gpu_host(
    tmp_path: Path,
) -> None:
    manifest, full = _plan(tmp_path)
    canary = fleet.build_plan(
        manifest,
        candidate=Path(full["candidate"]["source"]),
        champion=Path(full["champion"]["source"]),
        **_binding_kwargs(full),
        internal_pairs=24,
        external_pairs=12,
        internal_base_seed=6_192_000_000,
        external_base_seed=6_192_001_000,
        workers_per_gpu=2,
        iteration_id="a1-canary",
        scope="canary",
        repo_commit="a" * 40,
        tool_hashes=full["tool_hashes"],
    )
    assert canary["scope"] == "canary"
    assert len(canary["jobs"]) == 24
    for phase in ("internal", "external"):
        jobs = [job for job in canary["jobs"] if job["phase"] == phase]
        assert len(jobs) == 12
        assert {job["alias"] for job in jobs} == {"c1", "h100-8a"}
        assert {(job["alias"], job["gpu"]) for job in jobs} == {
            *(("c1", gpu) for gpu in range(4)),
            *(("h100-8a", gpu) for gpu in range(8)),
        }


def test_every_job_is_cuda_pinned_and_has_exact_n128_infoset_d6_recipe(
    tmp_path: Path,
) -> None:
    manifest, plan = _plan(tmp_path)
    required = {
        "--n-full": "128",
        "--c-visit": "50.0",
        "--sigma-eval": "0.98",
        "--determinization-particles": "4",
        "--determinization-min-simulations": "32",
        "--symmetry-averaged-eval-threshold": "20",
        "--value-readout": "scalar",
        "--gate-config": "flywheel",
    }
    for job in plan["jobs"]:
        argv = job["argv"]
        for flag, expected in required.items():
            assert argv[argv.index(flag) + 1] == expected
        if job["phase"] == "internal":
            assert argv[argv.index("--candidate-c-scale") + 1] == "0.03"
            assert argv[argv.index("--baseline-c-scale") + 1] == "0.03"
            assert argv[argv.index("--candidate-value-squash") + 1] == "tanh"
            assert argv[argv.index("--baseline-value-squash") + 1] == "tanh"
            assert "--c-scale" not in argv
        else:
            assert argv[argv.index("--c-scale") + 1] == "0.03"
        for flag in (
            "--lazy-interior-chance",
            "--correct-rust-chance-spectra",
            "--public-observation",
            "--information-set-search",
            "--no-belief-chance-spectra",
            "--symmetry-averaged-eval",
            "--evaluator-rust-featurize",
            "--native-mcts-hot-loop",
        ):
            assert argv.count(flag) == 1
        assert "--device" in argv and argv[argv.index("--device") + 1] == "cuda"
        if job["phase"] == "external":
            assert argv[argv.index("--vps-to-win") + 1] == "10"
            assert (
                argv[argv.index("--max-player-trade-offers-per-turn") + 1] == "0"
            )
    rendered = fleet.dry_run_commands(manifest, plan, "internal")
    assert len(rendered["hosts"]) == 8
    all_shell = "\n".join(row["ssh_command"][-1] for row in rendered["hosts"])
    for gpu in range(8):
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in all_shell
    assert "B200" not in all_shell
    assert "query-compute-apps=process_name" in all_shell
    assert "nvidia-cuda-mps-server" in all_shell
    assert "memory.used" in all_shell
    assert "PYTHONPATH=" in all_shell
    assert "/home/ubuntu/catan-zero-v1/src" in all_shell
    assert "catanatron_rs-0.1.8-cp311-cp311-manylinux_2_34_x86_64.whl" in all_shell
    assert "catanatron-rs" in all_shell
    assert "0.1.8" in all_shell
    assert "gumbel_search_capabilities" in all_shell
    assert "sigma_reference_visits" in all_shell
    assert "belief_target_evidence" in all_shell
    assert "initial_road_d1_scope" in all_shell
    assert "public_award_feature_parity" in all_shell
    assert "/home/ubuntu/catan-zero-v1/tools/fleet/launch_detached.sh" in all_shell
    assert "\ntools/fleet/launch_detached.sh " not in all_shell


def test_role_specific_search_calibration_is_sealed_into_jobs_and_identity(
    tmp_path: Path,
) -> None:
    manifest, default = _plan(tmp_path)
    calibrated = fleet.build_plan(
        manifest,
        candidate=Path(default["candidate"]["source"]),
        champion=Path(default["champion"]["source"]),
        **_binding_kwargs(default),
        internal_pairs=600,
        external_pairs=500,
        internal_base_seed=6_190_000_000,
        external_base_seed=6_191_000_000,
        workers_per_gpu=8,
        candidate_c_scale=0.10,
        champion_c_scale=0.03,
        repo_commit="a" * 40,
        tool_hashes=default["tool_hashes"],
    )
    assert calibrated["run_id"] != default["run_id"]
    assert calibrated["science_config_hash"] != default["science_config_hash"]
    assert calibrated["role_search_config"] == {
        "candidate": {"c_scale": 0.10, "value_squash": "tanh"},
        "champion": {"c_scale": 0.03, "value_squash": "tanh"},
    }
    internal = next(job for job in calibrated["jobs"] if job["phase"] == "internal")
    assert internal["argv"][internal["argv"].index("--candidate-c-scale") + 1] == "0.1"
    assert internal["argv"][internal["argv"].index("--baseline-c-scale") + 1] == "0.03"
    for job in (job for job in calibrated["jobs"] if job["phase"] == "external"):
        expected = "0.1" if job["role"] == "candidate" else "0.03"
        assert job["argv"][job["argv"].index("--c-scale") + 1] == expected


def test_role_specific_value_squash_is_sealed_into_plan_and_internal_jobs(
    tmp_path: Path,
) -> None:
    manifest, default = _plan(tmp_path)
    diagnostic = fleet.build_plan(
        manifest,
        candidate=Path(default["candidate"]["source"]),
        champion=Path(default["champion"]["source"]),
        **_binding_kwargs(default),
        internal_pairs=600,
        external_pairs=500,
        internal_base_seed=6_190_000_000,
        external_base_seed=6_191_000_000,
        workers_per_gpu=8,
        candidate_c_scale=0.03,
        champion_c_scale=0.03,
        candidate_value_squash="clip",
        champion_value_squash="tanh",
        repo_commit="a" * 40,
        tool_hashes=default["tool_hashes"],
    )

    assert diagnostic["run_id"] != default["run_id"]
    assert diagnostic["science_config_hash"] != default["science_config_hash"]
    assert diagnostic["role_search_config"] == {
        "candidate": {"c_scale": 0.03, "value_squash": "clip"},
        "champion": {"c_scale": 0.03, "value_squash": "tanh"},
    }
    for job in (job for job in diagnostic["jobs"] if job["phase"] == "internal"):
        argv = job["argv"]
        assert argv[argv.index("--candidate-value-squash") + 1] == "clip"
        assert argv[argv.index("--baseline-value-squash") + 1] == "tanh"
        assert job["command_hash"] == fleet._digest(argv)  # noqa: SLF001


def test_corrected_belief_gameplay_operator_is_sealed_per_role(tmp_path: Path) -> None:
    manifest, default = _plan(tmp_path)
    diagnostic = fleet.build_plan(
        manifest,
        candidate=Path(default["candidate"]["source"]),
        champion=Path(default["champion"]["source"]),
        **_binding_kwargs(default),
        internal_pairs=600,
        external_pairs=500,
        internal_base_seed=6_192_000_000,
        external_base_seed=6_193_000_000,
        workers_per_gpu=8,
        candidate_c_scale=0.03,
        champion_c_scale=0.03,
        candidate_gameplay_policy_aggregation="aggregate_q_then_improve",
        champion_gameplay_policy_aggregation="mean_improved_policy",
        candidate_rescale_noise_floor_c=1.0,
        champion_rescale_noise_floor_c=0.0,
        candidate_sigma_eval=0.98,
        champion_sigma_eval=0.98,
        candidate_sigma_reference_visits=8,
        champion_sigma_reference_visits=None,
        comparison_mode="historical_comparison",
        historical_comparison_reason="same-net corrected belief operator diagnostic",
        repo_commit="a" * 40,
        tool_hashes=default["tool_hashes"],
    )
    roles = diagnostic["role_search_config"]
    assert roles["candidate"]["gameplay_policy_aggregation"] == (
        "aggregate_q_then_improve"
    )
    assert roles["candidate"]["sigma_reference_visits"] == 8
    assert roles["candidate"]["rescale_noise_floor_c"] == 1.0
    assert roles["champion"]["gameplay_policy_aggregation"] == (
        "mean_improved_policy"
    )
    assert roles["champion"]["sigma_reference_visits"] is None
    internal = next(job for job in diagnostic["jobs"] if job["phase"] == "internal")
    argv = internal["argv"]
    assert argv[argv.index("--candidate-gameplay-policy-aggregation") + 1] == (
        "aggregate_q_then_improve"
    )
    assert "--baseline-gameplay-policy-aggregation" in argv
    assert "--candidate-sigma-reference-visits" in argv
    assert "--baseline-sigma-reference-visits" not in argv
    external_candidate = next(
        job
        for job in diagnostic["jobs"]
        if job["phase"] == "external" and job["role"] == "candidate"
    )
    external_champion = next(
        job
        for job in diagnostic["jobs"]
        if job["phase"] == "external" and job["role"] == "champion"
    )
    candidate_argv = external_candidate["argv"]
    champion_argv = external_champion["argv"]
    assert candidate_argv[
        candidate_argv.index("--gameplay-policy-aggregation") + 1
    ] == "aggregate_q_then_improve"
    assert candidate_argv[candidate_argv.index("--sigma-reference-visits") + 1] == "8"
    assert champion_argv[
        champion_argv.index("--gameplay-policy-aggregation") + 1
    ] == "mean_improved_policy"
    assert "--sigma-reference-visits" not in champion_argv
    plan_path = tmp_path / "corrected-plan.json"
    fleet.write_new_readonly(plan_path, diagnostic)
    assert fleet.load_plan(plan_path, manifest)["plan_hash"] == diagnostic["plan_hash"]


def test_same_checkpoint_clip_vs_tanh_plan_is_diagnostic_and_executable(
    tmp_path: Path,
) -> None:
    manifest, default = _plan(tmp_path)
    checkpoint = Path(default["champion"]["source"])
    diagnostic = fleet.build_plan(
        manifest,
        candidate=checkpoint,
        champion=checkpoint,
        candidate_parent=checkpoint,
        registry=ChampionRegistry.load(
            Path(default["evaluation_binding"]["registry"]["path"])
        ),
        internal_pairs=600,
        external_pairs=500,
        internal_base_seed=6_190_000_000,
        external_base_seed=6_191_000_000,
        candidate_c_scale=0.03,
        champion_c_scale=0.03,
        candidate_value_squash="clip",
        champion_value_squash="tanh",
        comparison_mode="historical_comparison",
        historical_comparison_reason="same-checkpoint value transform diagnostic",
        repo_commit="a" * 40,
        tool_hashes=default["tool_hashes"],
    )

    assert diagnostic["candidate"]["sha256"] == diagnostic["champion"]["sha256"]
    assert diagnostic["evaluation_binding"]["promotion_eligible"] is False
    internal = next(job for job in diagnostic["jobs"] if job["phase"] == "internal")
    assert internal["argv"][internal["argv"].index("--candidate-value-squash") + 1] == "clip"
    assert internal["argv"][internal["argv"].index("--baseline-value-squash") + 1] == "tanh"


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_role_specific_search_calibration_rejects_invalid_values(
    tmp_path: Path, value: float
) -> None:
    manifest, default = _plan(tmp_path)
    with pytest.raises(fleet.FleetError, match="candidate_c_scale"):
        fleet.build_plan(
            manifest,
            candidate=Path(default["candidate"]["source"]),
            champion=Path(default["champion"]["source"]),
            **_binding_kwargs(default),
            internal_pairs=600,
            external_pairs=500,
            internal_base_seed=6_190_000_000,
            external_base_seed=6_191_000_000,
            workers_per_gpu=8,
            candidate_c_scale=value,
            repo_commit="a" * 40,
            tool_hashes=default["tool_hashes"],
        )


def test_launch_command_creates_fresh_tree_and_ignores_ssh_working_directory(
    tmp_path: Path,
) -> None:
    remote_repo = tmp_path / "remote repo"
    launcher = remote_repo / "tools" / "fleet" / "launch_detached.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "shift 3\n"
        "test \"$1\" = --\n"
        "shift\n"
        '"$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    remote_root = tmp_path / "remote output"
    job_dir = (
        remote_root / "runs" / "a1-eval-0123456789abcdef" / "internal" / "job-0"
    )
    report = job_dir / "report.json"
    job = {
        "job_id": "job-0",
        "job_dir": str(job_dir),
        "report": str(report),
        "gpu": 0,
        "argv": [
            "python3",
            "-c",
            f"from pathlib import Path; Path({str(report)!r}).write_text('ok')",
        ],
    }
    command = fleet._launch_job_command(  # noqa: SLF001
        {"remote_repo": str(remote_repo), "remote_root": str(remote_root)},
        job,
    )
    unrelated_cwd = tmp_path / "unrelated-ssh-cwd"
    unrelated_cwd.mkdir()
    assert not remote_root.exists()

    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=unrelated_cwd,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert remote_root.is_dir()
    assert (remote_root / "runs" / "a1-eval-0123456789abcdef" / "internal").is_dir()
    assert report.read_text(encoding="utf-8") == "ok"
    assert (job_dir / ".done").is_file()
    assert (job_dir / ".rc").read_text(encoding="utf-8") == "0\n"
    assert not (job_dir / ".failed").exists()


def test_launch_refuses_symlinked_remote_root_before_writing_job(
    tmp_path: Path,
) -> None:
    remote_repo = tmp_path / "repo"
    launcher = remote_repo / "tools" / "fleet" / "launch_detached.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    launcher.chmod(0o755)
    escaped = tmp_path / "recovery-data"
    escaped.mkdir()
    remote_root = tmp_path / "eval-root"
    remote_root.symlink_to(escaped, target_is_directory=True)
    job_dir = remote_root / "runs" / "a1-eval-0123456789abcdef" / "internal" / "job-0"
    job = {
        "job_id": "job-0",
        "job_dir": str(job_dir),
        "report": str(job_dir / "report.json"),
        "gpu": 0,
        "argv": ["true"],
    }

    completed = subprocess.run(
        ["bash", "-lc", fleet._launch_job_command(  # noqa: SLF001
            {"remote_repo": str(remote_repo), "remote_root": str(remote_root)},
            job,
        )],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert list(escaped.iterdir()) == []


def test_checkpoint_staging_creates_both_distinct_remote_parent_dirs(
    tmp_path: Path,
) -> None:
    manifest, plan = _plan(tmp_path)
    plan = copy.deepcopy(plan)
    plan["candidate"]["remote"] = "/srv/a1/candidates/candidate.pt"
    plan["champion"]["remote"] = "/srv/a1/champions/champion.pt"
    commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    fleet._prepare_remote_host(  # noqa: SLF001
        manifest, plan, manifest["hosts"][0], runner=runner
    )

    assert "mkdir -p /srv/a1/candidates /srv/a1/champions" in commands[0][-1]


def test_plan_hash_and_checkpoint_bytes_are_replayed_on_load(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    path = tmp_path / "plan.json"
    fleet.write_new_readonly(path, plan)
    assert fleet.load_plan(path, manifest)["plan_hash"] == plan["plan_hash"]
    assert plan["candidate"]["remote"] == plan["candidate"]["source"]
    assert plan["champion"]["remote"] == plan["champion"]["source"]
    Path(plan["candidate"]["source"]).chmod(0o644)
    Path(plan["candidate"]["source"]).write_bytes(b"changed")
    with pytest.raises(fleet.FleetError, match="candidate checkpoint bytes drifted"):
        fleet.load_plan(path, manifest)


def test_load_plan_rejects_rehashed_internal_role_squash_command_drift(
    tmp_path: Path,
) -> None:
    manifest, plan = _plan(tmp_path)
    drifted = copy.deepcopy(plan)
    job = next(job for job in drifted["jobs"] if job["phase"] == "internal")
    index = job["argv"].index("--candidate-value-squash")
    del job["argv"][index : index + 2]
    job["command_hash"] = fleet._digest(job["argv"])  # noqa: SLF001
    drifted["plan_hash"] = fleet._digest(  # noqa: SLF001
        {key: value for key, value in drifted.items() if key != "plan_hash"}
    )
    path = tmp_path / "role-squash-command-drift.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")

    with pytest.raises(fleet.FleetError, match="internal shard contract drift"):
        fleet.load_plan(path, manifest)


def test_remote_run_identity_binds_code_manifest_and_worker_packing(
    tmp_path: Path,
) -> None:
    manifest, template = _plan(tmp_path)
    candidate = Path(template["candidate"]["source"])
    champion = Path(template["champion"]["source"])

    def rebuild(**overrides: object) -> dict:
        options = {
            "candidate": candidate,
            "champion": champion,
            **_binding_kwargs(template),
            "internal_pairs": 600,
            "external_pairs": 500,
            "internal_base_seed": 6_190_000_000,
            "external_base_seed": 6_191_000_000,
            "workers_per_gpu": 8,
            "repo_commit": "a" * 40,
            "tool_hashes": template["tool_hashes"],
        }
        options.update(overrides)
        return fleet.build_plan(manifest, **options)

    assert rebuild()["run_id"] == template["run_id"]
    assert rebuild(workers_per_gpu=16)["run_id"] != template["run_id"]
    assert rebuild(repo_commit="b" * 40)["run_id"] != template["run_id"]
    changed_tools = dict(template["tool_hashes"])
    changed_tools["tools/gumbel_search_cross_net_h2h.py"] = "sha256:" + "9" * 64
    assert rebuild(tool_hashes=changed_tools)["run_id"] != template["run_id"]

    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["remote_root"] = "/home/ubuntu/a1-evaluation-other"
    changed_manifest["manifest_hash"] = fleet._digest(  # noqa: SLF001
        {
            key: value
            for key, value in changed_manifest.items()
            if key != "manifest_hash"
        }
    )
    assert fleet.build_plan(
        changed_manifest,
        candidate=candidate,
        champion=champion,
        **_binding_kwargs(template),
        internal_pairs=600,
        external_pairs=500,
        internal_base_seed=6_190_000_000,
        external_base_seed=6_191_000_000,
        workers_per_gpu=8,
        repo_commit="a" * 40,
        tool_hashes=template["tool_hashes"],
    )["run_id"] != template["run_id"]


def test_resume_selects_only_missing_failed_or_stale_jobs(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    jobs = [job for job in plan["jobs"] if job["phase"] == "internal"]
    status = {
        "jobs": [
            {
                "job_id": job["job_id"],
                "state": ("done", "active", "failed", "stale", "missing")[index % 5],
            }
            for index, job in enumerate(jobs)
        ]
    }
    selected = fleet.jobs_to_resume(plan, status, "internal")
    expected = {
        row["job_id"]
        for row in status["jobs"]
        if row["state"] in {"failed", "stale", "missing"}
    }
    assert selected == expected
    rendered = fleet.dry_run_commands(
        manifest, plan, "internal", selected_job_ids=selected
    )
    assert sum(row["jobs"] for row in rendered["hosts"]) == len(selected)


def test_ray_spec_advertises_no_b200_gpu_and_all_40_h100_slots(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    spec = fleet.ray_cluster_spec(manifest, plan)
    assert spec["head"]["num_gpus"] == 0
    assert sum(worker["num_gpus"] for worker in spec["workers"]) == 40
    assert spec["scheduler_contract"]["actor_resources"] == {
        "num_gpus": 1,
        "resources": {"H100": 1},
    }
    eight = next(worker for worker in spec["workers"] if worker["alias"] == "h100-8a")
    assert eight["resources"] == {"H100": 8}


def test_plan_rejects_overlapping_internal_external_seed_claims(tmp_path: Path) -> None:
    manifest = fleet.load_manifest(_manifest_file(tmp_path))
    candidate = tmp_path / "candidate.pt"
    champion = tmp_path / "champion.pt"
    candidate.write_bytes(b"candidate")
    champion.write_bytes(b"champion")
    registry = _registry(tmp_path, champion)
    with pytest.raises(fleet.FleetError, match="seed intervals overlap"):
        fleet.build_plan(
            manifest,
            candidate=candidate,
            champion=champion,
            candidate_parent=champion,
            registry=registry,
            internal_pairs=600,
            external_pairs=500,
            internal_base_seed=6_190_000_000,
            external_base_seed=6_190_000_500,
            repo_commit="a" * 40,
            tool_hashes={},
        )


def test_load_plan_rejects_semantic_tamper_even_if_old_hash_remains(
    tmp_path: Path,
) -> None:
    manifest, plan = _plan(tmp_path)
    tampered = copy.deepcopy(plan)
    tampered["jobs"][0]["pairs"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(fleet.FleetError, match="plan hash does not replay"):
        fleet.load_plan(path, manifest)


def test_load_plan_replays_run_identity_and_confines_remote_job_paths(
    tmp_path: Path,
) -> None:
    manifest, plan = _plan(tmp_path)

    wrong_run = copy.deepcopy(plan)
    wrong_run["run_id"] = "a1-eval-0000000000000000"
    wrong_run["plan_hash"] = fleet._digest(  # noqa: SLF001
        {key: value for key, value in wrong_run.items() if key != "plan_hash"}
    )
    wrong_run_path = tmp_path / "wrong-run.json"
    wrong_run_path.write_text(json.dumps(wrong_run), encoding="utf-8")
    with pytest.raises(fleet.FleetError, match="run identity does not replay"):
        fleet.load_plan(wrong_run_path, manifest)

    escaped = copy.deepcopy(plan)
    job = escaped["jobs"][0]
    job["job_dir"] = "/home/ubuntu/catan-zero-production/recovery/active"
    job["report"] = f"{job['job_dir']}/report.json"
    job["argv"][-1] = job["report"]
    job["command_hash"] = fleet._digest(job["argv"])  # noqa: SLF001
    escaped["plan_hash"] = fleet._digest(  # noqa: SLF001
        {key: value for key, value in escaped.items() if key != "plan_hash"}
    )
    escaped_path = tmp_path / "escaped.json"
    escaped_path.write_text(json.dumps(escaped), encoding="utf-8")
    with pytest.raises(fleet.FleetError, match="path escapes its sealed run"):
        fleet.load_plan(escaped_path, manifest)


def test_validation_claim_is_atomic_idempotent_and_journaled(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    assert fleet.claim_validation_ranges(manifest, plan) == "claimed"
    assert fleet.claim_validation_ranges(manifest, plan) == "adopted"
    ledger = Path(manifest["validation_ledger"])
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["event"] == "claim"
    assert events[0]["plan_hash"] == plan["plan_hash"]
    claim_files = list(Path(str(ledger) + ".claims").glob("*.json"))
    assert len(claim_files) == 1
    assert (claim_files[0].stat().st_mode & 0o777) == 0o444


def test_validation_claim_rejects_overlap_from_concurrent_plan(tmp_path: Path) -> None:
    manifest, first = _plan(tmp_path)
    assert fleet.claim_validation_ranges(manifest, first) == "claimed"
    candidate = tmp_path / "candidate-2.pt"
    candidate.write_bytes(b"candidate-two")
    second = fleet.build_plan(
        manifest,
        candidate=candidate,
        champion=Path(first["champion"]["source"]),
        **_binding_kwargs(first),
        internal_pairs=600,
        external_pairs=500,
        internal_base_seed=6_190_000_300,
        external_base_seed=6_192_000_000,
        iteration_id="a2",
        repo_commit="a" * 40,
        tool_hashes=first["tool_hashes"],
    )
    with pytest.raises(fleet.FleetError, match="VAL-only seed overlap"):
        fleet.claim_validation_ranges(manifest, second)


def test_explicit_common_seed_cohort_allows_only_exact_interval_reuse(
    tmp_path: Path,
) -> None:
    manifest, template = _plan(tmp_path)
    champion = Path(template["champion"]["source"])

    def make(name: str, *, internal_base: int = 6_190_000_000) -> dict:
        candidate = tmp_path / f"{name}.pt"
        candidate.write_bytes(name.encode())
        return fleet.build_plan(
            manifest,
            candidate=candidate,
            champion=champion,
            **_binding_kwargs(template),
            internal_pairs=600,
            external_pairs=500,
            internal_base_seed=internal_base,
            external_base_seed=6_191_000_000,
            iteration_id=name,
            seed_cohort_id="dual-arm-common-v1",
            repo_commit="a" * 40,
            tool_hashes=template["tool_hashes"],
        )

    first = make("candidate-one")
    second = make("candidate-two")
    assert first["plan_hash"] != second["plan_hash"]
    assert fleet.claim_validation_ranges(manifest, first) == "claimed"
    assert fleet.claim_validation_ranges(manifest, second) == "claimed"

    calibrated = fleet.build_plan(
        manifest,
        candidate=Path(second["candidate"]["source"]),
        champion=champion,
        **_binding_kwargs(template),
        internal_pairs=600,
        external_pairs=500,
        internal_base_seed=6_190_000_000,
        external_base_seed=6_191_000_000,
        iteration_id="candidate-two-calibrated",
        seed_cohort_id="dual-arm-common-v1",
        candidate_c_scale=0.10,
        champion_c_scale=0.03,
        repo_commit="a" * 40,
        tool_hashes=template["tool_hashes"],
    )
    assert calibrated["science_config_hash"] != second["science_config_hash"]
    assert fleet.claim_validation_ranges(manifest, calibrated) == "claimed"

    partial = make("candidate-partial", internal_base=6_190_000_001)
    with pytest.raises(fleet.FleetError, match="VAL-only seed overlap"):
        fleet.claim_validation_ranges(manifest, partial)


def test_common_seed_reuse_requires_an_explicit_shared_cohort(tmp_path: Path) -> None:
    manifest, first = _plan(tmp_path)
    assert fleet.claim_validation_ranges(manifest, first) == "claimed"
    candidate = tmp_path / "candidate-without-cohort.pt"
    candidate.write_bytes(b"other")
    second = fleet.build_plan(
        manifest,
        candidate=candidate,
        champion=Path(first["champion"]["source"]),
        **_binding_kwargs(first),
        internal_pairs=600,
        external_pairs=500,
        internal_base_seed=6_190_000_000,
        external_base_seed=6_191_000_000,
        iteration_id="no-common-cohort",
        repo_commit="a" * 40,
        tool_hashes=first["tool_hashes"],
    )
    with pytest.raises(fleet.FleetError, match="VAL-only seed overlap"):
        fleet.claim_validation_ranges(manifest, second)


def test_validation_status_adopts_claim_and_appends_event(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    fleet.claim_validation_ranges(manifest, plan)
    fleet.record_validation_status(manifest, plan, status="internal_collected")
    events = [
        json.loads(line)
        for line in Path(manifest["validation_ledger"]).read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["claim", "status"]
    assert events[-1]["status"] == "internal_collected"


def test_claim_adoption_repairs_crash_between_claim_file_and_journal(
    tmp_path: Path,
) -> None:
    manifest, plan = _plan(tmp_path)
    ledger = Path(manifest["validation_ledger"])
    claims = Path(str(ledger) + ".claims")
    claims.mkdir(parents=True)
    claim = claims / f"{plan['plan_hash'][7:]}.json"
    claim.write_text(json.dumps(fleet._claim_payload(plan)))  # noqa: SLF001
    assert fleet.claim_validation_ranges(manifest, plan) == "adopted"
    event = json.loads(ledger.read_text().strip())
    assert event["event"] == "claim"
    assert event["recovered"] is True
