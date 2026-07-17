from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.fleet import a1_live_canary as canary
from tools import prelaunch_guard


def _source_argv(out_dir: str, base_seed: int, claim: str) -> list[str]:
    return [
        "tools/generate_gumbel_selfplay_data.py",
        "--out-dir",
        out_dir,
        "--games",
        "245",
        "--workers",
        "16",
        "--checkpoint",
        "/models/champion.pt",
        "--device",
        "cuda",
        "--n-full",
        "128",
        "--n-fast",
        "16",
        "--p-full",
        "0.25",
        "--c-visit",
        "50.0",
        "--c-scale",
        "0.1",
        "--rescale-noise-floor-c",
        "0.0",
        "--sigma-eval",
        "0.98",
        "--wide-candidates-threshold",
        "24",
        "--max-depth",
        "80",
        "--base-seed",
        str(base_seed),
        "--ledger-claim-label",
        claim,
        "--symmetry-averaged-eval-threshold",
        "20",
        "--determinization-particles",
        "4",
        "--determinization-min-simulations",
        "32",
        "--symmetry-averaged-eval",
        "--no-wide-roots-always-full",
        "--lazy-interior-chance",
        "--no-belief-chance-spectra",
        "--information-set-search",
        "--public-observation",
        "--no-native-mcts-hot-loop",
        "--no-rust-featurize",
        "--no-eval-server",
        "--seed-claim",
        "--resume",
    ]


def _source_config_provenance(base_seed: int) -> dict:
    config = canary.contract.GenerateConfig(
        games=245,
        base_seed=base_seed,
        n_full=128,
        information_set_search=True,
        determinization_particles=4,
        determinization_min_simulations=32,
        public_observation=True,
        native_mcts_hot_loop=False,
        rust_featurize=False,
    )
    value = {
        "pipeline": "generate",
        "config_hash": config.config_hash(),
        "full_config_hash": config.full_config_hash(),
        "config": config.canonical_payload(),
    }
    value["provenance_sha256"] = canary._digest(value)
    return value


def _current_coherent_lock_job() -> tuple[dict, dict]:
    lock = {
        "schema_version": canary.contract.LOCK_SCHEMA,
        "science": {
            "search_operator": canary.current_science.search(),
            "evaluator": canary.current_science.evaluator(),
        },
        "generation": canary.current_science.generation(),
        "checkpoints": [
            {
                "role": "producer",
                "path": "/models/current-coherent-producer.pt",
                "sha256": "sha256:" + "1" * 64,
            }
        ],
        "provenance": {
            "guard_config": {
                "path": (
                    "configs/guards/"
                    "a1_generation_coherent_public_n128_adaptive256_forced_value_v3.json"
                )
            }
        },
    }
    job = {
        "category": "current_producer",
        "output_dir": "/home/ubuntu/catan-zero-production/current-coherent",
        "attempts": 150,
        "base_seed": 400_000_000_000,
        "claim_label": "current-coherent-render",
    }
    return lock, job


def _current_coherent_source_argv() -> list[str]:
    """Render one current-producer command through the production renderer."""

    lock, job = _current_coherent_lock_job()
    return canary.contract._generator_argv(lock, job, mix_paths={})  # noqa: SLF001


def test_current_coherent_render_accepts_disabled_adaptive_budget() -> None:
    lock, job = _current_coherent_lock_job()
    source = _current_coherent_source_argv()
    derived = canary._replace_values(
        source,
        {
            "--out-dir": "/home/ubuntu/gen_out/a1-live-canary-coherent/gpu0",
            "--games": "16",
            "--base-seed": str(canary.contract.VAL_ONLY_SEED_RANGE[0]),
            "--ledger-claim-label": "val-current-coherent-gpu0",
        },
    )

    # The adopted base-n128 operator omits nullable adaptive values and renders
    # both false and true boolean science fields explicitly.
    assert "--n-full-wide" not in source
    assert "--n-full-wide-threshold" not in source
    assert "--no-wide-roots-always-full" in source
    assert "--record-automatic-transitions" in source
    assert canary._flag_value(source, "--workers") == "128"
    assert "--eval-server" in source
    assert canary._flag_value(source, "--eval-server-max-batch") == "96"
    assert "--eval-server-request-collector" in source
    assert canary._flag_value(source, "--eval-server-matmul-precision") == "highest"
    bucket_index = source.index("--eval-server-cuda-graph-batch-buckets")
    assert source[bucket_index + 1 : bucket_index + 13] == [
        str(value)
        for value in lock["generation"]["eval_server_cuda_graph_batch_buckets"]
    ]
    canary._assert_exact_recipe(source, derived)

    provenance = canary.contract._expected_generate_config_provenance(  # noqa: SLF001
        lock, job, opponent_mix_manifest=None
    )
    projected = provenance["config"]["fields"]
    generation = canary.current_science.generation()
    for science_field, config_field in (
        canary.current_science.PRODUCTION_GENERATION_RUNTIME_FIELD_MAP.items()
    ):
        assert projected[config_field] == generation[science_field]


def test_opponent_mix_local_mps_render_is_not_production_generation() -> None:
    lock, job = _current_coherent_lock_job()
    job["category"] = "recent_history"
    mix_path = Path("/sealed/recent_history.opponent-mix.json")
    source = canary.contract._generator_argv(  # noqa: SLF001
        lock, job, mix_paths={"recent_history": mix_path}
    )
    derived = canary._replace_values(
        source,
        {
            "--out-dir": "/home/ubuntu/gen_out/a1-live-canary-mix/gpu0",
            "--games": "16",
            "--base-seed": str(canary.contract.VAL_ONLY_SEED_RANGE[0]),
            "--ledger-claim-label": "val-current-mix-gpu0",
        },
    )
    assert canary._flag_value(source, "--workers") == "16"
    assert "--no-eval-server" in source
    assert "--eval-server" not in source
    assert canary._flag_value(source, "--opponent-mix-manifest") == str(mix_path)
    canary._assert_exact_recipe(source, derived)

    guard_path = (
        canary.contract.REPO_ROOT
        / "configs/guards/a1_generation_coherent_public_n128_adaptive256_forced_value_v3.json"
    )
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    lint = next(item["args"] for item in guard["guards"] if item["name"] == "cli_flag_lint")
    result = prelaunch_guard.guard_cli_flag_lint(
        argv=source[1:],
        critical_flags=lint["critical_flags"],
        expected_values=lint["expected_values"],
        forbidden_flags=lint.get("forbidden_flags", ()),
        parser=canary.contract.generation_cli.build_parser(),
    )
    assert not result.passed
    assert "--workers=16 (expected 128)" in result.reason
    assert "--eval-server=False (expected True)" in result.reason

    provenance = canary.contract._expected_generate_config_provenance(  # noqa: SLF001
        lock, job, opponent_mix_manifest=str(mix_path)
    )
    assert provenance["config"]["fields"]["workers"] == 16
    assert provenance["config"]["fields"]["eval_server"] is False


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(canary.executor, "_repo_artifacts", lambda *_a, **_k: [])
    lock_path = tmp_path / "lock.json"
    render_path = tmp_path / "render.json"
    hosts_path = tmp_path / "hosts.json"
    receipt_path = tmp_path / "receipt.json"
    for path in (lock_path, render_path, hosts_path):
        path.write_text("{}\n", encoding="utf-8")
    os.chmod(hosts_path, 0o600)
    production_ledger = tmp_path / "PRODUCTION_LEDGER.md"
    production_ledger.write_text("# production\n", encoding="utf-8")
    lock = {
        "contract_sha256": "sha256:" + "a" * 64,
        "science": {"evaluator": {"value_readout": "scalar"}},
        "checkpoints": [
            {
                "role": "producer",
                "metadata": {
                    "mask_hidden_info": True,
                    "legacy_scalar_readout_attestation": {
                        "schema_version": "legacy-scalar-readout-attestation-v1"
                    },
                },
            }
        ],
        "fleet": {"output_root": "/home/ubuntu/catan-zero-production/runs/selfplay"},
    }
    aliases = {
        "c1": 4,
        "c2": 4,
        "c3": 4,
        "c4": 4,
        "c5": 4,
        "c6": 4,
        "h100-8a": 8,
        "h100-8b": 8,
    }
    lanes: dict[str, list[dict]] = {}
    commands: list[dict] = []
    counter = 0
    for alias, count in aliases.items():
        for gpu in range(count):
            worker = f"{alias.replace('h100-', 'h')}_gpu{gpu}"
            lane = []
            previous = None
            for category in canary.CATEGORY_ORDER:
                job = f"{worker}__{category}"
                source_attestation = tmp_path / "attestations" / f"{job}.json"
                source_attestation.parent.mkdir(exist_ok=True)
                payload = {"job_id": job, "contract_sha256": lock["contract_sha256"]}
                source_attestation.write_text(
                    json.dumps(payload) + "\n", encoding="utf-8"
                )
                argv = _source_argv(
                    f"/home/ubuntu/catan-zero-production/runs/selfplay/{job}",
                    300_000_000_000 + counter * 1000,
                    f"production-{counter}",
                )
                command = {
                    "job_id": job,
                    "worker_id": worker,
                    "host_alias": alias,
                    "gpu": gpu,
                    "category": category,
                    "argv": argv,
                    "argv_sha256": canary._digest(argv),
                    "environment": {
                        **canary.contract.SEALED_RUNTIME_ENVIRONMENT,
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        **canary.executor.CLIENT_ENVIRONMENT,
                        "CATAN_SEED_LEDGER": str(production_ledger),
                        "CATAN_A1_CONTRACT_SHA256": lock["contract_sha256"],
                        "CATAN_ZERO_CONFIG_REGISTRY": str(
                            Path(argv[argv.index("--out-dir") + 1])
                            / "config_registry.jsonl"
                        ),
                    },
                    "config_provenance": _source_config_provenance(
                        300_000_000_000 + counter * 1000
                    ),
                    "ledger_claim": {"path": str(production_ledger)},
                    "output_attestation": {
                        "source": str(source_attestation),
                        "source_file_sha256": canary._sha256(source_attestation),
                        "destination": f"/production/{job}/a1_contract.json",
                        "payload_sha256": canary._digest(payload),
                    },
                    "must_run_after": [] if previous is None else [previous],
                }
                command["environment_sha256"] = canary._digest(
                    command["environment"]
                )
                lane.append(command)
                commands.append(command)
                previous = job
                counter += 1
            lanes[worker] = lane
    rendered = {
        "render_sha256": "sha256:" + "b" * 64,
        "commands": commands,
        "required_artifacts": {
            "seed_ledger": {
                "path": str(production_ledger),
                "sha256": canary._sha256(production_ledger),
            },
            "checkpoints": [],
            "rendered_opponent_mix": [],
        },
    }
    hosts = {
        "schema_version": canary.executor.HOST_SCHEMA,
        "ssh_user": "ubuntu",
        "ssh_key": str(tmp_path / "key"),
        "remote_root": "/home/ubuntu/a1-production",
        "python": "/home/ubuntu/catan-zero-v1/.venv/bin/python",
        "hosts": {alias: f"192.0.2.{index + 1}" for index, alias in enumerate(aliases)},
    }
    allowed = tmp_path / "gen_out"
    root = allowed / "a1-live-canary-shapecheck"
    plan = canary.derive_canary_plan(
        lock=lock,
        rendered=rendered,
        lanes=lanes,
        hosts=hosts,
        lock_path=lock_path,
        render_path=render_path,
        hosts_path=hosts_path,
        receipt_path=receipt_path,
        canary_id="shapecheck",
        base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0],
        canary_root=root,
        allowed_root=allowed,
    )
    return plan, lock, rendered, lanes, hosts, root, production_ledger


def test_derives_exact_twelve_lane_validation_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _lock, _rendered, _lanes, _hosts, root, production_ledger = _fixture(
        tmp_path, monkeypatch
    )
    assert plan["validation_only"] is True
    assert plan["lane_count"] == 12
    assert plan["job_count"] == 36
    assert plan["claim_count"] == 0
    assert plan["canary_claim_count"] == 36
    assert plan["production_claims_consumed"] == 0
    assert set(plan["_private"]["hosts"]["hosts"]) == {"c1", "h100-8a"}
    assert plan["canary_seed_ledger"] != str(production_ledger)
    assert Path(plan["canary_seed_ledger"]).is_relative_to(root)
    assert production_ledger.read_text() == "# production\n"
    parsed_claims = canary.contract.parse_seed_ledger(plan["canary_seed_ledger"])
    assert len(parsed_claims) == 36
    assert all(
        "VAL-ONLY" in label and "claim=val-" in label for _, _, label in parsed_claims
    )
    assert [
        lane[0]["gpu"]
        for lane in plan["_private"]["lanes"].values()
        if lane[0]["host_alias"] == "c1"
    ] == [0, 1, 2, 3]
    assert sorted(
        lane[0]["gpu"]
        for lane in plan["_private"]["lanes"].values()
        if lane[0]["host_alias"] == "h100-8a"
    ) == list(range(8))
    assert all(
        command["config_provenance"]["config"]["fields"]["rust_featurize"]
        is False
        and "--no-rust-featurize" in command["argv"]
        and "--rust-featurize" not in command["argv"]
        for lane in plan["_private"]["lanes"].values()
        for command in lane
    )
    canary.validate_canary_plan(plan)


def test_derives_operator_selected_cohort_and_game_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        _default,
        lock,
        rendered,
        lanes,
        hosts,
        _root,
        _production_ledger,
    ) = _fixture(tmp_path, monkeypatch)
    chosen_root = tmp_path / "allowed" / "a1-live-canary-n128pilot"
    plan = canary.derive_canary_plan(
        lock=lock,
        rendered=rendered,
        lanes=lanes,
        hosts=hosts,
        lock_path=tmp_path / "lock.json",
        render_path=tmp_path / "render.json",
        hosts_path=tmp_path / "hosts.json",
        receipt_path=tmp_path / "pilot.receipt.json",
        canary_id="n128pilot",
        base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0] + 10_000,
        canary_root=chosen_root,
        allowed_root=tmp_path / "allowed",
        canary_aliases={"c2": 4, "h100-8b": 8},
        games_per_job=8,
        native_runtime=True,
        categories=("current_producer",),
    )
    assert plan["canary_aliases"] == {"c2": 4, "h100-8b": 8}
    assert plan["games_per_job"] == 8
    assert plan["native_runtime"] is True
    assert plan["lane_count"] == 12
    assert plan["job_count"] == 12
    assert plan["category_order"] == ["current_producer"]
    assert set(plan["_private"]["hosts"]["hosts"]) == {"c2", "h100-8b"}
    assert all(
        canary._flag_value(command["argv"], "--games") == "8"
        for lane in plan["_private"]["lanes"].values()
        for command in lane
    )
    assert all(
        "--native-mcts-hot-loop" in command["argv"]
        and "--rust-featurize" in command["argv"]
        and "--no-native-mcts-hot-loop" not in command["argv"]
        and "--no-rust-featurize" not in command["argv"]
        and command["config_provenance"]["config"]["fields"]["rust_featurize"]
        is True
        for lane in plan["_private"]["lanes"].values()
        for command in lane
    )
    canary.validate_canary_plan(plan)


def test_only_identity_seed_output_and_attempt_values_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _lock, rendered, _lanes, _hosts, root, _ledger = _fixture(
        tmp_path, monkeypatch
    )
    source = {command["job_id"]: command for command in rendered["commands"]}
    ranges = []
    for lane in plan["_private"]["lanes"].values():
        assert [item["category"] for item in lane] == list(canary.CATEGORY_ORDER)
        for command in lane:
            original = source[command["source_production_job_id"]]
            canary._assert_exact_recipe(original["argv"], command["argv"])
            assert canary._flag_value(command["argv"], "--games") == "16"
            assert Path(canary._flag_value(command["argv"], "--out-dir")).parent == root
            start = int(canary._flag_value(command["argv"], "--base-seed"))
            ranges.append((start, start + canary.GAMES_PER_JOB))
            assert (
                command["environment"]["CATAN_SEED_LEDGER"]
                == plan["canary_seed_ledger"]
            )
            assert command["environment"]["CATAN_A1_CANARY_ID"] == "shapecheck"
            assert command["environment"]["CATAN_ZERO_CONFIG_REGISTRY"] == str(
                Path(canary._flag_value(command["argv"], "--out-dir"))
                / "config_registry.jsonl"
            )
            assert "/catan-zero-production/" not in command["environment"][
                "CATAN_ZERO_CONFIG_REGISTRY"
            ]
            assert command["environment_sha256"] == canary._digest(
                command["environment"]
            )
            attestation = json.loads(
                Path(command["output_attestation"]["source"]).read_text()
            )
            assert attestation["validation_only"] is True
            assert (
                attestation["target_information_regime"]
                == "public_conservation_pimc_v1"
            )
            assert attestation["source_job"]["job_id"] == original["job_id"]
    assert len(ranges) == len(set(ranges)) == 36
    assert ranges == sorted(ranges)


@pytest.mark.parametrize(
    ("seed_delta", "root_name", "message"),
    [
        (-1, "a1-live-canary-shapecheck", "VAL-ONLY"),
        (0, "wrong-root", "root must be exactly"),
    ],
)
def test_rejects_seed_or_output_boundary_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed_delta: int,
    root_name: str,
    message: str,
) -> None:
    plan, lock, rendered, lanes, hosts, _root, _ledger = _fixture(tmp_path, monkeypatch)
    with pytest.raises(canary.CanaryError, match=message):
        canary.derive_canary_plan(
            lock=lock,
            rendered=rendered,
            lanes=lanes,
            hosts=hosts,
            lock_path=Path(plan["lock"]),
            render_path=tmp_path / "render.json",
            hosts_path=tmp_path / "hosts.json",
            receipt_path=tmp_path / "other.receipt.json",
            canary_id="shapecheck",
            base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0] + seed_delta,
            canary_root=tmp_path / "gen_out" / root_name,
            allowed_root=tmp_path / "gen_out",
        )


def test_rejects_missing_shape_or_scalar_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, lock, rendered, lanes, hosts, root, _ledger = _fixture(tmp_path, monkeypatch)
    broken_lanes = dict(lanes)
    broken_lanes.pop("c1_gpu3")
    with pytest.raises(canary.CanaryError, match="c1 gpu0-3"):
        canary.derive_canary_plan(
            lock=lock,
            rendered=rendered,
            lanes=broken_lanes,
            hosts=hosts,
            lock_path=Path(plan["lock"]),
            render_path=tmp_path / "render.json",
            hosts_path=tmp_path / "hosts.json",
            receipt_path=tmp_path / "missing.receipt.json",
            canary_id="shapecheck",
            base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0],
            canary_root=root,
            allowed_root=tmp_path / "gen_out",
        )


def test_rejects_any_source_science_or_guard_bypass_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, lock, rendered, lanes, hosts, root, _ledger = _fixture(tmp_path, monkeypatch)
    broken_lanes = copy.deepcopy(lanes)
    command = broken_lanes["c1_gpu0"][0]
    n_full_index = command["argv"].index("--n-full") + 1
    command["argv"][n_full_index] = "64"
    with pytest.raises(canary.CanaryError, match="exact recipe requires --n-full=128"):
        canary.derive_canary_plan(
            lock=lock,
            rendered={
                **rendered,
                "commands": [item for lane in broken_lanes.values() for item in lane],
            },
            lanes=broken_lanes,
            hosts=hosts,
            lock_path=Path(plan["lock"]),
            render_path=tmp_path / "render.json",
            hosts_path=tmp_path / "hosts.json",
            receipt_path=tmp_path / "science.receipt.json",
            canary_id="shapecheck",
            base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0],
            canary_root=root,
            allowed_root=tmp_path / "gen_out",
        )

    broken_lanes = copy.deepcopy(lanes)
    command = broken_lanes["c1_gpu0"][0]
    command["argv"].append("--skip-guards")
    with pytest.raises(canary.CanaryError, match="switch drift"):
        canary.derive_canary_plan(
            lock=lock,
            rendered={
                **rendered,
                "commands": [item for lane in broken_lanes.values() for item in lane],
            },
            lanes=broken_lanes,
            hosts=hosts,
            lock_path=Path(plan["lock"]),
            render_path=tmp_path / "render.json",
            hosts_path=tmp_path / "hosts.json",
            receipt_path=tmp_path / "guard.receipt.json",
            canary_id="shapecheck",
            base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0],
            canary_root=root,
            allowed_root=tmp_path / "gen_out",
        )
    broken_lock = copy.deepcopy(lock)
    broken_lock["checkpoints"][0]["metadata"].pop("legacy_scalar_readout_attestation")
    with pytest.raises(canary.CanaryError, match="typed legacy scalar"):
        canary.derive_canary_plan(
            lock=broken_lock,
            rendered=rendered,
            lanes=lanes,
            hosts=hosts,
            lock_path=Path(plan["lock"]),
            render_path=tmp_path / "render.json",
            hosts_path=tmp_path / "hosts.json",
            receipt_path=tmp_path / "scalar.receipt.json",
            canary_id="shapecheck",
            base_seed=canary.contract.VAL_ONLY_SEED_RANGE[0],
            canary_root=root,
            allowed_root=tmp_path / "gen_out",
        )


def test_validation_rejects_any_production_ledger_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, *_ = _fixture(tmp_path, monkeypatch)
    plan["_private"]["rendered"]["required_artifacts"]["seed_ledger"]["path"] = plan[
        "production_seed_ledger"
    ]
    with pytest.raises(canary.CanaryError, match="not the canary ledger"):
        canary.validate_canary_plan(plan)


def test_validation_rejects_production_registry_leak_even_when_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _lock, _rendered, _lanes, _hosts, _root, _ledger = _fixture(
        tmp_path, monkeypatch
    )
    command = next(iter(plan["_private"]["lanes"].values()))[0]
    command["environment"]["CATAN_ZERO_CONFIG_REGISTRY"] = (
        "/home/ubuntu/catan-zero-production/runs/selfplay/config_registry.jsonl"
    )
    command["environment_sha256"] = canary._digest(command["environment"])
    with pytest.raises(canary.CanaryError, match="exact environment drift"):
        canary.validate_canary_plan(plan)


def test_status_stop_and_execute_reuse_hardened_receipt_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, *_ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(canary, "build_canary_plan", lambda **_kwargs: plan)
    calls = []
    monkeypatch.setattr(
        canary.executor,
        "execute",
        lambda built, *, receipt_path, resume: (
            calls.append(("run", built, receipt_path, resume)) or {"status": "launched"}
        ),
    )
    monkeypatch.setattr(
        canary.executor,
        "status",
        lambda built, *, receipt_path: (
            calls.append(("status", built, receipt_path)) or {"status": "ok"}
        ),
    )
    monkeypatch.setattr(
        canary.executor,
        "stop_execution",
        lambda built, *, receipt_path, go: (
            calls.append(("stop", built, receipt_path, go)) or {"status": "stopped"}
        ),
    )
    monkeypatch.setattr(
        canary,
        "attest_mps_runtime",
        lambda built, **_kwargs: (
            calls.append(("mps-runtime", built))
            or {"required_nofile_soft": canary.executor.REQUIRED_NOFILE_SOFT}
        ),
    )
    common = [
        "--lock",
        str(tmp_path / "lock"),
        "--render",
        str(tmp_path / "render"),
        "--hosts",
        str(tmp_path / "hosts"),
        "--receipt",
        str(tmp_path / "receipt"),
        "--canary-id",
        "shapecheck",
        "--base-seed",
        str(canary.contract.VAL_ONLY_SEED_RANGE[0]),
    ]
    assert canary.main(["run", *common, "--resume", "--go"]) == 0
    assert canary.main(["status", *common]) == 0
    assert canary.main(["stop", *common, "--go"]) == 0
    assert [call[0] for call in calls] == ["run", "mps-runtime", "status", "stop"]
    assert calls[0][3] is True and calls[3][3] is True


def test_live_mps_runtime_attestation_rejects_low_server_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lanes = {}
    for alias, count in canary.CANARY_ALIASES.items():
        for gpu in range(count):
            worker_id = f"canary-{alias}-gpu{gpu}"
            lanes[worker_id] = [
                {
                    "job_id": f"{worker_id}__{category}",
                    "host_alias": alias,
                    "gpu": gpu,
                    "argv": [
                        "generate.py",
                        "--out-dir",
                        str(tmp_path / worker_id / category),
                        "--workers",
                        "16",
                        "--games",
                        "16",
                    ],
                }
                for category in canary.CATEGORY_ORDER
            ]
    plan = {
        "_private": {
            "hosts": {"python": "/venv/bin/python"},
            "lanes": lanes,
        }
    }
    monkeypatch.setattr(canary, "validate_canary_plan", lambda _plan: None)

    not_before = 1_700_000_000.0
    not_before_ns = int(not_before * 1_000_000_000)

    def response(
        alias: str,
        limit: object,
        *,
        drop_lane: bool = False,
        drop_worker: bool = False,
        stale_worker: bool = False,
    ) -> SimpleNamespace:
        progress = {
            item["worker_id"]: {
                "worker_id": item["worker_id"],
                "gpu": item["gpu"],
                "job_id": item["jobs"][0]["job_id"],
                "output": item["jobs"][0]["output"],
                "expected_workers": item["jobs"][0]["workers"],
                "workers": {
                    f"worker_{index:03d}": {
                        "progress": (
                            f"{item['jobs'][0]['output']}/"
                            f"worker_{index:03d}/progress.json"
                        ),
                        "rows": 10,
                        "simulations": 128,
                        "games_failed": 0,
                        "mtime_ns": not_before_ns + 1,
                    }
                    for index in range(item["jobs"][0]["workers"])
                },
            }
            for item in canary._runtime_lane_expectations(plan, alias)
        }
        if drop_lane:
            progress.pop(next(iter(progress)))
        if drop_worker:
            first = progress[next(iter(progress))]
            first["workers"].pop(next(iter(first["workers"])))
        if stale_worker:
            first = progress[next(iter(progress))]
            first["workers"][next(iter(first["workers"]))]["mtime_ns"] = (
                not_before_ns - 1
            )
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "required_nofile_soft": canary.executor.REQUIRED_NOFILE_SOFT,
                    "server_nofile_soft": {"4242": limit},
                    "canary_lane_progress": progress,
                }
            ),
            stderr="",
        )

    timeouts: list[float] = []

    def safe_ssh(_hosts, alias, _command, **kwargs):
        timeouts.append(kwargs["timeout_seconds"])
        return response(alias, canary.executor.REQUIRED_NOFILE_SOFT)

    monkeypatch.setattr(
        canary.executor,
        "_ssh",
        safe_ssh,
    )
    report = canary.attest_mps_runtime(
        plan, not_before_epoch=not_before, timeout_seconds=0.01
    )
    assert set(report["hosts"]) == {"c1", "h100-8a"}
    assert len(timeouts) == 2
    assert all(15.0 <= timeout <= 16.0 for timeout in timeouts)
    assert "timeout=" in canary.MPS_RUNTIME_ATTESTATION_SCRIPT

    monkeypatch.setattr(
        canary.executor,
        "_ssh",
        lambda _hosts, alias, _command, **_kwargs: response(alias, 1024),
    )
    with pytest.raises(canary.CanaryError, match="unsafe MPS runtime limit"):
        canary.attest_mps_runtime(
            plan, not_before_epoch=not_before, timeout_seconds=0.01
        )

    monkeypatch.setattr(
        canary.executor,
        "_ssh",
        lambda _hosts, alias, _command, **_kwargs: response(
            alias, canary.executor.REQUIRED_NOFILE_SOFT, drop_lane=True
        ),
    )
    with pytest.raises(canary.CanaryError, match="incomplete canary lane progress"):
        canary.attest_mps_runtime(
            plan, not_before_epoch=not_before, timeout_seconds=0.01
        )

    monkeypatch.setattr(
        canary.executor,
        "_ssh",
        lambda _hosts, alias, _command, **_kwargs: response(
            alias, canary.executor.REQUIRED_NOFILE_SOFT, drop_worker=True
        ),
    )
    with pytest.raises(canary.CanaryError, match="unsafe canary lane progress"):
        canary.attest_mps_runtime(
            plan, not_before_epoch=not_before, timeout_seconds=0.01
        )

    monkeypatch.setattr(
        canary.executor,
        "_ssh",
        lambda _hosts, alias, _command, **_kwargs: response(
            alias, canary.executor.REQUIRED_NOFILE_SOFT, stale_worker=True
        ),
    )
    with pytest.raises(canary.CanaryError, match="stale/unsafe canary worker progress"):
        canary.attest_mps_runtime(
            plan, not_before_epoch=not_before, timeout_seconds=0.01
        )


def test_live_mps_runtime_attestation_hard_bounds_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lanes = {
        f"canary-{alias}-gpu{gpu}": [
            {
                "job_id": f"canary-{alias}-gpu{gpu}__current_producer",
                "host_alias": alias,
                "gpu": gpu,
                "argv": [
                    "generate.py",
                    "--out-dir",
                    str(tmp_path / alias / f"gpu{gpu}"),
                    "--workers",
                    "16",
                    "--games",
                    "16",
                ],
            }
        ]
        for alias, count in canary.CANARY_ALIASES.items()
        for gpu in range(count)
    }
    plan = {
        "_private": {
            "hosts": {"python": "/venv/bin/python"},
            "lanes": lanes,
        }
    }
    monkeypatch.setattr(canary, "validate_canary_plan", lambda _plan: None)
    monkeypatch.setattr(
        canary.executor,
        "_ssh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("ssh", 1.0)
        ),
    )
    with pytest.raises(canary.CanaryError, match="transport timed out on c1"):
        canary.attest_mps_runtime(
            plan, not_before_epoch=1.0, timeout_seconds=0.01
        )


def test_go_stops_exact_canary_when_mps_runtime_attestation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {"plan_sha256": "sha256:mps-failure", "_private": {}}
    monkeypatch.setattr(canary, "build_canary_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        canary.executor,
        "execute",
        lambda *_args, **_kwargs: {"status": "launched"},
    )
    monkeypatch.setattr(
        canary,
        "attest_mps_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            canary.CanaryError("low MPS server limit")
        ),
    )
    stops: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        canary.executor,
        "stop_execution",
        lambda _plan, *, receipt_path, go: stops.append((receipt_path, go)),
    )
    receipt = tmp_path / "receipt.json"
    result = canary.main(
        [
            "run",
            "--lock",
            str(tmp_path / "lock"),
            "--render",
            str(tmp_path / "render"),
            "--hosts",
            str(tmp_path / "hosts"),
            "--receipt",
            str(receipt),
            "--canary-id",
            "shapecheck",
            "--base-seed",
            str(canary.contract.VAL_ONLY_SEED_RANGE[0]),
            "--go",
        ]
    )
    assert result == 2
    assert stops == [(receipt, True)]


def test_execute_partial_launch_failure_exact_stops_receipt_bound_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending_worker = "canary-c1-gpu0"
    plan = {
        "plan_sha256": "sha256:partial-launch",
        "_private": {"lanes": {pending_worker: [{"host_alias": "c1", "gpu": 0}]}},
    }
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(canary, "build_canary_plan", lambda **_kwargs: plan)

    def partial_execute(*_args, **_kwargs):
        receipt.write_text(
            json.dumps(
                {
                    "plan_sha256": plan["plan_sha256"],
                    "status": "launching",
                    "lane_pids": {},
                    "launch_pending_worker_id": pending_worker,
                }
            ),
            encoding="utf-8",
        )
        raise canary.executor.ExecutorError("later lane acknowledgement failed")

    monkeypatch.setattr(canary.executor, "execute", partial_execute)
    stops: list[bool] = []
    monkeypatch.setattr(
        canary.executor,
        "stop_execution",
        lambda *_args, **kwargs: stops.append(kwargs["go"]),
    )
    result = canary.main(
        [
            "run",
            "--lock",
            str(tmp_path / "lock"),
            "--render",
            str(tmp_path / "render"),
            "--hosts",
            str(tmp_path / "hosts"),
            "--receipt",
            str(receipt),
            "--canary-id",
            "shapecheck",
            "--base-seed",
            str(canary.contract.VAL_ONLY_SEED_RANGE[0]),
            "--go",
        ]
    )
    assert result == 2
    assert stops == [True]


def test_execute_preflight_failure_without_launch_receipt_does_not_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {"plan_sha256": "sha256:preflight", "_private": {}}
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(canary, "build_canary_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        canary.executor,
        "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            canary.executor.ExecutorError("preflight rejected")
        ),
    )
    stops: list[bool] = []
    monkeypatch.setattr(
        canary.executor,
        "stop_execution",
        lambda *_args, **kwargs: stops.append(kwargs["go"]),
    )
    result = canary.main(
        [
            "run",
            "--lock",
            str(tmp_path / "lock"),
            "--render",
            str(tmp_path / "render"),
            "--hosts",
            str(tmp_path / "hosts"),
            "--receipt",
            str(receipt),
            "--canary-id",
            "shapecheck",
            "--base-seed",
            str(canary.contract.VAL_ONLY_SEED_RANGE[0]),
            "--go",
        ]
    )
    assert result == 2
    assert stops == []


def test_go_stops_exact_canary_when_runtime_receipt_persistence_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {"plan_sha256": "sha256:persistence-failure", "_private": {}}
    monkeypatch.setattr(canary, "build_canary_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        canary.executor,
        "execute",
        lambda *_args, **_kwargs: {"status": "launched"},
    )
    monkeypatch.setattr(
        canary,
        "attest_mps_runtime",
        lambda *_args, **_kwargs: {"status": "safe"},
    )
    monkeypatch.setattr(
        canary.executor,
        "_atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    stops: list[bool] = []
    monkeypatch.setattr(
        canary.executor,
        "stop_execution",
        lambda *_args, **kwargs: stops.append(kwargs["go"]),
    )
    receipt = tmp_path / "receipt.json"
    result = canary.main(
        [
            "run",
            "--lock",
            str(tmp_path / "lock"),
            "--render",
            str(tmp_path / "render"),
            "--hosts",
            str(tmp_path / "hosts"),
            "--receipt",
            str(receipt),
            "--canary-id",
            "shapecheck",
            "--base-seed",
            str(canary.contract.VAL_ONLY_SEED_RANGE[0]),
            "--go",
        ]
    )
    assert result == 2
    assert stops == [True]


def test_go_keyboard_interrupt_exact_stops_then_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {"plan_sha256": "sha256:keyboard-interrupt", "_private": {}}
    monkeypatch.setattr(canary, "build_canary_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        canary.executor,
        "execute",
        lambda *_args, **_kwargs: {"status": "launched"},
    )
    monkeypatch.setattr(
        canary,
        "attest_mps_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    stops: list[bool] = []
    monkeypatch.setattr(
        canary.executor,
        "stop_execution",
        lambda *_args, **kwargs: stops.append(kwargs["go"]),
    )
    with pytest.raises(KeyboardInterrupt):
        canary.main(
            [
                "run",
                "--lock",
                str(tmp_path / "lock"),
                "--render",
                str(tmp_path / "render"),
                "--hosts",
                str(tmp_path / "hosts"),
                "--receipt",
                str(tmp_path / "receipt"),
                "--canary-id",
                "shapecheck",
                "--base-seed",
                str(canary.contract.VAL_ONLY_SEED_RANGE[0]),
                "--go",
            ]
        )
    assert stops == [True]


def test_go_surfaces_primary_and_exact_stop_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = {"plan_sha256": "sha256:combined-failure", "_private": {}}
    monkeypatch.setattr(canary, "build_canary_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        canary.executor,
        "execute",
        lambda *_args, **_kwargs: {"status": "launched"},
    )
    monkeypatch.setattr(
        canary,
        "attest_mps_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("transport lost")),
    )
    monkeypatch.setattr(
        canary.executor,
        "stop_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            canary.executor.ExecutorError("host c1 unreachable")
        ),
    )
    result = canary.main(
        [
            "run",
            "--lock",
            str(tmp_path / "lock"),
            "--render",
            str(tmp_path / "render"),
            "--hosts",
            str(tmp_path / "hosts"),
            "--receipt",
            str(tmp_path / "receipt"),
            "--canary-id",
            "shapecheck",
            "--base-seed",
            str(canary.contract.VAL_ONLY_SEED_RANGE[0]),
            "--go",
        ]
    )
    assert result == 2
    error = capsys.readouterr().err
    assert "transport lost" in error
    assert "exact-stop also failed" in error
    assert "host c1 unreachable" in error


def test_audit_requires_clean_public_information_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, *_ = _fixture(tmp_path, monkeypatch)

    def fake_ssh(_hosts, _alias, command):
        encoded = command.split()[-1]
        # The shell-quoted JSON is easiest to recover from the expected plan,
        # while still proving one result is required per remote command.
        del encoded
        alias_commands = [
            item
            for lane in plan["_private"]["lanes"].values()
            for item in lane
            if item["host_alias"] == _alias
        ]
        payload = [
            {
                "job_id": item["job_id"],
                "rows": 10,
                "simulations": 128,
                "manifest_sha256": "sha256:m",
                "attestation_sha256": item["output_attestation"]["source_file_sha256"],
                "config_hash": item["config_provenance"]["config_hash"],
                "full_config_hash": item["config_provenance"]["full_config_hash"],
                "config_registry_sha256": "sha256:registry",
            }
            for item in alias_commands
        ]
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(canary.executor, "_ssh", fake_ssh)
    report = canary.audit_canary(plan)
    assert report["status"] == "PASS"
    assert report["job_count"] == 36
    assert report["rows"] == 360
    assert report["simulations"] == 36 * 128
    assert report["audit_sha256"].startswith("sha256:")


def test_remote_audit_script_rejects_hidden_truth_target_regime(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    attestation = tmp_path / "attestation.json"
    registry = tmp_path / "config_registry.jsonl"
    attestation.write_text('{"validation_only":true}\n', encoding="utf-8")
    registry.write_text(
        '{"config_hash":"generate-test","pipeline":"generate"}\n',
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "games_requested": 16,
                "games_completed": 16,
                "games_failed": 0,
                "errors": [],
                "base_seed": canary.contract.VAL_ONLY_SEED_RANGE[0],
                "rows": 100,
                "simulations_used_total": 1280,
                "target_information_regime": "authoritative_hidden_state_search_v1",
                "config_hash": "generate-test",
            }
        ),
        encoding="utf-8",
    )
    expected = [
        {
            "job_id": "unsafe",
            "manifest": str(manifest),
            "attestation": str(attestation),
            "attestation_sha256": canary._sha256(attestation),
            "config_registry": str(registry),
            "games": 16,
            "base_seed": canary.contract.VAL_ONLY_SEED_RANGE[0],
        }
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            canary.REMOTE_AUDIT_SCRIPT,
            json.dumps(expected),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unsafe target regime: unsafe" in (result.stderr or result.stdout)


def test_remote_audit_rejects_forged_minimal_config_registry(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    attestation = tmp_path / "attestation.json"
    registry = tmp_path / "config_registry.jsonl"
    provenance = _source_config_provenance(canary.contract.VAL_ONLY_SEED_RANGE[0])
    attestation.write_text('{"validation_only":true}\n', encoding="utf-8")
    registry.write_text(
        json.dumps(
            {
                "config_hash": provenance["config_hash"],
                "pipeline": "generate",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry.chmod(0o444)
    manifest.write_text(
        json.dumps(
            {
                "games_requested": 16,
                "games_completed": 16,
                "games_failed": 0,
                "errors": [],
                "base_seed": canary.contract.VAL_ONLY_SEED_RANGE[0],
                "rows": 100,
                "simulations_used_total": 1280,
                "target_information_regime": "public_conservation_pimc_v1",
                "config_hash": provenance["config_hash"],
            }
        ),
        encoding="utf-8",
    )
    expected = [
        {
            "job_id": "forged",
            "manifest": str(manifest),
            "attestation": str(attestation),
            "attestation_sha256": canary._sha256(attestation),
            "config_registry": str(registry),
            "config_provenance": provenance,
            "games": 16,
            "base_seed": canary.contract.VAL_ONLY_SEED_RANGE[0],
        }
    ]
    result = subprocess.run(
        [sys.executable, "-c", canary.REMOTE_AUDIT_SCRIPT, json.dumps(expected)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "registry fields mismatch: forged" in (result.stderr or result.stdout)
