from __future__ import annotations

import hashlib
import io
import json
import os
import resource
import shlex
import signal
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from tools import a1_pre_wave_contract as contract
from tools.fleet import a1_lane_supervisor as supervisor
from tools.fleet import a1_production_executor as executor


def _sha(path: Path) -> str:
    return executor._sha256(path)


def _fixture(
    tmp_path: Path, *, current_v3: bool = False
) -> tuple[Path, Path, dict, dict]:
    ledger = tmp_path / "seed-ledger.md"
    ledger.write_text("# ledger\n", encoding="utf-8")
    checkpoint = tmp_path / "champion.pt"
    checkpoint.write_bytes(b"checkpoint")
    mix_paths = {}
    mix_records = []
    for category in ("recent_history", "hard_negative"):
        path = tmp_path / "mix" / f"{category}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"category": category}) + "\n", encoding="utf-8")
        mix_paths[category] = path
        mix_records.append({"path": str(path), "sha256": _sha(path)})
    search = {
        "n_full": 128,
        "n_fast": 16,
        "p_full": 0.25,
        "c_visit": 50.0,
        "c_scale": 0.03,
        "rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.98,
        "wide_candidates_threshold": 24,
        "max_depth": 80,
        "symmetry_averaged_eval": True,
        "wide_roots_always_full": False,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": True,
        "exact_budget_sh": False,
        "exact_budget_sh_min_n": 0,
        "belief_chance_spectra": False,
        "information_set_search": True,
        "determinization_particles": 4,
        "determinization_min_simulations": 32,
        "n_full_wide": None,
        "n_full_wide_threshold": None,
        "raw_policy_above_width": None,
        "symmetry_averaged_eval_threshold": 20,
    }
    lock = {
        "schema_version": (
            contract.LOCK_SCHEMA if current_v3 else contract.LEGACY_LOCK_SCHEMA
        ),
        "contract_sha256": "sha256:" + "a" * 64,
        "science": {
            "search_operator": search,
            "evaluator": {
                "prior_temperature": 1.0,
                "value_scale": 1.0,
                "value_readout": "scalar",
                "cache_size": 0,
                "public_observation": True,
                "rust_featurize": False,
            },
        },
        "generation": {
            "workers_per_gpu": 16,
            "device": "cuda",
            "max_decisions": 600,
            "temperature_decisions": 90,
            "temperature_high": 1.0,
            "temperature_low": 0.0,
            "late_temperature": 0.0,
            "late_temperature_decisions": None,
            "track": "2p_no_trade",
            "vps_to_win": 10,
            "obs_width": 806,
            "shard_size": 512,
            "format": "npz",
            "eval_server": False,
        },
        "checkpoints": [
            {"id": "producer", "role": "producer", "path": str(checkpoint), "sha256": _sha(checkpoint)}
        ],
        "fleet": {"seed_ledger": {"path": str(ledger)}, "jobs": []},
        "game_contract": {
            **(
                {
                    "profile": contract.CURRENT_GAME_CONTRACT_PROFILE,
                    "worker_count": 64,
                    "job_count": 192,
                }
                if current_v3
                else {}
            ),
            "total_complete_games": 12_000,
            "category_games": {
                "current_producer": 9_600,
                "recent_history": 1_800,
                "hard_negative": 600,
            },
            "total_attempts": 12_512 if current_v3 else 12_320,
            "category_attempts": (
                {
                    "current_producer": 9_920,
                    "recent_history": 1_928,
                    "hard_negative": 664,
                }
                if current_v3
                else {
                    "current_producer": 9_800,
                    "recent_history": 1_880,
                    "hard_negative": 640,
                }
            ),
            "selection_rule": "lowest_seed_complete_per_job",
            "selection_before_row_expansion": True,
        },
    }
    commands = []
    categories = executor.CATEGORY_ORDER
    for lane_index in range(64 if current_v3 else 40):
        alias = f"h{lane_index // 4:02d}"
        gpu = lane_index % 4
        worker_id = f"{alias}_gpu{gpu}"
        previous = None
        for category_index, category in enumerate(categories):
            job_id = f"{worker_id}__{category}"
            base_seed = 10_000_000_000 + lane_index * 100_000 + category_index * 1_000
            job = {
                "job_id": job_id,
                "worker_id": worker_id,
                "host_alias": alias,
                "gpu": gpu,
                "category": category,
                "output_dir": str(tmp_path / "outputs" / job_id),
                "attempts": 100,
                "games": 100,
                "base_seed": base_seed,
                "seed_end": base_seed + 100,
                "claim_label": f"claim-{job_id}",
            }
            lock["fleet"]["jobs"].append(job)
            argv = contract._generator_argv(lock, job, mix_paths=mix_paths)
            attestation = tmp_path / "attestations" / f"{job_id}.json"
            attestation.parent.mkdir(parents=True, exist_ok=True)
            attestation.write_text(json.dumps({"job_id": job_id}) + "\n", encoding="utf-8")
            claim_row = contract._ledger_claim_row(lock, job)
            environment = contract._job_environment(lock, job)
            config_provenance = contract._expected_generate_config_provenance(
                lock,
                job,
                opponent_mix_manifest=(
                    None if category == "current_producer" else str(mix_paths[category])
                ),
            )
            commands.append(
                {
                    **{key: job[key] for key in ("job_id", "worker_id", "host_alias", "gpu", "category")},
                    "environment": environment,
                    "environment_sha256": contract._digest_value(environment),
                    "config_provenance": config_provenance,
                    "python": "python",
                    "argv": argv,
                    "argv_sha256": contract._digest_value(argv),
                    "ledger_claim": {"path": str(ledger), "row": claim_row, "row_sha256": contract._digest_value(claim_row)},
                    "output_attestation": {
                        "source": str(attestation),
                        "source_file_sha256": _sha(attestation),
                        "destination": str(Path(job["output_dir"]) / "a1_contract.json"),
                    },
                    "must_run_after": [] if previous is None else [previous],
                }
            )
            previous = job_id
    rendered = {
        "schema_version": contract.RENDER_SCHEMA,
        "contract_sha256": lock["contract_sha256"],
        "required_artifacts": {
            "seed_ledger": {"path": str(ledger), "sha256": _sha(ledger)},
            "rendered_opponent_mix": mix_records,
        },
        "commands": commands,
    }
    rendered["render_sha256"] = contract._digest_value(rendered)
    lock_path = tmp_path / "contract.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    render_path = tmp_path / "commands.json"
    render_path.write_text(json.dumps(rendered), encoding="utf-8")
    return lock_path, render_path, lock, rendered


def _verifier(lock: dict):
    def verify(_path: Path, *, require_all_job_claims: bool = False) -> dict:
        assert require_all_job_claims is True
        return lock

    return verify


def _hosts(tmp_path: Path, rendered: dict) -> Path:
    key = tmp_path / "id_ed25519"
    key.write_text("private", encoding="utf-8")
    path = tmp_path / "hosts.json"
    aliases = {command["host_alias"] for command in rendered["commands"]}
    path.write_text(
        json.dumps(
            {
                "schema_version": executor.HOST_SCHEMA,
                "ssh_user": "ubuntu",
                "ssh_key": str(key),
                "remote_root": "/home/ubuntu/a1-production",
                "python": "/usr/bin/python3",
                "hosts": {alias: f"10.0.0.{index + 1}" for index, alias in enumerate(sorted(aliases))},
            }
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return path


def test_remote_install_exact_hit_skips_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "artifact"
    source.write_bytes(b"exact")
    ssh_calls: list[str] = []
    scp_calls: list[tuple] = []

    def exact_precheck(_hosts: dict, _alias: str, command: str):
        ssh_calls.append(command)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(executor, "_ssh", exact_precheck)
    monkeypatch.setattr(
        executor, "_scp", lambda *args: scp_calls.append(args)
    )
    executor._remote_install(
        {
            "python": sys.executable,
            "remote_root": "/remote",
        },
        "h00",
        source,
        "/remote/exact",
        _sha(source),
    )

    assert len(ssh_calls) == 1
    assert not scp_calls


@pytest.mark.parametrize("kind", ["mismatch", "symlink"])
def test_remote_install_precheck_refuses_mismatch_and_symlink(
    tmp_path: Path, kind: str
) -> None:
    destination = tmp_path / "destination"
    expected_source = tmp_path / "expected"
    expected_source.write_bytes(b"expected")
    if kind == "mismatch":
        destination.write_bytes(b"different")
        message = "different bytes"
    else:
        target = tmp_path / "target"
        target.write_bytes(b"expected")
        destination.symlink_to(target)
        message = "regular non-symlink"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            executor._REMOTE_INSTALL_PRECHECK_SCRIPT,
            str(destination),
            _sha(expected_source),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_stage_files_are_alias_scoped_except_global_artifacts(tmp_path: Path) -> None:
    _lock_path, _render_path, lock, rendered = _fixture(tmp_path)
    required = rendered["required_artifacts"] | {
        "checkpoints": lock["checkpoints"]
    }
    lanes = {
        "h00_gpu0": rendered["commands"][:3],
        "h01_gpu0": rendered["commands"][12:15],
    }

    staged = executor._stage_files_by_alias(required, lanes)
    by_alias = {
        alias: {destination for _source, destination, _digest in records}
        for alias, records in staged.items()
    }
    globals_expected = {
        *(item["path"] for item in required["checkpoints"]),
        *(item["path"] for item in required["rendered_opponent_mix"]),
    }
    h00_attestations = {
        command["output_attestation"]["source"] for command in lanes["h00_gpu0"]
    }
    h01_attestations = {
        command["output_attestation"]["source"] for command in lanes["h01_gpu0"]
    }

    assert by_alias["h00"] == globals_expected | h00_attestations
    assert by_alias["h01"] == globals_expected | h01_attestations
    assert by_alias["h00"].isdisjoint(h01_attestations)
    assert by_alias["h01"].isdisjoint(h00_attestations)


def test_repo_artifact_plan_binds_fixed_executor_and_supervisor() -> None:
    supervisor_path = Path(supervisor.__file__).resolve()
    rendered = {
        "required_artifacts": {
            "guard_config": {
                "path": str(supervisor_path),
                "sha256": _sha(supervisor_path),
            },
            "generator_code": [],
            "runtime_code_tree": [],
        }
    }

    artifacts = {
        record["path"]: record for record in executor._repo_artifacts(rendered)
    }

    assert artifacts["tools/fleet/a1_lane_supervisor.py"]["sha256"] == _sha(
        supervisor_path
    )
    assert artifacts["tools/fleet/a1_production_executor.py"]["sha256"] == _sha(
        Path(executor.__file__).resolve()
    )


def _historical_repo_pair(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    historical = tmp_path / "catan-db1c8b1-campaign"
    current = tmp_path / "current"
    relative = Path("configs/guards/generate_gumbel_selfplay_data.json")
    for root in (historical, current):
        artifact = root / relative
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"identical-runtime-bytes")
    for source in (
        Path(executor.__file__).resolve(),
        Path(supervisor.__file__).resolve(),
        executor._REPO_ROOT / "tools/fleet/a1_stop_helper.py",
    ):
        destination = current / source.relative_to(executor._REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return historical, current, relative, _sha(historical / relative)


def test_live_shaped_db1_runtime_paths_relocate_to_identical_current_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    historical, current, relative, digest = _historical_repo_pair(tmp_path)
    campaign = historical / "configs/operations/a1-dual-arm-56gpu-20260710/contract.json"
    monkeypatch.setattr(executor, "HISTORICAL_DB1_REPO_ROOT", historical)
    monkeypatch.setattr(executor, "HISTORICAL_DB1_CAMPAIGN_PATH", campaign)
    lock = {
        "source_campaign": {
            "path": str(campaign),
            "sha256": contract.HISTORICAL_DB1_CAMPAIGN_FILE_SHA256,
        },
        "provenance": {
            "executor": {
                "kind": "executor",
                "path": "tools/fleet/a1_production_executor.py",
                "sha256": contract.HISTORICAL_DB1_EXECUTOR_SHA256,
            }
        },
    }
    historical_fixed_records = []
    for fixed_relative in (
        Path("tools/fleet/a1_lane_supervisor.py"),
        Path("tools/fleet/a1_production_executor.py"),
    ):
        frozen_fixed = historical / fixed_relative
        frozen_fixed.parent.mkdir(parents=True, exist_ok=True)
        frozen_fixed.write_bytes(b"historical-sealed-control-plane")
        historical_fixed_records.append(
            {"path": str(frozen_fixed), "sha256": _sha(frozen_fixed)}
        )
    rendered = {
        "required_artifacts": {
            "guard_config": {
                "path": str(historical / relative),
                "sha256": digest,
            },
            "generator_code": [],
            "runtime_code_tree": historical_fixed_records,
        }
    }

    artifacts = executor._repo_artifacts(
        rendered,
        repo_root=current,
        historical_root=executor._historical_runtime_root(lock),
    )
    by_path = {record["path"]: record for record in artifacts}

    assert by_path[str(relative)]["sha256"] == digest
    assert by_path["tools/fleet/a1_lane_supervisor.py"]["sha256"] == _sha(
        current / "tools/fleet/a1_lane_supervisor.py"
    )
    assert by_path["tools/fleet/a1_production_executor.py"]["sha256"] == _sha(
        current / "tools/fleet/a1_production_executor.py"
    )

    (current / relative).write_bytes(b"changed-current-control-plane")
    artifacts = executor._repo_artifacts(
        rendered,
        repo_root=current,
        historical_root=executor._historical_runtime_root(lock),
    )
    by_path = {record["path"]: record for record in artifacts}
    assert by_path[str(relative)]["source_path"] == str(historical / relative)
    assert "source_path" not in by_path["tools/fleet/a1_lane_supervisor.py"]
    assert "source_path" not in by_path["tools/fleet/a1_production_executor.py"]
    archive = tmp_path / "repo.tar"
    executor._build_repo_tar(
        artifacts,
        executor._repo_files(artifacts, repo_root=current),
        archive,
    )
    with tarfile.open(archive) as staged:
        assert staged.extractfile(str(relative)).read() == b"identical-runtime-bytes"


@pytest.mark.parametrize(
    "mutation", ["historical_mismatch", "symlink", "current_symlink", "other_root"]
)
def test_db1_runtime_relocation_rejects_drift_symlinks_and_other_roots(
    tmp_path: Path, mutation: str
) -> None:
    historical, current, relative, digest = _historical_repo_pair(tmp_path)
    source = historical / relative
    if mutation == "historical_mismatch":
        source.write_bytes(b"drift")
    elif mutation == "symlink":
        source.unlink()
        source.symlink_to(current / relative)
    elif mutation == "current_symlink":
        current_source = current / relative
        current_source.unlink()
        current_source.symlink_to(source)
    elif mutation == "other_root":
        source = tmp_path / "other" / relative
        source.parent.mkdir(parents=True)
        source.write_bytes(b"identical-runtime-bytes")

    with pytest.raises(executor.ExecutorError):
        executor._relocate_historical_artifact(
            {"path": str(source), "sha256": digest},
            historical_root=historical,
            current_root=current,
        )


def test_dry_plan_is_exact_40_lane_120_job_n128_mps_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, render_path, lock, rendered = _fixture(tmp_path)
    hosts = _hosts(tmp_path, rendered)
    monkeypatch.setattr(
        executor, "_repo_artifacts", lambda _rendered, **_kwargs: []
    )
    plan = executor.build_plan(
        lock_path=lock_path,
        render_path=render_path,
        hosts_path=hosts,
        receipt_path=tmp_path / "receipt.json",
        verify_lock_fn=_verifier(lock),
    )
    assert plan["lane_count"] == 40
    assert plan["job_count"] == plan["claim_count"] == 120
    assert plan["client_environment"] == executor.CLIENT_ENVIRONMENT
    assert plan["category_order"] == list(executor.CATEGORY_ORDER)
    assert all("--resume" in command["argv"] for command in rendered["commands"])
    assert all(
        command["environment"]["CATAN_ZERO_CONFIG_REGISTRY"]
        == str(Path(command["argv"][command["argv"].index("--out-dir") + 1]) / "config_registry.jsonl")
        for command in rendered["commands"]
    )
    assert all(command["argv"][command["argv"].index("--n-full") + 1] == "128" for command in rendered["commands"])
    assert not any(flag in command["argv"] for command in rendered["commands"] for flag in executor.FORBIDDEN_ADAPTIVE_ARGV)


def test_current_v3_dry_plan_is_exact_64_lane_192_job_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, render_path, lock, rendered = _fixture(tmp_path, current_v3=True)
    hosts = _hosts(tmp_path, rendered)
    monkeypatch.setattr(executor, "_repo_artifacts", lambda _rendered, **_kwargs: [])

    plan = executor.build_plan(
        lock_path=lock_path,
        render_path=render_path,
        hosts_path=hosts,
        receipt_path=tmp_path / "receipt.json",
        verify_lock_fn=_verifier(lock),
    )

    assert plan["lane_count"] == 64
    assert plan["job_count"] == plan["claim_count"] == 192


def test_dual_arm_n256_profile_is_exact_28_lane_84_job_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, render_path, lock, rendered = _fixture(tmp_path)
    lock["schema_version"] = contract.GENERATION_ARM_LOCK_SCHEMA
    lock["game_contract"] = {
        "profile": "dual_arm_generation_v1",
        "arm_id": "n256",
        "worker_count": 28,
        "job_count": 84,
    }
    lock["science"]["search_operator"]["n_full"] = 256
    lock["science"]["search_operator"]["prior_temperature"] = 1.0
    lock["science"]["search_operator_sha256"] = contract._digest_value(  # noqa: SLF001
        lock["science"]["search_operator"]
    )
    lock["science"]["effective_search_config_sha256"] = "sha256:" + "1" * 64
    lock["science"]["evaluator_sha256"] = "sha256:" + "2" * 64
    lock["science"]["value_readout"] = "scalar"
    lock["provenance"] = {
        "guard_configs": {
            category: {"path": "configs/guards/a1_generation_n256.json"}
            for category in executor.CATEGORY_ORDER
        },
        "runtime_code_tree_sha256": "sha256:" + "3" * 64,
    }
    lock["fleet"]["jobs"] = lock["fleet"]["jobs"][:84]
    lock["fleet"]["seed_plan_sha256"] = contract._digest_value(  # noqa: SLF001
        lock["fleet"]["jobs"]
    )
    lock["checkpoints"].extend(
        [
            {**lock["checkpoints"][0], "id": "history", "role": "history"},
            {
                **lock["checkpoints"][0],
                "id": "hard-negative",
                "role": "hard_negative",
            },
        ]
    )
    lock["source_categories"] = [
        {"name": "current_producer", "mode": "self", "checkpoint_ids": []},
        {
            "name": "recent_history",
            "mode": "checkpoint_list",
            "checkpoint_ids": ["history"],
        },
        {
            "name": "hard_negative",
            "mode": "checkpoint_list",
            "checkpoint_ids": ["hard-negative"],
        },
    ]
    mix_paths = {
        Path(item["path"]).stem: Path(item["path"])
        for item in rendered["required_artifacts"]["rendered_opponent_mix"]
    }
    rendered["commands"] = rendered["commands"][:84]
    for job, command in zip(lock["fleet"]["jobs"], rendered["commands"]):
        job["arm_id"] = "n256"
        job["c_scale"] = 0.1
        command["arm_id"] = "n256"
        command["argv"] = contract._generator_argv(  # noqa: SLF001
            lock, job, mix_paths=mix_paths
        )
        command["argv_sha256"] = contract._digest_value(command["argv"])  # noqa: SLF001
        command["config_provenance"] = contract._expected_generate_config_provenance(  # noqa: SLF001
            lock,
            job,
            opponent_mix_manifest=(
                None
                if job["category"] == "current_producer"
                else str(mix_paths[job["category"]])
            ),
        )
        attestation = contract._job_attestation(lock, job)  # noqa: SLF001
        source = Path(command["output_attestation"]["source"])
        source.write_text(json.dumps(attestation))
        command["output_attestation"]["source_file_sha256"] = _sha(source)
        command["output_attestation"]["payload_sha256"] = contract._digest_value(  # noqa: SLF001
            attestation
        )
    rendered["render_sha256"] = contract._digest_value(  # noqa: SLF001
        {key: value for key, value in rendered.items() if key != "render_sha256"}
    )
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    render_path.write_text(json.dumps(rendered), encoding="utf-8")
    hosts = _hosts(tmp_path, rendered)
    monkeypatch.setattr(executor, "_repo_artifacts", lambda _rendered, **_kwargs: [])

    plan = executor.build_plan(
        lock_path=lock_path,
        render_path=render_path,
        hosts_path=hosts,
        receipt_path=tmp_path / "receipt.json",
        verify_lock_fn=_verifier(lock),
    )

    assert plan["lane_count"] == 28
    assert plan["job_count"] == plan["claim_count"] == 84
    assert all(command["arm_id"] == "n256" for command in rendered["commands"])
    assert all(
        command["argv"][command["argv"].index("--n-full") + 1] == "256"
        for command in rendered["commands"]
    )


def test_private_host_config_and_render_environment_fail_closed(tmp_path: Path) -> None:
    lock_path, render_path, lock, rendered = _fixture(tmp_path)
    hosts = _hosts(tmp_path, rendered)
    os.chmod(hosts, 0o644)
    with pytest.raises(executor.ExecutorError, match="mode 0600"):
        executor.load_hosts(hosts, {"h00"})
    os.chmod(hosts, 0o600)
    rendered["commands"][0]["environment"]["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/wrong"
    rendered.pop("render_sha256")
    rendered["render_sha256"] = contract._digest_value(rendered)
    render_path.write_text(json.dumps(rendered), encoding="utf-8")
    with pytest.raises(executor.ExecutorError, match="client environment drift"):
        executor.verify_render(lock_path, render_path, verify_lock_fn=_verifier(lock))


def test_executor_refuses_promoted_checkpoint_under_different_c_scale(
    tmp_path: Path,
) -> None:
    lock_path, render_path, lock, _rendered = _fixture(tmp_path)
    producer = contract._producer(lock)  # noqa: SLF001
    deployed = {"c_scale": 0.10, "n_full": 128}
    lock["promotion_handoff"] = {
        "mode": contract.POST_PROMOTION_HANDOFF_MODE,
        "document_schema": contract.promotion_handoff.HANDOFF_SCHEMA,
        "producer_checkpoint": {
            "path": producer["path"],
            "sha256": producer["sha256"],
        },
        "producer_identity_sha256": "sha256:" + "2" * 64,
        "producer_search_config": deployed,
        "producer_search_config_sha256": contract._digest_value(deployed),
    }
    with pytest.raises(
        executor.ExecutorError,
        match=r"unsafe promoted producer identity.*executes c_scale=0\.03",
    ):
        executor.verify_render(lock_path, render_path, verify_lock_fn=_verifier(lock))


def test_render_environment_digest_and_per_job_registry_fail_closed(
    tmp_path: Path,
) -> None:
    lock_path, render_path, lock, rendered = _fixture(tmp_path)
    command = rendered["commands"][0]
    command["environment"]["CATAN_ZERO_CONFIG_REGISTRY"] = str(
        tmp_path / "shared-or-production-registry.jsonl"
    )
    command["environment_sha256"] = contract._digest_value(command["environment"])
    rendered.pop("render_sha256")
    rendered["render_sha256"] = contract._digest_value(rendered)
    render_path.write_text(json.dumps(rendered), encoding="utf-8")
    with pytest.raises(executor.ExecutorError, match="client environment drift"):
        executor.verify_render(lock_path, render_path, verify_lock_fn=_verifier(lock))

    command["environment"] = contract._job_environment(
        lock, lock["fleet"]["jobs"][0]
    )
    command["environment_sha256"] = "sha256:" + "0" * 64
    rendered.pop("render_sha256")
    rendered["render_sha256"] = contract._digest_value(rendered)
    render_path.write_text(json.dumps(rendered), encoding="utf-8")
    with pytest.raises(executor.ExecutorError, match="environment digest mismatch"):
        executor.verify_render(lock_path, render_path, verify_lock_fn=_verifier(lock))


def _lane(tmp_path: Path, commands: list[dict]) -> tuple[Path, dict]:
    lock_copy = tmp_path / "remote" / "contract.lock.json"
    render_copy = tmp_path / "remote" / "commands.json"
    lock_copy.parent.mkdir(parents=True, exist_ok=True)
    lock_copy.write_text("lock\n", encoding="utf-8")
    render_copy.write_text("render\n", encoding="utf-8")
    materialized = [
        executor._materialize_job_environment(command, repo_dir=str(tmp_path))
        for command in commands
    ]
    lane = {
        "schema_version": supervisor.SCHEMA,
        "worker_id": commands[0]["worker_id"],
        "host_alias": commands[0]["host_alias"],
        "gpu": commands[0]["gpu"],
        "repo_dir": str(tmp_path),
        "python": sys.executable,
        "receipt_dir": str(tmp_path / "receipts"),
        "quarantine_dir": str(tmp_path / "quarantine"),
        "log_dir": str(tmp_path / "logs"),
        "lane_lock": str(tmp_path / "lane.lock"),
        "client_environment": dict(supervisor.CLIENT_ENVIRONMENT),
        "operator_manifests": {
            "lock": {"path": str(lock_copy), "sha256": _sha(lock_copy)},
            "render": {"path": str(render_copy), "sha256": _sha(render_copy)},
        },
        "commands": materialized,
    }
    lane["lane_sha256"] = supervisor._digest(lane)
    path = tmp_path / "lane.json"
    path.write_text(json.dumps(lane), encoding="utf-8")
    return path, lane


def _arm_lane(tmp_path: Path, arm_id: str) -> tuple[Path, dict]:
    _lock_path, _render_path, _lock, rendered = _fixture(tmp_path)
    commands = rendered["commands"][:3]
    for command in commands:
        command["arm_id"] = arm_id
        argv = command["argv"]
        argv[argv.index("--n-full") + 1] = str(supervisor.ARM_N_FULL[arm_id])
        argv.extend(["--generation-arm-id", arm_id])
        command["argv_sha256"] = supervisor._digest(argv)
    return _lane(tmp_path, commands)


@pytest.mark.parametrize("arm_id", ["n128", "n256"])
def test_supervisor_accepts_live_shaped_arm_lane(
    tmp_path: Path, arm_id: str
) -> None:
    lane_path, _lane_payload = _arm_lane(tmp_path, arm_id)

    loaded = supervisor.load_lane(lane_path)

    assert {command["arm_id"] for command in loaded["commands"]} == {arm_id}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_command_arm", "requires one command arm_id"),
        ("missing_argv_arm", "requires one command arm_id"),
        ("unknown_arm", "unknown generation arm"),
        ("mismatched_markers", "generation arms mismatch"),
        ("mixed_jobs", "mixes generation arms"),
        ("wrong_budget", "requires --n-full 256"),
        ("adaptive_override", "forbid adaptive/wide overrides"),
    ],
)
def test_supervisor_rejects_malformed_live_shaped_arm_lane(
    tmp_path: Path, mutation: str, message: str
) -> None:
    lane_path, lane = _arm_lane(tmp_path, "n256")
    command = lane["commands"][1]
    argv = command["argv"]
    arm_index = argv.index("--generation-arm-id")
    if mutation == "missing_command_arm":
        command.pop("arm_id")
    elif mutation == "missing_argv_arm":
        del argv[arm_index : arm_index + 2]
    elif mutation == "unknown_arm":
        command["arm_id"] = "n512"
        argv[arm_index + 1] = "n512"
    elif mutation == "mismatched_markers":
        argv[arm_index + 1] = "n128"
    elif mutation == "mixed_jobs":
        command["arm_id"] = "n128"
        argv[arm_index + 1] = "n128"
        argv[argv.index("--n-full") + 1] = "128"
    elif mutation == "wrong_budget":
        argv[argv.index("--n-full") + 1] = "128"
    elif mutation == "adaptive_override":
        argv.extend(["--n-full-wide", "512"])
    command["argv_sha256"] = supervisor._digest(argv)
    lane["lane_sha256"] = supervisor._digest(
        {key: value for key, value in lane.items() if key != "lane_sha256"}
    )
    lane_path.write_text(json.dumps(lane), encoding="utf-8")

    with pytest.raises(supervisor.SupervisorError, match=message):
        supervisor.load_lane(lane_path)


def _complete_output(command: dict) -> None:
    argv = command["argv"]
    out = Path(argv[argv.index("--out-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    attempts = int(argv[argv.index("--games") + 1])
    base_seed = int(argv[argv.index("--base-seed") + 1])
    provenance = command["config_provenance"]
    (out / "manifest.json").write_text(
        json.dumps({"games_requested": attempts, "games_completed": attempts, "games_failed": 0, "errors": [], "base_seed": base_seed, "config_hash": provenance["config_hash"]}),
        encoding="utf-8",
    )
    registry = Path(command["environment"]["CATAN_ZERO_CONFIG_REGISTRY"])
    registry.write_text(
        json.dumps(
            {
                "config_hash": provenance["config_hash"],
                "full_config_hash": provenance["full_config_hash"],
                "pipeline": "generate",
                "timestamp": "2026-07-10T00:00:00+00:00",
                "purpose": "test",
                "config": provenance["config"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = Path(command["output_attestation"]["source"])
    destination = Path(command["output_attestation"]["destination"])
    if not destination.exists():
        destination.write_bytes(source.read_bytes())
    else:
        assert destination.read_bytes() == source.read_bytes()


def test_completed_receipts_are_validated_and_never_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lock_path, _render_path, _lock, rendered = _fixture(tmp_path)
    commands = rendered["commands"][:3]
    lane_path, lane = _lane(tmp_path, commands)
    Path(lane["receipt_dir"]).mkdir(parents=True)
    for command in lane["commands"]:
        _complete_output(command)
        completed = supervisor._validate_completed(command)
        receipt = {
            "schema_version": supervisor.RECEIPT_SCHEMA,
            "job_id": command["job_id"],
            "lane_sha256": lane["lane_sha256"],
            "argv_sha256": command["argv_sha256"],
            "status": "complete",
            **completed,
        }
        Path(lane["receipt_dir"], f"{command['job_id']}.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_a, **_k: pytest.fail("completed job reran"))
    assert supervisor.run_lane(lane_path)["status"] == "complete"


def test_incomplete_lane_runs_exact_resume_sequentially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    monkeypatch.setenv("PYTHONHOME", "/unsealed/ambient-python")
    _lock_path, _render_path, _lock, rendered = _fixture(tmp_path)
    lane_path, lane = _lane(tmp_path, rendered["commands"][:3])
    calls = []

    class Process:
        pid = 424242

        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs["env"], kwargs))
            command = next(item for item in lane["commands"] if item["argv"] == argv[1:])
            _complete_output(command)

        def wait(self):
            return 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", Process)
    assert supervisor.run_lane(lane_path)["status"] == "complete"
    assert len(calls) == 3
    assert all("--resume" in argv for argv, _environment, _kwargs in calls)
    assert all(environment["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/mps_pipe_host" for _argv, environment, _kwargs in calls)
    assert all(
        environment["CATAN_ZERO_CONFIG_REGISTRY"].endswith("/config_registry.jsonl")
        for _argv, environment, _kwargs in calls
    )
    assert all(kwargs.get("pass_fds") for _argv, _environment, kwargs in calls)
    assert all(environment == command["environment"] for (_argv, environment, _kwargs), command in zip(calls, lane["commands"], strict=True))
    assert all("NVIDIA_TF32_OVERRIDE" not in environment for _argv, environment, _kwargs in calls)
    assert all("PYTHONHOME" not in environment for _argv, environment, _kwargs in calls)
    assert [command["category"] for command in lane["commands"]] == list(supervisor.CATEGORY_ORDER)


def test_completed_job_rejects_forged_minimal_registry_record(tmp_path: Path) -> None:
    _lock_path, _render_path, _lock, rendered = _fixture(tmp_path)
    _lane_path, lane = _lane(tmp_path, rendered["commands"][:3])
    command = lane["commands"][0]
    _complete_output(command)
    registry = Path(command["environment"]["CATAN_ZERO_CONFIG_REGISTRY"])
    registry.write_text(
        json.dumps(
            {
                "pipeline": "generate",
                "config_hash": command["config_provenance"]["config_hash"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(supervisor.SupervisorError, match="registry fields drifted"):
        supervisor._validate_completed(command)
    assert stat.S_IMODE(registry.stat().st_mode) == 0o444


def test_completed_receipt_rejects_post_completion_registry_mutation(
    tmp_path: Path,
) -> None:
    _lock_path, _render_path, _lock, rendered = _fixture(tmp_path)
    _lane_path, lane = _lane(tmp_path, rendered["commands"][:3])
    command = lane["commands"][0]
    _complete_output(command)
    completed = supervisor._validate_completed(command)
    receipt_path = Path(lane["receipt_dir"]) / f"{command['job_id']}.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": supervisor.RECEIPT_SCHEMA,
                "job_id": command["job_id"],
                "lane_sha256": lane["lane_sha256"],
                "argv_sha256": command["argv_sha256"],
                "status": "complete",
                **completed,
            }
        ),
        encoding="utf-8",
    )
    registry = Path(command["environment"]["CATAN_ZERO_CONFIG_REGISTRY"])
    registry.chmod(0o644)
    record = json.loads(registry.read_text(encoding="utf-8"))
    record["purpose"] = "mutated-after-completion"
    registry.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(supervisor.SupervisorError, match="config_registry_sha256 drift"):
        supervisor._run_job(lane, command)


def test_lane_rejects_registry_escape_even_with_rehashed_environment(
    tmp_path: Path,
) -> None:
    _lock_path, _render_path, _lock, rendered = _fixture(tmp_path)
    commands = rendered["commands"][:3]
    commands[0]["environment"]["CATAN_ZERO_CONFIG_REGISTRY"] = str(
        tmp_path / "shared.jsonl"
    )
    commands[0]["environment_sha256"] = supervisor._digest(
        commands[0]["environment"]
    )
    lane_path, _lane_payload = _lane(tmp_path, commands)
    with pytest.raises(supervisor.SupervisorError, match="sealed inside its output"):
        supervisor.load_lane(lane_path)


def test_completed_output_without_receipt_refuses(tmp_path: Path) -> None:
    _lock_path, _render_path, _lock, rendered = _fixture(tmp_path)
    lane_path, lane = _lane(tmp_path, rendered["commands"][:3])
    loaded = supervisor.load_lane(lane_path)
    _complete_output(loaded["commands"][0])
    with pytest.raises(supervisor.SupervisorError, match="without O_EXCL receipt"):
        supervisor._run_job(loaded, loaded["commands"][0])


def test_exact_incomplete_attempt_is_quarantined_before_fresh_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lock_path, _render_path, _lock, rendered = _fixture(tmp_path)
    lane_path, lane = _lane(tmp_path, rendered["commands"][:3])
    command = lane["commands"][0]
    receipt_dir = Path(lane["receipt_dir"])
    receipt_dir.mkdir(parents=True)
    receipt = {
        "schema_version": supervisor.RECEIPT_SCHEMA,
        "job_id": command["job_id"],
        "lane_sha256": lane["lane_sha256"],
        "argv_sha256": command["argv_sha256"],
        "status": "failed",
        "attempts": 1,
    }
    (receipt_dir / f"{command['job_id']}.json").write_text(json.dumps(receipt), encoding="utf-8")
    out = Path(command["argv"][command["argv"].index("--out-dir") + 1])
    out.mkdir(parents=True)
    (out / "partial.bin").write_bytes(b"forensic")

    class Process:
        pid = 424243

        def __init__(self, argv, **kwargs):
            assert kwargs["pass_fds"]
            _complete_output(command)

        def wait(self):
            return 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", Process)
    result = supervisor._run_job(supervisor.load_lane(lane_path), command)
    assert result["status"] == "complete"
    preserved = list(Path(lane["quarantine_dir"]).glob(f"{command['job_id']}.attempt-*"))
    preserved = [path for path in preserved if not path.name.endswith(".receipt.json")]
    assert len(preserved) == 1
    assert (preserved[0] / "partial.bin").read_bytes() == b"forensic"
    sidecar = Path(str(preserved[0]) + ".receipt.json")
    assert json.loads(sidecar.read_text())["status"] == "complete"


def test_invalid_manifest_is_forensic_then_replayed_not_stuck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lock_path, _render_path, _lock, rendered = _fixture(tmp_path)
    lane_path, lane = _lane(tmp_path, rendered["commands"][:3])
    command = lane["commands"][0]
    receipt_dir = Path(lane["receipt_dir"])
    receipt_dir.mkdir(parents=True)
    receipt = {
        "schema_version": supervisor.RECEIPT_SCHEMA,
        "job_id": command["job_id"],
        "lane_sha256": lane["lane_sha256"],
        "argv_sha256": command["argv_sha256"],
        "status": "running",
        "attempts": 1,
    }
    (receipt_dir / f"{command['job_id']}.json").write_text(json.dumps(receipt), encoding="utf-8")
    _complete_output(command)
    manifest = Path(command["argv"][command["argv"].index("--out-dir") + 1]) / "manifest.json"
    broken = json.loads(manifest.read_text())
    broken["games_failed"] = 1
    manifest.write_text(json.dumps(broken), encoding="utf-8")

    class Process:
        pid = 424244

        def __init__(self, argv, **kwargs):
            _complete_output(command)

        def wait(self):
            return 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", Process)
    assert supervisor._run_job(supervisor.load_lane(lane_path), command)["status"] == "complete"
    forensic_manifests = list(Path(lane["quarantine_dir"]).glob("*/manifest.json"))
    assert len(forensic_manifests) == 1
    assert json.loads(forensic_manifests[0].read_text())["games_failed"] == 1


def test_append_only_ledger_update_contract() -> None:
    assert executor._append_only_bytes(b"prefix\n", b"prefix\nclaim\n") == b"prefix\nclaim\n"
    assert executor._append_only_bytes(b"live\n", b"live\n") == b"live\n"
    with pytest.raises(executor.ExecutorError, match="not an exact prefix"):
        executor._append_only_bytes(b"diverged\n", b"prefix\nclaim\n")


def test_ssh_propagates_hard_command_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed.update({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    result = executor._ssh(
        {
            "ssh_key": "/tmp/key",
            "ssh_user": "ubuntu",
            "hosts": {"c1": "192.0.2.1"},
        },
        "c1",
        "true",
        timeout_seconds=17.5,
    )
    assert result.returncode == 0
    assert observed["timeout"] == 17.5


def test_supervisor_launch_intent_is_durable_before_detached_spawn() -> None:
    source = Path(executor.__file__).read_text(encoding="utf-8")
    execute_start = source.index("def execute(")
    pending = source.index('"launch_pending_worker_id": worker_id', execute_start)
    spawn = source.index("result = _ssh(hosts, alias, launch)", pending)
    pid = source.index("lane_pids[worker_id] = int", spawn)
    clear = source.index('receipt.pop("launch_pending_worker_id", None)', pid)
    assert pending < spawn < pid < clear


def test_resume_refuses_unresolved_pending_supervisor_launch(tmp_path: Path) -> None:
    public = {"plan_sha256": "sha256:pending"}
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": executor.RECEIPT_SCHEMA,
                "plan_sha256": public["plan_sha256"],
                "status": "launching",
                "lane_pids": {},
                "launch_pending_worker_id": "c1_gpu0",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(executor.ExecutorError, match="unresolved pending"):
        executor._resume_receipt(receipt, public, resume=True)


def test_exact_stop_ssh_is_hard_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor,
        "_ssh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("ssh", executor.STOP_SSH_TIMEOUT_SECONDS)
        ),
    )
    plan = {
        "repo_artifacts_sha256": "sha256:repo",
        "_private": {
            "hosts": {
                "remote_root": "/remote",
                "python": "/venv/bin/python",
            }
        },
    }
    with pytest.raises(executor.ExecutorError, match="stop timed out"):
        executor._stop_helper_call(
            plan,
            "c1_gpu0",
            [{"host_alias": "c1"}],
            action="stop",
            supervisor_pid=0,
        )


def test_preflight_accepts_only_exact_report(monkeypatch: pytest.MonkeyPatch) -> None:
    report = {
        "gpu_indices": [0, 1, 2, 3],
        "compute_apps": "mps_only_or_empty",
        "mps_active": "active",
        "mps_enabled": "enabled",
        "mps_main_pid": 123,
        "mps_unit_sha256": executor._sha256(executor.MPS_UNIT_PATH),
        "mps_limit_nofile_soft": executor.REQUIRED_NOFILE_SOFT,
        "client_environment": dict(executor.CLIENT_ENVIRONMENT),
        "python": "/venv/bin/python",
        "torch_version": "x",
        "torch_cuda_version": "x",
        "catanatron_rs_version": executor.NATIVE_WHEEL_VERSION,
        "native_wheel_sha256": executor._native_wheel_release_identity()["sha256"],
        "native_mcts_capabilities": sorted(executor.NATIVE_REQUIRED_CAPABILITIES),
        "required_nofile_soft": executor.REQUIRED_NOFILE_SOFT,
        "nofile_soft_before": 1024,
        "nofile_soft": executor.REQUIRED_NOFILE_SOFT,
        "nofile_hard": 1_048_576,
    }
    monkeypatch.setattr(
        executor,
        "_ssh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, json.dumps(report), ""),
    )
    assert executor._preflight_host({"python": "/venv/bin/python"}, "h00", [0, 1, 2, 3]) == report
    report["client_environment"] = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/wrong"}
    with pytest.raises(executor.ExecutorError, match="environment drift"):
        executor._preflight_host({"python": "/venv/bin/python"}, "h00", [0, 1, 2, 3])
    report["client_environment"] = dict(executor.CLIENT_ENVIRONMENT)
    report["mps_unit_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(executor.ExecutorError, match="unit digest drift"):
        executor._preflight_host({"python": "/venv/bin/python"}, "h00", [0, 1, 2, 3])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("catanatron_rs_version", "0.1.4", "version drift"),
        ("native_wheel_sha256", "sha256:" + "0" * 64, "wheel digest drift"),
        ("native_mcts_capabilities", [], "capability drift"),
        ("native_mcts_capabilities", "not-a-list", "capability drift"),
    ],
)
def test_preflight_fails_closed_on_native_runtime_report(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object, message: str
) -> None:
    report = {
        "gpu_indices": [0, 1, 2, 3],
        "compute_apps": "mps_only_or_empty",
        "mps_active": "active",
        "mps_enabled": "enabled",
        "mps_main_pid": 123,
        "mps_unit_sha256": executor._sha256(executor.MPS_UNIT_PATH),
        "mps_limit_nofile_soft": executor.REQUIRED_NOFILE_SOFT,
        "client_environment": dict(executor.CLIENT_ENVIRONMENT),
        "python": "/venv/bin/python",
        "torch_version": "x",
        "torch_cuda_version": "x",
        "catanatron_rs_version": executor.NATIVE_WHEEL_VERSION,
        "native_wheel_sha256": executor._native_wheel_release_identity()["sha256"],
        "native_mcts_capabilities": sorted(executor.NATIVE_REQUIRED_CAPABILITIES),
        "required_nofile_soft": executor.REQUIRED_NOFILE_SOFT,
        "nofile_soft_before": 1024,
        "nofile_soft": executor.REQUIRED_NOFILE_SOFT,
        "nofile_hard": 1_048_576,
    }
    report[field] = value
    monkeypatch.setattr(
        executor,
        "_ssh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps(report), ""
        ),
    )
    with pytest.raises(executor.ExecutorError, match=message):
        executor._preflight_host(
            {"python": "/venv/bin/python"}, "h00", [0, 1, 2, 3]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("required_nofile_soft", 1024, "required soft RLIMIT_NOFILE drift"),
        ("nofile_soft", 1024, "soft RLIMIT_NOFILE"),
        ("nofile_hard", 1024, "hard RLIMIT_NOFILE"),
        ("nofile_soft", "65536", "invalid RLIMIT_NOFILE report"),
        ("mps_limit_nofile_soft", 1024, "MPS LimitNOFILESoft"),
        ("mps_limit_nofile_soft", "65536", "invalid MPS LimitNOFILESoft report"),
    ],
)
def test_preflight_fails_closed_on_nofile_report(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object, message: str
) -> None:
    report = {
        "gpu_indices": [0, 1, 2, 3],
        "compute_apps": "mps_only_or_empty",
        "mps_active": "active",
        "mps_enabled": "enabled",
        "mps_main_pid": 123,
        "mps_unit_sha256": executor._sha256(executor.MPS_UNIT_PATH),
        "mps_limit_nofile_soft": executor.REQUIRED_NOFILE_SOFT,
        "client_environment": dict(executor.CLIENT_ENVIRONMENT),
        "python": "/venv/bin/python",
        "torch_version": "x",
        "torch_cuda_version": "x",
        "catanatron_rs_version": executor.NATIVE_WHEEL_VERSION,
        "native_wheel_sha256": executor._native_wheel_release_identity()["sha256"],
        "native_mcts_capabilities": sorted(executor.NATIVE_REQUIRED_CAPABILITIES),
        "required_nofile_soft": executor.REQUIRED_NOFILE_SOFT,
        "nofile_soft_before": 1024,
        "nofile_soft": executor.REQUIRED_NOFILE_SOFT,
        "nofile_hard": 1_048_576,
    }
    report[field] = value
    monkeypatch.setattr(
        executor,
        "_ssh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps(report), ""
        ),
    )
    with pytest.raises(executor.ExecutorError, match=message):
        executor._preflight_host(
            {"python": "/venv/bin/python"}, "h00", [0, 1, 2, 3]
        )


def test_supervisor_launch_raises_nofile_before_new_session_and_env(
    tmp_path: Path,
) -> None:
    _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < executor.REQUIRED_NOFILE_SOFT:
        pytest.skip("test host hard RLIMIT_NOFILE is below the production requirement")
    observation = tmp_path / "observation.json"
    child = tmp_path / "supervisor.py"
    child.write_text(
        """\
import json,os,pathlib,resource,sys,time
path=pathlib.Path(os.environ['A1_TEST_OBSERVATION'])
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
path.write_text(json.dumps({'argv':sys.argv[1:],'soft':soft,'hard':hard,'pid':os.getpid(),'sid':os.getsid(0),'pgid':os.getpgrp(),'pipe':os.environ.get('CUDA_MPS_PIPE_DIRECTORY'),'pythonpath':os.environ.get('PYTHONPATH'),'dont_write_bytecode':os.environ.get('PYTHONDONTWRITEBYTECODE'),'tf32_override':os.environ.get('NVIDIA_TF32_OVERRIDE'),'pythonhome':os.environ.get('PYTHONHOME')}))
time.sleep(30)
""",
        encoding="utf-8",
    )
    log = tmp_path / "logs" / "lane.log"
    command = executor._supervisor_launch_command(
        python=sys.executable,
        supervisor=str(child),
        remote_lane="/sealed/lane.json",
        log=str(log),
        repo_dir="/sealed/repo",
        extra_environment={"A1_TEST_OBSERVATION": str(observation)},
    )
    def low_soft_nofile() -> None:
        _current_soft, current_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, current_hard))

    hostile_environment = os.environ.copy()
    hostile_environment["NVIDIA_TF32_OVERRIDE"] = "1"
    hostile_environment["PYTHONHOME"] = "/unsealed/ambient-python"
    result = subprocess.run(
        shlex.split(command), text=True, capture_output=True, check=False,
        preexec_fn=low_soft_nofile,
        env=hostile_environment,
    )
    assert result.returncode == 0, result.stderr
    pid = int(result.stdout.strip())
    try:
        deadline = time.monotonic() + 5
        while not observation.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        observed = json.loads(observation.read_text(encoding="utf-8"))
        assert observed["argv"] == ["run", "--lane", "/sealed/lane.json"]
        assert observed["soft"] >= executor.REQUIRED_NOFILE_SOFT
        assert observed["pid"] == observed["sid"] == observed["pgid"] == pid
        assert observed["pipe"] == executor.CLIENT_ENVIRONMENT["CUDA_MPS_PIPE_DIRECTORY"]
        assert observed["pythonpath"] == "/sealed/repo/src:/sealed/repo"
        assert observed["dont_write_bytecode"] == "1"
        assert observed["tf32_override"] is None
        assert observed["pythonhome"] is None
    finally:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_staged_repo_is_read_only_bytecode_clean_and_resume_verifiable(
    tmp_path: Path,
) -> None:
    files = {
        "pkg/module.py": (b"VALUE = 7\n", 0o444),
        "bin/tool.py": (b"#!/usr/bin/env python3\n", 0o555),
    }
    archive_path = tmp_path / "repo.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, (payload, mode) in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = mode
            archive.addfile(info, io.BytesIO(payload))
    manifest = {
        "schema_version": "a1-production-repo-v1",
        "repo_tar_sha256": executor._sha256(archive_path),
        "artifacts": [
            {
                "path": name,
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "mode": mode,
            }
            for name, (payload, mode) in files.items()
        ],
    }
    manifest["manifest_sha256"] = executor._digest(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    root = tmp_path / "sealed-repo"
    receipt = tmp_path / "receipt.json"
    argv = [
        sys.executable, "-c", executor._STAGE_REPO_SCRIPT,
        str(archive_path), str(root), str(manifest_path), str(receipt),
    ]

    first = subprocess.run(argv, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    assert stat.S_IMODE(root.stat().st_mode) == 0o555
    assert stat.S_IMODE((root / "pkg").stat().st_mode) == 0o555
    assert stat.S_IMODE((root / "pkg/module.py").stat().st_mode) == 0o444
    assert stat.S_IMODE((root / "bin/tool.py").stat().st_mode) == 0o555

    imported = subprocess.run(
        [sys.executable, "-c", "import pkg.module; assert pkg.module.VALUE == 7"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert not list(root.rglob("__pycache__"))
    assert not list(root.rglob("*.pyc"))
    if os.geteuid() != 0:
        direct_environment = {**os.environ, "PYTHONPATH": str(root)}
        direct_environment.pop("PYTHONDONTWRITEBYTECODE", None)
        direct_environment.pop("PYTHONPYCACHEPREFIX", None)
        direct_import = subprocess.run(
            [sys.executable, "-c", "import pkg.module"],
            cwd=root,
            env=direct_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert direct_import.returncode == 0, direct_import.stderr
        assert not list(root.rglob("__pycache__"))
        assert not list(root.rglob("*.pyc"))

    os.chmod(root / "pkg", 0o755)
    legacy_cache = root / "pkg/__pycache__"
    legacy_cache.mkdir()
    (legacy_cache / "module.cpython-311.pyc").write_bytes(b"legacy bytecode")
    resumed = subprocess.run(argv, text=True, capture_output=True, check=False)
    assert resumed.returncode == 0, resumed.stderr
    assert not legacy_cache.exists()
    assert stat.S_IMODE((root / "pkg").stat().st_mode) == 0o555
    clean_resume = subprocess.run(argv, text=True, capture_output=True, check=False)
    assert clean_resume.returncode == 0, clean_resume.stderr
