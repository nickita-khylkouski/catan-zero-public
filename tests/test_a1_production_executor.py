from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path

import pytest

from tools import a1_pre_wave_contract as contract
from tools.fleet import a1_lane_supervisor as supervisor
from tools.fleet import a1_production_executor as executor


def _sha(path: Path) -> str:
    return executor._sha256(path)


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
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
        "n_full_wide": None,
        "n_full_wide_threshold": None,
        "raw_policy_above_width": None,
        "symmetry_averaged_eval_threshold": 20,
    }
    lock = {
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
    }
    commands = []
    categories = executor.CATEGORY_ORDER
    for lane_index in range(40):
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
            commands.append(
                {
                    **{key: job[key] for key in ("job_id", "worker_id", "host_alias", "gpu", "category")},
                    "environment": {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        **executor.CLIENT_ENVIRONMENT,
                        "CATAN_SEED_LEDGER": str(ledger),
                        "CATAN_A1_CONTRACT_SHA256": lock["contract_sha256"],
                    },
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


def test_dry_plan_is_exact_40_lane_120_job_n128_mps_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, render_path, lock, rendered = _fixture(tmp_path)
    hosts = _hosts(tmp_path, rendered)
    monkeypatch.setattr(executor, "_repo_artifacts", lambda _rendered: [])
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
    assert all(command["argv"][command["argv"].index("--n-full") + 1] == "128" for command in rendered["commands"])
    assert not any(flag in command["argv"] for command in rendered["commands"] for flag in executor.FORBIDDEN_ADAPTIVE_ARGV)


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


def _lane(tmp_path: Path, commands: list[dict]) -> tuple[Path, dict]:
    lock_copy = tmp_path / "remote" / "contract.lock.json"
    render_copy = tmp_path / "remote" / "commands.json"
    lock_copy.parent.mkdir(parents=True, exist_ok=True)
    lock_copy.write_text("lock\n", encoding="utf-8")
    render_copy.write_text("render\n", encoding="utf-8")
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
        "commands": commands,
    }
    lane["lane_sha256"] = supervisor._digest(lane)
    path = tmp_path / "lane.json"
    path.write_text(json.dumps(lane), encoding="utf-8")
    return path, lane


def _complete_output(command: dict) -> None:
    argv = command["argv"]
    out = Path(argv[argv.index("--out-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    attempts = int(argv[argv.index("--games") + 1])
    base_seed = int(argv[argv.index("--base-seed") + 1])
    (out / "manifest.json").write_text(
        json.dumps({"games_requested": attempts, "games_completed": attempts, "games_failed": 0, "errors": [], "base_seed": base_seed}),
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
    for command in commands:
        _complete_output(command)
        receipt = {
            "schema_version": supervisor.RECEIPT_SCHEMA,
            "job_id": command["job_id"],
            "lane_sha256": lane["lane_sha256"],
            "argv_sha256": command["argv_sha256"],
            "status": "complete",
        }
        Path(lane["receipt_dir"], f"{command['job_id']}.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_a, **_k: pytest.fail("completed job reran"))
    assert supervisor.run_lane(lane_path)["status"] == "complete"


def test_incomplete_lane_runs_exact_resume_sequentially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert all(kwargs.get("pass_fds") for _argv, _environment, kwargs in calls)
    assert [command["category"] for command in lane["commands"]] == list(supervisor.CATEGORY_ORDER)


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


def test_preflight_accepts_only_exact_report(monkeypatch: pytest.MonkeyPatch) -> None:
    report = {
        "gpu_indices": [0, 1, 2, 3],
        "compute_apps": "mps_only_or_empty",
        "mps_active": "active",
        "mps_enabled": "enabled",
        "mps_main_pid": 123,
        "client_environment": dict(executor.CLIENT_ENVIRONMENT),
        "python": "/venv/bin/python",
        "torch_version": "x",
        "torch_cuda_version": "x",
        "catanatron_rs_version": "x",
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
