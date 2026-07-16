import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time

import pytest

from tools.fleet import build_a1_neutral_panel_jobset as historical_panel
from tools.fleet import gpu_fleet as fleet


_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_historical_panel_snapshot_cannot_impersonate_generic_authority(tmp_path):
    raw = historical_panel.manifest(ssh_key="/tmp/key")
    assert raw["schema_version"] == historical_panel.HISTORICAL_MANIFEST_SCHEMA
    path = tmp_path / "historical-panel.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(fleet.FleetError, match="unsupported fleet manifest schema"):
        fleet.load_manifest(path)


def _manifest(tmp_path: Path, commits: dict[str, str] | None = None):
    commits = commits or {}
    hosts = []
    for alias, (address, count) in fleet.EXPECTED_HOSTS.items():
        hosts.append(
            {
                "alias": alias,
                "address": address,
                "gpu_count": count,
                "accelerator": fleet.EXPECTED_ACCELERATOR,
                "repo_commit": commits.get(alias, "a" * 40),
            }
        )
    path = tmp_path / "fleet.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": fleet.MANIFEST_SCHEMA,
                "fleet_authority": fleet.FLEET_AUTHORITY,
                "ssh_user": "ubuntu",
                "ssh_key": None,
                "remote_repo": "/opt/catan",
                "remote_root": "/runs",
                "hosts": hosts,
            }
        )
    )
    return fleet.load_manifest(path)


def _jobset(tmp_path: Path, jobs):
    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps(
            {"schema_version": fleet.JOBSET_SCHEMA, "run_id": "run1", "jobs": jobs}
        )
    )
    return fleet.load_jobset(path)


def test_committed_legacy_manifest_retains_exact_56_gpu_authority():
    manifest = fleet.load_manifest(_REPO_ROOT / "configs" / "gpu_fleet_56.json")
    assert fleet.fleet_authority(manifest) == fleet.LEGACY_FLEET_AUTHORITY
    assert manifest["manifest_hash"] == (
        "sha256:8d7713a5ec68528d13de3f92bbdc5fa24d218d70627dd85a61ffadf147e8b2d9"
    )
    assert len(manifest["hosts"]) == 10
    assert sum(host["gpu_count"] for host in manifest["hosts"]) == 56
    assert [(host["alias"], host["gpu_count"]) for host in manifest["hosts"][-2:]] == [
        ("h100-8c", 8),
        ("h100-8d", 8),
    ]


def test_committed_manifest_is_the_canonical_exact_64_gpu_authority():
    manifest = fleet.load_manifest(_REPO_ROOT / "configs" / "gpu_fleet_64.json")
    assert manifest["fleet_authority"] == fleet.FLEET_AUTHORITY
    assert fleet.fleet_authority(manifest) == fleet.FLEET_AUTHORITY
    assert len(manifest["hosts"]) == 12
    assert sum(host["gpu_count"] for host in manifest["hosts"]) == 64
    assert [(host["alias"], host["gpu_count"]) for host in manifest["hosts"]] == [
        (alias, count) for alias, count in fleet.EXPECTED_SHAPES.items()
    ]
    assert manifest["ssh_key"] == "/home/ubuntu/.ssh/catan_fleet_ed25519"


def test_new_eight_gpu_hosts_are_allocatable_only_at_their_audited_commit(tmp_path):
    manifest = fleet.load_manifest(_REPO_ROOT / "configs" / "gpu_fleet_56.json")
    jobs = _jobset(
        tmp_path,
        [
            {"job_id": "new-c", "gpus": 8, "argv": ["true"]},
            {"job_id": "new-d", "gpus": 8, "argv": ["true"]},
        ],
    )
    plan = fleet.build_plan(
        manifest,
        jobs,
        repo_commit="589e747ac5d2fcf857c8910df78e7b61d5b05da5",
    )
    assert [row["alias"] for row in plan["assignments"]] == ["h100-8c", "h100-8d"]


def test_manifest_is_exactly_the_canonical_64_gpu_shape(tmp_path):
    manifest = _manifest(tmp_path)
    assert sum(host["gpu_count"] for host in manifest["hosts"]) == 64
    manifest["hosts"].append(
        {
            "alias": "h100-8e",
            "address": "203.0.113.9",
            "gpu_count": 8,
            "accelerator": fleet.EXPECTED_ACCELERATOR,
            "repo_commit": "a" * 40,
        }
    )
    path = tmp_path / "expanded.json"
    path.write_text(
        json.dumps({k: v for k, v in manifest.items() if k != "manifest_hash"})
    )
    with pytest.raises(fleet.FleetError, match="mapping drift"):
        fleet.load_manifest(path)


def test_v2_manifest_refuses_missing_or_legacy_authority(tmp_path):
    manifest = _manifest(tmp_path)
    raw = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    for authority in (None, fleet.LEGACY_FLEET_AUTHORITY):
        if authority is None:
            raw.pop("fleet_authority", None)
        else:
            raw["fleet_authority"] = authority
        path = tmp_path / f"wrong-authority-{authority}.json"
        path.write_text(json.dumps(raw))
        with pytest.raises(fleet.FleetError, match="fleet_authority"):
            fleet.load_manifest(path)


def test_v1_manifest_cannot_smuggle_exact64_authority(tmp_path):
    raw = json.loads((_REPO_ROOT / "configs" / "gpu_fleet_56.json").read_text())
    raw["fleet_authority"] = fleet.FLEET_AUTHORITY
    path = tmp_path / "v1-smuggled-authority.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(fleet.FleetError, match="must not declare"):
        fleet.load_manifest(path)


def test_c7_c8_allocate_only_at_their_audited_commit(tmp_path):
    manifest = fleet.load_manifest(_REPO_ROOT / "configs" / "gpu_fleet_64.json")
    jobs = _jobset(
        tmp_path,
        [
            {"job_id": "c7-job", "host": "c7", "gpus": 4, "argv": ["true"]},
            {"job_id": "c8-job", "host": "c8", "gpus": 4, "argv": ["true"]},
        ],
    )
    plan = fleet.build_plan(
        manifest,
        jobs,
        repo_commit="4e95bc0cc81cf3d121410aeb0093f3975c492af6",
    )
    assert [row["alias"] for row in plan["assignments"]] == ["c7", "c8"]


def test_exact64_new_hosts_allocate_only_at_their_deployed_commit(tmp_path):
    manifest = fleet.load_manifest(_REPO_ROOT / "configs" / "gpu_fleet_64.json")
    jobs = _jobset(
        tmp_path,
        [
            {"job_id": "h100-8c-job", "host": "h100-8c", "gpus": 8, "argv": ["true"]},
            {"job_id": "h100-8d-job", "host": "h100-8d", "gpus": 8, "argv": ["true"]},
        ],
    )
    plan = fleet.build_plan(
        manifest,
        jobs,
        repo_commit="4e95bc0cc81cf3d121410aeb0093f3975c492af6",
    )
    assert [row["alias"] for row in plan["assignments"]] == ["h100-8c", "h100-8d"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda hosts: hosts[0].update(address="203.0.113.9"), "mapping drift"),
        (lambda hosts: hosts[0].update(gpu_count=8), "mapping drift"),
        (lambda hosts: hosts[1].update(address=hosts[0]["address"]), "duplicate"),
        (lambda hosts: hosts[0].update(accelerator="H100"), "must be exactly"),
    ],
)
def test_manifest_rejects_wrong_duplicate_and_lookalike_hosts(
    tmp_path, mutation, message
):
    manifest = _manifest(tmp_path)
    raw = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    mutation(raw["hosts"])
    path = tmp_path / f"bad-{message.replace(' ', '-')}.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(fleet.FleetError, match=message):
        fleet.load_manifest(path)


def test_deterministic_heterogeneous_allocation(tmp_path):
    manifest = _manifest(tmp_path)
    jobs = _jobset(
        tmp_path,
        [
            {"job_id": "train", "gpus": 4, "argv": ["python", "train.py"]},
            {"job_id": "eval", "gpus": 8, "argv": ["python", "eval.py"]},
        ],
    )
    first = fleet.build_plan(manifest, jobs, repo_commit="a" * 40)
    second = fleet.build_plan(manifest, jobs, repo_commit="a" * 40)
    assert first == second
    assert [(row["alias"], row["gpu_ids"]) for row in first["assignments"]] == [
        ("c1", [0, 1, 2, 3]),
        ("h100-8a", list(range(8))),
    ]


def test_commit_filter_and_preference_fail_closed(tmp_path):
    commits = {alias: "a" * 40 for alias in fleet.EXPECTED_SHAPES}
    commits["h100-8b"] = "b" * 40
    manifest = _manifest(tmp_path, commits)
    jobs = _jobset(tmp_path, [{"job_id": "new", "gpus": 8, "argv": ["true"]}])
    plan = fleet.build_plan(manifest, jobs, repo_commit="b" * 40)
    assert plan["assignments"][0]["alias"] == "h100-8b"
    pinned = _jobset(
        tmp_path,
        [{"job_id": "bad", "gpus": 4, "host": "c1", "argv": ["true"]}],
    )
    with pytest.raises(fleet.FleetError, match="not"):
        fleet.build_plan(manifest, pinned, repo_commit="b" * 40)


def test_rejects_shell_shaped_env_name(tmp_path):
    with pytest.raises(fleet.FleetError, match="invalid env"):
        _jobset(
            tmp_path,
            [
                {
                    "job_id": "x",
                    "argv": ["true"],
                    "env": {"A;touch /tmp/pwn": "x"},
                }
            ],
        )


def test_plan_tamper_is_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    jobs = _jobset(tmp_path, [{"job_id": "x", "argv": ["true"]}])
    plan = fleet.build_plan(manifest, jobs, repo_commit="a" * 40)
    plan["assignments"][0]["gpu_ids"] = [3]
    plan["plan_hash"] = fleet._digest(
        {key: value for key, value in plan.items() if key != "plan_hash"}
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(fleet.FleetError, match="allocation"):
        fleet.load_plan(path, manifest)


def test_submit_is_dry_run_and_shell_quotes_argv(tmp_path):
    manifest = _manifest(tmp_path)
    jobs = _jobset(tmp_path, [{"job_id": "x", "argv": ["python", "a b", "$(bad)"]}])
    plan = fleet.build_plan(manifest, jobs, repo_commit="a" * 40)
    assert fleet.submit(manifest, plan, go=False)["dry_run"] is True
    command = fleet._launch_command(manifest, plan, plan["assignments"][0])
    assert "'a b'" in command
    assert "'$(bad)'" in command
    assert command.index(":done") < command.index("memory.used")
    assert command.index(":active") < command.index("memory.used")
    assert "allocation.lock" in command
    assert "flock --exclusive --close" in command
    assert 'exec {lease_fd}>"$lock_root/gpu-0.lock"' in command
    assert command.index("gpu-0.lock") < command.index("memory.used")
    assert command.index("allocation.lock") < command.index("git -C")
    assert command.index("git -C") < command.index("receipt.json")
    assert "grep -Fxc" in command
    assert fleet.EXPECTED_ACCELERATOR in command


def test_gpu_lease_is_exclusive_across_concurrent_processes(tmp_path):
    if shutil.which("flock") is None:
        pytest.skip("util-linux flock behavioral test requires Linux")
    lease = tmp_path / "gpu-0.lock"
    holder = subprocess.Popen(["flock", "--exclusive", str(lease), "sleep", "0.5"])
    try:
        for _ in range(50):
            probe = subprocess.run(["flock", "--nonblock", str(lease), "true"])
            if probe.returncode != 0:
                break
        assert probe.returncode != 0
    finally:
        holder.wait(timeout=2)
    assert subprocess.run(["flock", "--nonblock", str(lease), "true"]).returncode == 0


@pytest.mark.skipif(sys.platform != "linux", reason="rendered fleet launch is Linux")
def test_two_rendered_controller_plans_cannot_share_one_gpu(tmp_path):
    """Run two rendered transactions; zero GPU memory makes the lease decisive."""
    remote_repo = tmp_path / "remote-repo"
    launcher = remote_repo / "tools/fleet/launch_detached.sh"
    launcher.parent.mkdir(parents=True)
    source_launcher = (
        Path(__file__).resolve().parents[1] / "tools/fleet/launch_detached.sh"
    )
    shutil.copy2(source_launcher, launcher)
    launcher.chmod(0o755)
    subprocess.run(["git", "init", "-q", str(remote_repo)], check=True)
    subprocess.run(
        ["git", "-C", str(remote_repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(remote_repo), "config", "user.name", "Fleet Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(remote_repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(remote_repo), "commit", "-qm", "fixture"], check=True
    )
    commit = subprocess.check_output(
        ["git", "-C", str(remote_repo), "rev-parse", "HEAD"], text=True
    ).strip()

    fake_nvidia = tmp_path / "nvidia-smi"
    fake_nvidia.write_text(
        """#!/usr/bin/env bash
case "$*" in
  *--query-compute-apps=*) exit 0 ;;
  *--query-gpu=index*) printf '0\\n1\\n2\\n3\\n' ;;
  *--query-gpu=name*) printf 'NVIDIA H100 80GB HBM3\\n%.0s' {1..4} ;;
  *--query-gpu=uuid*) gpu=0; while [ "$#" -gt 0 ]; do [ "$1" = -i ] && { gpu="$2"; break; }; shift; done; printf 'GPU-%s\\n' "$gpu" ;;
  *--query-gpu=memory.used*) printf '0\\n' ;;
  *) exit 2 ;;
esac
"""
    )
    fake_nvidia.chmod(0o755)

    manifest = _manifest(tmp_path)
    manifest["remote_repo"] = str(remote_repo)
    manifest["remote_root"] = str(tmp_path / "runs")
    for host in manifest["hosts"]:
        host["repo_commit"] = commit

    def rendered(run_id, job_id, argv):
        jobs = _jobset(
            tmp_path,
            [{"job_id": job_id, "host": "c1", "gpus": 1, "argv": argv}],
        )
        jobs["run_id"] = run_id
        plan = fleet.build_plan(manifest, jobs, repo_commit=commit)
        row = plan["assignments"][0]
        command = fleet._launch_command(manifest, plan, row).replace(
            "nvidia-smi", shlex.quote(str(fake_nvidia))
        )
        return row, command

    row_a, command_a = rendered("rendered-a", "hold", ["sleep", "3"])
    row_b, command_b = rendered("rendered-b", "collide", ["sleep", "1"])
    first = subprocess.run(
        ["bash", "-c", command_a], text=True, capture_output=True, timeout=5
    )
    assert first.returncode == 0, first.stderr
    assert Path(row_a["job_dir"], "receipt.json").is_file()

    collision = subprocess.run(
        ["bash", "-c", command_b], text=True, capture_output=True, timeout=5
    )
    assert collision.returncode != 0
    assert "gpu-lease-busy" in collision.stderr
    assert not Path(row_b["job_dir"], "receipt.json").exists()

    deadline = time.monotonic() + 10
    while not Path(row_a["job_dir"], ".done").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert Path(row_a["job_dir"], ".done").exists()

    # The heartbeat can retain its inherited lease for one 5-second cadence.
    retry = None
    while time.monotonic() < deadline:
        retry = subprocess.run(
            ["bash", "-c", command_b], text=True, capture_output=True, timeout=5
        )
        if retry.returncode == 0:
            break
        time.sleep(0.1)
    assert retry is not None and retry.returncode == 0, retry.stderr
    assert Path(row_b["job_dir"], "receipt.json").is_file()


def test_adversarial_argv_and_env_are_executed_as_literal_data(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["remote_repo"] = str(tmp_path)
    output = tmp_path / "observed.json"
    pwned = tmp_path / "pwned"
    strange = f"$(touch {pwned}) ; `touch {pwned}`"
    jobs = _jobset(
        tmp_path,
        [
            {
                "job_id": "literal",
                "argv": [
                    sys.executable,
                    "-c",
                    "import json,os,sys;json.dump([sys.argv[1],os.environ['WEIRD']],open(sys.argv[2],'w'))",
                    strange,
                    str(output),
                ],
                "env": {"WEIRD": strange},
            }
        ],
    )
    plan = fleet.build_plan(manifest, jobs, repo_commit="a" * 40)
    row = plan["assignments"][0]
    row["job_dir"] = str(tmp_path / "job")
    Path(row["job_dir"]).mkdir()
    _, inner = fleet._runtime(manifest, plan, row)
    subprocess.run(["bash", "-lc", inner], check=True)
    assert json.loads(output.read_text()) == [strange, strange]
    assert not pwned.exists()


def test_inventory_validates_shape_commit_and_busy_state(tmp_path):
    manifest = _manifest(tmp_path)

    def runner(argv):
        return subprocess.CompletedProcess(
            argv,
            0,
            "hostname=n\ngpu_count=4\ngpu_names=NVIDIA H100\n"
            "busy_gpus=0\nrepo_commit=" + "a" * 40 + "\n",
            "",
        )

    # The fake response has four GPUs for every host, so 8-GPU declarations fail.
    result = fleet.inventory(manifest, runner=runner)
    assert result["valid"] is False
    assert result["fleet_authority"] == fleet.FLEET_AUTHORITY
    assert result["gpu_capacity"] == 64


def _local_status_runner(argv):
    result = subprocess.run(
        ["bash", "-lc", argv[-1]],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return subprocess.CompletedProcess(
        argv, result.returncode, result.stdout, result.stderr
    )


def _write_receipt(manifest, plan, row, *, mutate=False):
    receipt, _ = fleet._runtime(manifest, plan, row)
    if mutate:
        receipt["plan_hash"] = "sha256:" + "0" * 64
    path = Path(row["job_dir"])
    path.mkdir(parents=True)
    (path / "receipt.json").write_bytes(fleet._canonical(receipt))
    os.chmod(path / "receipt.json", 0o444)


@pytest.mark.skipif(sys.platform != "linux", reason="remote status contract is Linux")
def test_status_rejects_cross_plan_receipt(tmp_path):
    manifest = _manifest(tmp_path)
    jobs = _jobset(tmp_path, [{"job_id": "x", "argv": ["true"]}])
    plan = fleet.build_plan(manifest, jobs, repo_commit="a" * 40)
    plan["assignments"][0]["job_dir"] = str(tmp_path / "cross-plan")
    _write_receipt(manifest, plan, plan["assignments"][0], mutate=True)
    result = fleet.status(manifest, plan, runner=_local_status_runner)
    assert result["jobs"][0]["status"] == "DRIFT"


@pytest.mark.skipif(sys.platform != "linux", reason="remote status contract is Linux")
def test_status_rejects_stale_or_reused_pid(tmp_path):
    manifest = _manifest(tmp_path)
    jobs = _jobset(tmp_path, [{"job_id": "x", "argv": ["true"]}])
    plan = fleet.build_plan(manifest, jobs, repo_commit="a" * 40)
    row = plan["assignments"][0]
    row["job_dir"] = str(tmp_path / "stale-pid")
    _write_receipt(manifest, plan, row)
    path = Path(row["job_dir"])
    (path / ".pid").write_text(str(os.getpid()))
    (path / ".heartbeat").write_text(f"now pid={os.getpid()}\n")
    result = fleet.status(manifest, plan, runner=_local_status_runner)
    assert result["jobs"][0]["status"] == "STALE_IDENTITY"
