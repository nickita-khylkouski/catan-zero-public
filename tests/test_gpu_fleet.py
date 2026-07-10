import json
from pathlib import Path
import subprocess

import pytest

from tools.fleet import gpu_fleet as fleet


def _manifest(tmp_path: Path, commits: dict[str, str] | None = None):
    commits = commits or {}
    hosts = []
    for alias, count in fleet.EXPECTED_SHAPES.items():
        hosts.append(
            {
                "alias": alias,
                "address": f"{alias}.example",
                "gpu_count": count,
                "accelerator": "H100",
                "repo_commit": commits.get(alias, "a" * 40),
            }
        )
    path = tmp_path / "fleet.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": fleet.MANIFEST_SCHEMA,
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


def test_manifest_is_exactly_the_canonical_40_gpu_shape(tmp_path):
    manifest = _manifest(tmp_path)
    assert sum(host["gpu_count"] for host in manifest["hosts"]) == 40
    manifest["hosts"].append(
        {
            "alias": "h100-8c",
            "address": "excluded.example",
            "gpu_count": 8,
            "accelerator": "H100",
            "repo_commit": "a" * 40,
        }
    )
    path = tmp_path / "expanded.json"
    path.write_text(
        json.dumps({k: v for k, v in manifest.items() if k != "manifest_hash"})
    )
    with pytest.raises(fleet.FleetError, match="exactly six"):
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
    assert result["gpu_capacity"] == 40
