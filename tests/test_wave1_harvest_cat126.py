"""CAT-126 #19: canonical parallel fleet harvest (tools/wave1_harvest.sh).

OPS change (no data content touched): rsync runs parallel across boxes AND dirs
with SSH ControlMaster reuse. Fleet HOST/DIRS come from an external non-committed
config (repo is public) — CAT-131 FLEET.md is the source; DIRS re-verified at
Wave-2 launch.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "wave1_harvest.sh"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _mock_harvest_env(
    tmp_path: Path,
    *,
    fail_source: str = "",
    empty_source: str = "",
) -> tuple[dict[str, str], Path]:
    """Build a deterministic five-box fleet with a recording rsync stub."""
    conf = tmp_path / "fleet.conf"
    conf.write_text(
        """declare -A HOST=(
  [c1]=host-c1 [c2]=host-c2 [c3]=host-c3
  [c5]=host-c5 [c6]=host-c6
)
declare -A DIRS=(
  [c1]="runs/volume-c1-a runs/volume-c1-b"
  [c2]="runs/teacher-c2"
  [c3]="runs/teacher-c3"
  [c5]="runs/volume-c5"
  [c6]="runs/teacher-c6"
)
GPU_SSH_KEY=/dev/null
""",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "rsync-calls"
    calls.mkdir()
    _write_executable(
        fake_bin / "rsync",
        """#!/usr/bin/env bash
set -u
printf '%s\n' "$@" > "$RSYNC_CALL_DIR/$BASHPID"
source_arg="${@: -2:1}"
dest="${@: -1}"
if [ -n "${FAKE_RSYNC_FAIL:-}" ] && [[ "$source_arg" == *"$FAKE_RSYNC_FAIL"* ]]; then
  exit 23
fi
mkdir -p "$dest/gpu0"
touch "$dest/gpu0/manifest.json" "$dest/gpu0/progress.jsonl"
touch "$dest/gpu0/run.log" "$dest/gpu0/quality.json"
if [ -n "${FAKE_RSYNC_EMPTY:-}" ] && [[ "$source_arg" == *"$FAKE_RSYNC_EMPTY"* ]]; then
  exit 0
fi
touch "$dest/gpu0/gumbel_self_play_shard_000000.npz"
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "FLEET_CONF": str(conf),
            "HARV_DIR": str(tmp_path / "harvest"),
            "HOME": str(tmp_path),
            # Keep the current Bash (associative arrays require Bash >=4) while
            # placing only rsync ahead of the real command set.
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RSYNC_CALL_DIR": str(calls),
            "FAKE_RSYNC_FAIL": fail_source,
            "FAKE_RSYNC_EMPTY": empty_source,
        }
    )
    return env, calls


def _recorded_rsync_calls(calls: Path) -> list[list[str]]:
    return [path.read_text(encoding="utf-8").splitlines() for path in calls.iterdir()]


def test_script_exists_and_syntax_ok():
    assert SCRIPT.exists()
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_refuses_without_fleet_config():
    r = subprocess.run(
        ["bash", str(SCRIPT), "harvest-volume"],
        capture_output=True, text=True,
        env={"FLEET_CONF": "/nonexistent/fleet.conf", "HOME": "/tmp", "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 2
    assert ("FLEET_CONF" in r.stderr) or ("fleet config" in r.stderr)  # fleet_lib reports missing config


def test_has_parallel_technique_and_controlmaster():
    src = SCRIPT.read_text()
    assert "ControlMaster=auto" in src
    assert "pull_dirs_parallel" in src
    # backgrounded rsync + wait (the parallelism):
    assert "&\n" in src or "& pids+=" in src
    assert "wait " in src
    assert "fleet/fleet_lib.sh" in src       # CAT-126 dedup: sources the ONE resolver
    assert "fleet_host" in src                # uses the resolver helper, not ${HOST[..]}


def test_mocked_harvest_preserves_reconciliation_artifacts_and_current_roles(tmp_path: Path):
    env, calls_dir = _mock_harvest_env(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), "harvest-all"],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    calls = _recorded_rsync_calls(calls_dir)
    assert len(calls) == 6  # two c1 roots plus one root on each other active generation box
    sources = {call[-2] for call in calls}
    assert sources == {
        "ubuntu@host-c1:runs/volume-c1-a/",
        "ubuntu@host-c1:runs/volume-c1-b/",
        "ubuntu@host-c5:runs/volume-c5/",
        "ubuntu@host-c2:runs/teacher-c2/",
        "ubuntu@host-c3:runs/teacher-c3/",
        "ubuntu@host-c6:runs/teacher-c6/",
    }
    assert not any("host-c4" in arg for call in calls for arg in call)
    for call in calls:
        assert "--include=*/" in call
        assert "--include=gumbel_self_play_shard_*.npz" in call
        assert "--include=manifest.json" in call
        assert "--include=*progress*" in call
        assert "--include=*.log" in call
        assert "--include=*.json" in call
        assert "--include=*.jsonl" in call
        assert call.index("--exclude=*") > call.index("--include=*.jsonl")

    harvest = tmp_path / "harvest"
    assert len(list(harvest.rglob("gumbel_self_play_shard_*.npz"))) == 6
    assert len(list(harvest.rglob("manifest.json"))) == 6
    assert len(list(harvest.rglob("progress.jsonl"))) == 6
    assert len(list(harvest.rglob("run.log"))) == 6
    assert len(list(harvest.rglob("quality.json"))) == 6


def test_mocked_harvest_all_returns_nested_rsync_failure_after_finishing_other_roles(
    tmp_path: Path,
):
    env, calls_dir = _mock_harvest_env(tmp_path, fail_source="volume-c1-b")
    result = subprocess.run(
        ["bash", str(SCRIPT), "harvest-all"],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert result.returncode == 23
    assert "rsync failed: box=c1 role=volume dir=runs/volume-c1-b rc=23" in result.stderr
    assert "box failed: box=c1 role=volume rc=23" in result.stderr

    # All sources were started/waited, including the teacher role launched after
    # the volume failure; its later success must not erase the failure status.
    calls = _recorded_rsync_calls(calls_dir)
    assert len(calls) == 6
    assert any("teacher-c6" in arg for call in calls for arg in call)


def test_mocked_successful_rsync_with_zero_box_shards_fails_closed(tmp_path: Path):
    env, calls_dir = _mock_harvest_env(tmp_path, empty_source="volume-c5")
    result = subprocess.run(
        ["bash", str(SCRIPT), "harvest-volume"],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert result.returncode != 0
    assert "c5 (volume) runs/volume-c5: 0 npz shards" in result.stdout
    assert "no NPZ shards harvested: box=c5 role=volume root=runs/volume-c5" in result.stderr
    assert "box failed: box=c5 role=volume" in result.stderr

    # Every rsync itself succeeded and emitted metadata; zero accepted NPZs is
    # what makes the authoritative harvest fail.
    assert len(_recorded_rsync_calls(calls_dir)) == 3
    assert not (tmp_path / "harvest/volume/c5").exists()


def test_multisource_box_fails_when_only_one_current_root_is_empty(tmp_path: Path):
    env, calls_dir = _mock_harvest_env(tmp_path, empty_source="volume-c1-b")

    result = subprocess.run(
        ["bash", str(SCRIPT), "harvest-volume"],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode != 0
    assert "c1 (volume) runs/volume-c1-a: 1 npz shards" in result.stdout
    assert "c1 (volume) runs/volume-c1-b: 0 npz shards" in result.stdout
    assert "no NPZ shards harvested: box=c1 role=volume root=runs/volume-c1-b" in result.stderr
    assert "box failed: box=c1 role=volume" in result.stderr
    assert len(_recorded_rsync_calls(calls_dir)) == 3
    # The successful sibling source is not published as a complete box harvest.
    assert not (tmp_path / "harvest/volume/c1").exists()


def test_reused_harvest_dir_does_not_let_stale_shards_mask_empty_pull(tmp_path: Path):
    env, calls_dir = _mock_harvest_env(tmp_path)
    first = subprocess.run(
        ["bash", str(SCRIPT), "harvest-volume"],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    accepted = sorted((tmp_path / "harvest/volume").rglob("*.npz"))
    assert len(accepted) == 3

    env["FAKE_RSYNC_EMPTY"] = "volume-c"
    second = subprocess.run(
        ["bash", str(SCRIPT), "harvest-volume"],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert second.returncode != 0
    assert "runs/volume-c1-a: 0 npz shards" in second.stdout
    assert "runs/volume-c1-b: 0 npz shards" in second.stdout
    assert "runs/volume-c5: 0 npz shards" in second.stdout
    assert "box failed: box=c1 role=volume" in second.stderr
    assert "box failed: box=c5 role=volume" in second.stderr
    assert len(_recorded_rsync_calls(calls_dir)) == 6
    # Transactional staging preserves the last accepted harvest, but validation
    # is based only on the failed invocation's fresh staging trees.
    assert sorted((tmp_path / "harvest/volume").rglob("*.npz")) == accepted


def test_no_hardcoded_fleet_ips_in_public_repo():
    src = SCRIPT.read_text()
    # No literal IPv4 addresses committed (topology stays in the external config).
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", src), \
        "hardcoded IP found — fleet topology must stay in the external FLEET_CONF"
