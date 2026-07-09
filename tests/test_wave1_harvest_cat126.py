"""CAT-126 #19: canonical parallel fleet harvest (tools/wave1_harvest.sh).

OPS change (no data content touched): rsync runs parallel across boxes AND dirs
with SSH ControlMaster reuse. Fleet HOST/DIRS come from an external non-committed
config (repo is public) — CAT-131 FLEET.md is the source; DIRS re-verified at
Wave-2 launch.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "wave1_harvest.sh"


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


def test_no_hardcoded_fleet_ips_in_public_repo():
    src = SCRIPT.read_text()
    # No literal IPv4 addresses committed (topology stays in the external config).
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", src), \
        "hardcoded IP found — fleet topology must stay in the external FLEET_CONF"
