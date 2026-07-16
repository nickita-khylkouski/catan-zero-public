from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "entrypoint",
    ("ppo_distributed_learner.py", "run_local_entity_ppo_shards.py"),
)
def test_ppo_entrypoint_help_ignores_stale_ambient_pythonpath(
    entrypoint: str, tmp_path: Path
) -> None:
    stale = tmp_path / "catan_zero" / "rl"
    stale.mkdir(parents=True)
    (stale.parent / "__init__.py").write_text("", encoding="utf-8")
    (stale / "__init__.py").write_text("", encoding="utf-8")
    (stale / "ppo_run_manifest.py").write_text(
        "raise RuntimeError('loaded stale ambient package')\n", encoding="utf-8"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / entrypoint), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "loaded stale ambient package" not in completed.stderr
    assert "--run-manifest" in completed.stdout

