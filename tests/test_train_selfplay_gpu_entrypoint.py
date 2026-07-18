from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = ROOT / "configs" / "selfplay" / "ppo_2p_no_trade_v2.json"


def test_legacy_gpu_trainer_refuses_canonical_manifest_before_runtime_setup() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if value
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "train_selfplay_gpu.py"),
            "--config",
            str(CANONICAL_MANIFEST),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "does not accept canonical_entity_ppo_run_v2 manifests" in result.stderr
    assert "tools/run_local_entity_ppo_shards.py" in result.stderr
    assert "tools/ppo_distributed_learner.py" in result.stderr
    assert "KeyError" not in result.stderr
