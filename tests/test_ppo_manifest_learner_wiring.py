from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_zero.rl import ppo_distributed as dist
from tools import ppo_distributed_learner as learner


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/selfplay/ppo_2p_no_trade_v2.json"


def _bound_manifest(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"exact-initializer")
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    payload["status"] = "bound"
    payload["spec"]["identity"]["initializer_sha256"] = (
        f"sha256:{dist.checkpoint_sha256(checkpoint)}"
    )
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, checkpoint


def test_manifest_resolution_retains_path_and_binds_v2_root(tmp_path: Path) -> None:
    manifest, checkpoint = _bound_manifest(tmp_path)
    config, _ = learner.resolve_config(
        [
            "--run-manifest",
            str(manifest),
            "--init-checkpoint",
            str(checkpoint),
            "--run-base",
            str(tmp_path),
            "--run-name",
            "run",
        ]
    )
    root = dist.run_root(config.run_base, config.run_name)
    dist.ensure_run_dirs(root)

    binding = learner._bind_configured_run_identity(root, config)  # noqa: SLF001

    assert config.run_manifest_path == str(manifest)
    assert binding["manifest_sha256"] == config.run_manifest_sha256
    assert dist.run_manifest_path(root).is_file()
    assert not dist.run_contract_path(root).exists()


def test_manifest_drift_after_resolution_is_refused(tmp_path: Path) -> None:
    manifest, checkpoint = _bound_manifest(tmp_path)
    config, _ = learner.resolve_config(
        [
            "--run-manifest",
            str(manifest),
            "--init-checkpoint",
            str(checkpoint),
        ]
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["spec"]["learner"]["lr"] = 0.0001
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed after configuration resolution"):
        learner._bind_configured_run_identity(tmp_path / "run", config)  # noqa: SLF001


def test_v2_stale_sweep_refuses_foreign_shards_without_consuming(
    tmp_path: Path,
) -> None:
    expected = "sha256:" + "1" * 64
    foreign = "sha256:" + "2" * 64
    shard = dist.write_trajectory_shard(
        tmp_path,
        "worker",
        0,
        [],
        policy_version=0,
        run_manifest_sha256=foreign,
    )

    with pytest.raises(dist.RunManifestError, match="manifest mismatch"):
        dist.sweep_drop_outside_policy_window(
            tmp_path,
            min_policy_version=1,
            max_policy_version=2,
            expected_run_manifest_sha256=expected,
        )

    assert shard.is_file()
    assert not any(dist.consumed_dir(tmp_path).iterdir())

