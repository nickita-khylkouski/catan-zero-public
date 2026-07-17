from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest

from tools.fleet import a1_h100_generation_preflight as preflight


COMMIT = "a" * 40
WHEEL_SHA = "b" * 64
EXTENSION_SHA = "c" * 64


def _manifest(tmp_path: Path) -> dict[str, object]:
    host = {
        "alias": "h100-8a",
        "address": "192.222.53.175",
        "gpu_count": 8,
        "accelerator": "NVIDIA H100 80GB HBM3",
    }
    return {
        "git_url": "https://github.com/example/catan-zero.git",
        "ssh_key": str(tmp_path / "fleet-key"),
        "ssh_user": "ubuntu",
        "strict_host_key_checking": "yes",
        "coordinator_alias": "h100-8a",
        "remote_repo": "/home/ubuntu/catan-zero-fleet-v2",
        "remote_python": "/home/ubuntu/catan-zero-fleet-v2/.venv/bin/python",
        "remote_root": "/home/ubuntu/a1-generation",
        "checkpoint": {
            "path": "/home/ubuntu/a1-artifacts/champion.pt",
            "sha256": "d" * 64,
        },
        "native_wheel": {
            "path": "/home/ubuntu/a1-artifacts/catanatron_rs.whl",
            "sha256": WHEEL_SHA,
        },
        "seed_authority": {
            "path": "/home/ubuntu/a1-generation/seed-authority.json",
            "range_start": 1_000_000,
            "host_stride": 100_000,
            "gpu_stride": 10_000,
        },
        "resource_minima": {},
        "manifest_sha256": "e" * 64,
        "hosts": [host],
    }


def _runtime() -> dict[str, str]:
    return {
        "python_version": "3.11.15",
        "torch_version": "2.11.0+cu128",
        "torch_cuda_version": "12.8",
        "catanatron_rs_version": "0.1.13",
        "catanatron_rs_extension_sha256": EXTENSION_SHA,
    }


def test_reused_environment_force_reinstalls_and_attests_native_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    host = manifest["hosts"][0]
    commands: list[list[str]] = []

    monkeypatch.setattr(preflight, "_assert_idle_host", lambda *_args: None)
    monkeypatch.setattr(preflight, "_retire_mps_host", lambda *_args: None)
    monkeypatch.setattr(preflight, "_stage_to_host", lambda *_args: None)

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(preflight, "_run", fake_run)
    preflight._bootstrap_host(  # noqa: SLF001
        manifest,
        repo_commit=COMMIT,
        host=host,
        runtime=_runtime(),
        checkpoint_source=tmp_path / "checkpoint.pt",
        wheel_source=tmp_path / "native.whl",
    )

    assert len(commands) == 1
    command = commands[0][-1]
    assert 'if [ "$reused" = 1 ]' in command
    assert 'pip install --force-reinstall --no-deps "$wheel"' in command
    assert WHEEL_SHA in command
    assert EXTENSION_SHA in command
    assert preflight.REQUIRED_LEARNER_ENTITY_ADAPTER in command
    assert "supported_action_context_adapter_versions" in command
    assert "loaded extension drift" in command
    assert "installed wheel provenance drift" in command


def test_inspection_binds_wheel_loaded_elf_and_v6_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    host = manifest["hosts"][0]
    captured: dict[str, object] = {}

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        token = argv[-1]
        captured.update(
            json.loads(
                base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
            )
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            json.dumps({"checks": {}, "details": {}, "ready": True}),
            "",
        )

    monkeypatch.setattr(preflight, "_run", fake_run)
    report = preflight._inspect_host(  # noqa: SLF001
        manifest,
        repo_commit=COMMIT,
        host=host,
        runtime=_runtime(),
    )

    assert report["ready"] is True
    assert captured["wheel_sha256"] == WHEEL_SHA
    assert captured["rust_extension_sha256"] == EXTENSION_SHA
    assert (
        captured["required_entity_adapter"]
        == preflight.REQUIRED_LEARNER_ENTITY_ADAPTER
    )
    assert "rust_wheel_install_binding" in preflight.REMOTE_INSPECT
    assert "rust_extension_identity" in preflight.REMOTE_INSPECT
    assert "rust_v6_action_context" in preflight.REMOTE_INSPECT

