from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from tools.fleet import a1_executor_bridge as bridge
from tools.fleet import a1_production_executor as executor


def _plan() -> dict:
    public = {
        "schema_version": executor.RECEIPT_SCHEMA,
        "status": "dry_run",
        "contract_sha256": "sha256:" + "a" * 64,
        "repo_artifacts_sha256": executor._digest([]),
    }
    public["plan_sha256"] = executor._digest(public)
    return {**public, "_private": {"repo_artifacts": []}}


def _frozen_repo(tmp_path: Path, plan: dict) -> Path:
    root = tmp_path / "frozen"
    source = root / "tools/fleet/a1_production_executor.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import json, pathlib\n"
        f"PLAN = json.loads({json.dumps(json.dumps(plan))})\n"
        "ROOT = pathlib.Path(__file__).resolve().parents[2]\n"
        "def build_plan(**_kwargs):\n"
        "    if pathlib.Path.cwd().resolve() != ROOT: raise RuntimeError('wrong frozen cwd')\n"
        "    return PLAN\n",
        encoding="utf-8",
    )
    return root


def test_frozen_plan_bridge_preserves_public_plan_and_binds_both_executors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _plan()
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps({"contract_sha256": original["contract_sha256"]}),
        encoding="utf-8",
    )
    original["operator_manifests"] = {
        "lock": {"sha256": executor._sha256(lock_path)}
    }
    public = executor._public(original)
    public.pop("plan_sha256")
    original["plan_sha256"] = executor._digest(public)
    root = _frozen_repo(tmp_path, original)
    monkeypatch.setenv("PYTHONPATH", str(Path(executor.__file__).resolve().parents[2]))
    monkeypatch.setenv("PYTHONHOME", "/invalid/hardened-python-home")
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    monkeypatch.setattr(executor, "build_plan", lambda **_kwargs: original)
    frozen_executor = root / "tools/fleet/a1_production_executor.py"
    result = bridge.build_bridged_plan(
        frozen_repo=root,
        frozen_executor_sha256=executor._sha256(frozen_executor),
        hardened_executor_sha256=executor._sha256(Path(executor.__file__)),
        bridge_sha256=executor._sha256(Path(bridge.__file__)),
        lock_path=lock_path,
        render_path=tmp_path / "render.json",
        hosts_path=tmp_path / "hosts.json",
        receipt_path=tmp_path / "executor.receipt.json",
    )

    assert executor._public(result) == executor._public(original)
    assert result["plan_sha256"] == original["plan_sha256"]
    typed = result["_private"]["executor_bridge"]
    assert typed["frozen_executor"] == {
        "path": str(frozen_executor),
        "sha256": executor._sha256(frozen_executor),
    }
    assert typed["hardened_executor"]["sha256"] == executor._sha256(
        Path(executor.__file__)
    )
    assert typed["bridge_tool"] == {
        "path": str(Path(bridge.__file__).resolve()),
        "sha256": executor._sha256(Path(bridge.__file__)),
    }
    assert executor._execution_repo_root(result) == root


def test_bridge_rejects_either_code_digest_drift(tmp_path: Path) -> None:
    original = _plan()
    root = _frozen_repo(tmp_path, original)
    frozen_digest = executor._sha256(
        root / "tools/fleet/a1_production_executor.py"
    )
    hardened_digest = executor._sha256(Path(executor.__file__))
    bridge_digest = executor._sha256(Path(bridge.__file__))
    with pytest.raises(bridge.BridgeError, match="frozen executor digest"):
        bridge.bind_plan(
            original,
            frozen_repo=root,
            expected_frozen_executor_sha256="sha256:" + "0" * 64,
            expected_hardened_executor_sha256=hardened_digest,
            expected_bridge_sha256=bridge_digest,
        )
    with pytest.raises(bridge.BridgeError, match="hardened executor digest"):
        bridge.bind_plan(
            original,
            frozen_repo=root,
            expected_frozen_executor_sha256=frozen_digest,
            expected_hardened_executor_sha256="sha256:" + "0" * 64,
            expected_bridge_sha256=bridge_digest,
        )
    with pytest.raises(bridge.BridgeError, match="bridge tool digest"):
        bridge.bind_plan(
            original,
            frozen_repo=root,
            expected_frozen_executor_sha256=frozen_digest,
            expected_hardened_executor_sha256=hardened_digest,
            expected_bridge_sha256="sha256:" + "0" * 64,
        )


def test_bridge_receipt_is_immutable_and_exactly_replayable(tmp_path: Path) -> None:
    original = _plan()
    root = _frozen_repo(tmp_path, original)
    plan = bridge.bind_plan(
        original,
        frozen_repo=root,
        expected_frozen_executor_sha256=executor._sha256(
            root / "tools/fleet/a1_production_executor.py"
        ),
        expected_hardened_executor_sha256=executor._sha256(Path(executor.__file__)),
        expected_bridge_sha256=executor._sha256(Path(bridge.__file__)),
    )
    path = tmp_path / "bridge.receipt.json"
    first = bridge.seal_bridge_receipt(path, plan)
    assert bridge.seal_bridge_receipt(path, plan) == first
    assert stat.S_IMODE(path.stat().st_mode) == 0o444

    os.chmod(path, 0o600)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(bridge.BridgeError, match="binds different execution code"):
        bridge.seal_bridge_receipt(path, plan)


def test_repo_sources_are_taken_from_frozen_root_and_rehashed(tmp_path: Path) -> None:
    root = tmp_path / "frozen"
    source = root / "pkg/runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 'frozen'\n", encoding="utf-8")
    record = {
        "path": "pkg/runtime.py",
        "sha256": executor._sha256(source),
        "mode": 0o444,
    }
    assert executor._repo_files([record], repo_root=root) == [source]
    source.write_text("VALUE = 'drift'\n", encoding="utf-8")
    with pytest.raises(executor.ExecutorError, match="source drift"):
        executor._repo_files([record], repo_root=root)
