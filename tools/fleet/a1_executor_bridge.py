#!/usr/bin/env python3
"""Run a frozen A1 plan with a separately hash-bound hardened executor.

The frozen executor builds and verifies the exact historical public/private
plan in an isolated interpreter.  This bridge adds only private execution
metadata, so the sealed ``plan_sha256`` and receipt identity do not change.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.fleet import a1_production_executor as hardened  # noqa: E402


class BridgeError(RuntimeError):
    pass


BRIDGE_RECEIPT_SCHEMA = "a1-frozen-plan-hardened-executor-bridge-receipt-v1"


def seal_bridge_receipt(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Create or exactly replay one immutable bridge identity receipt."""
    payload = {
        "schema_version": BRIDGE_RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "bridge": plan["_private"]["executor_bridge"],
    }
    payload["receipt_sha256"] = hardened._digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BridgeError(f"cannot read immutable bridge receipt: {error}") from error
        if existing != payload:
            raise BridgeError("immutable bridge receipt binds different execution code")
        return payload
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        return seal_bridge_receipt(path, plan)
    os.fchmod(descriptor, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return payload


def _load_frozen_plan(
    *,
    frozen_repo: Path,
    lock_path: Path,
    render_path: Path,
    hosts_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Build the plan with no current-checkout modules in the interpreter."""
    script = r'''import importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(strict=True)
executor_path=(root/'tools/fleet/a1_production_executor.py').resolve(strict=True)
sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('_a1_frozen_executor',executor_path)
if spec is None or spec.loader is None: raise SystemExit('cannot load frozen executor spec')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
plan=module.build_plan(lock_path=pathlib.Path(sys.argv[2]),render_path=pathlib.Path(sys.argv[3]),hosts_path=pathlib.Path(sys.argv[4]),receipt_path=pathlib.Path(sys.argv[5]))
print(json.dumps(plan,sort_keys=True,separators=(',',':')))'''
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(frozen_repo),
            str(lock_path),
            str(render_path),
            str(hosts_path),
            str(receipt_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise BridgeError(f"frozen executor refused plan construction: {detail}")
    try:
        plan = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise BridgeError("frozen executor returned invalid plan JSON") from error
    if not isinstance(plan, dict) or not isinstance(plan.get("_private"), dict):
        raise BridgeError("frozen executor returned an invalid typed plan")
    return plan


def bind_plan(
    plan: dict[str, Any],
    *,
    frozen_repo: Path,
    expected_frozen_executor_sha256: str,
    expected_hardened_executor_sha256: str,
) -> dict[str, Any]:
    """Privately bind two executors without changing the public plan digest."""
    hardened._verify_plan_digest(plan)
    public_before = hardened._public(plan)
    root = frozen_repo.resolve(strict=True)
    frozen_path = (root / "tools/fleet/a1_production_executor.py").resolve(strict=True)
    hardened_path = Path(hardened.__file__).resolve(strict=True)
    frozen_digest = hardened._sha256(frozen_path)
    hardened_digest = hardened._sha256(hardened_path)
    if frozen_digest != expected_frozen_executor_sha256:
        raise BridgeError("frozen executor digest does not equal explicit binding")
    if hardened_digest != expected_hardened_executor_sha256:
        raise BridgeError("hardened executor digest does not equal explicit binding")
    bridge = {
        "schema_version": hardened.BRIDGE_SCHEMA,
        "frozen_repo_root": str(root),
        "frozen_executor": {"path": str(frozen_path), "sha256": frozen_digest},
        "hardened_executor": {
            "path": str(hardened_path),
            "sha256": hardened_digest,
        },
        "plan_sha256": plan["plan_sha256"],
        "repo_artifacts_sha256": plan["repo_artifacts_sha256"],
    }
    bridge["bridge_sha256"] = hardened._digest(bridge)
    private = dict(plan["_private"])
    private["executor_bridge"] = bridge
    result = {**plan, "_private": private}
    if hardened._public(result) != public_before:
        raise BridgeError("bridge changed the frozen public execution plan")
    hardened._execution_repo_root(result)
    return result


def build_bridged_plan(
    *,
    frozen_repo: Path,
    frozen_executor_sha256: str,
    hardened_executor_sha256: str,
    lock_path: Path,
    render_path: Path,
    hosts_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    plan = _load_frozen_plan(
        frozen_repo=frozen_repo,
        lock_path=lock_path,
        render_path=render_path,
        hosts_path=hosts_path,
        receipt_path=receipt_path,
    )
    portable_plan = hardened.build_plan(
        lock_path=lock_path,
        render_path=render_path,
        hosts_path=hosts_path,
        receipt_path=receipt_path,
        repo_root=frozen_repo,
    )
    if portable_plan != plan:
        raise BridgeError(
            "portable replay does not exactly equal the frozen executor plan"
        )
    return bind_plan(
        portable_plan,
        frozen_repo=frozen_repo,
        expected_frozen_executor_sha256=frozen_executor_sha256,
        expected_hardened_executor_sha256=hardened_executor_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "stop"):
        item = sub.add_parser(name)
        item.add_argument("--frozen-repo", required=True, type=Path)
        item.add_argument("--frozen-executor-sha256", required=True)
        item.add_argument("--hardened-executor-sha256", required=True)
        item.add_argument("--lock", required=True, type=Path)
        item.add_argument("--render", required=True, type=Path)
        item.add_argument("--hosts", required=True, type=Path)
        item.add_argument("--receipt", required=True, type=Path)
        item.add_argument("--bridge-receipt", required=True, type=Path)
    sub.choices["run"].add_argument("--resume", action="store_true")
    sub.choices["run"].add_argument("--go", action="store_true")
    sub.choices["stop"].add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_bridged_plan(
            frozen_repo=args.frozen_repo,
            frozen_executor_sha256=args.frozen_executor_sha256,
            hardened_executor_sha256=args.hardened_executor_sha256,
            lock_path=args.lock,
            render_path=args.render,
            hosts_path=args.hosts,
            receipt_path=args.receipt,
        )
        seal_bridge_receipt(args.bridge_receipt, plan)
        if args.command == "status":
            result = hardened.status(plan, receipt_path=args.receipt)
        elif args.command == "stop":
            result = hardened.stop_execution(
                plan, receipt_path=args.receipt, go=bool(args.go)
            )
        elif args.go:
            result = hardened.execute(
                plan, receipt_path=args.receipt, resume=bool(args.resume)
            )
        else:
            result = {
                "schema_version": hardened.BRIDGE_SCHEMA,
                "bridge": plan["_private"]["executor_bridge"],
                "plan": hardened._public(plan),
            }
    except (BridgeError, hardened.ExecutorError, OSError) as error:
        print(f"REFUSING: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
