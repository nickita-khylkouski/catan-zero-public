#!/usr/bin/env python3
"""Seal a catanatron-rs wheel to the extension imported by a worker venv."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping


RUNTIME_IDENTITY_SCHEMA = "a1-native-runtime-identity-v1"
WHEEL_RECEIPT_SCHEMA = "catanatron-rs-wheel-build-receipt-v2"


class RuntimeIdentityError(RuntimeError):
    """The installed native runtime cannot be tied to the sealed wheel."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _stable_regular_file(path: Path) -> tuple[bytes, str]:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            raise RuntimeIdentityError(
                f"artifact is not a canonical regular file: {path}"
            )
        data = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise RuntimeIdentityError(f"cannot read artifact {path}: {error}") from error
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(before, field) != getattr(after, field) for field in identity_fields
    ):
        raise RuntimeIdentityError(f"artifact changed while being read: {path}")
    return data, "sha256:" + hashlib.sha256(data).hexdigest()


def _normalized_sha256(value: object, *, field: str) -> str:
    text = str(value).removeprefix("sha256:")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeIdentityError(f"{field} is not a canonical SHA-256 digest")
    return "sha256:" + text


def _git(command: list[str], *, repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", *command],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeIdentityError(f"Git identity probe failed: {error}") from error


def _wheel_extension_identity(wheel: Path) -> dict[str, str]:
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.endswith(".so") and not name.startswith("/")
            ]
            if len(members) != 1:
                raise RuntimeIdentityError(
                    f"native wheel must contain exactly one extension, found {members}"
                )
            member = members[0]
            if ".." in Path(member).parts:
                raise RuntimeIdentityError(
                    f"native wheel has an unsafe extension path: {member}"
                )
            digest = hashlib.sha256()
            with archive.open(member) as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeIdentityError(
            f"cannot inspect native wheel {wheel}: {error}"
        ) from error
    return {"member": member, "sha256": "sha256:" + digest.hexdigest()}


def _installed_runtime(
    *, repo: Path, python: Path, expected_wheel_sha256: str
) -> dict[str, Any]:
    """Run the existing fleet probes under the exact worker interpreter."""

    probe = r"""
import importlib
import json
import sys
from importlib.metadata import distribution
from pathlib import Path
from tools.fleet.a1_h100_eval_fleet import (
    NATIVE_REQUIRED_CAPABILITIES,
    _assert_installed_native_wheel_sha256,
    _native_runtime_sha256,
)
import catanatron_rs

expected = sys.argv[1]
_assert_installed_native_wheel_sha256(expected)
metadata = distribution('catanatron-rs')
native = importlib.import_module('catanatron_rs.catanatron_rs')
capability_fn = getattr(catanatron_rs, 'gumbel_search_capabilities', None)
if not callable(capability_fn):
    raise RuntimeError('catanatron_rs has no gumbel_search_capabilities')
capabilities = sorted(set(map(str, capability_fn())))
missing = sorted(set(NATIVE_REQUIRED_CAPABILITIES) - set(capabilities))
if missing:
    raise RuntimeError(f'native runtime lacks required capabilities: {missing}')
print(json.dumps({
    'distribution_name': str(metadata.metadata.get('Name') or 'catanatron-rs'),
    'distribution_version': str(metadata.version),
    'package_path': str(Path(catanatron_rs.__file__).resolve(strict=True)),
    'extension_path': str(Path(native.__file__).resolve(strict=True)),
    'extension_sha256': _native_runtime_sha256(),
    'capabilities': capabilities,
}, sort_keys=True))
"""
    environment = os.environ.copy()
    roots = [str(repo / "src"), str(repo)]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    try:
        completed = subprocess.run(
            [str(python), "-c", probe, expected_wheel_sha256],
            cwd=repo,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "")
        raise RuntimeIdentityError(
            "native runtime attestation failed under the worker interpreter: "
            f"{stderr or error}"
        ) from error
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeIdentityError(
            f"native runtime attestation emitted malformed output: {completed.stdout!r}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeIdentityError("native runtime attestation is not a JSON object")
    return value


def inspect(receipt_path: Path, *, repo: Path, python: Path) -> dict[str, Any]:
    """Bind build receipt, release inventory, wheel, install, and loaded ELF."""

    receipt_path = Path(os.path.abspath(os.fspath(receipt_path.expanduser())))
    receipt_bytes, receipt_file_sha256 = _stable_regular_file(receipt_path)
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeIdentityError(f"malformed native wheel receipt: {error}") from error
    if not isinstance(receipt, dict) or receipt.get("schema_version") != WHEEL_RECEIPT_SCHEMA:
        raise RuntimeIdentityError("native wheel build receipt schema drift")
    filename = receipt.get("wheel_filename")
    if (
        not isinstance(filename, str)
        or not filename.endswith(".whl")
        or Path(filename).name != filename
    ):
        raise RuntimeIdentityError("native wheel receipt has an unsafe filename")
    wheel = receipt_path.parent / filename
    _wheel_bytes, wheel_sha256 = _stable_regular_file(wheel)
    expected_wheel_sha256 = _normalized_sha256(
        receipt.get("wheel_sha256"), field="wheel receipt SHA-256"
    )
    if wheel_sha256 != expected_wheel_sha256:
        raise RuntimeIdentityError("native wheel bytes differ from build receipt")

    source_commit = str(receipt.get("source_commit", ""))
    source_tree = str(receipt.get("source_tree", ""))
    if len(source_commit) != 40 or len(source_tree) != 40:
        raise RuntimeIdentityError("native receipt has malformed Git source identity")
    if _git(["rev-parse", f"{source_commit}^{{tree}}"], repo=repo) != source_tree:
        raise RuntimeIdentityError("native receipt source tree does not match Git")

    inventory = repo / "native/catanatron-rs/WHEEL_SHA256SUMS"
    inventory_bytes, inventory_sha256 = _stable_regular_file(inventory)
    expected_line = expected_wheel_sha256.removeprefix("sha256:") + "  " + filename
    try:
        lines = inventory_bytes.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise RuntimeIdentityError("native release inventory is not UTF-8") from error
    if lines != [expected_line]:
        raise RuntimeIdentityError(
            "repository release inventory does not seal the supplied native wheel"
        )

    wheel_extension = _wheel_extension_identity(wheel)
    installed = _installed_runtime(
        repo=repo, python=python, expected_wheel_sha256=expected_wheel_sha256
    )
    runtime_sha256 = _normalized_sha256(
        installed.get("extension_sha256"), field="loaded extension SHA-256"
    )
    if runtime_sha256 != wheel_extension["sha256"]:
        raise RuntimeIdentityError(
            "loaded native extension bytes differ from the sealed wheel member"
        )
    capabilities = installed.get("capabilities")
    if not isinstance(capabilities, list) or capabilities != sorted(set(capabilities)):
        raise RuntimeIdentityError("native capability attestation is not canonical")

    value: dict[str, Any] = {
        "schema_version": RUNTIME_IDENTITY_SCHEMA,
        "wheel_build_receipt": {
            "path": str(receipt_path),
            "file_sha256": receipt_file_sha256,
            "schema_version": WHEEL_RECEIPT_SCHEMA,
            "source_commit": source_commit,
            "source_tree": source_tree,
        },
        "distribution": {
            "name": str(installed.get("distribution_name", "")),
            "version": str(installed.get("distribution_version", "")),
            "wheel_path": str(wheel),
            "wheel_filename": filename,
            "wheel_sha256": expected_wheel_sha256,
        },
        "extension": {
            "path": str(installed.get("extension_path", "")),
            "sha256": runtime_sha256,
            "wheel_member": wheel_extension["member"],
            "wheel_member_sha256": wheel_extension["sha256"],
        },
        "package_path": str(installed.get("package_path", "")),
        "release_inventory": {
            "path": str(inventory),
            "sha256": inventory_sha256,
        },
        "capabilities": capabilities,
    }
    value["identity_sha256"] = _digest(value)
    return value


def verify_record(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeIdentityError(
            "launch receipt lacks native runtime identity; restart with a sealed wheel"
        )
    unhashed = dict(value)
    declared = unhashed.pop("identity_sha256", None)
    if declared != _digest(unhashed):
        raise RuntimeIdentityError("native runtime identity semantic digest mismatch")
    extension = value.get("extension")
    capabilities = value.get("capabilities")
    if (
        value.get("schema_version") != RUNTIME_IDENTITY_SCHEMA
        or not isinstance(value.get("distribution"), Mapping)
        or not isinstance(extension, Mapping)
        or extension.get("sha256") != extension.get("wheel_member_sha256")
        or not isinstance(capabilities, list)
        or capabilities != sorted(set(capabilities))
    ):
        raise RuntimeIdentityError("native runtime identity is malformed")
    return value
