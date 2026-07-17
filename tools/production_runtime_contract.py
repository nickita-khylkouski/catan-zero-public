#!/usr/bin/env python3
"""Load the exact interpreter/runtime identity admitted by production A1."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPO_ROOT / "configs/runtime/a1_production_runtime.json"
SCHEMA = "a1-production-runtime-v1"
REQUIRED_KEYS = {
    "schema_version",
    "python_version",
    "torch_version",
    "torch_cuda_version",
    "catanatron_rs_version",
    "catanatron_rs_wheel_filename",
    "catanatron_rs_wheel_sha256",
    "numpy_version",
    "networkx_version",
    "nvidia_driver_version",
    "gymnasium_version",
    "zstandard_version",
    "scipy_version",
    "whr_version",
}
OPTIONAL_KEYS = {"catanatron_rs_extension_sha256"}
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+(?:\+[A-Za-z0-9.]+)?$")
_WHEEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*\.whl$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RuntimeContractError(RuntimeError):
    """The production runtime contract is missing or malformed."""


def load_runtime_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, str]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeContractError(f"cannot load production runtime contract: {error}") from error
    if (
        not isinstance(payload, dict)
        or not REQUIRED_KEYS.issubset(payload)
        or set(payload).difference(REQUIRED_KEYS | OPTIONAL_KEYS)
    ):
        raise RuntimeContractError("production runtime contract key set drift")
    if payload.get("schema_version") != SCHEMA:
        raise RuntimeContractError("production runtime contract schema drift")
    for key in (
        "python_version",
        "torch_version",
        "torch_cuda_version",
        "catanatron_rs_version",
        "numpy_version",
        "networkx_version",
        "nvidia_driver_version",
        "gymnasium_version",
        "zstandard_version",
        "scipy_version",
        "whr_version",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or not _VERSION.fullmatch(value):
            raise RuntimeContractError(f"invalid production runtime {key}")
    wheel = payload.get("catanatron_rs_wheel_filename")
    if not isinstance(wheel, str) or not _WHEEL.fullmatch(wheel):
        raise RuntimeContractError("invalid production runtime wheel filename")
    wheel_sha256 = payload.get("catanatron_rs_wheel_sha256")
    if not isinstance(wheel_sha256, str) or not _SHA256.fullmatch(wheel_sha256):
        raise RuntimeContractError("invalid production runtime wheel SHA-256")
    extension_sha256 = payload.get("catanatron_rs_extension_sha256")
    if extension_sha256 is not None and (
        not isinstance(extension_sha256, str)
        or not _SHA256.fullmatch(extension_sha256)
    ):
        raise RuntimeContractError("invalid production runtime extension SHA-256")
    expected_prefix = f"catanatron_rs-{payload['catanatron_rs_version']}-"
    if not wheel.startswith(expected_prefix):
        raise RuntimeContractError("native wheel filename/version drift")
    return {str(key): str(value) for key, value in payload.items()}


def interpreter_version(executable: str) -> str | None:
    """Return an isolated interpreter's exact patch version, or ``None``.

    The installer must not treat an arbitrary ``python3.11`` as the contracted
    3.11.15 runtime.  Probe the executable itself before allowing it to create
    the production venv; any execution error, extra output, or timeout selects
    the exact ``uv`` bootstrap path instead.
    """

    try:
        result = subprocess.run(
            [
                executable,
                "-I",
                "-c",
                "import platform; print(platform.python_version())",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.splitlines()
    if result.returncode != 0 or result.stderr or len(lines) != 1:
        return None
    return lines[0]


def _stable_file_sha256(path: Path) -> str:
    """Hash one canonical regular file without accepting replacement races."""

    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RuntimeContractError(
                f"native runtime must be a regular non-symlink file: {path}"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeContractError(
                    f"native runtime changed while opening: {path}"
                )
            digest = hashlib.sha256()
            while block := os.read(descriptor, 8 * 1024 * 1024):
                digest.update(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise RuntimeContractError(
            f"cannot attest native runtime {path}: {error}"
        ) from error
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(opened, field) != getattr(after, field)
        for field in identity_fields
    ):
        raise RuntimeContractError(f"native runtime changed while hashing: {path}")
    return digest.hexdigest()


def assert_native_runtime_contract(
    path: Path = DEFAULT_CONTRACT,
) -> dict[str, str]:
    """Prove this interpreter loaded the one sealed native wheel.

    A Git checkout update cannot update an existing virtualenv.  Version alone
    is also insufficient because two local wheel builds can share a version.
    Bind all three identities once at process launch: distribution version,
    PEP 610 wheel archive digest, and loaded extension bytes.
    """

    expected = load_runtime_contract(path)
    expected_extension = expected.get("catanatron_rs_extension_sha256")
    if expected_extension is None:
        raise RuntimeContractError(
            "production runtime contract has no native extension SHA-256"
        )
    try:
        distribution = metadata.distribution("catanatron-rs")
    except metadata.PackageNotFoundError as error:
        raise RuntimeContractError(
            "catanatron-rs is not installed under the executing interpreter"
        ) from error
    if distribution.version != expected["catanatron_rs_version"]:
        raise RuntimeContractError(
            "catanatron-rs version drift: "
            f"expected {expected['catanatron_rs_version']}, "
            f"got {distribution.version}"
        )

    try:
        direct_url_raw = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_raw) if direct_url_raw is not None else None
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeContractError(
            f"installed catanatron-rs has malformed direct_url.json: {error}"
        ) from error
    archive = direct_url.get("archive_info") if isinstance(direct_url, dict) else None
    stated: set[str] = set()
    if isinstance(archive, dict):
        direct_hash = archive.get("hash")
        if isinstance(direct_hash, str):
            stated.add(direct_hash)
        hashes = archive.get("hashes")
        if isinstance(hashes, dict) and isinstance(hashes.get("sha256"), str):
            stated.add(f"sha256={hashes['sha256']}")
    expected_archive = "sha256=" + expected["catanatron_rs_wheel_sha256"]
    if stated != {expected_archive}:
        raise RuntimeContractError(
            "installed catanatron-rs wheel archive drift: "
            f"expected {expected_archive}, got {sorted(stated)}"
        )

    extension_records = [
        record
        for record in (distribution.files or ())
        if str(record).endswith((".so", ".pyd", ".dylib"))
    ]
    if len(extension_records) != 1:
        raise RuntimeContractError(
            "installed catanatron-rs must contain exactly one native extension; "
            f"found {len(extension_records)}"
        )
    relative_extension = Path(str(extension_records[0]))
    if relative_extension.is_absolute() or ".." in relative_extension.parts:
        raise RuntimeContractError(
            f"installed native extension record is unsafe: {relative_extension}"
        )
    located_extension = Path(
        os.path.abspath(os.fspath(distribution.locate_file(extension_records[0])))
    )
    try:
        if located_extension.is_symlink():
            raise RuntimeContractError(
                "installed native extension itself may not be a symlink: "
                f"{located_extension}"
            )
        extension = located_extension.resolve(strict=True)
    except OSError as error:
        raise RuntimeContractError(
            f"cannot resolve installed native extension {located_extension}: {error}"
        ) from error

    try:
        native = importlib.import_module("catanatron_rs.catanatron_rs")
    except ImportError as error:
        raise RuntimeContractError(
            f"cannot import catanatron-rs native extension: {error}"
        ) from error
    loaded_raw = getattr(native, "__file__", None)
    if not isinstance(loaded_raw, str) or not loaded_raw:
        raise RuntimeContractError(
            "loaded catanatron-rs native extension has no file identity"
        )
    try:
        loaded = Path(os.path.abspath(loaded_raw)).resolve(strict=True)
    except OSError as error:
        raise RuntimeContractError(
            f"cannot resolve loaded native extension {loaded_raw}: {error}"
        ) from error
    if loaded != extension:
        raise RuntimeContractError(
            "loaded catanatron-rs extension path drift: "
            f"installed={extension} loaded={loaded}"
        )
    actual_extension = _stable_file_sha256(extension)
    if actual_extension != expected_extension:
        raise RuntimeContractError(
            "loaded catanatron-rs extension drift: "
            f"expected {expected_extension}, got {actual_extension}"
        )
    return {
        "version": distribution.version,
        "wheel_sha256": expected["catanatron_rs_wheel_sha256"],
        "extension_path": str(extension),
        "extension_sha256": actual_extension,
    }


def assert_native_runtime_for_python(
    executable: str | Path,
    path: Path = DEFAULT_CONTRACT,
) -> None:
    """Run the exact native attestation under a prospective child interpreter."""

    command = [
        str(executable),
        str(Path(__file__).resolve()),
        "--contract",
        str(path.resolve()),
        "--check-native",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeContractError(
            f"cannot attest child native runtime: {error}"
        ) from error
    if result.returncode != 0:
        detail = result.stdout.strip() or result.stderr.strip() or "no diagnostic"
        raise RuntimeContractError(f"child native runtime refused: {detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--format",
        choices=("json", "lines"),
        default="json",
        help="lines emits the installer-safe fixed field order",
    )
    parser.add_argument(
        "--check-python",
        metavar="EXECUTABLE",
        help="succeed only when EXECUTABLE is the contracted Python patch",
    )
    parser.add_argument(
        "--check-native",
        action="store_true",
        help="succeed only when this interpreter loaded the sealed native wheel",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = load_runtime_contract(args.contract)
    except RuntimeContractError as error:
        print(f"REFUSED: {error}")
        return 2
    if args.check_python is not None:
        actual = interpreter_version(args.check_python)
        expected = payload["python_version"]
        if actual != expected:
            print(
                f"REFUSED: Python patch drift: expected {expected}, got "
                f"{actual or 'unavailable'}"
            )
            return 3
        print(f"Python runtime exact: {actual}")
        return 0
    if args.check_native:
        try:
            identity = assert_native_runtime_contract(args.contract)
        except RuntimeContractError as error:
            print(f"REFUSED: {error}")
            return 4
        print(json.dumps(identity, sort_keys=True, separators=(",", ":")))
        return 0
    if args.format == "lines":
        for key in (
            "python_version",
            "torch_version",
            "torch_cuda_version",
            "catanatron_rs_version",
            "catanatron_rs_wheel_filename",
            "catanatron_rs_wheel_sha256",
            "numpy_version",
            "networkx_version",
            "gymnasium_version",
            "zstandard_version",
            "scipy_version",
            "whr_version",
            "nvidia_driver_version",
        ):
            print(payload[key])
    else:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
