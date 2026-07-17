#!/usr/bin/env python3
"""Load the exact interpreter/runtime identity admitted by production A1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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
