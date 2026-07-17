from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest

from tools import production_runtime_contract as runtime


def test_canonical_production_runtime_is_exact_and_self_consistent() -> None:
    payload = runtime.load_runtime_contract()

    assert payload == {
        "schema_version": runtime.SCHEMA,
        "python_version": "3.11.15",
        "torch_version": "2.11.0+cu128",
        "torch_cuda_version": "12.8",
        "catanatron_rs_version": "0.1.13",
        "catanatron_rs_extension_sha256": (
            "e9f85102c65b98a7d0e3f89209c28e10b84159a006e0ef5589c776cc60dbefa4"
        ),
        "catanatron_rs_wheel_filename": (
            "catanatron_rs-0.1.13-cp311-cp311-manylinux_2_34_x86_64.whl"
        ),
        "catanatron_rs_wheel_sha256": (
            "2e0c6fda344ae85dd1bccd1f5474acea92218a0b561cdde0e43b386ec284fb97"
        ),
        "numpy_version": "2.4.6",
        "networkx_version": "3.6.1",
        "nvidia_driver_version": "580.105.08",
        "gymnasium_version": "1.3.0",
        "zstandard_version": "0.25.0",
        "scipy_version": "1.17.1",
        "whr_version": "2.2.0",
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.pop("torch_version"),
        lambda value: value.__setitem__("python_version", "3.11.invalid"),
        lambda value: value.__setitem__("catanatron_rs_wheel_sha256", "0" * 63),
        lambda value: value.__setitem__("catanatron_rs_extension_sha256", "0" * 63),
        lambda value: value.__setitem__(
            "catanatron_rs_wheel_filename", "catanatron_rs-0.1.7-fake.whl"
        ),
    ),
)
def test_runtime_contract_rejects_missing_or_inconsistent_identity(
    tmp_path: Path, mutation
) -> None:
    payload = runtime.load_runtime_contract()
    mutation(payload)
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runtime.RuntimeContractError):
        runtime.load_runtime_contract(path)


def test_legacy_runtime_contract_without_extension_digest_remains_readable(
    tmp_path: Path,
) -> None:
    payload = runtime.load_runtime_contract()
    payload.pop("catanatron_rs_extension_sha256")
    path = tmp_path / "legacy-runtime-v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert runtime.load_runtime_contract(path) == payload


def test_installer_line_protocol_has_fixed_complete_order(capsys) -> None:
    assert runtime.main(["--format", "lines"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "3.11.15",
        "2.11.0+cu128",
        "12.8",
        "0.1.13",
        "catanatron_rs-0.1.13-cp311-cp311-manylinux_2_34_x86_64.whl",
        "2e0c6fda344ae85dd1bccd1f5474acea92218a0b561cdde0e43b386ec284fb97",
        "2.4.6",
        "3.6.1",
        "1.3.0",
        "0.25.0",
        "1.17.1",
        "2.2.0",
        "580.105.08",
    ]


def _fake_python(path: Path, *, version: str, stderr: str = "") -> Path:
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' {version!r}\n"
        + (f"printf '%s\\n' {stderr!r} >&2\n" if stderr else ""),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_python_selector_accepts_only_exact_contracted_patch(
    tmp_path: Path, capsys
) -> None:
    exact = _fake_python(tmp_path / "python-exact", version="3.11.15")
    drifted = _fake_python(tmp_path / "python-drifted", version="3.11.14")
    noisy = _fake_python(
        tmp_path / "python-noisy", version="3.11.15", stderr="unexpected"
    )

    assert runtime.interpreter_version(str(exact)) == "3.11.15"
    assert runtime.main(["--check-python", str(exact)]) == 0
    assert "Python runtime exact: 3.11.15" in capsys.readouterr().out
    for executable in (drifted, noisy, tmp_path / "missing"):
        assert runtime.main(["--check-python", str(executable)]) == 3
        assert "REFUSED: Python patch drift" in capsys.readouterr().out
