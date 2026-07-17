from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace

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


class _FakeDistribution:
    def __init__(
        self,
        *,
        version: str,
        extension: Path,
        wheel_sha256: str,
    ) -> None:
        self.version = version
        self.files = [Path("catanatron_rs/catanatron_rs.so")]
        self._extension = extension
        self._wheel_sha256 = wheel_sha256

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return json.dumps(
            {
                "archive_info": {
                    "hashes": {"sha256": self._wheel_sha256},
                }
            }
        )

    def locate_file(self, _record: object) -> Path:
        return self._extension


def _native_contract_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, str]]:
    extension = tmp_path / "catanatron_rs.so"
    extension.write_bytes(b"sealed native extension")
    payload = runtime.load_runtime_contract()
    payload["catanatron_rs_extension_sha256"] = hashlib.sha256(
        extension.read_bytes()
    ).hexdigest()
    payload["catanatron_rs_wheel_sha256"] = "a" * 64
    contract = tmp_path / "runtime.json"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    distribution = _FakeDistribution(
        version=payload["catanatron_rs_version"],
        extension=extension,
        wheel_sha256=payload["catanatron_rs_wheel_sha256"],
    )
    monkeypatch.setattr(runtime.metadata, "distribution", lambda _name: distribution)
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(extension)),
    )
    return contract, extension, payload


def test_native_runtime_attestation_binds_archive_and_loaded_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, extension, payload = _native_contract_fixture(tmp_path, monkeypatch)

    identity = runtime.assert_native_runtime_contract(contract)

    assert identity == {
        "version": payload["catanatron_rs_version"],
        "wheel_sha256": payload["catanatron_rs_wheel_sha256"],
        "extension_path": str(extension),
        "extension_sha256": payload["catanatron_rs_extension_sha256"],
    }


def test_native_runtime_attestation_accepts_shared_venv_parent_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real-venv"
    real.mkdir()
    extension = real / "catanatron_rs.so"
    extension.write_bytes(b"sealed native extension")
    alias = tmp_path / "checkout-venv"
    alias.symlink_to(real, target_is_directory=True)
    located = alias / extension.name
    payload = runtime.load_runtime_contract()
    payload["catanatron_rs_extension_sha256"] = hashlib.sha256(
        extension.read_bytes()
    ).hexdigest()
    payload["catanatron_rs_wheel_sha256"] = "a" * 64
    contract = tmp_path / "runtime.json"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    distribution = _FakeDistribution(
        version=payload["catanatron_rs_version"],
        extension=located,
        wheel_sha256=payload["catanatron_rs_wheel_sha256"],
    )
    monkeypatch.setattr(runtime.metadata, "distribution", lambda _name: distribution)
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(extension)),
    )

    identity = runtime.assert_native_runtime_contract(contract)

    assert identity["extension_path"] == str(extension)


@pytest.mark.parametrize("drift", ("version", "wheel", "extension"))
def test_native_runtime_attestation_refuses_installed_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    contract, extension, payload = _native_contract_fixture(tmp_path, monkeypatch)
    if drift == "version":
        distribution = _FakeDistribution(
            version="0.1.12",
            extension=extension,
            wheel_sha256=payload["catanatron_rs_wheel_sha256"],
        )
        monkeypatch.setattr(
            runtime.metadata, "distribution", lambda _name: distribution
        )
    elif drift == "wheel":
        distribution = _FakeDistribution(
            version=payload["catanatron_rs_version"],
            extension=extension,
            wheel_sha256="b" * 64,
        )
        monkeypatch.setattr(
            runtime.metadata, "distribution", lambda _name: distribution
        )
    else:
        extension.write_bytes(b"unsealed replacement")

    with pytest.raises(runtime.RuntimeContractError, match=drift):
        runtime.assert_native_runtime_contract(contract)
