from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import production_runtime_contract as runtime


def test_canonical_production_runtime_is_exact_and_self_consistent() -> None:
    payload = runtime.load_runtime_contract()

    assert payload == {
        "schema_version": runtime.SCHEMA,
        "python_version": "3.11.15",
        "torch_version": "2.11.0+cu128",
        "torch_cuda_version": "12.8",
        "catanatron_rs_version": "0.1.8",
        "catanatron_rs_wheel_filename": (
            "catanatron_rs-0.1.8-cp311-cp311-manylinux_2_34_x86_64.whl"
        ),
        "catanatron_rs_wheel_sha256": (
            "f311673efa4d1e697736415cdff38ebb1e7eed3f109b241d5a5097cfb6d7dc2e"
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


def test_installer_line_protocol_has_fixed_complete_order(capsys) -> None:
    assert runtime.main(["--format", "lines"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "3.11.15",
        "2.11.0+cu128",
        "12.8",
        "0.1.8",
        "catanatron_rs-0.1.8-cp311-cp311-manylinux_2_34_x86_64.whl",
        "f311673efa4d1e697736415cdff38ebb1e7eed3f109b241d5a5097cfb6d7dc2e",
        "2.4.6",
        "3.6.1",
        "1.3.0",
        "0.25.0",
        "1.17.1",
        "2.2.0",
        "580.105.08",
    ]
