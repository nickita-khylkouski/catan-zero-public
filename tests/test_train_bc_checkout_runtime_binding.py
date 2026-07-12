from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import train_bc


REPO = Path(__file__).resolve().parents[1]


def test_runtime_binding_names_exact_checkout_and_critical_modules() -> None:
    binding = train_bc._assert_checkout_runtime_binding()

    assert binding["schema_version"] == "train-bc-checkout-runtime-v1"
    assert Path(binding["trainer"]).samefile(REPO / "tools" / "train_bc.py")
    assert Path(binding["source_root"]).samefile(REPO / "src")
    assert binding["binding_sha256"].startswith("sha256:")
    modules = binding["modules"]
    for name in (
        "catan_zero",
        "catan_zero.rl.entity_token_policy",
        "catan_zero.rl.optim_state",
    ):
        path = Path(modules[name]["path"])
        assert path.is_relative_to((REPO / "src").resolve())
        assert modules[name]["sha256"].startswith("sha256:")


def test_runtime_binding_refuses_preloaded_foreign_project_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = sys.modules["catan_zero.rl.optim_state"]
    foreign = tmp_path / "old-checkout/src/catan_zero/rl/optim_state.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("# stale\n", encoding="utf-8")
    monkeypatch.setattr(module, "__file__", str(foreign))

    with pytest.raises(RuntimeError, match="checkout/runtime package skew"):
        train_bc._assert_checkout_runtime_binding()


def test_direct_script_ignores_ambient_stale_pythonpath(tmp_path: Path) -> None:
    stale = tmp_path / "stale"
    package = stale / "catan_zero"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('ambient stale package imported')\n", encoding="utf-8"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(stale)
    completed = subprocess.run(
        [sys.executable, str(REPO / "tools" / "train_bc.py"), "--help"],
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ambient stale package imported" not in completed.stderr


def test_checkpoint_metadata_and_report_persist_runtime_binding() -> None:
    binding = train_bc._assert_checkout_runtime_binding()
    args = SimpleNamespace(
        value_head_type="mse",
        value_loss_weight=0.25,
        value_categorical_loss_weight=0.0,
        hlgauss_scalar_aux_loss_weight=0.0,
        value_target_lambda=1.0,
        value_hlgauss_sigma_ratio=0.75,
        value_root_blend_phases="",
        truncated_vp_margin_value_weight=0.0,
        checkout_runtime_binding=binding,
        a1_contract_sha256=None,
    )
    metadata = train_bc._value_training_metadata(
        args,
        scalar_weight=0.25,
        categorical_weight=0.0,
        categorical_bins=0,
        optimizer_steps=1,
        completed_epochs=1,
        scalar_training_weight_sum=1.0,
        categorical_training_weight_sum=0.0,
    )
    assert metadata["checkout_runtime_binding"] == binding

    source = (REPO / "tools" / "train_bc.py").read_text(encoding="utf-8")

    assert '"checkout_runtime_binding": checkout_runtime_binding' in source
