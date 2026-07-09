"""Startup guard tests for the LEGACY CPU factory (FIX 5, task #85 hygiene
batch): tools/modal_gumbel_factory.py has no `public_observation` knob, so
it's an armed footgun against masked-trained checkpoints. Both
local_entrypoints must refuse to run unless explicitly acknowledged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
# APPEND (not insert-at-0) so a real site-packages `modal` always wins over
# anything under tools/. History: tools/ used to hold a `modal/` package
# (tools/modal/modal_gumbel_factory_gpu.py) that shadowed the real Modal SDK and
# raised AttributeError at COLLECTION when tools/ was inserted at the front.
# CAT-134 DELETED that shadow package (regression-guarded by
# tests/test_modal_gpu_container_cap.py); the append is kept as a defensive default.
if str(_TOOLS_DIR) not in sys.path:
    sys.path.append(str(_TOOLS_DIR))

# modal is an optional dep (pyproject `.[modal]` extra) needed ONLY by this
# legacy-guard test. Skip the module cleanly when modal is absent OR when the only
# importable `modal` is the non-SDK tools/modal/ shadow (no `.Image`).
_modal = pytest.importorskip("modal")
if not hasattr(_modal, "Image"):
    pytest.skip(
        "importable 'modal' is not the Modal SDK (shadowed or stubbed); "
        "install the real SDK via `pip install -e '.[modal]'` to run this guard.",
        allow_module_level=True,
    )

import modal_gumbel_factory as factory  # type: ignore  # noqa: E402


def test_refuse_helper_raises_when_not_acknowledged():
    with pytest.raises(SystemExit, match="LEGACY CPU-only factory"):
        factory._refuse_unless_legacy_cpu_factory_acknowledged(False)


def test_refuse_helper_mentions_the_gpu_replacement_and_escape_hatch():
    with pytest.raises(SystemExit) as excinfo:
        factory._refuse_unless_legacy_cpu_factory_acknowledged(False)
    message = str(excinfo.value)
    assert "modal_gumbel_factory_gpu.py" in message
    assert "i_know_this_is_the_legacy_cpu_factory=True" in message


def test_refuse_helper_is_a_noop_when_acknowledged():
    factory._refuse_unless_legacy_cpu_factory_acknowledged(True)  # must not raise


def test_launch_gumbel_pilot_refuses_by_default():
    with pytest.raises(SystemExit, match="LEGACY CPU-only factory"):
        factory.launch_gumbel_pilot(run_name="smoke", checkpoint_rel="checkpoints/x/checkpoint.pt")


def test_launch_gumbel_pilot_acknowledged_gets_past_the_guard():
    # Not deployed to Modal in this test process, so the underlying
    # .map() call raises modal's own ExecutionError -- proof the guard
    # itself did NOT block execution once acknowledged.
    import modal.exception

    with pytest.raises(modal.exception.ExecutionError):
        factory.launch_gumbel_pilot(
            run_name="smoke",
            checkpoint_rel="checkpoints/x/checkpoint.pt",
            i_know_this_is_the_legacy_cpu_factory=True,
        )


def test_launch_gumbel_gen_refuses_by_default():
    with pytest.raises(SystemExit, match="LEGACY CPU-only factory"):
        factory.launch_gumbel_gen(run_name="wave", checkpoint_rel="checkpoints/x/checkpoint.pt")


def test_launch_gumbel_gen_acknowledged_gets_past_the_guard():
    import modal.exception

    with pytest.raises(modal.exception.ExecutionError):
        factory.launch_gumbel_gen(
            run_name="wave",
            checkpoint_rel="checkpoints/x/checkpoint.pt",
            i_know_this_is_the_legacy_cpu_factory=True,
        )
