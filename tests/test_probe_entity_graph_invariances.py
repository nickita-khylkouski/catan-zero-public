from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _probe_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "probe_entity_graph_invariances.py"
    )
    spec = importlib.util.spec_from_file_location(
        "probe_entity_graph_invariances", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def result():
    return _probe_module().run_probe()


def test_incumbent_ignores_legal_action_target_ids(result):
    assert result["incumbent_target_id_diff"] == {
        "logits": 0.0,
        "value": 0.0,
        "final_vp": 0.0,
        "q_values": 0.0,
    }


def test_enabled_target_gather_makes_target_ids_observable(result):
    diff = result["enabled_gather_target_id_diff"]
    assert diff["logits"] > 0.0
    assert diff["q_values"] > 0.0
    assert diff["value"] == 0.0
    assert diff["final_vp"] == 0.0


def test_incumbent_is_blind_to_within_type_board_token_permutation(result):
    # Dense attention has no vertex/edge position or incidence input. Numerical
    # reduction order may introduce tiny roundoff even though the function is
    # mathematically permutation invariant.
    assert max(result["incumbent_token_permutation_diff"].values()) < 1.0e-5
