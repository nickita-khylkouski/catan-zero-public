"""Fail-closed contracts for value reanalysis."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import reanalyze_lite as rl  # type: ignore  # noqa: E402


@pytest.mark.parametrize("component", ["root_value", "root_prior_value"])
def test_lite_reanalysis_refuses_search_value_columns(component: str) -> None:
    with pytest.raises(SystemExit, match="single stored-feature forward"):
        rl.validate_v_component(component)


def test_lite_reanalysis_only_exposes_provenance_bound_search_q_columns() -> None:
    assert set(rl.V_COMPONENTS) == {"target_scores"}
    assert all(
        spec == {"forward_output": "q_values", "kind": "per_action"}
        for spec in rl.V_COMPONENTS.values()
    )
