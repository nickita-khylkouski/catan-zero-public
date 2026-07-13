from __future__ import annotations

import json
from pathlib import Path

from catan_zero.search.gumbel_chance_mcts import information_set_particle_budgets


REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / "configs/guards/a1_generation_adaptive_n256_wide40.json"


def test_adaptive_n256_guard_spends_extra_budget_on_particles_not_particle_depth() -> None:
    payload = json.loads(GUARD.read_text())
    lint = payload["guards"][0]
    assert lint["name"] == "cli_flag_lint"
    expected = lint["args"]["expected_values"]

    assert expected["--n-full"] == 128
    assert expected["--n-full-wide"] == 256
    assert expected["--n-full-wide-threshold"] == 40
    assert expected["--wide-roots-always-full"] is True
    assert expected["--determinization-particles"] == 8
    assert expected["--determinization-min-simulations"] == 32

    base = information_set_particle_budgets(
        expected["--n-full"],
        expected["--determinization-particles"],
        expected["--determinization-min-simulations"],
    )
    wide = information_set_particle_budgets(
        expected["--n-full-wide"],
        expected["--determinization-particles"],
        expected["--determinization-min-simulations"],
    )
    assert base == (32,) * 4
    assert wide == (32,) * 8


def test_adaptive_n256_guard_binds_every_adaptive_flag_as_critical() -> None:
    payload = json.loads(GUARD.read_text())
    lint = payload["guards"][0]["args"]
    critical = set(lint["critical_flags"])
    assert {
        "--n-full-wide",
        "--n-full-wide-threshold",
        "--wide-roots-always-full",
        "--determinization-particles",
        "--determinization-min-simulations",
    } <= critical
