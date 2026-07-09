"""RUN-6 CAT-88 regression: the continuous flywheel must pin the GENERATION search
config EXPLICITLY, and RAISE LOUD if it is unset -- never inherit
generate_gumbel_selfplay_data.py's own CLI defaults.

POSTURE (team-lead decision): loud-fail-if-unset, NO hardcoded gen defaults. The
flywheel's generate() previously omitted these flags, so every generation subprocess
silently resolved the tool defaults -- "a whole unvalidated preset incl D1" (the tool
defaults DIFFER from canonical: c_scale 0.1 vs 0.03, temperature-decisions 45 vs 90,
lazy-interior-chance OFF vs ON). Because gen config is RUN-DEPENDENT (volume n64/p0.25
vs teacher n128/p1.0) there is no safe default: FlywheelConfig.resolve_gen_search_argv()
RAISES if any field is unset, forcing the operator to specify it.
"""
from __future__ import annotations

import pytest

from catan_zero.rl.flywheel.config import FlywheelConfig


def test_gen_search_config_unset_raises_loud() -> None:
    cfg = FlywheelConfig()  # no gen_* set
    with pytest.raises(ValueError, match="CAT-88"):
        cfg.resolve_gen_search_argv()


def test_gen_search_config_partial_raises_and_names_missing() -> None:
    cfg = FlywheelConfig(gen_n_full=64, gen_n_fast=16, gen_p_full=0.25)  # rest unset
    with pytest.raises(ValueError) as excinfo:
        cfg.resolve_gen_search_argv()
    msg = str(excinfo.value)
    # names a still-unset field, does NOT name an already-set one
    assert "gen_c_scale" in msg
    assert "gen_n_full" not in msg


def test_gen_search_config_fully_set_returns_explicit_argv() -> None:
    cfg = FlywheelConfig(
        gen_n_full=64, gen_n_fast=16, gen_p_full=0.25, gen_c_visit=50.0,
        gen_c_scale=0.03, gen_max_decisions=600, gen_max_depth=80,
        gen_temperature_decisions=90, gen_lazy_interior_chance=True,
        gen_correct_rust_chance_spectra=True,
    )
    argv = cfg.resolve_gen_search_argv()
    # every canonical value present and explicit (NOT the tool defaults 0.1/45/OFF)
    assert "--c-scale" in argv and argv[argv.index("--c-scale") + 1] == "0.03"
    assert "--temperature-decisions" in argv and argv[argv.index("--temperature-decisions") + 1] == "90"
    assert "--lazy-interior-chance" in argv and "--no-lazy-interior-chance" not in argv
    assert "--correct-rust-chance-spectra" in argv
    assert argv[argv.index("--n-full") + 1] == "64"


def test_gen_search_config_teacher_override_and_boolean_off_forms() -> None:
    cfg = FlywheelConfig(
        gen_n_full=128, gen_n_fast=32, gen_p_full=1.0, gen_c_visit=50.0,
        gen_c_scale=0.03, gen_max_decisions=600, gen_max_depth=80,
        gen_temperature_decisions=90, gen_lazy_interior_chance=False,
        gen_correct_rust_chance_spectra=False,
    ).validate()
    argv = cfg.resolve_gen_search_argv()
    assert argv[argv.index("--n-full") + 1] == "128"
    assert argv[argv.index("--p-full") + 1] == "1.0"
    # booleans set False -> explicit --no-x form (never left to a default)
    assert "--no-lazy-interior-chance" in argv and "--lazy-interior-chance" not in argv
    assert "--no-correct-rust-chance-spectra" in argv
