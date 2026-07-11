from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from tools.generate_gumbel_selfplay_data import _validate_science_args


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "p_full": 0.25,
        "c_visit": 50.0,
        "c_scale": 0.03,
        "rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.98,
        "temperature_high": 1.0,
        "temperature_low": 0.0,
        "late_temperature": 0.0,
        "prior_temperature": 1.0,
        "value_scale": 1.0,
        "temperature_move_fraction": None,
        "public_observation": True,
        "information_set_search": True,
        "belief_chance_spectra": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"information_set_search": False},
            "--public-observation requires --information-set-search",
        ),
        (
            {"public_observation": False},
            "--information-set-search requires --public-observation",
        ),
        (
            {"belief_chance_spectra": True},
            "--information-set-search cannot be combined with --belief-chance-spectra",
        ),
    ],
)
def test_generation_rejects_hidden_information_mismatches(
    overrides: dict[str, object], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = argparse.ArgumentParser(prog="generate")
    with pytest.raises(SystemExit, match="2"):
        _validate_science_args(_args(**overrides), parser)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "public_observation": True,
            "information_set_search": True,
            "belief_chance_spectra": False,
        },
        {
            "public_observation": False,
            "information_set_search": False,
            "belief_chance_spectra": True,
        },
    ],
)
def test_generation_accepts_coherent_search_regimes(
    overrides: dict[str, object],
) -> None:
    _validate_science_args(_args(**overrides), argparse.ArgumentParser())
