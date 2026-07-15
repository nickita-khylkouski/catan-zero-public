from __future__ import annotations

from catan_zero.rl.gumbel_self_play import (
    GumbelSelfPlayConfig,
    _temperature_for_decision,
)


def _config(**overrides) -> GumbelSelfPlayConfig:
    values = {
        "max_decisions": 600,
        "temperature_move_fraction": 0.075,  # cutoff = round(600 * 0.075) = 45
        "temperature_high": 1.0,
        "temperature_low": 0.0,
    }
    values.update(overrides)
    return GumbelSelfPlayConfig(**values)


# --------------------------------------------------------------------------- default (no-op)


def test_late_temperature_disabled_by_default_matches_two_stage_schedule() -> None:
    config = _config()
    assert config.late_temperature_move_fraction is None

    for decision_index in (0, 44):
        assert (
            _temperature_for_decision(
                decision_index, config=config, eval_override=False
            )
            == config.temperature_high
        )
    for decision_index in (45, 90, 599):
        assert (
            _temperature_for_decision(
                decision_index, config=config, eval_override=False
            )
            == config.temperature_low
        )


def test_eval_override_ignores_late_temperature_even_when_configured() -> None:
    config = _config(late_temperature_move_fraction=0.25, late_temperature=0.3)
    assert (
        _temperature_for_decision(50, config=config, eval_override=True)
        == config.temperature_low
    )


# --------------------------------------------------------------------------- late window enabled


def test_late_temperature_window_extends_past_the_opening_cutoff() -> None:
    # late cutoff = round(600 * 0.25) = 150
    config = _config(late_temperature_move_fraction=0.25, late_temperature=0.3)

    assert (
        _temperature_for_decision(44, config=config, eval_override=False)
        == config.temperature_high
    )
    assert _temperature_for_decision(45, config=config, eval_override=False) == 0.3
    assert _temperature_for_decision(149, config=config, eval_override=False) == 0.3
    assert (
        _temperature_for_decision(150, config=config, eval_override=False)
        == config.temperature_low
    )
    assert (
        _temperature_for_decision(599, config=config, eval_override=False)
        == config.temperature_low
    )


def test_late_temperature_cutoff_never_precedes_the_opening_cutoff() -> None:
    """A late fraction smaller than (or equal to) the opening fraction must not shrink
    the window to zero or negative width -- it degenerates to the plain two-stage
    schedule (late cutoff clamped up to the opening cutoff)."""
    config = _config(
        late_temperature_move_fraction=0.01, late_temperature=0.3
    )  # would be < cutoff

    for decision_index in (45, 90, 599):
        assert (
            _temperature_for_decision(
                decision_index, config=config, eval_override=False
            )
            == config.temperature_low
        )


def test_nonforced_choice_clock_ignores_prompt_index() -> None:
    config = _config(temperature_clock="nonforced_choice")

    assert (
        _temperature_for_decision(
            120,
            config=config,
            eval_override=False,
            nonforced_choice_index=44,
        )
        == config.temperature_high
    )
    assert (
        _temperature_for_decision(
            120,
            config=config,
            eval_override=False,
            nonforced_choice_index=45,
        )
        == config.temperature_low
    )


def test_nonforced_choice_clock_requires_explicit_choice_index() -> None:
    config = _config(temperature_clock="nonforced_choice")

    try:
        _temperature_for_decision(0, config=config, eval_override=False)
    except ValueError as error:
        assert "requires a choice index" in str(error)
    else:  # pragma: no cover - defensive assertion without pytest dependency.
        raise AssertionError("missing nonforced choice index was accepted")
