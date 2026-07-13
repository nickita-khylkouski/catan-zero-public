from __future__ import annotations

import argparse
from unittest import mock

import pytest

from tools import train_bc


class _CapturedParser(Exception):
    """Raised by the parse_args() intercept below to hand back the fully-built parser
    without executing any training -- main() builds the whole parser (pure
    add_argument calls, no side effects) before it ever calls parse_args()."""

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        super().__init__("captured parser")
        self.parser = parser


def _intercept(self: argparse.ArgumentParser, *args, **kwargs):
    raise _CapturedParser(self)


def _capture_parser() -> argparse.ArgumentParser:
    with mock.patch.object(argparse.ArgumentParser, "parse_args", _intercept):
        try:
            train_bc.main()
        except _CapturedParser as captured:
            return captured.parser
    raise AssertionError("train_bc.main() did not reach parser.parse_args()")


def test_truncated_vp_margin_value_weight_default_feeds_the_value_loss() -> None:
    """AUDIT FIX: the CLI default used to be 0.0 (truncated rows contributed zero value
    signal by default). It must now default to a nonzero weight -- 0.25, matching the
    already-validated value-repair-v2/v3 recipe -- while 0.0 is still available to
    restore the old behavior explicitly."""
    parser = _capture_parser()
    assert parser.get_default("truncated_vp_margin_value_weight") == pytest.approx(0.25)


def test_lr_schedule_default_is_flat_no_behavior_change() -> None:
    parser = _capture_parser()
    assert parser.get_default("lr_schedule") == "flat"
    assert set(parser._option_string_actions["--lr-schedule"].choices) == {
        "flat",
        "cosine",
        "linear",
    }


def test_weight_decay_and_optimizer_defaults_unchanged() -> None:
    """FIX 1 must not change the DEFAULT config (adam, weight_decay=0.0) -- only make a
    nonzero --weight-decay with --optimizer adam fail loud instead of silently no-op."""
    parser = _capture_parser()
    assert parser.get_default("optimizer") == "adam"
    assert parser.get_default("weight_decay") == pytest.approx(0.0)
    assert parser.get_default("resume_optimizer") is True
    assert parser.get_default("action_module_lr_mult") == pytest.approx(1.0)
    assert parser.get_default("trunk_lr_mult") == pytest.approx(1.0)
    assert parser.get_default("objective_gradient_interference_every_batches") == 0


def test_policy_surprise_weight_defaults_off() -> None:
    """CAT-45: policy-surprise sampling must default to OFF (0.0), so existing runs'
    epoch order and reproducibility are unaffected unless explicitly opted in."""
    parser = _capture_parser()
    assert parser.get_default("policy_surprise_weight") == pytest.approx(0.0)
    assert parser.get_default("policy_surprise_cap") == pytest.approx(4.0)


def test_soft_target_source_default_is_policy() -> None:
    """REGRESSION (8th default-override incident class): the prefer_scores default
    silently degrades the policy target to visited-only/one-hot at low-coverage
    54-action nodes. The default MUST be "policy" so launchers that don't pin
    --soft-target-source get the safe Gumbel completed-Q visit target.
    prefer_scores remains a valid explicit choice (still in choices=)."""
    parser = _capture_parser()
    assert parser.get_default("soft_target_source") == "policy"
