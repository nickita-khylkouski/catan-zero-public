from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import train_bc  # noqa: E402


def _args(**overrides):
    values = {
        "arch": "entity_graph",
        "policy_kl_anchor_direction": "forward",
        "policy_kl_anchor_weight": 0.2,
        "policy_kl_target": 0.1,
        "policy_kl_dual_lr": 0.5,
        "policy_kl_max_weight": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_projected_dual_controller_updates_from_eligible_row_mean() -> None:
    controller = train_bc.AdaptivePolicyKLController(
        target_kl=0.1,
        dual_lr=0.5,
        max_weight=1.0,
        coefficient=0.2,
    )
    first = controller.update(global_kl_sum=4.0, global_eligible_rows=20)
    assert first["observed_kl"] == pytest.approx(0.2)
    assert first["coefficient_after"] == pytest.approx(0.25)

    second = controller.update(global_kl_sum=0.0, global_eligible_rows=20)
    assert second["coefficient_after"] == pytest.approx(0.20)
    empty = controller.update(global_kl_sum=0.0, global_eligible_rows=0)
    assert empty["updated"] is False

    state = controller.state_dict()
    assert state["updates"] == 2
    assert state["eligible_rows"] == 40
    assert state["observed_kl_mean"] == pytest.approx(0.1)
    assert state["metric_scope"] == (
        "ddp_global_training_sampler_authenticated_stored_prior_multi_action_rows"
    )
    assert state["functional_parent_kl_authority"] == "none_measure_mismatch"
    assert state["promotion_claim_authority"] is False


def test_adaptive_controller_forces_diagnostic_only_promotion_contract() -> None:
    controller = train_bc.AdaptivePolicyKLController(
        target_kl=0.03,
        dual_lr=0.5,
        max_weight=1.0,
        coefficient=0.2,
    )
    report = {
        "diagnostic_only": False,
        "promotion_eligible": True,
    }

    train_bc._apply_adaptive_policy_kl_promotion_guard(  # noqa: SLF001
        report, controller
    )

    assert report["diagnostic_only"] is True
    assert report["promotion_eligible"] is False
    authority = report["adaptive_policy_kl_claim_authority"]
    assert authority["status"] == "diagnostic_only_measure_mismatch"
    assert authority["controller_metric_scope"] != authority["selection_metric_scope"]
    assert (
        authority["controller_target_does_not_certify_functional_parent_kl"] is True
    )
    assert authority["promotion_eligible"] is False
    assert len(authority["required_implementation_before_promotion"]) == 4


def test_disabled_controller_does_not_change_promotion_contract() -> None:
    report = {
        "diagnostic_only": False,
        "promotion_eligible": True,
    }

    train_bc._apply_adaptive_policy_kl_promotion_guard(  # noqa: SLF001
        report, None
    )

    assert report == {
        "diagnostic_only": False,
        "promotion_eligible": True,
    }


def test_checkpoint_surface_does_not_mislabel_training_sampler_kl_as_parent_kl() -> None:
    controller = train_bc.AdaptivePolicyKLController(
        target_kl=0.03,
        dual_lr=0.5,
        max_weight=1.0,
        coefficient=0.2,
    )

    surface = train_bc._policy_kl_controller_surface(  # noqa: SLF001
        {"base": "surface"}, controller
    )

    assert surface is not None
    assert surface["base"] == "surface"
    assert "adaptive_parent_policy_kl" not in surface
    assert surface["adaptive_stored_prior_policy_kl"]["metric_scope"] == (
        "ddp_global_training_sampler_authenticated_stored_prior_multi_action_rows"
    )


def test_controller_resume_restores_dynamic_coefficient_exactly() -> None:
    original = train_bc._adaptive_policy_kl_controller(  # noqa: SLF001
        _args(), resume_progress=None
    )
    assert original is not None
    original.update(global_kl_sum=6.0, global_eligible_rows=20)
    progress = {"policy_kl_controller_state": original.state_dict()}
    restored = train_bc._adaptive_policy_kl_controller(  # noqa: SLF001
        _args(), resume_progress=progress
    )
    assert restored is not None
    assert restored.state_dict() == original.state_dict()

    with pytest.raises(SystemExit, match="differs from command"):
        train_bc._adaptive_policy_kl_controller(  # noqa: SLF001
            _args(policy_kl_target=0.2), resume_progress=progress
        )

    malformed = original.state_dict()
    malformed["updates"] = 1.5
    with pytest.raises(SystemExit, match="malformed"):
        train_bc._adaptive_policy_kl_controller(  # noqa: SLF001
            _args(),
            resume_progress={"policy_kl_controller_state": malformed},
        )


def test_controller_uses_one_ddp_global_numerator_and_denominator(monkeypatch) -> None:
    observed = {}

    def fake_reduce(values, ddp):
        observed.update(values)
        assert ddp["enabled"] is True
        return {"kl_sum": 9.0, "eligible_rows": 30.0}

    monkeypatch.setattr(train_bc, "_reduce_named_sums", fake_reduce)
    numerator, denominator = train_bc._global_policy_kl_controller_parts(  # noqa: SLF001
        2.0,
        7.0,
        {"enabled": True, "world_size": 8, "rank": 0, "local_rank": 0},
    )
    assert observed == {"kl_sum": 2.0, "eligible_rows": 7.0}
    assert (numerator, denominator) == (9.0, 30.0)


def test_target_is_explicit_opt_in_and_forward_only() -> None:
    disabled = _args(policy_kl_target=None)
    train_bc._validate_adaptive_policy_kl_args(disabled)  # noqa: SLF001
    assert train_bc._adaptive_policy_kl_controller(  # noqa: SLF001
        disabled, resume_progress=None
    ) is None

    with pytest.raises(SystemExit, match="omits --policy-kl-target"):
        train_bc._adaptive_policy_kl_controller(  # noqa: SLF001
            disabled,
            resume_progress={"policy_kl_controller_state": {"coefficient": 0.2}},
        )
    with pytest.raises(SystemExit, match="lacks controller state"):
        train_bc._adaptive_policy_kl_controller(  # noqa: SLF001
            _args(), resume_progress={}
        )

    with pytest.raises(SystemExit, match="direction forward"):
        train_bc._validate_adaptive_policy_kl_args(  # noqa: SLF001
            _args(policy_kl_anchor_direction="reverse")
        )
    with pytest.raises(SystemExit, match="initial policy-KL anchor weight"):
        train_bc._validate_adaptive_policy_kl_args(  # noqa: SLF001
            _args(policy_kl_anchor_weight=2.0)
        )
