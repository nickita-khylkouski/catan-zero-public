from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from catan_zero.rl.pipeline_configs import TrainConfig
from tools import a1_pre_wave_contract, train_bc


def _weight_contract_args(
    *,
    loser_sample_weight: float,
    policy_loss_weight: float = 1.0,
    acknowledged: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        loser_sample_weight=loser_sample_weight,
        policy_loss_weight=policy_loss_weight,
        acknowledge_diagnostic_outcome_conditioned_policy_distillation=acknowledged,
    )


def test_default_mcts_policy_mass_is_unbiased_between_winners_and_losers() -> None:
    parser = train_bc.build_parser()
    loser_weight = parser.get_default("loser_sample_weight")
    assert loser_weight == pytest.approx(1.0)
    assert TrainConfig().loser_sample_weight == pytest.approx(1.0)

    data = {
        "action_taken": np.asarray([1, 2, 3, 4], dtype=np.int16),
        "legal_action_ids": np.asarray(
            [[1, 5], [2, 6], [3, 7], [4, 8]], dtype=np.int16
        ),
        "player": np.asarray(["RED", "BLUE", "RED", "BLUE"]),
        "winner": np.asarray(["RED", "RED", "BLUE", "BLUE"]),
        "truncated": np.zeros(4, dtype=np.bool_),
        "policy_weight_multiplier": np.ones(4, dtype=np.float32),
    }
    weights = train_bc.build_sample_weights(
        data,
        teacher_weights={},
        phase_weights={},
        forced_action_weight=0.0,
        winner_sample_weight=1.0,
        loser_sample_weight=loser_weight,
        vp_margin_weight=0.0,
        vps_to_win=10,
    )
    won = data["player"] == data["winner"]
    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert float(weights[won].sum()) == pytest.approx(float(weights[~won].sum()))


def test_production_refuses_silent_outcome_conditioned_policy_distillation() -> None:
    with pytest.raises(SystemExit, match="outcome-conditions MCTS policy"):
        train_bc._validate_outcome_conditioned_policy_distillation(
            _weight_contract_args(loser_sample_weight=0.3),
            a1_preflight_meta=None,
        )


def test_explicit_diagnostic_authorizations_preserve_historical_replay() -> None:
    acknowledged = train_bc._validate_outcome_conditioned_policy_distillation(
        _weight_contract_args(loser_sample_weight=0.3, acknowledged=True),
        a1_preflight_meta=None,
    )
    assert acknowledged["diagnostic_only"] is True
    assert acknowledged["promotion_eligible"] is False
    assert acknowledged["authorization"] == "explicit_cli_acknowledgement"

    historical = train_bc._validate_outcome_conditioned_policy_distillation(
        _weight_contract_args(loser_sample_weight=0.3),
        a1_preflight_meta={"diagnostic_only": True, "promotion_eligible": False},
    )
    assert historical["authorization"] == "authenticated_diagnostic_descriptor"


def test_zero_policy_objective_does_not_require_irrelevant_acknowledgement() -> None:
    contract = train_bc._validate_outcome_conditioned_policy_distillation(
        _weight_contract_args(loser_sample_weight=0.3, policy_loss_weight=0.0),
        a1_preflight_meta=None,
    )
    assert contract["outcome_conditioned"] is False
    assert contract["authorization"] == "not_required"


def test_pre_wave_authoring_defaults_to_unbiased_policy_mass() -> None:
    assert (
        a1_pre_wave_contract.EXPECTED_LEARNER_TRAINING_RECIPE[
            "loser_sample_weight"
        ]
        == 1.0
    )


def test_factory_defaults_safe_and_requires_diagnostic_ack_for_old_weight(tmp_path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in ("src", env.get("PYTHONPATH", "")) if value
    )
    safe_dir = tmp_path / "safe"
    safe = subprocess.run(
        [
            sys.executable,
            "tools/start_training_factory.py",
            "--run-dir",
            str(safe_dir),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert safe.returncode == 0, safe.stderr
    manifest = json.loads((safe_dir / "pipeline_manifest.json").read_text())
    train_command = next(
        command for command in manifest["commands"] if "tools/train_bc.py" in command
    )
    index = train_command.index("--loser-sample-weight")
    assert train_command[index + 1] == "1.0"
    assert (
        "--acknowledge-diagnostic-outcome-conditioned-policy-distillation"
        not in train_command
    )

    refused = subprocess.run(
        [
            sys.executable,
            "tools/start_training_factory.py",
            "--run-dir",
            str(tmp_path / "refused"),
            "--dry-run",
            "--loser-sample-weight",
            "0.3",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert refused.returncode != 0
    assert "diagnostic-only" in refused.stderr
