from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/a1_n256_lr_response_b200.sh"


def test_n256_lr_response_launcher_is_syntax_clean_and_fail_closed() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "6e-5|0.00006" in text
    assert "2.4e-4|0.00024" in text
    assert "--lr must be exactly 6e-5 or 2.4e-4" in text
    assert "all-196k-corrective-lr120u-loser1" in text
    assert 'effective.get("lr") != 0.00012' in text
    assert 'effective.get("loser_sample_weight") != 1.0' in text
    assert "midpoint learner lock binds another spec" in text
    assert "midpoint contract is writable" in text
    assert "inspect-spec" not in text
    assert text.count('a1_dual_learner_contract.py" seal') == 1
    assert "midpoint_spec=$midpoint_dir/learner.spec.json" in text
    assert "midpoint_lock=$midpoint_dir/learner.lock.json" in text
    assert "shared_contract_dir=$root/training/n256-lr-response-contracts/$repo_commit" in text
    assert 'flock "$contract_fd"' in text
    assert 'cp "$midpoint_spec" "$spec_tmp"' in text
    assert "spec=$dose/learner.spec.json" not in text
    assert "lock=$dose/learner.lock.json" not in text
    midpoint_verifier = text.split("verify_midpoint() {", 1)[1].split(
        "prepare_shared_contract() {", 1
    )[0]
    assert "a1-dual-arm-learner-lock-v1" in midpoint_verifier
    assert "midpoint historical runtime binding drift" in midpoint_verifier
    assert "contract.verify_lock" not in midpoint_verifier
    assert 'set(a["recipe_drift"]) == {"lr", "loser_sample_weight"}' in text
    assert 'a["effective_recipe"]["epochs"] == 1' in text
    assert 'p["world_size"] == 8 and p["global_batch_size"] == 4096' in text
    assert 'a["diagnostic_only"], a["promotion_eligible"]' in text
    assert 'systemctl is-active nvidia-mps.service' in text
    assert 'trap restore_mps EXIT' in text
    assert "partial LR-response outputs exist without a completed receipt" in text
    assert "authenticated completed diagnostic LR-response dose; no retraining" in text
    assert "$root/n128" not in text


@pytest.mark.parametrize("bad_lr", ["1.2e-4", "0.00012", "3e-5", "0", "banana"])
def test_n256_lr_response_launcher_rejects_undeclared_lr_before_io(
    bad_lr: str,
) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--lr", bad_lr],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert "--lr must be exactly 6e-5 or 2.4e-4" in result.stderr


def test_n256_lr_response_launcher_requires_explicit_lr() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr
