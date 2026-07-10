from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import a0_policy_drift as probe  # noqa: E402


def _metrics(*, policy_loss: float = 1.0, prior_kl: float = 0.5) -> dict:
    return {
        "samples": 100,
        "accuracy_active_count": 80,
        "prior_kl_rows": 60,
        "policy_loss": policy_loss,
        "prior_kl_model_prior_mean": prior_kl,
    }


def test_stage_comparison_enforces_two_percent_absolute_drift() -> None:
    passed = probe.compare_stage_metrics(
        _metrics(), _metrics(policy_loss=1.02, prior_kl=0.51)
    )
    assert passed["pass"] is True

    failed = probe.compare_stage_metrics(
        _metrics(), _metrics(policy_loss=0.97, prior_kl=0.5)
    )
    assert failed["pass"] is False
    assert failed["metrics"]["unforced_policy_loss"]["pass"] is False


def test_stage_comparison_refuses_different_row_populations() -> None:
    hl = _metrics()
    hl["prior_kl_rows"] = 59
    with pytest.raises(probe.a0.ContractError, match="row counts differ"):
        probe.compare_stage_metrics(_metrics(), hl)


def test_checkpoint_stages_bind_all_saved_epochs_and_final(tmp_path: Path) -> None:
    paths = probe._checkpoint_stages(tmp_path / "checkpoint.pt")
    assert list(paths) == ["epoch1", "epoch2", "epoch3", "final"]
    assert paths["epoch1"].name == "checkpoint_epoch0001.pt"
    assert paths["epoch3"].name == "checkpoint_epoch0003.pt"
    assert paths["final"].name == "checkpoint.pt"


def test_exact_trainer_seed_manifests_must_match_lock_and_each_other(
    tmp_path: Path,
) -> None:
    seeds = np.asarray([11, 7, 9], dtype=np.int64)
    canonical = np.sort(seeds).astype("<i8", copy=False)
    seed_sha = "sha256:" + hashlib.sha256(canonical.tobytes()).hexdigest()
    arm_contracts = {"matched_common_sha256": "matched"}
    for arm in ("scalar", "hlgauss33"):
        report = tmp_path / arm / "report.json"
        report.parent.mkdir()
        report.write_text("{}", encoding="utf-8")
        manifest = report.with_suffix(".validation_seeds.json")
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "train-validation-game-seeds-v1",
                    "validation_game_seed_count": 3,
                    "validation_game_seed_set_sha256": seed_sha,
                    "game_seeds": seeds.tolist(),
                }
            ),
            encoding="utf-8",
        )
        arm_contracts[arm] = {"report": str(report)}
    lock = {
        "validation": {
            "validation_game_seed_set_sha256": seed_sha,
            "validation_game_seed_count_after_row_cap": 3,
        },
        "arm_contracts": arm_contracts,
    }

    selected, evidence = probe._load_exact_validation_seeds(lock, tmp_path)
    assert selected.tolist() == [7, 9, 11]
    assert evidence["validation_game_seed_set_sha256"] == seed_sha

    hl_manifest = Path(arm_contracts["hlgauss33"]["report"]).with_suffix(
        ".validation_seeds.json"
    )
    payload = json.loads(hl_manifest.read_text(encoding="utf-8"))
    payload["game_seeds"] = [7, 9, 13]
    changed = np.asarray(payload["game_seeds"], dtype="<i8")
    payload["validation_game_seed_set_sha256"] = (
        "sha256:" + hashlib.sha256(np.sort(changed).tobytes()).hexdigest()
    )
    hl_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(probe.a0.ContractError, match="seed set/count drift"):
        probe._load_exact_validation_seeds(lock, tmp_path)
