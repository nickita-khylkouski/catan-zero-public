from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tools import a1_b200_active_policy_campaign as campaign
from tools.fleet import a1_active_policy_eval_matrix as matrix


def test_eval_authority_uses_selected_dose_not_terminal_checkpoint(
    tmp_path: Path, monkeypatch,
) -> None:
    campaign_path = tmp_path / "campaign.json"
    selection_path = tmp_path / "selection.json"
    fingerprint_path = tmp_path / "p100.fingerprint.json"
    upgrade_path = tmp_path / "upgrade.json"
    registry_path = tmp_path / "registry.json"
    f7 = tmp_path / "f7.pt"
    initializer = tmp_path / "f7-upgraded.pt"
    v5 = tmp_path / "v5.pt"
    selected_checkpoint = tmp_path / "p100-step0032.pt"
    terminal_checkpoint = tmp_path / "p100-terminal.pt"
    for path in (
        campaign_path,
        selection_path,
        fingerprint_path,
        upgrade_path,
        registry_path,
        f7,
        initializer,
        v5,
        selected_checkpoint,
        terminal_checkpoint,
    ):
        path.write_bytes(path.name.encode())

    initializer_sha = "sha256:" + "1" * 64
    selected_sha = "sha256:" + "2" * 64
    terminal_sha = "sha256:" + "3" * 64
    fingerprint_file_sha = "sha256:" + "4" * 64
    campaign_file_sha = "sha256:" + "5" * 64
    campaign_payload = {
        "campaign_sha256": "sha256:" + "6" * 64,
        "inputs": {"architecture_upgrade_receipt": str(upgrade_path)},
        "lineage_contract": {"upgraded_initializer_sha256": initializer_sha},
    }
    selected = {
        "arm": "P100",
        "step": 32,
        "checkpoint": str(selected_checkpoint),
        "checkpoint_sha256": selected_sha,
        "parent_kl": 0.02,
        "trunk_relative_l2": 0.01,
        "teacher_gap_closure": 0.2,
        "within_drift_budgets": True,
        "positive_teacher_gap_closure": True,
        "eligible": True,
    }
    selection_payload = {
        "eligible_arms": ["P100"],
        "winner": "P100",
        "arm_fingerprints": {
            "P100": {
                "path": str(fingerprint_path),
                "file_sha256": fingerprint_file_sha,
                "fingerprint_sha256": "sha256:" + "7" * 64,
                "has_eligible_checkpoint": True,
                "selected_checkpoint": selected,
            }
        },
    }
    fingerprint_payload = {
        "arm": "P100",
        "fingerprint_sha256": "sha256:" + "7" * 64,
        "checkpoints": [
            {
                "step": 32,
                "checkpoint": str(selected_checkpoint),
                "checkpoint_sha256": selected_sha,
                "functional": {
                    "parent_kl": 0.02,
                    "teacher_gap_closure": 0.2,
                },
                "layer_drift": {"trunk_relative_l2": 0.01},
            },
            {
                "step": 128,
                "checkpoint": str(terminal_checkpoint),
                "checkpoint_sha256": terminal_sha,
                "functional": {
                    "parent_kl": 0.08,
                    "teacher_gap_closure": 0.25,
                },
                "layer_drift": {"trunk_relative_l2": 0.07},
            },
        ],
    }
    upgrade = {
        "source": {
            "path": str(f7),
            "sha256": campaign.EXPECTED_F7_PARENT_SHA256,
        },
        "upgraded_initializer": {
            "path": str(initializer),
            "sha256": initializer_sha,
        },
    }

    digests = {
        campaign_path: campaign_file_sha,
        fingerprint_path: fingerprint_file_sha,
        f7: campaign.EXPECTED_F7_PARENT_SHA256,
        initializer: initializer_sha,
        selected_checkpoint: selected_sha,
        terminal_checkpoint: terminal_sha,
        v5: campaign.EXPECTED_CORPUS_PRODUCER_SHA256,
    }
    monkeypatch.setattr(
        matrix.active_campaign,
        "_load_campaign",
        lambda _path: (campaign_path, campaign_payload),
    )
    monkeypatch.setattr(
        matrix,
        "_load_selection",
        lambda _path, **_kwargs: (selection_path, selection_payload),
    )
    monkeypatch.setattr(
        matrix.architecture_upgrade, "verify_receipt", lambda _path: upgrade
    )
    monkeypatch.setattr(
        matrix.ChampionRegistry,
        "load",
        lambda _path: SimpleNamespace(
            get_role=lambda _role: SimpleNamespace(checkpoint_path=str(v5))
        ),
    )
    monkeypatch.setattr(
        matrix.active_campaign,
        "_load_signed",
        lambda _path, **_kwargs: (fingerprint_path, fingerprint_payload),
    )
    monkeypatch.setattr(
        matrix.active_campaign,
        "_verify_completed_arm",
        lambda _campaign, _arm: {
            "checkpoint": str(terminal_checkpoint),
            "checkpoint_sha256": terminal_sha,
        },
    )
    monkeypatch.setattr(matrix, "_file_sha256", lambda path: digests[Path(path)])
    monkeypatch.setattr(matrix.fleet, "_sha256", lambda path: digests[Path(path)])

    authority = matrix._load_authority(
        campaign_path=campaign_path,
        selection_path=selection_path,
        registry_path=registry_path,
    )

    assert authority["completed"]["P100"]["checkpoint"] == str(terminal_checkpoint)
    assert authority["candidates"]["P100"] == {
        "arm": "P100",
        "step": 32,
        "checkpoint": str(selected_checkpoint),
        "checkpoint_sha256": selected_sha,
        "parent_kl": 0.02,
        "trunk_relative_l2": 0.01,
        "teacher_gap_closure": 0.2,
    }
    assert authority["fingerprints"]["P100"]["selected_checkpoint"]["step"] == 32
