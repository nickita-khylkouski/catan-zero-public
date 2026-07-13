from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from catan_zero.rl.entity_token_policy import EntityGraphConfig
from tools import a1_selected_dose_hlgauss_arm as arm
from tools import train_bc


def _source_overrides() -> dict[str, object]:
    return {
        "per_game_policy_weight": False,
        "per_game_policy_weight_mode": "equal",
        "value_head_type": "mse",
        "value_loss_weight": 0.25,
    }


def test_hlgauss_descriptor_changes_only_primary_value_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = _source_overrides()
    source = {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "components": [],
        "learner_recipe_overrides": overrides,
        "learner_recipe_overrides_sha256": arm.bridge.corrected._digest(overrides),  # noqa: SLF001
    }
    stable = {
        "component_ids": ["n128", "n256", "gen3_replay"],
        "component_game_sampling_ratios": [0.5714286, 0.2285714, 0.2],
        "policy_kl_anchor_component_ids": ["gen3_replay"],
        "policy_distillation_component_ids": ["n128", "n256", "gen3_replay"],
        "value_training_component_ids": ["n128", "n256", "gen3_replay"],
        "stored_policy_component_temperatures": {
            "n128": 1.0,
            "n256": 1.11,
            "gen3_replay": 0.52,
        },
    }

    def preflight(path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            **stable,
            "learner_recipe_overrides": payload["learner_recipe_overrides"],
        }, arm.bridge.corrected._file_ref(path)  # noqa: SLF001

    monkeypatch.setattr(arm.value_axis, "_preflight_descriptor", preflight)
    meta, ref = arm._write_hlgauss_descriptor(  # noqa: SLF001
        source, {**stable, "learner_recipe_overrides": overrides}, tmp_path / "hl.json"
    )
    treatment = json.loads(Path(ref["path"]).read_text(encoding="utf-8"))
    assert treatment["learner_recipe_overrides"] == {
        **overrides,
        "value_head_type": "hlgauss",
    }
    assert treatment["learner_recipe_overrides_sha256"] == arm.bridge.corrected._digest(  # noqa: SLF001
        treatment["learner_recipe_overrides"]
    )
    assert meta["component_ids"] == stable["component_ids"]


def _source_command(tmp_path: Path) -> list[str]:
    values = {
        "--data": str(tmp_path / "source.json"),
        "--max-steps": "128",
        "--batch-size": "512",
        "--grad-accum-steps": "1",
        "--lr": "3e-05",
        "--lr-warmup-steps": "100",
        "--policy-loss-weight": "1.0",
        "--soft-target-weight": "0.9",
        "--value-loss-weight": "0.25",
        "--value-target-lambda": "1.0",
        "--validation-game-sentinel-manifest": str(tmp_path / "source.sentinel.json"),
        "--init-checkpoint": str(tmp_path / "f7.pt"),
        "--value-head-type": "mse",
        "--checkpoint": str(tmp_path / "old.pt"),
        "--report": str(tmp_path / "old.report.json"),
    }
    command = [
        "/venv/bin/python",
        "-m",
        "torch.distributed.run",
        str(tmp_path / "source/tools/train_bc.py"),
    ]
    for flag, value in values.items():
        command.extend((flag, value))
    return command


def test_command_delta_is_exact_selected_dose_hlgauss_axis(tmp_path: Path) -> None:
    source = _source_command(tmp_path)
    treatment_trainer = tmp_path / "runtime/tools/train_bc.py"
    treatment_trainer.parent.mkdir(parents=True)
    treatment_trainer.write_text("# repaired trainer\n", encoding="utf-8")
    command, changes = arm._derive_command(  # noqa: SLF001
        source,
        source_descriptor=tmp_path / "source.json",
        treatment_descriptor=tmp_path / "hl.json",
        source_sentinel=tmp_path / "source.sentinel.json",
        treatment_sentinel=tmp_path / "hl.sentinel.json",
        source_init=tmp_path / "f7.pt",
        treatment_init=tmp_path / "f7-catbins33.pt",
        treatment_trainer=treatment_trainer,
        output_root=tmp_path / "out",
    )
    assert set(changes) == {
        "--data",
        "--validation-game-sentinel-manifest",
        "--init-checkpoint",
        "--value-head-type",
        "--checkpoint",
        "--report",
        "trainer",
    }
    option = arm.bridge.corrected._option  # noqa: SLF001
    assert option(command, "--value-head-type") == "hlgauss"
    assert option(command, "--value-loss-weight") == "0.25"
    assert option(command, "--max-steps") == "128"
    assert option(command, "--lr") == "3e-05"
    assert option(command, "--init-checkpoint").endswith("f7-catbins33.pt")


def _checkpoint_pair(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "f7.pt"
    upgraded = tmp_path / "f7-catbins33.pt"
    source_config = dataclasses.asdict(
        EntityGraphConfig(action_size=607, static_action_feature_size=1)
    )
    source_config["value_categorical_bins"] = 0
    source_raw = {
        "config": {"fields": source_config},
        "model": {"trunk.weight": torch.arange(4, dtype=torch.float32)},
        "trained_value_readouts": ["scalar"],
    }
    torch.save(source_raw, source)
    source_sha = arm.bridge.corrected._file_ref(source)["sha256"].removeprefix(  # noqa: SLF001
        "sha256:"
    )
    upgraded_config = dict(source_config)
    upgraded_config["value_categorical_bins"] = 33
    upgraded_raw = {
        **source_raw,
        "config": {"fields": upgraded_config},
        "model": {
            **source_raw["model"],
            "value_categorical_head.weight": torch.ones(34, 4),
            "value_categorical_head.bias": torch.zeros(34),
        },
        "upgrade_provenance": {
            "schema_version": "entity-graph-upgrade-v1",
            "source_checkpoint_sha256": source_sha,
            "flags": {"value_categorical_bins": 33},
            "initialization_seed": 1,
            "trained_value_readouts_added": [],
            "forward_max_diff": 0.0,
            "forward_identical_at_init": True,
        },
    }
    torch.save(upgraded_raw, upgraded)
    return source, upgraded


def test_initializer_contract_accepts_only_additive_untrained_cat_head(
    tmp_path: Path,
) -> None:
    source, upgraded = _checkpoint_pair(tmp_path)
    contract = arm._verify_categorical_initializer(source, upgraded)  # noqa: SLF001
    assert contract["config_delta"] == {
        "value_categorical_bins": {"source": 0, "treatment": 33}
    }
    assert set(contract["added_parameter_keys"]) == {
        "value_categorical_head.weight",
        "value_categorical_head.bias",
    }
    assert contract["upgrade_provenance"]["forward_identical_at_init"] is True


def test_initializer_contract_rejects_inherited_weight_drift(tmp_path: Path) -> None:
    source, upgraded = _checkpoint_pair(tmp_path)
    raw = torch.load(upgraded, map_location="cpu", weights_only=False)
    raw["model"]["trunk.weight"][0] += 1
    torch.save(raw, upgraded)
    with pytest.raises(arm.HLGaussArmError, match="non-additive weights"):
        arm._verify_categorical_initializer(source, upgraded)  # noqa: SLF001


def test_declared_contract_keeps_shared_trunk_lr_semantics_explicit() -> None:
    assert arm._matched_contract() == {  # noqa: SLF001
        "global_row_dose": 524_288,
        "optimizer_steps": 128,
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "fresh_f7_initialization": True,
        "fresh_adam": True,
        "candidate_chaining": False,
        "sample_order_unchanged": True,
        "lr_trajectory_unchanged": True,
        "policy_objective_unchanged": True,
        "value_targets_and_component_scope_unchanged": True,
        "value_lr_mult": 0.3,
        "shared_trunk_uses_base_lr": True,
    }


def test_categorical_bin_width_unwraps_ddp_model() -> None:
    policy = SimpleNamespace(
        model=SimpleNamespace(module=SimpleNamespace(value_categorical_bins=33))
    )
    assert train_bc._policy_value_categorical_bins(policy) == 33  # noqa: SLF001
    with pytest.raises(RuntimeError, match="value_categorical_bins >= 2"):
        train_bc._policy_value_categorical_bins(  # noqa: SLF001
            SimpleNamespace(
                model=SimpleNamespace(
                    module=SimpleNamespace(value_categorical_bins=0)
                )
            )
        )
