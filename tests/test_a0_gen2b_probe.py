from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import a0_gen2b_probe as probe  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _manifest() -> dict:
    return {
        "schema_version": probe.SCHEMA,
        "experiment_id": "test",
        "inputs": {"expected_rows": 12},
        "historical_value_trace": [0.6, 0.8, 0.84],
        "output_dir": "runs/a0",
        "upgrade": {"bins": 33, "seed": 1, "checkpoint": "runs/a0/cat.pt"},
        "physical_gpus": {"scalar": "0", "hlgauss": "1"},
        "recipe": {
            "arch": "entity_graph",
            "epochs": 3,
            "mask_hidden_info": True,
            "symmetry_augment": False,
            "seed": 1,
            "validation_fraction": probe._REPORT_SENTINEL,
            "validation_max_samples": probe._REPORT_SENTINEL,
            "validation_seed": probe._REPORT_SENTINEL,
            "optimizer": "adamw",
            "fused_optimizer": False,
        },
        "required_historical_report_fields": [
            "arch",
            "samples",
            "epochs",
            "mask_hidden_info",
            "seed",
            "validation_fraction",
            "validation_max_samples",
            "validation_seed",
            "optimizer",
            "fused_optimizer",
        ],
    }


def _report() -> dict:
    return {
        "arch": "entity_graph",
        "samples": 12,
        "epochs": 3,
        "mask_hidden_info": True,
        "seed": 1,
        "validation_fraction": 0.25,
        "validation_max_samples": 3,
        "validation_seed": 17,
        "optimizer": "adamw",
        "fused_optimizer": False,
    }


def test_manifest_requires_distinct_single_gpu_visibility() -> None:
    manifest = _manifest()
    probe._validate_manifest(manifest)
    manifest["physical_gpus"]["hlgauss"] = "0"
    with pytest.raises(probe.ContractError, match="distinct physical GPU"):
        probe._validate_manifest(manifest)
    manifest["physical_gpus"]["hlgauss"] = "1,2"
    with pytest.raises(probe.ContractError, match="exactly one physical GPU"):
        probe._validate_manifest(manifest)


def test_recipe_is_report_locked_and_report_derived() -> None:
    manifest = _manifest()
    recipe = probe._resolve_recipe(manifest, _report())
    assert recipe["validation_fraction"] == 0.25
    assert recipe["validation_max_samples"] == 3
    assert recipe["validation_seed"] == 17

    drifted = _report()
    drifted["optimizer"] = "adam"
    with pytest.raises(
        probe.ContractError, match="historical recipe drift for optimizer"
    ):
        probe._resolve_recipe(manifest, drifted)

    missing = _report()
    del missing["validation_seed"]
    with pytest.raises(probe.ContractError, match="missing recipe fields"):
        probe._resolve_recipe(manifest, missing)


def test_checked_in_manifest_resolves_and_both_full_commands_parse() -> None:
    manifest = json.loads(
        (_REPO / "configs/experiments/a0_gen2b_hlgauss.json").read_text(
            encoding="utf-8"
        )
    )
    probe._validate_manifest(manifest)
    report = {
        "samples": manifest["inputs"]["expected_rows"],
        "data": manifest["inputs"]["corpus_dir"],
        "init_checkpoint": manifest["inputs"]["source_checkpoint"],
    }
    for key in manifest["required_historical_report_fields"]:
        value = manifest["recipe"].get(key)
        if value == probe._REPORT_SENTINEL:
            value = {
                "validation_fraction": 0.05,
                "validation_max_samples": 200000,
                "validation_seed": 17,
                "fused_optimizer": False,
                "trust_curated_data_quality": True,
                "value_target_lambda": 1.0,
            }[key]
        if value is not None:
            report[key] = value
    recipe = probe._resolve_recipe(manifest, report)
    validation = probe._validation_and_order_contract(
        np.repeat(np.arange(100, 120, dtype=np.int64), 3),
        validation_fraction=float(recipe["validation_fraction"]),
        validation_seed=int(recipe["validation_seed"]),
        validation_max_samples=int(recipe["validation_max_samples"]),
        train_seed=int(recipe["seed"]),
        epochs=int(recipe["epochs"]),
    )
    contracts = probe._build_arm_contracts(manifest, recipe, validation)

    import train_bc

    parser = train_bc.build_parser()
    scalar = parser.parse_args(contracts["scalar"]["argv"][1:])
    hl = parser.parse_args(contracts["hlgauss33"]["argv"][1:])
    assert scalar.optimizer == "adamw"
    assert scalar.value_loss_weight == 1.0
    assert scalar.final_vp_loss_weight == 0.1
    assert scalar.lr_warmup_steps == 134
    assert scalar.lr_schedule == "cosine"
    assert scalar.resume_optimizer is False
    assert scalar.acknowledge_diagnostic_outcome_conditioned_policy_distillation is True
    assert hl.value_head_type == "hlgauss"
    assert hl.value_categorical_bins == 33
    assert hl.hlgauss_scalar_aux_loss_weight == 0.0


def test_checked_in_manifest_uses_explicit_inert_legacy_lambda_default() -> None:
    manifest = json.loads(
        (_REPO / "configs/experiments/a0_gen2b_hlgauss.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["recipe"]["value_target_lambda"] == 1.0
    assert "value_target_lambda" not in manifest["required_historical_report_fields"]


def test_recipe_cli_is_mapping_order_independent() -> None:
    forward = {"epochs": 3, "lr": 1.0e-4, "batch_size": 4096}
    reverse = dict(reversed(list(forward.items())))

    assert probe._recipe_cli(forward) == probe._recipe_cli(reverse)


def test_python_environment_preserves_venv_style_symlink(
    tmp_path: Path, monkeypatch
) -> None:
    link = tmp_path / "venv-python"
    link.symlink_to(Path(sys.executable))
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "python": "3.11",
                    "numpy": "test",
                    "torch": "test",
                    "torch_cuda": "test",
                }
            )
        )

    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    environment = probe._python_environment(link, _REPO)

    assert environment["executable"] == str(link.absolute())
    assert captured["command"][0] == str(link)
    assert environment["python"]


def test_seal_records_venv_entry_point_without_resolving_symlink(
    tmp_path: Path,
) -> None:
    system_python = tmp_path / "system-python"
    system_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(system_python)

    # The lock field is intentionally tested through the same expression used
    # by seal: canonicalizing this path would silently escape the venv.
    recorded = probe._preserved_executable_path(venv_python)

    assert recorded == str(venv_python)
    assert recorded != str(venv_python.resolve())


def test_upgrade_validation_accepts_durable_name_keyed_config(tmp_path: Path) -> None:
    import torch

    from catan_zero.rl.config_serialization import config_to_dict
    from catan_zero.rl.entity_token_policy import EntityGraphConfig

    checkpoint = tmp_path / "categorical-init.pt"
    source_sha256 = "a" * 64
    torch.save(
        {
            "config": config_to_dict(
                EntityGraphConfig(
                    action_size=607,
                    static_action_feature_size=1,
                    value_categorical_bins=33,
                )
            ),
            "trained_value_readouts": [],
            "upgrade_provenance": {
                "schema_version": "entity-graph-upgrade-v1",
                "source_checkpoint_sha256": source_sha256,
                "flags": {"value_categorical_bins": 33},
                "initialization_seed": 1,
                "trained_value_readouts_added": [],
            },
        },
        checkpoint,
    )

    probe._validate_upgrade_checkpoint(
        checkpoint,
        source_sha256=source_sha256,
        bins=33,
        seed=1,
    )


def test_validation_contract_freezes_seed_set_and_exact_three_epoch_order() -> None:
    # Four games with three rows each. The validation cap deliberately samples
    # rows after selecting complete games, matching train_bc's two-stage split.
    seeds = np.repeat(np.asarray([10, 20, 30, 40], dtype=np.int64), 3)
    contract_a = probe._validation_and_order_contract(
        seeds,
        validation_fraction=0.25,
        validation_seed=17,
        validation_max_samples=2,
        train_seed=1,
        epochs=3,
    )
    contract_b = probe._validation_and_order_contract(
        seeds,
        validation_fraction=0.25,
        validation_seed=17,
        validation_max_samples=2,
        train_seed=1,
        epochs=3,
    )
    assert contract_a == contract_b
    assert contract_a["train_rows"] == 9
    assert contract_a["validation_rows"] == 2
    assert contract_a["selected_game_seed_count_before_row_cap"] == 1
    assert contract_a["selected_game_seed_ranges_cli"]
    assert contract_a["epoch_row_order_sha256"].startswith("sha256:")

    changed = probe._validation_and_order_contract(
        seeds,
        validation_fraction=0.25,
        validation_seed=17,
        validation_max_samples=2,
        train_seed=2,
        epochs=3,
    )
    assert changed["epoch_row_order_sha256"] != contract_a["epoch_row_order_sha256"]


def test_arm_contracts_pin_fresh_optimizer_and_only_add_cat_head_to_hl() -> None:
    manifest = _manifest()
    manifest["inputs"].update(
        {
            "corpus_dir": "runs/corpus",
            "source_checkpoint": "runs/gen1.pt",
        }
    )
    recipe = probe._resolve_recipe(manifest, _report())
    validation = probe._validation_and_order_contract(
        np.repeat(np.asarray([1, 2, 3, 4]), 3),
        validation_fraction=0.25,
        validation_seed=17,
        validation_max_samples=3,
        train_seed=1,
        epochs=3,
    )
    contracts = probe._build_arm_contracts(manifest, recipe, validation)
    scalar = contracts["scalar"]["argv"]
    hl = contracts["hlgauss33"]["argv"]
    for argv in (scalar, hl):
        assert "--no-resume-optimizer" in argv
        assert "--save-each-epoch" in argv
        assert "--allow-concurrent-bc" in argv
        assert argv[argv.index("--device") + 1] == "cuda:0"
        assert argv[argv.index("--validation-game-seed-ranges") + 1]
    assert scalar[scalar.index("--value-head-type") + 1] == "mse"
    assert "--value-categorical-bins" not in scalar
    assert hl[hl.index("--value-head-type") + 1] == "hlgauss"
    assert hl[hl.index("--value-categorical-bins") + 1] == "33"
    assert contracts["matched_common"]["fresh_optimizer"] is True

    import train_bc

    parser = train_bc.build_parser()
    parsed_scalar = parser.parse_args(scalar[1:])
    parsed_hl = parser.parse_args(hl[1:])
    assert parsed_scalar.resume_optimizer is False
    assert parsed_hl.value_categorical_bins == 33


def test_arm_contracts_support_separate_code_and_artifact_roots(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["inputs"].update(
        {
            "corpus_dir": "runs/corpus",
            "source_checkpoint": "runs/gen1.pt",
        }
    )
    recipe = probe._resolve_recipe(manifest, _report())
    validation = probe._validation_and_order_contract(
        np.repeat(np.asarray([1, 2, 3, 4]), 3),
        validation_fraction=0.25,
        validation_seed=17,
        validation_max_samples=3,
        train_seed=1,
        epochs=3,
    )
    contracts = probe._build_arm_contracts(
        manifest, recipe, validation, artifact_root=tmp_path
    )
    scalar = contracts["scalar"]
    assert scalar["init_checkpoint"] == str(tmp_path / "runs/gen1.pt")
    assert scalar["checkpoint"] == str(tmp_path / "runs/a0/scalar/checkpoint.pt")
    assert scalar["argv"][scalar["argv"].index("--data") + 1] == str(
        tmp_path / "runs/corpus"
    )


def test_inventory_hashes_every_regular_file_and_refuses_symlinks(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_meta.json").write_text("{}", encoding="utf-8")
    (corpus / "game_seed.dat").write_bytes(b"one")
    entries, digest = probe._corpus_inventory(corpus)
    assert [entry["path"] for entry in entries] == ["corpus_meta.json", "game_seed.dat"]
    assert digest.startswith("sha256:")
    (corpus / "game_seed.dat").write_bytes(b"two")
    _, changed = probe._corpus_inventory(corpus)
    assert changed != digest

    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    (corpus / "link").symlink_to(target)
    with pytest.raises(probe.ContractError, match="refuses symlink"):
        probe._corpus_inventory(corpus)


def test_historical_trace_is_exactly_anchored() -> None:
    probe._validate_historical_trace(
        [0.66521, 0.80899, 0.84179], [0.6652, 0.8090, 0.8418]
    )
    with pytest.raises(probe.ContractError, match="historical value trace epoch 2"):
        probe._validate_historical_trace(
            [0.6652, 0.70, 0.8418], [0.6652, 0.8090, 0.8418]
        )


def test_postflight_requires_fresh_optimizer_and_identical_holdout(
    tmp_path: Path,
) -> None:
    validation_sha = probe._int64_set_sha(np.asarray([7, 9], dtype=np.int64))
    lock = {
        "historical_value_trace": [0.6652, 0.8090, 0.8418],
        "resolved_recipe": {"epochs": 3, "optimizer": "adamw"},
        "inputs": {"source_checkpoint_sha256": "source"},
        "upgrade": {"checkpoint_sha256": "upgraded"},
        "validation": {
            "validation_game_seed_set_sha256": validation_sha,
            "validation_game_seed_count_after_row_cap": 2,
        },
        "arm_contracts": {
            "scalar": {"report": "scalar.json"},
            "hlgauss33": {"report": "hl.json"},
        },
    }
    common = {
        "resume_optimizer": False,
        "optimizer_restored": False,
        "validation_game_seed_set_sha256": validation_sha,
        "validation_game_seed_count": 2,
        "steps_completed": 10,
        "epochs": 3,
        "optimizer": "adamw",
    }
    scalar = {
        **common,
        "init_checkpoint_sha256": "sha256:source",
        "value_training": {
            "schema_version": "value-training-v1",
            "primary_readout": "scalar",
            "trained_value_readouts": ["scalar"],
        },
        "metrics": [
            {"validation": {"primary_value_loss": value}}
            for value in (0.6652, 0.8090, 0.8418)
        ],
    }
    hl = {
        **common,
        "init_checkpoint_sha256": "sha256:upgraded",
        "value_training": {
            "schema_version": "value-training-v1",
            "primary_readout": "categorical",
            "trained_value_readouts": ["categorical"],
        },
        "metrics": [
            {"validation": {"primary_value_loss": value}} for value in (1.0, 0.99, 0.98)
        ],
    }
    (tmp_path / "scalar.json").write_text(json.dumps(scalar), encoding="utf-8")
    (tmp_path / "hl.json").write_text(json.dumps(hl), encoding="utf-8")
    (tmp_path / "scalar.pt").write_bytes(b"scalar")
    (tmp_path / "hl.pt").write_bytes(b"hl")
    lock["arm_contracts"]["scalar"]["checkpoint"] = "scalar.pt"
    lock["arm_contracts"]["hlgauss33"]["checkpoint"] = "hl.pt"
    verdict = probe._postflight(lock, tmp_path)
    assert verdict["a0_training_loss_gate_pass"] is True

    hl["optimizer_restored"] = True
    (tmp_path / "hl.json").write_text(json.dumps(hl), encoding="utf-8")
    with pytest.raises(probe.ContractError, match="optimizer state was restored"):
        probe._postflight(lock, tmp_path)
