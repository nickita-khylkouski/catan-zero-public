from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_mixed_value_objective_probe as probe
from tools import train_bc


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _corpus(root: Path, name: str, seeds: tuple[int, int]) -> tuple[Path, Path]:
    corpus = root / name
    corpus.mkdir()
    (corpus / "row_offsets.dat").write_bytes(np.asarray([0, 0, 0], dtype="<i8").tobytes())
    (corpus / "game_seed.dat").write_bytes(np.asarray(seeds, dtype="<i8").tobytes())
    inventory = []
    for filename in ("game_seed.dat", "row_offsets.dat"):
        path = corpus / filename
        inventory.append(
            {
                "filename": filename,
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    meta = {
        "schema": "memmap_corpus_v1",
        "row_count": 2,
        "legal_width": 1,
        "flat_count": 0,
        "columns": {"game_seed": {"kind": "fixed", "dtype": "<i8", "inner_shape": []}},
        "payload_inventory_schema": "memmap-payload-inventory-v1",
        "payload_inventory": inventory,
        "payload_inventory_sha256": _canonical(inventory),
        "selected_game_seed_manifest": {"a1_contract_sha256": "sha256:" + "1" * 64},
        "a1_post_wave_audit": {"contract_sha256": "sha256:" + "1" * 64},
    }
    (corpus / "corpus_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    validation = root / f"{name}.validation.json"
    validation.write_text("{}\n", encoding="utf-8")
    return corpus, validation


def _argv(tmp_path: Path, lr: str = "1.2e-4") -> list[str]:
    n256, n256_validation = _corpus(tmp_path, "n256", (10, 11))
    n128, n128_validation = _corpus(tmp_path, "n128", (20, 21))
    checkpoint = tmp_path / "categorical-init.pt"
    checkpoint.write_bytes(b"shared categorical-capable init")
    return [
        "--lr",
        lr,
        "--n256-corpus",
        str(n256),
        "--n256-validation",
        str(n256_validation),
        "--n128-corpus",
        str(n128),
        "--n128-validation",
        str(n128_validation),
        "--categorical-init-checkpoint",
        str(checkpoint),
        "--output-root",
        str(tmp_path / "out"),
    ]


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_prepare_builds_matched_nonpromotable_no_copy_world8_plan(tmp_path, monkeypatch):
    called = False

    def refuse_launch(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("dry preparation launched work")

    monkeypatch.setattr(probe.subprocess, "run", refuse_launch)
    probe.main(_argv(tmp_path))
    assert called is False

    manifest = json.loads((tmp_path / "out/experiment.manifest.json").read_text())
    assert manifest["diagnostic_only"] is True
    assert manifest["promotion_eligible"] is False
    assert manifest["lr_curve_verdict_selection"] == "1.2e-4"
    assert manifest["lr"] == pytest.approx(0.00012)
    assert manifest["topology"] == {
        "world_size": 8,
        "local_batch_size": 512,
        "grad_accum_steps": 1,
        "global_batch_size": 4096,
        "data_format": "memmap_composite_v1",
        "global_row_shuffle": True,
        "no_copy": True,
    }
    assert [Path(row["corpus_dir"]).name for row in manifest["components"]] == [
        "n256",
        "n128",
    ]

    mse = manifest["arms"]["mse"]
    hlgauss = manifest["arms"]["hlgauss"]
    mse_common = dict(mse["recipe"])
    hlgauss_common = dict(hlgauss["recipe"])
    assert mse_common.pop("value_head_type") == "mse"
    assert hlgauss_common.pop("value_head_type") == "hlgauss"
    assert mse_common == hlgauss_common
    assert mse_common["loser_sample_weight"] == 1.0
    assert mse_common["per_game_policy_weight"] is True
    assert mse_common["per_game_policy_weight_mode"] == "equal"
    assert mse_common["per_game_value_weight"] is True
    assert mse_common["per_game_value_weight_mode"] == "sqrt"
    assert mse_common["forced_row_value_weight"] == 0.1
    assert mse_common["value_categorical_bins"] == 33
    assert mse_common["value_hlgauss_sigma_ratio"] == 0.75

    for arm in (mse, hlgauss):
        command = arm["command"]
        assert "--nproc-per-node=8" in command
        assert _option(command, "--batch-size") == "512"
        assert _option(command, "--grad-accum-steps") == "1"
        assert _option(command, "--validation-max-samples") == "0"
        assert Path(arm["checkpoint"]).parent.name == arm["arm"]
        assert Path(arm["receipt"]).parent.name == arm["arm"]


@pytest.mark.parametrize("lr", tuple(probe.ALLOWED_LRS))
def test_supported_lr_is_resolved_and_bound(tmp_path, lr):
    args = probe.build_parser().parse_args(_argv(tmp_path, lr))
    manifest, _ = probe.prepare(args)
    assert manifest["lr"] == probe.ALLOWED_LRS[lr]
    assert all(
        arm["recipe"]["lr"] == probe.ALLOWED_LRS[lr]
        for arm in manifest["arms"].values()
    )


def test_unsupported_lr_is_rejected_before_preparation(tmp_path):
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(_argv(tmp_path, "3e-4"))
    assert not (tmp_path / "out").exists()


def test_descriptors_authenticate_full_recipe_and_refuse_objective_drift(tmp_path):
    args = probe.build_parser().parse_args(_argv(tmp_path))
    manifest, _ = probe.prepare(args)
    for arm_name in probe.ARMS:
        arm = manifest["arms"][arm_name]
        verified = train_bc._preflight_memmap_composite_descriptor(arm["descriptor"])
        matching = type("Args", (), arm["recipe"])()
        train_bc._validate_composite_learner_recipe_authorization(matching, verified)
        drifted_recipe = dict(arm["recipe"])
        drifted_recipe["value_head_type"] = (
            "hlgauss" if arm_name == "mse" else "mse"
        )
        drifted = type("Args", (), drifted_recipe)()
        with pytest.raises(SystemExit, match="authenticated diagnostic learner recipe"):
            train_bc._validate_composite_learner_recipe_authorization(drifted, verified)


def test_existing_prepared_plan_is_idempotent_but_recipe_drift_is_refused(tmp_path):
    args = probe.build_parser().parse_args(_argv(tmp_path))
    first, first_path = probe.prepare(args)
    second, second_path = probe.prepare(args)
    assert first == second
    assert first_path == second_path
    descriptor = tmp_path / "out/mse/memmap_composite.json"
    payload = json.loads(descriptor.read_text())
    payload["learner_recipe_overrides"]["loser_sample_weight"] = 0.3
    descriptor.chmod(0o644)
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="prepared artifact drift"):
        probe.prepare(args)
