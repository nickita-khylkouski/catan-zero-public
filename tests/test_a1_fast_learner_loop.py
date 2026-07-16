from __future__ import annotations

import json
from pathlib import Path

from tools import a1_fast_learner_loop as runner
from tools import train_bc


def _source_manifest(tmp_path: Path, *, include_readout: bool) -> Path:
    init = tmp_path / "init.pt"
    data = tmp_path / "memmap-composite.json"
    sentinel = tmp_path / "validation.json"
    init.write_bytes(b"initializer")
    data.write_text("{}\n", encoding="utf-8")
    sentinel.write_text("{}\n", encoding="utf-8")
    command = [
        "torchrun",
        "--nproc-per-node",
        "8",
        str(Path(train_bc.__file__).resolve()),
        "--data",
        str(data),
        "--checkpoint",
        str(tmp_path / "old-candidate.pt"),
        "--report",
        str(tmp_path / "old-report.json"),
        "--init-checkpoint",
        str(init),
        "--validation-game-sentinel-manifest",
        str(sentinel),
        "--max-steps",
        "128",
        "--batch-size",
        "512",
        "--lr",
        "3e-05",
        "--lr-schedule",
        "flat",
        "--no-resume-optimizer",
        "--soft-target-weight",
        "0.9",
    ]
    if include_readout:
        command.extend(
            (
                "--scalar-value-loss-readout=raw",
                "--scalar-value-loss-scale",
                "2.0",
            )
        )
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"command": command}), encoding="utf-8")
    return path


def test_deployed_tanh_arm_emits_the_real_train_bc_flags(tmp_path: Path) -> None:
    source = _source_manifest(tmp_path, include_readout=False)
    command, _manifest = runner.derive_run(
        source_manifest_path=source,
        output_root=tmp_path / "out",
        arm="deployed_tanh_value",
    )

    assert "--scalar-value-loss-transform" not in command
    assert runner._option(command, "--scalar-value-loss-readout") == "deployed_tanh"
    assert runner._option(command, "--scalar-value-loss-scale") == "1.0"
    trainer_index = next(
        index for index, item in enumerate(command) if item.endswith("/tools/train_bc.py")
    )
    parsed = train_bc.build_parser().parse_args(command[trainer_index + 1 :])
    assert parsed.scalar_value_loss_readout == "deployed_tanh"
    assert parsed.scalar_value_loss_scale == 1.0


def test_deployed_tanh_arm_replaces_explicit_source_readout(tmp_path: Path) -> None:
    source = _source_manifest(tmp_path, include_readout=True)
    command, _manifest = runner.derive_run(
        source_manifest_path=source,
        output_root=tmp_path / "out",
        arm="deployed_tanh_value",
    )

    assert runner._option(command, "--scalar-value-loss-readout") == "deployed_tanh"
    assert runner._option(command, "--scalar-value-loss-scale") == "2.0"
