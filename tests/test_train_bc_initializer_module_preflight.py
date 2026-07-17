from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tools import train_bc  # noqa: E402


def _args(checkpoint, required: str) -> SimpleNamespace:
    return SimpleNamespace(
        init_checkpoint=str(checkpoint),
        arch="entity_graph",
        require_feature_learning_signal_modules=required,
    )


def test_initializer_preflight_rejects_missing_required_modules_before_launch(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "legacy-v6.pt"
    torch.save(
        {
            "model": {
                "event_encoder.weight": torch.zeros(2, 2),
                "value_head.weight": torch.zeros(1, 2),
            }
        },
        checkpoint,
    )

    with pytest.raises(
        SystemExit,
        match=(
            "before GPU launch: "
            "missing=action_cross_blocks,v6_exact_resource_residual"
        ),
    ):
        train_bc._preflight_init_checkpoint_architecture(
            _args(
                checkpoint,
                "action_cross_blocks,event_encoder,"
                "v6_exact_resource_residual,value_head",
            ),
            {"rank": 0},
        )


def test_initializer_preflight_accepts_required_modules_and_ddp_prefix(
    tmp_path,
    capsys,
) -> None:
    checkpoint = tmp_path / "v7.pt"
    torch.save(
        {
            "model": {
                "module.action_cross_blocks.0.q_proj.weight": torch.zeros(2, 2),
                "module.event_encoder.weight": torch.zeros(2, 2),
                "module.value_head.weight": torch.zeros(1, 2),
            }
        },
        checkpoint,
    )

    train_bc._preflight_init_checkpoint_architecture(
        _args(checkpoint, "action_cross_blocks,event_encoder,value_head"),
        {"rank": 0},
    )

    assert '"ok": true' in capsys.readouterr().out


def test_initializer_preflight_requires_model_mapping_when_modules_are_required(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "metadata-only.pt"
    torch.save({"policy_type": "entity_graph"}, checkpoint)

    with pytest.raises(SystemExit, match="missing model tensor mapping"):
        train_bc._preflight_init_checkpoint_architecture(
            _args(checkpoint, "action_cross_blocks"),
            {"rank": 0},
        )
