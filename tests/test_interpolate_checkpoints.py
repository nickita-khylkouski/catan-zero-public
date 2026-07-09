from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.interpolate_checkpoints import interpolate_checkpoints


def _write_checkpoint(path: Path, *, bias: float, hidden_size: int = 2) -> None:
    torch.save(
        {
            "observation_size": 3,
            "action_size": 4,
            "hidden_size": hidden_size,
            "architecture": "candidate",
            "use_action_id_embedding": True,
            "context_action_feature_size": 1,
            "action_features": torch.full((4, 2), bias),
            "model": {"0.weight": torch.full((2, 3), bias)},
            "actor": {"weight": torch.full((2, 2), bias)},
            "critic": {"weight": torch.full((1, 2), bias)},
            "q_head": None,
            "q_state": {"weight": torch.full((2, 2), bias)},
            "q_action_encoder": {"0.weight": torch.full((2, 3), bias)},
            "q_action_bias": {"weight": torch.full((1, 3), bias)},
            "action_encoder": {"0.weight": torch.full((2, 3), bias)},
            "action_id_embedding": {"weight": torch.full((4, 2), bias)},
            "action_bias": {"weight": torch.full((1, 3), bias)},
        },
        path,
    )


def test_interpolate_checkpoints_writes_multiple_alpha_outputs(tmp_path: Path) -> None:
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(base, bias=0.0)
    _write_checkpoint(candidate, bias=1.0)

    outputs = interpolate_checkpoints(
        base=base,
        candidate=candidate,
        alphas=(0.1, 0.25),
        output_template=str(tmp_path / "blend_a{alpha}.pt"),
    )

    assert [path.name for path in outputs] == ["blend_a0p1.pt", "blend_a0p25.pt"]
    first = torch.load(outputs[0], map_location="cpu")
    second = torch.load(outputs[1], map_location="cpu")
    assert torch.allclose(first["model"]["0.weight"], torch.full((2, 3), 0.1))
    assert torch.allclose(second["action_bias"]["weight"], torch.full((1, 3), 0.25))
    assert first["hidden_size"] == 2


def test_interpolate_checkpoints_rejects_incompatible_metadata(tmp_path: Path) -> None:
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    _write_checkpoint(base, bias=0.0, hidden_size=2)
    _write_checkpoint(candidate, bias=1.0, hidden_size=4)

    with pytest.raises(ValueError, match="hidden_size"):
        interpolate_checkpoints(
            base=base,
            candidate=candidate,
            alphas=(0.1,),
            output_template=str(tmp_path / "blend.pt"),
        )
