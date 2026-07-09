from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.ema_average_checkpoints import (
    compute_ema_weights,
    ema_average_checkpoints,
)


def _checkpoint(*, bias: float, step: int, mask_hidden_info: bool = True, hidden_size: int = 8):
    return {
        "policy_type": "entity_graph",
        "config": {"hidden_size": hidden_size, "graph_layers": 2},
        "action_mask_version": "v1",
        "mask_hidden_info": mask_hidden_info,
        "static_action_features_sha256": "deadbeef",
        "static_action_features": torch.zeros(3, 3),
        "model": {
            "trunk.weight": torch.full((4, 4), bias),
            "num_batches_tracked": torch.tensor(step, dtype=torch.int64),
        },
    }


def _write(path: Path, **kwargs) -> Path:
    torch.save(_checkpoint(**kwargs), path)
    return path


# --------------------------------------------------------------------------- weight math


def test_compute_ema_weights_sums_to_one_and_favors_newest() -> None:
    weights = compute_ema_weights(4, 0.75)
    assert sum(weights) == pytest.approx(1.0)
    assert weights == sorted(weights)  # monotonically increasing -> newest heaviest


def test_compute_ema_weights_zero_decay_uses_only_the_newest() -> None:
    weights = compute_ema_weights(3, 0.0)
    assert weights == pytest.approx([0.0, 0.0, 1.0])


def test_compute_ema_weights_unit_decay_is_uniform_swa_average() -> None:
    weights = compute_ema_weights(4, 1.0)
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_compute_ema_weights_rejects_decay_out_of_range() -> None:
    with pytest.raises(ValueError):
        compute_ema_weights(3, 1.1)
    with pytest.raises(ValueError):
        compute_ema_weights(3, -0.1)


def test_compute_ema_weights_rejects_empty() -> None:
    with pytest.raises(ValueError):
        compute_ema_weights(0, 0.75)


# --------------------------------------------------------------------------- averaging


def test_ema_average_checkpoints_weights_float_tensors_by_ema_decay(tmp_path: Path) -> None:
    paths = [
        _write(tmp_path / "a.pt", bias=0.0, step=0),
        _write(tmp_path / "b.pt", bias=1.0, step=1),
        _write(tmp_path / "c.pt", bias=2.0, step=2),
    ]

    result = ema_average_checkpoints(checkpoints=paths, decay=0.5)

    weights = compute_ema_weights(3, 0.5)
    expected = sum(w * b for w, b in zip(weights, (0.0, 1.0, 2.0)))
    assert float(result["model"]["trunk.weight"][0, 0]) == pytest.approx(expected)


def test_ema_average_checkpoints_decay_one_is_plain_swa_uniform_mean(tmp_path: Path) -> None:
    """decay=1.0 is the plain-SWA special case: a uniform mean across all checkpoints,
    not just newest-heaviest."""
    paths = [
        _write(tmp_path / "a.pt", bias=0.0, step=0),
        _write(tmp_path / "b.pt", bias=3.0, step=1),
        _write(tmp_path / "c.pt", bias=6.0, step=2),
    ]

    result = ema_average_checkpoints(checkpoints=paths, decay=1.0)

    assert result["ema_weights"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert float(result["model"]["trunk.weight"][0, 0]) == pytest.approx(3.0)  # plain mean of 0,3,6


def test_ema_average_checkpoints_carries_over_integer_buffers_from_newest(tmp_path: Path) -> None:
    """Non-floating-point buffers (e.g. BatchNorm num_batches_tracked) must not be
    fractionally averaged -- they're carried through from the newest checkpoint."""
    paths = [
        _write(tmp_path / "a.pt", bias=0.0, step=10),
        _write(tmp_path / "b.pt", bias=1.0, step=20),
        _write(tmp_path / "c.pt", bias=2.0, step=30),
    ]

    result = ema_average_checkpoints(checkpoints=paths, decay=0.75)

    assert int(result["model"]["num_batches_tracked"]) == 30


def test_ema_average_checkpoints_carries_over_newest_metadata(tmp_path: Path) -> None:
    paths = [
        _write(tmp_path / "a.pt", bias=0.0, step=0, hidden_size=8),
        _write(tmp_path / "b.pt", bias=1.0, step=1, hidden_size=8),
    ]

    result = ema_average_checkpoints(checkpoints=paths, decay=0.75)

    assert result["config"] == {"hidden_size": 8, "graph_layers": 2}
    assert result["mask_hidden_info"] is True
    assert result["policy_type"] == "entity_graph"
    assert result["ema_decay"] == pytest.approx(0.75)
    assert result["ema_weights"] == pytest.approx(compute_ema_weights(2, 0.75))
    assert [Path(p) for p in result["ema_source_checkpoints"]] == paths


def test_ema_average_checkpoints_writes_output_file_that_round_trips(tmp_path: Path) -> None:
    paths = [
        _write(tmp_path / "a.pt", bias=0.0, step=0),
        _write(tmp_path / "b.pt", bias=1.0, step=1),
    ]
    output = tmp_path / "ema.pt"

    result = ema_average_checkpoints(checkpoints=paths, decay=0.5, output=output)

    assert output.exists()
    reloaded = torch.load(output, map_location="cpu", weights_only=False)
    assert torch.allclose(reloaded["model"]["trunk.weight"], result["model"]["trunk.weight"])


def test_ema_average_checkpoints_single_checkpoint_is_identity(tmp_path: Path) -> None:
    path = _write(tmp_path / "only.pt", bias=3.0, step=0)

    result = ema_average_checkpoints(checkpoints=[path], decay=0.75)

    assert float(result["model"]["trunk.weight"][0, 0]) == pytest.approx(3.0)
    assert result["ema_weights"] == pytest.approx([1.0])


# --------------------------------------------------------------------------- refusal paths


def test_ema_average_checkpoints_refuses_mask_hidden_info_mismatch(tmp_path: Path) -> None:
    paths = [
        _write(tmp_path / "a.pt", bias=0.0, step=0, mask_hidden_info=True),
        _write(tmp_path / "b.pt", bias=1.0, step=1, mask_hidden_info=False),
    ]

    with pytest.raises(ValueError, match="mask_hidden_info"):
        ema_average_checkpoints(checkpoints=paths, decay=0.5)


def test_ema_average_checkpoints_refuses_arch_config_mismatch(tmp_path: Path) -> None:
    paths = [
        _write(tmp_path / "a.pt", bias=0.0, step=0, hidden_size=8),
        _write(tmp_path / "b.pt", bias=1.0, step=1, hidden_size=16),
    ]

    with pytest.raises(ValueError, match="config"):
        ema_average_checkpoints(checkpoints=paths, decay=0.5)


def test_ema_average_checkpoints_refuses_state_dict_key_mismatch(tmp_path: Path) -> None:
    a = _checkpoint(bias=0.0, step=0)
    b = _checkpoint(bias=1.0, step=1)
    b["model"]["extra_layer.weight"] = torch.zeros(2, 2)
    torch.save(a, tmp_path / "a.pt")
    torch.save(b, tmp_path / "b.pt")

    with pytest.raises(ValueError, match="state_dict keys"):
        ema_average_checkpoints(checkpoints=[tmp_path / "a.pt", tmp_path / "b.pt"], decay=0.5)


def test_ema_average_checkpoints_refuses_tensor_shape_mismatch(tmp_path: Path) -> None:
    a = _checkpoint(bias=0.0, step=0)
    b = _checkpoint(bias=1.0, step=1)
    b["model"]["trunk.weight"] = torch.zeros(8, 8)
    torch.save(a, tmp_path / "a.pt")
    torch.save(b, tmp_path / "b.pt")

    with pytest.raises(ValueError, match="shape mismatch"):
        ema_average_checkpoints(checkpoints=[tmp_path / "a.pt", tmp_path / "b.pt"], decay=0.5)
