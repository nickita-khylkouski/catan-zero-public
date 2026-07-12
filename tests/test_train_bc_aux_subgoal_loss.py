"""Unit tests for tools/train_bc.py::_aux_subgoal_loss (CAT-100 loss wiring).

The helper must combine only the aux heads that have BOTH a model output and a
corpus target field, mask -1 rows for categorical heads, be a pure no-op (0.0, 0)
when NO aux head is built, and (CAT-105) RAISE loud when heads ARE built but the
corpus carries none of the aux target columns (requested-but-inert footgun).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import train_bc  # noqa: E402


_N = 4
_DEVICE = torch.device("cpu")


def _outputs(all_heads=True):
    out = {
        "logits": torch.randn(_N, 5),
        "value": torch.randn(_N),
    }
    out["aux_longest_road"] = torch.randn(_N, requires_grad=True)
    out["aux_largest_army"] = torch.randn(_N, requires_grad=True)
    out["aux_vp_in_n"] = torch.randn(_N, requires_grad=True)
    out["aux_next_settlement"] = torch.randn(_N, 54, requires_grad=True)
    out["aux_robber_target"] = torch.randn(_N, 19, requires_grad=True)
    return out


def test_heads_built_but_corpus_lacks_aux_fields_raises_loud():
    """CAT-105: heads built (aux outputs present) + weight>0 (this fn is only
    called then) but the corpus carries NONE of the aux columns => the objective
    would silently train as a no-op. Must RAISE, not return (0.0, 0)."""
    out = _outputs()
    data = {}  # corpus without any aux target column
    with pytest.raises(ValueError, match="CAT-105"):
        train_bc._aux_subgoal_loss(out, data, np.arange(_N), _DEVICE)


def test_heads_absent_is_noop():
    """Heads-off model (no aux outputs): fn is a pure no-op, never raises."""
    out = {"logits": torch.randn(_N, 5), "value": torch.randn(_N)}
    data = {}
    loss, active = train_bc._aux_subgoal_loss(out, data, np.arange(_N), _DEVICE)
    assert active == 0
    assert float(loss) == 0.0


def test_all_heads_present_counts_five_and_is_differentiable():
    out = _outputs()
    data = {
        "aux_longest_road": np.array([0, 1, 1, 0], dtype=np.float32),
        "aux_largest_army": np.array([0, 0, 1, 0], dtype=np.float32),
        "aux_vp_in_n": np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float32),
        "aux_next_settlement": np.array([5, 12, -1, 40], dtype=np.int64),
        "aux_robber_target": np.array([3, -1, -1, 7], dtype=np.int64),
    }
    loss, active = train_bc._aux_subgoal_loss(out, data, np.arange(_N), _DEVICE)
    assert active == 5
    assert torch.isfinite(loss)
    loss.backward()
    # Categorical head with some valid rows receives gradient.
    assert out["aux_next_settlement"].grad is not None
    assert out["aux_next_settlement"].grad.abs().sum().item() > 0.0


def test_all_ignored_categorical_head_is_skipped():
    out = _outputs()
    data = {
        "aux_next_settlement": np.array([-1, -1, -1, -1], dtype=np.int64),
    }
    loss, active = train_bc._aux_subgoal_loss(out, data, np.arange(_N), _DEVICE)
    # The head's targets are entirely ignore rows -> no active head, no loss.
    assert active == 0
    assert float(loss) == 0.0


def test_partial_head_subset_counted():
    out = _outputs()
    data = {
        "aux_longest_road": np.array([1, 0, 1, 1], dtype=np.float32),
        "aux_vp_in_n": np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32),
    }
    loss, active = train_bc._aux_subgoal_loss(out, data, np.arange(_N), _DEVICE)
    assert active == 2
    assert torch.isfinite(loss) and float(loss) > 0.0


def test_spatial_aux_labels_follow_the_exact_sampled_d6_orientation():
    """D6-augmented inputs must not retain absolute-coordinate aux labels."""
    class _FakeSymmetry:
        fwd_vertex = np.stack(
            [np.arange(54), np.roll(np.arange(54), -1)], axis=0
        )
        fwd_hex = np.stack(
            [np.arange(19), np.roll(np.arange(19), -1)], axis=0
        )

        @staticmethod
        def _remap_values(ids, table_g):
            rows = np.arange(ids.shape[0])
            return np.where(ids < 0, -1, table_g[rows, np.maximum(ids, 0)])

    symmetry = _FakeSymmetry()
    batch = np.arange(2)
    g = np.array([1, 1], dtype=np.int64)
    settlement = np.array([5, 31], dtype=np.int64)
    robber = np.array([3, 14], dtype=np.int64)
    mapped_settlement = symmetry.fwd_vertex[g, settlement]
    mapped_robber = symmetry.fwd_hex[g, robber]

    settlement_logits = torch.full((2, 54), -20.0, requires_grad=True)
    robber_logits = torch.full((2, 19), -20.0, requires_grad=True)
    with torch.no_grad():
        settlement_logits[torch.arange(2), torch.as_tensor(mapped_settlement)] = 20.0
        robber_logits[torch.arange(2), torch.as_tensor(mapped_robber)] = 20.0
    outputs = {
        "aux_next_settlement": settlement_logits,
        "aux_robber_target": robber_logits,
    }
    data = {
        "aux_next_settlement": settlement,
        "aux_robber_target": robber,
    }

    loss, active = train_bc._aux_subgoal_loss(
        outputs,
        data,
        batch,
        _DEVICE,
        symmetry=symmetry,
        symmetry_ids=g,
    )
    assert active == 2
    assert float(loss) < 1.0e-6
