"""OPT-4: bit-identical vectorization of the active masked-logits helper."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import torch

from catan_zero.rl.torch_ppo import _masked_logits  # noqa: E402


def _ref_masked_logits(logits, valid_actions):
    mask = torch.full_like(logits, -1e9)
    for row, actions in enumerate(valid_actions):
        if actions:
            mask[row, list(actions)] = 0.0
    return logits + mask


def _random_legal(rng, batch, max_legal, action_size):
    """A (batch, max_legal) int array of action ids with -1 padding, some rows empty."""
    out = np.full((batch, max_legal), -1, dtype=np.int64)
    for r in range(batch):
        k = int(rng.integers(0, max_legal + 1))  # 0 => empty row
        if k:
            out[r, :k] = rng.choice(action_size, size=k, replace=False)
    return out


def test_masked_logits_bit_identical():
    rng = np.random.default_rng(2)
    action_size = 54
    for _ in range(20):
        legal = _random_legal(rng, 130, 12, action_size)
        valid = [tuple(int(a) for a in row if int(a) >= 0) for row in legal]
        logits = torch.randn(130, action_size, dtype=torch.float32)
        got = _masked_logits(logits, valid, action_size)
        ref = _ref_masked_logits(logits, valid)
        assert torch.equal(got, ref)


def test_masked_logits_all_empty_rows():
    action_size = 54
    valid = [() for _ in range(8)]
    logits = torch.randn(8, action_size, dtype=torch.float32)
    got = _masked_logits(logits, valid, action_size)
    ref = _ref_masked_logits(logits, valid)
    assert torch.equal(got, ref)
