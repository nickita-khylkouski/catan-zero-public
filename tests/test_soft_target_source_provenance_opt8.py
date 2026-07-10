"""OPT-8: checkpoints record which soft policy target they trained against.

A checkpoint trained with the degenerate prefer_scores target was previously
indistinguishable from a policy-target one without digging through report.json.
EntityGraphPolicy.save now stamps soft_target_source into the checkpoint dict
(mirroring the mask_hidden_info provenance field).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import train_bc  # type: ignore  # noqa: E402

from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)

ACTION_SIZE = 8
STATIC_FEATURE_SIZE = 4


def _tiny_policy() -> EntityGraphPolicy:
    config = EntityGraphConfig(
        action_size=ACTION_SIZE,
        static_action_feature_size=STATIC_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    static = np.zeros((ACTION_SIZE, STATIC_FEATURE_SIZE), dtype=np.float32)
    return EntityGraphPolicy(config, static, device="cpu")


def _load_raw(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def test_save_records_soft_target_source(tmp_path):
    path = tmp_path / "ckpt.pt"
    _tiny_policy().save(path, mask_hidden_info=False, soft_target_source="policy")
    assert _load_raw(path)["soft_target_source"] == "policy"


def test_save_records_prefer_scores_when_that_was_used(tmp_path):
    path = tmp_path / "ckpt.pt"
    _tiny_policy().save(path, soft_target_source="prefer_scores")
    assert _load_raw(path)["soft_target_source"] == "prefer_scores"


def test_soft_target_source_defaults_to_empty_when_omitted(tmp_path):
    path = tmp_path / "ckpt.pt"
    _tiny_policy().save(path)  # neither kwarg
    data = _load_raw(path)
    assert data["soft_target_source"] == ""
    # OPT-8 must not disturb the existing provenance field.
    assert data["mask_hidden_info"] is False


def test_distributed_checkpoint_writer_records_soft_target_source(tmp_path):
    path = tmp_path / "ddp_ckpt.pt"
    policy = _tiny_policy()
    train_bc._write_entity_checkpoint(
        policy,
        str(path),
        policy.model.state_dict(),
        True,
        soft_target_source="policy",
    )
    data = _load_raw(path)
    assert data["soft_target_source"] == "policy"
    assert data["mask_hidden_info"] is True
