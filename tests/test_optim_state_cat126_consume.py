"""CAT-126 #8 (consume-side, audit-fixer): supplementary coverage on top of
fsdp-builder's optim_state.py (CAT-128) starter tests. Focus: the atomic-write,
non-rank0 contract, provenance tag, and the arch-mismatch fail-safe (the
grow-from-checkpoint path the design specifically called out). DDP/single-GPU +
sidecar are covered by test_optim_state_cat128.py; these do not duplicate them.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.optim_state import (
    load_optimizer_state,
    optimizer_sidecar_path,
    save_optimizer_state,
)

_DDP0 = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
_DDP_RANK1 = {"enabled": True, "world_size": 2, "rank": 1, "local_rank": 1}


def _stepped_adam(model=None):
    model = model or torch.nn.Linear(8, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(5, 8)).sum().backward()
        opt.step()
    return model, opt


def test_save_is_atomic_no_tmp_left(tmp_path):
    ckpt = tmp_path / "sub" / "ckpt.pt"
    model, opt = _stepped_adam()
    save_optimizer_state(ckpt, model, opt, _DDP0)
    assert optimizer_sidecar_path(ckpt).exists()
    # atomic temp+rename must leave no partial file behind
    assert not list(tmp_path.glob("**/*.tmp*"))
    assert not list(tmp_path.glob("**/.*.tmp*"))


def test_sidecar_records_plain_format_tag(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    model, opt = _stepped_adam()
    save_optimizer_state(ckpt, model, opt, _DDP0)
    blob = torch.load(optimizer_sidecar_path(ckpt), map_location="cpu", weights_only=False)
    assert blob["format"] == "plain"  # DDP/single provenance


def test_nonrank0_does_not_write(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    model, opt = _stepped_adam()
    # DDP save is a rank0-only WRITE (all ranks call, only rank0 touches the file).
    assert save_optimizer_state(ckpt, model, opt, _DDP_RANK1) is None
    assert not optimizer_sidecar_path(ckpt).exists()


def test_arch_mismatch_load_is_failsafe(tmp_path):
    """grow-from-checkpoint: a sidecar from a different arch (param-count mismatch)
    must NOT crash the resume -- load returns False and the fresh optimizer is
    untouched (trains from zero-state)."""
    ckpt = tmp_path / "ckpt.pt"
    small, small_opt = _stepped_adam(torch.nn.Linear(8, 4))          # 2 params
    save_optimizer_state(ckpt, small, small_opt, _DDP0)

    big = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))  # 4 params
    big_opt = torch.optim.Adam(big.parameters(), lr=1e-3)
    assert load_optimizer_state(ckpt, big, big_opt, _DDP0) is False
    assert len(big_opt.state) == 0  # fresh optimizer left intact
