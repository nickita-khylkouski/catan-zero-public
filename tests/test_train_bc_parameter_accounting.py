from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tools import train_bc  # noqa: E402


class _AccountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.active = torch.nn.Linear(2, 2)  # 6 params: trained + executed
        self.frozen_active = torch.nn.Linear(2, 1)  # 3: frozen, still executed
        self.q_head = torch.nn.Linear(2, 1)  # 3: frozen + skipped


def _policy() -> SimpleNamespace:
    model = _AccountingModel()
    for parameter in model.frozen_active.parameters():
        parameter.requires_grad = False
    for parameter in model.q_head.parameters():
        parameter.requires_grad = False
    model._forward_inactive_parameter_modules = frozenset({"q_head"})
    return SimpleNamespace(model=model)


def test_parameter_accounting_distinguishes_all_three_surfaces() -> None:
    report = train_bc._parameter_accounting(_policy())

    assert report == {
        "schema_version": "model-parameter-accounting-v1",
        "total_parameters": 12,
        "trainable_parameters": 6,
        "forward_active_parameters": 9,
        "forward_inactive_parameters": 3,
        "forward_inactive_submodules": ["q_head"],
        "total_parameter_contract": (
            "serialized_checkpoint_structure_including_frozen_or_skipped_heads"
        ),
    }


def test_35m_guard_remains_a_total_checkpoint_size_contract() -> None:
    policy = _policy()
    accepted = SimpleNamespace(
        require_35m_model=True,
        arch="entity_graph",
        min_35m_params=12,
        max_35m_params=12,
    )
    train_bc._enforce_35m_model_size(policy, accepted)

    rejected = SimpleNamespace(
        require_35m_model=True,
        arch="entity_graph",
        min_35m_params=13,
        max_35m_params=20,
    )
    with pytest.raises(SystemExit, match="total checkpoint parameter count"):
        train_bc._enforce_35m_model_size(policy, rejected)


def test_parameter_accounting_gate_is_not_checkpoint_state() -> None:
    policy = _policy()

    assert not any(
        "forward_inactive" in name for name in policy.model.state_dict()
    )
    assert train_bc._parameter_accounting(policy)["total_parameters"] == sum(
        parameter.numel() for parameter in policy.model.parameters()
    )
