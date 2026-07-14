from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from tools import a1_scientific_evidence as scientific_evidence
from tools import train_bc


def _sha(character: str) -> str:
    return "sha256:" + character * 64


class _CentralComposite:
    component_ids = (
        "current_producer",
        "recent_history",
        "hard_negative",
        "historical_replay",
    )

    def __init__(self, rows_per_component: int = 32) -> None:
        self.component_offsets = (
            np.arange(5, dtype=np.int64) * rows_per_component
        )
        rows = int(self.component_offsets[-1])
        self._legal = np.tile(np.arange(3, dtype=np.int64), (rows, 1))
        self._prior = np.full((rows, 3), 1.0 / 3.0, dtype=np.float32)

    def component_indices_for_rows(self, rows):
        return np.searchsorted(self.component_offsets[1:], rows, side="right")

    def __getitem__(self, key: str):
        return {
            "legal_action_ids": self._legal,
            "prior_policy": self._prior,
        }[key]


def _meta() -> dict[str, object]:
    return {
        "components": [
            {
                "component_id": component_id,
                "payload_inventory_sha256": _sha(str(index + 1)),
            }
            for index, component_id in enumerate(_CentralComposite.component_ids)
        ]
    }


def test_realized_order_hashes_physical_rows_not_train_positions() -> None:
    data = _CentralComposite()
    train = np.asarray([95, 3, 65, 34, 1, 96, 66, 35], dtype=np.int64)
    order = np.asarray([1, 0, 3, 2, 1, 7], dtype=np.int64)
    result = train_bc._a1_central_sample_order_evidence(  # noqa: SLF001
        data,
        train_indices=train,
        validation_indices=np.asarray([0, 2, 4], dtype=np.int64),
        global_order_positions=order,
        composite_meta=_meta(),
        expected_sample_dose=len(order),
    )
    digest = hashlib.sha256()
    for draw_index, physical_row in enumerate(train[order]):
        component_index = int(
            np.searchsorted(data.component_offsets[1:], physical_row, side="right")
        )
        component_id = data.component_ids[component_index]
        identity = scientific_evidence.canonical_row_identity(
            payload_member_sha256=_sha(str(component_index + 1)),
            row_offset=int(physical_row - data.component_offsets[component_index]),
            component_id=component_id,
            prior_policy_present=True,
            legal_action_count=3,
        )
        train_bc._a1_order_digest_update(  # noqa: SLF001
            digest, index=draw_index, row_identity_sha256=identity
        )
    assert result["physical_row_identity"] is True
    assert result["sample_dose"] == len(order)
    assert result["sample_order_sha256"] == "sha256:" + digest.hexdigest()

    tampered = train_bc._a1_central_sample_order_evidence(  # noqa: SLF001
        data,
        train_indices=train,
        validation_indices=np.asarray([0, 2, 4], dtype=np.int64),
        global_order_positions=order[::-1],
        composite_meta=_meta(),
        expected_sample_dose=len(order),
    )
    assert tampered["sample_order_sha256"] != result["sample_order_sha256"]


def test_realized_order_rejects_validation_overlap_and_position_drift() -> None:
    data = _CentralComposite()
    train = np.arange(16, dtype=np.int64)
    with pytest.raises(RuntimeError, match="validation-excluded physical row"):
        train_bc._a1_central_sample_order_evidence(  # noqa: SLF001
            data,
            train_indices=train,
            validation_indices=np.asarray([7], dtype=np.int64),
            global_order_positions=np.arange(8, dtype=np.int64),
            composite_meta=_meta(),
            expected_sample_dose=8,
        )
    with pytest.raises(RuntimeError, match="non-training position"):
        train_bc._a1_central_sample_order_evidence(  # noqa: SLF001
            data,
            train_indices=train,
            validation_indices=np.asarray([31], dtype=np.int64),
            global_order_positions=np.asarray([0, 16], dtype=np.int64),
            composite_meta=_meta(),
            expected_sample_dose=2,
        )


def test_rank_stride_reinterleave_exact_524288_without_padding() -> None:
    dose = 524_288
    world = 8
    global_order = np.arange(dose, dtype=np.int64)
    rank_orders = [global_order[rank::world] for rank in range(world)]
    replay = train_bc._a1_reinterleave_rank_stride_orders(  # noqa: SLF001
        rank_orders, expected_global_size=dose
    )
    assert np.array_equal(replay, global_order)
    with pytest.raises(RuntimeError, match="padding or changed dose"):
        train_bc._a1_reinterleave_rank_stride_orders(  # noqa: SLF001
            [np.append(value, index) for index, value in enumerate(rank_orders)],
            expected_global_size=dose,
        )


def test_epoch_order_observer_does_not_advance_sampler_rng() -> None:
    weights = np.linspace(1.0, 2.0, 128, dtype=np.float64)
    ddp = {"enabled": True, "world_size": 8, "rank": 3}
    seed = 424242
    observed: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    baseline = np.random.default_rng(seed)
    before = copy.deepcopy(rng.bit_generator.state)
    local = train_bc._epoch_order(  # noqa: SLF001
        rng,
        128,
        4,
        ddp,
        sample_weights=weights,
        max_samples=64,
        global_order_observer=lambda order: observed.append(order.copy()),
    )
    baseline_local = train_bc._epoch_order(  # noqa: SLF001
        baseline,
        128,
        4,
        ddp,
        sample_weights=weights,
        max_samples=64,
    )
    assert before != rng.bit_generator.state
    assert np.array_equal(local, baseline_local)
    assert np.array_equal(local, observed[0][3::8])
    assert rng.random() == baseline.random()


def test_realized_order_refuses_tampered_authority_before_training() -> None:
    realized = {
        "schema_version": "a1-realized-central-sample-order-v1",
        "sample_dose": 8,
        "sample_order_sha256": _sha("1"),
        "row_set_sha256": _sha("2"),
        "unique_row_count": 7,
        "kl_eligible_rows": 2,
        "kl_eligible_mass_decimal": "0.25",
        "kl_ordered_evidence_sha256": _sha("3"),
        "kl_eligible_evidence_sha256": _sha("4"),
        "physical_row_identity": True,
        "validation_rows_excluded": True,
    }
    binding = {
        key: value
        for key, value in realized.items()
        if key
        in {
            "sample_dose",
            "sample_order_sha256",
            "row_set_sha256",
            "unique_row_count",
            "kl_eligible_rows",
            "kl_eligible_mass_decimal",
            "kl_ordered_evidence_sha256",
            "kl_eligible_evidence_sha256",
        }
    }
    assert train_bc._a1_verify_realized_central_sample_order(  # noqa: SLF001
        realized, binding
    ) == realized
    tampered = dict(binding)
    tampered["sample_order_sha256"] = _sha("f")
    with pytest.raises(RuntimeError, match="sample_order_sha256"):
        train_bc._a1_verify_realized_central_sample_order(  # noqa: SLF001
            realized, tampered
        )
