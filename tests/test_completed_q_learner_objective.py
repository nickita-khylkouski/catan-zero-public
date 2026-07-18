from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from catan_zero.rl.target_reliability import (
    TARGET_RELIABILITY_CONFIDENCE_FORMULA,
    TARGET_RELIABILITY_SCHEMA,
)
from tools import train_bc


def _bound_completed_q_data() -> dict:
    operator = "sha256:" + "1" * 64
    data = {
        "action_taken": np.asarray([10, 20], dtype=np.int64),
        "legal_action_ids": np.asarray(
            [[10, 11, -1], [20, 21, 22]], dtype=np.int64
        ),
        "completed_q_values": np.asarray(
            [[0.8, -0.2, np.nan], [0.3, 0.1, -0.4]], dtype=np.float32
        ),
        "completed_q_mask": np.asarray(
            [[True, True, False], [True, True, True]], dtype=np.bool_
        ),
        "target_scores": np.asarray(
            [[99.0, -99.0, np.nan], [99.0, -99.0, 55.0]], dtype=np.float32
        ),
        "target_reliability_version": np.ones(2, dtype=np.uint8),
        "target_reliability_audited": np.asarray([True, False]),
        "target_reliability_js_divergence": np.asarray(
            [0.5 * math.log(2.0), np.nan], dtype=np.float32
        ),
        "target_reliability_policy_top1_agreement": np.asarray(
            [True, False]
        ),
        "target_reliability_q_top1_agreement": np.asarray([True, False]),
        "target_reliability_q_margin_primary": np.asarray(
            [1.0, np.nan], dtype=np.float32
        ),
        "target_reliability_q_margin_duplicate": np.asarray(
            [0.9, np.nan], dtype=np.float32
        ),
        "target_reliability_confidence": np.asarray(
            [0.5, 1.0], dtype=np.float32
        ),
        "target_information_regime": np.asarray(
            ["public_belief_single_tree_v1", "public_belief_single_tree_v1"]
        ),
    }
    data["meta"] = {
        "columns": {
            "completed_q_values": {"kind": "ragged2d", "dtype": "float32"},
            "completed_q_mask": {"kind": "ragged2d", "dtype": "bool"},
        },
        "stage_c_policy_overlay": {
            "completed_q_binding": {
                "schema_version": "a1-stage-c-completed-q-binding-v1",
                "columns": {
                    "values": "completed_q_values",
                    "mask": "completed_q_mask",
                },
                "semantics": {
                    "range": [-1.0, 1.0],
                    "support": "every_legal_action_on_selected_stage_c_rows",
                    "target_scores_relation": (
                        "separate_raw_visited_q_column_never_overwritten"
                    ),
                },
                "row_identity": {"selected_rows": 2},
                "operator_identity": {
                    "target_policy_target_identity_sha256": operator,
                    "q_values_root_perspective": True,
                    "legacy_or_unbound_q_allowed": False,
                },
                "reliability_identity": {
                    "schema_version": TARGET_RELIABILITY_SCHEMA,
                    "version": 1,
                    "receipt_sha256": "sha256:" + "2" * 64,
                },
            }
        },
    }
    return data


class _Corpus(dict):
    def __len__(self):
        return len(self["action_taken"])

    @property
    def meta(self):
        return self["meta"]


def test_completed_q_loss_is_return_scale_and_reliability_weighted() -> None:
    data = _bound_completed_q_data()
    q_values = torch.zeros((2, 3), dtype=torch.float32)
    loss, weighted_sum, denominator = train_bc._completed_q_loss_parts(  # noqa: SLF001
        q_values,
        data,
        np.asarray([0, 1], dtype=np.int64),
        torch.ones(2),
        torch.device("cpu"),
        confidence_floor=0.25,
        policy_weights_include_reliability=False,
    )
    row0 = (0.8**2 + (-0.2) ** 2) / 2.0
    row1 = (0.3**2 + 0.1**2 + (-0.4) ** 2) / 3.0
    expected_sum = 0.5 * row0 + row1
    assert denominator.item() == pytest.approx(1.5)
    assert weighted_sum.item() == pytest.approx(expected_sum)
    assert loss.item() == pytest.approx(expected_sum / 1.5)


def test_completed_q_admission_binds_operator_and_never_target_scores() -> None:
    corpus = _Corpus(_bound_completed_q_data())
    report = train_bc._completed_q_binding_admission(  # noqa: SLF001
        corpus,
        loss_weight=0.2,
        confidence_floor=0.25,
        policy_weights_include_reliability=False,
    )
    assert report["enabled"] is True
    assert report["legacy_target_scores_consumed"] is False
    assert report["semantics"] == "return_scale_root_actor_perspective_mse"
    assert (
        report["reliability_weighting"]["formula"]
        == TARGET_RELIABILITY_CONFIDENCE_FORMULA
    )

    del corpus["meta"]["stage_c_policy_overlay"]
    with pytest.raises(SystemExit, match="unbound values"):
        train_bc._completed_q_binding_admission(  # noqa: SLF001
            corpus,
            loss_weight=0.2,
        )
