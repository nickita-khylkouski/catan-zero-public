from __future__ import annotations

import json
import subprocess

import pytest

from tools import a1_production_l1_rerun as l1


def _historical() -> list[str]:
    return [
        "python", "-m", "torch.distributed.run", "--nproc-per-node=8",
        "train_bc.py", "--arch", "entity_graph", "--hidden-size", "640",
        "--graph-layers", "6", "--attention-heads", "8", "--epochs", "1",
        "--max-steps", "1024", "--batch-size", "512", "--grad-accum-steps", "1",
        "--optimizer", "adam", "--no-resume-optimizer", "--no-fused-optimizer",
        "--lr", "3e-05", "--lr-warmup-steps", "100", "--soft-target-weight", "0.9",
        "--value-loss-weight", "0.25", "--forced-action-weight", "0.0",
        "--loser-sample-weight", "1.0", "--mask-hidden-info",
        "--graph-history-features", "--trust-curated-data-quality",
    ]


def test_latest_main_additions_project_to_exact_historical_l1() -> None:
    historical = _historical()
    inventories = ["sha256:" + char * 64 for char in "abc"]
    derived = historical + [
        "--max-grad-norm", "1.0", "--policy-aux-active-batch-size", "0",
        l1.ACK_FLAG, inventories[0], l1.ACK_FLAG, inventories[1],
        l1.ACK_FLAG, inventories[2], l1.CROP_FLAG,
    ]

    assert l1._historical_projection(derived, inventories) == historical
    l1._validate_historical_recipe(historical)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (("--max-steps", "2048"), "historical L1 drift"),
        (("--loser-sample-weight", "0.3"), "historical L1 drift"),
        (("--soft-target-weight", "1.0"), "historical L1 drift"),
    ],
)
def test_historical_l1_recipe_fails_closed_on_causal_drift(
    mutation: tuple[str, str], match: str
) -> None:
    command = _historical()
    command[command.index(mutation[0]) + 1] = mutation[1]
    with pytest.raises(l1.L1Error, match=match):
        l1._validate_historical_recipe(command)


def test_idle_probe_requires_exactly_eight_b200s_and_no_compute() -> None:
    calls = 0

    def runner(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess([], 0, stdout="NVIDIA B200\n" * 8)
        return subprocess.CompletedProcess([], 0, stdout="")

    assert l1._idle_b200s(runner) == []


def test_idle_probe_reports_non_mps_compute_but_ignores_mps() -> None:
    outputs = iter(
        [
            "NVIDIA B200\n" * 8,
            "12, /usr/bin/nvidia-cuda-mps-server\n13, python\n",
        ]
    )

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout=next(outputs))

    assert l1._idle_b200s(runner) == ["13, python"]


def test_projection_rejects_extra_or_reordered_inventory_ack() -> None:
    inventories = ["a", "b", "c"]
    derived = _historical() + [
        "--max-grad-norm", "1.0", "--policy-aux-active-batch-size", "0",
        l1.ACK_FLAG, "b", l1.ACK_FLAG, "a", l1.ACK_FLAG, "c", l1.CROP_FLAG,
    ]
    with pytest.raises(l1.L1Error, match="order drift"):
        l1._historical_projection(derived, inventories)


def test_python_binding_preserves_lexical_venv_symlink(tmp_path) -> None:
    real = tmp_path / "python-real"
    real.write_bytes(b"#!/bin/sh\n")
    real.chmod(0o755)
    lexical = tmp_path / "venv-python"
    lexical.symlink_to(real)

    binding = l1._python_binding(lexical)

    assert binding["lexical_path"] == str(lexical)
    assert binding["resolved_path"] == str(real)
    assert l1._verify_python_binding(binding) == str(lexical)


def _empty_event_surface(inventories: list[str]) -> dict[str, object]:
    scan: dict[str, object] = {
        "schema": "training-empty-event-mask-scan-v1",
        "row_count": 47_620_447,
        "padded_event_width": 64,
        "nonzero_event_mask_count": 0,
    }
    scan["scan_sha256"] = l1._digest(scan)
    return {
        "schema": "a1-training-event-history-contract-v1",
        "status": "empty_payloads_acknowledged",
        "graph_history_observation_schema": True,
        "event_history_consumer_enabled": True,
        "training_event_history_trainable": False,
        "event_history_end_to_end_usable": False,
        "training_event_tensor_width": 0,
        "empty_payload_inventory_acknowledgements": sorted(inventories),
        "empty_event_mask_scan": scan,
    }


def test_completed_report_proves_authenticated_empty_event_fast_path() -> None:
    inventories = ["sha256:" + char * 64 for char in "abc"]
    report = {"training_information_surface": _empty_event_surface(inventories)}

    l1._verify_authenticated_empty_event_report(report, inventories)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_event_tensor_width", 64),
        ("training_event_history_trainable", True),
        ("event_history_end_to_end_usable", True),
    ],
)
def test_completed_report_rejects_empty_event_fast_path_drift(
    field: str, value: object
) -> None:
    inventories = ["sha256:" + char * 64 for char in "abc"]
    surface = _empty_event_surface(inventories)
    surface[field] = value

    with pytest.raises(l1.L1Error, match="fast-path drift"):
        l1._verify_authenticated_empty_event_report(
            {"training_information_surface": surface}, inventories
        )


def test_completed_report_rejects_forged_empty_event_scan() -> None:
    inventories = ["sha256:" + char * 64 for char in "abc"]
    surface = _empty_event_surface(inventories)
    scan = json.loads(json.dumps(surface["empty_event_mask_scan"]))
    scan["row_count"] = 1
    surface["empty_event_mask_scan"] = scan

    with pytest.raises(l1.L1Error, match="scan is malformed or drifted"):
        l1._verify_authenticated_empty_event_report(
            {"training_information_surface": surface}, inventories
        )
