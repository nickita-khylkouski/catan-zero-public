from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_completion_telemetry_amendment as amendment


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _profile(checkpoint: Path, milliseconds: float) -> dict[str, object]:
    summary = {
        "mean": milliseconds,
        "median": milliseconds,
        "p95": milliseconds * 1.1,
        "min": milliseconds * 0.9,
    }
    return {
        "device": "NVIDIA B200",
        "checkpoint": str(checkpoint),
        "strict_fp32": {
            "matmul_precision": "highest",
            "cuda_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "autocast": False,
        },
        "shape": {
            "batch_size": 48,
            "legal_width": 54,
            "event_width": 0,
            "valid_players": 2,
        },
        "warmup": 20,
        "iterations": 100,
        "return_q": True,
        "exact_window": {"cuda_ms": summary, "wall_ms": summary},
        "exact_vs_attributed_output_parity": {
            "logits": {"max_abs": 0.0, "mean_abs": 0.0},
            "value": {"max_abs": 0.0, "mean_abs": 0.0},
        },
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    parent = tmp_path / "parent.pt"
    candidate = tmp_path / "candidate.pt"
    old_finalizer = tmp_path / "old_finalizer.py"
    fix = tmp_path / "fixed_finalizer.py"
    parent.write_bytes(b"parent")
    candidate.write_bytes(b"candidate")
    old_finalizer.write_text("def main(): return base.main()\n")
    fix.write_text("def main():\n    value = finalize(manifest)\n")

    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "completion_finalizer": amendment._file_ref(old_finalizer),
        "inference_cost_contract": {
            "required_before_completion": True,
            "strict_fp32": True,
            "reference_checkpoint": amendment._file_ref(parent),
            "matched_shape": {
                "batch_size": 48,
                "legal_width": 54,
                "event_width": 0,
                "valid_players": 2,
                "warmup": 20,
                "iterations": 100,
                "return_q": True,
            },
        },
    }
    manifest["manifest_sha256"] = amendment._digest(manifest)
    _write_json(manifest_path, manifest)

    historical_path = tmp_path / "historical.json"
    candidate_ref: dict[str, object] = amendment._file_ref(candidate)
    candidate_ref["size_bytes"] = candidate.stat().st_size
    finalizer_ref: dict[str, object] = amendment._file_ref(old_finalizer)
    finalizer_ref["size_bytes"] = old_finalizer.stat().st_size
    historical = {
        "status": "complete_nonpromotable",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "manifest": amendment._file_ref(manifest_path),
        "completion_finalizer": finalizer_ref,
        "checkpoint": candidate_ref,
    }
    historical["receipt_sha256"] = amendment._digest(historical)
    _write_json(historical_path, historical)

    reference_profile = tmp_path / "reference.json"
    candidate_profile = tmp_path / "candidate.json"
    _write_json(reference_profile, _profile(parent, 2.0))
    _write_json(candidate_profile, _profile(candidate, 3.0))
    return {
        "historical": historical_path,
        "manifest": manifest_path,
        "reference": reference_profile,
        "candidate": candidate_profile,
        "fix": fix,
    }


def _build(paths: dict[str, Path]) -> dict[str, object]:
    return amendment.build_receipt(
        historical_completion=paths["historical"],
        manifest_path=paths["manifest"],
        reference_profile=paths["reference"],
        candidate_profile=paths["candidate"],
        dispatcher_fix=paths["fix"],
        created_at_unix_ns=123,
    )


def test_amendment_replays_historical_completion_and_cost(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt = _build(paths)
    output = tmp_path / "amendment.json"
    _write_json(output, receipt)

    assert amendment.verify(output) == receipt
    telemetry = receipt["inference_cost_telemetry"]
    assert isinstance(telemetry, dict)
    assert telemetry["candidate_reference_ratios"]["cuda_mean_slowdown"] == 1.5
    assert receipt["promotion_eligible"] is False


def test_amendment_rejects_nonzero_profile_parity(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    value = json.loads(paths["candidate"].read_text())
    value["exact_vs_attributed_output_parity"]["logits"]["max_abs"] = 1e-6
    _write_json(paths["candidate"], value)

    with pytest.raises(amendment.AmendmentError, match="bit-exact"):
        _build(paths)


def test_amendment_rejects_historical_receipt_tamper(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    value = json.loads(paths["historical"].read_text())
    value["promotion_eligible"] = True
    _write_json(paths["historical"], value)

    with pytest.raises(amendment.AmendmentError, match="semantic digest"):
        _build(paths)
