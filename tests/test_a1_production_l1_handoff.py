from __future__ import annotations

import json

import pytest

from tools import a1_production_l1_handoff as handoff


def test_pending_bundle_digest_and_authorization_fail_closed(tmp_path) -> None:
    payload = {
        "schema_version": handoff.SCHEMA,
        "promotion_ready": False,
        "pointer_mutation_authorized": False,
        "learner": {},
        "evidence": {},
        "authoritative_transaction_audit": {},
    }
    payload["bundle_sha256"] = handoff._digest(payload)
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((KeyError, handoff.HandoffError)):
        handoff.verify(path)


def test_checkpoint_sha_refuses_missing_binding() -> None:
    with pytest.raises(handoff.HandoffError, match="candidate checkpoint SHA"):
        handoff._checkpoint_sha({}, "test")


def test_selected_completion_replays_typed_dose_and_actual_draws(
    tmp_path, monkeypatch
) -> None:
    candidate_path = tmp_path / "candidate.pt"
    candidate_path.write_bytes(b"candidate")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    submission_path = tmp_path / "submission.json"
    submission_path.write_text("{}")
    selected = handoff.learner_dose.PARETO_SELECTED_DOSE
    f7_sha = "sha256:" + "f" * 64
    report_payload = {
        "max_steps": selected.optimizer_steps,
        "steps_completed": selected.optimizer_steps,
        "world_size": selected.world_size,
        "batch_size": selected.per_rank_batch_size,
        "grad_accum_steps": selected.grad_accum_steps,
        "effective_global_batch_size": selected.effective_global_batch_size,
        "training_row_draws": selected.global_samples,
        "base_training_row_draws": selected.global_samples,
        "policy_aux_training_row_draws": 0,
        "total_training_row_draws": selected.global_samples,
        "init_checkpoint_sha256": f7_sha,
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload))
    candidate = handoff._ref(candidate_path)
    report = handoff._ref(report_path)
    manifest = handoff._ref(manifest_path)
    completion_payload = {
        "schema_version": handoff.production_l1.COMPLETION_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "created_at_unix_ns": 1,
        "manifest": manifest,
        "submission": handoff._ref(submission_path),
        "checkpoint": candidate,
        "report": report,
        "unit_state": {
            "ActiveState": "inactive",
            "Result": "success",
            "ExecMainStatus": "0",
        },
        "dose_contract": selected.payload(),
    }
    completion_payload["receipt_sha256"] = handoff._digest(completion_payload)
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(json.dumps(completion_payload))
    completion = handoff._ref(completion_path)
    monkeypatch.setattr(
        handoff.production_l1,
        "verify",
        lambda _path: {"manifest_ref": manifest, "dose": selected},
    )

    assert handoff._verify_selected_completion(
        candidate=candidate,
        report=report,
        completion=completion,
        f7_sha256=f7_sha,
    ) == completion_payload["receipt_sha256"]

    report_payload["training_row_draws"] = 4_194_304
    report_path.write_text(json.dumps(report_payload))
    report = handoff._ref(report_path)
    completion_payload["report"] = report
    completion_payload["receipt_sha256"] = handoff._digest(
        {
            key: value
            for key, value in completion_payload.items()
            if key != "receipt_sha256"
        }
    )
    completion_path.write_text(json.dumps(completion_payload))
    with pytest.raises(handoff.HandoffError, match="training_row_draws"):
        handoff._verify_selected_completion(
            candidate=candidate,
            report=report,
            completion=handoff._ref(completion_path),
            f7_sha256=f7_sha,
        )
