from __future__ import annotations

import json
from pathlib import Path
import shlex
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tools import a1_pre_wave_contract as contract
from tools import a1_experimental_corpus_finalizer as finalizer
from tools import build_memmap_corpus as corpus


SHA = "sha256:" + "a" * 64


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sources(tmp_path: Path, arm: str = "n128") -> tuple[Path, Path, Path, Path]:
    lock = tmp_path / "lock.json"
    render = tmp_path / "render.json"
    _write_json(lock, {})
    _write_json(render, {})
    current = [
        {
            "arm_id": arm,
            "job_id": f"{arm}_gpu{i:02d}__current_producer",
            "worker_id": f"{arm}_gpu{i:02d}",
        }
        for i in range(28)
    ]
    reconstruction = {
        "classification": finalizer.LABEL,
        "arms": {
            arm: {"arm_selected_games": finalizer.QUOTAS[arm]["current_producer"]}
        },
        "job_bindings": current,
    }
    recovery_lanes = []
    for i in range(28):
        recovery_lanes.append(
            {
                "arm_id": arm,
                "lane_id": f"{arm}-lane-{i:02d}",
                "host_alias": "h0",
                "receipt": f"/remote/receipts/{arm}-{i:02d}.json",
                "commands": [
                    {
                        "arm_id": arm,
                        "job_id": f"{arm}_gpu{i:02d}__{category}",
                        "worker_id": f"{arm}_gpu{i:02d}",
                        "host_alias": "h0",
                        "category": category,
                        "output_dir": f"/remote/{arm}/{i}/{category}",
                    }
                    for category in ("recent_history", "hard_negative")
                ],
            }
        )
    recovery = {
        "label": finalizer.LABEL,
        "source_artifacts": {
            arm: {
                "lock": {"path": str(lock), "sha256": finalizer._file_sha(lock)},
                "lock_sha256": SHA,
                "render": {"path": str(render), "sha256": finalizer._file_sha(render)},
                "render_sha256": SHA,
            }
        },
        "lanes": recovery_lanes,
    }
    recovery["plan_sha256"] = finalizer._digest(recovery)
    reconstruction_path = tmp_path / "reconstruction.json"
    recovery_path = tmp_path / "recovery.json"
    _write_json(reconstruction_path, reconstruction)
    _write_json(recovery_path, recovery)
    return reconstruction_path, recovery_path, lock, render


def test_plan_is_immutable_nonpromotable_and_exact(tmp_path: Path) -> None:
    reconstruction, recovery, _lock, _render = _sources(tmp_path)
    out = tmp_path / "plan.json"
    value = finalizer.build_plan(reconstruction, recovery, "n128", out)
    assert value["production_eligible"] is False
    assert value["classification"] == finalizer.LABEL
    assert len(value["current_jobs"]) == 28
    assert len(value["recovery_jobs"]) == 56
    assert finalizer._verified_plan(out) == value
    with pytest.raises(finalizer.FinalizerError, match="immutable output drift"):
        finalizer._atomic_json(out, {"different": True})


def test_harvest_resumes_and_rehashes_every_published_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reconstruction, recovery, _lock, _render = _sources(tmp_path)
    plan_path = tmp_path / "plan.json"
    finalizer.build_plan(reconstruction, recovery, "n128", plan_path)

    def fake_run(argv: list[str], **_kwargs) -> SimpleNamespace:
        destination = Path(argv[-1])
        destination.mkdir(parents=True, exist_ok=True)
        np.savez(destination / "shard.npz", game_seed=np.asarray([1]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(finalizer.subprocess, "run", fake_run)
    monkeypatch.setattr(
        finalizer,
        "remote_completion",
        lambda *_args, **_kwargs: {"ready": True, "complete": 28, "total": 28},
    )
    root = tmp_path / "harvest"
    first = finalizer.harvest(plan_path, root, ["ssh"], parallelism=4)
    assert len(first["jobs"]) == 56
    assert finalizer.harvest(plan_path, root, ["ssh"], parallelism=4) == first
    (root / "jobs" / first["jobs"][0]["job_id"] / "shard.npz").write_bytes(b"drift")
    with pytest.raises(finalizer.FinalizerError, match="drifted"):
        finalizer.harvest(plan_path, root, ["ssh"], parallelism=4)


def test_parallel_harvest_is_bounded_deterministic_and_resumable_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reconstruction, recovery, _lock, _render = _sources(tmp_path)
    plan_path = tmp_path / "plan.json"
    finalizer.build_plan(reconstruction, recovery, "n128", plan_path)
    monkeypatch.setattr(
        finalizer,
        "remote_completion",
        lambda *_args, **_kwargs: {"ready": True, "complete": 28, "total": 28},
    )
    mutex = threading.Lock()
    active = 0
    maximum = 0
    calls: list[str] = []
    fail_once = {"needle": "recent_history", "armed": True}

    def fake_run(argv: list[str], **_kwargs) -> SimpleNamespace:
        nonlocal active, maximum
        source = argv[-2]
        with mutex:
            active += 1
            maximum = max(maximum, active)
            calls.append(source)
        try:
            time.sleep(0.005)
            if fail_once["armed"] and fail_once["needle"] in source:
                fail_once["armed"] = False
                return SimpleNamespace(returncode=23)
            destination = Path(argv[-1])
            destination.mkdir(parents=True, exist_ok=True)
            np.savez(destination / "shard.npz", game_seed=np.asarray([1]))
            return SimpleNamespace(returncode=0)
        finally:
            with mutex:
                active -= 1

    monkeypatch.setattr(finalizer.subprocess, "run", fake_run)
    root = tmp_path / "harvest"
    with pytest.raises(finalizer.FinalizerError, match="resumable"):
        finalizer.harvest(plan_path, root, ["ssh"], parallelism=4)
    assert maximum == 4
    assert not (root / "harvest.receipt.json").exists()
    first_call_count = len(calls)
    value = finalizer.harvest(plan_path, root, ["ssh"], parallelism=4)
    assert len(calls) == first_call_count + 1
    assert [row["job_id"] for row in value["jobs"]] == sorted(
        row["job_id"] for row in value["jobs"]
    )


def test_harvest_schedule_round_robins_hosts_without_changing_receipt_order() -> None:
    jobs = [
        {"job_id": "job-04", "host_alias": "h2"},
        {"job_id": "job-01", "host_alias": "h1"},
        {"job_id": "job-03", "host_alias": "h1"},
        {"job_id": "job-02", "host_alias": "h2"},
        {"job_id": "job-00", "host_alias": "h0"},
        {"job_id": "job-05", "host_alias": "h2"},
    ]

    scheduled = finalizer._host_round_robin_jobs(jobs)

    assert [(row["host_alias"], row["job_id"]) for row in scheduled] == [
        ("h0", "job-00"),
        ("h1", "job-01"),
        ("h2", "job-02"),
        ("h1", "job-03"),
        ("h2", "job-04"),
        ("h2", "job-05"),
    ]
    assert sorted(row["job_id"] for row in scheduled) == sorted(
        row["job_id"] for row in jobs
    )


def test_remote_receipt_gate_requires_exact_complete_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reconstruction, recovery, _lock, _render = _sources(tmp_path)
    plan_path = tmp_path / "plan.json"
    finalizer.build_plan(reconstruction, recovery, "n128", plan_path)

    def fake_run(argv: list[str], **_kwargs) -> SimpleNamespace:
        lanes = json.loads(shlex.split(argv[-1])[-1])
        rows = [{"lane_id": lane["lane_id"], "status": "complete"} for lane in lanes]
        return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")

    monkeypatch.setattr(finalizer.subprocess, "run", fake_run)
    ready = finalizer.remote_completion(plan_path, ["ssh"])
    assert ready["ready"] is True and ready["complete"] == 28

    def incomplete(*_args, **_kwargs):
        return {"ready": False, "complete": 27, "total": 28}

    monkeypatch.setattr(finalizer, "remote_completion", incomplete)
    with pytest.raises(finalizer.FinalizerError, match="27/28"):
        finalizer.wait_ready(plan_path, ["ssh"], poll_seconds=0.001, timeout_seconds=0)


def _shard(path: Path, seed: int, category: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        game_seed=np.asarray([seed, seed]),
        terminated=np.asarray([True, True]),
        truncated=np.asarray([False, False]),
        target_information_regime=np.asarray([finalizer.REQUIRED_REGIME] * 2),
        action_taken=np.asarray([0, 0]),
        legal_action_ids=np.asarray([[0], [0]]),
        category=np.asarray([category, category]),
        opponent_tag=np.asarray(
            ["" if category == "current_producer" else category] * 2
        ),
    )


def test_finalize_selects_lowest_complete_seed_and_emits_training_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        finalizer.QUOTAS,
        "n128",
        {"current_producer": 28, "recent_history": 28, "hard_negative": 28},
    )
    reconstruction, recovery, _lock, _render = _sources(tmp_path)
    plan_path = tmp_path / "plan.json"
    finalizer.build_plan(reconstruction, recovery, "n128", plan_path)
    plan = json.loads(plan_path.read_text())
    current_root = tmp_path / "current"
    recovery_root = tmp_path / "recovered"
    for index, row in enumerate(plan["current_jobs"]):
        _shard(
            current_root / "n128" / row["job_id"] / "x.npz",
            1_000 + index,
            "current_producer",
        )
    for index, row in enumerate(plan["recovery_jobs"]):
        root = recovery_root / "jobs" / row["job_id"]
        _shard(
            root / "x.npz",
            2_000 + index,
            row["category"],
        )
        producer = "sha256:" + "1" * 64
        opponent = (
            "sha256:" + ("2" if row["category"] == "recent_history" else "3") * 64
        )
        _write_json(
            root / "a1_contract.json",
            {
                "schema_version": "a1-generation-job-attestation-v2",
                "contract_sha256": SHA,
                "job_id": row["job_id"],
                "worker_id": row["worker_id"],
                "category": row["category"],
                "arm_id": "n128",
                "games": 1,
                "producer_checkpoint_sha256": producer,
                "opponent_checkpoint_sha256": [opponent],
                "search_operator_sha256": SHA,
                "effective_search_config_sha256": SHA,
                "evaluator_sha256": SHA,
                "runtime_code_tree_sha256": SHA,
            },
        )
    monkeypatch.setattr(
        contract,
        "verify_lock",
        lambda _path: {
            "checkpoints": [
                {"role": "producer", "sha256": "sha256:" + "1" * 64},
                {"role": "history", "sha256": "sha256:" + "2" * 64},
                {"role": "hard_negative", "sha256": "sha256:" + "3" * 64},
            ]
        },
    )
    out = tmp_path / "out"
    receipt = finalizer.finalize(plan_path, current_root, recovery_root, out)
    assert receipt["production_eligible"] is False
    selected = json.loads((out / "n128.selected_games.json").read_text())
    assert selected["selected_game_count"] == 84
    assert selected["category_game_counts"] == finalizer.QUOTAS["n128"]
    audit = json.loads((out / "n128.audit.json").read_text())
    assert audit["passed"] is True and audit["classification"] == finalizer.LABEL
    assert audit["rows"] == 168
    validation = json.loads((out / "n128.validation_seeds.json").read_text())
    assert validation["validation_row_count"] > 0
    monkeypatch.setitem(
        corpus.DUAL_ARM_SUBSET_CATEGORY_COUNTS,
        ("n128", "full-140k"),
        finalizer.QUOTAS["n128"],
    )
    loaded_selection = corpus._load_a1_selected_game_manifest(  # noqa: SLF001
        out / "n128.selected_games.json"
    )
    loaded_audit = corpus._load_a1_post_wave_audit(  # noqa: SLF001
        out / "n128.audit.json", loaded_selection
    )
    assert loaded_audit["arm_id"] == "n128"
    assert loaded_audit["selected_row_count"] == 168
    assert loaded_audit["harvest_relocation"]["arm_id"] == "n128"


def test_v3_attestation_rejects_tampered_realized_search_identity(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    producer = "sha256:" + "1" * 64
    opponent = "sha256:" + "2" * 64
    _write_json(
        root / "a1_contract.json",
        {
            "schema_version": "a1-generation-job-attestation-v3",
            "contract_sha256": SHA,
            "job_id": "lane__recent_history",
            "worker_id": "lane",
            "category": "recent_history",
            "arm_id": "n128",
            "games": 1,
            "producer_checkpoint_sha256": producer,
            "opponent_checkpoint_sha256": [opponent],
            "search_operator_sha256": "sha256:" + "3" * 64,
            "effective_search_config_sha256": "sha256:" + "4" * 64,
            "evaluator_sha256": SHA,
            "runtime_code_tree_sha256": SHA,
        },
    )
    job = {
        "job_id": "lane__recent_history",
        "worker_id": "lane",
        "category": "recent_history",
        "arm_id": "n128",
        "producer_checkpoint_sha256": producer,
        "opponent_checkpoint_sha256": [opponent],
    }
    with pytest.raises(finalizer.FinalizerError, match="realized search identity drift"):
        finalizer._verify_job_attestation(  # noqa: SLF001
            root,
            job,
            contract_sha256=SHA,
            selected_quota=1,
            expected_search_operator_sha256="sha256:" + "5" * 64,
            expected_effective_search_config_sha256="sha256:" + "4" * 64,
        )
