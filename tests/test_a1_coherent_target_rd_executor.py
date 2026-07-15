from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from tools.fleet import a1_coherent_target_rd_executor as executor


def test_idle_mps_server_is_the_only_compute_process_exempted() -> None:
    raw = "\n".join(
        (
            "100, nvidia-cuda-mps-server, 80",
            "101, /usr/bin/nvidia-cuda-mps-server, 80",
            "102, python3, 22000",
            "103, nvidia-cuda-mps-server-helper, 80",
            "malformed-row",
        )
    )

    assert executor._non_mps_compute_processes(raw) == [
        "102, python3, 22000",
        "103, nvidia-cuda-mps-server-helper, 80",
        "malformed-row",
    ]


def test_progress_snapshot_aggregates_workers_and_marks_missing_stale(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker_000"
    worker.mkdir()
    (worker / "progress.json").write_text(
        json.dumps(
            {
                "games_requested": 3,
                "game_index_start": 0,
                "base_seed": 700,
                "games_succeeded": 2,
                "games_failed": 0,
                "games_truncated": 0,
                "rows_confirmed": 17,
                "simulations_used_total": 128,
                "confirmed_shards": [{"index": 0}],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    now = dt.datetime.now(dt.timezone.utc)

    snapshot = executor._progress_snapshot(
        tmp_path,
        lane={"games": 5, "base_seed": 700},
        workers=2,
        observed_at=now,
        launched_at=now - dt.timedelta(minutes=20),
        stale_seconds=900.0,
    )

    assert snapshot["workers"][0]["state"] == "running"
    assert snapshot["workers"][1]["state"] == "missing_stale"
    assert snapshot["workers"][1]["expected_games"] == 2
    assert snapshot["workers"][1]["game_index_start"] == 3
    assert snapshot["totals"] == {
        "games_completed": 2,
        "games_failed": 0,
        "games_truncated": 0,
        "rows": 17,
        "simulations_used_total": 128,
    }


def _coherent_shard(path: Path, *, corrupt_visit_sum: bool = False) -> None:
    visits = np.asarray([5, 3, 6 if corrupt_visit_sum else 7], dtype=np.uint16)
    np.savez_compressed(
        path,
        game_seed=np.asarray([901, 901, 901], dtype=np.uint64),
        decision_index=np.asarray([0, 1, 2], dtype=np.int32),
        seat=np.asarray([0, 1, 0], dtype=np.int8),
        terminated=np.asarray([False, False, True]),
        truncated=np.asarray([False, False, False]),
        policy_weight_multiplier=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        target_information_regime=np.asarray(
            ["coherent", "coherent", "coherent"], dtype="<U8"
        ),
        legal_action_mask=np.asarray(
            [[True, True, False], [True, False, False], [False, True, False]]
        ),
        simulations_used=np.asarray([8, 0, 7], dtype=np.uint16),
        search_evidence_version=np.asarray(1, dtype=np.uint8),
        search_evidence_offsets=np.asarray([0, 2, 3], dtype=np.uint32),
        search_visit_counts_flat=visits,
        search_completed_q_flat=np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
    )


def test_shard_closure_authenticates_search_evidence_and_full_trajectory(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shard.npz"
    _coherent_shard(shard)
    trace = {
        "seen": set(),
        "current_seed": None,
        "last_decision": None,
        "current_complete": False,
    }

    result = executor._verify_shard_arrays(
        shard,
        contract={"target_information_regime": "coherent"},
        trace=trace,
    )

    assert result == {"rows": 3, "policy_active_rows": 2}
    assert trace["seen"] == {901}
    assert trace["current_complete"] is True


def test_shard_closure_rejects_search_visit_sum_drift(tmp_path: Path) -> None:
    shard = tmp_path / "shard.npz"
    _coherent_shard(shard, corrupt_visit_sum=True)

    with pytest.raises(executor.ExecutorError, match="visit sum mismatch"):
        executor._verify_shard_arrays(
            shard,
            contract={"target_information_regime": "coherent"},
            trace={
                "seen": set(),
                "current_seed": None,
                "last_decision": None,
                "current_complete": False,
            },
        )


def test_wait_polls_existing_launch_then_collects_without_execute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshots = iter(
        (
            {
                "state": "running",
                "observed_at": "2026-07-15T00:00:00+00:00",
                "totals": {"games_completed": 5, "games_requested": 8, "rows": 100},
                "failed_lanes": [],
                "stale_lanes": [],
            },
            {
                "state": "complete_uncollected",
                "observed_at": "2026-07-15T00:00:01+00:00",
                "totals": {"games_completed": 8, "games_requested": 8, "rows": 160},
                "failed_lanes": [],
                "stale_lanes": [],
            },
        )
    )
    monkeypatch.setattr(executor, "status", lambda *_args, **_kwargs: next(snapshots))
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        executor,
        "collect",
        lambda *_args, **_kwargs: {"status": "complete", "games": 8},
    )
    monkeypatch.setattr(
        executor,
        "execute",
        lambda *_args, **_kwargs: pytest.fail("wait must never launch work"),
    )

    assert executor.wait_for_completion(
        tmp_path / "contract.json",
        host_address="b200",
        poll_seconds=0.01,
    ) == {"status": "complete", "games": 8}


def test_cli_preserves_launch_and_adds_lifecycle_modes() -> None:
    parser = executor.build_parser()
    launch = parser.parse_args(
        ["--host-address", "b200", "--python", "/venv/bin/python", "--go"]
    )
    lifecycle = parser.parse_args(["--host-address", "b200", "--status"])

    assert launch.go is True
    assert launch.python == Path("/venv/bin/python")
    assert lifecycle.status is True
    assert lifecycle.python is None
