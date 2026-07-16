from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytest

from tools import cluster_big_push


def test_parser_defaults_leave_legacy_live_controller_unarmed() -> None:
    args = cluster_big_push.build_parser().parse_args([])

    assert args.dry_run is False
    assert args.acknowledge_legacy_gcp_big_push is False


def test_default_live_run_refuses_without_explicit_acknowledgement() -> None:
    args = cluster_big_push.build_parser().parse_args([])

    with pytest.raises(SystemExit, match="may automatically launch stale GCP"):
        cluster_big_push._refuse_unacknowledged_legacy_live_run(args)


def test_dry_run_remains_available_without_legacy_acknowledgement() -> None:
    args = cluster_big_push.build_parser().parse_args(["--dry-run"])

    cluster_big_push._refuse_unacknowledged_legacy_live_run(args)


def test_explicit_acknowledgement_arms_intentional_legacy_live_run() -> None:
    args = cluster_big_push.build_parser().parse_args(
        [cluster_big_push.LEGACY_LIVE_ACK_FLAG]
    )

    cluster_big_push._refuse_unacknowledged_legacy_live_run(args)


def test_main_refuses_before_creating_run_directory_or_running_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cluster_big_push"
    monkeypatch.setattr(cluster_big_push, "RUN_DIR", run_dir)
    monkeypatch.setattr(sys, "argv", ["cluster_big_push.py"])
    monkeypatch.setattr(
        cluster_big_push,
        "run_cycle",
        lambda *_args, **_kwargs: pytest.fail("legacy cycle ran before refusal"),
    )

    with pytest.raises(SystemExit, match="tools/fleet"):
        cluster_big_push.main()

    assert not run_dir.exists()


def test_main_preserves_unacknowledged_dry_run_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cluster_big_push"
    calls = []
    monkeypatch.setattr(cluster_big_push, "RUN_DIR", run_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["cluster_big_push.py", "--dry-run", "--no-refresh"],
    )

    def fake_run_cycle(args, *, cycle_index):
        calls.append((args.dry_run, args.no_refresh, cycle_index))
        return {"dry_run": True}

    monkeypatch.setattr(cluster_big_push, "run_cycle", fake_run_cycle)

    cluster_big_push.main()

    assert run_dir.is_dir()
    assert calls == [(True, True, 0)]


def test_direct_run_cycle_refuses_before_refresh_or_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        dry_run=False,
        acknowledge_legacy_gcp_big_push=False,
        no_refresh=False,
    )
    monkeypatch.setattr(
        cluster_big_push,
        "refresh_state",
        lambda _args: pytest.fail("remote refresh ran before refusal"),
    )

    with pytest.raises(SystemExit, match=cluster_big_push.LEGACY_LIVE_ACK_FLAG):
        cluster_big_push.run_cycle(args, cycle_index=0)
