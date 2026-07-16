from __future__ import annotations

import sys

import pytest

from tools import ssh_gpu_cluster


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "hostname"],
        ["sync"],
        ["sync", "--delete"],
        ["setup"],
        ["launch-train"],
    ],
)
def test_live_retired_fleet_actions_refuse_without_acknowledgement(
    argv: list[str],
) -> None:
    args = ssh_gpu_cluster.build_parser().parse_args(argv)

    with pytest.raises(SystemExit, match="retired JSON-fleet utility"):
        ssh_gpu_cluster._refuse_live_retired_fleet_action(args)


@pytest.mark.parametrize(
    "argv",
    [
        ["inventory"],
        ["status"],
        ["sync", "--dry-run"],
        ["setup", "--dry-run"],
        ["launch-train", "--dry-run"],
    ],
)
def test_read_only_and_dry_run_actions_remain_available(argv: list[str]) -> None:
    args = ssh_gpu_cluster.build_parser().parse_args(argv)

    ssh_gpu_cluster._refuse_live_retired_fleet_action(args)


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "hostname"],
        ["sync"],
        ["setup"],
        ["launch-train"],
    ],
)
def test_explicit_retired_fleet_acknowledgement_allows_live_action(
    argv: list[str],
) -> None:
    args = ssh_gpu_cluster.build_parser().parse_args(
        [ssh_gpu_cluster.RETIRED_FLEET_ACK_FLAG, *argv]
    )

    ssh_gpu_cluster._refuse_live_retired_fleet_action(args)


def test_main_refuses_before_loading_stale_host_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["ssh_gpu_cluster.py", "launch-train"])
    monkeypatch.setattr(
        ssh_gpu_cluster,
        "load_config",
        lambda _path: pytest.fail("stale host config was loaded before refusal"),
    )

    with pytest.raises(SystemExit, match="tools/fleet"):
        ssh_gpu_cluster.main()
