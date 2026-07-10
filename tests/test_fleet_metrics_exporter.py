from __future__ import annotations

import json
from pathlib import Path

from tools.fleet import fleet_metrics_exporter as exporter


def _write_json(path: Path, payload: object, *, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.touch()
    import os

    os.utime(path, (mtime, mtime))


def _config(base_seed: int = 1000, games: int = 20, n_full: int = 128) -> dict:
    return {
        "pipeline": "generate",
        "schema_version": 4,
        "fields": {
            "base_seed": base_seed,
            "games": games,
            "n_full": n_full,
            "p_full": 0.25,
        },
    }


def test_live_progress_exports_process_health_config_seed_and_counters(
    tmp_path: Path,
) -> None:
    now = 10_000.0
    root = tmp_path / "gen_out"
    gpu = root / "claim-a" / "gpu2"
    config = _config()
    _write_json(gpu / "config.json", config, mtime=now - 10)
    _write_json(
        gpu / "worker_000" / "progress.json",
        {
            "games_requested": 10,
            "games_completed_local": 4,
            "rows": 1200,
            "simulations_used_total": 88_000,
            "shard_count_confirmed": 2,
            "games_failed": 0,
            "games_truncated": 1,
        },
        mtime=now - 5,
    )
    _write_json(
        gpu / "worker_001" / "progress.json",
        {
            "games_requested": 10,
            "games_completed_local": 3,
            "rows": 900,
            "simulations_used_total": 70_000,
            "shard_count_confirmed": 1,
            "games_failed": 0,
            "games_truncated": 0,
        },
        mtime=now - 4,
    )
    processes = {str(gpu.resolve()): {41}}

    snapshots = exporter.collect_snapshots(
        [root],
        host="c1",
        processes=processes,
        now=now,
        stale_after_seconds=60,
        max_run_age_seconds=3600,
    )
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.gpu == "2"
    assert snapshot.pipeline == "0"
    assert snapshot.alias == "c1"
    assert snapshot.category == "legacy"
    assert snapshot.config_hash == exporter._config_hash(config)
    assert snapshot.seed_start == 1000
    assert snapshot.seed_end == 1020
    assert snapshot.games_completed == 7
    assert snapshot.rows == 2100
    assert snapshot.simulations == 158_000
    assert snapshot.shards == 3
    assert snapshot.truncations == 1
    assert snapshot.process_count == 1
    assert snapshot.healthy is True

    metrics = exporter.render_metrics(
        snapshots, host="c1", roots=[], now=now
    )
    assert 'config_hash="' + exporter._config_hash(config) + '"' in metrics
    assert 'seed_range="[1000,1020)"' in metrics
    assert "catan_fleet_generator_rows{" in metrics and "} 2100" in metrics
    assert "catan_fleet_generator_simulations{" in metrics and "} 158000" in metrics
    assert "catan_fleet_generator_healthy{" in metrics and "} 1" in metrics


def test_sealed_a1_layout_discovers_every_alias_and_category_with_labels(
    tmp_path: Path,
) -> None:
    now = 15_000.0
    root = tmp_path / "gen_out"
    run = root / "a1-fresh-mixed-12000games"
    aliases = ("c1", "c2", "c3", "c4", "c5", "c6", "h100-8a", "h100-8b")
    category_rows = {
        "current_producer": 100,
        "recent_history": 10,
        "hard_negative": 1,
    }
    processes: dict[str, set[int]] = {}
    for alias_index, alias in enumerate(aliases):
        for category_index, (category, rows) in enumerate(category_rows.items()):
            output = run / f"{alias}_gpu0__{category}"
            _write_json(
                output / "config.json",
                _config(base_seed=10_000 + alias_index * 100 + category_index),
                mtime=now - 2,
            )
            _write_json(
                output / "worker_000" / "progress.json",
                {
                    "games_requested": 20,
                    "games_completed_local": 1,
                    "rows": rows,
                    "simulations_used_total": rows * 128,
                    "shard_count_confirmed": 1,
                    "games_failed": 0,
                    "games_truncated": 0,
                },
                mtime=now - 1,
            )
            processes[str(output.resolve())] = {
                1000 + alias_index * len(category_rows) + category_index
            }

    snapshots = []
    for alias in aliases:
        for category in category_rows:
            output = run / f"{alias}_gpu0__{category}"
            snapshot = exporter.snapshot_run(
                output,
                host="fleet-test",
                processes=processes,
                now=now,
                stale_after_seconds=60,
            )
            assert snapshot is not None
            snapshots.append(snapshot)

    assert len(snapshots) == len(aliases) * len(category_rows)
    assert {(item.alias, item.category) for item in snapshots} == {
        (alias, category) for alias in aliases for category in category_rows
    }
    assert {
        category: sum(item.rows for item in snapshots if item.category == category)
        for category in category_rows
    } == {category: rows * len(aliases) for category, rows in category_rows.items()}
    assert all(item.run == "a1-fresh-mixed-12000games" for item in snapshots)
    assert all(item.gpu == "0" and item.pipeline == "0" for item in snapshots)
    assert all(item.role == "teacher" and item.healthy for item in snapshots)

    metrics = exporter.render_metrics(
        snapshots, host="fleet-test", roots=[], now=now
    )
    for alias in aliases:
        for category in category_rows:
            assert f'alias="{alias}",category="{category}"' in metrics
    assert 'run="a1-fresh-mixed-12000games"' in metrics


def test_sealed_a1_categories_share_physical_gpu_slot_arbitration(
    tmp_path: Path,
) -> None:
    now = 16_000.0
    root = tmp_path / "gen_out"
    run = root / "a1-fresh-mixed-12000games"
    current = run / "c1_gpu2__current_producer"
    recent = run / "c1_gpu2__recent_history"
    hard = run / "c1_gpu2__hard_negative"
    for index, output in enumerate((current, recent, hard)):
        _write_json(
            output / "config.json",
            _config(base_seed=20_000 + index),
            mtime=now - index,
        )

    snapshots = exporter.collect_snapshots(
        [root],
        host="c1",
        processes={str(recent.resolve()): {123}},
        now=now,
        stale_after_seconds=60,
        max_run_age_seconds=3600,
    )

    assert len(snapshots) == 1
    assert snapshots[0].output_dir == recent
    assert snapshots[0].alias == "c1"
    assert snapshots[0].category == "recent_history"


def test_stale_active_progress_is_unhealthy(tmp_path: Path) -> None:
    now = 20_000.0
    gpu = tmp_path / "runs" / "claim" / "gpu0"
    _write_json(gpu / "config.json", _config(), mtime=now - 500)
    _write_json(
        gpu / "worker_000" / "progress.json",
        {
            "games_requested": 20,
            "games_completed_local": 2,
            "rows": 3,
            "simulations_used_total": 4,
            "shard_count_confirmed": 0,
            "games_failed": 0,
            "games_truncated": 0,
        },
        mtime=now - 400,
    )
    snapshot = exporter.snapshot_run(
        gpu,
        host="c1",
        processes={str(gpu.resolve()): {7}},
        now=now,
        stale_after_seconds=300,
    )
    assert snapshot is not None
    assert snapshot.stale_seconds == 400
    assert snapshot.process_count == 1
    assert snapshot.healthy is False


def test_clean_completed_manifest_is_healthy_without_process(tmp_path: Path) -> None:
    now = 30_000.0
    gpu = tmp_path / "runs" / "claim" / "gpu3"
    _write_json(gpu / "config.json", _config(), mtime=now - 100)
    _write_json(
        gpu / "manifest.json",
        {
            "config_hash": "sha256:0123456789abcdef",
            "base_seed": 1000,
            "games_requested": 20,
            "games_completed": 20,
            "games_failed": 0,
            "games_truncated": 0,
            "rows": 5000,
            "simulations_used_total": 90000,
            "shards": ["a", "b"],
            "errors": [],
            "n_full": 128,
            "p_full": 0.25,
        },
        mtime=now - 80,
    )
    snapshot = exporter.snapshot_run(
        gpu, host="c1", processes={}, now=now, stale_after_seconds=30
    )
    assert snapshot is not None
    assert snapshot.complete is True
    assert snapshot.healthy is True
    assert snapshot.process_count == 0


def test_failed_or_fatal_manifest_is_never_complete_or_healthy(tmp_path: Path) -> None:
    now = 31_000.0
    gpu = tmp_path / "runs" / "claim" / "gpu3"
    _write_json(gpu / "config.json", _config(), mtime=now - 10)
    _write_json(
        gpu / "manifest.json",
        {
            "games_requested": 20,
            "games_completed": 20,
            "games_failed": 1,
            "errors": [],
            "fatal_execution_error": {
                "type": "RuntimeError",
                "message": "server died",
            },
            "n_full": 128,
        },
        mtime=now - 5,
    )
    snapshot = exporter.snapshot_run(
        gpu, host="c1", processes={}, now=now, stale_after_seconds=300
    )
    assert snapshot is not None
    assert snapshot.complete is False
    assert snapshot.healthy is False


def test_discover_generators_reads_out_dir_from_proc_cmdline(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cmdline = proc / "123" / "cmdline"
    cmdline.parent.mkdir(parents=True)
    out = tmp_path / "gen" / "gpu0"
    cmdline.write_bytes(
        b"python\0tools/generate_gumbel_selfplay_data.py\0--out-dir\0"
        + str(out).encode()
        + b"\0"
    )
    (proc / "not-a-pid").mkdir()

    assert exporter.discover_generators(proc) == {str(out.resolve()): {123}}


def test_only_latest_or_active_run_per_gpu_is_exposed(tmp_path: Path) -> None:
    now = 40_000.0
    root = tmp_path / "runs"
    old = root / "old" / "gpu0"
    active = root / "active" / "gpu0"
    _write_json(old / "config.json", _config(base_seed=1), mtime=now - 20)
    _write_json(active / "config.json", _config(base_seed=100), mtime=now - 100)
    snapshots = exporter.collect_snapshots(
        [root],
        host="c1",
        processes={str(active.resolve()): {9}},
        now=now,
        stale_after_seconds=300,
        max_run_age_seconds=1000,
    )
    assert [snapshot.run for snapshot in snapshots] == ["active"]


def test_dual_pipeline_directories_are_both_exposed(tmp_path: Path) -> None:
    now = 45_000.0
    root = tmp_path / "runs"
    first = root / "dual" / "gpu2_pipeline0"
    second = root / "dual" / "gpu2_pipeline1"
    _write_json(first / "config.json", _config(base_seed=100), mtime=now - 2)
    _write_json(second / "config.json", _config(base_seed=110), mtime=now - 1)

    snapshots = exporter.collect_snapshots(
        [root],
        host="c1",
        processes={str(first.resolve()): {10}, str(second.resolve()): {11}},
        now=now,
        stale_after_seconds=300,
        max_run_age_seconds=1000,
    )
    assert [(item.gpu, item.pipeline) for item in snapshots] == [
        ("2", "0"),
        ("2", "1"),
    ]
    metrics = exporter.render_metrics(snapshots, host="c1", roots=[], now=now)
    assert 'pipeline="0"' in metrics
    assert 'pipeline="1"' in metrics


def test_launcher_dumps_typed_config_before_generation() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "tools/fleet/fleet_launch.sh"
    ).read_text(encoding="utf-8")
    assert '--dump-config "$PIPELINE_OUT/config.json"' in source
    assert '--config-purpose "fleet-$PIPELINE_ID"' in source
