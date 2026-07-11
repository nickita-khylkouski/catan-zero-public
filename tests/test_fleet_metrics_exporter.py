from __future__ import annotations

import json
from pathlib import Path

import pytest

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
            "public_observation": True,
            "information_set_search": True,
            "determinization_particles": 4,
            "determinization_min_simulations": 32,
            "n_full": n_full,
            "n_fast": 16,
            "p_full": 0.25,
            "symmetry_averaged_eval": True,
            "symmetry_averaged_eval_threshold": 20,
            "c_scale": 0.03,
            "c_visit": 50.0,
            "max_depth": 80,
            "lazy_interior_chance": True,
            "belief_chance_spectra": False,
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
    assert snapshot.recipe_safe is True
    assert snapshot.target_information_regime == "public_conservation_pimc_v1"
    assert snapshot.target_information_regime_attested is False

    metrics = exporter.render_metrics(snapshots, host="c1", roots=[], now=now)
    assert 'config_hash="' + exporter._config_hash(config) + '"' in metrics
    assert 'seed_range="[1000,1020)"' in metrics
    assert "catan_fleet_generator_rows{" in metrics and "} 2100" in metrics
    assert "catan_fleet_generator_simulations{" in metrics and "} 158000" in metrics
    assert "catan_fleet_generator_healthy{" in metrics and "} 1" in metrics
    assert "catan_fleet_generator_recipe_safe{" in metrics and "} 1" in metrics
    assert 'information_set_search="true"' in metrics
    assert 'determinization_particles="4"' in metrics
    assert 'target_information_regime="public_conservation_pimc_v1"' in metrics
    assert 'catan_fleet_generator_lanes_active_total{host="c1"} 1' in metrics
    assert 'catan_fleet_generator_lanes_recipe_safe_total{host="c1"} 1' in metrics


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

    metrics = exporter.render_metrics(snapshots, host="fleet-test", roots=[], now=now)
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


@pytest.mark.parametrize(
    ("arm", "category", "c_scale"),
    [
        ("n128", "current_producer", 0.1),
        ("n128", "recent_history", 0.03),
        ("n256", "current_producer", 0.1),
        ("n256", "hard_negative", 0.03),
    ],
)
def test_nested_dual_arm_layout_is_discovered_with_category_recipe(
    tmp_path: Path, arm: str, category: str, c_scale: float
) -> None:
    now = 16_500.0
    root = tmp_path / "runs" / "selfplay"
    output = root / "campaign-r1" / arm / f"{arm}_gpu00__{category}"
    config = _config(n_full=int(arm.removeprefix("n")))
    config["fields"]["c_scale"] = c_scale
    _write_json(output / "config.json", config, mtime=now - 2)
    _write_json(
        output / "worker_000" / "progress.json",
        {
            "games_requested": 20,
            "games_completed_local": 1,
            "rows": 100,
            "simulations_used_total": 1000,
            "shard_count_confirmed": 1,
            "games_failed": 0,
            "games_truncated": 0,
        },
        mtime=now - 1,
    )

    snapshots = exporter.collect_snapshots(
        [root],
        host="fleet-host",
        processes={str(output.resolve()): {123}},
        now=now,
        stale_after_seconds=60,
        max_run_age_seconds=3600,
    )

    assert len(snapshots) == 1
    assert snapshots[0].run == arm
    assert snapshots[0].category == category
    assert snapshots[0].n_full == int(arm.removeprefix("n"))
    assert snapshots[0].c_scale == c_scale
    assert snapshots[0].recipe_safe is True


def test_nested_dual_arm_recipe_rejects_cross_arm_budget(tmp_path: Path) -> None:
    now = 16_600.0
    root = tmp_path / "runs" / "selfplay"
    output = root / "campaign-r1" / "n256" / "n256_gpu00__current_producer"
    config = _config(n_full=128)
    config["fields"]["c_scale"] = 0.1
    _write_json(output / "config.json", config, mtime=now - 1)

    snapshots = exporter.collect_snapshots(
        [root],
        host="fleet-host",
        processes={str(output.resolve()): {123}},
        now=now,
        stale_after_seconds=60,
        max_run_age_seconds=3600,
    )

    assert len(snapshots) == 1
    assert snapshots[0].recipe_safe is False


def test_a1_contract_and_live_argv_supply_typed_attestation_labels(
    tmp_path: Path,
) -> None:
    now = 17_000.0
    root = tmp_path / "gen_out"
    output = root / "a1-fresh-mixed-12000games" / "c1_gpu0__current_producer"
    contract_hash = "sha256:" + "a" * 64
    _write_json(
        output / "a1_contract.json",
        {
            "schema_version": "a1-generation-job-attestation-v2",
            "base_seed": 300_000_000_000,
            "seed_end": 300_000_000_245,
            "games": 240,
            "attempts": 245,
            "effective_search_config_sha256": contract_hash,
        },
        mtime=now - 2,
    )
    _write_json(
        output / "worker_000" / "progress.json",
        {
            "games_requested": 245,
            "games_completed_local": 2,
            "rows": 900,
            "simulations_used_total": 12_000,
            "shard_count_confirmed": 1,
            "games_failed": 0,
            "games_truncated": 0,
        },
        mtime=now - 1,
    )
    resolved = str(output.resolve())
    argv = (
        "python",
        "tools/generate_gumbel_selfplay_data.py",
        "--out-dir",
        resolved,
        "--n-full",
        "128",
        "--n-fast",
        "16",
        "--p-full",
        "0.25",
        "--public-observation",
        "--information-set-search",
        "--determinization-particles",
        "4",
        "--determinization-min-simulations",
        "32",
        "--symmetry-averaged-eval",
        "--symmetry-averaged-eval-threshold",
        "20",
        "--c-scale",
        "0.03",
        "--c-visit",
        "50",
        "--max-depth",
        "80",
        "--lazy-interior-chance",
        "--no-belief-chance-spectra",
    )

    snapshot = exporter.snapshot_run(
        output,
        host="fleet-host",
        processes={resolved: {123}},
        generator_argv={resolved: argv},
        now=now,
        stale_after_seconds=60,
    )

    assert snapshot is not None
    assert snapshot.config_hash == contract_hash
    assert snapshot.n_full == 128
    assert snapshot.p_full == 0.25
    assert snapshot.role == "teacher"
    assert snapshot.seed_start == 300_000_000_000
    assert snapshot.seed_end == 300_000_000_245
    assert snapshot.games_requested == 245
    assert snapshot.recipe_safe is True
    assert snapshot.target_information_regime == "public_conservation_pimc_v1"
    assert snapshot.target_information_regime_attested is False


def test_recipe_mismatch_and_attested_unsafe_regime_are_exported(
    tmp_path: Path,
) -> None:
    now = 18_000.0
    root = tmp_path / "gen_out"
    mismatch = root / "a1" / "gpu0"
    config = _config()
    config["fields"]["p_full"] = 0.4
    _write_json(mismatch / "config.json", config, mtime=now - 2)
    resolved = str(mismatch.resolve())
    snapshot = exporter.snapshot_run(
        mismatch,
        host="c1",
        processes={resolved: {10}},
        now=now,
        stale_after_seconds=60,
    )
    assert snapshot is not None
    assert snapshot.p_full == 0.4
    assert snapshot.recipe_safe is False

    _write_json(
        mismatch / "manifest.json",
        {
            "games_requested": 20,
            "games_completed": 1,
            "games_failed": 0,
            "target_information_regime": "authoritative_hidden_state_search_v1",
            "cli_args": config["fields"],
        },
        mtime=now - 1,
    )
    attested = exporter.snapshot_run(
        mismatch,
        host="c1",
        processes={resolved: {10}},
        now=now,
        stale_after_seconds=60,
    )
    assert attested is not None
    assert attested.target_information_regime_attested is True
    assert attested.target_information_regime == "authoritative_hidden_state_search_v1"
    assert attested.recipe_safe is False
    metrics = exporter.render_metrics([attested], host="c1", roots=[], now=now)
    assert "catan_fleet_generator_target_information_regime_attested{" in metrics
    assert 'target_information_regime="authoritative_hidden_state_search_v1"' in metrics
    assert 'catan_fleet_generator_lanes_recipe_safe_total{host="c1"} 0' in metrics


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("public_observation", False),
        ("information_set_search", False),
        ("determinization_particles", 2),
        ("determinization_min_simulations", 16),
        ("n_full", 256),
        ("n_fast", 8),
        ("p_full", 0.4),
        ("symmetry_averaged_eval", False),
        ("symmetry_averaged_eval_threshold", 16),
        ("c_scale", 0.1),
        ("c_visit", 25.0),
        ("max_depth", 64),
        ("lazy_interior_chance", False),
        ("belief_chance_spectra", True),
        ("target_information_regime", "authoritative_hidden_state_search_v1"),
    ],
)
def test_every_exact_recipe_field_is_safety_binding(
    tmp_path: Path, field: str, drifted: object
) -> None:
    now = 19_000.0
    output = tmp_path / "gen_out" / field / "gpu0"
    config = _config()
    config["fields"][field] = drifted
    _write_json(output / "config.json", config, mtime=now - 1)
    resolved = str(output.resolve())
    snapshot = exporter.snapshot_run(
        output,
        host="c1",
        processes={resolved: {11}},
        now=now,
        stale_after_seconds=60,
    )
    assert snapshot is not None
    assert snapshot.recipe_safe is False, field


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
    assert exporter.discover_generator_argv(proc) == {
        str(out.resolve()): (
            "python",
            "tools/generate_gumbel_selfplay_data.py",
            "--out-dir",
            str(out),
        )
    }


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
