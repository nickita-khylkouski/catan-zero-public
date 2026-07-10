from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest

# `tools/generate_gumbel_selfplay_data.py` does bare sibling imports
# (`from factory_common import ...`), so it only works with `tools/` itself on
# sys.path (matches the bootstrap pattern in tests/test_gumbel_self_play.py).
_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_gumbel_selfplay_data as cli  # type: ignore  # noqa: E402
from catan_zero.rl.pipeline_configs import GenerateConfig  # noqa: E402


def _base_args(**overrides):
    values = {
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "games": 4,
        "n_full": 4,
        "n_fast": 2,
        "p_full": 1.0,
        "checkpoint": None,
        "base_seed": 1,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": False,
        "exact_budget_sh": False,
        "exact_budget_sh_min_n": 0,
        "rust_featurize": False,
    }
    values.update(overrides)

    class _Args:
        pass

    args = _Args()
    for key, value in values.items():
        setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# _worker_entry must never let an exception escape to pool.map -- one worker
# crashing (OOM, bad checkpoint, etc.) must not abort every other worker's
# already-completed results.
# ---------------------------------------------------------------------------


def test_worker_entry_catches_a_fatal_crash_and_returns_an_error_summary(monkeypatch):
    def _boom(worker_args):
        raise RuntimeError("synthetic fatal worker crash")

    monkeypatch.setattr(cli, "_run_worker", _boom)

    worker_args = {
        "worker_index": 2,
        "games": 3,
        "out_dir": "/tmp/does-not-matter",
    }

    result = cli._worker_entry(worker_args)

    assert result["worker_index"] == 2
    assert result["games_completed"] == 0
    assert result["games_failed"] == 3
    assert result["shards"] == []
    assert result["errors"]
    assert "synthetic fatal worker crash" in result["errors"][0]["error"]


def test_dual_pipeline_topology_is_validated_and_recorded(tmp_path):
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "--fleet-pipelines-per-gpu",
            "2",
            "--fleet-pipeline-index",
            "1",
            "--fleet-pipeline-id",
            "claim-gpu0-pipeline1",
        ]
    )
    summary = cli._merge_worker_summaries(
        [], out_dir=tmp_path, elapsed_sec=1.0, args=args
    )
    assert summary["fleet_pipelines_per_gpu"] == 2
    assert summary["fleet_pipeline_index"] == 1
    assert summary["fleet_pipeline_id"] == "claim-gpu0-pipeline1"


def test_dual_pipeline_index_must_fit_pipeline_count(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--skip-guards",
                "--out-dir",
                str(tmp_path),
                "--games",
                "0",
                "--fleet-pipeline-index",
                "1",
            ]
        )
    assert exc_info.value.code == 2
    assert "fleet-pipeline-index" in capsys.readouterr().err


def test_config_pipeline_count_is_applied_before_index_validation(tmp_path):
    out_dir = tmp_path / "out"
    config_path = tmp_path / "generate.json"
    config_path.write_text(
        json.dumps(
            GenerateConfig(
                games=0,
                fleet_pipelines_per_gpu=2,
            ).canonical_payload()
        )
    )

    cli.main(
        [
            "--skip-guards",
            "--out-dir",
            str(out_dir),
            "--config",
            str(config_path),
            "--fleet-pipeline-index",
            "1",
        ]
    )
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["fleet_pipelines_per_gpu"] == 2
    assert manifest["fleet_pipeline_index"] == 1


def test_config_is_applied_before_auto_shard_and_science_validation(tmp_path):
    out_dir = tmp_path / "out"
    config = GenerateConfig(games=0, n_full=128, shard_size=777, p_full=0.5)
    config_path = tmp_path / "generate.json"
    config_path.write_text(json.dumps(config.canonical_payload()))

    cli.main(
        [
            "--skip-guards",
            "--out-dir",
            str(out_dir),
            "--config",
            str(config_path),
        ]
    )

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["n_full"] == 128
    assert manifest["p_full"] == 0.5
    # A config-supplied shard size is pinned, not replaced by n=128 auto-size.
    assert manifest["cli_args"]["shard_size"] == 777


def test_config_filled_values_are_explicit_to_prelaunch_guard(tmp_path):
    parser = cli.build_parser()
    config = GenerateConfig(
        games=0,
        c_scale=0.03,
        temperature_decisions=90,
        public_observation=True,
        lazy_interior_chance=True,
    )
    config_path = tmp_path / "generate.json"
    config_path.write_text(json.dumps(config.canonical_payload()))
    raw_argv = ["--out-dir", str(tmp_path / "out"), "--config", str(config_path)]
    args = parser.parse_args(raw_argv)
    filled = cli.apply_config_file(
        args,
        parser,
        argv=raw_argv,
        expected_pipeline=GenerateConfig.PIPELINE,
    )

    effective_argv = cli._guard_argv_with_config_values(
        args, parser, raw_argv, filled
    )
    reparsed = parser.parse_args(effective_argv)

    assert "--c-scale" in effective_argv
    assert "--public-observation" in effective_argv
    assert "--lazy-interior-chance" in effective_argv
    assert reparsed.c_scale == 0.03
    assert reparsed.temperature_decisions == 90


def test_config_rejects_invalid_choice_before_guard_or_output(tmp_path, capsys):
    out_dir = tmp_path / "out"
    payload = GenerateConfig(games=0).canonical_payload()
    payload["fields"]["value_readout"] = "not-a-readout"
    config_path = tmp_path / "generate.json"
    config_path.write_text(json.dumps(payload))

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--skip-guards",
                "--out-dir",
                str(out_dir),
                "--config",
                str(config_path),
            ]
        )
    assert exc_info.value.code == 2
    assert "value-readout" in capsys.readouterr().err
    assert not out_dir.exists()


@pytest.mark.parametrize("bad_p_full", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_p_full_from_config_fails_before_output(tmp_path, bad_p_full, capsys):
    out_dir = tmp_path / "out"
    config = GenerateConfig(games=0, p_full=bad_p_full)
    config_path = tmp_path / "generate.json"
    config_path.write_text(json.dumps(config.canonical_payload()))

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--skip-guards",
                "--out-dir",
                str(out_dir),
                "--config",
                str(config_path),
            ]
        )
    assert exc_info.value.code == 2
    assert "p-full" in capsys.readouterr().err
    assert not out_dir.exists()


def test_manifest_and_config_hash_bind_checkpoint_bytes(tmp_path):
    out_dir = tmp_path / "out"
    checkpoint = tmp_path / "champion.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    expected_sha = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    cli.main(
        [
            "--skip-guards",
            "--out-dir",
            str(out_dir),
            "--games",
            "0",
            "--checkpoint",
            str(checkpoint),
        ]
    )

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["producer_checkpoint_sha256"] == expected_sha
    staged = Path(manifest["producer_checkpoint_staged_path"])
    assert staged.read_bytes() == b"checkpoint bytes"
    assert staged.stat().st_mode & 0o222 == 0
    expected_config = GenerateConfig(
        checkpoint=str(checkpoint), games=0, producer_checkpoint_sha256=expected_sha
    )
    assert manifest["config_hash"] == expected_config.config_hash()


def test_staged_checkpoint_closes_source_path_toctou(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    source = tmp_path / "champion.pt"
    source.write_bytes(b"bytes workers must load")

    staged_path, digest = cli._stage_producer_checkpoint(str(source), output)
    source.write_bytes(b"replacement after staging")

    assert staged_path is not None
    assert Path(staged_path).read_bytes() == b"bytes workers must load"
    assert digest == "sha256:" + hashlib.sha256(b"bytes workers must load").hexdigest()


def test_precreated_fleet_run_log_does_not_make_output_stale(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "run.log").touch()

    cli.main(
        [
            "--skip-guards",
            "--out-dir",
            str(out_dir),
            "--games",
            "0",
        ]
    )
    assert (out_dir / "manifest.json").is_file()


@pytest.mark.parametrize(
    ("completed", "failed"),
    [(1, 1), (1, 0)],
    ids=("failed-game", "missing-game"),
)
def test_partial_generation_writes_manifest_then_exits_nonzero(
    tmp_path, monkeypatch, completed, failed
):
    out_dir = tmp_path / "out"

    def _partial(worker_args):
        return {
            "worker_index": worker_args["worker_index"],
            "out_dir": worker_args["out_dir"],
            "games_completed": completed,
            "games_failed": failed,
            "games_truncated": 0,
            "rows": 0,
            "decisions_total": 0,
            "forced_decisions_total": 0,
            "simulations_used_total": 0,
            "wins_by_color": {},
            "shards": [],
            "errors": [],
        }

    monkeypatch.setattr(cli, "_worker_entry", _partial)
    with pytest.raises(SystemExit, match="generation incomplete"):
        cli.main(
            [
                "--skip-guards",
                "--no-seed-claim",
                "--out-dir",
                str(out_dir),
                "--games",
                "2",
            ]
        )

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["games_requested"] == 2
    assert manifest["games_completed"] == completed
    assert manifest["games_failed"] == failed


def test_eval_server_bootstrap_failure_preserves_top_manifest(
    tmp_path, monkeypatch
):
    out_dir = tmp_path / "out"
    checkpoint = tmp_path / "champion.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")

    def _fail_bootstrap(*_args, **_kwargs):
        raise RuntimeError("synthetic EvalServer startup failure")

    monkeypatch.setattr(cli, "_run_eval_server_batch", _fail_bootstrap)
    with pytest.raises(SystemExit, match="fatal=RuntimeError"):
        cli.main(
            [
                "--skip-guards",
                "--no-seed-claim",
                "--out-dir",
                str(out_dir),
                "--games",
                "1",
                "--checkpoint",
                str(checkpoint),
                "--eval-server",
            ]
        )

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["games_requested"] == 1
    assert manifest["games_completed"] == 0
    assert manifest["games_failed"] == 1
    assert manifest["fatal_execution_error"] == {
        "type": "RuntimeError",
        "message": "synthetic EvalServer startup failure",
    }


# ---------------------------------------------------------------------------
# _merge_worker_summaries must not reference a worker manifest.json path that
# was never written (a worker that crashed before/without writing one via
# run_worker_games's atomic write), and must still merge survivors' shards/
# games correctly. It must also drop any shard path that does not exist on
# disk (defensive, mirrors the same existence check already applied to
# `shards` in the current implementation).
# ---------------------------------------------------------------------------


def test_merge_worker_summaries_excludes_manifest_path_for_a_crashed_worker(tmp_path):
    good_out_dir = tmp_path / "worker_000"
    good_out_dir.mkdir()
    (good_out_dir / "manifest.json").write_text("{}", encoding="utf-8")
    present_shard = good_out_dir / "gumbel_self_play_shard_00000.npz"
    present_shard.write_bytes(b"fake")
    missing_shard = str(good_out_dir / "gumbel_self_play_shard_00001.npz")

    good_result = {
        "worker_index": 0,
        "out_dir": str(good_out_dir),
        "games_completed": 4,
        "games_failed": 0,
        "games_truncated": 0,
        "rows": 40,
        "decisions_total": 40,
        "forced_decisions_total": 4,
        "simulations_used_total": 400,
        "wins_by_color": {"RED": 2, "BLUE": 2},
        # One real shard plus one path that doesn't exist on disk -- the
        # merge must drop the latter.
        "shards": [str(present_shard), missing_shard],
        "errors": [],
    }
    # A crashed worker: _worker_entry's except-block summary. Its out_dir was
    # never created (crash happened before run_worker_games could write
    # anything), so no manifest.json exists there.
    crashed_result = {
        "worker_index": 1,
        "out_dir": str(tmp_path / "worker_001"),
        "games_completed": 0,
        "games_failed": 3,
        "games_truncated": 0,
        "rows": 0,
        "decisions_total": 0,
        "forced_decisions_total": 0,
        "simulations_used_total": 0,
        "wins_by_color": {},
        "shards": [],
        "errors": [
            {
                "worker_index": 1,
                "game_index": None,
                "game_seed": None,
                "error": "RuntimeError('synthetic fatal worker crash')",
            }
        ],
    }

    args = _base_args(games=7)
    summary = cli._merge_worker_summaries(
        [good_result, crashed_result], out_dir=tmp_path, elapsed_sec=1.0, args=args
    )

    assert summary["games_completed"] == 4
    assert summary["games_failed"] == 3
    assert summary["shards"] == [str(present_shard)]
    assert len(summary["errors"]) == 1
    assert "synthetic fatal worker crash" in summary["errors"][0]["error"]
    # The crashed worker's manifest.json was never written -- its path must
    # not appear in worker_summaries (this used to unconditionally list
    # every worker's out_dir/manifest.json regardless of whether it exists).
    assert (
        str(Path(crashed_result["out_dir"]) / "manifest.json")
        not in summary["worker_summaries"]
    )
    assert str(good_out_dir / "manifest.json") in summary["worker_summaries"]


# ---------------------------------------------------------------------------
# CAT-54 opponent-mix CLI wiring: --opponent-mix-manifest validation, and
# _merge_worker_summaries's per-tag aggregation.
# ---------------------------------------------------------------------------


def _minimal_self_only_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "mix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "categories": [
                    {"name": "producer_self_play", "weight": 1, "source": "self"}
                ]
            }
        )
    )
    return manifest_path


def test_opponent_mix_manifest_requires_checkpoint(tmp_path, monkeypatch):
    manifest_path = _minimal_self_only_manifest(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_gumbel_selfplay_data.py",
            "--skip-guards",
            "--out-dir",
            str(tmp_path / "out"),
            "--games",
            "0",
            "--opponent-mix-manifest",
            str(manifest_path),
        ],
    )
    with pytest.raises(SystemExit, match="requires --checkpoint"):
        cli.main()


def test_opponent_mix_manifest_mutually_exclusive_with_pool_manifest(
    tmp_path, monkeypatch
):
    manifest_path = _minimal_self_only_manifest(tmp_path)
    pool_manifest_path = tmp_path / "pool.json"
    pool_manifest_path.write_text(
        json.dumps({"opponents": [{"checkpoint": "/fake.pt", "version": 0}]})
    )
    fake_checkpoint = tmp_path / "champion.pt"
    fake_checkpoint.write_bytes(b"fake")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_gumbel_selfplay_data.py",
            "--skip-guards",
            "--out-dir",
            str(tmp_path / "out"),
            "--games",
            "0",
            "--checkpoint",
            str(fake_checkpoint),
            "--opponent-pool-manifest",
            str(pool_manifest_path),
            "--opponent-mix-manifest",
            str(manifest_path),
        ],
    )
    with pytest.raises(SystemExit, match="mutually exclusive"):
        cli.main()


def test_merge_worker_summaries_default_path_leaves_mix_disabled(tmp_path):
    """Regression: the default path (no --opponent-mix-manifest at all, the
    same `_base_args()` every pre-CAT-54 test in this file already uses) must
    keep reporting opponent_mix as fully disabled/empty -- CAT-54 is purely
    additive."""
    args = _base_args(games=1)
    summary = cli._merge_worker_summaries(
        [], out_dir=tmp_path, elapsed_sec=1.0, args=args
    )
    assert summary["opponent_mix_enabled"] is False
    assert summary["opponent_mix_manifest"] is None
    assert summary["opponent_mix_effective_weights"] == {}
    assert summary["opponent_mix_pool_games"] == 0
    assert summary["opponent_mix_pool_fraction_realized"] == 0.0
    assert summary["opponent_mix_tags_used"] == []
    assert summary["opponent_mix_per_tag_champion_winrate"] == {}


def test_merge_worker_summaries_aggregates_mix_per_tag_stats_across_workers(tmp_path):
    """Tag propagation + sum-then-divide aggregation: two workers each report
    raw (games, champion_wins) per tag; the merged summary must SUM the raw
    counts before dividing (not average each worker's own win-rate), exactly
    mirroring the existing opponent_pool_per_version_stats convention."""
    worker_a = {
        "worker_index": 0,
        "out_dir": str(tmp_path / "worker_000"),
        "games_completed": 10,
        "shards": [],
        "errors": [],
        "wins_by_color": {},
        "opponent_mix_pool_games": 3,
        "opponent_mix_per_tag_stats": {
            "hard_experimental": {"games": 3, "champion_wins": 1},
            "producer_self_play": {"games": 7, "champion_wins": 4},
        },
    }
    worker_b = {
        "worker_index": 1,
        "out_dir": str(tmp_path / "worker_001"),
        "games_completed": 10,
        "shards": [],
        "errors": [],
        "wins_by_color": {},
        "opponent_mix_pool_games": 2,
        "opponent_mix_per_tag_stats": {
            "hard_experimental": {"games": 2, "champion_wins": 2},
            "producer_self_play": {"games": 8, "champion_wins": 5},
        },
    }
    args = _base_args(games=20, opponent_mix_manifest="dummy_mix.json")
    summary = cli._merge_worker_summaries(
        [worker_a, worker_b], out_dir=tmp_path, elapsed_sec=1.0, args=args
    )

    assert summary["opponent_mix_enabled"] is True
    assert summary["opponent_mix_pool_games"] == 5
    assert summary["opponent_mix_pool_fraction_realized"] == pytest.approx(5 / 20)
    assert set(summary["opponent_mix_tags_used"]) == {
        "hard_experimental",
        "producer_self_play",
    }
    winrates = summary["opponent_mix_per_tag_champion_winrate"]
    # hard_experimental: (1+2) champion_wins over (3+2) games = 3/5
    assert winrates["hard_experimental"] == pytest.approx(3 / 5)
    # producer_self_play: (4+5) over (7+8) = 9/15
    assert winrates["producer_self_play"] == pytest.approx(9 / 15)


# ---------------------------------------------------------------------------
# CAT-56 exploiter lane CLI wiring: --exploiter-fraction resolve/scale/cap and
# _merge_worker_summaries's per-engine aggregation.
# ---------------------------------------------------------------------------


def _exploiter_manifest(tmp_path: Path, *, external_weight: float = 3.0) -> Path:
    manifest_path = tmp_path / "exploiter_mix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "categories": [
                    {"name": "producer_self_play", "weight": 97, "source": "self"},
                    {
                        "name": "catanatron_value",
                        "weight": external_weight,
                        "source": "external_engine",
                        "engine": "catanatron_value",
                    },
                ]
            }
        )
    )
    return manifest_path


def test_resolve_mix_with_exploiter_default_uses_manifest_weights(tmp_path):
    from catan_zero.rl.flywheel.opponent_mix import external_engine_effective_fraction

    manifest_path = _exploiter_manifest(tmp_path, external_weight=3.0)
    config = cli._resolve_mix_with_exploiter(str(manifest_path), None)
    assert external_engine_effective_fraction(config) == pytest.approx(0.03)


def test_resolve_mix_with_exploiter_scales_to_flag(tmp_path):
    from catan_zero.rl.flywheel.opponent_mix import external_engine_effective_fraction

    manifest_path = _exploiter_manifest(tmp_path, external_weight=3.0)
    config = cli._resolve_mix_with_exploiter(str(manifest_path), 0.02)
    assert external_engine_effective_fraction(config) == pytest.approx(0.02)


def test_resolve_mix_with_exploiter_rejects_over_cap(tmp_path):
    # 97/20 -> external share ~0.17, over the 0.05 R9 cap; must fail fast.
    manifest_path = _exploiter_manifest(tmp_path, external_weight=20.0)
    with pytest.raises(SystemExit, match="cap"):
        cli._resolve_mix_with_exploiter(str(manifest_path), None)


def test_resolve_mix_with_exploiter_flag_over_cap_rejected(tmp_path):
    manifest_path = _exploiter_manifest(tmp_path, external_weight=3.0)
    with pytest.raises(SystemExit):
        cli._resolve_mix_with_exploiter(str(manifest_path), 0.20)


def test_resolve_mix_with_exploiter_flag_without_external_category_errors(tmp_path):
    manifest_path = _minimal_self_only_manifest(tmp_path)
    with pytest.raises(SystemExit, match="no effective external_engine"):
        cli._resolve_mix_with_exploiter(str(manifest_path), 0.03)


def test_exploiter_fraction_requires_mix_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_gumbel_selfplay_data.py",
            "--skip-guards",
            "--out-dir",
            str(tmp_path / "out"),
            "--games",
            "0",
            "--exploiter-fraction",
            "0.03",
        ],
    )
    with pytest.raises(SystemExit, match="requires --opponent-mix-manifest"):
        cli.main()


def test_merge_worker_summaries_default_path_leaves_exploiter_disabled(tmp_path):
    args = _base_args(games=1)
    summary = cli._merge_worker_summaries(
        [], out_dir=tmp_path, elapsed_sec=1.0, args=args
    )
    assert summary["exploiter_enabled"] is False
    assert summary["exploiter_games"] == 0
    assert summary["exploiter_engines_used"] == []
    assert summary["exploiter_per_engine_stats"] == {}
    assert summary["exploiter_divergence_topics"] == {}


def test_merge_worker_summaries_aggregates_exploiter_per_engine_stats(tmp_path):
    """Two workers each report raw (games, champion_wins, divergences) per external
    engine; the merge SUMS before dividing, and merges divergence topic counts."""
    worker_a = {
        "worker_index": 0,
        "out_dir": str(tmp_path / "worker_000"),
        "games_completed": 10,
        "shards": [],
        "errors": [],
        "wins_by_color": {},
        "exploiter_games": 3,
        "exploiter_per_engine_stats": {
            "catanatron_value": {"games": 3, "champion_wins": 2, "divergences": 1}
        },
        "exploiter_divergence_topics": {"rules_adjudication_needed_longest_road": 1},
    }
    worker_b = {
        "worker_index": 1,
        "out_dir": str(tmp_path / "worker_001"),
        "games_completed": 10,
        "shards": [],
        "errors": [],
        "wins_by_color": {},
        "exploiter_games": 2,
        "exploiter_per_engine_stats": {
            "catanatron_value": {"games": 2, "champion_wins": 1, "divergences": 0}
        },
        "exploiter_divergence_topics": {"rules_adjudication_needed_longest_road": 2},
    }
    args = _base_args(games=20, opponent_mix_manifest="dummy_mix.json")
    summary = cli._merge_worker_summaries(
        [worker_a, worker_b], out_dir=tmp_path, elapsed_sec=1.0, args=args
    )
    assert summary["exploiter_enabled"] is True
    assert summary["exploiter_games"] == 5
    assert summary["exploiter_fraction_realized"] == pytest.approx(5 / 20)
    assert summary["exploiter_engines_used"] == ["catanatron_value"]
    stats = summary["exploiter_per_engine_stats"]["catanatron_value"]
    assert stats == {"games": 5, "champion_wins": 3, "divergences": 1}
    # (2+1) champion_wins over (3+2) graded games
    assert summary["exploiter_per_engine_champion_winrate"][
        "catanatron_value"
    ] == pytest.approx(3 / 5)
    assert summary["exploiter_divergence_topics"] == {
        "rules_adjudication_needed_longest_road": 3
    }
    assert summary["exploiter_fraction_cap"] == pytest.approx(0.05)
