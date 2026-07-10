from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import continuous_flywheel as flywheel  # noqa: E402
from catan_zero.rl.flywheel import (  # noqa: E402
    FlywheelConfig,
    ensure_dirs,
    promote,
    publish_candidate,
    seed_champion,
)
from tools.champion_registry import ChampionRegistry  # noqa: E402


def _runner(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    candidate_value_readout: str = "scalar",
    baseline_value_readout: str = "scalar",
) -> flywheel.Runner:
    loop_dir = tmp_path / "loop"
    loop_dir.mkdir()
    return flywheel.Runner(
        FlywheelConfig(
            gate_candidate_value_readout=candidate_value_readout,
            gate_baseline_value_readout=baseline_value_readout,
        ),
        loop_dir,
        dry_run=dry_run,
        workers=4,
        device="cuda:0",
        base_seed=123,
    )


def test_scoreboard_gate_cannot_promote_masked_flywheel(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner.cfg.gate_style = "scoreboard"
    result = runner.gate("candidate.pt", "champion.pt", round_idx=0)
    assert result == {
        "ok": False,
        "pass": False,
        "verdict": "unsafe_gate_style",
        "reason": (
            "scoreboard evaluation has no public-information search boundary; "
            "use gate_style='h2h'"
        ),
    }


def test_masked_h2h_gate_uses_named_flywheel_policy_and_extends(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(
        tmp_path,
        candidate_value_readout="categorical",
        baseline_value_readout="scalar",
    )
    commands: list[list[str]] = []
    decisions = iter(("continue", "H1"))
    eval_calls: list[dict] = []

    def fake_run(cmd: list[str], _log_path: Path) -> int:
        commands.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                    {
                        "public_observation": True,
                        "information_set_search": True,
                        "determinization_particles": 4,
                        "determinization_min_simulations": 32,
                    "candidate_value_readout": cmd[
                        cmd.index("--candidate-value-readout") + 1
                    ],
                    "baseline_value_readout": cmd[
                        cmd.index("--baseline-value-readout") + 1
                    ],
                    "candidate_win_rate": 0.55,
                    "games_played": int(cmd[cmd.index("--pairs") + 1]) * 2,
                    "games": [],
                }
            )
        )
        return 0

    def fake_evaluate(_pair_scores, **kwargs):
        eval_calls.append(kwargs)
        return {
            "decision": next(decisions),
            "ll_pairs": 10,
            "split_pairs": 20,
            "ww_pairs": 30,
            **kwargs,
        }

    monkeypatch.setattr(flywheel, "_run", fake_run)
    monkeypatch.setattr(
        flywheel,
        "pair_scores_from_h2h_games",
        lambda _games: ([0.0, 0.5, 1.0], {"ll_pairs": 1, "split_pairs": 1, "ww_pairs": 1, "incomplete_pairs": 0}),
    )
    monkeypatch.setattr(flywheel, "evaluate_pentanomial_sprt", fake_evaluate)

    result = runner._gate_h2h("candidate.pt", "champion.pt", round_idx=2)

    assert result["ok"] is True
    assert result["pass"] is True
    assert result["verdict"] == "promote"
    assert result["gate_config_params"] == {
        "gate_config": "flywheel",
        "elo0": -10.0,
        "elo1": 15.0,
        "alpha": 0.05,
        "beta": 0.05,
        "n_sims": 16,
        "base_games": 300,
        "max_games": 600,
    }
    assert [int(cmd[cmd.index("--pairs") + 1]) for cmd in commands] == [150, 300]
    assert all("--gate-config" in cmd and cmd[cmd.index("--gate-config") + 1] == "flywheel" for cmd in commands)
    assert all("--public-observation" in cmd for cmd in commands)
    assert all(cmd[cmd.index("--n-full") + 1] == "16" for cmd in commands)
    assert all(
        cmd[cmd.index("--candidate-value-readout") + 1] == "categorical"
        for cmd in commands
    )
    assert all(
        cmd[cmd.index("--baseline-value-readout") + 1] == "scalar"
        for cmd in commands
    )
    assert eval_calls == [
        {"elo0": -10.0, "elo1": 15.0, "alpha": 0.05, "beta": 0.05},
        {"elo0": -10.0, "elo1": 15.0, "alpha": 0.05, "beta": 0.05},
    ]

    artifact = json.loads((runner.loop_dir / "gates" / "round_002.json").read_text())
    assert artifact["gate_config_params"]["gate_config"] == "flywheel"
    assert artifact["verdict"] == "promote"
    assert artifact["public_observation"] is True
    assert artifact["candidate_value_readout"] == "categorical"
    assert artifact["baseline_value_readout"] == "scalar"


def test_flywheel_timeout_canary_is_generator_only(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], _log_path: Path) -> int:
        commands.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                    {
                        "public_observation": True,
                        "information_set_search": True,
                        "determinization_particles": 4,
                        "determinization_min_simulations": 32,
                    "candidate_value_readout": cmd[
                        cmd.index("--candidate-value-readout") + 1
                    ],
                    "baseline_value_readout": cmd[
                        cmd.index("--baseline-value-readout") + 1
                    ],
                    "candidate_win_rate": 0.51,
                    "games": [],
                }
            )
        )
        return 0

    monkeypatch.setattr(flywheel, "_run", fake_run)
    monkeypatch.setattr(
        flywheel,
        "pair_scores_from_h2h_games",
        lambda _games: ([0.5], {"ll_pairs": 1, "split_pairs": 8, "ww_pairs": 2, "incomplete_pairs": 0}),
    )
    monkeypatch.setattr(
        flywheel,
        "evaluate_pentanomial_sprt",
        lambda _scores, **kwargs: {
            "decision": "continue",
            "ll_pairs": 1,
            "split_pairs": 8,
            "ww_pairs": 2,
            **kwargs,
        },
    )
    monkeypatch.setattr(
        flywheel,
        "r9_timeout_verdict",
        lambda *args, **kwargs: {"canary_eligible": True, "verdict": "canary_promote"},
    )

    result = runner._gate_h2h("candidate.pt", "champion.pt", round_idx=0)

    assert len(commands) == 2
    assert result["pass"] is True
    assert result["verdict"] == "canary_promote"
    assert result["promotion_scope"] == "generator_champion_only"
    assert result["public_champion_updated"] is False


def test_masking_mismatch_fails_closed(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    def fake_run(cmd: list[str], _log_path: Path) -> int:
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"public_observation": False, "games": []}))
        return 0

    monkeypatch.setattr(flywheel, "_run", fake_run)
    result = runner._gate_h2h("candidate.pt", "champion.pt", round_idx=0)

    assert result["ok"] is False
    assert result["pass"] is False
    assert result["verdict"] == "masking_mismatch"


def test_value_readout_mismatch_fails_closed(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(
        tmp_path,
        candidate_value_readout="categorical",
        baseline_value_readout="scalar",
    )

    def fake_run(cmd: list[str], _log_path: Path) -> int:
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                    {
                        "public_observation": True,
                        "information_set_search": True,
                        "determinization_particles": 4,
                        "determinization_min_simulations": 32,
                    # Simulate an old/broken subprocess silently evaluating
                    # both sides through the scalar readout.
                    "candidate_value_readout": "scalar",
                    "baseline_value_readout": "scalar",
                    "games": [],
                }
            )
        )
        return 0

    monkeypatch.setattr(flywheel, "_run", fake_run)
    result = runner._gate_h2h("candidate.pt", "champion.pt", round_idx=0)

    assert result["ok"] is False
    assert result["pass"] is False
    assert result["verdict"] == "value_readout_mismatch"
    assert result["candidate_value_readout"] == "categorical"
    assert result["baseline_value_readout"] == "scalar"


def test_flywheel_value_readout_config_validates_and_round_trips() -> None:
    cfg = FlywheelConfig(
        gate_candidate_value_readout="categorical",
        gate_baseline_value_readout="scalar",
    ).validate()
    assert FlywheelConfig.from_dict(cfg.to_dict()) == cfg
    for field in ("gate_candidate_value_readout", "gate_baseline_value_readout"):
        with pytest.raises(ValueError, match=field):
            FlywheelConfig(**{field: "unknown"}).validate()


def test_registry_promotion_updates_generator_only_and_is_recovery_idempotent(tmp_path: Path) -> None:
    loop_dir = tmp_path / "loop"
    ensure_dirs(loop_dir)
    seed = tmp_path / "seed.pt"
    seed.write_bytes(b"seed")
    public = tmp_path / "public.pt"
    public.write_bytes(b"public-gen3")
    seed_champion(loop_dir, seed, version=0)

    registry_path = loop_dir / "champion_registry.json"
    registry = ChampionRegistry(registry_path)
    public_before = registry.set_role("public_champion", public, version=3, reason="pinned public gen-3")
    registry.save()

    candidate = publish_candidate(loop_dir, lambda p: Path(p).write_bytes(b"candidate"), step=10)
    new_champion = promote(loop_dir, candidate, gate={"verdict": "promote"})
    gate = {
        "verdict": "promote",
        "gate_config_params": {"gate_config": "flywheel", "elo0": -10.0, "elo1": 15.0},
    }

    first = flywheel.record_generator_promotion(
        registry_path,
        new_champion,
        round_idx=7,
        gate=gate,
    )
    second = flywheel.record_generator_promotion(
        registry_path,
        new_champion,
        round_idx=7,
        gate=gate,
    )

    reloaded = ChampionRegistry.load(registry_path)
    generator = reloaded.get_role("generator_champion")
    public_after = reloaded.get_role("public_champion")
    assert generator is not None
    assert generator.checkpoint_path == new_champion.path
    assert generator.version == new_champion.version
    assert generator.provenance["gate_config_params"]["gate_config"] == "flywheel"
    assert public_after == public_before
    assert reloaded.promotion_count("generator_champion") == 1
    assert first["promotion_count"] == second["promotion_count"] == 1
    assert first["public_champion_updated"] is False
    assert second["already_recorded"] is True
