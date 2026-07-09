"""Tests for the high-regret restart system (task #64).

Covers:
  * reconstruction round-trip (reconstructed featurisation matches stored rows
    bit/fp16-for-bit, and the recomputed legal policy ids match exactly),
  * reconstruction determinism,
  * regret scoring on a synthetic shard (value_surprise / KL / forced gating),
  * restart play-from-state (schema, start_mode tagging, policy/value weights),
  * RestartShardWriter round-trips the extra columns,
  * mixing-recipe planner arithmetic.

Data-dependent tests skip cleanly when the B200 self-play corpus is absent, so
the suite still runs in a bare checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import regret_common  # noqa: E402
import reconstruct_state as rs  # noqa: E402

_PILOT = Path("/home/ubuntu/catan-zero/runs/selfplay/gen1_pilot_20260704")


def _rust_available() -> bool:
    try:
        from catan_zero.search.rust_mcts import _require_rust_module

        _require_rust_module()
        return True
    except Exception:
        return False


def _first_pilot_shard() -> Path | None:
    if not _PILOT.exists():
        return None
    shards = sorted(_PILOT.rglob("gumbel_self_play_shard_*.npz"))
    return shards[0] if shards else None


# --------------------------------------------------------------------------- #
# Reconstruction round-trip (the core correctness proof)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _rust_available(), reason="rust engine unavailable")
def test_reconstruction_round_trip_pilot():
    shard = _first_pilot_shard()
    if shard is None:
        pytest.skip("pilot corpus not present")
    result = rs.round_trip_shard_rows(shard, max_rows=24, correct_rust_chance_spectra=None, seed=1)
    assert result["rows_checked"] > 0
    # Every sampled row must reconstruct exactly under the resolved flag.
    assert result["pass_rate"] == 1.0, result
    # fp16 tolerance is generous; real diffs are ~0 for a true replay.
    assert result["max_abs_diff_overall"] <= 1e-2, result


@pytest.mark.skipif(not _rust_available(), reason="rust engine unavailable")
def test_reconstruction_deterministic():
    shard = _first_pilot_shard()
    if shard is None:
        pytest.skip("pilot corpus not present")
    data = regret_common.load_shard(shard)
    gseed = int(np.asarray(data["game_seed"]).reshape(-1)[0])
    seq = rs.gather_game_action_sequence(shard.parent, gseed)
    target = min(6, len(seq))
    g1 = rs.reconstruct_state(gseed, seq.actions, target)
    g2 = rs.reconstruct_state(gseed, seq.actions, target)
    f1 = rs.featurize_state(g1)
    f2 = rs.featurize_state(g2)
    assert f1["legal_policy_ids"] == f2["legal_policy_ids"]
    assert np.array_equal(f1["features"]["vertex_tokens"], f2["features"]["vertex_tokens"])


# --------------------------------------------------------------------------- #
# Regret scoring on a synthetic shard (no engine needed)
# --------------------------------------------------------------------------- #
def _synthetic_shard() -> dict[str, np.ndarray]:
    # 3 rows, legal width 3. Row0: clean win, taken Q far from z (+high surprise).
    # Row1: forced (1 legal action). Row2: clean loss, argmax mismatch.
    lids = np.array([[10, 11, 12], [20, -1, -1], [30, 31, -1]], dtype=np.int16)
    action_taken = np.array([10, 20, 31], dtype=np.int16)
    target_scores = np.array(
        [[-0.9, 0.1, 0.2], [np.nan, np.nan, np.nan], [0.8, 0.7, np.nan]], dtype=np.float32
    )
    target_scores_mask = np.isfinite(target_scores)
    target_policy = np.array(
        [[0.7, 0.2, 0.1], [1.0, 0.0, 0.0], [0.2, 0.8, 0.0]], dtype=np.float32
    )
    prior_policy = np.array(
        [[0.6, 0.3, 0.1], [1.0, 0.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float16
    )
    return {
        "action_taken": action_taken,
        "legal_action_ids": lids,
        "target_scores": target_scores,
        "target_scores_mask": target_scores_mask,
        "target_policy": target_policy,
        "prior_policy": prior_policy,
        "winner": np.array(["RED", "RED", "BLUE"]),
        "player": np.array(["RED", "RED", "RED"]),
        "truncated": np.array([False, False, False]),
        "phase": np.array(["BUILD_INITIAL_SETTLEMENT", "ROLL", "MOVE_ROBBER"]),
        "is_forced": np.array([False, True, False]),
    }


def test_score_shard_value_surprise_and_gating():
    shard = _synthetic_shard()
    scored = regret_common.score_shard(shard, regret_common.RegretConfig())
    # Row0: z=+1, q_taken=-0.9 -> value_surprise = 1.9
    assert scored["value_surprise"][0] == pytest.approx(1.9, abs=1e-4)
    assert scored["has_value_surprise"][0]
    # Row1 forced -> not a candidate.
    assert scored["is_forced"][1]
    assert not scored["is_candidate"][1]
    # Row2: z=-1, q_taken=0.7 -> value_surprise = 1.7; argmax(target)=1, argmax(prior)=0 -> mismatch & lost.
    assert scored["value_surprise"][2] == pytest.approx(1.7, abs=1e-4)
    assert scored["argmax_mismatch_lost"][2] == pytest.approx(1.0)
    # Row0 argmax matches (both 0) -> no mismatch bonus.
    assert scored["argmax_mismatch_lost"][0] == pytest.approx(0.0)


def test_score_shard_kl_zero_when_target_equals_prior():
    shard = _synthetic_shard()
    shard["prior_policy"] = shard["target_policy"].astype(np.float16)
    scored = regret_common.score_shard(shard, regret_common.RegretConfig())
    assert np.allclose(scored["kl_disagreement"], 0.0, atol=1e-3)


def test_score_shard_raw_uses_provided_values():
    shard = _synthetic_shard()
    # Strip searched Q to simulate a raw shard.
    shard["target_scores_mask"] = np.zeros_like(shard["target_scores_mask"])
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # v(s)=0 everywhere
    scored = regret_common.score_shard(shard, regret_common.RegretConfig(), values=values)
    # Row0: |0 - (+1)| = 1.0
    assert scored["value_surprise"][0] == pytest.approx(1.0, abs=1e-4)
    # Row2: |0 - (-1)| = 1.0
    assert scored["value_surprise"][2] == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# Mixing-recipe planner
# --------------------------------------------------------------------------- #
def test_plan_start_mix_sums_and_ratios():
    import generate_restart_selfplay as gr

    counts = gr.plan_start_mix(1000)
    assert sum(counts.values()) == 1000
    assert counts["normal"] == 600
    assert counts["opening"] == 200
    assert counts["robber_dev"] == 100
    assert counts["random_archived"] == 100


def test_plan_start_mix_rounding_goes_to_normal():
    import generate_restart_selfplay as gr

    counts = gr.plan_start_mix(7)
    assert sum(counts.values()) == 7


# --------------------------------------------------------------------------- #
# Restart play-from-state + writer round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _rust_available(), reason="rust engine unavailable")
def test_restart_play_from_state_and_writer(tmp_path):
    import generate_restart_selfplay as gr
    from catan_zero.search.gumbel_chance_mcts import HeuristicRustEvaluator

    shard = _first_pilot_shard()
    if shard is None:
        pytest.skip("pilot corpus not present")
    data = regret_common.load_shard(shard)
    gseed = int(np.asarray(data["game_seed"]).reshape(-1)[0])
    seq = rs.gather_game_action_sequence(shard.parent, gseed)
    start_dec = min(8, max(1, len(seq) // 4))

    evaluator = HeuristicRustEvaluator(score_actions=False)
    action_size = rs.action_size_for_colors(("RED", "BLUE"))
    game, chance_rng = rs.reconstruct_state(
        gseed, seq.actions, start_dec, return_rng=True, action_size=action_size
    )
    config = gr.RestartSelfPlayConfig(max_continuation_decisions=40, restart_temperature_decisions=4)
    pairs = gr.play_restart_game_from_state(
        evaluator,
        game,
        config=config,
        start_mode=gr.START_ARCHIVED,
        start_bucket="opening",
        archived_game_seed=gseed,
        archived_decision_index=start_dec,
        restart_select_seed=123,
        action_size=action_size,
        chance_rng=chance_rng,
    )
    assert len(pairs) > 0
    for row, _features in pairs:
        assert row["start_mode"] == gr.START_ARCHIVED
        assert row["teacher_name"] == gr.TEACHER_NAME
        assert float(row["policy_weight_multiplier"]) == 0.0
        assert float(row["value_weight_multiplier"]) == 1.0
        assert int(row["archived_game_seed"]) == gseed
        # legal_action_ids valid entries are within action space.
        lids = np.asarray(row["legal_action_ids"])
        assert (lids[lids >= 0] < action_size).all()

    # Writer round-trip: restart columns survive to disk.
    writer = gr.RestartShardWriter(tmp_path, shard_size=1000)
    for row, features in pairs:
        writer.add(row, features)
    writer.close()
    out = sorted(tmp_path.glob("restart_self_play_shard_*.npz"))
    assert out
    loaded = regret_common.load_shard(out[0])
    for key in gr.RESTART_KEYS:
        assert key in loaded, key
    assert loaded["start_mode"][0] == gr.START_ARCHIVED
    assert int(loaded["archived_decision_index"][0]) == start_dec
    # Shared schema intact.
    assert "hex_tokens" in loaded and "target_policy" in loaded


# --------------------------------------------------------------------------- #
# Extractor helpers over real pilot data (light)
# --------------------------------------------------------------------------- #
def test_topk_keeps_highest():
    import extract_regret_states as ex

    topk = ex._TopK(3)
    for score in [0.1, 0.9, 0.5, 0.3, 0.95, 0.2]:
        topk.offer(score, {"regret_score": score})
    kept = [r["regret_score"] for r in topk.sorted_desc()]
    assert kept == [0.95, 0.9, 0.5]


# --------------------------------------------------------------------------- #
# --public-observation threading into _build_evaluator (cat92-public-obs-fix)
# --------------------------------------------------------------------------- #
import argparse  # noqa: E402


def _eval_args(**over):
    base = dict(
        checkpoint="dummy.pt", device="cpu", value_scale=1.0, value_squash="tanh",
        public_observation=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_build_evaluator_threads_public_observation(monkeypatch):
    """_build_evaluator must pass args.public_observation into the eval config
    (the whole point of the fix). Monkeypatch from_checkpoint to capture it, so
    this is deterministic and needs no checkpoint load."""
    import generate_restart_selfplay as gr
    import catan_zero.search.neural_rust_mcts as nrm

    captured = {}

    @classmethod
    def _fake_from_checkpoint(cls, checkpoint, *, device, config):  # noqa: ANN001
        captured["public_observation"] = config.public_observation
        captured["value_scale"] = config.value_scale
        return object()  # stand-in evaluator; we only inspect the config

    monkeypatch.setattr(
        nrm.BatchedEntityGraphRustEvaluator, "from_checkpoint", _fake_from_checkpoint
    )
    gr._build_evaluator(_eval_args(public_observation=True))
    assert captured["public_observation"] is True
    captured.clear()
    gr._build_evaluator(_eval_args(public_observation=False))
    assert captured["public_observation"] is False


def _masked_champion() -> Path | None:
    for p in (
        Path("/home/ubuntu/flywheel_backup_20260708/champion/champion_v0.pt"),
        Path("/home/ubuntu/catan-zero/runs/bc/gen3_20260706/checkpoint.pt"),
    ):
        if p.exists():
            return p
    return None


@pytest.mark.skipif(not _rust_available(), reason="rust engine unavailable")
def test_masked_champion_loads_and_asserts_regime():
    """Real masked champion loads through _build_evaluator with
    public_observation=True (regime assert passes), and REFUSES with
    public_observation=False (safety net intact)."""
    import generate_restart_selfplay as gr

    ckpt = _masked_champion()
    if ckpt is None:
        pytest.skip("masked champion checkpoint not present")
    # public_observation=True -> masked champion accepted.
    ev = gr._build_evaluator(_eval_args(checkpoint=str(ckpt), public_observation=True))
    assert ev is not None
    # public_observation=False on a masked-trained net -> loud regime refusal.
    with pytest.raises(Exception):
        gr._build_evaluator(_eval_args(checkpoint=str(ckpt), public_observation=False))


@pytest.mark.skipif(not _rust_available(), reason="rust engine unavailable")
def test_restart_game_with_masked_champion_completes(tmp_path):
    """End-to-end: one restart-sampled game runs with the masked champion
    evaluator (public_observation=True), producing value-only rows."""
    import generate_restart_selfplay as gr

    ckpt = _masked_champion()
    shard = _first_pilot_shard()
    if ckpt is None or shard is None:
        pytest.skip("masked champion or pilot corpus not present")
    evaluator = gr._build_evaluator(_eval_args(checkpoint=str(ckpt), public_observation=True))
    data = regret_common.load_shard(shard)
    gseed = int(np.asarray(data["game_seed"]).reshape(-1)[0])
    seq = rs.gather_game_action_sequence(shard.parent, gseed)
    start_dec = min(8, max(1, len(seq) // 4))
    action_size = rs.action_size_for_colors(("RED", "BLUE"))
    game, chance_rng = rs.reconstruct_state(
        gseed, seq.actions, start_dec, return_rng=True, action_size=action_size
    )
    config = gr.RestartSelfPlayConfig(max_continuation_decisions=30, restart_temperature_decisions=4)
    pairs = gr.play_restart_game_from_state(
        evaluator, game, config=config, start_mode=gr.START_ARCHIVED,
        start_bucket="opening", archived_game_seed=gseed, archived_decision_index=start_dec,
        restart_select_seed=7, action_size=action_size, chance_rng=chance_rng,
    )
    assert len(pairs) > 0
    for row, _features in pairs:
        assert float(row["value_weight_multiplier"]) == 1.0
        assert float(row["policy_weight_multiplier"]) == 0.0
