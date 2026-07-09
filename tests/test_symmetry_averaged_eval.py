"""Tests for the wide-root symmetry-averaged evaluator (f74b).

The gating logic (off => no-op, on+wide => averaged, on+narrow => plain,
missing-method => graceful fallback) is tested with a mock evaluator against a
real rust opening root. A real-checkpoint integration test confirms the neural
evaluate_symmetry_averaged actually denoises (differs from a single-orientation
evaluate, stays a valid distribution/value, lands within the orientation
envelope)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("catanatron_rs")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from catan_zero.search.gumbel_chance_mcts import (  # noqa: E402
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    _GNode,
)
from catan_zero.search.rust_mcts import _require_rust_module  # noqa: E402

COLORS = ("RED", "BLUE")
CKPT = "/home/ubuntu/catan-zero/runs/bc/entity_graph_35m_value_repair_v2_raw_selfplay_20260704/checkpoint.pt"


class _MockEvaluator:
    """Returns a distinct sentinel value from each path so we can see which the
    MCTS called. Priors are uniform over the real legal actions."""

    def __init__(self, *, with_symmetry: bool):
        self.plain_value = 0.10
        self.avg_value = 0.90
        self.calls = {"evaluate": 0, "symmetry": 0}
        if with_symmetry:
            self.evaluate_symmetry_averaged = self._symmetry  # type: ignore[attr-defined]

    def _uniform(self, legal_actions):
        p = 1.0 / max(len(legal_actions), 1)
        return {int(a): p for a in legal_actions}

    def evaluate(self, game, legal_actions, *, root_color, colors):
        self.calls["evaluate"] += 1
        return self._uniform(legal_actions), self.plain_value

    def _symmetry(self, game, legal_actions, *, root_color, colors):
        self.calls["symmetry"] += 1
        return self._uniform(legal_actions), self.avg_value


def _opening_game(seed=1):
    rs = _require_rust_module()
    return rs.Game.simple(list(COLORS), seed=seed)


def _expand_root(config, evaluator, game):
    mcts = GumbelChanceMCTS(config, evaluator)
    root = _GNode(game=game.copy(), root_color=str(game.current_color()))
    mcts._expand(root, at_root=True)
    return root


def test_off_state_is_noop():
    game = _opening_game()
    ev = _MockEvaluator(with_symmetry=True)
    cfg = GumbelChanceMCTSConfig(colors=COLORS, symmetry_averaged_eval=False,
                                 wide_candidates_threshold=24)
    root = _expand_root(cfg, ev, game)
    assert ev.calls == {"evaluate": 1, "symmetry": 0}
    assert abs(root.prior_value - ev.plain_value) < 1e-9


def test_on_wide_root_uses_symmetry_average():
    game = _opening_game()  # opening has ~54 settlement candidates > 24
    ev = _MockEvaluator(with_symmetry=True)
    cfg = GumbelChanceMCTSConfig(colors=COLORS, symmetry_averaged_eval=True,
                                 wide_candidates_threshold=24)
    root = _expand_root(cfg, ev, game)
    assert ev.calls == {"evaluate": 0, "symmetry": 1}
    assert abs(root.prior_value - ev.avg_value) < 1e-9


def test_on_narrow_root_uses_plain_eval():
    game = _opening_game()
    ev = _MockEvaluator(with_symmetry=True)
    # threshold above the ~54 opening width => treated as narrow => no averaging.
    cfg = GumbelChanceMCTSConfig(colors=COLORS, symmetry_averaged_eval=True,
                                 wide_candidates_threshold=100)
    root = _expand_root(cfg, ev, game)
    assert ev.calls == {"evaluate": 1, "symmetry": 0}
    assert abs(root.prior_value - ev.plain_value) < 1e-9


def test_missing_method_falls_back_gracefully():
    game = _opening_game()
    ev = _MockEvaluator(with_symmetry=False)  # no evaluate_symmetry_averaged
    assert not hasattr(ev, "evaluate_symmetry_averaged")
    cfg = GumbelChanceMCTSConfig(colors=COLORS, symmetry_averaged_eval=True,
                                 wide_candidates_threshold=24)
    root = _expand_root(cfg, ev, game)  # must not raise
    assert ev.calls == {"evaluate": 1, "symmetry": 0}
    assert abs(root.prior_value - ev.plain_value) < 1e-9


def test_leaf_expansion_never_averages():
    """A non-root expansion (at_root defaults False) always uses plain eval,
    even with the flag on and a wide node -- averaging is a root-only cost."""
    game = _opening_game()
    ev = _MockEvaluator(with_symmetry=True)
    cfg = GumbelChanceMCTSConfig(colors=COLORS, symmetry_averaged_eval=True,
                                 wide_candidates_threshold=24)
    mcts = GumbelChanceMCTS(cfg, ev)
    node = _GNode(game=game.copy(), root_color=str(game.current_color()))
    mcts._expand(node)  # at_root defaults to False
    assert ev.calls == {"evaluate": 1, "symmetry": 0}


@pytest.mark.skipif(not Path(CKPT).exists(), reason="checkpoint not present")
def test_real_evaluator_symmetry_average_denoises():
    import torch
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.hex_symmetry import build_hex_symmetry, N_SYMMETRIES
    from catan_zero.search.neural_rust_mcts import (
        EntityGraphRustEvaluator, rust_game_to_entity_batch,
        rust_action_context_batch, rust_policy_action_ids,
    )

    policy = EntityGraphPolicy.load(CKPT, device="cpu")
    policy.model.eval()
    ev = EntityGraphRustEvaluator(policy)
    game = _opening_game(seed=1)
    legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
    root_color = str(game.current_color())

    priors_plain, value_plain = ev.evaluate(game, legal, root_color=root_color, colors=COLORS)
    priors_avg, value_avg = ev.evaluate_symmetry_averaged(game, legal, root_color=root_color, colors=COLORS)

    # valid distribution + value
    assert abs(sum(priors_avg.values()) - 1.0) < 1e-4
    assert -1.0 <= value_avg <= 1.0
    assert set(priors_avg) == set(priors_plain)

    # denoising actually changed something (model is symmetry-inconsistent)
    assert abs(value_avg - value_plain) > 1e-4

    # averaged value must lie within the envelope of the 12 single-orientation
    # values (it is their mean on the raw scale, monotone through the squash).
    pids = rust_policy_action_ids(game, legal, colors=COLORS, action_size=int(policy.action_size))
    entity = rust_game_to_entity_batch(game, legal, actor=root_color, colors=COLORS,
                                       action_size=int(policy.action_size), policy_action_ids=pids,
                                       public_observation=bool(ev.config.public_observation))
    context = rust_action_context_batch(game, legal, actor=root_color, colors=COLORS,
                                        action_size=int(policy.action_size),
                                        fill=float(ev.config.context_fill), policy_action_ids=pids)
    legal_ids = np.asarray(pids, dtype=np.int64)[None, :]
    sym = build_hex_symmetry()
    ent_n = sym.orientations_entity(entity)
    legal_n = np.repeat(legal_ids, N_SYMMETRIES, axis=0)
    ctx_n = np.repeat(context, N_SYMMETRIES, axis=0)
    with torch.no_grad():
        out = policy.forward_legal_np(ent_n, legal_n, ctx_n, return_q=False)
    raw_vals = out["value"].detach().float().cpu().numpy().reshape(-1)
    squashed = np.array([ev._apply_value_squash(float(v)) for v in raw_vals])
    if root_color != str(game.current_color()):
        squashed = -squashed
    lo, hi = float(squashed.min()), float(squashed.max())
    assert lo - 1e-6 <= value_avg <= hi + 1e-6
