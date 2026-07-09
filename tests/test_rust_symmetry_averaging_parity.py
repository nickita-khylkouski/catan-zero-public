"""Naive-reference parity for D6 root symmetry averaging on the RUST featurizer
(CAT-65 / task #81 verification item (d)).

The existing `hex_symmetry` suite proves the 12 permutation tables are valid
board automorphisms (bijection / group-closure / incidence-automorphism /
round-trip against the live rust featurizer id space), and
`test_rust_action_context_evaluator_wiring.test_evaluate_symmetry_averaged_matches_between_rust_featurize_flag`
proves `evaluate_symmetry_averaged` gives the same answer whether the entity
tensor came from the Rust or the Python featurizer. What neither covers is that
the *optimized* averaging path -- featurize ONCE, then `average_forward` tiles
that single featurization to a B=12 batch, permutes it, and does ONE batched
forward -- equals a slow, naive reference that permutes and forwards each of the
12 orientations INDEPENDENTLY (a fresh B=1 permute + a separate forward per
orientation) and averages in Python.

This is the "matching a slow, naive 12x reference implementation on a handful of
test roots" gate: it is what would catch a batch-axis aliasing bug, a wrong
`np.repeat`/tile, or a mis-broadcast canonical-coordinate restore inside
`average_forward`/`orientations_entity` -- bugs that the flag-parity test above
cannot see because BOTH of its sides run the same batched code.

Scope note (honest): this does NOT construct a physically rotated live Catan
board and re-featurize it -- catanatron_rs has no public "rotate the board"
constructor, so a from-a-rotated-game reference is not achievable here. The
claim that permuting the featurized tensor equals featurizing a rotated board
rests on the board-automorphism property, which is verified independently by
`tests/test_hex_symmetry.py`. This test closes the remaining gap: that the
batched averaging optimization equals the un-batched per-orientation computation
on the Rust-featurized tensor, end-to-end through a real forward pass.

Needs the catanatron_rs extension WITH the task-#81 entity/context featurizers.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import catanatron_rs

    _HAS_RUST_FEATURIZE = hasattr(catanatron_rs, "build_entity_features_flat") and hasattr(
        catanatron_rs, "build_action_context_flat"
    )
except ImportError:
    catanatron_rs = None  # type: ignore[assignment]
    _HAS_RUST_FEATURIZE = False

needs_rust_featurize = pytest.mark.skipif(
    not _HAS_RUST_FEATURIZE,
    reason="catanatron_rs with the task-#81 entity/context featurizers not installed",
)

COLORS: tuple[str, ...] = ("RED", "BLUE")


def _tiny_real_policy():
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    policy.model.eval()  # create() leaves train mode; active Dropout would break equality.
    return policy


def _advance_to_wide_root(game, *, min_legal: int = 3, max_steps: int = 300):
    for _ in range(max_steps):
        if game.winning_color() is not None:
            break
        legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
        if len(legal) >= min_legal:
            return game
        game.play_tick()
    raise AssertionError(f"did not reach a state with >= {min_legal} legal actions")


def _rust_entity_and_context(policy, game, legal_actions, acting_color):
    """Reproduce `evaluate_symmetry_averaged`'s Rust featurize inputs exactly:
    the single-state (B=1) entity dict, the (1, A) legal-id array, and the
    (1, A, C) context array, all via the `rust_featurize=True` seam."""
    from catan_zero.search.neural_rust_mcts import (
        EntityGraphRustEvaluator,
        EntityGraphRustEvaluatorConfig,
        _resolve_entity_adapter,
        rust_policy_action_ids,
    )

    evaluator = EntityGraphRustEvaluator(
        policy,
        config=EntityGraphRustEvaluatorConfig(public_observation=False, rust_featurize=True),
    )
    policy_action_ids = rust_policy_action_ids(
        game, legal_actions, colors=COLORS, action_size=int(policy.action_size)
    )
    resolved = _resolve_entity_adapter(
        game,
        legal_actions,
        colors=COLORS,
        action_size=int(policy.action_size),
        policy_action_ids=policy_action_ids,
        snapshot=None,
        action_by_id=None,
        public_observation=False,
        perspective=acting_color,
    )
    entity = evaluator._entity_batch_via_rust(
        game,
        colors=COLORS,
        policy_action_ids=policy_action_ids,
        acting_color=acting_color,
        adapter=resolved[1],
    )
    context = evaluator._context_batch_via_rust(
        game,
        acting_color=acting_color,
        adapter=resolved[1],
    )
    legal_ids = np.asarray(policy_action_ids, dtype=np.int64)[None, :]
    return entity, legal_ids, context


def _make_forward_fn(policy):
    import torch

    def forward_fn(entity_n, legal_n, ctx_n, return_q):
        with torch.no_grad():
            out = policy.forward_legal_np(entity_n, legal_n, ctx_n, return_q=return_q)
        return {
            "logits": out["logits"].detach().float().cpu().numpy(),
            "value": out["value"].detach().float().cpu().numpy().reshape(-1),
        }

    return forward_fn


@needs_rust_featurize
def test_batched_average_matches_naive_per_orientation_loop():
    """The optimized `average_forward` (tile-to-12 + one batched permute + one
    batched forward) must equal a naive loop of 12 independent B=1 permute +
    separate forward calls, averaged in Python -- on the Rust-featurized tensor,
    for both value and per-candidate logits, across several wide roots."""
    from catan_zero.rl.hex_symmetry import N_SYMMETRIES, build_hex_symmetry

    policy = _tiny_real_policy()
    forward_fn = _make_forward_fn(policy)
    sym = build_hex_symmetry()

    compared = 0
    for seed in (101, 102, 103, 104):
        game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
        game = _advance_to_wide_root(game, min_legal=3)
        acting_color = str(game.current_color())
        legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))

        entity, legal_ids, context = _rust_entity_and_context(policy, game, legal, acting_color)

        # Optimized path (the code under test).
        avg = sym.average_forward(entity, legal_ids, context, forward_fn, return_q=False)
        opt_value = float(avg["value"])
        opt_logits = np.asarray(avg["logits"], dtype=np.float64)

        # Naive reference: permute + forward each orientation INDEPENDENTLY.
        naive_values: list[float] = []
        naive_logits: list[np.ndarray] = []
        for g in range(N_SYMMETRIES):
            ent_g = sym.permute_entity_batch(entity, g)  # B=1, single symmetry
            out_g = forward_fn(ent_g, legal_ids, context, False)
            naive_values.append(float(np.asarray(out_g["value"]).reshape(-1)[0]))
            naive_logits.append(np.asarray(out_g["logits"], dtype=np.float64)[0])
        naive_value = float(np.mean(naive_values))
        naive_logit = np.mean(np.stack(naive_logits, axis=0), axis=0)

        assert np.isclose(opt_value, naive_value, atol=1e-6, rtol=1e-5), (
            f"seed={seed}: batched value {opt_value!r} != naive {naive_value!r}"
        )
        assert opt_logits.shape == naive_logit.shape, (
            f"seed={seed}: logits shape {opt_logits.shape} != {naive_logit.shape}"
        )
        assert np.allclose(opt_logits, naive_logit, atol=1e-6, rtol=1e-5), (
            f"seed={seed}: batched logits differ from naive per-orientation mean "
            f"(max abs diff {np.max(np.abs(opt_logits - naive_logit)):.3e})"
        )
        compared += 1

    assert compared >= 3


@needs_rust_featurize
def test_identity_orientation_reproduces_the_unpermuted_forward():
    """Orientation g=0 (identity) must leave the Rust-featurized tensor's forward
    output unchanged -- a guard that the permutation machinery is a no-op at the
    identity and that `average_forward`'s orientation 0 is the canonical state."""
    from catan_zero.rl.hex_symmetry import build_hex_symmetry

    policy = _tiny_real_policy()
    forward_fn = _make_forward_fn(policy)
    sym = build_hex_symmetry()

    game = catanatron_rs.Game.simple(list(COLORS), seed=101)
    game = _advance_to_wide_root(game, min_legal=3)
    acting_color = str(game.current_color())
    legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
    entity, legal_ids, context = _rust_entity_and_context(policy, game, legal, acting_color)

    base = forward_fn(entity, legal_ids, context, False)
    ent_identity = sym.permute_entity_batch(entity, 0)
    permuted0 = forward_fn(ent_identity, legal_ids, context, False)

    assert np.allclose(
        np.asarray(base["value"]).reshape(-1),
        np.asarray(permuted0["value"]).reshape(-1),
        atol=1e-6,
        rtol=1e-5,
    )
    assert np.allclose(
        np.asarray(base["logits"], dtype=np.float64),
        np.asarray(permuted0["logits"], dtype=np.float64),
        atol=1e-6,
        rtol=1e-5,
    )
