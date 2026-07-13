"""Opt-in HL-Gauss value readout wiring for search and its launchers."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.pipeline_configs import EvalConfig, GenerateConfig
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
import catan_zero.search.neural_rust_mcts as neural_rust_mcts


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import generate_gumbel_selfplay_data as generate_cli  # type: ignore  # noqa: E402
import gumbel_search_cross_net_h2h as cross_net_cli  # type: ignore  # noqa: E402
import train_bc  # type: ignore  # noqa: E402


class _FakeGame:
    def current_color(self) -> str:
        return "RED"


def test_categorical_training_support_width_unwraps_ddp_model() -> None:
    policy = SimpleNamespace(
        model=SimpleNamespace(module=SimpleNamespace(value_categorical_bins=33))
    )
    assert train_bc._policy_value_categorical_bins(policy) == 33  # noqa: SLF001
    with pytest.raises(RuntimeError, match="value_categorical_bins >= 2"):
        train_bc._policy_value_categorical_bins(  # noqa: SLF001
            SimpleNamespace(
                model=SimpleNamespace(
                    module=SimpleNamespace(value_categorical_bins=0)
                )
            )
        )


class _SplitValuePolicy:
    """Minimal policy contract; evaluator path tests do not need the Rust wheel."""

    action_size = 16
    trained_with_masked_hidden_info = False

    def __init__(self, *, categorical: bool) -> None:
        self.trained_value_readouts = (
            ("scalar", "categorical") if categorical else ("scalar",)
        )
        self.model = SimpleNamespace(
            value_categorical_bins=9 if categorical else 0,
            value_categorical_head=object() if categorical else None,
        )

    def forward_legal_np(self, _entity, legal_ids, _context, *, return_q=False):
        del return_q
        import torch

        rows, width = legal_ids.shape
        return {
            "logits": torch.zeros((rows, width), dtype=torch.float32),
            "value": torch.full((rows,), 0.75, dtype=torch.float32),
            "value_categorical": torch.full((rows,), -0.25, dtype=torch.float32),
        }


@pytest.fixture()
def split_value_policy():
    return _SplitValuePolicy(categorical=True)


@pytest.fixture()
def evaluator_inputs(monkeypatch):
    """Stub only the Rust/feature boundary; exercise each real evaluator path."""
    legal = (3, 7)
    game = _FakeGame()
    monkeypatch.setattr(
        neural_rust_mcts,
        "_fetch_leaf_decision_inputs",
        lambda _game, _colors, *, include_snapshot=True: (
            "{}" if include_snapshot else None,
            {3: {}, 7: {}},
        ),
    )
    monkeypatch.setattr(
        neural_rust_mcts,
        "rust_policy_action_ids",
        lambda _game, actions, **_kwargs: tuple(actions),
    )
    monkeypatch.setattr(
        neural_rust_mcts,
        "_resolve_entity_adapter",
        lambda *_args, **_kwargs: ({}, object(), []),
    )
    monkeypatch.setattr(
        neural_rust_mcts,
        "rust_game_to_entity_batch",
        lambda *_args, **_kwargs: {"dummy": np.zeros((1, 1, 1), dtype=np.float32)},
    )
    monkeypatch.setattr(
        neural_rust_mcts,
        "rust_action_context_batch",
        lambda *_args, **_kwargs: np.zeros((1, len(legal), 1), dtype=np.float32),
    )

    import catan_zero.rl.hex_symmetry as hex_symmetry

    class _IdentityAverage:
        def average_forward(
            self,
            entity,
            legal_ids,
            context,
            forward_fn,
            *,
            return_q,
            action_size=None,
        ):
            del action_size
            out = forward_fn(entity, legal_ids, context, return_q)
            return {"logits": out["logits"][0], "value": float(out["value"][0])}

    monkeypatch.setattr(hex_symmetry, "build_hex_symmetry", lambda: _IdentityAverage())
    return game, legal


def test_default_is_scalar_and_categorical_requires_a_real_head() -> None:
    assert EntityGraphRustEvaluatorConfig().value_readout == "scalar"
    scalar_policy = _SplitValuePolicy(categorical=False)
    EntityGraphRustEvaluator(scalar_policy, config=EntityGraphRustEvaluatorConfig())
    with pytest.raises(ValueError, match="requires a checkpoint with a trained HL-Gauss"):
        EntityGraphRustEvaluator(
            scalar_policy,
            config=EntityGraphRustEvaluatorConfig(value_readout="categorical"),
        )
    with pytest.raises(ValueError, match="unknown value_readout"):
        EntityGraphRustEvaluator(
            scalar_policy,
            config=EntityGraphRustEvaluatorConfig(value_readout="mystery"),
        )


def test_checkpoint_config_only_head_upgrade_fails_closed(split_value_policy) -> None:
    split_value_policy._checkpoint_missing_state_keys = (
        "value_categorical_head.0.weight",
    )
    with pytest.raises(ValueError, match="trained weights are absent"):
        EntityGraphRustEvaluator(
            split_value_policy,
            config=EntityGraphRustEvaluatorConfig(value_readout="categorical"),
        )


def test_checkpoint_head_without_positive_training_provenance_fails_closed(
    split_value_policy,
) -> None:
    split_value_policy.trained_value_readouts = ("scalar",)
    with pytest.raises(ValueError, match="no positive value-training-v1 provenance"):
        EntityGraphRustEvaluator(
            split_value_policy,
            config=EntityGraphRustEvaluatorConfig(value_readout="categorical"),
        )


def test_checkpoint_value_training_provenance_distinguishes_upgrade_from_training(
    tmp_path,
) -> None:
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    base = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    categorical = EntityGraphPolicy(
        replace(base.config, value_categorical_bins=9),
        base.static_action_features.detach().cpu().numpy(),
        device="cpu",
    )
    upgrade_only_path = tmp_path / "upgrade_only.pt"
    categorical.save(upgrade_only_path)
    upgrade_only = EntityGraphPolicy.load(upgrade_only_path, device="cpu")
    assert upgrade_only.trained_value_readouts == ("scalar",)
    with pytest.raises(ValueError, match="no positive value-training-v1 provenance"):
        EntityGraphRustEvaluator(
            upgrade_only,
            config=EntityGraphRustEvaluatorConfig(value_readout="categorical"),
        )

    trained_path = tmp_path / "trained.pt"
    categorical.save(
        trained_path,
        value_training={
            "schema_version": "value-training-v1",
            "primary_readout": "categorical",
            "trained_value_readouts": ["categorical"],
            "resolved_scalar_mse_weight": 0.0,
            "resolved_categorical_ce_weight": 0.25,
            "hlgauss_bins": 9,
            "optimizer_steps": 10,
            "completed_epochs": 1,
            "scalar_training_weight_sum": 0.0,
            "categorical_training_weight_sum": 128.0,
        },
    )
    trained = EntityGraphPolicy.load(trained_path, device="cpu")
    assert trained.trained_value_readouts == ("categorical",)
    EntityGraphRustEvaluator(
        trained,
        config=EntityGraphRustEvaluatorConfig(value_readout="categorical"),
    )

    forged_path = tmp_path / "forged_zero_step.pt"
    categorical.save(
        forged_path,
        value_training={
            "schema_version": "value-training-v1",
            "primary_readout": "categorical",
            "trained_value_readouts": ["categorical"],
            "resolved_scalar_mse_weight": 0.0,
            "resolved_categorical_ce_weight": 0.25,
            "hlgauss_bins": 9,
            "optimizer_steps": 0,
            "completed_epochs": 1,
            "scalar_training_weight_sum": 0.0,
            "categorical_training_weight_sum": 128.0,
        },
    )
    forged = EntityGraphPolicy.load(forged_path, device="cpu")
    assert "categorical" not in forged.trained_value_readouts
    with pytest.raises(ValueError, match="no positive value-training-v1 provenance"):
        EntityGraphRustEvaluator(
            forged,
            config=EntityGraphRustEvaluatorConfig(value_readout="categorical"),
        )


def test_sync_and_evaluate_many_use_the_selected_readout(
    split_value_policy, evaluator_inputs
) -> None:
    game, legal = evaluator_inputs
    root = str(game.current_color())
    scalar = EntityGraphRustEvaluator(
        split_value_policy,
        config=EntityGraphRustEvaluatorConfig(value_readout="scalar", cache_size=0),
    )
    default = EntityGraphRustEvaluator(
        split_value_policy,
        config=EntityGraphRustEvaluatorConfig(cache_size=0),
    )
    categorical = EntityGraphRustEvaluator(
        split_value_policy,
        config=EntityGraphRustEvaluatorConfig(value_readout="categorical", cache_size=0),
    )

    scalar_result = scalar.evaluate(game, legal, root_color=root, colors=("RED", "BLUE"))
    assert default.evaluate(game, legal, root_color=root, colors=("RED", "BLUE")) == scalar_result
    assert scalar_result[1] == pytest.approx(math.tanh(0.75))
    assert categorical.evaluate(
        game, legal, root_color=root, colors=("RED", "BLUE")
    )[1] == pytest.approx(-0.25)
    assert categorical.evaluate_many(
        [(game, legal)], root_color=root, colors=("RED", "BLUE")
    )[0][1] == pytest.approx(-0.25)


def test_async_batched_worker_uses_categorical_readout(
    split_value_policy, evaluator_inputs
) -> None:
    game, legal = evaluator_inputs
    root = str(game.current_color())
    evaluator = BatchedEntityGraphRustEvaluator(
        split_value_policy,
        config=EntityGraphRustEvaluatorConfig(value_readout="categorical", cache_size=0),
        max_batch_size=8,
        max_wait_ms=0.0,
    )
    try:
        _priors, value = evaluator.evaluate(
            game, legal, root_color=root, colors=("RED", "BLUE")
        )
    finally:
        evaluator.close()
    assert value == pytest.approx(-0.25)


def test_d6_symmetry_averaging_uses_categorical_readout(
    split_value_policy, evaluator_inputs
) -> None:
    game, legal = evaluator_inputs
    root = str(game.current_color())
    evaluator = EntityGraphRustEvaluator(
        split_value_policy,
        config=EntityGraphRustEvaluatorConfig(value_readout="categorical", cache_size=0),
    )
    _priors, value = evaluator.evaluate_symmetry_averaged(
        game, legal, root_color=root, colors=("RED", "BLUE")
    )
    assert value == pytest.approx(-0.25)


def test_remote_eval_client_carries_categorical_head_metadata() -> None:
    from catan_zero.search.eval_server import RemoteEvalClient

    with pytest.raises(ValueError, match="requires a checkpoint with a trained HL-Gauss"):
        RemoteEvalClient(
            object(),
            object(),
            0,
            action_size=16,
            trained_with_masked_hidden_info=False,
            config=EntityGraphRustEvaluatorConfig(value_readout="categorical"),
        )
    client = RemoteEvalClient(
        object(),
        object(),
        0,
        action_size=16,
        trained_with_masked_hidden_info=False,
        value_categorical_bins=9,
        value_categorical_head_available=True,
        config=EntityGraphRustEvaluatorConfig(value_readout="categorical"),
    )
    assert client.config.value_readout == "categorical"


def test_generator_cli_and_typed_hash_record_value_readout() -> None:
    default = generate_cli.build_parser().parse_args(["--out-dir", "/tmp/readout-default"])
    selected = generate_cli.build_parser().parse_args(
        ["--out-dir", "/tmp/readout-cat", "--value-readout", "categorical"]
    )
    assert default.value_readout == "scalar"
    assert selected.value_readout == "categorical"
    assert GenerateConfig().value_readout == "scalar"
    assert GenerateConfig(value_readout="categorical").config_hash() != GenerateConfig().config_hash()
    assert EvalConfig(value_readout="categorical").config_hash() != EvalConfig().config_hash()
    mixed = EvalConfig(
        mode="cross_net",
        candidate_value_readout="categorical",
        baseline_value_readout="scalar",
    )
    shared = EvalConfig(
        mode="cross_net",
        candidate_value_readout="scalar",
        baseline_value_readout="scalar",
    )
    assert mixed.config_hash() != shared.config_hash()


def test_cross_net_readout_resolution_preserves_shared_fallback_and_role_overrides() -> None:
    assert cross_net_cli._resolve_value_readouts(
        SimpleNamespace(
            value_readout="scalar",
            candidate_value_readout=None,
            baseline_value_readout=None,
        )
    ) == ("scalar", "scalar")
    assert cross_net_cli._resolve_value_readouts(
        SimpleNamespace(
            value_readout="scalar",
            candidate_value_readout="categorical",
            baseline_value_readout=None,
        )
    ) == ("categorical", "scalar")
    assert cross_net_cli._resolve_value_readouts(
        {
            "value_readout": "categorical",
            "baseline_value_readout": "scalar",
        }
    ) == ("categorical", "scalar")


def test_generator_rejects_categorical_readout_without_checkpoint() -> None:
    from test_generate_gumbel_selfplay_data_cat12_flags import _worker_args

    with pytest.raises(ValueError, match="requires a neural checkpoint"):
        generate_cli._run_worker(_worker_args(value_readout="categorical"))


def test_generator_worker_and_cross_net_builder_thread_value_readout(monkeypatch) -> None:
    captured: list[EntityGraphRustEvaluatorConfig] = []

    def fake_from_checkpoint(_checkpoint, *, device, config, **_kwargs):
        del device
        captured.append(config)
        return object()

    monkeypatch.setattr(
        generate_cli.BatchedEntityGraphRustEvaluator,
        "from_checkpoint",
        fake_from_checkpoint,
    )
    monkeypatch.setattr(
        generate_cli,
        "run_worker_games",
        lambda **_kwargs: {
            "games_completed": 0,
            "games_failed": 0,
            "games_truncated": 0,
            "rows": 0,
            "decisions_total": 0,
            "forced_decisions_total": 0,
            "simulations_used_total": 0,
            "wins_by_color": {},
            "shards": [],
            "errors": [],
        },
    )
    from test_generate_gumbel_selfplay_data_cat12_flags import _worker_args

    generate_cli._run_worker(
        _worker_args(
            checkpoint="categorical.pt",
            value_readout="categorical",
            rust_featurize=False,
            eval_cache_size=0,
        )
    )
    assert captured[-1].value_readout == "categorical"

    monkeypatch.setattr(
        cross_net_cli.BatchedEntityGraphRustEvaluator,
        "from_checkpoint",
        fake_from_checkpoint,
    )
    cross_net_cli._build_evaluator(
        "categorical.pt",
        {
            "device": "cpu",
            "value_scale": 1.0,
            "prior_temperature": 1.0,
            "value_squash": "tanh",
            "value_readout": "categorical",
            "public_observation": False,
        },
    )
    assert captured[-1].value_readout == "categorical"

    mixed_worker_args = {
        "device": "cpu",
        "value_scale": 1.0,
        "prior_temperature": 1.0,
        "value_squash": "tanh",
        "value_readout": "scalar",
        "candidate_value_readout": "categorical",
        "baseline_value_readout": "scalar",
        "public_observation": False,
    }
    cross_net_cli._build_evaluator(
        "candidate.pt", mixed_worker_args, role="candidate"
    )
    cross_net_cli._build_evaluator(
        "baseline.pt", mixed_worker_args, role="baseline"
    )
    assert [config.value_readout for config in captured[-2:]] == [
        "categorical",
        "scalar",
    ]
