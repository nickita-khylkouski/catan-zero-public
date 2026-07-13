from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from catan_zero.rl import entity_token_features_rust as feature_path
from catan_zero.rl.pipeline_configs import EvalConfig


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def test_complete_native_feature_api_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    incomplete = SimpleNamespace(
        EntityTopology=lambda *args: None,
        build_entity_features_flat=lambda *args: None,
        gumbel_search_capabilities=lambda: ["public_award_feature_parity"],
    )
    monkeypatch.setitem(sys.modules, "catanatron_rs", incomplete)

    assert feature_path.rust_feature_path_available() is False
    with pytest.raises(RuntimeError, match="build_action_context_flat.*refusing Python fallback"):
        feature_path.require_rust_feature_path()


def test_complete_native_feature_api_passes_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = SimpleNamespace(
        EntityTopology=lambda *args: None,
        build_entity_features_flat=lambda *args: None,
        build_action_context_flat=lambda *args: None,
        gumbel_search_capabilities=lambda: ["public_award_feature_parity"],
    )
    monkeypatch.setitem(sys.modules, "catanatron_rs", complete)

    assert feature_path.rust_feature_path_available() is True
    feature_path.require_rust_feature_path()


def test_native_feature_path_rejects_stale_public_award_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = SimpleNamespace(
        EntityTopology=lambda *args: None,
        build_entity_features_flat=lambda *args: None,
        build_action_context_flat=lambda *args: None,
        gumbel_search_capabilities=lambda: ["initial_road_d1_scope"],
    )
    monkeypatch.setitem(sys.modules, "catanatron_rs", stale)

    assert feature_path.rust_feature_path_available() is False
    with pytest.raises(RuntimeError, match="stale public-award semantics"):
        feature_path.require_rust_feature_path()


def test_eval_config_hash_seals_native_feature_choice() -> None:
    reference = EvalConfig(mode="cross_net", candidate="a.pt", baseline="b.pt")
    native = EvalConfig(
        mode="cross_net",
        candidate="a.pt",
        baseline="b.pt",
        evaluator_rust_featurize=True,
    )
    assert reference.evaluator_rust_featurize is False
    assert native.config_hash() != reference.config_hash()


def test_neutral_harness_seals_native_feature_path_in_recipe() -> None:
    from catanatron_neutral_harness_match import _search_recipe, build_parser

    args = build_parser().parse_args(
        [
            "--checkpoint",
            "candidate.pt",
            "--opponent",
            "random",
            "--mode",
            "search",
            "--evaluator-rust-featurize",
            "--out",
            "out.json",
        ]
    )
    assert args.evaluator_rust_featurize is True
    assert _search_recipe(args)["evaluator_rust_featurize"] is True


@pytest.mark.parametrize(
    "module_name,checkpoint_key",
    [
        ("gumbel_search_cross_net_h2h", "candidate.pt"),
        ("gumbel_search_vs_bot_h2h", "candidate.pt"),
        ("catanatron_neutral_harness_match", "candidate.pt"),
    ],
)
def test_canonical_evaluator_builders_thread_native_feature_flag(
    monkeypatch: pytest.MonkeyPatch, module_name: str, checkpoint_key: str
) -> None:
    module = __import__(module_name)
    captured = {}

    def fake_from_checkpoint(checkpoint, *, device, config):
        captured.update(checkpoint=checkpoint, device=device, config=config)
        return object()

    monkeypatch.setattr(
        module.BatchedEntityGraphRustEvaluator,
        "from_checkpoint",
        fake_from_checkpoint,
    )
    worker_args = {
        "device": "cuda:0",
        "value_scale": 1.0,
        "prior_temperature": 1.0,
        "value_squash": "tanh",
        "value_readout": "scalar",
        "public_observation": False,
        "evaluator_rust_featurize": True,
        "checkpoint": checkpoint_key,
    }
    if module_name == "gumbel_search_cross_net_h2h":
        module._build_evaluator(checkpoint_key, worker_args)
    elif module_name == "gumbel_search_vs_bot_h2h":
        module._build_evaluator(checkpoint_key, worker_args)
    else:
        module._build_evaluator(worker_args)

    assert captured["config"].rust_featurize is True
