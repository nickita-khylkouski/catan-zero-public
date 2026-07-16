"""Generator-side production-safety wiring for resolved opponent mixes."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_gumbel_selfplay_data as cli  # type: ignore  # noqa: E402


def test_generator_failfast_resolution_binds_mix_to_producer_bytes(tmp_path: Path) -> None:
    producer = tmp_path / "producer.pt"
    producer.write_bytes(b"producer-weights")
    aliased_opponent = tmp_path / "old-name.pt"
    aliased_opponent.write_bytes(producer.read_bytes())
    manifest = tmp_path / "mix.json"
    manifest.write_text(
        json.dumps(
            {
                "categories": [
                    {"name": "producer_self_play", "weight": 90, "source": "self"},
                    {
                        "name": "previous_public_champion",
                        "weight": 10,
                        "source": "checkpoint_list",
                        "checkpoints": [{"path": str(aliased_opponent)}],
                    },
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="producer checkpoint"):
        cli._resolve_mix_with_exploiter(
            str(manifest),
            None,
            producer_checkpoint=str(producer),
        )


def test_worker_uses_main_process_resolved_config_without_registry_reresolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolved config is handed through unchanged; workers do not race a
    mutable registry. Stop immediately after mix construction so no engine or
    checkpoint load is needed for this wiring regression."""
    from catan_zero.rl.flywheel.opponent_mix import MixCategory, OpponentMixConfig

    resolved = OpponentMixConfig(
        categories=(MixCategory(name="producer_self_play", weight=1, source="self"),)
    )

    def _must_not_resolve(*_args, **_kwargs):
        raise AssertionError("worker re-resolved the mutable manifest")

    monkeypatch.setattr(cli, "_resolve_mix_with_exploiter", _must_not_resolve)

    # A deliberately incomplete worker dict reaches evaluator construction
    # before mix construction, so intercept the evaluator and then the runtime.
    class _StopAfterMix(RuntimeError):
        pass

    evaluator_configs = []

    class _FakeEvaluator:
        def __init__(self):
            self.policy = SimpleNamespace(config=SimpleNamespace())

        @classmethod
        def from_checkpoint(cls, *_args, **kwargs):
            evaluator_configs.append(kwargs["config"])
            return cls()

    class _FakeMixRuntime:
        def __init__(self, *, config, evaluator_factory):
            assert config is resolved
            assert callable(evaluator_factory)
            # Construct one mixed-opponent evaluator now so this wiring test
            # proves it receives the same Rust featurizer setting as the
            # producer evaluator rather than stopping at a merely callable
            # closure.
            evaluator_factory("opponent.pt")
            assert evaluator_configs[-1].rust_featurize is True
            raise _StopAfterMix

    monkeypatch.setattr(cli, "BatchedEntityGraphRustEvaluator", _FakeEvaluator)
    monkeypatch.setattr(cli, "MixRuntime", _FakeMixRuntime)
    worker_args = {
        "checkpoint": "not-opened-by-fake.pt",
        "device": "cpu",
        "value_scale": 1.0,
        "prior_temperature": 1.0,
        "value_readout": "scalar",
        "public_observation": True,
        "rust_featurize": True,
        "eval_cache_size": 1,
        "score_actions": False,
        "opponent_pool_manifest": None,
        "opponent_mix_manifest": "mutable-registry-manifest.json",
        "opponent_mix_config": resolved,
    }

    with pytest.raises(_StopAfterMix):
        cli._run_worker(worker_args)
