"""Search-consistent value materialization and provenance contracts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import reanalyze_lite as rl  # type: ignore  # noqa: E402
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)


def test_scalar_materialization_matches_search_tanh_scale_and_final_clip() -> None:
    torch = pytest.importorskip("torch")
    spec = rl.resolve_root_value_materialization(
        value_readout="scalar", value_squash="tanh", value_scale=2.0
    )
    raw = np.asarray([-3.0, -0.25, 0.0, 0.25, 3.0], dtype=np.float32)

    got = rl.materialize_search_root_values({"value": torch.tensor(raw)}, spec)

    np.testing.assert_allclose(got, np.tanh(raw * 2.0), rtol=0, atol=1e-7)
    assert spec["forward_output"] == "value"
    assert spec["applied_value_squash"] == "tanh"
    assert np.all((-1.0 <= got) & (got <= 1.0))


def test_scalar_clip_and_categorical_readout_match_search_without_fallback() -> None:
    torch = pytest.importorskip("torch")
    outputs = {
        "value": torch.tensor([-0.4, 0.4]),
        "value_categorical": torch.tensor([-0.8, 0.8]),
    }
    scalar = rl.resolve_root_value_materialization(
        value_readout="scalar", value_squash="clip", value_scale=2.0
    )
    categorical = rl.resolve_root_value_materialization(
        value_readout="categorical", value_squash="tanh", value_scale=2.0
    )

    np.testing.assert_array_equal(
        rl.materialize_search_root_values(outputs, scalar),
        np.asarray([-0.8, 0.8], dtype=np.float32),
    )
    # Categorical search deliberately bypasses scalar tanh and final-clips.
    np.testing.assert_array_equal(
        rl.materialize_search_root_values(outputs, categorical),
        np.asarray([-1.0, 1.0], dtype=np.float32),
    )
    assert categorical["forward_output"] == "value_categorical"
    assert categorical["applied_value_squash"] == "none"


@pytest.mark.parametrize("readout", ["scalar", "categorical"])
@pytest.mark.parametrize("squash", ["tanh", "clip"])
def test_materializer_is_locked_to_the_real_search_squash_contract(
    readout: str,
    squash: str,
) -> None:
    torch = pytest.importorskip("torch")
    raw = np.asarray([-2.0, -0.2, 0.0, 0.4, 3.0], dtype=np.float32)
    config = EntityGraphRustEvaluatorConfig(
        value_readout=readout,
        value_squash=squash,
        value_scale=1.7,
    )
    evaluator = object.__new__(EntityGraphRustEvaluator)
    evaluator.config = config
    expected = np.asarray(
        [np.clip(evaluator._apply_value_squash(float(value)), -1.0, 1.0) for value in raw],
        dtype=np.float32,
    )
    key = "value" if readout == "scalar" else "value_categorical"
    spec = rl.resolve_root_value_materialization(
        value_readout=readout,
        value_squash=squash,
        value_scale=1.7,
    )

    got = rl.materialize_search_root_values({key: torch.tensor(raw)}, spec)

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"value_readout": "other"}, "value-readout"),
        ({"value_squash": "sigmoid"}, "value-squash"),
        ({"value_scale": 0.0}, "value-scale"),
        ({"value_scale": float("nan")}, "value-scale"),
    ],
)
def test_materialization_config_fails_closed(kwargs: dict, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        rl.resolve_root_value_materialization(**kwargs)


def test_materialization_rejects_missing_readout_and_nonfinite_output() -> None:
    torch = pytest.importorskip("torch")
    categorical = rl.resolve_root_value_materialization(value_readout="categorical")
    with pytest.raises(SystemExit, match="requires forward output 'value_categorical'"):
        rl.materialize_search_root_values({"value": torch.tensor([0.0])}, categorical)

    scalar = rl.resolve_root_value_materialization()
    with pytest.raises(SystemExit, match="non-finite"):
        rl.materialize_search_root_values(
            {"value": torch.tensor([float("nan")])}, scalar
        )


class _TinyCorpus:
    def __init__(self, n: int) -> None:
        self.n = n
        self.meta = {"columns": {}}

    def __len__(self) -> int:
        return self.n

    def __contains__(self, key: str) -> bool:
        return False


def test_rewrite_embeds_durable_provenance_and_rejects_bypassed_range(
    tmp_path: Path,
) -> None:
    corpus = _TinyCorpus(3)
    (tmp_path / "corpus_meta.json").write_text(
        json.dumps({"columns": {}}), encoding="utf-8"
    )
    spec = rl.resolve_root_value_materialization()

    with pytest.raises(SystemExit, match="out-of-range"):
        rl.rewrite_per_state_column(
            corpus,
            tmp_path,
            "root_value",
            np.asarray([0.0, 1.01, -0.5], dtype=np.float32),
            column_provenance=spec,
        )

    result = rl.rewrite_per_state_column(
        corpus,
        tmp_path,
        "root_value",
        np.asarray([0.0, 1.0, -1.0], dtype=np.float32),
        column_provenance=spec,
    )
    meta = json.loads((tmp_path / "corpus_meta.json").read_text(encoding="utf-8"))
    schema = meta["columns"]["root_value"]
    assert result["meta_changed"] is True
    assert schema["target_semantics"] == rl.ROOT_VALUE_TARGET_SEMANTICS
    assert schema["materialization"] == spec


def test_legacy_or_tampered_root_value_provenance_is_rejected() -> None:
    with pytest.raises(SystemExit, match="legacy/raw-forward"):
        rl.validate_root_value_materialization(None, v_component="root_value")

    tampered = rl.resolve_root_value_materialization()
    tampered["forward_output"] = "value_categorical"
    with pytest.raises(SystemExit, match="mismatched"):
        rl.validate_root_value_materialization(tampered, v_component="root_value")
