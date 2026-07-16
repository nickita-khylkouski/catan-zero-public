from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.pipeline_configs import TrainConfig
from tools import train_bc


def _identity() -> dict[str, object]:
    _key, identity = train_bc._derived_array_cache_key(
        {
            "data_fingerprint": "sha256:" + "a" * 64,
            "row_count": 6,
            "recipe": {"per_game_value_weight": True},
        }
    )
    return identity


def test_derived_array_cache_round_trip_is_exact_read_only_mmap(tmp_path):
    arrays = {
        "train_indices": np.asarray([5, 1, 4, 2], dtype=np.int64),
        "policy_sample_weights": np.asarray([0.0, 0.5, 2.0], dtype=np.float32),
    }
    identity = _identity()
    directory = train_bc._write_derived_array_cache(tmp_path, identity, arrays)
    loaded = train_bc._load_derived_array_cache(tmp_path, identity)

    assert directory.name == train_bc._canonical_json_sha256(identity).split(":", 1)[1]
    assert set(loaded) == set(arrays)
    for name, expected in arrays.items():
        assert isinstance(loaded[name], np.memmap)
        assert loaded[name].flags.writeable is False
        assert loaded[name].dtype == expected.dtype
        assert np.array_equal(loaded[name], expected)


def test_derived_array_cache_canonicalizes_tuple_inputs_for_json_round_trip(
    tmp_path,
):
    _key, identity = train_bc._derived_array_cache_key(
        {
            "validation_game_seed_ranges": [(10, 19), (30, 39)],
            "nested": {"tuple": ("a", 2)},
        }
    )
    assert identity["inputs"]["validation_game_seed_ranges"] == [
        [10, 19],
        [30, 39],
    ]
    directory = train_bc._write_derived_array_cache(
        tmp_path,
        identity,
        {"train_indices": np.arange(3, dtype=np.int64)},
    )
    assert directory.is_dir()
    loaded = train_bc._load_derived_array_cache(tmp_path, identity)
    np.testing.assert_array_equal(loaded["train_indices"], [0, 1, 2])


def test_derived_array_cache_fails_closed_on_payload_tamper(tmp_path):
    identity = _identity()
    directory = train_bc._write_derived_array_cache(
        tmp_path, identity, {"train_indices": np.arange(8, dtype=np.int64)}
    )
    payload = directory / "train_indices.npy"
    payload.chmod(0o644)
    with open(payload, "r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 0x01]))

    with pytest.raises(SystemExit, match="file digest mismatch"):
        train_bc._load_derived_array_cache(tmp_path, identity)


def test_derived_array_cache_fails_closed_on_inventory_tamper(tmp_path):
    identity = _identity()
    directory = train_bc._write_derived_array_cache(
        tmp_path, identity, {"train_indices": np.arange(3, dtype=np.int64)}
    )
    manifest_path = directory / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["arrays"]["train_indices"]["shape"] = [99]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SystemExit, match="shape/dtype drift"):
        train_bc._load_derived_array_cache(tmp_path, identity)


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("policy_loss_weight", 0.0),
        ("final_vp_loss_weight", 0.0),
        ("q_loss_weight", 0.25),
        ("policy_kl_anchor_weight", 0.25),
        ("value_uncertainty_loss_weight", 0.25),
        ("aux_subgoal_loss_weight", 0.25),
        ("belief_resource_loss_weight", 0.25),
        ("moe_routed_experts", 4),
    ),
)
def test_derived_array_cache_binds_coverage_scope_objectives(
    field: str,
    changed: float | int,
) -> None:
    baseline = SimpleNamespace(
        **TrainConfig().field_values(),
        base_sampler=train_bc.BASE_SAMPLER_COVERAGE_IMPORTANCE_V1,
    )
    treatment = SimpleNamespace(**vars(baseline))
    setattr(treatment, field, changed)

    baseline_fields = train_bc._derived_training_scope_cache_fields(  # noqa: SLF001
        baseline,
        resolved_scalar_value_loss_weight=0.1,
        resolved_categorical_value_loss_weight=0.0,
    )
    treatment_fields = train_bc._derived_training_scope_cache_fields(  # noqa: SLF001
        treatment,
        resolved_scalar_value_loss_weight=0.1,
        resolved_categorical_value_loss_weight=0.0,
    )

    assert train_bc._derived_array_cache_key(baseline_fields)[0] != (  # noqa: SLF001
        train_bc._derived_array_cache_key(treatment_fields)[0]  # noqa: SLF001
    )


def test_derived_array_cache_binds_resolved_value_objectives() -> None:
    args = SimpleNamespace(
        **TrainConfig().field_values(),
        base_sampler=train_bc.BASE_SAMPLER_COVERAGE_IMPORTANCE_V1,
    )
    identities = []
    for scalar_weight, categorical_weight in (
        (0.1, 0.0),
        (0.2, 0.0),
        (0.1, 0.25),
    ):
        fields = train_bc._derived_training_scope_cache_fields(  # noqa: SLF001
            args,
            resolved_scalar_value_loss_weight=scalar_weight,
            resolved_categorical_value_loss_weight=categorical_weight,
        )
        identities.append(train_bc._derived_array_cache_key(fields)[0])  # noqa: SLF001

    assert len(set(identities)) == len(identities)


def test_coverage_rejects_unsupported_objective_before_cache_build() -> None:
    args = SimpleNamespace(
        **TrainConfig(q_loss_weight=0.25).field_values(),
        base_sampler=train_bc.BASE_SAMPLER_COVERAGE_IMPORTANCE_V1,
    )

    with pytest.raises(SystemExit, match="q_loss_weight"):
        train_bc._validate_coverage_sampler_configuration(  # noqa: SLF001
            args,
            categorical_value_loss_weight=0.0,
        )


def test_policy_signal_floor_cannot_be_inert_under_weighted_sampler() -> None:
    args = SimpleNamespace(
        **TrainConfig().field_values(),
        base_sampler=train_bc.BASE_SAMPLER_WEIGHTED_REPLACEMENT_V1,
        minimum_policy_effective_rows_per_global_batch=32.0,
    )

    with pytest.raises(SystemExit, match="requires --base-sampler"):
        train_bc._validate_coverage_sampler_configuration(  # noqa: SLF001
            args,
            categorical_value_loss_weight=0.0,
        )


def test_policy_signal_floor_requires_enabled_policy_objective() -> None:
    args = SimpleNamespace(
        **TrainConfig(policy_loss_weight=0.0).field_values(),
        base_sampler=train_bc.BASE_SAMPLER_COVERAGE_IMPORTANCE_V1,
        minimum_policy_effective_rows_per_global_batch=32.0,
    )

    with pytest.raises(SystemExit, match="positive --policy-loss-weight"):
        train_bc._validate_coverage_sampler_configuration(  # noqa: SLF001
            args,
            categorical_value_loss_weight=0.0,
        )


def test_weighted_epoch_cap_is_exact_historical_prefix():
    n = 10_000
    weights = np.linspace(0.1, 2.0, n, dtype=np.float64)
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    full = train_bc._epoch_order(
        np.random.default_rng(123), n, 512, ddp, sample_weights=weights
    )
    capped = train_bc._epoch_order(
        np.random.default_rng(123),
        n,
        512,
        ddp,
        sample_weights=weights,
        max_samples=2_048,
    )
    assert np.array_equal(capped, full[:2_048])


def test_uniform_epoch_cap_is_ignored_to_preserve_permutation_semantics():
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    uncapped = train_bc._epoch_order(np.random.default_rng(7), 100, 8, ddp)
    requested_cap = train_bc._epoch_order(
        np.random.default_rng(7), 100, 8, ddp, max_samples=10
    )
    assert len(requested_cap) == 100
    assert np.array_equal(requested_cap, uncapped)
