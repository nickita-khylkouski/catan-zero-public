import numpy as np
import pytest

from catan_zero.rl.pipeline_configs import TrainConfig
from tools import train_bc


MODE = train_bc.POLICY_AUX_SAMPLING_WEIGHTED_CYCLES_V1


def _order(weights, *, draws, seed=7, ddp=None, offset=0):
    return train_bc._policy_aux_epoch_order(  # noqa: SLF001
        np.random.default_rng(seed),
        len(weights),
        np.asarray(weights, dtype=np.float64),
        local_draws=draws,
        ddp=ddp or {"enabled": False, "world_size": 1, "rank": 0},
        mode=MODE,
        global_draw_offset=offset,
    )


def test_weighted_cycle_has_no_duplicates_before_exhaustion() -> None:
    order = _order([0.0, 1.0, 4.0, 2.0, 0.5], draws=8)

    assert len(set(order[:4].tolist())) == 4
    assert len(set(order[4:8].tolist())) == 4


def test_weighted_cycle_order_is_seeded_and_weight_sensitive() -> None:
    weights = [0.0, 1.0, 4.0, 2.0, 0.5]

    assert _order(weights, draws=4, seed=7).tolist() == [4, 2, 3, 1]
    assert np.array_equal(
        _order(weights, draws=4, seed=7),
        _order(weights, draws=4, seed=7),
    )
    assert not np.array_equal(
        _order(weights, draws=4, seed=7),
        _order(weights, draws=4, seed=81),
    )


def test_ddp_rank_stride_reinterleaves_to_one_global_weighted_stream() -> None:
    weights = [1.0, 3.0, 2.0, 0.0, 0.5]
    world = 3
    local_draws = 5
    rank_orders = [
        _order(
            weights,
            draws=local_draws,
            seed=19,
            ddp={"enabled": True, "world_size": world, "rank": rank},
        )
        for rank in range(world)
    ]
    reinterleaved = np.column_stack(rank_orders).reshape(-1)
    global_order = _order(
        weights,
        draws=local_draws * world,
        seed=19,
    )

    assert np.array_equal(reinterleaved, global_order)


def test_weighted_cycles_repeat_only_after_each_eligible_row() -> None:
    order = _order([1.0, 2.0, 0.0], draws=5, seed=31)

    assert set(order[:2].tolist()) == {0, 1}
    assert set(order[2:4].tolist()) == {0, 1}
    assert order[4] in {0, 1}
    counts = np.bincount(order, minlength=3)
    assert int(counts.max()) == 3


def test_cycle_continues_across_epoch_slices_before_restart() -> None:
    weights = [1.0, 4.0, 2.0, 0.5]
    first = _order(weights, draws=2, seed=43, offset=0)
    second = _order(weights, draws=2, seed=43, offset=2)
    combined = np.concatenate((first, second))

    assert len(set(combined.tolist())) == 4
    assert np.array_equal(combined, _order(weights, draws=4, seed=43))


def test_full_partial_and_resumed_epochs_match_one_uninterrupted_global_stream() -> None:
    weights = [1.0, 4.0, 2.0, 0.5, 3.0]
    seed = 47
    ddp = {"enabled": True, "world_size": 2}
    offset = 0
    segments = []

    # A full epoch, a capped trailing epoch, then a resumed epoch. The persisted
    # offset must describe actual globally consumed draws, not
    # epoch_number*current_epoch_length.
    for local_draws in (4, 1):
        rank_orders = [
            _order(
                weights,
                draws=local_draws,
                seed=seed,
                ddp={**ddp, "rank": rank},
                offset=offset,
            )
            for rank in range(2)
        ]
        segments.append(np.column_stack(rank_orders).reshape(-1))
        offset = train_bc._advance_policy_aux_global_draw_offset(  # noqa: SLF001
            offset,
            local_draws=local_draws,
            ddp={**ddp, "rank": 0},
        )

    resumed_offset = train_bc._restore_policy_aux_global_draw_offset(  # noqa: SLF001
        {"policy_aux_global_draw_offset": offset},
        required=True,
    )
    resumed_local_draws = 3
    resumed_rank_orders = [
        _order(
            weights,
            draws=resumed_local_draws,
            seed=seed,
            ddp={**ddp, "rank": rank},
            offset=resumed_offset,
        )
        for rank in range(2)
    ]
    segments.append(np.column_stack(resumed_rank_orders).reshape(-1))
    offset = train_bc._advance_policy_aux_global_draw_offset(  # noqa: SLF001
        resumed_offset,
        local_draws=resumed_local_draws,
        ddp={**ddp, "rank": 0},
    )

    realized = np.concatenate(segments)
    uninterrupted = _order(weights, draws=offset, seed=seed)

    assert realized.tolist() == uninterrupted.tolist()
    assert offset == (4 + 1 + 3) * 2


def test_resumed_midcycle_slice_uses_slice_relative_reuse_and_coverage_bounds() -> None:
    short = train_bc._policy_aux_stream_slice_contract(  # noqa: SLF001
        global_draw_offset=4,
        global_draws=5,
        eligible_rows=5,
    )
    with_full_cycle = train_bc._policy_aux_stream_slice_contract(  # noqa: SLF001
        global_draw_offset=4,
        global_draws=6,
        eligible_rows=5,
    )

    # Five draws beginning at the final row of one cycle touch two cycles and
    # may repeat a row, but do not contain any complete cycle.
    assert short["maximum_source_row_reuse"] == 2
    assert short["contains_complete_cycle"] is False
    # One more draw completes the next aligned cycle, proving full coverage.
    assert with_full_cycle["maximum_source_row_reuse"] == 2
    assert with_full_cycle["contains_complete_cycle"] is True


def test_zero_weight_rows_are_never_drawn() -> None:
    order = _order([0.0, 5.0, 0.0, 1.0, 0.0], draws=100, seed=5)

    assert set(order.tolist()) == {1, 3}


def test_cycle_report_binds_coverage_ess_and_reuse_cap() -> None:
    report = train_bc._policy_aux_sampling_cycle_report(  # noqa: SLF001
        np.asarray([0.0, 1.0, 2.0, 1.0]),
        local_draws=4,
        ddp={"enabled": True, "world_size": 2, "rank": 0},
        mode=MODE,
    )

    assert report["global_draws"] == 8
    assert report["eligible_positive_mass_rows"] == 3
    assert report["sampling_weight_effective_sample_size"] == pytest.approx(
        16.0 / 6.0
    )
    assert report["complete_cycles_before_slice"] == 0
    assert report["complete_cycles_after_slice"] == 2
    assert report["cycle_boundaries_crossed"] == 2
    assert report["partial_cycle_draws_at_end"] == 2
    assert report["maximum_source_row_reuse_by_construction"] == 3
    assert report["duplicates_before_cycle_exhaustion"] is False


def test_train_config_binds_policy_aux_sampling_mode() -> None:
    assert (
        TrainConfig().policy_aux_sampling_mode
        == train_bc.POLICY_AUX_SAMPLING_LEGACY_REPLACEMENT_V1
    )
    assert TrainConfig().full_config_hash() != TrainConfig(
        policy_aux_sampling_mode=MODE
    ).full_config_hash()
