"""Unit tests for `catan_zero.rl.flywheel.opponent_mix` (CAT-54): pure-stdlib
categorical opponent-mix sampling. No rust engine / torch needed."""
from __future__ import annotations

import json

import pytest

from catan_zero.rl.flywheel.opponent_mix import (
    MixCategory,
    MixCheckpointRef,
    OpponentMixConfig,
    choose_checkpoint_in_category,
    choose_mix_category,
    choose_mix_opponent,
    config_to_dict,
    read_opponent_mix_manifest,
    realized_mix_fractions,
)


def _r9_categories(*, catanatron_pending: bool = True) -> tuple[MixCategory, ...]:
    """The adopted CAT-5 R9 exact mix: 75/10/5/5/5."""
    return (
        MixCategory(name="producer_self_play", weight=75.0, source="self"),
        MixCategory(
            name="previous_public_champion",
            weight=10.0,
            source="checkpoint_list",
            checkpoints=(MixCheckpointRef(path="/arch/champion_v3.pt", version=3, md5="aaa"),),
        ),
        MixCategory(
            name="older_champion",
            weight=5.0,
            source="checkpoint_list",
            checkpoints=(
                MixCheckpointRef(path="/arch/champion_v0.pt", version=0, md5="bbb"),
                MixCheckpointRef(path="/arch/champion_v1.pt", version=1, md5="ccc"),
            ),
        ),
        MixCategory(
            name="hard_experimental",
            weight=5.0,
            source="checkpoint_list",
            checkpoints=(MixCheckpointRef(path="/arch/exploiter_v0.pt", version=-1, md5="ddd"),),
        ),
        MixCategory(
            name="catanatron_value",
            weight=5.0,
            source="external_engine",
            engine="catanatron_value",
            pending=catanatron_pending,
        ),
    )


# --------------------------------------------------------------------------- construction / validation
def test_mixcategory_rejects_negative_weight():
    with pytest.raises(ValueError):
        MixCategory(name="bad", weight=-1.0, source="self")


def test_mixcategory_rejects_unknown_source():
    with pytest.raises(ValueError):
        MixCategory(name="bad", weight=1.0, source="not_a_real_source")


def test_mixcategory_checkpoint_list_requires_checkpoints_unless_pending():
    with pytest.raises(ValueError):
        MixCategory(name="bad", weight=1.0, source="checkpoint_list", checkpoints=())
    # pending=True tolerates an empty checkpoint list.
    MixCategory(name="ok", weight=1.0, source="checkpoint_list", checkpoints=(), pending=True)


def test_mixcategory_external_engine_requires_engine_name():
    with pytest.raises(ValueError):
        MixCategory(name="bad", weight=1.0, source="external_engine", pending=True)


def test_mixcategory_non_pending_unwired_external_engine_is_not_implemented():
    """Scope boundary: an external_engine category with an UNWIRED engine name
    (not in WIRED_EXTERNAL_ENGINES) must still fail loudly at construction --
    never silently no-op or get sampled and then crash a worker. CAT-56 wires
    catanatron_value/ab3/ab4/ab5; anything else is still unimplemented.

    (Under CAT-54, EVERY external_engine was unwired and this raised for
    catanatron_value too; CAT-56 flipped that for the wired names -- see
    test_mixcategory_wired_external_engine_is_sampleable below.)"""
    with pytest.raises(NotImplementedError):
        MixCategory(name="bad", weight=1.0, source="external_engine", engine="not_a_real_bot", pending=False)


def test_mixcategory_wired_external_engine_is_sampleable():
    """CAT-56: a WIRED external engine (catanatron_value) now constructs
    non-pending and is sampled as an external opponent (is_external, carrying the
    engine name, no checkpoint)."""
    from catan_zero.rl.flywheel.opponent_mix import WIRED_EXTERNAL_ENGINES

    assert "catanatron_value" in WIRED_EXTERNAL_ENGINES
    category = MixCategory(
        name="catanatron_value", weight=5.0, source="external_engine", engine="catanatron_value"
    )
    assert category.is_external_engine is True
    assert category.is_effective is True
    config = OpponentMixConfig(
        categories=(MixCategory(name="self", weight=95.0, source="self"), category)
    )
    saw_external = False
    for game_index in range(3000):
        choice = choose_mix_opponent(game_index, config.categories)
        if choice.tag == "catanatron_value":
            assert choice.is_external is True
            assert choice.engine == "catanatron_value"
            assert choice.is_pool is True  # opponent occupies the non-producer seat
            assert choice.path == "" and choice.version == -1  # no checkpoint
            saw_external = True
    assert saw_external, "the wired external engine must actually be sampled"


def test_opponentmixconfig_rejects_duplicate_category_names():
    with pytest.raises(ValueError):
        OpponentMixConfig(
            categories=(
                MixCategory(name="dup", weight=1.0, source="self"),
                MixCategory(name="dup", weight=1.0, source="self"),
            )
        )


def test_opponentmixconfig_rejects_all_pending_or_zero_weight():
    with pytest.raises(ValueError):
        OpponentMixConfig(
            categories=(
                MixCategory(name="a", weight=0.0, source="self"),
                MixCategory(name="b", weight=1.0, source="self", pending=True),
            )
        )


# --------------------------------------------------------------------------- effective_weights
def test_effective_weights_renormalize_over_non_pending_categories():
    config = OpponentMixConfig(categories=_r9_categories(catanatron_pending=True))
    weights = config.effective_weights()
    assert "catanatron_value" not in weights
    assert set(weights) == {
        "producer_self_play",
        "previous_public_champion",
        "older_champion",
        "hard_experimental",
    }
    assert abs(weights["producer_self_play"] - 75.0 / 95.0) < 1e-12
    assert abs(sum(weights.values()) - 1.0) < 1e-12


# --------------------------------------------------------------------------- sampling: determinism
def test_choose_mix_opponent_is_deterministic_given_game_index():
    categories = _r9_categories()
    for game_index in (0, 1, 2, 7, 42, 1000, 999_999):
        first = choose_mix_opponent(game_index, categories)
        second = choose_mix_opponent(game_index, categories)
        assert first == second


def test_choose_mix_opponent_never_samples_a_pending_category():
    categories = _r9_categories(catanatron_pending=True)
    for game_index in range(3000):
        assert choose_mix_opponent(game_index, categories).tag != "catanatron_value"


def test_choose_mix_category_raises_with_no_effective_categories():
    with pytest.raises(ValueError):
        choose_mix_category(0, (MixCategory(name="a", weight=1.0, source="self", pending=True),))


# --------------------------------------------------------------------------- sampling: distribution
def test_mix_sampling_distribution_matches_configured_weights_seeded():
    """Mix-sampling distribution test (seeded): over many deterministic draws,
    each effective category's realized fraction should land close to its
    configured (renormalized) weight -- this is the exact-mix correctness
    property the ticket asks for (75/10/5/5/5, or here 75/10/5/5 once
    catanatron_value's pending 5% is excluded)."""
    categories = _r9_categories(catanatron_pending=True)
    fractions = realized_mix_fractions(50_000, categories)
    expected = OpponentMixConfig(categories=categories).effective_weights()
    for name, expected_fraction in expected.items():
        assert abs(fractions[name] - expected_fraction) < 0.01, (name, fractions[name], expected_fraction)


def test_choose_checkpoint_in_category_is_deterministic_and_covers_all_entries():
    category = MixCategory(
        name="older_champion",
        weight=5.0,
        source="checkpoint_list",
        checkpoints=(
            MixCheckpointRef(path="/a.pt", version=0),
            MixCheckpointRef(path="/b.pt", version=1),
            MixCheckpointRef(path="/c.pt", version=2),
        ),
    )
    seen_paths = set()
    for game_index in range(3000):
        ref = choose_checkpoint_in_category(game_index, category)
        assert ref == choose_checkpoint_in_category(game_index, category)
        seen_paths.add(ref.path)
    assert seen_paths == {"/a.pt", "/b.pt", "/c.pt"}


def test_choose_mix_opponent_self_category_has_no_checkpoint():
    categories = (MixCategory(name="producer_self_play", weight=1.0, source="self"),)
    choice = choose_mix_opponent(0, categories)
    assert choice.is_pool is False
    assert choice.tag == "producer_self_play"
    assert choice.path == ""
    assert choice.version == -1


# --------------------------------------------------------------------------- manifest I/O
def test_read_opponent_mix_manifest_round_trips(tmp_path):
    manifest = {
        "categories": [
            {"name": "producer_self_play", "weight": 75, "source": "self"},
            {
                "name": "previous_public_champion",
                "weight": 10,
                "source": "checkpoint_list",
                "checkpoints": [{"path": "/arch/v3.pt", "version": 3, "md5": "aaa"}],
            },
            {
                "name": "catanatron_value",
                "weight": 5,
                "source": "external_engine",
                "engine": "catanatron_value",
                "pending": True,
            },
        ]
    }
    manifest_path = tmp_path / "mix.json"
    manifest_path.write_text(json.dumps(manifest))

    config = read_opponent_mix_manifest(manifest_path)
    assert {c.name for c in config.categories} == {
        "producer_self_play",
        "previous_public_champion",
        "catanatron_value",
    }
    dumped = config_to_dict(config)
    assert dumped["effective_weights"]["producer_self_play"] == pytest.approx(75.0 / 85.0)


def test_read_opponent_mix_manifest_rejects_empty_categories(tmp_path):
    manifest_path = tmp_path / "mix.json"
    manifest_path.write_text(json.dumps({"categories": []}))
    with pytest.raises(ValueError):
        read_opponent_mix_manifest(manifest_path)
