"""Transcript-equivalence tests between catanatron_rs (Rust) and vendor/catanatron
(Python reference engine).

See `src/catan_zero/adapters/engine_equivalence.py` for the harness design notes
(map alignment, seating alignment, chance-outcome forcing) and
`tools/engine_equivalence_sweep.py` for the bulk CLI sweep.

Four genuine engine divergences were found while building this harness. Two
(#1, #4) were confirmed Rust-side chance-spectrum bugs, fixed upstream in
catanatron_rs 0.1.2; the tests below now assert the *fixed* behavior (spectra
are hand-weighted / card-count-weighted with no phantom outcomes) while still
tolerating a pre-fix wheel via `apply_chance_step`'s materialize-and-diff
handling (see its module-level docstring). Two (#2, #3) are pinned as
characterization tests that assert only that the divergence is real and
correctly bucketed for rules adjudication, not that one engine is correct:

1. (FIXED in catanatron_rs 0.1.2) Rust's MOVE_ROBBER-with-victim chance
   spectrum used to be uniform over the 5 resource types regardless of the
   victim's actual hand, with a silent no-op for resources not held. Fixed
   wheels return exactly the hand-weighted outcomes that can really happen.
2. Rust's longest-road award transfers on an exact tie (Python's road-length
   bookkeeping requires strictly-greater length to transfer the bonus).
   Repro: seed=2, TOURNAMENT map, RED/BLUE, step 392.
3. Rust disallows a BUILD_ROAD that Python's `buildable_edges` permits: once a
   node is reached by a player's road, Python keeps it in that player's
   connected-component set even if an opponent later places an (initial-phase)
   settlement there, permitting further "through the opponent's building"
   road-building. Repro: seed=5, TOURNAMENT map, RED/BLUE, step 997.

   NOTE on #2 and #3: a 2026-07-02 rules audit independently found pre-existing
   bugs in the vendored Python engine's own longest-road/road-buildability
   logic (upstream issues #376 and #378 -- roads buildable through enemy
   settlements, and both-ends-enemy-capped roads undercounting length by 1),
   since fixed (A15-A17). Even with both sides patched, this harness still
   does NOT auto-attribute #2/#3 to "Rust is wrong" or "Python is wrong" --
   it tags them `rules_adjudication_needed_longest_road` /
   `rules_adjudication_needed_buildable_edge_near_enemy` (see
   `classify_state_divergence_topic` / `classify_legal_action_mismatch_topic`)
   so that if either bug ever regresses, or a new one appears with the same
   shape, it's still routed for human adjudication rather than silently
   trusted either way.
4. (FIXED in catanatron_rs 0.1.2) Rust's BUY_DEVELOPMENT_CARD chance spectrum
   used to be able to return a "phantom" outcome with substantial nonzero
   probability that drew no card at all (deck count and hand unchanged).
   Fixed wheels never return a phantom outcome.
"""

from __future__ import annotations

import json

import pytest

from catan_zero.adapters.engine_equivalence import (
    EquivalenceConfig,
    RustModuleUnavailable,
    apply_chance_step,
    build_paired_games,
    canonical_python_action_key,
    canonical_rust_action_key,
    classify_legal_action_mismatch_topic,
    classify_state_divergence_topic,
    diff_state_views,
    is_chance_action,
    legal_action_diff,
    play_one_game,
    python_state_view,
    raw_action_to_python_action,
    require_rust_module,
    run_sweep,
    rust_legal_actions,
    rust_state_view,
    vendor_symbols,
)


def _rust():
    try:
        return require_rust_module()
    except RustModuleUnavailable as error:
        pytest.skip(str(error))


_FIXED_CATANATRON_RS_VERSION = (0, 1, 2)


def _require_fixed_rust_wheel():
    """Skip (rather than fail) tests that assert the *post-fix* chance-spectrum
    behavior when the installed catanatron_rs predates the fix (< 0.1.2). Older
    wheels are still exercised by the version-agnostic tests elsewhere in this
    file (e.g. `apply_chance_step`'s materialize-and-diff handling, and the
    "never misclassified" tests), just not held to the corrected-spectrum
    standard these specific tests check.
    """
    _rust()
    import importlib.metadata as metadata

    try:
        raw_version = metadata.version("catanatron-rs")
    except metadata.PackageNotFoundError:
        pytest.skip("catanatron-rs package metadata not found; cannot check fix version")
        return
    parts = tuple(int(part) for part in raw_version.split(".")[:3])
    if parts < _FIXED_CATANATRON_RS_VERSION:
        pytest.skip(
            f"installed catanatron_rs {raw_version} predates the chance-spectrum fix "
            f"({'.'.join(map(str, _FIXED_CATANATRON_RS_VERSION))}); skipping post-fix assertion"
        )


# --------------------------------------------------------------------------
# Map + setup alignment
# --------------------------------------------------------------------------


def test_tournament_map_tiles_match_between_engines():
    """The fixed TOURNAMENT map must be bit-identical (no independent shuffles)."""
    rs = _rust()
    from catan_zero.rl._catanatron import import_catanatron_module

    build_map = import_catanatron_module("catanatron.models.map").build_map

    python_map = build_map("TOURNAMENT")
    python_tiles = {coord: (tile.resource, tile.number) for coord, tile in python_map.land_tiles.items()}

    game = rs.Game(colors=["RED", "BLUE"], seed=123, map_kind="TOURNAMENT")
    snapshot = json.loads(game.json_snapshot())
    rust_tiles = {}
    for entry in snapshot["tiles"]:
        coord = tuple(entry["coordinate"])
        tile = entry["tile"]
        if tile["type"] == "RESOURCE_TILE":
            rust_tiles[coord] = (tile["resource"], tile["number"])
        elif tile["type"] == "DESERT":
            rust_tiles[coord] = (None, None)

    assert rust_tiles == python_tiles


def test_tournament_map_node_ids_match_between_engines():
    rs = _rust()
    from catan_zero.rl._catanatron import import_catanatron_module

    build_map = import_catanatron_module("catanatron.models.map").build_map

    python_map = build_map("TOURNAMENT")
    node_by_coord_dir = {}
    for coord, tile in python_map.tiles.items():
        if not hasattr(tile, "nodes"):
            continue
        for direction, node_id in tile.nodes.items():
            node_by_coord_dir[(coord, direction.value)] = node_id

    game = rs.Game(colors=["RED", "BLUE"], seed=123, map_kind="TOURNAMENT")
    snapshot = json.loads(game.json_snapshot())
    checked = 0
    for node_info in snapshot["nodes"].values():
        key = (tuple(node_info["tile_coordinate"]), node_info["direction"])
        expected = node_by_coord_dir.get(key)
        assert expected == int(node_info["id"]), f"node id mismatch at {key}"
        checked += 1
    assert checked == len(snapshot["nodes"])


def test_build_paired_games_seating_is_aligned():
    _rust()
    config = EquivalenceConfig()
    rust_game, python_game, seated_colors = build_paired_games(7, config)

    py_colors = tuple(color.name for color in python_game.state.colors)
    assert py_colors == seated_colors
    assert str(rust_game.current_color()) == seated_colors[0]
    assert python_game.state.current_color().name == seated_colors[0]


# --------------------------------------------------------------------------
# Chance outcome forcing
# --------------------------------------------------------------------------


def test_spectrum_probabilities_for_roll_are_exact():
    """Rust's ROLL chance spectrum must match the true two-dice distribution."""
    rs = _rust()
    config = EquivalenceConfig()
    rust_game, _python_game, seated_colors = build_paired_games(321, config)

    # Advance (rust-only) until a ROLL action is offered.
    for _ in range(60):
        ids = rust_game.playable_action_indices(list(seated_colors), config.map_kind)
        raw_actions = json.loads(rust_game.playable_actions_json())
        roll = next((raw for raw in raw_actions if raw[1] == "ROLL"), None)
        if roll is not None:
            break
        rust_game.execute_action_index(ids[0], list(seated_colors), config.map_kind)
    else:
        pytest.fail("did not reach a ROLL prompt")

    spectrum = json.loads(rust_game.spectrum_json(json.dumps(roll)))
    assert len(spectrum) == 11
    probabilities = [entry["probability"] for entry in spectrum]
    expected = [count / 36 for count in (1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1)]
    assert probabilities == pytest.approx(expected, abs=1e-12)


def test_move_robber_spectrum_is_hand_weighted_with_no_phantom_outcomes():
    """MOVE_ROBBER-with-victim spectrum outcomes must all be genuine
    single-resource steals (exactly one resource the victim holds decrements
    by 1, nothing else changes), with combined probability per resource
    proportional to that resource's true count in the victim's hand -- not
    uniform over all 5 resource types regardless of what's actually held.
    Pre-fix Rust wheels (< 0.1.2) failed this (see this file's module
    docstring, finding #1); this test holds the raw engine API to the
    corrected standard directly, independent of the harness's own
    materialize-and-diff workaround in `apply_chance_step`.
    """
    rs = _rust()
    _require_fixed_rust_wheel()
    game = rs.Game(colors=["RED", "BLUE"], seed=1, map_kind="TOURNAMENT")
    game.set_player_hand("BLUE", [3, 1, 0, 0, 0], [0, 0, 0, 0, 0])  # 3x WOOD, 1x BRICK
    action = ["RED", "MOVE_ROBBER", [[0, 0, 0], "BLUE"]]

    spectrum = json.loads(game.spectrum_json(json.dumps(action)))
    before = json.loads(game.json_snapshot())["player_state"][1]["resources"]

    probability_by_resource: dict[str, float] = {}
    for outcome_index, entry in enumerate(spectrum):
        outcome_game = game.apply_chance_outcome(json.dumps(action), outcome_index)
        after = json.loads(outcome_game.json_snapshot())["player_state"][1]["resources"]
        delta = {k: after[k] - before[k] for k in before if after[k] != before[k]}
        assert delta and list(delta.values()) == [-1], (
            f"outcome {outcome_index} was not a clean single-resource steal (phantom "
            f"no-op outcomes are the pre-fix bug this test guards against): delta={delta}"
        )
        (resource,) = delta.keys()
        probability_by_resource[resource] = probability_by_resource.get(resource, 0.0) + float(
            entry["probability"]
        )

    total_cards = sum(before.values())
    expected = {resource: count / total_cards for resource, count in before.items() if count > 0}
    assert set(probability_by_resource) == set(expected)
    for resource, expected_probability in expected.items():
        assert probability_by_resource[resource] == pytest.approx(expected_probability, abs=1e-9)


def test_buy_development_card_spectrum_has_no_phantom_outcomes_across_playthrough():
    """BUY_DEVELOPMENT_CARD spectrum outcomes must all be genuine single-card
    draws: deck count decrements by exactly 1 and exactly one dev-card type in
    the actor's hand increments by exactly 1. Pre-fix Rust wheels (< 0.1.2)
    could return a "phantom" outcome that drew nothing (finding #4); this
    replays the exact scenario that used to trigger it (seed=30) and asserts
    `apply_chance_step` never has to filter out a phantom outcome on a fixed
    wheel.
    """
    _require_fixed_rust_wheel()
    symbols = vendor_symbols()
    config = EquivalenceConfig(max_steps=3000)
    rust_game, python_game, seated_colors = build_paired_games(30, config)

    import random

    harness_rng = random.Random(30 * 2 + 1)
    buy_dev_card_events = 0
    for _step in range(1500):
        if rust_game.winning_color() is not None or python_game.winning_color() is not None:
            break
        ids, raw_actions = rust_legal_actions(rust_game, seated_colors, config.map_kind)
        choice = harness_rng.randrange(len(ids))
        chosen_id, chosen_raw = ids[choice], raw_actions[choice]
        if is_chance_action(chosen_raw):
            rust_game, outcome = apply_chance_step(
                rust_game, python_game, chosen_raw, symbols=symbols, harness_rng=harness_rng
            )
            if outcome.action_type == "BUY_DEVELOPMENT_CARD":
                buy_dev_card_events += 1
                assert outcome.detail["phantom_outcomes"] == [], (
                    "expected zero phantom BUY_DEVELOPMENT_CARD outcomes on a fixed wheel, "
                    f"got {outcome.detail!r}"
                )
        else:
            rust_game.execute_action_index(chosen_id, list(seated_colors), config.map_kind)
            python_game.execute(raw_action_to_python_action(chosen_raw, symbols))

    assert buy_dev_card_events > 0, (
        "expected at least one BUY_DEVELOPMENT_CARD chance event for seed=30 within "
        "1500 steps to actually exercise the check above"
    )
    # The harness must still keep both engines in sync.
    rust_view = rust_state_view(rust_game)
    python_view = python_state_view(python_game, symbols)
    assert diff_state_views(rust_view, python_view) == []


def test_apply_chance_step_forces_matching_dice_and_robber_outcomes():
    """The harness's own chance-forcing must keep both engines' hands in sync."""
    _rust()
    symbols = vendor_symbols()
    config = EquivalenceConfig()
    rust_game, python_game, seated_colors = build_paired_games(9, config)

    import random

    harness_rng = random.Random(0)
    steps = 0
    saw_roll = False
    while steps < 400 and not saw_roll:
        ids, raw_actions = rust_legal_actions(rust_game, seated_colors, config.map_kind)
        roll_slot = next(((i, raw) for i, raw in zip(ids, raw_actions) if raw[1] == "ROLL"), None)
        if roll_slot is not None:
            chosen_id, chosen_raw = roll_slot
        else:
            chosen_id, chosen_raw = ids[0], raw_actions[0]
        if is_chance_action(chosen_raw):
            rust_game, outcome = apply_chance_step(
                rust_game, python_game, chosen_raw, symbols=symbols, harness_rng=harness_rng
            )
            if outcome.action_type == "ROLL":
                saw_roll = True
        else:
            rust_game.execute_action_index(chosen_id, list(seated_colors), config.map_kind)
            python_game.execute(raw_action_to_python_action(chosen_raw, symbols))
        steps += 1

    assert saw_roll, "did not reach a ROLL action within the step budget"
    rust_view = rust_state_view(rust_game)
    python_view = python_state_view(python_game, symbols)
    assert diff_state_views(rust_view, python_view) == []


# --------------------------------------------------------------------------
# Legal action set + state equivalence over full games
# --------------------------------------------------------------------------


def test_canonical_action_keys_agree_for_matching_actions():
    _rust()
    symbols = vendor_symbols()
    raw = ["RED", "BUILD_ROAD", [4, 15]]
    py_action = raw_action_to_python_action(raw, symbols)
    assert canonical_rust_action_key(raw) == canonical_python_action_key(py_action)

    raw_reversed = ["RED", "BUILD_ROAD", [15, 4]]
    assert canonical_rust_action_key(raw) == canonical_rust_action_key(raw_reversed)


def _base_player(**overrides):
    player = {
        "victory_points": 2,
        "actual_victory_points": 2,
        "resources": {"WOOD": 1, "BRICK": 1, "SHEEP": 0, "WHEAT": 0, "ORE": 0},
        "dev_cards": {"KNIGHT": 0, "YEAR_OF_PLENTY": 0, "MONOPOLY": 0, "ROAD_BUILDING": 0, "VICTORY_POINT": 0},
        "longest_road_length": 3,
        "roads_available": 10,
        "settlements_available": 3,
        "cities_available": 4,
        "has_army": False,
        "played_dev_cards": {"KNIGHT": 0, "YEAR_OF_PLENTY": 0, "MONOPOLY": 0, "ROAD_BUILDING": 0, "VICTORY_POINT": 0},
    }
    player.update(overrides)
    return player


def _base_view(*, player_a, player_b):
    return {
        "colors": ["RED", "BLUE"],
        "current_color": "RED",
        "players": [player_a, player_b],
        "resource_bank": {"WOOD": 10, "BRICK": 10, "SHEEP": 10, "WHEAT": 10, "ORE": 10},
        "robber_coordinate": (0, 0, 0),
        "buildings": {},
        "roads": {},
        "winner": None,
        "development_deck_count": 5,
        "is_road_building": False,
        "free_roads_available": 0,
    }


def test_classify_longest_road_tie_swap_is_bucketed_as_rules_adjudication():
    """The confirmed repro shape: both engines agree the road length is tied
    at 5 for both players, but disagree on who holds the +2VP bonus. Every
    other tracked field (resources, dev cards, availability, army evidence)
    matches, so this should still route to the longest-road bucket even
    though `longest_road_length` itself is identical between engines.
    """
    rust_view = _base_view(
        player_a=_base_player(victory_points=2, actual_victory_points=2, longest_road_length=5),
        player_b=_base_player(victory_points=4, actual_victory_points=4, longest_road_length=5),
    )
    python_view = _base_view(
        player_a=_base_player(victory_points=4, actual_victory_points=4, longest_road_length=5),
        player_b=_base_player(victory_points=2, actual_victory_points=2, longest_road_length=5),
    )
    mismatches = diff_state_views(rust_view, python_view)
    assert mismatches
    assert classify_state_divergence_topic(rust_view, python_view, mismatches) == (
        "rules_adjudication_needed_longest_road"
    )


def test_classify_largest_army_miscount_is_not_masked_as_longest_road():
    """Adversarial-review regression: a VP mismatch caused by a largest-army
    discrepancy (has_army/played KNIGHT counts differ) must NOT be bucketed
    as the known longest-road issue -- it's a different, novel bug and should
    surface as "unclassified" so it gets investigated.
    """
    rust_view = _base_view(
        player_a=_base_player(
            victory_points=2, actual_victory_points=2, has_army=False,
            played_dev_cards={"KNIGHT": 2, "YEAR_OF_PLENTY": 0, "MONOPOLY": 0, "ROAD_BUILDING": 0, "VICTORY_POINT": 0},
        ),
        player_b=_base_player(
            victory_points=4, actual_victory_points=4, has_army=True,
            played_dev_cards={"KNIGHT": 3, "YEAR_OF_PLENTY": 0, "MONOPOLY": 0, "ROAD_BUILDING": 0, "VICTORY_POINT": 0},
        ),
    )
    python_view = _base_view(
        player_a=_base_player(
            victory_points=4, actual_victory_points=4, has_army=True,
            played_dev_cards={"KNIGHT": 2, "YEAR_OF_PLENTY": 0, "MONOPOLY": 0, "ROAD_BUILDING": 0, "VICTORY_POINT": 0},
        ),
        player_b=_base_player(
            victory_points=2, actual_victory_points=2, has_army=False,
            played_dev_cards={"KNIGHT": 3, "YEAR_OF_PLENTY": 0, "MONOPOLY": 0, "ROAD_BUILDING": 0, "VICTORY_POINT": 0},
        ),
    )
    mismatches = diff_state_views(rust_view, python_view)
    assert mismatches
    assert classify_state_divergence_topic(rust_view, python_view, mismatches) == "unclassified"


def test_classify_buildable_road_mismatch_requires_enemy_adjacency():
    """Adversarial-review regression: a BUILD_ROAD legal-action mismatch must
    only be bucketed as the known buildable-edge issue when the mismatched
    edge is actually adjacent to an enemy-occupied node. An edge with no such
    adjacency is a different, novel bug and must surface as "unclassified".
    """
    only_python = {("RED", "BUILD_ROAD", (7, 8))}
    buildings_with_enemy_at_8 = {8: ("BLUE", "SETTLEMENT")}
    assert classify_legal_action_mismatch_topic(set(), only_python, buildings_with_enemy_at_8) == (
        "rules_adjudication_needed_buildable_edge_near_enemy"
    )

    buildings_no_enemy_nearby = {8: ("RED", "SETTLEMENT")}
    assert classify_legal_action_mismatch_topic(set(), only_python, buildings_no_enemy_nearby) == "unclassified"

    buildings_empty = {}
    assert classify_legal_action_mismatch_topic(set(), only_python, buildings_empty) == "unclassified"


def test_single_game_reaches_completion_or_a_documented_divergence():
    """End-to-end smoke test: play one full game and require the outcome to be
    either a clean completion or one of the two documented, characterized bugs
    (longest-road tie-break, or the buildable-edges-through-enemy-node gap).
    Any *other* kind of divergence is a new, undocumented finding and should
    fail this test so it gets investigated.
    """
    _rust()
    config = EquivalenceConfig(max_steps=3000)
    result = play_one_game(seed=1, config=config)
    assert result.outcome in ("completed", "step_limit")


def test_longest_road_divergence_seed_2_is_never_misclassified():
    """Seed=2 is the original repro for finding #2 (Rust's longest-road bonus
    transferring on an exact tie) against pre-fix engines. Whether that bug
    (or the pre-existing Python-side longest-road bugs from the 2026-07-02
    audit, upstream #376/#378) is present depends on which engine versions are
    installed -- this test does not pin a specific outcome. It only asserts
    the harness itself never breaks (outcome != "error") and, IF a
    state_divergence occurs, it is never silently misclassified: either it's
    correctly routed to `rules_adjudication_needed_longest_road` (see the
    `classify_state_divergence_topic` unit tests above for the positive-
    evidence guarantee behind that), or it's "unclassified" -- meaning this
    harness run surfaced a genuinely new finding worth reporting, not a
    regression in this test.
    """
    _rust()
    config = EquivalenceConfig(max_steps=3000)
    result = play_one_game(seed=2, config=config)
    assert result.outcome != "error", result.detail
    if result.outcome == "state_divergence":
        assert result.topic in ("rules_adjudication_needed_longest_road", "unclassified"), (
            f"unexpected topic {result.topic!r}: {result.detail}"
        )


def test_buildable_edge_divergence_seed_5_is_never_misclassified():
    """Seed=5 is the original repro for finding #3 (Python's `buildable_edges`
    permitting a road through an enemy-occupied node) against pre-fix
    engines. Same reasoning as the longest-road test above: no specific
    outcome is pinned since the fix landscape evolves, but the harness must
    never crash, and any legal_action_mismatch here must be correctly
    bucketed (`rules_adjudication_needed_buildable_edge_near_enemy`, per the
    enemy-adjacency evidence checked in `classify_legal_action_mismatch_topic`)
    or fall through to "unclassified" as a genuinely new finding.
    """
    _rust()
    config = EquivalenceConfig(max_steps=3000)
    result = play_one_game(seed=5, config=config)
    assert result.outcome != "error", result.detail
    if result.outcome == "legal_action_mismatch":
        assert result.topic in (
            "rules_adjudication_needed_buildable_edge_near_enemy",
            "unclassified",
        ), f"unexpected topic {result.topic!r}: {result.detail}"


@pytest.mark.parametrize(
    "seed", [97, 324, 425, 449, 605, 701, 717, 969]
)
def test_a24_component_duplicate_node_seeds_no_longer_diverge(seed):
    """FIX A24: these 8 seeds all reproduced the same legal_action_mismatch
    (Python under-offering a legal BUILD_ROAD) in a full parallel equivalence
    scan against the post-A17-refinement (87a7393) engine -- see
    runs/engine_equivalence/parallel_scan_findings.json on B200. Root cause:
    a settlement-triggered cut in Board.build_settlement legitimately leaves
    the cut node duplicated across two connected_components entries (their
    shared boundary), but buildable_edges()'s per-component is_friendly_node
    filter discarded an entire piece (and any node ONLY reachable within it)
    whenever that piece alone had no friendly building -- even when it was
    still reachable via the shared boundary node from a genuinely anchored
    piece. See test_board.py::test_a24_buildable_edges_reaches_through_a_shared_boundary_node_after_a_settlement_cut
    for a minimal, isolated unit reconstruction of the exact shape.

    Unlike the other single-seed tests in this file (which don't pin an
    outcome, since other/unrelated engine bugs are tracked separately), this
    IS meant to pin an outcome: none of these 8 seeds should hit a
    legal_action_mismatch anymore after this fix. A later, unrelated
    state_divergence from a different open issue is not a regression in this
    fix and is tolerated here.
    """
    _rust()
    config = EquivalenceConfig(max_steps=3000)
    result = play_one_game(seed=seed, config=config)
    assert result.outcome != "error", result.detail
    assert result.outcome != "legal_action_mismatch", (
        f"seed={seed} still hits a legal_action_mismatch after the A24 fix: {result.detail}"
    )


@pytest.mark.parametrize("seed", [26, 179, 194, 388, 331])
def test_a26_settlement_cut_tie_seeds_no_longer_diverge(seed):
    """FIX A26: these 5 seeds reproduced a Longest-Road-card winner-flip on a
    settlement-triggered cut (root cause: build_settlement's recompute picked
    the GLOBAL max across ALL colors' road_lengths and unconditionally
    reassigned, instead of requiring a challenger to STRICTLY exceed the
    CURRENT incumbent -- the same rule build_road already enforces). Seed 331
    was the most severe case: a tie flip actually changed the reported
    WINNER of the game. See
    test_board.py::test_a26_settlement_cut_tie_with_another_color_keeps_the_incumbent
    for a minimal, isolated unit reconstruction.

    Unlike A24's test above, this DOES pin outcome == "completed": empirically
    (checked directly against the real engines) all 5 of these seeds now run
    clean end to end under the fix, so tolerating a residual divergence here
    would mask a real regression rather than accommodate a separate known
    issue.
    """
    _rust()
    config = EquivalenceConfig(max_steps=3000)
    result = play_one_game(seed=seed, config=config)
    assert result.outcome == "completed", (
        f"seed={seed} no longer completes cleanly after the A26 fix: "
        f"outcome={result.outcome!r} topic={result.topic!r} detail={result.detail}"
    )


@pytest.mark.parametrize("seed", [587, 954])
def test_a27_enemy_cut_node_dfs_start_seeds_no_longer_diverge(seed):
    """FIX A27: these 2 seeds reproduced a longest-road LENGTH undercount by
    exactly 1 (root cause: an enemy-owned cut-node with TWO OR MORE friendly
    exits -- e.g. a road cycle closing back through it -- was never tried as
    a fresh DFS start in longest_acyclic_path, only reachable as a mid-path
    stop from one direction, which correctly counts the edge INTO it (FIX
    A16) but never gets to explore its OTHER exit(s)). See
    test_board.py::test_a27_longest_acyclic_path_closes_a_cycle_through_an_enemy_cut_node
    for a minimal, isolated unit reconstruction of the exact real topology
    (seed=587's RED road network: a closed loop through a BLUE city).

    Pins outcome == "completed", same reasoning as A26's test above: checked
    directly against the real engines, both seeds now run clean end to end.
    """
    _rust()
    config = EquivalenceConfig(max_steps=3000)
    result = play_one_game(seed=seed, config=config)
    assert result.outcome == "completed", (
        f"seed={seed} no longer completes cleanly after the A27 fix: "
        f"outcome={result.outcome!r} topic={result.topic!r} detail={result.detail}"
    )


@pytest.mark.parametrize("seed", [167, 491, 731])
def test_a28_incumbent_still_on_top_seeds_no_longer_diverge(seed):
    """FIX A28: closes a gap the A26 fix left -- these 3 seeds reproduced a
    Longest-Road-card mismatch where a severed incumbent was wrongly kept
    despite an UNTOUCHED rival's already-existing length now strictly
    exceeding the incumbent's reduced length (A26 only checked that the
    incumbent still qualified at >=5, not that they were still on top). See
    test_board.py::test_a28_settlement_cut_below_an_untouched_rival_transfers_the_card
    for a minimal, isolated unit reconstruction, and the companion
    test_a28_settlement_cut_tying_an_untouched_rival_keeps_the_incumbent for
    the mirror-image tie-never-transfers sanity check.

    Pins outcome == "completed", same reasoning as A26/A27's tests: checked
    directly against the real engines, all 3 seeds now run clean end to end.
    """
    _rust()
    config = EquivalenceConfig(max_steps=3000)
    result = play_one_game(seed=seed, config=config)
    assert result.outcome == "completed", (
        f"seed={seed} no longer completes cleanly after the A28 fix: "
        f"outcome={result.outcome!r} topic={result.topic!r} detail={result.detail}"
    )


@pytest.mark.parametrize("start_seed", [1000])
def test_small_sweep_runs_without_crashing(start_seed):
    """Smoke test for the bulk sweep runner itself (not a correctness assertion
    about game outcomes, since real engine bugs are expected to surface as
    divergences within a handful of games -- see test file docstring)."""
    _rust()
    config = EquivalenceConfig(max_steps=1500)
    report = run_sweep(num_games=5, start_seed=start_seed, config=config)
    assert report.num_games == 5
    assert sum(report.outcomes.values()) == 5
    for divergence in report.divergences:
        assert divergence.outcome != "error", divergence.detail
