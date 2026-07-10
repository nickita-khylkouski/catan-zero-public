from __future__ import annotations

import json

import pytest

pytest.importorskip("catanatron_rs")

from catan_zero.search.gumbel_chance_mcts import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    HeuristicRustEvaluator,
    SearchResult,
    _GAction,
    _GNode,
    _decision_context,
    batch_api_available,
    sequential_halving_schedule,
)
from catan_zero.search.rust_mcts import (
    _heuristic_value,
    _playable_action_json_by_index,
    _require_rust_module,
    _spectrum,
)


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def _pure_mcts(config: GumbelChanceMCTSConfig) -> GumbelChanceMCTS:
    """Build the algorithm shell without invoking the optional engine guard.

    Node-level completed-Q and policy math only reads ``config``. Constructing
    that shell directly keeps those pure-Python tests runnable when the Rust
    extension is not installed, while live search tests continue through
    ``_rust()`` and the real constructor.
    """
    mcts = object.__new__(GumbelChanceMCTS)
    mcts.config = config
    return mcts


def _catanatron_rs_version() -> tuple[int, ...]:
    try:
        from importlib.metadata import version

        raw = version("catanatron_rs")
    except Exception:
        return (0, 0, 0)
    parts: list[int] = []
    for part in raw.split(".")[:3]:
        digits = "".join(char for char in part if char.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


# A19 (MOVE_ROBBER-with-victim uniform-spectrum bug) and A20 (BUY_DEVELOPMENT_CARD
# phantom outcome bug) were fixed in catanatron_rs 0.1.1; the "raw engine bug is
# present" assertions below are stale (by design) from that version onward.
_A19_A20_FIXED_ON_WHEEL = _catanatron_rs_version() >= (0, 1, 1)


def _advance_to_roll(catanatron_rs, *, seed: int):
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=seed)
    for _ in range(300):
        playable = json.loads(game.playable_actions_json())
        if playable and playable[0][1] == "ROLL":
            return game
        game.play_tick()
    raise AssertionError("did not reach a ROLL prompt")


def _advance_to_multi_action_state(catanatron_rs, *, seed: int, min_legal: int = 2):
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=seed)
    for _ in range(300):
        game.play_tick()
        if game.winning_color() is not None:
            break
        playable = json.loads(game.playable_actions_json())
        if len(playable) >= min_legal:
            return game
    raise AssertionError(f"did not reach a state with >= {min_legal} legal actions")


def _action_values(game, legal_actions, *, colors, root_color):
    """Independent ground-truth: expected heuristic value of taking each action.

    Mirrors the pre-softmax scoring HeuristicRustEvaluator computes internally,
    but exposed standalone so tests can compare policies against it without
    coupling to the evaluator's own (softmaxed, sign-flipped) prior output.
    """
    action_json = _playable_action_json_by_index(game, legal_actions, colors, None)
    values: dict[int, float] = {}
    for action_id in legal_actions:
        outcomes = _spectrum(game, action_json[action_id])
        expected = 0.0
        for outcome_index, probability in outcomes:
            outcome = game.apply_chance_outcome(
                json.dumps(action_json[action_id]), outcome_index
            )
            expected += probability * _heuristic_value(
                outcome, root_color=root_color, colors=colors
            )
        values[int(action_id)] = expected
    return values


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


def test_search_result_actions_are_all_legal():
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=11)
    legal = set(json.loads(game.playable_actions_json()) and _legal_ids(game))

    mcts = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=3, n_full=32, p_full=1.0))
    result = mcts.search(game, force_full=True)

    assert result.selected_action in legal
    assert set(result.improved_policy) == legal
    assert set(result.visit_counts) == legal
    assert set(result.priors) == legal
    assert set(result.q_values).issubset(legal)


def _legal_ids(game):
    return set(int(a) for a in game.playable_action_indices(["RED", "BLUE"], None))


def test_single_legal_action_takes_forced_fast_path():
    catanatron_rs = _rust()
    game = _advance_to_roll(catanatron_rs, seed=5)
    legal = _legal_ids(game)
    assert len(legal) == 1

    mcts = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=0))
    result = mcts.search(game, force_full=True)

    assert result.selected_action in legal
    assert result.simulations_used == 0
    assert result.improved_policy == {next(iter(legal)): 1.0}


def test_forced_roll_still_enumerates_dice_and_reports_real_afterstate_value():
    # ROLL is nearly always the sole legal action at its own decision point, so
    # the forced single-action fast path must still enumerate all 11 outcomes
    # rather than reporting a fake placeholder root_value/afterstate_value --
    # this is the only place a real dice roll's afterstate signal is produced.
    catanatron_rs = _rust()
    game = _advance_to_roll(catanatron_rs, seed=5)
    action = next(iter(_legal_ids(game)))

    mcts = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=0))
    result = mcts.search(game, force_full=True)

    assert action in result.afterstate_values
    assert -1.0 <= result.afterstate_values[action] <= 1.0
    assert -1.0 <= result.root_value <= 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_search_is_deterministic_for_same_seed():
    catanatron_rs = _rust()
    config = GumbelChanceMCTSConfig(seed=99, n_full=24, p_full=1.0)

    first_game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=19)
    second_game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=19)

    a = GumbelChanceMCTS(config).search(first_game, force_full=True)
    b = GumbelChanceMCTS(config).search(second_game, force_full=True)

    assert a.selected_action == b.selected_action
    assert a.visit_counts == b.visit_counts
    assert a.improved_policy == b.improved_policy
    assert a.simulations_used == b.simulations_used


# ---------------------------------------------------------------------------
# Dice spectrum enumeration in the chance-node backup.
# ---------------------------------------------------------------------------


def test_roll_chance_node_enumerates_all_eleven_dice_outcomes_with_exact_weights():
    catanatron_rs = _rust()
    game = _advance_to_roll(catanatron_rs, seed=5)
    mcts = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=1))

    node = _GNode(game=game.copy(), root_color=str(game.current_color()))
    mcts._expand(node)
    roll_action_id = next(
        action_id
        for action_id, action_json in node.action_json.items()
        if action_json[1] == "ROLL"
    )
    stats = node.actions[roll_action_id]

    mcts._traverse_roll(node, roll_action_id, stats, depth=0)

    assert len(stats.probabilities) == 11
    expected = [count / 36.0 for count in (1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1)]
    for outcome_index, expected_probability in enumerate(expected):
        assert abs(stats.probabilities[outcome_index] - expected_probability) < 1.0e-9
    assert abs(sum(stats.probabilities.values()) - 1.0) < 1.0e-9
    assert stats.afterstate_value is not None
    assert -1.0 <= stats.afterstate_value <= 1.0

    # The backed-up value is the exact probability-weighted average of the
    # (evaluator-leaf) child values, not a single-sample estimate.
    expected_value = sum(
        stats.probabilities[index] * stats.children[index].value for index in stats.children
    )
    assert abs(stats.value_sum / stats.visits - expected_value) < 1.0e-9


def test_traverse_roll_does_not_crash_when_enumeration_returns_no_outcomes():
    # Regression test: if _enumerate_roll_outcomes ever returns no children
    # (e.g. a pathological spectrum with all-zero probabilities), sampling
    # from the empty outcome set used to raise IndexError on outcomes[-1].
    # It must instead fall back to the node's own leaf estimate.
    catanatron_rs = _rust()
    game = _advance_to_roll(catanatron_rs, seed=5)
    mcts = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=1))

    node = _GNode(game=game.copy(), root_color=str(game.current_color()))
    mcts._expand(node)
    roll_action_id = next(
        action_id
        for action_id, action_json in node.action_json.items()
        if action_json[1] == "ROLL"
    )
    stats = node.actions[roll_action_id]

    mcts._enumerate_roll_outcomes = lambda *args, **kwargs: ({}, {}, 0.0)  # type: ignore[method-assign]

    value = mcts._traverse_roll(node, roll_action_id, stats, depth=0)

    assert value == node.prior_value
    assert stats.children == {}
    assert stats.afterstate_value == node.prior_value
    assert stats.visits == 1


def test_buy_development_card_uses_expectation_backup_not_single_sample():
    # F7 regression: BUY_DEVELOPMENT_CARD is enumerated + expectation-backed
    # up exactly like ROLL now, not single-sampled -- ALL real (non-phantom)
    # candidates get materialized and expanded on the FIRST traversal (the
    # phantom-filtering pass already pays that cost to classify real vs.
    # phantom), and every traversal's backed-up value is the exact
    # probability-weighted average over every materialized child's current
    # value, not a 1-sample estimate.
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=42)
    dev_card_action = None
    for _ in range(2000):
        playable = json.loads(game.playable_actions_json())
        candidates = [a for a in playable if a[1] == "BUY_DEVELOPMENT_CARD"]
        if candidates:
            dev_card_action = candidates[0]
            break
        game.play_tick()
    assert dev_card_action is not None

    spectrum = json.loads(game.spectrum_json(json.dumps(dev_card_action)))
    assert len(spectrum) == 5

    mcts = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=2))
    node = _GNode(game=game.copy(), root_color=str(game.current_color()))
    mcts._expand(node)
    action_id = next(
        action_id
        for action_id, action_json in node.action_json.items()
        if action_json[1] == "BUY_DEVELOPMENT_CARD"
    )
    stats = node.actions[action_id]

    mcts._traverse_robber_or_dev(node, action_id, stats, depth=0)

    # Every real (non-phantom) candidate is materialized on the FIRST
    # traversal -- not just the sampled one.
    assert 1 <= len(stats.children) <= len(spectrum)
    assert abs(sum(stats.probabilities.values()) - 1.0) < 1.0e-9
    materialized_children = dict(stats.children)
    expected_value = sum(
        stats.probabilities[index] * stats.children[index].value for index in stats.children
    )
    assert abs(stats.value_sum / stats.visits - expected_value) < 1.0e-9

    # A second traversal reuses the already-materialized children -- same
    # objects, no re-materialization (the sampled child's subtree may
    # deepen, changing its `.value`, so the exact expectation isn't
    # re-derivable after the fact without per-call history; what must hold
    # is that materialization happened exactly once).
    mcts._traverse_robber_or_dev(node, action_id, stats, depth=0)
    assert stats.visits == 2
    assert stats.children.keys() == materialized_children.keys()
    for index, child in materialized_children.items():
        assert stats.children[index] is child


# ---------------------------------------------------------------------------
# Corrected chance spectra (A19/A20 mitigation for verified Rust engine bugs).
# ---------------------------------------------------------------------------


def _find_move_robber_with_victim(catanatron_rs, *, seed: int, max_ticks: int = 300):
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=seed)
    for _ in range(max_ticks):
        playable = json.loads(game.playable_actions_json())
        candidates = [
            action for action in playable if action[1] == "MOVE_ROBBER" and action[2][1] is not None
        ]
        if candidates:
            return game, candidates[0]
        game.play_tick()
    raise AssertionError("did not reach a MOVE_ROBBER-with-victim decision within budget")


def _find_buy_development_card_with_phantom_outcome(catanatron_rs, *, seed: int, max_ticks: int = 4000):
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=seed)
    for _ in range(max_ticks):
        if game.winning_color() is not None:
            break
        playable = json.loads(game.playable_actions_json())
        candidates = [action for action in playable if action[1] == "BUY_DEVELOPMENT_CARD"]
        if candidates:
            action_json = candidates[0]
            spectrum = json.loads(game.spectrum_json(json.dumps(action_json)))
            snapshot = json.loads(game.json_snapshot())
            colors = [str(color) for color in snapshot["colors"]]
            actor_index = colors.index(str(action_json[0]))
            before_cards = snapshot["player_state"][actor_index]["dev_cards"]
            before_deck = int(snapshot["development_deck_count"])
            phantom_indices = []
            for outcome_index, _entry in enumerate(spectrum):
                candidate_game = game.apply_chance_outcome(json.dumps(action_json), outcome_index)
                candidate_snapshot = json.loads(candidate_game.json_snapshot())
                candidate_cards = candidate_snapshot["player_state"][actor_index]["dev_cards"]
                gained = [
                    card
                    for card in DEVELOPMENT_CARDS
                    if int(candidate_cards.get(card, 0)) - int(before_cards.get(card, 0)) == 1
                ]
                deck_decreased = (
                    int(candidate_snapshot["development_deck_count"]) == before_deck - 1
                )
                if not (len(gained) == 1 and deck_decreased):
                    phantom_indices.append(outcome_index)
            if phantom_indices:
                return game, action_json, tuple(phantom_indices)
        game.play_tick()
    raise AssertionError(
        "did not reach a BUY_DEVELOPMENT_CARD decision with a phantom outcome within budget "
        "(engine bug A20 may not reproduce for this seed range)"
    )


def test_move_robber_with_victim_uses_the_victims_real_hand_not_uniform():
    # Property test for our OWN correction logic (`_corrected_move_robber_outcome`):
    # it must be hand-weighted regardless of whether the underlying engine
    # bug (A19) is present on the installed wheel -- unconditional, unlike
    # the "raw engine bug is present" checks below which are wheel-version-gated.
    catanatron_rs = _rust()
    game, action_json = _find_move_robber_with_victim(catanatron_rs, seed=0)

    victim_name = action_json[2][1]
    snapshot = json.loads(game.json_snapshot())
    colors = [str(color) for color in snapshot["colors"]]
    victim_hand = snapshot["player_state"][colors.index(victim_name)]["resources"]
    held_resources = {resource for resource in RESOURCES if victim_hand[resource] > 0}
    assert held_resources, "test fixture requires a victim holding at least one card"
    assert held_resources != set(RESOURCES), (
        "test fixture requires the victim to be missing at least one resource, "
        "otherwise this test can't distinguish corrected weights from uniform"
    )

    mcts = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=1, correct_rust_chance_spectra=True))
    drawn_resources: dict[str, int] = {resource: 0 for resource in RESOURCES}
    for _ in range(200):
        # NOTE: `resource_index` is a NATIVE engine outcome index (which,
        # since the A19 fix, is no longer guaranteed to be the resource's
        # RESOURCES position -- it may be a shorter, hand-filtered index
        # space). Derive the actually-stolen resource from the returned
        # child game's hand diff instead of indexing RESOURCES directly.
        _outcome_index, child_game = mcts._corrected_move_robber_outcome(game, action_json)
        after_hand = json.loads(child_game.json_snapshot())["player_state"][
            colors.index(victim_name)
        ]["resources"]
        stolen = [
            resource
            for resource in RESOURCES
            if int(victim_hand[resource]) - int(after_hand[resource]) == 1
        ]
        assert len(stolen) == 1, f"expected exactly one resource stolen, got delta vs {stolen}"
        drawn_resources[stolen[0]] += 1

    # Every draw must be a resource the victim actually holds.
    for resource in RESOURCES:
        if resource not in held_resources:
            assert drawn_resources[resource] == 0, (
                f"corrected sampling drew {resource!r}, which the victim does not hold"
            )
    assert sum(drawn_resources[resource] for resource in held_resources) == 200

    # Weights should be proportional to the hand (not uniform among held cards
    # either, unless the hand happens to be uniform itself).
    total_cards = sum(victim_hand[resource] for resource in held_resources)
    for resource in held_resources:
        expected_fraction = victim_hand[resource] / total_cards
        observed_fraction = drawn_resources[resource] / 200.0
        assert abs(observed_fraction - expected_fraction) < 0.15, (
            f"{resource}: expected~{expected_fraction:.3f} observed={observed_fraction:.3f}"
        )


@pytest.mark.skipif(
    _A19_A20_FIXED_ON_WHEEL,
    reason="A19 (MOVE_ROBBER-with-victim uniform spectrum) was fixed in catanatron_rs "
    "0.1.1; this assertion is stale by design on newer wheels.",
)
def test_move_robber_raw_spectrum_bug_is_present_before_0_1_1():
    catanatron_rs = _rust()
    game, action_json = _find_move_robber_with_victim(catanatron_rs, seed=0)
    raw_spectrum = json.loads(game.spectrum_json(json.dumps(action_json)))
    assert [entry["probability"] for entry in raw_spectrum] == [0.2] * 5


@pytest.mark.skipif(
    not _A19_A20_FIXED_ON_WHEEL,
    reason="requires catanatron_rs >= 0.1.1 (A19 fix) to have anything to A/B against",
)
def test_move_robber_correction_flag_is_a_noop_on_the_fixed_wheel():
    """The A/B conclusion this restructuring is meant to put on record:
    correct_rust_chance_spectra=True and False now produce the same
    real-world stolen-resource distribution on a wheel with A19 fixed
    (the raw spectrum is itself already hand-weighted)."""
    catanatron_rs = _rust()
    game, action_json = _find_move_robber_with_victim(catanatron_rs, seed=0)

    def _stolen_resource_counts(*, correct_rust_chance_spectra: bool) -> dict[str, int]:
        mcts = GumbelChanceMCTS(
            GumbelChanceMCTSConfig(seed=1, correct_rust_chance_spectra=correct_rust_chance_spectra),
            HeuristicRustEvaluator(score_actions=False),
        )
        counts: dict[str, int] = {resource: 0 for resource in RESOURCES}
        for _ in range(200):
            node = _GNode(game=game.copy(), root_color=str(game.current_color()))
            mcts._expand(node)
            action_id = next(
                aid
                for aid, stored in node.action_json.items()
                if stored[1] == "MOVE_ROBBER" and stored[2][1] is not None
            )
            stats = node.actions[action_id]
            before = json.loads(game.json_snapshot())
            victim_index = [str(c) for c in before["colors"]].index(action_json[2][1])
            before_hand = before["player_state"][victim_index]["resources"]
            mcts._traverse_robber_or_dev(node, action_id, stats, depth=0)
            # `stats.children` holds every real candidate (F7: enumerated +
            # expectation-backed-up, not single-sampled) -- the traversal
            # only *recurses* into one, so pick that one rather than
            # assuming a single entry.
            child = next(candidate for candidate in stats.children.values() if candidate.visits > 0)
            after_hand = json.loads(child.game.json_snapshot())["player_state"][victim_index][
                "resources"
            ]
            stolen = [
                resource for resource in RESOURCES if before_hand[resource] - after_hand[resource] == 1
            ]
            if stolen:
                counts[stolen[0]] += 1
        return counts

    corrected = _stolen_resource_counts(correct_rust_chance_spectra=True)
    raw = _stolen_resource_counts(correct_rust_chance_spectra=False)
    total_corrected = sum(corrected.values())
    total_raw = sum(raw.values())
    assert total_corrected > 0 and total_raw > 0
    for resource in RESOURCES:
        corrected_fraction = corrected[resource] / total_corrected
        raw_fraction = raw[resource] / total_raw
        assert abs(corrected_fraction - raw_fraction) < 0.15, (
            f"{resource}: corrected~{corrected_fraction:.3f} raw~{raw_fraction:.3f} -- "
            "flag states should behave identically once the wheel is fixed"
        )


def test_move_robber_correction_can_be_disabled_via_config_flag():
    catanatron_rs = _rust()
    game, action_json = _find_move_robber_with_victim(catanatron_rs, seed=0)

    mcts = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=1, correct_rust_chance_spectra=False))
    node = _GNode(game=game.copy(), root_color=str(game.current_color()))
    mcts._expand(node)
    action_id = next(
        action_id
        for action_id, stored_action_json in node.action_json.items()
        if stored_action_json[1] == "MOVE_ROBBER" and stored_action_json[2][1] is not None
    )
    stats = node.actions[action_id]

    # With the flag off, the (buggy, uniform) raw spectrum is enumerated
    # natively (F7: still expectation-backed-up, just not hand-weight
    # corrected) -- this should not raise, and should exercise the
    # uncorrected codepath, not `move_robber_victim_outcome_weights`.
    mcts._traverse_robber_or_dev(node, action_id, stats, depth=0)
    assert stats.visits == 1


def test_buy_development_card_corrected_outcome_always_gains_exactly_one_real_card():
    # Property test for our OWN correction logic (`_corrected_buy_dev_card_outcome`):
    # every sampled outcome must be real (gains exactly one card, decrements
    # the deck) on ANY real BUY_DEVELOPMENT_CARD decision -- unconditional,
    # regardless of whether the installed wheel still has phantom outcomes to
    # filter (if it doesn't, this correction is a no-op that still holds).
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=42)
    for _ in range(2000):
        playable = json.loads(game.playable_actions_json())
        candidates = [a for a in playable if a[1] == "BUY_DEVELOPMENT_CARD"]
        if candidates:
            action_json = candidates[0]
            break
        game.play_tick()
    else:
        raise AssertionError("did not reach a BUY_DEVELOPMENT_CARD decision within budget")

    mcts = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=2, correct_rust_chance_spectra=True))
    snapshot = json.loads(game.json_snapshot())
    colors = [str(color) for color in snapshot["colors"]]
    actor_index = colors.index(str(action_json[0]))
    before_cards = snapshot["player_state"][actor_index]["dev_cards"]
    before_deck = int(snapshot["development_deck_count"])

    for _ in range(100):
        _outcome_index, child_game = mcts._corrected_buy_dev_card_outcome(game, action_json)
        child_snapshot = json.loads(child_game.json_snapshot())
        child_cards = child_snapshot["player_state"][actor_index]["dev_cards"]
        gained = [
            card
            for card in DEVELOPMENT_CARDS
            if int(child_cards.get(card, 0)) - int(before_cards.get(card, 0)) == 1
        ]
        assert len(gained) == 1, "every sampled outcome must actually draw exactly one card"
        assert int(child_snapshot["development_deck_count"]) == before_deck - 1


@pytest.mark.skipif(
    _A19_A20_FIXED_ON_WHEEL,
    reason="A20 (BUY_DEVELOPMENT_CARD phantom outcome) was fixed in catanatron_rs "
    "0.1.1; this assertion is stale by design on newer wheels.",
)
def test_buy_development_card_phantom_bug_is_present_before_0_1_1():
    catanatron_rs = _rust()
    _game, _action_json, phantom_indices = _find_buy_development_card_with_phantom_outcome(
        catanatron_rs, seed=6
    )
    assert phantom_indices, "test fixture requires at least one phantom outcome"


@pytest.mark.skipif(
    not _A19_A20_FIXED_ON_WHEEL,
    reason="requires catanatron_rs >= 0.1.1 (A20 fix) to have anything to A/B against",
)
def test_buy_development_card_correction_flag_is_a_noop_on_the_fixed_wheel():
    """The A/B conclusion this restructuring is meant to put on record:
    correct_rust_chance_spectra=True and False now produce the same
    real-world drawn-card distribution on a wheel with A20 fixed (the raw
    spectrum has no phantom outcomes left to filter)."""
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=42)
    for _ in range(2000):
        playable = json.loads(game.playable_actions_json())
        candidates = [a for a in playable if a[1] == "BUY_DEVELOPMENT_CARD"]
        if candidates:
            action_json = candidates[0]
            break
        game.play_tick()
    else:
        raise AssertionError("did not reach a BUY_DEVELOPMENT_CARD decision within budget")

    snapshot = json.loads(game.json_snapshot())
    colors = [str(color) for color in snapshot["colors"]]
    actor_index = colors.index(str(action_json[0]))
    before_cards = snapshot["player_state"][actor_index]["dev_cards"]

    def _drawn_card_counts(*, correct_rust_chance_spectra: bool) -> dict[str, int]:
        mcts = GumbelChanceMCTS(
            GumbelChanceMCTSConfig(seed=2, correct_rust_chance_spectra=correct_rust_chance_spectra),
            HeuristicRustEvaluator(score_actions=False),
        )
        counts: dict[str, int] = {card: 0 for card in DEVELOPMENT_CARDS}
        for _ in range(200):
            node = _GNode(game=game.copy(), root_color=str(game.current_color()))
            mcts._expand(node)
            action_id = next(
                aid for aid, stored in node.action_json.items() if stored[1] == "BUY_DEVELOPMENT_CARD"
            )
            stats = node.actions[action_id]
            mcts._traverse_robber_or_dev(node, action_id, stats, depth=0)
            # `stats.children` holds every real (non-phantom) candidate (F7:
            # enumerated + expectation-backed-up, not single-sampled) --
            # only one is ever actually *visited* per traversal, so pick that
            # one rather than assuming a single entry.
            child = next(candidate for candidate in stats.children.values() if candidate.visits > 0)
            child_cards = json.loads(child.game.json_snapshot())["player_state"][actor_index][
                "dev_cards"
            ]
            gained = [
                card
                for card in DEVELOPMENT_CARDS
                if int(child_cards.get(card, 0)) - int(before_cards.get(card, 0)) == 1
            ]
            if gained:
                counts[gained[0]] += 1
        return counts

    corrected = _drawn_card_counts(correct_rust_chance_spectra=True)
    raw = _drawn_card_counts(correct_rust_chance_spectra=False)
    total_corrected = sum(corrected.values())
    total_raw = sum(raw.values())
    assert total_corrected > 0 and total_raw > 0
    for card in DEVELOPMENT_CARDS:
        corrected_fraction = corrected[card] / total_corrected
        raw_fraction = raw[card] / total_raw
        assert abs(corrected_fraction - raw_fraction) < 0.15, (
            f"{card}: corrected~{corrected_fraction:.3f} raw~{raw_fraction:.3f} -- "
            "flag states should behave identically once the wheel is fixed"
        )


# ---------------------------------------------------------------------------
# Improved policy properties.
# ---------------------------------------------------------------------------


def test_improved_policy_from_search_sums_to_one():
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=7, min_legal=3)
    mcts = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=4, n_full=32, p_full=1.0))
    result = mcts.search(game, force_full=True)

    assert abs(sum(result.improved_policy.values()) - 1.0) < 1.0e-9


def test_improved_policy_puts_more_mass_on_higher_completed_q_actions():
    mcts = _pure_mcts(GumbelChanceMCTSConfig(seed=0))
    node = _GNode(game=None, root_color="RED")
    node.actions = {0: _GAction(prior=0.5), 1: _GAction(prior=0.5)}
    node.action_logits = {0: 0.0, 1: 0.0}

    completed_q = {0: 0.9, 1: -0.9}
    policy = mcts._improved_policy(node, completed_q)

    assert abs(sum(policy.values()) - 1.0) < 1.0e-9
    assert policy[0] > policy[1]


class _DistinctPriorEvaluator:
    """Test double returning strictly non-uniform, deterministic priors.

    `HeuristicRustEvaluator` legitimately ties priors on many real states (e.g.
    symmetric BUILD_ROAD candidates score identically since it only looks at
    aggregate player stats, not board topology), which would make a
    prior_temperature scaling test a no-op regardless of correctness. This
    double isolates the prior_temperature mechanism from evaluator quirks.
    """

    def evaluate(self, game, legal_actions, *, root_color, colors):
        del game, root_color, colors
        raw = {int(action): float(rank + 1) for rank, action in enumerate(legal_actions)}
        total = sum(raw.values())
        return {action: value / total for action, value in raw.items()}, 0.0


def test_prior_temperature_scales_action_logits_and_sharpens_policy():
    # Regression test: prior_temperature was a dead config knob (defined but
    # never applied). It must now scale action_logits as log(prior)/temperature
    # and, in turn, sharpen/soften the resulting improved policy.
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=7, min_legal=3)
    evaluator = _DistinctPriorEvaluator()

    sharp = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=0, prior_temperature=0.5), evaluator)
    soft = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=0, prior_temperature=2.0), evaluator)

    sharp_node = _GNode(game=game.copy(), root_color=str(game.current_color()))
    soft_node = _GNode(game=game.copy(), root_color=str(game.current_color()))
    sharp._expand(sharp_node)
    soft._expand(soft_node)

    assert set(sharp_node.action_logits) == set(soft_node.action_logits)
    for action_id in sharp_node.action_logits:
        # temperature=0.5 -> logits = log(p)/0.5 = 4x temperature=2.0's log(p)/2.
        assert soft_node.action_logits[action_id] == pytest.approx(
            sharp_node.action_logits[action_id] * 0.25, rel=1.0e-6
        )

    zero_q_sharp = {action_id: 0.0 for action_id in sharp_node.actions}
    zero_q_soft = {action_id: 0.0 for action_id in soft_node.actions}
    policy_sharp = sharp._improved_policy(sharp_node, zero_q_sharp)
    policy_soft = soft._improved_policy(soft_node, zero_q_soft)

    assert policy_sharp != policy_soft
    # Lower prior_temperature sharpens (more concentrated) the distribution.
    assert max(policy_sharp.values()) >= max(policy_soft.values())


def test_improved_policy_uses_mixed_value_for_unvisited_actions_not_zero():
    mcts = _pure_mcts(
        GumbelChanceMCTSConfig(seed=0, c_visit=50.0, c_scale=1.0)
    )

    class _FakeGame:
        def current_color(self):
            return "RED"

    node = _GNode(game=_FakeGame(), root_color="RED", prior_value=0.6)
    visited = _GAction(prior=0.5, visits=10, value_sum=8.0)  # q = 0.8
    unvisited = _GAction(prior=0.5, visits=0, value_sum=0.0)
    node.actions = {0: visited, 1: unvisited}

    completed_q = mcts._completed_q(node)

    assert completed_q[0] == pytest.approx(0.8)
    # Unvisited action must get the mixed value estimate, not a bare 0.0.
    assert completed_q[1] != 0.0
    expected_v_mix = (node.prior_value + 10 * 0.8) / (1.0 + 10)
    assert completed_q[1] == pytest.approx(expected_v_mix)


# ---------------------------------------------------------------------------
# F1: completed-Q/sigma ported from mctx (rescale + prior-weighted v_mix).
# ---------------------------------------------------------------------------


def test_v_mix_weights_visited_actions_by_prior_not_by_visit_count():
    """F1c regression: two visited actions with DIFFERENT visit counts but
    the SAME prior must contribute EQUALLY to v_mix (prior-weighted), not in
    proportion to how many times Gumbel-Top-k/Sequential Halving happened to
    visit them (the old, buggy visit-weighted behavior)."""
    mcts = _pure_mcts(GumbelChanceMCTSConfig(seed=0))

    class _FakeGame:
        def current_color(self):
            return "RED"

    node = _GNode(game=_FakeGame(), root_color="RED", prior_value=0.0)
    # Same prior (0.4 each), very different visit counts and Qs.
    heavily_visited_low_q = _GAction(prior=0.4, visits=100, value_sum=-50.0)  # q = -0.5
    lightly_visited_high_q = _GAction(prior=0.4, visits=1, value_sum=1.0)  # q = 1.0
    unvisited = _GAction(prior=0.2, visits=0, value_sum=0.0)
    node.actions = {0: heavily_visited_low_q, 1: lightly_visited_high_q, 2: unvisited}

    completed_q = mcts._completed_q(node)

    # Prior-weighted: weighted_q = (0.4*-0.5 + 0.4*1.0) / (0.4+0.4) = 0.25.
    # Visit-weighted (old, buggy): weighted_q = (100*-0.5 + 1*1.0) / 101 ~= -0.485.
    total_visits = 100 + 1
    expected_v_mix_prior_weighted = (0.0 + total_visits * 0.25) / (1.0 + total_visits)
    assert completed_q[2] == pytest.approx(expected_v_mix_prior_weighted)

    visit_weighted_q = (100 * -0.5 + 1 * 1.0) / 101
    expected_v_mix_visit_weighted = (0.0 + total_visits * visit_weighted_q) / (1.0 + total_visits)
    assert completed_q[2] != pytest.approx(expected_v_mix_visit_weighted)


def test_rescale_completed_q_maps_min_max_to_zero_one():
    rescaled = GumbelChanceMCTS._rescale_completed_q({0: -0.9, 1: 0.3, 2: 0.9})
    assert rescaled[0] == pytest.approx(0.0, abs=1.0e-6)
    assert rescaled[2] == pytest.approx(1.0, abs=1.0e-6)
    assert 0.0 < rescaled[1] < 1.0


def test_rescale_completed_q_handles_uniform_values_without_dividing_by_zero():
    rescaled = GumbelChanceMCTS._rescale_completed_q({0: 0.5, 1: 0.5})
    assert rescaled[0] == pytest.approx(0.0, abs=1.0e-3)
    assert rescaled[1] == pytest.approx(0.0, abs=1.0e-3)


def test_improved_policy_rescales_completed_q_before_sigma_bounding_sharpening():
    """F1a regression: a completed_q spread far outside [0, 1] must NOT
    translate into proportionally unbounded sharpening -- it must first be
    rescaled to [0, 1], so sigma's effect is bounded by
    (c_visit + max_visits) * c_scale regardless of Q's raw magnitude."""
    mcts = _pure_mcts(
        GumbelChanceMCTSConfig(seed=0, c_visit=50.0, c_scale=0.1)
    )
    node = _GNode(game=None, root_color="RED")
    node.actions = {0: _GAction(prior=0.5), 1: _GAction(prior=0.5)}
    node.action_logits = {0: 0.0, 1: 0.0}

    # A huge raw Q spread (e.g. an unclamped/buggy value estimate).
    huge_spread_q = {0: 1000.0, 1: -1000.0}
    policy_huge = mcts._improved_policy(node, huge_spread_q)

    # A modest, already-in-[0,1]-ish spread with the SAME relative ordering.
    modest_spread_q = {0: 0.9, 1: -0.9}
    policy_modest = mcts._improved_policy(node, modest_spread_q)

    # Both spreads rescale to the same [0, 1] endpoints, so the two policies
    # must be identical -- if the raw magnitude leaked through un-rescaled,
    # the huge spread would produce a near-one-hot policy the modest one
    # wouldn't.
    assert policy_huge[0] == pytest.approx(policy_modest[0], rel=1.0e-6)
    assert policy_huge[1] == pytest.approx(policy_modest[1], rel=1.0e-6)
    # And neither is anywhere near one-hot: max mass is bounded by sigma's
    # (c_visit + max_visits) * c_scale = (50 + 0) * 0.1 = 5.0 logit spread.
    assert max(policy_huge.values()) < 0.995


# ---------------------------------------------------------------------------
# Sequential Halving budget accounting.
# ---------------------------------------------------------------------------


def test_sequential_halving_schedule_matches_exact_totals_when_well_provisioned():
    schedule = sequential_halving_schedule(16, 64)
    assert schedule == [(16, 1), (8, 2), (4, 4), (2, 8)]
    assert sum(count * budget for count, budget in schedule) == 64


def test_sequential_halving_schedule_halves_candidates_each_round():
    schedule = sequential_halving_schedule(32, 256)
    counts = [count for count, _ in schedule]
    assert counts == [32, 16, 8, 4, 2]
    for count, budget in schedule:
        assert budget >= 1


def test_sequential_halving_schedule_never_starves_a_round_below_one_sim():
    # Under-provisioned budgets (n < m * num_rounds) still give >=1 sim per
    # candidate per round; total usage may exceed n_simulations in this case,
    # which is expected standard Sequential Halving behavior, not a bug.
    schedule = sequential_halving_schedule(32, 8)
    for _count, budget in schedule:
        assert budget >= 1
    total = sum(count * budget for count, budget in schedule)
    assert total >= 8


def test_root_search_reports_simulations_used_consistent_with_schedule():
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=11, min_legal=2)
    legal = _legal_ids(game)
    num_legal = len(legal)
    m = min(num_legal, 32) if num_legal > 24 else min(num_legal, 16)

    mcts = GumbelChanceMCTS(GumbelChanceMCTSConfig(seed=3, n_full=64, p_full=1.0))
    result = mcts.search(game, force_full=True)

    expected_total = sum(
        count * budget for count, budget in sequential_halving_schedule(m, 64)
    )
    assert result.simulations_used == expected_total
    assert sum(result.visit_counts.values()) == result.simulations_used


# ---------------------------------------------------------------------------
# Policy-improvement smoke test.
# ---------------------------------------------------------------------------


def test_policy_improvement_smoke_over_twenty_midgame_states():
    catanatron_rs = _rust()
    colors = ("RED", "BLUE")
    evaluator = HeuristicRustEvaluator()
    config = GumbelChanceMCTSConfig(seed=0, n_full=24, p_full=1.0)

    prior_evs: list[float] = []
    improved_evs: list[float] = []

    for seed in range(20):
        game = _advance_to_multi_action_state(catanatron_rs, seed=seed, min_legal=2)
        root_color = str(game.current_color())
        legal_actions = tuple(sorted(_legal_ids(game)))

        mcts = GumbelChanceMCTS(config, evaluator)
        result = mcts.search(game, force_full=True)

        values = _action_values(game, legal_actions, colors=colors, root_color=root_color)

        prior_ev = sum(result.priors[a] * values[a] for a in legal_actions)
        improved_ev = sum(result.improved_policy[a] * values[a] for a in legal_actions)
        prior_evs.append(prior_ev)
        improved_evs.append(improved_ev)

    mean_prior = sum(prior_evs) / len(prior_evs)
    mean_improved = sum(improved_evs) / len(improved_evs)

    assert mean_improved >= mean_prior - 1.0e-6


# ---------------------------------------------------------------------------
# 0.1.2 batch API integration (decision_context_json / apply_chance_outcomes_batch).
# Gated on batch_api_available() since the local dev mirror only has wheel
# 0.1.0 -- these are the tests that actually exercise the new path on a box
# with wheel >= 0.1.2. The fallback path (use_batch_api has no effect when
# the wheel lacks it) is already exercised unconditionally by every other
# test in this file, since batch_api_available() is False locally.
# ---------------------------------------------------------------------------

requires_batch_api = pytest.mark.skipif(
    not batch_api_available(),
    reason="requires catanatron_rs wheel >= 0.1.2 (decision_context_json / "
    "apply_chance_outcomes_batch); not available on this install",
)


@requires_batch_api
def test_decision_context_matches_legacy_legal_actions_and_action_json():
    catanatron_rs = _rust()
    from catan_zero.search.rust_mcts import _legal_action_indices

    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=7)
    legal_actions, action_json_by_id, _spectrum_by_id = _decision_context(
        game, colors=("RED", "BLUE"), map_kind=None
    )
    legacy_legal_actions = _legal_action_indices(game, colors=("RED", "BLUE"), map_kind=None)
    legacy_action_json = _playable_action_json_by_index(
        game, legacy_legal_actions, ("RED", "BLUE"), None
    )

    assert set(legal_actions) == set(legacy_legal_actions)
    for action_id in legal_actions:
        assert action_json_by_id[action_id] == legacy_action_json[action_id]


@requires_batch_api
def test_decision_context_spectrum_matches_legacy_spectrum_json_for_roll():
    catanatron_rs = _rust()
    game = _advance_to_roll(catanatron_rs, seed=5)
    legal_actions, action_json_by_id, spectrum_by_id = _decision_context(
        game, colors=("RED", "BLUE"), map_kind=None
    )
    roll_action_id = next(
        action_id for action_id in legal_actions if action_json_by_id[action_id][1] == "ROLL"
    )
    cached_spectrum = spectrum_by_id[roll_action_id]
    legacy_spectrum = _spectrum(game, action_json_by_id[roll_action_id])
    # Tiny floating-point differences are expected: decision_context_json and
    # spectrum_json compute/normalize probabilities via slightly different
    # code paths on the Rust side (both correct, not required to be
    # bit-identical) -- compare with a tolerance, not exact equality.
    assert len(cached_spectrum) == len(legacy_spectrum)
    for (cached_index, cached_probability), (legacy_index, legacy_probability) in zip(
        cached_spectrum, legacy_spectrum
    ):
        assert cached_index == legacy_index
        assert cached_probability == pytest.approx(legacy_probability, abs=1.0e-9)


@requires_batch_api
def test_search_result_identical_with_and_without_batch_api():
    """Regression: the batch-API path must be a pure performance optimization
    -- identical SearchResult (same seed, same evaluator) with
    use_batch_api=True vs False."""
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=11)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    result_new = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=16, p_full=1.0, use_batch_api=True), evaluator
    ).search(game.copy(), force_full=True)
    result_old = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=16, p_full=1.0, use_batch_api=False), evaluator
    ).search(game.copy(), force_full=True)

    assert result_new.selected_action == result_old.selected_action
    assert result_new.visit_counts == result_old.visit_counts
    assert result_new.improved_policy == result_old.improved_policy
    assert result_new.q_values == result_old.q_values
    assert result_new.root_value == pytest.approx(result_old.root_value)
    assert result_new.simulations_used == result_old.simulations_used


@requires_batch_api
def test_search_result_identical_with_and_without_batch_api_at_a_roll_decision():
    """Same identity check specifically at a forced-ROLL root, which exercises
    _enumerate_roll_outcomes' batched materialization + batched evaluation."""
    catanatron_rs = _rust()
    game = _advance_to_roll(catanatron_rs, seed=5)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    result_new = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=9, use_batch_api=True), evaluator
    ).search(game.copy())
    result_old = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=9, use_batch_api=False), evaluator
    ).search(game.copy())

    assert result_new.selected_action == result_old.selected_action
    assert result_new.root_value == pytest.approx(result_old.root_value)
    assert result_new.afterstate_values.keys() == result_old.afterstate_values.keys()
    for action_id in result_new.afterstate_values:
        assert result_new.afterstate_values[action_id] == pytest.approx(
            result_old.afterstate_values[action_id]
        )


@requires_batch_api
def test_evaluate_many_matches_per_request_evaluate_for_heuristic_evaluator():
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=2)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    legal_actions = tuple(game.playable_action_indices(["RED", "BLUE"], None))
    root_color = str(game.current_color())

    single_results = [
        evaluator.evaluate(game, legal_actions, root_color=root_color, colors=("RED", "BLUE"))
    ]
    many_results = evaluator.evaluate_many(
        [(game, legal_actions)], root_color=root_color, colors=("RED", "BLUE")
    )

    assert many_results == single_results


# ---------------------------------------------------------------------------
# f70 D1: noise-floor attenuation of the completed-Q rescale (flag-gated).
# ---------------------------------------------------------------------------


class _FakeColorGame:
    """Minimal stand-in exposing `current_color()` for node-level unit tests
    that never touch the Rust engine."""

    def __init__(self, color: str = "RED") -> None:
        self._color = color

    def current_color(self) -> str:
        return self._color


def _node_with(actions: dict[int, _GAction], *, prior_value: float = 0.0) -> _GNode:
    node = _GNode(game=_FakeColorGame("RED"), root_color="RED", prior_value=prior_value)
    node.actions = actions
    node.action_logits = {action_id: 0.0 for action_id in actions}
    return node


def test_noise_floor_disabled_is_exact_rescale_noop():
    """rescale_noise_floor_c == 0 -> `_rescaled_completed_q` is bit-identical
    to the raw min-max `_rescale_completed_q` (pure no-op default)."""
    mcts = _pure_mcts(
        GumbelChanceMCTSConfig(seed=0, rescale_noise_floor_c=0.0)
    )
    node = _node_with(
        {
            0: _GAction(prior=0.4, visits=3, value_sum=0.9),
            1: _GAction(prior=0.3, visits=2, value_sum=-0.2),
            2: _GAction(prior=0.3, visits=0),
        }
    )
    completed = mcts._completed_q(node)
    assert mcts._rescaled_completed_q(node, completed) == mcts._rescale_completed_q(completed)


def test_noise_floor_exact_tie_maps_all_to_half():
    """An exact tie (raw_spread == 0) -> alpha == 0 -> every rescaled value
    is 0.5 (a constant, so the prior's order is what survives)."""
    mcts = _pure_mcts(
        GumbelChanceMCTSConfig(seed=0, rescale_noise_floor_c=1.0)
    )
    node = _node_with(
        {
            0: _GAction(prior=0.5, visits=4, value_sum=1.2),  # q = 0.3
            1: _GAction(prior=0.5, visits=4, value_sum=1.2),  # q = 0.3
        }
    )
    completed = mcts._completed_q(node)  # both 0.3 -> tie
    out = mcts._rescaled_completed_q(node, completed)
    for value in out.values():
        assert value == pytest.approx(0.5)


def test_noise_floor_converges_to_plain_rescale_at_high_visits():
    """mean_visits -> large drives the noise floor -> 0 and alpha -> 1, so
    the attenuated rescale converges to the plain rescale."""
    mcts = _pure_mcts(
        GumbelChanceMCTSConfig(
            seed=0, rescale_noise_floor_c=1.0, sigma_eval=0.79
        )
    )
    v = 10**10
    node = _node_with(
        {
            0: _GAction(prior=0.4, visits=v, value_sum=0.9 * v),  # q = 0.9
            1: _GAction(prior=0.3, visits=v, value_sum=0.1 * v),  # q = 0.1
            2: _GAction(prior=0.3, visits=v, value_sum=-0.3 * v),  # q = -0.3
        }
    )
    completed = mcts._completed_q(node)
    plain = mcts._rescale_completed_q(completed)
    attenuated = mcts._rescaled_completed_q(node, completed)
    for action_id in plain:
        assert attenuated[action_id] == pytest.approx(plain[action_id], abs=1.0e-3)


def test_noise_floor_pulls_low_visit_spread_toward_half():
    """A modest raw spread observed over very few visits (spread near the
    noise floor) gets pulled toward the neutral 0.5, shrinking the rescaled
    range that would otherwise fill [0, 1]."""
    mcts = _pure_mcts(
        GumbelChanceMCTSConfig(
            seed=0, rescale_noise_floor_c=1.0, sigma_eval=0.79
        )
    )
    node = _node_with(
        {
            0: _GAction(prior=0.4, visits=1, value_sum=0.03),  # q = 0.03
            1: _GAction(prior=0.3, visits=1, value_sum=-0.01),  # q = -0.01
            2: _GAction(prior=0.3, visits=1, value_sum=0.00),  # q = 0.0
        }
    )
    completed = mcts._completed_q(node)
    plain = mcts._rescale_completed_q(completed)
    attenuated = mcts._rescaled_completed_q(node, completed)
    # Every attenuated value is strictly closer to 0.5 than the plain rescale.
    for action_id in plain:
        assert abs(attenuated[action_id] - 0.5) < abs(plain[action_id] - 0.5) + 1.0e-12
    # And the overall spread is compressed.
    assert (max(attenuated.values()) - min(attenuated.values())) < (
        max(plain.values()) - min(plain.values())
    )


def test_noise_floor_does_not_change_selection_when_disabled():
    """Full real search is unaffected when the flag is off (integration
    no-op): identical selected action / improved policy / visits with and
    without the field present at its default."""
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=7)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    base = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=32, n_fast=32, p_full=1.0), evaluator
    )
    off = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=32, n_fast=32, p_full=1.0, rescale_noise_floor_c=0.0),
        evaluator,
    )
    r_base = base.search(game.copy(), force_full=True)
    r_off = off.search(game.copy(), force_full=True)
    assert r_off.selected_action == r_base.selected_action
    assert r_off.visit_counts == r_base.visit_counts
    assert r_off.improved_policy == r_base.improved_policy


# ---------------------------------------------------------------------------
# f70 D2: variance-aware completed-Q shrinkage (flag-gated).
# ---------------------------------------------------------------------------


def test_variance_aware_q_disabled_is_noop():
    """variance_aware_q == False -> `_completed_q` is unchanged from the
    plain completed-Q (default OFF)."""
    plain = _pure_mcts(GumbelChanceMCTSConfig(seed=0, variance_aware_q=False))
    node = _node_with(
        {
            0: _GAction(prior=0.5, visits=10, value_sum=8.0, value_sq_sum=7.0),
            1: _GAction(prior=0.5, visits=3, value_sum=0.3, value_sq_sum=0.9),
        },
        prior_value=0.2,
    )
    completed = plain._completed_q(node)
    assert completed[0] == pytest.approx(0.8)  # sign*q, untouched
    assert completed[1] == pytest.approx(0.1)


def test_variance_aware_zero_per_action_variance_keeps_q():
    """A candidate whose per-visit backups were all identical (zero
    per-action variance -> SE == 0) keeps its exact completed-Q even with
    the flag on: shrink == 1."""
    aware = _pure_mcts(GumbelChanceMCTSConfig(seed=0, variance_aware_q=True))
    # value_sq_sum = visits * q^2 -> variance exactly 0 for both.
    node = _node_with(
        {
            0: _GAction(prior=0.5, visits=10, value_sum=8.0, value_sq_sum=10 * 0.8 * 0.8),
            1: _GAction(prior=0.5, visits=10, value_sum=2.0, value_sq_sum=10 * 0.2 * 0.2),
        }
    )
    completed = aware._completed_q(node)
    assert completed[0] == pytest.approx(0.8)
    assert completed[1] == pytest.approx(0.2)


def test_variance_aware_high_variance_candidate_shrinks_toward_v_mix():
    """A high-per-visit-variance, few-visit candidate is pulled toward v_mix,
    while a precisely-estimated (zero-variance) candidate is left alone."""
    prior_value = 0.0
    precise = _GAction(prior=0.5, visits=10, value_sum=8.0, value_sq_sum=10 * 0.8 * 0.8)  # q=0.8, var=0
    noisy = _GAction(prior=0.5, visits=2, value_sum=1.0, value_sq_sum=1.0)  # q=0.5, var=0.25
    node = _node_with({0: precise, 1: noisy}, prior_value=prior_value)

    plain = _pure_mcts(
        GumbelChanceMCTSConfig(seed=0, variance_aware_q=False)
    )._completed_q(node)
    aware = _pure_mcts(
        GumbelChanceMCTSConfig(seed=0, variance_aware_q=True, variance_aware_k=1.0)
    )._completed_q(node)

    # v_mix is prior-weighted: weighted_q = (0.8 + 0.5)/2 = 0.65; total_visits=12.
    v_mix = (prior_value + 12 * 0.65) / 13.0
    # Precise candidate: SE == 0 -> unchanged.
    assert aware[0] == pytest.approx(plain[0])
    assert aware[0] == pytest.approx(0.8)
    # Noisy candidate: moved from its raw q (0.5) toward v_mix.
    assert plain[1] == pytest.approx(0.5)
    assert abs(aware[1] - v_mix) < abs(plain[1] - v_mix)
    assert aware[1] != pytest.approx(plain[1])


def test_variance_aware_single_visited_candidate_is_untouched():
    """With fewer than 2 visited candidates there is no between-candidate
    signal to preserve, so shrinkage is skipped entirely."""
    aware = _pure_mcts(GumbelChanceMCTSConfig(seed=0, variance_aware_q=True))
    node = _node_with(
        {
            0: _GAction(prior=0.5, visits=3, value_sum=1.5, value_sq_sum=2.0),  # q=0.5, var>0
            1: _GAction(prior=0.5, visits=0),  # unvisited
        }
    )
    completed = aware._completed_q(node)
    assert completed[0] == pytest.approx(0.5)


def test_value_sq_sum_accumulates_during_real_search():
    """The running sum of squares is populated by real backups (so D2 has a
    variance estimate to use), and its mean-square dominates the squared mean
    (i.e. non-negative empirical variance)."""
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=5)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    mcts = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=1, n_full=32, n_fast=32, p_full=1.0), evaluator
    )
    root = _GNode(game=game.copy(), root_color=str(game.current_color()))
    mcts._expand(root)
    mcts._run_root_search(root, 32)
    visited = [stats for stats in root.actions.values() if stats.visits > 0]
    assert visited, "expected at least one visited root action"
    for stats in visited:
        assert stats.value_sq_sum >= 0.0
        assert stats.q_variance >= 0.0
        # sum of squares must be at least (sum^2 / visits) (Cauchy-Schwarz).
        assert stats.value_sq_sum >= (stats.value_sum * stats.value_sum) / stats.visits - 1.0e-9
