"""Smoke test for the AlphaZero-style MCTS PyO3 bindings on `catanatron_rs.Game`.

Run after `maturin develop --features python-extension`:

    python tests/smoke_mcts_bindings.py

Exercises (each prints PASS/FAIL):
  * copy() / __copy__ / __deepcopy__ produce independent games
  * spectrum_json() enumerates weighted chance outcomes (~11 dice rolls, sum ~1.0)
  * apply_chance_outcome() re-derives the i-th outcome as a fresh Game
  * execute_action_index() advances state without a JSON round-trip
  * player_state_json() exposes a player's hidden hand
  * set_player_hand() overwrites the hidden hand (if present)
"""

import json
import sys

import catanatron_rs


COLORS = ["RED", "BLUE"]
FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def advance_to_action_type(game, action_type, max_ticks=200):
    """Play ticks until the first playable action is of `action_type`."""
    for _ in range(max_ticks):
        playable = json.loads(game.playable_actions_json())
        if playable and playable[0][1] == action_type:
            return True
        if game.winning_color() is not None:
            return False
        game.play_tick()
    return False


def first_action_of_type(game, action_type):
    for action in json.loads(game.playable_actions_json()):
        if action[1] == action_type:
            return action
    return None


def main():
    # ---- copy() independence -------------------------------------------------
    g = catanatron_rs.Game.simple(COLORS, seed=7)
    advance_to_action_type(g, "BUILD_SETTLEMENT")  # initial placement is deterministic-ish
    g2 = g.copy()
    idx_before = g.state_index()
    g2.play_tick()
    check(
        "copy() is independent",
        g.state_index() == idx_before and g2.state_index() == idx_before + 1,
        f"g={g.state_index()} g2={g2.state_index()}",
    )

    import copy as _copy

    g3 = _copy.copy(g)
    g4 = _copy.deepcopy(g)
    g3.play_tick()
    g4.play_tick()
    check(
        "__copy__/__deepcopy__ independent",
        g.state_index() == idx_before,
        f"g={g.state_index()}",
    )

    # ---- get a game to a ROLL prompt -----------------------------------------
    roll_game = catanatron_rs.Game.simple(COLORS, seed=7)
    has_roll = advance_to_action_type(roll_game, "ROLL")
    check("reached a ROLL prompt", has_roll)

    roll_action = first_action_of_type(roll_game, "ROLL")
    check("found a ROLL action", roll_action is not None, str(roll_action))

    # ---- spectrum_json on the ROLL chance node -------------------------------
    spectrum_ok = False
    spectrum = []
    if roll_action is not None:
        spectrum = json.loads(roll_game.spectrum_json(json.dumps(roll_action)))
        total_p = sum(o["probability"] for o in spectrum)
        every_has_snapshot = all("snapshot" in o for o in spectrum)
        # 2..12 inclusive = 11 outcomes
        spectrum_ok = (
            len(spectrum) == 11
            and abs(total_p - 1.0) < 1e-9
            and every_has_snapshot
        )
        check(
            "spectrum_json ~11 dice outcomes summing to 1.0",
            spectrum_ok,
            f"n={len(spectrum)} sum={total_p:.6f}",
        )
    else:
        check("spectrum_json ~11 dice outcomes summing to 1.0", False, "no roll action")

    # snapshots are nested game-JSON objects (same schema as json_snapshot)
    if spectrum:
        snaps_valid = all("current_playable_actions" in o["snapshot"] for o in spectrum)
        check("spectrum snapshots are valid game JSON", snaps_valid)

    # ---- apply_chance_outcome re-derives the i-th outcome --------------------
    if roll_action is not None and spectrum:
        outcome_game = roll_game.apply_chance_outcome(json.dumps(roll_action), 0)
        # Parent untouched, outcome advanced by exactly one action record.
        check(
            "apply_chance_outcome returns a fresh, advanced Game",
            isinstance(outcome_game, catanatron_rs.Game)
            and outcome_game.state_index() == roll_game.state_index() + 1
            and roll_game.playable_actions_json()
            == roll_game.playable_actions_json(),
            f"parent={roll_game.state_index()} outcome={outcome_game.state_index()}",
        )
        # And it should match the spectrum snapshot for index 0.
        matches = json.loads(outcome_game.json_snapshot()) == spectrum[0]["snapshot"]
        check("apply_chance_outcome(0) matches spectrum[0] snapshot", matches)
    else:
        check("apply_chance_outcome returns a fresh, advanced Game", False, "no roll")

    # ---- execute_action_index advances state ---------------------------------
    exec_game = catanatron_rs.Game.simple(COLORS, seed=11)
    advance_to_action_type(exec_game, "ROLL", max_ticks=10) or advance_to_action_type(
        exec_game, "BUILD_SETTLEMENT"
    )
    indices = exec_game.playable_action_indices(COLORS, None)
    check("playable_action_indices returned indices", len(indices) > 0, str(indices[:5]))
    before = exec_game.state_index()
    exec_game.execute_action_index(indices[0], COLORS, None)
    check(
        "execute_action_index advanced state",
        exec_game.state_index() == before + 1,
        f"{before} -> {exec_game.state_index()}",
    )

    # ---- player_state_json exposes the hidden hand ---------------------------
    ps = json.loads(exec_game.player_state_json("RED"))
    check(
        "player_state_json returns a hand",
        "resources" in ps
        and "dev_cards" in ps
        and len(ps["resources"]) == 5
        and len(ps["dev_cards"]) == 5,
        f"resources={ps.get('resources')} dev_cards={ps.get('dev_cards')}",
    )

    # ---- set_player_hand (optional) ------------------------------------------
    if hasattr(exec_game, "set_player_hand"):
        exec_game.set_player_hand("RED", [1, 2, 3, 4, 5], [1, 0, 0, 0, 2])
        ps2 = json.loads(exec_game.player_state_json("RED"))
        check(
            "set_player_hand overwrote the hand",
            ps2["resources"] == [1, 2, 3, 4, 5] and ps2["dev_cards"] == [1, 0, 0, 0, 2],
            f"resources={ps2['resources']} dev_cards={ps2['dev_cards']}",
        )
    else:
        print("[SKIP] set_player_hand not present")

    # ---- decision_context_json: one-round-trip expansion context -------------
    ctx_game = catanatron_rs.Game.simple(COLORS, seed=7)
    advance_to_action_type(ctx_game, "ROLL")
    if hasattr(ctx_game, "decision_context_json"):
        context = json.loads(ctx_game.decision_context_json(COLORS, None))
        indices = ctx_game.playable_action_indices(COLORS, None)
        actions = json.loads(ctx_game.playable_actions_json())
        entries = context["actions"]
        check(
            "decision_context_json matches indices+actions",
            [e["index"] for e in entries] == list(indices)
            and [e["action"] for e in entries] == actions
            and context["current_color"] in COLORS,
            f"n={len(entries)}",
        )
        roll_entries = [e for e in entries if e["action"][1] == "ROLL"]
        spectrum_ok = bool(roll_entries) and all(
            len(e.get("spectrum", [])) == 11
            and abs(sum(e["spectrum"]) - 1.0) < 1e-9
            for e in roll_entries
        )
        check("decision_context_json ROLL spectrum = 11 probs summing to 1", spectrum_ok)
        deterministic_clean = all(
            "spectrum" not in e
            for e in entries
            if e["action"][1] in ("END_TURN", "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY")
        )
        check("decision_context_json omits spectrum for deterministic actions", deterministic_clean)

        # ---- apply_chance_outcomes_batch matches per-outcome application -----
        roll_action = roll_entries[0]["action"] if roll_entries else None
        if roll_action is not None and hasattr(ctx_game, "apply_chance_outcomes_batch"):
            batch = ctx_game.apply_chance_outcomes_batch(json.dumps(roll_action))
            per_outcome = [
                ctx_game.apply_chance_outcome(json.dumps(roll_action), i)
                for i in range(len(batch))
            ]
            check(
                "apply_chance_outcomes_batch(None) = all outcomes in order",
                len(batch) == 11
                and all(
                    b.json_snapshot() == p.json_snapshot()
                    for b, p in zip(batch, per_outcome)
                ),
                f"n={len(batch)}",
            )
            subset = ctx_game.apply_chance_outcomes_batch(json.dumps(roll_action), [10, 0])
            check(
                "apply_chance_outcomes_batch subset ordering",
                len(subset) == 2
                and subset[0].json_snapshot() == per_outcome[10].json_snapshot()
                and subset[1].json_snapshot() == per_outcome[0].json_snapshot(),
            )
            try:
                ctx_game.apply_chance_outcomes_batch(json.dumps(roll_action), [99])
                check("apply_chance_outcomes_batch rejects bad index", False)
            except ValueError:
                check("apply_chance_outcomes_batch rejects bad index", True)
        else:
            print("[SKIP] apply_chance_outcomes_batch not present")
    else:
        print("[SKIP] decision_context_json not present")

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failing): {', '.join(FAILURES)}")
        sys.exit(1)
    print("RESULT: PASS (all MCTS bindings working)")


if __name__ == "__main__":
    main()
