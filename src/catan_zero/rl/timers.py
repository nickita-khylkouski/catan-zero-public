from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TimerProfileName = Literal["very_fast", "fast", "normal", "slow", "very_slow"]
TimerPhase = Literal[
    "initial_build",
    "roll",
    "robber",
    "discard",
    "trade_response",
    "main_turn",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class ColonistTimerProfile:
    """Virtual Colonist-style timing profile.

    These are training/evaluation budgets, not wall-clock sleeps. Public
    Colonist timer notes provide exact values for initial build, robber/discard,
    and roll phases. Main-turn values are benchmark policy, so they remain
    explicit config rather than claimed Colonist facts.
    """

    name: TimerProfileName
    initial_build_seconds: int
    robber_discard_seconds: int
    roll_seconds: int
    main_turn_seconds: int
    trade_response_seconds: int

    def budget_for_phase(self, phase: TimerPhase) -> int:
        if phase == "initial_build":
            return self.initial_build_seconds
        if phase in ("robber", "discard"):
            return self.robber_discard_seconds
        if phase == "roll":
            return self.roll_seconds
        if phase == "trade_response":
            return self.trade_response_seconds
        return self.main_turn_seconds


COLONIST_TIMER_PROFILES: dict[TimerProfileName, ColonistTimerProfile] = {
    "very_fast": ColonistTimerProfile("very_fast", 60, 10, 10, 30, 10),
    "fast": ColonistTimerProfile("fast", 120, 20, 10, 60, 20),
    "normal": ColonistTimerProfile("normal", 180, 40, 20, 90, 40),
    "slow": ColonistTimerProfile("slow", 360, 80, 60, 180, 80),
    "very_slow": ColonistTimerProfile("very_slow", 18000, 3000, 3000, 18000, 3000),
}


def timer_phase_from_prompt(prompt_name: str, playable_action_types: tuple[str, ...]) -> TimerPhase:
    if prompt_name in ("BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD"):
        return "initial_build"
    if prompt_name == "DISCARD":
        return "discard"
    if prompt_name == "MOVE_ROBBER":
        return "robber"
    if prompt_name in ("DECIDE_TRADE", "DECIDE_ACCEPTEES"):
        return "trade_response"
    if "ROLL" in playable_action_types:
        return "roll"
    if prompt_name == "PLAY_TURN":
        return "main_turn"
    return "unknown"
