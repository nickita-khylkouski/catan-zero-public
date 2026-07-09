from __future__ import annotations

from dataclasses import dataclass

from catan_zero.engine import CatanEngine
from catan_zero.schemas import SeedBundle


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    ruleset_id: str = "CatanBench-4P-Full-v1"
    games_per_lineup: int = 10_000
    decision_time_ms: int = 1000
    structured_trade: bool = True
    free_form_chat: bool = False


@dataclass(frozen=True, slots=True)
class GameResult:
    seed_bundle: SeedBundle
    winner_seat: int
    final_victory_points: tuple[int, int, int, int]
    turns: int


def run_seeded_game(engine: CatanEngine, seed_bundle: SeedBundle) -> GameResult:
    """Run one game once policies are wired to the engine.

    Stage 0 defines the return contract. Policy orchestration will be added
    after the simulator adapter is mapped and tested.
    """

    engine.reset(seed_bundle)
    raise NotImplementedError("policy orchestration is added after adapter certification")

