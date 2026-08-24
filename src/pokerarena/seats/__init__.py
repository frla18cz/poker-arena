"""The seats the arena ships with, and the registry that holds them.

Importing this module registers ``heuristic``, ``llm`` and ``solver``.
``human`` needs no factory — the table waits for a person instead of asking a
strategy.

To add your own, register a factory and the table will accept it as a seat
kind without any change here:

    from pokerarena.engine.seat_registry import register

    register("my_bot", lambda build: MyStrategy(seed=build.seat.name))
"""
from __future__ import annotations

from ..engine.seat_registry import SeatBuild, register
from .heuristic import HeuristicSeat
from .llm import LlmSeat
from .solver import SolverSeat

__all__ = ["HeuristicSeat", "LlmSeat", "SolverSeat", "SeatBuild", "register"]


def _heuristic(build: SeatBuild) -> HeuristicSeat:
    return HeuristicSeat()


def _llm(build: SeatBuild) -> LlmSeat:
    seat = build.seat
    return LlmSeat(
        seat.decision_model,
        name=seat.name,
        temperature=0.2 if seat.temperature is None else seat.temperature,
        timeout_s=seat.timeout_s or build.config.timeout_s,
        big_blind=float(getattr(build.config, "big_blind", 1.0) or 1.0),
        variant=seat.prompt_variant,
    )


def _solver(build: SeatBuild) -> SolverSeat:
    seat = build.seat
    return SolverSeat(budget_s=seat.solver_budget_s, seed=build.config.seed)


register("heuristic", _heuristic)
register("llm", _llm)
register("solver", _solver)
