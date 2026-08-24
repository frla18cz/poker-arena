"""The seat registry: a ``kind`` maps to a factory that builds a strategy.

The table can deal a hand with anything that has ``decide(gs, rng) -> Action``.
Who builds that strategy is not the table's business, so it asks the registry
instead of knowing the answer.

``human`` has no factory — the table waits for a person rather than asking a
strategy. Everything else, built in or not, arrives here through ``register``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .table_config import SeatConfig, TableConfig


@dataclass(frozen=True)
class SeatBuild:
    """Everything a factory needs to build a strategy for a seat."""

    seat: SeatConfig
    config: TableConfig
    out_dir: Path | None
    iterations: int
    # The table's opponent profiler, keyed by seat name, or ``None``.
    profiler: object = None


# Returns a strategy, or ``None`` when the seat has none (a human).
SeatFactory = Callable[[SeatBuild], object]

_factories: dict[str, SeatFactory] = {}


def register(kind: str, factory: SeatFactory) -> None:
    """Register a factory for a seat kind; a later call replaces an earlier one."""
    _factories[kind] = factory


def factory_for(kind: str) -> SeatFactory | None:
    return _factories.get(kind)


def registered_kinds() -> tuple[str, ...]:
    return tuple(sorted(_factories))


# --- opponent profiling ----------------------------------------------------
#
# The table does not build a profiler: reading opponents is a strategy concern,
# not a rule of poker. Whoever has one registers it here; without one, the game
# is played without reads.
ProfileFeedFactory = Callable[[TableConfig, Path | None, Path | None], object]

_profile_feed: ProfileFeedFactory | None = None


def register_profile_feed(factory: ProfileFeedFactory | None) -> None:
    global _profile_feed
    _profile_feed = factory


def build_profile_feed(config: TableConfig, *, out_dir: Path | None = None,
                       history_root: Path | None = None):
    """A profiler for this table, or ``None`` if nobody supplied one."""
    if _profile_feed is None:
        return None
    return _profile_feed(config, out_dir, history_root)


# --- decision logging ------------------------------------------------------
#
# Writing decisions to a log is an audit concern, not part of the game. With
# nothing registered, the table simply does not log.
DecisionLogFactory = Callable[[SeatConfig, Path], object]

_decision_log: DecisionLogFactory | None = None


def register_decision_log(factory: DecisionLogFactory | None) -> None:
    global _decision_log
    _decision_log = factory


def build_decision_log(seat: SeatConfig, out_dir: Path):
    if _decision_log is None:
        return None
    return _decision_log(seat, out_dir)


def clear() -> None:
    """For tests: empty the registry."""
    _factories.clear()
    register_profile_feed(None)
    register_decision_log(None)
