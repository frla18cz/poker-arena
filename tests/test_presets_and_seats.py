"""Every shipped preset must be playable, and every kind it names must exist.

The arena was extracted from a larger project, and the extraction renamed the
seat vocabulary in one place but not the others: presets went on building
`bot`, `gto` and `chart` seats that `SEAT_KINDS` no longer knew, so six of the
nine presets raised the moment anyone chose them. Nothing caught it, because
the tests only ever built seats by hand.

These tests walk the shipped configuration itself, which is the part a stranger
touches first.
"""
from __future__ import annotations

import random

import pytest

import pokerarena.seats                       # registers the built-in kinds
from pokerarena.contract.game_state import ActionType, GameState, Seat, Street
from pokerarena.engine.seat_registry import registered_kinds
from pokerarena.engine.table_config import (
    PRESETS, SEAT_LABELS, SeatConfig, known_seat_kinds,
)
from pokerarena.seats.solver import SolverSeat


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_validates(name: str) -> None:
    """A preset in the menu is a promise that choosing it starts a table."""
    PRESETS[name]().validate()


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_seat_kind_is_known(name: str) -> None:
    known = known_seat_kinds()
    for seat in PRESETS[name]().seats:
        assert seat.kind in known, f"{name}: seat {seat.name!r} has kind {seat.kind!r}"


def test_default_seat_kind_is_usable() -> None:
    """The editor builds new seats from the defaults, so they must validate."""
    SeatConfig("new seat").validate()


def test_every_kind_has_a_label_and_a_factory() -> None:
    """The UI shows labels and the table needs factories; `human` has no factory."""
    for kind in known_seat_kinds():
        assert kind in SEAT_LABELS, f"{kind} has no label for the UI"
        if kind != "human":
            assert kind in registered_kinds(), f"{kind} has no factory"


# -- the solver seat -------------------------------------------------------

def _spot(hole, board, pot, to_call, opponents=1) -> GameState:
    seats = [Seat(seat_no=0, name="hero", is_hero=True, in_hand=True)]
    seats += [Seat(seat_no=i + 1, name=f"v{i}", in_hand=True)
              for i in range(opponents)]
    street = (Street.PREFLOP if not board
              else Street.FLOP if len(board) == 3
              else Street.TURN if len(board) == 4 else Street.RIVER)
    return GameState(street=street, hero_cards=hole, board=board, pot=pot,
                     to_call=to_call, seats=seats)


def _seat() -> SolverSeat:
    return SolverSeat(budget_s=1.0, seed=7)


def test_solver_folds_the_worst_hand_and_raises_the_best() -> None:
    """The two ends of the range, where pot odds leave no room for argument."""
    seat = _seat()
    assert seat.decide(_spot(["7d", "2c"], [], 10, 3)).type is ActionType.FOLD
    assert seat.decide(_spot(["As", "Ah"], [], 10, 3)).type is ActionType.RAISE


def test_solver_never_folds_when_checking_is_free() -> None:
    action = _seat().decide(_spot(["7d", "2c"], ["Qs", "Js", "9h"], 10, 0))
    assert action.type is not ActionType.FOLD


def test_solver_folds_air_facing_a_river_bet() -> None:
    """No draws left and nothing made: the price cannot be right."""
    action = _seat().decide(
        _spot(["8d", "4c"], ["Qs", "Js", "2s", "7h", "3d"], 20, 15))
    assert action.type is ActionType.FOLD


def test_preflop_realisation_discounts_raw_equity() -> None:
    """Raw equity overstates a weak unpaired hand; the seat records both."""
    action = _seat().decide(_spot(["7d", "2c"], [], 10, 3))
    assert action.meta["equity"] < action.meta["raw_equity"]
    # A pair keeps nearly all of its equity, because it knows what it flopped.
    pair = _seat().decide(_spot(["2d", "2c"], [], 10, 3))
    assert pair.meta["realisation"] > action.meta["realisation"]


def test_solver_equity_is_reproducible() -> None:
    """A fixed seed must give a fixed answer, or nothing here is testable."""
    spot = _spot(["As", "Ks"], ["Qs", "Js", "2h"], 20, 6)
    first = SolverSeat(budget_s=1.0, seed=11).decide(spot).meta["raw_equity"]
    second = SolverSeat(budget_s=1.0, seed=11).decide(spot).meta["raw_equity"]
    assert first == second


def test_solver_asks_for_more_equity_against_more_opponents() -> None:
    """The same hand is worth less the more people can beat it."""
    heads_up = _seat().decide(_spot(["Ad", "Kd"], ["7c", "2h", "9s"], 20, 5, 1))
    multiway = _seat().decide(_spot(["Ad", "Kd"], ["7c", "2h", "9s"], 20, 5, 4))
    assert multiway.meta["raw_equity"] < heads_up.meta["raw_equity"]


# -- the registry has to be populated for a *stranger's* process -----------

def test_a_fresh_process_can_play_without_importing_seats() -> None:
    """`poker-arena` must deal a hand with no import beyond the entry point.

    Only the test modules imported `pokerarena.seats`, so nothing registered
    the built-in kinds when the CLI ran. Every bot seat got `strategy = None`
    and the table died on its first decision with

        AttributeError: 'NoneType' object has no attribute 'decide'

    which meant the published arena could not play a single hand against its
    own built-in bot — the one thing the README promises works out of the box.
    The tests missed it precisely because they did the import themselves, so
    this one runs in a subprocess that does not.
    """
    import subprocess
    import sys

    script = (
        "from pokerarena.engine.table_config import SeatConfig, TableConfig\n"
        "from pokerarena.engine.table import Table\n"
        "config = TableConfig(seats=tuple(\n"
        "    SeatConfig(f'Bot{i}', kind='heuristic') for i in range(1, 4)))\n"
        "table = Table(config)\n"
        "table.play_hand()\n"
        "print('OK', sum(p.hands for p in table.players))\n"
    )
    done = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True)
    assert done.returncode == 0, (
        f"a fresh process could not play a hand:\n{done.stderr}")
    assert done.stdout.startswith("OK")
