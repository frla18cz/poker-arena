"""A bot of your own at the table, in about thirty lines.

Run it against the built-in opponents::

    python examples/custom_seat.py

Nothing here is special to this file. A seat is any object with

    decide(state, rng=None) -> Action

and registering it under a name makes that name a legal seat kind. The table
never learns what is behind the chair — it asks, and plays the answer.

That also means your bot does not have to live in this repository. Import
`pokerarena`, register your factory, and build a table: the strategy can stay
in your own private project, closed source, and still sit down here.
"""
from __future__ import annotations

import random

from pokerarena.contract.game_state import Action, ActionType, GameState
from pokerarena.engine.seat_registry import SeatBuild, register
from pokerarena.engine.table import Table
from pokerarena.engine.table_config import SeatConfig, TableConfig


class CallingStation:
    """Calls anything it is offered, bets when checked to. Terrible, on purpose.

    It is here to show the shape of a strategy, not to win. Note what it does
    *not* do: it never returns an action the table did not offer, and it says
    why, so the hand log can show its reasoning next to everyone else's.
    """

    def decide(self, state: GameState, rng: random.Random | None = None) -> Action:
        to_call = float(state.to_call or 0.0)
        if to_call > 0:
            return Action(ActionType.CALL, amount=to_call,
                          reason="curiosity beats discipline")
        pot = float(state.pot or 0.0)
        if pot > 0:
            return Action(ActionType.BET, amount=round(pot * 0.5, 2),
                          reason="nobody bet, so I will")
        return Action(ActionType.CHECK, reason="nothing to do")


# The factory receives everything the table knows about the seat it is filling:
# `build.seat` is the SeatConfig, `build.config` the table. Use the table's seed
# if your strategy is random, so a game can be replayed exactly.
register("calling_station", lambda build: CallingStation())


def main() -> None:
    config = TableConfig(
        seats=(
            SeatConfig("station", kind="calling_station"),
            SeatConfig("house-1", kind="heuristic"),
            SeatConfig("house-2", kind="heuristic"),
        ),
        start_stack=200,
        # Off, so the chips at the end are the chips from the start. With
        # rebuys on, a busted seat is topped back up and the stacks no longer
        # add up to what everyone sat down with.
        rebuy=False,
    )
    table = Table(config)
    for _ in range(20):
        if len(table.seated()) < 2:
            print("everyone else is broke; the game ends early\n")
            break
        table.play_hand()

    print(f"{'seat':12} {'kind':16} {'stack':>8} {'net':>7}")
    for player in table.players:
        net = player.stack - config.start_stack
        print(f"{player.name:12} {player.config.kind:16} "
              f"{player.stack:8.0f} {net:+7.0f}")
    total = sum(p.stack for p in table.players)
    print(f"{'':12} {'total':16} {total:8.0f}   "
          f"(everyone sat down with {config.start_stack * len(table.players)})")


if __name__ == "__main__":
    main()
