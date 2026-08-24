"""A seat that plays from equity rather than from a read.

Where ``HeuristicSeat`` guesses at its own strength from the shape of the hand,
this one asks `pokersolver <https://github.com/frla18cz/poker-solver>`_ for a
number: Monte Carlo equity against a range for every opponent still in the
hand, then a decision made against the price it is being offered.

That makes it slower than the built-in bot and considerably harder to beat by
bluffing, because it does not have a read to exploit — only arithmetic. It pays
in time rather than money: no key, no model, a second or two per decision.

The library is an optional dependency::

    pip install 'pokerarena[solver]'

The opponent range is deliberately one fixed, sane continuing range rather than
anything adaptive. A seat that modelled the table would be a different project;
this one exists to put a floor under the table — an opponent that is never
free money and never needs an account.
"""
from __future__ import annotations

import random

from ..contract.game_state import Action, ActionType, GameState, Street

# What the seat assumes an opponent is still holding. Wide enough that folding
# to it is rarely right, tight enough that it does not call off against a range
# nobody actually plays.
DEFAULT_RANGE = "22+, A2s+, K7s+, Q8s+, J8s+, T8s+, 97s+, 86s+, 75s+, 65s, 54s, A8o+, KTo+, QTo+, JTo"

# Equity margin demanded over the raw price before calling. Monte Carlo has a
# sampling error of its own, and the rake is not in these numbers at all.
CALL_MARGIN = 0.03
# Where value betting starts. Lower than the heuristic bot's, because this seat
# actually knows where it stands.
RAISE_EQUITY = 0.62
BET_FRACTION = 0.65
# Iterations per second of budget. Measured on the library's Monte Carlo; the
# point is that a bigger budget buys precision, not a different strategy.
ITERATIONS_PER_S = 2500
MIN_ITERATIONS = 400

# Raw equity is not what a hand actually wins. Preflop, a weak unpaired holding
# has to survive three more betting rounds it will usually be guessing through,
# so it realises less than its share of the pot; a pair realises nearly all of
# it, because it knows what it has flopped. Without this the seat called raises
# with 72o at 28% equity against a 23% price and lost money doing it.
PREFLOP_REALISATION = 0.82
PREFLOP_REALISATION_PAIR = 0.97


class SolverUnavailable(RuntimeError):
    """pokersolver is not installed."""


def _pokersolver():
    try:
        from pokersolver import equity as equity_mod
        from pokersolver.ranges.range import Range
    except ImportError as exc:                    # pragma: no cover - install hint
        raise SolverUnavailable(
            "solver seats need the optional dependency: "
            "pip install 'pokerarena[solver]'"
        ) from exc
    return equity_mod, Range


class SolverSeat:
    """Decides from Monte Carlo equity against a fixed opponent range."""

    def __init__(self, *, budget_s: float = 6.0,
                 opponent_range: str = DEFAULT_RANGE,
                 call_margin: float = CALL_MARGIN,
                 raise_equity: float = RAISE_EQUITY,
                 seed: int | None = None) -> None:
        self.budget_s = budget_s
        self.opponent_range = opponent_range
        self.call_margin = call_margin
        self.raise_equity = raise_equity
        self._seed = seed

    # -- equity ------------------------------------------------------------

    def _iterations(self, opponents: int) -> int:
        """The sampling budget, shared across however many opponents there are."""
        total = int(self.budget_s * ITERATIONS_PER_S / max(1, opponents))
        return max(MIN_ITERATIONS, total)

    def equity(self, hole: list[str], board: list[str], opponents: int,
               rng: random.Random) -> float:
        equity_mod, Range = _pokersolver()
        dead = set(hole) | set(board)
        combos = Range.parse(self.opponent_range).combos(dead_cards=dead)
        iterations = self._iterations(opponents)
        if not combos:
            # Every combo blocked is not a real situation, but the library
            # falls back to uniform rather than raising, so mirror that.
            return equity_mod.equity(hole, board, opponents, iterations, rng)
        if opponents <= 1:
            return equity_mod.equity_vs_range(
                hole, board, combos, 1, iterations, rng)
        return equity_mod.equity_vs_ranges(
            hole, board, [combos] * opponents, iterations, rng)

    # -- decision ----------------------------------------------------------

    def _realisation(self, hole: list[str], street: Street) -> float:
        """How much of its raw equity this hand can expect to actually win."""
        if street != Street.PREFLOP:
            # Postflop the board is out and equity is much closer to reality.
            return 1.0
        return (PREFLOP_REALISATION_PAIR if hole[0][0].upper() == hole[1][0].upper()
                else PREFLOP_REALISATION)

    def decide(self, state: GameState, rng: random.Random | None = None) -> Action:
        rng = rng or random.Random(self._seed)
        hole = [c for c in (state.hero_cards or []) if c]
        board = [c for c in (state.board or []) if c]
        if len(hole) < 2:
            # Nothing to compute with; never fold at no price.
            return Action(ActionType.CHECK if not state.to_call else ActionType.FOLD,
                          reason="no hole cards")

        live = [s for s in (state.seats or []) if getattr(s, "in_hand", True)
                and not getattr(s, "is_hero", False)]
        opponents = max(1, len(live))
        raw = self.equity(hole, board, opponents, rng)
        realisation = self._realisation(hole, state.street)
        eq = raw * realisation

        to_call = float(state.to_call or 0.0)
        pot = float(state.pot or 0.0)
        street = "preflop" if state.street == Street.PREFLOP else state.street.value
        meta = {"equity": round(eq, 4), "raw_equity": round(raw, 4),
                "realisation": realisation, "opponents": opponents,
                "iterations": self._iterations(opponents), "street": street}

        if to_call <= 0:
            if eq >= self.raise_equity and pot > 0:
                return Action(ActionType.BET, amount=round(pot * BET_FRACTION, 2),
                              reason=f"equity {eq:.0%} vs {opponents}", meta=meta)
            return Action(ActionType.CHECK,
                          reason=f"equity {eq:.0%}, checking is free", meta=meta)

        price = to_call / (pot + to_call) if (pot + to_call) > 0 else 1.0
        meta["price"] = round(price, 4)
        if eq >= self.raise_equity:
            amount = round((pot + to_call) * BET_FRACTION + to_call, 2)
            return Action(ActionType.RAISE, amount=amount,
                          reason=f"equity {eq:.0%} over price {price:.0%}", meta=meta)
        if eq >= price + self.call_margin:
            return Action(ActionType.CALL, amount=to_call,
                          reason=f"equity {eq:.0%} beats price {price:.0%}", meta=meta)
        return Action(ActionType.FOLD,
                      reason=f"equity {eq:.0%} under price {price:.0%}", meta=meta)
