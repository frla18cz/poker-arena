"""A pot-odds bot with no model behind it — the table's built-in opponent.

Deliberately simple, and deliberately honest about it. It counts outs on the
flop and turn, compares its rough equity to the price it is being offered, and
raises only when it is well ahead of that price. No ranges, no reads, no
bluffing.

That makes it a fair sparring partner for a beginner and a punching bag for
anyone else — which is the point. It exists so a table is playable the moment
you install the package, with no API key and no model download. Anything
stronger plugs in through ``seat_registry``.
"""
from __future__ import annotations

import random

from ..contract.game_state import Action, ActionType, GameState, Street

# How much equity the bot demands over the price before calling. Without a
# margin it would call every coinflip, and over time the rake and its own
# imprecision would eat it alive.
CALL_MARGIN = 0.04
# Where betting replaces calling. Deliberately high: the bot never bluffs, so a
# bet from it always means strength — which makes it something to learn against.
RAISE_EQUITY = 0.66
# What share of the pot it bets when it does bet.
BET_FRACTION = 0.6

_RANKS = "23456789TJQKA"


def _rank(card: str) -> int:
    return _RANKS.index(card[0].upper()) if card and card[0].upper() in _RANKS else 0


def _preflop_strength(hole: list[str]) -> float:
    """Rough starting-hand strength in 0..1. No chart, just the shape of it."""
    if len(hole) < 2:
        return 0.0
    high, low = sorted((_rank(hole[0]), _rank(hole[1])), reverse=True)
    suited = hole[0][-1].lower() == hole[1][-1].lower()
    gap = high - low
    if gap == 0:                                  # a pair
        return 0.5 + high / 24
    score = 0.30 + (high + low) / 60
    if suited:
        score += 0.06
    if gap <= 2:                                  # connected, or a small gap
        score += 0.05
    elif gap >= 6:
        score -= 0.06
    return max(0.0, min(0.95, score))


def _postflop_strength(hole: list[str], board: list[str]) -> float:
    """Equity estimated from what the hand hit; draws count as a chance."""
    cards = [c for c in hole + board if c]
    ranks = [c[0].upper() for c in cards]
    suits = [c[-1].lower() for c in cards]
    hole_ranks = [c[0].upper() for c in hole]

    pairs = {r for r in hole_ranks if ranks.count(r) >= 2}
    trips = {r for r in hole_ranks if ranks.count(r) >= 3}
    flush_draw = any(suits.count(s) == 4 for s in set(suits))
    flush = any(suits.count(s) >= 5 for s in set(suits))

    if flush:
        return 0.88
    if trips:
        return 0.82
    if len(pairs) >= 2:
        return 0.70
    strength = 0.30
    if pairs:
        # Top pair is a different hand from bottom pair, so the height of the
        # board decides.
        board_high = max((_rank(c) for c in board), default=0)
        hole_pair_high = max(_RANKS.index(r) for r in pairs)
        strength = 0.62 if hole_pair_high >= board_high else 0.46
    if flush_draw:
        # A draw with two cards to come is worth roughly two thirds of a made
        # hand; on the river there is nothing left to hit, so it adds nothing.
        strength = max(strength, 0.52 if len(board) < 5 else strength)
    return strength


class HeuristicSeat:
    """Decides from the price and a rough read of its own hand."""

    def __init__(self, *, call_margin: float = CALL_MARGIN,
                 raise_equity: float = RAISE_EQUITY) -> None:
        self.call_margin = call_margin
        self.raise_equity = raise_equity

    def decide(self, state: GameState, rng: random.Random | None = None) -> Action:
        hole = list(state.hero_cards or [])
        board = list(state.board or [])
        strength = (_preflop_strength(hole) if state.street == Street.PREFLOP
                    else _postflop_strength(hole, board))

        to_call = float(state.to_call or 0.0)
        pot = float(state.pot or 0.0)

        if to_call <= 0:
            # Never fold when checking is free. Folding at no price is the one
            # mistake a complete beginner spots immediately.
            if strength >= self.raise_equity and pot > 0:
                return Action(ActionType.BET, amount=round(pot * BET_FRACTION, 2),
                              reason=f"hand looks {strength:.0%}")
            return Action(ActionType.CHECK, reason="checking is free")

        price = to_call / (pot + to_call) if (pot + to_call) > 0 else 1.0
        if strength >= self.raise_equity:
            return Action(ActionType.RAISE, amount=round((pot + to_call) * BET_FRACTION
                                                         + to_call, 2),
                          reason=f"hand looks {strength:.0%} vs price {price:.0%}")
        if strength >= price + self.call_margin:
            return Action(ActionType.CALL, amount=to_call,
                          reason=f"hand looks {strength:.0%} vs price {price:.0%}")
        return Action(ActionType.FOLD,
                      reason=f"hand looks {strength:.0%} under price {price:.0%}")
