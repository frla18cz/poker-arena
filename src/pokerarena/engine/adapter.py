"""The bridge between PokerKit and this package's contract.

Builds a `GameState` from PokerKit's state so any strategy can play, and maps an
`Action` back to a PokerKit move. Amounts are in chips — blinds 1/2, so
big_blind=2 by default, and a table with another structure passes its own.
Seating is SB at index 0, BB at index 1, button at index n-1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..contract.game_state import Action, ActionType, GameState, Seat, Street

_STREET = {0: Street.PREFLOP, 1: Street.FLOP, 2: Street.TURN, 3: Street.RIVER}
BIG_BLIND = 2


def card(c) -> str:
    """A PokerKit Card as 'Ts' or 'Ah'; repr(Card) is exactly that already."""
    return repr(c)


@dataclass
class HandLine:
    """The action line of a hand, recorded for every seat.

    It records **all** seats, and ``summary()`` selects from them by who is
    looking. Filtering during recording against a fixed hero meant that building
    a state for any other seat systematically lost the stats about the real
    hero — and at a table of independent strategies it quietly returned nothing.
    For a single hero the result is unchanged, since ``summary()`` excludes the
    observer anyway.
    """
    pf_raise_count: int = 0
    checkraise_seats: set = field(default_factory=set)
    bet_streets_by_seat: dict = field(default_factory=dict)   # seat -> {street_index}
    call_streets_by_seat: dict = field(default_factory=dict)  # seat -> {street_index}
    checked: set = field(default_factory=set)   # (street_index, seat) checks this hand
    last_aggressor: int | None = None

    def note(self, street_index: int, seat: int,
             aggressive: bool, called: bool, checked: bool) -> None:
        if checked:
            self.checked.add((street_index, seat))
        if aggressive:
            self.last_aggressor = seat
            if street_index == 0:
                self.pf_raise_count += 1
            else:
                self.bet_streets_by_seat.setdefault(seat, set()).add(street_index)
                if (street_index, seat) in self.checked:
                    self.checkraise_seats.add(seat)
        elif called and street_index >= 1:
            self.call_streets_by_seat.setdefault(seat, set()).add(street_index)

    def summary(self, hero_index: int, to_call: float, in_hand: list[int]) -> dict:
        """A summary of the relevant opponent."""
        opp = [s for s in in_hand if s != hero_index]
        if to_call > 0 and self.last_aggressor is not None and self.last_aggressor != hero_index:
            s = self.last_aggressor
            return {"checkraise": s in self.checkraise_seats,
                    "call_streets": len(self.call_streets_by_seat.get(s, ())),
                    "bet_streets": len(self.bet_streets_by_seat.get(s, ()))}
        return {"checkraise": any(s in self.checkraise_seats for s in opp),
                "call_streets": max((len(self.call_streets_by_seat.get(s, ())) for s in opp), default=0),
                "bet_streets": max((len(self.bet_streets_by_seat.get(s, ())) for s in opp), default=0)}


def build_gamestate(st, hero_index: int, pf_aggressor: int | None,
                    line: HandLine | None = None, *,
                    button_seat: int | None = None,
                    timeout_s: float | None = None,
                    history: list | None = None,
                    hand_id: str = "", table_id: str = "",
                    names: list[str] | None = None,
                    big_blind: int = BIG_BLIND,
                    small_blind: int | None = None) -> GameState:
    """A snapshot of the PokerKit state through the eyes of ``hero_index``.

    ``button_seat`` is optional because PokerKit posts blinds at indexes 0 and 1,
    which puts the button on the last seat. At a table with a moving button the
    players shift between hands, so a caller can ask for a different layout.

    ``timeout_s`` sets the budget for the decision. Without it,
    ``remaining_action_time_s()`` answers ``None`` and a model or solver runs
    with no time limit at all — which is fine in a study run and wrong at a
    table where someone is waiting.
    nikdo.

    ``big_blind`` and ``small_blind`` are in CHIPS and have to match what the
    table dealt with. Strategies work almost entirely in big blinds — matrices,
    SPR, sizing — so a wrong value here breaks nothing visibly; it quietly
    rescales every decision. ``small_blind=None`` means half the big blind.

    ``history`` is the chronology of ``ObservedAction`` — the story of the hand.
    Without it a model decides with less context than it would have in a real
    game, and no line is visible when reviewing the spot afterwards.

    ``names`` are the seat names in PokerKit's order. Without them seats carry
    ``name=None`` and an opponent profiler has nothing to look up, so any reads
    stay silent even when there is history to draw on.
    """
    n = st.player_count
    board = [card(c) for row in st.board_cards for c in row]
    to_call = st.checking_or_calling_amount or 0
    min_to = st.min_completion_betting_or_raising_to_amount
    bet_level = max(st.bets) if st.bets else 0
    min_raise = max(big_blind,
                    (min_to - bet_level) if min_to is not None else big_blind)

    seats = [
        Seat(seat_no=i, name=(names[i] if names and i < len(names) else None),
             stack=float(st.stacks[i]), committed=float(st.bets[i]),
             is_hero=(i == hero_index), in_hand=bool(st.statuses[i]))
        for i in range(n)
    ]
    gs = GameState(
        table_id=table_id,
        hand_id=hand_id,
        street=_STREET.get(st.street_index, Street.RIVER),
        hero_cards=[card(c) for c in st.hole_cards[hero_index]],
        board=board,
        pot=float(st.total_pot_amount),
        to_call=float(to_call),
        small_blind=float(big_blind / 2 if small_blind is None else small_blind),
        big_blind=float(big_blind),
        min_raise=float(min_raise),
        seats=seats,
        hero_seat=hero_index,
        button_seat=(n - 1) if button_seat is None else button_seat,
        sb_seat=0,
        bb_seat=1,
        hero_pf_aggressor=(pf_aggressor == hero_index),
        action_timeout_s=timeout_s,
        action_history=list(history or []),
    )
    if timeout_s is not None:
        # The deadline wins over `action_timeout_s`, so the budget really does
        # run down over the hand.
        gs.action_deadline_monotonic = time.monotonic() + timeout_s
    if line is not None:
        in_hand = [i for i in range(n) if bool(st.statuses[i])]
        summ = line.summary(hero_index, float(to_call), in_hand)
        gs.pf_raise_count = line.pf_raise_count
        gs.villain_checkraise = summ["checkraise"]
        gs.villain_bet_streets = summ["bet_streets"]
        gs.villain_call_streets = summ["call_streets"]
        gs.last_aggressor_seat = line.last_aggressor
    return gs


def apply_action(st, action: Action) -> bool:
    """Apply an Action to the PokerKit state.

    Returns True when it was aggression — a bet, raise or jam — which is what
    tracking the preflop aggressor needs."""
    t = action.type
    if t == ActionType.FOLD:
        if st.can_fold():
            st.fold()
        else:                       # nothing to fold to -> check
            st.check_or_call()
        return False
    if t in (ActionType.CHECK, ActionType.CALL):
        st.check_or_call()
        return False
    # BET / RAISE
    lo = st.min_completion_betting_or_raising_to_amount
    hi = st.max_completion_betting_or_raising_to_amount
    if lo is None:                  # cannot raise -> call instead
        st.check_or_call()
        return False
    amt = int(round(action.amount))
    amt = max(lo, min(hi, amt))
    st.complete_bet_or_raise_to(amt)
    return True
