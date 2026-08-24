"""The contract: the table, its seats, and any strategy speak only this.

Everything depends on this model and nothing else, so a table, a bot and a UI
can be developed and tested against fixtures independently. The table fills the
state in, a strategy reads it and answers with an Action, the table applies it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass
class Seat:
    """One seat at the table."""
    seat_no: int
    name: str | None = None
    stack: float = 0.0
    committed: float = 0.0     # what they have put in this betting round
    is_hero: bool = False
    in_hand: bool = True       # False po foldu
    is_active: bool = True     # seated and playing, not sitting out
    # True when ``name`` is a per-table alias ("Player 3") rather than an
    # identity: it must not be used to look a player up across games. New fields
    # belong at the end — some code constructs Seat positionally.
    is_anonymous: bool = False


@dataclass(frozen=True)
class ObservedAction:
    """One action that actually happened, in the hand's chronology.

    ``amount`` is the increment this action put in; ``committed_after`` is the
    player's total for the street once it was applied. Keeping both means a
    reader never has to guess whether a number is a raise-to or an increment.
    """
    street: Street
    seat_no: int
    position: str
    action: str
    amount: float = 0.0
    committed_after: float = 0.0


@dataclass
class Action:
    """A decision, for the table to carry out."""
    type: ActionType
    amount: float = 0.0        # the total to bet for BET/RAISE; 0 otherwise
    reason: str = ""           # human-readable explanation, for display and logs
    meta: dict = field(default_factory=dict)  # anything behind it: equity, position, range

    def __str__(self) -> str:
        base = self.type.value.upper()
        if self.amount:
            base += f" {self.amount:g}"
        return base


@dataclass
class GameState:
    """The complete state of a table at the moment someone has to act."""
    table_id: str = ""
    street: Street = Street.PREFLOP
    hero_cards: list[str] = field(default_factory=list)   # 2 karty
    board: list[str] = field(default_factory=list)        # 0–5 karet
    pot: float = 0.0
    to_call: float = 0.0        # what it costs to call
    small_blind: float = 0.0
    big_blind: float = 0.0
    min_raise: float = 0.0      # the smallest legal raise
    seats: list[Seat] = field(default_factory=list)
    hero_seat: int | None = None
    button_seat: int | None = None
    sb_seat: int | None = None
    bb_seat: int | None = None
    hero_pf_aggressor: bool = False   # were they the last preflop raiser?
    hand_id: str = ""
    # --- the line of the hand ---
    pf_raise_count: int = 0           # preflop raises: open=1, 3bet=2, 4bet=3
    villain_checkraise: bool = False  # an opponent check-raised on any street
    villain_bet_streets: int = 0      # postflop streets where an opponent bet or raised
    villain_call_streets: int = 0     # postflop streets where an opponent called
    last_aggressor_seat: int | None = None  # seat of the last bet or raise
    # The preflop sequence in order: [(seat, "F"|"C"|"R")], fold/call/raise only —
    # blinds and checks are left out. An empty list is meaningful (UTG first to
    # act is a valid RFI); ``None`` means nobody filled it in, which is a
    # different thing entirely, and callers depend on telling the two apart.
    preflop_actions: list[tuple[int, str]] | None = None
    # The full chronology, postflop included, per seat.
    action_history: list[ObservedAction] = field(default_factory=list)
    # The base time to act. Whoever decides and whoever acts share one absolute
    # deadline; any time bank is tracked separately so it cannot be counted twice.
    action_timeout_s: float | None = None
    # Time bank metadata. The deadline is a monotonic timestamp on this host.
    time_bank_state: int | None = None
    time_bank_left_s: float | None = None
    action_received_monotonic: float | None = None
    action_deadline_monotonic: float | None = None
    # Hard inconsistencies found while building the state (pot, to-call,
    # committed). A non-empty list means nothing here can be trusted to bet on.
    state_errors: list[dict] = field(default_factory=list)

    @property
    def hero(self) -> Seat | None:
        for s in self.seats:
            if s.is_hero:
                return s
        return None

    @property
    def n_opponents(self) -> int:
        """How many opponents are still in the hand."""
        return sum(1 for s in self.seats if s.in_hand and not s.is_hero)

    @property
    def hero_can_check(self) -> bool:
        return self.to_call <= 0.0

    def remaining_action_time_s(self, now: float | None = None) -> float | None:
        """How much time is actually left to act."""
        if self.action_deadline_monotonic is not None:
            return max(0.0, self.action_deadline_monotonic
                       - (time.monotonic() if now is None else now))
        if self.action_timeout_s is None:
            return None
        bank = (self.time_bank_left_s or 0.0) if self.time_bank_state == 2 else 0.0
        return max(0.0, self.action_timeout_s + bank)

    def validate(self) -> None:
        """A basic sanity check; raises ValueError when the state contradicts itself."""
        if self.hero_cards and len(self.hero_cards) != 2:
            raise ValueError(f"a hand is 2 cards, got {len(self.hero_cards)}")
        if len(self.board) > 5:
            raise ValueError(f"a board is at most 5 cards, got {len(self.board)}")
        if self.to_call < 0:
            raise ValueError("to_call cannot be negative")
