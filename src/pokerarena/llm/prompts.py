"""The arena's own prompt: state in, one action out.

One variant, kept short on purpose. It describes the spot, lists what is
actually legal, and asks for a single line of JSON. There is no range
estimation pass, no critic, no mixed strategy — a table needs a decision, and
every extra round trip is a human waiting.

If you have a better prompt, you do not edit this one: register your own
catalogue through ``engine.prompt_catalog.set_catalog`` and keep it in your own
repository.
"""
from __future__ import annotations

from ..contract.game_state import GameState, Street

SYSTEM = """You are playing No-Limit Texas Hold'em at a friendly table.

Play to win chips, not to look clever. Fold weak hands rather than calling out \
of curiosity, bet when you are ahead, and do not bluff into two opponents \
without a reason. Position matters: act tighter early, wider on the button.

Reply with one line of JSON and nothing else:

  {"action": "fold|check|call|bet|raise", "amount": <number>, "why": "<8 words>"}

Rules for the reply:
* `action` must be one of the legal actions listed in the prompt.
* `amount` is the total you put in for bet/raise, in chips. Omit it for \
fold/check/call.
* Never invent an action that is not listed. If unsure, prefer check over \
call, and call over raise.
"""

_STREET_NAME = {
    Street.PREFLOP: "preflop",
    Street.FLOP: "flop",
    Street.TURN: "turn",
    Street.RIVER: "river",
}


def render(state: GameState, legal: list[str], *, seat_name: str = "you",
           big_blind: float = 1.0) -> str:
    """The user half of the prompt — the spot as plain text."""
    board = " ".join(state.board or []) or "—"
    hole = " ".join(state.hero_cards or []) or "??"
    bb = big_blind or 1.0

    lines = [
        f"Street: {_STREET_NAME.get(state.street, 'preflop')}",
        f"Your cards: {hole}",
        f"Board: {board}",
        f"Pot: {state.pot:.2f} ({state.pot / bb:.1f} bb)",
    ]
    if state.to_call:
        price = state.to_call / (state.pot + state.to_call) if state.pot else 1.0
        lines.append(f"To call: {state.to_call:.2f} "
                     f"({state.to_call / bb:.1f} bb) — you need "
                     f"{price:.0%} equity to break even")
    else:
        lines.append("To call: 0 — checking is free")
    if getattr(state, "hero_stack", None) is not None:
        lines.append(f"Your stack: {state.hero_stack:.2f} "
                     f"({state.hero_stack / bb:.1f} bb)")
    lines.append(f"Players still in the hand: {len(state.seats) or 2}")
    lines.append(f"Legal actions: {', '.join(legal)}")
    lines.append("")
    lines.append(f"You are {seat_name}. Reply with the JSON line only.")
    return "\n".join(lines)


# --- a two-pass variant ----------------------------------------------------
#
# The one above asks for a decision straight away. This one asks what the
# opponents are holding first, and only then what to do about it — the shape
# most serious poker prompting takes, because a decision without a read is
# guesswork dressed up as arithmetic.
#
# It is here as a working reference, not as a strong strategy: the text is
# deliberately plain so that the *structure* is the thing you copy. Swap in
# your own rules through `prompt_catalog.set_catalog` and the arena will play
# them instead. Two calls means twice the cost and twice the wait, which is
# the trade this variant exists to let you measure.

RANGE_SYSTEM = """You read No-Limit Texas Hold'em hands for a living.

Given a spot, say what each opponent still in the hand is likely holding. Work
from what they have actually done: position, whether they raised or only
called, and how the board interacts with the hands they would play that way.

Be concrete and brief. Use standard notation (AKs, QQ+, T9s-76s). At most one
short line per opponent, and no more than four lines in total. Do not give
advice and do not mention what the hero should do — that is the next question,
not this one.
"""

DECISION_SYSTEM = SYSTEM.replace(
    "You are playing No-Limit Texas Hold'em at a friendly table.",
    "You are playing No-Limit Texas Hold'em at a friendly table.\n\n"
    "A read of the opponents' ranges is supplied. Use it: fold hands that are "
    "behind the range you are facing, and bet the ones that are ahead of it.",
)


def render_range(state: GameState, legal: list[str], *, seat_name: str = "you",
                 big_blind: float = 1.0) -> str:
    """The first pass: describe the spot, ask for reads."""
    bb = big_blind or 1.0
    live = [s for s in (state.seats or []) if getattr(s, "in_hand", True)]
    lines = [
        f"Street: {_STREET_NAME.get(state.street, 'preflop')}",
        f"Board: {' '.join(state.board or []) or '—'}",
        f"Pot: {state.pot / bb:.1f} bb",
        f"Players still in the hand: {len(live) or 2}",
    ]
    # Naming the hero's own seat matters: without it a model happily hands the
    # hero a range too, and then reasons about "the opponent" that is you.
    hero_seat = next((s for s in (state.seats or [])
                      if getattr(s, "is_hero", False)), None)
    hero_pos = ""
    if state.action_history and hero_seat is not None:
        hero_pos = next((a.position for a in state.action_history
                         if a.seat_no == hero_seat.seat_no), "")
    others = [s.name for s in live
              if not getattr(s, "is_hero", False) and s.name]
    if state.action_history:
        lines.append("Action so far: " + " · ".join(
            f"{a.position} {a.action}"
            + (f" {a.amount / bb:.1f}bb" if a.amount else "")
            for a in state.action_history))
    lines.append("")
    who = f"You are {seat_name}"
    if hero_pos:
        who += f", in {hero_pos}"
    lines.append(who + ". Do NOT give a range for yourself.")
    if others:
        lines.append("Give one line for each of: " + ", ".join(others) + ".")
    else:
        lines.append("Give one line for each opponent still in the hand.")
    return "\n".join(lines)


def render_decision(state: GameState, legal: list[str], *,
                    seat_name: str = "you", big_blind: float = 1.0,
                    ranges: str = "") -> str:
    """The second pass: the same spot, plus the reads from the first."""
    spot = render(state, legal, seat_name=seat_name, big_blind=big_blind)
    if not ranges:
        return spot
    return f"Reads on the opponents:\n{ranges.strip()}\n\n{spot}"
