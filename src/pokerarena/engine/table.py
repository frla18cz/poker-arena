"""A table you sit at: the button moves, stacks carry over, seats differ.

A benchmark harness measures **one** strategy: the hero rotates through seats,
stacks reset every hand, and every opponent shares one instance and one RNG.
That is enough to measure a win rate and not enough to be a table. This is the
real thing — every seat has its own strategy, its own RNG and its own stack that
survives between hands, and the button moves.

PokerKit still does the rules: dealing, streets, showdown, side pots. What is
here is the orchestration on top.

Running it:

    python -m pokerarena.engine.table --preset cheap --hands 50
    python -m pokerarena.engine.table --preset mixed --hands 20   # COSTS MONEY

Output goes to a directory per game: ``hands.txt`` in hand-history format,
``<seat>.decisions.jsonl`` for each seat that logs, and
``table.json`` se souhrnem.

The results rest on PokerKit's own payoffs. ``hands.txt`` is a reconstruction,
but it balances to the chip — verified across 2 to 6 players. Should the two
ever disagree, a reader can detect it and decline to report a result for that
hand.

To replay one:

    python -m pokerarena.server.host \\
        --hh lab/table/<label>/hands.txt \\
        --decisions lab/table/<label>/<seat>.decisions.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from pokerkit import Automation, NoLimitTexasHoldem

from ..contract.game_state import Action, ActionType, ObservedAction, Street
from .adapter import HandLine, apply_action, build_gamestate, card
from . import seat_registry
from .table_config import CHIP_SCALE, PRESETS, SeatConfig, TableConfig

_AUT = (
    Automation.ANTE_POSTING, Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING, Automation.CARD_BURNING,
    Automation.HOLE_DEALING, Automation.BOARD_DEALING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
)
_STREET = {0: Street.PREFLOP, 1: Street.FLOP, 2: Street.TURN, 3: Street.RIVER}
# PokerKit posts blinds at indexes 0 and 1, which puts the button last.
_POSITIONS = {
    # Heads-up, PokerKit has it the other way: index 0 is the big blind and
    # index 1 is the button and small blind.
    2: ("BB", "SB"),
    3: ("SB", "BB", "BTN"),
    4: ("SB", "BB", "CO", "BTN"),
    5: ("SB", "BB", "UTG", "CO", "BTN"),
    6: ("SB", "BB", "UTG", "HJ", "CO", "BTN"),
}
DEFAULT_OUT = Path("lab/table")
# How many hands before a player is classified at all.
#
# Against a pool of thousands, judging someone from fifty hands is unwise, and a
# hundred is the usual threshold. A home game is the opposite case: it is the
# same four people every time, and a session is around fifty hands — so a
# hundred would still report "no history" about someone in their fourth evening
# who plays 89% of hands. Confidence still scales with sample size, so a thin
# sample is trusted less on its own.
TABLE_MIN_SAMPLE_HANDS = 30


def _payoff(st, index: int) -> float:
    """A player's net result for the hand.

    Equivalent to PokerKit's payoffs, computed as the difference from the
    starting stack: by the end of a hand every payout has happened, so it does
    not matter
    kdy PokerKit pot posunul.

    **Zero does not mean they did not play.** In a split pot everyone gets
    exactly their contribution back, so the result is zero even after a big
    hand — it cannot be used to infer whether someone called or folded.
    """
    return float(st.stacks[index]) - float(st.starting_stacks[index])


class NeedHumanAction(RuntimeError):
    """A human is seated, but nobody supplied a way to ask them."""


@dataclass
class Player:
    config: SeatConfig
    stack: int
    rng: random.Random
    strategy: object | None = None      # Strategy s .decide(gs, rng)
    log: object | None = None           # a decision log, for anything not human
    net: int = 0
    hands: int = 0
    rebuys: int = 0
    cost_usd: float = 0.0
    fallbacks: int = 0

    @property
    def name(self) -> str:
        return self.config.name


class TableOver(RuntimeError):
    """No hand can be dealt: fewer than two seats still have chips."""


@dataclass
class HandRecord:
    hand_no: int
    hand_id: str
    button: str
    order: list[str]                    # names in PokerKit's seat order
    start_stacks: dict[str, int]
    hole_cards: dict[str, list[str]]
    board: list[str]
    actions: list[dict]
    nets: dict[str, int]
    cost_usd: float = 0.0
    text: str = ""                      # the hand history text
    all_in: bool = False


def build_players(config: TableConfig, *, out_dir: Path | None = None,
                  iterations: int = 250, profiler=None) -> list[Player]:
    """Build the players from the config: each its own instance and own RNG.

    A shared strategy instance and a shared RNG are fine when measuring a single
    hero, but at a table they would turn five "independent" bots into one player
    occupying five seats.

    ``seat_registry`` knows who builds which kind. An unregistered kind — and
    ``human`` always — is left without a strategy. For a person that is the whole
    point; anywhere else it means nobody registered a factory.
    """
    players: list[Player] = []
    for index, seat in enumerate(config.seats):
        rng = random.Random(config.seed * 1000 + index)
        player = Player(config=seat, stack=config.start_stack, rng=rng)
        factory = seat_registry.factory_for(seat.kind)
        if factory is not None:
            player.strategy = factory(seat_registry.SeatBuild(
                seat=seat, config=config, out_dir=out_dir,
                iterations=iterations, profiler=profiler))
        # Decisions are logged for **everyone who is not human**. A model seat
        # leaves a trace with its prompts, and a solver seat leaves its route,
        # equity and provenance in the action's meta — without a record there
        # is nothing to review later.
        if out_dir is not None and seat.kind != "human":
            player.log = seat_registry.build_decision_log(seat, out_dir)
        players.append(player)
    return players


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name)


_BOARD_AT = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def showdown_hands(hole: dict[str, list[str]], board: list[str],
                   contenders: list[str]) -> list[dict]:
    """What each player who got there showed, and which of them won.

    **PokerKit** evaluates it — the same engine that split the pot. Evaluating
    it separately could show a different five cards from the ones someone was
    paid for, and that is exactly the kind of contradiction nobody at a table
    will believe.

    ``won`` allows several players, for a split pot. With no board — everyone
    folded preflop — nothing is returned, because there is nothing to show.
    """
    if len(board) < 5 or len(contenders) < 2:
        return []
    from pokerkit import StandardHighHand

    rows: list[dict] = []
    for name in contenders:
        cards = hole.get(name) or []
        if len(cards) != 2:
            continue
        try:
            made = StandardHighHand.from_game("".join(cards), "".join(board))
        except (ValueError, KeyError):
            continue                   # unknown cards: no showdown to show
        label = str(made).split(" (")[0]
        rows.append({"name": name,
                     "best": [repr(c) for c in made.cards],
                     "label": label.lower(),
                     "_hand": made})
    if not rows:
        return []
    best = max(row["_hand"] for row in rows)
    for row in rows:
        row["won"] = row.pop("_hand") == best
    return rows


def replay_steps(*, order: list[str], start_stacks: dict[str, int],
                 actions: list[dict], board: list[str],
                 big_blind: int, small_blind: int | None = None) -> list[dict]:
    """The state **before** each action, so a finished hand can be stepped through.

    Derived from the actions rather than from PokerKit, which discards the state
    once the hand is done. Blinds are not in ``actions`` — nobody decided them —
    so they are seeded separately; without that the preflop pot and the stacks
    would each be short by one and a half blinds.

    The last position of the slider is the finished hand rather than a step,
    which is why there are exactly as many records as there were actions.
    """
    n = len(order)
    # `small_blind=None` means half the big blind. In a 2/5 structure it is 0.4
    # of a blind instead, which cannot be inferred and has to be passed in.
    sb = float(big_blind / 2 if small_blind is None else small_blind)
    blinds = ({order[0]: float(big_blind), order[1]: sb} if n == 2
              else {order[0]: sb, order[1]: float(big_blind)})
    bets = {name: blinds.get(name, 0.0) for name in order}      # vklad na ULICI
    stacks = {name: float(start_stacks[name]) - blinds.get(name, 0.0)
              for name in order}
    pot = float(sum(blinds.values()))
    folded: set[str] = set()
    street = "preflop"
    steps: list[dict] = []
    for act in actions:
        if act["street"] != street:
            street = act["street"]
            bets = {name: 0.0 for name in order}
        steps.append({
            "street": street,
            "board": board[:_BOARD_AT.get(street, 0)],
            "pot": pot,
            "acting": act["player"],
            "action": act,
            "seats": [{"name": name, "stack": stacks[name], "bet": bets[name],
                       "folded": name in folded} for name in order],
        })
        amount = float(act["amount"])
        bets[act["player"]] += amount
        stacks[act["player"]] -= amount
        pot += amount
        if act["action"] == "fold":
            folded.add(act["player"])
    return steps


class Table:
    """One table. ``play_hand()`` plays a hand and moves the button."""

    def __init__(self, config: TableConfig, *, out_dir: Path | None = None,
                 human_action=None, iterations: int = 250, on_state=None,
                 stacks: dict | None = None, button: int | None = None,
                 hand_no: int = 0) -> None:
        """``stacks``, ``button`` and ``hand_no`` continue a game — see ``TableSession``."""
        config.validate()
        self.config = config
        self.out_dir = out_dir
        self.human_action = human_action
        # An optional subscriber to the state as a hand progresses. Nothing
        # changes without one; the web UI needs it to draw the table between
        # your own turns.
        self.on_state = on_state
        if config.humans and human_action is None:
            raise NeedHumanAction(
                "a human is seated, but no `human_action` was supplied")
        # Opponent profiles are built BEFORE the players, so a seat gets them in
        # its constructor and has reads from the first hand rather than the
        # second.
        self.profiles = seat_registry.build_profile_feed(config, out_dir=out_dir)
        self.profiler = getattr(self.profiles, "profiler", None)
        self.players = build_players(config, out_dir=out_dir,
                                     iterations=iterations,
                                     profiler=self.profiler)
        # Who the hand history calls the hero: the first model seat, or the
        # human. Positions and results are computed relative to them.
        self.hero_name = next(
            (p.name for p in self.players if p.config.kind in ("llm", "human")),
            self.players[0].name)
        self.button = len(self.players) - 1     # first hand: player 0 is the SB
        # Continuing an earlier game: stacks by name, since the seat order may
        # have changed, and the button where it was left.
        for player in self.players:
            if stacks and player.name in stacks:
                player.stack = int(stacks[player.name])
        if button is not None:
            self.button = int(button) % len(self.players)
        # How many hands have been played. It has to be restored when resuming:
        # `hand_id` is built from it, so starting at zero would collide with the
        # earlier hands and overwrite them in the history.
        self.hand_no = int(hand_no)
        self.hands: list[HandRecord] = []

    # -- seating order -------------------------------------------------------

    def seated(self) -> list[Player]:
        """Players who still have chips. Without rebuys, busts leave for good."""
        return [p for p in self.players if p.stack > 0]

    def _seating(self) -> list[Player]:
        """The players in the order PokerKit expects; the button lands last.

        Heads-up, PokerKit handles it itself and **the opposite way round from
        what you would expect**: with blinds of (1, 2), index 0 puts in two chips
        and index 1 puts in one, and index 1 acts first. In other words the big
        blind is at index 0 and the small blind — the button — at index 1. The
        rule "start after the button" satisfies that just as it does a full
        table, so there must be no special case for two players. While there was
        one, heads-up had the players swapped throughout the hand history.
        """
        live = self.seated()
        n = len(live)
        start = (self.button + 1) % n
        return [live[(start + k) % n] for k in range(n)]

    def _positions(self, n: int) -> tuple[str, ...]:
        return _POSITIONS.get(n, _POSITIONS[6][:n])

    @staticmethod
    def _blinds(n: int, big_blind: int, small_blind: int) -> dict[int, float]:
        """Who posted what, in PokerKit seat indices."""
        if n == 2:
            return {0: float(big_blind), 1: float(small_blind)}
        return {0: float(small_blind), 1: float(big_blind)}

    # -- single hand -------------------------------------------------------

    def play_hand(self) -> HandRecord:
        config = self.config
        # With rebuys off a seat can bust for good, and PokerKit refuses a
        # non-positive starting stack — it raised "Non-positive starting stacks
        # was supplied" from deep inside the deal, which says nothing about the
        # game being over. Two players with chips is the real precondition.
        if len(self.seated()) < 2:
            raise TableOver(
                "the game is over: fewer than two seats still have chips")
        order = self._seating()
        n = len(order)
        self.hand_no += 1
        # It has to be numeric: the hand-history header is parsed as
        # `Hand #(\d+)`, and without a match a decision log cannot be joined to it.
        hand_id = f"{config.seed}{self.hand_no:05d}"

        # PokerKit deals from the global `random`, so the only way to make a
        # game reproducible is to seed it ourselves before each hand.
        #
        # Derived from **(seed, hand number)**, not from a running sequence. A
        # sequence restarts when a game is resumed, so the fourth hand after
        # coming back dealt exactly the cards the fourth hand had before. This
        # way a hand is determined by its number and resuming does not touch it.
        random.seed((config.seed * 1_000_003 + self.hand_no) % (2 ** 32))
        st = NoLimitTexasHoldem.create_state(
            _AUT, True, 0, (config.sb, config.big_blind),
            config.big_blind, tuple(p.stack for p in order), n)

        positions = self._positions(n)
        start_stacks = {p.name: p.stack for p in order}
        hole = {p.name: [card(c) for c in st.hole_cards[i]]
                for i, p in enumerate(order)}
        line = HandLine()
        history: list[ObservedAction] = []
        actions: list[dict] = []
        pf_aggressor: int | None = None
        cost = 0.0
        guard = 0
        # Tracking street contributions here, because `st.bets` cannot be used:
        # once a betting round closes, PokerKit collects it into the pot and
        # zeroes it, which makes the last action of a street come out negative.
        current_street = 0
        street_totals: dict[int, float] = self._blinds(n, config.big_blind,
                                                       config.sb)
        # Total contributed over the hand, blinds included. PokerKit's payoffs
        # count them too, so leaving them out would make the last action wrong by
        # a blind.
        contributed: dict[int, float] = dict(street_totals)

        while st.actor_index is not None and guard < 400:
            guard += 1
            index = st.actor_index
            player = order[index]
            street_index = st.street_index or 0
            if street_index != current_street:
                current_street, street_totals = street_index, {}
            street = _STREET.get(street_index, Street.RIVER)
            to_call_before = st.checking_or_calling_amount or 0
            before = street_totals.get(index, 0.0)
            stack_before = float(st.stacks[index])
            could_fold = st.can_fold()
            # Raise bounds have to be read BEFORE the action; afterwards they
            # describe the next player.
            raise_lo = st.min_completion_betting_or_raising_to_amount
            raise_hi = st.max_completion_betting_or_raising_to_amount
            # Published BEFORE deciding, so the table shows who is being waited
            # on — a model can take several seconds.
            self._publish(st, order, positions, hand_id, hole, street_totals,
                          index, actions)

            action, meta_cost, _applied = self._decide(
                st, index, order, pf_aggressor, line, history, hand_id)
            cost += meta_cost
            aggressive = apply_action(st, action)

            # The contribution comes from the ACTION PLAYED, not from chip
            # movement or payoffs. The last action of a hand is accompanied by
            # the pot being paid out, and in a split pot both players get exactly
            # their contribution back — so stack difference and payoff both come
            # to zero, and a call would be recorded as a fold. (Seen on a real
            # hand: p0 called 47 and was written down as "fold".)
            if aggressive and raise_lo is not None:
                target = min(raise_hi, max(raise_lo, round(action.amount)))
                increment = max(0.0, float(target) - before)
                verb = "raise" if (to_call_before > 0 or before > 0) else "bet"
            elif action.type == ActionType.FOLD and could_fold:
                increment, verb = 0.0, "fold"
            else:
                increment = min(float(to_call_before), stack_before)
                verb = "call" if to_call_before > 0 else "check"
            contributed[index] = contributed.get(index, 0.0) + increment
            after = before + increment
            street_totals[index] = after
            folded = verb == "fold"
            history.append(ObservedAction(
                street=street, seat_no=index, position=positions[index],
                action=verb, amount=increment, committed_after=after))
            actions.append({"street": street.value, "player": player.name,
                            "position": positions[index], "action": verb,
                            "amount": increment, "total": after,
                            # Why the seat played it, when the seat says. A
                            # strategy that folds because its key expired has
                            # to be able to say so at the table.
                            "reason": (action.reason or "")[:120],
                            "failed": bool((action.meta or {}).get("failed")),
                            # Which decision of theirs this is within the hand.
                            # The decision log is append-only in the same order,
                            # so this indexes straight into it.
                            "decision": sum(1 for a in actions
                                            if a["player"] == player.name)})
            line.note(street_index, index, aggressive,
                      called=(not aggressive and not folded and to_call_before > 0),
                      checked=(not aggressive and not folded and to_call_before == 0))
            if street_index == 0 and aggressive:
                pf_aggressor = index

        board = [card(c) for row in st.board_cards for c in row]
        nets = {}
        for index, player in enumerate(order):
            net = int(_payoff(st, index))
            player.stack = int(st.stacks[index])
            player.net += net
            player.hands += 1
            nets[player.name] = net

        record = HandRecord(
            hand_no=self.hand_no, hand_id=hand_id,
            button=order[-1].name,          # the button always lands last
            order=[p.name for p in order], start_stacks=start_stacks,
            hole_cards=hole, board=board, actions=actions, nets=nets,
            cost_usd=round(cost, 6),
            all_in=any(contributed.get(i, 0.0) >= start_stacks[p.name]
                       for i, p in enumerate(order)))
        record.text = format_hand(record, config, positions, hero=self.hero_name)
        self.hands.append(record)
        # Statistics update on a finished hand, never a running one: VPIP, cbet
        # and WTSD are all defined over a whole hand. This must not be allowed to
        # fail — reads are a bonus, not a condition of playing.
        if self.profiles is not None:
            try:
                self.profiles.ingest_hand_text(record.text)
            except Exception:                  # noqa: BLE001
                pass
        # The closing snapshot, with the result and the showdown revealed — this
        # is the first point where it may be shown. ``runout`` carries the
        # streets that arrived WITH NO decisions, after an all-in, so the table
        # can deal them one at a time instead of appearing with a full board.
        self._publish(st, order, positions, hand_id, hole, street_totals,
                      None, actions, finished=True, nets=nets,
                      start_stacks=start_stacks,
                      runout=self._runout_streets(order, actions, board))

        self._settle()
        # A frozen button means fixed positions. The whole seating comes from
        # `self.button` via `_seating`, so this is the only place a position
        # changes — players never move between seats themselves.
        if config.rotate_button:
            self.button = (self.button + 1) % n
        return record

    @staticmethod
    def _runout_streets(order: list[Player], actions: list[dict],
                        board: list[str]) -> list[dict]:
        """The streets dealt with no decisions at all, after an all-in.

        They are recognisable because the board is longer than the last action
        reached: while there is still betting, someone opens each new street with
        a check or a bet. Once everyone is all-in, PokerKit deals the rest at
        once — and the table would show the hand with a complete board, leaving
        nothing to watch.

        A heads-up fold is not this: nothing is dealt, because there is nobody
        to deal to.
        """
        folded = {a["player"] for a in actions if a["action"] == "fold"}
        if len([p for p in order if p.name not in folded]) < 2:
            return []
        seen = _BOARD_AT.get(actions[-1]["street"] if actions else "preflop", 0)
        names = {3: "flop", 4: "turn", 5: "river"}
        return [{"street": names[count], "board": board[:count]}
                for count in (3, 4, 5) if seen < count <= len(board)]

    def _snapshot(self, st, order, positions, hand_id, hole, street_totals,
                  acting: int | None, actions: list[dict], *,
                  finished: bool = False, nets: dict | None = None,
                  start_stacks: dict | None = None,
                  runout: list[dict] | None = None) -> dict:
        """The state of the hand in progress, for drawing the table.

        The cards of **every** player are here — this is the server's truth. Who
        gets to see what is decided by ``TableSession.state``: a player sees
        their own, plus any marked ``revealed``.

        ``revealed`` is set at the end of a hand and **only for those who reached
        showdown**. Showing the cards of someone who folded would be more than a
        real table gives; showing them earlier would give away a live hand.
        """
        street = _STREET.get(st.street_index or 0, Street.RIVER)
        folded = {a["player"] for a in actions if a["action"] == "fold"}
        if finished and actions:
            # PokerKit zeroes `street_index` once the hand is over, so a hand
            # played to the river reported itself as "preflop". The last action
            # knows better.
            street = Street(actions[-1]["street"])
        showdown = finished and sum(
            1 for p in order if p.name not in folded) > 1
        board = [card(c) for row in st.board_cards for c in row]
        # Stepping through only makes sense for a finished hand; on a live one
        # it would reveal how many streets are still to come.
        steps = replay_steps(
            order=[p.name for p in order], start_stacks=start_stacks,
            actions=actions, board=board, big_blind=self.config.big_blind,
            small_blind=self.config.sb,
        ) if finished and start_stacks else []
        # After the hand `total_pot_amount` is zero, because the pot has been
        # paid out. The result should still show what was played for, so it is
        # recomputed from the actions.
        pot = float(st.total_pot_amount)
        if steps:
            pot = steps[-1]["pot"] + float(actions[-1]["amount"])
        # The hand shown. Computed only for a finished hand with a showdown; on
        # a live one it would give away what people hold.
        shown = showdown_hands(
            hole, board, [p.name for p in order if p.name not in folded]
        ) if showdown else []
        return {
            "hand_id": hand_id,
            "street": street.value,
            "board": board,
            "steps": steps,
            "runout": list(runout or []),
            "showdown_hands": shown,
            # Who took the pot. `nets` is not enough: in a split pot everyone
            # gets their contribution back and the result is zero even after a
            # big hand.
            "winners": (([row["name"] for row in shown if row["won"]]
                         or [p.name for p in order if p.name not in folded])
                        if finished else []),
            "start_stacks": dict(start_stacks or {}),
            "pot": pot,
            "button": order[-1].name,
            "acting": order[acting].name if acting is not None else None,
            "finished": finished,
            "nets": dict(nets or {}),
            "actions": list(actions),
            "seats": [
                {"name": p.name, "position": positions[i],
                 "avatar": getattr(p.config, "avatar", ""),
                 "stack": float(st.stacks[i]),
                 # Once the hand is over, the chips in front of players have
                 # been collected into the pot and reflected in the stacks;
                 # leaving them out there would show them twice.
                 "bet": 0.0 if finished else float(street_totals.get(i, 0.0)),
                 "folded": p.name in folded,
                 "revealed": showdown and p.name not in folded,
                 "cards": hole.get(p.name, [])}
                for i, p in enumerate(order)],
        }

    def _publish(self, *args, **kwargs) -> None:
        if self.on_state is None:
            return
        try:
            self.on_state(self._snapshot(*args, **kwargs))
        except Exception:                      # noqa: BLE001 — must not break the game
            pass

    def _decide(self, st, index: int, order: list[Player],
                pf_aggressor: int | None, line: HandLine,
                history: list[ObservedAction], hand_id: str,
                ) -> tuple[Action, float, bool]:
        player = order[index]
        # A slow seat may ask for its own ceiling; otherwise the table's applies.
        budget = player.config.timeout_s or self.config.timeout_s
        gs = build_gamestate(
            st, index, pf_aggressor, line, button_seat=len(order) - 1,
            # Strategies work in big blinds; without the real structure, a 2/5
            # table would read every amount at five times its value.
            big_blind=self.config.big_blind, small_blind=self.config.sb,
            timeout_s=budget, history=history,
            hand_id=hand_id, table_id="table",
            names=[p.name for p in order])
        if player.config.kind == "human":
            # The bounds come from PokerKit, not from `GameState`. That carries
            # an increment and a remaining stack, while an action is read as a
            # total for the street — and crucially it cannot show that betting is
            # closed (a short all-in does not reopen it), so a raise would be
            # offered that the engine quietly performs as a call.
            action = self.human_action(gs, player, {
                "raise_to_min": st.min_completion_betting_or_raising_to_amount,
                "raise_to_max": st.max_completion_betting_or_raising_to_amount,
                "can_fold": st.can_fold(),
            })
        else:
            action = player.strategy.decide(gs, player.rng)
        cost = float((action.meta or {}).get("cost_usd") or 0.0)
        player.cost_usd += cost
        if (action.meta or {}).get("fallback"):
            player.fallbacks += 1
        if player.log is not None:
            player.log.log(gs, action)
        return action, cost, False

    def _settle(self) -> None:
        """Anyone below the threshold tops back up to a full stack.

        ``net`` deliberately does not change: the loss is already in the result
        of the hand where the chips went. A top-up is a refill, not another loss,
        and subtracting it again would count the same loss twice. The ceiling is
        always ``start_stack``; nobody sits deeper than that.
        """
        if not self.config.rebuy:
            return
        threshold = self.config.rebuy_below_bb * self.config.big_blind
        for player in self.players:
            if player.stack < threshold:
                player.rebuys += 1
                player.stack = self.config.start_stack

    # -- output --------------------------------------------------------------

    def standings(self) -> list[dict]:
        bb = self.config.big_blind
        rows = []
        for player in self.players:
            rows.append({
                "name": player.name, "kind": player.config.kind,
                "model": (player.config.decision_model
                          if player.config.kind == "llm" else player.config.kind),
                "hands": player.hands, "stack": player.stack,
                "net_bb": round(player.net / bb, 1),
                "bb100": (round(100 * (player.net / bb) / player.hands, 1)
                          if player.hands else None),
                "cost_usd": round(player.cost_usd, 5),
                "rebuys": player.rebuys, "fallbacks": player.fallbacks,
            })
        return sorted(rows, key=lambda r: -(r["net_bb"]))


def format_hand(record: HandRecord, config: TableConfig,
                positions: tuple[str, ...], hero: str | None = None) -> str:
    """One hand in hand-history format, so any reader can open it.

    Cards are shown for **everyone** who reached the end: here they are known,
    and when studying how seats played, that is the most valuable thing the
    record can hold.
    """
    bb = config.big_blind
    sb = config.sb
    # The format wants a currency, but the table plays in chips. The rate is
    # therefore **one displayed unit = €1.00**, so the record carries exactly the
    # numbers visible at the table. Using an NL2 scale instead meant the same
    # raise read as "3 bb" at the table and "€0.06" in the history.
    unit = 1.0 / CHIP_SCALE
    money = lambda chips: f"€{chips * unit:.2f}"        # noqa: E731
    lines = [
        f"PokerStars Hand #{record.hand_id}:  Hold'em No Limit "
        f"({money(sb)}/{money(bb)} EUR) - "
        f"{time.strftime('%Y/%m/%d %H:%M:%S')} CEST",
        # The button is always the last index, heads-up included.
        f"Table 'sim' {len(record.order)}-max Seat "
        f"#{len(record.order)} is the button",
    ]
    for index, name in enumerate(record.order, start=1):
        lines.append(f"Seat {index}: {name} ({money(record.start_stacks[name])} in chips)")
    # Heads-up, index 1 posts the small blind — the button; otherwise index 0
    # does. Who is who comes from the ORDER, not the amount: where the small
    # blind is not half the big one, matching on value could pick the wrong seat.
    small, big = ((record.order[1], record.order[0]) if len(record.order) == 2
                  else (record.order[0], record.order[1]))
    blinds = {small: sb, big: bb}
    lines.append(f"{small}: posts small blind {money(sb)}")
    lines.append(f"{big}: posts big blind {money(bb)}")
    lines.append("*** HOLE CARDS ***")
    # "Dealt to" has to name the seat being followed, not whoever is on the
    # small blind, or every hand would be reported as played from the SB.
    if hero not in record.hole_cards:
        hero = next((n for n in record.order if n in record.hole_cards), None)
    if hero:
        lines.append(f"Dealt to {hero} [{' '.join(record.hole_cards[hero])}]")

    street_cards = {"flop": record.board[:3], "turn": record.board[:4],
                    "river": record.board[:5]}
    seen = "preflop"
    committed: dict[str, float] = {}
    for action in record.actions:
        if action["street"] != seen:
            seen = action["street"]
            committed = {}
            cards = street_cards.get(seen, [])
            if seen == "flop":
                lines.append(f"*** FLOP *** [{' '.join(cards)}]")
            elif seen == "turn":
                lines.append(f"*** TURN *** [{' '.join(cards[:3])}] [{cards[3]}]")
            elif seen == "river":
                lines.append(f"*** RIVER *** [{' '.join(cards[:4])}] [{cards[4]}]")
        name, verb = action["player"], action["action"]
        if verb == "fold":
            lines.append(f"{name}: folds")
        elif verb == "check":
            lines.append(f"{name}: checks")
        elif verb == "call":
            lines.append(f"{name}: calls {money(action['amount'])}")
        elif verb == "bet":
            lines.append(f"{name}: bets {money(action['amount'])}")
        else:
            lines.append(f"{name}: raises {money(action['amount'])} "
                         f"to {money(action['total'])}")
        committed[name] = action["total"]

    # What each player really put in. PokerKit posts the blinds itself, so they
    # are not among the actions and have to be added separately.
    contributions = {}
    for name in record.order:
        contributions[name] = blinds.get(name, 0) + sum(
            a["amount"] for a in record.actions if a["player"] == name)

    # An uncalled bet comes back. It shows on the last street with action: the
    # largest contribution exceeds the second largest and nobody covered the
    # difference. Without this, the record would claim more went into the pot
    # than came out of it.
    if record.actions:
        last_street = record.actions[-1]["street"]
        totals: dict[str, float] = {}
        for action in record.actions:
            if action["street"] == last_street:
                totals[action["player"]] = action["total"]
        if last_street == "preflop":
            for name, posted in blinds.items():
                totals.setdefault(name, posted)
        ranked = sorted(totals.items(), key=lambda kv: -kv[1])
        if len(ranked) > 1 and ranked[0][1] > ranked[1][1]:
            back = ranked[0][1] - ranked[1][1]
            contributions[ranked[0][0]] -= back
            lines.append(f"Uncalled bet ({money(back)}) returned to {ranked[0][0]}")

    survivors = [n for n in record.order
                 if not any(a["player"] == n and a["action"] == "fold"
                            for a in record.actions)]
    if len(survivors) > 1:
        lines.append("*** SHOW DOWN ***")
        for name in survivors:
            lines.append(f"{name}: shows [{' '.join(record.hole_cards[name])}]")
    for name in record.order:
        if name not in survivors:
            # Someone who folded collects nothing. With side pots the
            # reconstruction of contributions can be off by a few chips, and
            # without this guard that produced a "collected" line for a player
            # who had long since left the hand.
            continue
        collected = record.nets[name] + contributions[name]
        if collected > 0.0001:
            lines.append(f"{name} collected {money(collected)} from pot")
    lines.append("*** SUMMARY ***")
    lines.append(f"Total pot {money(sum(contributions.values()))} | Rake €0.00")
    if record.board:
        lines.append(f"Board [{' '.join(record.board)}]")
    return "\n".join(lines) + "\n"


def run_table(config: TableConfig, hands: int, *, out_dir: Path,
              progress: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-seat logs are opened in append mode while `hands.txt` is rewritten.
    # Without cleaning up, a second game with the same label would leave hands
    # and decisions misaligned — hand ids come from the seed and the order, so
    # they repeat.
    for stale in out_dir.glob("*.jsonl"):
        stale.unlink()
    table = Table(config, out_dir=out_dir)
    hh_path = out_dir / "hands.txt"
    with hh_path.open("w", encoding="utf-8") as handle:
        for index in range(hands):
            record = table.play_hand()
            handle.write(record.text + "\n")
            handle.flush()
            if progress and (index + 1) % 10 == 0:
                spent = sum(p.cost_usd for p in table.players)
                print(f"  … {index + 1}/{hands} hands, ${spent:.4f} spent")
    report = {
        "config": config.as_dict(), "hands": hands,
        "standings": table.standings(),
        "cost_usd": round(sum(p.cost_usd for p in table.players), 5),
    }
    (out_dir / "table.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="cheap", choices=sorted(PRESETS),
                    help="the table line-up (some presets COST MONEY)")
    ap.add_argument("--config", type=Path, help="a JSON table configuration")
    ap.add_argument("--hands", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=250,
                    help="Monte Carlo iterations for equity")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.config:
        config = TableConfig.from_dict(json.loads(args.config.read_text("utf-8")))
    else:
        config = PRESETS[args.preset]()
    config = TableConfig(seats=config.seats, big_blind=config.big_blind,
                         start_stack=config.start_stack, seed=args.seed,
                         timeout_s=config.timeout_s, rebuy=config.rebuy)
    if config.humans:
        raise SystemExit("a table with a human is played from the UI, not the CLI")

    estimate = config.estimated_cost_per_hand() * args.hands
    label = args.out or (DEFAULT_OUT / f"{args.preset}_{args.hands}h_seed{args.seed}")
    print(f"▶ {args.preset}: {args.hands} hands, {len(config.seats)} seats")
    for seat in config.seats:
        detail = (f"{seat.range_model} → {seat.decision_model} · {seat.prompt_variant}"
                  + (f" · kritik {seat.critic_model}" if seat.critic_model else "")
                  if seat.kind == "llm" else seat.kind)
        print(f"   {seat.name:12} {seat.kind:6} {detail}")
    if estimate > 0:
        print(f"   estimated cost: ${estimate:.2f}")

    report = run_table(config, args.hands, out_dir=label)
    print(f"\n{'name':12} {'kind':6} {'hands':>6} {'bb/100':>8} {'cost':>9} {'fallback':>9}")
    for row in report["standings"]:
        print(f"{row['name']:12} {row['kind']:6} {row['hands']:6} "
              f"{row['bb100'] if row['bb100'] is not None else '—':>8} "
              f"${row['cost_usd']:.5f} {row['fallbacks']:>9}")
    print(f"\ncelkem utraceno ${report['cost_usd']:.4f} → {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
