"""A table running in the background, with room for a human to sit at it.

``Table.play_hand()`` is synchronous and stops at a human seat until it is given
an action. A web request cannot sit and wait like that, so the game runs on its
own thread and exchanges two messages with the browser:

* ``state()`` — what is visible **through one player's eyes**; other people's
  cards are not sent,
* ``act()`` — the action the game is waiting for.

The first of those is the whole privacy model for playing with other people: the
server simply never sends someone else's cards, so there is nothing in the
browser to reveal.
"""
from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path

from ..contract.game_state import Action, ActionType, GameState
from .table import Table, TableOver
from .table_config import CHIP_SCALE, TableConfig

# How long the browser holds each street of a runout (`RUNOUT_STAGE_MS` in the
# page). The server extends the pause between hands to match — if the two
# numbers disagree, the table deals over the top of the animation.
RUNOUT_STAGE_S = 1.6


class TableSession:
    """One running table: a thread plays, HTTP only asks and answers."""

    STATE_FILE = "state.json"
    SNAPSHOTS_FILE = "hands.jsonl"

    @classmethod
    def load(cls, out_dir: Path, **kwargs) -> "TableSession":
        """Pick up an earlier game: same line-up, same stacks, same button.

        Named ``load`` rather than ``resume`` because the instance method
        ``resume()`` already means "deal the next hand", and sharing the name
        would quietly override this.

        A session otherwise lives only in memory, so closing the server or
        starting a new game would lose a game in progress for good. The state is
        therefore written to ``state.json`` after every hand.

        The **seat tokens** come back too. Regenerating them would mean handing
        everyone a new link purely because the game was continued.
        """
        import json

        state = json.loads((Path(out_dir) / cls.STATE_FILE).read_text("utf-8"))
        played = int(state.get("hands_played", 0))
        session = cls(TableConfig.from_dict(state["config"]), out_dir=Path(out_dir),
                      stacks=state.get("stacks"), button=state.get("button"),
                      hand_no=played, **kwargs)
        session.hands_played = played
        for name, token in (state.get("tokens") or {}).items():
            if name in session.tokens:
                session.tokens[name] = token
        session._load_snapshots()
        return session

    def __init__(self, config: TableConfig, *, out_dir: Path | None = None,
                 hand_pause_s: float = 4.0,
                 stacks: dict | None = None, button: int | None = None,
                 hand_no: int = 0) -> None:
        config.validate()
        self.config = config
        self.out_dir = out_dir
        # A pause after each hand, so there is time to read who had what.
        # `paused` holds the table until it is let go.
        self.hand_pause_s = hand_pause_s
        # Extra pause because the last hand had a board to run out. The browser
        # shows flop, turn and river one at a time and holds the winning hand;
        # dealing over that would mean the all-in was never actually seen.
        self._extra_pause = 0.0
        self.paused = False
        self._resume = threading.Event()
        self._live: dict | None = None
        self._finished: dict[str, dict] = {}
        self.table = Table(config, out_dir=out_dir, human_action=self._ask_human,
                           on_state=self._on_state, stacks=stacks, button=button,
                           hand_no=hand_no)
        self._lock = threading.RLock()
        self._answered = threading.Event()
        self._pending: dict | None = None      # what is being waited on
        self._answer: Action | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.hands_played = 0
        self.hands_target = 0
        self.events: list[str] = []
        self.error: str | None = None
        self.label = out_dir.name if out_dir is not None else ""
        # The seat's secret. Without it there are no cards and no legal moves —
        # the whole privacy model when playing over a network.
        self.tokens: dict[str, str] = {
            seat.name: secrets.token_urlsafe(9)
            for seat in config.seats if seat.kind == "human"}

    def seat_for(self, token: str | None) -> str | None:
        """Whose seat a token is. With no token: the only human, if there is one.

        That exception is for playing on one computer. With a single human at the
        table there is nobody to overhear them, so there is no reason to make
        them use a link.
        """
        if token:
            return next((name for name, value in self.tokens.items()
                         if secrets.compare_digest(value, token)), None)
        return next(iter(self.tokens)) if len(self.tokens) == 1 else None

    # -- running -----------------------------------------------------------

    def start(self, hands: int) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("the table is already running")
        self._stop.clear()
        self.hands_target = hands
        self.error = None
        self._thread = threading.Thread(target=self._run, args=(hands,),
                                        daemon=True, name="table")
        self._thread.start()

    def _run(self, hands: int) -> None:
        handle = None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            handle = (self.out_dir / "hands.txt").open("a", encoding="utf-8")
        try:
            for _ in range(hands):
                if self._stop.is_set():
                    break
                try:
                    record = self.table.play_hand()
                except TableOver as over:
                    # Not a failure: without rebuys someone eventually wins
                    # everything, and the game simply ends there.
                    self._note(str(over))
                    break
                self.hands_played += 1
                if handle is not None:
                    handle.write(record.text + "\n")
                    handle.flush()
                # Reported in big blinds, not chips, to match the table.
                bb = self.config.big_blind
                self._note(f"hand #{record.hand_id}: " + ", ".join(
                    f"{name} {value / bb:+g} bb"
                    for name, value in sorted(record.nets.items(),
                                              key=lambda kv: -kv[1]) if value))
                self._save_state()
                self._wait_between_hands()
        except Exception as exc:                 # noqa: BLE001 — surface it
            self.error = f"{type(exc).__name__}: {exc}"
            self._note(f"the table crashed: {self.error}")
        finally:
            if handle is not None:
                handle.close()
            with self._lock:
                self._pending = None
            self._answered.set()

    def _save_state(self) -> None:
        """Write the state after a hand, so the game survives the server closing."""
        if self.out_dir is None:
            return
        import json
        try:
            (self.out_dir / self.STATE_FILE).write_text(json.dumps({
                "label": self.label,
                "config": self.config.as_dict(),
                "stacks": {p.name: p.stack for p in self.table.players},
                "button": self.table.button,
                "hands_played": self.hands_played,
                "tokens": self.tokens,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass                      # saving state must never break the game

    def _wait_between_hands(self) -> None:
        """The gap after a hand: a fixed delay, or held until released."""
        self._resume.clear()
        deadline = time.monotonic() + self.hand_pause_s + self._extra_pause
        self._extra_pause = 0.0
        while not self._stop.is_set():
            if self.paused:
                if self._resume.wait(timeout=0.2):
                    return
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self._resume.wait(timeout=min(remaining, 0.2)):
                return

    def resume(self) -> None:
        """Deal the next hand now, even if the table is paused."""
        self.paused = False
        self._resume.set()

    def stop(self, wait: float = 5.0) -> None:
        """Stop the table and **wait for the thread to actually finish**.

        Without waiting, the old thread keeps writing into a directory the new
        table has already cleaned out, and two games end up in one log.
        """
        self._stop.set()
        self._resume.set()
        # If the game is waiting on a person, wake it or the thread hangs.
        with self._lock:
            if self._pending is not None:
                self._answer = Action(ActionType.FOLD, 0.0, "table stopped")
        self._answered.set()
        thread = self._thread
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=wait)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _on_state(self, snapshot: dict) -> None:
        with self._lock:
            self._live = snapshot
            if snapshot.get("finished"):
                self._extra_pause = RUNOUT_STAGE_S * len(snapshot.get("runout") or [])
                # Finished hands are kept so they can be reviewed while playing.
                # A few kilobytes each, and only the last 200 are held.
                self._finished[snapshot["hand_id"]] = snapshot
                for stale in list(self._finished)[:-200]:
                    del self._finished[stale]
                self._append_snapshot(snapshot)

    def _append_snapshot(self, snapshot: dict) -> None:
        """Write a finished hand, so it can still be reviewed after resuming.

        ``hands.txt`` is the hand-history format, but stepping through a hand
        needs cards, stacks and the individual steps, none of which fit in it.
        Without this file, the history is empty after ``load()`` and someone
        returning to a game has nothing to click.
        """
        if self.out_dir is None:
            return
        import json
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with (self.out_dir / self.SNAPSHOTS_FILE).open(
                    "a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        except OSError:
            pass                      # saving history must never break the game

    def _load_snapshots(self) -> None:
        """Load finished hands from an earlier run; a corrupt line is skipped."""
        if self.out_dir is None:
            return
        import json
        path = self.out_dir / self.SNAPSHOTS_FILE
        if not path.exists():
            return
        rows: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue          # truncated by a server crash
        with self._lock:
            for snapshot in rows[-200:]:
                if snapshot.get("hand_id"):
                    self._finished[str(snapshot["hand_id"])] = snapshot

    def history(self) -> list[dict]:
        """Hands played, newest first — what the review list is built from."""
        with self._lock:
            rows = list(self._finished.values())
        return [{"hand_id": h["hand_id"], "board": h["board"], "nets": h["nets"]}
                for h in reversed(rows)]

    def finished_hand(self, hand_id: str, token: str | None = None) -> dict:
        """One finished hand, through the eyes of whoever holds the token.

        It reveals **what the end of the hand revealed**: the showdown, and your
        own cards. Showing what an opponent mucked, after the fact, would be more
        than anyone at the table is entitled to.
        """
        with self._lock:
            snapshot = self._finished.get(str(hand_id))
        if snapshot is None:
            raise LookupError(f"no hand {hand_id} in this game")
        viewer = self.seat_for(token)
        return {**snapshot, "seats": [
            {**seat, "cards": (seat["cards"]
                               if seat["name"] == viewer or seat.get("revealed")
                               else [])}
            for seat in snapshot["seats"]]}

    def _live_for(self, viewer: str | None) -> dict | None:
        """The hand in progress, with other people's cards removed."""
        with self._lock:
            live = self._live
        if live is None:
            return None
        return {**live, "seats": [
            {**seat, "cards": (seat["cards"]
                               if seat["name"] == viewer or seat.get("revealed")
                               else [])}
            for seat in live["seats"]]}

    def _note(self, text: str) -> None:
        with self._lock:
            self.events.append(text)
            del self.events[:-40]

    # -- the human seat ------------------------------------------------------

    def _ask_human(self, state: GameState, player,
                   bounds: dict | None = None) -> Action:
        """Called on the game thread; blocks until an action arrives from the web.

        ``bounds`` carries the limits straight from the engine — without them the
        offer would only be
        odhadovala z ``GameState``.
        """
        with self._lock:
            # The cards and the shape of the spot have to be in `pending`:
            # `state()` only knows the last **finished** hand, so the player
            # would be deciding blind.
            self._pending = {
                "seat": player.name, "options": legal_options(state, bounds),
                "hero_cards": list(state.hero_cards), "board": list(state.board),
                "street": state.street.value, "pot": round(state.pot, 2),
                "to_call": round(state.to_call, 2),
                "stack": round(float(state.hero.stack) if state.hero else 0.0, 2),
                "asked_at": time.time()}
            self._answer = None
            self._answered.clear()
        # The wait has **no timeout**: nobody's action gets played for them. The
        # thread is a daemon and `stop()` sets the event, so it cannot get stuck —
        # the only ways out are to act or to stop the table.
        self._answered.wait()
        with self._lock:
            action, self._pending = self._answer, None
        return action or _default_action(state)

    def act(self, kind: str, amount: float = 0.0, token: str | None = None) -> None:
        viewer = self.seat_for(token)
        with self._lock:
            if self._pending is None:
                raise LookupError("the table is not waiting for an action")
            if self._pending["seat"] != viewer:
                # Without this check, anyone on the network could play someone
                # else's seat.
                raise PermissionError(
                    f"na tahu je {self._pending['seat']}, ne ty")
            options = self._pending["options"]
            if kind not in {o["action"] for o in options}:
                raise ValueError(f"{kind!r} is not legal here")
            chosen = next(o for o in options if o["action"] == kind)
            if chosen.get("min") is not None:
                amount = max(chosen["min"], min(chosen["max"], float(amount)))
            self._answer = Action(_ACTIONS[kind], float(amount), "human")
        self._answered.set()

    # -- the browser's view --------------------------------------------------

    def state(self, token: str | None = None) -> dict:
        """The table **through the eyes of whoever holds the token**.

        Cards and legal moves go only to the person they belong to. Everyone
        else learns at most that someone is being waited on — otherwise editing
        a name in the URL would be enough to see an opponent's hand.
        """
        viewer = self.seat_for(token)
        with self._lock:
            pending = dict(self._pending) if self._pending else None
        if pending is not None and pending["seat"] != viewer:
            pending = {"seat": pending["seat"], "mine": False}
        elif pending is not None:
            pending["mine"] = True
        record = self.table.hands[-1] if self.table.hands else None
        # How much history any reads are based on. Without this number there is
        # no telling whether a bot is reading the group's history or playing
        # blind — a difference you can feel in its play but could not see.
        feed = getattr(self.table, "profiles", None)
        return {
            "running": self.running,
            "hands_played": self.hands_played,
            "hands_target": self.hands_target,
            "error": self.error,
            "viewer": viewer,
            "paused": self.paused,
            "label": self.label,
            # The UI shows big blinds; PokerKit needs whole numbers, so
            # everything is counted in chips and converted here.
            "big_blind": self.config.big_blind,
            # The small blind is sent explicitly: in a 2/5 structure it is not
            # half the big blind, so the UI must not infer it.
            "small_blind": self.config.sb,
            "blinds_label": self.config.blinds_label,
            # How many chips one displayed unit is. The UI converts EVERY amount
            # by it — dividing by the big blind instead showed 0.4 where a 2/5
            # table meant 2.
            "chip_scale": CHIP_SCALE,
            "start_stack": self.config.start_stack,
            "waiting_for": pending,
            "live": self._live_for(viewer),
            "seats": [
                {"name": p.name, "kind": p.config.kind, "avatar": getattr(p.config, "avatar", ""), "stack": p.stack,
                 "net_bb": round(p.net / self.config.big_blind, 1),
                 "cost_usd": round(p.cost_usd, 5), "rebuys": p.rebuys,
                 "cards": (list(record.hole_cards.get(p.name, ()))
                           if record and viewer and p.name == viewer else [])}
                for p in self.table.players],
            "last_hand": ({"hand_id": record.hand_id, "board": record.board,
                           "nets": record.nets} if record else None),
            "exploits": ({"players": feed.tracked_players,
                          "hands": feed.hands_ingested,
                          "from_history": getattr(feed, "seeded_hands", 0)}
                         if feed is not None else None),
            "cost_usd": round(sum(p.cost_usd for p in self.table.players), 5),
            "events": list(self.events[-12:]),
        }


_ACTIONS = {"fold": ActionType.FOLD, "check": ActionType.CHECK,
            "call": ActionType.CALL, "bet": ActionType.BET,
            "raise": ActionType.RAISE}


def legal_options(state: GameState, bounds: dict | None = None) -> list[dict]:
    """What a player may do, shaped so it can be drawn as buttons directly.

    ``min`` and ``max`` are the **total for the street** — "raise to" — because
    that is exactly how `Table.play_hand` reads an action.

    ``bounds`` comes straight from the engine (``raise_to_min``,
    ``raise_to_max``, ``can_fold``) and wins when present. Without it the values
    are derived from ``GameState``, which is only **approximate**: the state
    carries an increment and a remaining stack, and crucially cannot show that
    betting is closed — after a short all-in, a raise button appeared and the
    engine performed it as a call. ``raise_to_min = None`` means raising is not
    allowed and no button is offered.
    """
    hero = state.hero
    stack = float(hero.stack) if hero else 0.0
    committed = float(hero.committed) if hero else 0.0
    bounds = bounds or {}
    options: list[dict] = []
    can_fold = bounds.get("can_fold", state.to_call > 0)
    if state.to_call > 0:
        if can_fold:
            options.append({"action": "fold", "label": "Fold"})
        options.append({"action": "call", "label": f"Call {state.to_call:g}",
                        "amount": min(state.to_call, stack)})
    else:
        options.append({"action": "check", "label": "Check"})

    if "raise_to_min" in bounds:
        low, high = bounds["raise_to_min"], bounds["raise_to_max"]
        if low is None or high is None:
            return options                       # betting is closed
        low, high = float(low), float(high)
    elif stack > state.to_call:
        # The largest bet on this street; the minimum raise is measured from it.
        level = max((float(s.committed) for s in state.seats), default=0.0)
        high = stack + committed                 # all-in, including what is in
        low = min(level + float(state.min_raise), high)
    else:
        return options
    kind = "raise" if state.to_call > 0 else "bet"
    options.append({"action": kind, "label": kind.capitalize(),
                    "min": low, "max": high, "pot": round(state.pot, 2)})
    return options


def _default_action(state: GameState) -> Action:
    """The safest move when nobody answers: check if free, otherwise fold."""
    if state.to_call <= 0:
        return Action(ActionType.CHECK, 0.0, "no answer")
    return Action(ActionType.FOLD, 0.0, "no answer")


__all__ = ["TableSession", "legal_options"]
