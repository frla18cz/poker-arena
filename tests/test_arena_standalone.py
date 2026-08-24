"""The arena has to work on its own — that is the whole point of the package.

These tests deliberately assert the boundary rather than the poker: a table
that quietly needs a private bot to be installed is not a table anyone else
can use.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import pokerarena.seats                                   # registers the seats
from pokerarena.contract.game_state import Action, ActionType, GameState, Street
from pokerarena.engine.seat_registry import registered_kinds
from pokerarena.engine.table import Table
from pokerarena.engine.table_config import (
    SeatConfig, TableConfig, TableConfigError, known_seat_kinds,
)
from pokerarena.seats.heuristic import HeuristicSeat
from pokerarena.seats.llm import legal_actions, parse
from pokerarena.server.host import build_server


def test_importing_the_arena_pulls_in_nothing_unexpected():
    """Importing the arena must reach nothing but the stdlib and its own deps.

    A private strategy plugs in through `seat_registry`, which means the arena
    must never import it — and more generally must not quietly acquire a
    dependency nobody declared. Checked as an allowlist rather than a blocklist:
    a blocklist only catches the packages you thought to name.
    """
    script = (
        "import sys, pokerarena.seats, pokerarena.engine.table, "
        "pokerarena.server.host\n"
        "print(' '.join(sorted({m.split('.')[0] for m in sys.modules "
        "if not m.startswith('_')})))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, check=True).stdout.split()

    allowed = set(sys.stdlib_module_names) | {
        "pokerarena",        # itself
        "pokerkit",          # dealing, declared in [play]
        "pokersolver",       # solver seats, declared in [solver]
        # Site hooks Python itself injects; nothing to do with the arena.
        "sitecustomize", "usercustomize",
    }
    unexpected = sorted(set(out) - allowed)
    assert not unexpected, (
        f"importing the arena pulled in {unexpected} — anything beyond the "
        "stdlib and the declared extras has no business being imported at "
        "module level"
    )


def test_a_table_of_built_in_bots_plays_itself():
    """Install, run, watch — no key, no model, no configuration."""
    config = TableConfig(
        seats=[SeatConfig(f"Bot{i}", kind="heuristic") for i in range(1, 4)],
        seed=7)
    table = Table(config)

    for _ in range(5):
        table.play_hand()

    # Chips are conserved: nobody invents or loses money in the accounting.
    assert sum(p.stack for p in table.players) == 3 * config.start_stack
    assert all(p.hands > 0 for p in table.players)


def test_seat_kinds_grow_with_the_registry():
    """Registering a factory is enough to make a seat kind legal."""
    from pokerarena.engine.seat_registry import register

    assert "heuristic" in registered_kinds()
    assert "my_bot" not in known_seat_kinds()
    with pytest.raises(TableConfigError):
        SeatConfig("X", kind="my_bot").validate()

    register("my_bot", lambda build: HeuristicSeat())
    try:
        assert "my_bot" in known_seat_kinds()
        SeatConfig("X", kind="my_bot").validate()          # no longer raises
    finally:
        from pokerarena.engine import seat_registry
        seat_registry._factories.pop("my_bot", None)


def test_heuristic_never_folds_when_checking_is_free():
    """The one mistake a beginner would spot immediately."""
    bot = HeuristicSeat()
    state = GameState(street=Street.FLOP, hero_cards=["2c", "7d"],
                      board=["As", "Kh", "9s"], pot=10.0, to_call=0.0)
    assert bot.decide(state).type is not ActionType.FOLD


def test_heuristic_folds_a_bad_hand_facing_a_big_bet():
    bot = HeuristicSeat()
    state = GameState(street=Street.FLOP, hero_cards=["2c", "7d"],
                      board=["As", "Kh", "9s"], pot=10.0, to_call=30.0)
    assert bot.decide(state).type is ActionType.FOLD


class TestLlmAnswers:
    """The LLM seat must never turn a bad answer into an illegal action."""

    def _facing_bet(self) -> GameState:
        return GameState(street=Street.TURN, hero_cards=["Ah", "Kd"],
                         board=["As", "7h", "2c", "9d"], pot=20.0, to_call=10.0)

    def test_reads_a_clean_answer(self):
        state = self._facing_bet()
        action = parse('{"action": "call", "why": "top pair"}', state,
                       legal_actions(state))
        assert action.type is ActionType.CALL
        assert action.amount == 10.0

    def test_digs_the_json_out_of_chatter(self):
        state = self._facing_bet()
        text = 'Sure!\n```json\n{"action": "fold"}\n```\nHope that helps.'
        assert parse(text, state, legal_actions(state)).type is ActionType.FOLD

    def test_refuses_an_action_that_is_not_on_offer(self):
        """Checking is not legal when there is a bet to answer."""
        state = self._facing_bet()
        assert parse('{"action": "check"}', state, legal_actions(state)) is None

    def test_refuses_a_raise_without_a_size(self):
        state = self._facing_bet()
        assert parse('{"action": "raise"}', state, legal_actions(state)) is None

    def test_refuses_nonsense(self):
        state = self._facing_bet()
        assert parse("no idea, sorry", state, legal_actions(state)) is None


class TestServer:
    """One game at a time, and seat links only for the owner."""

    def _serve(self, tmp_path: Path, **kwargs):
        server = build_server("127.0.0.1", 0, tables_root=tmp_path,
                              public_base="http://testhost", **kwargs)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, f"http://127.0.0.1:{port}"

    def _get(self, base: str, path: str):
        with urllib.request.urlopen(base + path) as response:
            return json.loads(response.read())

    def _post(self, base: str, path: str, body: dict):
        request = urllib.request.Request(
            base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    def test_starts_a_game_and_hands_out_seat_links(self, tmp_path: Path):
        server, base = self._serve(tmp_path)
        try:
            assert self._get(base, "/api/table/state")["running"] is False
            self._post(base, "/api/table", {
                "config": {"seats": [{"name": "You", "kind": "human"},
                                     {"name": "Bot", "kind": "heuristic"}]},
                "hands": 2, "label": "test"})
            time.sleep(0.5)

            links = self._get(base, "/api/table/seats")
            assert [row["seat"] for row in links] == ["You"], (
                "only human seats need a link — a bot has nowhere to click")
            assert links[0]["url"].startswith("http://testhost/?seat=")
            assert links[0]["qr"].lstrip().startswith("<svg")
        finally:
            server.shutdown()

    def test_seat_links_are_not_public(self, tmp_path: Path):
        """A seat link is a chair at the table: whoever holds it, plays it."""
        server, base = self._serve(tmp_path, lock_setup=True, trust_local=False,
                                   owner_token="secret")
        try:
            self._post(base, "/api/table?owner=secret", {
                "config": {"seats": [{"name": "You", "kind": "human"},
                                     {"name": "Bot", "kind": "heuristic"}]},
                "hands": 1})
            time.sleep(0.4)

            with pytest.raises(urllib.error.HTTPError) as caught:
                self._get(base, "/api/table/seats")
            assert caught.value.code == 403

            owner_view = self._get(base, "/api/table/seats?owner=secret")
            assert owner_view and owner_view[0]["seat"] == "You"
        finally:
            server.shutdown()

    def test_a_stranger_cannot_start_a_game(self, tmp_path: Path):
        server, base = self._serve(tmp_path, lock_setup=True, trust_local=False,
                                   owner_token="secret")
        try:
            with pytest.raises(urllib.error.HTTPError) as caught:
                self._post(base, "/api/table", {"preset": "friends"})
            assert caught.value.code == 403
        finally:
            server.shutdown()
