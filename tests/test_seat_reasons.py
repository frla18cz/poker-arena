"""A seat has to be able to say why — and a broken one has to say it is broken.

`LlmSeat` parsed the model's `why` into `last_reason` and set `last_error` when
the model could not be reached, and then nobody ever read either. A seat whose
key was missing folded every hand in silence and looked like a nit. The class
docstring promised the opposite: "a seat that quietly plays badly because a key
expired is worse than one that visibly does nothing."

So the reason travels on the Action now, and these tests hold it there.
"""
from __future__ import annotations

import pytest

from pokerarena.contract.game_state import ActionType, GameState, Seat, Street
from pokerarena.llm import client
from pokerarena.seats.heuristic import HeuristicSeat
from pokerarena.seats.llm import LlmSeat


def spot(to_call: float = 5.0) -> GameState:
    return GameState(street=Street.FLOP, hero_cards=["As", "Kd"],
                     board=["Ac", "7h", "2s"], pot=10.0, to_call=to_call,
                     seats=[Seat(seat_no=0, name="hero", is_hero=True),
                            Seat(seat_no=1, name="v")])


@pytest.fixture
def answer(monkeypatch):
    def use(text):
        monkeypatch.setattr(client, "complete", lambda *a, **k: text)
    return use


def test_the_models_why_reaches_the_action(answer) -> None:
    answer('{"action":"call","why":"top pair, cheap"}')
    action = LlmSeat("openai/gpt-5.2").decide(spot())
    assert action.type is ActionType.CALL
    assert action.reason == "top pair, cheap"


def test_an_unreachable_model_says_so_on_the_action(monkeypatch) -> None:
    def unavailable(*a, **k):
        raise client.LlmUnavailable("no key")
    monkeypatch.setattr(client, "complete", unavailable)

    action = LlmSeat("openai/gpt-5.2").decide(spot())
    assert action.type is ActionType.FOLD
    assert "no answer from" in action.reason and "no key" in action.reason
    assert action.meta["failed"] is True


def test_an_unusable_answer_says_so_too(answer) -> None:
    answer("I reckon I fold, mate")
    action = LlmSeat("openai/gpt-5.2").decide(spot())
    assert action.meta["failed"] is True
    assert "unusable answer" in action.reason


def test_a_working_model_is_not_flagged_as_failed(answer) -> None:
    answer('{"action":"call","why":"fine"}')
    action = LlmSeat("openai/gpt-5.2").decide(spot())
    assert not (action.meta or {}).get("failed")


def test_the_built_in_bot_explains_itself() -> None:
    for to_call in (0.0, 5.0):
        assert HeuristicSeat().decide(spot(to_call)).reason, (
            f"no reason given when to_call={to_call}")


def test_the_reason_reaches_the_hand_snapshot() -> None:
    """The table has to carry it; the page reads it from there."""
    from pokerarena.engine.table import Table
    from pokerarena.engine.table_config import SeatConfig, TableConfig

    config = TableConfig(seats=tuple(
        SeatConfig(f"Bot{i}", kind="heuristic") for i in range(1, 4)))
    table = Table(config)
    record = table.play_hand()
    assert record.actions, "the hand recorded no actions"
    assert any(a.get("reason") for a in record.actions), (
        "no action carried a reason into the snapshot")
