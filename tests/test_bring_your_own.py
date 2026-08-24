"""The two seams a stranger plugs into: a seat of their own, prompts of their own.

Both were advertised and only one worked. `prompt_variant` was never passed to
the seat and the catalogue could not supply a prompt at all, so "register your
own catalogue" renamed the menu and changed nothing a model ever saw. These
tests hold both seams open by checking the thing that actually matters: that
the registered code is what gets played.
"""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

import pokerarena.seats  # noqa: F401 — registers the built-ins
from pokerarena.contract.game_state import Action, ActionType, GameState, Seat, Street
from pokerarena.engine import prompt_catalog, seat_registry
from pokerarena.engine.prompt_catalog import PromptBundle, PromptCatalog
from pokerarena.engine.table import Table, TableOver
from pokerarena.engine.table_config import SeatConfig, TableConfig, known_seat_kinds
from pokerarena.llm import client
from pokerarena.seats.llm import LlmSeat

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def spot() -> GameState:
    return GameState(street=Street.FLOP, hero_cards=["As", "Kd"],
                     board=["Ac", "7h", "2s"], pot=10.0, to_call=5.0,
                     seats=[Seat(seat_no=0, name="hero", is_hero=True),
                            Seat(seat_no=1, name="v")])


# -- a seat of your own ----------------------------------------------------

def test_a_registered_seat_is_dealt_into_a_real_hand() -> None:
    """Registering is the whole integration: no edit to the arena required."""
    seen = []

    class Nit:
        def decide(self, state, rng=None):
            seen.append(state.street)
            return Action(ActionType.FOLD, reason="not today")

    seat_registry.register("nit_for_test", lambda build: Nit())
    try:
        assert "nit_for_test" in known_seat_kinds()
        config = TableConfig(seats=(
            SeatConfig("nit", kind="nit_for_test"),
            SeatConfig("house-1", kind="heuristic"),
            SeatConfig("house-2", kind="heuristic")))
        Table(config).play_hand()
        assert seen, "the registered seat was never asked to decide"
    finally:
        seat_registry._factories.pop("nit_for_test", None)


# -- prompts of your own ---------------------------------------------------

@pytest.fixture
def own_catalog():
    """Install a catalogue, and put the built-in one back afterwards."""
    sent = {}

    def render(state, legal, *, seat_name="you", big_blind=1.0):
        return f"MINE: {seat_name} may {'/'.join(legal)}"

    prompt_catalog.set_catalog(PromptCatalog(
        variants=("mine", "other"),
        recommended="mine",
        bayes_chain=frozenset(),
        rule_ids=lambda _v: frozenset(),
        supports_streaming=lambda _m: False,
        bundle=lambda v: PromptBundle(system=f"SYSTEM-{v}", render=render),
    ))
    try:
        yield sent
    finally:
        prompt_catalog.set_catalog(None)


def test_a_registered_catalog_is_what_the_model_receives(own_catalog, monkeypatch) -> None:
    """The point of the seam: your prompt reaches the provider, not the arena's."""
    captured = {}

    def fake_complete(model, system, user, **kw):
        captured["system"], captured["user"] = system, user
        return '{"action":"call","why":"ok"}'

    monkeypatch.setattr(client, "complete", fake_complete)
    LlmSeat("openai/gpt-5.2", name="Bot 1", variant="other").decide(spot())

    assert captured["system"] == "SYSTEM-other", "the seat played the wrong variant"
    assert captured["user"].startswith("MINE:"), "the arena's own renderer was used"


def test_an_unknown_variant_falls_back_instead_of_raising(own_catalog, monkeypatch) -> None:
    """A bad name must not kill a hand mid-game."""
    captured = {}
    monkeypatch.setattr(client, "complete", lambda m, s, u, **k: (
        captured.update(system=s) or '{"action":"call"}'))
    LlmSeat("openai/gpt-5.2", variant="does-not-exist").decide(spot())
    assert captured["system"] == "SYSTEM-mine"     # the recommended one


def test_the_built_in_catalog_still_supplies_a_prompt() -> None:
    from pokerarena.llm import prompts
    bundle = prompt_catalog.bundle_for(prompt_catalog.recommended_variant())
    assert bundle.system == prompts.SYSTEM


# -- the examples have to run ----------------------------------------------

@pytest.mark.parametrize("name", ["custom_seat.py", "custom_prompts.py"])
def test_the_examples_run(name: str) -> None:
    """A README that points at a broken example is worse than no example."""
    done = subprocess.run([sys.executable, str(EXAMPLES / name)],
                          capture_output=True, text=True)
    assert done.returncode == 0, f"{name} failed:\n{done.stderr}"


# -- busting out without rebuys -------------------------------------------

def test_a_table_without_rebuys_ends_instead_of_crashing() -> None:
    """PokerKit refuses a zero starting stack; the game is simply over."""
    config = TableConfig(
        seats=(SeatConfig("a", kind="heuristic"), SeatConfig("b", kind="heuristic")),
        start_stack=200, rebuy=False)
    table = Table(config)
    for _ in range(400):
        if len(table.seated()) < 2:
            break
        table.play_hand()
    else:
        pytest.skip("nobody busted in 400 hands")

    with pytest.raises(TableOver):
        table.play_hand()
    assert sum(p.stack for p in table.players) == 2 * config.start_stack


# -- two passes: reads first, then a decision ------------------------------

def two_pass_spot() -> GameState:
    from pokerarena.contract.game_state import ObservedAction
    state = GameState(
        street=Street.FLOP, hero_cards=["As", "Kd"], board=["Ac", "7h", "2s"],
        pot=12.0, to_call=6.0,
        seats=[Seat(seat_no=0, name="Hero", is_hero=True, in_hand=True),
               Seat(seat_no=1, name="Villain", in_hand=True)])
    state.action_history = [
        ObservedAction(street=Street.PREFLOP, seat_no=1, position="BTN",
                       action="raise", amount=6.0),
        ObservedAction(street=Street.PREFLOP, seat_no=0, position="BB",
                       action="call", amount=4.0)]
    return state


def test_the_two_pass_variant_makes_two_calls_and_uses_the_first(monkeypatch) -> None:
    """The reads have to reach the decision, or the second call is just cost."""
    calls = []

    def fake_complete(model, system, user, **kw):
        calls.append({"system": system, "user": user})
        if len(calls) == 1:
            return "BTN: AK, AQ, sets."
        return '{"action":"raise","amount":18,"why":"top pair"}'

    monkeypatch.setattr(client, "complete", fake_complete)
    seat = LlmSeat("openai/gpt-5.2", name="Hero", variant="reads_first")
    action = seat.decide(two_pass_spot())

    assert len(calls) == 2, "a two-pass variant must ask twice"
    assert "BTN: AK, AQ, sets." in calls[1]["user"], (
        "the decision pass never saw the reads")
    assert action.meta["calls"] == 2
    assert action.meta["ranges"] == "BTN: AK, AQ, sets."


def test_the_default_variant_still_makes_one_call(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(client, "complete", lambda m, s, u, **k: (
        calls.append(u) or '{"action":"call"}'))
    action = LlmSeat("openai/gpt-5.2", variant="default").decide(two_pass_spot())
    assert len(calls) == 1
    assert action.meta["calls"] == 1


def test_the_range_pass_does_not_eat_the_whole_budget(monkeypatch) -> None:
    """Reads that time out the decision are worse than no reads."""
    budgets = []
    monkeypatch.setattr(client, "complete", lambda m, s, u, **k: (
        budgets.append(k.get("timeout_s")) or "reads"
        if len(budgets) == 0 else '{"action":"call"}'))

    def capture(model, system, user, **kw):
        budgets.append(kw.get("timeout_s"))
        return "reads" if len(budgets) == 1 else '{"action":"call"}'

    monkeypatch.setattr(client, "complete", capture)
    LlmSeat("openai/gpt-5.2", timeout_s=20.0, variant="reads_first").decide(two_pass_spot())
    assert len(budgets) == 2
    assert sum(budgets) <= 20.0 + 1e-9, "the two passes together overran the budget"
    assert all(b > 0 for b in budgets)


def test_the_range_prompt_does_not_ask_the_hero_about_himself() -> None:
    """A model given the hero's own seat will hand the hero a range too."""
    from pokerarena.llm import prompts
    text = prompts.render_range(two_pass_spot(), ["fold", "call", "raise"],
                                seat_name="Hero", big_blind=2.0)
    assert "Do NOT give a range for yourself" in text
    assert "Villain" in text


def test_two_pass_is_offered_by_the_built_in_catalog() -> None:
    from pokerarena.engine.prompt_catalog import TWO_PASS_VARIANT, bundle_for, catalog
    assert TWO_PASS_VARIANT in catalog().variants
    assert bundle_for(TWO_PASS_VARIANT).two_pass
    assert not bundle_for(catalog().recommended).two_pass, (
        "the recommended variant should stay the cheap single call")
