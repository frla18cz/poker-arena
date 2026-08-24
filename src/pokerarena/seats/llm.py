"""A seat that asks a language model what to do.

One call per decision, one line of JSON back. If the model is unreachable, or
answers with something that is not a legal action, the seat falls back to the
cheapest safe move — check when it is free, fold when it is not — and says so
in ``last_error``. It never guesses an action the table did not offer.

That fallback is not a hidden strategy: a seat that quietly plays badly because
a key expired is worse than one that visibly does nothing.
"""
from __future__ import annotations

import json
import random
import re

from ..contract.game_state import Action, ActionType, GameState
from ..engine import prompt_catalog
from ..llm import client

_ACTIONS = {
    "fold": ActionType.FOLD,
    "check": ActionType.CHECK,
    "call": ActionType.CALL,
    "bet": ActionType.BET,
    "raise": ActionType.RAISE,
}
# Models like to wrap the JSON in an explanation or a fenced block, however
# firmly the prompt says otherwise. Pulling out the first object is cheaper
# than arguing about it.
_JSON = re.compile(r"\{.*?\}", re.DOTALL)

# How much of a seat's budget the first pass of a two-pass variant may take.
# The reads are worth nothing if the decision then times out.
RANGE_PASS_SHARE = 0.45
# Reads are meant to be a few short lines; without a ceiling a model writes an
# essay and spends the decision's budget doing it.
RANGE_PASS_TOKENS = 300


def legal_actions(state: GameState) -> list[str]:
    """What is legal here, named the way the model will see it."""
    if state.to_call and state.to_call > 0:
        return ["fold", "call", "raise"]
    return ["check", "bet"]


def parse(text: str, state: GameState, legal: list[str]) -> Action | None:
    """The model's answer as an action, or ``None`` if it cannot be used."""
    match = _JSON.search(text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None

    name = str(data.get("action", "")).strip().lower()
    if name not in legal or name not in _ACTIONS:
        return None

    kind = _ACTIONS[name]
    if kind in (ActionType.FOLD, ActionType.CHECK):
        return Action(kind)
    if kind is ActionType.CALL:
        return Action(kind, amount=float(state.to_call or 0.0))
    try:
        amount = float(data.get("amount") or 0.0)
    except (TypeError, ValueError):
        return None
    if amount <= 0:
        return None
    return Action(kind, amount=round(amount, 2))


class LlmSeat:
    """A seat driven by a language model."""

    def __init__(self, model: str, *, name: str = "you",
                 temperature: float = 0.2, timeout_s: float = 20.0,
                 big_blind: float = 1.0, variant: str = "") -> None:
        self.model = model
        self.name = name
        # Resolved per decision, not here: a catalogue can be registered after
        # the table is built, and the seat should pick it up.
        self.variant = variant
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.big_blind = big_blind
        self.last_error: str = ""
        self.last_reason: str = ""
        # The reads from a two-pass variant's first call, for the hand log.
        self.last_ranges: str = ""

    def decide(self, state: GameState, rng: random.Random | None = None) -> Action:
        legal = legal_actions(state)
        self.last_error = ""
        self.last_ranges = ""
        bundle = prompt_catalog.bundle_for(
            self.variant or prompt_catalog.recommended_variant())
        extra: dict[str, str] = {}
        calls = 1
        try:
            if bundle.two_pass:
                # The first pass gets a share of the budget, not all of it:
                # spending the whole ceiling on reads would leave the decision
                # with nothing, and the seat would fall back having paid twice.
                self.last_ranges = client.complete(
                    self.model, bundle.range_system,
                    bundle.range_render(state, legal, seat_name=self.name,
                                        big_blind=self.big_blind),
                    temperature=self.temperature,
                    timeout_s=self.timeout_s * RANGE_PASS_SHARE,
                    max_tokens=RANGE_PASS_TOKENS).strip()
                extra["ranges"] = self.last_ranges
                calls = 2
            text = client.complete(
                self.model, bundle.system,
                bundle.render(state, legal, seat_name=self.name,
                              big_blind=self.big_blind, **extra),
                temperature=self.temperature,
                timeout_s=self.timeout_s * ((1 - RANGE_PASS_SHARE) if calls == 2 else 1))
        except client.LlmUnavailable as exc:
            self.last_error = str(exc)
            return self._safe(state, str(exc))

        action = parse(text, state, legal)
        if action is None:
            self.last_error = f"unusable answer: {text[:120]!r}"
            return self._safe(state, self.last_error)
        match = re.search(r'"why"\s*:\s*"([^"]{0,80})"', text)
        self.last_reason = match.group(1) if match else ""
        action.reason = self.last_reason
        action.meta = {**(action.meta or {}), "model": self.model,
                       "variant": self.variant or prompt_catalog.recommended_variant(),
                       "calls": calls}
        if self.last_ranges:
            action.meta["ranges"] = self.last_ranges
        return action

    def _safe(self, state: GameState, why: str = "") -> Action:
        """Check when it is free, fold when it is not. No guessing.

        The reason travels with the action. A seat that folds every hand
        because a key is missing has to say so at the table — otherwise it
        looks like a bad player, which is exactly what this failure hid
        behind before: `last_error` was set and then read by nobody.
        """
        kind = ActionType.CHECK if not state.to_call else ActionType.FOLD
        return Action(kind,
                      reason=(f"no answer from {self.model}: {why}" if why else ""),
                      meta={"model": self.model, "failed": bool(why)})
