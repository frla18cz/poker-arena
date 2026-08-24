"""pokerarena — a poker table you can actually sit at.

Deal, bet, show down; humans in a browser, bots in process, or both at the
same table. The engine speaks one contract — ``pokerarena.contract.GameState``
and ``Action`` — and everything that decides is a plugin behind
``seat_registry``. Bring a heuristic, a solver, or an LLM; the table does not
care which.
"""
from __future__ import annotations

# Importing the package registers the built-in seat kinds. Without this the
# registry stays empty for every entry point except the tests, which import
# `pokerarena.seats` themselves — and a table then hands every bot seat a
# strategy of ``None`` and dies on its first decision. That is exactly what
# happened: `poker-arena` could not play a hand against its own built-in bot.
from . import seats  # noqa: F401,E402  — imported for the registration

__all__ = ["contract", "engine", "seats", "server"]
