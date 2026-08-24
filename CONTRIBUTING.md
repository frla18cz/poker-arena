# Contributing

The most useful thing you can add here is **a new kind of seat**. The table was
built so that it never needs to know what is behind a chair — only that it can
decide — so a strategy is a contribution that touches almost nothing else.

## A new seat

A seat is any object with a `decide(state, rng=None) -> Action`. It reads a
`GameState` and answers with an `Action` from the legal list. That is the whole
contract; see `src/pokerarena/contract/game_state.py`.

```python
from pokerarena.engine.seat_registry import register

register("my_bot", lambda build: MyStrategy(seed=build.config.seed))
```

Registering is enough to make the kind legal — `known_seat_kinds()` grows with
the registry, so there is no list to edit. If your seat ships inside the arena
rather than your own repo, add it under `src/pokerarena/seats/`, register it in
`seats/__init__.py`, and give it a label in `SEAT_LABELS`.

Two tests guard that wiring, and they will tell you if you missed a step:
`test_every_kind_has_a_label_and_a_factory` and the preset checks beside it.

## Ground rules

- **Run the tests.** `pip install -e '.[dev]'` then `python -m pytest -q`.
- **A seat must never invent an action the table did not offer.** When a
  strategy cannot answer — a model is unreachable, a solver runs out of budget —
  check if that is free and fold if it is not, and record why in the action's
  `reason`. Silently folding a playable hand looks like a bad player rather than
  a broken setup, which is exactly the bug that is hard to find later.
- **Keep the core dependency-free.** The table itself is stdlib-only; anything
  your seat needs belongs in an optional extra, imported lazily so that
  `pip install pokerarena` stays small. `seats/solver.py` is the pattern.
- **English**, in code, comments and commit messages.
- **Say what a strategy does not do.** An honest limit in a docstring is worth
  more than an optimistic one.

## Things that are unlikely to be merged

- Anything that turns this into a poker *site*: accounts, real money, chips that
  mean something. It is a table for an evening with friends, not a platform.
- A strategy that only works against this arena's own bots.
- Vendored model weights or API keys of any kind.

## Reporting something broken

An issue with the preset you chose, the seats at the table and what you expected
instead is plenty. If it involves a language model, say which one — most of the
surprising behaviour lives there rather than in the engine.
