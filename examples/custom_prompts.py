"""Your own prompts at the table, without forking the arena.

    python examples/custom_prompts.py

The arena ships one prompt: short, one call per decision, no range pass. If you
have a better one — or a whole family of them you are A/B testing — you do not
edit `llm/prompts.py`. You register a catalogue, and every LLM seat can then be
pointed at any variant in it by name, from the setup panel or from a config.

The catalogue is the seam that keeps a private prompt private. This file could
live in your own repository, import `pokerarena`, and the arena would never
need to know what your prompts say.
"""
from __future__ import annotations

from pokerarena.contract.game_state import GameState, Street
from pokerarena.engine.prompt_catalog import (
    PromptBundle, PromptCatalog, set_catalog,
)

# --- a prompt of your own --------------------------------------------------

TIGHT_SYSTEM = """You are a tight, position-aware No-Limit Hold'em player.

Fold most hands out of position. Bet for value with strong hands and give up
cheaply with weak ones. Do not bluff into more than one opponent.

Reply with one line of JSON and nothing else:

  {"action": "fold|check|call|bet|raise", "amount": <number>, "why": "<8 words>"}

`action` must be one of the legal actions listed. `amount` is the total you put
in for bet/raise. Never invent an action that is not listed.
"""

_STREETS = {Street.PREFLOP: "preflop", Street.FLOP: "flop",
            Street.TURN: "turn", Street.RIVER: "river"}


def render_tight(state: GameState, legal: list[str], *, seat_name: str = "you",
                 big_blind: float = 1.0) -> str:
    """The spot, in the terms this prompt cares about."""
    bb = big_blind or 1.0
    lines = [
        f"Street: {_STREETS.get(state.street, 'preflop')}",
        f"Your cards: {' '.join(state.hero_cards or []) or '??'}",
        f"Board: {' '.join(state.board or []) or '—'}",
        f"Pot: {state.pot / bb:.1f} bb",
        f"To call: {(state.to_call or 0) / bb:.1f} bb",
        f"Opponents left: {max(1, len(state.seats or []) - 1)}",
        f"Legal actions: {', '.join(legal)}",
        "",
        f"You are {seat_name}. JSON only.",
    ]
    return "\n".join(lines)


# --- registering it --------------------------------------------------------

BUNDLES = {
    "tight": PromptBundle(system=TIGHT_SYSTEM, render=render_tight),
}


def install() -> None:
    """Make "tight" a variant every LLM seat can be pointed at."""
    set_catalog(PromptCatalog(
        variants=("tight",),
        recommended="tight",
        # Empty unless a variant runs a second, critic pass.
        bayes_chain=frozenset(),
        rule_ids=lambda _variant: frozenset(),
        supports_streaming=lambda _model: False,
        bundle=lambda variant: BUNDLES[variant],
    ))


def main() -> None:
    from pokerarena.engine.prompt_catalog import bundle_for, catalog

    install()
    print("variants on offer:", catalog().variants)

    state = GameState(street=Street.FLOP, hero_cards=["As", "Kd"],
                      board=["Ac", "7h", "2s"], pot=10.0, to_call=5.0)
    bundle = bundle_for("tight")
    print("\n--- system ---")
    print(bundle.system.strip()[:160], "…")
    print("\n--- the spot this seat would be sent ---")
    print(bundle.render(state, ["fold", "call", "raise"],
                        seat_name="Bot 1", big_blind=2.0))
    print("\nAn `llm` seat with prompt_variant='tight' now plays this prompt.")


if __name__ == "__main__":
    main()
