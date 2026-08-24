"""The prompt variants a table knows about — a registry, not an import.

Seat configuration needs three things: which variants exist, which is
recommended, and which support a second critic pass. The arena itself ships one
variant (`BUILTIN_VARIANT`): a simple prompt turning a game state and its legal
actions into a decision.

Anyone with their own catalogue registers it through `set_catalog`. That is how
a private bot plugs its prompts in without the arena having to know they exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# The name of the built-in variant — what the table calls "whatever the
# provider was given".
BUILTIN_VARIANT = "default"
# A second built-in: read the opponents first, then decide. Two calls per
# decision. Shipped as a worked example of the shape, not as a strong strategy.
TWO_PASS_VARIANT = "reads_first"


@dataclass(frozen=True)
class PromptBundle:
    """The two halves of a prompt: the standing instruction and the spot.

    ``render`` is called as ``render(state, legal, seat_name=..., big_blind=...)``
    and returns the user half as plain text.

    **Two passes.** Fill in ``range_system`` and ``range_render`` and the seat
    makes two calls instead of one: first it asks the model what the opponents
    are holding, then it asks for a decision with that answer in hand. A
    two-pass ``render`` is additionally given ``ranges=<the first answer>``, so
    it has to accept that keyword.

    Two passes cost twice the calls and twice the wait. Whether the reads are
    worth it is exactly the kind of thing this seam exists to let you measure:
    register both variants and sit them at the same table.
    """

    system: str
    render: Callable[..., str]
    range_system: str = ""
    range_render: Callable[..., str] | None = None

    @property
    def two_pass(self) -> bool:
        return bool(self.range_system and self.range_render)


@dataclass(frozen=True)
class PromptCatalog:
    """What a table needs to know about prompts to validate and play a seat."""

    variants: tuple[str, ...]
    recommended: str
    # Variants with a two-pass critic; `critic_model` only means anything there.
    bayes_chain: frozenset[str]
    # The schema rules of a variant — `stream_early_stop` asks for
    # `field_order`. An unknown variant answers empty rather than raising.
    rule_ids: Callable[[str], frozenset[str]]
    # Can the provider answer as a stream? Without an LLM layer, nobody can.
    supports_streaming: Callable[[str], bool]
    # The prompt a variant actually plays. Without this a registered catalogue
    # could rename the menu but never change a single decision — which is what
    # happened: seats ignored `prompt_variant` and always played the built-in
    # prompt, so "register your own catalogue" did nothing at all.
    bundle: Callable[[str], PromptBundle]


def _builtin_bundle(variant: str) -> PromptBundle:
    # Imported here: the catalogue is engine-level and the prompt lives in the
    # LLM layer, which is an optional extra.
    from ..llm import prompts
    if variant == TWO_PASS_VARIANT:
        return PromptBundle(
            system=prompts.DECISION_SYSTEM, render=prompts.render_decision,
            range_system=prompts.RANGE_SYSTEM, range_render=prompts.render_range)
    return PromptBundle(system=prompts.SYSTEM, render=prompts.render)


def _builtin_catalog() -> PromptCatalog:
    return PromptCatalog(
        variants=(BUILTIN_VARIANT, TWO_PASS_VARIANT),
        # The single-pass one stays recommended: a table is people waiting, and
        # the second call doubles both the bill and the wait.
        recommended=BUILTIN_VARIANT,
        bayes_chain=frozenset(),
        rule_ids=lambda _variant: frozenset(),
        supports_streaming=lambda _model: False,
        bundle=_builtin_bundle,
    )


_catalog: PromptCatalog | None = None


def catalog() -> PromptCatalog:
    """The current catalogue; derived once, then kept."""
    global _catalog
    if _catalog is None:
        _catalog = _builtin_catalog()
    return _catalog


def set_catalog(value: PromptCatalog | None) -> None:
    """Set the catalogue; ``None`` restores the built-in one."""
    global _catalog
    _catalog = value


def bundle_for(variant: str) -> PromptBundle:
    """The prompt a seat should play. Unknown variants fall back to the
    recommended one rather than raising mid-hand."""
    cat = catalog()
    if variant not in cat.variants:
        variant = cat.recommended
    return cat.bundle(variant)


def recommended_variant() -> str:
    return catalog().recommended


# --- the "live" variant ----------------------------------------------------
#
# Some presets name a bot seat after whichever variant is being played for real
# elsewhere. A seat name reads as an opponent's identity at the table, so a
# hardcoded one would mislead anyone building reads from names. Whoever has such
# a setting registers a provider for it; without one, the recommended variant
# is used.
_live_provider: Callable[[], str] | None = None


def set_live_variant_provider(provider: Callable[[], str] | None) -> None:
    global _live_provider
    _live_provider = provider


def live_variant() -> str:
    """The variant being played for real; the recommended one if unset."""
    provider = _live_provider or recommended_variant
    try:
        return provider()
    except Exception:                    # noqa: BLE001 — a missing setting is fine
        return recommended_variant()
