"""The page and the server have to agree on the shape of `options`.

The seat editor read `options.kinds` while the server sent `seat_kinds`, so the
kind dropdown was built from `undefined` and the whole seat panel died on the
first render. Nothing failed loudly: the server answered 200 and the tests
never loaded the page.

So the contract is checked here instead — every `options.X` the page reads must
be something `_options()` actually sends.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pokerarena.server.host import PAGE, ArenaHandler

# `find` and `map` are Array methods called on an options list, not keys.
NOT_KEYS = {"find", "map", "filter", "includes", "length"}
# Sent only when the provider layer knows something about a model; the page is
# written to degrade to an empty object, and `caps()` is tested for that.
OPTIONAL = {"model_caps"}


def _keys_the_page_reads() -> set[str]:
    page = Path(PAGE).read_text(encoding="utf-8")
    return {m for m in re.findall(r"options\.([a-zA-Z_][a-zA-Z0-9_]*)", page)} - NOT_KEYS


def _options_payload() -> dict:
    # `_options` needs nothing from a live request beyond the lock flag.
    handler = ArenaHandler.__new__(ArenaHandler)
    handler.lock_setup = False
    return ArenaHandler._options(handler)


def test_page_reads_only_keys_the_server_sends() -> None:
    sent = set(_options_payload())
    missing = sorted(_keys_the_page_reads() - sent - OPTIONAL)
    assert not missing, f"the page reads options that are never sent: {missing}"


@pytest.mark.parametrize("key", ["seat_kinds", "seat_labels", "defaults", "presets",
                                 "models", "prompt_variants",
                                 "blind_structures", "needs_owner"])
def test_options_carries_what_the_editor_needs(key: str) -> None:
    assert key in _options_payload()


def test_defaults_describe_a_seat_the_editor_can_build() -> None:
    """The editor clones `defaults` for a new seat, so it must be a legal one."""
    payload = _options_payload()
    assert payload["defaults"]["kind"] in payload["seat_kinds"]


def test_every_kind_the_server_offers_has_a_label() -> None:
    payload = _options_payload()
    for kind in payload["seat_kinds"]:
        assert kind in payload["seat_labels"]
