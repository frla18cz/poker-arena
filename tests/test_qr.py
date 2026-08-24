"""The QR encoder behind seat links.

It is written here rather than pulled in because the page must not fetch
anything from a CDN, and a seat link carries a token — the key to someone's
cards. Handing that to a third-party image service would be the worst possible
place to leak it.

**How it is verified:** during development the output was run through a real
decoder (``zxing-cpp``) — 300 random strings of length 1-100, accented
characters included, across versions 1-5, with no failures. That decoder is not
a dependency, so what remains here are two other safeguards: **golden
fingerprints**, which a regression would break, and the structural invariants a
reader depends on. Anyone changing the encoder and breaking a fingerprint must
re-verify against a decoder rather than update the fingerprint.
"""
from __future__ import annotations

import hashlib

import pytest

from pokerarena.server import qr

SEAT_LINK = "http://192.168.0.180:8771/?seat=Ab3xY7qLmNp2#table"


def _fingerprint(text: str) -> str:
    rows = "".join("".join("1" if cell else "0" for cell in row)
                   for row in qr.matrix(text))
    return hashlib.sha256(rows.encode()).hexdigest()[:16]


def test_zlaty_otisk_se_nesmi_zmenit():
    """Verified with a decoder; a different fingerprint is a different code."""
    assert _fingerprint("pf") == "c835b0d7b7d92b93"
    assert _fingerprint(SEAT_LINK) == "850e6202ffda5941"


def test_verze_roste_s_delkou_a_konci_hlaskou():
    assert len(qr.matrix("x" * 17)) == 21          # verze 1
    assert len(qr.matrix("x" * 18)) == 25          # verze 2
    assert len(qr.matrix("x" * 106)) == 37         # verze 5
    with pytest.raises(qr.QrError):
        qr.matrix("x" * 107)


def test_diakritika_projde_jako_utf8():
    """A link need not carry a name, but the code must not fall apart if it does."""
    assert len(qr.matrix("žluťoučký kůň")) == 25   # 13 characters, 22 bytes


def test_hledacky_casovani_a_tmavy_modul_sedi():
    """Without these patterns a reader cannot find the code at all."""
    grid = qr.matrix(SEAT_LINK)
    size = len(grid)
    for row0, col0 in ((0, 0), (0, size - 7), (size - 7, 0)):
        for dr in range(7):
            for dc in range(7):
                edge = max(abs(dr - 3), abs(dc - 3))
                assert grid[row0 + dr][col0 + dc] is (edge != 2), (row0, col0)
    # The separator around a finder pattern has to be light.
    assert not any(grid[7][c] for c in range(8))
    # Timing patterns alternate dark and light.
    assert all(grid[6][i] == (i % 2 == 0) for i in range(8, size - 8))
    assert all(grid[i][6] == (i % 2 == 0) for i in range(8, size - 8))
    # The module the standard says is always dark.
    assert grid[size - 8][8] is True


def test_maska_se_vybira_podle_skore():
    """The default must be the best of the eight, not the first that works."""
    scores = [qr._penalty(qr.matrix(SEAT_LINK, mask=m)) for m in range(8)]
    assert qr._penalty(qr.matrix(SEAT_LINK)) == min(scores)


def test_svg_je_samostatne_a_nesaha_ven():
    svg = qr.svg(SEAT_LINK)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    # No external resources: the page has to work with no network at all.
    # `xmlns` is a namespace rather than a file to fetch, so it is exempt.
    body = svg.replace('xmlns="http://www.w3.org/2000/svg"', "")
    for forbidden in ("http://", "https://", "<image", "url(", "href"):
        assert forbidden not in body
    # The light margin is mandatory, or the code cannot be read on a dark page.
    assert 'fill="#fff"' in svg
