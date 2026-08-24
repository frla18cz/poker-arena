"""A minimal QR encoder — exactly as much as a seat link needs.

Written rather than pulled in because the package is stdlib-only and the page
must not fetch anything from a CDN. More to the point, a seat link carries a
token — the key to someone's cards — and handing that to a third-party image
service would be the worst possible place to leak it.

The scope is deliberately narrow:

* **byte mode**, which covers any ASCII or UTF-8 URL,
* **error correction level L**, the most data per area; these codes are scanned
  from a screen held in someone's hand, not off a scuffed crate,
* **versions 1-5**, up to about 106 bytes. Through version 5, level L always
  uses **a single block**, which avoids block interleaving — the part of the
  specification that goes wrong most often. Longer input raises `QrError`
  rather than truncating silently.

The output is a `list[list[bool]]` matrix, where True is a dark module, or SVG
directly. Verified against `segno`: for every version and all eight masks the
matrix matches bit for bit (`tests/test_qr.py`).
"""
from __future__ import annotations

# Data and error-correction codewords per version at level L:
# version -> (data codewords, EC codewords per block). Through 5 there is one block.
_CAPACITY_L = {1: (19, 7), 2: (34, 10), 3: (55, 15), 4: (80, 20), 5: (108, 26)}
# Centre of the alignment pattern; version 1 has none.
_ALIGNMENT = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30]}
_EC_L_BITS = 0b01           # level L indicator in the format field


class QrError(ValueError):
    """The input does not fit the supported versions."""


# -- the GF(256) field --------------------------------------------------------

_EXP = [0] * 512
_LOG = [0] * 256


def _init_tables() -> None:
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        x <<= 1
        if x & 0x100:           # 0x11D, the QR primitive polynomial
            x ^= 0x11D
    for i in range(255, 512):
        _EXP[i] = _EXP[i - 255]


_init_tables()


def _mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _generator(degree: int) -> list[int]:
    """The Reed-Solomon generator polynomial of a given degree."""
    poly = [1]
    for i in range(degree):
        nxt = [0] * (len(poly) + 1)
        for j, coef in enumerate(poly):
            nxt[j] ^= _mul(coef, 1)
            nxt[j + 1] ^= _mul(coef, _EXP[i])
        poly = nxt
    return poly


def _ec_codewords(data: list[int], count: int) -> list[int]:
    gen = _generator(count)
    rest = list(data) + [0] * count
    for i in range(len(data)):
        coef = rest[i]
        if coef:
            for j, g in enumerate(gen):
                rest[i + j] ^= _mul(g, coef)
    return rest[len(data):]


# -- building the bit stream --------------------------------------------------

def _pick_version(length: int) -> int:
    for version, (data_words, _) in sorted(_CAPACITY_L.items()):
        # 4 mode bits + the count (8 bits through version 9) + data + 4 terminator bits.
        if 4 + 8 + length * 8 <= data_words * 8:
            return version
    raise QrError(f"{length} bytes will not fit version 5 at level L (~106 max)")


def _bitstream(payload: bytes, version: int) -> list[int]:
    data_words, _ = _CAPACITY_L[version]
    bits: list[int] = []

    def put(value: int, width: int) -> None:
        for shift in range(width - 1, -1, -1):
            bits.append((value >> shift) & 1)

    put(0b0100, 4)                       # byte mode
    put(len(payload), 8)                 # character count, 8 bits through version 9
    for byte in payload:
        put(byte, 8)
    # The terminator: at most four zero bits.
    put(0, min(4, data_words * 8 - len(bits)))
    while len(bits) % 8:                 # pad out to a whole codeword
        bits.append(0)
    words = [int("".join(str(b) for b in bits[i:i + 8]), 2)
             for i in range(0, len(bits), 8)]
    # Pad with the specified alternating pair until the block is full.
    for i in range(data_words - len(words)):
        words.append(0xEC if i % 2 == 0 else 0x11)
    return words


# -- laying out the matrix ----------------------------------------------------

def _blank(size: int):
    return [[None] * size for _ in range(size)]


def _place_function_patterns(grid, version: int) -> None:
    size = len(grid)

    def finder(row: int, col: int) -> None:
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                r, c = row + dr, col + dc
                if not (0 <= r < size and 0 <= c < size):
                    continue
                edge = max(abs(dr - 3), abs(dc - 3))
                grid[r][c] = edge != 2 and edge <= 3

    finder(0, 0)
    finder(0, size - 7)
    finder(size - 7, 0)

    for i in range(8, size - 8):         # timing patterns
        grid[6][i] = i % 2 == 0
        grid[i][6] = i % 2 == 0

    centers = _ALIGNMENT[version]
    for r in centers:
        for c in centers:
            # Skip the corners, where a finder pattern already sits.
            if (r < 8 and c < 8) or (r < 8 and c > size - 9) \
                    or (r > size - 9 and c < 8):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    grid[r + dr][c + dc] = max(abs(dr), abs(dc)) != 1

    grid[size - 8][8] = True             # the always-dark module


def _format_positions(size: int):
    """Both copies of the format field; the order follows bits 14 down to 0."""
    first = [(8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8),
             (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8)]
    second = [(size - 1, 8), (size - 2, 8), (size - 3, 8), (size - 4, 8),
              (size - 5, 8), (size - 6, 8), (size - 7, 8),
              (8, size - 8), (8, size - 7), (8, size - 6), (8, size - 5),
              (8, size - 4), (8, size - 3), (8, size - 2), (8, size - 1)]
    return first, second


def _format_bits(mask: int) -> int:
    """The 15-bit format field: 5 data bits, a BCH(15,5) remainder, masked 0x5412."""
    value = (_EC_L_BITS << 3) | mask
    rest = value << 10
    while rest.bit_length() >= 11:
        rest ^= 0b10100110111 << (rest.bit_length() - 11)
    return ((value << 10) | rest) ^ 0b101010000010010


def _place_data(grid, words: list[int]) -> None:
    """Data bits, zigzagging up from the bottom right."""
    size = len(grid)
    bits = [(word >> shift) & 1 for word in words for shift in range(7, -1, -1)]
    index = 0
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:                     # skip the timing pattern column
            col -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for row in rows:
            for c in (col, col - 1):
                if grid[row][c] is None:
                    grid[row][c] = bool(bits[index]) if index < len(bits) else False
                    index += 1
        upward = not upward
        col -= 2


def _mask_bit(mask: int, row: int, col: int) -> bool:
    if mask == 0:
        return (row + col) % 2 == 0
    if mask == 1:
        return row % 2 == 0
    if mask == 2:
        return col % 3 == 0
    if mask == 3:
        return (row + col) % 3 == 0
    if mask == 4:
        return (row // 2 + col // 3) % 2 == 0
    if mask == 5:
        return (row * col) % 2 + (row * col) % 3 == 0
    if mask == 6:
        return ((row * col) % 2 + (row * col) % 3) % 2 == 0
    return ((row + col) % 2 + (row * col) % 3) % 2 == 0


def _penalty(matrix) -> int:
    """The four penalty rules from the standard; lower is better."""
    size = len(matrix)
    score = 0

    # 1) runs of five or more identical modules
    for line in list(matrix) + [list(col) for col in zip(*matrix)]:
        run, prev = 1, line[0]
        for cell in line[1:]:
            if cell == prev:
                run += 1
            else:
                if run >= 5:
                    score += 3 + (run - 5)
                run, prev = 1, cell
        if run >= 5:
            score += 3 + (run - 5)

    # 2) bloky 2×2
    for r in range(size - 1):
        for c in range(size - 1):
            quad = (matrix[r][c], matrix[r][c + 1],
                    matrix[r + 1][c], matrix[r + 1][c + 1])
            if all(quad) or not any(quad):
                score += 3

    # 3) the 1:1:3:1:1 pattern with four light modules beside it
    needle_a = [True, False, True, True, True, False, True,
                False, False, False, False]
    needle_b = list(reversed(needle_a))
    for line in list(matrix) + [list(col) for col in zip(*matrix)]:
        for i in range(size - 10):
            window = line[i:i + 11]
            if window == needle_a or window == needle_b:
                score += 40

    # 4) how far the share of dark modules is from half
    dark = sum(sum(1 for cell in row if cell) for row in matrix)
    ratio = dark * 100 // (size * size)
    score += 10 * min(abs(ratio - 50) // 5, abs(ratio - 50 + 4) // 5)
    return score


def matrix(text: str, *, mask: int | None = None) -> list[list[bool]]:
    """The QR matrix for ``text``; ``True`` is a dark module.

    ``mask`` is normally left alone — the one with the lowest penalty is chosen,
    as the standard says. It can be forced to compare against a reference
    encoder in tests.
    """
    payload = text.encode("utf-8")
    version = _pick_version(len(payload))
    data_words = _bitstream(payload, version)
    _, ec_count = _CAPACITY_L[version]
    words = data_words + _ec_codewords(data_words, ec_count)

    size = version * 4 + 17
    base = _blank(size)
    _place_function_patterns(base, version)
    reserved = [[cell is not None for cell in row] for row in base]
    # The format field depends on the mask, so for now only reserve its space.
    first, second = _format_positions(size)
    for r, c in first + second:
        reserved[r][c] = True
        if base[r][c] is None:
            base[r][c] = False
    for row in range(size):
        for col in range(size):
            if reserved[row][col]:
                continue
            base[row][col] = None
    _place_data(base, words)

    best = None
    for candidate in (range(8) if mask is None else [mask]):
        grid = [[bool(cell) for cell in row] for row in base]
        for row in range(size):
            for col in range(size):
                if not reserved[row][col] and _mask_bit(candidate, row, col):
                    grid[row][col] = not grid[row][col]
        bits = _format_bits(candidate)
        for index, (r, c) in enumerate(first):
            grid[r][c] = bool((bits >> (14 - index)) & 1)
        for index, (r, c) in enumerate(second):
            grid[r][c] = bool((bits >> (14 - index)) & 1)
        grid[size - 8][8] = True
        score = _penalty(grid)
        if best is None or score < best[0]:
            best = (score, grid)
    return best[1]


def svg(text: str, *, module: int = 4, quiet: int = 4) -> str:
    """The QR code as standalone SVG. ``quiet`` is the mandatory light margin.

    The colours are hardcoded black on white: a QR code is read by contrast, and
    a dark page theme could invert it.
    """
    grid = matrix(text)
    size = len(grid)
    side = (size + quiet * 2) * module
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" '
             f'height="{side}" viewBox="0 0 {side} {side}" '
             f'shape-rendering="crispEdges" role="img" '
             f'aria-label="QR code for the link">',
             f'<rect width="{side}" height="{side}" fill="#fff"/>']
    for row in range(size):
        for col in range(size):
            if grid[row][col]:
                x = (col + quiet) * module
                y = (row + quiet) * module
                parts.append(f'<rect x="{x}" y="{y}" width="{module}" '
                             f'height="{module}" fill="#000"/>')
    parts.append("</svg>")
    return "".join(parts)


__all__ = ["QrError", "matrix", "svg"]
