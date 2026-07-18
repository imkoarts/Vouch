"""A deterministic, deliberately limited approximation of X weighted length.

The codepoint weights and transformed HTTP(S) URL length follow common
twitter-text rules, but the local regular expression is not a complete port of
the official URL parser.  In particular, callers must not treat it as covering
every Unicode, bare-domain, or punctuation-boundary edge case.
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_SINGLE_WEIGHT_RANGES = (
    (0, 4351),
    (8192, 8205),
    (8208, 8223),
    (8242, 8247),
)
TRANSFORMED_URL_LENGTH = 23


def _codepoint_weight(character: str) -> int:
    codepoint = ord(character)
    return 1 if any(start <= codepoint <= end for start, end in _SINGLE_WEIGHT_RANGES) else 2


def weighted_length(text: str) -> int:
    """Return the local approximate weight; recognized HTTP(S) URLs count as 23."""

    total = 0
    cursor = 0
    for match in _URL_RE.finditer(text):
        total += sum(_codepoint_weight(character) for character in text[cursor : match.start()])
        total += TRANSFORMED_URL_LENGTH
        cursor = match.end()
    total += sum(_codepoint_weight(character) for character in text[cursor:])
    return total
