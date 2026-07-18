"""Deterministic utility screening for automatic single-source references."""

from __future__ import annotations

import re
from dataclasses import dataclass

_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,15}\b")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'$%.-]*")
_SUBSTANCE = re.compile(
    r"\b(?:because|since|but|although|however|means?|requires?|increas(?:e|ed|es|ing)|"
    r"decreas(?:e|ed|es|ing)|rais(?:e|ed|es|ing)|cut|sell|sold|buy|bought|"
    r"believe|think|better|worse|market|capital|price|cost|revenue|launch|"
    r"announce(?:d|s)?|charging|percent|ecosystem)\b|[$%]|\d",
    re.IGNORECASE,
)
_LOW_CONTEXT_REACTION = re.compile(
    r"^(?:lmao|lol|wow|nice|true|same|agreed|happy to see|good to see|love to see)\b",
    re.IGNORECASE,
)
_EVENT = re.compile(
    r"\b(?:breaking|live|announce(?:d|s)?|launch(?:ed|es)?|rais(?:e|ed|es)?|"
    r"cut|sign(?:ed|s)?|charg(?:e|ed|es|ing)|acquir(?:e|ed|es)|report(?:ed|s)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ReferenceEligibility:
    eligible: bool
    utility_score: float
    reason_code: str
    editorial_intent: str


def assess_reference_eligibility(text: str) -> ReferenceEligibility:
    """Keep useful source propositions while cheaply rejecting conversational debris."""

    normalized = " ".join(text.split()).strip()
    without_links = _URL.sub(" ", normalized)
    without_scaffolding = _MENTION.sub(" ", without_links)
    words = _WORD.findall(without_scaffolding)
    word_count = len(words)
    has_substance = bool(_SUBSTANCE.search(without_scaffolding))

    if not normalized or not words:
        return ReferenceEligibility(False, 0.0, "EMPTY_OR_URL_ONLY", "quote_reaction")
    if word_count < 5:
        return ReferenceEligibility(False, 0.05, "TOO_LITTLE_CONTEXT", "quote_reaction")
    if word_count <= 8 and (_LOW_CONTEXT_REACTION.search(without_scaffolding) or not has_substance):
        return ReferenceEligibility(False, 0.12, "LOW_CONTEXT_REACTION", "quote_reaction")

    length_score = min(1.0, word_count / 24)
    structure_score = min(
        1.0,
        0.25
        + 0.25 * float(has_substance)
        + 0.20 * float(bool(re.search(r"[.!?]\s+|\n", text)))
        + 0.15 * float(bool(re.search(r"[$%]|\d", text)))
        + 0.15 * float(bool(re.search(r"\b(?:but|because|since|although|however)\b", text, re.I))),
    )
    utility = round(0.55 * length_score + 0.45 * structure_score, 4)
    intent = "report_event" if _EVENT.search(without_scaffolding) else "quote_reaction"
    return ReferenceEligibility(True, utility, "REFERENCE_USEFUL", intent)
