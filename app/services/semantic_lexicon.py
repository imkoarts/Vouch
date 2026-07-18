"""Bounded lexical normalization for the semantic intermediate representation.

This module maps surface forms into typed lexical categories. It never decides humor safety,
human harm, or reply similarity.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.schemas.content import SemanticEntity

_WORD = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?", re.UNICODE)
_DETERMINERS = frozenset({"a", "an", "the", "this", "that", "these", "those"})

PERSON_SINGULAR = frozenset(
    {
        "person",
        "child",
        "worker",
        "employee",
        "passenger",
        "civilian",
        "patient",
        "driver",
        "resident",
        "victim",
        "founder",
        "officer",
        "soldier",
        "miner",
        "nurse",
        "clinician",
        "survivor",
        "woman",
        "man",
        "boy",
        "girl",
        "mother",
        "father",
        "adult",
        "teenager",
        "technician",
        "contractor",
        "engineer",
        "doctor",
        "physician",
        "paramedic",
        "medic",
        "responder",
        "operator",
        "staffer",
        "crewman",
        "assistant",
    }
)
PERSON_PLURAL = frozenset(
    {
        "people",
        "persons",
        "children",
        "workers",
        "employees",
        "passengers",
        "civilians",
        "patients",
        "drivers",
        "residents",
        "victims",
        "founders",
        "officers",
        "soldiers",
        "miners",
        "nurses",
        "clinicians",
        "survivors",
        "women",
        "men",
        "boys",
        "girls",
        "families",
        "adults",
        "teenagers",
        "technicians",
        "contractors",
        "engineers",
        "doctors",
        "physicians",
        "paramedics",
        "medics",
        "responders",
        "operators",
        "staffers",
        "crew",
        "personnel",
        "assistants",
        "staff",
    }
)
HUMAN_PRONOUNS = frozenset({"he", "she", "him", "her", "they", "them", "their", "his"})
GROUP_PHRASES = (
    "response team",
    "medical team",
    "rescue team",
    "hospital staff",
    "crew members",
)

SOFTWARE_PROCESS = frozenset(
    {
        "process",
        "thread",
        "job",
        "request",
        "session",
        "container",
        "worker process",
        "browser process",
        "database process",
        "pipeline",
        "build",
        "deployment",
    }
)
TECHNICAL_SYSTEM = frozenset(
    {
        "api",
        "app",
        "application",
        "browser",
        "database",
        "network",
        "node",
        "server",
        "service",
        "system",
        "grid",
        "model",
        "component",
        "platform",
        "website",
        "site",
        "infrastructure",
        "chain",
    }
)
PRODUCT = frozenset(
    {
        "product",
        "feature",
        "launch",
        "brand",
        "meme",
        "code",
        "link",
        "battery",
        "battery life",
        "conversation",
        "project",
    }
)
ORGANIZATION = frozenset(
    {
        "company",
        "organization",
        "hospital",
        "agency",
        "startup",
        "regulator",
        "government",
        "team",
        "committee",
    }
)
CRYPTO = frozenset(
    {
        "token",
        "protocol",
        "crypto",
        "bitcoin",
        "ethereum",
        "blockchain",
        "chain",
        "coin",
        "stablecoin",
        "altcoin",
        "dao",
        "ledger",
        "validator",
        "staking pool",
        "liquidity pool",
    }
)
BUSINESS_OBJECT = frozenset(
    {
        "deal",
        "market",
        "campaign",
        "engagement",
        "momentum",
        "performance",
        "traffic",
        "revenue",
        "sales",
        "funding",
        "proposal",
        "strategy",
        "productivity",
        "experience",
        "index",
        "bond",
        "stock",
        "share",
        "fund",
        "etf",
        "derivative",
        "option",
        "future",
        "security",
        "instrument",
        "asset",
        "portfolio",
        "contract",
        "offering",
        "issuance",
    }
)
LEGAL_POLITICAL_OBJECT = frozenset(
    {
        "bill",
        "law",
        "measure",
        "proposal",
        "policy",
        "amendment",
        "resolution",
        "motion",
        "nomination",
        "ordinance",
        "campaign",
        "referendum",
        "ballot",
        "initiative",
        "legislation",
        "act",
        "candidacy",
        "ticket",
        "manifesto",
        "petition",
    }
)
PHYSICAL_EVENT = frozenset(
    {
        "outage",
        "crash",
        "collision",
        "blast",
        "explosion",
        "fire",
        "attack",
        "collapse",
        "incident",
        "accident",
        "earthquake",
        "flood",
        "disaster",
        "blackout",
    }
)
ABSTRACT_CONCEPT = frozenset(
    {
        "wording",
        "statement",
        "headline",
        "label",
        "name",
        "announcement",
        "idea",
        "concept",
        "definition",
        "formula",
        "timing",
        "execution",
        "work",
        "showing",
        "success",
        "failure",
    }
)

POSITIVE_MODIFIERS = frozenset(
    {
        "great",
        "excellent",
        "nice",
        "beautiful",
        "brilliant",
        "flawless",
        "perfect",
        "impeccable",
        "stellar",
        "superb",
        "splendid",
        "magnificent",
        "wonderful",
        "amazing",
        "remarkable",
        "masterful",
        "outstanding",
        "lovely",
        "glorious",
        "textbook",
        "heroic",
        "strong",
        "solid",
        "good",
        "fancy",
    }
)
PRAISE_NOUNS = frozenset(
    {
        "work",
        "job",
        "timing",
        "execution",
        "showing",
        "performance",
        "display",
        "effort",
        "result",
        "response",
        "stuff",
    }
)
SUCCESS_PREDICATES = frozenset(
    {
        "delivered",
        "excelled",
        "triumphed",
        "succeeded",
        "won",
        "nailed",
        "aced",
        "crushed",
        "shined",
        "showed",
        "outdid",
    }
)
REWARD_NOUNS = frozenset(
    {
        "raise",
        "bonus",
        "award",
        "promotion",
        "medal",
        "trophy",
        "applause",
        "ovation",
        "prize",
        "commendation",
    }
)
CELEBRATION_NOUNS = frozenset(
    {
        "win",
        "victory",
        "triumph",
        "success",
        "applause",
        "ovation",
        "cheers",
        "toast",
        "celebration",
    }
)
RATING_NOUNS = frozenset({"star", "stars", "points", "marks", "score"})
PACKAGING_MARKERS = frozenset(
    {
        "glorified",
        "fancy",
        "rebranded",
        "repackaged",
        "renamed",
        "disguised",
        "dressed",
        "pretending",
        "cosplaying",
        "branding",
        "wrapper",
        "clothing",
        "paint",
        "ux",
        "lipstick",
        "costume",
        "makeover",
        "suit",
        "polish",
        "polished",
        "gloss",
        "glossy",
        "veneer",
        "facade",
        "façade",
        "packaging",
        "presentation",
        "aesthetics",
        "coated",
        "wrapped",
        "lacquered",
        "styled",
        "spin",
        "cosmetic",
        "cosmetics",
        "surface",
        "window dressing",
    }
)
REDUCTION_MARKERS = frozenset(
    {
        "just",
        "merely",
        "basically",
        "essentially",
        "effectively",
        "nothing",
        "only",
        "simply",
        "glorified",
        "fancy",
        "rebranded",
        "repackaged",
    }
)
POSITIVE_EVENT_WORDS = frozenset(
    {
        "restored",
        "recovered",
        "completed",
        "resolved",
        "fixed",
        "safely",
        "success",
        "successful",
        "improved",
        "protected",
        "rescued",
        "saved",
        "stabilized",
    }
)
NEGATIVE_EVENT_WORDS = frozenset(
    {
        "outage",
        "failure",
        "failed",
        "error",
        "crash",
        "collapse",
        "damage",
        "damaged",
        "destroyed",
        "attack",
        "blast",
        "explosion",
        "fire",
        "incident",
        "accident",
        "breach",
        "loss",
        "lost",
    }
)

_ENTITY_HEAD_TYPES: tuple[tuple[frozenset[str], str], ...] = (
    (SOFTWARE_PROCESS, "software_process"),
    (TECHNICAL_SYSTEM, "technical_system"),
    (PRODUCT, "product"),
    (CRYPTO, "crypto_or_protocol"),
    (BUSINESS_OBJECT, "business_or_financial_object"),
    (LEGAL_POLITICAL_OBJECT, "legal_or_political_object"),
    (ORGANIZATION, "company_or_organization"),
    (PHYSICAL_EVENT, "physical_event"),
    (ABSTRACT_CONCEPT, "abstract_concept"),
)

_CRYPTO_SUFFIXES = ("coin", "chain", "token", "protocol")
_BUSINESS_COMPOUND_HEADS = frozenset(
    {
        "bond",
        "stock",
        "share",
        "fund",
        "etf",
        "index",
        "contract",
        "security",
        "instrument",
        "asset",
        "portfolio",
        "offering",
        "issuance",
    }
)
_LEGAL_POLITICAL_COMPOUND_HEADS = frozenset(
    {
        "bill",
        "law",
        "measure",
        "amendment",
        "resolution",
        "motion",
        "nomination",
        "ordinance",
        "referendum",
        "ballot",
        "initiative",
        "legislation",
        "act",
        "candidacy",
        "ticket",
        "petition",
    }
)


def normalize_surface(value: str) -> str:
    """Normalize typography and whitespace without assigning semantic meaning."""

    return " ".join(
        value.casefold()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .split()
    ).strip(" \t\n\r.,!?;:\"'")


def lexical_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        match.group(0).casefold().replace("\u2019", "'") for match in _WORD.finditer(value)
    )


def token_set(value: str) -> frozenset[str]:
    return frozenset(lexical_tokens(value))


def contains_any(value: str, terms: Iterable[str]) -> bool:
    tokens = token_set(value)
    return bool(tokens & frozenset(terms))


def _contains_lexical_phrase(value: str, phrase: str) -> bool:
    value_tokens = lexical_tokens(value)
    phrase_tokens = lexical_tokens(phrase)
    if not phrase_tokens or len(phrase_tokens) > len(value_tokens):
        return False
    width = len(phrase_tokens)
    return any(
        value_tokens[index : index + width] == phrase_tokens
        for index in range(len(value_tokens) - width + 1)
    )


def _trim_entity(value: str) -> str:
    tokens = list(lexical_tokens(value))
    while tokens and (tokens[0] in _DETERMINERS or tokens[0].isdigit()):
        tokens.pop(0)
    return " ".join(tokens) or normalize_surface(value)


def classify_entity_span(value: str) -> SemanticEntity:
    """Map a surface span to a bounded entity type; capitalization is ignored."""

    normalized = _trim_entity(value)
    tokens = token_set(normalized)
    normalized_tokens = frozenset(
        token[:-1] if token.endswith("s") and not token.endswith("ss") and len(token) > 3 else token
        for token in tokens
    )
    ordered_tokens = lexical_tokens(normalized)
    terminal_head = ordered_tokens[-1] if ordered_tokens else ""
    singular_terminal_head = (
        terminal_head[:-1]
        if terminal_head.endswith("s")
        and not terminal_head.endswith("ss")
        and len(terminal_head) > 3
        else terminal_head
    )
    singular = tokens & PERSON_SINGULAR
    plural = tokens & PERSON_PLURAL
    # Multiword and exact non-human referents take precedence over embedded human-role tokens
    # such as ``worker`` in ``worker process`` or ``employee`` in ``employee engagement``.
    if (
        normalized in SOFTWARE_PROCESS
        or normalized_tokens & SOFTWARE_PROCESS
        or any(_contains_lexical_phrase(normalized, phrase) for phrase in SOFTWARE_PROCESS)
    ):
        entity_type = "software_process"
        confidence = "high"
    elif (
        normalized in BUSINESS_OBJECT
        or normalized_tokens & BUSINESS_OBJECT
        or any(_contains_lexical_phrase(normalized, phrase) for phrase in BUSINESS_OBJECT)
    ):
        entity_type = "business_or_financial_object"
        confidence = "high"
    elif (
        normalized in PRODUCT
        or normalized_tokens & PRODUCT
        or any(_contains_lexical_phrase(normalized, phrase) for phrase in PRODUCT)
    ):
        entity_type = "product"
        confidence = "high"
    elif plural or any(_contains_lexical_phrase(normalized, phrase) for phrase in GROUP_PHRASES):
        entity_type = "group_of_people"
        confidence = "high"
    elif singular or tokens & HUMAN_PRONOUNS:
        entity_type = "person"
        confidence = "high" if singular else "medium"
    elif terminal_head.endswith(_CRYPTO_SUFFIXES) or singular_terminal_head in CRYPTO:
        entity_type = "crypto_or_protocol"
        confidence = "high"
    elif singular_terminal_head in _BUSINESS_COMPOUND_HEADS:
        entity_type = "business_or_financial_object"
        confidence = "high"
    elif singular_terminal_head in _LEGAL_POLITICAL_COMPOUND_HEADS:
        entity_type = "legal_or_political_object"
        confidence = "high"
    elif normalized in CRYPTO or normalized_tokens & CRYPTO:
        entity_type = "crypto_or_protocol"
        confidence = "high"
    elif normalized in LEGAL_POLITICAL_OBJECT or normalized_tokens & LEGAL_POLITICAL_OBJECT:
        entity_type = "legal_or_political_object"
        confidence = "high"
    elif normalized in TECHNICAL_SYSTEM or normalized_tokens & TECHNICAL_SYSTEM:
        entity_type = "technical_system"
        confidence = "high"
    elif normalized in ORGANIZATION or normalized_tokens & ORGANIZATION:
        entity_type = "company_or_organization"
        confidence = "high"
    elif normalized in PHYSICAL_EVENT or normalized_tokens & PHYSICAL_EVENT:
        entity_type = "physical_event"
        confidence = "high"
    elif normalized in ABSTRACT_CONCEPT or normalized_tokens & ABSTRACT_CONCEPT:
        entity_type = "abstract_concept"
        confidence = "medium"
    else:
        compound_type = next(
            (
                entity_type
                for heads, entity_type in _ENTITY_HEAD_TYPES
                if singular_terminal_head in heads
            ),
            None,
        )
        if compound_type is not None:
            entity_type = compound_type
            confidence = "high"
        else:
            entity_type = "unknown"
            confidence = "low"
    return SemanticEntity(
        text_span=value.strip(),
        canonical_name=normalized or None,
        entity_type=entity_type,
        confidence=confidence,
    )
