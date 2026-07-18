"""Canonical claim extraction and application-owned proposition adjudication.

The deterministic layer is deliberately conservative. Provider output may propose richer
structures, but it never replaces the meaning of rendered text without application validation.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal, cast

from app.schemas.content import (
    AuxiliaryEvidenceRole,
    CanonicalClaim,
    CanonicalClause,
    ClaimChange,
    ClaimKind,
    ClaimModality,
    ClaimPolarity,
    ClaimSupport,
    EventIdentityMatch,
    EventQualifier,
    EvidenceCandidateMatch,
    QuantityMention,
)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_URL = re.compile(r"https?://[^\s)\]}>]+", re.I)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_QUANTITY = re.compile(
    r"(?<!\w)(?:[$€£]\s*)?\d+(?:[.,]\d+)?(?:\s*(?:%|percent|percentage points?|bps|bp|"
    r"million|billion|thousand|k|m|bn|usd|eur|gbp))?(?!\w)",
    re.I,
)
_DATE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)"
    r"(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:,?\s+\d{4})?\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.I,
)
_ENTITY = re.compile(
    r"(?<![@#\w])(?:[A-ZÁÉÍÓÖŐÚÜŰ][\wÁÉÍÓÖŐÚÜŰáéíóöőúüű'\u2019.-]*"
    r"(?:\s+(?:[A-ZÁÉÍÓÖŐÚÜŰ][\wÁÉÍÓÖŐÚÜŰáéíóöőúüű'\u2019.-]*|of|the|and)){0,4})"
)

_STOP = {
    "a",
    "an",
    "the",
    "on",
    "in",
    "at",
    "to",
    "from",
    "of",
    "for",
    "and",
    "or",
    "that",
    "this",
    "it",
    "its",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "by",
    "with",
    "as",
    "after",
    "before",
    "latest",
    "move",
    "company",
}

_PREDICATE_FORMS: dict[str, tuple[str, ...]] = {
    "decrease": ("decreased", "decrease", "reduced", "reduce", "cut", "cuts", "lowered", "lower"),
    "increase": ("increased", "increase", "raised", "raise", "hiked", "hike"),
    "exclude": ("excluded", "excludes", "exclude", "omitted", "omits", "omit", "left out"),
    "include": ("included", "includes", "include", "contains", "contain"),
    "admit": ("admitted", "admits", "admit", "acknowledged", "acknowledges"),
    "deny": ("denied", "denies", "deny", "rejected the allegation"),
    "remove": ("voted to remove", "votes to remove", "remove", "removed", "ousted", "dismissed"),
    "keep": ("voted to keep", "votes to keep", "keep", "kept", "retain", "retained"),
    "sign": ("signed", "signs", "sign", "signing"),
    "buy": (
        "acquired",
        "acquires",
        "acquire",
        "purchased",
        "purchases",
        "purchase",
        "bought",
        "buys",
        "buy",
    ),
    "sell": ("sold", "sells", "sell"),
    "resign": ("resigned", "resigns", "resign"),
    "appoint": ("was appointed", "were appointed", "appointed", "appoints", "appoint"),
    "file": ("filed", "files", "file"),
    "arrest": ("was arrested", "were arrested", "arrested", "arrests", "arrest"),
    "die": ("died", "dies", "die"),
    "merge": ("merged", "merges", "merge"),
    "approve": ("approved", "approves", "approve"),
    "reject": ("rejected", "rejects", "reject"),
    "close": ("closed", "closes", "close"),
    "open": ("opened", "opens", "open"),
    "delay": ("was delayed", "were delayed", "delayed", "delays", "delay"),
    "cancel": ("was cancelled", "were cancelled", "cancelled", "canceled", "cancels", "cancel"),
    "win": ("won", "wins", "win", "defeated", "beats", "beat"),
    "lose": ("lost", "loses", "lose", "was defeated by"),
    "launch": ("launched", "launches", "launch", "shipped", "ships", "ship"),
    "add": ("added", "adds", "add"),
    "export": ("exported", "exports", "export"),
    "require": ("required", "requires", "require", "needed", "needs"),
    "ask": ("asked whether", "asks whether", "asked", "asks"),
    "report": ("reported", "reports", "said", "says", "claimed", "claims"),
    "cause": ("caused", "causes", "cause"),
    "reach": ("has reached", "have reached", "reached", "reaches", "reach"),
    "turn": ("has turned", "have turned", "turned", "turns", "turn"),
    "show": ("shows", "showed", "show"),
    "take": ("takes", "took", "take"),
    "give": ("gives", "gave", "give"),
    "dismantle": ("dismantled", "dismantles", "dismantle"),
    "control": ("controlled", "controls", "control"),
    "found": ("founded", "founds", "found"),
}

_OPPOSITES = frozenset(
    {
        frozenset(("remove", "keep")),
        frozenset(("increase", "decrease")),
        frozenset(("include", "exclude")),
        frozenset(("admit", "deny")),
        frozenset(("win", "lose")),
        frozenset(("buy", "sell")),
        frozenset(("approve", "reject")),
        frozenset(("open", "close")),
        frozenset(("launch", "cancel")),
    }
)

_OPINION_MARKERS = re.compile(
    r"\b(?:i think|i believe|i prefer|i like|i dislike|in my view|to me|feels? like|"
    r"seems? funny|hilarious|ridiculous|beautiful|boring)\b",
    re.I,
)
_ADVICE_MARKERS = re.compile(
    r"^(?:for .{1,100},\s*)?(?:start|write down|treat|choose|record|keep|state|make|compare|"
    r"let|have|use|avoid)\b|\b(?:should|recommend|next step|bounded workflow)\b",
    re.I,
)
_REACTION_IDIOM = re.compile(
    r"(?:[\"'“”][^\"'“”]{1,80}[\"'“”]\s+is doing (?:a lot|heavy lifting)|"
    r"\bis doing (?:a lot|heavy lifting) of work\b|\bhas left the chat\b)",
    re.I,
)
_WORDING_COMMENTARY = re.compile(
    r"\b(?:moves? (?:the )?(?:sentence|phrase|wording|line) toward|"
    r"the (?:sentence|phrase|wording|line) (?:leans|moves|shifts)|"
    r"doing (?:a lot|heavy lifting|heroic amounts) of work)\b",
    re.I,
)
_DECLARATIVE_EVENT = re.compile(r"^[A-Z0-9][^!?]{1,300}[.!]?$", re.UNICODE)
_METADATA_ONLY = re.compile(
    r"^(?:see|source|link|read|email|contact)\b.*(?:https?://\S+|\S+@\S+).*$",
    re.I,
)

_SOURCE_NAME = (
    r"(?:@[A-Za-z0-9_]{1,32}|the cited source|"
    r"[A-Z][\w'\u2019.-]*(?:\s+(?:[A-Z][\w'\u2019.-]*|of|the|and)){0,4})"
)
_ACCORDING_TO = re.compile(rf"^according to\s+(?P<source>{_SOURCE_NAME})\s*,\s*(?P<body>.+)$", re.I)
_SPEAKER_VERB = re.compile(
    rf"^(?P<source>{_SOURCE_NAME})\s+(?P<verb>said|says|reported|reports|claimed|claims|"
    rf"announced|announces)\s+(?:that\s+)?(?P<body>.+)$",
    re.I,
)
_SPEAKER_DENIAL = re.compile(
    rf"^(?P<source>{_SOURCE_NAME})\s+(?P<verb>denied|denies)\s+that\s+(?P<body>.+)$",
    re.I,
)
_COLON_ATTRIBUTION = re.compile(rf"^(?P<source>{_SOURCE_NAME}):\s*[\"“]?(?P<body>.+?)[\"”]?$", re.I)

_COPULAR = re.compile(
    r"^(?P<subject>.+?)\s+(?P<copula>is|are|was|were)\s+"
    r"(?P<neg>not\s+|no longer\s+)?(?P<state>unavailable|available|unprofitable|profitable|"
    r"closed|open|active|inactive|legal|illegal|valid|invalid|complete|incomplete|bankrupt)\b(?P<rest>.*)$",
    re.I,
)
_STATE_NORMALIZATION: dict[str, tuple[str, ClaimPolarity]] = {
    "unavailable": ("available", "negative"),
    "available": ("available", "positive"),
    "unprofitable": ("profitable", "negative"),
    "profitable": ("profitable", "positive"),
    "closed": ("open", "negative"),
    "open": ("open", "positive"),
    "inactive": ("active", "negative"),
    "active": ("active", "positive"),
    "illegal": ("legal", "negative"),
    "legal": ("legal", "positive"),
    "invalid": ("valid", "negative"),
    "valid": ("valid", "positive"),
    "incomplete": ("complete", "negative"),
    "complete": ("complete", "positive"),
    "bankrupt": ("bankrupt", "positive"),
}

_NEGATIVE_PREFIX = re.compile(
    r"(?:\bnot|\bnever|\bno longer|\bdid not|\bdidn't|\bdoes not|\bdoesn't|\bhas not|"
    r"\bhasn't|\bhave not|\bhaven't|\bhad not|\bhadn't|\bcannot|\bcan't|\bfailed to|"
    r"\brefused to|\bdeclined to|\bhas yet to|\bhave yet to|\bhasn't yet|\bhaven't yet|"
    r"\bwithout)\s*$",
    re.I,
)
_NEGATIVE_AFTER = re.compile(r"^\s*(?:no\b|neither\b|nor\b)", re.I)
_POSSIBLE_PREFIX = re.compile(r"\b(?:may|might|could|possibly)\s*$", re.I)
_PROBABLE_PREFIX = re.compile(r"\b(?:will likely|is likely to|are likely to|probably)\s*$", re.I)
_REPORT_MARKER = re.compile(r"\b(?:reportedly|allegedly)\b", re.I)

_CONDITIONAL_START = re.compile(r"^\s*(?:if|unless)\b", re.I)
_COUNTERFACTUAL = re.compile(r"^\s*had\b.+\bwould\b|\bwould\b.+\bif\b", re.I)
_PLAN = re.compile(r"\b(?:plans?|planned|intends?|intended|aims?|aimed)\s+to\b", re.I)
_ATTEMPT = re.compile(r"\b(?:attempted|attempts?|tried|tries)\s+to\b", re.I)
_NEAR_EVENT = re.compile(r"\b(?:almost|nearly)\b|\b(?:was|were|is|are)\s+about\s+to\b", re.I)
_EXPECTED = re.compile(r"\b(?:is|are|was|were)\s+expected\s+to\b", re.I)
_FUTURE = re.compile(r"\bwill\b", re.I)
_ONGOING = re.compile(r"\b(?:is|are|was|were)\s+\w+ing\b", re.I)

_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "inc.", "ltd.", "e.g.", "i.e."}

_DOTTED_INITIALISM = re.compile(r"(?:\b[A-Za-z]\.){2,}$")
_HONORIFIC_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "inc.",
    "ltd.",
    "co.",
    "corp.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
    "a.i.",
    "gov.",
    "sen.",
    "rep.",
    "gen.",
    "lt.",
    "col.",
    "no.",
    "dept.",
    "jan.",
    "feb.",
    "mar.",
    "apr.",
    "jun.",
    "jul.",
    "aug.",
    "sep.",
    "sept.",
    "oct.",
    "nov.",
    "dec.",
}

_MATRIX_SCOPE_PATTERNS: tuple[tuple[re.Pattern[str], str, str, str], ...] = (
    (
        re.compile(
            r"^(?P<subject>.+?)\s+(?:is|was|are|were)\s+scheduled\s+to\s+(?P<body>.+)$", re.I
        ),
        "planned",
        "planned",
        "scheduled",
    ),
    (
        re.compile(r"^(?P<subject>.+?)\s+(?:agreed|agrees)\s+to\s+(?P<body>.+)$", re.I),
        "promised",
        "planned",
        "agreed",
    ),
    (
        re.compile(r"^(?P<subject>.+?)\s+(?:promised|promises)\s+to\s+(?P<body>.+)$", re.I),
        "promised",
        "planned",
        "promised",
    ),
    (
        re.compile(
            r"^(?P<subject>.+?)\s+(?:hoped|hopes|wanted|wants|sought|seeks)\s+to\s+(?P<body>.+)$",
            re.I,
        ),
        "desired",
        "planned",
        "desired",
    ),
    (
        re.compile(
            r"^(?P<subject>.+?)\s+(?:is|was|are|were)\s+authorized\s+to\s+(?P<body>.+)$", re.I
        ),
        "authorized",
        "planned",
        "authorized",
    ),
    (
        re.compile(
            r"^(?P<subject>.+?)\s+(?:is|was|are|were)\s+prevented\s+from\s+(?P<body>.+)$", re.I
        ),
        "prevented",
        "attempted",
        "prevented",
    ),
)

_UNCERTAINTY_SCOPE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^whether\s+(?P<body>.+?)\s+remains?\s+unclear\.?$", re.I), "not_established"),
    (re.compile(r"^(?:it\s+is|it's)\s+unclear\s+whether\s+(?P<body>.+)$", re.I), "not_established"),
    (
        re.compile(r"^there\s+is\s+no\s+evidence\s+(?:that\s+)?(?P<body>.+)$", re.I),
        "not_established",
    ),
    (
        re.compile(r"^it\s+has\s+not\s+been\s+established\s+that\s+(?P<body>.+)$", re.I),
        "not_established",
    ),
    (re.compile(r"^reports?\s+dispute\s+whether\s+(?P<body>.+)$", re.I), "questioned"),
    (re.compile(r"^the\s+claim\s+that\s+(?P<body>.+?)\s+is\s+false\.?$", re.I), "denied"),
)

_CONDITION_TAIL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?P<body>.+?)\s+only\s+if\s+(?P<condition>.+)$", re.I),
    re.compile(r"^(?P<body>.+?)\s+provided\s+that\s+(?P<condition>.+)$", re.I),
    re.compile(r"^(?P<body>.+?)\s+when\s+(?P<condition>.+)$", re.I),
    re.compile(r"^(?P<body>.+?)\s+once\s+(?P<condition>.+)$", re.I),
    re.compile(r"^(?P<body>.+?)\s+subject\s+to\s+(?P<condition>.+)$", re.I),
)

_QUALIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "seller_or_counterparty",
        re.compile(r"\bfrom\s+(?P<value>[A-Z][\w.'\u2019 -]{0,80})\b", re.I),
    ),
    (
        "instrument",
        re.compile(
            r"\b(?:in|with)\s+(?P<value>cash|bitcoin|crypto|stock|shares|a wire transfer)\b", re.I
        ),
    ),
    (
        "purpose",
        re.compile(
            r"\bto\s+(?P<value>avoid\s+[^,.;]{1,80}|prevent\s+[^,.;]{1,80}|secure\s+[^,.;]{1,80})",
            re.I,
        ),
    ),
    (
        "approval",
        re.compile(
            r"\bwith\s+(?P<value>(?:government|regulatory|board|committee)\s+approval)\b", re.I
        ),
    ),
    ("legal_status", re.compile(r"\b(?P<value>illegally|legally|unlawfully|lawfully)\b", re.I)),
    ("secrecy", re.compile(r"\b(?P<value>secretly|covertly|privately)\b", re.I)),
    ("voluntariness", re.compile(r"\b(?P<value>voluntarily|reluctantly)\b", re.I)),
    ("pressure", re.compile(r"\b(?P<value>under\s+pressure|under\s+duress)\b", re.I)),
    ("margin", re.compile(r"\b(?P<value>narrowly|by\s+a\s+narrow\s+margin)\b", re.I)),
)

_GENERIC_HEADS = {
    "bill",
    "agreement",
    "policy",
    "proposal",
    "app",
    "application",
    "deal",
    "service",
    "product",
    "plan",
    "report",
}


@dataclass(frozen=True, slots=True)
class ClaimComparison:
    support: ClaimSupport
    explanation: str
    evidence_claim: CanonicalClaim | None = None
    match: EvidenceCandidateMatch | None = None
    issue_codes: tuple[str, ...] = ()
    human_verification_required: bool = False
    supporting_claims: tuple[CanonicalClaim, ...] = ()
    partial_claims: tuple[CanonicalClaim, ...] = ()
    conflicting_claims: tuple[CanonicalClaim, ...] = ()
    attributed_claims: tuple[CanonicalClaim, ...] = ()
    uncertain_claims: tuple[CanonicalClaim, ...] = ()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {
        token.casefold()
        for token in _WORD.findall(unicodedata.normalize("NFC", value))
        if token.casefold() not in _STOP and len(token) > 1
    }


def _alignment(
    left: str | None,
    right: str | None,
    *,
    same_threshold: float = 0.8,
    partial_threshold: float = 0.4,
) -> Literal["same", "partial", "different", "unknown"]:
    if not left or not right:
        return "unknown"
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return "same" if normalize_text(left) == normalize_text(right) else "different"
    overlap = len(a & b) / max(len(a | b), 1)
    if normalize_text(left) == normalize_text(right) or overlap >= same_threshold:
        return "same"
    if overlap >= partial_threshold:
        return "partial"
    return "different"


_ALIAS_TOKENS = {
    "president",
    "prime",
    "minister",
    "parliament",
    "company",
    "corporation",
    "corp",
    "inc",
    "limited",
    "ltd",
    "the",
}


def _component_identity(
    left: str | None,
    right: str | None,
    *,
    object_component: bool,
) -> Literal["same", "alias", "qualified_same_core", "different", "uncertain"]:
    if not left or not right:
        return "uncertain"
    if normalize_text(left) == normalize_text(right):
        return "same"
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return "different"
    if left_tokens == right_tokens:
        return "alias"
    shared = left_tokens & right_tokens
    if not shared:
        return "different"
    left_only = left_tokens - right_tokens
    right_only = right_tokens - left_tokens
    if object_component:
        shared_generic = shared & _GENERIC_HEADS
        left_discriminators = left_only - _ALIAS_TOKENS
        right_discriminators = right_only - _ALIAS_TOKENS
        if shared_generic and left_discriminators and right_discriminators:
            return "different"
        if shared_generic and (left_discriminators or right_discriminators):
            return "qualified_same_core"
    extras = left_only | right_only
    if extras and extras <= _ALIAS_TOKENS:
        return "alias"
    jaccard = len(shared) / max(len(left_tokens | right_tokens), 1)
    if jaccard >= 0.8:
        return "alias"
    if object_component and (left_tokens <= right_tokens or right_tokens <= left_tokens):
        return "qualified_same_core"
    return "different"


def _qualifier_key(qualifier: EventQualifier) -> tuple[str, frozenset[str]]:
    return qualifier.qualifier_type, frozenset(_tokens(qualifier.text))


def _unsupported_qualifiers(
    draft: CanonicalClaim,
    evidence: CanonicalClaim,
) -> tuple[EventQualifier, ...]:
    evidence_keys = {_qualifier_key(item) for item in evidence.material_qualifiers}
    unsupported: list[EventQualifier] = []
    for qualifier in draft.material_qualifiers:
        qualifier_type, tokens = _qualifier_key(qualifier)
        matched = any(
            qualifier_type == evidence_type
            and bool(tokens)
            and len(tokens & evidence_tokens) / max(len(tokens | evidence_tokens), 1) >= 0.75
            for evidence_type, evidence_tokens in evidence_keys
        )
        if not matched:
            unsupported.append(qualifier)
    return tuple(unsupported)


def _embedded_assertion_status(claim: CanonicalClaim) -> str | None:
    for clause in claim.clauses:
        if clause.clause_role in {"embedded_proposition", "denied_content", "question_content"}:
            return clause.assertion_status
    return None


def _protected_period(index: int, text: str, protected: tuple[tuple[int, int], ...]) -> bool:
    if any(start <= index < end for start, end in protected):
        return True
    if (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    ):
        return True
    if text[max(0, index - 2) : index + 2].count(".") >= 2:
        return True
    token_start = max(text.rfind(" ", 0, index), text.rfind("\n", 0, index)) + 1
    token = text[token_start : index + 1]
    lowered = token.casefold().strip('"“”()[]{}')
    if lowered in _HONORIFIC_ABBREVIATIONS or _DOTTED_INITIALISM.search(token):
        return True
    # Preserve the first period in initialisms such as U.S. and J.D.
    return bool(
        index > 0
        and text[index - 1].isalpha()
        and index + 2 < len(text)
        and text[index + 1].isalpha()
        and text[index + 2] == "."
    )


def _sentence_spans(text: str) -> tuple[tuple[int, int, str], ...]:
    """Bounded scanner preserving decimals, abbreviations, URLs, emails, and ellipses."""

    protected = tuple(
        (match.start(), match.end())
        for pattern in (_URL, _EMAIL)
        for match in pattern.finditer(text)
    )
    spans: list[tuple[int, int, str]] = []
    start = 0
    length = len(text)
    index = 0
    while index < length:
        char = text[index]
        boundary = False
        if char == "\n":
            boundary = True
        elif char in "!?":
            boundary = not any(begin <= index < end for begin, end in protected)
        elif char == ".":
            boundary = not _protected_period(index, text, protected)
        if boundary:
            end = index + (0 if char == "\n" else 1)
            raw = text[start:end]
            stripped = raw.strip()
            if stripped and stripped not in {".", "..", "..."}:
                leading = len(raw) - len(raw.lstrip())
                actual_start = start + leading
                spans.append((actual_start, actual_start + len(stripped), stripped))
            start = index + 1
        index += 1
    if start < length:
        raw = text[start:]
        stripped = raw.strip()
        if stripped and stripped not in {".", "..", "..."}:
            leading = len(raw) - len(raw.lstrip())
            actual_start = start + leading
            spans.append((actual_start, actual_start + len(stripped), stripped))
    return tuple(spans)


@dataclass(frozen=True, slots=True)
class _ClauseSegment:
    exact_text: str
    semantic_text: str
    relative_start: int
    relative_end: int
    relation: Literal[
        "main",
        "relative_clause",
        "coordinated_clause",
        "discourse_linked_clause",
        "approval_modifier",
        "unknown",
    ]
    parent_segment: int | None = None
    unconsumed_material_text: tuple[str, ...] = ()


_RELATIVE_TAIL = re.compile(
    r"^(?P<main>.+?),\s*(?P<pronoun>which|who|whose|where)\s+(?P<tail>.+?)\s*[.]?$",
    re.I,
)
_LINKED_TAIL = re.compile(
    r"^(?P<main>.+?),\s*(?P<link>while|whereas|but|then|as|although|though|because|after|before|and)\s+"
    r"(?P<tail>.+?)\s*[.]?$",
    re.I,
)
_APPROVAL_MODIFIER_TAIL = re.compile(
    r"^(?P<main>.+?),\s*in\s+a\s+deal\s+approved\s+by\s+(?P<actor>.+?)\s*[.]?$",
    re.I,
)
_MATERIAL_TAIL = re.compile(
    r",\s*(?:which|who|whose|where|while|whereas|but|then|as|although|though|because|after|before|and|"
    r"in\s+a\s+deal\s+approved\s+by)\b",
    re.I,
)


def _tail_contains_additional_material_clause(tail: str) -> bool:
    """Return true when a bounded tail contains another proposition we cannot split safely."""

    if _MATERIAL_TAIL.search(tail):
        return True
    return bool(
        re.search(
            r"\b(?:and|but|while|whereas|then|although|though|because|after|before)\s+"
            r"[A-Z][^\s,;:]*\s+",
            tail,
        )
        and _canonical_predicate(tail.split(maxsplit=1)[-1])[0] is not None
    )


def _relative_semantic_text(main: str, pronoun: str, tail: str) -> str | None:
    predicate, match = _canonical_predicate(main)
    subject, obj = _subject_object(main, match)
    del predicate
    antecedent = obj or subject
    if not antecedent or len(_named_entities(antecedent)) > 1:
        return None
    clean_tail = tail.strip().rstrip(".")
    if _tail_contains_additional_material_clause(clean_tail):
        return None
    if pronoun.casefold() in {"which", "who"}:
        if re.match(r"^(?:is|are|was|were)\b", clean_tail, re.I):
            return f"{antecedent} {clean_tail}"
        tail_predicate, tail_match = _canonical_predicate(clean_tail)
        if tail_predicate is None:
            return None
        tail_subject, tail_object = _subject_object(clean_tail, tail_match)
        if tail_subject and not tail_object:
            return f"{clean_tail} {antecedent}"
        if not tail_subject:
            return f"{antecedent} {clean_tail}"
        return clean_tail
    # ``whose`` and ``where`` need antecedent/possessive resolution that the bounded fallback
    # cannot prove. Preserve the material tail as uncertain rather than inventing a relation.
    return None


def _clause_segments(body: str) -> tuple[_ClauseSegment, ...]:
    """Split bounded trailing material clauses without silently discarding their propositions."""

    stripped = body.strip()
    approval = _APPROVAL_MODIFIER_TAIL.match(stripped)
    if approval is not None:
        main = approval.group("main").strip()
        actor = approval.group("actor").strip().rstrip(".")
        main_start = body.find(main)
        tail_start = approval.start("actor")
        tail_exact_start = body.rfind(",", 0, tail_start) + 1
        tail_exact = body[tail_exact_start:].strip()
        return (
            _ClauseSegment(main, main, main_start, main_start + len(main), "main"),
            _ClauseSegment(
                tail_exact,
                f"{actor} approved the deal",
                tail_exact_start
                + len(body[tail_exact_start:])
                - len(body[tail_exact_start:].lstrip()),
                len(body),
                "approval_modifier",
                0,
            ),
        )

    relative = _RELATIVE_TAIL.match(stripped)
    if relative is not None:
        main = relative.group("main").strip()
        pronoun = relative.group("pronoun")
        tail = relative.group("tail").strip()
        main_start = body.find(main)
        pronoun_start = relative.start("pronoun")
        exact_tail = body[pronoun_start:].strip()
        semantic = _relative_semantic_text(main, pronoun, tail)
        if semantic is None:
            return (
                _ClauseSegment(main, main, main_start, main_start + len(main), "main"),
                _ClauseSegment(
                    exact_tail,
                    exact_tail,
                    pronoun_start,
                    len(body),
                    "unknown",
                    0,
                    (exact_tail,),
                ),
            )
        return (
            _ClauseSegment(main, main, main_start, main_start + len(main), "main"),
            _ClauseSegment(
                exact_tail,
                semantic,
                pronoun_start,
                len(body),
                "relative_clause",
                0,
            ),
        )

    linked = _LINKED_TAIL.match(stripped)
    if linked is not None:
        main = linked.group("main").strip()
        tail = linked.group("tail").strip().rstrip(".")
        main_start = body.find(main)
        link_start = linked.start("link")
        exact_tail = body[link_start:].strip()
        if _tail_contains_additional_material_clause(tail):
            return (
                _ClauseSegment(main, main, main_start, main_start + len(main), "main"),
                _ClauseSegment(
                    exact_tail,
                    tail,
                    link_start,
                    len(body),
                    "unknown",
                    0,
                    (exact_tail,),
                ),
            )
        relation = (
            "coordinated_clause"
            if linked.group("link").casefold() in {"while", "whereas", "but", "then", "and"}
            else "discourse_linked_clause"
        )
        # The connector itself carries discourse meaning, but the proposition body is the tail.
        if _canonical_predicate(tail)[0] is None and _copular_parts(tail) is None:
            if _REACTION_IDIOM.search(tail) or _WORDING_COMMENTARY.search(tail):
                return (
                    _ClauseSegment(main, main, main_start, main_start + len(main), "main"),
                    _ClauseSegment(
                        exact_tail,
                        tail,
                        link_start,
                        len(body),
                        "discourse_linked_clause",
                        0,
                    ),
                )
            return (
                _ClauseSegment(main, main, main_start, main_start + len(main), "main"),
                _ClauseSegment(
                    exact_tail,
                    tail,
                    link_start,
                    len(body),
                    "unknown",
                    0,
                    (exact_tail,),
                ),
            )
        return (
            _ClauseSegment(main, main, main_start, main_start + len(main), "main"),
            _ClauseSegment(
                exact_tail,
                tail,
                link_start,
                len(body),
                cast(
                    Literal[
                        "coordinated_clause",
                        "discourse_linked_clause",
                    ],
                    relation,
                ),
                0,
            ),
        )

    marker = _MATERIAL_TAIL.search(stripped)
    if marker is not None:
        main = stripped[: marker.start()].rstrip(" ,")
        tail = stripped[marker.start() + 1 :].strip()
        return (
            _ClauseSegment(main, main, 0, len(main), "main"),
            _ClauseSegment(
                tail,
                tail,
                marker.start() + 1,
                len(stripped),
                "unknown",
                0,
                (tail,),
            ),
        )
    return (_ClauseSegment(stripped, stripped, 0, len(stripped), "main"),)


def _form_pattern(form: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w]){r'\s+'.join(re.escape(x) for x in form.split())}(?![\w])", re.I)


_PREDICATE_PATTERNS = tuple(
    sorted(
        (
            (canonical, form, _form_pattern(form))
            for canonical, forms in _PREDICATE_FORMS.items()
            for form in forms
        ),
        key=lambda item: len(item[1]),
        reverse=True,
    )
)


def _looks_like_nominal_use(sentence: str, form: str, match: re.Match[str]) -> bool:
    if " " in form:
        return False
    prefix = sentence[: match.start()].rstrip()
    previous = re.search(r"([^\W_]+)$", prefix, re.UNICODE)
    return bool(previous and previous.group(1).casefold() in {"a", "an", "the", "this", "that"})


def _canonical_predicate(sentence: str) -> tuple[str | None, re.Match[str] | None]:
    best: tuple[str, str, re.Match[str]] | None = None
    for canonical, form, pattern in _PREDICATE_PATTERNS:
        match = pattern.search(sentence)
        if match is None or _looks_like_nominal_use(sentence, form, match):
            continue
        if (
            best is None
            or match.start() < best[2].start()
            or (match.start() == best[2].start() and len(form) > len(best[1]))
        ):
            best = (canonical, form, match)
    return (None, None) if best is None else (best[0], best[2])


def _clean(value: str) -> str:
    return value.strip(" \t\n,.:;!?()[]{}\"'")


def _date_values(sentence: str) -> tuple[str, ...]:
    values = []
    for match in _DATE.finditer(sentence):
        value = match.group(0).strip()
        if value == "may":
            continue
        values.append(value)
    return tuple(values)


def _named_entities(sentence: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _ENTITY.finditer(sentence):
        value = _clean(match.group(0))
        if normalize_text(value) in {"the", "a", "an", "according", "breaking", "source", "draft"}:
            continue
        values.append(value)
    return tuple(dict.fromkeys(values))


def _strip_subject_tail(subject: str) -> str:
    subject = re.sub(r"^(?:breaking|just in|update|news)\s*:?\s*", "", subject, flags=re.I)
    subject = re.sub(
        r"\b(?:did not|does not|do not|has not|have not|had not|may have|might have|"
        r"could have|was about to|were about to|is about to|are about to|"
        r"is expected to|are expected to|was expected to|were expected to|"
        r"failed to|refused to|declined to|has yet to|have yet to|will likely|"
        r"is likely to|are likely to|plans to|plan to|attempted to|tries to|tried to|"
        r"did|does|do|has|have|had|was|were|is|are|may|might|could|would|will|can|"
        r"probably|almost|nearly|reportedly|allegedly)\s*$",
        "",
        subject,
        flags=re.I,
    ).strip()
    return subject


def _scope_clauses(
    sentence: str,
    body: str,
    *,
    claim_prefix: str,
) -> tuple[str, tuple[CanonicalClause, ...], str | None]:
    """Return the embedded proposition body without promoting its assertion scope."""

    for pattern, assertion_status, event_status, matrix_predicate in _MATRIX_SCOPE_PATTERNS:
        match = pattern.match(body.strip())
        if match is None:
            continue
        subject = _clean(match.group("subject"))
        embedded = match.group("body").strip().rstrip(".")
        effective = f"{subject} {embedded}"
        embedded_predicate, embedded_match = _canonical_predicate(effective)
        embedded_subject, embedded_object = _subject_object(effective, embedded_match)
        clauses = (
            CanonicalClause(
                clause_id=f"{claim_prefix}-M",
                exact_text=sentence,
                clause_role="main_assertion",
                assertion_status="asserted",
                subject=subject,
                predicate=matrix_predicate,
                object=embedded,
            ),
            CanonicalClause(
                clause_id=f"{claim_prefix}-E",
                exact_text=embedded,
                clause_role="embedded_proposition",
                assertion_status=assertion_status,
                subject=embedded_subject,
                predicate=embedded_predicate,
                object=embedded_object,
                parent_clause_id=f"{claim_prefix}-M",
            ),
        )
        return effective, clauses, event_status

    for pattern, assertion_status in _UNCERTAINTY_SCOPE_PATTERNS:
        match = pattern.match(body.strip())
        if match is None:
            continue
        embedded = match.group("body").strip().rstrip(".")
        embedded_predicate, embedded_match = _canonical_predicate(embedded)
        embedded_subject, embedded_object = _subject_object(embedded, embedded_match)
        role = "denied_content" if assertion_status == "denied" else "embedded_proposition"
        clauses = (
            CanonicalClause(
                clause_id=f"{claim_prefix}-M",
                exact_text=sentence,
                clause_role="main_assertion",
                assertion_status="asserted",
                predicate="deny" if assertion_status == "denied" else "establish",
                object=embedded,
            ),
            CanonicalClause(
                clause_id=f"{claim_prefix}-E",
                exact_text=embedded,
                clause_role=role,
                assertion_status=assertion_status,
                subject=embedded_subject,
                predicate=embedded_predicate,
                object=embedded_object,
                parent_clause_id=f"{claim_prefix}-M",
            ),
        )
        return embedded, clauses, "unknown"

    for pattern in _CONDITION_TAIL_PATTERNS:
        match = pattern.match(body.strip().rstrip("."))
        if match is None:
            continue
        embedded = match.group("body").strip()
        condition = match.group("condition").strip()
        embedded_predicate, embedded_match = _canonical_predicate(embedded)
        embedded_subject, embedded_object = _subject_object(embedded, embedded_match)
        condition_predicate, condition_match = _canonical_predicate(condition)
        condition_subject, condition_object = _subject_object(condition, condition_match)
        clauses = (
            CanonicalClause(
                clause_id=f"{claim_prefix}-E",
                exact_text=embedded,
                clause_role="embedded_proposition",
                assertion_status="conditional",
                subject=embedded_subject,
                predicate=embedded_predicate,
                object=embedded_object,
            ),
            CanonicalClause(
                clause_id=f"{claim_prefix}-C",
                exact_text=condition,
                clause_role="condition",
                assertion_status="conditional",
                subject=condition_subject,
                predicate=condition_predicate,
                object=condition_object,
                parent_clause_id=f"{claim_prefix}-E",
            ),
        )
        return embedded, clauses, "conditional"
    return body, (), None


def _extract_material_qualifiers(body: str) -> tuple[EventQualifier, ...]:
    qualifiers: list[EventQualifier] = []
    for qualifier_type, pattern in _QUALIFIER_PATTERNS:
        for match in pattern.finditer(body):
            text = _clean(match.group(0))
            qualifiers.append(EventQualifier(qualifier_type=qualifier_type, text=text))
    unique: dict[tuple[str, str], EventQualifier] = {}
    for item in qualifiers:
        unique[(item.qualifier_type, normalize_text(item.text))] = item
    return tuple(unique.values())


def _strip_qualifiers(value: str | None, qualifiers: Sequence[EventQualifier]) -> str | None:
    if not value:
        return value
    cleaned = value
    for qualifier in qualifiers:
        cleaned = re.sub(re.escape(qualifier.text), " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    return cleaned or None


def _subject_object(sentence: str, match: re.Match[str] | None) -> tuple[str | None, str | None]:
    if match is None:
        return None, None
    subject = sentence[: match.start()].strip(" \t\n,.:;-—") or None
    obj = sentence[match.end() :].strip(" \t\n,.:;-—") or None
    if subject:
        subject = _strip_subject_tail(subject)
    if obj:
        obj = re.sub(r"^(?:to|by)\s+", "", obj, flags=re.I).strip()
        obj = re.split(
            r",\s+(?:in|which|while|whereas|as|then|but|the latest|latest)\b",
            obj,
            maxsplit=1,
            flags=re.I,
        )[0].strip()
    if (
        subject
        and obj
        and re.search(r"\b(?:was|were|is|are)\s*$", sentence[: match.start()], re.I)
        and re.match(r"^by\s+", sentence[match.end() :].lstrip(), re.I)
    ):
        actor = re.sub(r"^by\s+", "", sentence[match.end() :].strip(), flags=re.I).strip(" .,!?:;")
        passive_object = re.sub(r"\b(?:was|were|is|are)\s*$", "", subject, flags=re.I).strip()
        return actor or None, passive_object or None
    return subject or None, obj or None


def _copular_parts(sentence: str) -> tuple[str, str, ClaimPolarity] | None:
    match = _COPULAR.match(sentence.strip())
    if match is None:
        return None
    state, base = _STATE_NORMALIZATION[match.group("state").casefold()]
    explicit_neg = bool(match.group("neg"))
    polarity: ClaimPolarity = base
    if explicit_neg:
        polarity = "negative" if base == "positive" else "positive"
    return _clean(match.group("subject")), state, polarity


def _parse_attribution(
    sentence: str,
) -> tuple[
    str | None,
    str,
    Literal["assertion", "report", "claim", "denial", "quotation", "unknown"],
    Literal["said", "reported", "claimed", "announced", "denied", "acknowledged", "unknown"] | None,
]:
    match = _ACCORDING_TO.match(sentence)
    if match is not None:
        return _clean(match.group("source")), match.group("body").strip(), "report", "reported"
    match = _SPEAKER_DENIAL.match(sentence)
    if match is not None:
        return _clean(match.group("source")), match.group("body").strip(), "denial", "denied"
    match = _SPEAKER_VERB.match(sentence)
    if match is not None:
        raw = match.group("verb").casefold()
        act = {
            "said": "said",
            "says": "said",
            "reported": "reported",
            "reports": "reported",
            "claimed": "claimed",
            "claims": "claimed",
            "announced": "announced",
            "announces": "announced",
        }[raw]
        speech = "claim" if act == "claimed" else "report"
        return _clean(match.group("source")), match.group("body").strip(), speech, act  # type: ignore[return-value]
    match = _COLON_ATTRIBUTION.match(sentence)
    if match is not None:
        source = _clean(match.group("source"))
        if normalize_text(source) not in {"breaking", "just in", "update", "news"}:
            return source, match.group("body").strip(), "quotation", "said"
    return None, sentence, "assertion", None


def _polarity(sentence: str, match: re.Match[str] | None) -> ClaimPolarity:
    if match is None:
        return "unknown"
    prefix = sentence[: match.start()].rstrip()
    suffix = sentence[match.end() :]
    if _NEGATIVE_PREFIX.search(prefix) or _NEGATIVE_AFTER.search(suffix):
        return "negative"
    if re.search(r"\bneither\b", prefix, re.I) and re.search(r"\bnor\b", sentence, re.I):
        return "negative"
    return "positive"


def _semantic_dimensions(
    sentence: str,
    body: str,
    match: re.Match[str] | None,
    outer_speech: str,
    attribution_act: str | None,
) -> tuple[str, str, str, ClaimModality]:
    speech_act = "question" if sentence.rstrip().endswith("?") else outer_speech
    prefix = body[: match.start()].rstrip() if match is not None else body
    if _REPORT_MARKER.search(prefix):
        epistemic = "alleged" if re.search(r"\ballegedly\b", prefix, re.I) else "reported"
    elif _POSSIBLE_PREFIX.search(prefix) or re.search(
        r"\b(?:may|might|could|possibly)\b", body, re.I
    ):
        epistemic = "possible"
    elif _PROBABLE_PREFIX.search(prefix) or re.search(r"\b(?:likely|probably)\b", body, re.I):
        epistemic = "probable"
    else:
        epistemic = "asserted"

    if _COUNTERFACTUAL.search(body):
        event_status = "counterfactual"
    elif _CONDITIONAL_START.search(body) or re.search(r"\bunless\b", body, re.I):
        event_status = "conditional"
    elif _ATTEMPT.search(prefix):
        event_status = "attempted"
    elif _NEAR_EVENT.search(prefix):
        event_status = "near_event"
    elif _PLAN.search(prefix):
        event_status = "planned"
    elif _EXPECTED.search(prefix) or _FUTURE.search(prefix):
        event_status = "future"
    elif _ONGOING.search(body):
        event_status = "ongoing"
    elif speech_act == "question":
        event_status = "unknown"
    else:
        event_status = "completed"

    legacy: ClaimModality
    if attribution_act == "denied":
        legacy = "denied"
    elif outer_speech in {"report", "claim", "quotation"} or epistemic in {"reported", "alleged"}:
        legacy = "reported"
    elif epistemic == "possible":
        legacy = "possible"
    elif epistemic == "probable":
        legacy = "probable"
    else:
        legacy = "asserted"
    return speech_act, epistemic, event_status, legacy


def _claim_kind(
    sentence: str,
    *,
    predicate: str | None,
    attribution: str | None,
    copular: bool,
    speech_act: str,
) -> ClaimKind:
    if (
        _REACTION_IDIOM.search(sentence)
        or _WORDING_COMMENTARY.search(sentence)
        or _ADVICE_MARKERS.search(sentence)
    ):
        return "opinion"
    if _OPINION_MARKERS.search(sentence):
        return "interpretation" if predicate or copular else "opinion"
    if speech_act == "question":
        return "uncertainty"
    if attribution is not None or _REPORT_MARKER.search(sentence):
        return "attributed_claim"
    if predicate or copular or _QUANTITY.search(sentence) or _DATE.search(sentence):
        return "factual"
    if re.match(r"^\s*i\s+(?:like|love|hate|prefer|enjoy|dislike)\b", sentence, re.I):
        return "opinion"
    if _DECLARATIVE_EVENT.match(sentence) and (
        _named_entities(sentence)
        or re.search(
            r"\b(?:company|service|product|board|parliament|government|team|court)\b",
            sentence,
            re.I,
        )
    ):
        return "material_claim_unknown"
    return "opinion"


def _normalized_quantity(value: str) -> str:
    raw = normalize_text(value).replace(",", "")
    currency = ""
    if raw and raw[0] in "$€£":
        currency, raw = raw[0], raw[1:].strip()
    match = re.fullmatch(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>%|percent|percentage point|percentage points|"
        r"million|billion|thousand|k|m|bn|bps|bp|usd|eur|gbp)?",
        raw,
    )
    if match is None:
        return normalize_text(value)
    try:
        number = Decimal(match.group("num"))
    except InvalidOperation:
        return normalize_text(value)
    number_text = format(number, "f")
    if "." in number_text:
        number_text = number_text.rstrip("0").rstrip(".")
    unit = match.group("unit") or ""
    unit = {
        "percent": "%",
        "million": "m",
        "billion": "bn",
        "thousand": "k",
        "percentage point": "percentage points",
        "bp": "bps",
    }.get(unit, unit)
    return f"{currency}{number_text}:{unit}"


def _normalized_quantities(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_normalized_quantity(value) for value in values)


def _quantity_mentions(body: str, predicate: str | None) -> tuple[QuantityMention, ...]:
    matches = list(_QUANTITY.finditer(body))
    mentions: list[QuantityMention] = []
    for index, match in enumerate(matches):
        text = match.group(0).strip()
        normalized = _normalized_quantity(text)
        prefix = body[max(0, match.start() - 16) : match.start()].casefold()
        role: Literal[
            "start_value",
            "end_value",
            "delta",
            "total",
            "rate",
            "minimum",
            "maximum",
            "count",
            "unknown",
        ] = "unknown"
        if re.search(r"\bfrom\s*$", prefix):
            role = "start_value"
        elif re.search(r"\bto\s*$", prefix):
            role = "end_value"
        elif re.search(r"\bby\s*$", prefix):
            role = "delta"
        elif re.search(r"\bat\s*$", prefix):
            role = "total"
        elif re.search(r"\b(?:minimum|min\.?|at least)\s*$", prefix):
            role = "minimum"
        elif re.search(r"\b(?:maximum|max\.?|at most)\s*$", prefix):
            role = "maximum"
        elif text.casefold().endswith(("%", "percent", "bps", "bp")):
            role = "rate"
        elif predicate in {"increase", "decrease"} and len(matches) == 1:
            role = "end_value"
        elif len(matches) == 1:
            role = "total"
        elif index == len(matches) - 1 and predicate in {"increase", "decrease"}:
            role = "end_value"
        unit = normalized.partition(":")[2] or None
        mentions.append(
            QuantityMention(
                text=text,
                normalized_value=normalized,
                role=role,
                unit=unit,
            )
        )
    return tuple(mentions)


def _extract_segment_claim(
    segment: _ClauseSegment,
    *,
    claim_id: str,
    sentence: str,
    sentence_start: int,
    body_offset: int,
    inherited_attribution: str | None,
    inherited_speech: Literal["assertion", "report", "claim", "denial", "quotation", "unknown"],
    inherited_attribution_act: Literal[
        "said", "reported", "claimed", "announced", "denied", "acknowledged", "unknown"
    ]
    | None,
    parent_claim_id: str | None,
) -> CanonicalClaim:
    semantic_text = segment.semantic_text.strip()
    if segment.unconsumed_material_text:
        return CanonicalClaim(
            claim_id=claim_id,
            exact_text=segment.exact_text.strip(),
            kind="material_claim_unknown",
            named_entities=_named_entities(segment.exact_text),
            extraction_provenance="deterministic",
            structure_validated=True,
            clause_relation=segment.relation,
            parent_claim_id=parent_claim_id,
            unconsumed_material_text=segment.unconsumed_material_text,
            start=sentence_start + body_offset + segment.relative_start,
            end=sentence_start + body_offset + segment.relative_end,
        )

    attribution, body, outer_speech, attribution_act = _parse_attribution(semantic_text)
    if attribution is None and inherited_attribution is not None:
        attribution = inherited_attribution
        outer_speech = inherited_speech
        attribution_act = inherited_attribution_act
    body, clauses, scoped_event_status = _scope_clauses(
        sentence,
        body,
        claim_prefix=claim_id,
    )
    qualifiers = _extract_material_qualifiers(body)
    copular = _copular_parts(body)
    subject: str | None
    predicate: str | None
    obj: str | None
    if copular is not None:
        subject, state, polarity = copular
        predicate = "be"
        obj = state
        predicate_match: re.Match[str] | None = None
    else:
        predicate, predicate_match = _canonical_predicate(body)
        subject, obj = _subject_object(body, predicate_match)
        subject = _strip_qualifiers(subject, qualifiers)
        obj = _strip_qualifiers(obj, qualifiers)
        polarity = _polarity(body, predicate_match)
    speech_act, epistemic, event_status, legacy_modality = _semantic_dimensions(
        sentence, body, predicate_match, outer_speech, attribution_act
    )
    if scoped_event_status is not None:
        event_status = scoped_event_status
    if clauses and clauses[-1].assertion_status in {"denied", "not_established", "questioned"}:
        epistemic = "unknown"
        legacy_modality = "unknown"
    kind = _claim_kind(
        semantic_text,
        predicate=predicate,
        attribution=attribution,
        copular=copular is not None,
        speech_act=speech_act,
    )
    if kind in {"opinion", "uncertainty"} and predicate is None and copular is None:
        polarity = "unknown"
        if kind == "opinion":
            legacy_modality = "unknown"
            epistemic = "unknown"
            event_status = "unknown"
    quantity_mentions = _quantity_mentions(body, predicate)
    relation_clause: tuple[CanonicalClause, ...] = ()
    if segment.relation != "main" and not clauses:
        relation_clause = (
            CanonicalClause(
                clause_id=f"{claim_id}-R",
                exact_text=segment.exact_text.strip(),
                clause_role="main_assertion" if segment.relation != "unknown" else "unknown",
                assertion_status="asserted" if segment.relation != "unknown" else "unknown",
                subject=subject,
                predicate=predicate,
                object=obj,
                parent_clause_id=parent_claim_id,
                relation=(
                    segment.relation
                    if segment.relation
                    in {
                        "relative_clause",
                        "coordinated_clause",
                        "discourse_linked_clause",
                        "approval_modifier",
                    }
                    else "unknown"
                ),
            ),
        )
    return CanonicalClaim(
        claim_id=claim_id,
        exact_text=segment.exact_text.strip(),
        kind=kind,
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity=polarity,
        modality=legacy_modality,
        quantities=tuple(item.text for item in quantity_mentions),
        quantity_mentions=quantity_mentions,
        dates=_date_values(body),
        named_entities=_named_entities(body),
        attribution_source=attribution,
        speech_act=speech_act,
        epistemic_modality=epistemic,
        event_status=event_status,
        attribution_act=attribution_act,
        extraction_provenance="deterministic",
        structure_validated=True,
        clauses=tuple((*clauses, *relation_clause)),
        material_qualifiers=qualifiers,
        clause_relation=segment.relation,
        parent_claim_id=parent_claim_id,
        start=sentence_start + body_offset + segment.relative_start,
        end=sentence_start + body_offset + segment.relative_end,
    )


def extract_claims(text: str, *, prefix: str = "C") -> tuple[CanonicalClaim, ...]:
    claims: list[CanonicalClaim] = []
    claim_counter = 0
    for sentence_start, _sentence_end, sentence in _sentence_spans(text):
        if _METADATA_ONLY.match(sentence.strip()):
            continue
        attribution, body, outer_speech, attribution_act = _parse_attribution(sentence)
        body_offset = sentence.find(body)
        if body_offset < 0:
            body_offset = 0
        segments = _clause_segments(body)
        parent_ids: dict[int, str] = {}
        for segment_index, segment in enumerate(segments):
            claim_counter += 1
            claim_id = f"{prefix}{claim_counter}"
            parent_claim_id = (
                parent_ids.get(segment.parent_segment)
                if segment.parent_segment is not None
                else None
            )
            exact_segment = segment
            if segment_index == 0 and attribution is not None:
                # Preserve the exact attribution prefix in the main claim span.
                exact_end = body_offset + segment.relative_end
                exact_segment = _ClauseSegment(
                    exact_text=sentence[:exact_end].strip(),
                    semantic_text=segment.semantic_text,
                    relative_start=-body_offset,
                    relative_end=segment.relative_end,
                    relation=segment.relation,
                    parent_segment=segment.parent_segment,
                    unconsumed_material_text=segment.unconsumed_material_text,
                )
            claim = _extract_segment_claim(
                exact_segment,
                claim_id=claim_id,
                sentence=sentence,
                sentence_start=sentence_start,
                body_offset=body_offset,
                inherited_attribution=attribution,
                inherited_speech=outer_speech,
                inherited_attribution_act=attribution_act,
                parent_claim_id=parent_claim_id,
            )
            claims.append(claim)
            parent_ids[segment_index] = claim_id
    return tuple(claims)


def extract_urls(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(0).rstrip(".,;:!?") for match in _URL.finditer(text)))


def _opposite(left: str | None, right: str | None) -> bool:
    return bool(left and right and frozenset((left, right)) in _OPPOSITES)


def _tuple_alignment(
    draft: tuple[str, ...], evidence: tuple[str, ...], *, quantities: bool = False
) -> Literal["same", "different", "missing", "not_applicable"]:
    if not draft and not evidence:
        return "not_applicable"
    if draft and not evidence:
        return "missing"
    if not draft and evidence:
        return "not_applicable"
    left = _normalized_quantities(draft) if quantities else tuple(map(normalize_text, draft))
    right = _normalized_quantities(evidence) if quantities else tuple(map(normalize_text, evidence))
    return "same" if left == right else "different"


def _quantity_alignment(
    draft: CanonicalClaim,
    evidence: CanonicalClaim,
) -> Literal["same", "different", "missing", "not_applicable"]:
    """Compare only quantity roles asserted by the draft.

    Evidence may contain additional established detail (for example a start value or delta) that a
    concise draft omits. The reverse is not true: a draft-added role is missing support, and a
    different value for the same role is a conflict.
    """

    draft_mentions = draft.quantity_mentions
    evidence_mentions = evidence.quantity_mentions
    if not draft_mentions and not evidence_mentions:
        return "not_applicable"
    if not draft_mentions:
        return "not_applicable"
    if not evidence_mentions:
        return "missing"
    by_role: dict[str, set[str]] = {}
    for mention in evidence_mentions:
        by_role.setdefault(mention.role, set()).add(mention.normalized_value)
    missing = False
    for mention in draft_mentions:
        evidence_values = by_role.get(mention.role)
        if evidence_values is None:
            # Unknown roles may align conservatively by exact normalized value only.
            if mention.role == "unknown" and any(
                item.normalized_value == mention.normalized_value for item in evidence_mentions
            ):
                continue
            missing = True
            continue
        if mention.normalized_value not in evidence_values:
            return "different"
    return "missing" if missing else "same"


def _epistemic_alignment(
    draft: CanonicalClaim, evidence: CanonicalClaim
) -> Literal["compatible", "escalated", "different", "unknown"]:
    d: str = draft.epistemic_modality
    e: str = evidence.epistemic_modality
    if "unknown" in {d, e}:
        # Legacy artifacts fall back to the old field.
        d = (
            draft.modality
            if draft.modality in {"asserted", "probable", "possible", "reported"}
            else "unknown"
        )
        e = (
            evidence.modality
            if evidence.modality in {"asserted", "probable", "possible", "reported"}
            else "unknown"
        )
    if "unknown" in {d, e}:
        return "unknown"
    if e == "asserted" and d in {"asserted", "probable", "possible"}:
        return "compatible"
    if e == "probable" and d in {"probable", "possible"}:
        return "compatible"
    if e == "possible" and d == "possible":
        return "compatible"
    if e in {"reported", "alleged"} and d in {"reported", "alleged"}:
        return "compatible"
    if (e, d) in {
        ("possible", "probable"),
        ("possible", "asserted"),
        ("probable", "asserted"),
        ("reported", "asserted"),
        ("alleged", "asserted"),
    }:
        return "escalated"
    return "different"


def _event_status_alignment(
    draft: CanonicalClaim, evidence: CanonicalClaim
) -> Literal["compatible", "escalated", "different", "unknown"]:
    d, e = draft.event_status, evidence.event_status
    if "unknown" in {d, e}:
        return "unknown"
    if d == e:
        return "compatible"
    if e == "completed" and d in {"unknown", "completed"}:
        return "compatible"
    if (
        e in {"future", "planned", "attempted", "near_event", "conditional", "counterfactual"}
        and d == "completed"
    ):
        return "escalated"
    return "different"


def _speech_act_alignment(
    draft: CanonicalClaim, evidence: CanonicalClaim
) -> Literal["compatible", "question_only", "different", "unknown"]:
    if evidence.speech_act == "question":
        return "question_only" if draft.speech_act != "question" else "compatible"
    if draft.speech_act == evidence.speech_act:
        return "compatible"
    if evidence.speech_act == "assertion" and draft.speech_act in {
        "report",
        "claim",
        "quotation",
    }:
        return "compatible"
    if (
        evidence.speech_act in {"report", "claim", "quotation", "denial"}
        and draft.speech_act == "assertion"
    ):
        return "different"
    if "unknown" in {draft.speech_act, evidence.speech_act}:
        return "unknown"
    return "different"


def _clause_scope_alignment(
    draft: CanonicalClaim,
    evidence: CanonicalClaim,
) -> Literal["compatible", "not_asserted", "different", "unknown"]:
    evidence_status = _embedded_assertion_status(evidence)
    draft_status = _embedded_assertion_status(draft)
    if evidence_status is None and draft_status is None:
        return "compatible"
    if evidence_status in {
        "questioned",
        "not_established",
        "denied",
        "planned",
        "promised",
        "desired",
        "authorized",
        "prevented",
        "conditional",
        "possible",
    }:
        if draft_status == evidence_status:
            return "compatible"
        if draft_status is None and draft.event_status == "completed":
            return "not_asserted"
        return "different"
    if evidence_status == "asserted" and draft_status in {None, "asserted"}:
        return "compatible"
    return "unknown"


def _attribution_alignment(
    draft: CanonicalClaim, evidence: CanonicalClaim
) -> Literal["same", "lost", "different", "not_applicable"]:
    if evidence.attribution_source and not draft.attribution_source:
        return "lost"
    if not evidence.attribution_source and not draft.attribution_source:
        return "not_applicable"
    if draft.attribution_source and not evidence.attribution_source:
        # Adding an attribution weakens an otherwise objective assertion. The source identity is
        # validated by the evidence packet boundary rather than treated as a factual conflict.
        return "not_applicable"
    if draft.attribution_source and evidence.attribution_source:
        same_source = (
            _alignment(draft.attribution_source, evidence.attribution_source, same_threshold=0.9)
            == "same"
        )
        same_act = (
            draft.attribution_act == evidence.attribution_act
            or "unknown" in {draft.attribution_act, evidence.attribution_act}
            or draft.attribution_act is None
            or evidence.attribution_act is None
        )
        return "same" if same_source and same_act else "different"
    return "different"


def align_claim_to_evidence(
    claim: CanonicalClaim, evidence: CanonicalClaim
) -> EvidenceCandidateMatch:
    speech_relation = bool(
        claim.predicate in {"admit", "report", "deny"}
        and evidence.attribution_source
        and _alignment(claim.subject, evidence.attribution_source, same_threshold=0.85) == "same"
        and _alignment(
            claim.object,
            evidence.object or evidence.exact_text,
            same_threshold=0.6,
            partial_threshold=0.25,
        )
        in {"same", "partial"}
    )
    if speech_relation:
        actor: Literal["same", "partial", "different", "unknown"] = "same"
        obj: Literal["same", "partial", "different", "unknown"] = "same"
        evidence_speech = "deny" if evidence.attribution_act == "denied" else "report"
        if claim.predicate == evidence_speech or (
            claim.predicate == "admit" and evidence_speech == "report"
        ):
            predicate: Literal["same", "opposite", "different", "unknown"] = "same"
        elif claim.predicate == "admit" and evidence_speech == "deny":
            predicate = "opposite"
        else:
            predicate = "different"
    else:
        actor_identity = _component_identity(
            claim.subject,
            evidence.subject,
            object_component=False,
        )
        object_identity = _component_identity(
            claim.object,
            evidence.object,
            object_component=True,
        )
        actor = cast(
            Literal["same", "partial", "different", "unknown"],
            {
                "same": "same",
                "alias": "same",
                "qualified_same_core": "partial",
                "different": "different",
                "uncertain": "unknown",
            }[actor_identity],
        )
        obj = cast(
            Literal["same", "partial", "different", "unknown"],
            {
                "same": "same",
                "alias": "same",
                "qualified_same_core": "partial",
                "different": "different",
                "uncertain": "unknown",
            }[object_identity],
        )
        if claim.predicate and evidence.predicate:
            if claim.predicate == evidence.predicate:
                predicate = "same"
            elif _opposite(claim.predicate, evidence.predicate):
                predicate = "opposite"
            else:
                predicate = "different"
        else:
            predicate = "unknown"

    quantity_alignment = _quantity_alignment(claim, evidence)
    date_alignment = _tuple_alignment(claim.dates, evidence.dates)

    def object_is_quantity_only(value: str | None, quantities: tuple[str, ...]) -> bool:
        if not value or not quantities:
            return False
        residual = normalize_text(value)
        for quantity in quantities:
            residual = residual.replace(normalize_text(quantity), " ")
        residual = re.sub(r"\b(?:to|from|at|of|the|a|an)\b", " ", residual)
        return not _tokens(residual)

    quantity_equivalent_object = bool(
        quantity_alignment == "same"
        and object_is_quantity_only(claim.object, claim.quantities)
        and object_is_quantity_only(evidence.object, evidence.quantities)
    )
    directly_competing_quantity_event = bool(
        predicate == "opposite"
        and actor in {"same", "partial"}
        and claim.quantities
        and evidence.quantities
        and object_is_quantity_only(claim.object, claim.quantities)
        and object_is_quantity_only(evidence.object, evidence.quantities)
    )
    if quantity_equivalent_object:
        obj = "same"
        if not speech_relation:
            object_identity = "same"
    if directly_competing_quantity_event and not speech_relation:
        obj = "same"
        object_identity = "same"

    exact = normalize_text(claim.exact_text) == normalize_text(evidence.exact_text)
    reversed_direction = bool(
        claim.subject
        and claim.object
        and evidence.subject
        and evidence.object
        and predicate == "same"
        and _alignment(claim.subject, evidence.object, same_threshold=0.75) == "same"
        and _alignment(claim.object, evidence.subject, same_threshold=0.75) == "same"
    )
    same_or_competing = predicate in {"same", "opposite"}
    relation_overlap = actor in {"same", "partial"} and obj in {"same", "partial"}
    one_side = same_or_competing and (
        (actor in {"same", "partial"} and obj == "unknown")
        or (obj in {"same", "partial"} and actor == "unknown")
    )
    if (
        exact
        or reversed_direction
        or directly_competing_quantity_event
        or (same_or_competing and relation_overlap)
    ):
        event: Literal["same", "partial", "unrelated"] = "same"
    elif (same_or_competing and obj == "same" and actor == "different") or (
        predicate == "same"
        and actor == "same"
        and obj == "unknown"
        and _attribution_alignment(claim, evidence) == "different"
    ):
        event = "same"
    elif one_side or (
        predicate == "same" and (actor in {"same", "partial"} or obj in {"same", "partial"})
    ):
        event = "partial"
    else:
        event = "unrelated"

    if speech_relation:
        actor_identity_value: Literal["same", "alias", "different", "uncertain"] = "same"
        object_identity_value: Literal[
            "same", "alias", "qualified_same_core", "different", "uncertain"
        ] = "same"
    else:
        actor_identity_value = (
            "alias" if actor_identity == "qualified_same_core" else actor_identity
        )
        object_identity_value = object_identity
    material_extra_details = tuple(item.text for item in _unsupported_qualifiers(claim, evidence))
    core_event_same = bool(
        predicate in {"same", "opposite"}
        and actor_identity_value in {"same", "alias"}
        and object_identity_value in {"same", "alias", "qualified_same_core"}
        and object_identity_value != "different"
    )
    event_identity = EventIdentityMatch(
        actor_identity=actor_identity_value,
        predicate_identity={
            "same": "same",
            "opposite": "opposite",
            "different": "different",
            "unknown": "uncertain",
        }[predicate],
        object_identity=object_identity_value,
        core_event_same=core_event_same,
        material_extra_details=material_extra_details,
    )

    # Similar generic heads with different discriminators are not the same event.
    if (
        not core_event_same
        and not exact
        and not reversed_direction
        and object_identity_value == "different"
    ):
        event = "unrelated" if predicate in {"same", "opposite"} else event

    attribution_alignment = _attribution_alignment(claim, evidence)
    speech_act_alignment = _speech_act_alignment(claim, evidence)
    if speech_relation:
        attribution_alignment = "same"
        speech_act_alignment = "compatible"
    return EvidenceCandidateMatch(
        evidence_claim_id=evidence.claim_id,
        event_alignment=event,
        actor_alignment=actor,
        predicate_alignment=predicate,
        object_alignment=obj,
        quantity_alignment=quantity_alignment,
        date_alignment=date_alignment,
        modality_alignment=_epistemic_alignment(claim, evidence),
        attribution_alignment=attribution_alignment,
        speech_act_alignment=speech_act_alignment,
        event_status_alignment=_event_status_alignment(claim, evidence),
        clause_scope_alignment=_clause_scope_alignment(claim, evidence),
        event_identity=event_identity,
    )


def contradiction_fields(
    claim: CanonicalClaim,
    evidence: CanonicalClaim,
    match: EvidenceCandidateMatch | None = None,
) -> tuple[str, ...]:
    match = match or align_claim_to_evidence(claim, evidence)
    if match.event_alignment == "unrelated":
        return ()
    fields: list[str] = []
    if match.predicate_alignment == "opposite":
        fields.append("predicate")
    if (
        match.predicate_alignment == "same"
        and claim.polarity != evidence.polarity
        and "unknown" not in {claim.polarity, evidence.polarity}
    ):
        fields.append("polarity")
    if match.event_alignment == "same" and match.quantity_alignment == "different":
        fields.append("quantities")
    if match.event_alignment == "same" and match.date_alignment == "different":
        fields.append("dates")
    if match.attribution_alignment == "different":
        fields.append("attribution")
    if match.modality_alignment == "escalated":
        fields.append("modality")
    if match.event_status_alignment == "escalated":
        fields.append("event_status")
    if match.speech_act_alignment == "question_only":
        fields.append("question_only")
    if match.clause_scope_alignment == "not_asserted":
        fields.append("embedded_event_not_asserted")
    reversed_direction = bool(
        claim.subject
        and claim.object
        and evidence.subject
        and evidence.object
        and match.predicate_alignment == "same"
        and _alignment(claim.subject, evidence.object, same_threshold=0.75) == "same"
        and _alignment(claim.object, evidence.subject, same_threshold=0.75) == "same"
    )
    if reversed_direction:
        fields.append("source_target_reversed")
    elif (
        match.event_alignment != "unrelated"
        and match.actor_alignment == "different"
        and match.predicate_alignment in {"same", "opposite"}
        and match.object_alignment == "same"
    ):
        fields.append("actor")
    return tuple(dict.fromkeys(fields))


def _candidate_score(match: EvidenceCandidateMatch) -> int:
    score = {"same": 30, "partial": 12, "unrelated": -100}[match.event_alignment]
    score += {"same": 7, "partial": 3, "different": -8, "unknown": 0}[match.actor_alignment]
    score += {"same": 12, "opposite": 10, "different": -10, "unknown": 0}[match.predicate_alignment]
    score += {"same": 7, "partial": 3, "different": -8, "unknown": 0}[match.object_alignment]
    score += {"same": 4, "different": -5, "missing": -2, "not_applicable": 0}[
        match.quantity_alignment
    ]
    score += {"same": 4, "different": -5, "missing": -2, "not_applicable": 0}[match.date_alignment]
    score += {"compatible": 4, "escalated": -7, "different": -4, "unknown": 0}[
        match.modality_alignment
    ]
    score += {"compatible": 4, "escalated": -8, "different": -4, "unknown": 0}[
        match.event_status_alignment
    ]
    score += {"compatible": 2, "question_only": -10, "different": -4, "unknown": 0}[
        match.speech_act_alignment
    ]
    score += {"compatible": 3, "not_asserted": -12, "different": -6, "unknown": 0}[
        match.clause_scope_alignment
    ]
    score += {"same": 3, "lost": -6, "different": -6, "not_applicable": 0}[
        match.attribution_alignment
    ]
    return score


def compare_claim_to_evidence(
    claim: CanonicalClaim, evidence_claims: Sequence[CanonicalClaim]
) -> ClaimComparison:
    """Aggregate every same-event proposition before application-owned support adjudication."""

    if claim.extraction_conflict or not claim.structure_validated:
        return ClaimComparison(
            "extraction_uncertain",
            "Claim extractors disagree with the rendered text structure.",
            issue_codes=("CLAIM_EXTRACTION_CONFLICT",),
            human_verification_required=True,
        )
    if claim.kind == "opinion":
        return ClaimComparison(
            "not_applicable", "The sentence is genuinely non-factual commentary."
        )
    if claim.kind == "material_claim_unknown":
        return ClaimComparison(
            "extraction_uncertain",
            "The sentence appears materially factual, but its proposition is uncertain.",
            issue_codes=("CLAIM_EXTRACTION_UNCERTAIN",),
            human_verification_required=True,
        )

    ranked: list[tuple[int, CanonicalClaim, EvidenceCandidateMatch]] = []
    for evidence in evidence_claims:
        match = align_claim_to_evidence(claim, evidence)
        if match.event_alignment != "unrelated":
            ranked.append((_candidate_score(match), evidence, match))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return ClaimComparison(
            "unsupported", "No evidence proposition describes the same event or relation."
        )

    supporting: list[CanonicalClaim] = []
    partial: list[CanonicalClaim] = []
    conflicting: list[CanonicalClaim] = []
    attributed: list[CanonicalClaim] = []
    uncertain: list[CanonicalClaim] = []
    issues: list[str] = []
    conflict_fields_seen: list[str] = []
    best_claim: CanonicalClaim | None = None
    best_match: EvidenceCandidateMatch | None = None

    for _, evidence, match in ranked:
        if best_claim is None:
            best_claim, best_match = evidence, match
        conflicts = contradiction_fields(claim, evidence, match)
        if "question_only" in conflicts:
            uncertain.append(evidence)
            issues.append("QUESTION_ONLY_EVIDENCE")
            continue
        if "modality" in conflicts:
            issues.append("MODALITY_ESCALATION")
        if "event_status" in conflicts:
            issues.extend(("EVENT_STATUS_ESCALATION", "TEMPORAL_STATUS_CHANGED"))
            if evidence.event_status in {"attempted", "near_event", "planned"}:
                issues.append("NON_COMPLETION_AS_COMPLETION")
        if "embedded_event_not_asserted" in conflicts:
            issues.append("EMBEDDED_EVENT_NOT_ASSERTED")
            embedded_status = _embedded_assertion_status(evidence)
            if embedded_status in {"not_established", "questioned"}:
                issues.extend(("EVIDENTIAL_SCOPE_LOST", "UNCERTAINTY_AS_ASSERTION"))
            if embedded_status == "denied":
                issues.append("DENIED_PROPOSITION_AS_FACT")
            if embedded_status == "prevented":
                issues.append("NON_COMPLETION_AS_COMPLETION")
        if conflicts:
            conflict_fields_seen.extend(conflicts)
            conflicting.append(evidence)
            continue
        if match.attribution_alignment == "lost" or (
            evidence.epistemic_modality in {"reported", "alleged"}
            and claim.epistemic_modality == "asserted"
        ):
            attributed.append(evidence)
            issues.append("ATTRIBUTION_LOST")
            continue
        unsupported_qualifiers = _unsupported_qualifiers(claim, evidence)
        if unsupported_qualifiers:
            partial.append(evidence)
            issues.append("UNSUPPORTED_EVENT_DETAIL")
            issues.extend(
                f"UNSUPPORTED_EVENT_DETAIL:{item.qualifier_type}" for item in unsupported_qualifiers
            )
            continue
        partial_reasons = (
            match.event_alignment == "partial"
            or match.actor_alignment in {"partial", "unknown"}
            or match.object_alignment in {"partial", "unknown"}
            or match.quantity_alignment == "missing"
            or match.date_alignment == "missing"
            or match.modality_alignment in {"different", "unknown"}
            or match.event_status_alignment in {"different", "unknown"}
            or match.speech_act_alignment in {"different", "unknown"}
            or match.clause_scope_alignment in {"different", "unknown"}
            or (
                match.event_identity is not None
                and match.event_identity.object_identity == "qualified_same_core"
            )
        )
        if partial_reasons:
            partial.append(evidence)
        elif match.predicate_alignment == "same" or normalize_text(
            claim.exact_text
        ) == normalize_text(evidence.exact_text):
            supporting.append(evidence)
        else:
            uncertain.append(evidence)

    buckets = dict(
        supporting_claims=tuple(supporting),
        partial_claims=tuple(partial),
        conflicting_claims=tuple(conflicting),
        attributed_claims=tuple(attributed),
        uncertain_claims=tuple(uncertain),
    )
    if conflicting:
        return ClaimComparison(
            "source_conflict",
            "Relevant canonical evidence contains a directly competing proposition; "
            "conflicting fields: " + ", ".join(dict.fromkeys(conflict_fields_seen)) + ".",
            best_claim,
            best_match,
            tuple(dict.fromkeys(issues)),
            True,
            **buckets,
        )
    if supporting and attributed:
        return ClaimComparison(
            "partially_supported",
            "The proposition has direct support and additional attributed-only evidence.",
            supporting[0],
            align_claim_to_evidence(claim, supporting[0]),
            tuple(dict.fromkeys(issues)),
            True,
            **buckets,
        )
    if supporting:
        return ClaimComparison(
            "supported",
            "All relevant directly competing evidence is compatible with the proposition.",
            supporting[0],
            align_claim_to_evidence(claim, supporting[0]),
            tuple(dict.fromkeys(issues)),
            False,
            **buckets,
        )
    if attributed:
        return ClaimComparison(
            "attributed_only",
            "Evidence establishes only that the proposition was reported or attributed.",
            attributed[0],
            align_claim_to_evidence(claim, attributed[0]),
            tuple(dict.fromkeys(issues)),
            True,
            **buckets,
        )
    if partial:
        return ClaimComparison(
            "partially_supported",
            "Evidence establishes only part of the material proposition.",
            partial[0],
            align_claim_to_evidence(claim, partial[0]),
            tuple(dict.fromkeys(issues)),
            True,
            **buckets,
        )
    if uncertain:
        return ClaimComparison(
            "unsupported",
            "Relevant evidence does not assert the proposed answer or completed event.",
            uncertain[0],
            align_claim_to_evidence(claim, uncertain[0]),
            tuple(dict.fromkeys(issues)),
            True,
            **buckets,
        )
    return ClaimComparison(
        "unsupported", "No compatible evidence proposition establishes the material claim."
    )


def _claim_corresponds_to_text(claim: CanonicalClaim, final_text: str) -> bool:
    if claim.start is not None and claim.end is not None:
        return (
            claim.end <= len(final_text) and final_text[claim.start : claim.end] == claim.exact_text
        )
    return claim.exact_text in final_text


def claim_structure_matches_text(claim: CanonicalClaim, text: str) -> bool:
    """Validate a provider proposal against deterministic surface structure where possible."""

    if not _claim_corresponds_to_text(claim, text):
        return False
    local = extract_claims(claim.exact_text, prefix="V")
    if not local:
        return False
    candidate = local[0]
    if claim.subject and _alignment(claim.subject, candidate.subject, same_threshold=0.9) != "same":
        return False
    if claim.object and _alignment(claim.object, candidate.object, same_threshold=0.85) not in {
        "same",
        "partial",
    }:
        return False
    if claim.predicate and (candidate.predicate is None or claim.predicate != candidate.predicate):
        return False
    if (
        claim.polarity != "unknown"
        and candidate.polarity != "unknown"
        and claim.polarity != candidate.polarity
    ):
        return False
    if claim.quantities and _normalized_quantities(claim.quantities) != _normalized_quantities(
        candidate.quantities
    ):
        return False
    if claim.dates and tuple(map(normalize_text, claim.dates)) != tuple(
        map(normalize_text, candidate.dates)
    ):
        return False
    if (
        claim.attribution_source
        and _alignment(claim.attribution_source, candidate.attribution_source, same_threshold=0.9)
        != "same"
    ):
        return False
    return (
        not (
            claim.event_status != "unknown"
            and candidate.event_status != "unknown"
            and claim.event_status != candidate.event_status
        )
        and (
            not claim.material_qualifiers
            or _qualifier_signature_for_validation(claim.material_qualifiers)
            == _qualifier_signature_for_validation(candidate.material_qualifiers)
        )
        and (
            not claim.clauses
            or _clause_signature_for_validation(claim.clauses)
            == _clause_signature_for_validation(candidate.clauses)
        )
    )


def _qualifier_signature_for_validation(
    qualifiers: Sequence[EventQualifier],
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((item.qualifier_type, normalize_text(item.text)) for item in qualifiers))


def _clause_signature_for_validation(
    clauses: Sequence[CanonicalClause],
) -> tuple[tuple[str, str, str | None, str | None, str | None], ...]:
    return tuple(
        (
            item.clause_role,
            item.assertion_status,
            normalize_text(item.subject or "") or None,
            item.predicate,
            normalize_text(item.object or "") or None,
        )
        for item in clauses
    )


def merge_claim_hints(
    extracted: Sequence[CanonicalClaim],
    hint_texts: Iterable[str],
    *,
    final_text: str | None = None,
) -> tuple[CanonicalClaim, ...]:
    hints: list[CanonicalClaim] = []
    for index, text in enumerate(hint_texts, start=1):
        if final_text is not None and text not in final_text:
            continue
        for hint in extract_claims(text, prefix=f"H{index}-"):
            hint = hint.model_copy(update={"extraction_provenance": "hint"})
            if any(
                normalize_text(hint.exact_text) == normalize_text(item.exact_text)
                for item in (*extracted, *hints)
            ):
                continue
            hints.append(hint)
    return tuple(hints)


def reconcile_claims(
    final_text: str,
    local_claims: Sequence[CanonicalClaim],
    provider_claims: Sequence[CanonicalClaim],
) -> tuple[CanonicalClaim, ...]:
    """Reconcile extractors without turning provider hallucinations into draft claims."""

    merged = list(local_claims)
    for provider in provider_claims:
        if not _claim_corresponds_to_text(provider, final_text):
            continue
        provider = provider.model_copy(update={"extraction_provenance": "provider"})
        index = next(
            (
                i
                for i, local in enumerate(merged)
                if normalize_text(local.exact_text) == normalize_text(provider.exact_text)
            ),
            None,
        )
        if index is None:
            # Provider-only material claims remain uncertain unless deterministic structure proves
            # exact correspondence. They are never silently promoted to supported.
            validated = claim_structure_matches_text(provider, final_text)
            merged.append(
                provider.model_copy(
                    update={
                        "structure_validated": validated,
                        "extraction_conflict": not validated,
                        "kind": provider.kind if validated else "material_claim_unknown",
                    }
                )
            )
            continue
        local = merged[index]
        validated = claim_structure_matches_text(provider, final_text)
        fields = (
            "subject",
            "predicate",
            "object",
            "polarity",
            "speech_act",
            "epistemic_modality",
            "event_status",
            "quantities",
            "dates",
            "attribution_source",
            "attribution_act",
            "clauses",
            "material_qualifiers",
        )
        disagrees = not validated or any(
            getattr(local, field) != getattr(provider, field) for field in fields
        )
        if disagrees:
            merged[index] = local.model_copy(update={"extraction_conflict": True})
            continue
        # A validated provider proposal may enrich missing legacy fields but does not replace a
        # confident deterministic proposition.
        if local.kind == "material_claim_unknown" or local.predicate is None:
            merged[index] = local.model_copy(update={"extraction_conflict": True})
    return tuple(merged)


def compare_claim_sets(
    previous: Sequence[CanonicalClaim], current: Sequence[CanonicalClaim]
) -> tuple[ClaimChange, ...]:
    remaining = list(previous)
    changes: list[ClaimChange] = []
    fields = (
        "subject",
        "predicate",
        "object",
        "polarity",
        "speech_act",
        "epistemic_modality",
        "event_status",
        "quantities",
        "dates",
        "attribution_source",
        "attribution_act",
        "kind",
        "clauses",
        "material_qualifiers",
    )
    for new in current:
        exact = next(
            (
                old
                for old in remaining
                if normalize_text(old.exact_text) == normalize_text(new.exact_text)
            ),
            None,
        )
        if exact is not None:
            remaining.remove(exact)
            changed_list = [
                field for field in fields if getattr(exact, field) != getattr(new, field)
            ]
            if "epistemic_modality" in changed_list:
                changed_list.append("modality")
            changed = tuple(dict.fromkeys(changed_list))
            changes.append(
                ClaimChange(
                    change_type="changed" if changed else "unchanged",
                    old_claim_id=exact.claim_id,
                    new_claim_id=new.claim_id,
                    fields_changed=changed,
                    explanation=(
                        f"Claim {exact.claim_id} changed in {', '.join(changed)}."
                        if changed
                        else f"Claim {new.claim_id} is unchanged."
                    ),
                )
            )
            continue
        best: CanonicalClaim | None = None
        score_best = -1
        for old in remaining:
            score = 0
            if old.predicate == new.predicate or _opposite(old.predicate, new.predicate):
                score += 4
            if _alignment(old.subject, new.subject) in {"same", "partial"}:
                score += 2
            if _alignment(old.object, new.object) in {"same", "partial"}:
                score += 2
            if score > score_best:
                score_best, best = score, old
        if best is not None and score_best >= 2:
            remaining.remove(best)
            changed_list = [
                field for field in fields if getattr(best, field) != getattr(new, field)
            ]
            if "epistemic_modality" in changed_list:
                changed_list.append("modality")
            changed = tuple(dict.fromkeys(changed_list)) or ("wording",)
            changes.append(
                ClaimChange(
                    change_type="changed",
                    old_claim_id=best.claim_id,
                    new_claim_id=new.claim_id,
                    fields_changed=changed,
                    explanation=f"Claim {best.claim_id} changed in {', '.join(changed)}.",
                )
            )
        else:
            changes.append(
                ClaimChange(
                    change_type="added",
                    new_claim_id=new.claim_id,
                    explanation=f"A new material claim was added: {new.exact_text}",
                )
            )
    for old in remaining:
        changes.append(
            ClaimChange(
                change_type="removed",
                old_claim_id=old.claim_id,
                explanation=f"Claim {old.claim_id} was removed.",
            )
        )
    return tuple(changes)


def approval_invalidation_reason(changes: Sequence[ClaimChange]) -> str:
    for change in changes:
        if change.change_type == "added":
            return "Approval revoked: a new factual claim was added and has no validated evidence."
        if change.change_type != "changed":
            continue
        fields = set(change.fields_changed)
        if "polarity" in fields:
            return "Approval revoked: a claim polarity changed."
        if "predicate" in fields:
            return "Approval revoked: a claim action changed."
        if "epistemic_modality" in fields or "event_status" in fields or "modality" in fields:
            return "Approval revoked: a claim modality or event status changed."
        if "quantities" in fields:
            return "Approval revoked: an approved quantity changed."
        if "dates" in fields:
            return "Approval revoked: an approved date changed."
        if "subject" in fields:
            return "Approval revoked: the named actor changed."
        if "attribution_source" in fields or "attribution_act" in fields:
            return "Approval revoked: the attribution changed."
        if "material_qualifiers" in fields:
            return "Approval revoked: a material event detail changed."
        if "clauses" in fields:
            return "Approval revoked: proposition scope or assertion status changed."
        return "Approval revoked: a material claim changed."
    return "Approval revoked: the canonical draft content changed."


def _contains_conjunction_expansion(anchor: str | None, auxiliary: str | None) -> bool:
    if not anchor or not auxiliary:
        return False
    return " and " in normalize_text(auxiliary) and normalize_text(auxiliary) != normalize_text(
        anchor
    )


def proposition_relation(
    anchor: CanonicalClaim, auxiliary: CanonicalClaim, role: AuxiliaryEvidenceRole
) -> tuple[bool, str]:
    match = align_claim_to_evidence(anchor, auxiliary)
    exact_actor = match.actor_alignment == "same" and not _contains_conjunction_expansion(
        anchor.subject, auxiliary.subject
    )
    exact_object = match.object_alignment == "same" and not _contains_conjunction_expansion(
        anchor.object, auxiliary.object
    )
    compatible = (
        exact_actor
        and exact_object
        and match.predicate_alignment == "same"
        and anchor.polarity == auxiliary.polarity
        and match.modality_alignment == "compatible"
        and match.event_status_alignment == "compatible"
        and match.speech_act_alignment == "compatible"
        and match.attribution_alignment in {"same", "not_applicable"}
    )
    if role in {"confirms_anchor_claim", "reports_same_event"}:
        quantities = not anchor.quantities or _normalized_quantities(
            anchor.quantities
        ) == _normalized_quantities(auxiliary.quantities)
        dates = not anchor.dates or tuple(map(normalize_text, anchor.dates)) == tuple(
            map(normalize_text, auxiliary.dates)
        )
        valid = compatible and quantities and dates
        return (
            valid,
            "same actor-action-object proposition"
            if valid
            else "not a complete same-event confirmation",
        )
    if role == "provides_counterclaim":
        valid = (
            match.actor_alignment == "same"
            and match.object_alignment in {"same", "partial"}
            and (
                match.predicate_alignment == "opposite"
                or (match.predicate_alignment == "same" and anchor.polarity != auxiliary.polarity)
            )
        )
        return valid, "opposed proposition" if valid else "no proposition-level counterclaim"
    if role == "adds_missing_detail":
        same_core = (
            match.actor_alignment == "same"
            and match.predicate_alignment == "same"
            and match.object_alignment in {"same", "partial"}
            and anchor.polarity == auxiliary.polarity
        )
        extra = bool(
            (not anchor.quantities and auxiliary.quantities)
            or (not anchor.dates and auxiliary.dates)
            or (
                anchor.object
                and auxiliary.object
                and match.object_alignment == "partial"
                and len(_tokens(auxiliary.object)) > len(_tokens(anchor.object))
            )
        )
        return (
            bool(same_core and extra),
            "adds a missing proposition detail"
            if same_core and extra
            else "does not add a necessary detail",
        )
    if role in {"provides_direct_response", "supplies_required_context"}:
        related = (
            match.event_alignment in {"same", "partial"}
            and match.predicate_alignment != "different"
        )
        anchor_object_terms = _tokens(anchor.object)
        auxiliary_object_terms = _tokens(auxiliary.object)
        shared_object_terms = anchor_object_terms & auxiliary_object_terms
        if role == "provides_direct_response" and auxiliary.predicate in {"ask", "report"}:
            # A scoped question or direct report may respond to the anchor without asserting the
            # same actor/action relation. Require a concrete anchor-object concept, rather than
            # accepting topic labels or entity overlap alone.
            related = related or bool(shared_object_terms)
        if role == "supplies_required_context":
            # Required context may use a different actor and predicate, but it must connect a
            # concrete object from the anchor to an explicit requirement or dependency relation.
            related = related or bool(
                len(shared_object_terms) >= 2 and auxiliary.predicate in {"require", "need"}
            )
        return related, "propositionally related context" if related else "topic overlap only"
    return False, "unsupported auxiliary role"
