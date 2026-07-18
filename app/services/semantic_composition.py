"""Composable lexical, clause, relation, and coverage extraction.

This module owns all raw-text reading for the semantic safety/diversity core.  It deliberately
keeps lexical normalization, clause framing, relation binding, and policy-neutral coverage as
separate testable stages.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from app.schemas.content import (
    ClauseFrame,
    EvaluationRelation,
    HumanFunctionImpact,
    LexicalAtom,
    LiteralPropositionFrame,
    PredicateScope,
    RelationCandidate,
    ReplyProposition,
    ReplySemanticCompleteness,
    ResultComplementFrame,
    ResultStateCandidate,
    ResultStateMateriality,
    SemanticCoverage,
    SemanticEntity,
    SemanticEvent,
    SemanticInspection,
    SemanticReplyShell,
    SemanticUnresolvedItem,
)
from app.services.semantic_lexicon import (
    ABSTRACT_CONCEPT,
    BUSINESS_OBJECT,
    CELEBRATION_NOUNS,
    CRYPTO,
    GROUP_PHRASES,
    LEGAL_POLITICAL_OBJECT,
    ORGANIZATION,
    PACKAGING_MARKERS,
    PERSON_PLURAL,
    PERSON_SINGULAR,
    PHYSICAL_EVENT,
    POSITIVE_EVENT_WORDS,
    POSITIVE_MODIFIERS,
    PRAISE_NOUNS,
    PRODUCT,
    RATING_NOUNS,
    REDUCTION_MARKERS,
    REWARD_NOUNS,
    SOFTWARE_PROCESS,
    SUCCESS_PREDICATES,
    TECHNICAL_SYSTEM,
    classify_entity_span,
    lexical_tokens,
    normalize_surface,
    token_set,
)

_WORD = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+|\n+")
_CONJUNCTION_BOUNDARY = re.compile(r",\s+(?=(?:and|but)\b)", re.I)
_SUBORDINATOR = (
    r"after|before|when|once|until|following|because|since|as|although|though|"
    r"even\s+though|if|unless|provided\s+that|whereas|while"
)
_SUBORDINATE_BOUNDARY = re.compile(rf"(?:,\s*|\s+)(?=(?:{_SUBORDINATOR})\b)", re.I)
_LEADING_SUBORDINATE = re.compile(
    rf"^(?P<subordinator>{_SUBORDINATOR})\s+"
    r"(?P<subordinate>[^,;]{1,180}),\s*(?P<matrix>.+)$",
    re.I,
)
_SPACE = re.compile(r"\s+")
_QUOTES = re.compile(r"[\"“”][^\"“”]+[\"“”]")

AssertionState = Literal[
    "asserted",
    "negated",
    "possible",
    "planned",
    "feared",
    "reported",
    "conditional",
    "historical_comparison",
    "unknown",
]
ScopeRelation = Literal[
    "matrix",
    "temporal",
    "causal",
    "concessive",
    "conditional",
    "contrastive",
    "relative",
    "participial",
    "other",
]


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    lemma: str
    category: Literal[
        "harm_predicate",
        "harm_modifier",
        "medical_state",
        "medical_destination",
        "medical_escalation",
        "physiological_state",
        "medical_support",
        "functional_dependency",
        "ordinary_result_state",
        "failure_predicate",
        "positive_event",
        "state_predicate",
        "positive_evaluation",
        "evaluation_head",
        "negative_evaluation",
        "reward_relation",
        "rating_relation",
        "result_link",
        "packaging_modifier",
        "reduction_operator",
        "reveal_operator",
        "copula",
        "negation",
        "modality",
        "attribution",
        "conditional",
    ]
    outcome: str | None = None


# Individual lemmas and short compounds only.  These resources normalize surface vocabulary into
# semantic categories; they do not encode candidate safety or a complete sentence.
_HARM_TERMS: dict[str, str] = {
    "die": "death",
    "died": "death",
    "dead": "death",
    "deceased": "death",
    "killed": "death",
    "kill": "death",
    "injured": "serious_injury",
    "injure": "serious_injury",
    "injuries": "serious_injury",
    "claimed": "death",
    "took": "death",
    "ended": "death",
    "lost": "death",
    "perished": "death",
    "slain": "death",
    "lifeless": "death",
    "passed away": "death",
    "pronounced dead": "death",
    "declared deceased": "death",
    "found dead": "death",
    "fatal": "death",
    "fatality": "death",
    "fatalities": "death",
    "death": "death",
    "deaths": "death",
    "death toll": "death",
    "body count": "death",
    "lost lives": "death",
    "lost their lives": "death",
    "lost his life": "death",
    "lost her life": "death",
    "claimed lives": "death",
    "took lives": "death",
    "ended lives": "death",
    "bodies recovered": "death",
    "did not survive": "non_survival",
    "didn't survive": "non_survival",
    "failed to survive": "non_survival",
    "no survivors": "non_survival",
    "never made it out": "non_survival",
    "did not make it home": "non_survival",
    "succumbed": "death",
    "seriously injured": "serious_injury",
    "severely injured": "serious_injury",
    "critically injured": "serious_injury",
    "fatally injured": "death",
    "serious injuries": "serious_injury",
    "severe injuries": "serious_injury",
    "mortal injuries": "death",
    "wounded": "serious_injury",
    "critically wounded": "serious_injury",
    "life-threatening injuries": "serious_injury",
    "permanently injured": "permanent_injury",
    "permanent injuries": "permanent_injury",
}
_MEDICAL_TERMS: dict[str, str] = {
    "critical condition": "critical_condition",
    "in critical condition": "critical_condition",
    "coma": "critical_condition",
    "in a coma": "critical_condition",
    "in coma": "critical_condition",
    "life support": "critical_condition",
    "on life support": "critical_condition",
    "fighting for life": "critical_condition",
    "fighting for his life": "critical_condition",
    "fighting for her life": "critical_condition",
    "fighting for their life": "critical_condition",
    "hospitalized": "hospitalization",
    "hospitalised": "hospitalization",
    "admitted to hospital": "hospitalization",
    "admitted to the hospital": "hospitalization",
    "intensive care": "intensive_care",
    "in intensive care": "intensive_care",
    "icu": "intensive_care",
    "hospitalization": "hospitalization",
    "hospitalisation": "hospitalization",
}
_MEDICAL_DESTINATIONS: dict[str, str] = {
    "hospital": "hospitalization",
    "the hospital": "hospitalization",
    "emergency room": "hospitalization",
    "emergency department": "hospitalization",
    "intensive care unit": "intensive_care",
}
_HARM_MODIFIERS: dict[str, str] = {
    "life threatening": "serious_injury",
    "life-threatening": "serious_injury",
    "severe": "serious_injury",
    "serious": "serious_injury",
    "critical": "critical_condition",
    "mortal": "death",
    "fatal": "death",
    "permanent": "permanent_injury",
    "permanently": "permanent_injury",
    "lasting": "permanent_injury",
    "irreversible": "permanent_injury",
}
_HARM_OUTCOME_HEADS: dict[str, str] = {
    "injury": "serious_injury",
    "injuries": "serious_injury",
    "wound": "serious_injury",
    "wounds": "serious_injury",
    "trauma": "serious_injury",
    "impairment": "permanent_injury",
    "physical damage": "serious_injury",
    "bodily damage": "serious_injury",
    "disability": "permanent_injury",
    "disabilities": "permanent_injury",
    "paralysis": "permanent_injury",
    "amputation": "permanent_injury",
    "amputations": "permanent_injury",
    "burns": "serious_injury",
}
_MEDICAL_ESCALATION_TERMS = {
    "deteriorated",
    "deteriorating",
    "worsened",
    "worsening",
    "decompensated",
    "decompensating",
    "unstable",
    "unresponsive",
}
_PHYSIOLOGICAL_STATE_TERMS: dict[str, str] = {
    "unconscious": "critical_condition",
    "barely conscious": "critical_condition",
    "semi-conscious": "critical_condition",
    "semiconscious": "critical_condition",
    "reduced consciousness": "critical_condition",
    "reduced state of consciousness": "critical_condition",
    "unable to breathe": "critical_condition",
    "unable to breathe unaided": "critical_condition",
    "unable to walk": "functional_impairment",
    "unable to move": "functional_impairment",
    "immobile": "functional_impairment",
    "loss of mobility": "functional_impairment",
    "loss of movement": "functional_impairment",
    "respiratory distress": "critical_condition",
    "breathing difficulty": "critical_condition",
}
_MEDICAL_SUPPORT_TERMS: dict[str, str] = {
    "respiratory support": "critical_condition",
    "breathing support": "critical_condition",
    "assisted breathing": "critical_condition",
    "assisted ventilation": "critical_condition",
    "mechanical ventilation": "critical_condition",
    "ventilator": "critical_condition",
    "respirator": "critical_condition",
    "oxygen support": "critical_condition",
    "medical support": "critical_condition",
    "dialysis": "critical_condition",
}
_FUNCTIONAL_DEPENDENCY_TERMS = {
    "dependent",
    "reliant",
    "depending",
    "relying",
    "requires",
    "required",
}
_ORDINARY_RESULT_STATE_TERMS = {
    "waiting",
    "outside",
    "behind schedule",
    "uncertain",
    "responsive",
    "ready",
    "available",
    "open",
    "closed",
}
_DIMINISHED_STATE_MODIFIERS = {
    "barely",
    "minimally",
    "poorly",
    "weakly",
    "intermittently",
    "partially",
    "scarcely",
}
_ORDINARY_FUNCTION_MARKERS = {
    "account",
    "application",
    "app",
    "backup",
    "bus",
    "button",
    "calendar",
    "dashboard",
    "export",
    "interface",
    "menu",
    "node",
    "page",
    "platform",
    "schedule",
    "screen",
    "service",
    "station",
    "supplier",
    "system",
    "tool",
    "train",
    "vendor",
    "website",
}
_ORDINARY_FUNCTION_VERBS = {
    "access",
    "click",
    "export",
    "find",
    "locate",
    "navigate",
    "open",
    "schedule",
    "use",
}
_IMPAIRMENT_STRUCTURE = re.compile(
    r"(?:\b(?:unable|incapable)\s+to\b|\b(?:cannot|can't|could not)\b|"
    r"\btoo\s+[a-z][a-z-]{1,30}\s+to\b|"
    r"\bneed(?:s|ed|ing)?\s+(?:help|assistance|support)\s+to\b|"
    r"\b(?:loss|lack|limitation|reduction)\s+of\b|"
    r"\b(?:reduced|limited|impaired)\s+(?:ability|capacity|function)\b|"
    r"\bdifficulty\s+(?:with|in|to)\b)",
    re.I,
)
_ASSISTANCE_NEED = re.compile(
    r"\b(?:need(?:s|ed|ing)?|require(?:s|d|ing)?)\b[^.!?;]{0,50}"
    r"\b(?:help|assistance|support|care|aid)\b",
    re.I,
)
_ABNORMAL_STATE_MODIFIER = re.compile(
    r"\b(?:abnormally|incoherently|involuntarily|uncontrollably|erratically|"
    r"severely|poorly|barely|minimally|partially|weakly|diminished|impaired|"
    r"reduced|limited)\b",
    re.I,
)
_ORDINARY_LOCATION_ROLE = re.compile(
    r"^[^.!?;]{1,45}(?:ing|ed)\b[^.!?;]{0,35}\b"
    r"(?:at|by|inside|near|outside|under|within|beside|behind)\b[^.!?;]{1,60}$",
    re.I,
)
_RESULT_GERUND_HEAD = re.compile(r"^[^\W\d_][^\W_]{1,40}ing\b", re.I | re.UNICODE)
_RESULT_PARTICIPLE_HEAD = re.compile(
    r"^(?:[a-z][a-z-]{1,40}(?:ed|en)|(?:barely|minimally|partially|poorly|weakly)\s+"
    r"[a-z][a-z-]{1,40}(?:ed|en|ive|ous|less))\b",
    re.I,
)
_MAIN_NEGATION = re.compile(
    r"(?:^|\b)(?:no\s+(?:one|person|people|worker|workers|patient|patients|resident|residents)\b|"
    r"(?:do|does|did|has|have|had|is|are|was|were|will|would|can|could|should|may|might|must)"
    r"\s+not\b|\bnever\s+(?!mind\b))",
    re.I,
)
_EVIDENTIAL_PARTICIPLE = re.compile(
    r"\b(?:confirmed|verified|established|determined|identified|substantiated|proven|resolved|"
    r"explained|disclosed|published|released)\b",
    re.I,
)
_EVIDENTIAL_PROCESS_HEAD = re.compile(
    r"\b(?:investigation|examination|inquiry|review|analysis|verification|"
    r"determination|assessment)\b",
    re.I,
)
_REMEDIATION_PREDICATE = re.compile(
    r"\b(?:need|needs|needed|require|requires|required|warrant|warrants|"
    r"call(?:s|ed)?\s+for|demand|demands|demanded)\b",
    re.I,
)
_REMEDIATION_OBJECT = re.compile(
    r"\b(?:audit|inspection|investigation|review|repair|remediation|response|"
    r"action|follow[- ]?up|explanation|accountability|verification|assessment|"
    r"replacement|substitution|preservation|isolation|quarantine)\b",
    re.I,
)
_ACTION_RECOMMENDATION = re.compile(
    r"^(?P<subject>[^.!?;]{1,80}?)\s+"
    r"(?P<modal>should|must|ought\s+to|need(?:s|ed)?\s+to|has\s+to|have\s+to|"
    r"is\s+required\s+to|are\s+required\s+to)\s+"
    r"(?P<action>[a-z][a-z-]{1,32})\s+(?P<object>[^.!?;]{1,160})$",
    re.I,
)
_ACKNOWLEDGEMENT_PREDICATE = re.compile(
    r"^(?P<subject>[^.!?;]{1,120}?)\s+(?:make|makes|made)\s+sense$",
    re.I,
)
_NORMATIVE_RESULT_PREDICATE = re.compile(
    r"^(?P<subject>[^.!?;]{1,120}?)\s+(?:make|makes|made|render|renders|rendered)\s+"
    r"(?P<object>[^.!?;]{1,120}?)\s+"
    r"(?P<state>inappropriate|unsuitable|unacceptable|unsafe|unwise|improper|"
    r"disrespectful|inadvisable)$",
    re.I,
)
_FACTUAL_ACTION = re.compile(
    r"\b(?:publish|published|release|released|provide|provided|file|filed|issue|issued|"
    r"disclose|disclosed|submit|submitted|share|shared|report|reported|document|documented|"
    r"record|recorded|complete|completed|restore|restored|repair|repaired|update|updated)\b",
    re.I,
)
_ORDINARY_SOCIAL_STATES = {
    "stunned",
    "surprised",
    "speechless",
    "uncertain",
    "confused about the decision",
    "quiet",
    "withdrawn",
    "reserved",
    "pleased",
    "happy",
    "upset",
}
_ORDINARY_BUSINESS_OBJECTS = {
    "invoice",
    "invoices",
    "contract",
    "contracts",
    "order",
    "orders",
    "budget",
    "budgets",
    "document",
    "documents",
    "request",
    "requests",
    "expense",
    "expenses",
    "payment",
    "payments",
}
_ORDINARY_BUSINESS_VERBS = {
    "approve",
    "authorize",
    "sign",
    "process",
    "submit",
    "review",
    "schedule",
    "access",
    "locate",
    "find",
    "open",
}
_ORDINARY_TRANSPORT_HEADS = {
    "bus",
    "ferry",
    "train",
    "shuttle",
    "tram",
    "subway",
    "route",
    "station",
}
_HUMAN_FUNCTION_STRUCTURE = re.compile(
    r"\b(?:see|seeing|sight|vision|visual|speak|speaking|speech|word|words|"
    r"breathe|breathing|respiratory|oxygen|walk|walking|stand|standing|balance|"
    r"move|moving|movement|coordinate|coordinating|remember|remembering|recognize|"
    r"recognizing|orientation|oriented|conscious|consciousness|responsive|"
    r"hand|hands|arm|arms|leg|legs|ear|ears|eye|eyes|grip|sensation|sensory|"
    r"numb|numbness)\b",
    re.I,
)
_RESULT_TAIL_FRAME = re.compile(
    r"^(?P<affected>(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"a|an|the|any|some|several|many)\s+)?(?:[a-z][a-z'-]*\s+){0,3}?"
    r"[a-z][a-z'-]*?)\s+(?P<complement>"
    r"(?:with\s+.+)|(?:unable|incapable)\s+to\s+.+|(?:cannot|can't|could not)\s+.+|"
    r"(?:dependent|reliant|depending|relying|requiring|required)\s+.+|"
    r"[a-z][a-z'-]*ing\s+.+)$",
    re.I,
)
_RESULT_LINK_TERMS = {
    "sent",
    "send",
    "sends",
    "left",
    "leave",
    "leaves",
    "put",
    "puts",
    "placed",
    "place",
    "places",
    "landed",
    "lands",
    "drove",
    "driven",
    "forced",
    "forces",
    "caused",
    "causes",
    "resulted",
    "results",
    "led",
    "leads",
    "rushed",
    "rushes",
    "transported",
    "transports",
    "transferred",
    "transfers",
    "airlifted",
    "airlifts",
    "brought",
    "brings",
    "taken",
    "takes",
    "rendered",
    "renders",
    "render",
    "made",
    "makes",
    "make",
    "kept",
    "keeps",
    "keep",
}
_NON_HARM_EVENT_TERMS: dict[str, str] = {
    "displaced": "displacement",
    "displace": "displacement",
    "evacuated": "evacuation",
    "evacuate": "evacuation",
    "damaged": "damage",
    "destroyed": "damage",
}
_FAILURE_TERMS = {
    "failed",
    "failure",
    "died",
    "dead",
    "killed",
    "crashed",
    "crash",
    "collapsed",
    "collapse",
    "outage",
    "blackout",
    "blast",
    "explosion",
    "fire",
    "attack",
    "incident",
    "accident",
    "earthquake",
    "flood",
    "disaster",
    "offline",
    "stalled",
    "blocked",
    "rejected",
}
_POSITIVE_EVENT_TERMS = set(POSITIVE_EVENT_WORDS) | {
    "restored power",
    "completed recovery",
    "returned to service",
}
_NEGATION_TERMS = {"no", "not", "never", "without", "neither", "nor"}
_MODAL_TERMS = {
    "may",
    "might",
    "could",
    "possibly",
    "perhaps",
    "allegedly",
    "reportedly",
    "expected",
    "likely",
    "unlikely",
}
_PLAN_TERMS = {"plan", "plans", "planned", "intend", "intends", "intended", "scheduled"}
_FEAR_TERMS = {"fear", "fears", "feared", "worried", "concerned"}
_ATTRIBUTION_TERMS = {
    "reported",
    "reports",
    "said",
    "says",
    "according to",
    "announced",
    "confirmed",
}
_CONDITIONAL_TERMS = {"if", "unless", "would", "could have", "might have"}
_COPULAS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "remains",
    "remained",
    "equals",
    "equaled",
    "equalled",
}
_INVERSE_PRAISE_PREDICATES = {
    "says",
    "screams",
    "captures",
    "embodies",
    "expresses",
    "signals",
    "communicates",
    "demonstrates",
    "shows",
}
_SAFE_EVENT_COMPOUNDS = {
    "fire sale",
    "fire callback",
    "fire request",
    "fire event",
    "growth explosion",
    "sales explosion",
    "explosion of interest",
    "explosion in users",
    "market collapse",
    "price collapse",
    "server collapse",
    "system collapse",
    "cyber attack",
    "ddos attack",
    "prompt attack",
    "adversarial attack",
}
_RATING_PATTERN = re.compile(r"\b\d+\s*(?:/|out\s+of)\s*\d+\b", re.I)
_REVEAL_TERMS = {
    "turns out",
    "turned out",
    "as it turns out",
    "apparently",
    "all along",
    "who knew",
    "surprise",
}
_REDUCTION_PHRASES = set(REDUCTION_MARKERS) | {
    "nothing but",
    "little more than",
    "amounts to",
    "boils down to",
    "comes down to",
    "in other words",
    "what it is",
    "really is",
}
_PACKAGING_PHRASES = set(PACKAGING_MARKERS) | {
    "dressed up",
    "dressed up as",
    "under a new name",
    "fresh coat of paint",
    "coat of paint",
    "in disguise",
    "pretending to be",
    "cosplaying as",
    "better branding",
    "better ux",
    "product clothing",
    "wearing product branding",
    "with lipstick",
    "with a makeover",
    "in a costume",
    "fancier",
    "polished",
    "glossy",
    "gloss",
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
}
_STATE_TERMS = {
    "active",
    "available",
    "bad",
    "broken",
    "closed",
    "complete",
    "down",
    "early",
    "good",
    "healthy",
    "inactive",
    "incomplete",
    "late",
    "live",
    "offline",
    "online",
    "open",
    "operational",
    "ready",
    "safe",
    "stable",
    "unavailable",
    "unsafe",
    "unstable",
    "up",
    "useful",
    "useless",
}

_POSITIVE_EVALUATION_PHRASES = (
    set(POSITIVE_MODIFIERS)
    | set(CELEBRATION_NOUNS)
    | set(SUCCESS_PREDICATES)
    | {
        "top marks",
        "full marks",
        "five stars",
        "great timing",
        "chef's kiss",
        "round of applause",
        "standing ovation",
        "big win",
        "good stuff",
        "employee of the month",
        "could not be prouder",
        "couldn't be prouder",
        "could not have asked for better",
        "couldn't have asked for better",
        "love",
        "needed",
        "better",
        "proud",
        "mission accomplished",
        "win",
        "victory",
        "triumph",
        "applause",
        "ovation",
        "hats off",
        "here's to",
        "take a bow",
    }
)
_EVALUATION_HEADS = set(PRAISE_NOUNS) | {
    "credit",
    "recognition",
    "honor",
    "honors",
    "honour",
    "honours",
    "kudos",
    "salute",
    "bow",
    "acclaim",
    "praise",
    "congratulations",
    "congrats",
    "encore",
    "props",
    "shoutout",
    "shout-out",
    "laurels",
    "flowers",
}
_RECOGNITION_RELATIONS = {
    "credit",
    "recognition",
    "honor",
    "honors",
    "honour",
    "honours",
    "kudos",
    "salute",
    "bow",
    "acclaim",
    "praise",
    "congratulations",
    "congrats",
    "encore",
    "props",
    "shoutout",
    "shout-out",
    "laurels",
    "flowers",
}
_REWARD_PHRASES = (
    set(REWARD_NOUNS)
    | _RECOGNITION_RELATIONS
    | {
        "deserves",
        "earned",
        "earns",
        "wins",
        "should get",
        "gets",
        "got",
        "receives",
        "received",
        "takes",
        "took",
    }
)
_RATING_PHRASES = set(RATING_NOUNS) | {"top marks", "full marks"}

_HEAD_CATEGORIES: tuple[tuple[set[str] | frozenset[str], str], ...] = (
    (set(PERSON_SINGULAR) | set(PERSON_PLURAL) | set(GROUP_PHRASES), "human_head"),
    (set(SOFTWARE_PROCESS) | set(TECHNICAL_SYSTEM) | set(PRODUCT) | set(CRYPTO), "technical_head"),
    (set(ORGANIZATION), "organization_head"),
    (set(BUSINESS_OBJECT), "business_head"),
    (set(LEGAL_POLITICAL_OBJECT), "legal_political_head"),
    (set(PHYSICAL_EVENT), "physical_event_head"),
    (set(ABSTRACT_CONCEPT), "abstract_head"),
)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    pieces = [re.escape(piece) for piece in phrase.split()]
    return re.compile(r"\b" + r"\s+".join(pieces) + r"\b", re.I)


def _iter_phrase_atoms(
    text: str,
    phrases: Iterable[str],
    category: str,
    *,
    lemma_map: dict[str, str] | None = None,
) -> Iterable[LexicalAtom]:
    for phrase in sorted(set(phrases), key=lambda value: (-len(value.split()), -len(value), value)):
        for match in _phrase_pattern(phrase).finditer(text):
            yield LexicalAtom(
                span=match.group(0),
                lemma=(lemma_map or {}).get(phrase, normalize_surface(phrase)),
                category=category,
                start=match.start(),
                end=match.end(),
            )


def extract_lexical_atoms(clause: str) -> tuple[LexicalAtom, ...]:
    """Return non-overlapping lexical atoms ordered by surface position."""

    candidates: list[LexicalAtom] = []
    for phrases, category in _HEAD_CATEGORIES:
        candidates.extend(_iter_phrase_atoms(clause, phrases, category))
    candidates.extend(
        _iter_phrase_atoms(clause, _HARM_TERMS, "harm_predicate", lemma_map=_HARM_TERMS)
    )
    candidates.extend(
        _iter_phrase_atoms(clause, _MEDICAL_TERMS, "medical_state", lemma_map=_MEDICAL_TERMS)
    )
    candidates.extend(
        _iter_phrase_atoms(
            clause,
            _MEDICAL_DESTINATIONS,
            "medical_destination",
            lemma_map=_MEDICAL_DESTINATIONS,
        )
    )
    candidates.extend(
        _iter_phrase_atoms(
            clause,
            _HARM_MODIFIERS,
            "harm_modifier",
            lemma_map=_HARM_MODIFIERS,
        )
    )
    candidates.extend(
        _iter_phrase_atoms(
            clause,
            _HARM_OUTCOME_HEADS,
            "harm_predicate",
            lemma_map=_HARM_OUTCOME_HEADS,
        )
    )
    candidates.extend(_iter_phrase_atoms(clause, _MEDICAL_ESCALATION_TERMS, "medical_escalation"))
    candidates.extend(
        _iter_phrase_atoms(
            clause,
            _PHYSIOLOGICAL_STATE_TERMS,
            "physiological_state",
            lemma_map=_PHYSIOLOGICAL_STATE_TERMS,
        )
    )
    candidates.extend(
        _iter_phrase_atoms(
            clause,
            _MEDICAL_SUPPORT_TERMS,
            "medical_support",
            lemma_map=_MEDICAL_SUPPORT_TERMS,
        )
    )
    candidates.extend(
        _iter_phrase_atoms(clause, _FUNCTIONAL_DEPENDENCY_TERMS, "functional_dependency")
    )
    candidates.extend(
        _iter_phrase_atoms(clause, _ORDINARY_RESULT_STATE_TERMS, "ordinary_result_state")
    )
    candidates.extend(
        _iter_phrase_atoms(
            clause,
            _DIMINISHED_STATE_MODIFIERS,
            "diminished_state_modifier",
        )
    )
    candidates.extend(_iter_phrase_atoms(clause, _RESULT_LINK_TERMS, "result_link"))
    candidates.extend(
        _iter_phrase_atoms(clause, _NON_HARM_EVENT_TERMS, "other", lemma_map=_NON_HARM_EVENT_TERMS)
    )
    candidates.extend(_iter_phrase_atoms(clause, _FAILURE_TERMS, "failure_predicate"))
    candidates.extend(_iter_phrase_atoms(clause, _POSITIVE_EVENT_TERMS, "positive_event"))
    candidates.extend(_iter_phrase_atoms(clause, _STATE_TERMS, "state_predicate"))
    candidates.extend(
        _iter_phrase_atoms(clause, _POSITIVE_EVALUATION_PHRASES, "positive_evaluation")
    )
    candidates.extend(_iter_phrase_atoms(clause, _EVALUATION_HEADS, "evaluation_head"))
    candidates.extend(_iter_phrase_atoms(clause, _REWARD_PHRASES, "reward_relation"))
    candidates.extend(_iter_phrase_atoms(clause, _RATING_PHRASES, "rating_relation"))
    candidates.extend(_iter_phrase_atoms(clause, _PACKAGING_PHRASES, "packaging_modifier"))
    candidates.extend(_iter_phrase_atoms(clause, _REDUCTION_PHRASES, "reduction_operator"))
    candidates.extend(_iter_phrase_atoms(clause, _REVEAL_TERMS, "reveal_operator"))
    candidates.extend(_iter_phrase_atoms(clause, _COPULAS, "copula"))
    candidates.extend(_iter_phrase_atoms(clause, _NEGATION_TERMS, "negation"))
    candidates.extend(
        _iter_phrase_atoms(clause, _MODAL_TERMS | _PLAN_TERMS | _FEAR_TERMS, "modality")
    )
    candidates.extend(_iter_phrase_atoms(clause, _ATTRIBUTION_TERMS, "attribution"))
    candidates.extend(_iter_phrase_atoms(clause, _CONDITIONAL_TERMS, "conditional"))
    candidates.extend(_iter_phrase_atoms(clause, _INVERSE_PRAISE_PREDICATES, "positive_evaluation"))
    for match in _RATING_PATTERN.finditer(clause):
        candidates.append(
            LexicalAtom(
                span=match.group(0),
                lemma="numeric_rating",
                category="rating_relation",
                start=match.start(),
                end=match.end(),
            )
        )

    # Prefer longer atoms at the same position and suppress contained atoms of the same semantic
    # family.  Cross-category overlap is retained (e.g. ``died`` can be a harm or failure
    # predicate); relation binding resolves it from the referent type.
    candidates.sort(key=lambda item: (item.start, -(item.end - item.start), item.category))
    result: list[LexicalAtom] = []
    for atom in candidates:
        if any(
            existing.category == atom.category
            and existing.start <= atom.start
            and existing.end >= atom.end
            for existing in result
        ):
            continue
        result.append(atom)
    return tuple(sorted(result, key=lambda item: (item.start, item.end, item.category)))


def _main_clause_negated(clause: str, atoms: Sequence[LexicalAtom]) -> bool:
    """Return whether negation scopes over the main proposition, not an embedded complement."""

    normalized = normalize_surface(clause)
    asserted_non_survival = any(
        atom.category == "harm_predicate" and atom.lemma == "non_survival" for atom in atoms
    )
    if asserted_non_survival or "no survivors" in normalized:
        return False
    return bool(_MAIN_NEGATION.search(normalized))


def _assertion_state(clause: str, atoms: Sequence[LexicalAtom]) -> AssertionState:
    normalized = normalize_surface(clause)
    categories = {atom.category for atom in atoms}
    tokens = token_set(normalized)
    if "historical comparison" in normalized or normalized.startswith("compared with"):
        return "historical_comparison"
    if "conditional" in categories or normalized.startswith("if "):
        return "conditional"
    if tokens & _PLAN_TERMS:
        return "planned"
    if tokens & _FEAR_TERMS:
        return "feared"
    if "modality" in categories:
        return "possible"
    if _main_clause_negated(clause, atoms):
        return "negated"
    if "attribution" in categories:
        return "reported"
    return "asserted"


def _clean_role(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip(" \t\n\r,;:.!?\"'“”")
    cleaned = re.sub(
        r"^(?:and|but|while|whereas|after|before|during|following|then|that)\s+",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"^(?:a|an|the)\s+", lambda m: m.group(0), cleaned, flags=re.I)
    return cleaned or None


def _predicate_atom(atoms: Sequence[LexicalAtom]) -> LexicalAtom | None:
    priority = {
        "harm_predicate": 0,
        "medical_state": 1,
        "medical_escalation": 2,
        "failure_predicate": 3,
        "other": 4,
        "positive_event": 5,
        "state_predicate": 6,
        "positive_evaluation": 7,
        "evaluation_head": 8,
        "reward_relation": 9,
        "rating_relation": 10,
        "reduction_operator": 11,
        "packaging_modifier": 12,
        "reveal_operator": 13,
        "copula": 14,
    }
    material = [atom for atom in atoms if atom.category in priority]
    if not material:
        return None
    return sorted(
        material, key=lambda atom: (priority[atom.category], atom.start, -len(atom.span))
    )[0]


def build_clause_frame(
    clause: str,
    *,
    clause_id: str = "unscoped",
    scope_id: str = "unscoped",
    parent_scope_id: str | None = None,
    relation_to_parent: ScopeRelation = "matrix",
    coordination_group_id: str | None = None,
    shared_subject_clause_id: str | None = None,
) -> ClauseFrame:
    """Build a lightweight clause frame independently of event/humor policy."""

    atoms = tuple(
        atom.model_copy(update={"clause_id": clause_id}) for atom in extract_lexical_atoms(clause)
    )
    predicate = _predicate_atom(atoms)
    subject: str | None = None
    object_span: str | None = None
    complement: str | None = None
    predicate_span: str | None = predicate.span if predicate else None
    if predicate is not None:
        before = clause[: predicate.start]
        after = clause[predicate.end :]
        # Remove attribution wrappers from the grammatical subject where possible.
        before = re.sub(
            r"^(?:according to\s+[^,]+,\s*|[^,]{1,80}\s+(?:reported|said|confirmed|announced)\s+)",
            "",
            before,
            flags=re.I,
        )
        before = re.sub(rf"^(?:{_SUBORDINATOR})\s+", "", before, flags=re.I)
        if predicate.category in {
            "harm_predicate",
            "medical_state",
            "medical_escalation",
            "failure_predicate",
            "positive_event",
            "other",
        }:
            before = re.sub(
                r"\b(?:is|are|was|were|be|been|being|has|have|had|to)\s*$",
                "",
                before,
                flags=re.I,
            )
        subject = _clean_role(before)
        object_span = _clean_role(after)
        complement = object_span

    # Copular clauses use the copula as the structural predicate even when the complement also
    # contains an evaluation or packaging atom.
    copula_atoms = [atom for atom in atoms if atom.category == "copula"]
    if copula_atoms:
        copula = copula_atoms[0]
        later_event_predicate = next(
            (
                atom
                for atom in atoms
                if atom.start >= copula.end
                and atom.category
                in {
                    "harm_predicate",
                    "medical_state",
                    "medical_escalation",
                    "failure_predicate",
                    "positive_event",
                    "other",
                }
            ),
            None,
        )
        if later_event_predicate is None:
            copula_subject = _clean_role(clause[: copula.start])
            copula_complement = _clean_role(clause[copula.end :])
            if copula_subject and copula_complement:
                subject = copula_subject
                predicate_span = copula.span
                complement = copula_complement
                object_span = copula_complement

    # Colon frames are a punctuation variant of a copular relation.
    if ":" in clause and not _QUOTES.search(clause):
        left, right = clause.split(":", 1)
        if _clean_role(left) and _clean_role(right):
            subject = _clean_role(left)
            complement = _clean_role(right)
            object_span = complement
            predicate_span = ":"

    state = _assertion_state(clause, atoms)
    modal = next((atom.span for atom in atoms if atom.category == "modality"), None)
    attributed = any(atom.category == "attribution" for atom in atoms)
    conditional = any(atom.category == "conditional" for atom in atoms) or normalize_surface(
        clause
    ).startswith("if ")
    main_negated = _main_clause_negated(clause, atoms)
    return ClauseFrame(
        clause_id=clause_id,
        scope_id=scope_id,
        parent_scope_id=parent_scope_id,
        relation_to_parent=relation_to_parent,
        coordination_group_id=coordination_group_id,
        shared_subject_clause_id=shared_subject_clause_id,
        clause_span=clause.strip(),
        subject_span=subject,
        predicate_span=predicate_span,
        object_span=object_span,
        complement_span=complement,
        modifiers=tuple(
            atom.span
            for atom in atoms
            if atom.category in {"packaging_modifier", "reduction_operator", "reveal_operator"}
        ),
        negated=main_negated,
        modal=modal,
        attributed=attributed,
        conditional=conditional,
        assertion_state=state,
        atoms=atoms,
    )


def _contains_predicate_like_surface(text: str) -> bool:
    normalized = normalize_surface(text)
    atoms = extract_lexical_atoms(text)
    if any(atom.category == "result_link" for atom in atoms):
        return True
    if re.search(
        r"\b(?:is|are|was|were|be|been|being|remains|remained|has|have|had|do|does|did|"
        r"will|would|can|could|should|may|might|must)\b",
        normalized,
        re.I,
    ):
        return True
    return bool(
        re.search(
            r"\b[^\W\d_]+(?:ed|en|izes|ises|ized|ised|ified|ifies)\b",
            normalized,
            re.I | re.UNICODE,
        )
    )


def _split_subordinate_scopes(text: str) -> tuple[str, ...]:
    """Split finite subordinate predicates from matrix arguments before role binding."""

    leading = _LEADING_SUBORDINATE.match(text.strip())
    if leading is not None:
        subordinate = f"{leading.group('subordinator')} {leading.group('subordinate')}".strip()
        matrix = leading.group("matrix").strip()
        if _contains_predicate_like_surface(subordinate) and _contains_predicate_like_surface(
            matrix
        ):
            return (
                *_split_subordinate_scopes(subordinate),
                *_split_subordinate_scopes(matrix),
            )

    matches = tuple(_SUBORDINATE_BOUNDARY.finditer(text))
    if not matches:
        participial = re.match(
            r"^(?P<matrix>.+?),\s*(?P<subordinate>[^\W\d_][^\W_]{1,40}(?:ing|ed)\b.+)$",
            text.strip(),
            re.I | re.UNICODE,
        )
        subordinate_atoms = (
            extract_lexical_atoms(participial.group("subordinate"))
            if participial is not None
            else ()
        )
        subordinate_starts_nonfinite = bool(
            participial is not None
            and re.match(
                r"^[^\W\d_][^\W_]{1,40}(?:ing|ed)\b",
                participial.group("subordinate"),
                re.I | re.UNICODE,
            )
            and not any(atom.category == "copula" for atom in subordinate_atoms)
        )
        if (
            participial is not None
            and _contains_predicate_like_surface(participial.group("matrix"))
            and subordinate_starts_nonfinite
            and not any(
                atom.category == "copula"
                for atom in extract_lexical_atoms(participial.group("matrix"))
            )
        ):
            return (
                participial.group("matrix").strip(),
                participial.group("subordinate").strip(),
            )
        return (text.strip(),)
    parts: list[str] = []
    cursor = 0
    for match in matches:
        suffix = text[match.end() :].strip()
        prefix = text[cursor : match.start()].strip()
        if not prefix or not suffix or not _contains_predicate_like_surface(suffix):
            continue
        parts.append(prefix)
        cursor = match.start()
        # One finite subordinate scope is enough; any nested connector is handled on the next pass.
        break
    if not parts:
        return (text.strip(),)
    tail = text[cursor:].strip()
    nested = _split_subordinate_scopes(tail) if tail != text.strip() else (tail,)
    return tuple((*parts, *nested))


def _scope_relation(text: str) -> ScopeRelation:
    normalized = normalize_surface(text)
    if any(atom.category == "reveal_operator" for atom in extract_lexical_atoms(text)):
        return "matrix"
    if re.match(r"^(?:after|before|when|once|until|following)\b", normalized):
        return "temporal"
    if re.match(r"^(?:because|since|as)\b", normalized):
        return "causal"
    if re.match(r"^(?:although|though|even though)\b", normalized):
        return "concessive"
    if re.match(r"^(?:if|unless|provided that)\b", normalized):
        return "conditional"
    if re.match(r"^(?:whereas|while)\b", normalized):
        return "contrastive"
    first = lexical_tokens(normalized)
    if first and re.fullmatch(r"[^\W\d_]+(?:ing|ed)", first[0], re.I | re.UNICODE):
        return "participial"
    return "matrix"


def segment_clauses(
    text: str,
    *,
    split_coordination: bool = True,
) -> tuple[ClauseFrame, ...]:
    """Segment text into stable clause and subclause scopes before role binding."""

    frames: list[ClauseFrame] = []
    clause_counter = 0
    group_counter = 0
    for sentence in _SENTENCE_BOUNDARY.split(text):
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            continue
        coordination_parts = (
            tuple(
                part.strip()
                for part in _CONJUNCTION_BOUNDARY.split(cleaned_sentence)
                if part.strip()
            )
            if split_coordination
            else (cleaned_sentence,)
        )
        parts = tuple(
            scoped
            for part in coordination_parts
            for scoped in _split_subordinate_scopes(part)
            if scoped
        )
        group_counter += 1
        coordination_group_id = f"G{group_counter}" if len(parts) > 1 else None
        relations = tuple(_scope_relation(part) for part in parts)
        scope_ids = tuple(f"C{clause_counter + offset + 1}" for offset in range(len(parts)))
        matrix_ids = tuple(
            scope_id
            for scope_id, relation in zip(scope_ids, relations, strict=True)
            if relation == "matrix"
        )
        parent_scope_id = matrix_ids[0] if matrix_ids else None
        previous_clause_id: str | None = None
        for part, relation, scope_id in zip(parts, relations, scope_ids, strict=True):
            clause_counter += 1
            clause_id = f"C{clause_counter}"
            probe = extract_lexical_atoms(part)
            has_subject = any(atom.category.endswith("_head") for atom in probe)
            shared_subject_clause_id = (
                previous_clause_id
                if previous_clause_id is not None and not has_subject and coordination_group_id
                else None
            )
            frames.append(
                build_clause_frame(
                    part,
                    clause_id=clause_id,
                    scope_id=scope_id,
                    parent_scope_id=(parent_scope_id if relation != "matrix" else None),
                    relation_to_parent=relation,
                    coordination_group_id=coordination_group_id,
                    shared_subject_clause_id=shared_subject_clause_id,
                )
            )
            previous_clause_id = clause_id
    return tuple(frames)


def build_predicate_scope_graph(
    text: str,
    frames: Sequence[ClauseFrame],
    *,
    namespace: str,
) -> tuple[PredicateScope, ...]:
    """Expose request-local scope identities, including embedded relative clauses."""

    id_map = {frame.scope_id: f"{namespace}-{frame.scope_id}" for frame in frames}
    output = [
        PredicateScope(
            scope_id=id_map[frame.scope_id],
            parent_scope_id=(
                id_map.get(frame.parent_scope_id) if frame.parent_scope_id is not None else None
            ),
            relation_to_parent=frame.relation_to_parent,
            text_span=frame.clause_span,
        )
        for frame in frames
    ]
    matrix_scope = next(
        (item.scope_id for item in output if item.relation_to_parent == "matrix"),
        output[0].scope_id if output else None,
    )
    relative_pattern = re.compile(
        r",\s*(?P<relative>(?:who|which|that)\s+[^,;]{1,180}),",
        re.I | re.UNICODE,
    )
    for index, match in enumerate(relative_pattern.finditer(text), start=1):
        relative = match.group("relative").strip()
        if not _contains_predicate_like_surface(relative):
            continue
        output.append(
            PredicateScope(
                scope_id=f"{namespace}-REL{index}",
                parent_scope_id=matrix_scope,
                relation_to_parent="relative",
                text_span=relative,
            )
        )
    return tuple(output)


def _dedupe_entities(values: Iterable[SemanticEntity]) -> tuple[SemanticEntity, ...]:
    result: list[SemanticEntity] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        key = (normalize_surface(value.text_span), value.entity_type, value.clause_id or "")
        if not key[0] or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)


def _entity_spans(frame: ClauseFrame) -> tuple[str, ...]:
    spans: list[str] = []
    for atom in frame.atoms:
        if atom.category.endswith("_head"):
            spans.append(atom.span)
    for value in (frame.subject_span, frame.object_span, frame.complement_span):
        if value and len(lexical_tokens(value)) <= 8:
            entity = classify_entity_span(value)
            specific_human_head = bool(
                token_set(value) & (set(PERSON_SINGULAR) | set(PERSON_PLURAL))
            )
            if entity.entity_type != "person" or specific_human_head:
                spans.append(value)
    return tuple(dict.fromkeys(spans))


def extract_entity_candidates(frames: Sequence[ClauseFrame]) -> tuple[SemanticEntity, ...]:
    return _dedupe_entities(
        classify_entity_span(span).model_copy(update={"clause_id": frame.clause_id})
        for frame in frames
        for span in _entity_spans(frame)
    )


def _nearest_entity(
    frame: ClauseFrame,
    *,
    before: int | None = None,
    after: int | None = None,
    types: set[str] | None = None,
) -> SemanticEntity | None:
    atoms = [atom for atom in frame.atoms if atom.category.endswith("_head")]
    if before is not None:
        atoms = [atom for atom in atoms if atom.end <= before]
        atoms.sort(key=lambda atom: before - atom.end)
    elif after is not None:
        atoms = [atom for atom in atoms if atom.start >= after]
        atoms.sort(key=lambda atom: atom.start - after)
    for atom in atoms:
        entity = classify_entity_span(atom.span).model_copy(update={"clause_id": frame.clause_id})
        if types is None or entity.entity_type in types:
            return entity
    return None


def _role_entity(value: str | None, *, clause_id: str | None = None) -> SemanticEntity | None:
    if not value:
        return None
    # Keep the role compact: subordinate event tails are not entity names.
    compact = re.split(
        r"\b(?:after|before|during|when|while|because|following|at|in|on|by|from|with)\b",
        value,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    entity = classify_entity_span(compact or value)
    return entity.model_copy(update={"clause_id": clause_id}) if clause_id else entity


def _physical_event(frame: ClauseFrame) -> SemanticEntity | None:
    atom = next((a for a in frame.atoms if a.category == "physical_event_head"), None)
    if atom is None:
        return None
    return classify_entity_span(atom.span).model_copy(update={"clause_id": frame.clause_id})


def _implicit_people(span: str) -> SemanticEntity:
    return SemanticEntity(
        text_span=span,
        canonical_name="people",
        entity_type="group_of_people",
        confidence="high",
    )


def _event_type_for(outcome: str, referent: SemanticEntity | None) -> str:
    if outcome in {
        "death",
        "serious_injury",
        "critical_condition",
        "hospitalization",
        "intensive_care",
        "permanent_injury",
        "non_survival",
    }:
        if referent is not None and referent.entity_type in {"person", "group_of_people"}:
            return outcome
        if referent is not None and referent.entity_type in {
            "technical_system",
            "software_process",
            "product",
            "company_or_organization",
            "crypto_or_protocol",
            "business_or_financial_object",
            "legal_or_political_object",
            "abstract_concept",
        }:
            return (
                "technical_failure"
                if referent.entity_type in {"technical_system", "software_process"}
                else "business_failure"
            )
        return "unknown"
    return outcome


def _event_polarity(event_type: str, state: AssertionState) -> str:
    if state in {
        "negated",
        "possible",
        "planned",
        "feared",
        "conditional",
        "historical_comparison",
    }:
        return "unknown"
    if event_type == "success":
        return "positive"
    if event_type in {
        "failure",
        "technical_failure",
        "business_failure",
        "death",
        "serious_injury",
        "critical_condition",
        "hospitalization",
        "intensive_care",
        "permanent_injury",
        "non_survival",
        "damage",
        "displacement",
        "evacuation",
        "abuse",
        "self_harm",
        "personal_tragedy",
    }:
        return "negative"
    return "neutral"


def _composed_harm_outcome(frame: ClauseFrame, atom: LexicalAtom) -> str:
    """Resolve an outcome head with independent severity/permanence modifiers."""

    outcome = atom.lemma
    modifiers = [
        item
        for item in frame.atoms
        if item.category == "harm_modifier" and abs(item.start - atom.start) <= 48
    ]
    modifier_outcomes = {item.lemma for item in modifiers}
    if "death" in modifier_outcomes:
        return "death"
    if "permanent_injury" in modifier_outcomes:
        return "permanent_injury"
    if "critical_condition" in modifier_outcomes and outcome == "serious_injury":
        return "serious_injury"
    if "serious_injury" in modifier_outcomes:
        return "serious_injury"
    return outcome


def _predicate_outcome(frame: ClauseFrame, atom: LexicalAtom) -> str:
    if atom.category == "harm_predicate":
        return _composed_harm_outcome(frame, atom)
    if atom.category in {"medical_state", "medical_destination"}:
        return atom.lemma
    if atom.category == "medical_escalation":
        return "unknown"
    if atom.category == "positive_event":
        if atom.lemma == "recovered" and token_set(frame.clause_span) & {"body", "bodies"}:
            return "death"
        return "success"
    if atom.category == "other":
        return atom.lemma
    if atom.category == "failure_predicate":
        return "failure"
    return "unknown"


def _result_link_before(frame: ClauseFrame, atom: LexicalAtom) -> LexicalAtom | None:
    links = [
        item for item in frame.atoms if item.category == "result_link" and item.end <= atom.start
    ]
    if not links:
        return None
    return sorted(links, key=lambda item: atom.start - item.end)[0]


def _medical_destination_is_result(frame: ClauseFrame, atom: LexicalAtom) -> bool:
    link = _result_link_before(frame, atom)
    if link is None:
        return False
    bridge = normalize_surface(frame.clause_span[link.end : atom.start])
    return bool(re.search(r"\b(?:to|into|in|on|at|for|with)\b", bridge))


def _nearest_human_between(
    frame: ClauseFrame,
    *,
    start: int,
    end: int,
) -> SemanticEntity | None:
    atoms = [
        atom
        for atom in frame.atoms
        if atom.category == "human_head" and atom.start >= start and atom.end <= end
    ]
    if not atoms:
        return None
    return classify_entity_span(sorted(atoms, key=lambda item: item.start)[-1].span)


def _result_complement_binding(
    frame: ClauseFrame,
    atom: LexicalAtom,
) -> tuple[SemanticEntity | None, SemanticEntity | None, tuple[str, ...]]:
    """Bind causal/result syntax independently from the medical outcome vocabulary."""

    link = _result_link_before(frame, atom)
    if link is None:
        return None, None, ()
    affected = _nearest_human_between(frame, start=link.end, end=atom.start)
    if affected is None:
        affected = _nearest_entity(
            frame,
            before=atom.start,
            types={"person", "group_of_people"},
        )
    if affected is None:
        affected = _nearest_entity(
            frame,
            after=atom.end,
            types={"person", "group_of_people"},
        )
    actor = _nearest_entity(frame, before=link.start)
    if actor is None:
        actor = _physical_event(frame)
    return actor, affected, (link.span,)


_HUMAN_ENTITY_TYPES = {"person", "group_of_people"}
_NON_HUMAN_ENTITY_TYPES = {
    "technical_system",
    "software_process",
    "product",
    "company_or_organization",
    "crypto_or_protocol",
    "business_or_financial_object",
    "legal_or_political_object",
    "abstract_concept",
}


def _result_link_atoms(frame: ClauseFrame) -> tuple[LexicalAtom, ...]:
    """Return resultative/causative links, with copulas as a bounded fallback."""

    explicit = tuple(atom for atom in frame.atoms if atom.category == "result_link")
    if explicit:
        return explicit
    return tuple(atom for atom in frame.atoms if atom.category == "copula")


def _role_head_after(frame: ClauseFrame, position: int) -> LexicalAtom | None:
    heads = [
        atom for atom in frame.atoms if atom.category.endswith("_head") and atom.start >= position
    ]
    return sorted(heads, key=lambda item: item.start)[0] if heads else None


def _role_head_before(frame: ClauseFrame, position: int) -> LexicalAtom | None:
    heads = [
        atom for atom in frame.atoms if atom.category.endswith("_head") and atom.end <= position
    ]
    return sorted(heads, key=lambda item: position - item.end)[0] if heads else None


def _clean_result_complement(value: str) -> str | None:
    cleaned = _clean_role(value)
    if cleaned is None:
        return None
    cleaned = re.split(
        r"\s+(?:after|before|during|following|because of|from)\s+(?=(?:a|an|the)\s+)?",
        cleaned,
        maxsplit=1,
        flags=re.I,
    )[0]
    cleaned = re.sub(r"^(?:to|into|in|on|at|with|as)\s+", "", cleaned, flags=re.I)
    return _clean_role(cleaned)


def _fronted_result_complement(frame: ClauseFrame, link: LexicalAtom) -> str | None:
    """Return a comma-delimited state phrase fronted before a copular link."""

    before = frame.clause_span[: link.start].strip()
    if "," not in before:
        return None
    affected_head = _role_head_before(frame, link.start)
    for part in (item.strip() for item in before.split(",")):
        cleaned = _clean_role(part)
        if cleaned is None:
            continue
        if affected_head is not None and affected_head.span in cleaned:
            continue
        if token_set(cleaned) & PHYSICAL_EVENT:
            continue
        if classify_entity_span(cleaned).entity_type != "unknown":
            continue
        return cleaned
    return None


def _result_tail_roles(
    frame: ClauseFrame,
    link: LexicalAtom,
) -> tuple[SemanticEntity | None, str | None]:
    """Split an explicit resultative tail into affected role and state complement."""

    tail = frame.clause_span[link.end :].strip()
    match = _IMPAIRMENT_STRUCTURE.search(tail)
    if match is None:
        return None, None
    affected_span = _clean_role(tail[: match.start()])
    complement = _clean_result_complement(tail[match.start() :])
    if affected_span is None or complement is None:
        return None, None
    affected = classify_entity_span(affected_span).model_copy(update={"clause_id": frame.clause_id})
    if affected.entity_type == "unknown":
        tokens = lexical_tokens(affected_span)
        has_count_or_person_determiner = bool(
            token_set(affected_span)
            & {
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "a",
                "an",
                "the",
            }
        )
        if has_count_or_person_determiner and tokens:
            affected = SemanticEntity(
                clause_id=frame.clause_id,
                text_span=affected_span,
                canonical_name=tokens[-1],
                entity_type="group_of_people",
                confidence="medium",
            )
    return affected, complement


def _result_complement_form(raw_tail: str, complement: str | None) -> str:
    normalized_tail = normalize_surface(raw_tail)
    normalized = normalize_surface(complement or "")
    if not normalized:
        return "unknown"
    if re.match(r"^(?:unable|incapable)\s+to\b|^(?:cannot|can't|could not)\b", normalized, re.I):
        return "infinitival"
    if re.match(r"^(?:with|having)\b", normalized_tail, re.I):
        return "with_state"
    if re.match(r"^(?:dependent|reliant|depending|relying|requiring|required)\b", normalized, re.I):
        return "dependency"
    if " and " in normalized or " or " in normalized:
        return "coordinated"
    if _RESULT_GERUND_HEAD.match(normalized):
        return "gerund"
    if _RESULT_PARTICIPLE_HEAD.match(normalized):
        return "participial"
    if len(lexical_tokens(normalized)) <= 12:
        return "adjectival"
    return "unknown"


def _coordinated_result_tail_roles(
    frame: ClauseFrame,
    tail: str,
) -> tuple[SemanticEntity | None, str | None, str] | None:
    """Bind a coordinated state without letting the first state leak into the affected role."""

    coordinator = re.search(r"\s+(?:and|or)\s+", tail, re.I)
    if coordinator is None:
        return None
    right = _clean_result_complement(tail[coordinator.end() :])
    if right is None or not (
        _IMPAIRMENT_STRUCTURE.search(right)
        or _RESULT_GERUND_HEAD.match(right)
        or _RESULT_PARTICIPLE_HEAD.match(right)
    ):
        return None
    left_surface = tail[: coordinator.start()]
    if re.search(r"\b(?:with|having)\b", left_surface, re.I):
        return None
    left_tokens = lexical_tokens(left_surface)
    if len(left_tokens) < 2:
        return None
    for state_width in range(1, min(4, len(left_tokens))):
        affected_span = " ".join(left_tokens[:-state_width])
        first_state = " ".join(left_tokens[-state_width:])
        affected = classify_entity_span(affected_span).model_copy(
            update={"clause_id": frame.clause_id}
        )
        counted = bool(
            token_set(affected_span)
            & {
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "a",
                "an",
                "some",
                "several",
                "many",
            }
        )
        if affected.entity_type == "unknown" and counted:
            head_tokens = lexical_tokens(affected_span)
            affected = SemanticEntity(
                clause_id=frame.clause_id,
                text_span=affected_span,
                canonical_name=head_tokens[-1] if head_tokens else None,
                entity_type="group_of_people",
                confidence="medium",
            )
        if affected.entity_type not in {"person", "group_of_people"}:
            continue
        first_clean = _clean_result_complement(first_state)
        if first_clean is None or _ordinary_function_control(first_clean):
            continue
        complement = f"{first_clean} and {right}"
        return affected, complement, "coordinated"
    return None


def _structural_result_tail_roles(
    frame: ClauseFrame,
    link: LexicalAtom,
) -> tuple[SemanticEntity | None, str | None, str] | None:
    tail = _clean_role(frame.clause_span[link.end :])
    if tail is None:
        return None
    coordinated = _coordinated_result_tail_roles(frame, tail)
    if coordinated is not None:
        return coordinated
    match = _RESULT_TAIL_FRAME.match(tail)
    if match is None:
        return None
    affected_span = _clean_role(match.group("affected"))
    raw_complement = _clean_role(match.group("complement"))
    complement = _clean_result_complement(raw_complement or "")
    if affected_span is None or complement is None:
        return None
    affected = classify_entity_span(affected_span).model_copy(update={"clause_id": frame.clause_id})
    if affected.entity_type == "unknown":
        head = lexical_tokens(affected_span)[-1] if lexical_tokens(affected_span) else ""
        human_structure = bool(
            _HUMAN_FUNCTION_STRUCTURE.search(complement)
            or re.search(r"\b(?:his|her|their)\b", complement, re.I)
            or re.search(r"(?:ist|ician|ian)$", head, re.I)
        )
        counted = bool(
            token_set(affected_span)
            & {
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "a",
                "an",
                "some",
                "several",
                "many",
            }
        )
        if human_structure and counted:
            affected = SemanticEntity(
                clause_id=frame.clause_id,
                text_span=affected_span,
                canonical_name=head or None,
                entity_type="group_of_people",
                confidence="medium",
            )
    return affected, complement, _result_complement_form(raw_complement or "", complement)


def _result_state_roles(
    frame: ClauseFrame,
    link: LexicalAtom,
) -> tuple[SemanticEntity | None, str | None, str]:
    """Bind the affected role and complete result complement independently from outcome typing."""

    clause = frame.clause_span
    passive = bool(
        re.search(
            r"\b(?:is|are|was|were|be|been|being)\s*$",
            clause[: link.start],
            re.I,
        )
    )
    subject = _role_entity(frame.subject_span, clause_id=frame.clause_id)
    if link.category == "copula" or passive:
        nearest_head = _role_head_before(frame, link.start)
        affected = classify_entity_span(nearest_head.span) if nearest_head is not None else subject
        raw_tail = clause[link.end :]
        complement = _clean_result_complement(raw_tail)
        if complement is None or re.match(
            r"^(?:after|before|during|following|because|from)\b",
            raw_tail.strip(),
            re.I,
        ):
            complement = _fronted_result_complement(frame, link)
        return affected, complement, _result_complement_form(raw_tail, complement)

    structural = _structural_result_tail_roles(frame, link)
    if structural is not None:
        structural_affected, structural_complement, structural_form = structural
        if structural_affected is not None and structural_affected.entity_type != "unknown":
            return structural_affected, structural_complement, structural_form

    head = _role_head_after(frame, link.end)
    if head is not None:
        affected = classify_entity_span(head.span).model_copy(update={"clause_id": frame.clause_id})
        raw_tail = clause[head.end :]
        complement = _clean_result_complement(raw_tail)
        return affected, complement, _result_complement_form(raw_tail, complement)

    if structural is not None:
        return structural

    structural_affected, structural_complement = _result_tail_roles(frame, link)
    if structural_affected is not None and structural_complement is not None:
        raw_tail = clause[link.end :]
        return (
            structural_affected,
            structural_complement,
            _result_complement_form(raw_tail, structural_complement),
        )

    affected = subject
    raw_tail = clause[link.end :]
    complement = _clean_result_complement(raw_tail)
    return affected, complement, _result_complement_form(raw_tail, complement)


def _atoms_in_span(frame: ClauseFrame, span: str | None) -> tuple[LexicalAtom, ...]:
    if not span:
        return ()
    start = frame.clause_span.find(span)
    if start < 0:
        return ()
    end = start + len(span)
    return tuple(atom for atom in frame.atoms if atom.start < end and atom.end > start)


def _ordinary_function_control(complement: str | None) -> bool:
    tokens = token_set(complement or "")
    normalized = normalize_surface(complement or "")
    if tokens & _ORDINARY_FUNCTION_MARKERS:
        return True
    if tokens & _ORDINARY_FUNCTION_VERBS and tokens & _ORDINARY_FUNCTION_MARKERS:
        return True
    if tokens & _ORDINARY_BUSINESS_VERBS and tokens & _ORDINARY_BUSINESS_OBJECTS:
        return True
    if tokens & _ORDINARY_TRANSPORT_HEADS and tokens & {
        "dependent",
        "reliant",
        "depending",
        "relying",
        "requires",
        "requiring",
    }:
        return True
    if any(state == normalized or state in normalized for state in _ORDINARY_SOCIAL_STATES):
        return True
    return any(
        marker in normalized
        for marker in (
            "behind schedule",
            "waiting outside",
            "waiting inside",
            "quiet and withdrawn",
            "socially reserved",
            "preferred option",
            "working remotely",
            "working late",
            "standing by",
        )
    )


def _structural_impairment(complement: str | None) -> bool:
    if not complement or _ordinary_function_control(complement):
        return False
    return bool(_IMPAIRMENT_STRUCTURE.search(complement))


def _result_state_category(
    frame: ClauseFrame,
    affected: SemanticEntity | None,
    complement: str | None,
) -> tuple[str, str]:
    """Classify semantic role of a result complement without deciding candidate policy."""

    if affected is not None and affected.entity_type in _NON_HUMAN_ENTITY_TYPES:
        return "non_human_state", "unknown"
    atoms = _atoms_in_span(frame, complement)
    categories = {atom.category for atom in atoms}
    outcomes = {
        atom.lemma
        for atom in atoms
        if atom.category
        in {
            "harm_predicate",
            "harm_modifier",
            "medical_state",
            "medical_destination",
            "physiological_state",
            "medical_support",
        }
    }
    if "death" in outcomes:
        return "typed_harm", "death"
    if "functional_dependency" in categories and "medical_support" in categories:
        # A support dependency is safety-relevant, but the deterministic layer must not invent a
        # diagnosis or severity from the support device alone. Preserve the bound complement for
        # live provider resolution or candidate-local offline fail-closed behavior.
        return "medical_dependency", "unknown"
    for outcome in (
        "permanent_injury",
        "intensive_care",
        "hospitalization",
        "critical_condition",
        "serious_injury",
    ):
        if outcome in outcomes:
            category = (
                "medical_dependency"
                if "medical_support" in categories or "functional_dependency" in categories
                else "physiological_or_medical"
            )
            return category, outcome
    if "medical_escalation" in categories:
        return "physiological_or_medical", "unknown"
    if "functional_dependency" in categories:
        dependency_object = re.search(
            r"\b(?:on|upon|to)\s+(.+)$",
            complement or "",
            re.I,
        )
        if dependency_object is not None:
            object_entity = classify_entity_span(dependency_object.group(1))
            if object_entity.entity_type in _NON_HUMAN_ENTITY_TYPES | {
                "company_or_organization"
            } or _ordinary_function_control(dependency_object.group(1)):
                return "ordinary_business_or_schedule", "unknown"
        return "unknown", "unknown"
    normalized = normalize_surface(complement or "")
    if (
        "ordinary_result_state" in categories
        and "diminished_state_modifier" in categories
        and affected is not None
        and affected.entity_type in _HUMAN_ENTITY_TYPES
    ):
        return "physiological_or_medical", "unknown"
    if "ordinary_result_state" in categories:
        if any(term in normalized for term in ("waiting", "outside")):
            return "ordinary_location_or_activity", "unknown"
        if any(term in normalized for term in ("schedule", "ready", "available", "open", "closed")):
            return "ordinary_business_or_schedule", "unknown"
        return "ordinary_cognitive_state", "unknown"
    if _ordinary_function_control(complement):
        return "ordinary_business_or_schedule", "unknown"
    if _structural_impairment(complement):
        return "physiological_or_medical", "unknown"
    return "unknown", "unknown"


def inspect_result_state_materiality(
    frame: ClauseFrame,
    candidate: ResultStateCandidate,
) -> ResultStateMateriality:
    """Classify bound result-state materiality without assigning candidate policy."""

    complement = candidate.state_complement_span or ""
    normalized = normalize_surface(complement)
    atoms = _atoms_in_span(frame, complement)
    categories = {atom.category for atom in atoms}
    asserted = not candidate.negated and not candidate.modal and not candidate.conditional
    relation: Literal[
        "inability",
        "loss_or_reduction",
        "assistance_need",
        "support_dependency",
        "abnormal_physiological_state",
        "explicit_medical_outcome",
        "ordinary_activity",
        "ordinary_location",
        "ordinary_business_or_ui_state",
        "unknown",
    ] = "unknown"

    if candidate.affected_entity_type in _NON_HUMAN_ENTITY_TYPES:
        relation = "ordinary_business_or_ui_state"
    elif candidate.state_category in {"typed_harm", "physiological_or_medical"} and (
        candidate.outcome_type != "unknown"
        or categories
        & {
            "harm_predicate",
            "harm_modifier",
            "medical_state",
            "medical_destination",
            "medical_escalation",
            "physiological_state",
        }
    ):
        relation = "explicit_medical_outcome"
    elif candidate.state_category == "medical_dependency" or (
        "functional_dependency" in categories and "medical_support" in categories
    ):
        relation = "support_dependency"
    elif _ordinary_function_control(complement):
        relation = (
            "ordinary_location"
            if _ORDINARY_LOCATION_ROLE.match(normalized)
            or any(token in token_set(normalized) for token in {"outside", "inside"})
            else "ordinary_business_or_ui_state"
        )
    elif _ORDINARY_LOCATION_ROLE.match(normalized):
        relation = "ordinary_location"
    elif re.search(r"\b(?:unable|incapable)\s+to\b|\b(?:cannot|can't|could not)\b", normalized):
        relation = "inability"
    elif re.search(
        r"\b(?:loss|lack|limitation|reduction)\s+of\b|"
        r"\b(?:reduced|limited|impaired)\s+(?:ability|capacity|function|mobility|vision|speech)\b",
        normalized,
        re.I,
    ):
        relation = "loss_or_reduction"
    elif _ASSISTANCE_NEED.search(normalized):
        relation = "assistance_need"
    elif "functional_dependency" in categories:
        relation = "support_dependency"
    elif (
        _HUMAN_FUNCTION_STRUCTURE.search(normalized)
        or _ABNORMAL_STATE_MODIFIER.search(normalized)
        or "diminished_state_modifier" in categories
        or (
            candidate.causal_event_span
            and candidate.complement_form
            in {"adjectival", "participial", "with_state", "coordinated"}
        )
    ):
        relation = "abnormal_physiological_state"
    elif candidate.complement_form in {"gerund", "participial"}:
        relation = "ordinary_activity"
    elif candidate.state_category in {
        "ordinary_location_or_activity",
        "ordinary_business_or_schedule",
        "ordinary_cognitive_state",
    }:
        relation = "ordinary_business_or_ui_state"

    material_relations = {
        "inability",
        "loss_or_reduction",
        "assistance_need",
        "support_dependency",
        "abnormal_physiological_state",
        "explicit_medical_outcome",
    }
    safety_material = bool(
        asserted
        and candidate.affected_entity_type in _HUMAN_ENTITY_TYPES
        and relation in material_relations
    )
    return ResultStateMateriality(
        clause_id=candidate.clause_id,
        affected_entity_type=candidate.affected_entity_type,
        relation_asserted=asserted,
        impairment_relation=relation,
        safety_material=safety_material,
        evidence_spans=tuple(
            dict.fromkeys(
                span
                for span in (
                    candidate.affected_entity_span,
                    candidate.linking_predicate_span,
                    candidate.state_complement_span,
                )
                if span
            )
        ),
    )


def _material_human_result_state(
    frame: ClauseFrame,
    candidate: ResultStateCandidate,
) -> bool:
    """Preserve asserted, non-harmless human result states as possible harm."""

    materiality = inspect_result_state_materiality(frame, candidate)
    return bool(
        materiality.safety_material
        and candidate.outcome_type == "unknown"
        and candidate.state_complement_span
    )


def inspect_human_function_impact(
    frame: ClauseFrame,
    candidate: ResultStateCandidate,
) -> HumanFunctionImpact:
    """Project bound impairment roles without requiring a known diagnosis."""

    materiality = inspect_result_state_materiality(frame, candidate)
    normalized = normalize_surface(candidate.state_complement_span or "")
    tokens = token_set(normalized)
    domain: Literal[
        "sensory",
        "respiratory",
        "alertness",
        "cognitive",
        "speech",
        "mobility",
        "motor",
        "other",
        "unknown",
    ] = "unknown"
    if tokens & {"vision", "hearing", "sight", "ears", "eyes"}:
        domain = "sensory"
    elif tokens & {"breathe", "breathing", "respiration", "oxygen"}:
        domain = "respiratory"
    elif tokens & {"conscious", "consciousness", "alert", "rousable", "awake"}:
        domain = "alertness"
    elif tokens & {"coherent", "incoherent", "confused", "memory", "thinking"}:
        domain = "cognitive"
    elif tokens & {"speech", "speaking", "talk", "voice"}:
        domain = "speech"
    elif tokens & {"balance", "walking", "walk", "mobility", "standing"}:
        domain = "mobility"
    elif tokens & {"movement", "moving", "grip", "coordination"}:
        domain = "motor"
    elif materiality.impairment_relation != "unknown":
        domain = "other"

    impact_by_relation = {
        "inability": "lost",
        "loss_or_reduction": "reduced",
        "assistance_need": "assistance_required",
        "support_dependency": "support_required",
        "abnormal_physiological_state": "impaired",
        "explicit_medical_outcome": "impaired",
        "ordinary_activity": "normal",
        "ordinary_location": "normal",
        "ordinary_business_or_ui_state": "normal",
        "unknown": "unknown",
    }
    return HumanFunctionImpact(
        clause_id=candidate.clause_id,
        function_domain=domain,
        impact=impact_by_relation[materiality.impairment_relation],
        asserted=materiality.relation_asserted,
        safety_material=materiality.safety_material,
        evidence_spans=materiality.evidence_spans,
    )


def extract_result_complement_frames(
    frames: Sequence[ClauseFrame],
) -> tuple[ResultComplementFrame, ...]:
    """Bind clause-local affected/result/complement roles before semantic state typing."""

    output: list[ResultComplementFrame] = []
    for frame in frames:
        for link in _result_link_atoms(frame):
            affected, complement, complement_form = _result_state_roles(frame, link)
            if complement is None:
                continue
            causal = next(
                (
                    atom.span
                    for atom in frame.atoms
                    if atom.category == "physical_event_head" and atom.start < link.start
                ),
                None,
            )
            output.append(
                ResultComplementFrame(
                    clause_id=frame.clause_id,
                    affected_entity_span=affected.text_span if affected else None,
                    affected_entity_type=(affected.entity_type if affected else "unknown"),
                    result_link_span=link.span,
                    complement_span=complement,
                    complement_form=complement_form,
                    assertion_scope=frame.assertion_state,
                    causal_context_span=causal,
                    consumed_spans=tuple(
                        dict.fromkeys(
                            span
                            for span in (
                                link.span,
                                affected.text_span if affected else None,
                            )
                            if span
                        )
                    ),
                )
            )
    return tuple(output)


def extract_result_state_candidates(
    frames: Sequence[ClauseFrame],
) -> tuple[ResultStateCandidate, ...]:
    """Type result-state frames without inventing a diagnosis for unknown complements."""

    frame_by_id = {frame.clause_id: frame for frame in frames}
    candidates: list[ResultStateCandidate] = []
    for result_frame in extract_result_complement_frames(frames):
        frame = frame_by_id[result_frame.clause_id]
        affected = (
            SemanticEntity(
                clause_id=frame.clause_id,
                text_span=result_frame.affected_entity_span,
                canonical_name=(
                    lexical_tokens(result_frame.affected_entity_span)[-1]
                    if lexical_tokens(result_frame.affected_entity_span)
                    else None
                ),
                entity_type=result_frame.affected_entity_type,
                confidence=("medium" if result_frame.affected_entity_type != "unknown" else "low"),
            )
            if result_frame.affected_entity_span
            else None
        )
        complement = result_frame.complement_span
        category, outcome = _result_state_category(frame, affected, complement)
        affected_type = "unknown" if affected is None else affected.entity_type
        candidates.append(
            ResultStateCandidate(
                clause_id=frame.clause_id,
                affected_entity_span=affected.text_span if affected else None,
                affected_entity_type=affected_type,
                linking_predicate_span=result_frame.result_link_span,
                state_complement_span=complement,
                complement_form=result_frame.complement_form,
                assertion_scope=result_frame.assertion_scope,
                causal_event_span=result_frame.causal_context_span,
                state_category=category,
                outcome_type=outcome,
                negated=frame.negated,
                modal=frame.assertion_state in {"possible", "planned", "feared"},
                conditional=frame.conditional,
                attributed=frame.attributed,
                confidence=(
                    "high"
                    if affected is not None
                    and affected.entity_type != "unknown"
                    and category != "unknown"
                    else "medium"
                ),
                consumed_spans=result_frame.consumed_spans,
            )
        )
    return tuple(candidates)


def bind_result_state_relations(
    frames: Sequence[ClauseFrame],
) -> tuple[
    tuple[SemanticEvent, ...],
    tuple[RelationCandidate, ...],
    tuple[ResultStateCandidate, ...],
]:
    """Project typed result states to events and preserve unknown human states for coverage."""

    result_states = extract_result_state_candidates(frames)
    events: list[SemanticEvent] = []
    relations: list[RelationCandidate] = []
    for index, candidate in enumerate(result_states, start=1):
        affected = (
            classify_entity_span(candidate.affected_entity_span).model_copy(
                update={"clause_id": candidate.clause_id}
            )
            if candidate.affected_entity_span
            else None
        )
        asserted = not candidate.negated and not candidate.modal and not candidate.conditional
        typed = candidate.outcome_type != "unknown"
        resolved_without_provider = candidate.state_category in {
            "ordinary_location_or_activity",
            "ordinary_business_or_schedule",
            "ordinary_cognitive_state",
            "non_human_state",
        }
        relation_consumed = tuple(
            dict.fromkeys(
                (
                    *candidate.consumed_spans,
                    *(
                        (candidate.state_complement_span,)
                        if typed or not asserted or resolved_without_provider
                        else ()
                    ),
                )
            )
        )
        relations.append(
            RelationCandidate(
                clause_id=candidate.clause_id,
                relation_kind="event" if typed else "unknown",
                affected_span=candidate.affected_entity_span,
                predicate_span=candidate.linking_predicate_span,
                complement_span=candidate.state_complement_span,
                confidence=candidate.confidence,
                consumed_spans=relation_consumed,
            )
        )
        if not typed or affected is None:
            continue
        event_type = _event_type_for(candidate.outcome_type, affected)
        events.append(
            SemanticEvent(
                clause_id=candidate.clause_id,
                event_id=f"RS{index}",
                predicate=(
                    candidate.linking_predicate_span or candidate.state_complement_span or "state"
                ),
                affected_entity=affected,
                object=(
                    classify_entity_span(candidate.causal_event_span).model_copy(
                        update={"clause_id": candidate.clause_id}
                    )
                    if candidate.causal_event_span
                    else None
                ),
                event_type=event_type,
                polarity=_event_polarity(
                    event_type,
                    "reported" if candidate.attributed else "asserted" if asserted else "possible",
                ),
                explicit_outcome=bool(
                    asserted
                    and event_type
                    in {
                        "death",
                        "serious_injury",
                        "critical_condition",
                        "hospitalization",
                        "intensive_care",
                        "permanent_injury",
                        "non_survival",
                    }
                ),
                assertion_state=(
                    "reported"
                    if candidate.attributed
                    else "negated"
                    if candidate.negated
                    else "conditional"
                    if candidate.conditional
                    else "possible"
                    if candidate.modal
                    else "asserted"
                ),
                evidence_spans=tuple(
                    dict.fromkeys(
                        span
                        for span in (
                            candidate.linking_predicate_span,
                            candidate.state_complement_span,
                        )
                        if span
                    )
                ),
            )
        )
    return tuple(events), tuple(relations), result_states


def _safe_physical_event_usage(frame: ClauseFrame, atom: LexicalAtom) -> bool:
    normalized = normalize_surface(frame.clause_span)
    if any(compound in normalized for compound in _SAFE_EVENT_COMPOUNDS):
        return True
    if any(item.category == "positive_event" for item in frame.atoms):
        event_tail = normalized[atom.start :]
        if re.match(r"(?:outage|blackout|incident)\s+(?:response|recovery|repair)", event_tail):
            return True
        if atom.span.casefold() in {"outage", "blackout", "incident", "crash", "collapse"}:
            return True
    return False


def _implicit_human_outcome(frame: ClauseFrame, atom: LexicalAtom) -> SemanticEntity | None:
    normalized = normalize_surface(frame.clause_span)
    outcome_heads = {
        "life",
        "lives",
        "body",
        "bodies",
        "survivor",
        "survivors",
        "fatality",
        "fatalities",
        "death",
        "deaths",
    }
    tokens = token_set(normalized)
    if tokens & outcome_heads:
        return _implicit_people(atom.span)
    if (
        atom.lemma in {"serious_injury", "death"}
        and re.search(
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|dozens?|hundreds?|\d+)\b",
            normalized,
        )
        and any(item.category == "physical_event_head" for item in frame.atoms)
    ):
        return _implicit_people(atom.span)
    return None


def _unknown_entity(span: str) -> SemanticEntity:
    return SemanticEntity(
        text_span=span,
        canonical_name=normalize_surface(span) or None,
        entity_type="unknown",
        confidence="low",
    )


def _bind_harm_referent(
    frame: ClauseFrame,
    atom: LexicalAtom,
) -> tuple[SemanticEntity | None, SemanticEntity | None, tuple[str, ...]]:
    """Bind actor and affected roles for a harm/medical predicate."""

    subject = _role_entity(frame.subject_span, clause_id=frame.clause_id)
    before = _nearest_entity(frame, before=atom.start)
    after = _nearest_entity(frame, after=atom.end)
    actor: SemanticEntity | None = None
    affected: SemanticEntity | None = None
    binding_spans: tuple[str, ...] = ()
    normalized = normalize_surface(frame.clause_span)
    predicate = normalize_surface(atom.span)

    implicit = _implicit_human_outcome(frame, atom)
    if implicit is not None:
        affected = implicit

    # Passive/result-state frames bind the grammatical subject.
    if subject is not None and subject.entity_type in {"person", "group_of_people"}:
        affected = subject

    result_actor, result_affected, result_spans = _result_complement_binding(frame, atom)
    if result_affected is not None:
        actor = result_actor
        affected = result_affected
        binding_spans = result_spans

    # Active transitive predicates bind the post-predicate argument.
    active_predicates = {
        "kill",
        "killed",
        "injure",
        "injured",
        "wounded",
        "hospitalized",
        "hospitalised",
        "claimed",
        "took",
        "ended",
    }
    passive_voice = bool(
        re.search(r"\b(?:is|are|was|were|be|been|being)\s*$", frame.clause_span[: atom.start], re.I)
    )
    if predicate in active_predicates and after is not None and not passive_voice:
        actor = before or subject
        affected = after

    # ``left <referent> <medical-state/outcome>`` is a result-state relation.
    prefix = frame.clause_span[: atom.start]
    left_match = re.search(r"\bleft\s+(.+?)\s*$", prefix, re.I)
    if left_match and affected is None:
        role = _clean_role(left_match.group(1))
        if role:
            affected = classify_entity_span(role)
            if affected.entity_type == "unknown" and re.search(
                r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|dozens?|hundreds?|\d+)\b",
                role,
                re.I,
            ):
                affected = _implicit_people(role)
        actor = _nearest_entity(frame, before=left_match.start())

    # ``suffered <outcome>`` and similar predicates bind the human subject.
    if (
        affected is None
        and re.search(r"\b(?:suffered|sustained|remains?|was|were)\b", normalized)
        and subject is not None
        and subject.entity_type in {"person", "group_of_people"}
    ):
        affected = subject

    # A non-human subject owns an intransitive death metaphor.
    if (
        subject is not None
        and subject.entity_type not in {"person", "group_of_people", "unknown", "physical_event"}
        and predicate in {"die", "died", "dead", "killed", "kill", "lost"}
    ):
        affected = subject

    # In ``component killed it`` the subject is the actor, while the pronoun referent is unknown.
    if predicate in {"kill", "killed"} and re.search(r"\bkilled?\s+it\b", normalized):
        actor = subject or before
        affected = _unknown_entity("it")

    if affected is None:
        affected = (
            after
            or (subject if subject is not None and subject.entity_type != "unknown" else None)
            or before
            or subject
        )
    return actor, affected, binding_spans


def bind_event_relations(
    frames: Sequence[ClauseFrame],
) -> tuple[tuple[SemanticEvent, ...], tuple[RelationCandidate, ...]]:
    """Compose events from entity, predicate, argument binding, and assertion state."""

    events: list[SemanticEvent] = []
    candidates: list[RelationCandidate] = []
    counter = 0
    for frame in frames:
        material = [
            atom
            for atom in frame.atoms
            if atom.category
            in {
                "harm_predicate",
                "medical_state",
                "medical_destination",
                "medical_escalation",
                "failure_predicate",
                "positive_event",
                "other",
            }
        ]
        selected: list[LexicalAtom] = []
        priority = {
            "harm_predicate": 0,
            "medical_state": 1,
            "medical_destination": 2,
            "medical_escalation": 3,
            "positive_event": 4,
            "other": 5,
            "failure_predicate": 6,
        }
        for atom in sorted(
            material,
            key=lambda item: (item.start, priority[item.category], -(item.end - item.start)),
        ):
            if any(
                existing.start <= atom.start
                and existing.end >= atom.end
                and existing.category != "positive_event"
                for existing in selected
            ):
                continue
            if atom.category == "failure_predicate" and any(
                existing.start <= atom.start and existing.end >= atom.end for existing in selected
            ):
                continue
            selected.append(atom)

        selected = [
            atom
            for atom in selected
            if atom.category != "medical_destination" or _medical_destination_is_result(frame, atom)
        ]

        for atom in selected:
            outcome = _predicate_outcome(frame, atom)
            subject = _role_entity(frame.subject_span, clause_id=frame.clause_id)
            actor: SemanticEntity | None = None
            affected: SemanticEntity | None = None
            object_entity = _physical_event(frame)

            binding_spans: tuple[str, ...] = ()
            if atom.category in {
                "harm_predicate",
                "medical_state",
                "medical_destination",
                "medical_escalation",
            }:
                actor, affected, binding_spans = _bind_harm_referent(frame, atom)
            elif atom.category == "positive_event":
                if outcome == "death":
                    affected = _implicit_human_outcome(frame, atom)
                else:
                    affected = subject or _nearest_entity(frame, before=atom.start)
            elif atom.category == "other":
                actor = subject or _nearest_entity(frame, before=atom.start)
                affected = _nearest_entity(frame, after=atom.end)
                if affected is None or affected.entity_type == "unknown":
                    affected = subject or _nearest_entity(frame, before=atom.start)
            else:
                before_entity = _nearest_entity(frame, before=atom.start)
                if atom.span.casefold() in PHYSICAL_EVENT:
                    compound_owner = (
                        before_entity
                        if before_entity is not None
                        and before_entity.entity_type
                        in {
                            "technical_system",
                            "software_process",
                            "product",
                            "company_or_organization",
                            "crypto_or_protocol",
                            "business_or_financial_object",
                            "legal_or_political_object",
                        }
                        else None
                    )
                    if compound_owner is None and _safe_physical_event_usage(frame, atom):
                        continue
                    object_entity = classify_entity_span(atom.span)
                    affected = compound_owner
                else:
                    affected = subject or before_entity

            event_type = _event_type_for(outcome, affected)
            if outcome == "failure":
                if atom.span.casefold() in PHYSICAL_EVENT and affected is None:
                    event_type = "failure"
                elif affected is not None and affected.entity_type in {
                    "technical_system",
                    "software_process",
                }:
                    event_type = "technical_failure"
                elif affected is not None and affected.entity_type in {
                    "product",
                    "company_or_organization",
                    "crypto_or_protocol",
                    "business_or_financial_object",
                    "legal_or_political_object",
                    "abstract_concept",
                }:
                    event_type = "business_failure"
                else:
                    event_type = "unknown"
            elif outcome == "success":
                event_type = "success"

            event_polarity = _event_polarity(event_type, frame.assertion_state)
            if event_type in {"technical_failure", "business_failure"} and any(
                item.category == "positive_event" for item in frame.atoms
            ):
                event_polarity = "neutral"

            counter += 1
            explicit = bool(
                (
                    atom.category
                    in {
                        "harm_predicate",
                        "medical_state",
                        "medical_destination",
                    }
                    or (atom.category == "positive_event" and outcome == "death")
                )
                and event_type
                in {
                    "death",
                    "serious_injury",
                    "critical_condition",
                    "hospitalization",
                    "intensive_care",
                    "permanent_injury",
                    "non_survival",
                }
                and frame.assertion_state in {"asserted", "reported"}
            )
            modifier_spans = (
                tuple(
                    item.span
                    for item in frame.atoms
                    if item.category == "harm_modifier" and abs(item.start - atom.start) <= 48
                )
                if atom.category == "harm_predicate"
                else ()
            )
            evidence_spans = tuple(dict.fromkeys((*modifier_spans, atom.span)))
            event = SemanticEvent(
                clause_id=frame.clause_id,
                event_id=f"E{counter}",
                predicate=normalize_surface(atom.span),
                actor=actor,
                affected_entity=affected,
                object=object_entity,
                event_type=event_type,
                polarity=event_polarity,
                explicit_outcome=explicit,
                assertion_state=frame.assertion_state,
                evidence_spans=evidence_spans,
            )
            events.append(event)
            candidates.append(
                RelationCandidate(
                    clause_id=frame.clause_id,
                    relation_kind="event",
                    actor_span=actor.text_span if actor else None,
                    affected_span=affected.text_span if affected else None,
                    predicate_span=atom.span,
                    confidence=(
                        "high"
                        if affected is not None and affected.entity_type != "unknown"
                        else "medium"
                    ),
                    consumed_spans=(
                        ()
                        if atom.category == "medical_escalation"
                        else tuple(dict.fromkeys((*binding_spans, *modifier_spans, atom.span)))
                    ),
                )
            )

    unique: list[SemanticEvent] = []
    seen: set[tuple[str, str, str, str]] = set()
    for event in events:
        key = (
            event.event_type,
            normalize_surface(event.predicate),
            normalize_surface(event.affected_entity.text_span) if event.affected_entity else "",
            event.assertion_state,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return tuple(unique), tuple(candidates)


def _source_alignment(
    target: SemanticEntity | None,
    source_entities: Sequence[SemanticEntity],
    source_events: Sequence[SemanticEvent],
    *,
    allow_targetless: bool,
) -> str | None:
    negative = [event for event in source_events if event.polarity == "negative"]
    if target is None:
        if not allow_targetless or not negative:
            return None
        explicit_harm = [event for event in negative if event.explicit_outcome]
        return (explicit_harm or negative)[0].event_id
    target_tokens = token_set(target.text_span)
    for event in negative:
        relation_entities = tuple(
            entity
            for entity in (event.actor, event.affected_entity, event.object)
            if entity is not None
        )
        if any(target_tokens & token_set(entity.text_span) for entity in relation_entities):
            return event.event_id
        if (
            target.entity_type == "physical_event"
            and event.object is not None
            and event.object.entity_type == "physical_event"
        ):
            return event.event_id
        # Infrastructure aliases can target the source failure event without substring identity.
        if (target.canonical_name or "") in {
            "grid",
            "network",
            "system",
            "power",
            "service",
        } and (
            event.event_type in {"failure", "technical_failure"}
            or (event.object is not None and event.object.entity_type == "physical_event")
        ):
            return event.event_id
    for entity in source_entities:
        if target_tokens & token_set(entity.text_span):
            linked = [
                event
                for event in source_events
                if entity in {event.actor, event.affected_entity, event.object}
            ]
            if linked:
                return linked[0].event_id
            if negative:
                explicit_harm = [event for event in negative if event.explicit_outcome]
                return (explicit_harm or negative)[0].event_id
    return None


def _target_after_preposition(clause: str, atom: LexicalAtom) -> str | None:
    tail = clause[atom.end :]
    match = re.search(r"\b(?:from|by|for|to|of)\s+([^.!?;,:]{1,80})", tail, re.I)
    if match:
        return _clean_role(match.group(1))
    return None


def _target_before_relation(clause: str, atom: LexicalAtom) -> str | None:
    head = _clean_role(clause[: atom.start])
    if not head:
        return None
    # Keep the nearest noun phrase, not an entire preceding clause.
    parts = re.split(r"[,;:]|\b(?:and|but|while)\b", head, flags=re.I)
    return _clean_role(parts[-1])


def _evaluation_target_span(frame: ClauseFrame, atoms: Sequence[LexicalAtom]) -> str | None:
    """Bind an evaluation target independently from evaluation classification."""

    ordered = sorted(atoms, key=lambda item: item.start)
    normalized_clause = normalize_surface(frame.clause_span)
    preposed = re.match(r"^(?:for|to|of)\s+(.+?)\s*:\s*", frame.clause_span, re.I)
    if preposed is not None:
        target = _clean_role(preposed.group(1))
        if target:
            return target
    relation_verbs = {
        "deserves",
        "earned",
        "earns",
        "wins",
        "won",
        "should get",
        "nailed",
        "aced",
        "crushed",
        "delivered",
        "outdid",
        "showed",
        "gets",
        "got",
        "receives",
        "received",
        "takes",
        "took",
    }
    verb = next(
        (atom for atom in ordered if normalize_surface(atom.span) in relation_verbs),
        None,
    )
    if verb is not None:
        target = _target_before_relation(frame.clause_span, verb)
        if target:
            target = re.sub(
                r"\b(?:gets?|got|receives?|received|takes?|took|earns?|earned)\s*$",
                "",
                target,
                flags=re.I,
            )
            return target

    imperative = re.match(
        r"^(?:take\s+a\s+bow|hats\s+off|bravo|congratulations|congrats)\s*[,;:]?\s*(?:to\s+)?(.+)$",
        frame.clause_span,
        re.I,
    )
    if imperative is not None:
        target = _clean_role(imperative.group(1))
        if target:
            return target

    # Explicit target prepositions bind the following noun phrase.
    for atom in ordered:
        target = _target_after_preposition(frame.clause_span, atom)
        if target:
            return target

    # Beneficiary relations bind the phrase immediately before ``needed``.
    needed = next((atom for atom in ordered if normalize_surface(atom.span) == "needed"), None)
    if needed is not None:
        head = _clean_role(frame.clause_span[: needed.start])
        if head:
            head = re.sub(r"^(?:exactly|just|precisely)\s+what\s+", "", head, flags=re.I)
            return _clean_role(head)

    # Comparative admiration binds the object of the comparison.
    better = next((atom for atom in ordered if normalize_surface(atom.span) == "better"), None)
    if better is not None:
        tail = _clean_role(frame.clause_span[better.end :])
        if tail:
            return tail

    # Copular praise binds its grammatical subject.

    if frame.subject_span and any(
        normalize_surface(atom.span) in POSITIVE_MODIFIERS
        or "proud" in normalize_surface(atom.span)
        for atom in ordered
    ):
        return frame.subject_span
    if normalized_clause.startswith("here's to "):
        return _clean_role(frame.clause_span[len("here's to ") :])
    return None


def _evaluation_head_is_material(frame: ClauseFrame, atom: LexicalAtom) -> bool:
    """Require an evaluation frame around otherwise ambiguous nouns such as ``performance``."""

    normalized = normalize_surface(frame.clause_span)
    head = normalize_surface(atom.span)
    if head in _RECOGNITION_RELATIONS:
        return bool(
            re.search(
                rf"(?:^|\b)(?:take\s+a\s+|give\s+|gets?\s+|got\s+|receives?\s+|"
                rf"earned?\s+|deserves?\s+|to\s+|for\s+|from\s+|by\s+)?{re.escape(head)}\b",
                normalized,
                re.I,
            )
        )
    if re.search(rf"\bwhat\s+(?:a|an)\s+[^.!?;]{{0,40}}\b{re.escape(head)}\b", normalized):
        return True
    if any(
        item.category in {"positive_evaluation", "reward_relation", "rating_relation"}
        and item is not atom
        for item in frame.atoms
    ):
        return True
    return bool(
        re.search(
            rf"\b{re.escape(head)}\b\s+(?:from|by|for|of)\s+",
            normalized,
            re.I,
        )
    )


def _evaluation_type(atoms: Sequence[LexicalAtom]) -> tuple[str, str]:
    normalized = {normalize_surface(atom.span) for atom in atoms}
    tokens = token_set(" ".join(atom.span for atom in atoms))
    if tokens & {"applause", "ovation", "cheers", "toast"}:
        return "celebration", "celebration"
    if any(atom.category == "rating_relation" for atom in atoms):
        return "rating", "rating"
    if tokens & {"win", "victory"} or normalized & set(SUCCESS_PREDICATES):
        return "success_attribution", "success_attribution"
    if tokens & {"triumph", "celebration"}:
        return "celebration", "celebration"
    if any("proud" in value for value in normalized):
        return "admiration", "admiration"
    if normalized & {"love", "needed"}:
        return "praise", "beneficiary"
    if any(atom.category == "reward_relation" for atom in atoms):
        if tokens & {"credit", "recognition", "honor", "honors", "honour", "honours"}:
            return "admiration", "admiration"
        return "reward", "reward"
    if "timing" in tokens:
        return "praise", "positive_timing"
    return "praise", "praise"


def bind_evaluation_relations(
    frames: Sequence[ClauseFrame],
    source_entities: Sequence[SemanticEntity],
    source_events: Sequence[SemanticEvent],
) -> tuple[tuple[EvaluationRelation, ...], tuple[RelationCandidate, ...]]:
    """Compose evaluations from a typed head, polarity, target role, and source alignment."""

    relations: list[EvaluationRelation] = []
    candidates: list[RelationCandidate] = []
    for frame in frames:
        # A negative normative result proposition may contain a celebration or reward noun as
        # its grammatical object.  In that role the noun is being rejected, not praised.
        if _normative_response_frame(frame):
            continue
        eval_atoms = [
            atom
            for atom in frame.atoms
            if atom.category
            in {
                "positive_evaluation",
                "reward_relation",
                "rating_relation",
                "evaluation_head",
            }
            and (atom.category != "evaluation_head" or _evaluation_head_is_material(frame, atom))
        ]
        if not eval_atoms:
            continue
        ordered = sorted(eval_atoms, key=lambda item: item.start)
        start = min(item.start for item in ordered)
        end = max(item.end for item in ordered)
        evidence = frame.clause_span[start:end]
        target_span = _evaluation_target_span(frame, ordered)
        target = classify_entity_span(target_span) if target_span else None
        evaluation_type, subtype = _evaluation_type(ordered)
        allow_targetless = bool(
            any(
                normalize_surface(item.span)
                in {"chef's kiss", "mission accomplished", "nailed", "aced", "crushed"}
                for item in ordered
            )
        )
        aligned = _source_alignment(
            target,
            source_entities,
            source_events,
            allow_targetless=allow_targetless,
        )
        asserted_source = [
            event for event in source_events if event.assertion_state in {"asserted", "reported"}
        ]
        has_negative = any(event.polarity == "negative" for event in asserted_source)
        has_positive = any(event.polarity == "positive" for event in asserted_source)
        literal = bool(has_positive and not has_negative)
        relation = EvaluationRelation(
            clause_id=frame.clause_id,
            target_entity=target,
            target_event_id=aligned,
            evaluation_type=evaluation_type,
            relation_subtype=subtype,
            polarity="positive",
            literal_reading_plausible=literal,
            confidence="high" if aligned or literal else "medium",
            evidence_spans=(evidence,),
        )
        relations.append(relation)
        candidates.append(
            RelationCandidate(
                clause_id=frame.clause_id,
                relation_kind="evaluation",
                target_span=target_span,
                predicate_span=evidence,
                confidence=relation.confidence,
                consumed_spans=tuple(
                    dict.fromkeys(
                        (*((target_span,) if target_span else ()), *(atom.span for atom in ordered))
                    )
                ),
            )
        )
    unique: list[EvaluationRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in relations:
        key = (
            relation.evaluation_type,
            relation.relation_subtype,
            normalize_surface(relation.target_entity.text_span) if relation.target_entity else "",
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(relation)
    return tuple(unique), tuple(candidates)


def _coordinated_tail_start(frame: ClauseFrame) -> int | None:
    """Return a comma boundary that starts a new pronominal clause."""

    for match in re.finditer(r",\s+(?=(?:i|you|he|she|it|we|they)\b)", frame.clause_span, re.I):
        return match.start()
    return None


def _atom_in_coordinated_tail(frame: ClauseFrame, atom: LexicalAtom) -> bool:
    boundary = _coordinated_tail_start(frame)
    return boundary is not None and atom.start > boundary


def _evidential_qualification_frame(frame: ClauseFrame) -> bool:
    """Recognize passive evidence/status uncertainty before shell interpretation."""

    normalized = normalize_surface(frame.clause_span)
    passive = re.search(
        r"\b(?:has|have|had|is|are|was|were)\s+(?:(?:not|never)\s+)?been\s+"
        r"(?P<participle>[^\W\d_]+(?:ed|en))\b",
        normalized,
        re.I,
    )
    if passive is not None and _EVIDENTIAL_PARTICIPLE.fullmatch(passive.group("participle")):
        return True
    if re.search(
        r"\b(?:need|needs|needed|require|requires|required)\s+"
        r"(?:confirmation|verification|corroboration|evidence|review)\b",
        normalized,
        re.I,
    ):
        return True
    if re.search(
        r"\b(?:remains?|is|are|was|were)\s+(?:the\s+subject\s+of\s+|under\s+)",
        normalized,
        re.I,
    ) and _EVIDENTIAL_PROCESS_HEAD.search(normalized):
        return True
    return bool(
        re.search(
            r"\b(?:remains?|is|are|was|were)\s+(?:unconfirmed|unverified|unclear|unknown|"
            r"undetermined|unresolved|not established)\b",
            normalized,
            re.I,
        )
    )


def _factual_assertion_frame(frame: ClauseFrame) -> bool:
    """Recognize a bounded neutral factual action with subject, predicate, and object roles."""

    if any(
        atom.category
        in {
            "positive_evaluation",
            "reward_relation",
            "rating_relation",
            "packaging_modifier",
            "reduction_operator",
            "reveal_operator",
        }
        for atom in frame.atoms
    ):
        return False
    normalized = normalize_surface(frame.clause_span)
    action = _FACTUAL_ACTION.search(normalized)
    if action is None:
        return False
    prefix = normalized[: action.start()].strip()
    suffix = normalized[action.end() :].strip()
    if not prefix or not suffix:
        return False
    # A bare determiner before an action-shaped noun ("a report for ...") is a noun phrase,
    # not a factual assertion.  Require a lexical subject role before accepting the action frame.
    subject_tokens = [
        token
        for token in lexical_tokens(prefix)
        if token not in {"a", "an", "the", "one", "this", "that"}
    ]
    return bool(subject_tokens)


def _finite_intransitive_roles(frame: ClauseFrame) -> tuple[str, str] | None:
    """Bind a complete subject plus finite intransitive predicate proposition."""

    match = re.fullmatch(
        r"(?:(?:and|but|yet)\s+)?(?P<subject>[^.!?;,:]{1,120}?)\s+"
        r"(?:still\s+)?(?P<predicate>[^\W\d_]{2,40}(?:s|ed))",
        normalize_surface(frame.clause_span),
        re.I | re.UNICODE,
    )
    if match is None:
        return None
    subject = match.group("subject").strip()
    predicate = match.group("predicate").strip()
    if not lexical_tokens(subject):
        return None
    return subject, predicate


def _complete_copular_literal(frame: ClauseFrame) -> bool:
    """Accept role-complete copular status while excluding evaluative/shell operators."""

    return bool(
        frame.subject_span
        and frame.predicate_span
        and (frame.object_span or frame.complement_span)
        and any(atom.category == "copula" for atom in frame.atoms)
        and not any(
            atom.category
            in {
                "positive_evaluation",
                "negative_evaluation",
                "evaluation_head",
                "reward_relation",
                "rating_relation",
                "packaging_modifier",
                "reduction_operator",
                "reveal_operator",
            }
            for atom in frame.atoms
        )
    )


def _direct_remediation_frame(frame: ClauseFrame) -> bool:
    """Recognize a complete necessity or action-recommendation proposition."""

    normalized = normalize_surface(frame.clause_span)
    if any(
        atom.category
        in {
            "positive_evaluation",
            "reward_relation",
            "rating_relation",
            "packaging_modifier",
            "reduction_operator",
            "reveal_operator",
        }
        for atom in frame.atoms
    ):
        return False

    predicate = _REMEDIATION_PREDICATE.search(normalized)
    if predicate is not None and _REMEDIATION_OBJECT.search(normalized):
        subject = normalized[: predicate.start()].strip()
        object_span = normalized[predicate.end() :].strip()
        if subject and object_span and lexical_tokens(subject) and lexical_tokens(object_span):
            return True

    recommendation = _ACTION_RECOMMENDATION.fullmatch(normalized)
    if recommendation is None:
        return False
    return all(
        lexical_tokens(recommendation.group(role)) for role in ("subject", "action", "object")
    )


def _literal_acknowledgement_frame(frame: ClauseFrame) -> bool:
    """Recognize a complete literal agreement/acknowledgement proposition."""

    match = _ACKNOWLEDGEMENT_PREDICATE.fullmatch(normalize_surface(frame.clause_span))
    return bool(match is not None and lexical_tokens(match.group("subject")))


def _normative_response_frame(frame: ClauseFrame) -> bool:
    """Recognize a complete causal/normative response proposition.

    The rule binds a proposition subject, a causative predicate, an affected response/action
    object, and a negative normative state.  It is intentionally role-based rather than tied to
    any one incident, celebration noun, or holdout sentence.
    """

    match = _NORMATIVE_RESULT_PREDICATE.fullmatch(normalize_surface(frame.clause_span))
    if match is None:
        return False
    return all(lexical_tokens(match.group(role)) for role in ("subject", "object", "state"))


def _negated_factual_assertion_frame(frame: ClauseFrame) -> bool:
    return bool(frame.negated and _factual_assertion_frame(frame))


def _literal_control(frame: ClauseFrame) -> bool:
    normalized = normalize_surface(frame.clause_span)
    if (
        _evidential_qualification_frame(frame)
        or _factual_assertion_frame(frame)
        or _direct_remediation_frame(frame)
        or _literal_acknowledgement_frame(frame)
        or _normative_response_frame(frame)
    ):
        return True
    has_state_predicate = any(
        atom.category in {"state_predicate", "positive_evaluation", "negative_evaluation"}
        and not _atom_in_coordinated_tail(frame, atom)
        for atom in frame.atoms
    )
    has_shell_operator = any(
        atom.category in {"packaging_modifier", "reduction_operator", "reveal_operator"}
        and not _atom_in_coordinated_tail(frame, atom)
        for atom in frame.atoms
    )
    if (
        has_state_predicate
        and any(atom.category == "copula" for atom in frame.atoms)
        and not has_shell_operator
    ):
        return True
    if _QUOTES.search(frame.clause_span):
        return True
    strip_match = re.match(r"^strip\s+away\s+(.+?)\s*:", frame.clause_span, re.I)
    if strip_match is not None and not _decoration_role(strip_match.group(1)):
        return True
    tokens = token_set(normalized)
    formula = {"sum", "ratio", "divided", "multiplied", "velocity", "distance", "equation"}
    definition = {
        "triangle",
        "polygon",
        "checksum",
        "definition",
        "protocol",
        "method",
        "system",
        "value",
        "used",
    }
    if tokens & formula:
        return True
    if "equals" in tokens and not any(
        atom.category in {"packaging_modifier", "reduction_operator"} for atom in frame.atoms
    ):
        decoration_tail = re.search(
            r"\b(?:plus|with|but)\s+(?:a\s+|an\s+|the\s+)?(.+)$",
            frame.clause_span,
            re.I,
        )
        if decoration_tail is None or not _decoration_role(decoration_tail.group(1)):
            return True
    comparative = re.search(
        r"\b(?:more|less|better|worse|harder|easier|faster|slower|larger|smaller)\b"
        r"[^.!?;]{0,120}\bthan\b",
        normalized,
        re.I,
    )
    if (
        comparative is not None
        and not re.match(r"^(?:nothing|few\s+things)\b", normalized, re.I)
        and not any(
            atom.category in {"packaging_modifier", "reduction_operator", "reveal_operator"}
            for atom in frame.atoms
        )
    ):
        return True
    if (
        tokens & definition
        and (frame.predicate_span is not None or frame.complement_span is not None)
        and not any(atom.category == "packaging_modifier" for atom in frame.atoms)
    ):
        return True
    subject_entity = _role_entity(frame.subject_span, clause_id=frame.clause_id)
    return bool(
        subject_entity is not None
        and subject_entity.entity_type == "legal_or_political_object"
        and not any(
            atom.category in {"packaging_modifier", "reduction_operator", "reveal_operator"}
            for atom in frame.atoms
        )
    )


def _strip_reveal_surface(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(
        r"^(?:as\s+it\s+turns\s+out|turns?\s+out|apparently|so\s+apparently|who\s+knew|surprise)\s*,?\s*",
        "",
        value,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+all\s+along$", "", cleaned, flags=re.I)
    cleaned = re.sub(r",?\s+in\s+other\s+words$", "", cleaned, flags=re.I)
    return _clean_role(cleaned)


def _decoration_role(value: str) -> bool:
    tokens = token_set(value)
    return bool(
        tokens
        & {
            "branding",
            "brand",
            "ux",
            "wrapper",
            "paint",
            "strategy",
            "clothing",
            "costume",
            "makeover",
            "lipstick",
            "progress",
            "bar",
            "fancier",
            "fancy",
            "glorified",
            "rebranded",
            "repackaged",
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
            "window",
            "dressing",
        }
    )


def _fronted_relation_roles(
    frame: ClauseFrame,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Parse a fronted cosmetic/reduction phrase independently from the copular relation."""

    match = re.match(
        r"^(?P<cues>with|behind|beneath|underneath|under|for\s+all|past|beyond)\s+"
        r"(?P<decoration>[^,:;]{1,80})\s*[,;:]\s*(?P<relation>.+)$",
        frame.clause_span,
        re.I,
    )
    if match is None or not _decoration_role(match.group("decoration")):
        return None, None, None, None, None
    subject, complement, predicate = _parse_copular_roles(match.group("relation"))
    if not subject or not complement:
        return None, None, None, None, None
    cue = normalize_surface(match.group("cues"))
    operator = "decorated_identity" if cue == "with" else "reduction"
    return subject, complement, predicate, _clean_role(match.group("decoration")), operator


def _expanded_decoration_span(frame: ClauseFrame, atom: LexicalAtom) -> str:
    """Attach a cosmetic atom to its local punctuation/preposition phrase."""

    left = frame.clause_span.rfind(",", 0, atom.start)
    semicolon = frame.clause_span.rfind(";", 0, atom.start)
    boundary = max(left, semicolon)
    if boundary >= 0:
        candidate = _clean_role(frame.clause_span[boundary + 1 :])
        if candidate and _decoration_role(candidate):
            return candidate
    prefix = frame.clause_span[: atom.start]
    connector = list(
        re.finditer(
            r"\b(?:with|plus|but|in|under|behind|wearing)\s+[^,;:]{0,80}$",
            prefix,
            re.I,
        )
    )
    if connector:
        candidate = _clean_role(frame.clause_span[connector[-1].start() :])
        if candidate and _decoration_role(candidate):
            return candidate
    return atom.span


def _parse_copular_roles(text: str) -> tuple[str | None, str | None, str | None]:
    match = re.search(
        r"\b(?:is|are|was|were|equals|equaled|equalled|amounts?\s+to|boils?\s+down\s+to|comes?\s+down\s+to)\b",
        text,
        re.I,
    )
    if match is None:
        return None, None, None
    return (
        _clean_role(text[: match.start()]),
        _clean_role(text[match.end() :]),
        match.group(0),
    )


def _setup_twist_shell(
    frame: ClauseFrame,
    trailing: ClauseFrame | None = None,
) -> SemanticReplyShell | None:
    """Bind a setup/contrast relation from clause roles rather than a topic phrase."""

    evidence: tuple[str, ...]
    if trailing is not None:
        connector_match = re.match(
            r"^(?:but|then|actually|apparently|somehow)\s+(.+)$",
            trailing.clause_span,
            re.I,
        )
        if connector_match is None:
            return None
        trailing_content = connector_match.group(1)
        if trailing.atoms and all(atom.category == "packaging_modifier" for atom in trailing.atoms):
            return None
        subject = _clean_role(frame.clause_span)
        complement = _clean_role(trailing_content)
        evidence = (frame.clause_span, trailing.clause_span)
        opener = normalize_surface(trailing.clause_span.split(maxsplit=1)[0])
    else:
        plot_match = re.match(r"^plot\s+twist\s*:\s*(.+)$", frame.clause_span, re.I)
        if plot_match is not None:
            subject = "setup"
            complement = _clean_role(plot_match.group(1))
            evidence = (frame.clause_span,)
            opener = "plot_twist"
        else:
            explicit_match = re.match(
                r"^(.+?)\s+(but\s+(?:make|call)\s+it)\s+(.+)$",
                frame.clause_span,
                re.I,
            )
            connector_match = explicit_match or re.match(
                r"^(.+?)\s*[,;—-]\s*(but|then|actually|apparently|somehow)\s+(.+)$",
                frame.clause_span,
                re.I,
            )
            if connector_match is None:
                return None
            tail_atoms = extract_lexical_atoms(connector_match.group(3))
            if tail_atoms and all(atom.category == "packaging_modifier" for atom in tail_atoms):
                return None
            subject = _clean_role(connector_match.group(1))
            complement = _clean_role(connector_match.group(3))
            evidence = (frame.clause_span,)
            opener = normalize_surface(connector_match.group(2)).replace(" ", "_")
    if not subject or not complement:
        return None
    return SemanticReplyShell(
        clause_id=frame.clause_id,
        operator="setup_twist",
        subject_span=subject,
        complement_span=complement,
        opener_family=opener,
        generic_reusability=True,
        evidence_spans=evidence,
    )


def _inverse_praise_shell(frame: ClauseFrame) -> SemanticReplyShell | None:
    match = re.match(
        r"^(?:nothing|few\s+things)\s+"
        r"(?:says?|screams?|captures?|embod(?:y|ies)|expresses?|signals?|communicates?|demonstrates?|shows?)\s+"
        r"(.+?)\s+(?:quite\s+)?(?:like|better\s+than)\s+(.+)$",
        frame.clause_span,
        re.I,
    )
    if match is None:
        return None
    return SemanticReplyShell(
        clause_id=frame.clause_id,
        operator="inverse_praise",
        subject_span=_clean_role(match.group(1)),
        complement_span=_clean_role(match.group(2)),
        opener_family="inverse_praise",
        generic_reusability=True,
        evidence_spans=(frame.clause_span,),
    )


def bind_reply_shell_relations(
    frames: Sequence[ClauseFrame],
) -> tuple[tuple[SemanticReplyShell, ...], tuple[RelationCandidate, ...]]:
    """Compose reply shells from canonical operators and bound role slots."""

    shells: list[SemanticReplyShell] = []
    candidates: list[RelationCandidate] = []
    consumed_frame_indexes: set[int] = set()
    for index, frame in enumerate(frames):
        if index in consumed_frame_indexes:
            continue
        if _literal_control(frame) or frame.clause_span.rstrip().endswith("?"):
            continue

        trailing = frames[index + 1] if index + 1 < len(frames) else None
        setup = _setup_twist_shell(frame, trailing) if trailing is not None else None
        if setup is None:
            setup = _setup_twist_shell(frame)
        if setup is not None:
            shells.append(setup)
            candidates.append(
                RelationCandidate(
                    clause_id=frame.clause_id,
                    relation_kind="setup_twist",
                    actor_span=setup.subject_span,
                    complement_span=setup.complement_span,
                    confidence="high",
                    consumed_spans=setup.evidence_spans,
                )
            )
            if trailing is not None and len(setup.evidence_spans) == 2:
                consumed_frame_indexes.add(index + 1)
            continue

        inverse = _inverse_praise_shell(frame)
        if inverse is not None:
            shells.append(inverse)
            candidates.append(
                RelationCandidate(
                    clause_id=frame.clause_id,
                    relation_kind="inverse_praise",
                    actor_span=inverse.subject_span,
                    complement_span=inverse.complement_span,
                    confidence="high",
                    consumed_spans=(frame.clause_span,),
                )
            )
            continue

        packaging = [
            atom
            for atom in frame.atoms
            if atom.category == "packaging_modifier" and not _atom_in_coordinated_tail(frame, atom)
        ]
        reduction = [
            atom
            for atom in frame.atoms
            if atom.category == "reduction_operator"
            and not _atom_in_coordinated_tail(frame, atom)
            and normalize_surface(atom.span)
            not in {"glorified", "fancy", "rebranded", "repackaged"}
        ]
        reveal = [
            atom
            for atom in frame.atoms
            if atom.category == "reveal_operator" and not _atom_in_coordinated_tail(frame, atom)
        ]

        subject = _strip_reveal_surface(frame.subject_span)
        complement = _strip_reveal_surface(frame.complement_span or frame.object_span)
        predicate = frame.predicate_span
        decoration: str | None = (
            _expanded_decoration_span(frame, packaging[0]) if packaging else None
        )
        marker: str | None = normalize_surface(packaging[0].span) if packaging else None
        forced_operator: str | None = None

        (
            fronted_subject,
            fronted_complement,
            fronted_predicate,
            fronted_decoration,
            fronted_operator,
        ) = _fronted_relation_roles(frame)
        if fronted_subject and fronted_complement:
            subject = fronted_subject
            complement = fronted_complement
            predicate = fronted_predicate
            decoration = fronted_decoration
            forced_operator = fronted_operator
            marker = normalize_surface(fronted_decoration or "") or None

        # ``strip away <packaging>: X is Y`` is an explicit reduction grammar. A physical or
        # factual removal (e.g. insulation) remains a literal control.
        strip_match = re.match(r"^strip\s+away\s+(.+?)\s*:\s*(.+)$", frame.clause_span, re.I)
        if strip_match is not None:
            removed, relation_text = strip_match.groups()
            if not _decoration_role(removed):
                continue
            subject, complement, predicate = _parse_copular_roles(relation_text)
            if not subject or not complement:
                continue
            reduction = [
                LexicalAtom(
                    span="strip away",
                    lemma="strip away",
                    category="reduction_operator",
                    start=0,
                    end=10,
                )
            ]
            packaging = []
            decoration = None
            marker = None

        # Colon identity/repackaging frames bind left and right roles directly.
        if predicate == ":" and strip_match is None:
            subject = _clean_role(frame.subject_span)
            complement = _clean_role(frame.complement_span)

        if (
            subject
            and complement
            and index + 1 < len(frames)
            and normalize_surface(frames[index + 1].clause_span).startswith("but ")
            and any(atom.category == "packaging_modifier" for atom in frames[index + 1].atoms)
        ):
            trailing = frames[index + 1]
            packaging = [
                *packaging,
                *(atom for atom in trailing.atoms if atom.category == "packaging_modifier"),
            ]
            decoration = trailing.clause_span
            marker = normalize_surface(packaging[-1].span)

        if not subject or not complement:
            # A trailing ``but fancier`` fragment decorates the preceding copular clause.
            if packaging and index > 0 and normalize_surface(frame.clause_span).startswith("but "):
                previous = frames[index - 1]
                if previous.subject_span and previous.complement_span:
                    subject = previous.subject_span
                    complement = previous.complement_span
                    decoration = frame.clause_span
                    marker = normalize_surface(packaging[0].span)
                else:
                    continue
            else:
                continue

        relation_has_state = any(
            atom.category in {"state_predicate", "positive_evaluation", "negative_evaluation"}
            and not _atom_in_coordinated_tail(frame, atom)
            for atom in frame.atoms
        )
        if relation_has_state and not (reduction or packaging or reveal):
            continue

        # A connector introduces an independent decoration role when its tail is cosmetic.
        connector = re.search(r"\b(?:with|plus|but)\s+(.+)$", complement, re.I)
        if connector and _decoration_role(connector.group(1)):
            decoration = _clean_role(connector.group(1))
            marker = marker or normalize_surface(connector.group(0).split()[0])

        operator: str = "none"
        opener: str | None = None
        if forced_operator is not None:
            operator = forced_operator
        elif reveal:
            operator = "reveal"
            opener = normalize_surface(reveal[0].span).replace(" ", "_")
        elif reduction:
            operator = "reduction"
            marker = None
            decoration = None
        elif packaging or decoration:
            operator = (
                "repackaging"
                if marker
                and token_set(marker)
                & {"rebranded", "repackaged", "renamed", "branding", "paint", "wrapper"}
                else "decorated_identity"
            )
        elif predicate and normalize_surface(predicate) in {"equals", "equaled", "equalled"}:
            # Semantic ``equals`` with an explicit cosmetic tail is decorated identity; literal
            # arithmetic/definitions were filtered above.
            operator = "decorated_identity" if decoration else "identity"
        elif predicate and normalize_surface(predicate) in {
            "is",
            "are",
            "was",
            "were",
            "amounts to",
            "boils down to",
            "comes down to",
        }:
            operator = "identity"

        if operator == "none":
            continue

        complement_clean = complement
        if decoration:
            decoration_pattern = re.escape(decoration)
            if re.match(r"^(?:with|plus|but|in|under|behind|wearing)\b", decoration, re.I):
                complement_clean = re.sub(
                    r"(?:[,;:]\s*|\s+)" + decoration_pattern + r"$",
                    "",
                    complement_clean,
                    flags=re.I,
                )
            else:
                complement_clean = re.sub(
                    r"(?:[,;:]\s*|\b(?:with|plus|but|in|under|behind|wearing)\s+)"
                    + decoration_pattern
                    + r"$",
                    "",
                    complement_clean,
                    flags=re.I,
                )
        for atom in (*reduction, *packaging, *reveal):
            complement_clean = re.sub(re.escape(atom.span), " ", complement_clean, flags=re.I)
        complement_clean = _SPACE.sub(" ", complement_clean).strip(" ,:;-") or complement
        subject = _strip_reveal_surface(subject)
        complement_clean = _strip_reveal_surface(complement_clean) or complement_clean

        shell = SemanticReplyShell(
            clause_id=frame.clause_id,
            operator=operator,
            subject_span=subject,
            complement_span=complement_clean,
            decoration_span=decoration,
            repackaging_marker=marker,
            opener_family=opener,
            generic_reusability=True,
            evidence_spans=(frame.clause_span,),
        )
        shells.append(shell)
        candidates.append(
            RelationCandidate(
                clause_id=frame.clause_id,
                relation_kind=(
                    "repackaging" if operator in {"decorated_identity", "repackaging"} else operator
                ),
                actor_span=subject,
                complement_span=complement_clean,
                decoration_span=decoration,
                confidence="high",
                consumed_spans=(frame.clause_span,),
            )
        )
    return tuple(shells), tuple(candidates)


def _covered(atom: LexicalAtom, consumed: Sequence[str], clause: str) -> bool:
    atom_norm = normalize_surface(atom.span)
    return any(
        atom_norm and atom_norm in normalize_surface(span)
        for span in consumed
        if span in clause or span == atom.span
    )


_PREDICATE_TOKEN = re.compile(
    r"\b[^\W\d_]+(?:ed|en|ing|izes?|ised|ises|ified|ifies)\b",
    re.I | re.UNICODE,
)


def _unknown_finite_predicate_spans(frame: ClauseFrame) -> tuple[str, ...]:
    """Find role-position finite predicates without assigning their semantic meaning."""

    represented = {normalize_surface(atom.span) for atom in frame.atoms}
    text = frame.clause_span
    candidates: list[str] = []
    head_atoms = [atom for atom in frame.atoms if atom.category.endswith("_head")]
    search_start = head_atoms[0].end if head_atoms else 0
    tail = text[search_start:]
    for match in re.finditer(r"\b[^\W\d_]+(?:s|ed)\b", tail, re.I | re.UNICODE):
        span = match.group(0)
        normalized = normalize_surface(span)
        if normalized in represented or normalized in {
            "has",
            "does",
            "was",
            "is",
            "this",
            "news",
            "status",
            "analysis",
        }:
            continue
        absolute_end = search_start + match.end()
        if not _clean_role(text[absolute_end:]):
            continue
        candidates.append(span)
        break
    return tuple(candidates)


def _unknown_predicate_spans(frame: ClauseFrame) -> tuple[str, ...]:
    """Return predicate-like spans not represented by a lexical atom.

    This is coverage accounting only. It never assigns an event or policy outcome.
    """

    represented = tuple(normalize_surface(atom.span) for atom in frame.atoms)
    spans: list[str] = []
    for match in _PREDICATE_TOKEN.finditer(frame.clause_span):
        span = match.group(0)
        normalized = normalize_surface(span)
        if any(normalized and normalized in atom for atom in represented):
            continue
        if normalized in {
            "according",
            "following",
            "during",
            "morning",
            "evening",
            "meeting",
            "branding",
            "clothing",
            "wording",
            "showing",
            "timing",
        }:
            continue
        spans.append(span)
    spans.extend(_unknown_finite_predicate_spans(frame))
    return tuple(dict.fromkeys(spans))


def _unknown_predicate_category(
    frame: ClauseFrame,
    span: str,
    *,
    context: Literal["source", "reply"],
) -> str:
    if context == "reply":
        return "unknown_predicate"
    human_context = any(atom.category == "human_head" for atom in frame.atoms)
    event_or_medical_context = any(
        atom.category
        in {
            "physical_event_head",
            "medical_state",
            "medical_destination",
            "medical_escalation",
            "harm_predicate",
            "harm_modifier",
        }
        for atom in frame.atoms
    )
    passive_or_result_state = bool(
        re.search(
            r"\b(?:was|were|is|are|been|left|remains?|found|declared|proved|suffered|sustained)\b"
            rf"[^.!?;]{{0,80}}\b{re.escape(span)}\b",
            frame.clause_span,
            re.I,
        )
    )
    result_link_context = any(atom.category == "result_link" for atom in frame.atoms)
    medical_trajectory_context = any(
        atom.category in {"medical_escalation", "medical_state", "harm_modifier"}
        for atom in frame.atoms
    )
    if (
        human_context
        and event_or_medical_context
        and (passive_or_result_state or result_link_context or medical_trajectory_context)
    ):
        return "possible_harm"
    return "unknown_predicate"


def _unknown_evaluation_spans(frame: ClauseFrame) -> tuple[str, ...]:
    """Return syntax-level evaluative material not covered by a known evaluation head."""

    spans: list[str] = []
    exclamative = re.search(
        r"\bwhat\s+(?:a|an)\s+([^.!?;,:]{1,80})",
        frame.clause_span,
        re.I,
    )
    if exclamative is not None:
        spans.append(exclamative.group(0))
    recognition = re.search(
        r"^(?:take\s+a\s+bow|hats\s+off|here(?:'|\u2019)s\s+to)\b[^.!?;]{0,100}",
        frame.clause_span,
        re.I,
    )
    if recognition is not None:
        spans.append(recognition.group(0))
    # Unknown reward/celebration noun phrases still carry material evaluative syntax.  The
    # deterministic layer does not guess their polarity; it preserves them for provider
    # extraction or candidate-level fail-closed handling.
    target_dedication = re.match(
        r"^(?:(?:a|an|the|one)\s+)?[^.!?;]{1,80}\s+(?:for|to)\s+[^.!?;]{1,50}$",
        frame.clause_span,
        re.I,
    )
    if target_dedication is not None and not _contains_predicate_like_surface(frame.clause_span):
        spans.append(target_dedication.group(0))
    modal_attribution = re.match(
        r"^[^.!?;]{1,60}\b(?:can|could|should|would)\s+[^.!?;]{1,80}$",
        frame.clause_span,
        re.I,
    )
    if modal_attribution is not None and frame.predicate_span is None:
        spans.append(modal_attribution.group(0))
    return tuple(dict.fromkeys(_clean_role(span) or span for span in spans))


def _unbound_shell_modifier_spans(
    frame: ClauseFrame,
    relation_candidates: Sequence[RelationCandidate],
) -> tuple[str, ...]:
    """Preserve syntactic decoration/framing that no shell role consumed."""

    shell_candidates = tuple(
        item
        for item in relation_candidates
        if item.relation_kind in {"identity", "reduction", "repackaging", "reveal"}
        and frame.clause_span in item.consumed_spans
    )
    if not shell_candidates or any(item.decoration_span for item in shell_candidates):
        return ()
    spans: list[str] = []
    complement = frame.complement_span or ""
    trailing = re.search(
        r"\b(?:in|under|with|behind|beneath|inside|outside|through|via|as)\s+"
        r"[^,;:.!?]{1,60}$",
        complement,
        re.I,
    )
    if trailing is not None:
        spans.append(trailing.group(0))
    participial = re.search(
        r"\b[^,;:.!?]{1,40}\b(?:wearing|wrapped|marketed|presented|styled|packaged|"
        r"labeled|labelled)\b[^,;:.!?]{0,50}$",
        complement,
        re.I,
    )
    if participial is not None:
        spans.append(participial.group(0))
    fronted = re.match(r"^([^,;]{2,60}),\s+.+\b(?:is|are|was|were)\b", frame.clause_span, re.I)
    if fronted is not None:
        spans.append(fronted.group(1).strip())
    return tuple(dict.fromkeys(spans))


def _semantic_item_id(
    *,
    side: Literal["source", "reply"],
    clause_id: str,
    category: str,
    span: str,
) -> str:
    payload = "\0".join((side, clause_id, category, normalize_surface(span))).encode()
    return f"{side}:{clause_id}:{hashlib.sha256(payload).hexdigest()[:16]}"


def _coverage_unresolved_items(
    *,
    side: Literal["source", "reply"],
    clause_id: str,
    clause_span: str,
    unresolved_spans: Sequence[str],
    unresolved_categories: Sequence[str],
) -> tuple[SemanticUnresolvedItem, ...]:
    spans = tuple(dict.fromkeys(unresolved_spans))
    categories = tuple(dict.fromkeys(unresolved_categories))
    if len(spans) == len(categories):
        pairs = tuple(zip(spans, categories, strict=True))
    elif len(categories) == 1:
        pairs = tuple((span, categories[0]) for span in spans)
    elif len(spans) == 1:
        pairs = tuple((spans[0], category) for category in categories)
    else:
        pairs = tuple((span, category) for span in spans for category in categories)
    return tuple(
        SemanticUnresolvedItem(
            item_id=_semantic_item_id(side=side, clause_id=clause_id, category=category, span=span),
            side=side,
            clause_id=clause_id,
            clause_span=clause_span,
            unresolved_span=span,
            category=category,
        )
        for span, category in pairs
        if category
        in {
            "possible_harm",
            "possible_evaluation",
            "possible_reduction",
            "unknown_predicate",
            "unknown_referent",
            "other",
        }
    )


def _refresh_coverage_items(
    coverage: SemanticCoverage, *, side: Literal["source", "reply"]
) -> SemanticCoverage:
    return coverage.model_copy(
        update={
            "coverage_id": f"{side}:{coverage.clause_id}",
            "unresolved_items": _coverage_unresolved_items(
                side=side,
                clause_id=coverage.clause_id,
                clause_span=coverage.clause_span,
                unresolved_spans=coverage.unresolved_spans,
                unresolved_categories=coverage.unresolved_categories,
            ),
        }
    )


def build_coverage(
    frames: Sequence[ClauseFrame],
    relation_candidates: Sequence[RelationCandidate],
    *,
    context: Literal["source", "reply"],
    result_state_candidates: Sequence[ResultStateCandidate] = (),
) -> tuple[SemanticCoverage, ...]:
    """Account for consumed and unresolved material in every clause."""

    results: list[SemanticCoverage] = []
    for frame in frames:
        consumed = tuple(
            dict.fromkeys(
                span
                for candidate in relation_candidates
                for span in candidate.consumed_spans
                if span
                and (
                    span in frame.clause_span
                    or normalize_surface(span) in normalize_surface(frame.clause_span)
                )
            )
        )
        unresolved_spans: list[str] = []
        unresolved_categories: list[str] = []
        literal_control = context == "reply" and _literal_control(frame)
        frame_result_states = tuple(
            candidate
            for candidate in result_state_candidates
            if candidate.linking_predicate_span
            and candidate.linking_predicate_span in frame.clause_span
            and candidate.state_complement_span
            and candidate.state_complement_span in frame.clause_span
        )
        ordinary_result_control = bool(
            context == "source"
            and frame_result_states
            and not any(
                atom.category
                in {
                    "harm_predicate",
                    "harm_modifier",
                    "medical_state",
                    "medical_destination",
                    "medical_escalation",
                    "failure_predicate",
                    "physical_event_head",
                }
                for atom in frame.atoms
            )
            and all(
                inspect_result_state_materiality(frame, candidate).impairment_relation
                in {
                    "ordinary_activity",
                    "ordinary_location",
                    "ordinary_business_or_ui_state",
                }
                for candidate in frame_result_states
            )
        )
        for atom in frame.atoms:
            if _covered(atom, consumed, frame.clause_span) or literal_control:
                continue
            if atom.category == "state_predicate":
                continue
            if atom.category == "reduction_operator" and _atom_in_coordinated_tail(frame, atom):
                continue
            category: str | None = None
            if atom.category in {
                "harm_predicate",
                "harm_modifier",
                "medical_state",
                "medical_destination",
                "medical_escalation",
            }:
                if atom.category != "medical_destination" or _medical_destination_is_result(
                    frame, atom
                ):
                    category = "possible_harm"
            elif (
                context == "reply"
                and atom.category
                in {
                    "positive_evaluation",
                    "evaluation_head",
                    "reward_relation",
                    "rating_relation",
                }
                and (
                    atom.category != "evaluation_head" or _evaluation_head_is_material(frame, atom)
                )
            ):
                category = "possible_evaluation"
            elif context == "reply" and atom.category in {
                "copula",
                "reduction_operator",
                "packaging_modifier",
                "reveal_operator",
            }:
                category = "possible_reduction"
            elif atom.category == "failure_predicate":
                category = "unknown_predicate"
            elif atom.category == "result_link" and any(
                item.category
                in {
                    "harm_predicate",
                    "harm_modifier",
                    "medical_state",
                    "medical_escalation",
                }
                or (
                    item.category == "medical_destination"
                    and _medical_destination_is_result(frame, item)
                )
                for item in frame.atoms
            ):
                category = "possible_harm"
            if category is not None:
                unresolved_spans.append(atom.span)
                unresolved_categories.append(category)
        if context == "source":
            for candidate in frame_result_states:
                if _material_human_result_state(frame, candidate):
                    complement_span = candidate.state_complement_span
                    if complement_span is None:
                        continue
                    unresolved_spans.append(complement_span)
                    unresolved_categories.append("possible_harm")
        for span in _unknown_predicate_spans(frame):
            if ordinary_result_control:
                continue
            if context == "source" and any(
                _material_human_result_state(frame, candidate)
                and candidate.state_complement_span is not None
                and normalize_surface(span) in normalize_surface(candidate.state_complement_span)
                for candidate in frame_result_states
            ):
                continue
            if (
                any(normalize_surface(span) in normalize_surface(item) for item in consumed)
                or literal_control
            ):
                continue
            unresolved_spans.append(span)
            unresolved_categories.append(_unknown_predicate_category(frame, span, context=context))
        if context == "reply" and not literal_control:
            has_evaluation_relation = any(
                candidate.relation_kind == "evaluation"
                and any(
                    normalize_surface(span) in normalize_surface(frame.clause_span)
                    for span in candidate.consumed_spans
                )
                for candidate in relation_candidates
            )
            for span in _unknown_evaluation_spans(frame):
                if has_evaluation_relation:
                    continue
                if any(normalize_surface(span) in normalize_surface(item) for item in consumed):
                    continue
                unresolved_spans.append(span)
                unresolved_categories.append("possible_evaluation")
            for span in _unbound_shell_modifier_spans(frame, relation_candidates):
                if any(normalize_surface(span) in normalize_surface(item) for item in consumed):
                    # Whole-clause consumption is not role coverage.  Only an explicit decoration
                    # role may consume this material.
                    shell_has_role = any(
                        candidate.decoration_span
                        and normalize_surface(span) in normalize_surface(candidate.decoration_span)
                        for candidate in relation_candidates
                    )
                    if shell_has_role:
                        continue
                unresolved_spans.append(span)
                unresolved_categories.append("possible_reduction")

        # A copular clause with semantic content but no shell relation is material uncertainty,
        # unless it is a known literal definition/formula/quotation control.
        if context == "reply" and not literal_control:
            has_copula = (
                any(atom.category == "copula" for atom in frame.atoms) or ":" in frame.clause_span
            )
            has_evaluation = any(
                candidate.relation_kind == "evaluation"
                and any(
                    normalize_surface(span) in normalize_surface(frame.clause_span)
                    for span in candidate.consumed_spans
                )
                for candidate in relation_candidates
            )
            if (
                has_copula
                and not has_evaluation
                and not any(
                    candidate.relation_kind in {"identity", "reduction", "repackaging", "reveal"}
                    and candidate.consumed_spans
                    and candidate.consumed_spans[0] == frame.clause_span
                    for candidate in relation_candidates
                )
            ):
                unresolved_spans.append(frame.predicate_span or frame.clause_span)
                unresolved_categories.append("possible_reduction")
        unresolved_spans = list(dict.fromkeys(unresolved_spans))
        unresolved_categories = list(dict.fromkeys(unresolved_categories))
        results.append(
            SemanticCoverage(
                coverage_id=f"{context}:{frame.clause_id}",
                clause_id=frame.clause_id,
                clause_span=frame.clause_span,
                consumed_spans=consumed,
                unresolved_items=_coverage_unresolved_items(
                    side=context,
                    clause_id=frame.clause_id,
                    clause_span=frame.clause_span,
                    unresolved_spans=tuple(unresolved_spans),
                    unresolved_categories=tuple(unresolved_categories),
                ),
                unresolved_spans=tuple(unresolved_spans),
                unresolved_categories=tuple(unresolved_categories),
                complete=not unresolved_spans,
            )
        )
    return tuple(results)


def _augment_unknown_referent_coverage(
    coverage_items: tuple[SemanticCoverage, ...],
    events: tuple[SemanticEvent, ...],
) -> tuple[SemanticCoverage, ...]:
    """A consumed predicate is incomplete when its material affected role is unresolved."""

    output: list[SemanticCoverage] = []
    for coverage in coverage_items:
        unresolved_spans = list(coverage.unresolved_spans)
        unresolved_categories = list(coverage.unresolved_categories)
        for event in events:
            if not any(span in coverage.clause_span for span in event.evidence_spans):
                continue
            affected = event.affected_entity
            if affected is not None and affected.entity_type != "unknown":
                continue
            if event.event_type not in {
                "unknown",
                "death",
                "serious_injury",
                "critical_condition",
                "hospitalization",
                "intensive_care",
                "permanent_injury",
                "non_survival",
                "technical_failure",
                "business_failure",
            }:
                continue
            span = (
                affected.text_span
                if affected is not None
                else event.predicate
                if event.predicate in coverage.clause_span
                else event.evidence_spans[0]
            )
            unresolved_spans.append(span)
            unresolved_categories.append("unknown_referent")
        updated = coverage.model_copy(
            update={
                "unresolved_spans": tuple(dict.fromkeys(unresolved_spans)),
                "unresolved_categories": tuple(dict.fromkeys(unresolved_categories)),
                "complete": not unresolved_spans,
            }
        )
        output.append(_refresh_coverage_items(updated, side="source"))
    return tuple(output)


def _merge_semantic_events(
    *groups: Sequence[SemanticEvent],
) -> tuple[SemanticEvent, ...]:
    """Merge independently bound event families without duplicating the same typed relation."""

    output: list[SemanticEvent] = []
    seen: set[tuple[str, str, str, str]] = set()
    for event in (event for group in groups for event in group):
        key = (
            event.event_type,
            normalize_surface(event.affected_entity.text_span) if event.affected_entity else "",
            event.assertion_state,
            "|".join(normalize_surface(span) for span in event.evidence_spans),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(event)
    return tuple(output)


def extract_reply_propositions(
    frames: Sequence[ClauseFrame],
    coverage: Sequence[SemanticCoverage],
) -> tuple[ReplyProposition, ...]:
    """Project proposition completeness and literal communicative roles per reply clause."""

    coverage_by_clause = {item.clause_id: item for item in coverage}
    output: list[ReplyProposition] = []
    for frame in frames:
        normalized = normalize_surface(frame.clause_span)
        intransitive_roles = _finite_intransitive_roles(frame)
        subject_span = frame.subject_span
        predicate_span = frame.predicate_span
        object_or_complement_span = frame.object_span or frame.complement_span
        if intransitive_roles is not None:
            subject_span, predicate_span = intransitive_roles
        function: Literal[
            "plain_observation",
            "direct_response",
            "qualification",
            "question",
            "other",
            "unknown",
        ] = "unknown"
        complete = False
        confidence: Literal["high", "medium", "low"] = "low"
        if frame.clause_span.rstrip().endswith("?") or re.match(
            r"^(?:what|why|how|when|where|who|which|is|are|was|were|do|does|did|"
            r"can|could|should|would)\b",
            normalized,
        ):
            function, complete, confidence = "question", True, "high"
        elif _evidential_qualification_frame(frame):
            function, complete, confidence = "qualification", True, "high"
        elif (
            _direct_remediation_frame(frame)
            or _literal_acknowledgement_frame(frame)
            or _normative_response_frame(frame)
        ):
            function, complete, confidence = "direct_response", True, "high"
        elif _factual_assertion_frame(frame) or (
            (_literal_control(frame) or _complete_copular_literal(frame))
            and frame.subject_span
            and frame.predicate_span
            and (frame.object_span or frame.complement_span)
        ):
            function, complete, confidence = "plain_observation", True, "high"
        elif intransitive_roles is not None:
            function = (
                "qualification" if normalized.startswith(("but ", "yet ")) else "plain_observation"
            )
            complete, confidence = True, "high"
        else:
            clause_coverage = coverage_by_clause.get(frame.clause_id)
            complete = bool(
                frame.subject_span
                and frame.predicate_span
                and (frame.object_span or frame.complement_span)
                and clause_coverage is not None
                and clause_coverage.complete
            )
        output.append(
            ReplyProposition(
                clause_id=frame.clause_id,
                subject_span=subject_span,
                predicate_span=predicate_span,
                object_or_complement_span=object_or_complement_span,
                assertion_state=frame.assertion_state,
                proposition_complete=complete,
                communicative_function=function,
                confidence=confidence,
                evidence_spans=(frame.clause_span,) if complete else (),
            )
        )
    return tuple(output)


def _reply_communicative_function(
    reply_text: str,
    frames: Sequence[ClauseFrame],
    coverage: Sequence[SemanticCoverage],
    evaluations: Sequence[EvaluationRelation],
    shells: Sequence[SemanticReplyShell],
) -> tuple[str, str, tuple[str, ...], tuple[ReplyProposition, ...]]:
    cleaned = reply_text.strip()
    if not cleaned:
        return "unknown", "low", (), ()
    propositions = extract_reply_propositions(frames, coverage)
    literal = tuple(
        proposition
        for proposition in propositions
        if proposition.communicative_function
        in {"plain_observation", "direct_response", "qualification", "question"}
        and proposition.confidence == "high"
    )
    if literal and len(literal) == len(propositions) and not evaluations and not shells:
        functions = {item.communicative_function for item in literal}
        function = next(iter(functions)) if len(functions) == 1 else "qualification"
        return (
            function,
            "high",
            tuple(span for item in literal for span in item.evidence_spans),
            propositions,
        )
    if evaluations or shells:
        return (
            "other",
            "high",
            tuple(
                dict.fromkeys(
                    (
                        *(span for relation in evaluations for span in relation.evidence_spans),
                        *(span for shell in shells for span in shell.evidence_spans),
                    )
                )
            ),
            propositions,
        )
    return "unknown", "low", (), propositions


def extract_literal_proposition_frames(
    frames: Sequence[ClauseFrame],
    propositions: Sequence[ReplyProposition],
) -> tuple[LiteralPropositionFrame, ...]:
    """Type complete literal roles without embedding candidate or safety policy."""

    by_clause = {item.clause_id: item for item in propositions}
    output: list[LiteralPropositionFrame] = []
    for frame in frames:
        proposition = by_clause.get(frame.clause_id)
        subject = frame.subject_span or (proposition.subject_span if proposition else None)
        predicate = frame.predicate_span or (proposition.predicate_span if proposition else None)
        complement = (
            frame.object_span
            or frame.complement_span
            or (proposition.object_or_complement_span if proposition else None)
        )
        normalized = normalize_surface(frame.clause_span)
        action_recommendation = _ACTION_RECOMMENDATION.fullmatch(normalized)
        remediation_predicate = _REMEDIATION_PREDICATE.search(normalized)
        passive_evidence = re.match(
            r"^(?P<subject>.+?)\s+(?:has|have|had|is|are|was|were)\s+"
            r"(?:(?:not|never)\s+)?been\s+(?P<participle>[^\W\d_]+(?:ed|en))\b",
            normalized,
            re.I,
        )
        if proposition and proposition.communicative_function == "direct_response":
            if action_recommendation is not None:
                subject = action_recommendation.group("subject")
                predicate = action_recommendation.group("action")
                complement = action_recommendation.group("object")
            elif remediation_predicate is not None:
                subject = normalized[: remediation_predicate.start()].strip()
                predicate = remediation_predicate.group(0)
                complement = normalized[remediation_predicate.end() :].strip()
        elif passive_evidence is not None:
            subject = passive_evidence.group("subject")
            predicate = passive_evidence.group("participle")
            complement = passive_evidence.group("participle")
        if not (
            proposition
            and proposition.proposition_complete
            and proposition.confidence == "high"
            and subject
            and predicate
            and complement
        ):
            continue
        proposition_type: Literal[
            "status",
            "evidence_availability",
            "investigation_state",
            "verification_state",
            "publication_state",
            "remediation_recommendation",
            "replacement_recommendation",
            "inspection_recommendation",
            "preservation_or_isolation_action",
        ] = "status"
        if re.search(r"\b(?:investigation|examination|inquiry|review|analysis)\b", normalized):
            proposition_type = "investigation_state"
        elif re.search(r"\b(?:verification|verified|confirmed|corroborated)\b", normalized):
            proposition_type = "verification_state"
        elif re.search(r"\b(?:published|released|disclosed|publication)\b", normalized):
            proposition_type = "publication_state"
        elif re.search(r"\b(?:evidence|documentation|records?|substantiated|proven)\b", normalized):
            proposition_type = "evidence_availability"

        if proposition.communicative_function == "direct_response":
            if re.search(r"\b(?:replace|replacement|substitute|substitution)\b", normalized):
                proposition_type = "replacement_recommendation"
            elif re.search(
                r"\b(?:preserve|preservation|isolate|isolation|quarantine|retain)\b",
                normalized,
            ):
                proposition_type = "preservation_or_isolation_action"
            elif re.search(r"\b(?:inspect|inspection|examine|examination|audit)\b", normalized):
                proposition_type = "inspection_recommendation"
            else:
                proposition_type = "remediation_recommendation"
        output.append(
            LiteralPropositionFrame(
                clause_id=frame.clause_id,
                proposition_type=proposition_type,
                subject_span=subject,
                predicate_span=predicate,
                object_or_complement_span=complement,
                assertion_state=frame.assertion_state,
                evidence_spans=(frame.clause_span,),
            )
        )
    return tuple(output)


def _reconcile_reply_semantic_completeness(
    frames: Sequence[ClauseFrame],
    coverage: Sequence[SemanticCoverage],
    propositions: Sequence[ReplyProposition],
    evaluations: Sequence[EvaluationRelation],
    shells: Sequence[SemanticReplyShell],
) -> tuple[tuple[SemanticCoverage, ...], tuple[ReplySemanticCompleteness, ...]]:
    """Fail closed when material reply surface has no complete or typed semantic role."""

    proposition_by_clause = {item.clause_id: item for item in propositions}
    coverage_by_clause = {item.clause_id: item for item in coverage}
    reconciled: list[SemanticCoverage] = []
    completeness: list[ReplySemanticCompleteness] = []
    for frame in frames:
        proposition = proposition_by_clause.get(frame.clause_id)
        clause_coverage = coverage_by_clause[frame.clause_id]
        typed_evaluation = any(
            any(span and span in frame.clause_span for span in relation.evidence_spans)
            for relation in evaluations
        )
        typed_shell = any(
            any(span and span in frame.clause_span for span in shell.evidence_spans)
            for shell in shells
        )
        validated_literal = bool(
            proposition
            and proposition.proposition_complete
            and proposition.confidence == "high"
            and proposition.communicative_function
            in {"plain_observation", "direct_response", "qualification", "question"}
            and proposition.evidence_spans
        )
        material_surface = bool(lexical_tokens(frame.clause_span))
        incomplete = bool(
            material_surface
            and not (proposition and proposition.proposition_complete)
            and not typed_evaluation
            and not typed_shell
            and not validated_literal
        )
        if incomplete and "incomplete_proposition" not in clause_coverage.unresolved_categories:
            clause_coverage = clause_coverage.model_copy(
                update={
                    "unresolved_spans": (*clause_coverage.unresolved_spans, frame.clause_span),
                    "unresolved_categories": (
                        *clause_coverage.unresolved_categories,
                        "incomplete_proposition",
                    ),
                    "complete": False,
                }
            )
            clause_coverage = _refresh_coverage_items(clause_coverage, side="reply")
        reconciled.append(clause_coverage)
        completeness.append(
            ReplySemanticCompleteness(
                clause_id=frame.clause_id,
                proposition_complete=bool(proposition and proposition.proposition_complete),
                typed_evaluation_present=typed_evaluation,
                typed_shell_present=typed_shell,
                validated_literal_function=validated_literal,
                material_surface_present=material_surface,
                unresolved_category="incomplete_proposition" if incomplete else None,
                evidence_spans=(frame.clause_span,) if material_surface else (),
            )
        )
    return tuple(reconciled), tuple(completeness)


def extract_compositional_semantics(source_text: str, reply_text: str = "") -> SemanticInspection:
    """Run the full compositional extraction pipeline and emit canonical typed IR."""

    source_frames = segment_clauses(source_text)
    reply_frames = segment_clauses(reply_text) if reply_text else ()
    reply_shell_frames = segment_clauses(reply_text, split_coordination=False) if reply_text else ()
    source_entities = extract_entity_candidates(source_frames)
    source_events, source_candidates = bind_event_relations(source_frames)
    result_events, result_candidates, result_states = bind_result_state_relations(source_frames)
    result_entities = tuple(
        SemanticEntity(
            clause_id=candidate.clause_id,
            text_span=candidate.affected_entity_span,
            canonical_name=(
                lexical_tokens(candidate.affected_entity_span)[-1]
                if lexical_tokens(candidate.affected_entity_span)
                else None
            ),
            entity_type=candidate.affected_entity_type,
            confidence=("medium" if candidate.affected_entity_type != "unknown" else "low"),
        )
        for candidate in result_states
        if candidate.affected_entity_span and candidate.affected_entity_type != "unknown"
    )
    source_entities = _dedupe_entities((*source_entities, *result_entities))
    source_events = _merge_semantic_events(source_events, result_events)
    evaluations, evaluation_candidates = bind_evaluation_relations(
        reply_frames,
        source_entities,
        source_events,
    )
    shells, shell_candidates = bind_reply_shell_relations(reply_shell_frames)
    reply_entities = _dedupe_entities(
        (
            *extract_entity_candidates(reply_frames),
            *(
                relation.target_entity
                for relation in evaluations
                if relation.target_entity is not None
            ),
        )
    )
    source_coverage = _augment_unknown_referent_coverage(
        build_coverage(
            source_frames,
            (*source_candidates, *result_candidates),
            context="source",
            result_state_candidates=result_states,
        ),
        source_events,
    )
    reply_coverage = build_coverage(
        reply_frames,
        (*evaluation_candidates, *shell_candidates),
        context="reply",
    )
    communicative_function, communicative_confidence, communicative_spans, reply_propositions = (
        _reply_communicative_function(reply_text, reply_frames, reply_coverage, evaluations, shells)
    )
    reply_coverage, reply_semantic_completeness = _reconcile_reply_semantic_completeness(
        reply_frames,
        reply_coverage,
        reply_propositions,
        evaluations,
        shells,
    )
    literal_proposition_frames = extract_literal_proposition_frames(
        reply_frames, reply_propositions
    )
    unresolved = tuple(
        dict.fromkeys(
            span
            for coverage in (*source_coverage, *reply_coverage)
            for span in coverage.unresolved_spans
        )
    )
    return SemanticInspection(
        source_entities=source_entities,
        predicate_scopes=(
            *build_predicate_scope_graph(source_text, source_frames, namespace="source"),
            *build_predicate_scope_graph(reply_text, reply_frames, namespace="reply"),
        ),
        source_events=source_events,
        reply_entities=reply_entities,
        reply_events=(),
        evaluations=evaluations,
        reply_shells=shells,
        result_state_materiality=tuple(
            inspect_result_state_materiality(
                next(frame for frame in source_frames if frame.clause_id == candidate.clause_id),
                candidate,
            )
            for candidate in result_states
        ),
        human_function_impacts=tuple(
            inspect_human_function_impact(
                next(frame for frame in source_frames if frame.clause_id == candidate.clause_id),
                candidate,
            )
            for candidate in result_states
        ),
        reply_propositions=reply_propositions,
        literal_proposition_frames=literal_proposition_frames,
        reply_semantic_completeness=reply_semantic_completeness,
        source_coverage=source_coverage,
        reply_coverage=reply_coverage,
        reply_communicative_function=communicative_function,
        reply_communicative_function_confidence=communicative_confidence,
        reply_communicative_evidence_spans=communicative_spans,
        reply_self_reference=bool(
            re.search(r"\b(?:i|i'm|i've|me|my|mine|myself)\b", reply_text, re.I)
        ),
        unresolved_material=unresolved,
    )
