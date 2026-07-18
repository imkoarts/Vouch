"""Deterministic publication-context and source-dependency assessment.

The drafting model writes the copy, but it does not get to decide whether readers must see the
source post. This module classifies the finished text as standalone, quote-dependent, or needing
one bounded context rewrite. It also measures whether a standalone draft adds a concrete lens
beyond paraphrasing the source.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.content import AngleType, UntrustedSourceData

PublicationContextFormat = Literal["standalone_post", "quote_post", "rewrite_with_context"]
AngleContextPreference = Literal["standalone_post", "quote_post", "either"]

_WORD = re.compile(r"[A-Za-z0-9_@#'-]{2,}")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_QUOTED_OPENING = re.compile(
    r"^\s*[\"'\u201c\u2018][^\"'\u201d\u2019\n]{2,100}[\"'\u201d\u2019](?:\s|[,:;.!?])",
    re.IGNORECASE,
)
_DEICTIC_OPENING = re.compile(
    r"^\s*(?:this|that|these|those|it|the wording|the phrase|the sentence|the line|"
    r"the headline|the caption|the framing)\b",
    re.IGNORECASE,
)
_WORDING_COMMENTARY = re.compile(
    r"\b(?:the wording|the phrase|the sentence|that sentence|this sentence|the line|that line|"
    r"this line|the headline|the caption|the framing|choice of words|wording choice|"
    r"is doing a lot of work|is carrying a lot of (?:weight|freight)|doing careful rhetorical "
    r"work|one way to (?:put|describe|frame) it|notice how|careful way to describe|"
    r"kept at (?:a )?(?:polite|careful) distance|avoids? saying|doesn['\u2019]t say|"
    r"lets? .* occupy .* sentence|within the sentence)\b",
    re.IGNORECASE,
)
_SOURCE_OBJECT = re.compile(
    r"\b(?:this|that|the) (?:post|tweet|caption|headline|claim|wording|sentence|line|framing)\b",
    re.IGNORECASE,
)
_ACTION = re.compile(
    r"\b(?:says?|said|launch(?:es|ed|ing)?|press(?:es|ed|ing)?|reject(?:s|ed|ing)?|"
    r"isolat(?:e|es|ed|ing)|cause(?:s|d|ing)?|strike(?:s|d|ing)?|improv(?:e|es|ed|ing)|"
    r"cut(?:s|ting)?|raise(?:s|d|ing)?|block(?:s|ed|ing)?|ban(?:s|ned|ning)?|"
    r"approve(?:s|d|ing)?|sign(?:s|ed|ing)?|win(?:s|ning)?|lose(?:s|lost|losing)?|"
    r"defeat(?:s|ed|ing)?|announce(?:s|d|ing)?|ask(?:s|ed|ing)?|choose(?:s|chose|choosing)?|"
    r"support(?:s|ed|ing)?|avoid(?:s|ed|ing)?|turn(?:s|ed|ing)?|make(?:s|made|making)?|"
    r"become(?:s|became|becoming)?|move(?:s|d|ing)?|hold(?:s|held|ing)?|"
    r"acknowledge(?:s|d|ing)?|describe(?:s|d|ing)?|frame(?:s|d|ing)?)\b",
    re.IGNORECASE,
)
_GENERIC_IMPLICATION = re.compile(
    r"\b(?:the (?:immediate )?(?:effect|implication|point) is|unusually concrete|"
    r"not just about .{0,90} but whether|a different frame|the same conversation|"
    r"question of who|take a position on it|makes? it about|turns? the conversation into|"
    r"this is about more than|the broader point|the real question)\b",
    re.IGNORECASE,
)
_CONCRETE_NOVEL = re.compile(
    r"\b(?:government|court|all(?:y|ies)|jurisdiction|sanction|visa|security|oil|product|"
    r"supply|refinery|export|import|price|vote|law|treaty|campaign|authority|strike|"
    r"production|shipment|border|model|api|contract|fee|deadline|final|trophy|match)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "but",
    "can",
    "could",
    "does",
    "for",
    "from",
    "have",
    "into",
    "just",
    "more",
    "most",
    "not",
    "only",
    "other",
    "over",
    "same",
    "should",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
    "would",
}
_ENTITY_EXCLUSIONS = {
    "A",
    "An",
    "And",
    "But",
    "If",
    "In",
    "It",
    "One",
    "Some",
    "That",
    "The",
    "This",
    "Those",
    "When",
}


class SourceDependencyAssessment(BaseModel):
    """Publication context decision persisted with the draft."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recommended_format: PublicationContextFormat
    source_dependency: float = Field(ge=0.0, le=1.0)
    standalone_clarity: float = Field(ge=0.0, le=1.0)
    wording_commentary: bool
    missing_context: tuple[str, ...] = ()
    reason: str
    rewrite_instruction: str = ""
    lexical_overlap: float = Field(default=0.0, ge=0.0, le=1.0)
    added_value_score: float = Field(default=1.0, ge=0.0, le=1.0)
    can_convert_to_standalone: bool = True


def _tokens(text: str) -> set[str]:
    return {
        token.casefold().strip("'\"")
        for token in _WORD.findall(text)
        if token.casefold().strip("'\"") not in _STOPWORDS
    }


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip())


def _overlap(draft: str, source: str) -> float:
    source_tokens = _tokens(source)
    if not source_tokens:
        return 0.0
    return min(1.0, len(_tokens(draft) & source_tokens) / len(source_tokens))


def _actor_is_clear(text: str) -> bool:
    if re.search(r"(?<!\w)@[A-Za-z0-9_]{2,}\b", text):
        return True
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9'-]{2,}\b", text)
    return any(candidate not in _ENTITY_EXCLUSIONS for candidate in candidates)


def _novel_added_value(draft: str, source: str) -> float:
    """Estimate proposition contribution without rewarding vocabulary novelty.

    A one-source standalone commentary cannot prove a new relation merely by introducing new
    abstract nouns, adjectives, or comparison language. The conservative score rewards only a
    compact source-specific reaction or an explicit relation whose actor, action, and object are
    all already visible in the evidence. Semantic review makes the final publication decision.
    """

    draft_tokens = _tokens(draft)
    source_tokens = _tokens(source)
    if not draft_tokens or not source_tokens:
        return 0.0
    overlap = len(draft_tokens & source_tokens) / max(len(draft_tokens), 1)
    draft_actions = {item.casefold() for item in _ACTION.findall(draft)}
    source_actions = {item.casefold() for item in _ACTION.findall(source)}
    supported_actions = draft_actions & source_actions
    generic_packaging = bool(_GENERIC_IMPLICATION.search(draft))
    invented_comparison = bool(
        re.search(
            r"\b(?:rather than|more (?:tangible|concrete|meaningful) than|"
            r"general (?:discussion|argument)|discussed in the abstract)\b",
            draft,
            re.IGNORECASE,
        )
    )
    depersonalized = bool(
        re.search(
            r"\b(?:a|the) (?:named )?(?:officeholder|figure|organization|company|actor)\b",
            draft,
            re.IGNORECASE,
        )
    )
    if generic_packaging or invented_comparison or depersonalized:
        return 0.05
    if len(draft) <= 140 and overlap < 0.45:
        # A compact source-specific reaction can be useful in quote/reply context.
        return 0.52
    if supported_actions and overlap >= 0.45:
        # This is grounded, but it remains a source rewrite rather than new commentary.
        return 0.30
    return 0.12


def assess_source_dependency(
    *,
    draft_text: str,
    source: UntrustedSourceData,
    angle_type: AngleType,
    angle_context_preference: AngleContextPreference = "either",
    source_context_strategy: str = "standalone",
) -> SourceDependencyAssessment:
    """Classify how the finished draft depends on its sole source post."""

    text = " ".join(draft_text.split())
    source_text = " ".join(source.content.split())
    wording_commentary = bool(
        _WORDING_COMMENTARY.search(text)
        or _SOURCE_OBJECT.search(text)
        or angle_context_preference == "quote_post"
    )
    opening_fragment = bool(_QUOTED_OPENING.search(text))
    deictic_opening = bool(_DEICTIC_OPENING.search(text))
    lexical_overlap = _overlap(text, source_text)
    actor_clear = _actor_is_clear(text)
    event_clear = bool(_ACTION.search(text))
    object_clear = len(_tokens(text)) >= 7
    no_source_required = not (wording_commentary or deictic_opening or _SOURCE_OBJECT.search(text))
    standalone_clarity = sum((actor_clear, event_clear, object_clear, no_source_required)) / 4.0
    missing_context: list[str] = []
    if not actor_clear:
        missing_context.append("central actor")
    if not event_clear:
        missing_context.append("event or action")
    if not object_clear:
        missing_context.append("object or concrete subject")
    if not no_source_required:
        missing_context.append("reason for the reaction without the source")

    risky_verification = source.verification_status in {"single_source", "unverified", "unknown"}
    attribution_importance = 1.0 if risky_verification else 0.25
    missing_context_score = 1.0 - standalone_clarity
    low_entity_coverage = 0.0 if actor_clear and event_clear else 1.0
    quote_score = (
        0.35 * float(wording_commentary)
        + 0.20 * missing_context_score
        + 0.15 * float(opening_fragment or deictic_opening)
        + 0.10 * lexical_overlap
        + 0.10 * low_entity_coverage
        + 0.10 * attribution_importance
    )
    if angle_context_preference == "quote_post":
        quote_score += 0.12
    elif angle_context_preference == "standalone_post":
        quote_score -= 0.08
    quote_score = max(0.0, min(quote_score, 1.0))

    added_value_score = _novel_added_value(text, source_text)

    if source_context_strategy == "attach_source_media":
        return SourceDependencyAssessment(
            recommended_format="standalone_post",
            source_dependency=max(quote_score, 0.55),
            standalone_clarity=max(standalone_clarity, 0.75),
            wording_commentary=wording_commentary,
            missing_context=(),
            reason="The attached source media supplies the context required by the commentary.",
            rewrite_instruction="",
            lexical_overlap=lexical_overlap,
            # A short reaction can be the added value when the original media remains visible.
            added_value_score=max(added_value_score, 0.65),
            can_convert_to_standalone=True,
        )
    if source_context_strategy == "quote_post":
        return SourceDependencyAssessment(
            recommended_format="quote_post",
            source_dependency=max(quote_score, 0.85),
            standalone_clarity=standalone_clarity,
            wording_commentary=wording_commentary,
            missing_context=tuple(missing_context),
            reason="The original X post is already the required publication context.",
            rewrite_instruction="",
            lexical_overlap=lexical_overlap,
            added_value_score=added_value_score,
            can_convert_to_standalone=standalone_clarity >= 0.5,
        )

    force_quote = wording_commentary and (opening_fragment or lexical_overlap >= 0.42)
    if force_quote or quote_score >= 0.65:
        return SourceDependencyAssessment(
            recommended_format="quote_post",
            source_dependency=max(quote_score, 0.65),
            standalone_clarity=standalone_clarity,
            wording_commentary=wording_commentary,
            missing_context=tuple(missing_context),
            reason=(
                "The draft comments on the source's exact wording or framing, so the source is "
                "part of the thought rather than background evidence."
            ),
            rewrite_instruction=(
                "To make this standalone, name the actor and event before the reaction and replace "
                "wording commentary with a self-contained observation."
            ),
            lexical_overlap=lexical_overlap,
            added_value_score=added_value_score,
            can_convert_to_standalone=actor_clear or event_clear,
        )

    if quote_score >= 0.40 or standalone_clarity < 0.50:
        return SourceDependencyAssessment(
            recommended_format="rewrite_with_context",
            source_dependency=quote_score,
            standalone_clarity=standalone_clarity,
            wording_commentary=wording_commentary,
            missing_context=tuple(missing_context),
            reason=(
                "The observation may work standalone, but the current copy omits necessary context."
            ),
            rewrite_instruction=(
                "Rewrite once as a standalone post. Preserve the observation and tone. Add only "
                "the minimum source-supported context needed to identify who did or said what, "
                "what happened, and why the observation matters. Do not mention the post, report, "
                "caption, wording, or assume the reader can see the source."
            ),
            lexical_overlap=lexical_overlap,
            added_value_score=added_value_score,
            can_convert_to_standalone=True,
        )

    return SourceDependencyAssessment(
        recommended_format="standalone_post",
        source_dependency=quote_score,
        standalone_clarity=standalone_clarity,
        wording_commentary=wording_commentary,
        missing_context=tuple(missing_context),
        reason="The draft identifies the actor and event and can be understood without the source.",
        rewrite_instruction="",
        lexical_overlap=lexical_overlap,
        added_value_score=added_value_score,
        can_convert_to_standalone=True,
    )
