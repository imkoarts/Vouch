"""Grounded publishable-signal selection before contribution planning."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.content import (
    ContentModeDecision,
    SignalDecision,
    UntrustedSourceData,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_ACTION = re.compile(
    r"\b(?:said|says|announc(?:e[sd]?|ing)|launch(?:ed|es|ing)?|sign(?:ed|s|ing)?|"
    r"vote[sd]?|remov(?:e[sd]?|ing)|cut(?:s|ting)?|reduc(?:e[sd]?|ing)|"
    r"increas(?:e[sd]?|ing)|won|lost|score[sd]?|defeat(?:ed|s)?|beat(?:s|en)?|"
    r"add(?:ed|s|ing)?|buy|buys|bought|acquir(?:e[sd]?|ing)|sell|sells|sold|"
    r"found(?:ed|s|ing)?|approve[sd]?|reject(?:ed|s)?|exclude[sd]?|include[sd]?|"
    r"publish(?:ed|es|ing)?|"
    r"report(?:ed|s|ing)?|confirm(?:ed|s|ing)?|den(?:y|ied|ies)|requires?)\b",
    re.I,
)
_NEWS = re.compile(
    r"\b(?:breaking|reportedly|according to|announced|confirmed|votes?|signed)\b", re.I
)
_ANALYSIS = re.compile(r"\b(?:because|therefore|means|mechanism|analysis|explains?|why)\b", re.I)
_OPINION = re.compile(
    r"\b(?:i think|i believe|should|bad idea|good idea|opinion|argue[sd]?)\b", re.I
)
_THESIS = re.compile(
    r"\b(?:the point is|the thesis|the real issue|fundamentally|ultimately)\b", re.I
)
_DISCUSSION = re.compile(r"\?$|\b(?:debate|discussion|question)\b", re.I)
_TREND = re.compile(r"(?:^|\s)#[\w-]+|\btrending\b", re.I)
_CREATIVE = re.compile(r"\b(?:imagine|write|prompt|joke|meme|story)\b", re.I)


@dataclass(frozen=True, slots=True)
class SignalSelectionOutcome:
    anchor: UntrustedSourceData | None
    decision: SignalDecision | None
    decisions: tuple[SignalDecision, ...]
    status: str
    reason_code: str


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip())


def _source_type(text: str) -> str:
    if _CREATIVE.search(text):
        return "creative_prompt"
    if _NEWS.search(text):
        return "news_update"
    if _THESIS.search(text):
        return "thesis"
    if _OPINION.search(text):
        return "opinion"
    if _ANALYSIS.search(text):
        return "analysis"
    if _DISCUSSION.search(text):
        return "discussion"
    if _TREND.search(text):
        return "trend"
    if _ACTION.search(text):
        return "factual_event"
    if text.strip():
        return "observation"
    return "unknown"


def _semantically_complete(text: str) -> bool:
    sentences = _sentences(text)
    if not sentences:
        return False
    core = sentences[0]
    words = core.split()
    return len(words) >= 5 and bool(
        _ACTION.search(core)
        or re.search(r"\b(?:is|are|was|were|has|have)\b", core, re.I)
        # A news lead such as "BREAKING: Company weighs charging ..." can contain a complete
        # proposition even when its verb is outside the deliberately small action lexicon.
        or (_NEWS.search(core) and re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}", core))
    )


def evaluate_signal(
    source: UntrustedSourceData,
    *,
    content_mode: ContentModeDecision,
    has_auxiliary_evidence: bool = False,
) -> SignalDecision:
    """Evaluate one ranked source independently; application code owns the decision."""

    text = source.content.strip()
    sentences = _sentences(text)
    core = sentences[0] if sentences else text or "Empty source"
    source_type = _source_type(text)
    complete = _semantically_complete(text)
    quote_or_reply = content_mode.subtype in {"quote_commentary", "reply_commentary"}
    direct = content_mode.mode in {"factual_update", "summary"}
    creative = content_mode.mode == "opinion_or_creative"
    unresolved = bool(sentences and sentences[-1].endswith("?"))

    contribution_count = 0
    if (
        (direct and complete)
        or (quote_or_reply and text)
        or (creative and text)
        or has_auxiliary_evidence
        or (unresolved and content_mode.mode == "commentary")
    ):
        contribution_count = 1

    supports = contribution_count > 0
    source_echo = (
        "high" if complete and content_mode.mode == "commentary" and not quote_or_reply else "low"
    )
    forced_novelty = "high" if complete and not supports else "medium" if not complete else "low"
    abstract_filler = (
        "high"
        if complete and not supports
        else "medium"
        if source_type in {"thesis", "analysis"}
        else "low"
    )
    thesis_inflation = (
        "high"
        if source_type in {"factual_event", "news_update"} and not direct and not quote_or_reply
        else "low"
    )
    unsupported = "high" if not supports and content_mode.mode == "commentary" else "low"
    source_stronger = (
        "high"
        if complete and not supports
        else "medium"
        if complete and content_mode.mode == "commentary"
        else "low"
    )

    if not text:
        action = "skip"
        reason = "EMPTY_SOURCE"
        user_reason = "The ranked source has no usable text."
    elif direct and not complete:
        action = "reject_insufficient_depth"
        reason = "SOURCE_NOT_SEMANTICALLY_COMPLETE"
        user_reason = "The source does not contain a complete factual proposition."
    elif (
        supports
        and content_mode.factual_inspection_required
        and source.verification_status
        in {
            "single_source",
            "unverified",
            "unknown",
        }
    ):
        action = "generate_with_verification"
        reason = "SUPPORTED_SIGNAL_REQUIRES_VERIFICATION"
        user_reason = (
            "A supported factual transformation exists, but verification remains required."
        )
    elif supports:
        action = "generate"
        reason = "SUPPORTED_MODE_TRANSFORMATION"
        user_reason = "The source supports a grounded transformation for the requested mode."
    elif source_stronger == "high":
        action = "reject_source_stronger_than_draft"
        reason = "SOURCE_ALREADY_SUFFICIENT"
        user_reason = (
            "The source is already complete and no supported transformation exists for this mode."
        )
    elif forced_novelty == "high":
        action = "reject_high_inflation_risk"
        reason = "HIGH_FORCED_NOVELTY_RISK"
        user_reason = "Generating from this source would require an unsupported adjacent thesis."
    else:
        action = "reject_no_supported_transformation"
        reason = "NO_SUPPORTED_TRANSFORMATION"
        user_reason = "No grounded, mode-appropriate contribution is available."

    return SignalDecision(
        source_id=source.source_id or "source",
        source_type=source_type,
        core_proposition=core,
        semantically_complete=complete,
        supports_requested_mode=supports,
        supported_contribution_count=contribution_count,
        risk_source_echo=source_echo,
        risk_forced_novelty=forced_novelty,
        risk_abstract_filler=abstract_filler,
        risk_thesis_inflation=thesis_inflation,
        risk_unsupported_inference=unsupported,
        risk_source_stronger_than_draft=source_stronger,
        recommended_mode=content_mode.mode,
        action=action,
        reason_code=reason,
        user_reason=user_reason,
    )


def select_publishable_signal(
    sources: Sequence[UntrustedSourceData],
    *,
    content_mode: ContentModeDecision,
    max_attempts: int = 5,
    has_auxiliary_evidence: bool = False,
) -> SignalSelectionOutcome:
    """Accept the first ranked source with a supported mode transformation."""

    decisions: list[SignalDecision] = []
    for source in sources[: max(1, max_attempts)]:
        decision = evaluate_signal(
            source,
            content_mode=content_mode,
            has_auxiliary_evidence=has_auxiliary_evidence,
        )
        decisions.append(decision)
        if decision.action in {"generate", "generate_with_verification"}:
            return SignalSelectionOutcome(
                anchor=source,
                decision=decision,
                decisions=tuple(decisions),
                status="ok",
                reason_code=decision.reason_code,
            )
    reason = decisions[-1].reason_code if decisions else "NO_CANDIDATE_SOURCES"
    return SignalSelectionOutcome(
        anchor=None,
        decision=None,
        decisions=tuple(decisions),
        status="no_publishable_signal",
        reason_code=reason,
    )
