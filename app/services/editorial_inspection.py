"""Categorical semantic editorial gate for grounded generation.

The inspector compares propositions and evidence relationships. It does not rewrite and it does
not turn the absence of regex matches into a synthetic quality score.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.schemas.content import (
    AngleCandidate,
    ContentModeDecision,
    DraftSemanticInspection,
    EditorialIntent,
    EditorialIssue,
    EditorialReport,
    EvidencePacket,
    GenerationVariant,
    SourceCoverage,
)
from app.services.generation_pipeline import extract_source_coverage
from app.services.semantic_classifier import inspect_reaction_function
from app.services.semantic_review import assess_added_value, review_semantics

CANONICAL_EDITORIAL_ISSUES: frozenset[str] = frozenset(
    {
        "SOURCE_ECHO",
        "IMPLIED_POINT_RESTATEMENT",
        "LOW_INFORMATION_GAIN",
        "UNSUPPORTED_CONTRIBUTION",
        "FORCED_NOVELTY",
        "ABSTRACT_FILLER",
        "ANALYTICAL_PACKAGING",
        "MANUFACTURED_CONTRAST",
        "INTERNAL_REPETITION",
        "PROMISED_INSIGHT_NOT_DELIVERED",
        "THESIS_INFLATION",
        "SOURCE_STRONGER_THAN_DRAFT",
        "FORCED_LENGTH",
        "VARIANTS_NOT_DISTINCT",
        "INCOHERENT_SYNTHESIS",
        "EVIDENCE_INVALID",
        "CONTRIBUTION_CHANGED_DURING_REVISION",
        "CONTRIBUTION_DETAIL_DROPPED",
        "CONTRIBUTION_PROPOSITION_DROPPED",
        "CONTRIBUTION_PROPOSITION_CHANGED",
        "CONTRIBUTION_CLAUSE_RELATION_CHANGED",
        "CONTRIBUTION_QUALIFIER_CHANGED",
        "REACTION_LABEL_WITHOUT_FUNCTION",
        "PLAIN_PARAPHRASE_AS_REACTION",
        "HUMOR_ON_TRAGEDY",
        "HUMOR_INTENT_UNCERTAIN",
        "HUMOR_INTENT_CONFLICT",
        "CERTAINTY_CHANGED_DURING_REVISION",
        "ATTRIBUTION_CHANGED_DURING_REVISION",
    }
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[^\W_]{3,}", re.UNICODE)
_ANALYTICAL_PACKAGING = re.compile(
    r"\b(?:narrow|practical|real[- ]world) test\b|"
    r"\bthe (?:practical|real|deeper|key) question is\b|"
    r"\bnot (?:only|just)\b.{0,180}\b(?:but|it is|it's)\b|"
    r"\b(?:framework|product shift|operating model|format legibility|category expansion)\b|"
    r"\b(?:reveals?|turns? .{0,55} into|becomes? part of .{0,60} itself|"
    r"makes? .{0,55} concrete|more tangible development|narrow promise|"
    r"contract problem gets messy|who owns (?:that )?trace)\b",
    re.IGNORECASE | re.DOTALL,
)
_ABSTRACT_FILLER = re.compile(
    r"\b(?:shift|development|dynamic|framework|signal|implication|importance|"
    r"tangible|concrete|meaningful|legibility|format|product strategy|"
    r"combination|category|promise|trace|ownership)\b",
    re.IGNORECASE,
)
_MANUFACTURED_CONTRAST = re.compile(
    r"\b(?:rather than|more (?:tangible|concrete|meaningful) than|"
    r"general (?:discussion|argument|debate)|discussed in the abstract|"
    r"not (?:only|just).{0,160}(?:but|it is|it's))\b",
    re.IGNORECASE | re.DOTALL,
)
_PROMISED_INSIGHT = re.compile(
    r"\b(?:the (?:real|deeper|interesting|important) (?:point|question|part) is|"
    r"what this (?:really )?(?:shows|reveals|means)|here is why this matters)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_ADJACENCY = re.compile(
    r"\b(?:ownership|contract|legal|liability|private context|employee(?:'s)? judgment|"
    r"migration failure|technical incompatibility|stakeholder|who owns|take it to another model)\b",
    re.IGNORECASE,
)
_ABSTRACT_TEST = re.compile(r"\b(?:narrow|practical|real[- ]world) test\b", re.I)
_PRODUCT_MEMO = re.compile(
    r"\b(?:product shift|operating model|active trading conditions|category expansion|"
    r"combination remains legible|user demand|format)\b",
    re.I,
)
_STAGED_THESIS = re.compile(
    r"\bthe (?:practical|real|deeper) question is\b|\bnot only\b.{0,160}\bit is\b",
    re.I | re.S,
)

_STOP = {
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "into",
    "than",
    "rather",
    "being",
    "have",
    "has",
    "was",
    "were",
    "for",
    "but",
    "not",
    "about",
    "its",
    "are",
    "is",
    "of",
    "to",
    "in",
    "on",
    "a",
    "an",
}


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip())


def _tokens(text: str) -> set[str]:
    return {word.casefold() for word in _WORD.findall(text) if word.casefold() not in _STOP}


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(min(len(left_tokens), len(right_tokens)), 1)


def _source_overlap(sentence: str, coverage: SourceCoverage) -> float:
    claims = (*coverage.explicit_claims, *coverage.strong_implications)
    return max((_similarity(sentence, claim) for claim in claims), default=0.0)


def _issue(
    code: str,
    span: str,
    explanation: str,
    *,
    severity: str = "block",
    subtypes: Sequence[str] = (),
) -> EditorialIssue:
    if code not in CANONICAL_EDITORIAL_ISSUES:
        raise ValueError(f"Unknown canonical editorial issue: {code}")
    return EditorialIssue(
        code=code,
        severity=severity,
        evidence_spans=(span or "[semantic relation]",),
        explanation=explanation,
        subtypes=tuple(subtypes),
    )


def _add_once(issues: list[EditorialIssue], issue: EditorialIssue) -> None:
    key = (issue.code, issue.evidence_spans, issue.subtypes)
    if any((item.code, item.evidence_spans, item.subtypes) == key for item in issues):
        return
    issues.append(issue)


def _independent_source_relation(packet: EvidencePacket, evidence_ids: set[str]) -> bool:
    selected = [item for item in packet.items if item.evidence_id in evidence_ids]
    if len(selected) < 2:
        return True
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            if _similarity(left.text, right.text) >= 0.42:
                return True
    return False


def inspect_editorial(
    text: str,
    *,
    packet: EvidencePacket,
    contribution: AngleCandidate | None,
    content_mode: ContentModeDecision,
    coverage: SourceCoverage | None = None,
    other_drafts: Sequence[str] = (),
    preferred_length_min: int | None = None,
) -> EditorialReport:
    """Inspect one draft against source coverage and the selected grounded contribution."""

    source_map = coverage or extract_source_coverage(packet)
    sentences = _sentences(text)
    issues: list[EditorialIssue] = []
    overlaps = tuple(_source_overlap(sentence, source_map) for sentence in sentences)
    max_overlap = max(overlaps, default=0.0)
    average_overlap = sum(overlaps) / max(len(overlaps), 1)
    source_overlap = (
        "high"
        if max_overlap >= 0.72 or average_overlap >= 0.52
        else ("partial" if max_overlap >= 0.34 else "none")
    )

    direct_transmission = content_mode.mode in {"factual_update", "summary"}
    open_composition = content_mode.mode == "opinion_or_creative"
    reaction_mode = content_mode.subtype in {"reply_commentary", "quote_commentary"}
    inspection_intent: EditorialIntent = (
        "report_event"
        if content_mode.mode == "factual_update"
        else "rewrite_existing"
        if content_mode.mode == "summary"
        else "explain_topic"
        if open_composition
        else "reply_reaction"
        if content_mode.subtype == "reply_commentary"
        else "quote_reaction"
        if content_mode.subtype == "quote_commentary"
        else "comment_on_source"
    )
    semantic = review_semantics(
        text,
        packet=packet,
        coverage=source_map,
        editorial_intent=inspection_intent,
    )
    added_value = assess_added_value(
        text,
        packet=packet,
        coverage=source_map,
        editorial_intent=inspection_intent,
    )

    source_specific_reaction = False
    if contribution is None:
        _add_once(
            issues,
            _issue(
                "UNSUPPORTED_CONTRIBUTION",
                text,
                "No validated contribution was supplied to the drafter.",
            ),
        )
        contribution_support = "unsupported"
        contribution_delivered = False
    else:
        available_ids = {item.evidence_id for item in packet.items}
        invalid_ids = set(contribution.evidence_ids) - available_ids
        if invalid_ids:
            _add_once(
                issues,
                _issue(
                    "EVIDENCE_INVALID",
                    ", ".join(sorted(invalid_ids)),
                    "The selected contribution cites evidence IDs that do not exist.",
                ),
            )
        if (
            contribution.requires_new_assumptions
            or contribution.unsupported_assumptions
            or contribution.support_status == "unsupported"
        ):
            assumptions = "; ".join(contribution.unsupported_assumptions) or contribution.thesis
            _add_once(
                issues,
                _issue(
                    "UNSUPPORTED_CONTRIBUTION",
                    assumptions,
                    "The selected contribution requires a relation not established by evidence.",
                ),
            )
            _add_once(
                issues,
                _issue(
                    "FORCED_NOVELTY",
                    assumptions,
                    (
                        "The contribution is distinct only because it adds an unsupported "
                        "adjacent complication."
                    ),
                ),
            )
            contribution_support = "unsupported"
        elif contribution.support_status == "partially_supported":
            contribution_support = "partial"
        else:
            contribution_support = "supported"
        thesis_similarity = max(
            (_similarity(sentence, contribution.thesis) for sentence in sentences), default=0.0
        )
        cited_by_semantic_job = bool(
            set(contribution.evidence_ids)
            & {evidence_id for job in semantic.sentence_jobs for evidence_id in job.evidence_ids}
        )
        reaction_inspection = inspect_reaction_function(packet, text) if reaction_mode else None
        source_specific_reaction = bool(
            reaction_mode
            and contribution.contribution_type
            in {
                "plain_observation",
                "direct_response",
                "specific_qualification",
                "genuine_question",
                "contextual_extension",
                "dry_humor",
                "dry_reframe",
                "joke",
            }
            and len(text.split()) <= 40
            and reaction_inspection is not None
            and reaction_inspection.source_specific
            and not reaction_inspection.source_echo
            and not reaction_inspection.unsupported_detail
        )
        contribution_delivered = (
            thesis_similarity >= 0.22 or cited_by_semantic_job or source_specific_reaction
        )
        if not contribution_delivered:
            _add_once(
                issues,
                _issue(
                    "PROMISED_INSIGHT_NOT_DELIVERED",
                    contribution.thesis,
                    (
                        "The draft does not deliver the validated contribution supplied "
                        "by the planner."
                    ),
                ),
            )

    source_echo_sentences = [
        sentence for sentence, overlap in zip(sentences, overlaps, strict=True) if overlap >= 0.55
    ]
    if (
        source_echo_sentences
        and not direct_transmission
        and not open_composition
        and not source_specific_reaction
    ):
        _add_once(
            issues,
            _issue(
                "SOURCE_ECHO",
                source_echo_sentences[0],
                (
                    "The draft repeats a source proposition instead of adding a "
                    "mode-appropriate contribution."
                ),
            ),
        )
    implied_spans = [
        sentence
        for sentence, overlap in zip(sentences, overlaps, strict=True)
        if 0.34 <= overlap < 0.72 and _ABSTRACT_FILLER.search(sentence)
    ]
    if (
        implied_spans
        and not direct_transmission
        and not open_composition
        and not source_specific_reaction
    ):
        _add_once(
            issues,
            _issue(
                "IMPLIED_POINT_RESTATEMENT",
                implied_spans[0],
                "The sentence repackages a strong source implication in more abstract language.",
            ),
        )

    if _ANALYTICAL_PACKAGING.search(text):
        subtypes: list[str] = []
        if _ABSTRACT_TEST.search(text):
            subtypes.append("abstract_test_framing")
        if _STAGED_THESIS.search(text):
            subtypes.append("staged_thesis")
        if _PRODUCT_MEMO.search(text):
            subtypes.append("product_memo_voice")
        if len(_ABSTRACT_FILLER.findall(text)) >= 2:
            subtypes.append("abstract_noun_density")
        _add_once(
            issues,
            _issue(
                "ANALYTICAL_PACKAGING",
                next(
                    (sentence for sentence in sentences if _ANALYTICAL_PACKAGING.search(sentence)),
                    text,
                ),
                (
                    "A direct observation was inflated into an abstract test, question, "
                    "reveal, or product memo."
                ),
                subtypes=subtypes,
            ),
        )
    if _MANUFACTURED_CONTRAST.search(text):
        _add_once(
            issues,
            _issue(
                "MANUFACTURED_CONTRAST",
                next(
                    (sentence for sentence in sentences if _MANUFACTURED_CONTRAST.search(sentence)),
                    text,
                ),
                (
                    "The comparison target or contrast was created by the draft rather "
                    "than supplied by evidence."
                ),
            ),
        )
    if _PROMISED_INSIGHT.search(text) and not added_value.supported_new_relations:
        _add_once(
            issues,
            _issue(
                "PROMISED_INSIGHT_NOT_DELIVERED",
                next(
                    (sentence for sentence in sentences if _PROMISED_INSIGHT.search(sentence)), text
                ),
                "The draft announces an insight but does not add a supported proposition.",
            ),
        )

    repeated_span: str | None = None
    for left_index, left in enumerate(sentences):
        for right in sentences[left_index + 1 :]:
            if _similarity(left, right) >= 0.48:
                repeated_span = f"{left} || {right}"
                break
        if repeated_span:
            break
    if (
        repeated_span
        or {"ANALYTICAL_PARAPHRASE_LOOP", "SAME_POINT_MULTIPLE_WAYS"} & set(semantic.issue_codes)
        or (
            len(sentences) >= 3
            and _ANALYTICAL_PACKAGING.search(text)
            and sum(bool(_ABSTRACT_FILLER.search(sentence)) for sentence in sentences) >= 2
        )
    ):
        _add_once(
            issues,
            _issue(
                "INTERNAL_REPETITION",
                repeated_span or text,
                "Multiple sentences perform the same semantic job through different wording.",
            ),
        )

    unsupported_adjacent = _UNSUPPORTED_ADJACENCY.search(text)
    if unsupported_adjacent and not any(
        unsupported_adjacent.group(0).casefold() in item.text.casefold() for item in packet.items
    ):
        _add_once(
            issues,
            _issue(
                "FORCED_NOVELTY",
                unsupported_adjacent.group(0),
                (
                    "The draft invents an ownership, contract, legal, stakeholder, or "
                    "technical problem to appear analytical."
                ),
            ),
        )
        _add_once(
            issues,
            _issue(
                "UNSUPPORTED_CONTRIBUTION",
                unsupported_adjacent.group(0),
                "The introduced relationship is absent from canonical evidence.",
            ),
        )

    supported_relations = (
        bool(added_value.supported_new_relations)
        or direct_transmission
        or open_composition
        or (reaction_mode and contribution_delivered)
    )
    information_gain = (
        "clear"
        if supported_relations
        else ("weak" if sentences and not added_value.unsupported_new_relations else "none")
    )
    if information_gain in {"weak", "none"} and not direct_transmission and not open_composition:
        _add_once(
            issues,
            _issue(
                "LOW_INFORMATION_GAIN",
                text,
                (
                    "After source restatement and rhetorical framing are removed, no "
                    "clear supported addition remains."
                ),
            ),
        )
    if (
        source_echo_sentences
        and not direct_transmission
        and not open_composition
        and (_ANALYTICAL_PACKAGING.search(text) or _ABSTRACT_FILLER.search(text))
        and not any(issue.code == "IMPLIED_POINT_RESTATEMENT" for issue in issues)
    ):
        _add_once(
            issues,
            _issue(
                "IMPLIED_POINT_RESTATEMENT",
                source_echo_sentences[0],
                (
                    "The draft compresses or reframes a source conclusion that the source "
                    "already established."
                ),
            ),
        )

    source_length = max(
        (len(item.text) for item in packet.items if item.source_type != "user_input"), default=0
    )
    source_stronger = (
        source_length > 0
        and len(text) > source_length * 1.25
        and information_gain != "clear"
        and bool(_ABSTRACT_FILLER.search(text))
    )
    if source_stronger:
        _add_once(
            issues,
            _issue(
                "SOURCE_STRONGER_THAN_DRAFT",
                text,
                (
                    "The expansion is longer and more abstract than its source without "
                    "adding useful supported information."
                ),
            ),
        )
    if len(_ABSTRACT_FILLER.findall(text)) >= 3 and information_gain != "clear":
        _add_once(
            issues,
            _issue(
                "ABSTRACT_FILLER",
                text,
                (
                    "Abstract nouns and framing replace concrete actors, actions, or "
                    "evidence-backed relations."
                ),
            ),
        )
        _add_once(
            issues,
            _issue(
                "THESIS_INFLATION",
                text,
                (
                    "The draft turns a bounded source observation into a larger thesis "
                    "without evidence."
                ),
            ),
        )

    if (
        preferred_length_min is not None
        and len(text) >= preferred_length_min
        and information_gain != "clear"
        and (
            any(issue.code == "INTERNAL_REPETITION" for issue in issues)
            or any(issue.code == "ABSTRACT_FILLER" for issue in issues)
        )
    ):
        _add_once(
            issues,
            _issue(
                "FORCED_LENGTH",
                text,
                (
                    "The preferred length appears to have been reached through repetition "
                    "or abstraction rather than supported content."
                ),
            ),
        )

    evidence_ids = set(contribution.evidence_ids) if contribution is not None else set()
    if len(evidence_ids) > 1 and not _independent_source_relation(packet, evidence_ids):
        _add_once(
            issues,
            _issue(
                "INCOHERENT_SYNTHESIS",
                ", ".join(sorted(evidence_ids)),
                (
                    "The cited sources share a topic but no propositional relationship "
                    "justifies synthesis."
                ),
                subtypes=("topic_overlap_only", "no_propositional_relation", "artificial_contrast"),
            ),
        )

    for other in other_drafts:
        if _similarity(text, other) >= 0.76:
            _add_once(
                issues,
                _issue(
                    "VARIANTS_NOT_DISTINCT",
                    f"{text} || {other}",
                    (
                        "Two variants express the same proposition; wording changes do not "
                        "create substantive diversity."
                    ),
                ),
            )
            break

    blocking = any(issue.severity == "block" for issue in issues)
    status = "block" if blocking else "warning" if issues else "pass"
    return EditorialReport(
        status=status,
        source_overlap=source_overlap,
        contribution_support=contribution_support,
        information_gain=information_gain,
        internal_redundancy=(
            "present" if any(issue.code == "INTERNAL_REPETITION" for issue in issues) else "none"
        ),
        source_stronger_than_draft=source_stronger,
        contribution_delivered=contribution_delivered,
        issues=tuple(issues),
    )


def inspect_variants(
    variants: Sequence[GenerationVariant],
    *,
    packet: EvidencePacket,
    contributions: Sequence[AngleCandidate],
    content_mode: ContentModeDecision,
    coverage: SourceCoverage | None = None,
    preferred_length_min: int | None = None,
) -> tuple[EditorialReport, ...]:
    """Inspect zero to three variants and compare their propositions with each other."""

    contribution_by_id = {item.angle_id: item for item in contributions}
    reports: list[EditorialReport] = []
    texts = [variant.text for variant in variants]
    for index, variant in enumerate(variants):
        report = inspect_editorial(
            variant.text,
            packet=packet,
            contribution=contribution_by_id.get(variant.angle_id),
            content_mode=content_mode,
            coverage=coverage,
            other_drafts=(*texts[:index], *texts[index + 1 :]),
            preferred_length_min=preferred_length_min,
        )
        reports.append(report)
    return tuple(reports)


def apply_structured_semantic_inspection(
    report: EditorialReport,
    inspection: DraftSemanticInspection,
) -> EditorialReport:
    """Merge structured semantic findings; categorical blocks override legacy scores."""

    if not isinstance(inspection, DraftSemanticInspection):
        raise TypeError("inspection must be DraftSemanticInspection")
    issues = list(report.issues)
    spans = inspection.evidence_spans or ("[structured semantic relation]",)
    explanations = {
        "SOURCE_ECHO": "The draft repeats an explicit source proposition.",
        "IMPLIED_POINT_RESTATEMENT": (
            "The draft restates a source implication through different wording."
        ),
        "LOW_INFORMATION_GAIN": "Removing source restatement leaves no useful grounded addition.",
        "ANALYTICAL_PACKAGING": (
            "The draft inflates a direct point into abstract analytical packaging."
        ),
        "MANUFACTURED_CONTRAST": "The draft creates a contrast not required by evidence.",
        "INTERNAL_REPETITION": "Multiple sentences perform the same proposition-level job.",
        "SOURCE_STRONGER_THAN_DRAFT": (
            "The generated expansion is longer and less useful than its source."
        ),
        "PROMISED_INSIGHT_NOT_DELIVERED": (
            "The selected contribution is not delivered in final text."
        ),
        "UNSUPPORTED_CONTRIBUTION": "The structured inspection found no supported contribution.",
        "FORCED_NOVELTY": "The draft relies on an unsupported adjacent complication.",
        "CONTRIBUTION_CHANGED_DURING_REVISION": (
            "The revised draft changed the selected contribution."
        ),
        "CONTRIBUTION_DETAIL_DROPPED": "The revised draft dropped a material contribution detail.",
        "CONTRIBUTION_PROPOSITION_DROPPED": (
            "The revised draft removed one of the planned contribution propositions."
        ),
        "CONTRIBUTION_PROPOSITION_CHANGED": (
            "The revised draft replaced a planned contribution proposition."
        ),
        "CONTRIBUTION_CLAUSE_RELATION_CHANGED": (
            "The revised draft changed how planned contribution clauses relate."
        ),
        "REACTION_LABEL_WITHOUT_FUNCTION": (
            "The provider reaction label is not supported by a semantic reaction function."
        ),
        "PLAIN_PARAPHRASE_AS_REACTION": (
            "The draft is a source paraphrase with reaction-shaped packaging."
        ),
        "HUMOR_ON_TRAGEDY": (
            "The reply uses humor on victims, deaths, serious injury, or another sensitive harm."
        ),
        "HUMOR_INTENT_UNCERTAIN": (
            "Humor intent remains uncertain for a sensitive source and requires fail-closed review."
        ),
        "HUMOR_INTENT_CONFLICT": (
            "Deterministic and provider humor-intent inspections materially disagree."
        ),
        "CONTRIBUTION_QUALIFIER_CHANGED": (
            "The revised draft removed or changed a material event qualifier."
        ),
        "CERTAINTY_CHANGED_DURING_REVISION": "The revised draft changed certainty or event status.",
        "ATTRIBUTION_CHANGED_DURING_REVISION": "The revised draft changed or removed attribution.",
    }
    for code in inspection.issue_codes:
        if code not in CANONICAL_EDITORIAL_ISSUES:
            continue
        _add_once(
            issues,
            _issue(
                code,
                spans[0],
                explanations.get(code, "Structured semantic inspection blocked this relation."),
            ),
        )
    if inspection.no_grounded_contribution and not any(
        item.code == "LOW_INFORMATION_GAIN" for item in issues
    ):
        _add_once(
            issues,
            _issue(
                "LOW_INFORMATION_GAIN",
                spans[0],
                "No grounded contribution remains after source restatement is removed.",
            ),
        )
    blocked = any(item.severity == "block" for item in issues)
    return report.model_copy(
        update={
            "status": "block" if blocked else report.status,
            "source_overlap": ("high" if inspection.source_echo else report.source_overlap),
            "information_gain": (
                "none"
                if inspection.no_grounded_contribution
                else "weak"
                if inspection.implied_restatement
                else report.information_gain
            ),
            "internal_redundancy": (
                "present" if inspection.internal_repetition else report.internal_redundancy
            ),
            "source_stronger_than_draft": (
                report.source_stronger_than_draft or inspection.source_stronger_than_draft
            ),
            "contribution_delivered": (
                report.contribution_delivered and inspection.contribution_delivered
            ),
            "issues": tuple(issues),
        }
    )
