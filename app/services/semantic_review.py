"""Independent proposition- and paragraph-level semantic review.

This module deliberately does not rewrite prose. It classifies what each sentence is doing,
compares those jobs with the source coverage, and can decide that publication adds no value.
"""

from __future__ import annotations

import re

from app.schemas.content import (
    AddedValueAssessment,
    EditorialIntent,
    EvidencePacket,
    ParagraphJob,
    SemanticQualityReport,
    SentenceJob,
    SourceCoverage,
)
from app.services.generation_pipeline import extract_source_coverage

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[^\W_]{3,}", re.UNICODE)
_ABSTRACT_RESTATEMENT = re.compile(
    r"\b(?:change|shift|development|signal|dynamic|framework|implication|"
    r"concrete|tangible|meaningful|significant|part of .{0,35} itself|"
    r"turns? .{0,30} into|makes? .{0,30} concrete)\b",
    re.IGNORECASE,
)
_COMPARISON_TARGET = re.compile(
    r"\b(?:rather than|more (?:tangible|concrete|meaningful|significant) than|"
    r"general (?:discussion|argument|debate)|discussed in the abstract|"
    r"compared with (?:a )?(?:general|hypothetical))\b",
    re.IGNORECASE,
)
_DEPERSONALIZING = re.compile(
    r"\b(?:a|the) (?:named )?(?:officeholder|figure|organization|company|actor|entity)\b",
    re.IGNORECASE,
)
_OVEREXPLAIN = re.compile(
    r"\b(?:makes? (?:the )?.{0,45} concrete|more tangible development|"
    r"more concrete development|this makes .{0,40} meaningful|"
    r"the significance is|what makes this important)\b",
    re.IGNORECASE,
)
_RECAP = re.compile(
    r"^(?:in short|to recap|overall|in summary|put differently|in other words)\b", re.I
)
_TRANSITION = re.compile(
    r"^(?:first|second|third|next|meanwhile|more broadly|at the same time|that said|"
    r"on the other hand|this brings us to|the key point is)\b",
    re.I,
)
_FORCED_SYNTHESIS = re.compile(r"\b(?:taken together|all of this means|the broader lesson)\b", re.I)
_UNIVERSAL_LESSON = re.compile(
    r"\b(?:always|never|everyone|all companies|all teams|in life)\b", re.I
)
_RHETORICAL_QUESTION = re.compile(r"\?")
_QUOTE_CARD = re.compile(r"^(?:[A-Z][^.!?]{2,55}[.!?])$")
_IMPORTANCE_FRAME = re.compile(
    r"\b(?:the uncomfortable truth is|this is (?:an )?important (?:distinction|question)|"
    r"this distinction matters|this raises another important question|"
    r"here is why this matters|it is important to understand)\b",
    re.I,
)
_ANNOUNCEMENT_BEFORE_CONTENT = re.compile(
    r"^(?:the )?(?:interesting part|deeper question|overlooked risk) is that\b",
    re.I,
)
_FALSE_AGENCY = re.compile(
    r"^(?:the )?(?:market|internet|algorithm|platform|industry) "
    r"(?:decided|wanted|believed|chose|realized|refused)\b",
    re.I,
)
_UNASKED_REASSURANCE = re.compile(r"\b(?:and )?that is okay\.?$", re.I)
_PERMISSION_ENDING = re.compile(
    r"\b(?:you can leave|not every .{0,60} needs?)\b.*(?:that is okay)?\.?$",
    re.I | re.S,
)
_MECHANICAL_TRIAD = re.compile(
    r"\b(?:should|must|needs? to) be "
    r"[a-z][a-z-]{2,},\s+[a-z][a-z-]{2,},\s+and\s+[a-z][a-z-]{2,}\b",
    re.I,
)
_ACTORLESS_CAUSALITY = re.compile(
    r"^(?:the )?(?:decision|change|outcome|result|policy) "
    r"(?:emerged|happened|occurred|followed|came about)\b",
    re.I,
)
_BROAD_NARRATOR = re.compile(r"\b(?:people|you|everyone|companies|teams) always\b", re.I)
_EXPLAINS_JOKE = re.compile(r"\bthis is funny because\b", re.I)
_ABSTRACT_TEST = re.compile(
    r"\b(?:gives?|creates?|turns?|provides?) .{0,55}\btest\b|"
    r"\b(?:real-world|narrow|practical) test\b|\btest of (?:user )?demand\b",
    re.I,
)
_MANUFACTURED_QUESTION = re.compile(r"\bthe (?:real |practical |deeper )?question is\b", re.I)
_NOT_ONLY_REVEAL = re.compile(
    r"\bnot (?:only|just)\b.{0,160}\b(?:but|it is|it's)\b|"
    r"\bnot [^.!?]{2,80}, but\b",
    re.I | re.S,
)
_PRODUCT_MEMO = re.compile(
    r"\b(?:product shift|operating model|category expansion|active trading conditions|"
    r"format (?:remains|creates|provides|expands)|combination legibility|"
    r"practical question|user demand)\b",
    re.I,
)
_ABSTRACT_PREDICATE = re.compile(
    r"\b(?:gives?|creates?|turns?|becomes?|represents?|signals?|reflects?|increases?)\b"
    r".{0,75}\b(?:test|shift|development|importance|framework|dynamic|implication|legibility)\b",
    re.I,
)
_STRUCTURE_ABSTRACT_NOUN = re.compile(
    r"\b(?:format|test|question|combination|legibility|expansion|adoption|experience|"
    r"implication|behavior|utility|development|shift|framework|dynamic|ownership|legible|"
    r"intelligence|memory|model|strategy|importance|affordability)\b",
    re.I,
)
_RECAP_BRIDGE = re.compile(
    r"\b(?:this brings us to|because .{0,90}(?:omitted|missing|changed)|"
    r"in conclusion|to recap|in summary)\b",
    re.I,
)
_EMPTY_TRANSITION = re.compile(
    r"\b(?:this raises another important question|this brings us to|moving forward|"
    r"with that in mind|the next issue is|first, it is important to understand)\b",
    re.I,
)
_FORCED_SYNTHESIS_EXTENDED = re.compile(
    r"\b(?:together,? (?:these|the) developments|taken together|all of this means|"
    r"the broader lesson)\b",
    re.I,
)
_ABSTRACTION_CHAIN = re.compile(
    r"\b(?:institutional memory|organizational intelligence|operating model|"
    r"digital ownership|learning stack)\b",
    re.I,
)
_UNIVERSAL_FROM_EXAMPLE = re.compile(
    r"\b(?:this proves|this shows) that (?:every|all)\b|"
    r"\b(?:every|all) (?:company|team|organization)s? must\b",
    re.I,
)
_FORMULAIC_ENDING = re.compile(
    r"(?:that is|that's) the real (?:issue|risk|problem|gap|cost)\.?$|"
    r"\bnot [^.!?]{2,70}, but [^.!?]{2,70}\.?$",
    re.I,
)
_SYNTHETIC_COHESION = re.compile(
    r"\bfirst, it is important to understand\b.*\bmoving forward\b.*"
    r"\bwith that in mind\b",
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
    "part",
    "itself",
    "more",
    "general",
    "being",
    "now",
    "has",
    "have",
    "was",
    "were",
    "for",
    "but",
    "not",
    "about",
    "it",
    "its",
    "are",
    "is",
    "a",
    "an",
    "of",
    "to",
    "in",
}

_CRITICAL_NO_POST = {
    "NO_GROUNDED_CONTRIBUTION",
    "ANALYTICAL_PARAPHRASE_LOOP",
    "LEXICAL_NOVELTY_ONLY",
    "SEMANTIC_SOURCE_ECHO",
}


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in _SENTENCE_SPLIT.split(text.strip()) if item.strip()]


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _WORD.findall(text) if token.casefold() not in _STOP}


def _similarity(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(min(len(a), len(b)), 1)


def _source_overlap(sentence: str, coverage: SourceCoverage) -> float:
    claims = (*coverage.explicit_claims, *coverage.strong_implications)
    return max((_similarity(sentence, claim) for claim in claims), default=0.0)


def _evidence_ids(sentence: str, packet: EvidencePacket) -> tuple[str, ...]:
    scored = [(item.evidence_id, _similarity(sentence, item.text)) for item in packet.items]
    return tuple(item_id for item_id, score in scored if score >= 0.28)


def _sentence_job(sentence: str, coverage: SourceCoverage) -> str:
    lowered = sentence.casefold()
    if sentence.endswith("?"):
        return "question"
    if re.match(
        r"^(?:for .{1,80}, )?(?:start|define|name|record|use|note|keep|compare|"
        r"document|stop|choose|write|review|check|preserve|state|make|let)\b",
        sentence,
        re.IGNORECASE,
    ):
        return "action"
    if _ABSTRACT_RESTATEMENT.search(sentence):
        return "restate"
    if any(marker in lowered for marker in ("according to", "said", "says", "reported")):
        return "attribution"
    if any(marker in lowered for marker in ("because", "by ", "through ", "requires")):
        return "mechanism"
    if any(marker in lowered for marker in ("but", "while", "whereas", "without")):
        return "distinction"
    if any(marker in lowered for marker in ("if ", "unless", "only when")):
        return "condition"
    if any(marker in lowered for marker in ("however", "caveat", "except")):
        return "caveat"
    if _source_overlap(sentence, coverage) >= 0.45:
        return "fact"
    return "framing"


def _paragraph_jobs(text: str) -> tuple[ParagraphJob, ...]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    result: list[ParagraphJob] = []
    for index, paragraph in enumerate(paragraphs):
        overlaps = tuple(
            prior for prior in range(index) if _similarity(paragraph, paragraphs[prior]) >= 0.55
        )
        job = _sentence_job(paragraph, SourceCoverage())
        new_information = None if overlaps else paragraph
        result.append(
            ParagraphJob(
                paragraph_index=index,
                job=job,
                new_information=new_information,
                overlaps_paragraphs=overlaps,
            )
        )
    return tuple(result)


def review_structure(text: str, *, mode: str | None = None) -> tuple[str, ...]:
    """Return semantic/structural issue codes without rewriting the draft.

    The checks combine paragraph jobs, proposition overlap, cadence, and a small set of
    high-confidence rhetorical markers. The mode is used only for medium-specific controls.
    """

    sentences = _sentences(text)
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    lowered_mode = (mode or "").casefold()
    long_form = lowered_mode in {"long_form", "x_long_post", "long_post", "professional"} or (
        len(paragraphs) >= 2 and len(text) >= 120
    )
    issues: list[str] = []

    if _ANNOUNCEMENT_BEFORE_CONTENT.search(text) or any(
        _ANNOUNCEMENT_BEFORE_CONTENT.search(paragraph) for paragraph in paragraphs
    ):
        issues.append("ANNOUNCEMENT_BEFORE_CONTENT")
    if _FALSE_AGENCY.search(text):
        issues.extend(("FALSE_AGENCY", "HIDDEN_ACTOR"))
    if _UNASKED_REASSURANCE.search(text) and not re.search(
        r"\b(?:do not need to reply|take the time you need)\b", text, re.I
    ):
        issues.append("UNASKED_REASSURANCE")
    if _PERMISSION_ENDING.search(text):
        issues.extend(("PERMISSION_GRANTING_ENDING", "CUTTABILITY_FAILURE"))
    if _MECHANICAL_TRIAD.search(text):
        issues.append("MECHANICAL_TRIAD")

    if _IMPORTANCE_FRAME.search(text):
        issues.append("IMPORTANCE_ANNOUNCEMENT")
    if re.search(r"\bhere is why this matters\b|\bhere's why\b", text, re.I):
        issues.append("INSIGHT_PREVIEW")
    if _ACTORLESS_CAUSALITY.search(text):
        issues.extend(("ACTORLESS_CAUSALITY", "HIDDEN_ACTOR"))
    if _BROAD_NARRATOR.search(text):
        issues.extend(("NARRATOR_DISTANCE", "ARMCHAIR_GENERALIZATION"))
    if _EXPLAINS_JOKE.search(text):
        issues.extend(("EXPLANATION_AFTER_PUNCHLINE", "CUTTABILITY_FAILURE"))
    if re.search(r"\b(?:and that is okay|and that's okay|not every .{0,45} needs?)\b", text, re.I):
        issues.append("CUTTABILITY_FAILURE")
    if re.search(r"\bthis distinction matters\b", text, re.I):
        issues.extend(("SENTENCE_WITHOUT_JOB", "CUTTABILITY_FAILURE"))

    abstract_test = bool(_ABSTRACT_TEST.search(text))
    manufactured_question = bool(_MANUFACTURED_QUESTION.search(text))
    product_memo = bool(_PRODUCT_MEMO.search(text))
    abstract_predicate = bool(_ABSTRACT_PREDICATE.search(text))
    abstract_nouns = len(_STRUCTURE_ABSTRACT_NOUN.findall(text))
    if abstract_test:
        issues.extend(("ABSTRACT_TEST_FRAMING", "ANALYTICAL_PACKAGING"))
    if manufactured_question:
        issues.extend(("MANUFACTURED_QUESTION", "ANALYTICAL_PACKAGING"))
        if len(sentences) >= 2:
            issues.append("STAGED_THESIS")
    if _NOT_ONLY_REVEAL.search(text):
        issues.extend(("NOT_ONLY_BUT_REVEAL", "STAGED_THESIS"))
    if product_memo or (manufactured_question and re.search(r"\bformat|product|legib", text, re.I)):
        issues.append("PRODUCT_MEMO_VOICE")
    if abstract_predicate:
        issues.append("ABSTRACT_PREDICATE")
    if abstract_nouns >= (2 if manufactured_question else 3) and (
        abstract_predicate or manufactured_question or product_memo
    ):
        issues.append("ABSTRACT_NOUN_DENSITY")
    if re.search(r"\b(?:people|users|the market|user demand)\b", text, re.I):
        issues.append("VAGUE_USER_PROXY")
    if len(sentences) >= 3 and (
        abstract_test or manufactured_question or _NOT_ONLY_REVEAL.search(text)
    ):
        issues.append("MINI_ESSAY_ARC")

    if len(sentences) >= 3:
        sentence_openings = [
            " ".join(re.findall(r"[A-Za-z]+", sentence.casefold())[:2]) for sentence in sentences
        ]
        sentence_lengths = [len(re.findall(r"[A-Za-z]+", sentence)) for sentence in sentences]
        repeated_opening = max(
            (sentence_openings.count(item) for item in set(sentence_openings) if item),
            default=0,
        )
        uniform_length = max(sentence_lengths) - min(sentence_lengths) <= 3
        if repeated_opening >= 3 and uniform_length:
            issues.append("METRONOMIC_RHYTHM")

    if long_form:
        paragraph_jobs = _paragraph_jobs(text)
        overlap_count = sum(bool(job.overlaps_paragraphs) for job in paragraph_jobs)
        opener_families = [
            bool(
                re.match(
                    r"^(?:the interesting part|the deeper question|the overlooked risk)", p, re.I
                )
            )
            for p in paragraphs
        ]
        formulaic_endings = sum(bool(_FORMULAIC_ENDING.search(p)) for p in paragraphs)
        same_shape = False
        if len(paragraphs) >= 3:
            sentence_counts = [len(_sentences(p)) for p in paragraphs]
            same_shape = max(sentence_counts) - min(sentence_counts) <= 1
        if len(paragraphs) >= 3 and (
            sum(opener_families) >= 2
            or formulaic_endings >= 2
            or (same_shape and overlap_count >= 1)
        ):
            issues.append("REPEATED_PARAGRAPH_ARC")
        if len(paragraphs) >= 3 and (formulaic_endings >= 2 or sum(opener_families) >= 2):
            issues.append("PARAGRAPH_THESIS_LOOP")
        if formulaic_endings >= 2:
            issues.append("FORMULAIC_PARAGRAPH_ENDING")

        if len(paragraphs) >= 2:
            for index, paragraph in enumerate(paragraphs[1:], start=1):
                prior = " ".join(paragraphs[:index])
                recap_overlap = _similarity(paragraph, prior)
                if _RECAP_BRIDGE.search(paragraph) and (
                    recap_overlap >= 0.20 or re.search(r"\bthis brings us to\b", paragraph, re.I)
                ):
                    issues.append("RECAP_TRANSITION")
                if _EMPTY_TRANSITION.search(paragraph):
                    issues.append("TRANSITION_SCAFFOLDING")
                if _similarity(paragraph, paragraphs[index - 1]) >= 0.32 or (
                    paragraph.casefold().startswith("because ")
                    and len(_tokens(paragraph) & _tokens(paragraphs[index - 1])) >= 2
                ):
                    issues.append("PARAGRAPH_RESTATEMENT")
            if _EMPTY_TRANSITION.search(paragraphs[0]):
                issues.append("TRANSITION_SCAFFOLDING")

        for paragraph in paragraphs:
            content = _tokens(paragraph)
            only_bridge = bool(_EMPTY_TRANSITION.fullmatch(paragraph.rstrip(".? "))) or bool(
                re.fullmatch(
                    r"(?:this is an important distinction that deserves attention|"
                    r"this raises another important question)\.?",
                    paragraph,
                    re.I,
                )
            )
            if only_bridge or (len(content) <= 6 and _IMPORTANCE_FRAME.search(paragraph)):
                issues.append("PARAGRAPH_WITHOUT_JOB")
                issues.append("CUTTABILITY_FAILURE")

        if len(paragraphs) >= 2 and re.match(r"^in conclusion\b", paragraphs[-1], re.I):
            issues.extend(("CONCLUSION_RESTATEMENT", "PARAGRAPH_RESTATEMENT"))
        if _FORCED_SYNTHESIS_EXTENDED.search(text):
            issues.extend(("FORCED_SYNTHESIS", "ANALYTICAL_PACKAGING"))
        if _ABSTRACTION_CHAIN.search(text) and (
            len(_ABSTRACTION_CHAIN.findall(text)) >= 2 or _FORCED_SYNTHESIS_EXTENDED.search(text)
        ):
            issues.append("ABSTRACTION_CREEP")
        if _UNIVERSAL_FROM_EXAMPLE.search(text):
            issues.extend(("EXAMPLE_TO_UNIVERSAL_LESSON", "ARMCHAIR_GENERALIZATION"))
        if len(_RHETORICAL_QUESTION.findall(text)) >= 3:
            issues.append("RHETORICAL_QUESTION_CHAIN")
        if len(paragraphs) >= 3:
            counts = [len(_sentences(p)) for p in paragraphs]
            word_counts = [len(p.split()) for p in paragraphs]
            if max(counts) - min(counts) <= 0 and max(word_counts) - min(word_counts) <= 4:
                issues.append("UNIFORM_PARAGRAPH_CADENCE")
                paragraph_openings = [
                    " ".join(re.findall(r"[A-Za-z]+", paragraph.casefold())[:2])
                    for paragraph in paragraphs
                ]
                repeated_paragraph_opening = max(
                    (paragraph_openings.count(item) for item in set(paragraph_openings) if item),
                    default=0,
                )
                if repeated_paragraph_opening >= 3:
                    issues.append("METRONOMIC_RHYTHM")
        short_sentence_count = sum(len(sentence.split()) <= 12 for sentence in sentences)
        if len(sentences) >= 3 and short_sentence_count / len(sentences) >= 0.66:
            issues.append("QUOTE_CARD_SENTENCE_CLUSTER")
        if (
            lowered_mode in {"long_form", "x_long_post", "long_post"}
            and len(sentences) >= 3
            and all(len(sentence.split()) <= 4 for sentence in sentences)
            and len(text.split()) <= 16
        ):
            issues.extend(("LONG_FORM_OVERCOMPRESSION", "QUOTE_CARD_SENTENCE_CLUSTER"))
        if _SYNTHETIC_COHESION.search(text):
            issues.extend(("SYNTHETIC_COHESION", "TRANSITION_SCAFFOLDING"))

    return tuple(dict.fromkeys(issues))


def assess_added_value(
    text: str,
    *,
    packet: EvidencePacket,
    coverage: SourceCoverage | None = None,
    editorial_intent: EditorialIntent = "comment_on_source",
) -> AddedValueAssessment:
    """Score relation novelty, never vocabulary novelty."""

    source_map = coverage or extract_source_coverage(packet)
    propositions = tuple(_sentences(text))
    external_evidence = any(item.source_type != "user_input" for item in packet.items)
    open_composition = (
        editorial_intent in {"explain_topic", "long_form_analysis"} and not external_evidence
    )
    source_echo = tuple(
        sentence for sentence in propositions if _source_overlap(sentence, source_map) >= 0.38
    )
    supported_new: list[str] = []
    unsupported_new: list[str] = []
    source_item_hits: list[set[str]] = []
    for sentence in propositions:
        hits = set(_evidence_ids(sentence, packet))
        source_item_hits.append(hits)
        overlap = _source_overlap(sentence, source_map)
        combines_independent_evidence = (
            len(hits) >= 2
            and overlap < 0.80
            and bool(
                re.search(
                    r"\b(?:but|without|cannot|can't|requires?|therefore|so|which means)\b",
                    sentence,
                    re.I,
                )
            )
        )
        if combines_independent_evidence or (overlap < 0.38 and len(hits) >= 2):
            supported_new.append(sentence)
        elif (
            overlap < 0.22
            and not open_composition
            and not (
                editorial_intent in {"reply_reaction", "quote_reaction"}
                and len(sentence.split()) <= 24
            )
        ):
            unsupported_new.append(sentence)
        elif open_composition:
            # A user-supplied topic is an instruction to compose, not a factual source whose
            # every sentence must lexically overlap. Factual safety remains in deterministic
            # checks; this layer must not mistake ordinary exposition for an evidence gap.
            supported_new.append(sentence)
    if editorial_intent in {
        "report_event",
        "rewrite_existing",
        "explain_topic",
        "long_form_analysis",
        "reply_reaction",
        "quote_reaction",
    }:
        decision = "pass" if propositions and not unsupported_new else "reject"
    elif supported_new:
        decision = "pass"
    elif unsupported_new:
        decision = "reject"
    else:
        decision = "no_post_needed"
    relation_novelty = min(1.0, len(supported_new) / max(len(propositions), 1))
    grounding = max(
        0.0,
        min(1.0, 1.0 - len(unsupported_new) / max(len(propositions), 1)),
    )
    return AddedValueAssessment(
        direct_source_claims=source_map.explicit_claims,
        draft_propositions=propositions,
        supported_new_relations=tuple(supported_new),
        unsupported_new_relations=tuple(unsupported_new),
        source_echo_propositions=source_echo,
        relation_novelty_score=relation_novelty,
        grounding_score=grounding,
        decision=decision,
    )


def review_semantics(
    text: str,
    *,
    packet: EvidencePacket,
    editorial_intent: EditorialIntent = "comment_on_source",
    coverage: SourceCoverage | None = None,
) -> SemanticQualityReport:
    """Classify sentence/paragraph jobs and decide accept, revise, or no-post."""

    source_map = coverage or extract_source_coverage(packet)
    sentences = _sentences(text)
    external_evidence = any(item.source_type != "user_input" for item in packet.items)
    open_composition = (
        editorial_intent in {"explain_topic", "long_form_analysis"} and not external_evidence
    )
    jobs: list[SentenceJob] = []
    issue_codes: list[str] = []
    overlaps: list[float] = []
    for index, sentence in enumerate(sentences):
        overlap = _source_overlap(sentence, source_map)
        overlaps.append(overlap)
        job = _sentence_job(sentence, source_map)
        evidence_ids = _evidence_ids(sentence, packet)
        if (
            editorial_intent in {"reply_reaction", "quote_reaction"}
            and not evidence_ids
            and len(sentence.split()) <= 24
        ):
            job = "reaction"
        jobs.append(
            SentenceJob(
                sentence_index=index,
                job=job,
                proposition=sentence,
                evidence_ids=evidence_ids,
            )
        )
        if job in {"framing", "importance_announcement"} and not evidence_ids:
            issue_codes.append("SENTENCE_WITHOUT_JOB")
        if job == "restate" and not evidence_ids:
            issue_codes.append("SENTENCE_WITHOUT_JOB")
        if not open_composition and _ABSTRACT_RESTATEMENT.search(sentence):
            issue_codes.append("ABSTRACT_RESTATEMENT")
        if not open_composition and _COMPARISON_TARGET.search(sentence):
            issue_codes.append("INVENTED_COMPARISON_TARGET")
        if not open_composition and _DEPERSONALIZING.search(sentence) and source_map.named_entities:
            issue_codes.append("DEPERSONALIZING_RENAME")
        if not open_composition and _OVEREXPLAIN.search(sentence):
            issue_codes.append("SOURCE_FACT_OVEREXPLANATION")
    paragraph_jobs = _paragraph_jobs(text)
    if any(item.overlaps_paragraphs for item in paragraph_jobs):
        issue_codes.extend(("PARAGRAPH_JOB_DUPLICATION", "REPEATED_PARAGRAPH_ARC"))
    if len(sentences) >= 2 and not open_composition:
        abstract_jobs = sum(job.job == "restate" for job in jobs)
        repeated_pairs = sum(
            _similarity(sentences[left], sentences[right]) >= 0.32
            or (jobs[left].job == jobs[right].job == "restate")
            for left in range(len(sentences))
            for right in range(left + 1, len(sentences))
        )
        if abstract_jobs >= 2 or repeated_pairs >= 2:
            issue_codes.extend(("ANALYTICAL_PARAPHRASE_LOOP", "SAME_POINT_MULTIPLE_WAYS"))
    value = assess_added_value(
        text,
        packet=packet,
        coverage=source_map,
        editorial_intent=editorial_intent,
    )
    source_overlap = sum(overlaps) / max(len(overlaps), 1)
    source_words = set().union(*(_tokens(claim) for claim in source_map.explicit_claims))
    draft_words = _tokens(text)
    lexical_novelty = len(draft_words - source_words) / max(len(draft_words), 1)
    if value.decision == "no_post_needed":
        issue_codes.extend(("NO_GROUNDED_CONTRIBUTION", "SEMANTIC_SOURCE_ECHO"))
    if (
        editorial_intent == "comment_on_source"
        and not value.supported_new_relations
        and lexical_novelty >= 0.30
    ):
        issue_codes.append("LEXICAL_NOVELTY_ONLY")
    if (
        value.source_echo_propositions
        and not value.supported_new_relations
        and editorial_intent == "comment_on_source"
    ):
        issue_codes.append("SEMANTIC_SOURCE_ECHO")
    if value.unsupported_new_relations:
        issue_codes.append("UNSUPPORTED_RELATION")
    # Long-form semantic structure.
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    if len(paragraphs) >= 2:
        if any(_RECAP.search(item) for item in paragraphs[1:]):
            issue_codes.append("RECAP_TRANSITION")
        if sum(bool(_TRANSITION.search(item)) for item in paragraphs) >= 2:
            issue_codes.append("TRANSITION_SCAFFOLDING")
        if _FORCED_SYNTHESIS.search(paragraphs[-1]):
            issue_codes.append("FORCED_SYNTHESIS")
        if _similarity(paragraphs[-1], " ".join(paragraphs[:-1])) >= 0.45:
            issue_codes.append("CONCLUSION_RESTATEMENT")
        lengths = [len(_sentences(item)) for item in paragraphs]
        if len(lengths) >= 3 and max(lengths) - min(lengths) <= 1:
            issue_codes.append("UNIFORM_PARAGRAPH_CADENCE")
        if sum(bool(_ABSTRACT_RESTATEMENT.search(item)) for item in paragraphs) >= 2:
            issue_codes.append("ABSTRACTION_CREEP")
    if _UNIVERSAL_LESSON.search(text) and len(packet.items) <= 2:
        issue_codes.append("EXAMPLE_TO_UNIVERSAL_LESSON")
    if len(_RHETORICAL_QUESTION.findall(text)) >= 3:
        issue_codes.append("RHETORICAL_QUESTION_CHAIN")
    if len(sentences) >= 3 and sum(bool(_QUOTE_CARD.match(item)) for item in sentences) >= 3:
        issue_codes.append("QUOTE_CARD_SENTENCE_CLUSTER")
    if (
        len(text) > 500
        and len(sentences) >= 8
        and sum(len(item.split()) <= 8 for item in sentences) >= 6
    ):
        issue_codes.append("LONG_FORM_OVERCOMPRESSION")
    issue_codes.extend(review_structure(text, mode=packet.requested_format))
    deduped = tuple(dict.fromkeys(issue_codes))
    if editorial_intent == "comment_on_source" and (
        value.decision == "no_post_needed" or _CRITICAL_NO_POST.intersection(deduped)
    ):
        decision = "no_post_needed"
    elif any(
        code in deduped
        for code in (
            "UNSUPPORTED_RELATION",
            "INVENTED_COMPARISON_TARGET",
            "DEPERSONALIZING_RENAME",
            "ANALYTICAL_PARAPHRASE_LOOP",
            "PARAGRAPH_JOB_DUPLICATION",
        )
    ):
        decision = "revise"
    else:
        decision = "accept"
    grounded = value.supported_new_relations[0] if value.supported_new_relations else None
    literal_core = next(
        (job.proposition for job in jobs if job.job in {"fact", "mechanism", "distinction"}),
        sentences[0] if sentences else "",
    )
    return SemanticQualityReport(
        decision=decision,
        literal_core=literal_core,
        source_coverage_overlap=max(0.0, min(source_overlap, 1.0)),
        grounded_contribution=grounded,
        sentence_jobs=tuple(jobs),
        paragraph_jobs=paragraph_jobs,
        issue_codes=deduped,
        unsupported_relations=value.unsupported_new_relations,
        explanation=(
            "No supported proposition remains after source echoes and framing are removed."
            if decision == "no_post_needed"
            else "Semantic sentence and paragraph jobs were reviewed independently of phrase lint."
        ),
    )
