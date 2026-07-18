"""Strict contracts for evidence-grounded content generation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import ContentType, GenerationMode, MediaType
from app.domain.enums import FactCheckStatus as FactCheckStatus

ContentFormat = ContentType
VerificationStatus = Literal[
    "direct_source",
    "corroborated",
    "single_source",
    "unverified",
    "opinion",
    "unknown",
]
EvidenceSourceType = Literal[
    "user_input",
    "x_post",
    "x_thread",
    "x_search",
    "x_home_timeline",
    "x_activity",
    "official_statement",
    "official_document",
    "news_report",
    "onchain_data",
    "market_data",
    "approved_post",
    "existing_draft",
]
AngleType = Literal[
    "plain_update",
    "consequence",
    "mechanism",
    "incentive",
    "tradeoff",
    "skeptical_caveat",
    "contradiction",
    "comparison",
    "timing",
    "practical_implication",
    "specific_question",
    "source_quality",
    "dry_observation",
    "market_structure",
    "concise_joke",
]
Confidence = Literal["high", "medium", "low"]
RevisionStatus = Literal["not_applicable", "revised", "no_change_needed", "insufficient_context"]
EditorialIntent = Literal[
    "report_event",
    "comment_on_source",
    "quote_reaction",
    "reply_reaction",
    "rewrite_existing",
    "explain_topic",
    "long_form_analysis",
]
GenerationStatus = Literal["ok", "no_post_needed", "insufficient_context"]
PipelineTerminalStatus = Literal[
    "ok",
    "no_publishable_signal",
    "no_grounded_contribution",
    "insufficient_evidence",
    "source_already_sufficient",
    "generation_failed",
]
SignalSourceType = Literal[
    "factual_event",
    "news_update",
    "analysis",
    "opinion",
    "thesis",
    "observation",
    "discussion",
    "trend",
    "creative_prompt",
    "unknown",
]
SignalRisk = Literal["low", "medium", "high"]
SignalAction = Literal[
    "generate",
    "generate_with_verification",
    "reject_no_supported_transformation",
    "reject_insufficient_depth",
    "reject_high_inflation_risk",
    "reject_source_stronger_than_draft",
    "skip",
]
ContentMode = Literal["factual_update", "commentary", "opinion_or_creative", "summary"]
SourceRole = Literal[
    "evidence",
    "factual_premise",
    "subject_of_commentary",
    "inspiration",
    "quotation",
]
AuxiliaryEvidenceRole = Literal[
    "confirms_anchor_claim",
    "adds_missing_detail",
    "provides_counterclaim",
    "provides_direct_response",
    "reports_same_event",
    "supplies_required_context",
]
ContributionType = Literal[
    "direct_update",
    "distinction",
    "mechanism",
    "correction",
    "counterexample",
    "caveat",
    "bounded_consequence",
    "source_quality",
    "connection",
    "genuine_question",
    "plain_observation",
    "direct_response",
    "specific_qualification",
    "contextual_extension",
    "dry_humor",
    "dry_reframe",
    "joke",
]


class MediaPlanMetadata(BaseModel):
    """Strict model-generated media metadata accepted by Structured Outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    duration_seconds: float | None = Field(default=None, ge=0)
    alt_text: str = ""
    prompt: str = ""


class MediaPlan(BaseModel):
    """Media inputs that participate in approval hashing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: MediaType = MediaType.NONE
    reason: str = ""
    required_files: tuple[str, ...] = ()
    metadata: MediaPlanMetadata = Field(default_factory=MediaPlanMetadata)


class EvidenceItem(BaseModel):
    """One normalized, untrusted evidence item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    source_type: EvidenceSourceType
    author_or_source: str | None = None
    text: str = Field(min_length=1)
    created_at: datetime | None = None
    url: str | None = None
    verification_status: VerificationStatus = "unknown"
    relevance_score: float | None = Field(default=None, ge=0.0)
    is_primary_source: bool = False


class EvidencePacket(BaseModel):
    """Normalized evidence boundary passed to the drafting model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: GenerationMode
    topic: str = Field(min_length=1)
    items: tuple[EvidenceItem, ...]
    known_facts: tuple[str, ...] = ()
    attributed_claims: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    interpretations_allowed: tuple[str, ...] = ()
    prohibited_inferences: tuple[str, ...] = ()
    language: str = Field(min_length=2, max_length=16)
    requested_format: str = "short_post"

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> EvidencePacket:
        ids = [item.evidence_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence item IDs must be unique")
        return self


class EvidenceSpan(BaseModel):
    """Exact evidence text used to support a semantic relation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)


class SupportedRelation(BaseModel):
    """A normalized subject-predicate-object relation backed by evidence spans."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str | None = None
    qualifiers: tuple[str, ...] = ()
    evidence_spans: tuple[EvidenceSpan, ...] = Field(min_length=1)


class SourceCoverage(BaseModel):
    """Semantic map of what the source already says or strongly implies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    explicit_claims: tuple[str, ...] = ()
    strong_implications: tuple[str, ...] = ()
    named_entities: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    supported_relations: tuple[SupportedRelation, ...] = ()
    recommendations: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    genuinely_unresolved_questions: tuple[str, ...] = ()
    prohibited_inferences: tuple[str, ...] = ()


class AvailableEvidence(BaseModel):
    """Anchor coverage plus explicitly role-bound auxiliary evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_coverage: SourceCoverage
    additional_facts: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    allowed_interpretations: tuple[str, ...] = ()
    prohibited_inferences: tuple[str, ...] = ()


class SignalDecision(BaseModel):
    """Validated publishable-signal decision made before prose generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    source_type: SignalSourceType
    core_proposition: str = Field(min_length=1)
    semantically_complete: bool
    supports_requested_mode: bool
    supported_contribution_count: int = Field(ge=0, le=3)
    risk_source_echo: SignalRisk
    risk_forced_novelty: SignalRisk
    risk_abstract_filler: SignalRisk
    risk_thesis_inflation: SignalRisk
    risk_unsupported_inference: SignalRisk
    risk_source_stronger_than_draft: SignalRisk
    recommended_mode: str | None = None
    action: SignalAction
    reason_code: str = Field(min_length=1)
    user_reason: str = Field(min_length=1)


class AuxiliaryEvidenceBinding(BaseModel):
    """Explicit role for evidence that is not the primary anchor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    role: AuxiliaryEvidenceRole
    relation_to_anchor: str = Field(min_length=1)


class AuxiliaryRelation(BaseModel):
    """Proposition-level justification for using non-anchor evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    auxiliary_evidence_id: str = Field(min_length=1)
    role: AuxiliaryEvidenceRole
    anchor_proposition: str = Field(min_length=1)
    auxiliary_proposition: str = Field(min_length=1)
    relationship: str = Field(min_length=1)
    necessity: str = Field(min_length=1)
    anchor_evidence_spans: tuple[EvidenceSpan, ...] = Field(min_length=1)
    auxiliary_evidence_spans: tuple[EvidenceSpan, ...] = Field(min_length=1)


class ContentModeDecision(BaseModel):
    """Mode-specific policy chosen before contribution planning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ContentMode
    subtype: str | None = None
    source_role: SourceRole
    factual_inspection_required: bool
    attribution_required: bool
    requested_format: str = Field(min_length=1)
    preferred_length_min: int | None = Field(default=None, ge=1)
    preferred_length_max: int | None = Field(default=None, ge=1)
    hard_length_max: int | None = Field(default=None, ge=1)
    failure_conditions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def lengths_are_ordered(self) -> ContentModeDecision:
        if (
            self.preferred_length_min is not None
            and self.preferred_length_max is not None
            and self.preferred_length_min > self.preferred_length_max
        ):
            raise ValueError("preferred length minimum must not exceed maximum")
        if (
            self.preferred_length_max is not None
            and self.hard_length_max is not None
            and self.preferred_length_max > self.hard_length_max
        ):
            raise ValueError("preferred length maximum must not exceed hard maximum")
        return self


class AngleCandidate(BaseModel):
    """One defensible drafting angle selected before prose generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    angle_id: str = Field(min_length=1)
    angle_type: AngleType
    thesis: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    why_interesting: str = Field(min_length=1)
    interpretation: str | None = None
    unsupported_assumptions: tuple[str, ...] = ()
    confidence: Confidence
    topic_specificity: float = Field(default=0.5, ge=0.0, le=1.0)
    contribution_type: ContributionType | None = None
    supporting_relations: tuple[SupportedRelation, ...] = ()
    source_claims_overlapped: tuple[str, ...] = ()
    source_overlap_score: float = Field(default=0.0, ge=0.0, le=1.0)
    support_score: float = Field(default=1.0, ge=0.0, le=1.0)
    usefulness_score: float = Field(default=0.5, ge=0.0, le=1.0)
    scope_fidelity_score: float = Field(default=1.0, ge=0.0, le=1.0)
    requires_new_assumptions: bool = False
    already_covered_by_source: bool = False
    support_status: Literal["supported", "partially_supported", "unsupported"] = "supported"
    usefulness: Literal["clear", "weak", "none"] = "clear"


ContributionCandidate = AngleCandidate


class LiteralDraftPlan(BaseModel):
    """Literal, evidence-bound plan handed to prose generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor: str | None = None
    action: str | None = None
    object: str | None = None
    condition: str | None = None
    supported_consequence: str | None = None
    contribution: str = Field(min_length=1)
    evidence_spans: tuple[EvidenceSpan, ...] = Field(min_length=1)


class SentenceJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sentence_index: int = Field(ge=0)
    job: Literal[
        "fact",
        "attribution",
        "reaction",
        "mechanism",
        "distinction",
        "condition",
        "caveat",
        "question",
        "action",
        "restate",
        "framing",
        "importance_announcement",
        "conclusion_only",
    ]
    proposition: str
    evidence_ids: tuple[str, ...] = ()


class ParagraphJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paragraph_index: int = Field(ge=0)
    job: str
    new_information: str | None = None
    overlaps_paragraphs: tuple[int, ...] = ()


class SemanticQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["accept", "revise", "no_post_needed"]
    literal_core: str
    source_coverage_overlap: float = Field(ge=0.0, le=1.0)
    grounded_contribution: str | None = None
    sentence_jobs: tuple[SentenceJob, ...] = ()
    paragraph_jobs: tuple[ParagraphJob, ...] = ()
    issue_codes: tuple[str, ...] = ()
    unsupported_relations: tuple[str, ...] = ()
    explanation: str = ""


class AddedValueAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    direct_source_claims: tuple[str, ...] = ()
    draft_propositions: tuple[str, ...] = ()
    supported_new_relations: tuple[str, ...] = ()
    unsupported_new_relations: tuple[str, ...] = ()
    source_echo_propositions: tuple[str, ...] = ()
    relation_novelty_score: float = Field(ge=0.0, le=1.0)
    grounding_score: float = Field(ge=0.0, le=1.0)
    decision: Literal["pass", "no_post_needed", "reject"]


class QualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    severity: Literal["error", "warning"]
    message: str = Field(min_length=1)
    sentence_index: int | None = Field(default=None, ge=0)
    suggested_action: (
        Literal[
            "delete",
            "flatten",
            "attribute",
            "ground",
            "shorten",
            "split",
            "return_insufficient_context",
            "regenerate_angle",
        ]
        | None
    ) = None


class QualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    specificity_score: float = Field(ge=0.0, le=1.0)
    evidence_score: float = Field(ge=0.0, le=1.0)
    naturalness_score: float = Field(ge=0.0, le=1.0)
    distinctiveness_score: float = Field(ge=0.0, le=1.0)
    thesis_score: float = Field(default=1.0, ge=0.0, le=1.0)
    medium_fit_score: float = Field(default=1.0, ge=0.0, le=1.0)
    voice_score: float = Field(default=1.0, ge=0.0, le=1.0)
    plainness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    rhetorical_density_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reply_fit_score: float = Field(default=1.0, ge=0.0, le=1.0)
    recent_similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sentence_job_uniqueness: float = Field(default=1.0, ge=0.0, le=1.0)
    semantic_repetition_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    supported_relation_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    abstract_restatement_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    rhetorical_scaffolding_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    paragraph_advancement_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    source_proposition_overlap: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_decision: Literal["accept", "revise", "no_post_needed"] | None = None
    issues: tuple[QualityIssue, ...] = ()

    @property
    def acceptable(self) -> bool:
        return self.passed

    @property
    def score(self) -> float:
        return (
            self.evidence_score * 0.25
            + self.specificity_score * 0.20
            + self.naturalness_score * 0.15
            + self.plainness_score * 0.15
            + self.reply_fit_score * 0.10
            + self.distinctiveness_score * 0.15
        )

    @property
    def blocking_flags(self) -> tuple[str, ...]:
        codes = [issue.code for issue in self.issues if issue.severity == "error"]
        if any(
            code in codes
            for code in (
                "UNSPECIFIC_UNCERTAINTY",
                "VAGUE_ATTRIBUTION",
                "GENERIC_VERIFICATION",
                "UNSUPPORTED_CERTAINTY",
            )
        ):
            codes.append("stock_opening")
        if "PLACEHOLDER_COMMENTARY" in codes:
            codes.append("noise_cliche")
        if any(
            code in codes
            for code in ("GENERIC_CONCLUSION", "QUOTABLE_ENDING", "OVEREXPLAINED_ENDING")
        ):
            codes.append("generic_takeaway")
        return tuple(dict.fromkeys(codes))

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues if issue.severity == "warning")


class GenerationVariant(BaseModel):
    """One structured LLM proposal tied to a selected angle and evidence IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    text: str = Field(min_length=1)
    parts: tuple[str, ...] = ()
    hook: str = ""
    cta: str = ""
    character_count: int = Field(default=0, ge=0)
    tone: tuple[str, ...] = ()
    claims: tuple[str, ...] = ()
    source_post_ids: tuple[str, ...] = ()
    similarity_risk: float = Field(ge=0.0, le=1.0)
    fact_check_required: bool = False
    angle_id: str = Field(min_length=1)
    angle_type: AngleType
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    confidence: Confidence
    factual_claims: tuple[str, ...]
    attributed_claims: tuple[str, ...]
    uncertainty_markers: tuple[str, ...]
    revision_status: RevisionStatus = "not_applicable"
    contribution_id: str | None = None
    contribution_type: str | None = None
    source_overlap_status: Literal["none", "partial", "high"] | None = None
    editorial_status: Literal["pass", "warning", "block"] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_derived_character_count(cls, data: object) -> object:
        """Never trust a model for a value derived deterministically from text."""

        if not isinstance(data, Mapping):
            return data
        normalized = dict(data)
        text = normalized.get("text")
        if isinstance(text, str):
            normalized["character_count"] = len(text)
        return normalized

    @model_validator(mode="after")
    def validate_thread_parts(self) -> GenerationVariant:
        if self.parts and any(not part.strip() for part in self.parts):
            raise ValueError("thread parts must be non-empty")
        return self


class HumanizerRevisionTarget(BaseModel):
    """One generated variant plus the named issues a bounded revision may fix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: GenerationVariant
    issue_codes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def issue_codes_are_unique(self) -> HumanizerRevisionTarget:
        if any(not code.strip() for code in self.issue_codes):
            raise ValueError("revision issue codes must be non-empty")
        if len(self.issue_codes) != len(set(self.issue_codes)):
            raise ValueError("revision issue codes must be unique")
        return self


class GenerationResult(BaseModel):
    """Structured generation response specified by the content workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GenerationStatus = "ok"
    reason_code: str | None = None
    idea_summary: str
    recommended_format: ContentFormat | None = None
    variants: tuple[GenerationVariant, ...] = Field(default=(), max_length=3)
    recommended_variant: int | None = Field(default=None, ge=0)
    recommendation_reason: str = ""
    media_plan: MediaPlan = Field(default_factory=MediaPlan)

    @model_validator(mode="before")
    @classmethod
    def normalize_non_thread_parts(cls, data: object) -> object:
        """Discard redundant thread fragments from an otherwise valid non-thread result.

        Structured-output JSON Schema cannot express the parent-dependent rule that ``parts``
        belongs only to thread variants. Providers can therefore return a complete non-thread
        ``text`` value and also populate ``parts`` even though those fragments are never used by
        the application. Treat that exact shape like the derived character count: normalize the
        redundant representation locally instead of rejecting the full generation result.

        The normalization is deliberately narrow. It runs only when the declared result format
        is non-thread and the variant already contains non-blank publishable text. Blank text,
        malformed variants, and incomplete thread results still fail strict validation.
        """

        if not isinstance(data, Mapping):
            return data
        normalized = dict(data)
        raw_format = normalized.get("recommended_format")
        format_value = raw_format.value if isinstance(raw_format, ContentFormat) else raw_format
        if format_value == ContentFormat.THREAD.value:
            return normalized

        variants = normalized.get("variants")
        if not isinstance(variants, (list, tuple)):
            return normalized

        normalized_variants: list[object] = []
        changed = False
        for variant in variants:
            if not isinstance(variant, Mapping):
                normalized_variants.append(variant)
                continue
            normalized_variant = dict(variant)
            text = normalized_variant.get("text")
            parts = normalized_variant.get("parts")
            if isinstance(text, str) and text.strip() and parts:
                normalized_variant["parts"] = []
                changed = True
            normalized_variants.append(normalized_variant)

        if changed:
            normalized["variants"] = normalized_variants
        return normalized

    @model_validator(mode="after")
    def validate_contract(self) -> GenerationResult:
        if self.status == "ok":
            if not self.variants:
                raise ValueError("status=ok requires one to three variants")
            if self.recommended_format is None:
                raise ValueError("status=ok requires recommended_format")
            if self.recommended_variant is None or self.recommended_variant >= len(self.variants):
                raise ValueError("recommended_variant must reference an existing variant")
        else:
            if self.variants or self.recommended_variant is not None:
                raise ValueError("non-ok generation results cannot contain public variants")
            return self
        angle_ids = [variant.angle_id for variant in self.variants]
        if len(angle_ids) != len(set(angle_ids)):
            raise ValueError("generation variants must use distinct angle IDs")
        if self.recommended_format is ContentFormat.THREAD:
            if any(len(variant.parts) < 2 for variant in self.variants):
                raise ValueError("every thread variant must contain at least two parts")
        elif any(variant.parts for variant in self.variants):
            raise ValueError("non-thread variants cannot contain thread parts")
        return self


class UnavailableVariant(BaseModel):
    """Explicit UI-safe representation of a grounded variant that was not generated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["not_generated"] = "not_generated"
    reason: str = Field(min_length=1)


class EditorialIssue(BaseModel):
    """Categorical semantic issue with the exact text or relation that caused it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    severity: Literal["warning", "block"]
    evidence_spans: tuple[str, ...] = Field(min_length=1)
    explanation: str = Field(min_length=1)
    subtypes: tuple[str, ...] = ()


class EditorialReport(BaseModel):
    """Semantic editorial inspection without synthetic numeric quality scores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["pass", "warning", "block"]
    source_overlap: Literal["none", "partial", "high"]
    contribution_support: Literal["supported", "partial", "unsupported"]
    information_gain: Literal["clear", "weak", "none"]
    internal_redundancy: Literal["none", "present"]
    source_stronger_than_draft: bool
    contribution_delivered: bool
    issues: tuple[EditorialIssue, ...] = ()

    @model_validator(mode="after")
    def status_matches_issues(self) -> EditorialReport:
        if any(issue.severity == "block" for issue in self.issues) and self.status != "block":
            raise ValueError("blocking editorial issues require status=block")
        return self


ClaimKind = Literal[
    "factual",
    "attributed_claim",
    "interpretation",
    "opinion",
    "uncertainty",
    "material_claim_unknown",
]
ClaimSupport = Literal[
    "supported",
    "partially_supported",
    "attributed_only",
    "unsupported",
    "source_conflict",
    "not_applicable",
    "extraction_uncertain",
]


ClaimPolarity = Literal["positive", "negative", "unknown"]
ClaimModality = Literal[
    "asserted",
    "possible",
    "probable",
    "reported",
    "denied",
    "unknown",
]


class QuantityMention(BaseModel):
    """A material quantity with its proposition-level semantic role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    normalized_value: str = Field(min_length=1)
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
    unit: str | None = None


class EventQualifier(BaseModel):
    """Material event detail that must be independently grounded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qualifier_type: Literal[
        "manner",
        "purpose",
        "cause",
        "instrument",
        "seller_or_counterparty",
        "location",
        "legal_status",
        "approval",
        "secrecy",
        "voluntariness",
        "pressure",
        "margin",
        "condition",
        "other_material",
    ]
    text: str = Field(min_length=1)
    supported_by_evidence_ids: tuple[str, ...] = ()


class CanonicalClause(BaseModel):
    """Clause-level proposition scope for matrix and embedded events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = Field(min_length=1)
    exact_text: str = Field(min_length=1)
    clause_role: Literal[
        "main_assertion",
        "embedded_proposition",
        "condition",
        "purpose",
        "cause",
        "attribution_content",
        "question_content",
        "denied_content",
        "unknown",
    ]
    assertion_status: Literal[
        "asserted",
        "questioned",
        "possible",
        "reported",
        "denied",
        "not_established",
        "planned",
        "promised",
        "desired",
        "authorized",
        "prevented",
        "conditional",
        "unknown",
    ]
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    parent_clause_id: str | None = None
    relation: (
        Literal[
            "matrix_embedding",
            "relative_clause",
            "coordinated_clause",
            "discourse_linked_clause",
            "approval_modifier",
            "condition",
            "attribution",
            "unknown",
        ]
        | None
    ) = None


class EventIdentityMatch(BaseModel):
    """Symmetric identity result for the core event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_identity: Literal["same", "alias", "different", "uncertain"]
    predicate_identity: Literal["same", "opposite", "different", "uncertain"]
    object_identity: Literal[
        "same",
        "alias",
        "qualified_same_core",
        "different",
        "uncertain",
    ]
    core_event_same: bool
    material_extra_details: tuple[str, ...] = ()


class CanonicalClaim(BaseModel):
    """Application-owned proposition extracted from rendered text or evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    exact_text: str = Field(min_length=1)
    kind: ClaimKind
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    polarity: ClaimPolarity = "unknown"
    modality: ClaimModality = "unknown"
    quantities: tuple[str, ...] = ()
    quantity_mentions: tuple[QuantityMention, ...] = ()
    dates: tuple[str, ...] = ()
    named_entities: tuple[str, ...] = ()
    attribution_source: str | None = None
    # Separated semantic dimensions. ``modality`` remains for backward-compatible artifacts,
    # while approval policy is derived from these application-owned fields.
    speech_act: Literal[
        "assertion",
        "question",
        "report",
        "claim",
        "denial",
        "quotation",
        "unknown",
    ] = "assertion"
    epistemic_modality: Literal[
        "asserted",
        "probable",
        "possible",
        "alleged",
        "reported",
        "unknown",
    ] = "unknown"
    event_status: Literal[
        "completed",
        "ongoing",
        "future",
        "planned",
        "attempted",
        "near_event",
        "conditional",
        "counterfactual",
        "unknown",
    ] = "unknown"
    attribution_act: (
        Literal[
            "said",
            "reported",
            "claimed",
            "announced",
            "denied",
            "acknowledged",
            "unknown",
        ]
        | None
    ) = None
    extraction_provenance: Literal["deterministic", "provider", "hint", "legacy"] = "legacy"
    structure_validated: bool = True
    extraction_conflict: bool = False
    clauses: tuple[CanonicalClause, ...] = ()
    material_qualifiers: tuple[EventQualifier, ...] = ()
    clause_relation: Literal[
        "main",
        "relative_clause",
        "coordinated_clause",
        "discourse_linked_clause",
        "approval_modifier",
        "unknown",
    ] = "main"
    parent_claim_id: str | None = None
    unconsumed_material_text: tuple[str, ...] = ()
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def offsets_are_valid(self) -> CanonicalClaim:
        if (self.start is None) != (self.end is None):
            raise ValueError("claim offsets must be both present or both absent")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("claim start must not exceed end")
        return self


class EvidenceCandidateMatch(BaseModel):
    """Application-owned alignment between one draft claim and one evidence proposition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_claim_id: str = Field(min_length=1)
    event_alignment: Literal["same", "partial", "unrelated"]
    actor_alignment: Literal["same", "partial", "different", "unknown"]
    predicate_alignment: Literal["same", "opposite", "different", "unknown"]
    object_alignment: Literal["same", "partial", "different", "unknown"]
    quantity_alignment: Literal["same", "different", "missing", "not_applicable"]
    date_alignment: Literal["same", "different", "missing", "not_applicable"]
    modality_alignment: Literal["compatible", "escalated", "different", "unknown"]
    attribution_alignment: Literal["same", "lost", "different", "not_applicable"]
    speech_act_alignment: Literal["compatible", "question_only", "different", "unknown"] = "unknown"
    event_status_alignment: Literal["compatible", "escalated", "different", "unknown"] = "unknown"
    clause_scope_alignment: Literal["compatible", "not_asserted", "different", "unknown"] = (
        "unknown"
    )
    event_identity: EventIdentityMatch | None = None


class ClaimExtractionContext(BaseModel):
    """Bounded context supplied to a claim extractor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: str = Field(min_length=1)
    strict: bool = True
    source_role: str | None = None


class ClaimExtractionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: tuple[CanonicalClaim, ...] = ()


class ClaimChange(BaseModel):
    """Claim-level difference used for manual-edit approval invalidation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    change_type: Literal["added", "removed", "changed", "unchanged"]
    old_claim_id: str | None = None
    new_claim_id: str | None = None
    fields_changed: tuple[str, ...] = ()
    explanation: str = Field(min_length=1)


class SourceSemanticProposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposition_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    status: Literal["explicit", "strongly_implied", "not_established"]
    relation: CanonicalClaim
    evidence_spans: tuple[EvidenceSpan, ...] = Field(min_length=1)
    validation_issues: tuple[str, ...] = ()


class SourceSemanticMap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    propositions: tuple[SourceSemanticProposition, ...] = ()
    conclusions: tuple[str, ...] = ()
    factual_premises: tuple[CanonicalClaim, ...] = ()


class ContributionInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contribution_id: str = Field(min_length=1)
    canonical_proposition: CanonicalClaim
    already_covered: bool
    requires_new_assumption: bool
    unsupported_adjacent_complications: tuple[str, ...] = ()
    grounded: bool
    reaction_function_present: bool = False
    source_specificity_present: bool = False
    plain_paraphrase_only: bool = False
    reaction_inspection: ReactionFunctionInspection | None = None


class LexicalAtom(BaseModel):
    """Normalized lexical evidence used only by compositional semantic extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    span: str = Field(min_length=1)
    lemma: str = Field(min_length=1)
    category: Literal[
        "human_head",
        "technical_head",
        "organization_head",
        "business_head",
        "legal_political_head",
        "physical_event_head",
        "abstract_head",
        "harm_predicate",
        "harm_modifier",
        "medical_state",
        "medical_destination",
        "medical_escalation",
        "physiological_state",
        "medical_support",
        "functional_dependency",
        "ordinary_result_state",
        "diminished_state_modifier",
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
        "other",
    ]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    clause_id: str | None = None


class SemanticClause(BaseModel):
    """Stable request-local clause scope used by every semantic extraction role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = Field(min_length=1)
    text_span: str = Field(min_length=1)
    coordination_group_id: str | None = None
    shared_subject_clause_id: str | None = None


class PredicateScope(BaseModel):
    """Smallest finite or explicit non-finite predicate scope used for role binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope_id: str = Field(min_length=1)
    parent_scope_id: str | None = None
    relation_to_parent: Literal[
        "matrix",
        "temporal",
        "causal",
        "concessive",
        "conditional",
        "contrastive",
        "relative",
        "participial",
        "other",
    ] = "matrix"
    text_span: str = Field(min_length=1)


class ClauseFrame(BaseModel):
    """Clause-level grammatical roles and assertion state before relation adjudication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
    scope_id: str = "unscoped"
    parent_scope_id: str | None = None
    relation_to_parent: Literal[
        "matrix",
        "temporal",
        "causal",
        "concessive",
        "conditional",
        "contrastive",
        "relative",
        "participial",
        "other",
    ] = "matrix"
    coordination_group_id: str | None = None
    shared_subject_clause_id: str | None = None
    clause_span: str = Field(min_length=1)
    subject_span: str | None = None
    predicate_span: str | None = None
    object_span: str | None = None
    complement_span: str | None = None
    modifiers: tuple[str, ...] = ()
    negated: bool = False
    modal: str | None = None
    attributed: bool = False
    conditional: bool = False
    assertion_state: Literal[
        "asserted",
        "negated",
        "possible",
        "planned",
        "feared",
        "reported",
        "conditional",
        "historical_comparison",
        "unknown",
    ] = "asserted"
    atoms: tuple[LexicalAtom, ...] = ()


class RelationCandidate(BaseModel):
    """Compositional relation assembled from clause roles and lexical categories."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
    relation_kind: Literal[
        "event",
        "evaluation",
        "identity",
        "reduction",
        "repackaging",
        "reveal",
        "inverse_praise",
        "setup_twist",
        "unknown",
    ]
    actor_span: str | None = None
    affected_span: str | None = None
    target_span: str | None = None
    predicate_span: str | None = None
    complement_span: str | None = None
    decoration_span: str | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    consumed_spans: tuple[str, ...] = ()


class ResultComplementFrame(BaseModel):
    """Clause-local result complement before outcome typing or safety adjudication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
    affected_entity_span: str | None = None
    affected_entity_type: Literal[
        "person",
        "group_of_people",
        "technical_system",
        "software_process",
        "product",
        "company_or_organization",
        "crypto_or_protocol",
        "business_or_financial_object",
        "legal_or_political_object",
        "physical_event",
        "abstract_concept",
        "unknown",
    ] = "unknown"
    result_link_span: str | None = None
    complement_span: str | None = None
    complement_form: Literal[
        "adjectival",
        "infinitival",
        "participial",
        "gerund",
        "with_state",
        "dependency",
        "coordinated",
        "unknown",
    ] = "unknown"
    assertion_scope: Literal[
        "asserted",
        "negated",
        "possible",
        "planned",
        "feared",
        "reported",
        "conditional",
        "historical_comparison",
        "unknown",
    ] = "asserted"
    causal_context_span: str | None = None
    consumed_spans: tuple[str, ...] = ()


class ResultStateCandidate(BaseModel):
    """Typed result-state roles before a final medical or harmless outcome is known."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
    affected_entity_span: str | None = None
    affected_entity_type: Literal[
        "person",
        "group_of_people",
        "technical_system",
        "software_process",
        "product",
        "company_or_organization",
        "crypto_or_protocol",
        "business_or_financial_object",
        "legal_or_political_object",
        "physical_event",
        "abstract_concept",
        "unknown",
    ] = "unknown"
    linking_predicate_span: str | None = None
    state_complement_span: str | None = None
    complement_form: Literal[
        "adjectival",
        "infinitival",
        "participial",
        "gerund",
        "with_state",
        "dependency",
        "coordinated",
        "unknown",
    ] = "unknown"
    assertion_scope: Literal[
        "asserted",
        "negated",
        "possible",
        "planned",
        "feared",
        "reported",
        "conditional",
        "historical_comparison",
        "unknown",
    ] = "asserted"
    causal_event_span: str | None = None
    state_category: Literal[
        "typed_harm",
        "physiological_or_medical",
        "medical_dependency",
        "ordinary_location_or_activity",
        "ordinary_business_or_schedule",
        "ordinary_cognitive_state",
        "non_human_state",
        "unknown",
    ] = "unknown"
    outcome_type: Literal[
        "death",
        "serious_injury",
        "critical_condition",
        "hospitalization",
        "intensive_care",
        "permanent_injury",
        "non_survival",
        "unknown",
    ] = "unknown"
    negated: bool = False
    modal: bool = False
    conditional: bool = False
    attributed: bool = False
    confidence: Literal["high", "medium", "low"] = "low"
    consumed_spans: tuple[str, ...] = ()


class ResultStateMateriality(BaseModel):
    """Policy-neutral materiality projection for one bound result-state relation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
    affected_entity_type: Literal[
        "person",
        "group_of_people",
        "technical_system",
        "software_process",
        "product",
        "company_or_organization",
        "crypto_or_protocol",
        "business_or_financial_object",
        "legal_or_political_object",
        "physical_event",
        "abstract_concept",
        "unknown",
    ] = "unknown"
    relation_asserted: bool = False
    impairment_relation: Literal[
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
    safety_material: bool = False
    evidence_spans: tuple[str, ...] = ()


class HumanFunctionImpact(BaseModel):
    """Role-bound effect on a human function, independent of surface wording."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
    function_domain: Literal[
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
    impact: Literal[
        "lost",
        "reduced",
        "impaired",
        "assistance_required",
        "support_required",
        "normal",
        "unknown",
    ] = "unknown"
    asserted: bool = False
    safety_material: bool = False
    evidence_spans: tuple[str, ...] = ()


class ReplyProposition(BaseModel):
    """Policy-neutral proposition completeness and communicative-role projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
    subject_span: str | None = None
    predicate_span: str | None = None
    object_or_complement_span: str | None = None
    assertion_state: str = "unknown"
    proposition_complete: bool = False
    communicative_function: Literal[
        "plain_observation",
        "direct_response",
        "qualification",
        "question",
        "other",
        "unknown",
    ] = "unknown"
    confidence: Literal["high", "medium", "low"] = "low"
    evidence_spans: tuple[str, ...] = ()


class LiteralPropositionFrame(BaseModel):
    """Policy-neutral typed literal relation projected from complete grammatical roles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
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
    ]
    subject_span: str = Field(min_length=1)
    predicate_span: str = Field(min_length=1)
    object_or_complement_span: str = Field(min_length=1)
    assertion_state: str = "unknown"
    proposition_complete: bool = True
    evidence_spans: tuple[str, ...] = ()


class ReplySemanticCompleteness(BaseModel):
    """Clause-local reconciliation of proposition, typed relations, and material surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
    proposition_complete: bool = False
    typed_evaluation_present: bool = False
    typed_shell_present: bool = False
    validated_literal_function: bool = False
    material_surface_present: bool = False
    unresolved_category: Literal["incomplete_proposition"] | None = None
    evidence_spans: tuple[str, ...] = ()


class ValidatedSpan(BaseModel):
    """Exact provider span plus local Unicode-token and clause-boundary checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    clause_id: str = "unscoped"
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    exact: bool = False
    token_start_boundary: bool = False
    token_end_boundary: bool = False


class SemanticUnresolvedItem(BaseModel):
    """One request-local unresolved semantic item with a stable identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    side: Literal["source", "reply"]
    clause_id: str = Field(min_length=1)
    clause_span: str = Field(min_length=1)
    unresolved_span: str = Field(min_length=1)
    category: Literal[
        "possible_harm",
        "possible_evaluation",
        "possible_reduction",
        "unknown_predicate",
        "unknown_referent",
        "incomplete_proposition",
        "other",
    ]


class SemanticCoverage(BaseModel):
    """Coverage accounting for every material clause."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    coverage_id: str = "coverage-unscoped"
    clause_id: str = "unscoped"
    clause_span: str = Field(min_length=1)
    consumed_spans: tuple[str, ...] = ()
    unresolved_items: tuple[SemanticUnresolvedItem, ...] = ()
    unresolved_spans: tuple[str, ...] = ()
    unresolved_categories: tuple[
        Literal[
            "possible_harm",
            "possible_evaluation",
            "possible_reduction",
            "unknown_predicate",
            "unknown_referent",
            "incomplete_proposition",
            "other",
        ],
        ...,
    ] = ()
    complete: bool = True


class SemanticEntity(BaseModel):
    """Typed entity reference emitted only by semantic extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str | None = None
    text_span: str = Field(min_length=1)
    canonical_name: str | None = None
    entity_type: Literal[
        "person",
        "group_of_people",
        "technical_system",
        "software_process",
        "product",
        "company_or_organization",
        "crypto_or_protocol",
        "business_or_financial_object",
        "legal_or_political_object",
        "physical_event",
        "abstract_concept",
        "unknown",
    ] = "unknown"
    confidence: Literal["high", "medium", "low"] = "low"


class SemanticEvent(BaseModel):
    """Typed predicate/referent relation; proximity alone is never an outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
    event_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    actor: SemanticEntity | None = None
    affected_entity: SemanticEntity | None = None
    object: SemanticEntity | None = None
    event_type: Literal[
        "success",
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
        "displacement",
        "evacuation",
        "damage",
        "abuse",
        "self_harm",
        "personal_tragedy",
        "other",
        "unknown",
    ] = "unknown"
    polarity: Literal["positive", "negative", "neutral", "unknown"] = "unknown"
    explicit_outcome: bool = False
    assertion_state: Literal[
        "asserted",
        "negated",
        "possible",
        "planned",
        "feared",
        "reported",
        "conditional",
        "historical_comparison",
        "unknown",
    ] = "asserted"
    evidence_spans: tuple[str, ...] = ()


class EvaluationRelation(BaseModel):
    """Typed evaluative relation extracted from final reply text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
    target_entity: SemanticEntity | None = None
    target_event_id: str | None = None
    evaluation_type: Literal[
        "praise",
        "admiration",
        "rating",
        "reward",
        "celebration",
        "success_attribution",
        "blame",
        "criticism",
        "dismissal",
        "other",
        "unknown",
    ] = "unknown"
    relation_subtype: Literal[
        "praise",
        "celebration",
        "rating",
        "reward",
        "admiration",
        "success_attribution",
        "positive_timing",
        "beneficiary",
        "none",
        "uncertain",
    ] = "none"
    polarity: Literal["positive", "negative", "neutral", "unknown"] = "unknown"
    literal_reading_plausible: bool = False
    confidence: Literal["high", "medium", "low"] = "low"
    evidence_spans: tuple[str, ...] = ()


class SemanticReplyShell(BaseModel):
    """Canonical semantic operator and role slots for reply diversity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clause_id: str = "unscoped"
    operator: Literal[
        "identity",
        "reduction",
        "decorated_identity",
        "repackaging",
        "inverse_praise",
        "reveal",
        "setup_twist",
        "none",
        "unknown",
    ] = "none"
    subject_span: str | None = None
    complement_span: str | None = None
    decoration_span: str | None = None
    repackaging_marker: str | None = None
    opener_family: str | None = None
    generic_reusability: bool = False
    evidence_spans: tuple[str, ...] = ()


class SemanticInspection(BaseModel):
    """Canonical semantic intermediate representation used by application adjudication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_entities: tuple[SemanticEntity, ...] = ()
    predicate_scopes: tuple[PredicateScope, ...] = ()
    source_events: tuple[SemanticEvent, ...] = ()
    reply_entities: tuple[SemanticEntity, ...] = ()
    reply_events: tuple[SemanticEvent, ...] = ()
    evaluations: tuple[EvaluationRelation, ...] = ()
    reply_shells: tuple[SemanticReplyShell, ...] = ()
    result_state_materiality: tuple[ResultStateMateriality, ...] = ()
    human_function_impacts: tuple[HumanFunctionImpact, ...] = ()
    reply_propositions: tuple[ReplyProposition, ...] = ()
    literal_proposition_frames: tuple[LiteralPropositionFrame, ...] = ()
    reply_semantic_completeness: tuple[ReplySemanticCompleteness, ...] = ()
    source_coverage: tuple[SemanticCoverage, ...] = ()
    reply_coverage: tuple[SemanticCoverage, ...] = ()
    reply_communicative_function: Literal[
        "plain_observation",
        "direct_response",
        "qualification",
        "question",
        "other",
        "unknown",
    ] = "unknown"
    reply_communicative_function_confidence: Literal["high", "medium", "low"] = "low"
    reply_communicative_evidence_spans: tuple[str, ...] = ()
    reply_self_reference: bool = False
    extraction_conflicts: tuple[str, ...] = ()
    unresolved_material: tuple[str, ...] = ()


class SemanticEscalationDecision(BaseModel):
    """Application-owned decision to request bounded provider semantic extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    required: bool = False
    material_categories: tuple[
        Literal[
            "possible_harm",
            "possible_evaluation",
            "possible_reduction",
            "unknown_referent",
            "unknown_predicate",
            "incomplete_proposition",
        ],
        ...,
    ] = ()
    clause_spans: tuple[str, ...] = ()
    unresolved_item_ids: tuple[str, ...] = ()
    reason: str = ""


class SemanticCoverageResolution(BaseModel):
    """Provider proposal for one deterministic unresolved coverage item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    side: Literal["source", "reply"]
    clause_id: str = Field(min_length=1)
    clause_span: str = Field(min_length=1)
    unresolved_span: str = Field(min_length=1)
    category: Literal[
        "possible_harm",
        "possible_evaluation",
        "possible_reduction",
        "unknown_referent",
        "unknown_predicate",
        "incomplete_proposition",
    ]
    disposition: Literal["typed_relation", "harmless", "unresolved"]
    reason: str = ""
    evidence_spans: tuple[str, ...] = ()


class SemanticProviderRequest(BaseModel):
    """Bounded semantic packet sent to a structured extraction provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_clauses: tuple[str, ...] = ()
    reply_clauses: tuple[str, ...] = ()
    request_id: str = Field(min_length=1)
    deterministic: SemanticInspection
    unresolved_items: tuple[SemanticUnresolvedItem, ...] = ()
    material_categories: tuple[
        Literal[
            "possible_harm",
            "possible_evaluation",
            "possible_reduction",
            "unknown_referent",
            "unknown_predicate",
            "incomplete_proposition",
        ],
        ...,
    ] = ()
    clause_spans: tuple[str, ...] = ()


class SemanticProviderProposal(BaseModel):
    """Typed provider extraction proposal without policy or publication authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inspection: SemanticInspection = Field(default_factory=SemanticInspection)
    coverage_resolutions: tuple[SemanticCoverageResolution, ...] = ()
    remaining_unresolved_items: tuple[SemanticCoverageResolution, ...] = ()


class SemanticInspectionResult(BaseModel):
    """Deterministic, provider, and canonical semantic extraction state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deterministic: SemanticInspection
    provider_proposal: SemanticProviderProposal | None = None
    canonical: SemanticInspection
    escalation: SemanticEscalationDecision
    provider_used: bool = False
    provider_validation_errors: tuple[str, ...] = ()
    unresolved_after_reconciliation: tuple[str, ...] = ()


class EvaluativeIronyInspection(BaseModel):
    """Bounded context-dependent inspection for evaluative irony in final reply text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_polarity: Literal["positive", "rewarding", "neutral", "unknown"] = "unknown"
    evaluation_relation: Literal[
        "praise",
        "celebration",
        "rating",
        "reward",
        "admiration",
        "success_attribution",
        "positive_timing",
        "beneficiary",
        "none",
        "uncertain",
    ] = "none"
    source_event_polarity: Literal["harmful", "negative", "positive", "mixed", "unknown"] = (
        "unknown"
    )
    evaluation_target: str | None = None
    target_source_alignment: bool = False
    positive_evaluation: bool = False
    negative_or_harmful_source_context: bool = False
    evaluation_target_grounded: bool = False
    literal_positive_reading_plausible: bool = False
    implicit_sarcasm: bool = False
    confidence: Literal["high", "medium", "low"] = "low"
    evidence_spans: tuple[str, ...] = ()


class SemanticHumorProposal(BaseModel):
    """Provider or deterministic proposal about final-text humor intent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: Literal[
        "literal_observation",
        "question",
        "qualification",
        "sarcasm",
        "irony",
        "callback",
        "wordplay",
        "incongruity",
        "dry_reframe",
        "none",
        "uncertain",
    ] = "none"
    humorous_treatment: bool = False
    confidence: Literal["high", "medium", "low"] = "low"
    evidence_spans: tuple[str, ...] = ()


class HumorIntentInspection(BaseModel):
    """Text-first determination of whether a reply uses humorous treatment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata_says_humor: bool = False
    text_appears_humorous: bool = False
    detected_reaction_type: Literal[
        "literalization",
        "incongruity",
        "callback",
        "wordplay",
        "dry_reframe",
        "template_humor",
        "sarcasm",
        "irony",
        "none",
        "uncertain",
    ] = "none"
    humor_safety_required: bool = False
    extraction_conflict: bool = False
    evaluative_irony: EvaluativeIronyInspection | None = None
    issue_codes: tuple[str, ...] = ()
    proposal: SemanticHumorProposal | None = None


class HumanHarmInspection(BaseModel):
    """Literal human-harm scope separated from technical or metaphorical death language."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    harm_referent: Literal["human", "non_human", "unknown"] = "unknown"
    harm_state: Literal[
        "death",
        "serious_injury",
        "critical_condition",
        "hospitalization",
        "intensive_care",
        "permanent_injury",
        "non_survival",
        "non_human_failure",
        "unknown",
    ] = "unknown"
    event_type: Literal[
        "physical_crash",
        "blast",
        "fire",
        "medical",
        "incident",
        "outage",
        "technical",
        "business_or_political",
        "unknown",
    ] = "unknown"
    literal_or_metaphorical_usage: Literal[
        "literal",
        "metaphorical_or_technical",
        "unknown",
    ] = "unknown"
    literal_harm: bool = False
    metaphorical_or_technical_usage: bool = False
    confidence: Literal["high", "medium", "low"] = "low"
    evidence_spans: tuple[str, ...] = ()
    reason: str | None = None


class HumorSafetyInspection(BaseModel):
    """Application-owned suitability decision for humorous treatment of source material."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sensitive_context: bool = False
    sensitive_categories: tuple[str, ...] = ()
    targets_victim_or_harm: bool = False
    targets_public_claim_or_system: bool = False
    self_deprecating: bool = False
    suitable_for_humor: bool = True
    issue_codes: tuple[str, ...] = ()
    context_confidence: Literal["high", "medium", "low"] = "high"
    ambiguous_sensitive_term: bool = False
    disambiguation_reason: str | None = None
    harm_referent: Literal["human", "non_human", "unknown"] = "unknown"
    literal_harm: bool = False
    metaphorical_or_technical_usage: bool = False


class ReplyShellInspection(BaseModel):
    """Bounded relational skeleton for reusable reply-shell comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operator: Literal[
        "identity",
        "reduction",
        "decorated_identity",
        "repackaging",
        "inverse_praise",
        "reveal",
        "setup_twist",
        "none",
        "uncertain",
        "unknown",
    ] = "none"
    subject_span: str | None = None
    complement_span: str | None = None
    decoration_span: str | None = None
    repackaging_marker: str | None = None
    opener_family: str | None = None
    source_specific: bool = False
    generic_reusability: bool = False
    evidence_spans: tuple[str, ...] = ()


class ReplyNaturalnessInspection(BaseModel):
    """Categorical reply-shape inspection used by application-owned selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_type: Literal[
        "plain_observation",
        "direct_response",
        "qualification",
        "question",
        "dry_humor",
        "contextual_extension",
    ]
    constructedness: Literal["low", "medium", "high"]
    punchline_compulsion: bool = False
    quote_card_risk: bool = False
    recent_syntax_reuse: bool = False
    false_brevity: bool = False
    source_specific: bool = True
    structural_families: tuple[str, ...] = ()
    recent_structural_match: bool = False
    generic_reusability: bool = False
    reply_shell: ReplyShellInspection | None = None
    humor_intent: HumorIntentInspection | None = None
    humor_safety: HumorSafetyInspection | None = None


class ReactionFunctionInspection(BaseModel):
    """Application-owned semantic validation of a reaction contribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reaction_type: Literal[
        "literalization",
        "incongruity",
        "callback",
        "wordplay",
        "dry_reframe",
        "none",
        "uncertain",
    ]
    source_specific: bool
    source_echo: bool
    unsupported_detail: bool
    generic_template_only: bool
    grounded: bool


class ContributionInspectionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    inspections: tuple[ContributionInspection, ...] = ()


class DraftSemanticInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    draft_propositions: tuple[CanonicalClaim, ...] = ()
    contribution_delivered: bool
    contribution_preserved: bool = True
    source_echo: bool
    implied_restatement: bool
    internal_repetition: bool
    analytical_packaging: bool
    source_stronger_than_draft: bool
    no_grounded_contribution: bool
    issue_codes: tuple[str, ...] = ()
    evidence_spans: tuple[str, ...] = ()
    humor_intent_proposal: SemanticHumorProposal | None = None
    humor_safety: HumorSafetyInspection | None = None
    semantic_inspection: SemanticInspection | None = None


class FactualClaimInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str = Field(min_length=1)
    kind: ClaimKind
    support: ClaimSupport
    evidence_ids: tuple[str, ...] = ()
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    explanation: str = ""
    issue_codes: tuple[str, ...] = ()
    human_verification_required: bool = False
    supporting_evidence_ids: tuple[str, ...] = ()
    partial_evidence_ids: tuple[str, ...] = ()
    conflicting_evidence_ids: tuple[str, ...] = ()
    attributed_evidence_ids: tuple[str, ...] = ()
    uncertain_evidence_ids: tuple[str, ...] = ()


class FactualInspectionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    status: Literal["pass", "block", "not_applicable"]
    claims: tuple[FactualClaimInspection, ...] = ()
    invalid_evidence_ids: tuple[str, ...] = ()
    invalid_excerpts: tuple[str, ...] = ()
    approval_eligible: bool
    extracted_claims: tuple[CanonicalClaim, ...] = ()
    provider_hint_claims: tuple[CanonicalClaim, ...] = ()
    claim_changes: tuple[ClaimChange, ...] = ()
    human_verification_required: bool = False
    semantic_reinspection_required: bool = False


class FinalTextInspection(BaseModel):
    """Canonical merged inspection result shared by every final-text workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extracted_claims: tuple[CanonicalClaim, ...] = ()
    source_semantic_map: SourceSemanticMap
    structured_semantic_inspection: DraftSemanticInspection
    semantic_extraction_result: SemanticInspectionResult | None = None
    editorial_report: EditorialReport
    factual_report: FactualInspectionReport


class UntrustedSourceData(BaseModel):
    """Compatibility representation of an external post as inert data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["untrusted_external_content"] = "untrusted_external_content"
    source_type: str = "x_post"
    source_id: str | None = None
    trusted: Literal[False] = False
    instructions_allowed: Literal[False] = False
    content: str
    author_username: str | None = None
    url: str | None = None
    created_at: datetime | None = None
    verification_status: VerificationStatus = "unknown"
    is_primary_source: bool = False
    public_metrics: dict[str, int | float] = Field(default_factory=dict)
    media_metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationRequest(BaseModel):
    """Provider-neutral request with explicit evidence and preselected angles."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    idea_summary: str = Field(min_length=1)
    content_type: ContentFormat
    language: str = Field(min_length=2)
    generation_mode: GenerationMode = GenerationMode.TREND_CONTEXT
    editorial_intent: EditorialIntent = "comment_on_source"
    evidence_packet: EvidencePacket
    source_coverage: SourceCoverage | None = None
    available_evidence: AvailableEvidence | None = None
    content_mode: ContentModeDecision | None = None
    signal_decision: SignalDecision | None = None
    auxiliary_evidence: tuple[AuxiliaryEvidenceBinding, ...] = ()
    auxiliary_relations: tuple[AuxiliaryRelation, ...] = ()
    angle_candidates: tuple[AngleCandidate, ...] = Field(default=(), max_length=3)
    literal_plans: tuple[LiteralDraftPlan, ...] = ()
    post_length_mode: str | None = None
    minimum_characters: int | None = Field(default=None, ge=1)
    maximum_characters: int | None = Field(default=None, ge=1)
    sources: tuple[UntrustedSourceData, ...] = ()
    feedback: str | None = None
    instructions: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    recent_posts: tuple[str, ...] = ()
    approved_style_examples: tuple[str, ...] = ()
    rejected_style_examples: tuple[str, ...] = ()
    revision_targets: tuple[HumanizerRevisionTarget, ...] = ()
    revision_issue_codes: tuple[str, ...] = ()
    output_schema: type[BaseModel] = Field(default=GenerationResult, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_request(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        normalized = dict(data)
        if "evidence_packet" in normalized and "angle_candidates" in normalized:
            return normalized
        topic = str(normalized.get("idea_summary") or "Untitled topic")
        language = str(normalized.get("language") or "en")
        mode = normalized.get("generation_mode") or GenerationMode.TOPIC_ONLY
        packet = EvidencePacket(
            mode=mode,
            topic=topic,
            items=(
                EvidenceItem(
                    evidence_id="legacy-evidence",
                    source_type="user_input",
                    author_or_source="operator",
                    text=topic,
                ),
            ),
            language=language,
        )
        normalized["generation_mode"] = mode
        normalized["evidence_packet"] = packet
        legacy_metadata = dict(normalized.get("metadata") or {})
        legacy_metadata["legacy_compatibility_request"] = True
        normalized["metadata"] = legacy_metadata
        normalized["angle_candidates"] = (
            AngleCandidate(
                angle_id="angle-1",
                angle_type="plain_update",
                thesis="Express one concrete observation.",
                evidence_ids=("legacy-evidence",),
                why_interesting="Compatibility angle for a legacy request.",
                confidence="medium",
            ),
            AngleCandidate(
                angle_id="angle-2",
                angle_type="skeptical_caveat",
                thesis="Express one precise caveat.",
                evidence_ids=("legacy-evidence",),
                why_interesting="Compatibility angle for a legacy request.",
                confidence="medium",
            ),
            AngleCandidate(
                angle_id="angle-3",
                angle_type="practical_implication",
                thesis="Express one practical implication.",
                evidence_ids=("legacy-evidence",),
                why_interesting="Compatibility angle for a legacy request.",
                confidence="medium",
            ),
        )
        return normalized

    @model_validator(mode="after")
    def validate_request(self) -> GenerationRequest:
        if (self.minimum_characters is None) != (self.maximum_characters is None):
            raise ValueError("minimum_characters and maximum_characters must be set together")
        if (
            self.minimum_characters is not None
            and self.maximum_characters is not None
            and self.minimum_characters > self.maximum_characters
        ):
            raise ValueError("minimum_characters must not exceed maximum_characters")
        if self.evidence_packet.mode is not self.generation_mode:
            raise ValueError("evidence packet mode must match generation mode")
        evidence_ids = {item.evidence_id for item in self.evidence_packet.items}
        angle_ids = [angle.angle_id for angle in self.angle_candidates]
        if len(angle_ids) != len(set(angle_ids)):
            raise ValueError("angle candidate IDs must be unique")
        candidate_by_id = {angle.angle_id: angle for angle in self.angle_candidates}
        for angle in self.angle_candidates:
            if not set(angle.evidence_ids).issubset(evidence_ids):
                raise ValueError("angle candidate references unknown evidence IDs")
        if self.signal_decision is not None and self.signal_decision.source_id not in evidence_ids:
            raise ValueError("signal decision must reference canonical evidence")
        auxiliary_ids = [binding.evidence_id for binding in self.auxiliary_evidence]
        if len(auxiliary_ids) != len(set(auxiliary_ids)):
            raise ValueError("auxiliary evidence bindings must be unique")
        if not set(auxiliary_ids).issubset(evidence_ids):
            raise ValueError("auxiliary evidence binding references unknown evidence")
        if self.signal_decision is not None and self.signal_decision.source_id in auxiliary_ids:
            raise ValueError("the primary anchor cannot also be auxiliary evidence")
        if self.metadata.get("one_anchor_policy") is True and self.signal_decision is None:
            raise ValueError("one-anchor policy requires a selected signal decision")
        if bool(self.revision_targets) != bool(self.revision_issue_codes):
            raise ValueError("revision targets and revision issue codes must be supplied together")
        if self.revision_targets:
            if len(self.revision_targets) != len(self.angle_candidates):
                raise ValueError("revision targets must align with supplied candidates")
            target_angle_ids = [target.variant.angle_id for target in self.revision_targets]
            if len(target_angle_ids) != len(set(target_angle_ids)):
                raise ValueError("revision target angle IDs must be unique")
            if set(target_angle_ids) != set(candidate_by_id):
                raise ValueError("revision targets must preserve the supplied angle candidates")
            union_codes = {code for target in self.revision_targets for code in target.issue_codes}
            if union_codes != set(self.revision_issue_codes):
                raise ValueError("revision issue-code summary must match target issue codes")
            for target in self.revision_targets:
                candidate = candidate_by_id[target.variant.angle_id]
                if target.variant.angle_type != candidate.angle_type:
                    raise ValueError("revision target changed the supplied angle type")
                if set(target.variant.evidence_ids) != set(candidate.evidence_ids):
                    raise ValueError("revision target changed the supplied evidence IDs")
        return self


class GenerationCritique(BaseModel):
    """Structured second-provider review used by the dual-review pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    originality: str
    topic_alignment: str
    unverified_claims: tuple[str, ...] = ()
    tone_notes: tuple[str, ...] = ()
    improvements: tuple[str, ...] = ()


class GenerationBundle(BaseModel):
    """All durable stages of a completed dual-review generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial: GenerationResult
    critique: GenerationCritique
    final: GenerationResult
    metadata: dict[str, Any] = Field(default_factory=dict)
