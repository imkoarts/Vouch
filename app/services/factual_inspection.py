"""Deterministic evidence validation and claim-structure factual inspection."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from app.schemas.content import (
    CanonicalClaim,
    EvidencePacket,
    EvidenceSpan,
    FactualClaimInspection,
    FactualInspectionReport,
    GenerationVariant,
)
from app.services.claim_structure import (
    compare_claim_sets,
    compare_claim_to_evidence,
    extract_claims,
    extract_urls,
    merge_claim_hints,
    reconcile_claims,
)

_NDA_BREACH_ASSERTION = re.compile(
    r"\b(?:broke|breached|violated)\s+(?P<count>one|two|three|\d+)?\s*"
    r"nda(?:s)?\b(?:\s+in\s+(?P<medium>one|a|\d+)\s+(?:tweet|post))?",
    re.IGNORECASE,
)


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return " ".join(normalized.split())


def validate_evidence_spans(
    packet: EvidencePacket,
    spans: Sequence[EvidenceSpan],
) -> tuple[tuple[EvidenceSpan, ...], tuple[str, ...], tuple[str, ...]]:
    """Return only canonical spans; never synthesize replacement wording."""

    by_id = {item.evidence_id: item for item in packet.items}
    valid: list[EvidenceSpan] = []
    invalid_ids: list[str] = []
    invalid_excerpts: list[str] = []
    for span in spans:
        item = by_id.get(span.evidence_id)
        if item is None:
            invalid_ids.append(span.evidence_id)
            continue
        canonical = unicodedata.normalize("NFC", item.text)
        quote = unicodedata.normalize("NFC", span.quote)
        if span.start is not None or span.end is not None:
            if (
                span.start is None
                or span.end is None
                or span.start > span.end
                or span.end > len(canonical)
            ):
                invalid_excerpts.append(span.quote)
                continue
            if canonical[span.start : span.end] != quote:
                invalid_excerpts.append(span.quote)
                continue
        elif _normalize(quote) not in _normalize(canonical):
            invalid_excerpts.append(span.quote)
            continue
        valid.append(span)
    return tuple(valid), tuple(dict.fromkeys(invalid_ids)), tuple(dict.fromkeys(invalid_excerpts))


def _evidence_claims(
    packet: EvidencePacket,
    selected_ids: set[str],
) -> tuple[tuple[CanonicalClaim, ...], dict[str, str]]:
    claims: list[CanonicalClaim] = []
    evidence_by_claim: dict[str, str] = {}
    for item_index, item in enumerate(packet.items, start=1):
        if item.evidence_id not in selected_ids:
            continue
        for claim in extract_claims(item.text, prefix=f"E{item_index}-"):
            claims.append(claim)
            evidence_by_claim[claim.claim_id] = item.evidence_id
    return tuple(claims), evidence_by_claim


def inspect_factual_integrity(
    variant: GenerationVariant,
    *,
    packet: EvidencePacket,
    evidence_spans: Sequence[EvidenceSpan] = (),
    factual_inspection_required: bool = False,
    previous_claims: Sequence[CanonicalClaim] = (),
    extracted_claims: Sequence[CanonicalClaim] | None = None,
    provider_claims: Sequence[CanonicalClaim] = (),
    semantic_reinspection_required: bool = False,
) -> FactualInspectionReport:
    """Inspect final rendered text independently of model-declared claim arrays.

    Provider arrays and live extractor results are proposals. Claims without an exact rendered-text
    span are ignored. Categorical support states, not legacy numeric scores, own approval policy.
    """

    valid_spans, invalid_ids, invalid_excerpts = validate_evidence_spans(packet, evidence_spans)
    available_ids = {item.evidence_id for item in packet.items}
    unknown_variant_ids = tuple(sorted(set(variant.evidence_ids) - available_ids))
    invalid_ids = tuple(dict.fromkeys((*invalid_ids, *unknown_variant_ids)))
    selected_ids = set(variant.evidence_ids) & available_ids
    evidence_claims, evidence_by_claim = _evidence_claims(packet, selected_ids)

    canonical_urls = {item.url for item in packet.items if item.url}
    invalid_urls = tuple(url for url in extract_urls(variant.text) if url not in canonical_urls)
    invalid_excerpts = tuple(
        dict.fromkeys((*invalid_excerpts, *(f"URL:{url}" for url in invalid_urls)))
    )

    local = (
        tuple(extracted_claims) if extracted_claims is not None else extract_claims(variant.text)
    )
    hint_claims = merge_claim_hints(
        local,
        (*variant.factual_claims, *variant.attributed_claims, *variant.uncertainty_markers),
        final_text=variant.text,
    )
    all_provider_claims = tuple((*provider_claims, *hint_claims))
    claims_to_inspect = reconcile_claims(variant.text, local, all_provider_claims)

    inspections: list[FactualClaimInspection] = []
    material_count = 0
    human_verification_required = semantic_reinspection_required
    nda_assertion = _NDA_BREACH_ASSERTION.search(variant.text)
    if nda_assertion is not None:
        matching_items = []
        draft_count = (nda_assertion.group("count") or "").casefold()
        draft_medium = (nda_assertion.group("medium") or "").casefold()
        for item in packet.items:
            source_match = _NDA_BREACH_ASSERTION.search(item.text)
            if source_match is None:
                continue
            source_count = (source_match.group("count") or "").casefold()
            source_medium = (source_match.group("medium") or "").casefold()
            if draft_count and source_count and draft_count != source_count:
                continue
            if draft_medium and source_medium and draft_medium != source_medium:
                continue
            matching_items.append(item)
        material_count += 1
        if matching_items:
            inspections.append(
                FactualClaimInspection(
                    claim=nda_assertion.group(0),
                    kind="factual",
                    support="supported",
                    evidence_ids=tuple(item.evidence_id for item in matching_items),
                    evidence_spans=tuple(
                        EvidenceSpan(evidence_id=item.evidence_id, quote=item.text)
                        for item in matching_items
                    ),
                    explanation=(
                        "The source explicitly establishes the NDA-breach assertion used by the "
                        "reply; humor does not expand the legal allegation."
                    ),
                )
            )
        else:
            inspections.append(
                FactualClaimInspection(
                    claim=nda_assertion.group(0),
                    kind="factual",
                    support="unsupported",
                    explanation=(
                        "The reply asserts an NDA breach, but the selected evidence does not "
                        "explicitly establish that legal/factual allegation."
                    ),
                    issue_codes=("UNSUPPORTED_FACT", "UNSUPPORTED_EVENT_DETAIL"),
                )
            )
    for claim in claims_to_inspect:
        comparison = compare_claim_to_evidence(claim, evidence_claims)
        support = comparison.support
        if claim.kind in {
            "factual",
            "attributed_claim",
            "interpretation",
            "material_claim_unknown",
        }:
            material_count += 1

        def ids_for(claims: Sequence[CanonicalClaim]) -> tuple[str, ...]:
            return tuple(
                dict.fromkeys(
                    evidence_by_claim[item.claim_id]
                    for item in claims
                    if item.claim_id in evidence_by_claim
                )
            )

        supporting_ids = ids_for(comparison.supporting_claims)
        partial_ids = ids_for(comparison.partial_claims)
        conflicting_ids = ids_for(comparison.conflicting_claims)
        attributed_ids = ids_for(comparison.attributed_claims)
        uncertain_ids = ids_for(comparison.uncertain_claims)
        evidence_ids = tuple(
            dict.fromkeys(
                (*supporting_ids, *partial_ids, *conflicting_ids, *attributed_ids, *uncertain_ids)
            )
        )
        if not evidence_ids and comparison.evidence_claim is not None:
            evidence_id = evidence_by_claim.get(comparison.evidence_claim.claim_id)
            if evidence_id:
                evidence_ids = (evidence_id,)
        matched_spans = tuple(span for span in valid_spans if span.evidence_id in evidence_ids)
        if not matched_spans:
            matched_spans = tuple(
                EvidenceSpan(evidence_id=evidence_by_claim[item.claim_id], quote=item.exact_text)
                for item in (
                    *comparison.supporting_claims,
                    *comparison.partial_claims,
                    *comparison.conflicting_claims,
                    *comparison.attributed_claims,
                    *comparison.uncertain_claims,
                )
                if item.claim_id in evidence_by_claim
            )
        issue_codes = list(comparison.issue_codes)
        if claim.unconsumed_material_text:
            issue_codes.extend(("CLAIM_EXTRACTION_UNCERTAIN", "UNINSPECTED_SECONDARY_PROPOSITION"))
        elif claim.clause_relation != "main" and support not in {"supported", "not_applicable"}:
            issue_codes.append("UNINSPECTED_SECONDARY_PROPOSITION")
        human_required = comparison.human_verification_required or support in {
            "partially_supported",
            "attributed_only",
            "extraction_uncertain",
        }
        human_verification_required = human_verification_required or human_required
        inspections.append(
            FactualClaimInspection(
                claim=claim.exact_text,
                kind=claim.kind,
                support=support,
                evidence_ids=evidence_ids,
                evidence_spans=matched_spans,
                explanation=comparison.explanation,
                issue_codes=tuple(dict.fromkeys(issue_codes)),
                human_verification_required=human_required,
                supporting_evidence_ids=supporting_ids,
                partial_evidence_ids=partial_ids,
                conflicting_evidence_ids=conflicting_ids,
                attributed_evidence_ids=attributed_ids,
                uncertain_evidence_ids=uncertain_ids,
            )
        )

    changes = compare_claim_sets(previous_claims, claims_to_inspect) if previous_claims else ()
    blocking_supports = {
        "partially_supported",
        "attributed_only",
        "unsupported",
        "source_conflict",
        "extraction_uncertain",
    }
    blocked = (
        bool(invalid_ids or invalid_excerpts)
        or semantic_reinspection_required
        or any(claim.support in blocking_supports for claim in inspections)
    )
    if factual_inspection_required and material_count == 0:
        blocked = True
        inspections.append(
            FactualClaimInspection(
                claim=variant.text,
                kind="material_claim_unknown",
                support="extraction_uncertain",
                explanation=(
                    "The selected mode requires factual inspection, but final rendered text "
                    "contains no confidently extractable material claim."
                ),
                issue_codes=("CLAIM_EXTRACTION_UNCERTAIN",),
                human_verification_required=True,
            )
        )
        human_verification_required = True

    applicable = material_count > 0 or factual_inspection_required or semantic_reinspection_required
    if not applicable and not invalid_ids and not invalid_excerpts:
        return FactualInspectionReport(
            applicable=False,
            status="not_applicable",
            claims=tuple(inspections)
            or (
                FactualClaimInspection(
                    claim=variant.text,
                    kind="opinion",
                    support="not_applicable",
                    explanation="No material factual premise requires evidence validation.",
                ),
            ),
            invalid_evidence_ids=invalid_ids,
            invalid_excerpts=invalid_excerpts,
            approval_eligible=True,
            extracted_claims=claims_to_inspect,
            provider_hint_claims=tuple(all_provider_claims),
            claim_changes=tuple(changes),
            human_verification_required=False,
            semantic_reinspection_required=False,
        )

    return FactualInspectionReport(
        applicable=True,
        status="block" if blocked else "pass",
        claims=tuple(inspections),
        invalid_evidence_ids=invalid_ids,
        invalid_excerpts=invalid_excerpts,
        approval_eligible=not blocked,
        extracted_claims=claims_to_inspect,
        provider_hint_claims=tuple(all_provider_claims),
        claim_changes=tuple(changes),
        human_verification_required=human_verification_required,
        semantic_reinspection_required=semantic_reinspection_required,
    )
