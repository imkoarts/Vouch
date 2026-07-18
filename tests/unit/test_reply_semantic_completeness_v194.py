from __future__ import annotations

import pytest

from app.schemas.content import SemanticProviderProposal
from app.services.semantic_adjudication import adjudicate_humor_intent
from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator
from app.services.semantic_reconciliation import reconcile_semantic_provider_proposal


def _intent(inspection):
    return adjudicate_humor_intent(
        inspection,
        metadata_says_humor=False,
        reaction_type="none",
        template_humor=False,
    )


@pytest.mark.parametrize(
    "reply",
    (
        "the deployment window",
        "however, the deployment window",
        "the deployment window (after review)",
    ),
)
def test_material_reply_fragment_cannot_safe_complete_sensitive_candidate(reply: str) -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "The collision killed a maintenance worker.", reply
    )
    inspection = result.canonical

    assert inspection.reply_semantic_completeness
    assert any(
        item.material_surface_present
        and not item.proposition_complete
        and item.unresolved_category == "incomplete_proposition"
        for item in inspection.reply_semantic_completeness
    )
    assert any(
        "incomplete_proposition" in coverage.unresolved_categories and not coverage.complete
        for coverage in inspection.reply_coverage
    )
    intent = _intent(inspection)
    assert "HUMOR_INTENT_UNCERTAIN" in intent.issue_codes
    assert not intent.text_appears_humorous


def test_complete_literal_proposition_remains_validated_on_sensitive_candidate() -> None:
    inspection = (
        SemanticExtractionCoordinator()
        .inspect_local(
            "The collision killed a maintenance worker.",
            "the incident requires an independent investigation",
        )
        .canonical
    )

    assert all(item.proposition_complete for item in inspection.reply_semantic_completeness)
    assert all(item.validated_literal_function for item in inspection.reply_semantic_completeness)
    assert all(item.complete for item in inspection.reply_coverage)
    assert "HUMOR_INTENT_UNCERTAIN" not in _intent(inspection).issue_codes


@pytest.mark.parametrize(
    "reply",
    (
        "what an impressive milestone for the response unit",
        "strip away the branding: onboarding is paperwork",
    ),
)
def test_typed_evaluation_or_shell_is_not_relabelled_as_incomplete_proposition(reply: str) -> None:
    inspection = (
        SemanticExtractionCoordinator()
        .inspect_local("The collision killed a maintenance worker.", reply)
        .canonical
    )

    assert inspection.evaluations or inspection.reply_shells
    assert inspection.reply_semantic_completeness
    assert all(item.unresolved_category is None for item in inspection.reply_semantic_completeness)


@pytest.mark.parametrize(
    ("source", "domain", "impact"),
    (
        ("The fumes left a technician with reduced vision.", "sensory", "reduced"),
        ("The impact left two riders unable to keep their balance.", "mobility", "lost"),
        (
            "The exposure left three workers needing assistance to breathe.",
            "respiratory",
            "assistance_required",
        ),
        (
            "The incident left one resident dependent on breathing support.",
            "respiratory",
            "support_required",
        ),
    ),
)
def test_human_function_impact_preserves_bound_roles(source: str, domain: str, impact: str) -> None:
    inspection = SemanticExtractionCoordinator().inspect_local(source).canonical

    assert any(
        item.function_domain == domain
        and item.impact == impact
        and item.asserted
        and item.safety_material
        for item in inspection.human_function_impacts
    )


def test_provider_none_preserves_application_owned_completeness_scope_and_roles() -> None:
    deterministic = (
        SemanticExtractionCoordinator()
        .inspect_local(
            "The fumes left a technician with reduced vision.", "however, the deployment window"
        )
        .deterministic
    )

    canonical = reconcile_semantic_provider_proposal(deterministic, SemanticProviderProposal())

    assert canonical.reply_semantic_completeness == deterministic.reply_semantic_completeness
    assert canonical.human_function_impacts == deterministic.human_function_impacts
    assert canonical.predicate_scopes == deterministic.predicate_scopes
    assert canonical.reply_coverage == deterministic.reply_coverage
