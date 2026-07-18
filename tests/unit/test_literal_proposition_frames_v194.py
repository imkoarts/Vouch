from __future__ import annotations

import pytest

from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator


@pytest.mark.parametrize(
    ("reply", "proposition_type"),
    (
        ("the service remains unavailable", "status"),
        ("the cause has not been confirmed", "verification_state"),
        ("the outage remains under investigation", "investigation_state"),
        ("the report has been published", "publication_state"),
        ("the controller requires an independent inspection", "inspection_recommendation"),
        ("the controller must replace the damaged relay", "replacement_recommendation"),
        ("the team must preserve the diagnostic records", "preservation_or_isolation_action"),
    ),
)
def test_complete_literal_relations_are_typed_without_policy(
    reply: str, proposition_type: str
) -> None:
    inspection = (
        SemanticExtractionCoordinator()
        .inspect_local("A worker died after an industrial incident.", reply)
        .canonical
    )

    frame = next(
        item
        for item in inspection.literal_proposition_frames
        if item.proposition_type == proposition_type
    )
    assert frame.proposition_complete
    assert frame.subject_span
    assert frame.predicate_span
    assert frame.object_or_complement_span
    assert frame.evidence_spans == (reply,)
