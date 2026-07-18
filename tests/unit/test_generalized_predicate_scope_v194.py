from __future__ import annotations

import pytest

from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator


@pytest.mark.parametrize(
    ("source", "relation"),
    (
        ("Although three operators were injured, the cooling controller failed.", "concessive"),
        ("If the cooling controller failed, three operators might be injured.", "conditional"),
        ("Three operators were injured when the cooling controller failed.", "temporal"),
        ("Three operators were injured because the cooling controller failed.", "causal"),
        ("Three operators were injured, whereas the cooling controller failed.", "contrastive"),
    ),
)
def test_subordinate_technical_predicate_cannot_borrow_matrix_human_role(
    source: str, relation: str
) -> None:
    inspection = SemanticExtractionCoordinator().inspect_local(source).canonical

    assert any(scope.relation_to_parent == relation for scope in inspection.predicate_scopes)
    technical_failures = [
        event for event in inspection.source_events if event.predicate in {"failed", "fails"}
    ]
    assert technical_failures
    assert all(
        event.affected_entity is None
        or event.affected_entity.entity_type not in {"person", "group_of_people"}
        for event in technical_failures
    )


def test_participial_scope_has_independent_explicit_human_role() -> None:
    inspection = (
        SemanticExtractionCoordinator()
        .inspect_local("The controller failed, leaving three operators waiting outside.")
        .canonical
    )

    assert any(scope.relation_to_parent == "participial" for scope in inspection.predicate_scopes)
    failure = next(event for event in inspection.source_events if event.predicate == "failed")
    assert (
        failure.affected_entity is None or failure.affected_entity.entity_type != "group_of_people"
    )


def test_embedded_relative_scope_is_explicit_and_request_local() -> None:
    inspection = (
        SemanticExtractionCoordinator()
        .inspect_local(
            "The controller, which failed overnight, delayed three operators.",
            "the report, which remains incomplete, needs an audit",
        )
        .canonical
    )

    relatives = [
        scope for scope in inspection.predicate_scopes if scope.relation_to_parent == "relative"
    ]
    assert {scope.text_span for scope in relatives} == {
        "which failed overnight",
        "which remains incomplete",
    }
    assert len({scope.scope_id for scope in inspection.predicate_scopes}) == len(
        inspection.predicate_scopes
    )
    assert all(scope.parent_scope_id for scope in relatives)


def test_nested_subordinate_scopes_keep_each_relation_bound_to_a_matrix() -> None:
    inspection = (
        SemanticExtractionCoordinator()
        .inspect_local(
            "Although the controller failed after the alarm sounded, two operators remained safe."
        )
        .canonical
    )

    relations = {scope.relation_to_parent for scope in inspection.predicate_scopes}
    assert {"matrix", "concessive", "temporal"} <= relations
    assert all(
        scope.parent_scope_id is not None
        for scope in inspection.predicate_scopes
        if scope.relation_to_parent != "matrix"
    )
