from __future__ import annotations

from app.domain.enums import ContentType, GenerationMode
from app.schemas.content import UntrustedSourceData
from app.services.content_mode import route_content_mode
from app.services.generation_pipeline import build_generation_context
from app.services.signal_selection import evaluate_signal, select_publishable_signal


def _source(source_id: str, text: str) -> UntrustedSourceData:
    return UntrustedSourceData(source_id=source_id, source_type="recent_search", content=text)


def test_selector_accepts_viable_factual_candidate() -> None:
    mode = route_content_mode(
        editorial_intent="report_event",
        generation_mode=GenerationMode.SOURCE_POST,
        requested_format=ContentType.SHORT_POST,
    )

    decision = evaluate_signal(
        _source("one", "Alice signed the bill on Tuesday."), content_mode=mode
    )

    assert decision.supports_requested_mode is True
    assert decision.action in {"generate", "generate_with_verification"}
    assert decision.supported_contribution_count == 1


def test_news_lead_with_unlisted_verb_is_still_semantically_complete() -> None:
    mode = route_content_mode(
        editorial_intent="report_event",
        generation_mode=GenerationMode.SOURCE_POST,
        requested_format=ContentType.SHORT_POST,
    )

    decision = evaluate_signal(
        _source(
            "one",
            "BREAKING: Trump Media weighs charging traders $100,000 a month for access.",
        ),
        content_mode=mode,
    )

    assert decision.semantically_complete is True
    assert decision.supports_requested_mode is True


def test_selector_rejects_complete_source_without_commentary_transformation() -> None:
    mode = route_content_mode(
        editorial_intent="comment_on_source",
        generation_mode=GenerationMode.SOURCE_POST,
        requested_format=ContentType.SHORT_POST,
    )

    decision = evaluate_signal(
        _source("one", "Alice signed the bill on Tuesday."), content_mode=mode
    )

    assert decision.supports_requested_mode is False
    assert decision.action == "reject_source_stronger_than_draft"
    assert decision.risk_forced_novelty == "high"


def test_candidate_iteration_moves_to_next_ranked_source() -> None:
    mode = route_content_mode(
        editorial_intent="report_event",
        generation_mode=GenerationMode.SOURCE_POST,
        requested_format=ContentType.SHORT_POST,
    )

    outcome = select_publishable_signal(
        (_source("empty", ""), _source("two", "Alice signed the bill on Tuesday.")),
        content_mode=mode,
        max_attempts=2,
    )

    assert outcome.status == "ok"
    assert outcome.anchor is not None
    assert outcome.anchor.source_id == "two"
    assert [item.source_id for item in outcome.decisions] == ["empty", "two"]


def test_no_accepted_candidates_returns_no_publishable_signal() -> None:
    mode = route_content_mode(
        editorial_intent="comment_on_source",
        generation_mode=GenerationMode.SOURCE_POST,
        requested_format=ContentType.SHORT_POST,
    )

    outcome = select_publishable_signal(
        (
            _source("one", "Alice signed the bill on Tuesday."),
            _source("two", "The company cut the fee from 2% to 1%."),
        ),
        content_mode=mode,
    )

    assert outcome.status == "no_publishable_signal"
    assert outcome.anchor is None
    assert len(outcome.decisions) == 2


def test_one_anchor_policy_drops_topic_overlap_without_explicit_role() -> None:
    context = build_generation_context(
        idea_summary="football",
        idea_explanation={"editorial_intent": "report_event"},
        language="en",
        sources=(
            _source("messi", "Messi scored for Inter Miami."),
            _source("fifa", "FIFA announced a tournament format change."),
        ),
        content_type=ContentType.SHORT_POST,
    )

    assert context.status == "ok"
    assert context.selected_anchor is not None
    assert context.selected_anchor.source_id == "messi"
    assert {item.evidence_id for item in context.evidence.items} == {"messi"}
    assert context.auxiliary_evidence == ()


def test_explicit_auxiliary_role_allows_bounded_grounded_connection() -> None:
    context = build_generation_context(
        idea_summary="evaluation export",
        idea_explanation={
            "editorial_intent": "comment_on_source",
            "auxiliary_evidence_roles": {"requirements": "supplies_required_context"},
        },
        language="en",
        sources=(
            _source("export", "The export includes scores but excludes test cases."),
            _source(
                "requirements", "The target model requires test cases to rerun the evaluation."
            ),
        ),
        content_type=ContentType.SHORT_POST,
    )

    assert context.status == "ok"
    assert context.signal_decision is not None
    assert context.signal_decision.source_id == "export"
    assert {item.evidence_id for item in context.evidence.items} == {"export", "requirements"}
    assert len(context.auxiliary_evidence) == 1
    assert len(context.angles) == 1
    assert set(context.angles[0].evidence_ids) == {"export", "requirements"}


def test_content_mode_receives_explicit_preferred_and_hard_length_bounds() -> None:
    context = build_generation_context(
        idea_summary="release update",
        idea_explanation={"editorial_intent": "report_event"},
        language="en",
        sources=(_source("release", "Alice published the release on Tuesday."),),
        content_type=ContentType.LONG_POST,
        preferred_length_min=300,
        preferred_length_max=500,
        hard_length_max=500,
    )

    assert context.status == "ok"
    assert context.content_mode.preferred_length_min == 300
    assert context.content_mode.preferred_length_max == 500
    assert context.content_mode.hard_length_max == 500


def test_single_complete_commentary_source_is_source_already_sufficient() -> None:
    context = build_generation_context(
        idea_summary="bill update",
        idea_explanation={"editorial_intent": "comment_on_source"},
        language="en",
        sources=(_source("bill", "Alice signed the bill on Tuesday."),),
        content_type=ContentType.SHORT_POST,
    )

    assert context.status == "no_post_needed"
    assert context.terminal_status == "source_already_sufficient"


def test_incomplete_factual_source_is_insufficient_evidence() -> None:
    context = build_generation_context(
        idea_summary="fragment",
        idea_explanation={"editorial_intent": "report_event"},
        language="en",
        sources=(_source("fragment", "Unconfirmed fragment"),),
        content_type=ContentType.SHORT_POST,
    )

    assert context.status == "no_post_needed"
    assert context.terminal_status == "insufficient_evidence"
