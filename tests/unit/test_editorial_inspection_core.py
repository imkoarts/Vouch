# ruff: noqa: RUF001
from __future__ import annotations

from app.domain.enums import ContentType, GenerationMode
from app.schemas.content import AngleCandidate, EvidenceItem, EvidencePacket, GenerationVariant
from app.services.content_mode import route_content_mode
from app.services.editorial_inspection import inspect_editorial, inspect_variants
from app.services.generation_pipeline import extract_source_coverage


def _packet(*texts: str) -> EvidencePacket:
    return EvidencePacket(
        mode=GenerationMode.SOURCE_POST,
        topic="regression",
        language="en",
        items=tuple(
            EvidenceItem(evidence_id=f"e{index}", source_type="x_post", text=text)
            for index, text in enumerate(texts)
        ),
    )


def _candidate(*evidence_ids: str, assumptions: tuple[str, ...] = ()) -> AngleCandidate:
    return AngleCandidate(
        angle_id="candidate",
        angle_type="practical_implication",
        contribution_type="connection",
        thesis="Use the validated evidence relation.",
        evidence_ids=tuple(evidence_ids),
        why_interesting="Grounded regression candidate.",
        confidence="high",
        unsupported_assumptions=assumptions,
        requires_new_assumptions=bool(assumptions),
        support_status="unsupported" if assumptions else "supported",
    )


def _commentary_mode():
    return route_content_mode(
        editorial_intent="comment_on_source",
        generation_mode=GenerationMode.SOURCE_POST,
        requested_format=ContentType.SHORT_POST,
    )


def test_hungarian_source_restatement_blocks_canonical_categories() -> None:
    source = (
        "BREAKING: Hungary’s parliament votes to remove President Tamás Sulyok, in the latest "
        "move to dismantle Viktor Orbán’s remaining influence."
    )
    draft = (
        "Hungary’s parliament voting to remove President Tamás Sulyok makes the change concrete: "
        "a named officeholder is being removed, rather than Viktor Orbán’s influence being "
        "discussed in the abstract. The presidency is now part of the political shift itself. "
        "That is a more tangible development than a general argument about influence."
    )
    packet = _packet(source)

    report = inspect_editorial(
        draft,
        packet=packet,
        contribution=_candidate("e0"),
        content_mode=_commentary_mode(),
        coverage=extract_source_coverage(packet),
    )

    assert report.status == "block"
    assert {
        "SOURCE_ECHO",
        "IMPLIED_POINT_RESTATEMENT",
        "LOW_INFORMATION_GAIN",
        "ANALYTICAL_PACKAGING",
        "INTERNAL_REPETITION",
        "MANUFACTURED_CONTRAST",
        "SOURCE_STRONGER_THAN_DRAFT",
    } <= {issue.code for issue in report.issues}
    assert all(issue.evidence_spans for issue in report.issues)


def test_polymarket_analytical_packaging_blocks_with_subtypes() -> None:
    packet = _packet("Polymarket launched combos and a trading tournament.")
    draft = (
        "A trading tournament around Polymarket's new combos product gives the format a narrow "
        "test: how people use it when there is a reason to trade it actively. If combos extend "
        "beyond sports, the practical question is not only whether more categories can be listed. "
        "It is whether the combination remains legible enough that a trader can understand what "
        "is being priced."
    )

    report = inspect_editorial(
        draft,
        packet=packet,
        contribution=_candidate("e0"),
        content_mode=_commentary_mode(),
        coverage=extract_source_coverage(packet),
    )

    codes = {issue.code for issue in report.issues}
    assert {
        "ANALYTICAL_PACKAGING",
        "LOW_INFORMATION_GAIN",
        "MANUFACTURED_CONTRAST",
        "INTERNAL_REPETITION",
    } <= codes
    packaging = next(issue for issue in report.issues if issue.code == "ANALYTICAL_PACKAGING")
    assert {
        "abstract_test_framing",
        "staged_thesis",
        "product_memo_voice",
        "abstract_noun_density",
    } <= set(packaging.subtypes)


def test_trace_ownership_is_forced_novelty_without_evidence() -> None:
    packet = _packet("We do not train on your data. You own your data.")
    draft = (
        "the contract problem gets messy fast: a correction mixes the model’s output with the "
        "company’s private context and an employee’s judgment. “you own your data” doesn’t answer "
        "who owns that trace — or whether you can take it to another model."
    )

    report = inspect_editorial(
        draft,
        packet=packet,
        contribution=_candidate("e0"),
        content_mode=_commentary_mode(),
        coverage=extract_source_coverage(packet),
    )

    assert {"FORCED_NOVELTY", "UNSUPPORTED_CONTRIBUTION", "ANALYTICAL_PACKAGING"} <= {
        issue.code for issue in report.issues
    }


def test_source_compression_is_not_commentary() -> None:
    packet = _packet(
        "Portability should include evals, corrections, traces, memory, and orchestration. "
        "Without them, switching models leaves part of the work behind."
    )
    draft = (
        "“we don’t train on your data” is a pretty narrow promise. if the corrections, evals and "
        "workflow history can’t move with the customer, switching models means leaving part of "
        "the work behind."
    )

    report = inspect_editorial(
        draft,
        packet=packet,
        contribution=_candidate("e0"),
        content_mode=_commentary_mode(),
        coverage=extract_source_coverage(packet),
    )

    assert {
        "SOURCE_ECHO",
        "IMPLIED_POINT_RESTATEMENT",
        "LOW_INFORMATION_GAIN",
        "ANALYTICAL_PACKAGING",
    } <= {issue.code for issue in report.issues}


def test_topic_overlap_only_is_incoherent_synthesis() -> None:
    packet = _packet(
        "Messi scored for Inter Miami.",
        "FIFA announced a tournament format change.",
    )
    report = inspect_editorial(
        "Messi scored while FIFA changed a tournament format, showing football is shifting.",
        packet=packet,
        contribution=_candidate("e0", "e1"),
        content_mode=_commentary_mode(),
        coverage=extract_source_coverage(_packet("Messi scored for Inter Miami.")),
    )

    issue = next(item for item in report.issues if item.code == "INCOHERENT_SYNTHESIS")
    assert {"topic_overlap_only", "no_propositional_relation", "artificial_contrast"} <= set(
        issue.subtypes
    )


def test_direct_factual_update_control_passes() -> None:
    packet = _packet("Alice signed the bill on Tuesday.")
    mode = route_content_mode(
        editorial_intent="report_event",
        generation_mode=GenerationMode.SOURCE_POST,
        requested_format=ContentType.SHORT_POST,
    )
    report = inspect_editorial(
        "Alice signed the bill on Tuesday.",
        packet=packet,
        contribution=_candidate("e0").model_copy(
            update={
                "angle_type": "plain_update",
                "contribution_type": "direct_update",
                "thesis": "Alice signed the bill on Tuesday.",
            }
        ),
        content_mode=mode,
        coverage=extract_source_coverage(packet),
    )

    assert report.status == "pass"
    assert report.issues == ()


def test_short_complete_quote_joke_is_not_padded_or_blocked() -> None:
    packet = _packet("The benchmark took 23 minutes to run.")
    mode = route_content_mode(
        editorial_intent="quote_reaction",
        generation_mode=GenerationMode.QUOTE_POST,
        requested_format=ContentType.QUOTE_COMMENTARY,
    )
    report = inspect_editorial(
        "23 minutes. the benchmark had time to reflect.",
        packet=packet,
        contribution=_candidate("e0").model_copy(
            update={"angle_type": "concise_joke", "contribution_type": "joke"}
        ),
        content_mode=mode,
        coverage=extract_source_coverage(packet),
        preferred_length_min=300,
    )

    assert "FORCED_LENGTH" not in {issue.code for issue in report.issues}


def test_semantically_duplicate_variants_are_blocked() -> None:
    packet = _packet("Alice signed the bill on Tuesday.")
    mode = route_content_mode(
        editorial_intent="report_event",
        generation_mode=GenerationMode.SOURCE_POST,
        requested_format=ContentType.SHORT_POST,
    )
    first = _candidate("e0").model_copy(
        update={
            "angle_id": "first",
            "angle_type": "plain_update",
            "contribution_type": "direct_update",
            "thesis": "Alice signed the bill on Tuesday.",
        }
    )
    second = first.model_copy(update={"angle_id": "second"})
    reports = inspect_variants(
        (
            GenerationVariant(
                label="first",
                angle_id="first",
                angle_type="plain_update",
                text="Alice signed the bill on Tuesday.",
                evidence_ids=("e0",),
                similarity_risk=0.0,
                confidence="high",
                factual_claims=("Alice signed the bill on Tuesday.",),
                attributed_claims=(),
                uncertainty_markers=(),
            ),
            GenerationVariant(
                label="second",
                angle_id="second",
                angle_type="plain_update",
                text="On Tuesday, Alice signed the bill.",
                evidence_ids=("e0",),
                similarity_risk=0.0,
                confidence="high",
                factual_claims=("On Tuesday, Alice signed the bill.",),
                attributed_claims=(),
                uncertainty_markers=(),
            ),
        ),
        packet=packet,
        contributions=(first, second),
        content_mode=mode,
        coverage=extract_source_coverage(packet),
    )

    assert all(
        "VARIANTS_NOT_DISTINCT" in {issue.code for issue in report.issues} for report in reports
    )


def test_preferred_length_padding_is_blocked_as_forced_length() -> None:
    packet = _packet("The company cut the fee from 2% to 1%.")
    draft = (
        "The fee reduction turns pricing into part of the product shift itself. "
        "This development makes the change more tangible. "
        "The shift is meaningful because pricing is now part of the framework."
    )
    report = inspect_editorial(
        draft,
        packet=packet,
        contribution=_candidate("e0"),
        content_mode=_commentary_mode(),
        coverage=extract_source_coverage(packet),
        preferred_length_min=120,
    )

    assert "FORCED_LENGTH" in {issue.code for issue in report.issues}


def test_natural_controls_allow_passive_technical_and_factual_list() -> None:
    packet = _packet("Write a compact technical release note.")
    mode = route_content_mode(
        editorial_intent="explain_topic",
        generation_mode=GenerationMode.TOPIC_ONLY,
        requested_format=ContentType.SHORT_POST,
    )
    texts = (
        "The migration was completed after the checksum passed.",
        "A capability token limits which process can call the storage service.",
        "The release includes the API, CLI, and migration guide.",
    )

    for index, text in enumerate(texts):
        contribution = _candidate("e0").model_copy(
            update={
                "angle_id": f"control-{index}",
                "contribution_type": "opinion",
                "thesis": text,
            }
        )
        report = inspect_editorial(
            text,
            packet=packet,
            contribution=contribution,
            content_mode=mode,
            coverage=extract_source_coverage(packet),
        )
        assert report.status == "pass", report
