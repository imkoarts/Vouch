"""Run an auditor-supplied clause-scoped hybrid-semantic holdout against a frozen release candidate.

The runner is network-free.  Cases may exercise deterministic-only behavior, offline unresolved
behavior, or an auditor-supplied structured provider proposal.  Provider proposals are validated
and reconciled by the same coordinator used by the application.  Exit status is 0 when all
expectations pass, 1 for semantic mismatches, and 2 for invalid input or runner errors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas.content import (  # noqa: E402
    SemanticInspection,
    SemanticInspectionResult,
    SemanticProviderProposal,
    SemanticProviderRequest,
)
from app.services.semantic_adjudication import (  # noqa: E402
    adjudicate_evaluative_irony,
    adjudicate_human_harm,
    adjudicate_humor_intent,
    adjudicate_humor_safety,
    adjudicate_reply_shell,
)
from app.services.semantic_extraction_coordinator import (  # noqa: E402
    SemanticExtractionCoordinator,
)

RunMode = Literal["deterministic_only", "offline_unresolved", "provider_assisted"]


class HoldoutExpectation(BaseModel):
    """Optional expectations; omitted fields are intentionally not asserted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    harm_referent: Literal["human", "non_human", "unknown"] | None = None
    harm_state: str | None = None
    literal_harm: bool | None = None
    implicit_sarcasm: bool | None = None
    text_appears_humorous: bool | None = None
    humor_safety_required: bool | None = None
    sensitive_context: bool | None = None
    suitable_for_humor: bool | None = None
    shell_operator: str | None = None
    semantic_candidate_eligible: bool | None = None
    escalation_required: bool | None = None
    provider_used: bool | None = None
    provider_validation_error_count: int | None = None
    required_material_categories: tuple[str, ...] = ()
    required_canonical_candidate_issue_codes: tuple[str, ...] = ()
    forbidden_canonical_candidate_issue_codes: tuple[str, ...] = ()
    required_provider_validation_errors: tuple[str, ...] = ()
    required_source_event_types: tuple[str, ...] = ()
    required_source_event_affected_types: tuple[str, ...] = ()
    required_source_event_assertion_states: tuple[str, ...] = ()
    required_source_event_evidence_spans: tuple[str, ...] = ()
    required_source_unresolved_categories: tuple[str, ...] = ()
    required_source_unresolved_spans: tuple[str, ...] = ()
    required_reply_unresolved_categories: tuple[str, ...] = ()
    required_unresolved_after_reconciliation: tuple[str, ...] = ()


class HoldoutCase(BaseModel):
    """One independent hybrid-semantic holdout case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    source: str
    reply: str = ""
    mode: RunMode = "deterministic_only"
    metadata_says_humor: bool = False
    reaction_type: Literal[
        "literalization",
        "incongruity",
        "callback",
        "wordplay",
        "dry_reframe",
        "none",
        "uncertain",
    ] = "none"
    template_humor: bool = False
    provider_proposal: SemanticProviderProposal | None = None
    expect: HoldoutExpectation


class HoldoutDocument(BaseModel):
    """Versioned external-holdout payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["4.0"]
    holdout_id: str = Field(min_length=1)
    cases: tuple[HoldoutCase, ...] = Field(min_length=1)


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    passed: bool
    failures: tuple[str, ...]
    observed: dict[str, object]


class _ProposalProvider:
    """Deterministic, network-free provider used only for auditor-supplied proposals."""

    def __init__(self, proposal: SemanticProviderProposal) -> None:
        self.proposal = proposal
        self.requests: list[SemanticProviderRequest] = []

    async def extract_semantics(self, request: SemanticProviderRequest) -> SemanticProviderProposal:
        self.requests.append(request)
        return self.proposal


def _flatten_coverage(
    inspection: SemanticInspection,
    *,
    source: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    coverage = inspection.source_coverage if source else inspection.reply_coverage
    categories = tuple(
        dict.fromkeys(category for item in coverage for category in item.unresolved_categories)
    )
    unresolved = tuple(dict.fromkeys(span for item in coverage for span in item.unresolved_spans))
    return categories, unresolved, all(item.complete for item in coverage) if coverage else True


def _inspection_summary(inspection: SemanticInspection) -> dict[str, Any]:
    source_categories, source_unresolved, source_complete = _flatten_coverage(
        inspection, source=True
    )
    reply_categories, reply_unresolved, reply_complete = _flatten_coverage(inspection, source=False)
    return {
        "source_event_types": tuple(event.event_type for event in inspection.source_events),
        "source_event_affected_types": tuple(
            event.affected_entity.entity_type
            for event in inspection.source_events
            if event.affected_entity is not None
        ),
        "source_event_assertion_states": tuple(
            event.assertion_state for event in inspection.source_events
        ),
        "source_events": tuple(
            {
                "clause_id": event.clause_id,
                "event_id": event.event_id,
                "predicate": event.predicate,
                "event_type": event.event_type,
                "affected_span": (
                    event.affected_entity.text_span if event.affected_entity is not None else None
                ),
                "affected_type": (
                    event.affected_entity.entity_type if event.affected_entity is not None else None
                ),
                "assertion_state": event.assertion_state,
                "explicit_outcome": event.explicit_outcome,
                "evidence_spans": event.evidence_spans,
            }
            for event in inspection.source_events
        ),
        "source_unresolved_categories": source_categories,
        "reply_unresolved_categories": reply_categories,
        "source_unresolved_spans": source_unresolved,
        "reply_unresolved_spans": reply_unresolved,
        "source_unresolved_items": tuple(
            item.model_dump(mode="json")
            for coverage in inspection.source_coverage
            for item in coverage.unresolved_items
        ),
        "reply_unresolved_items": tuple(
            item.model_dump(mode="json")
            for coverage in inspection.reply_coverage
            for item in coverage.unresolved_items
        ),
        "reply_communicative_function": inspection.reply_communicative_function,
        "reply_communicative_function_confidence": (
            inspection.reply_communicative_function_confidence
        ),
        "source_coverage_complete": source_complete,
        "reply_coverage_complete": reply_complete,
        "extraction_conflicts": inspection.extraction_conflicts,
        "source_evidence_spans": tuple(
            dict.fromkeys(
                span for event in inspection.source_events for span in event.evidence_spans
            )
        ),
        "reply_evidence_spans": tuple(
            dict.fromkeys(
                (
                    *(
                        span
                        for relation in inspection.evaluations
                        for span in relation.evidence_spans
                    ),
                    *(span for shell in inspection.reply_shells for span in shell.evidence_spans),
                )
            )
        ),
    }


def _canonical_decision(
    inspection: SemanticInspection,
    case: HoldoutCase,
) -> dict[str, Any]:
    harm = adjudicate_human_harm(inspection)
    irony = adjudicate_evaluative_irony(inspection)
    intent = adjudicate_humor_intent(
        inspection,
        metadata_says_humor=case.metadata_says_humor,
        reaction_type=case.reaction_type,
        template_humor=case.template_humor,
    )
    safety = adjudicate_humor_safety(inspection)
    shell = adjudicate_reply_shell(inspection)
    intent_codes = tuple(dict.fromkeys(intent.issue_codes))
    raw_safety_codes = tuple(dict.fromkeys(safety.issue_codes))
    candidate_codes = tuple(
        dict.fromkeys((*intent_codes, *(raw_safety_codes if intent.humor_safety_required else ())))
    )
    blocking = {"HUMOR_ON_TRAGEDY", "HUMOR_INTENT_UNCERTAIN", "HUMOR_INTENT_CONFLICT"}
    return {
        "harm_referent": harm.harm_referent,
        "harm_state": harm.harm_state,
        "literal_harm": harm.literal_harm,
        "implicit_sarcasm": irony.implicit_sarcasm,
        "text_appears_humorous": intent.text_appears_humorous,
        "humor_safety_required": intent.humor_safety_required,
        "sensitive_context": safety.sensitive_context,
        "suitable_for_humor": safety.suitable_for_humor,
        "shell_operator": shell.operator,
        "humor_intent_issue_codes": intent_codes,
        "raw_humor_safety_issue_codes": raw_safety_codes,
        "canonical_candidate_issue_codes": candidate_codes,
        "semantic_candidate_eligible": not bool(blocking & set(candidate_codes)),
    }


async def _run_case(
    case: HoldoutCase,
) -> tuple[SemanticInspectionResult, tuple[SemanticProviderRequest, ...]]:
    if case.mode == "provider_assisted":
        if case.provider_proposal is None:
            raise ValueError("provider_assisted case requires provider_proposal")
        provider = _ProposalProvider(case.provider_proposal)
        result = await SemanticExtractionCoordinator(provider).inspect(
            case.source,
            case.reply,
            allow_live=True,
        )
        return result, tuple(provider.requests)
    coordinator = SemanticExtractionCoordinator()
    if case.mode == "offline_unresolved":
        return await coordinator.inspect(case.source, case.reply, allow_live=False), ()
    if case.provider_proposal is not None:
        raise ValueError("provider_proposal is allowed only in provider_assisted mode")
    return coordinator.inspect_local(case.source, case.reply), ()


async def _observe(case: HoldoutCase) -> dict[str, Any]:
    result, provider_requests = await _run_case(case)
    canonical_summary = _inspection_summary(result.canonical)
    decision = _canonical_decision(result.canonical, case)
    return {
        "mode": case.mode,
        "provider_request_count": len(provider_requests),
        "provider_requests": tuple(
            {
                "request_id": request.request_id,
                "unresolved_item_ids": tuple(item.item_id for item in request.unresolved_items),
                "source_clauses": request.source_clauses,
                "reply_clauses": request.reply_clauses,
            }
            for request in provider_requests
        ),
        "deterministic_inspection": _inspection_summary(result.deterministic),
        "escalation_decision": result.escalation.model_dump(mode="json"),
        "provider_proposal": (
            result.provider_proposal.model_dump(mode="json")
            if result.provider_proposal is not None
            else None
        ),
        "provider_used": result.provider_used,
        "provider_validation_errors": result.provider_validation_errors,
        "canonical_inspection": canonical_summary,
        "unresolved_after_reconciliation": result.unresolved_after_reconciliation,
        **canonical_summary,
        **decision,
    }


def _compare(case: HoldoutCase, observed: dict[str, Any]) -> tuple[str, ...]:
    expected = case.expect
    failures: list[str] = []
    scalar_fields = (
        "harm_referent",
        "harm_state",
        "literal_harm",
        "implicit_sarcasm",
        "text_appears_humorous",
        "humor_safety_required",
        "sensitive_context",
        "suitable_for_humor",
        "shell_operator",
        "semantic_candidate_eligible",
        "provider_used",
    )
    for field in scalar_fields:
        expected_value = getattr(expected, field)
        if expected_value is not None and observed[field] != expected_value:
            failures.append(f"{field}: expected {expected_value!r}, observed {observed[field]!r}")
    if (
        expected.escalation_required is not None
        and observed["escalation_decision"]["required"] != expected.escalation_required
    ):
        failures.append(
            "escalation_required: expected "
            f"{expected.escalation_required!r}, observed "
            f"{observed['escalation_decision']['required']!r}"
        )
    if (
        expected.provider_validation_error_count is not None
        and len(observed["provider_validation_errors"]) != expected.provider_validation_error_count
    ):
        failures.append(
            "provider_validation_error_count: expected "
            f"{expected.provider_validation_error_count}, observed "
            f"{len(observed['provider_validation_errors'])}"
        )

    subset_checks = (
        (
            expected.required_material_categories,
            observed["escalation_decision"]["material_categories"],
            "material_categories",
        ),
        (
            expected.required_canonical_candidate_issue_codes,
            observed["canonical_candidate_issue_codes"],
            "canonical_candidate_issue_codes",
        ),
        (
            expected.required_provider_validation_errors,
            observed["provider_validation_errors"],
            "provider_validation_errors",
        ),
        (
            expected.required_source_event_types,
            observed["source_event_types"],
            "source_event_types",
        ),
        (
            expected.required_source_event_affected_types,
            observed["source_event_affected_types"],
            "source_event_affected_types",
        ),
        (
            expected.required_source_event_assertion_states,
            observed["source_event_assertion_states"],
            "source_event_assertion_states",
        ),
        (
            expected.required_source_event_evidence_spans,
            observed["source_evidence_spans"],
            "source_event_evidence_spans",
        ),
        (
            expected.required_source_unresolved_categories,
            observed["source_unresolved_categories"],
            "source_unresolved_categories",
        ),
        (
            expected.required_source_unresolved_spans,
            observed["source_unresolved_spans"],
            "source_unresolved_spans",
        ),
        (
            expected.required_reply_unresolved_categories,
            observed["reply_unresolved_categories"],
            "reply_unresolved_categories",
        ),
        (
            expected.required_unresolved_after_reconciliation,
            observed["unresolved_after_reconciliation"],
            "unresolved_after_reconciliation",
        ),
    )
    for required_values, actual_values, label in subset_checks:
        missing = sorted(set(required_values) - set(actual_values))
        if missing:
            failures.append(f"{label}: missing required values {missing!r}")
    forbidden = set(expected.forbidden_canonical_candidate_issue_codes) & set(
        observed["canonical_candidate_issue_codes"]
    )
    if forbidden:
        failures.append(
            f"canonical_candidate_issue_codes: contained forbidden values {sorted(forbidden)!r}"
        )
    return tuple(failures)


def _current_unresolved_items(source: str, reply: str) -> tuple[dict[str, Any], ...]:
    local = SemanticExtractionCoordinator().inspect_local(source, reply)
    return tuple(
        item.model_dump(mode="json")
        for coverage in (*local.deterministic.source_coverage, *local.deterministic.reply_coverage)
        for item in coverage.unresolved_items
    )


def migrate_legacy_document(raw: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Migrate schema 2 provider resolutions to request-local item identities explicitly."""

    if raw.get("schema_version") == "4.0":
        return raw, ()
    if raw.get("schema_version") != "2.0":
        raise ValueError("unsupported holdout schema version")
    migrated = json.loads(json.dumps(raw))
    migrations: list[str] = []
    migrated["schema_version"] = "4.0"
    for case in migrated.get("cases", []):
        proposal = case.get("provider_proposal")
        if not proposal:
            continue
        current = _current_unresolved_items(case.get("source", ""), case.get("reply", ""))
        for group_name in ("coverage_resolutions", "remaining_unresolved_items"):
            for resolution in proposal.get(group_name, []):
                if resolution.get("item_id") and resolution.get("clause_id"):
                    continue
                candidates = [
                    item
                    for item in current
                    if item["side"] == resolution.get("side")
                    and item["unresolved_span"] == resolution.get("unresolved_span")
                ]
                exact_category = [
                    item for item in candidates if item["category"] == resolution.get("category")
                ]
                chosen = (
                    exact_category[0]
                    if len(exact_category) == 1
                    else (candidates[0] if len(candidates) == 1 else None)
                )
                if chosen is None:
                    migrations.append(
                        f"{case.get('id')}:unmapped:{resolution.get('unresolved_span')}"
                    )
                    continue
                old_category = resolution.get("category")
                resolution.update(
                    {
                        "item_id": chosen["item_id"],
                        "clause_id": chosen["clause_id"],
                        "clause_span": chosen["clause_span"],
                        "category": chosen["category"],
                    }
                )
                migrations.append(
                    f"{case.get('id')}:{chosen['item_id']}:{old_category}->{chosen['category']}"
                )
    return migrated, tuple(migrations)


async def run_holdout(document: HoldoutDocument) -> dict[str, Any]:
    """Evaluate all cases and return a deterministic JSON-serializable summary."""

    results: list[CaseResult] = []
    for case in document.cases:
        observed = await _observe(case)
        failures = _compare(case, observed)
        results.append(
            CaseResult(id=case.id, passed=not failures, failures=failures, observed=observed)
        )
    passed = sum(result.passed for result in results)
    return {
        "schema_version": document.schema_version,
        "holdout_id": document.holdout_id,
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "all_passed": passed == len(results),
        "results": [result.model_dump(mode="json") for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", help="auditor-supplied holdout JSON")
    parser.add_argument("--output", type=Path, help="write result JSON to this path")
    parser.add_argument("--write-schema", type=Path, help="write the canonical input JSON schema")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.write_schema is not None:
        arguments.write_schema.parent.mkdir(parents=True, exist_ok=True)
        arguments.write_schema.write_text(
            json.dumps(HoldoutDocument.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if arguments.input is None:
            return 0
    if arguments.input is None:
        print(
            "A holdout input file is required unless only --write-schema is used", file=sys.stderr
        )
        return 2
    try:
        raw = json.loads(arguments.input.read_text(encoding="utf-8"))
        migrated, compatibility_migrations = migrate_legacy_document(raw)
        document = HoldoutDocument.model_validate(migrated)
        summary = asyncio.run(run_holdout(document))
        summary["compatibility_migrations"] = compatibility_migrations
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Invalid holdout input: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
