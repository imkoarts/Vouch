"""Run an auditor-supplied semantic holdout against a frozen release candidate.

The input JSON is validated strictly. This runner never performs network access and does not
contain development or certification examples. Exit status is 0 when every case passes, 1 when
one or more expectations fail, and 2 for invalid input or runner errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas.content import SemanticInspection  # noqa: E402
from app.services.semantic_adjudication import (  # noqa: E402
    adjudicate_evaluative_irony,
    adjudicate_human_harm,
    adjudicate_humor_intent,
    adjudicate_humor_safety,
    adjudicate_reply_shell,
)
from app.services.semantic_extraction import extract_semantic_inspection  # noqa: E402
from app.services.semantic_reconciliation import reconcile_semantic_inspections  # noqa: E402


class HoldoutExpectation(BaseModel):
    """Optional expectations; omitted fields are intentionally not asserted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    harm_referent: Literal["human", "non_human", "unknown"] | None = None
    harm_state: str | None = None
    literal_harm: bool | None = None
    implicit_sarcasm: bool | None = None
    irony_confidence: Literal["high", "medium", "low"] | None = None
    text_appears_humorous: bool | None = None
    humor_safety_required: bool | None = None
    sensitive_context: bool | None = None
    suitable_for_humor: bool | None = None
    shell_operator: str | None = None
    source_coverage_complete: bool | None = None
    reply_coverage_complete: bool | None = None
    semantic_candidate_eligible: bool | None = None
    required_issue_codes: tuple[str, ...] = ()
    forbidden_issue_codes: tuple[str, ...] = ()
    required_humor_intent_issue_codes: tuple[str, ...] = ()
    required_raw_humor_safety_issue_codes: tuple[str, ...] = ()
    required_canonical_candidate_issue_codes: tuple[str, ...] = ()
    forbidden_canonical_candidate_issue_codes: tuple[str, ...] = ()
    required_source_event_types: tuple[str, ...] = ()
    required_source_unresolved_categories: tuple[str, ...] = ()
    required_reply_unresolved_categories: tuple[str, ...] = ()
    required_source_evidence_spans: tuple[str, ...] = ()
    required_reply_evidence_spans: tuple[str, ...] = ()


class HoldoutCase(BaseModel):
    """One independent semantic holdout case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    source: str
    reply: str = ""
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
    provider_proposal: SemanticInspection | None = None
    expect: HoldoutExpectation


class HoldoutDocument(BaseModel):
    """Versioned external-holdout payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    holdout_id: str = Field(min_length=1)
    cases: tuple[HoldoutCase, ...] = Field(min_length=1)


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    passed: bool
    failures: tuple[str, ...]
    observed: dict[str, object]


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
    complete = bool(coverage) and all(item.complete for item in coverage)
    if not coverage:
        complete = True
    return categories, unresolved, complete


def _observe(case: HoldoutCase) -> dict[str, object]:
    deterministic = extract_semantic_inspection(case.source, case.reply)
    inspection = reconcile_semantic_inspections(
        deterministic,
        case.provider_proposal,
        source_text=case.source,
        reply_text=case.reply,
    )
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
    source_categories, source_unresolved, source_complete = _flatten_coverage(
        inspection, source=True
    )
    reply_categories, reply_unresolved, reply_complete = _flatten_coverage(inspection, source=False)
    humor_intent_issue_codes = tuple(dict.fromkeys(intent.issue_codes))
    raw_humor_safety_issue_codes = tuple(dict.fromkeys(safety.issue_codes))
    canonical_candidate_issue_codes = tuple(
        dict.fromkeys(
            (
                *humor_intent_issue_codes,
                *(raw_humor_safety_issue_codes if intent.humor_safety_required else ()),
            )
        )
    )
    blocking_codes = {"HUMOR_ON_TRAGEDY", "HUMOR_INTENT_UNCERTAIN", "HUMOR_INTENT_CONFLICT"}
    semantic_candidate_eligible = not bool(blocking_codes & set(canonical_candidate_issue_codes))
    source_evidence = tuple(
        dict.fromkeys(span for event in inspection.source_events for span in event.evidence_spans)
    )
    reply_evidence = tuple(
        dict.fromkeys(
            span
            for relation in (*inspection.evaluations, *inspection.reply_shells)
            for span in relation.evidence_spans
        )
    )
    return {
        "harm_referent": harm.harm_referent,
        "harm_state": harm.harm_state,
        "literal_harm": harm.literal_harm,
        "implicit_sarcasm": irony.implicit_sarcasm,
        "irony_confidence": irony.confidence,
        "text_appears_humorous": intent.text_appears_humorous,
        "humor_safety_required": intent.humor_safety_required,
        "sensitive_context": safety.sensitive_context,
        "suitable_for_humor": safety.suitable_for_humor,
        "shell_operator": shell.operator,
        "source_coverage_complete": source_complete,
        "reply_coverage_complete": reply_complete,
        "semantic_candidate_eligible": semantic_candidate_eligible,
        "humor_intent_issue_codes": humor_intent_issue_codes,
        "raw_humor_safety_issue_codes": raw_humor_safety_issue_codes,
        "canonical_candidate_issue_codes": canonical_candidate_issue_codes,
        # Backward-compatible alias now reflects production gating rather than raw context.
        "issue_codes": canonical_candidate_issue_codes,
        "source_event_types": tuple(event.event_type for event in inspection.source_events),
        "source_unresolved_categories": source_categories,
        "reply_unresolved_categories": reply_categories,
        "source_unresolved_spans": source_unresolved,
        "reply_unresolved_spans": reply_unresolved,
        "source_evidence_spans": source_evidence,
        "reply_evidence_spans": reply_evidence,
        "extraction_conflicts": inspection.extraction_conflicts,
    }


def _compare(case: HoldoutCase, observed: dict[str, object]) -> tuple[str, ...]:
    expected = case.expect
    failures: list[str] = []
    scalar_fields = (
        "harm_referent",
        "harm_state",
        "literal_harm",
        "implicit_sarcasm",
        "irony_confidence",
        "text_appears_humorous",
        "humor_safety_required",
        "sensitive_context",
        "suitable_for_humor",
        "shell_operator",
        "source_coverage_complete",
        "reply_coverage_complete",
        "semantic_candidate_eligible",
    )
    for field in scalar_fields:
        expected_value = getattr(expected, field)
        if expected_value is not None and observed[field] != expected_value:
            failures.append(f"{field}: expected {expected_value!r}, observed {observed[field]!r}")

    subset_checks = (
        ("required_issue_codes", "issue_codes"),
        ("required_humor_intent_issue_codes", "humor_intent_issue_codes"),
        ("required_raw_humor_safety_issue_codes", "raw_humor_safety_issue_codes"),
        ("required_canonical_candidate_issue_codes", "canonical_candidate_issue_codes"),
        ("required_source_event_types", "source_event_types"),
        ("required_source_unresolved_categories", "source_unresolved_categories"),
        ("required_reply_unresolved_categories", "reply_unresolved_categories"),
        ("required_source_evidence_spans", "source_evidence_spans"),
        ("required_reply_evidence_spans", "reply_evidence_spans"),
    )
    for expected_field, observed_field in subset_checks:
        required = set(getattr(expected, expected_field))
        actual = set(observed[observed_field])
        missing = sorted(required - actual)
        if missing:
            failures.append(f"{observed_field}: missing required values {missing!r}")
    forbidden = set(expected.forbidden_issue_codes) & set(observed["issue_codes"])
    if forbidden:
        failures.append(f"issue_codes: contained forbidden values {sorted(forbidden)!r}")
    forbidden_canonical = set(expected.forbidden_canonical_candidate_issue_codes) & set(
        observed["canonical_candidate_issue_codes"]
    )
    if forbidden_canonical:
        failures.append(
            "canonical_candidate_issue_codes: contained forbidden values "
            f"{sorted(forbidden_canonical)!r}"
        )
    return tuple(failures)


def run_holdout(document: HoldoutDocument) -> dict[str, object]:
    """Evaluate all cases and return a deterministic JSON-serializable summary."""

    results: list[CaseResult] = []
    for case in document.cases:
        observed = _observe(case)
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
        document = HoldoutDocument.model_validate_json(arguments.input.read_text(encoding="utf-8"))
        summary = run_holdout(document)
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
