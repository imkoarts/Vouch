"""Deterministic, offline generation used by tests and the local mock MVP."""

from __future__ import annotations

import re

from app.schemas.content import (
    ContentFormat,
    GenerationRequest,
    GenerationResult,
    GenerationVariant,
    MediaPlan,
)


def _fit_mock_text(base: str, minimum: int | None, maximum: int | None) -> str:
    """Apply only the hard maximum; never pad mock text with unrelated filler."""

    del minimum
    if maximum is None or len(base) <= maximum:
        return base
    rendered = base[:maximum].rstrip(" ,;:-")
    if rendered and rendered[-1] not in ".!?":
        cut = max(rendered.rfind("."), rendered.rfind("!"), rendered.rfind("?"))
        if cut >= 0:
            rendered = rendered[: cut + 1]
    return rendered


def _evidence_keyword(request: GenerationRequest) -> tuple[str, bool]:
    external = [item for item in request.evidence_packet.items if item.source_type != "user_input"]
    selected = external[0] if external else request.evidence_packet.items[0]
    words = re.findall(r"[^\W_]{4,}", selected.text, flags=re.UNICODE)
    ignored = {
        "about",
        "after",
        "before",
        "could",
        "should",
        "their",
        "there",
        "these",
        "only",
        "time",
        "with",
        "from",
        "this",
        "that",
    }
    keyword = next((word for word in words if word.casefold() not in ignored), "review")
    return keyword, bool(external)


def _base_text(
    request: GenerationRequest,
    *,
    index: int,
    keyword: str,
    contribution_type: str | None = None,
) -> str:
    external = [item for item in request.evidence_packet.items if item.source_type != "user_input"]
    if request.content_type is ContentFormat.THREAD and request.editorial_intent == "explain_topic":
        thread_openers = (
            "Record the evidence before anyone proposes a public post.",
            "Treat publication as the second step, not the default.",
            "The workflow starts with evidence, not a finished post.",
        )
        return thread_openers[index % len(thread_openers)]
    if request.editorial_intent in {"report_event", "rewrite_existing"} and external:
        # Exercise the valid direct-update control while preserving attribution for a
        # single-source news claim. The mock must obey the same evidence contract as a live
        # provider rather than relying on a test-only quality bypass.
        selected = external[min(index, len(external) - 1)]
        statement = selected.text.strip()
        if request.content_mode is not None and request.content_mode.attribution_required:
            source_name = selected.author_or_source or "the cited source"
            prefix = (
                f"According to @{source_name}, "
                if selected.author_or_source
                else f"According to {source_name}, "
            )
            if statement:
                statement = statement[0].lower() + statement[1:]
            return prefix + statement
        return statement
    if request.editorial_intent in {"reply_reaction", "quote_reaction"} and external:
        source_text = external[0].text.casefold()
        if request.editorial_intent == "reply_reaction":
            if "manual review" in source_text:
                replies = {
                    "plain_observation": "manual review is enough here; a public reply can wait",
                    "direct_response": (
                        "agreed, this belongs in manual review before any public reply"
                    ),
                    "specific_qualification": (
                        "manual review makes sense, especially before a public reply"
                    ),
                    "genuine_question": "does this need anything beyond manual review right now?",
                    "dry_humor": "manual review has the sensible job for once",
                    "contextual_extension": (
                        "manual review keeps the public response separate from the event"
                    ),
                }
                return replies.get(
                    contribution_type or "",
                    "manual review is enough here; a public reply can wait",
                )
            replies = {
                "plain_observation": f"the {keyword} detail is the part worth answering directly",
                "direct_response": (
                    f"the {keyword} point makes sense, but the constraint still matters"
                ),
                "specific_qualification": (
                    f"mostly yes, though the {keyword} detail changes the answer"
                ),
                "genuine_question": f"what does the {keyword} detail change in practice?",
                "dry_humor": f"the {keyword} detail is already doing enough work here",
                "contextual_extension": f"the {keyword} constraint is where this becomes practical",
            }
            return replies.get(
                contribution_type or "",
                f"the {keyword} detail is the part worth answering directly",
            )
        if "arrogant" in source_text:
            replies = {
                "plain_observation": 'one clip is not much evidence for "the only time."',
                "direct_response": 'one moment is a thin sample for "the only time."',
                "specific_qualification": '"the only time" needs more than one moment.',
                "genuine_question": 'does one moment really support "the only time"?',
                "dry_humor": '"the only time" is doing heroic amounts of work here.',
                "contextual_extension": (
                    "the caption makes a bigger claim than the moment can carry"
                ),
            }
            return replies.get(
                contribution_type or "",
                '"the only time" is doing heroic amounts of work here.',
            )
        if "synthetic manually imported post" in source_text:
            quotes = (
                "a synthetic post is a sensible boundary for testing manual generation",
                "manual generation is easier to trust when the source stays visible beside it",
                "the synthetic source can do the setup; the human still gets the final say",
            )
        else:
            quotes = (
                f"the {keyword} detail is the part worth responding to directly",
                f"the source already makes the {keyword} point clearly enough",
                f"the {keyword} detail is doing most of the work here",
            )
        return quotes[index % len(quotes)]
    topic = request.idea_summary.strip().rstrip(".!?")
    variants = (
        (
            f"For {topic}, start with the concrete decision, state the reason plainly, "
            "and keep the next step small enough to review."
        ),
        (
            f"For {topic}, write down the constraint first. Then make one observable "
            "change and compare the result with the starting point."
        ),
        (
            f"Treat {topic} as a bounded workflow: choose one action, record the outcome, "
            "and let a human decide what follows."
        ),
    )
    return variants[index % len(variants)]


class MockLLMProvider:
    """Create one to three deterministic variants without network access."""

    name = "mock"
    model = "mock-structured-v4"

    async def generate_structured(self, request: GenerationRequest) -> GenerationResult:
        keyword, has_external_evidence = _evidence_keyword(request)
        external = [
            item for item in request.evidence_packet.items if item.source_type != "user_input"
        ]
        labels = ("direct observation", "analytical", "conversational")
        source_ids = tuple(item.evidence_id for item in request.evidence_packet.items)
        variants: list[GenerationVariant] = []
        for index, (label, angle) in enumerate(
            zip(labels[: len(request.angle_candidates)], request.angle_candidates, strict=True)
        ):
            parts: tuple[str, ...] = ()
            text = _base_text(
                request,
                index=index,
                keyword=keyword,
                contribution_type=angle.contribution_type,
            )
            rendered = _fit_mock_text(text, request.minimum_characters, request.maximum_characters)
            if (
                request.minimum_characters is not None
                and len(rendered) < request.minimum_characters
            ):
                return GenerationResult(
                    status="insufficient_context",
                    reason_code="MOCK_HARD_MINIMUM_UNSUPPORTED",
                    idea_summary=request.idea_summary,
                    recommendation_reason=(
                        "Mock evidence cannot satisfy the explicit minimum without "
                        "unrelated filler."
                    ),
                    media_plan=MediaPlan(),
                )
            if request.content_type is ContentFormat.THREAD:
                if request.editorial_intent == "explain_topic":
                    thread_follow_ups = (
                        "Keep publication behind an exact-draft human approval.",
                        "Nothing goes live until a human approves the exact draft.",
                        "Only the reviewed draft moves on to publication approval.",
                    )
                    follow_up = thread_follow_ups[index % len(thread_follow_ups)]
                else:
                    follow_up = (
                        "Record the result, then return it to a human before the next action."
                    )
                parts = (rendered, follow_up)
                rendered = "\n\n".join(parts)
            variants.append(
                GenerationVariant(
                    label=label,
                    text=rendered,
                    parts=parts,
                    hook="",
                    cta="",
                    character_count=len(rendered),
                    tone=(label,),
                    claims=(),
                    source_post_ids=source_ids,
                    similarity_risk=0.0,
                    fact_check_required=False,
                    angle_id=angle.angle_id,
                    angle_type=angle.angle_type,
                    evidence_ids=angle.evidence_ids,
                    confidence=angle.confidence,
                    factual_claims=(
                        (external[0].text,)
                        if has_external_evidence
                        and request.editorial_intent in {"report_event", "rewrite_existing"}
                        and not (
                            request.content_mode is not None
                            and request.content_mode.attribution_required
                        )
                        else ()
                    ),
                    attributed_claims=(
                        (external[0].text,)
                        if has_external_evidence
                        and request.editorial_intent == "report_event"
                        and request.content_mode is not None
                        and request.content_mode.attribution_required
                        else ()
                    ),
                    uncertainty_markers=(),
                    revision_status=("revised" if request.revision_targets else "not_applicable"),
                    contribution_id=angle.angle_id,
                    contribution_type=angle.contribution_type,
                )
            )
        return GenerationResult(
            idea_summary=request.idea_summary,
            recommended_format=request.content_type,
            variants=tuple(variants),
            recommended_variant=0,
            recommendation_reason="Evidence-bound synthetic candidates for human selection",
            media_plan=MediaPlan(),
        )
