"""Source ingestion, manual imports, and deterministic idea creation."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy.orm import Session

from app.models import Idea
from app.models.enums import ContentSourceKind, ContentType
from app.repositories.content import ContentRepository
from app.schemas.configuration import ContentSourcesConfiguration
from app.services.idea_ranker import IdeaRanker, RankingWeights

_X_POST_URL = re.compile(
    r"^https?://(?:www\.)?(?:x|twitter)\.com/(?:[^/]+/status|i/web/status)/(\d+)(?:[/?#].*)?$",
    re.IGNORECASE,
)


class XReadPort(Protocol):
    """Minimal direct-post reader used by manual URL imports."""

    async def get_post(self, post_id: str) -> Mapping[str, Any]: ...


class IdeaCollector:
    def __init__(
        self,
        session: Session,
        *,
        ranker: IdeaRanker | None = None,
        source_configuration: ContentSourcesConfiguration,
        similarity_threshold: float = 0.92,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between zero and one")
        self.session = session
        self.repository = ContentRepository(session)
        # Reserved for idea-level similarity checks. Raw source rows are never
        # deduplicated by text similarity or conversation ID.
        self.similarity_threshold = similarity_threshold
        self.source_configuration = source_configuration
        self.ranker = ranker or IdeaRanker(
            RankingWeights.from_mapping(self.source_configuration.ranking.model_dump())
        )

    def ingest(
        self,
        posts: Iterable[Mapping[str, Any]],
        *,
        source_kind: ContentSourceKind,
        source_name: str,
        source_identifier: str = "",
        persistence_kind: ContentSourceKind | None = None,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> list[str]:
        configured_source = self.source_configuration.source_for_kind(source_kind)
        if not configured_source.enabled:
            return []
        source = self.repository.get_or_create_source(
            # Some deployed databases predate a newly introduced domain source kind.
            # The explicit adapter keeps acquisition policy authoritative while the
            # metadata preserves provenance until the persistence enum is retired.
            kind=persistence_kind or source_kind,
            name=source_name,
            identifier=source_identifier,
            weight=configured_source.weight,
            configuration=dict(source_metadata or {}),
        )
        idea_ids: list[str] = []
        for payload in posts:
            source_post, _ = self.repository.import_source_post(
                payload=payload,
                source=source,
            )
            effective_priority = self.repository.effective_source_priority(source_post)
            result = self.ranker.rank(
                source_weight=effective_priority,
                published_at=source_post.published_at,
                public_metrics=source_post.public_metrics,
                media_potential=1.0 if source_post.media_metadata else 0.0,
            )
            title = source_post.text.splitlines()[0][:120]
            explanation: dict[str, Any] = {
                "why_selected": "deterministic source, recency, engagement and novelty score",
                "source": source_kind.value,
                "why_relevant": "ranked from the configured content source",
                "differentiation": "add independent analysis; do not paraphrase the source",
                "facts_to_verify": [],
                "recommended_media": (
                    "video"
                    if any(
                        str(item.get("type", "")).casefold() in {"video", "animated_gif"}
                        for item in source_post.media_metadata.get("items", [])
                        if isinstance(item, Mapping)
                    )
                    else "image"
                    if source_post.media_metadata
                    else "text"
                ),
            }
            idea, _ = self.repository.create_idea(
                source_post=source_post,
                title=title or "Content idea",
                summary=source_post.text[:500],
                score=result.score,
                components=result.components,
                explanation=explanation,
                recommended_format=ContentType.SHORT_POST,
                source_priority=effective_priority,
            )
            idea_ids.append(idea.id)
        return list(dict.fromkeys(idea_ids))

    def collect_mock(self) -> list[str]:
        """Create synthetic local data while preserving the real ranking priorities."""

        now = datetime.now(UTC)
        batches: tuple[tuple[ContentSourceKind, str, list[dict[str, Any]]], ...] = (
            (
                ContentSourceKind.HOME_TIMELINE,
                "Home timeline",
                [
                    {
                        "id": "900000000000000001",
                        "text": (
                            "A small team documented one repeatable release practice "
                            "and reduced handoff friction."
                        ),
                        "author_id": "700000000000000001",
                        "username": "synthetic_builder",
                        "lang": "en",
                        "created_at": (now - timedelta(hours=1)).isoformat(),
                        "conversation_id": "900000000000000001",
                        "public_metrics": {
                            "like_count": 18,
                            "retweet_count": 3,
                            "reply_count": 2,
                        },
                    }
                ],
            ),
            (
                ContentSourceKind.SELECTED_ACCOUNT,
                "Selected account",
                [
                    {
                        "id": "900000000000000002",
                        "text": (
                            "Synthetic selected-account note: useful automation begins with "
                            "a visible manual checkpoint."
                        ),
                        "author_id": "700000000000000002",
                        "username": "synthetic_researcher",
                        "lang": "en",
                        "created_at": (now - timedelta(hours=2)).isoformat(),
                        "conversation_id": "900000000000000002",
                        "public_metrics": {
                            "like_count": 12,
                            "retweet_count": 1,
                            "reply_count": 1,
                        },
                    }
                ],
            ),
            (
                ContentSourceKind.EVERGREEN,
                "Evergreen",
                [
                    {
                        "id": None,
                        "text": (
                            "Explain a reusable lesson from a current project without "
                            "exposing private details."
                        ),
                        "lang": "en",
                        "created_at": (now - timedelta(days=7)).isoformat(),
                        "conversation_id": None,
                        "public_metrics": {},
                    }
                ],
            ),
        )
        ids: list[str] = []
        for kind, name, posts in batches:
            ids.extend(
                self.ingest(
                    posts,
                    source_kind=kind,
                    source_name=name,
                )
            )
        unique_ids = list(dict.fromkeys(ids))
        for idea_id in unique_ids:
            idea = self.session.get(Idea, idea_id)
            if idea is None:
                continue
            idea.explanation = {**idea.explanation, "editorial_intent": "report_event"}
        self.session.flush()
        return unique_ids

    async def import_url(self, url: str, client: XReadPort) -> list[str]:
        match = _X_POST_URL.fullmatch(url.strip())
        if match is None:
            raise ValueError("Expected an x.com or twitter.com post URL with a numeric ID")
        payload = dict(await client.get_post(match.group(1)))
        payload.setdefault("url", url)
        idea_ids = self.ingest(
            [payload],
            source_kind=ContentSourceKind.MANUAL_URL,
            source_name="Manual X URL",
            source_identifier=match.group(1),
        )
        for idea_id in idea_ids:
            idea = self.session.get(Idea, idea_id)
            if idea is None:
                continue
            idea.recommended_format = ContentType.QUOTE_COMMENTARY
            idea.explanation = {
                **idea.explanation,
                "origin": "manual_url",
                "editorial_intent": "quote_reaction",
                "operator_requested_generation": True,
                "why_selected": "The operator supplied this exact X post URL.",
                "why_relevant": "Generate a reviewable reaction to the supplied post.",
                "differentiation": (
                    "Add one concise source-specific reaction without inventing facts or "
                    "summarizing the quoted post."
                ),
            }
        self.session.flush()
        return idea_ids

    def import_file(self, path: Path) -> list[str]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            rows: list[Any] = [
                {"text": line.strip()}
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            rows = value if isinstance(value, list) else [value]
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        else:
            raise ValueError("Only TXT, JSON and CSV imports are supported")
        posts = [row for row in rows if isinstance(row, Mapping)]
        return self.ingest(
            posts,
            source_kind=ContentSourceKind.IMPORT_FILE,
            source_name=f"Import {path.name}",
            source_identifier=str(path.resolve()),
        )
