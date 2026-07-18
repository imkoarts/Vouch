"""Reproducible, provider-independent idea ranking."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class RankingWeights:
    source_priority: float = 0.35
    recency_score: float = 0.20
    engagement_velocity: float = 0.15
    topic_relevance: float = 0.15
    novelty_score: float = 0.10
    media_potential: float = 0.05
    plagiarism_risk: float = 1.0
    safety_risk: float = 1.0
    engagement_age_floor_hours: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RankingWeights:
        known = {
            field: float(value.get(field, default))
            for field, default in {
                "source_priority": 0.35,
                "recency_score": 0.20,
                "engagement_velocity": 0.15,
                "topic_relevance": 0.15,
                "novelty_score": 0.10,
                "media_potential": 0.05,
                "plagiarism_risk": 1.0,
                "safety_risk": 1.0,
                "engagement_age_floor_hours": 1.0,
            }.items()
        }
        if any(number < 0 or not math.isfinite(number) for number in known.values()):
            raise ValueError("Ranking weights must be finite and non-negative")
        return cls(**known)


@dataclass(frozen=True, slots=True)
class RankingResult:
    score: float
    components: dict[str, float]


class IdeaRanker:
    """Compute the prompt's score without an LLM, so results are reproducible."""

    def __init__(self, weights: RankingWeights | None = None) -> None:
        self.weights = weights or RankingWeights()

    def rank(
        self,
        *,
        source_weight: float,
        published_at: datetime | None,
        public_metrics: Mapping[str, Any],
        topic_relevance: float = 0.5,
        novelty_score: float = 0.7,
        media_potential: float = 0.0,
        plagiarism_risk: float = 0.0,
        safety_risk: float = 0.0,
        now: datetime | None = None,
    ) -> RankingResult:
        now = now or datetime.now(UTC)
        if published_at is None:
            age_hours = 168.0
        else:
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=UTC)
            age_hours = max(0.0, (now - published_at).total_seconds() / 3600)
        recency = math.exp(-age_hours / 72.0)
        metrics = {
            name: max(0, int(public_metrics.get(name, 0) or 0))
            for name in ("like_count", "retweet_count", "reply_count", "quote_count")
        }
        engagement = (
            metrics["like_count"]
            + 2 * metrics["retweet_count"]
            + metrics["reply_count"]
            + metrics["quote_count"]
        )
        velocity = min(
            1.0,
            math.log1p(engagement / max(age_hours, self.weights.engagement_age_floor_hours)) / 5.0,
        )
        inputs = {
            "source_priority": min(1.0, max(0.0, source_weight)),
            "recency_score": recency,
            "engagement_velocity": velocity,
            "topic_relevance": min(1.0, max(0.0, topic_relevance)),
            "novelty_score": min(1.0, max(0.0, novelty_score)),
            "media_potential": min(1.0, max(0.0, media_potential)),
            "plagiarism_risk": min(1.0, max(0.0, plagiarism_risk)),
            "safety_risk": min(1.0, max(0.0, safety_risk)),
        }
        score = (
            inputs["source_priority"] * self.weights.source_priority
            + inputs["recency_score"] * self.weights.recency_score
            + inputs["engagement_velocity"] * self.weights.engagement_velocity
            + inputs["topic_relevance"] * self.weights.topic_relevance
            + inputs["novelty_score"] * self.weights.novelty_score
            + inputs["media_potential"] * self.weights.media_potential
            - inputs["plagiarism_risk"] * self.weights.plagiarism_risk
            - inputs["safety_risk"] * self.weights.safety_risk
        )
        return RankingResult(score=max(0.0, min(1.0, score)), components=inputs)
