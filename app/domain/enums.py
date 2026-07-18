"""Authoritative dependency-free enum contracts for the application."""

from __future__ import annotations

from enum import StrEnum


class ContentSourceKind(StrEnum):
    HOME_TIMELINE = "home_timeline"
    RECENT_SEARCH = "recent_search"
    X_ACTIVITY = "x_activity"
    SELECTED_ACCOUNT = "selected_account"
    X_LIST = "x_list"
    MANUAL_URL = "manual_url"
    IMPORT_FILE = "import_file"
    EVERGREEN = "evergreen"


class TopicKind(StrEnum):
    SEARCH = "search"
    EVERGREEN = "evergreen"
    PRODUCT = "product"
    PROJECT = "project"


class IdeaStatus(StrEnum):
    COLLECTED = "collected"
    RANKED = "ranked"
    SELECTED = "selected"
    ARCHIVED = "archived"


class GenerationMode(StrEnum):
    SOURCE_POST = "source_post"
    SOURCE_THREAD = "source_thread"
    NEWS_CLAIM = "news_claim"
    TOPIC_ONLY = "topic_only"
    TREND_CONTEXT = "trend_context"
    USER_IDEA = "user_idea"
    REWRITE_EXISTING = "rewrite_existing"
    REPLY = "reply"
    QUOTE_POST = "quote_post"
    LONG_POST = "long_post"
    THREAD = "thread"


class PostLengthMode(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONGREAD = "longread"
    RANDOM = "random"


class ContentType(StrEnum):
    SHORT_POST = "short_post"
    LONG_POST = "long_post"
    THREAD = "thread"
    REPLY = "reply"
    QUOTE_COMMENTARY = "quote_commentary"
    VIDEO_POST = "video_post"


class MediaType(StrEnum):
    NONE = "none"
    IMAGE = "image"
    VIDEO = "video"


class DraftStatus(StrEnum):
    GENERATED = "generated"
    NEEDS_REVIEW = "needs_review"
    REGENERATING = "regenerating"
    APPROVED = "approved"
    PUBLISH_PENDING_CONFIRMATION = "publish_pending_confirmation"
    PUBLISHED = "published"
    REJECTED = "rejected"
    DELETED = "deleted"
    PUBLISH_FAILED = "publish_failed"
    REMOTE_DELETED = "remote_deleted"


class FactCheckStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    VERIFIED = "verified"
    FAILED = "failed"


class ApprovalAction(StrEnum):
    APPROVED = "approved"
    REVOKED = "revoked"


class PublicationKind(StrEnum):
    POST = "post"
    THREAD_ITEM = "thread_item"
    REPLY = "reply"


class PublicationStatus(StrEnum):
    CLAIMED = "claimed"
    IN_FLIGHT = "in_flight"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    REMOTE_DELETED = "remote_deleted"


class ReplyCandidateStatus(StrEnum):
    DISCOVERED = "discovered"
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    DRAFT_CREATED = "draft_created"
    OPTED_OUT = "opted_out"


class VideoScriptStatus(StrEnum):
    GENERATED = "generated"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    GENERATION_REQUESTED = "generation_requested"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class VideoJobStatus(StrEnum):
    REQUESTED = "requested"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoGeneratorMode(StrEnum):
    DISABLED = "disabled"
    API = "api"
    PLUGIN_MANUAL = "plugin_manual"


class LLMRunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"


class ApiDirection(StrEnum):
    READ = "read"
    WRITE = "write"


__all__ = [
    "ApiDirection",
    "ApprovalAction",
    "ContentSourceKind",
    "ContentType",
    "DraftStatus",
    "FactCheckStatus",
    "GenerationMode",
    "IdeaStatus",
    "LLMRunStatus",
    "MediaType",
    "PublicationKind",
    "PublicationStatus",
    "ReplyCandidateStatus",
    "TopicKind",
    "VideoGeneratorMode",
    "VideoJobStatus",
    "VideoScriptStatus",
]
