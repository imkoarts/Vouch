"""Create the initial content moderation schema.

Revision ID: 0001_initial
Revises: None
Create Date: 2026-07-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _enum(name: str, *values: str) -> sa.Enum:
    return sa.Enum(
        *values,
        name=name,
        native_enum=False,
        create_constraint=True,
    )


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    )


def _updated_at() -> sa.Column:
    return sa.Column(
        "updated_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    )


def upgrade() -> None:
    op.create_table(
        "content_sources",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column(
            "kind",
            _enum(
                "content_source_kind",
                "home_timeline",
                "recent_search",
                "selected_account",
                "x_list",
                "manual_url",
                "import_file",
                "evergreen",
            ),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("identifier", sa.String(length=512), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        _created_at(),
        _updated_at(),
        sa.CheckConstraint(
            "weight >= 0 AND weight <= 1", name="ck_content_sources_weight_range"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_content_sources"),
        sa.UniqueConstraint(
            "kind", "identifier", name="uq_content_sources_kind_identifier"
        ),
    )

    op.create_table(
        "topics",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column(
            "kind",
            _enum("topic_kind", "search", "evergreen", "product", "project"),
            nullable=False,
        ),
        sa.Column("query", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        _created_at(),
        _updated_at(),
        sa.PrimaryKeyConstraint("id", name="pk_topics"),
        sa.UniqueConstraint("name", name="uq_topics_name"),
    )

    op.create_table(
        "source_posts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("x_post_id", sa.String(length=32), nullable=True),
        sa.Column("content_source_id", sa.String(length=36), nullable=True),
        sa.Column("url", sa.String(length=2048), nullable=True),
        sa.Column("author_id", sa.String(length=64), nullable=True),
        sa.Column("username", sa.String(length=64), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("public_metrics", sa.JSON(), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=True),
        sa.Column("media_metadata", sa.JSON(), nullable=False),
        sa.Column("source_labels", sa.JSON(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        _created_at(),
        _updated_at(),
        sa.ForeignKeyConstraint(
            ["content_source_id"],
            ["content_sources.id"],
            name="fk_source_posts_content_source_id_content_sources",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_posts"),
        sa.UniqueConstraint("x_post_id", name="uq_source_posts_x_post_id"),
        sa.UniqueConstraint("text_hash", name="uq_source_posts_text_hash"),
    )
    op.create_index(
        "ix_source_posts_content_source_id", "source_posts", ["content_source_id"]
    )
    op.create_index("ix_source_posts_author_id", "source_posts", ["author_id"])
    op.create_index("ix_source_posts_username", "source_posts", ["username"])
    op.create_index(
        "ix_source_posts_conversation_id", "source_posts", ["conversation_id"]
    )
    op.create_index("ix_source_posts_fetched_at", "source_posts", ["fetched_at"])

    op.create_table(
        "ideas",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column(
            "status",
            _enum("idea_status", "collected", "ranked", "selected", "archived"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column(
            "recommended_format",
            _enum(
                "idea_content_type",
                "short_post",
                "long_post",
                "thread",
                "reply",
                "quote_commentary",
                "video_post",
            ),
            nullable=True,
        ),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("score_components", sa.JSON(), nullable=False),
        sa.Column("explanation", sa.JSON(), nullable=False),
        sa.Column("dedupe_key", sa.String(length=64), nullable=True),
        _created_at(),
        _updated_at(),
        sa.CheckConstraint("score >= 0", name="ck_ideas_score_nonnegative"),
        sa.PrimaryKeyConstraint("id", name="pk_ideas"),
        sa.UniqueConstraint("dedupe_key", name="uq_ideas_dedupe_key"),
    )

    op.create_table(
        "idea_sources",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("idea_id", sa.String(length=36), nullable=False),
        sa.Column("source_post_id", sa.String(length=36), nullable=True),
        sa.Column("topic_id", sa.String(length=36), nullable=True),
        sa.Column("source_priority", sa.Float(), nullable=False),
        sa.Column("contribution", sa.Float(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "((source_post_id IS NOT NULL AND topic_id IS NULL) OR "
            "(source_post_id IS NULL AND topic_id IS NOT NULL))",
            name="ck_idea_sources_exactly_one_source",
        ),
        sa.CheckConstraint(
            "source_priority >= 0 AND contribution >= 0",
            name="ck_idea_sources_nonnegative_scores",
        ),
        sa.CheckConstraint(
            "position >= 0", name="ck_idea_sources_position_nonnegative"
        ),
        sa.ForeignKeyConstraint(
            ["idea_id"],
            ["ideas.id"],
            name="fk_idea_sources_idea_id_ideas",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["source_post_id"],
            ["source_posts.id"],
            name="fk_idea_sources_source_post_id_source_posts",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["topic_id"],
            ["topics.id"],
            name="fk_idea_sources_topic_id_topics",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_idea_sources"),
        sa.UniqueConstraint(
            "idea_id", "source_post_id", name="uq_idea_sources_idea_source_post"
        ),
        sa.UniqueConstraint(
            "idea_id", "topic_id", name="uq_idea_sources_idea_topic"
        ),
    )
    op.create_index("ix_idea_sources_idea_id", "idea_sources", ["idea_id"])
    op.create_index(
        "ix_idea_sources_source_post_id", "idea_sources", ["source_post_id"]
    )
    op.create_index("ix_idea_sources_topic_id", "idea_sources", ["topic_id"])

    op.create_table(
        "reply_candidates",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_post_id", sa.String(length=36), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=True),
        sa.Column("author_id", sa.String(length=64), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=True),
        sa.Column(
            "status",
            _enum(
                "reply_candidate_status",
                "discovered",
                "eligible",
                "ineligible",
                "draft_created",
                "opted_out",
            ),
            nullable=False,
        ),
        sa.Column(
            "direct_mention", sa.Boolean(), server_default=sa.false(), nullable=False
        ),
        sa.Column(
            "reply_to_owned_post",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column(
            "allowlisted", sa.Boolean(), server_default=sa.false(), nullable=False
        ),
        sa.Column("eligibility_reason", sa.Text(), nullable=True),
        sa.Column("conversation_context", sa.JSON(), nullable=False),
        _created_at(),
        _updated_at(),
        sa.ForeignKeyConstraint(
            ["source_post_id"],
            ["source_posts.id"],
            name="fk_reply_candidates_source_post_id_source_posts",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_reply_candidates"),
        sa.UniqueConstraint(
            "source_post_id", name="uq_reply_candidates_source_post_id"
        ),
    )
    op.create_index(
        "ix_reply_candidates_conversation_id", "reply_candidates", ["conversation_id"]
    )
    op.create_index(
        "ix_reply_candidates_author_id", "reply_candidates", ["author_id"]
    )
    op.create_index(
        "ix_reply_candidates_status_created",
        "reply_candidates",
        ["status", "created_at"],
    )

    op.create_table(
        "drafts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("idea_id", sa.String(length=36), nullable=True),
        sa.Column("reply_candidate_id", sa.String(length=36), nullable=True),
        sa.Column(
            "content_type",
            _enum(
                "draft_content_type",
                "short_post",
                "long_post",
                "thread",
                "reply",
                "quote_commentary",
                "video_post",
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            _enum(
                "draft_status",
                "generated",
                "needs_review",
                "regenerating",
                "approved",
                "publish_pending_confirmation",
                "published",
                "rejected",
                "deleted",
                "publish_failed",
                "remote_deleted",
            ),
            nullable=False,
        ),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("model", sa.String(length=200), nullable=True),
        sa.Column(
            "fact_check_status",
            _enum(
                "fact_check_status", "not_required", "required", "verified", "failed"
            ),
            nullable=False,
        ),
        sa.Column("blocking_safety_flags", sa.JSON(), nullable=False),
        sa.Column("media_plan", sa.JSON(), nullable=False),
        sa.Column("artifact_path", sa.String(length=2048), nullable=False),
        sa.Column("current_content_hash", sa.String(length=64), nullable=True),
        sa.Column("current_version_number", sa.Integer(), nullable=False),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        _created_at(),
        _updated_at(),
        sa.CheckConstraint(
            "current_version_number >= 0", name="ck_drafts_version_nonnegative"
        ),
        sa.CheckConstraint("row_version >= 1", name="ck_drafts_row_version_positive"),
        sa.ForeignKeyConstraint(
            ["idea_id"],
            ["ideas.id"],
            name="fk_drafts_idea_id_ideas",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["reply_candidate_id"],
            ["reply_candidates.id"],
            name="fk_drafts_reply_candidate_id_reply_candidates",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_drafts"),
        sa.UniqueConstraint(
            "reply_candidate_id", name="uq_drafts_reply_candidate_id"
        ),
    )
    op.create_index("ix_drafts_idea_id", "drafts", ["idea_id"])
    op.create_index(
        "ix_drafts_reply_candidate_id", "drafts", ["reply_candidate_id"]
    )
    op.create_index(
        "ix_drafts_current_content_hash", "drafts", ["current_content_hash"]
    )
    op.create_index(
        "ix_drafts_status_updated_at", "drafts", ["status", "updated_at"]
    )

    op.create_table(
        "draft_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("draft_id", sa.String(length=36), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.Column("rendered_text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("origin", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("model", sa.String(length=200), nullable=True),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("critique", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        _created_at(),
        sa.CheckConstraint(
            "version_number >= 1", name="ck_draft_versions_version_positive"
        ),
        sa.ForeignKeyConstraint(
            ["draft_id"],
            ["drafts.id"],
            name="fk_draft_versions_draft_id_drafts",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_draft_versions"),
        sa.UniqueConstraint(
            "draft_id", "version_number", name="uq_draft_versions_draft_version"
        ),
    )
    op.create_index("ix_draft_versions_draft_id", "draft_versions", ["draft_id"])
    op.create_index(
        "ix_draft_versions_draft_hash",
        "draft_versions",
        ["draft_id", "content_hash"],
    )

    op.create_table(
        "draft_approvals",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("draft_id", sa.String(length=36), nullable=False),
        sa.Column("draft_version_id", sa.String(length=36), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "action",
            _enum("approval_action", "approved", "revoked"),
            nullable=False,
        ),
        sa.Column("actor", sa.String(length=200), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("supersedes_id", sa.String(length=36), nullable=True),
        _created_at(),
        sa.CheckConstraint(
            "((action = 'approved' AND supersedes_id IS NULL) OR "
            "(action = 'revoked' AND supersedes_id IS NOT NULL))",
            name="ck_draft_approvals_action_reference",
        ),
        sa.ForeignKeyConstraint(
            ["draft_id"],
            ["drafts.id"],
            name="fk_draft_approvals_draft_id_drafts",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["draft_version_id"],
            ["draft_versions.id"],
            name="fk_draft_approvals_draft_version_id_draft_versions",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_id"],
            ["draft_approvals.id"],
            name="fk_draft_approvals_supersedes_id_draft_approvals",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_draft_approvals"),
        sa.UniqueConstraint(
            "supersedes_id", name="uq_draft_approvals_supersedes"
        ),
    )
    op.create_index("ix_draft_approvals_draft_id", "draft_approvals", ["draft_id"])
    op.create_index(
        "ix_draft_approvals_draft_version_id",
        "draft_approvals",
        ["draft_version_id"],
    )
    op.create_index(
        "ix_draft_approvals_draft_created",
        "draft_approvals",
        ["draft_id", "created_at"],
    )

    op.create_table(
        "publications",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("draft_id", sa.String(length=36), nullable=False),
        sa.Column("draft_version_id", sa.String(length=36), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "kind",
            _enum("publication_kind", "post", "thread_item", "reply"),
            nullable=False,
        ),
        sa.Column(
            "status",
            _enum(
                "publication_status",
                "claimed",
                "in_flight",
                "succeeded",
                "failed",
                "ambiguous",
                "remote_deleted",
            ),
            nullable=False,
        ),
        sa.Column("thread_position", sa.Integer(), nullable=False),
        sa.Column("request_payload_hash", sa.String(length=64), nullable=False),
        sa.Column("x_post_id", sa.String(length=32), nullable=True),
        sa.Column("parent_post_id", sa.String(length=32), nullable=True),
        sa.Column("account_id", sa.String(length=64), nullable=True),
        sa.Column("response_metadata", sa.JSON(), nullable=False),
        sa.Column("estimated_cost_usd", sa.Numeric(14, 6), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deletion_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_details", sa.Text(), nullable=True),
        _created_at(),
        _updated_at(),
        sa.CheckConstraint(
            "thread_position >= 0", name="ck_publications_position_nonnegative"
        ),
        sa.CheckConstraint(
            "attempt_count >= 0", name="ck_publications_attempt_nonnegative"
        ),
        sa.CheckConstraint(
            "estimated_cost_usd >= 0", name="ck_publications_cost_nonnegative"
        ),
        sa.ForeignKeyConstraint(
            ["draft_id"],
            ["drafts.id"],
            name="fk_publications_draft_id_drafts",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["draft_version_id"],
            ["draft_versions.id"],
            name="fk_publications_draft_version_id_draft_versions",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_publications"),
        sa.UniqueConstraint(
            "draft_id",
            "content_hash",
            "thread_position",
            name="uq_publications_draft_hash_position",
        ),
        sa.UniqueConstraint(
            "draft_id",
            "request_payload_hash",
            "thread_position",
            name="uq_publications_request_position",
        ),
        sa.UniqueConstraint("x_post_id", name="uq_publications_x_post_id"),
    )
    op.create_index("ix_publications_draft_id", "publications", ["draft_id"])
    op.create_index(
        "ix_publications_draft_version_id", "publications", ["draft_version_id"]
    )
    op.create_index(
        "ix_publications_draft_status", "publications", ["draft_id", "status"]
    )

    op.create_table(
        "video_scripts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("draft_id", sa.String(length=36), nullable=False),
        sa.Column("draft_version_id", sa.String(length=36), nullable=False),
        sa.Column("script_payload", sa.JSON(), nullable=False),
        sa.Column("script_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            _enum(
                "video_script_status",
                "generated",
                "needs_review",
                "approved",
                "generation_requested",
                "rendering",
                "completed",
                "failed",
                "rejected",
            ),
            nullable=False,
        ),
        sa.Column("approved_hash", sa.String(length=64), nullable=True),
        sa.Column("approved_by", sa.String(length=200), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        _created_at(),
        _updated_at(),
        sa.ForeignKeyConstraint(
            ["draft_id"],
            ["drafts.id"],
            name="fk_video_scripts_draft_id_drafts",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["draft_version_id"],
            ["draft_versions.id"],
            name="fk_video_scripts_draft_version_id_draft_versions",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_video_scripts"),
        sa.UniqueConstraint(
            "draft_id", "script_hash", name="uq_video_scripts_draft_hash"
        ),
    )
    op.create_index("ix_video_scripts_draft_id", "video_scripts", ["draft_id"])
    op.create_index(
        "ix_video_scripts_draft_version_id", "video_scripts", ["draft_version_id"]
    )
    op.create_index(
        "ix_video_scripts_status_updated",
        "video_scripts",
        ["status", "updated_at"],
    )

    op.create_table(
        "video_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("video_script_id", sa.String(length=36), nullable=False),
        sa.Column(
            "mode",
            _enum("video_generator_mode", "disabled", "api", "plugin_manual"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("external_job_id", sa.String(length=200), nullable=True),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            _enum("video_job_status", "requested", "rendering", "completed", "failed"),
            nullable=False,
        ),
        sa.Column("response_metadata", sa.JSON(), nullable=False),
        sa.Column("local_path", sa.String(length=2048), nullable=True),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("thumbnail_metadata", sa.JSON(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_details", sa.Text(), nullable=True),
        _created_at(),
        _updated_at(),
        sa.CheckConstraint("file_size >= 0", name="ck_video_jobs_file_size_nonnegative"),
        sa.ForeignKeyConstraint(
            ["video_script_id"],
            ["video_scripts.id"],
            name="fk_video_jobs_video_script_id_video_scripts",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_video_jobs"),
        sa.UniqueConstraint(
            "video_script_id", "request_hash", name="uq_video_jobs_script_request"
        ),
        sa.UniqueConstraint(
            "provider", "external_job_id", name="uq_video_jobs_provider_external"
        ),
    )
    op.create_index(
        "ix_video_jobs_video_script_id", "video_jobs", ["video_script_id"]
    )
    op.create_index(
        "ix_video_jobs_status_updated", "video_jobs", ["status", "updated_at"]
    )

    op.create_table(
        "llm_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("idea_id", sa.String(length=36), nullable=True),
        sa.Column("draft_id", sa.String(length=36), nullable=True),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("request_id", sa.String(length=255), nullable=True),
        sa.Column("operation", sa.String(length=100), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("estimated_cost_usd", sa.Numeric(14, 6), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            _enum("llm_run_status", "succeeded", "failed", "retrying"),
            nullable=False,
        ),
        sa.Column("request_payload_hash", sa.String(length=64), nullable=True),
        sa.Column("actual_model", sa.String(length=200), nullable=True),
        sa.Column("error_details", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        _created_at(),
        sa.CheckConstraint(
            "input_tokens >= 0", name="ck_llm_runs_input_tokens_nonnegative"
        ),
        sa.CheckConstraint(
            "output_tokens >= 0", name="ck_llm_runs_output_tokens_nonnegative"
        ),
        sa.CheckConstraint("latency_ms >= 0", name="ck_llm_runs_latency_nonnegative"),
        sa.CheckConstraint("retry_count >= 0", name="ck_llm_runs_retry_nonnegative"),
        sa.CheckConstraint(
            "estimated_cost_usd >= 0", name="ck_llm_runs_cost_nonnegative"
        ),
        sa.ForeignKeyConstraint(
            ["idea_id"],
            ["ideas.id"],
            name="fk_llm_runs_idea_id_ideas",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["draft_id"],
            ["drafts.id"],
            name="fk_llm_runs_draft_id_drafts",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_llm_runs"),
        sa.UniqueConstraint(
            "provider", "request_id", name="uq_llm_runs_provider_request"
        ),
    )
    op.create_index("ix_llm_runs_idea_id", "llm_runs", ["idea_id"])
    op.create_index("ix_llm_runs_draft_id", "llm_runs", ["draft_id"])
    op.create_index(
        "ix_llm_runs_provider_created", "llm_runs", ["provider", "created_at"]
    )

    op.create_table(
        "api_usage",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("operation", sa.String(length=100), nullable=False),
        sa.Column(
            "direction", _enum("api_direction", "read", "write"), nullable=False
        ),
        sa.Column("units", sa.Integer(), nullable=False),
        sa.Column("estimated_cost_usd", sa.Numeric(14, 6), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("request_id", sa.String(length=255), nullable=True),
        sa.Column("success", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        _created_at(),
        sa.CheckConstraint("units >= 0", name="ck_api_usage_units_nonnegative"),
        sa.CheckConstraint(
            "estimated_cost_usd >= 0", name="ck_api_usage_cost_nonnegative"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_api_usage"),
    )
    op.create_index(
        "ix_api_usage_provider_created", "api_usage", ["provider", "created_at"]
    )
    op.create_index(
        "ix_api_usage_direction_created", "api_usage", ["direction", "created_at"]
    )

    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("entity_type", sa.String(length=100), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=True),
        sa.Column("actor", sa.String(length=200), nullable=False),
        sa.Column("old_status", sa.String(length=64), nullable=True),
        sa.Column("new_status", sa.String(length=64), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_audit_events"),
    )
    op.create_index(
        "ix_audit_events_entity", "audit_events", ["entity_type", "entity_id"]
    )
    op.create_index(
        "ix_audit_events_action_created", "audit_events", ["action", "created_at"]
    )

    op.create_table(
        "opt_out_users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("x_user_id", sa.String(length=64), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("opted_out_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_opt_out_users"),
        sa.UniqueConstraint("x_user_id", name="uq_opt_out_users_x_user_id"),
    )

    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("updated_by", sa.String(length=200), nullable=True),
        sa.Column("is_secret", sa.Boolean(), server_default=sa.false(), nullable=False),
        _created_at(),
        _updated_at(),
        sa.CheckConstraint("is_secret = false", name="ck_app_settings_no_secrets"),
        sa.PrimaryKeyConstraint("key", name="pk_app_settings"),
    )


def downgrade() -> None:
    op.drop_table("app_settings")
    op.drop_table("opt_out_users")
    op.drop_index("ix_audit_events_action_created", table_name="audit_events")
    op.drop_index("ix_audit_events_entity", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index("ix_api_usage_direction_created", table_name="api_usage")
    op.drop_index("ix_api_usage_provider_created", table_name="api_usage")
    op.drop_table("api_usage")
    op.drop_index("ix_llm_runs_provider_created", table_name="llm_runs")
    op.drop_index("ix_llm_runs_draft_id", table_name="llm_runs")
    op.drop_index("ix_llm_runs_idea_id", table_name="llm_runs")
    op.drop_table("llm_runs")
    op.drop_index("ix_video_jobs_status_updated", table_name="video_jobs")
    op.drop_index("ix_video_jobs_video_script_id", table_name="video_jobs")
    op.drop_table("video_jobs")
    op.drop_index("ix_video_scripts_status_updated", table_name="video_scripts")
    op.drop_index("ix_video_scripts_draft_version_id", table_name="video_scripts")
    op.drop_index("ix_video_scripts_draft_id", table_name="video_scripts")
    op.drop_table("video_scripts")
    op.drop_index("ix_publications_draft_status", table_name="publications")
    op.drop_index("ix_publications_draft_version_id", table_name="publications")
    op.drop_index("ix_publications_draft_id", table_name="publications")
    op.drop_table("publications")
    op.drop_index("ix_draft_approvals_draft_created", table_name="draft_approvals")
    op.drop_index(
        "ix_draft_approvals_draft_version_id", table_name="draft_approvals"
    )
    op.drop_index("ix_draft_approvals_draft_id", table_name="draft_approvals")
    op.drop_table("draft_approvals")
    op.drop_index("ix_draft_versions_draft_hash", table_name="draft_versions")
    op.drop_index("ix_draft_versions_draft_id", table_name="draft_versions")
    op.drop_table("draft_versions")
    op.drop_index("ix_drafts_status_updated_at", table_name="drafts")
    op.drop_index("ix_drafts_current_content_hash", table_name="drafts")
    op.drop_index("ix_drafts_reply_candidate_id", table_name="drafts")
    op.drop_index("ix_drafts_idea_id", table_name="drafts")
    op.drop_table("drafts")
    op.drop_index("ix_reply_candidates_status_created", table_name="reply_candidates")
    op.drop_index("ix_reply_candidates_author_id", table_name="reply_candidates")
    op.drop_index(
        "ix_reply_candidates_conversation_id", table_name="reply_candidates"
    )
    op.drop_table("reply_candidates")
    op.drop_index("ix_idea_sources_topic_id", table_name="idea_sources")
    op.drop_index("ix_idea_sources_source_post_id", table_name="idea_sources")
    op.drop_index("ix_idea_sources_idea_id", table_name="idea_sources")
    op.drop_table("idea_sources")
    op.drop_table("ideas")
    op.drop_index("ix_source_posts_fetched_at", table_name="source_posts")
    op.drop_index("ix_source_posts_conversation_id", table_name="source_posts")
    op.drop_index("ix_source_posts_username", table_name="source_posts")
    op.drop_index("ix_source_posts_author_id", table_name="source_posts")
    op.drop_index("ix_source_posts_content_source_id", table_name="source_posts")
    op.drop_table("source_posts")
    op.drop_table("topics")
    op.drop_table("content_sources")

