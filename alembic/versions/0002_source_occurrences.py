"""Preserve source occurrences and deduplicate raw posts only by X post ID.

Revision ID: 0002_source_occurrences
Revises: 0001_initial
Create Date: 2026-07-11
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_source_occurrences"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_STAGE_TABLE = "_source_post_occurrence_stage"


def _require_downgrade_compatible_text_hashes(
    connection: sa.engine.Connection,
) -> None:
    """Refuse a lossy downgrade before creating or changing any table.

    Revision 0001 requires ``source_posts.text_hash`` to be unique, while this
    revision deliberately permits different posts with the same normalized
    text.  Choosing which row to delete or merge is an operator decision, not a
    safe migration default.
    """

    metadata = sa.MetaData()
    source_posts = sa.Table("source_posts", metadata, autoload_with=connection)
    duplicate = connection.execute(
        sa.select(
            source_posts.c.text_hash,
            sa.func.count(source_posts.c.id).label("row_count"),
        )
        .group_by(source_posts.c.text_hash)
        .having(sa.func.count(source_posts.c.id) > 1)
        .limit(1)
    ).first()
    if duplicate is None:
        return

    raise RuntimeError(
        "Downgrade from 0002_source_occurrences to 0001_initial is blocked: "
        "duplicate source_posts.text_hash values cannot satisfy the unique "
        "constraint restored by 0001. No schema or data changes were made by "
        "this downgrade. Create and verify a database backup, resolve duplicate "
        "source_posts manually, then retry; or keep the current schema. "
        "Automatic deletion or merging is intentionally declined."
    )


def _create_stage_table() -> None:
    op.create_table(
        _STAGE_TABLE,
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_post_id", sa.String(length=36), nullable=False),
        sa.Column("content_source_id", sa.String(length=36), nullable=False),
        sa.Column("source_weight", sa.Float(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("seen_count", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=f"pk_{_STAGE_TABLE}"),
    )


def _create_occurrence_table() -> None:
    op.create_table(
        "source_post_occurrences",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_post_id", sa.String(length=36), nullable=False),
        sa.Column("content_source_id", sa.String(length=36), nullable=False),
        sa.Column("source_weight", sa.Float(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("seen_count", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "seen_count >= 1",
            name=(
                "ck_source_post_occurrences_"
                "source_post_occurrences_seen_count_positive"
            ),
        ),
        sa.CheckConstraint(
            "source_weight >= 0 AND source_weight <= 1",
            name=(
                "ck_source_post_occurrences_"
                "source_post_occurrences_weight_range"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["content_source_id"],
            ["content_sources.id"],
            name=(
                "fk_source_post_occurrences_content_source_id_content_sources"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["source_post_id"],
            ["source_posts.id"],
            name="fk_source_post_occurrences_source_post_id_source_posts",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_post_occurrences"),
        sa.UniqueConstraint(
            "source_post_id",
            "content_source_id",
            name="uq_source_post_occurrences_post_source",
        ),
    )
    op.create_index(
        "ix_source_post_occurrences_content_source_id",
        "source_post_occurrences",
        ["content_source_id"],
    )
    op.create_index(
        "ix_source_post_occurrences_source_post",
        "source_post_occurrences",
        ["source_post_id", "source_weight"],
    )


def upgrade() -> None:
    _create_stage_table()
    connection = op.get_bind()
    metadata = sa.MetaData()
    source_posts = sa.Table("source_posts", metadata, autoload_with=connection)
    content_sources = sa.Table("content_sources", metadata, autoload_with=connection)
    stage = sa.Table(_STAGE_TABLE, metadata, autoload_with=connection)
    connection.execute(
        sa.insert(stage).from_select(
            [
                "id",
                "source_post_id",
                "content_source_id",
                "source_weight",
                "first_seen_at",
                "last_seen_at",
                "seen_count",
                "metadata",
            ],
            sa.select(
                source_posts.c.id,
                source_posts.c.id,
                source_posts.c.content_source_id,
                content_sources.c.weight,
                source_posts.c.fetched_at,
                source_posts.c.fetched_at,
                sa.literal(1),
                sa.literal({"migrated": True}, type_=sa.JSON()),
            )
            .join(
                content_sources,
                content_sources.c.id == source_posts.c.content_source_id,
            )
            .where(source_posts.c.content_source_id.is_not(None)),
        )
    )

    with op.batch_alter_table("source_posts", recreate="always") as batch_op:
        batch_op.drop_index("ix_source_posts_content_source_id")
        batch_op.drop_constraint(
            "fk_source_posts_content_source_id_content_sources",
            type_="foreignkey",
        )
        batch_op.drop_constraint("uq_source_posts_text_hash", type_="unique")
        batch_op.drop_column("content_source_id")

    _create_occurrence_table()
    metadata = sa.MetaData()
    stage = sa.Table(_STAGE_TABLE, metadata, autoload_with=connection)
    occurrences = sa.Table(
        "source_post_occurrences", metadata, autoload_with=connection
    )
    connection.execute(
        sa.insert(occurrences).from_select(
            [
                "id",
                "source_post_id",
                "content_source_id",
                "source_weight",
                "first_seen_at",
                "last_seen_at",
                "seen_count",
                "metadata",
            ],
            sa.select(
                stage.c.id,
                stage.c.source_post_id,
                stage.c.content_source_id,
                stage.c.source_weight,
                stage.c.first_seen_at,
                stage.c.last_seen_at,
                stage.c.seen_count,
                stage.c.metadata,
            ),
        )
    )
    op.drop_table(_STAGE_TABLE)


def downgrade() -> None:
    connection = op.get_bind()
    # This must remain the first operation.  SQLite batch alteration copies and
    # replaces tables, so discovering the conflict later could leave a partially
    # downgraded database even though Alembic did not advance its revision row.
    _require_downgrade_compatible_text_hashes(connection)

    _create_stage_table()
    metadata = sa.MetaData()
    stage = sa.Table(_STAGE_TABLE, metadata, autoload_with=connection)
    occurrences = sa.Table(
        "source_post_occurrences", metadata, autoload_with=connection
    )
    connection.execute(
        sa.insert(stage).from_select(
            [
                "id",
                "source_post_id",
                "content_source_id",
                "source_weight",
                "first_seen_at",
                "last_seen_at",
                "seen_count",
                "metadata",
            ],
            sa.select(
                occurrences.c.id,
                occurrences.c.source_post_id,
                occurrences.c.content_source_id,
                occurrences.c.source_weight,
                occurrences.c.first_seen_at,
                occurrences.c.last_seen_at,
                occurrences.c.seen_count,
                occurrences.c.metadata,
            ),
        )
    )
    op.drop_index(
        "ix_source_post_occurrences_source_post",
        table_name="source_post_occurrences",
    )
    op.drop_index(
        "ix_source_post_occurrences_content_source_id",
        table_name="source_post_occurrences",
    )
    op.drop_table("source_post_occurrences")

    with op.batch_alter_table("source_posts", recreate="always") as batch_op:
        batch_op.add_column(
            sa.Column("content_source_id", sa.String(length=36), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_source_posts_content_source_id_content_sources",
            "content_sources",
            ["content_source_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_unique_constraint(
            "uq_source_posts_text_hash",
            ["text_hash"],
        )
        batch_op.create_index(
            "ix_source_posts_content_source_id",
            ["content_source_id"],
        )

    metadata = sa.MetaData()
    source_posts = sa.Table("source_posts", metadata, autoload_with=connection)
    stage = sa.Table(_STAGE_TABLE, metadata, autoload_with=connection)
    preferred_source = (
        sa.select(stage.c.content_source_id)
        .where(stage.c.source_post_id == source_posts.c.id)
        .order_by(stage.c.source_weight.desc(), stage.c.content_source_id.asc())
        .limit(1)
        .scalar_subquery()
    )
    connection.execute(
        sa.update(source_posts).values(content_source_id=preferred_source)
    )
    op.drop_table(_STAGE_TABLE)
