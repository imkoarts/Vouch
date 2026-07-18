"""Versioned manifest for one cryptographically coherent release artifact set."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ArtifactDigest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)


class ModuleDigestPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wheel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReleaseArtifactSet(BaseModel):
    """Manifest binding source ZIP, wheel, semantic modules, and runtime identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    application_version: str = Field(min_length=1)
    source_zip: ArtifactDigest
    wheel: ArtifactDigest
    external_runner: ArtifactDigest
    external_schema: ArtifactDigest
    source_tree_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_module_digests: dict[str, ModuleDigestPair]
    source_skill_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    wheel_skill_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_runtime_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    wheel_runtime_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_count: int = Field(ge=0)
    migration_head: str = Field(min_length=1)
    release_file_count: int = Field(ge=1)
