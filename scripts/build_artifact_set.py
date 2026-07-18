"""Build and validate one coherent source-ZIP/wheel artifact set from a clean tree."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas.release_artifacts import (  # noqa: E402
    ArtifactDigest,
    ModuleDigestPair,
    ReleaseArtifactSet,
)
from app.services.humanizer_runtime import HumanizerRuntimeLoader  # noqa: E402
from scripts.build_release import build_release  # noqa: E402

VERSION_RE = re.compile(r'^version = "([^"]+)"$', re.MULTILINE)
KEY_SEMANTIC_MODULES = (
    "app/schemas/content.py",
    "app/services/semantic_composition.py",
    "app/services/semantic_extraction.py",
    "app/services/semantic_extraction_coordinator.py",
    "app/services/semantic_provider_validation.py",
    "app/services/semantic_reconciliation.py",
    "app/services/semantic_adjudication.py",
)


class ArtifactSetError(RuntimeError):
    """Raised when source, wheel, or manifest identities do not agree."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(root: Path) -> str:
    match = VERSION_RE.search((root / "pyproject.toml").read_text(encoding="utf-8"))
    if match is None:
        raise ArtifactSetError("pyproject.toml has no project version")
    return match.group(1)


def _source_tree_digest(root: Path, manifest: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for relative in manifest:
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256_file(root / relative).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _wheel_metadata_version(archive: zipfile.ZipFile) -> str:
    metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1:
        raise ArtifactSetError("wheel must contain exactly one METADATA file")
    text = archive.read(metadata_names[0]).decode("utf-8")
    for line in text.splitlines():
        if line.startswith("Version: "):
            return line.removeprefix("Version: ").strip()
    raise ArtifactSetError("wheel METADATA has no Version field")


def _runtime_from_wheel(wheel: Path) -> tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="xbot-wheel-runtime-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)
        bundled = root / "app" / "_bundled" / "personal-humanizer"
        prompt_root = root / "app" / "prompts"
        skill = bundled / "SKILL.md"
        if not skill.is_file():
            raise ArtifactSetError("wheel is missing bundled Personal Humanizer SKILL.md")
        runtime = HumanizerRuntimeLoader(
            include_references=True,
            bundled_root=bundled,
            prompt_root=prompt_root,
        ).load()
        return _sha256_file(skill), runtime.skill_hash


def _semantic_digests(root: Path, wheel: Path) -> dict[str, ModuleDigestPair]:
    values: dict[str, ModuleDigestPair] = {}
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for relative in KEY_SEMANTIC_MODULES:
            if relative not in names:
                raise ArtifactSetError(f"wheel is missing semantic module: {relative}")
            source_hash = _sha256_file(root / relative)
            wheel_hash = _sha256_bytes(archive.read(relative))
            if source_hash != wheel_hash:
                raise ArtifactSetError(f"source/wheel semantic module mismatch: {relative}")
            values[relative] = ModuleDigestPair(
                source_sha256=source_hash,
                wheel_sha256=wheel_hash,
            )
    return values


def _migration_head(root: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "heads"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
    if not line:
        raise ArtifactSetError("Alembic did not report a migration head")
    return line.split()[0]


def _artifact(path: Path) -> ArtifactDigest:
    return ArtifactDigest(
        filename=path.name,
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def validate_artifact_set(manifest_path: Path) -> ReleaseArtifactSet:
    """Validate hashes, versions, semantic modules, and Humanizer identity from manifest."""

    manifest = ReleaseArtifactSet.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    source_zip = base / manifest.source_zip.filename
    wheel = base / manifest.wheel.filename
    runner = base / manifest.external_runner.filename
    schema = base / manifest.external_schema.filename
    for artifact, path in (
        (manifest.source_zip, source_zip),
        (manifest.wheel, wheel),
        (manifest.external_runner, runner),
        (manifest.external_schema, schema),
    ):
        if not path.is_file():
            raise ArtifactSetError(f"artifact is missing: {path.name}")
        if _sha256_file(path) != artifact.sha256 or path.stat().st_size != artifact.size_bytes:
            raise ArtifactSetError(f"artifact digest mismatch: {path.name}")
    expected_wheel_name = f"vouch-{manifest.application_version}-py3-none-any.whl"
    if wheel.name != expected_wheel_name:
        raise ArtifactSetError(
            f"wrong wheel filename/version: {wheel.name!r} != {expected_wheel_name!r}"
        )
    with zipfile.ZipFile(wheel) as archive:
        if _wheel_metadata_version(archive) != manifest.application_version:
            raise ArtifactSetError("wheel metadata version does not match artifact manifest")
        for relative, pair in manifest.semantic_module_digests.items():
            if relative not in archive.namelist():
                raise ArtifactSetError(f"wheel is missing manifest module: {relative}")
            if _sha256_bytes(archive.read(relative)) != pair.wheel_sha256:
                raise ArtifactSetError(f"wheel module digest mismatch: {relative}")
            if pair.source_sha256 != pair.wheel_sha256:
                raise ArtifactSetError(f"source/wheel module parity failed: {relative}")
    wheel_skill_hash, wheel_runtime_hash = _runtime_from_wheel(wheel)
    if wheel_skill_hash != manifest.wheel_skill_hash:
        raise ArtifactSetError("wheel SKILL.md hash does not match artifact manifest")
    if wheel_runtime_hash != manifest.wheel_runtime_hash:
        raise ArtifactSetError("wheel runtime hash does not match artifact manifest")
    if manifest.source_skill_hash != manifest.wheel_skill_hash:
        raise ArtifactSetError("source/wheel SKILL.md hashes differ")
    if manifest.source_runtime_hash != manifest.wheel_runtime_hash:
        raise ArtifactSetError("source/wheel runtime hashes differ")
    with zipfile.ZipFile(source_zip) as archive:
        source_names = set(archive.namelist())
        for relative, pair in manifest.semantic_module_digests.items():
            if relative not in source_names:
                raise ArtifactSetError(f"source ZIP is missing manifest module: {relative}")
            if _sha256_bytes(archive.read(relative)) != pair.source_sha256:
                raise ArtifactSetError(f"source ZIP module digest mismatch: {relative}")
    return manifest


def build_artifact_set(
    root: Path, output_dir: Path, *, test_count: int
) -> tuple[ReleaseArtifactSet, Path]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    version = _version(root)
    if version != "0.19.7":
        raise ArtifactSetError(
            f"artifact-set build requires application version 0.19.7, got {version}"
        )
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    wheel_stage = output_dir / "wheel-stage"
    wheel_stage.mkdir()
    environment = os.environ.copy()
    environment.setdefault("SOURCE_DATE_EPOCH", "315532800")
    subprocess.run(  # noqa: S603 - fixed local interpreter and fixed arguments
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_stage),
        ],
        cwd=root,
        env=environment,
        check=True,
    )
    wheels = tuple(wheel_stage.glob("*.whl"))
    if len(wheels) != 1:
        raise ArtifactSetError("clean build must produce exactly one wheel")
    expected_name = f"vouch-{version}-py3-none-any.whl"
    if wheels[0].name != expected_name:
        raise ArtifactSetError(f"wheel filename/version mismatch: {wheels[0].name}")
    wheel = output_dir / expected_name
    wheels[0].replace(wheel)
    shutil.rmtree(wheel_stage)
    source_zip = output_dir / "Vouch_v0.19.7_RC.zip"
    _, release_manifest = build_release(root=root, destination=source_zip)
    source_tree_digest = _source_tree_digest(root, release_manifest)
    source_skill = root / "docs" / "personal-humanizer" / "SKILL.md"
    source_runtime = HumanizerRuntimeLoader(
        include_references=True,
        bundled_root=root / "docs" / "personal-humanizer",
        prompt_root=root / "app" / "prompts",
    ).load()
    wheel_skill_hash, wheel_runtime_hash = _runtime_from_wheel(wheel)
    semantic_digests = _semantic_digests(root, wheel)
    runner = output_dir / "run_external_semantic_holdout_v5.py"
    shutil.copyfile(root / "scripts" / runner.name, runner)
    schema = output_dir / "external-semantic-holdout-v5.schema.json"
    subprocess.run(  # noqa: S603 - fixed interpreter and release runner
        [
            sys.executable,
            str(root / "scripts" / runner.name),
            "--write-schema",
            str(schema),
        ],
        cwd=root,
        check=True,
    )
    manifest = ReleaseArtifactSet(
        application_version=version,
        source_zip=_artifact(source_zip),
        wheel=_artifact(wheel),
        external_runner=_artifact(runner),
        external_schema=_artifact(schema),
        source_tree_digest=source_tree_digest,
        semantic_module_digests=semantic_digests,
        source_skill_hash=_sha256_file(source_skill),
        wheel_skill_hash=wheel_skill_hash,
        source_runtime_hash=source_runtime.skill_hash,
        wheel_runtime_hash=wheel_runtime_hash,
        test_count=test_count,
        migration_head=_migration_head(root),
        release_file_count=len(release_manifest),
    )
    manifest_path = output_dir / "Vouch_v0.19.7_ARTIFACT_SET.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    validate_artifact_set(manifest_path)
    checksum_path = output_dir / "SHA256SUMS.txt"
    checksum_artifacts = (source_zip, wheel, manifest_path, runner, schema)
    checksum_path.write_text(
        "".join(f"{_sha256_file(path)}  {path.name}\n" for path in checksum_artifacts),
        encoding="utf-8",
    )
    return manifest, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist" / "artifact-set")
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        if arguments.validate is not None:
            manifest = validate_artifact_set(arguments.validate.resolve())
            print(f"Artifact set valid: {manifest.application_version}")
            return 0
        manifest, path = build_artifact_set(
            ROOT,
            arguments.output_dir,
            test_count=arguments.test_count,
        )
        print(
            f"Artifact set: {path} ({manifest.release_file_count} release files; "
            "source/wheel parity passed)"
        )
        return 0
    except (ArtifactSetError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print(f"Artifact-set build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
