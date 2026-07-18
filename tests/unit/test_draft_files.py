from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.domain.security import compute_content_hash
from app.services.draft_files import DraftArtifactError, DraftArtifactStore

_CREATED_AT = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
_TEXT_PLAN = {"type": "none", "required_files": []}


def _metadata(draft_id: str, content: str) -> dict[str, object]:
    content_hash = compute_content_hash(content, _TEXT_PLAN)
    return {
        "draft_id": draft_id,
        "status": "needs_review",
        "created_at": _CREATED_AT.isoformat(),
        "updated_at": _CREATED_AT.isoformat(),
        "content_type": "short_post",
        "language": "ru",
        "provider": "mock",
        "model": "mock-model",
        "source_count": 0,
        "character_count": len(content),
        "fact_check_status": "not_required",
        "approved_at": None,
        "content_hash": content_hash,
    }


class DraftBundleAtomicityTests(unittest.TestCase):
    def test_complete_bundle_is_revealed_only_after_all_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            store = DraftArtifactStore(Path(temporary_root))
            directory = store.create_bundle(
                draft_id="draft-atomic",
                created_at=_CREATED_AT,
                metadata=_metadata("draft-atomic", "Safe draft"),
                content="Safe draft",
                media_plan=_TEXT_PLAN,
            )

            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {
                    "draft.md",
                    "generations.json",
                    "media_manifest.json",
                    "media_plan.json",
                    "metadata.json",
                    "publication.json",
                    "review.md",
                    "sources.json",
                    "video_script.json",
                },
            )
            self.assertEqual(
                json.loads((directory / "media_manifest.json").read_text("utf-8")),
                {"version": 1, "files": []},
            )
            self.assertFalse(list(directory.parent.glob(".draft-atomic.*.tmp")))

    def test_failed_bundle_build_leaves_no_partial_final_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            store = DraftArtifactStore(Path(temporary_root))
            final_directory = store.draft_directory("draft-failure", _CREATED_AT)

            with (
                patch(
                    "app.services.draft_files._atomic_write_json",
                    side_effect=OSError("synthetic write failure"),
                ),
                self.assertRaises(OSError),
            ):
                store.create_bundle(
                    draft_id="draft-failure",
                    created_at=_CREATED_AT,
                    metadata=_metadata("draft-failure", "Safe draft"),
                    content="Safe draft",
                    media_plan=_TEXT_PLAN,
                )

            self.assertFalse(final_directory.exists())
            self.assertFalse(list(final_directory.parent.glob(".draft-failure.*.tmp")))


class DraftReconciliationTests(unittest.TestCase):
    def _bundle(
        self, root: Path, draft_id: str = "draft-reconcile"
    ) -> tuple[DraftArtifactStore, Path]:
        store = DraftArtifactStore(root)
        directory = store.create_bundle(
            draft_id=draft_id,
            created_at=_CREATED_AT,
            metadata=_metadata(draft_id, "Original text"),
            content="Original text",
            media_plan=_TEXT_PLAN,
        )
        return store, directory

    def test_database_content_type_is_authoritative_over_front_matter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            store, directory = self._bundle(Path(temporary_root))
            draft_path = directory / "draft.md"
            forged = draft_path.read_text("utf-8").replace(
                "content_type: short_post", "content_type: thread"
            )
            draft_path.write_text(forged, encoding="utf-8", newline="\n")

            with self.assertRaisesRegex(DraftArtifactError, "authoritative database"):
                store.read_markdown(
                    directory,
                    expected_hash=str(
                        _metadata("draft-reconcile", "Original text")["content_hash"]
                    ),
                    media_plan=_TEXT_PLAN,
                    expected_content_type="short_post",
                )

    def test_text_edit_changes_hash_without_trusting_front_matter_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            store, directory = self._bundle(Path(temporary_root))
            original_hash = str(_metadata("draft-reconcile", "Original text")["content_hash"])
            draft_path = directory / "draft.md"
            draft_path.write_text(
                draft_path.read_text("utf-8").replace("Original text", "Edited text"),
                encoding="utf-8",
                newline="\n",
            )

            snapshot = store.read_markdown(
                directory,
                expected_hash=original_hash,
                expected_content="Original text",
                media_plan=_TEXT_PLAN,
                expected_content_type="short_post",
            )

            self.assertTrue(snapshot.changed)
            self.assertTrue(snapshot.content_changed)
            self.assertFalse(snapshot.media_manifest_changed)
            self.assertTrue(snapshot.approval_fingerprint_changed)
            self.assertNotEqual(snapshot.content_hash, original_hash)

    def test_replacing_media_bytes_invalidates_hash_and_strict_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            store, directory = self._bundle(Path(temporary_root))
            media_directory = directory / "media"
            media_directory.mkdir()
            media_file = media_directory / "visual.png"
            media_file.write_bytes(b"first synthetic image bytes")
            media_plan = {
                "type": "image",
                "required_files": ["media/visual.png"],
                "metadata": {"width": 1200, "height": 675},
            }
            manifest = store.refresh_media_manifest(directory, media_plan)
            approved_hash = compute_content_hash("Original text", media_plan, manifest)

            unchanged = store.read_markdown(
                directory,
                expected_hash=approved_hash,
                expected_content="Original text",
                media_plan=media_plan,
                expected_content_type="short_post",
            )
            self.assertFalse(unchanged.changed)
            self.assertFalse(unchanged.content_changed)
            self.assertFalse(unchanged.approval_fingerprint_changed)
            self.assertEqual(
                set(unchanged.media_manifest["files"][0]),
                {"relative_path", "sha256", "mime_type", "file_size", "metadata"},
            )

            media_file.write_bytes(b"other synthetic image bytes")
            changed = store.read_markdown(
                directory,
                expected_hash=approved_hash,
                expected_content="Original text",
                media_plan=media_plan,
                expected_content_type="short_post",
            )

            self.assertTrue(changed.changed)
            self.assertFalse(changed.content_changed)
            self.assertTrue(changed.media_manifest_changed)
            self.assertTrue(changed.approval_fingerprint_changed)
            self.assertNotEqual(changed.content_hash, approved_hash)
            with self.assertRaisesRegex(DraftArtifactError, "approved manifest"):
                store.validate_media_manifest(directory, media_plan)

    def test_text_and_media_changes_are_reported_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            store, directory = self._bundle(Path(temporary_root))
            media_directory = directory / "media"
            media_directory.mkdir()
            media_file = media_directory / "visual.png"
            media_file.write_bytes(b"first synthetic image bytes")
            media_plan = {
                "type": "image",
                "required_files": ["media/visual.png"],
            }
            manifest = store.refresh_media_manifest(directory, media_plan)
            approved_hash = compute_content_hash("Original text", media_plan, manifest)
            draft_path = directory / "draft.md"
            draft_path.write_text(
                draft_path.read_text("utf-8").replace("Original text", "Edited text"),
                encoding="utf-8",
                newline="\n",
            )
            media_file.write_bytes(b"second synthetic image bytes")

            changed = store.read_markdown(
                directory,
                expected_hash=approved_hash,
                expected_content="Original text",
                media_plan=media_plan,
                expected_content_type="short_post",
            )

            self.assertTrue(changed.content_changed)
            self.assertTrue(changed.media_manifest_changed)
            self.assertTrue(changed.approval_fingerprint_changed)

    def test_media_paths_cannot_escape_the_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            store, directory = self._bundle(Path(temporary_root))
            with self.assertRaisesRegex(DraftArtifactError, "safe bundle-relative"):
                store.build_media_manifest(
                    directory,
                    {"type": "image", "required_files": ["../outside.png"]},
                )

    def test_validated_media_files_returns_only_manifest_verified_bundle_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            store, directory = self._bundle(Path(temporary_root))
            media_directory = directory / "media"
            media_directory.mkdir()
            media_file = media_directory / "visual.png"
            media_file.write_bytes(b"verified synthetic image")
            media_plan = {
                "type": "image",
                "required_files": ["media/visual.png"],
            }
            store.refresh_media_manifest(directory, media_plan)

            files = store.validated_media_files(directory, media_plan)

            self.assertEqual(files, (media_file.resolve(),))


class DraftQuarantineTests(unittest.TestCase):
    def _bundle(self, root: Path, draft_id: str) -> tuple[DraftArtifactStore, Path]:
        store = DraftArtifactStore(root)
        directory = store.create_bundle(
            draft_id=draft_id,
            created_at=_CREATED_AT,
            metadata=_metadata(draft_id, "Quarantine me"),
            content="Quarantine me",
            media_plan=_TEXT_PLAN,
        )
        return store, directory

    def test_local_removal_moves_complete_bundle_to_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            store, directory = self._bundle(Path(temporary_root), "draft-delete")
            destination = store.remove_bundle(directory, current_status="needs_review")

            self.assertFalse(directory.exists())
            self.assertTrue((destination / "draft.md").is_file())
            self.assertIn(store.quarantine_root, destination.parents)

    def test_published_bundle_cannot_be_ordinary_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            store, directory = self._bundle(Path(temporary_root), "draft-published")

            with self.assertRaisesRegex(DraftArtifactError, "does not permit"):
                store.remove_bundle(directory, current_status="published")

            self.assertTrue((directory / "draft.md").is_file())
            self.assertFalse(store.quarantine_root.exists())


if __name__ == "__main__":
    unittest.main()
