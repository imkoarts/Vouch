from __future__ import annotations

import unittest
from itertools import pairwise

from app.domain.state_machine import (
    ALLOWED_DRAFT_TRANSITIONS,
    DraftStatus,
    InvalidDraftStatusError,
    InvalidDraftTransitionError,
    allowed_targets,
    assert_transition_allowed,
    can_transition,
)


class DraftStateMachineTests(unittest.TestCase):
    def test_status_values_match_product_contract_exactly(self) -> None:
        self.assertEqual(
            {status.value for status in DraftStatus},
            {
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
            },
        )

    def test_every_state_has_an_explicit_allowlist_entry(self) -> None:
        self.assertEqual(set(ALLOWED_DRAFT_TRANSITIONS), set(DraftStatus))
        self.assertTrue(
            all(status not in targets for status, targets in ALLOWED_DRAFT_TRANSITIONS.items())
        )

    def test_review_and_publication_happy_path_is_allowed(self) -> None:
        path = (
            DraftStatus.GENERATED,
            DraftStatus.NEEDS_REVIEW,
            DraftStatus.APPROVED,
            DraftStatus.PUBLISH_PENDING_CONFIRMATION,
            DraftStatus.PUBLISHED,
            DraftStatus.REMOTE_DELETED,
        )
        for current, target in pairwise(path):
            self.assertTrue(can_transition(current, target))
            self.assertIs(assert_transition_allowed(current, target), target)

    def test_edit_after_approval_returns_to_review(self) -> None:
        self.assertTrue(can_transition("approved", "needs_review"))
        self.assertFalse(can_transition("approved", "published"))

    def test_failed_publish_requires_return_to_approved_before_new_attempt(self) -> None:
        self.assertTrue(can_transition("publish_failed", "approved"))
        self.assertFalse(can_transition("publish_failed", "publish_pending_confirmation"))

    def test_remote_and_local_deleted_states_are_terminal(self) -> None:
        self.assertEqual(allowed_targets(DraftStatus.DELETED), frozenset())
        self.assertEqual(allowed_targets(DraftStatus.REMOTE_DELETED), frozenset())

    def test_forbidden_transition_raises_safe_specific_exception(self) -> None:
        with self.assertRaises(InvalidDraftTransitionError) as captured:
            assert_transition_allowed("needs_review", "published")
        self.assertEqual(captured.exception.current, DraftStatus.NEEDS_REVIEW)
        self.assertEqual(captured.exception.target, DraftStatus.PUBLISHED)

    def test_unknown_status_does_not_echo_untrusted_value(self) -> None:
        secret_value = "not-a-status-with-secret-token"
        with self.assertRaises(InvalidDraftStatusError) as captured:
            can_transition(secret_value, DraftStatus.APPROVED)
        self.assertNotIn(secret_value, str(captured.exception))

    def test_allowlist_mapping_is_read_only(self) -> None:
        with self.assertRaises(TypeError):
            ALLOWED_DRAFT_TRANSITIONS[DraftStatus.DELETED] = frozenset(  # type: ignore[index]
                {DraftStatus.APPROVED}
            )


if __name__ == "__main__":
    unittest.main()
