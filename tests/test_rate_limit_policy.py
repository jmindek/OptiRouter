"""Tests for the shared 429 policy used by both forwarding transports.

The streaming and non-streaming paths had this logic duplicated; these tests
pin the policy once so the two cannot drift.
"""

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from src.main import _apply_rate_limit_policy


class TestApplyRateLimitPolicy(unittest.TestCase):
    def test_non_429_proceeds(self):
        with patch("src.main.rate_limiter") as rl:
            retry = _apply_rate_limit_policy(200, "m", attempt=0, max_retries=3)
        self.assertFalse(retry)
        rl.record_429.assert_not_called()

    def test_429_retries_while_attempts_remain(self):
        with (
            patch("src.main.rate_limiter") as rl,
            patch("src.main.rate_limit_events") as events,
        ):
            retry = _apply_rate_limit_policy(429, "m", attempt=0, max_retries=3)
        self.assertTrue(retry)
        rl.record_429.assert_called_once_with("m")
        events.add.assert_called_once_with(1, {"model": "m"})

    def test_429_on_last_attempt_raises(self):
        with patch("src.main.rate_limiter") as rl:
            with self.assertRaises(HTTPException) as ctx:
                _apply_rate_limit_policy(429, "m", attempt=2, max_retries=3)
        self.assertEqual(ctx.exception.status_code, 429)
        rl.record_429.assert_called_once_with("m")

    def test_records_before_deciding_to_retry(self):
        """The limit is recorded on every 429, including the final one."""
        with patch("src.main.rate_limiter") as rl:
            _apply_rate_limit_policy(429, "m", attempt=1, max_retries=3)
        rl.record_429.assert_called_once_with("m")

    def test_other_error_status_proceeds(self):
        """500 is not a rate limit: the caller's raise_for_status handles it."""
        with patch("src.main.rate_limiter") as rl:
            retry = _apply_rate_limit_policy(500, "m", attempt=0, max_retries=3)
        self.assertFalse(retry)
        rl.record_429.assert_not_called()


if __name__ == "__main__":
    unittest.main()
