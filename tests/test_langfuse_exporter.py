"""Tests for the Langfuse native-format span exporter.

Langfuse v2 has no OTLP endpoint, so spans are translated to its batch
ingestion format. These tests pin the translation and the never-raise policy.
"""

import base64
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Status, StatusCode

from src.langfuse_exporter import LangfuseSpanProcessor


def _make_processor(**kw) -> LangfuseSpanProcessor:
    opts = dict(
        host="http://localhost:3000", public_key="pk-lf-x", secret_key="sk-lf-y"
    )
    opts.update(kw)
    return LangfuseSpanProcessor(**opts)


def _finished_span(name="span-name", error=False):
    """Produce a real finished ReadableSpan via a TracerProvider."""
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("router.model", "google/gemini-4-argon")
        span.set_attribute("router.confidence", 0.8)
        span.set_attribute("router.tags", ["a", "b"])
        if error:
            span.set_status(Status(StatusCode.ERROR, "boom"))
    # ReadableSpan remains valid after the context manager exits.
    return span


class TestInit(unittest.TestCase):
    def test_builds_basic_auth_header(self):
        proc = _make_processor()
        expected = base64.b64encode(b"pk-lf-x:sk-lf-y").decode()
        self.assertEqual(proc._auth, f"Basic {expected}")

    def test_ingestion_url(self):
        proc = _make_processor(host="http://localhost:3000/")
        self.assertEqual(proc._url, "http://localhost:3000/api/public/ingestion")


class TestToBatch(unittest.TestCase):
    def test_emits_trace_and_span(self):
        proc = _make_processor()
        span = _finished_span()
        batch = proc._to_batch(span)
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0]["type"], "trace-create")
        self.assertEqual(batch[1]["type"], "span-create")

    def test_span_references_trace(self):
        proc = _make_processor()
        span = _finished_span()
        trace, sp = proc._to_batch(span)
        self.assertEqual(sp["body"]["traceId"], trace["body"]["id"])
        self.assertTrue(sp["body"]["id"])
        self.assertNotEqual(sp["body"]["id"], trace["body"]["id"])

    def test_ids_are_hex(self):
        proc = _make_processor()
        trace, sp = proc._to_batch(_finished_span())
        int(trace["body"]["id"], 16)
        int(sp["body"]["id"], 16)

    def test_timestamps_are_iso_z(self):
        proc = _make_processor()
        trace, sp = proc._to_batch(_finished_span())
        self.assertTrue(trace["body"]["timestamp"].endswith("Z"))
        self.assertIn("startTime", sp["body"])
        # Parses as a real datetime.
        datetime.fromisoformat(trace["body"]["timestamp"].replace("Z", "+00:00"))

    def test_attributes_become_metadata(self):
        proc = _make_processor()
        _, sp = proc._to_batch(_finished_span())
        meta = sp["body"]["metadata"]
        self.assertEqual(meta["router.model"], "google/gemini-4-argon")
        self.assertEqual(meta["router.confidence"], 0.8)
        self.assertEqual(meta["router.tags"], ["a", "b"])

    def test_error_status_sets_level(self):
        proc = _make_processor()
        _, sp = proc._to_batch(_finished_span(error=True))
        self.assertEqual(sp["body"]["level"], "ERROR")
        self.assertEqual(sp["body"]["statusMessage"], "boom")

    def test_ok_span_has_no_error_level(self):
        proc = _make_processor()
        _, sp = proc._to_batch(_finished_span(error=False))
        self.assertNotIn("level", sp["body"])

    def test_trace_carries_input_output(self):
        """Attributes promoted so Langfuse's trace view shows content."""
        proc = _make_processor()
        provider = TracerProvider()
        t = provider.get_tracer("t")
        with t.start_as_current_span("chat") as span:
            span.set_attribute("input", "the prompt")
            span.set_attribute("output", "model=vendor/x")
        trace, sp = proc._to_batch(span)
        self.assertEqual(trace["body"]["input"], "the prompt")
        self.assertEqual(trace["body"]["output"], "model=vendor/x")
        self.assertEqual(sp["body"]["input"], "the prompt")

    def test_absent_input_output_stay_null(self):
        """Background spans (e.g. inventory_refresh) have no content."""
        proc = _make_processor()
        trace, sp = proc._to_batch(_finished_span("inventory_refresh"))
        self.assertNotIn("input", trace["body"])
        self.assertNotIn("output", trace["body"])


class TestOnEnd(unittest.TestCase):
    def test_posts_batch_to_ingestion(self):
        proc = _make_processor()
        span = _finished_span()
        mock_resp = MagicMock()
        mock_resp.status_code = 207
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("src.langfuse_exporter.httpx.Client", return_value=mock_client):
            proc.on_end(span)

        args, kwargs = mock_client.post.call_args
        self.assertEqual(args[0], "http://localhost:3000/api/public/ingestion")
        self.assertTrue(kwargs["headers"]["Authorization"].startswith("Basic "))
        self.assertEqual(len(kwargs["json"]["batch"]), 2)

    def test_http_error_is_logged_not_raised(self):
        proc = _make_processor()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "nope"
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.langfuse_exporter.httpx.Client", return_value=mock_client),
            patch("src.langfuse_exporter.logger") as log,
        ):
            proc.on_end(_finished_span())  # must not raise
        log.warning.assert_called()

    def test_network_error_is_swallowed(self):
        proc = _make_processor()
        mock_client = MagicMock()
        mock_client.post.side_effect = RuntimeError("connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.langfuse_exporter.httpx.Client", return_value=mock_client),
            patch("src.langfuse_exporter.logger") as log,
        ):
            proc.on_end(_finished_span())  # telemetry must never break a request
        log.warning.assert_called()


if __name__ == "__main__":
    unittest.main()


class TestTraceContentPolicy(unittest.TestCase):
    """ROUTER_TRACE_INPUT controls what user content reaches Langfuse."""

    def _span_and_decision(self):
        from src.models import ChatMessage
        from src.router import RoutingDecision

        msgs = [ChatMessage(role="user", content="x" * 2000)]
        decision = RoutingDecision("vendor/model", 0.55, "ok")
        span = MagicMock()
        return span, msgs, decision

    def _record(self, policy, routing_detail=True, chars=500):
        from src.main import _record_trace_content

        span, msgs, decision = self._span_and_decision()
        with patch("src.main.settings") as s:
            s.trace_input = policy
            s.trace_input_chars = chars
            s.trace_routing_detail = routing_detail
            _record_trace_content(span, msgs, "vendor/model", decision)
        return span

    def test_truncated_clips_prompt(self):
        span = self._record("truncated", chars=500)
        kwargs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        self.assertEqual(len(kwargs["input"]), 500)

    def test_full_keeps_whole_prompt(self):
        span = self._record("full")
        kwargs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        self.assertGreater(len(kwargs["input"]), 2000)

    def test_decision_policy_sends_no_prompt(self):
        span = self._record("decision")
        keys = [c.args[0] for c in span.set_attribute.call_args_list]
        self.assertNotIn("input", keys)

    def test_none_policy_sends_no_prompt(self):
        span = self._record("none")
        keys = [c.args[0] for c in span.set_attribute.call_args_list]
        self.assertNotIn("input", keys)

    def test_output_records_decision(self):
        span = self._record("truncated")
        kwargs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        self.assertIn("model=vendor/model", kwargs["output"])
        self.assertIn("confidence=0.55", kwargs["output"])

    def test_routing_detail_off_omits_output(self):
        span = self._record("truncated", routing_detail=False)
        keys = [c.args[0] for c in span.set_attribute.call_args_list]
        self.assertNotIn("output", keys)

    def test_no_decision_still_ok(self):
        """A request that never reached classification must not crash."""
        from src.main import _record_trace_content

        span = MagicMock()
        with patch("src.main.settings") as s:
            s.trace_input = "truncated"
            s.trace_input_chars = 500
            s.trace_routing_detail = True
            _record_trace_content(span, [], "static-fallback", None)
        keys = [c.args[0] for c in span.set_attribute.call_args_list]
        self.assertNotIn("output", keys)
