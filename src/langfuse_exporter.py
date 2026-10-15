"""Translate OpenTelemetry spans into Langfuse's native ingestion format.

Langfuse v2 exposes no OTLP endpoint: `POST /api/public/otel` does not exist,
and nothing serves gRPC on 4317. Its ingestion API accepts its own JSON batch
format instead:

    POST {host}/api/public/ingestion
    Authorization: Basic base64(public_key:secret_key)
    {"batch": [
        {"id", "type": "trace-create", "timestamp", "body": {...}},
        {"id", "type": "span-create",  "timestamp", "body": {...}}
    ]}

This module is a `SpanProcessor`, so it plugs into the existing TracerProvider
instead of replacing it. The standard OTLP/console exporters keep working
alongside it.
"""

import base64
import logging
import uuid
from typing import Any, Sequence

import httpx
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

logger = logging.getLogger(__name__)

INGEST_PATH = "/api/public/ingestion"


class LangfuseSpanProcessor(SpanProcessor):
    """Send finished spans to a Langfuse v2 ingestion endpoint."""

    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        timeout: float = 5.0,
    ) -> None:
        self._url = f"{host.rstrip('/')}{INGEST_PATH}"
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self._auth = f"Basic {token}"
        self._timeout = timeout

    def on_start(self, span, parent_context=None) -> None:  # noqa: ANN001, ARG002
        """Nothing to do until the span ends."""

    def on_end(self, span: ReadableSpan) -> None:
        """Translate one finished span and post it.

        Delivery must never break a request, so every failure is logged and
        swallowed.
        """
        try:
            batch = self._to_batch(span)
            if not batch:
                return
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    self._url,
                    headers={
                        "Authorization": self._auth,
                        "Content-Type": "application/json",
                    },
                    json={"batch": batch},
                )
            if resp.status_code >= 300:
                logger.warning(
                    f"Langfuse ingestion failed: {resp.status_code} {resp.text[:200]}"
                )
        except Exception as e:  # noqa: BLE001 - telemetry must never raise
            logger.warning(f"Langfuse ingestion error: {e}")

    def shutdown(self) -> None:
        """No background state to flush."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002
        return True

    def _to_batch(self, span: ReadableSpan) -> list[dict[str, Any]]:
        """Build the Langfuse batch events for one span."""
        ctx = span.get_span_context()
        if ctx is None:
            return []
        trace_id = self._hex(ctx.trace_id)
        span_id = self._hex(ctx.span_id)
        if span.start_time is None:
            return []
        timestamp = self._iso(span.start_time)

        attrs = span.attributes or {}
        # Conventional names, promoted so Langfuse's trace view shows them.
        trace_input = attrs.get("input")
        trace_output = attrs.get("output")

        trace_body: dict[str, Any] = {
            "id": trace_id,
            "timestamp": timestamp,
            "name": span.name,
            "metadata": {"service": "optirouter"},
        }
        if trace_input is not None:
            trace_body["input"] = trace_input
        if trace_output is not None:
            trace_body["output"] = trace_output

        trace_event = {
            "id": str(uuid.uuid4()),
            "type": "trace-create",
            "timestamp": timestamp,
            "body": trace_body,
        }

        span_body: dict[str, Any] = {
            "id": span_id,
            "traceId": trace_id,
            "name": span.name,
            "startTime": timestamp,
        }
        if span.end_time:
            span_body["endTime"] = self._iso(span.end_time)
        if trace_input is not None:
            span_body["input"] = trace_input
        if trace_output is not None:
            span_body["output"] = trace_output

        if attrs:
            span_body["metadata"] = {k: self._jsonable(v) for k, v in attrs.items()}

        status = span.status
        # Langfuse uses level: ERROR for failures, DEFAULT otherwise.
        if status is not None and status.status_code.name == "ERROR":
            span_body["level"] = "ERROR"
            if status.description:
                span_body["statusMessage"] = status.description

        span_event = {
            "id": str(uuid.uuid4()),
            "type": "span-create",
            "timestamp": timestamp,
            "body": span_body,
        }

        return [trace_event, span_event]

    @staticmethod
    def _hex(value: int) -> str:
        """OTel ids are ints; Langfuse wants lowercase hex."""
        return format(value, "032x")

    @staticmethod
    def _iso(nanos: int) -> str:
        """Convert OTel nanosecond timestamps to an ISO-8601 UTC string."""
        from datetime import datetime, timezone

        seconds = nanos / 1e9
        return (
            datetime.fromtimestamp(seconds, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        """Coerce OTel attribute values into JSON-serialisable ones."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [LangfuseSpanProcessor._jsonable(v) for v in value]
        return str(value)
