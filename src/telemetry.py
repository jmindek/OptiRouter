from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from src.config import settings


def init_telemetry():
    """Initialize OpenTelemetry SDK.

    Traces go to Langfuse when it is configured, otherwise to the console.
    Metrics go to the console.
    """
    provider = TracerProvider()

    langfuse_configured = bool(
        settings.langfuse_host
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    )

    if langfuse_configured:
        from src.langfuse_exporter import LangfuseSpanProcessor

        provider.add_span_processor(
            LangfuseSpanProcessor(
                host=settings.langfuse_host,
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
            )
        )
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)

    metrics.set_meter_provider(
        MeterProvider(
            metric_readers=[PeriodicExportingMetricReader(ConsoleMetricExporter())]
        )
    )


tracer = trace.get_tracer("OptiRouter")
meter = metrics.get_meter("OptiRouter")

# Metrics
request_duration = meter.create_histogram(
    name="router.request.duration",
    unit="ms",
    description="Request latency in milliseconds",
)
model_selections = meter.create_counter(
    name="router.model.selections",
    unit="1",
    description="Number of model selections by model ID",
)
rate_limit_events = meter.create_counter(
    name="router.rate_limits",
    unit="1",
    description="Number of rate limit (429) events",
)
inventory_refreshes = meter.create_counter(
    name="router.inventory.refreshes",
    unit="1",
    description="Number of inventory refresh attempts",
)
inventory_errors = meter.create_counter(
    name="router.inventory.errors",
    unit="1",
    description="Number of inventory refresh failures",
)
handoff_events = meter.create_counter(
    name="router.handoffs",
    unit="1",
    description="Number of model handoff events",
)
errors_total = meter.create_counter(
    name="router.errors.total",
    unit="1",
    description="Total error count",
)
