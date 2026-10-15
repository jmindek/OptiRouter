from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # System One decision model (OpenRouter ID)
    model: str = Field(default="typesafe/jev-1.13", validation_alias="ROUTER_MODEL")
    api_base: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="ROUTER_API_BASE"
    )
    api_key: str = Field(default="", validation_alias="ROUTER_API_KEY")

    # OpenRouter
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    inventory_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="ROUTER_INVENTORY_URL"
    )

    # Routing mode: cloud, local, auto
    mode: str = Field(default="auto", validation_alias="ROUTER_MODE")

    # Local oMLX fallback
    omlx_base: str = Field(
        default="http://127.0.0.1:8000", validation_alias="ROUTER_OMLX_BASE"
    )

    # Discount threshold (0.0-1.0)
    min_discount: float = Field(default=0.25, validation_alias="ROUTER_MIN_DISCOUNT")

    # Minimum decision-model confidence (0.0-1.0) to accept a routing pick.
    # Below this the router abstains and uses low_confidence_choice.
    # Observed range for model selection is ~0.28-0.40; 0.55 abstained on
    # essentially every request. See README for the tuning note.
    min_confidence: float = Field(
        default=0.30, validation_alias="ROUTER_MIN_CONFIDENCE"
    )

    # Which feature to pick the best candidate by when the decision model
    # abstains due to low confidence. Options match the feature weights.
    low_confidence_choice: Literal[
        "discount",
        "latency",
        "uptime",
        "coding_index",
        "intelligence_index",
        "agentic_index",
        "cost",
        "price",
        "throughput",
        "context_window",
        "context_length",
        "reasoning",
        "composite_score",
        "model_tier",
    ] = Field(default="discount", validation_alias="ROUTER_LOW_CONFIDENCE_CHOICE")

    # Routing preference handed to the decision model as a tie-breaker.
    # "cost" prefers the cheaper candidate when models are otherwise comparable.
    # Live testing showed this is the only preference that raises confidence;
    # open/closed-weight and reasoning preferences lowered it.
    preference: str = Field(default="cost", validation_alias="ROUTER_PREFERENCE")

    # Fallback
    fallback_mode: str = Field(
        default="static", validation_alias="ROUTER_FALLBACK_MODE"
    )
    static_fallback: str = Field(
        default="openrouter/deepseek/deepseek-v3.1",
        validation_alias="ROUTER_STATIC_FALLBACK",
    )

    # Model handoff when selected model falls off discount list
    model_handoff: str = Field(
        default="next_best", validation_alias="ROUTER_MODEL_HANDOFF"
    )

    # Minimum context length in tokens
    min_context_length: int = Field(
        default=131072, validation_alias="ROUTER_MIN_CONTEXT_LENGTH"
    )

    # Endpoint trustworthiness floors for inventory sweep.
    # Endpoints below these floors are not trustworthy enough to record perf
    # numbers from (falling back to max discount overall only when none qualify).
    min_endpoint_uptime: float = Field(
        default=95.0, validation_alias="ROUTER_MIN_ENDPOINT_UPTIME"
    )
    min_endpoint_requests: int = Field(
        default=100, validation_alias="ROUTER_MIN_ENDPOINT_REQUESTS"
    )

    # Feature weights for model decision evaluation
    # Accepted format: string "discount=0.4,latency=0.2,..." or dict
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "discount": 0.3,
            "cost": 0.2,
            "latency": 0.15,
            "throughput": 0.15,
            "uptime": 0.1,
            "coding_index": 0.1,
            "intelligence_index": 0.0,
            "agentic_index": 0.0,
            "context_window": 0.0,
            "reasoning": 0.0,
            "composite_score": 0.0,
        },
        validation_alias="ROUTER_WEIGHTS",
    )

    @field_validator("weights", mode="before")
    @classmethod
    def _parse_weights(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("{"):
                import json

                return json.loads(v)
            parsed = {}
            for item in v.split(","):
                if "=" in item:
                    k, val = item.split("=", 1)
                    parsed[k.strip()] = float(val.strip())
            return parsed
        return v

    # Cache
    cache_file: str = Field(
        default="data/models.json", validation_alias="ROUTER_CACHE_FILE"
    )
    # Inventory refresh cron. Hourly by default: the sweep is browserless and
    # costs ~4s for the whole catalogue, so keeping the pool at most an hour
    # stale is cheap.
    refresh_cron: str = Field(
        default="0 * * * *", validation_alias="ROUTER_REFRESH_CRON"
    )

    # Langfuse (v2) ingestion API. Langfuse v2 has no OTLP endpoint, so spans
    # are translated to its native batch format and POSTed to
    # {langfuse_host}/api/public/ingestion with basic auth. When unset, traces
    # go to the console.
    langfuse_host: str = Field(default="", validation_alias="LANGFUSE_HOST")
    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", validation_alias="LANGFUSE_SECRET_KEY")

    # What to record as a trace's input/output in Langfuse.
    #   none      - record nothing (traces show timing and model only)
    #   decision  - record the routing decision as output, no prompt content
    #   truncated - record a clipped prompt as input, plus the decision (default)
    #   full      - record the whole prompt
    # Prompts are user content, so the default deliberately clips them.
    trace_input: Literal["none", "decision", "truncated", "full"] = Field(
        default="truncated", validation_alias="ROUTER_TRACE_INPUT"
    )

    # Character cap when trace_input is "truncated".
    trace_input_chars: int = Field(
        default=500, validation_alias="ROUTER_TRACE_INPUT_CHARS"
    )

    # Record the routing decision detail (chosen model, confidence, reason,
    # candidate count) as span attributes. Router metadata, not user content.
    trace_routing_detail: bool = Field(
        default=True, validation_alias="ROUTER_TRACE_ROUTING_DETAIL"
    )

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
