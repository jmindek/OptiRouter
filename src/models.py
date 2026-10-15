from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "router"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    tools: list | None = None
    tool_choice: str | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    stop: list[str] | str | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    response_format: dict | None = None
    seed: int | None = None
    user: str | None = None


class ModelInfo(BaseModel):
    id: str
    discount_pct: float
    provider: str
    input_price: float
    output_price: float
    context_length: int
    # Capability hints, captured from the OpenRouter /models payload.
    name: str = ""
    description: str = ""
    # What the model can consume/produce, e.g. ["text"] or ["text", "image"].
    input_modalities: list[str] = []
    output_modalities: list[str] = []
    # OpenAI-style params the model accepts, e.g. ["tools", "reasoning"].
    supported_parameters: list[str] = []
    # Benchmark scores (artificial_analysis). None when not published.
    intelligence_index: float | None = None
    coding_index: float | None = None
    agentic_index: float | None = None
    # Serving-endpoint facts, from the endpoint the router will actually use.
    # These describe the provider, not the model: the same model can be served
    # by many providers at different prices, speeds and reliabilities.
    endpoint_name: str = ""
    latency_p50_ms: float | None = None
    throughput_p50: float | None = None
    uptime_pct: float | None = None
    request_count: int = 0

    @property
    def effective_cost(self) -> float:
        """Effective blended price in USD per million tokens (75% input, 25% output)."""
        return (self.input_price * 0.75 + self.output_price * 0.25) * 1e6

    @property
    def has_reasoning(self) -> bool:
        """True if the model supports extended reasoning / thinking."""
        params = set(self.supported_parameters)
        if "reasoning" in params or "include_reasoning" in params:
            return True
        id_lower = self.id.lower()
        return any(
            k in id_lower for k in ("reasoning", "thinking", "-r1", "/r1", "o1", "o3")
        )

    @property
    def composite_score(self) -> float:
        """Composite capability index (0-100).

        Averages published benchmark indices (intelligence, coding, agentic).
        When benchmark scores are missing, estimates a capability tier score
        from parameter and model architecture hints so the decision model
        always has a non-zero, differentiated capability signal.
        """
        valid_benchmarks = [
            v
            for v in (self.intelligence_index, self.coding_index, self.agentic_index)
            if v is not None and v > 0.0
        ]
        if valid_benchmarks:
            return sum(valid_benchmarks) / len(valid_benchmarks)

        text = f"{self.id} {self.name} {self.description}".lower()
        if any(
            w in text
            for w in (
                "frontier",
                "flagship",
                "large",
                "70b",
                "405b",
                "671b",
                "deepseek-v3",
                "deepseek-v4",
                "opus",
            )
        ):
            return 60.0
        if any(w in text for w in ("medium", "32b", "35b", "27b", "sonnet", "pro")):
            return 45.0
        if any(
            w in text
            for w in ("mini", "flash", "small", "8b", "7b", "3b", "haiku", "nano")
        ):
            return 30.0
        return 35.0
