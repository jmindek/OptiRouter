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
