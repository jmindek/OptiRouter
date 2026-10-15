from src.models import ChatCompletionRequest, ModelInfo


def test_chat_request():
    req = ChatCompletionRequest(
        model="router",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert req.model == "router"
    assert len(req.messages) == 1


def test_model_info():
    m = ModelInfo(
        id="openrouter/deepseek/deepseek-v3.1",
        discount_pct=50.0,
        provider="openrouter",
        input_price=0.0000001,
        output_price=0.0000005,
        context_length=131072,
    )
    assert m.discount_pct == 50.0
    assert m.model_dump()["id"] == "openrouter/deepseek/deepseek-v3.1"
