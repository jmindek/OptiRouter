from src.inventory import load_cache, save_cache
from src.models import ModelInfo


def test_save_load_cache(tmp_path):
    models = [
        ModelInfo(
            id="openrouter/deepseek/deepseek-v3.1",
            discount_pct=50.0,
            provider="openrouter",
            input_price=0.0000001,
            output_price=0.0000005,
            context_length=131072,
        ),
        ModelInfo(
            id="openrouter/google/gemini-2.0-flash",
            discount_pct=75.0,
            provider="openrouter",
            input_price=0.000000075,
            output_price=0.0000003,
            context_length=1048576,
        ),
    ]

    cache_file = tmp_path / "models.json"
    save_cache(models, str(cache_file))

    loaded = load_cache(str(cache_file))
    assert len(loaded) == 2
    assert loaded[0].id == "openrouter/deepseek/deepseek-v3.1"
    assert loaded[1].id == "openrouter/google/gemini-2.0-flash"


def test_load_empty_cache(tmp_path):
    cache_file = tmp_path / "nonexistent.json"
    loaded = load_cache(str(cache_file))
    assert loaded == []
