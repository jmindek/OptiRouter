import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def _forbid_real_cache(tmp_path, monkeypatch):
    """Point the model cache at a temp file for every test.

    Tests that call the inventory sweep save a cache; without this they would
    overwrite the working data/models.json with fixture data. Tests that need
    a specific path still patch settings.cache_file themselves.
    """
    monkeypatch.setattr(settings, "cache_file", str(tmp_path / "models.json"))
