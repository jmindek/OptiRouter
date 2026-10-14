"""Model handoff logic for when selected model falls off discount list."""

import logging

from src.config import settings
from src.models import ModelInfo

logger = logging.getLogger(__name__)


def model_in_inventory(model: str, models: list[ModelInfo]) -> bool:
    """Check if a model ID is in the discounted inventory."""
    return any(m.id == model for m in models)


def handle_model_handoff(model: str, models: list[ModelInfo]) -> tuple[str, str | None]:
    """Handle when selected model is not in discounted inventory.

    Returns (model_to_use, error_detail_or_none).
    """
    handoff = settings.model_handoff

    if handoff == "stop":
        return model, (
            f"Model '{model}' is no longer discounted. "
            "Router is configured to stop. Please select a different model."
        )
    elif handoff == "next_best":
        best = models[0]  # already sorted by discount desc, price asc
        logger.info(f"Model '{model}' not discounted, switching to '{best.id}'")
        return best.id, None
    else:  # keep_current
        logger.info(f"Model '{model}' not discounted, continuing anyway")
        return model, None
