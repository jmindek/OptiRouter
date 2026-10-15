"""Decision model providers.

Abstracts the decision model call behind a provider interface so the same
POST /api/alpha/decisions contract is used regardless of backend.

Supported providers:
  - openrouter  : hosted via OpenRouter (default)
  - ollama      : local Ollama instance
  - vllm        : local vLLM inference server
  - systemone   : local /v1/systemone server (kev, ollaya, jevos)
"""

import abc
import logging
from typing import Any

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

# Path of the hosted Decisions API, relative to the OpenRouter host root.
DECISIONS_PATH = "/api/alpha/decisions"


class DecisionModelProvider(abc.ABC):
    """Interface for decision model backends."""

    @abc.abstractmethod
    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the decision model and return the response dict.

        The payload is the Decisions API format:
        {model, state, questions}.

        Returns the raw response dict (answers, model, etc.).
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...


class OpenRouterProvider(DecisionModelProvider):
    """Hosted decision model via OpenRouter."""

    @property
    def name(self) -> str:
        return "openrouter"

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = settings.api_key

        # The Decisions API lives on the OpenRouter host root, but ROUTER_API_BASE
        # conventionally includes the OpenAI-style /api/v1 suffix. Strip it, or
        # the path becomes /api/v1/api/alpha/decisions and 404s.
        api_base = settings.api_base.rstrip("/").removesuffix("/api/v1")
        endpoint = f"{api_base}{DECISIONS_PATH}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if resp.status_code != 200:
            raise ValueError(f"Decision model HTTP {resp.status_code}: {resp.text}")

        return resp.json()


class OllamaProvider(DecisionModelProvider):
    """Local decision model via Ollama."""

    @property
    def name(self) -> str:
        return "ollama"

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = settings.ollama_base
        model = settings.ollama_model or settings.model

        # Translate Decisions API payload → Ollama chat format
        system_msg = (
            payload.get("questions", {})
            .get("target", {})
            .get("instructions", "Choose the best model.")
        )
        criteria = payload.get("questions", {}).get("target", {}).get("criteria", {})
        user_msg = "Available candidate models:\n" + "\n".join(
            f"- {k}: {v}" for k, v in criteria.items()
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        endpoint = f"{base.rstrip('/')}/api/chat"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                endpoint,
                json={"model": model, "messages": messages, "stream": False},
            )

        if resp.status_code != 200:
            raise ValueError(f"Ollama HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        # Ollama returns {"message": {"role": "...", "content": "..."}}
        content = data.get("message", {}).get("content", "")

        # Parse the model's response — it should contain a model ID.
        # Try to extract a model ID from the response text.
        choice = self._extract_choice(content, criteria)
        if not choice:
            raise ValueError(
                f"Could not parse choice from Ollama response: {content[:200]}"
            )

        return {
            "answers": {
                "target": {
                    "choice": choice,
                    "confidence": 0.5,  # Ollama doesn't provide calibrated confidence
                }
            },
            "model": model,
        }

    @staticmethod
    def _extract_choice(content: str, criteria: dict) -> str | None:
        """Extract a model ID from the Ollama response text."""
        content = content.strip()
        # Direct match against criteria keys
        for key in criteria:
            if key.lower() in content.lower() or content.lower().startswith(
                key.lower()
            ):
                return key
        # Try to find any criteria key in the response
        for key in criteria:
            if key in content:
                return key
        return None


class VLLMProvider(DecisionModelProvider):
    """Local decision model via vLLM."""

    @property
    def name(self) -> str:
        return "vllm"

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = settings.vllm_base
        model = settings.vllm_model or settings.model

        # Translate Decisions API payload → OpenAI-compatible chat format
        system_msg = (
            payload.get("questions", {})
            .get("target", {})
            .get("instructions", "Choose the best model.")
        )
        criteria = payload.get("questions", {}).get("target", {}).get("criteria", {})
        user_msg = "Available candidate models:\n" + "\n".join(
            f"- {k}: {v}" for k, v in criteria.items()
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        endpoint = f"{base.rstrip('/')}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                endpoint,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 200,
                    "temperature": 0.0,
                },
            )

        if resp.status_code != 200:
            raise ValueError(f"vLLM HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        criteria_keys = list(criteria.keys())
        choice = OllamaProvider._extract_choice(
            content, dict(zip(criteria_keys, criteria.values()))
        )
        if not choice:
            raise ValueError(
                f"Could not parse choice from vLLM response: {content[:200]}"
            )

        return {
            "answers": {
                "target": {
                    "choice": choice,
                    "confidence": 0.5,
                }
            },
            "model": model,
        }


class SystemOneProvider(DecisionModelProvider):
    """Local decision model via /v1/systemone wire protocol.

    Compatible with kev, ollaya, jevos, and any server implementing the
    System One decision protocol. Returns typed, calibrated decisions.
    """

    @property
    def name(self) -> str:
        return "systemone"

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = settings.systemone_base
        model = settings.systemone_model or settings.model

        # Translate Decisions API payload → /v1/systemone format
        target = payload.get("questions", {}).get("target", {})
        instructions = target.get("instructions", "Choose the best model.")
        criteria = target.get("criteria", {})

        # Build state from the Decisions API payload
        state = {
            "request": payload.get("state", {}),
            "preference": payload.get("preference", ""),
        }

        # Build SystemOne questions
        questions = {
            "target": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {k: v for k, v in criteria.items()},
            }
        }

        endpoint = f"{base.rstrip('/')}/v1/systemone"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                endpoint,
                json={
                    "model": model,
                    "state": state,
                    "questions": questions,
                },
            )

        if resp.status_code != 200:
            raise ValueError(f"SystemOne HTTP {resp.status_code}: {resp.text}")

        data = resp.json()

        # Extract typed decision from SystemOne response
        answers = data.get("answers", {})
        target_answer = answers.get("target", {})

        choice = target_answer.get("choice")
        confidence = target_answer.get("confidence", 0.5)

        if not choice:
            raise ValueError(
                f"Could not extract choice from SystemOne response: {str(data)[:200]}"
            )

        answer: dict[str, Any] = {
            "choice": choice,
            "confidence": confidence,
        }
        # SystemOne returns the full distribution; keep it so callers can see
        # runner-ups instead of only the argmax.
        probabilities = target_answer.get("probabilities")
        if probabilities:
            answer["probabilities"] = probabilities

        return {
            "answers": {"target": answer},
            "model": model,
        }


def get_provider() -> DecisionModelProvider:
    """Return the configured decision model provider."""
    provider_name = settings.decision_model_provider.lower()

    providers: dict[str, type[DecisionModelProvider]] = {
        "openrouter": OpenRouterProvider,
        "ollama": OllamaProvider,
        "vllm": VLLMProvider,
        "systemone": SystemOneProvider,
    }

    cls = providers.get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown decision model provider: {provider_name}. "
            f"Supported: {list(providers.keys())}"
        )

    return cls()
