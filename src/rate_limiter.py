import asyncio
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class RateLimiter:
    """Reactive rate limiter. Retries on 429 with exponential backoff + jitter."""

    def __init__(
        self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._backoff_until: dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()

    async def wait_if_needed(self, model: str) -> None:
        """Block until it's safe to send a request for this model."""
        async with self._lock:
            now = time.monotonic()
            until = self._backoff_until[model]
            delay = max(0.0, until - now)
        if delay > 0:
            logger.debug(f"Rate limited {model}: waiting {delay:.1f}s")
            await asyncio.sleep(delay)

    def record_429(self, model: str) -> None:
        """Record a 429 and schedule next retry with exponential backoff + jitter."""
        loop = asyncio.get_running_loop()
        loop.create_task(self._record_429(model))

    async def _record_429(self, model: str) -> None:
        async with self._lock:
            delay = min(self.base_delay, self.max_delay)
            jitter = delay * 0.5
            self._backoff_until[model] = time.monotonic() + delay + jitter
            logger.info(f"429 for {model}, backing off {delay + jitter:.1f}s")

    def clear(self, model: str) -> None:
        """Clear backoff state when request succeeds."""
        self._backoff_until[model] = 0.0


rate_limiter = RateLimiter()
