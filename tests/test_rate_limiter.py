import asyncio
import time

from src.rate_limiter import RateLimiter


def test_rate_limiter_clear():
    rl = RateLimiter()
    rl._backoff_until["test-model"] = time.monotonic() + 100
    rl.clear("test-model")
    assert rl._backoff_until["test-model"] == 0.0


def test_rate_limiter_wait_no_block():
    rl = RateLimiter()
    start = time.monotonic()
    asyncio.run(rl.wait_if_needed("test-model"))
    elapsed = time.monotonic() - start
    assert elapsed < 0.1  # Should not block when no backoff


def test_rate_limiter_backoff():
    rl = RateLimiter(base_delay=0.01)
    rl._backoff_until["test-model"] = time.monotonic() + 0.01
    start = time.monotonic()
    asyncio.run(rl.wait_if_needed("test-model"))
    elapsed = time.monotonic() - start
    assert elapsed >= 0.01  # Should have waited


def test_rate_limiter_does_not_block_unthrottled_models():
    """Waiting on a throttled model must not hold the lock and block other models."""
    rl = RateLimiter()
    rl._backoff_until["throttled-model"] = time.monotonic() + 10.0

    async def check():
        task = asyncio.create_task(rl.wait_if_needed("throttled-model"))
        await asyncio.sleep(0.01)
        start = time.monotonic()
        await rl.wait_if_needed("fast-model")
        elapsed = time.monotonic() - start
        task.cancel()
        assert elapsed < 0.1

    asyncio.run(check())
