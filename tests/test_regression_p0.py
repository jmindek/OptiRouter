import unittest
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from src.config import settings
from src.inventory import fetch_discounted_models
from src.main import forward_to_openrouter
from src.models import ChatCompletionRequest


class TestP0Regressions(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_discounted_models_concurrent(self):
        """Test concurrent endpoint fetching with semaphore."""
        import tempfile
        from pathlib import Path

        models_payload = {
            "data": [
                {
                    "id": "model-1",
                    "architecture": {"output_modalities": ["text"]},
                    "context_length": 4096,
                },
                {
                    "id": "model-2",
                    "architecture": {"output_modalities": ["text"]},
                    "context_length": 4096,
                },
            ]
        }
        ep1_payload = {
            "data": {
                "endpoints": [
                    {
                        "provider": "prov1",
                        "pricing": {
                            "discount": 0.6,
                            "prompt": "0.001",
                            "completion": "0.002",
                        },
                        "context_length": 4096,
                    }
                ]
            }
        }
        ep2_payload = {
            "data": {
                "endpoints": [
                    {
                        "provider": "prov2",
                        "pricing": {
                            "discount": 0.1,
                            "prompt": "0.001",
                            "completion": "0.002",
                        },
                        "context_length": 4096,
                    }
                ]
            }
        }

        async def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.status_code = 200

            if url.endswith("/models"):
                resp.json.return_value = models_payload
            elif "/models/model-1/endpoints" in url:
                resp.json.return_value = ep1_payload
            elif "/models/model-2/endpoints" in url:
                resp.json.return_value = ep2_payload
            else:
                resp.status_code = 404
                resp.json.return_value = {}
            return resp

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=mock_get)

        # Isolate the cache: this call saves the discounted models, and must
        # never touch the working data/models.json.
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "src.inventory.settings.cache_file",
                str(Path(tmp_dir) / "models.json"),
            ):
                discounted = await fetch_discounted_models(client=mock_client)
        self.assertEqual(len(discounted), 1)
        self.assertEqual(discounted[0].id, "model-1")
        self.assertEqual(discounted[0].discount_pct, 60.0)

    async def test_forward_url_no_v1_duplication(self):
        """Regression test: Ensure openrouter base URL does not duplicate /v1."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        captured_url: str | None = None

        async def mock_post(url, **kwargs):
            nonlocal captured_url
            captured_url = url
            return mock_resp

        mock_client = MagicMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch.object(settings, "inventory_url", "https://openrouter.ai/api/v1"),
            patch.object(settings, "openrouter_api_key", "sk-test"),
        ):
            await forward_to_openrouter(
                messages=[{"role": "user", "content": "hi"}],
                model="model-1",
                stream=False,
                request=ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}]),
            )
            assert captured_url is not None
            self.assertIn("/v1/chat/completions", captured_url)
            self.assertNotIn("/v1/v1", captured_url)

    async def test_forward_to_openrouter_streaming_connection_lifecycle(self):
        """Regression test: the upstream connection must stay open while the
        StreamingResponse body is iterated.

        Returning the StreamingResponse while the stream context is still
        active closes the connection before FastAPI iterates the body, which
        truncates the stream and raises TransferEncodingError downstream.

        The fake response mimics a real one: reading from it after it has been
        closed raises, instead of quietly yielding. That is what makes this
        test fail when the connection-lifecycle bug is reintroduced; a mock
        whose close is a no-op cannot observe the bug at all.
        """
        fake_chunks = [
            b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        class FakeResponse:
            status_code = 200

            def __init__(self):
                self.closed = False

            async def aread(self):
                return b""

            async def aiter_raw(self):
                for chunk in fake_chunks:
                    if self.closed:
                        raise AssertionError(
                            "stream read after the connection was closed"
                        )
                    yield chunk

        class FakeStreamContext:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *exc):
                # A real stream is unusable once its context exits.
                self.response.closed = True
                return False

        fake_resp = FakeResponse()

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def stream(self, *args, **kwargs):
                return FakeStreamContext(fake_resp)

            async def aclose(self):
                fake_resp.closed = True

        with patch("httpx.AsyncClient", FakeClient):
            response = await forward_to_openrouter(
                messages=[{"role": "user", "content": "hi"}],
                model="model-1",
                stream=True,
                request=ChatCompletionRequest(
                    messages=[{"role": "user", "content": "hi"}], stream=True
                ),
            )

            response = cast(StreamingResponse, response)

            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

            self.assertEqual(len(chunks), 2)
            self.assertIn(b"hello", chunks[0])
            # Framing survives byte-for-byte, including the blank-line
            # separator that terminates each SSE event.
            self.assertEqual(b"".join(chunks), b"".join(fake_chunks))
            self.assertTrue(chunks[1].endswith(b"\n\n"))

    async def test_streaming_429_retries_and_records_backoff(self):
        """A 429 seen while opening the stream must be retried and recorded,
        exactly like the non-streaming path. Nothing has been sent at that
        point, so retrying is safe; before this was fixed a streaming 429 was
        turned into an empty 200 and never recorded."""
        from src.main import rate_limiter

        class FakeResponse:
            def __init__(self, status_code):
                self.status_code = status_code
                self.closed = False

            async def aread(self):
                return b'{"error": "rate limited"}'

            async def aiter_raw(self):
                if False:
                    yield b""

        class FakeStreamContext:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *exc):
                self.response.closed = True
                return False

        responses = [FakeResponse(429), FakeResponse(200)]

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def stream(self, *args, **kwargs):
                return FakeStreamContext(responses.pop(0))

            async def aclose(self):
                pass

        rate_limiter._backoff_until.clear()
        with patch("httpx.AsyncClient", FakeClient), patch(
            "asyncio.sleep", new=AsyncMock()
        ), patch.object(
            rate_limiter, "record_429", wraps=rate_limiter.record_429
        ) as record_spy:
            response = await forward_to_openrouter(
                messages=[{"role": "user", "content": "hi"}],
                model="stream-429-model",
                stream=True,
                request=ChatCompletionRequest(
                    messages=[{"role": "user", "content": "hi"}], stream=True
                ),
            )

        # The 429 was retried, and the successful attempt was returned.
        self.assertEqual(response.status_code, 200)
        # And the rate limit was recorded, so the next request backs off.
        # (A later success clears the timer, so assert on the recording, not
        # on the backoff still being set.)
        record_spy.assert_called_once_with("stream-429-model")

    async def test_streaming_upstream_error_raises_instead_of_empty_200(self):
        """An upstream error while opening the stream must surface as an
        error, not as a 200 with an empty body."""

        class FakeResponse:
            status_code = 500

            async def aread(self):
                return b'{"error": "boom"}'

            async def aiter_raw(self):
                if False:
                    yield b""

        class FakeStreamContext:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *exc):
                return False

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def stream(self, *args, **kwargs):
                return FakeStreamContext(FakeResponse())

            async def aclose(self):
                pass

        with patch("httpx.AsyncClient", FakeClient):
            with self.assertRaises(HTTPException) as ctx:
                await forward_to_openrouter(
                    messages=[{"role": "user", "content": "hi"}],
                    model="model-1",
                    stream=True,
                    request=ChatCompletionRequest(
                        messages=[{"role": "user", "content": "hi"}], stream=True
                    ),
                )

        self.assertEqual(ctx.exception.status_code, 502)


if __name__ == "__main__":
    unittest.main()
