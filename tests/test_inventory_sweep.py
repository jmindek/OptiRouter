import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.inventory import _is_chat_capable, fetch_discounted_models
from src.models import ModelInfo


def _mi(**kw) -> ModelInfo:
    base = dict(
        id="v/m",
        discount_pct=50.0,
        provider="v",
        input_price=0.0,
        output_price=0.0,
        context_length=4096,
        output_modalities=["text"],
    )
    base.update(kw)
    return ModelInfo(**base)


class TestIsChatCapable(unittest.TestCase):
    def test_text_output_ok(self):
        self.assertTrue(_is_chat_capable(_mi()))

    def test_audio_only_rejected(self):
        self.assertFalse(_is_chat_capable(_mi(output_modalities=["audio"])))

    def test_video_only_rejected(self):
        self.assertFalse(_is_chat_capable(_mi(output_modalities=["video"])))

    def test_empty_modalities_rejected(self):
        # Placeholder rows like "docs/quickstart" carry no modality.
        self.assertFalse(_is_chat_capable(_mi(output_modalities=[])))

    def test_zero_context_rejected(self):
        self.assertFalse(_is_chat_capable(_mi(context_length=0)))

    def test_text_plus_image_ok(self):
        self.assertTrue(_is_chat_capable(_mi(output_modalities=["text", "image"])))


MODELS_PAYLOAD = {
    "data": [
        {
            "id": "inception/mercury-2.5",
            "name": "Inception: Mercury 2.5",
            "description": "Mercury 2.5 is a fast reasoning model.",
            "pricing": {"prompt": "0.00000004", "completion": "0.00000015"},
            "context_length": 260000,
            "top_provider": {"name": "Inception"},
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
            "supported_parameters": ["tools"],
            "benchmarks": {"artificial_analysis": {"intelligence_index": 12.3}},
        },
        {
            "id": "upstage/solar-pro4",
            "name": "Upstage: Solar Pro 4",
            "pricing": {"prompt": "0.00000009", "completion": "0.00000036"},
            "context_length": 524288,
            "top_provider": {"name": "Upstage"},
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
        },
        {
            # Non-chat model: audio output only. Discounted, but unusable.
            "id": "elevenlabs/eleven-v4",
            "name": "ElevenLabs v4",
            "pricing": {"prompt": "0.00000001", "completion": "0"},
            "context_length": 0,
            "top_provider": {"name": "ElevenLabs"},
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["audio"],
            },
        },
    ]
}


def _endpoint_payload(discount: float, prompt: str, completion: str, ctx: int) -> dict:
    return {
        "data": {
            "endpoints": [
                {
                    "provider": "prov",
                    "pricing": {
                        "discount": discount,
                        "prompt": prompt,
                        "completion": completion,
                    },
                    "context_length": ctx,
                }
            ]
        }
    }


class TestSweepIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_discounted_models_sweeps_endpoints(self):
        endpoints = {
            "inception/mercury-2.5": _endpoint_payload(
                0.8, "0.00000004", "0.00000015", 260000
            ),
            "upstage/solar-pro4": _endpoint_payload(
                0.7, "0.00000009", "0.00000036", 524288
            ),
            # Audio model: discounted, must be filtered out.
            "elevenlabs/eleven-v4": _endpoint_payload(0.5, "0.00000001", "0", 0),
            # Below the default 50% threshold.
            "low-discount/model": _endpoint_payload(0.2, "0.1", "0.2", 8192),
        }

        async def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if url.endswith("/models"):
                resp.json.return_value = MODELS_PAYLOAD
            else:
                mid = url.split("/models/")[1].removesuffix("/endpoints")
                resp.json.return_value = endpoints.get(mid, {"data": {"endpoints": []}})
            return resp

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "models.json"
            with patch("src.inventory.settings.cache_file", str(cache_file)):
                mock_client = MagicMock()
                mock_client.get = AsyncMock(side_effect=mock_get)
                models = await fetch_discounted_models(client=mock_client)

                # Drops the 20% (below threshold) and the audio model.
                self.assertEqual(len(models), 2)
                self.assertEqual(models[0].id, "inception/mercury-2.5")
                self.assertEqual(models[0].discount_pct, 80.0)
                self.assertEqual(models[0].context_length, 260000)
                # Capability hints must be captured for the decision model.
                self.assertEqual(models[0].name, "Inception: Mercury 2.5")
                self.assertIn("reasoning", models[0].description)
                self.assertEqual(models[0].output_modalities, ["text"])
                self.assertEqual(models[0].supported_parameters, ["tools"])
                self.assertEqual(models[0].intelligence_index, 12.3)

                self.assertEqual(models[1].id, "upstage/solar-pro4")
                self.assertEqual(models[1].discount_pct, 70.0)

                self.assertNotIn("elevenlabs/eleven-v4", [m.id for m in models])
                self.assertTrue(cache_file.exists())

    async def test_falls_back_to_cache_when_sweep_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "models.json"
            with patch("src.inventory.settings.cache_file", str(cache_file)):
                from src.inventory import save_cache

                save_cache(
                    [
                        ModelInfo(
                            id="cached/model",
                            discount_pct=60.0,
                            provider="p",
                            input_price=0.001,
                            output_price=0.002,
                            context_length=8192,
                            output_modalities=["text"],
                        )
                    ]
                )

                mock_client = MagicMock()
                mock_client.get = AsyncMock(side_effect=RuntimeError("boom"))
                models = await fetch_discounted_models(client=mock_client)

                self.assertEqual([m.id for m in models], ["cached/model"])

    def test_pick_endpoint_respects_configured_floors(self):
        from src.inventory import _pick_endpoint

        eps = [
            {
                "provider_name": "HighDiscountLowUptime",
                "pricing": {"discount": 0.8},
                "latency_last_30m": {"p50": 400.0},
                "uptime_last_1d": 90.0,
                "perf_last_30m_by_workload": {
                    "text_generation": {"request_count": 500}
                },
            },
            {
                "provider_name": "ModerateDiscountHealthy",
                "pricing": {"discount": 0.5},
                "latency_last_30m": {"p50": 500.0},
                "uptime_last_1d": 98.0,
                "perf_last_30m_by_workload": {
                    "text_generation": {"request_count": 200}
                },
            },
        ]

        # Default floor (uptime >= 95, reqs >= 100) picks ModerateDiscountHealthy
        with (
            patch("src.inventory.settings.min_endpoint_uptime", 95.0),
            patch("src.inventory.settings.min_endpoint_requests", 100),
        ):
            picked = _pick_endpoint(eps)
            assert picked is not None
            self.assertEqual(picked["provider_name"], "ModerateDiscountHealthy")

        # Lowered uptime floor (uptime >= 85) allows HighDiscountLowUptime
        with (
            patch("src.inventory.settings.min_endpoint_uptime", 85.0),
            patch("src.inventory.settings.min_endpoint_requests", 100),
        ):
            picked = _pick_endpoint(eps)
            assert picked is not None
            self.assertEqual(picked["provider_name"], "HighDiscountLowUptime")


if __name__ == "__main__":
    unittest.main()
