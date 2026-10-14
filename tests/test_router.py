import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models import ChatMessage, ModelInfo
from src.router import (
    MAX_CANDIDATES,
    Requirements,
    RoutingDecision,
    _build_criteria,
    _describe_model,
    _estimate_tokens,
    _filter_candidates,
    _parse_decision,
    _rank_candidates,
    _render_state,
    _request_requirements,
    _truncate_middle,
    classify_request,
)

MODELS = [
    ModelInfo(
        id="anthropic/claude-3.5-sonnet",
        discount_pct=50.0,
        provider="anthropic",
        input_price=0.003,
        output_price=0.015,
        context_length=200000,
    ),
    ModelInfo(
        id="google/gemini-2.0-flash",
        discount_pct=75.0,
        provider="google",
        input_price=0.000075,
        output_price=0.0003,
        context_length=1048576,
    ),
]


def _response(body: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.text = str(body)
    return resp


def _patched_client(resp):
    """Patch httpx.AsyncClient so .post(*) returns `resp`."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return patch("src.router.httpx.AsyncClient", return_value=mock_client), mock_client


CLOUD_SETTINGS = {
    "model": "typesafe/jev-1.13",
    "api_base": "https://openrouter.ai/api/v1",
    "api_key": "sk-test",
    "mode": "cloud",
    "min_confidence": 0.30,
    "preference": "cost",
    "min_context_length": 131072,
    "weights": {
        "discount": 0.4,
        "latency": 0.2,
        "uptime": 0.2,
        "coding_index": 0.2,
        "intelligence_index": 0.0,
        "agentic_index": 0.0,
    },
}


def _patch_settings():
    m = MagicMock()
    for k, v in CLOUD_SETTINGS.items():
        setattr(m, k, v)
    return patch("src.router.settings", m)


class TestRenderState(unittest.TestCase):
    def test_roles_and_content(self):
        msgs = [
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="user", content="write a sort"),
        ]
        out = _render_state(msgs)
        self.assertIn("[system] be terse", out)
        self.assertIn("[user] write a sort", out)

    def test_content_parts(self):
        msgs = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ],
            )
        ]
        out = _render_state(msgs)
        self.assertIn("look at this", out)
        self.assertIn("<image_url>", out)

    def test_accepts_dicts(self):
        out = _render_state([{"role": "user", "content": "hi"}])
        self.assertIn("[user] hi", out)

    def test_empty(self):
        self.assertEqual(_render_state([]), "")

    def test_truncates_middle_of_long_transcript(self):
        msgs = [ChatMessage(role="user", content="start " + "x" * 100 + " end")]
        out = _truncate_middle(_render_state(msgs), limit=60)
        self.assertLessEqual(len(out), 120)  # limit plus the omission marker
        self.assertIn("omitted", out)


class TestRequirements(unittest.TestCase):
    def test_detects_image_part(self):
        msgs = [ChatMessage(role="user", content=[{"type": "image_url", "image_url": {}}])]
        self.assertTrue(_request_requirements(msgs).needs_image_input)

    def test_detects_tool_calls(self):
        msgs = [ChatMessage(role="assistant", tool_calls=[{"id": "1"}])]
        self.assertTrue(_request_requirements(msgs).needs_tools)

    def test_detects_tool_role(self):
        msgs = [ChatMessage(role="tool", content="result")]
        self.assertTrue(_request_requirements(msgs).needs_tools)

    def test_plain_text_has_no_requirements(self):
        msgs = [ChatMessage(role="user", content="hello")]
        req = _request_requirements(msgs)
        self.assertFalse(req.needs_image_input)
        self.assertFalse(req.needs_tools)

    def test_token_estimate_from_untruncated_text(self):
        msgs = [ChatMessage(role="user", content="a" * 4000)]
        self.assertGreaterEqual(_estimate_tokens(msgs), 900)


class TestFilterCandidates(unittest.TestCase):
    def _pool(self):
        return [
            ModelInfo(id="text/only", discount_pct=50, provider="p", input_price=0.001, output_price=0.002,
                      context_length=8192, input_modalities=["text"], supported_parameters=["tools"]),
            ModelInfo(id="vision/model", discount_pct=50, provider="p", input_price=0.001, output_price=0.002,
                      context_length=8192, input_modalities=["text", "image"], supported_parameters=[]),
            ModelInfo(id="small/ctx", discount_pct=50, provider="p", input_price=0.001, output_price=0.002,
                      context_length=4096, input_modalities=["text"], supported_parameters=[]),
        ]

    def test_image_filter(self):
        kept = _filter_candidates(self._pool(), Requirements(needs_image_input=True))
        self.assertEqual([m.id for m in kept], ["vision/model"])

    def test_tools_filter(self):
        kept = _filter_candidates(self._pool(), Requirements(needs_tools=True))
        self.assertEqual([m.id for m in kept], ["text/only"])

    def test_context_filter(self):
        kept = _filter_candidates(self._pool(), Requirements(min_context=8000))
        self.assertEqual([m.id for m in kept], ["text/only", "vision/model"])

    def test_no_requirements_keeps_all(self):
        kept = _filter_candidates(self._pool(), Requirements())
        self.assertEqual(len(kept), 3)

    def test_never_returns_empty(self):
        kept = _filter_candidates(self._pool(), Requirements(needs_tools=True, needs_image_input=True))
        self.assertEqual(len(kept), 3)  # falls back to the full pool


class TestDescribeModel(unittest.TestCase):
    def test_includes_capability_and_cost(self):
        m = ModelInfo(
            id="vendor/model-x",
            discount_pct=80.0,
            provider="vendor",
            input_price=0.000001,
            output_price=0.000002,
            context_length=131072,
            name="Vendor Model X",
            description="A frontier coding model. Second sentence.",
            input_modalities=["text", "image"],
            output_modalities=["text"],
            supported_parameters=["tools", "reasoning", "structured_outputs"],
        )
        hint = _describe_model(m)
        self.assertIn("Vendor Model X", hint)
        self.assertIn("A frontier coding model.", hint)
        self.assertNotIn("Second sentence", hint)  # only first sentence
        self.assertIn("accepts image", hint)
        self.assertIn("tool calling", hint)
        self.assertIn("reasoning", hint)
        self.assertIn("80% off", hint)
        self.assertIn("$1.00/Mtok in", hint)  # effective price tie-breaker

    def test_minimal_model(self):
        m = ModelInfo(
            id="vendor/bare",
            discount_pct=50.0,
            provider="",
            input_price=0.0,
            output_price=0.0,
            context_length=4096,
        )
        hint = _describe_model(m)
        self.assertIn("vendor/bare", hint)  # falls back to id
        self.assertIn("50% off", hint)


class TestBuildCriteria(unittest.TestCase):
    def test_sorted_by_discount_then_price(self):
        criteria = _build_criteria(MODELS)
        # Both have no benchmark scores, so rank falls back to discount order:
        # gemini 75% first, claude 50% second.
        keys = list(criteria)
        self.assertEqual(keys[0], "google/gemini-2.0-flash")
        self.assertEqual(keys[1], "anthropic/claude-3.5-sonnet")

    def test_criteria_is_id_to_description(self):
        criteria = _build_criteria(MODELS)
        self.assertIn("google/gemini-2.0-flash", criteria)
        desc = criteria["google/gemini-2.0-flash"]
        self.assertIn("google", desc)
        self.assertIn("75%", desc)

    def test_cap_at_max_candidates(self):
        many = [
            ModelInfo(
                id=f"p/m{i}",
                discount_pct=50.0,
                provider="p",
                input_price=0.001,
                output_price=0.002,
                context_length=4096,
            )
            for i in range(MAX_CANDIDATES + 20)
        ]
        self.assertEqual(len(_build_criteria(many)), MAX_CANDIDATES)

    def test_empty(self):
        self.assertEqual(_build_criteria([]), {})

    def test_blended_keeps_cheap_and_strong(self):
        """The capped pool must span cheap-weak to strong-expensive."""
        pool = [
            ModelInfo(id="cheap/weak", discount_pct=80, provider="p", input_price=0.00001,
                      output_price=0.00002, context_length=8192, coding_index=20.0),
            ModelInfo(id="cheap/weak2", discount_pct=75, provider="p", input_price=0.00002,
                      output_price=0.00004, context_length=8192, coding_index=25.0),
            ModelInfo(id="mid/ok", discount_pct=60, provider="p", input_price=0.0002,
                      output_price=0.0004, context_length=8192, coding_index=55.0),
            ModelInfo(id="strong/pricey", discount_pct=50, provider="p", input_price=0.002,
                      output_price=0.004, context_length=8192, coding_index=77.0),
            ModelInfo(id="strong/pricey2", discount_pct=50, provider="p", input_price=0.003,
                      output_price=0.006, context_length=8192, coding_index=76.0),
            ModelInfo(id="filler", discount_pct=50, provider="p", input_price=0.0001,
                      output_price=0.0002, context_length=8192, coding_index=30.0),
        ]
        with patch("src.router.MAX_CANDIDATES", 4):
            kept = list(_build_criteria(pool))
        self.assertIn("strong/pricey", kept)  # best score survives
        self.assertIn("cheap/weak", kept)  # best discount survives


class TestParseDecision(unittest.TestCase):
    def test_ok(self):
        data = {
            "model": "jev-1.13.0",
            "answers": {
                "target": {
                    "type": "choice",
                    "choice": "google/gemini-2.0-flash",
                    "probabilities": {"google/gemini-2.0-flash": 0.81},
                    "confidence": 0.7,
                }
            },
            "usage": {"input_tokens": 312, "output_tokens": 0},
        }
        with _patch_settings():
            d = _parse_decision(data)
        self.assertEqual(d.model_id, "google/gemini-2.0-flash")
        self.assertAlmostEqual(d.confidence, 0.7)
        self.assertEqual(d.reason, "ok")

    def test_low_confidence_abstains(self):
        data = {
            "answers": {
                "target": {
                    "type": "choice",
                    "choice": "google/gemini-2.0-flash",
                    "confidence": 0.2,
                }
            }
        }
        with _patch_settings():
            d = _parse_decision(data)
        self.assertIsNone(d.model_id)
        self.assertEqual(d.reason, "low_confidence")
        self.assertAlmostEqual(d.confidence, 0.2)

    def test_missing_answers(self):
        with _patch_settings():
            d = _parse_decision({})
        self.assertIsNone(d.model_id)
        self.assertEqual(d.reason, "parse_error")

    def test_answers_not_dict(self):
        with _patch_settings():
            d = _parse_decision({"answers": {"target": "nope"}})
        self.assertEqual(d.reason, "parse_error")

    def test_no_choice_field(self):
        with _patch_settings():
            d = _parse_decision({"answers": {"target": {"type": "choice"}}})
        self.assertEqual(d.reason, "parse_error")

    def test_non_numeric_confidence_treated_as_zero(self):
        data = {
            "answers": {
                "target": {"type": "choice", "choice": "m", "confidence": "high"}
            }
        }
        with _patch_settings():
            d = _parse_decision(data)
        self.assertEqual(d.reason, "low_confidence")
        self.assertEqual(d.confidence, 0.0)


class TestClassifyRequest(unittest.IsolatedAsyncioTestCase):
    async def test_no_models(self):
        d = await classify_request([ChatMessage(role="user", content="hi")], [])
        self.assertIsNone(d.model_id)
        self.assertEqual(d.reason, "no_models")

    async def test_no_api_key(self):
        with patch("src.router.settings") as m:
            m.mode = "auto"
            m.api_key = ""
            m.omlx_base = "http://localhost:8000"
            d = await classify_request([ChatMessage(role="user", content="hi")], MODELS)
        self.assertIsNone(d.model_id)
        self.assertEqual(d.reason, "no_api_key")

    async def test_success_posts_correct_shape(self):
        body = {
            "model": "jev-1.13.0",
            "answers": {
                "target": {
                    "type": "choice",
                    "choice": "google/gemini-2.0-flash",
                    "probabilities": {"google/gemini-2.0-flash": 0.81},
                    "confidence": 0.9,
                }
            },
        }
        ctx, mock_client = _patched_client(_response(body))
        with ctx, _patch_settings():
            d = await classify_request([ChatMessage(role="user", content="summarize this")], MODELS)

        self.assertEqual(d.model_id, "google/gemini-2.0-flash")
        self.assertEqual(d.reason, "ok")

        # Verify URL and payload shape.
        args, kwargs = mock_client.post.call_args
        self.assertEqual(args[0], "https://openrouter.ai/api/alpha/decisions")
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "typesafe/jev-1.13")
        self.assertIn("[user] summarize this", payload["state"]["request"])
        self.assertNotIn("task", payload["state"])  # no keyword tag
        self.assertIn("cheaper", payload["state"]["preference"])
        self.assertIn("effective price", payload["state"]["preference"])
        q = payload["questions"]["target"]
        self.assertEqual(q["type"], "choice")
        self.assertIn("instructions", q)
        self.assertIn("Tie-breaker", q["instructions"])
        self.assertIn("google/gemini-2.0-flash", q["criteria"])
        # The cost tie-breaker must be visible per candidate.
        self.assertIn("/Mtok in", q["criteria"]["google/gemini-2.0-flash"])
        # No chat fields must leak into a decision request.
        self.assertNotIn("messages", payload)
        self.assertNotIn("stream", payload)

    async def test_preference_none_omits_tiebreaker(self):
        body = {
            "answers": {
                "target": {
                    "type": "choice",
                    "choice": "google/gemini-2.0-flash",
                    "confidence": 0.5,
                }
            }
        }
        ctx, mock_client = _patched_client(_response(body))
        with ctx, _patch_settings() as m:
            m.preference = "none"
            await classify_request([ChatMessage(role="user", content="hi")], MODELS)
        _, kwargs = mock_client.post.call_args
        self.assertNotIn("Tie-breaker", kwargs["json"]["questions"]["target"]["instructions"])
        self.assertEqual(kwargs["json"]["state"]["preference"], "none")

    async def test_http_error(self):
        ctx, _ = _patched_client(_response({"error": "boom"}, status=400))
        with ctx, _patch_settings():
            d = await classify_request([ChatMessage(role="user", content="hi")], MODELS)
        self.assertIsNone(d.model_id)
        self.assertEqual(d.reason, "http_error")

    async def test_invalid_json(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        ctx, _ = _patched_client(resp)
        with ctx, _patch_settings():
            d = await classify_request([ChatMessage(role="user", content="hi")], MODELS)
        self.assertIsNone(d.model_id)
        self.assertEqual(d.reason, "parse_error")


if __name__ == "__main__":
    unittest.main()
