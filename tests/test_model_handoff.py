import unittest
from unittest.mock import patch

from src.handoff import handle_model_handoff, model_in_inventory
from src.models import ModelInfo


class TestModelInInventory(unittest.TestCase):
    def test_model_found(self):
        models = [
            ModelInfo(
                id="model-a",
                discount_pct=50.0,
                provider="p",
                input_price=0.001,
                output_price=0.002,
                context_length=4096,
            ),
            ModelInfo(
                id="model-b",
                discount_pct=75.0,
                provider="p",
                input_price=0.001,
                output_price=0.002,
                context_length=4096,
            ),
        ]
        self.assertTrue(model_in_inventory("model-a", models))
        self.assertTrue(model_in_inventory("model-b", models))

    def test_model_not_found(self):
        models = [
            ModelInfo(
                id="model-a",
                discount_pct=50.0,
                provider="p",
                input_price=0.001,
                output_price=0.002,
                context_length=4096,
            ),
        ]
        self.assertFalse(model_in_inventory("model-c", models))

    def test_empty_inventory(self):
        self.assertFalse(model_in_inventory("any", []))


class TestHandleModelHandoff(unittest.TestCase):
    def setUp(self):
        self.models = [
            ModelInfo(
                id="best-model",
                discount_pct=75.0,
                provider="p",
                input_price=0.001,
                output_price=0.002,
                context_length=4096,
            ),
            ModelInfo(
                id="second-model",
                discount_pct=50.0,
                provider="p",
                input_price=0.001,
                output_price=0.002,
                context_length=4096,
            ),
        ]

    def test_handoff_stop(self):
        with patch("src.handoff.settings") as mock_settings:
            mock_settings.model_handoff = "stop"
            model, error = handle_model_handoff("old-model", self.models)
            self.assertEqual(model, "old-model")
            self.assertIsNotNone(error)
            self.assertIn("no longer discounted", error)

    def test_handoff_next_best(self):
        with patch("src.handoff.settings") as mock_settings:
            mock_settings.model_handoff = "next_best"
            model, error = handle_model_handoff("old-model", self.models)
            self.assertEqual(model, "best-model")
            self.assertIsNone(error)

    def test_handoff_keep_current(self):
        with patch("src.handoff.settings") as mock_settings:
            mock_settings.model_handoff = "keep_current"
            model, error = handle_model_handoff("old-model", self.models)
            self.assertEqual(model, "old-model")
            self.assertIsNone(error)

    def test_next_best_uses_first_model(self):
        """First model in list is the best (sorted by discount desc)."""
        with patch("src.handoff.settings") as mock_settings:
            mock_settings.model_handoff = "next_best"
            model, _ = handle_model_handoff("old-model", self.models)
            self.assertEqual(model, "best-model")

    def test_parse_cron_validates_5_fields(self):
        from src.main import _parse_cron

        self.assertEqual(
            _parse_cron("0 * * * *"),
            {
                "minute": "0",
                "hour": "*",
                "day": "*",
                "month": "*",
                "day_of_week": "*",
            },
        )
        with self.assertRaises(ValueError):
            _parse_cron("0 0 * * * *")  # 6 fields
        with self.assertRaises(ValueError):
            _parse_cron("0 * *")  # 3 fields


if __name__ == "__main__":
    unittest.main()
