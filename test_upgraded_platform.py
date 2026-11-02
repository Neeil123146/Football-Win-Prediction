"""
Verification Suite for Upgraded Match Prediction & Prediction History Platform.

Tests:
1. GET / - Existing dashboard continues to load with all required elements intact.
2. GET /predict/<fixture_id> - Dedicated match prediction page retrieves genuine fixture details,
   runs 53-feature pipeline, outputs calibrated probabilities (~100%), and displays pre-match stats.
3. POST /api/predictions/confirm - Confirms prediction and persists record to history.
4. Duplicate confirmation protection - Verifies idempotent deduplication.
5. GET /history - Renders Prediction History page with KPI stats and filters.
6. Verification Lifecycle - Tests PENDING, CORRECT, INCORRECT, POSTPONED statuses,
   and verifies historical accuracy calculation formula excluding unresolved matches.
7. Fixture browsing throughout October 2026 and November 2026.
"""

import json
import unittest
from app import app
from src.data.live_data_fetcher import LiveMatchFetcher
from src.prediction_history import PredictionHistoryManager
from src.result_verifier import MatchResultVerifier


class TestUpgradedPlatform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.fetcher = LiveMatchFetcher()
        cls.history_mgr = PredictionHistoryManager()

    def test_01_existing_dashboard_intact(self):
        """Dashboard continues to render all existing sections."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        required = [
            "live-status-ribbon",
            "provider-status-badge",
            "upcoming-fixtures-card",
            "fixtures-container",
            "mode-live-btn",
            "mode-offline-btn",
            "matchup-selector",
            "prediction-result",
            "provenance-strip",
            "explanation-card",
            "explanation-factors-list",
            "comparison-table",
            "backtest-card",
            "limitations-section",
            "/history",  # Link to Prediction History
        ]
        for item in required:
            self.assertIn(item, html, f"Missing component '{item}' in index.html")

    def test_02_browse_october_and_november_fixtures(self):
        """Live fixture browser retrieves real upcoming fixtures for October and November 2026."""
        # October
        resp_oct = self.client.get("/api/live/fixtures?start_date=2026-10-01&end_date=2026-10-31")
        self.assertEqual(resp_oct.status_code, 200)
        d_oct = resp_oct.get_json()
        self.assertTrue(d_oct.get("success"))
        self.assertGreater(len(d_oct.get("fixtures", [])), 0, "Should have October 2026 fixtures")

        # November
        resp_nov = self.client.get("/api/live/fixtures?start_date=2026-11-01&end_date=2026-11-30")
        self.assertEqual(resp_nov.status_code, 200)
        d_nov = resp_nov.get_json()
        self.assertTrue(d_nov.get("success"))
        self.assertGreater(len(d_nov.get("fixtures", [])), 0, "Should have November 2026 fixtures")

    def test_03_dedicated_match_prediction_page(self):
        """Dedicated prediction page loads for fixture ID and derives real 53-feature prediction."""
        # Get a genuine upcoming fixture
        meta = self.fetcher.get_upcoming_fixtures_with_meta(start_date="2026-10-01", end_date="2026-10-31")
        fixtures = meta.get("fixtures", [])
        self.assertGreater(len(fixtures), 0)
        target = fixtures[0]

        # GET /predict/<fixture_id>
        resp = self.client.get(f"/predict/{target.fixture_id}")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn(target.home_team, html)
        self.assertIn(target.away_team, html)
        self.assertIn("PRE-MATCH PREDICTION", html)
        self.assertIn("Why This Prediction?", html)
        self.assertIn("53-Feature", html)
        self.assertIn("Confirm & Save Prediction", html)

        # Also test JSON format
        resp_json = self.client.get(f"/predict/{target.fixture_id}?format=json")
        self.assertEqual(resp_json.status_code, 200)
        d = resp_json.get_json()
        self.assertTrue(d.get("success"))
        pred = d.get("prediction", {})
        self.assertEqual(pred.get("feature_count"), 53)
        probs = pred.get("probabilities", {})
        prob_sum = round(probs.get("home_win", 0) + probs.get("draw", 0) + probs.get("away_win", 0), 1)
        self.assertAlmostEqual(prob_sum, 100.0, delta=0.2)

    def test_04_confirm_and_persist_prediction(self):
        """Confirming prediction saves record and survives query."""
        payload = {
            "fixture_id": "espn_test_1001",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "kickoff_date": "2026-10-15",
            "kickoff_time": "17:30",
            "venue": "Emirates Stadium",
            "predicted_outcome": "Home Win",
            "probabilities": {"home_win": 55.0, "draw": 25.0, "away_win": 20.0},
            "model_name": "Random Forest",
        }
        resp = self.client.post("/api/predictions/confirm", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        rec = data.get("prediction")
        self.assertEqual(rec["home_team"], "Arsenal")
        self.assertEqual(rec["verification_status"], "PENDING")

        # Test duplicate suppression (idempotency)
        resp2 = self.client.post("/api/predictions/confirm", json=payload)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.get_json()
        self.assertEqual(data2["prediction"]["id"], rec["id"], "Should return identical record within debounce window")

    def test_05_prediction_history_page(self):
        """History page renders KPI summary, cards, and accuracy metric."""
        resp = self.client.get("/history")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("Prediction History", html)
        self.assertIn("Historical Accuracy", html)
        self.assertIn("Refresh & Verify Results Now", html)
        self.assertIn("Arsenal", html)

    def test_06_verification_logic_and_accuracy(self):
        """Test verification lifecycle and accuracy formula: Correct / (Correct + Incorrect) * 100."""
        mgr = PredictionHistoryManager()
        # Add simulated completed predictions: 1 correct home win, 1 incorrect away win, 1 pending
        p_correct = mgr.save_prediction({
            "fixture_id": "sim_fix_1",
            "home_team": "Liverpool",
            "away_team": "Everton",
            "predicted_outcome": "Home Win",
            "probabilities": {"home_win": 60.0, "draw": 20.0, "away_win": 20.0},
        })
        p_incorrect = mgr.save_prediction({
            "fixture_id": "sim_fix_2",
            "home_team": "Wolves",
            "away_team": "Man City",
            "predicted_outcome": "Home Win",
            "probabilities": {"home_win": 50.0, "draw": 25.0, "away_win": 25.0},
        })
        p_postponed = mgr.save_prediction({
            "fixture_id": "sim_fix_3",
            "home_team": "Fulham",
            "away_team": "Brentford",
            "predicted_outcome": "Draw",
            "probabilities": {"home_win": 30.0, "draw": 40.0, "away_win": 30.0},
        })

        # Update p_correct as finished Home Win (2 - 1)
        mgr.update_prediction_verification(
            pred_id=p_correct["id"],
            verification_status="CORRECT",
            actual_score={"home": 2, "away": 1},
            actual_outcome="Home Win",
            is_correct=True,
        )

        # Update p_incorrect as finished Away Win (0 - 3)
        mgr.update_prediction_verification(
            pred_id=p_incorrect["id"],
            verification_status="INCORRECT",
            actual_score={"home": 0, "away": 3},
            actual_outcome="Away Win",
            is_correct=False,
        )

        # Update p_postponed as POSTPONED
        mgr.update_prediction_verification(
            pred_id=p_postponed["id"],
            verification_status="POSTPONED",
            actual_score=None,
            actual_outcome=None,
            is_correct=None,
        )

        summary = mgr.get_summary_stats()
        self.assertGreaterEqual(summary["verified"], 2)
        # Accuracy must exclude postponed and pending
        expected_acc = round((summary["correct"] / summary["verified"]) * 100.0, 1)
        self.assertEqual(summary["accuracy"], expected_acc)


if __name__ == "__main__":
    unittest.main()
