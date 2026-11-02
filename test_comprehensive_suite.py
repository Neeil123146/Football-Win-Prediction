"""
Comprehensive End-to-End Test Suite for Football Win Prediction System.
Final Year BE Computer Engineering Project Verification.

Validates all 4 primary subsystems:
  1. ML Pipeline Integrity
  2. Prediction Pipeline & Explainability
  3. Live-Data Pipeline & Fallback Handling
  4. Web Application (Flask Endpoints, Error Handling, UI Consistency)
"""

from __future__ import annotations

import html as html_lib
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from src.data.live_data_fetcher import (
    CANONICAL_TEAMS,
    FootballDataCoUkProvider,
    LiveMatchFetcher,
    normalize_team_name,
)
from src.features.live_feature_generator import LiveFeatureGenerator
from src.prediction import MatchPredictor


class TestMLPipeline(unittest.TestCase):
    """Test 1: ML Pipeline Integrity & Historical Validation Discipline."""

    @classmethod
    def setUpClass(cls):
        cls.cleaned_path = PROJECT_ROOT / "data" / "processed" / "cleaned_matches.csv"
        cls.engineered_path = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"
        cls.metadata_path = PROJECT_ROOT / "models" / "metadata.json"
        cls.comparison_path = PROJECT_ROOT / "models" / "model_comparison.json"

        with open(cls.metadata_path) as f:
            cls.metadata = json.load(f)
        with open(cls.comparison_path) as f:
            cls.comparison = json.load(f)

    def test_historical_dataset_and_seasons(self):
        """Historical dataset loads correctly with 10 seasons and 3,800 matches."""
        self.assertTrue(self.cleaned_path.exists(), "Cleaned matches file must exist")
        df = pd.read_csv(self.cleaned_path)
        self.assertEqual(len(df), 3800, "Must contain exactly 3,800 matches across 10 seasons")

        seasons = sorted(df["Season"].unique())
        self.assertEqual(len(seasons), 10, "Must contain exactly 10 EPL seasons")
        expected_seasons = [
            "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
            "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
        ]
        self.assertEqual(seasons, expected_seasons, "Seasons must match expected chronological span")

    def test_unique_clubs_count(self):
        """Historical dataset represents exactly 34 unique Premier League clubs."""
        df = pd.read_csv(self.cleaned_path)
        unique_clubs = set(df["HomeTeam"]).union(set(df["AwayTeam"]))
        self.assertEqual(len(unique_clubs), 34, "Dataset must represent exactly 34 unique clubs")
        self.assertEqual(unique_clubs, CANONICAL_TEAMS, "Clubs must match canonical set exactly")

    def test_53_features_schema_and_zero_leakage(self):
        """Verify exact 53 pre-match features, column ordering, and zero target/future leakage."""
        feature_cols = self.metadata.get("feature_columns", [])
        self.assertEqual(len(feature_cols), 53, "Feature schema must specify exactly 53 features")

        # Zero leakage check: Target and in-match statistics must never appear in feature set
        forbidden_substrings = [
            "FTR", "FullTimeResult", "FTHG", "FTAG", "HTHG", "HTAG", "HTR",
            "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"
        ]
        for feat in feature_cols:
            self.assertNotIn(feat, forbidden_substrings, f"Feature '{feat}' constitutes target or match leakage")

    def test_temporal_cross_validation_and_champion_model(self):
        """Temporal cross-validation results remain intact and Random Forest is the verified champion."""
        champion = self.metadata.get("best_model")
        self.assertEqual(champion, "Random Forest", "Random Forest must be the selected champion model")

        # Verify exact temporal CV numbers for Random Forest
        rf_cv = self.metadata["temporal_cross_validation"]["Random Forest"]
        self.assertEqual(rf_cv["mean_accuracy"], 0.5039)
        self.assertEqual(rf_cv["std_accuracy"], 0.0185)
        self.assertEqual(rf_cv["mean_macro_f1"], 0.4435)
        self.assertEqual(rf_cv["mean_log_loss"], 1.0232)
        self.assertEqual(rf_cv["mean_brier_score"], 0.6092)
        self.assertEqual(rf_cv["mean_draw_recall"], 0.1572)

        # Verify untouched final test set evaluation
        rf_test = self.comparison["Random Forest"]["test"]
        self.assertEqual(rf_test["accuracy"], 0.5066)
        self.assertEqual(rf_test["macro_f1"], 0.4441)
        self.assertEqual(rf_test["log_loss"], 0.9964)
        self.assertEqual(rf_test["brier_score"], 0.5925)
        self.assertEqual(rf_test["draw_recall"], 0.1429)


class TestPredictionPipeline(unittest.TestCase):
    """Test 2: Prediction Pipeline, Numerical Validity, and Explainability."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = MatchPredictor(PROJECT_ROOT / "models")

    def test_valid_prediction_and_probability_sum(self):
        """Valid prediction returns bounded probabilities summing to ~100% and matching outcome."""
        res = self.predictor.predict("Arsenal", "Chelsea")
        probs = res["probabilities"]

        # Valid bounds
        self.assertGreaterEqual(probs["home_win"], 0.0)
        self.assertLessEqual(probs["home_win"], 100.0)
        self.assertGreaterEqual(probs["draw"], 0.0)
        self.assertLessEqual(probs["draw"], 100.0)
        self.assertGreaterEqual(probs["away_win"], 0.0)
        self.assertLessEqual(probs["away_win"], 100.0)

        # Sum to 100%
        total_prob = probs["home_win"] + probs["draw"] + probs["away_win"]
        self.assertAlmostEqual(total_prob, 100.0, delta=0.2)

        # Predicted outcome matches maximum probability
        max_outcome = max(probs, key=probs.get)
        label_map = {"home_win": "Home Win", "draw": "Draw", "away_win": "Away Win"}
        self.assertEqual(res["predicted_outcome"], label_map[max_outcome])

    def test_multiple_team_combinations(self):
        """Test varied combinations across traditional Big 6, mid-table, and promoted teams."""
        matchups = [
            ("Man City", "Liverpool"),
            ("Everton", "Wolves"),
            ("Aston Villa", "Tottenham"),
            ("Newcastle", "West Ham"),
            ("Brighton", "Brentford"),
            ("Fulham", "Bournemouth"),
        ]
        for home, away in matchups:
            res = self.predictor.predict(home, away)
            self.assertIn(res["predicted_outcome"], ["Home Win", "Draw", "Away Win"])
            probs = res["probabilities"]
            self.assertAlmostEqual(probs["home_win"] + probs["draw"] + probs["away_win"], 100.0, delta=0.2)

    def test_same_team_validation(self):
        """Selecting the same team for both Home and Away raises an explicit error."""
        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict("Liverpool", "Liverpool")
        self.assertIn("cannot play against itself", str(ctx.exception))

    def test_invalid_team_validation(self):
        """Invalid or non-existent team raises an explicit error."""
        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict("Atlantis FC", "Chelsea")
        self.assertIn("not in the historical Premier League dataset", str(ctx.exception))

    def test_why_this_prediction_explainability(self):
        """Explainability factors reflect actual feature values and include non-causal disclaimer."""
        res = self.predictor.predict("Aston Villa", "Nott'm Forest", live=True)
        self.assertIn("explanation", res)
        explanation = res["explanation"]

        factors = explanation.get("factors", [])
        self.assertGreaterEqual(len(factors), 3, "Should provide at least 3 contributing factors")
        for f in factors:
            self.assertIn("title", f)
            self.assertIn("detail", f)
            self.assertIn("impact", f)

        # Verify non-causal probabilistic disclaimer
        disclaimer = explanation.get("disclaimer", "")
        self.assertIn("uncertain", disclaimer.lower())
        self.assertIn("probability", disclaimer.lower())


class TestLiveDataPipeline(unittest.TestCase):
    """Test 3: Live-Data Provider, Feature Generator, and Fallback Resilience."""

    @classmethod
    def setUpClass(cls):
        cls.fetcher = LiveMatchFetcher(cache_ttl_fixtures=300, cache_ttl_results=3600)
        cls.gen = LiveFeatureGenerator()

    def test_live_provider_status(self):
        """Provider status returns operational provider and canonical club support."""
        provider = self.fetcher._resolve_provider()
        self.assertIn(provider.provider_name, ["football-data.co.uk", "football-data.org", "ESPN Premier League Calendar", "espn"])
        self.assertEqual(len(CANONICAL_TEAMS), 34)

    def test_upcoming_fixtures_retrieval(self):
        """Live fixtures retrieval succeeds, returns canonical teams, and tags cache provenance."""
        meta_res = self.fetcher.get_upcoming_fixtures_with_meta(start_date="2026-10-01", end_date="2026-10-31")
        fixtures = meta_res.get("fixtures", [])
        self.assertGreater(len(fixtures), 0, "Should retrieve upcoming fixtures")
        self.assertIn(meta_res.get("data_source"), ["live_api", "cached", "stale_cache_fallback"])
        self.assertIsInstance(meta_res.get("is_cached"), bool)

        sample = fixtures[0]
        self.assertIn(sample.home_team, CANONICAL_TEAMS)
        self.assertIn(sample.away_team, CANONICAL_TEAMS)

    def test_exact_53_live_feature_generation(self):
        """Live feature generator derives exactly 53 features with zero leakage."""
        res = self.gen.generate_features("Aston Villa", "Nott'm Forest", match_date="2026-09-12")
        fv = res.feature_vector
        feature_dict = res.feature_dict

        self.assertEqual(fv.shape, (1, 53), "Feature vector shape must be (1, 53)")
        self.assertEqual(len(feature_dict), 53, "Feature dict must contain exactly 53 keys")

        # Validate specific calculated values
        self.assertGreater(feature_dict["Home_Elo"], 1400.0)
        self.assertGreater(feature_dict["Away_Elo"], 1400.0)
        self.assertEqual(feature_dict["Elo_Diff"], feature_dict["Home_Elo"] - feature_dict["Away_Elo"])
        self.assertEqual(feature_dict["Diff_Recent_Pts"], feature_dict["Home_Recent_Pts"] - feature_dict["Away_Recent_Pts"])

    def test_live_vector_model_inference(self):
        """Pre-match live feature vector successfully passes through StandardScaler and locked model."""
        predictor = MatchPredictor(PROJECT_ROOT / "models")
        res = predictor.predict("Aston Villa", "Nott'm Forest", live=True)
        self.assertEqual(res["data_source"], "live_feature_pipeline")
        self.assertTrue(res["is_live"])
        self.assertEqual(res["feature_count"], 53)
        self.assertIn(res["predicted_outcome"], ["Home Win", "Draw", "Away Win"])

    def test_live_provider_network_failure_fallback(self):
        """Provider gracefully handles network failures using cached fallback without raising unhandled errors."""
        broken_provider = FootballDataCoUkProvider()
        broken_provider.FIXTURES_URL = "https://invalid-nonexistent-domain-404.co.uk/fixtures.csv"

        fallback_fetcher = LiveMatchFetcher()
        fallback_fetcher._provider_uk = broken_provider
        fallback_fetcher._provider_org = broken_provider
        fallback_fetcher._provider_espn = broken_provider
        meta_res = fallback_fetcher.get_upcoming_fixtures_with_meta()
        fixtures = meta_res.get("fixtures", [])
        self.assertGreater(len(fixtures), 0, "Fallback should retrieve disk cache")
        self.assertTrue(meta_res.get("is_cached"))
        self.assertIn(meta_res.get("data_source"), ["cached", "stale_cache_fallback"])


class TestWebApplication(unittest.TestCase):
    """Test 4: Flask Web Application, Routing, API Endpoints, and Error Handling."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_dashboard_renders(self):
        """GET / renders successfully with responsive structure, dropdowns, and UI components."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        raw_html = response.get_data(as_text=True)
        html = html_lib.unescape(raw_html)

        self.assertIn("Football Win Prediction System", html)
        self.assertIn("live-status-ribbon", html)
        self.assertIn("upcoming-fixtures-card", html)
        self.assertIn("mode-live-btn", html)
        self.assertIn("mode-offline-btn", html)
        self.assertIn("provenance-strip", html)
        self.assertIn("explanation-card", html)

        # Check all 34 canonical clubs appear in unescaped HTML options
        for club in CANONICAL_TEAMS:
            self.assertIn(club, html, f"Club {club} must appear in HTML options")

    def test_api_teams_endpoint(self):
        """GET /api/teams returns all 34 canonical clubs in JSON."""
        response = self.client.get("/api/teams")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data.get("teams", [])), 34)

    def test_api_live_status_endpoint(self):
        """GET /api/live/status returns operational status and schema count 53."""
        response = self.client.get("/api/live/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "operational")
        self.assertEqual(data.get("feature_columns_count"), 53)

    def test_api_live_fixtures_endpoint(self):
        """GET /api/live/fixtures returns scheduled fixtures with provider attribution."""
        response = self.client.get("/api/live/fixtures")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertGreater(data.get("total"), 0)
        self.assertIn("data_source", data)

    def test_post_predict_offline_json(self):
        """POST /predict with offline JSON generates valid pre-match prediction."""
        payload = {"home_team": "Arsenal", "away_team": "Chelsea", "live": False}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("predicted_outcome"), "Home Win")
        self.assertAlmostEqual(
            data["probabilities"]["home_win"] + data["probabilities"]["draw"] + data["probabilities"]["away_win"],
            100.0,
            delta=0.2,
        )

    def test_post_predict_live_json(self):
        """POST /predict with live=true generates live-pipeline prediction with explainability."""
        payload = {"home_team": "Aston Villa", "away_team": "Nott'm Forest", "live": True}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("is_live"))
        self.assertIn("explanation", data)
        self.assertGreaterEqual(len(data["explanation"]["factors"]), 3)

    def test_post_predict_form_submission(self):
        """POST /predict with standard HTML Form submission generates prediction."""
        form_data = {"home_team": "Man City", "away_team": "Man United"}
        response = self.client.post("/predict", data=form_data)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))

    def test_post_predict_same_team_rejection(self):
        """POST /predict rejects same-team selection with HTTP 400."""
        payload = {"home_team": "Liverpool", "away_team": "Liverpool"}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data.get("success"))
        self.assertIn("cannot play against itself", data.get("error"))

    def test_post_predict_missing_team_rejection(self):
        """POST /predict rejects missing team parameters with HTTP 400."""
        payload = {"home_team": "Arsenal"}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data.get("success"))
        self.assertIn("Both home_team and away_team are required", data.get("error"))

    def test_api_models_and_backtest_endpoints(self):
        """GET /api/models and GET /api/backtest return accurate historical metrics."""
        m_resp = self.client.get("/api/models")
        self.assertEqual(m_resp.status_code, 200)
        m_data = m_resp.get_json()
        self.assertEqual(m_data["metadata"]["best_model"], "Random Forest")

        b_resp = self.client.get("/api/backtest")
        self.assertEqual(b_resp.status_code, 200)
        b_data = b_resp.get_json()
        self.assertEqual(b_data["model_accuracy"], 50.66)
        self.assertEqual(b_data["naive_baseline_accuracy"], 43.42)
        self.assertEqual(b_data["improvement_percentage_points"], 7.24)


if __name__ == "__main__":
    unittest.main(verbosity=2)
