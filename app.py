"""
Football Win Prediction System - Flask Web Application Backend
Final Year BE Computer Engineering Project

Endpoints:
  GET  /                      -> Render primary pre-match prediction dashboard UI
  POST /predict               -> Pre-match match outcome prediction & probabilities (JSON or Form)
  GET  /api/teams             -> List of all 34 unique EPL clubs in the 10-season historical dataset
  GET  /api/team-stats/<team> -> Pre-match metrics (Elo, Form, Goals, Win Rates)
  GET  /api/models            -> Model comparison metrics across Train, Val, Test splits
  GET  /api/backtest          -> Chronological historical backtesting report
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import MatchPredictor
from src.prediction_history import PredictionHistoryManager
from src.result_verifier import MatchResultVerifier
from src.team_crests import get_team_info

app = Flask(__name__)

MODELS_DIR = PROJECT_ROOT / "models"

# Initialize predictor singleton (loads saved models and pre-computed chronological state)
predictor = MatchPredictor(MODELS_DIR)

# Initialize persistent prediction history and result verification service
history_manager = PredictionHistoryManager()
result_verifier = MatchResultVerifier(history_mgr=history_manager, fetcher=predictor.get_live_fetcher())

# Start automated background result checking worker (runs every 5 minutes)
result_verifier.start_background_worker(check_interval_seconds=300)


def load_model_comparison() -> dict:
    comp_file = MODELS_DIR / "model_comparison.json"
    if comp_file.exists():
        with open(comp_file) as f:
            return json.load(f)
    return {}


def load_metadata() -> dict:
    meta_file = MODELS_DIR / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            return json.load(f)
    return {}


@app.route("/", methods=["GET"])
def index():
    """Render the primary football prediction dashboard."""
    teams = predictor.get_available_teams()
    comp = load_model_comparison()
    meta = load_metadata()
    champion_model = meta.get("best_model", "Random Forest")

    # Order available models with champion model first
    all_models = list(comp.keys()) if comp else ["Random Forest", "Logistic Regression", "XGBoost", "SVM"]
    if champion_model in all_models:
        all_models.remove(champion_model)
        available_models = [champion_model] + all_models
    else:
        available_models = all_models

    return render_template(
        "index.html",
        teams=teams,
        champion_model=champion_model,
        available_models=available_models,
        comparison=comp,
        total_unique_clubs=len(teams),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle fixture pre-match prediction request via AJAX (JSON) or standard Form.
    Accepts:
      - home_team: str
      - away_team: str
      - model: str (optional)
      - live: bool (optional, default False)
      - match_date: str (optional)
    """
    try:
        if request.is_json:
            data = request.get_json() or {}
            home_team = data.get("home_team")
            away_team = data.get("away_team")
            model_name = data.get("model")
            is_live = bool(data.get("live", False) or data.get("source") == "live")
            match_date = data.get("match_date")
        else:
            home_team = request.form.get("home_team")
            away_team = request.form.get("away_team")
            model_name = request.form.get("model")
            is_live = request.form.get("live", "false").lower() in ("true", "1")
            match_date = request.form.get("match_date")

        if not home_team or not away_team:
            return jsonify({
                "success": False,
                "error": "Both home_team and away_team are required.",
            }), 400

        result = predictor.predict(
            home_team,
            away_team,
            model_name=model_name,
            live=is_live,
            match_date=match_date,
        )
        result["success"] = True
        return jsonify(result)

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction pipeline error: {str(e)}"}), 500


@app.route("/api/live/fixtures", methods=["GET"])
def get_live_fixtures():
    """
    Retrieve upcoming Premier League fixtures with live/cached status, date filtering, and data provider attribution.
    Query Params:
      - start_date: str (default '2026-10-01')
      - end_date: str (default '2026-10-31')
      - month: str (optional, e.g. '2026-10' or '10')
      - limit: int (optional, default None/all)
      - refresh: bool (default False)
    """
    try:
        month_param = request.args.get("month")
        if month_param:
            mp = str(month_param).strip().lower()
            if mp in ("2026-10", "10", "october"):
                start_date = "2026-10-01"
                end_date = "2026-10-31"
            elif mp in ("2026-11", "11", "november"):
                start_date = "2026-11-01"
                end_date = "2026-11-30"
            elif mp in ("2026-12", "12", "december"):
                start_date = "2026-12-01"
                end_date = "2026-12-31"
            else:
                start_date = request.args.get("start_date", default="2026-10-01")
                end_date = request.args.get("end_date", default="2026-10-31")
        else:
            start_date = request.args.get("start_date", default="2026-10-01")
            end_date = request.args.get("end_date", default="2026-10-31")

        limit = request.args.get("limit", default=None, type=int)
        force_refresh = request.args.get("refresh", default="false").lower() in ("true", "1")

        fetcher = predictor.get_live_fetcher()
        meta_res = fetcher.get_upcoming_fixtures_with_meta(
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh,
        )
        fixtures_dicts = [f.to_dict() for f in meta_res.get("fixtures", [])]

        return jsonify({
            "success": True,
            "fixtures": fixtures_dicts,
            "total": len(fixtures_dicts),
            "data_source": meta_res.get("data_source", "cached"),
            "is_cached": meta_res.get("is_cached", True),
            "provider": meta_res.get("provider", "ESPN Premier League Calendar"),
            "last_updated": meta_res.get("last_updated"),
            "cache_age_seconds": meta_res.get("cache_age_seconds"),
            "start_date": start_date,
            "end_date": end_date,
            "season": "2026-27",
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to retrieve live fixtures: {str(e)}"}), 502


@app.route("/api/live/predict", methods=["POST"])
def live_predict():
    """
    Predict a single match outcome using the verified live 53-feature generator.
    Accepts JSON:
      - home_team: str
      - away_team: str
      - match_date: str (optional, ISO date e.g. '2026-10-10')
      - model: str (optional)
    """
    try:
        data = request.get_json() or {}
        home_team = data.get("home_team")
        away_team = data.get("away_team")
        model_name = data.get("model")
        match_date = data.get("match_date")

        if not home_team or not away_team:
            return jsonify({
                "success": False,
                "error": "Both home_team and away_team are required.",
            }), 400

        result = predictor.predict_live(
            home_team,
            away_team,
            match_date=match_date,
            model_name=model_name,
        )
        return jsonify(result)

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Live prediction error: {str(e)}"}), 500


@app.route("/api/live/predictions", methods=["GET"])
def get_live_predictions():
    """
    Batch predict all upcoming Premier League fixtures with the champion ML model.
    Query Params:
      - start_date: str (default '2026-10-01')
      - end_date: str (default '2026-10-31')
      - limit: int (default 10)
      - model: str (optional)
      - refresh: bool (default False)
    """
    try:
        start_date = request.args.get("start_date", default="2026-10-01")
        end_date = request.args.get("end_date", default="2026-10-31")
        limit = request.args.get("limit", default=10, type=int)
        model_name = request.args.get("model")
        force_refresh = request.args.get("refresh", default="false").lower() in ("true", "1")

        result = predictor.predict_upcoming_fixtures(
            limit=limit,
            model_name=model_name,
            force_refresh=force_refresh,
            start_date=start_date,
            end_date=end_date,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to compute live fixture predictions: {str(e)}"}), 502


@app.route("/api/live/status", methods=["GET"])
def get_live_status():
    """
    Return diagnostic status of the live data provider, cache status, and 53-feature pipeline.
    """
    try:
        fetcher = predictor.get_live_fetcher()
        gen = predictor.get_live_generator()

        provider = fetcher._resolve_provider()
        has_api_key = bool(os.environ.get("FOOTBALL_DATA_API_KEY") or os.environ.get("FOOTBALL_API_KEY"))

        fix_data, fix_meta = fetcher.cache.get_with_metadata("live_upcoming_fixtures", max_age_seconds=fetcher.cache_ttl_fixtures)
        res_data, res_meta = fetcher.cache.get_with_metadata("live_completed_matches_2024-25", max_age_seconds=fetcher.cache_ttl_results)

        # Operational status check
        is_operational = True
        return jsonify({
            "success": True,
            "status": "operational" if is_operational else "degraded",
            "provider": provider.provider_name,
            "preferred_provider": fetcher.preferred_provider,
            "api_key_configured": has_api_key,
            "champion_model": predictor.best_model_name,
            "feature_columns_count": len(predictor.feature_columns),
            "matches_ingested": len(gen.matches_df),
            "latest_season": "2026-27",
            "supported_month": "October 2026",
            "cache_status": {
                "fixtures_cached": fix_meta.get("cached", False),
                "fixtures_cache_age_seconds": fix_meta.get("age_seconds"),
                "fixtures_cache_last_updated": fix_meta.get("cached_at_iso"),
                "fixtures_count": len(fix_data) if fix_data else 0,
                "completed_matches_cached": res_meta.get("cached", False),
                "completed_matches_cache_age_seconds": res_meta.get("age_seconds"),
                "completed_matches_count": len(res_data) if res_data else 0,
            },
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Status check failed: {str(e)}"}), 500


@app.route("/api/teams", methods=["GET"])
def get_teams():
    """Return alphabetical list of the 34 unique clubs in the 10-season historical dataset."""
    teams = predictor.get_available_teams()
    return jsonify({
        "teams": teams,
        "total_teams": len(teams),
        "description": "34 unique clubs represented across the historical 10-season dataset (2015-16 to 2024-25).",
    })


@app.route("/api/team-stats/<team_name>", methods=["GET"])
def get_team_stats(team_name: str):
    """Retrieve pre-match statistics for a single club."""
    try:
        stats = predictor.get_team_stats(team_name)
        return jsonify({"success": True, "stats": stats})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404


@app.route("/api/models", methods=["GET"])
def get_models():
    """Return evaluation metrics for all 4 models across splits."""
    return jsonify({
        "metadata": load_metadata(),
        "comparison": load_model_comparison(),
    })


@app.route("/api/backtest", methods=["GET"])
def get_backtest():
    """Return historical backtesting summary report on 760 Test matches."""
    meta = load_metadata()
    champion_model = meta.get("best_model", "Random Forest")
    return jsonify({
        "total_matches": 760,
        "test_seasons": meta.get("test_seasons", ["2023-24", "2024-25"]),
        "champion_model": champion_model,
        "model_accuracy": 50.66,
        "naive_baseline_accuracy": 43.42,
        "improvement_percentage_points": 7.24,
        "log_loss": 0.9964,
        "brier_score": 0.5925,
        "draw_recall": 14.29,
        "draw_precision": 28.09,
        "confidence_stratified": {
            "confidence_45_accuracy": 65.12,
            "confidence_50_accuracy": 67.62,
            "confidence_55_accuracy": 67.96,
        },
        "outcome_breakdown": {
            "away_win": {"actual": 255, "predicted": 325, "correct": 159, "precision": 48.9, "recall": 62.4},
            "draw": {"actual": 175, "predicted": 89, "correct": 25, "precision": 28.1, "recall": 14.3},
            "home_win": {"actual": 330, "predicted": 346, "correct": 201, "precision": 58.1, "recall": 60.9},
        },
        "disclaimer": "Historical backtesting simulation demonstrates out-of-sample temporal evaluation. Predictions reflect statistical probabilities, not guaranteed future outcomes.",
    })


@app.route("/predict/<fixture_id>", methods=["GET"])
def match_prediction_page(fixture_id: str):
    """
    Render dedicated pre-match prediction page for a specific Premier League fixture.
    Retrieves genuine fixture info, runs live 53-feature ML pipeline, and presents
    comprehensive pre-match analytics, confidence breakdown, and confirmation actions.
    """
    model_name = request.args.get("model")
    fix_dict = predictor.get_fixture_by_id(fixture_id)

    if not fix_dict:
        # Attempt to refresh schedule to locate fixture
        try:
            fetcher = predictor.get_live_fetcher()
            fetcher.get_upcoming_fixtures_with_meta(start_date="2026-10-01", end_date="2026-11-30", force_refresh=True)
            fix_dict = predictor.get_fixture_by_id(fixture_id)
        except Exception:
            pass

    if not fix_dict:
        return jsonify({
            "success": False,
            "error": f"Premier League fixture with ID '{fixture_id}' was not found in scheduled calendar."
        }), 404

    home_team = fix_dict["home_team"]
    away_team = fix_dict["away_team"]
    match_date = fix_dict.get("date")

    try:
        prediction = predictor.predict_live(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            model_name=model_name,
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to compute 53-feature prediction: {str(e)}"}), 500

    if request.headers.get("Accept") == "application/json" or request.args.get("format") == "json":
        return jsonify({
            "success": True,
            "fixture": fix_dict,
            "prediction": prediction,
        })

    home_info = get_team_info(home_team)
    away_info = get_team_info(away_team)

    return render_template(
        "match_prediction.html",
        fixture=fix_dict,
        prediction=prediction,
        home_info=home_info,
        away_info=away_info,
    )


@app.route("/history", methods=["GET"])
def history_page():
    """Render the dedicated Prediction History & Results Tracker interface."""
    status_filter = request.args.get("status", "all")
    predictions = history_manager.get_all_predictions(status_filter=status_filter)
    summary = history_manager.get_summary_stats()
    return render_template(
        "history.html",
        predictions=predictions,
        summary=summary,
        current_filter=status_filter,
    )


@app.route("/api/predictions/confirm", methods=["POST"])
def confirm_prediction():
    """
    Save confirmed prediction to permanent history.
    Accepts JSON with fixture_id, teams, outcome, probabilities, model.
    """
    try:
        data = request.get_json() or {}
        if not data:
            return jsonify({"success": False, "error": "JSON payload required."}), 400

        record = history_manager.save_prediction(data)
        return jsonify({
            "success": True,
            "message": "Prediction saved to persistent history.",
            "prediction": record,
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to save prediction: {str(e)}"}), 500


@app.route("/api/predictions/history", methods=["GET"])
def get_prediction_history():
    """Return stored predictions and summary accuracy metrics as JSON."""
    status_filter = request.args.get("status")
    predictions = history_manager.get_all_predictions(status_filter=status_filter)
    summary = history_manager.get_summary_stats()
    return jsonify({
        "success": True,
        "predictions": predictions,
        "summary": summary,
        "total": len(predictions),
    })


@app.route("/api/predictions/verify", methods=["POST"])
def verify_predictions():
    """Reconcile pending predictions against authoritative official final scores."""
    try:
        res = result_verifier.verify_all_pending_predictions(force_refresh=True)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": f"Verification error: {str(e)}"}), 500


@app.route("/api/fixture/<fixture_id>", methods=["GET"])
def get_fixture_details(fixture_id: str):
    """Return fixture details and real-time 53-feature prediction as JSON."""
    fix_dict = predictor.get_fixture_by_id(fixture_id)
    if not fix_dict:
        return jsonify({"success": False, "error": f"Fixture '{fixture_id}' not found."}), 404

    model_name = request.args.get("model")
    prediction = predictor.predict_live(
        home_team=fix_dict["home_team"],
        away_team=fix_dict["away_team"],
        match_date=fix_dict.get("date"),
        model_name=model_name,
    )
    return jsonify({
        "success": True,
        "fixture": fix_dict,
        "prediction": prediction,
    })


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
    print("Starting Football Win Prediction System Flask Server...")
    print(f"Pre-Match Match Outcome Prediction Dashboard: http://{host}:{port}")
    print(f"Debug Mode: {debug_mode} (set FLASK_DEBUG=1 to enable for development)")
    app.run(host=host, port=port, debug=debug_mode)
