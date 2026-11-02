"""
Step 1: Project Audit & Baseline Verification Script.
Audits the existing 10-year historical dataset, the 53 feature columns,
the trained champion model, scaler, and external football data provider reachability.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
CLEANED_DATA = PROJECT_ROOT / "data" / "processed" / "cleaned_matches.csv"

def audit():
    print("=" * 70)
    print("STEP 1: PROJECT AUDIT & BASELINE SYSTEM VERIFICATION")
    print("=" * 70)

    # 1. Historical Dataset Audit
    print("\n[1/5] Auditing Historical Dataset...")
    if not CLEANED_DATA.exists():
        print(f"  [ERROR] Dataset missing at {CLEANED_DATA}")
        return False
    df = pd.read_csv(CLEANED_DATA)
    seasons = df["Season"].unique()
    teams = sorted(list(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())))
    print(f"  [OK] Cleaned matches file exists: {len(df):,} matches")
    print(f"  [OK] Total seasons covered: {len(seasons)} ({seasons[0]} to {seasons[-1]})")
    print(f"  [OK] Unique Premier League clubs: {len(teams)}")

    # 2. Model & Artifacts Audit
    print("\n[2/5] Auditing Trained Models & Metadata...")
    meta_path = MODELS_DIR / "metadata.json"
    scaler_path = MODELS_DIR / "scaler.pkl"
    best_model_path = MODELS_DIR / "best_model.pkl"

    if not meta_path.exists() or not scaler_path.exists() or not best_model_path.exists():
        print("  [ERROR] One or more core model artifacts missing in models/")
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    scaler = joblib.load(scaler_path)
    best_model = joblib.load(best_model_path)
    champion_name = meta.get("best_model", "Unknown")
    feature_cols = meta.get("feature_columns", [])

    print(f"  [OK] Champion Model: {champion_name} ({type(best_model).__name__})")
    print(f"  [OK] Scaler: {type(scaler).__name__} fitted with {scaler.n_features_in_} features")
    print(f"  [OK] Feature count in metadata: {len(feature_cols)} features")

    # 3. Verify Feature Dimensions
    print("\n[3/5] Verifying 53 Feature Dimension Compatibility...")
    if len(feature_cols) != 53 or scaler.n_features_in_ != 53:
        print(f"  [ERROR] Feature mismatch! Expected 53, got {len(feature_cols)} cols, {scaler.n_features_in_} in scaler")
        return False
    print("  [OK] Exact 53 features confirmed in metadata and scaler.")

    # 4. Prediction Engine Audit
    print("\n[4/5] Testing Existing Offline Prediction Engine...")
    try:
        from src.prediction import MatchPredictor
        predictor = MatchPredictor(MODELS_DIR)
        test_pred = predictor.predict("Arsenal", "Chelsea")
        print("  [OK] MatchPredictor initialized successfully.")
        print(f"  [OK] Sample Prediction (Arsenal vs Chelsea):")
        print(f"      Home Win: {test_pred['probabilities']['home_win']:.1f}%")
        print(f"      Draw:     {test_pred['probabilities']['draw']:.1f}%")
        print(f"      Away Win: {test_pred['probabilities']['away_win']:.1f}%")
        print(f"      Predicted Outcome: {test_pred['predicted_outcome']}")
    except Exception as e:
        print(f"  [ERROR] Prediction engine failed: {e}")
        return False

    # 5. External Live Data Provider Connectivity Check
    print("\n[5/5] Checking External Data Source Connectivity...")
    # Check football-data.co.uk
    try:
        req = urllib.request.Request("https://www.football-data.co.uk", headers={"User-Agent": "Mozilla/5.0"})
        res = urllib.request.urlopen(req, timeout=5)
        print(f"  [OK] football-data.co.uk is reachable (HTTP {res.status}).")
    except Exception as e:
        print(f"  [WARN] football-data.co.uk check returned: {e}")

    # Check football-data.org endpoint
    try:
        req = urllib.request.Request("https://api.football-data.org/v4/competitions/PL", headers={"User-Agent": "Mozilla/5.0"})
        res = urllib.request.urlopen(req, timeout=5)
        print(f"  [OK] api.football-data.org is reachable (HTTP {res.status}).")
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"  [OK] api.football-data.org is reachable (HTTP 403 Auth Required - Expected without API key header).")
        else:
            print(f"  [WARN] api.football-data.org returned HTTP {e.code}")
    except Exception as e:
        print(f"  [WARN] api.football-data.org network check: {e}")

    print("\n" + "=" * 70)
    print("AUDIT RESULT: SYSTEM HEALTHY & READY FOR LIVE PIPELINE UPGRADE")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = audit()
    sys.exit(0 if success else 1)
