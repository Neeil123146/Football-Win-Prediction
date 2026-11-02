"""
Step 7: Verification of Live UI Dashboard Integration.

Verifies:
1. Flask app renders index.html with all required live sections:
   - Live Status Ribbon
   - Upcoming EPL Fixtures Match Center
   - Pipeline Mode Switcher (Live vs Offline)
   - Match Probability Assessment & Provenance Strip
   - "Why This Prediction?" Explainability Section
   - Historical model evaluation matrix and backtesting
   - Limitations and non-causal disclaimers
2. End-to-end simulation of user actions via Flask test client.
3. Validates both live and offline prediction flows.
4. Validates error handling for same-team, missing team, and edge cases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app


def test_ui_dashboard():
    print("=" * 80)
    print("STEP 7: LIVE UI DASHBOARD INTEGRATION VERIFICATION")
    print("=" * 80)

    client = app.test_client()

    # 1. Render main dashboard HTML
    print("\n[1] Testing GET / (Main Dashboard Render)...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    html = res.get_data(as_text=True)

    required_elements = [
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
    ]

    for el in required_elements:
        assert el in html, f"Missing required element in HTML: {el}"
        print(f"  [OK] Found HTML component: '{el}'")

    print("  [OK] Dashboard HTML contains all required live and explainability sections.")

    # 2. Test Live Prediction Flow via /api/live/predict
    print("\n[2] Testing Live Prediction Flow (User selects Aston Villa vs Nott'm Forest)...")
    live_req = {
        "home_team": "Aston Villa",
        "away_team": "Nott'm Forest",
        "match_date": "2026-09-12",
        "model": "Random Forest",
    }
    res_live = client.post("/api/live/predict", json=live_req)
    assert res_live.status_code == 200
    live_data = res_live.get_json()

    print(f"  Predicted Outcome: {live_data['predicted_outcome']} ({live_data['winner_team']})")
    print(f"  Confidence: {live_data['confidence']}%")
    probs = live_data["probabilities"]
    prob_sum = round(probs["home_win"] + probs["draw"] + probs["away_win"], 1)
    print(f"  Probabilities: Home {probs['home_win']}%, Draw {probs['draw']}%, Away {probs['away_win']}% (Sum: {prob_sum}%)")
    print(f"  Data Source: {live_data['data_source']} | Is Live: {live_data['is_live']}")
    print(f"  Feature Count: {live_data['feature_count']} features passed to model")
    assert live_data["feature_count"] == 53
    assert abs(prob_sum - 100.0) <= 0.1

    # Check Explainability factors
    explanation = live_data.get("explanation", {})
    factors = explanation.get("factors", [])
    print(f"  Explainability Factors Count: {len(factors)}")
    for f in factors[:3]:
        print(f"    - [{f['title']}]: {f['detail']}")
    assert len(factors) > 0
    assert "disclaimer" in explanation
    print(f"  Disclaimer: {explanation['disclaimer'][:80]}...")

    # 3. Test Offline Prediction Flow via /predict (Backward Compatibility)
    print("\n[3] Testing Offline Prediction Flow (Arsenal vs Chelsea)...")
    offline_req = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "model": "Random Forest",
    }
    res_offline = client.post("/predict", json=offline_req)
    assert res_offline.status_code == 200
    off_data = res_offline.get_json()
    off_probs = off_data["probabilities"]
    print(f"  Offline Outcome: {off_data['predicted_outcome']} ({off_data['winner_team']})")
    print(f"  Offline Probabilities: Home {off_probs['home_win']}%, Draw {off_probs['draw']}%, Away {off_probs['away_win']}%")
    # Verify baseline stability
    assert off_probs["home_win"] == 42.6
    assert off_probs["draw"] == 29.4
    assert off_probs["away_win"] == 28.0
    print("  [OK] Offline prediction preserved with 100% baseline parity.")

    # 4. Error Handling Tests
    print("\n[4] Testing Error Handling in UI / API...")
    # Same team error
    res_same = client.post("/api/live/predict", json={"home_team": "Liverpool", "away_team": "Liverpool"})
    assert res_same.status_code == 400
    print(f"  [OK] Same team rejected gracefully: {res_same.get_json().get('error')}")

    # Missing team error
    res_missing = client.post("/api/live/predict", json={"home_team": "Liverpool"})
    assert res_missing.status_code == 400
    print(f"  [OK] Missing team rejected gracefully: {res_missing.get_json().get('error')}")

    # 5. Live Upcoming Fixtures API check
    print("\n[5] Testing Live Fixtures API (/api/live/fixtures)...")
    res_fix = client.get("/api/live/fixtures?limit=5")
    assert res_fix.status_code == 200
    fix_data = res_fix.get_json()
    print(f"  Total Live Fixtures Retrieved: {fix_data['total']}")
    print(f"  Provider: {fix_data['provider']} | Is Cached: {fix_data['is_cached']}")
    assert fix_data["total"] > 0
    first_f = fix_data["fixtures"][0]
    print(f"  First Fixture: {first_f['date']} {first_f['time']} | {first_f['home_team']} vs {first_f['away_team']}")

    print("\n" + "=" * 80)
    print("ALL STEP 7 UI DASHBOARD TESTS PASSED!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_ui_dashboard()
    sys.exit(0 if success else 1)
