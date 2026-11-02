"""
Step 6: Live Prediction Integration & API Endpoint Verification Suite.

Tests the full pipeline through the Flask application test client:
Live provider -> live data -> 53 features -> scaler -> locked model -> probabilities -> Flask API
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app


def run_tests():
    print("=" * 80)
    print("STEP 6: LIVE PREDICTION & FLASK API ENDPOINT VERIFICATION")
    print("=" * 80)

    client = app.test_client()

    # ------------------------------------------------------------------
    # Test 1: GET /api/live/status
    # ------------------------------------------------------------------
    print("\n[Test 1] Testing GET /api/live/status ...")
    res1 = client.get("/api/live/status")
    assert res1.status_code == 200, f"Expected 200, got {res1.status_code}"
    data1 = res1.get_json()
    print(f"  Status: {data1.get('status')}")
    print(f"  Provider: {data1.get('provider')}")
    print(f"  Champion Model: {data1.get('champion_model')}")
    print(f"  Features Schema Count: {data1.get('feature_columns_count')}")
    print(f"  Total Ingested Matches: {data1.get('matches_ingested')}")
    print(f"  Cache Status: Fixtures Cached={data1['cache_status']['fixtures_cached']}, Completed Cached={data1['cache_status']['completed_matches_cached']}")
    assert data1["feature_columns_count"] == 53
    assert data1["status"] == "operational"
    print("  [OK] /api/live/status passed.")

    # ------------------------------------------------------------------
    # Test 2: GET /api/live/fixtures
    # ------------------------------------------------------------------
    print("\n[Test 2] Testing GET /api/live/fixtures ...")
    res2 = client.get("/api/live/fixtures?limit=5")
    assert res2.status_code == 200, f"Expected 200, got {res2.status_code}"
    data2 = res2.get_json()
    print(f"  Total Fixtures: {data2.get('total')}")
    print(f"  Data Source: {data2.get('data_source')} | Is Cached: {data2.get('is_cached')} | Provider: {data2.get('provider')}")
    print(f"  Last Updated: {data2.get('last_updated')} (Age: {data2.get('cache_age_seconds')}s)")
    assert data2["total"] > 0
    assert "fixtures" in data2
    sample_f = data2["fixtures"][0]
    print(f"  Sample Fixture: {sample_f['date']} {sample_f['time']} | {sample_f['home_team']} vs {sample_f['away_team']} (Canonical: {sample_f['is_canonical']})")
    print("  [OK] /api/live/fixtures passed.")

    # ------------------------------------------------------------------
    # Test 3: POST /api/live/predict (Live 53-Feature Inference)
    # ------------------------------------------------------------------
    print("\n[Test 3] Testing POST /api/live/predict (Aston Villa vs Nott'm Forest) ...")
    payload = {
        "home_team": "Aston Villa",
        "away_team": "Nott'm Forest",
        "match_date": "2026-09-12",
        "model": "Random Forest",
    }
    res3 = client.post("/api/live/predict", json=payload)
    assert res3.status_code == 200, f"Expected 200, got {res3.status_code}: {res3.get_data(as_text=True)}"
    data3 = res3.get_json()
    print(f"  Predicted Outcome: {data3.get('predicted_outcome')} (Winner: {data3.get('winner_team')})")
    print(f"  Confidence: {data3.get('confidence')}%")
    probs = data3["probabilities"]
    prob_sum = round(probs["home_win"] + probs["draw"] + probs["away_win"], 1)
    print(f"  Probabilities: Home {probs['home_win']}%, Draw {probs['draw']}%, Away {probs['away_win']}% (Sum: {prob_sum}%)")
    print(f"  Data Source: {data3.get('data_source')} | Is Live: {data3.get('is_live')} | Feature Count: {data3.get('feature_count')}")
    assert data3["feature_count"] == 53
    assert data3["is_live"] is True
    assert abs(prob_sum - 100.0) <= 0.1
    print("  [OK] POST /api/live/predict passed.")

    # ------------------------------------------------------------------
    # Test 4: GET /api/live/predictions (Batch Upcoming Fixtures)
    # ------------------------------------------------------------------
    print("\n[Test 4] Testing GET /api/live/predictions (Batch live predictions) ...")
    res4 = client.get("/api/live/predictions?limit=5")
    assert res4.status_code == 200, f"Expected 200, got {res4.status_code}: {res4.get_data(as_text=True)}"
    data4 = res4.get_json()
    print(f"  Total Predictions: {data4.get('total_fixtures')}")
    print(f"  Data Source: {data4.get('data_source')} | Provider: {data4.get('provider')} | Is Cached: {data4.get('is_cached')}")
    for f in data4["fixtures"][:3]:
        p = f["probabilities"]
        print(f"  - {f['date']} | {f['home_team']:<14} vs {f['away_team']:<14} -> {f['predicted_outcome']} (H: {p['home_win']}%, D: {p['draw']}%, A: {p['away_win']}%)")
    print("  [OK] GET /api/live/predictions passed.")

    # ------------------------------------------------------------------
    # Test 5: Existing Offline POST /predict (Backward Compatibility)
    # ------------------------------------------------------------------
    print("\n[Test 5] Testing existing offline POST /predict (Arsenal vs Chelsea) ...")
    offline_payload = {"home_team": "Arsenal", "away_team": "Chelsea"}
    res5 = client.post("/predict", json=offline_payload)
    assert res5.status_code == 200, f"Expected 200, got {res5.status_code}"
    data5 = res5.get_json()
    print(f"  Offline Predicted Outcome: {data5.get('predicted_outcome')}")
    p5 = data5["probabilities"]
    print(f"  Offline Probabilities: Home {p5['home_win']}%, Draw {p5['draw']}%, Away {p5['away_win']}%")
    # Must match original offline values: Home 42.6%, Draw 29.4%, Away 28.0%
    assert p5["home_win"] == 42.6
    assert p5["draw"] == 29.4
    assert p5["away_win"] == 28.0
    print("  [OK] Exact offline prediction backward compatibility verified.")

    # ------------------------------------------------------------------
    # Test 6: POST /predict with live=true flag
    # ------------------------------------------------------------------
    print("\n[Test 6] Testing POST /predict with live=true flag (Aston Villa vs Nott'm Forest) ...")
    live_payload = {"home_team": "Aston Villa", "away_team": "Nott'm Forest", "live": True, "match_date": "2026-09-12"}
    res6 = client.post("/predict", json=live_payload)
    assert res6.status_code == 200
    data6 = res6.get_json()
    print(f"  Live Predicted Outcome: {data6.get('predicted_outcome')}")
    print(f"  Live Probabilities: {data6['probabilities']}")
    assert data6.get("is_live") is True
    print("  [OK] POST /predict with live flag passed.")

    # ------------------------------------------------------------------
    # Test 7: Error Handling & Edge Cases
    # ------------------------------------------------------------------
    print("\n[Test 7] Testing Error Handling & Edge Cases ...")
    # Same team
    res_err1 = client.post("/api/live/predict", json={"home_team": "Arsenal", "away_team": "Arsenal"})
    assert res_err1.status_code == 400
    print(f"  Same-team rejection: {res_err1.get_json().get('error')}")

    # Missing away team
    res_err2 = client.post("/api/live/predict", json={"home_team": "Arsenal"})
    assert res_err2.status_code == 400
    print(f"  Missing parameter rejection: {res_err2.get_json().get('error')}")

    # Non-canonical team graceful handling
    res_promo = client.post("/api/live/predict", json={"home_team": "Chelsea", "away_team": "Hull"})
    assert res_promo.status_code == 200
    data_promo = res_promo.get_json()
    print(f"  Promoted/Non-canonical team handled: Chelsea vs Hull -> {data_promo['predicted_outcome']} (Canonical: {data_promo['is_canonical']})")
    print("  [OK] Error handling tests passed.")

    print("\n" + "=" * 80)
    print("ALL 7 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
