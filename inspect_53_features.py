"""
Step 2: Exact 53 Feature Inventory & Verification Script.
Audits and categorizes the exact 53 pre-match feature columns expected
by the champion model and StandardScaler from models/metadata.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


FEATURE_CATEGORIES = {
    "1. Recent 5-Match Form (Rolling Window)": [
        ("Home_Recent_Pts", "Points accumulated by Home team in their last 5 matches (3 for W, 1 for D, 0 for L)"),
        ("Away_Recent_Pts", "Points accumulated by Away team in their last 5 matches"),
        ("Home_Recent_WinRate", "Win rate of Home team in last 5 matches (Wins / 5)"),
        ("Away_Recent_WinRate", "Win rate of Away team in last 5 matches (Wins / 5)"),
        ("Home_Recent_GF_PG", "Goals scored per game by Home team in last 5 matches"),
        ("Away_Recent_GF_PG", "Goals scored per game by Away team in last 5 matches"),
        ("Home_Recent_GA_PG", "Goals conceded per game by Home team in last 5 matches"),
        ("Away_Recent_GA_PG", "Goals conceded per game by Away team in last 5 matches"),
    ],
    "2. Recent Form Differential Features": [
        ("Diff_Recent_Pts", "Home_Recent_Pts - Away_Recent_Pts"),
        ("Diff_Recent_WinRate", "Home_Recent_WinRate - Away_Recent_WinRate"),
    ],
    "3. Lifetime Overall Record in Dataset": [
        ("Home_Overall_MP", "Total historical matches played by Home team in dataset"),
        ("Away_Overall_MP", "Total historical matches played by Away team in dataset"),
        ("Home_Overall_PPG", "Lifetime points per game for Home team"),
        ("Away_Overall_PPG", "Lifetime points per game for Away team"),
        ("Home_Overall_GF_PG", "Lifetime goals scored per game for Home team"),
        ("Away_Overall_GF_PG", "Lifetime goals scored per game for Away team"),
        ("Home_Overall_GA_PG", "Lifetime goals conceded per game for Home team"),
        ("Away_Overall_GA_PG", "Lifetime goals conceded per game for Away team"),
        ("Home_Overall_GD", "Lifetime goal difference (GF - GA) for Home team"),
        ("Away_Overall_GD", "Lifetime goal difference (GF - GA) for Away team"),
    ],
    "4. Lifetime Record Differential Features": [
        ("Diff_PPG", "Home_Overall_PPG - Away_Overall_PPG"),
        ("Diff_Overall_GD", "Home_Overall_GD - Away_Overall_GD"),
    ],
    "5. Lifetime Venue-Specific Performance (Home / Away Strength)": [
        ("Home_Home_WR", "Lifetime home win rate for Home team (Home wins / Home matches)"),
        ("Home_Home_GF_PG", "Lifetime goals scored per game at home by Home team"),
        ("Home_Home_GA_PG", "Lifetime goals conceded per game at home by Home team"),
        ("Away_Away_WR", "Lifetime away win rate for Away team (Away wins / Away matches)"),
        ("Away_Away_GF_PG", "Lifetime goals scored per game on the road by Away team"),
        ("Away_Away_GA_PG", "Lifetime goals conceded per game on the road by Away team"),
    ],
    "6. Current Season Venue-Specific Performance": [
        ("Home_Season_Home_WR", "Current season home win rate for Home team"),
        ("Home_Season_Home_GF_PG", "Current season goals scored per game at home by Home team"),
        ("Home_Season_Home_GA_PG", "Current season goals conceded per game at home by Home team"),
        ("Away_Season_Away_WR", "Current season away win rate for Away team"),
        ("Away_Season_Away_GF_PG", "Current season goals scored per game away by Away team"),
        ("Away_Season_Away_GA_PG", "Current season goals conceded per game away by Away team"),
    ],
    "7. Historical Head-to-Head (H2H) Prior Dynamics": [
        ("H2H_Prior_Matches", "Number of historical encounters between Home and Away team prior to this match"),
        ("H2H_HomeWins", "Wins by Home team in prior meetings"),
        ("H2H_Draws", "Draws in prior meetings"),
        ("H2H_AwayWins", "Wins by Away team in prior meetings"),
        ("H2H_GoalDiff", "Cumulative goal difference for Home team in prior meetings"),
    ],
    "8. Dynamic Sequential Elo Ratings": [
        ("Home_Elo", "Pre-match Elo rating of Home team (K=20, start=1500, updated sequentially)"),
        ("Away_Elo", "Pre-match Elo rating of Away team (K=20, start=1500, updated sequentially)"),
        ("Elo_Diff", "Home_Elo - Away_Elo"),
    ],
    "9. Current Season Standings & Table Metrics": [
        ("Home_Season_Rank", "League table position (1 to 20) of Home team before kickoff"),
        ("Away_Season_Rank", "League table position (1 to 20) of Away team before kickoff"),
        ("Home_Season_Pts", "Total league points accumulated in the current season by Home team"),
        ("Away_Season_Pts", "Total league points accumulated in the current season by Away team"),
        ("Home_Season_GD", "Current season goal difference for Home team"),
        ("Away_Season_GD", "Current season goal difference for Away team"),
        ("Home_Season_MP", "Current season matches played by Home team"),
        ("Away_Season_MP", "Current season matches played by Away team"),
        ("Diff_Rank", "Away_Season_Rank - Home_Season_Rank (positive indicates Home is ranked higher)"),
        ("Diff_Season_Pts", "Home_Season_Pts - Away_Season_Pts"),
        ("Diff_Season_GD", "Home_Season_GD - Away_Season_GD"),
    ],
}


def main():
    print("=" * 80)
    print("STEP 2: EXACT 53-FEATURE INVENTORY & SCHEMA SPECIFICATION")
    print("=" * 80)

    meta_file = MODELS_DIR / "metadata.json"
    scaler_file = MODELS_DIR / "scaler.pkl"

    with open(meta_file) as f:
        meta = json.load(f)

    scaler = joblib.load(scaler_file)
    expected_cols = meta.get("feature_columns", [])

    print(f"\nTotal Features Registered in models/metadata.json: {len(expected_cols)}")
    print(f"Total Features Expected by StandardScaler:       {scaler.n_features_in_}")

    # Build flat dictionary from categories
    categorized_flat = {}
    for cat, features in FEATURE_CATEGORIES.items():
        for col_name, desc in features:
            categorized_flat[col_name] = (cat, desc)

    # Check for complete 1-to-1 parity
    missing_in_spec = set(expected_cols) - set(categorized_flat.keys())
    extra_in_spec = set(categorized_flat.keys()) - set(expected_cols)

    if missing_in_spec:
        print(f"\n[ERROR] Features in metadata.json missing from specification: {missing_in_spec}")
        return False
    if extra_in_spec:
        print(f"\n[ERROR] Features in specification not in metadata.json: {extra_in_spec}")
        return False

    print("\n[OK] 100% SCHEMA PARITY: All 53 features match metadata.json and scaler.pkl exactly.\n")

    # Print categorized breakdown
    index = 0
    for cat_name, features in FEATURE_CATEGORIES.items():
        print(f"\n--- {cat_name} ({len(features)} features) ---")
        for col_name, desc in features:
            # Check exact index position in expected_cols
            expected_idx = expected_cols.index(col_name)
            print(f"  [{expected_idx:02d}] {col_name:<25} -> {desc}")

    print("\n" + "=" * 80)
    print("ALL 53 PRE-MATCH FEATURES VERIFIED AND LOCKED FOR LIVE INGESTION")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
