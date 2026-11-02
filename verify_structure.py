"""
Football Win Prediction System - Master System & Architecture Validator
Final Year BE Computer Engineering Project

Validates that all project components, datasets, trained models,
prediction pipelines, and web frontend files exist and operate correctly.

Run from the Football-Win-Prediction folder:
    python verify_structure.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "notebooks",
    ROOT / "models",
    ROOT / "src",
    ROOT / "templates",
    ROOT / "static",
    ROOT / "static" / "images",
]

REQUIRED_FILES = [
    ROOT / "requirements.txt",
    ROOT / "README.md",
    ROOT / "app.py",
    ROOT / "src" / "__init__.py",
    ROOT / "src" / "data_loader.py",
    ROOT / "src" / "data_preprocessing.py",
    ROOT / "src" / "feature_engineering.py",
    ROOT / "src" / "train_models.py",
    ROOT / "src" / "evaluate_models.py",
    ROOT / "src" / "prediction.py",
    ROOT / "templates" / "index.html",
    ROOT / "static" / "style.css",
    ROOT / "static" / "script.js",
    ROOT / "notebooks" / "football_analysis.ipynb",
]

REQUIRED_DATA = [
    ROOT / "data" / "processed" / "combined_raw.csv",
    ROOT / "data" / "processed" / "cleaned_matches.csv",
    ROOT / "data" / "processed" / "engineered_features.csv",
]

REQUIRED_MODELS = [
    ROOT / "models" / "best_model.pkl",
    ROOT / "models" / "scaler.pkl",
    ROOT / "models" / "metadata.json",
    ROOT / "models" / "model_comparison.json",
    ROOT / "models" / "feature_importance.csv",
    ROOT / "models" / "logistic_regression.pkl",
    ROOT / "models" / "random_forest.pkl",
    ROOT / "models" / "xgboost.pkl",
    ROOT / "models" / "svm.pkl",
]

REQUIRED_IMAGES = [
    ROOT / "static" / "images" / "confusion_matrix.png",
    ROOT / "static" / "images" / "calibration_curves.png",
    ROOT / "static" / "images" / "draw_class_analysis.png",
    ROOT / "static" / "images" / "model_accuracy_comparison.png",
    ROOT / "static" / "images" / "model_logloss_comparison.png",
    ROOT / "static" / "images" / "feature_importance.png",
    ROOT / "static" / "images" / "result_distribution.png",
]


def check_group(title: str, file_list: list[Path]) -> list[str]:
    print(f"\n{title}:")
    missing = []
    for path in file_list:
        exists = path.exists()
        mark = "OK" if exists else "MISSING"
        rel_path = path.relative_to(ROOT)
        print(f"  [{mark:<7}] {rel_path}")
        if not exists:
            missing.append(str(rel_path))
    return missing


def main() -> None:
    print("=" * 70)
    print("FOOTBALL WIN PREDICTION SYSTEM: END-TO-END VERIFICATION")
    print("=" * 70)
    print(f"Project directory: {ROOT}\n")

    all_missing = []
    all_missing.extend(check_group("1. Directory Architecture", REQUIRED_DIRS))
    all_missing.extend(check_group("2. Source & Web Files", REQUIRED_FILES))
    all_missing.extend(check_group("3. Processed Datasets (10 Seasons)", REQUIRED_DATA))
    all_missing.extend(check_group("4. Trained Models & Preprocessing Objects", REQUIRED_MODELS))
    all_missing.extend(check_group("5. Visualizations & Evaluation Charts", REQUIRED_IMAGES))

    raw_csvs = sorted((ROOT / "data" / "raw").glob("EPL_*.csv"))
    print(f"\n6. Raw EPL Season Datasets found: {len(raw_csvs)} / 10 seasons")
    for csv_path in raw_csvs:
        print(f"  - {csv_path.name}")

    # Live test of prediction engine
    print("\n7. Inference Engine Sanity Check:")
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.prediction import MatchPredictor
        p = MatchPredictor()
        test_pred = p.predict("Arsenal", "Chelsea")
        print(f"  [OK     ] Prediction Engine Operational")
        print(f"            Sample match: Arsenal vs Chelsea")
        print(f"            Predicted: {test_pred['predicted_outcome']} ({test_pred['winner_team']})")
        print(f"            Probabilities: {test_pred['probabilities']}")
    except Exception as exc:
        print(f"  [FAILED ] Inference Engine Error: {exc}")
        all_missing.append("Inference engine test")

    print("\n" + "=" * 70)
    if all_missing:
        print(f"STATUS: INCOMPLETE ({len(all_missing)} missing items)")
        for item in all_missing:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("STATUS: COMPLETE! All components, models, data, and web assets ready.")
        print("Run the web app using: python app.py")
        print("Open your browser at: http://127.0.0.1:5000")
    print("=" * 70)


if __name__ == "__main__":
    main()
