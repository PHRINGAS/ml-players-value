import os
import json
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")

FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "contextual_featured_dataset.csv")
MASTER_PATH = os.path.join(DATA_PROCESSED_DIR, "master_player_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_final_context_model.pkl")
MODEL_COLUMNS_PATH = os.path.join(MODELS_DIR, "model_columns.json")
SEGMENT_METRICS_PATH = os.path.join(MODELS_DIR, "metrics_by_segment.json")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_segments(
    y_true_eur: np.ndarray,
    y_pred_eur: np.ndarray,
    master: pd.DataFrame,
    segments: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    n = len(y_true_eur)
    df = master.iloc[:n].copy()

    for name, by_cols in segments.items():
        # create a composite key
        key = df[by_cols].astype(str).agg("|".join, axis=1)
        tmp = pd.DataFrame({
            "key": key,
            "y_true": y_true_eur,
            "y_pred": y_pred_eur,
        })
        agg = tmp.groupby("key").apply(lambda g: mae(g["y_true"].values, g["y_pred"].values)).to_dict()
        results[name] = agg
    return results


def main() -> None:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(MODEL_COLUMNS_PATH)):
        raise FileNotFoundError("Model artifacts missing. Run training first.")
    if not (os.path.exists(FEATURES_PATH) and os.path.exists(MASTER_PATH)):
        raise FileNotFoundError("Processed datasets missing. Run ETL and feature engineering first.")

    model = joblib.load(MODEL_PATH)
    with open(MODEL_COLUMNS_PATH, "r", encoding="utf-8") as f:
        model_cols = json.load(f)

    X = pd.read_csv(FEATURES_PATH)
    master = pd.read_csv(MASTER_PATH)

    # Ensure column order and dataset length alignment
    X = X[model_cols]
    n = min(len(X), len(master))
    X = X.iloc[:n].copy()
    master = master.iloc[:n].copy()

    # Generate predictions and convert to euro scale
    pred_log = model.predict(X)
    y_pred_eur = np.expm1(pred_log)
    y_true_eur = master["market_value_in_eur"].astype(float).values

    # Calculate global performance metrics
    metrics = {"MAE": mae(y_true_eur, y_pred_eur), "n": int(n)}

    # Define evaluation segments for detailed analysis
    segments = {
        "by_league": ["player_club_domestic_competition_id"],
    }
    if "position" in master.columns:
        segments["by_position"] = ["position"]

    # Create age buckets for additional segmentation
    if "age" in master.columns:
        bins = [0, 20, 23, 26, 30, 100]
        labels = ["<=20", "21-23", "24-26", "27-30", ">30"]
        master = master.copy()
        master["age_bucket"] = pd.cut(master["age"], bins=bins, labels=labels, right=True)
        segments["by_age_bucket"] = ["age_bucket"]

    seg_metrics = evaluate_segments(y_true_eur, y_pred_eur, master, segments)

    all_metrics = {"global": metrics, **seg_metrics}
    with open(SEGMENT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
