import os
import json
from typing import Dict, Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
CONFIGS_DIR = os.path.join(ROOT, "configs")

FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "contextual_featured_dataset.csv")
MASTER_PATH = os.path.join(DATA_PROCESSED_DIR, "master_player_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_final_context_model.pkl")
MODEL_COLUMNS_PATH = os.path.join(MODELS_DIR, "model_columns.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURE_IMPORTANCES_PATH = os.path.join(MODELS_DIR, "feature_importances.csv")
HPARAMS_PATH = os.path.join(CONFIGS_DIR, "hparams.yaml")


def load_hparams() -> Dict[str, Any]:
    with open(HPARAMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"No se encontró {FEATURES_PATH}. Ejecuta: python -m src.features.build_features"
        )
    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError(
            f"No se encontró {MASTER_PATH}. Ejecuta el ETL primero."
        )

    X = pd.read_csv(FEATURES_PATH)
    master = pd.read_csv(MASTER_PATH)

    # Align length (defensive)
    n = min(len(X), len(master))
    X = X.iloc[:n].copy()
    y = master.iloc[:n]["market_value_in_eur"].astype(float)

    # log transform target
    y_log = np.log1p(y.values)

    cfg = load_hparams()
    test_size = float(cfg.get("test_size", 0.2))
    random_state = int(cfg.get("random_state", 42))
    lgbm_params = cfg.get("lgbm", {})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=test_size, random_state=random_state
    )

    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    with open(MODEL_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f)

    # Metrics in EUR scale
    pred_log = model.predict(X_test)
    y_pred = np.expm1(pred_log)
    y_true = np.expm1(y_test)
    mae = float(mean_absolute_error(y_true, y_pred))

    metrics = {"MAE": mae, "n_train": int(len(X_train)), "n_test": int(len(X_test))}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Feature importances
    fi = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    fi.sort_values("importance", ascending=False).to_csv(FEATURE_IMPORTANCES_PATH, index=False)

    print("Entrenamiento completado.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
