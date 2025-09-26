import os
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "..", "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "..", "models")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "..", "configs")

MASTER_INPUT = os.path.join(DATA_PROCESSED_DIR, "master_player_dataset.csv")
FEATURED_OUTPUT = os.path.join(DATA_PROCESSED_DIR, "contextual_featured_dataset.csv")
MODEL_COLUMNS_PATH = os.path.join(MODELS_DIR, "model_columns.json")


def load_context_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_features(df: pd.DataFrame, context_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute the minimal set of features required by the final model.

    Expected inputs present in df:
      - age, height_in_cm, minutes_played, goals, assists,
        player_club_domestic_competition_id, current_club_name
    """
    # Calculate per-90-minute performance metrics
    minutes_per_90 = df["minutes_played"].fillna(0) / 90.0
    denom = minutes_per_90.replace(0, np.nan)
    df_feat = pd.DataFrame()
    df_feat["minutes_per_90"] = minutes_per_90.fillna(0)
    df_feat["goals_per_90"] = (df["goals"].fillna(0) / denom).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_feat["assists_per_90"] = (df["assists"].fillna(0) / denom).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_feat["goals_plus_assists_per_90"] = (
        (df["goals"].fillna(0) + df["assists"].fillna(0)) / denom
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Pass through basic player attributes
    df_feat["age"] = df["age"].astype(float)
    df_feat["height_in_cm"] = df["height_in_cm"].astype(float)

    # Calculate contract months remaining (if available)
    if "contract_months_remaining" in df.columns:
        df_feat["contract_months_remaining"] = df["contract_months_remaining"].fillna(0).astype(float)
    else:
        # Best effort: derive from expiration date if present
        if "contract_expiration_date" in df.columns:
            exp = pd.to_datetime(df["contract_expiration_date"], errors="coerce")
            today = pd.Timestamp.today().normalize()
            months = (exp - today).dt.days / 30.44
            df_feat["contract_months_remaining"] = months.clip(lower=0).fillna(0).astype(float)
        else:
            df_feat["contract_months_remaining"] = 0.0

    # Apply contextual adjustments based on competition and club
    league_map = context_cfg.get("league_strength_map_by_id", {})
    default_league = context_cfg.get("default_league_strength_factor", 0.4)
    df_feat["league_strength_factor"] = df["player_club_domestic_competition_id"].map(league_map).fillna(default_league)

    tier1 = set(context_cfg.get("club_tier_1", []))
    df_feat["club_tier"] = df["current_club_name"].isin(tier1).astype(int)

    # Order columns to match model expectations if available
    if os.path.exists(MODEL_COLUMNS_PATH):
        with open(MODEL_COLUMNS_PATH, "r", encoding="utf-8") as f:
            model_cols = json.load(f)
        df_feat = df_feat.reindex(columns=model_cols)

    return df_feat


def main():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    context_cfg = load_context_config(os.path.join(CONFIGS_DIR, "context.yaml"))

    if not os.path.exists(MASTER_INPUT):
        raise FileNotFoundError(f"Master dataset not found: {MASTER_INPUT}. Run ETL pipeline first.")

    df = pd.read_csv(MASTER_INPUT)
    feat = compute_features(df, context_cfg)

    # Persist the enhanced dataset
    feat.to_csv(FEATURED_OUTPUT, index=False)
    print(f"Enhanced features saved to: {FEATURED_OUTPUT}")
    print(f"Dataset dimensions: {feat.shape}")


if __name__ == "__main__":
    main()
