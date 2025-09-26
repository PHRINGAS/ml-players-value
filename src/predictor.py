import json
import logging
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator

LOGGER = logging.getLogger(__name__)


def _project_root() -> str:
    """Return repository root assuming this file lives in src/."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_context_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load contextual configuration (league strength, club tiers) from YAML file."""
    if config_path is None:
        config_path = os.path.join(_project_root(), "configs", "context.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class PlayerInput(BaseModel):
    age: float = Field(..., ge=14, le=50)
    height_in_cm: float = Field(..., ge=100, le=220)
    current_club_name: str
    player_club_domestic_competition_id: str = Field(..., min_length=2, max_length=5)

    minutes_played: int = Field(..., ge=0)
    goals: int = Field(..., ge=0)
    assists: int = Field(..., ge=0)

    # Optional fields for contract duration - either direct months or expiration date
    contract_months_remaining: Optional[float] = Field(default=None, ge=0)
    contract_expiration_date: Optional[str] = None  # Format: YYYY-MM-DD

    # Optional fields not used by current model but accepted for compatibility
    foot: Optional[str] = Field(default=None)
    position: Optional[str] = Field(default=None)

    @field_validator("contract_expiration_date")
    @classmethod
    def validate_date_format(cls, v: Optional[str]):
        if v is None:
            return v
        try:
            pd.to_datetime(v)
        except Exception as e:  # noqa: BLE001
            raise ValueError("contract_expiration_date must be in YYYY-MM-DD format") from e
        return v


class PlayerValuePredictor:
    """Machine learning predictor for football player market values.

    This class loads a pre-trained model and provides methods to predict
    player market values based on performance metrics and contextual factors.
    """

    def __init__(self, model_path, columns_path):
        """Initialize the predictor by loading model artifacts and configuration.

        Args:
            model_path: Path to the serialized LightGBM model
            columns_path: Path to the JSON file containing expected feature columns
        """
        LOGGER.info("Loading model artifacts...")
        self.model = joblib.load(model_path)
        with open(columns_path, "r", encoding="utf-8") as f:
            self.model_columns = json.load(f)

        # Load contextual configuration for league strength and club tiers
        self.context_cfg = load_context_config()
        self.league_strength_map = self.context_cfg.get("league_strength_map_by_id", {})
        self.default_league_strength = self.context_cfg.get("default_league_strength_factor", 0.4)
        self.tier1_clubs = set(self.context_cfg.get("club_tier_1", []))
        LOGGER.info("Predictor initialized successfully.")

    def preprocess_input(self, player_data: dict) -> pd.DataFrame:
        """Transform raw player data into model-ready features.

        This method validates input data, calculates normalized performance metrics,
        and applies contextual adjustments based on league strength and club tier.

        Args:
            player_data: Dictionary containing player information and statistics

        Returns:
            DataFrame with processed features ready for model prediction
        """
        # Validate and normalize input using Pydantic model
        data = PlayerInput(**player_data).model_dump()

        # Calculate contract months remaining from expiration date if not provided
        if data.get("contract_months_remaining") is None and data.get("contract_expiration_date"):
            try:
                exp_date = pd.to_datetime(data["contract_expiration_date"])  # type: ignore[arg-type]
                today = pd.Timestamp.today().normalize()
                data["contract_months_remaining"] = max(0.0, (exp_date - today).days / 30.44)
            except Exception:  # noqa: BLE001
                data["contract_months_remaining"] = 0.0

        # Calculate normalized performance metrics (per 90 minutes played)
        minutes_per_90 = (data.get("minutes_played", 0) or 0) / 90.0
        # Avoid division by zero for players with no minutes
        denom = minutes_per_90 if minutes_per_90 > 0 else np.nan
        goals_per_90 = (data.get("goals", 0) or 0) / denom
        assists_per_90 = (data.get("assists", 0) or 0) / denom
        combined_goals_assists_per_90 = ((data.get("goals", 0) or 0) + (data.get("assists", 0) or 0)) / denom

        # Apply contextual adjustments based on competition and club
        competition_id = data.get("player_club_domestic_competition_id")
        league_strength = self.league_strength_map.get(competition_id, self.default_league_strength)
        club_tier = 1 if data.get("current_club_name") in self.tier1_clubs else 0

        # Construct feature vector matching model expectations
        feature_row = {
            "age": data.get("age", 0.0),
            "height_in_cm": data.get("height_in_cm", 0.0),
            "minutes_per_90": minutes_per_90 if np.isfinite(minutes_per_90) else 0.0,
            "goals_per_90": goals_per_90 if np.isfinite(goals_per_90) else 0.0,
            "assists_per_90": assists_per_90 if np.isfinite(assists_per_90) else 0.0,
            "goals_plus_assists_per_90": combined_goals_assists_per_90 if np.isfinite(combined_goals_assists_per_90) else 0.0,
            "contract_months_remaining": float(data.get("contract_months_remaining") or 0.0),
            "league_strength_factor": float(league_strength),
            "club_tier": int(club_tier),
        }

        df = pd.DataFrame([feature_row], columns=self.model_columns)
        df = df.fillna(0)
        return df

    def predict(self, player_data: dict) -> float:
        """Predict market value for a single player.

        Args:
            player_data: Dictionary containing player information and statistics

        Returns:
            Predicted market value in euros
        """
        # Preprocess input data into model-ready features
        processed_df = self.preprocess_input(player_data)

        # Generate prediction on logarithmic scale
        log_prediction = self.model.predict(processed_df)

        # Transform back to original euro scale
        euro_prediction = np.expm1(log_prediction)

        return euro_prediction[0]


# =============================================================================
# Demo Section: Example Usage
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Define default paths for model artifacts
    ROOT = _project_root()
    DEFAULT_MODEL_PATH = os.path.join(ROOT, "models", "lgbm_final_context_model.pkl")
    DEFAULT_COLUMNS_PATH = os.path.join(ROOT, "models", "model_columns.json")

    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    columns_path = os.environ.get("COLUMNS_PATH", DEFAULT_COLUMNS_PATH)

    # Initialize the predictor with model artifacts
    predictor = PlayerValuePredictor(model_path=model_path, columns_path=columns_path)

    # Example player data for prediction
    sample_player = {
        "age": 22.5,
        "height_in_cm": 185.0,
        "current_club_name": "Bayer 04 Leverkusen",
        "player_club_domestic_competition_id": "DE1",  # Bundesliga
        "minutes_played": 2800,
        "goals": 15,
        "assists": 10,
        "contract_months_remaining": 36.0,  # 3 years remaining
    }

    # Generate market value prediction
    predicted_value = predictor.predict(sample_player)

    print("\n--- Market Value Prediction Demo ---")
    print(f"Player: {sample_player['age']:.1f} years old at {sample_player['current_club_name']}")
    print(f"Estimated Market Value: €{predicted_value:,.0f}")
