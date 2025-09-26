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
    """Load contextual configuration (league strength, club tiers) from YAML."""
    if config_path is None:
        config_path = os.path.join(_project_root(), "configs", "context.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")
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

    # Puedes pasar directamente los meses restantes o una fecha de expiración
    contract_months_remaining: Optional[float] = Field(default=None, ge=0)
    contract_expiration_date: Optional[str] = None  # YYYY-MM-DD

    # Campos opcionales no utilizados por el modelo actual, pero admitidos
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
            raise ValueError("contract_expiration_date debe tener formato YYYY-MM-DD") from e
        return v


class PlayerValuePredictor:
    def __init__(self, model_path, columns_path):
        """
        Inicializa el predictor cargando el modelo y las columnas necesarias.
        """
        LOGGER.info("Cargando artefactos del modelo...")
        self.model = joblib.load(model_path)
        with open(columns_path, "r", encoding="utf-8") as f:
            self.model_columns = json.load(f)
        # Cargar configuración contextual
        self.context_cfg = load_context_config()
        self.league_strength_map = self.context_cfg.get("league_strength_map_by_id", {})
        self.default_league_strength = self.context_cfg.get("default_league_strength_factor", 0.4)
        self.tier1_clubs = set(self.context_cfg.get("club_tier_1", []))
        LOGGER.info("Predictor inicializado exitosamente.")

    def preprocess_input(self, player_data: dict) -> pd.DataFrame:
        """
        Toma un diccionario con datos de un jugador y lo preprocesa
        para que coincida con el formato que el modelo espera.
        """
        # Validar/normalizar entrada con Pydantic
        data = PlayerInput(**player_data).model_dump()

        # Si no viene contract_months_remaining, derivarlo desde la fecha de expiración
        if data.get("contract_months_remaining") is None and data.get("contract_expiration_date"):
            try:
                exp = pd.to_datetime(data["contract_expiration_date"])  # type: ignore[arg-type]
                today = pd.Timestamp.today().normalize()
                data["contract_months_remaining"] = max(0.0, (exp - today).days / 30.44)
            except Exception:  # noqa: BLE001
                data["contract_months_remaining"] = 0.0

        # --- 1. Features de rendimiento normalizadas ---
        minutes_per_90 = (data.get("minutes_played", 0) or 0) / 90.0
        # Evitar divisiones por cero
        denom = minutes_per_90 if minutes_per_90 > 0 else np.nan
        goals_per_90 = (data.get("goals", 0) or 0) / denom
        assists_per_90 = (data.get("assists", 0) or 0) / denom
        ga_per_90 = ((data.get("goals", 0) or 0) + (data.get("assists", 0) or 0)) / denom

        # --- 2. Features contextuales ---
        comp_id = data.get("player_club_domestic_competition_id")
        league_strength = self.league_strength_map.get(comp_id, self.default_league_strength)
        club_tier = 1 if data.get("current_club_name") in self.tier1_clubs else 0

        row = {
            "age": data.get("age", 0.0),
            "height_in_cm": data.get("height_in_cm", 0.0),
            "minutes_per_90": minutes_per_90 if np.isfinite(minutes_per_90) else 0.0,
            "goals_per_90": goals_per_90 if np.isfinite(goals_per_90) else 0.0,
            "assists_per_90": assists_per_90 if np.isfinite(assists_per_90) else 0.0,
            "goals_plus_assists_per_90": ga_per_90 if np.isfinite(ga_per_90) else 0.0,
            "contract_months_remaining": float(data.get("contract_months_remaining") or 0.0),
            "league_strength_factor": float(league_strength),
            "club_tier": int(club_tier),
        }

        df = pd.DataFrame([row], columns=self.model_columns)
        df = df.fillna(0)
        return df

    def predict(self, player_data: dict) -> float:
        """
        Realiza una predicción de valor de mercado para un solo jugador.
        """
        # Preprocesar los datos de entrada
        processed_df = self.preprocess_input(player_data)

        # Predecir en escala logarítmica
        prediction_log = self.model.predict(processed_df)

        # Revertir a la escala original en euros
        prediction_eur = np.expm1(prediction_log)

        return prediction_eur[0]


# =============================================================================
# Bloque de Demostración: Cómo usar el predictor
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    # Rutas a los artefactos del modelo
    ROOT = _project_root()
    DEFAULT_MODEL_PATH = os.path.join(ROOT, "models", "lgbm_final_context_model.pkl")
    DEFAULT_COLUMNS_PATH = os.path.join(ROOT, "models", "model_columns.json")

    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    columns_path = os.environ.get("COLUMNS_PATH", DEFAULT_COLUMNS_PATH)

    # Crear una instancia del predictor
    predictor = PlayerValuePredictor(model_path=model_path, columns_path=columns_path)

    # --- Crear un usuario de ejemplo para predecir ---
    new_player = {
        "age": 22.5,
        "height_in_cm": 185.0,
        "current_club_name": "Bayer 04 Leverkusen",
        "player_club_domestic_competition_id": "DE1",  # Bundesliga
        "minutes_played": 2800,
        "goals": 15,
        "assists": 10,
        "contract_months_remaining": 36.0,  # 3 años de contrato
    }

    # Realizar la predicción
    predicted_value = predictor.predict(new_player)

    print("\n--- Demostración de Predicción ---")
    print(
        f"Datos del jugador: {new_player.get('age', 0):.1f} años en {new_player['current_club_name']}"
    )
    print(f"Valor de Mercado Estimado: €{predicted_value:,.0f}")
