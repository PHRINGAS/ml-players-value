import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .predictor import PlayerValuePredictor, PlayerInput, _project_root

app = FastAPI(title="Player Market Value API", version="1.0")


def get_predictor() -> PlayerValuePredictor:
    root = _project_root()
    model_path = os.environ.get("MODEL_PATH", os.path.join(root, "models", "lgbm_final_context_model.pkl"))
    columns_path = os.environ.get("COLUMNS_PATH", os.path.join(root, "models", "model_columns.json"))
    return PlayerValuePredictor(model_path=model_path, columns_path=columns_path)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


class PredictionResponse(BaseModel):
    market_value_eur: float


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PlayerInput):
    try:
        predictor = get_predictor()
        value = predictor.predict(payload.model_dump())
        return PredictionResponse(market_value_eur=float(value))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
