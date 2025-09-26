import json
import os
from typing import Optional

import typer

from .predictor import PlayerValuePredictor, _project_root

app = typer.Typer(help="CLI para el sistema de valoración de futbolistas")


@app.command()
def predict(
    input: Optional[str] = typer.Option(None, help="Ruta a un archivo JSON con los datos del jugador"),
    age: Optional[float] = typer.Option(None),
    height_in_cm: Optional[float] = typer.Option(None),
    current_club_name: Optional[str] = typer.Option(None),
    player_club_domestic_competition_id: Optional[str] = typer.Option(None),
    minutes_played: Optional[int] = typer.Option(None),
    goals: Optional[int] = typer.Option(None),
    assists: Optional[int] = typer.Option(None),
    contract_months_remaining: Optional[float] = typer.Option(None),
):
    """Realiza una predicción de valor de mercado a partir de un JSON o de opciones CLI."""
    root = _project_root()
    model_path = os.environ.get("MODEL_PATH", os.path.join(root, "models", "lgbm_final_context_model.pkl"))
    columns_path = os.environ.get("COLUMNS_PATH", os.path.join(root, "models", "model_columns.json"))

    predictor = PlayerValuePredictor(model_path=model_path, columns_path=columns_path)

    if input:
        with open(input, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {
            "age": age,
            "height_in_cm": height_in_cm,
            "current_club_name": current_club_name,
            "player_club_domestic_competition_id": player_club_domestic_competition_id,
            "minutes_played": minutes_played,
            "goals": goals,
            "assists": assists,
            "contract_months_remaining": contract_months_remaining,
        }

    value = predictor.predict(payload)
    typer.echo(json.dumps({"market_value_eur": round(float(value), 2)}, ensure_ascii=False))


if __name__ == "__main__":
    app()
