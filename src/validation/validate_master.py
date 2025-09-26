import os
import sys

import pandera as pa
import pandera.typing as pat
import pandas as pd

# project root: .../src/validation -> /src -> / (go up 3 levels from src/validation/validate_master.py)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
INPUT_PATH = os.path.join(DATA_PROCESSED_DIR, "master_player_dataset.csv")


class MasterSchema(pa.DataFrameModel):
    player_id: pat.Series[int]
    name: pat.Series[str]
    age: pat.Series[float] = pa.Field(ge=14, le=50)
    height_in_cm: pat.Series[float] = pa.Field(ge=100, le=220, nullable=True)
    current_club_name: pat.Series[str]
    player_club_domestic_competition_id: pat.Series[str]
    goals: pat.Series[int] = pa.Field(ge=0)
    assists: pat.Series[int] = pa.Field(ge=0)
    minutes_played: pat.Series[int] = pa.Field(ge=0)
    games_played: pat.Series[int] = pa.Field(ge=0)
    market_value_in_eur: pat.Series[float] = pa.Field(gt=0)

    # Campos opcionales que pueden ser nulos
    foot: pat.Series[str] = pa.Field(nullable=True)
    country_of_citizenship: pat.Series[str] = pa.Field(nullable=True)
    contract_expiration_date: pat.Series[str] = pa.Field(nullable=True)

    class Config:
        coerce = True


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        print(f"No se encontró {INPUT_PATH}. Ejecuta ETL primero.")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)

    # Subconjunto de columnas posibles
    cols = [c for c in MasterSchema.to_schema().columns.keys() if c in df.columns]
    df_sub = df[cols]

    MasterSchema.validate(df_sub, lazy=True)
    print("Validación exitosa del dataset maestro.")


if __name__ == "__main__":
    main()
