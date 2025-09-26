import os
import sys

import pandera as pa
import pandera.typing as pat
import pandas as pd

# Project root: navigate up 3 levels from src/validation/validate_master.py
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
INPUT_PATH = os.path.join(DATA_PROCESSED_DIR, "master_player_dataset.csv")


class MasterSchema(pa.DataFrameModel):
    """Pandera schema for validating the master player dataset structure and constraints."""

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

    # Optional fields that may contain null values
    foot: pat.Series[str] = pa.Field(nullable=True)
    country_of_citizenship: pat.Series[str] = pa.Field(nullable=True)
    contract_expiration_date: pat.Series[str] = pa.Field(nullable=True)

    class Config:
        coerce = True


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        print(f"Master dataset not found: {INPUT_PATH}. Run ETL pipeline first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)

    # Select only columns that exist in both schema and dataframe
    cols = [c for c in MasterSchema.to_schema().columns.keys() if c in df.columns]
    df_sub = df[cols]

    MasterSchema.validate(df_sub, lazy=True)
    print("Master dataset validation completed successfully.")


if __name__ == "__main__":
    main()
