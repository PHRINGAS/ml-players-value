import os
import json

import numpy as np
import pandas as pd

from src.features.build_features import compute_features


def test_compute_features_minimal(tmp_path):
    """Test feature computation with minimal required input data."""
    # Create minimal test dataset
    df = pd.DataFrame(
        {
            "age": [22.0],
            "height_in_cm": [180.0],
            "minutes_played": [900],
            "goals": [5],
            "assists": [3],
            "player_club_domestic_competition_id": ["DE1"],
            "current_club_name": ["Bayer 04 Leverkusen"],
        }
    )
    cfg = {
        "league_strength_map_by_id": {"DE1": 0.95},
        "default_league_strength_factor": 0.4,
        "club_tier_1": ["Real Madrid"],
    }
    feat = compute_features(df, cfg)

    # Verify expected feature columns are present
    expected = {
        "age",
        "height_in_cm",
        "minutes_per_90",
        "goals_per_90",
        "assists_per_90",
        "goals_plus_assists_per_90",
        "contract_months_remaining",
        "league_strength_factor",
        "club_tier",
    }
    assert set(feat.columns) >= expected

    # Validate computed feature values
    assert np.isclose(feat.loc[0, "minutes_per_90"], 10.0)
    assert np.isclose(feat.loc[0, "goals_per_90"], 0.5)
    assert np.isclose(feat.loc[0, "assists_per_90"], 0.3)
    assert np.isclose(feat.loc[0, "goals_plus_assists_per_90"], 0.8)
    assert np.isclose(feat.loc[0, "league_strength_factor"], 0.95)
    assert int(feat.loc[0, "club_tier"]) == 0
