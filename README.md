# Predictive System for Football Player Valuation (v1.0)

This repository documents the construction of an end-to-end Machine Learning system for estimating football players' market value. The project spans from data ingestion and cleaning to model training, optimization, and packaging of a production-ready predictive model.

---

## 🎯 The Challenge: The Hidden Value in Data

Football player valuation is a complex process with millions of euros at stake in every market decision. While scouts and analysts bring invaluable experience, the process often lacks an objective, quantifiable foundation.

This project addresses the following challenge: **Is it possible to build a model that learns hidden patterns in data to estimate a player's market value, based on their performance, demographic profile, and crucially, their competitive context?**

## ✨ Key Results: An Intelligent and Quantifiable Model

After a rigorous iterative modeling process, the final system can predict market value with a **Mean Absolute Error (MAE) of €2,921,504**.

- **Massive Improvement:** This represents a **26.63% error reduction** (more than €1,060,000 in added precision) compared to the initial baseline model.
- **Contextual Intelligence:** Feature importance analysis revealed that the most determining factors are not just goals or assists, but the **player's context**:
  1. **Contract Situation** (`contract_months_remaining`)
  2. **Age and Potential** (`age`)
  3. **Competition Quality** (`league_strength_factor`)
  4. **Club Prestige** (`club_tier`)

## 🏗️ The Process: From Raw Data to Predictive Model

The project's success is based on an iterative approach where each phase built upon the previous:

1. **Foundations and Baseline:** We established a robust ETL process to clean and unify multiple data sources. A simple Ridge Regression model set our baseline with an **MAE of €3.98M**.
2. **Feature Engineering (Iteration 1):** We created normalized performance features (e.g., `goals_per_90`) and business features (e.g., `contract_months_remaining`). This allowed a base LightGBM model to reduce MAE to **€3.23M**, an 18.7% improvement.
3. **Hyperparameter Optimization:** Using **Optuna**, we conducted an exhaustive search (300 trials) to find the optimal LightGBM configuration, refining its performance and robustness.
4. **Error Analysis and the "Context Factor" (Iteration 2):** We analyzed where the model failed, discovering systematic underestimation of elite players and overestimation of players in minor leagues. This guided the creation of **contextual features** (`league_strength_factor`, `club_tier`), leading the final model to its **€2.92M MAE**, the definitive qualitative leap.

## 🛠️ Technology Stack

- **Analysis and Modeling:** Python, Pandas, Scikit-learn, LightGBM
- **Configuration and Validation:** YAML (configs), Pydantic for input validation
- **Data/Model Versioning:** DVC (Data Version Control) with `dvc.yaml`
- **Environment and Prototyping:** Jupyter Notebooks (development) and modular `.py` scripts (production)
- **Interface:** CLI with Typer; optional API with FastAPI + Uvicorn
- **Quality/Automation:** pre-commit (black/ruff/isort/nbstripout), GitHub Actions (CI), Pytest
- **Hyperparameter Optimization (optional):** Optuna

## 📁 Repository Structure

```
├── data/                                  # Directory for all datasets (managed by DVC)
│   ├── raw/                               # Original, unmodified data
│   └── processed/                         # Clean and enriched datasets, ready for modeling
│
├── models/                                # Model artifacts
│   ├── lgbm_final_context_model.pkl       # Serialized model
│   ├── model_columns.json                 # Expected features list (order)
│   ├── metrics.json                       # Global metrics (MAE)
│   └── feature_importances.csv            # Feature importances
│
├── notebooks/                             # Exploration and development process (Jupyter Notebooks)
│   ├── 00_baselines.ipynb
│   ├── 01_eda_cariboo_dataset.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_error_analysis.ipynb
│   ├── 06_contextual_feature_engineering.ipynb
│   ├── 07_final_model_evaluation.ipynb
│   └── 08_final_error_analysis.ipynb
│
├── src/                                   # Production source code
│   ├── etl/
│   │   └── build_master_dataset.py        # ETL from raw tables → master
│   ├── features/
│   │   └── build_features.py              # Contextual feature engineering (production)
│   ├── validation/
│   │   └── validate_master.py             # Master dataset validation (Pandera)
│   ├── monitoring/
│   │   └── drift_report.py                # Drift report (Evidently)
│   ├── predictor.py                       # Predictor with Pydantic + external configuration (YAML)
│   ├── train.py                           # Final model training
│   └── evaluate.py                        # Global and segment evaluation
│
├── configs/                               # Versioned configuration (YAML)
│   ├── context.yaml                       # League strength map and tier-1 clubs
│   └── hparams.yaml                       # LightGBM hyperparameters and split
│
├── .gitignore
├── README.md                              # This file
├── requirements-dev.txt                   # Development/training dependencies
├── requirements-prod.txt                  # Minimal production/inference dependencies
├── dvc.yaml                               # DVC pipeline (ETL → features → train → evaluate)
├── Dockerfile                             # Image for serving the API
├── .dockerignore                          # Docker image exclusions
├── .pre-commit-config.yaml                # Quality hooks (black/ruff/isort/nbstripout)
├── pyproject.toml                         # Linter/formatter configuration
├── .github/workflows/ci.yml               # CI pipeline (lint + tests)
└── scoping.md                             # Project scope document
```

## 🚀 How to Run the Project

Follow these instructions for reproducibility (ETL → features → train → evaluate) and inference (CLI / API / Docker).

### Prerequisites

- Python 3.10+
- Git
- DVC installed (`pip install dvc`) or install from `requirements-dev.txt`

### 1) Environment Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
# source venv/bin/activate

# Development/training
pip install -r requirements-dev.txt

# Production/inference (minimal)
# pip install -r requirements-prod.txt
```

### 2) Get Data and Reproduce Pipeline

```bash
dvc pull

dvc repro
```

Generated artifacts:

- `data/processed/master_player_dataset.csv`
- `data/processed/contextual_featured_dataset.csv`
- `models/lgbm_final_context_model.pkl`, `models/model_columns.json`
- `models/metrics.json`, `models/feature_importances.csv`
- `models/metrics_by_segment.json`

### 3) Script Inference (Demo)

```bash
python src/predictor.py
```

Optional environment variables:

- `MODEL_PATH` (default: `models/lgbm_final_context_model.pkl`)
- `COLUMNS_PATH` (default: `models/model_columns.json`)

### 4) CLI (Typer)

```bash
python -m src.cli predict --age 22.5 --height-in-cm 185 \
  --current-club-name "Bayer 04 Leverkusen" \
  --player-club-domestic-competition-id DE1 \
  --minutes-played 2800 --goals 15 --assists 10 \
  --contract-months-remaining 36
```

Or with a JSON input file:

```bash
python -m src.cli predict --input player.json
```

### 5) API (FastAPI)

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Test:

- Health: http://localhost:8000/healthz
- Prediction (POST http://localhost:8000/predict)

Example payload:

```json
{
  "age": 22.5,
  "height_in_cm": 185,
  "current_club_name": "Bayer 04 Leverkusen",
  "player_club_domestic_competition_id": "DE1",
  "minutes_played": 2800,
  "goals": 15,
  "assists": 10,
  "contract_months_remaining": 36
}
```

### 6) Docker

```bash
docker build -t players-value:latest .
docker run --rm -p 8000:8000 players-value:latest
```

### 7) Quality and Testing

```bash
pre-commit install
pre-commit run --all-files
pytest -q
```

### 8) Validation and Monitoring

```bash
python -m src.validation.validate_master

python -m src.monitoring.drift_report \
  data/processed/contextual_featured_dataset.csv \
  data/processed/contextual_featured_dataset.csv \
  --output reports/drift_report.html
```

---

For training or retraining details, consult `src/train.py` and `configs/hparams.yaml`. To adjust context (league strength, tier-1 clubs), edit `configs/context.yaml`.
