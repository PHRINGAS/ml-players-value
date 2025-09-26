# Project Scope Document: Football Player Valuation System

This document details the objectives, metrics, decisions, and limitations of the player valuation project.

## 1. Business Vision and Objectives

**Vision:** Create a decision-support tool for football professionals (scouts, sporting directors) that is objective, data-driven, and capable of quantifying a football player's market value.

**Objective:** Develop a Machine Learning model that estimates the market value (in euros) of players from major leagues, surpassing a simple baseline model and providing insights into the factors that drive this value.

## 2. Machine Learning Problem Definition

*   **Problem Type:** Supervised Regression.
*   **Target Variable:** `market_value_in_eur`.
    *   **Transformation:** A logarithmic transformation (`np.log1p`) is applied to the target variable before training. This normalizes its distribution (which is heavily right-skewed) and reduces the disproportionate influence of extremely high-value players (outliers). The model predicts on the logarithmic scale and the result is converted back to euros (`np.expm1`) for interpretation.
*   **Unit of Analysis:** A player at a specific point in time, defined by the date of their last available market valuation.

## 3. Success Criteria and Metrics

*   **Primary Business Metric:** Mean Absolute Error (MAE) in Euros. The goal is to minimize this metric.
*   **Model Optimization Metric:** MAE on the transformed value (`MAE_log`). This was the metric used by Optuna for hyperparameter optimization.
*   **Baseline:** The final model's performance is compared against a simple Ridge model.
    *   **Initial Baseline MAE:** **€3,982,015**.
*   **Qualitative Analysis:** Success is also measured by the model's ability to identify the correct value drivers (see feature importance analysis) and by the mitigation of systematic biases identified in the error analysis.

## 4. Scope and Limitations (Out of Scope)

*   **Real-Time Scraping:** This project uses a static snapshot of Kaggle data. No continuous scraping system is implemented.
*   **Real-Time Predictions:** The system is designed for "batch" predictions through a script (`predictor.py`), not for deployment in a low-latency cloud API.
*   **Advanced Features Not Included:** More complex and difficult-to-quantify factors were deliberately excluded, such as: detailed injury history, social media popularity metrics, agent quality, or specific contractual clauses.
*   **League Coverage:** The model focuses primarily on the 5 major European leagues and other main competitions present in the dataset. Its accuracy in minor leagues is not guaranteed.

## 5. Key Project Decisions (Decision Log)

*   **Data Source:** We decided to use David Cariboo's Kaggle dataset as the primary source instead of multiple disparate sources. This decision was crucial as it provided unique player IDs, eliminating the need for complex and fragile data fusions based on names (fuzzy matching).
*   **Iterative Modeling Strategy:** We followed a continuous improvement process:
    1.  **Baseline (Ridge):** Established an initial MAE of ~€3.98M.
    2.  **Advanced Model (LightGBM):** The choice of a gradient boosting model and the first round of feature engineering reduced the MAE to ~€3.23M.
    3.  **Optimization (Optuna):** Fine-tuning of hyperparameters provided a marginal improvement, bringing the MAE to ~€3.22M.
    4.  **Final Model (Contextual):** The second round of feature engineering, focused on context (league strength, club prestige), achieved the most significant improvement, reaching a **final MAE of ~€2.92M**.
*   **Context Focus:** Error analysis of intermediate models revealed that the biggest weakness was the lack of context. Therefore, we prioritized the creation of `league_strength_factor` and `club_tier` features, which proved to be the key to the final quality leap of the model.