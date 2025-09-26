# Sistema Predictivo para la Valoración de Futbolistas (v1.0)

Este repositorio documenta la construcción de un sistema de Machine Learning de extremo a extremo para estimar el valor de mercado de futbolistas. El proyecto abarca desde la ingesta y limpieza de datos hasta el entrenamiento, la optimización y el empaquetado de un modelo predictivo listo para ser utilizado.

---

## 🚀 El Desafío: El Valor Oculto en los Datos

La valoración de futbolistas es un proceso complejo, con millones de euros en juego en cada decisión de mercado. Aunque los scouts y analistas aportan una experiencia invaluable, el proceso a menudo carece de una base objetiva y cuantificable.

Este proyecto aborda el siguiente desafío: **¿Es posible construir un modelo que aprenda los patrones ocultos en los datos para estimar el valor de mercado de un jugador, basándose en su rendimiento, perfil demográfico y, crucialmente, su contexto competitivo?**

## ✨ Resultados Clave: Un Modelo Inteligente y Cuantificable

Tras un riguroso proceso de modelado iterativo, el sistema final es capaz de predecir el valor de mercado con un **Error Absoluto Medio (MAE) de €2,921,504**.

*   **Mejora Masiva:** Esto representa una **reducción total del error del 26.63%** (más de €1,060,000 de precisión añadida) en comparación con el modelo baseline inicial.
*   **Inteligencia Contextual:** El análisis de importancia de características reveló que los factores más determinantes no son solo los goles o las asistencias, sino el **contexto del jugador**:
    1.  **Situación Contractual** (`contract_months_remaining`)
    2.  **Edad y Potencial** (`age`)
    3.  **Calidad de la Competición** (`league_strength_factor`)
    4.  **Prestigio del Club** (`club_tier`)

*(Aquí puedes insertar una imagen del gráfico de importancia de características para un impacto visual)*

## 🧠 El Proceso: De Datos Crudos a un Modelo Predictivo

El éxito del proyecto se basa en un enfoque iterativo, donde cada fase construyó sobre la anterior:

1.  **Fundamentos y Baseline:** Se estableció un proceso de ETL robusto para limpiar y unificar múltiples fuentes de datos. Un modelo simple de Regresión Ridge estableció nuestro baseline con un **MAE de €3.98M**.

2.  **Ingeniería de Características (Iteración 1):** Se crearon características de rendimiento normalizadas (ej. `goals_per_90`) y de negocio (ej. `contract_months_remaining`). Esto permitió a un modelo LightGBM base reducir el MAE a **€3.23M**, una mejora del 18.7%.

3.  **Optimización de Hiperparámetros:** Usando **Optuna**, se realizó una búsqueda exhaustiva (300 trials) para encontrar la configuración óptima del modelo LightGBM, afinando su rendimiento y robustez.

4.  **Análisis de Errores y el "Factor Contexto" (Iteración 2):** Se analizó dónde fallaba el modelo, descubriendo una subestimación sistemática de jugadores de élite y una sobreestimación de jugadores en ligas menores. Esto guio la creación de **features contextuales** (`league_strength_factor`, `club_tier`), que llevaron al modelo final a su MAE de **€2.92M**, el salto cualitativo definitivo.

## 🛠️ Stack Tecnológico

*   **Análisis y Modelado:** Python, Pandas, Scikit-learn, LightGBM
*   **Optimización de Hiperparámetros:** Optuna
*   **Versionado de Artefactos:** Git, DVC (Data Version Control) para los datasets
*   **Entorno y Prototipado:** Jupyter Notebooks
*   **Código de Producción:** Scripts `.py` modulares

## 📂 Estructura del Repositorio

├── data/ # Directorio para todos los datasets (gestionado por DVC).
│ ├── raw/ # Datos originales, sin modificar.
│ └── processed/ # Datasets limpios y enriquecidos, listos para modelado.
│
├── models/ # Contiene los artefactos del modelo final entrenado.
│ ├── lgbm_final_context_model.pkl # El objeto del modelo serializado.
│ └── model_columns.json # Lista de features que el modelo espera.
│
├── notebooks/ # Proceso de exploración y desarrollo (Jupyter Notebooks).
│ ├── 00_baselines.ipynb
│ ├── 01_eda_cariboo_dataset.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_modeling.ipynb
│ ├── 04_hyperparameter_tuning.ipynb
│ ├── 05_error_analysis.ipynb
│ ├── 06_contextual_feature_engineering.ipynb
│ ├── 07_final_model_evaluation.ipynb
│ └── 08_final_error_analysis.ipynb
│
├── src/ # Código fuente de producción.
│ ├── etl/ # Scripts para la ingesta y procesamiento de datos (ETL).
│ └── predictor.py # Script final empaquetado para realizar predicciones.
│
├── .gitignore # Especifica qué archivos ignorar en Git.
├── README.md # Documentación principal del proyecto (este archivo).
├── requirements.txt # Lista de dependencias de Python para reproducibilidad.
└── scoping.md # Documento de definición del alcance inicial del proyecto.


## 🏁 Cómo Ejecutar el Proyecto

Sigue estos pasos para configurar el entorno y realizar una predicción de ejemplo.

### Pre-requisitos
*   Python 3.9+
*   Git
*   DVC

### Pasos de Instalación y Ejecución

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/valoracion-jugadores.git
    cd valoracion-jugadores
    ```

2.  **Configura el entorno virtual y las dependencias:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Descarga los datos versionados con DVC:**
    *DVC gestiona los datasets para mantener el repositorio ligero. Este comando descarga los datos necesarios para que el proyecto funcione.*
    ```bash
    dvc pull
    ```

4.  **Ejecuta el predictor de ejemplo:**
    El script `src/predictor.py` cargará el modelo final y predecirá el valor de un jugador de ejemplo definido en el código.
    ```bash
    python src/predictor.py
    ```
    Output esperado:
    ```
    Cargando artefactos del modelo...
    Predictor inicializado exitosamente.

    --- Demostración de Predicción ---
    Datos del jugador: Attack de 22.5 años en Bayer 04 Leverkusen
    Valor de Mercado Estimado: €16,813,992
    ```