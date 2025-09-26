import pandas as pd
import numpy as np
import os

# Definir rutas de entrada y salida
DATA_DIR = 'data/raw/davidcariboo_dataset/'
OUTPUT_DIR = 'data/processed/'

def load_data():
    """Carga los dataframes necesarios desde la carpeta raw."""
    print("Cargando datos crudos...")
    players_df = pd.read_csv(f'{DATA_DIR}players.csv')
    valuations_df = pd.read_csv(f'{DATA_DIR}player_valuations.csv')
    appearances_df = pd.read_csv(f'{DATA_DIR}appearances.csv')
    return players_df, valuations_df, appearances_df

def get_latest_valuations(valuations_df):
    """Procesa el historial de valoraciones para obtener la más reciente por usuario.

    Conserva además el identificador de competición doméstica del club (`player_club_domestic_competition_id`)
    para su uso en features contextuales posteriores.
    """
    print("Procesando valoraciones...")
    valuations_df['date'] = pd.to_datetime(valuations_df['date'])
    valuations_df = valuations_df.sort_values(by=['player_id', 'date'], ascending=[True, False])
    # Mantener columnas relevantes
    keep_cols = ['player_id', 'date', 'market_value_in_eur']
    if 'player_club_domestic_competition_id' in valuations_df.columns:
        keep_cols.append('player_club_domestic_competition_id')
    latest_valuations = valuations_df[keep_cols].drop_duplicates(subset='player_id', keep='first')
    return latest_valuations

def get_seasonal_stats(appearances_df, start_date='2022-08-01', end_date='2023-07-31'):
    """Agrega las estadísticas de rendimiento por usuario para una temporada específica."""
    print(f"Agregando estadísticas de la temporada {start_date} a {end_date}...")
    appearances_df['date'] = pd.to_datetime(appearances_df['date'])
    season_apps = appearances_df[
        (appearances_df['date'] >= start_date) & (appearances_df['date'] <= end_date)
    ]
    player_stats = season_apps.groupby('player_id').agg(
        goals=('goals', 'sum'),
        assists=('assists', 'sum'),
        minutes_played=('minutes_played', 'sum'),
        games_played=('game_id', 'count')
    ).reset_index()
    return player_stats

def create_master_table(players_df, latest_valuations, player_stats):
    """Une todas las fuentes de datos en una tabla maestra, manejando conflictos de nombres."""
    print("Creando tabla maestra...")
    target_col = 'market_value_in_eur'
    
    # Merge que causa el conflicto
    # Unir players con las últimas valoraciones (manteniendo competition_id si existe)
    cols = ['player_id', 'date', target_col]
    if 'player_club_domestic_competition_id' in latest_valuations.columns:
        cols.append('player_club_domestic_competition_id')
    master_df = pd.merge(players_df, latest_valuations[cols], on='player_id', how='inner')

    # --- INICIO DE LA CORRECCIÓN ---
    # Replicar la lógica del notebook para manejar columnas duplicadas
    col_x = f'{target_col}_x'
    col_y = f'{target_col}_y'

    if col_y in master_df.columns:
        print(f"Detectadas columnas duplicadas. Corrigiendo...")
        # La columna '_y' viene de 'latest_valuations', que es la que queremos.
        master_df[target_col] = master_df[col_y]
        # Eliminar las columnas con sufijos que ya no necesitamos.
        master_df = master_df.drop(columns=[col_x, col_y])
    # --- FIN DE LA CORRECCIÓN ---

    final_df = pd.merge(master_df, player_stats, on='player_id', how='inner')
    return final_df

def clean_and_prepare_data(final_df):
    """Aplica filtros finales y crea la feature 'age'."""
    print("Limpiando y preparando datos para modelado...")
    target_col = 'market_value_in_eur'
    required_cols = [target_col, 'goals', 'assists', 'minutes_played', 'date_of_birth']
    
    df_filtered = final_df.dropna(subset=required_cols)
    df_filtered = df_filtered[df_filtered[target_col] > 0]
    df_filtered = df_filtered[df_filtered['minutes_played'] > 90]
    
    df_model = df_filtered.copy()
    
    # Feature Engineering: Calcular la edad en el momento de la valorización
    df_model['date_of_birth'] = pd.to_datetime(df_model['date_of_birth'])
    df_model['valuation_date'] = pd.to_datetime(df_model['date']) # Renombrar por claridad
    df_model['age'] = (df_model['valuation_date'] - df_model['date_of_birth']).dt.days / 365.25
    
    # Seleccionar y renombrar columnas finales
    final_cols = [
        'player_id', 'name', 'age', 'position', 'sub_position', 'foot', 'height_in_cm',
        'country_of_citizenship', 'current_club_name', 'contract_expiration_date',
        'player_club_domestic_competition_id',
        'goals', 'assists', 'minutes_played', 'games_played',
        'valuation_date', 'market_value_in_eur'
    ]
    # Filtrar para quedarse solo con las columnas que existen
    final_cols_exist = [col for col in final_cols if col in df_model.columns]
    df_model = df_model[final_cols_exist]

    return df_model

def main():
    """Función principal que orquesta todo el pipeline de ETL."""
    players, valuations, appearances = load_data()
    latest_vals = get_latest_valuations(valuations)
    stats = get_seasonal_stats(appearances)
    master = create_master_table(players, latest_vals, stats)
    model_ready_data = clean_and_prepare_data(master)
    
    # Asegurarse de que el directorio de salida exista
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Guardar el resultado
    output_path = f'{OUTPUT_DIR}master_player_dataset.csv'
    model_ready_data.to_csv(output_path, index=False)
    print(f"\n¡Proceso completado! Dataset maestro guardado en: {output_path}")
    print(f"Dimensiones finales: {model_ready_data.shape}")

if __name__ == '__main__':
    main()