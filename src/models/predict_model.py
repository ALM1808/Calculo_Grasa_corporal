# src/models/predict_model.py

import joblib
import pandas as pd
from pathlib import Path

# Ruta al modelo entrenado
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "rf_pipeline_optuna.pkl"

# Cargar pipeline entrenado
model = joblib.load(MODEL_PATH)

def predict_fat_percentage(new_data: pd.DataFrame) -> float:
    """
    Realiza una predicción de porcentaje de grasa corporal.
    
    Parámetros:
        new_data (pd.DataFrame): DataFrame con una sola fila con las mismas columnas que el entrenamiento.
    
    Retorna:
        float: Predicción de grasa corporal.
    """
    if new_data.shape[0] != 1:
        raise ValueError("Solo se admite una fila por predicción.")
    
    prediction = model.predict(new_data)
    return float(prediction[0])