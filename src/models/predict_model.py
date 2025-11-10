# src/models/predict_model.py

import joblib
import pandas as pd
from pathlib import Path


# --- Ruta al modelo entrenado ---
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "rf_pipeline_optuna.pkl"


# --- Función para cargar el modelo ---
def load_model(path: Path = MODEL_PATH):
    """
    Carga el modelo desde el archivo especificado.

    Parámetros:
        path (Path): Ruta al archivo .pkl del modelo.

    Retorna:
        Modelo cargado (pipeline de scikit-learn).
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {path}")
    model = joblib.load(path)
    return model


# --- Carga inicial (opcional, útil para evitar recarga repetida) ---
model = load_model()


# --- Función de predicción ---
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
