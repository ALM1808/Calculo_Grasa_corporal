# backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pandas as pd
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import os
import hashlib
from datetime import datetime
from dotenv import load_dotenv

# Cargar .env desde la raíz del proyecto
load_dotenv()

# --------------------- Config Hopsworks ---------------------
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
# Para hopsworks.login el host correcto es SOLO el dominio
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")

# --------------------- FastAPI ---------------------
app = FastAPI(
    title="Grasa Corporal API",
    description="API para predecir % de grasa corporal y registrar actividad en Hopsworks",
    version="1.0.0",
)

# Hilo para tareas en segundo plano (Hopsworks)
executor = ThreadPoolExecutor(max_workers=2)

# --------------------- Esquemas de entrada ---------------------
class PredictionInput(BaseModel):
    email: str
    age: int = Field(..., ge=10, le=100)
    gender: Literal["Male", "Female"]
    weight_kg: float = Field(..., ge=30.0, le=200.0)
    height_m: float = Field(..., ge=1.2, le=2.2)
    max_bpm: int
    avg_bpm: int
    resting_bpm: int
    session_duration_hours: float
    calories_burned: float
    workout_type: Literal["Cardio", "Strength", "Mixed"]
    water_intake_liters: float
    workout_frequency_days_week: int = Field(..., ge=1, le=7)
    experience_level: Literal["1", "2", "3"]


class FeedbackInput(BaseModel):
    email: str
    real_fat_percentage: float = Field(..., ge=1, le=80)
    predicted_fat_percentage: Optional[float] = None


# --------------------- Modelo ---------------------
@lru_cache(maxsize=1)
def get_model():
    from src.models.predict_model import load_model
    return load_model()


# --------------------- Mapping columnas ---------------------
SNAKE_TO_MODEL = {
    "age": "Age",
    "gender": "Gender",
    "weight_kg": "Weight (kg)",
    "height_m": "Height (m)",
    "max_bpm": "Max_BPM",
    "avg_bpm": "Avg_BPM",
    "resting_bpm": "Resting_BPM",
    "session_duration_hours": "Session_Duration (hours)",
    "calories_burned": "Calories_Burned",
    "workout_type": "Workout_Type",
    "water_intake_liters": "Water_Intake (liters)",
    "workout_frequency_days_week": "Workout_Frequency (days/week)",
    "experience_level": "Experience_Level",
}

EXPECTED_MODEL_COLS = [
    "Age",
    "Gender",
    "Weight (kg)",
    "Height (m)",
    "Max_BPM",
    "Avg_BPM",
    "Resting_BPM",
    "Session_Duration (hours)",
    "Calories_Burned",
    "Workout_Type",
    "Water_Intake (liters)",
    "Workout_Frequency (days/week)",
    "Experience_Level",
    "BMI",
    "Log_Age",
]


# --------------------- Utilidades ---------------------
def generate_user_id(email: str) -> str:
    return hashlib.sha256(email.lower().encode()).hexdigest()[:16]


def _get_hopsworks_project():
    """Lanza login a Hopsworks o lanza excepción si falta config."""
    if not (HOPSWORKS_PROJECT and HOPSWORKS_API_KEY):
        raise RuntimeError("Hopsworks env vars missing")

    import hopsworks

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
    )
    return project


# -------- Guardado de predicciones en Hopsworks (sin bloquear) --------
def _save_prediction_sync(df_model: pd.DataFrame, email: str, prediction: float):
    try:
        project = _get_hopsworks_project()
    except Exception:
        # Si falla Hopsworks, NO rompemos la API
        return

    fs = project.get_feature_store()

    # Mapear nombres del modelo -> snake_case usados en el FG
    model_to_snake = {v: k for k, v in SNAKE_TO_MODEL.items()}
    model_to_snake.update({"BMI": "bmi", "Log_Age": "log_age"})

    df = df_model.rename(columns=model_to_snake).copy()

    # Añadir metadatos
    now = datetime.utcnow()
    df["email"] = email
    df["user_id"] = df["email"].apply(generate_user_id)
    df["timestamp"] = now
    df["predicted_fat_percentage"] = float(prediction)
    if "real_fat_percentage" not in df.columns:
        df["real_fat_percentage"] = np.nan

    # Tipos
    int_cols = [
        "age",
        "workout_frequency_days_week",
        "experience_level",
        "max_bpm",
        "avg_bpm",
        "resting_bpm",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    float_cols = [
        "weight_kg",
        "height_m",
        "session_duration_hours",
        "calories_burned",
        "water_intake_liters",
        "bmi",
        "log_age",
    ]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    # Definir columnas del Feature Group principal
    expected_fg_cols = [
        "user_id",
        "email",
        "timestamp",
        "age",
        "gender",
        "weight_kg",
        "height_m",
        "max_bpm",
        "avg_bpm",
        "resting_bpm",
        "session_duration_hours",
        "calories_burned",
        "workout_type",
        "water_intake_liters",
        "workout_frequency_days_week",
        "experience_level",
        "bmi",
        "log_age",
        "predicted_fat_percentage",
        "real_fat_percentage",
    ]

    for c in expected_fg_cols:
        if c not in df.columns:
            df[c] = np.nan

    out = df[expected_fg_cols].copy()

    fg = fs.get_or_create_feature_group(
        name="user_fat_percentage",
        version=1,
        description="Registros de predicción de grasa corporal desde la API",
        primary_key=["user_id", "email", "timestamp"],
        event_time="timestamp",
        online_enabled=True,
    )

    fg.insert(out, write_options={"wait_for_job": False})


def save_prediction_to_hopsworks(df_model: pd.DataFrame, email: str, prediction: float):
    # Lanzamos en segundo plano; la petición /predict responde rápido.
    executor.submit(_save_prediction_sync, df_model.copy(), email, float(prediction))


# -------- Guardado de feedback en Hopsworks (sin bloquear) --------
def _save_feedback_sync(email: str, real_value: float, predicted: Optional[float]):
    try:
        project = _get_hopsworks_project()
    except Exception:
        return

    fs = project.get_feature_store()

    pfg = fs.get_or_create_feature_group(
        name="user_fat_feedback",
        version=1,
        description="Valores reales de grasa corporal enviados por usuarios",
        primary_key=["user_id", "email", "timestamp"],
        event_time="timestamp",
        online_enabled=False,
    )

    row = pd.DataFrame(
        [
            {
                "user_id": generate_user_id(email),
                "email": email,
                "timestamp": datetime.utcnow(),
                "real_fat_percentage": float(real_value),
                "predicted_fat_percentage": float(predicted)
                if predicted is not None
                else np.nan,
            }
        ]
    )

    pfg.insert(row, write_options={"wait_for_job": False})


def save_feedback_to_hopsworks(email: str, real_value: float, predicted: Optional[float]):
    executor.submit(_save_feedback_sync, email, float(real_value), predicted)


# --------------------- Endpoints ---------------------
@app.get("/")
def root():
    return {"message": "API de predicción de grasa corporal 🧠"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/hopsworks-status")
def hopsworks_status():
    try:
        _get_hopsworks_project()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # 1) Datos en snake_case (del request)
        data = input_data.dict()
        email = data.pop("email")
        df = pd.DataFrame([data])

        # 2) Features derivadas con nombres del modelo
        df["BMI"] = df["weight_kg"] / (df["height_m"] ** 2)
        df["Log_Age"] = np.log1p(df["age"])

        # 3) Renombrar a columnas esperadas por el pipeline
        df_model = df.rename(columns=SNAKE_TO_MODEL)

        # 4) Tipos correctos
        if "Experience_Level" in df_model.columns:
            df_model["Experience_Level"] = (
                pd.to_numeric(df_model["Experience_Level"], errors="coerce")
                .fillna(1)
                .astype(int)
            )
        for c in ("Gender", "Workout_Type"):
            if c in df_model.columns:
                df_model[c] = df_model[c].astype(str)

        # 5) Validar columnas
        missing = set(EXPECTED_MODEL_COLS) - set(df_model.columns)
        if missing:
            raise HTTPException(status_code=500, detail=f"columns are missing: {missing}")

        df_model = df_model[EXPECTED_MODEL_COLS]

        # 6) Predicción
        model = get_model()
        yhat = model.predict(df_model)[0]
        pred = round(float(yhat), 2)

        # 7) Guardar en Hopsworks en segundo plano
        save_prediction_to_hopsworks(df_model, email, pred)

        return {"predicted_fat_percentage": pred}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def feedback(data: FeedbackInput):
    try:
        save_feedback_to_hopsworks(
            email=data.email,
            real_value=data.real_fat_percentage,
            predicted=data.predicted_fat_percentage,
        )
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))








