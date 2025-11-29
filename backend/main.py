# backend/main.py
import sys

# Evitar nest_asyncio si ya está cargado
if "nest_asyncio" in sys.modules:
    del sys.modules["nest_asyncio"]

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional

import os
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import requests

from functools import lru_cache

app = FastAPI(title="Grasa Corporal API", version="2.1")


# ============================================================
# MODELOS DE ENTRADA
# ============================================================

class PredictionInput(BaseModel):
    email: str
    age: int
    gender: Literal["Male", "Female"]
    weight_kg: float
    height_m: float
    max_bpm: int
    avg_bpm: int
    resting_bpm: int
    session_duration_hours: float
    calories_burned: float
    workout_type: Literal["Cardio", "Strength", "Mixed"]
    water_intake_liters: float
    workout_frequency_days_week: int
    experience_level: Literal["1", "2", "3"]


class FeedbackInput(BaseModel):
    email: str
    real_fat_percentage: float
    predicted_fat_percentage: Optional[float] = None


# ============================================================
# CARGA DEL MODELO
# ============================================================

@lru_cache(maxsize=1)
def get_model():
    candidates = [
        Path("models") / "rf_pipeline.pkl",
        Path("/app/models") / "rf_pipeline.pkl",
    ]
    tried = []

    for p in candidates:
        tried.append(str(p))
        if p.exists():
            print(f"[MODEL] Cargado desde {p}", flush=True)
            return joblib.load(p)

    raise FileNotFoundError(
        "No se encontró rf_pipeline.pkl.\nRutas probadas:\n" +
        "\n".join(tried)
    )


# ============================================================
# UTILIDADES
# ============================================================

def generate_user_id(email: str) -> str:
    return hashlib.sha256(email.lower().encode()).hexdigest()[:16]


def current_timestamp():
    return datetime.utcnow().isoformat()


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


# ============================================================
# LOGS LOCALES EN CSV
# ============================================================

LOG_PATH = Path("/app/backend/data_logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)


def append_to_csv(filename: str, row: dict):
    fpath = LOG_PATH / filename
    df = pd.DataFrame([row])

    if fpath.exists():
        df.to_csv(fpath, mode="a", header=False, index=False)
    else:
        df.to_csv(fpath, index=False)


# ============================================================
# INSERCIÓN A HOPSWORKS VIA REST   (Opción A)
# ============================================================

def hopsworks_rest_insert(feature_group: dict, payload_row: dict):
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = os.getenv("HOPSWORKS_PROJECT")
    host = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")

    if not api_key or not project:
        print("[HOPSWORKS] Variables de entorno faltantes. No se inserta nada.")
        return

    url = (
        f"https://{host}/_api/fs/v1/projects/{project}/"
        f"featuregroups/{feature_group['name']}/insert"
    )

    headers = {
        "Authorization": f"ApiKey {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "featureGroupVersion": feature_group["version"],
        "dataframe": {
            "columns": list(payload_row.keys()),
            "records": [list(payload_row.values())]
        }
    }

    try:
        r = requests.post(url, headers=headers, json=payload)
        if 200 <= r.status_code < 300:
            print("[HOPSWORKS] Inserción OK")
        else:
            print("[HOPSWORKS] Error:", r.status_code, r.text)
    except Exception as e:
        print("[HOPSWORKS] Excepción REST:", e)


def save_prediction_to_hopsworks(row_snake: pd.Series, pred: float):
    try:
        email = row_snake["email"]

        payload = row_snake.to_dict()

        payload["bmi"] = payload["weight_kg"] / (payload["height_m"] ** 2)
        payload["log_age"] = float(np.log1p(payload["age"]))

        payload["user_id"] = generate_user_id(email)
        payload["timestamp"] = current_timestamp()
        payload["predicted_fat_percentage"] = float(pred)
        payload["real_fat_percentage"] = None

        fg = {"name": "user_fat_percentage", "version": 1}

        hopsworks_rest_insert(fg, payload)

    except Exception as e:
        print("[HOPSWORKS] Error save_prediction:", e)


def save_feedback_to_hopsworks(email: str, real: float, pred: Optional[float]):
    try:
        payload = {
            "user_id": generate_user_id(email),
            "email": email,
            "timestamp": current_timestamp(),
            "real_fat_percentage": float(real),
            "predicted_fat_percentage": float(pred) if pred else None,
        }

        fg = {"name": "user_fat_feedback", "version": 1}

        hopsworks_rest_insert(fg, payload)

    except Exception as e:
        print("[HOPSWORKS] Error save_feedback:", e)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {"message": "API GrasaCorporal OK"}


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        data = input_data.dict()
        email = data["email"].lower().strip()

        df_snake = pd.DataFrame([data])
        df_snake["email"] = email
        df_snake["experience_level"] = df_snake["experience_level"].astype(int)

        df_model = df_snake.rename(columns=SNAKE_TO_MODEL).copy()
        df_model["BMI"] = df_snake["weight_kg"] / (df_snake["height_m"] ** 2)
        df_model["Log_Age"] = np.log1p(df_snake["age"].astype(float))

        missing = set(EXPECTED_MODEL_COLS) - set(df_model.columns)
        if missing:
            raise HTTPException(
                500,
                f"Faltan columnas para el modelo: {missing}"
            )

        df_model = df_model[EXPECTED_MODEL_COLS]

        model = get_model()
        pred = round(float(model.predict(df_model)[0]), 2)

        # LOG local
        append_to_csv("predictions.csv", {
            "timestamp": current_timestamp(),
            "email": email,
            "predicted_fat_percentage": pred,
            **data
        })

        # Enviar a Hopsworks
        save_prediction_to_hopsworks(df_snake.iloc[0], pred)

        return {"predicted_fat_percentage": pred}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error en predicción: {e}")


@app.post("/feedback")
def feedback(input_data: FeedbackInput):
    try:
        email = input_data.email.lower().strip()

        append_to_csv("feedback.csv", {
            "timestamp": current_timestamp(),
            "email": email,
            "real_fat_percentage": input_data.real_fat_percentage,
            "predicted_fat_percentage": input_data.predicted_fat_percentage
        })

        save_feedback_to_hopsworks(
            email=email,
            real=input_data.real_fat_percentage,
            pred=input_data.predicted_fat_percentage
        )

        return {"status": "ok"}

    except Exception as e:
        raise HTTPException(500, f"Error en feedback: {e}")


# -------------------------------------------------------
# ENDPOINT /history
# -------------------------------------------------------
@app.get("/history/{email}")
def history(email: str):
    try:
        email = email.strip().lower()
        fpath = LOG_PATH / "predictions.csv"

        if not fpath.exists():
            return {"records": []}

        df = pd.read_csv(fpath)

        # Filtrar por email
        df = df[df["email"].str.lower() == email]

        # Ordenar por fecha
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

        # Convertir a registros JSON serializables
        records = df.to_dict(orient="records")

        return {"records": records}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo histórico: {e}")





















