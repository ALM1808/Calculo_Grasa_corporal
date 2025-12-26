# backend/main.py
import sys
if "nest_asyncio" in sys.modules:
    del sys.modules["nest_asyncio"]

import os
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from typing import Literal, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter


# ======================================================
# LOGGING
# ======================================================
logger = logging.getLogger("grasa-backend")
logger.setLevel(logging.INFO)

if os.getenv("K_SERVICE"):
    # Cloud Run
    try:
        import google.cloud.logging
        from google.cloud.logging_v2.handlers import CloudLoggingHandler

        logging_client = google.cloud.logging.Client()
        handler = CloudLoggingHandler(logging_client)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        if not any(isinstance(h, CloudLoggingHandler) for h in root_logger.handlers):
            root_logger.addHandler(handler)

        logger.info("Cloud Logging activado (Cloud Run)")
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logger.warning(f"No se pudo inicializar Cloud Logging: {e}")
else:
    logging.basicConfig(level=logging.INFO)
    logger.info("Logging local activado (sin Cloud Logging)")


# ======================================================
# APP
# ======================================================
app = FastAPI(title="Grasa Corporal API", version="2.6")


# ======================================================
# SCHEMAS
# ======================================================
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


class PredictionResponse(BaseModel):
    predicted_fat_percentage: float
    prediction_id: str
    timestamp: str  # ISO


class FeedbackInput(BaseModel):
    email: str
    real_fat_percentage: float
    predicted_fat_percentage: Optional[float] = None
    prediction_id: Optional[str] = None  # <- CLAVE para enlazar


# ======================================================
# MODEL
# ======================================================
@lru_cache(maxsize=1)
def get_model():
    for p in [Path("models/rf_pipeline.pkl"), Path("/app/models/rf_pipeline.pkl")]:
        if p.exists():
            logger.info(f"Modelo cargado desde {p}")
            return joblib.load(p)
    raise FileNotFoundError("rf_pipeline.pkl no encontrado")


# ======================================================
# UTILS
# ======================================================
def generate_user_id(email: str) -> str:
    return hashlib.sha256(email.lower().encode()).hexdigest()[:16]


def utc_now_sec() -> datetime:
    # datetime aware UTC sin microsegundos (consistente para IDs + orden)
    return datetime.now(timezone.utc).replace(microsecond=0)


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

EXPECTED_MODEL_COLS = list(SNAKE_TO_MODEL.values()) + ["BMI", "Log_Age"]


# CSV backup (local / best effort)
LOG_PATH = Path(os.getenv("LOG_PATH", "/tmp/data_logs"))
LOG_PATH.mkdir(parents=True, exist_ok=True)

def append_to_csv(filename: str, row: Dict[str, Any]) -> None:
    fpath = LOG_PATH / filename
    pd.DataFrame([row]).to_csv(
        fpath,
        mode="a" if fpath.exists() else "w",
        header=not fpath.exists(),
        index=False,
    )


# ======================================================
# FIRESTORE
# ======================================================
@lru_cache(maxsize=1)
def get_firestore():
    try:
        return firestore.Client()
    except Exception as e:
        logger.warning(f"Firestore no disponible: {e}")
        return None


def save_prediction_to_firestore(prediction_id: str, record: Dict[str, Any]) -> None:
    fs = get_firestore()
    if fs is None:
        logger.info("Firestore no inicializado: se omite save_prediction_to_firestore")
        return
    try:
        fs.collection("predictions").document(prediction_id).set(record)
    except Exception as e:
        logger.warning(f"Error guardando predicción en Firestore: {e}")


def update_prediction_with_feedback(
    prediction_id: str,
    real: float,
    pred: Optional[float],
) -> None:
    fs = get_firestore()
    if fs is None:
        logger.info("Firestore no inicializado: se omite update_prediction_with_feedback")
        return

    try:
        patch = {
            "real_fat_percentage": float(real),
            "feedback_timestamp": utc_now_sec(),
        }

        if pred is not None:
            patch["predicted_fat_percentage_feedback"] = float(pred)
            patch.update(compute_error_metrics(real, pred))

        fs.collection("predictions").document(prediction_id).set(
            patch,
            merge=True,
        )

    except Exception as e:
        logger.warning(f"Error actualizando métricas en Firestore: {e}")



def save_feedback_fallback(email: str, real: float, pred: Optional[float]) -> None:
    fs = get_firestore()
    if fs is None:
        logger.info("Firestore no inicializado: se omite save_feedback_fallback")
        return
    try:
        fs.collection("feedback").add({
            "email": email,
            "timestamp": utc_now_sec(),
            "real_fat_percentage": float(real),
            "predicted_fat_percentage": float(pred) if pred is not None else None,
        })
    except Exception as e:
        logger.warning(f"Fallo guardando feedback en Firestore (fallback): {e}")

def compute_error_metrics(real: float, pred: float) -> dict:
    abs_error = abs(real - pred)
    signed_error = real - pred
    relative_error = (abs_error / real * 100) if real != 0 else None

    return {
        "abs_error": round(abs_error, 3),
        "signed_error": round(signed_error, 3),
        "relative_error": round(relative_error, 2) if relative_error is not None else None,
    }

def aggregate_error_metrics(records: list) -> dict:
    """
    Calcula métricas agregadas a partir de records con feedback
    """
    df = pd.DataFrame(records)

    if df.empty:
        return {}

    # Solo registros con feedback real
    df = df[df["real_fat_percentage"].notna()].copy()

    if df.empty:
        return {}

    # Asegurar numéricos
    df["predicted_fat_percentage"] = pd.to_numeric(df["predicted_fat_percentage"], errors="coerce")
    df["real_fat_percentage"] = pd.to_numeric(df["real_fat_percentage"], errors="coerce")

    df["abs_error"] = (df["real_fat_percentage"] - df["predicted_fat_percentage"]).abs()
    df["signed_error"] = df["real_fat_percentage"] - df["predicted_fat_percentage"]
    df["relative_error"] = df["abs_error"] / df["real_fat_percentage"] * 100

    return {
        "n_feedback": int(len(df)),
        "mae": round(df["abs_error"].mean(), 3),
        "mean_signed_error": round(df["signed_error"].mean(), 3),
        "mean_relative_error_pct": round(df["relative_error"].mean(), 2),
    }

# ======================================================
# ENDPOINTS
# ======================================================
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    get_model()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput, background_tasks: BackgroundTasks):
    """
    /predict canónico:
    - Normaliza email
    - Crea timestamp UTC (aware)
    - Predice
    - Guarda en Firestore: predictions/{prediction_id}
    - Guarda CSV backup
    - Devuelve predicted + prediction_id + timestamp ISO
    """
    data = input_data.dict()
    email = data["email"].strip().lower()

    ts = utc_now_sec()
    prediction_id = f"{email.replace('@','_')}__{ts.isoformat()}"

    # DataFrame modelo
    df = pd.DataFrame([data])
    df["experience_level"] = df["experience_level"].astype(int)

    df_model = df.rename(columns=SNAKE_TO_MODEL)
    df_model["BMI"] = df["weight_kg"] / (df["height_m"] ** 2)
    df_model["Log_Age"] = np.log1p(df["age"])
    df_model = df_model[EXPECTED_MODEL_COLS]

    pred = round(float(get_model().predict(df_model)[0]), 2)

    # Registro “limpio” (sin age/gender en histórico si no lo quieres)
    record = {
        "prediction_id": prediction_id,
        "email": email,
        "timestamp": ts,  # Firestore timestamp nativo (datetime aware)
        "predicted_fat_percentage": float(pred),

        # campos útiles para histórico
        "weight_kg": float(data["weight_kg"]),
        "height_m": float(data["height_m"]),
        "workout_type": data["workout_type"],
        "session_duration_hours": float(data["session_duration_hours"]),
        "workout_frequency_days_week": int(data["workout_frequency_days_week"]),
        "avg_bpm": int(data["avg_bpm"]),
        "resting_bpm": int(data["resting_bpm"]),
        "max_bpm": int(data["max_bpm"]),
        "calories_burned": float(data["calories_burned"]),
        "water_intake_liters": float(data["water_intake_liters"]),
        "experience_level": int(data["experience_level"]),

        # feedback (se rellenará cuando llegue)
        "real_fat_percentage": None,
    }

    # Backup CSV (best effort)
    try:
        append_to_csv("predictions.csv", {
            **record,
            "timestamp": ts.isoformat(),  # en CSV lo guardamos como string
        })
    except Exception as e:
        logger.warning(f"No se pudo guardar CSV predictions.csv: {e}")

    # Firestore fuente de verdad
    background_tasks.add_task(save_prediction_to_firestore, prediction_id, record)

    return {
        "predicted_fat_percentage": pred,
        "prediction_id": prediction_id,
        "timestamp": ts.isoformat(),
    }


@app.post("/feedback")
def feedback(input_data: FeedbackInput, background_tasks: BackgroundTasks):
    """
    /feedback:
    - Si viene prediction_id -> actualiza ese doc en predictions (RECOMENDADO)
    - Si no viene -> guarda en colección feedback (fallback)
    """
    email = input_data.email.strip().lower()

    # CSV backup
    try:
        append_to_csv("feedback.csv", {
            "timestamp": utc_now_sec().isoformat(),
            "email": email,
            "real_fat_percentage": float(input_data.real_fat_percentage),
            "predicted_fat_percentage": float(input_data.predicted_fat_percentage) if input_data.predicted_fat_percentage is not None else None,
            "prediction_id": input_data.prediction_id,
        })
    except Exception as e:
        logger.warning(f"No se pudo guardar CSV feedback.csv: {e}")

    if input_data.prediction_id:
        background_tasks.add_task(
            update_prediction_with_feedback,
            input_data.prediction_id,
            float(input_data.real_fat_percentage),
            float(input_data.predicted_fat_percentage) if input_data.predicted_fat_percentage is not None else None,
        )
    else:
        background_tasks.add_task(
            save_feedback_fallback,
            email,
            float(input_data.real_fat_percentage),
            float(input_data.predicted_fat_percentage) if input_data.predicted_fat_percentage is not None else None,
        )

    return {"status": "ok"}


@app.get("/history")
def history(email: str = Query(..., min_length=3)):
    """
    /history (Firestore):
    - Lee de predictions filtrando por email
    - Ordena por timestamp
    - Devuelve timestamps ISO + tipos JSON-friendly
    """
    email = email.strip().lower()
    fs = get_firestore()
    if fs is None:
        logger.warning("Firestore no disponible")
        return {"records": []}

    try:
        query = (
            fs.collection("predictions")
            .where(filter=FieldFilter("email", "==", email))
            .order_by("timestamp")
        )

        records = []
        for doc in query.stream():
            data = doc.to_dict() or {}

            ts = data.get("timestamp")
            if ts is not None and hasattr(ts, "isoformat"):
                data["timestamp"] = ts.isoformat()
            else:
                # fallback por si algún doc viejo no lo tiene
                ct = getattr(doc, "create_time", None)
                data["timestamp"] = ct.isoformat() if (ct is not None and hasattr(ct, "isoformat")) else None

            # asegurar floats
            if data.get("predicted_fat_percentage") is not None:
                try:
                    data["predicted_fat_percentage"] = float(data["predicted_fat_percentage"])
                except Exception:
                    pass

            if data.get("real_fat_percentage") is not None:
                try:
                    data["real_fat_percentage"] = float(data["real_fat_percentage"])
                except Exception:
                    pass

            records.append(data)

        return {"records": jsonable_encoder(records)}

    except Exception as e:
        logger.exception(f"Error cargando histórico Firestore: {e}")
        return {"records": []}

@app.get("/metrics")
def metrics(email: Optional[str] = Query(None)):
    """
    Métricas agregadas:
    - Globales si no hay email
    - Por usuario si se pasa email
    """
    fs = get_firestore()
    if fs is None:
        return {"metrics": {}}

    try:
        query = fs.collection("predictions")

        if email:
            email = email.strip().lower()
            query = query.where(filter=FieldFilter("email", "==", email))

        records = []
        for doc in query.stream():
            data = doc.to_dict() or {}
            if "real_fat_percentage" in data:
                records.append(data)

        metrics = aggregate_error_metrics(records)
        return {"metrics": metrics}

    except Exception as e:
        logger.exception("Error calculando métricas agregadas")
        return {"metrics": {}}









































