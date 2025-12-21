# backend/main.py
import sys
if "nest_asyncio" in sys.modules:
    del sys.modules["nest_asyncio"]

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Literal, Optional
import os
import hashlib
from datetime import datetime
from pathlib import Path
from functools import lru_cache
from fastapi import Query
from fastapi.encoders import jsonable_encoder
import pandas as pd
import numpy as np
import joblib

from google.cloud import firestore
from datetime import timezone

from google.cloud.firestore_v1 import FieldFilter

# ======================================================
# CONFIGURACIÓN LOGGING
# ======================================================
import logging
import os

logger = logging.getLogger("grasa-backend")
logger.setLevel(logging.INFO)

# ¿Estamos en Cloud Run?
if os.getenv("K_SERVICE"):
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
    # Local / Docker local
    logging.basicConfig(level=logging.INFO)
    logger.info("Logging local activado (sin Cloud Logging)")


# ======================================================
# FASTAPI
# ======================================================
app = FastAPI(title="Grasa Corporal API", version="2.5")

# ======================================================
# Pydantic Schemas
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

class FeedbackInput(BaseModel):
    email: str
    real_fat_percentage: float
    predicted_fat_percentage: Optional[float] = None

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

def current_timestamp():
    return pd.Timestamp.utcnow()

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

LOG_PATH = Path(os.getenv("LOG_PATH", "/tmp/data_logs"))
LOG_PATH.mkdir(parents=True, exist_ok=True)

def append_to_csv(filename: str, row: dict):
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


def save_prediction_to_firestore(email: str, data: dict, pred: float):
    fs = get_firestore()
    if fs is None:
        logger.info("Firestore no inicializado: se omite save_prediction_to_firestore")
        return

    try:
        ts = datetime.now(timezone.utc).replace(microsecond=0)
        doc_id = f"{email.replace('@','_')}__{ts.isoformat()}"

        fs.collection("predictions").document(doc_id).set({
            "email": email,
            "timestamp": ts,  # Firestore lo guarda como timestamp nativo
            "predicted_fat_percentage": float(pred),
            **data,
        })
    except Exception as e:
        logger.warning(f"Fallo guardando predicción en Firestore: {e}")


def save_feedback_to_firestore(email: str, real: float, pred: Optional[float]):
    fs = get_firestore()
    if fs is None:
        logger.info("Firestore no inicializado: se omite save_feedback_to_firestore")
        return

    try:
        fs.collection("feedback").add({
            "email": email,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0),
            "real_fat_percentage": float(real),
            "predicted_fat_percentage": float(pred) if pred is not None else None,
        })
    except Exception as e:
        logger.warning(f"Fallo guardando feedback en Firestore: {e}")


def load_history_from_firestore(email: str):
    fs = get_firestore()
    if fs is None:
        logger.info("Firestore no inicializado: histórico vacío desde Firestore")
        return []

    try:
        docs = fs.collection("predictions").where(
            filter=("email", "==", email)
        ).stream()

        records = []
        for d in docs:
            r = d.to_dict()
            ts = r.get("timestamp")
            # Si Firestore devuelve datetime, lo pasamos a ISO; si no, lo dejamos tal cual
            if hasattr(ts, "isoformat"):
                r["timestamp"] = ts.isoformat()
            records.append(r)

        return sorted(records, key=lambda x: x.get("timestamp") or "")
    except Exception as e:
        logger.warning(f"Fallo cargando histórico desde Firestore: {e}")
        return []


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
    data = input_data.dict()
    email = data["email"].lower().strip()

    df = pd.DataFrame([data])
    df["experience_level"] = df["experience_level"].astype(int)

    df_model = df.rename(columns=SNAKE_TO_MODEL)
    df_model["BMI"] = df["weight_kg"] / df["height_m"]**2
    df_model["Log_Age"] = np.log1p(df["age"])
    df_model = df_model[EXPECTED_MODEL_COLS]

    pred = round(float(get_model().predict(df_model)[0]), 2)

    append_to_csv("predictions.csv", {
        "timestamp": current_timestamp(),
        "email": email,
        "predicted_fat_percentage": pred,
        **data,
    })

    background_tasks.add_task(save_prediction_to_firestore, email, data, pred)

    return {"predicted_fat_percentage": pred}

@app.post("/feedback")
def feedback(input_data: FeedbackInput, background_tasks: BackgroundTasks):
    email = input_data.email.lower().strip()

    append_to_csv("feedback.csv", {
        "timestamp": current_timestamp(),
        "email": email,
        "real_fat_percentage": input_data.real_fat_percentage,
        "predicted_fat_percentage": input_data.predicted_fat_percentage,
    })

    background_tasks.add_task(
        save_feedback_to_firestore,
        email,
        input_data.real_fat_percentage,
        input_data.predicted_fat_percentage,
    )

    return {"status": "ok"}

from google.cloud.firestore_v1 import FieldFilter

from google.cloud.firestore_v1 import FieldFilter
from fastapi import Query

@app.get("/history")
def history(email: str = Query(..., min_length=3)):
    email = email.strip().lower()

    fs = get_firestore()
    if fs is None:
        logger.warning("Firestore no disponible")
        return {"records": []}

    records = []

    try:
        # ---------- PREDICTIONS ----------
        pred_query = (
            fs.collection("predictions")
            .where(filter=FieldFilter("email", "==", email))
        )

        for doc in pred_query.stream():
            data = doc.to_dict() or {}

            # timestamp: usar campo si existe; si no, fallback a create_time del documento
            ts = data.get("timestamp")
            if ts is None:
                ts = getattr(doc, "create_time", None)

            if ts is not None and hasattr(ts, "isoformat"):
                data["timestamp"] = ts.isoformat()
            else:
                data["timestamp"] = None

            # asegurar tipos
            if "predicted_fat_percentage" in data and data["predicted_fat_percentage"] is not None:
                try:
                    data["predicted_fat_percentage"] = float(data["predicted_fat_percentage"])
                except Exception:
                    pass

            # en predictions normalmente no existe real_fat_percentage
            if "real_fat_percentage" in data and data["real_fat_percentage"] is not None:
                try:
                    data["real_fat_percentage"] = float(data["real_fat_percentage"])
                except Exception:
                    pass
            else:
                data["real_fat_percentage"] = None

            data["source"] = "prediction"
            records.append(data)

        # ---------- FEEDBACK ----------
        fb_query = (
            fs.collection("feedback")
            .where(filter=FieldFilter("email", "==", email))
        )

        for doc in fb_query.stream():
            data = doc.to_dict() or {}

            ts = data.get("timestamp")
            if ts is None:
                ts = getattr(doc, "create_time", None)

            if ts is not None and hasattr(ts, "isoformat"):
                data["timestamp"] = ts.isoformat()
            else:
                data["timestamp"] = None

            if "predicted_fat_percentage" in data and data["predicted_fat_percentage"] is not None:
                try:
                    data["predicted_fat_percentage"] = float(data["predicted_fat_percentage"])
                except Exception:
                    pass

            if "real_fat_percentage" in data and data["real_fat_percentage"] is not None:
                try:
                    data["real_fat_percentage"] = float(data["real_fat_percentage"])
                except Exception:
                    pass

            data["source"] = "feedback"
            records.append(data)

        # Orden robusto: primero los que tienen timestamp válido
        def _key(r):
            return (r.get("timestamp") is None, r.get("timestamp") or "")

        records.sort(key=_key)
        return {"records": records}

    except Exception as e:
        logger.exception(f"Error cargando histórico Firestore: {e}")
        return {"records": []}










































