import os
import pandas as pd
import numpy as np
from datetime import datetime
import hopsworks

# ================================================================
# CONFIG
# ================================================================

PREDICTIONS_CSV = "backend/data_logs/predictions.csv"
FEEDBACK_CSV = "backend/data_logs/feedback.csv"

PROJECT = os.getenv("HOPSWORKS_PROJECT")
API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOST = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")

# ================================================================
# UTILS
# ================================================================

def generate_user_id(email: str) -> str:
    import hashlib
    return hashlib.sha256(email.lower().encode()).hexdigest()[:16]

def log(msg):
    print(f"[SYNC] {msg}")


# ================================================================
# MAIN SYNC
# ================================================================

def sync_predictions(fs):
    """Sube predicciones al FG user_fat_percentage."""
    if not os.path.exists(PREDICTIONS_CSV):
        log("No existe predictions.csv — nada que sincronizar.")
        return

    df = pd.read_csv(PREDICTIONS_CSV)

    if df.empty:
        log("predictions.csv vacío.")
        return

    # Generar user_id
    df["user_id"] = df["email"].apply(generate_user_id)

    # Convertir timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Casting correcto
    df["experience_level"] = df["experience_level"].astype(int)
    df["workout_frequency_days_week"] = df["workout_frequency_days_week"].astype(int)

    # Campos extra que el FG espera
    df["real_fat_percentage"] = np.nan  # este FG contiene predicciones, no valores reales
    df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
    df["log_age"] = np.log1p(df["age"].astype(float))

    fg = fs.get_or_create_feature_group(
        name="user_fat_percentage",
        version=1,
        primary_key=["user_id", "email"],
        event_time="timestamp",
        description="Registros de predicción de grasa corporal",
        online_enabled=True,
    )

    log(f"Subiendo {len(df)} filas a user_fat_percentage…")
    fg.insert(df, write_options={"wait_for_job": False})
    log("✔ Predicciones sincronizadas correctamente.")


def sync_feedback(fs):
    """Sube feedback al FG user_fat_feedback."""
    if not os.path.exists(FEEDBACK_CSV):
        log("No existe feedback.csv — nada que sincronizar.")
        return

    df = pd.read_csv(FEEDBACK_CSV)

    if df.empty:
        log("feedback.csv vacío.")
        return

    df["user_id"] = df["email"].apply(generate_user_id)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    fg = fs.get_or_create_feature_group(
        name="user_fat_feedback",
        version=1,
        primary_key=["user_id", "email"],
        event_time="timestamp",
        description="Valores reales enviados por los usuarios",
        online_enabled=True,
    )

    log(f"Subiendo {len(df)} filas a user_fat_feedback…")
    fg.insert(df, write_options={"wait_for_job": False})
    log("✔ Feedback sincronizado correctamente.")


def main():
    if not (PROJECT and API_KEY):
        print("❌ Debes definir HOPSWORKS_PROJECT y HOPSWORKS_API_KEY en variables de entorno.")
        return

    log("Conectando a Hopsworks…")
    project = hopsworks.login(
        project=PROJECT,
        api_key_value=API_KEY,
        host=HOST
    )

    fs = project.get_feature_store()

    sync_predictions(fs)
    sync_feedback(fs)

    log("✔ Sincronización COMPLETADA.")


if __name__ == "__main__":
    main()
