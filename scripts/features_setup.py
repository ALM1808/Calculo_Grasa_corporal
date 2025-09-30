# scripts/features_setup.py
import os
import pandas as pd
from pathlib import Path
import hopsworks

# --- Conexión a Hopsworks ---
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT", "GrasaCorporal"),
    host=os.getenv("HOPSWORKS_HOST", "https://c.app.hopsworks.ai"),
)
fs = project.get_feature_store()

# --- Configuración ---
ENTITY_NAME, ENTITY_VER = "user_fat_percentage", 1
pk_cols = ["email", "user_id"]
event_time = "timestamp"
partition_key = "date"

# --- Crear o recuperar Feature Group ---
try:
    fg = fs.get_feature_group(ENTITY_NAME, version=ENTITY_VER)
    print("✅ Feature Group ya existe, usando el existente")
except:
    fg = fs.create_feature_group(
        name=ENTITY_NAME,
        version=ENTITY_VER,
        description="Registros usuario + features + labels de grasa corporal",
        primary_key=pk_cols,
        event_time=event_time,
        online_enabled=True,
        partition_key=partition_key,
    )
    print("📦 Feature Group creado")

# --- Backfill desde CSV local (opcional) ---
ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "feature_store" / "user_fat_percentage" / "features.csv"

if csv_path.exists():
    df_hist = pd.read_csv(csv_path)
    # Normaliza tipos
    df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], errors="coerce")
    if "date" not in df_hist.columns:
        df_hist["date"] = df_hist["timestamp"].dt.date.astype(str)
    # Inserta
    fg.insert(df_hist, write_options={"wait_for_job": True})
    print(f"📤 Datos insertados desde {csv_path}")
else:
    print("⚠️ No encontré el CSV local, omitiendo backfill")

# --- Crear o recuperar Feature View ---
feature_cols = [
    "Age","Gender","Weight (kg)","Height (m)","Max_BPM","Avg_BPM","Resting_BPM",
    "Session_Duration (hours)","Calories_Burned","Workout_Type",
    "Water_Intake (liters)","Workout_Frequency (days/week)","Experience_Level",
]
label_col = "real_fat_percentage"

query = fg.select(features=feature_cols + [
    "predicted_fat_percentage","real_fat_percentage","timestamp","email","user_id","date"
])

try:
    fv = fs.get_feature_view("user_fat_percentage_fv", version=1)
    print("✅ Feature View ya existe")
except:
    fv = fs.create_feature_view(
        name="user_fat_percentage_fv",
        version=1,
        query=query,
        labels=[label_col],
        description="FV para entrenamiento/inferencia",
    )
    print("📦 Feature View creada")
