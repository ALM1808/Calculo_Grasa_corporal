import requests
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timezone

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ======================================================
# CONFIG
# ======================================================
API_URL = "https://grasa-backend-648174663015.europe-west1.run.app"
TRAINING_DATA_URL = f"{API_URL}/training-data"

MODEL_V1_PATH = Path("models/rf_pipeline.pkl")
MODEL_V2_PATH = Path("models/rf_pipeline_v2.pkl")

RANDOM_STATE = 42

# ======================================================
# LOAD DATA
# ======================================================
print("📥 Descargando training data...")
resp = requests.get(TRAINING_DATA_URL, timeout=30)
resp.raise_for_status()

records = resp.json().get("records", [])
df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("❌ No hay datos (records vacío)")

print(f"✅ Registros disponibles: {len(df)}")

# ======================================================
# RECONSTRUIR INPUT EXACTO COMO /predict (V1)
# ======================================================
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

# Target
y = pd.to_numeric(df["real_fat_percentage"], errors="coerce")
mask = y.notna()
df = df.loc[mask].copy()
y = y.loc[mask].astype(float)

# Filtrar filas incompletas
required_cols = ["age", "gender", "weight_kg", "height_m"]
df = df.dropna(subset=required_cols).copy()
y = y.loc[df.index]

# Features
X = df.rename(columns=SNAKE_TO_MODEL).copy()

X["Age"] = pd.to_numeric(X["Age"], errors="coerce").astype(int)
X["Experience_Level"] = pd.to_numeric(
    X["Experience_Level"], errors="coerce"
).astype(int)

X["Gender"] = X["Gender"].astype(str)
X["Workout_Type"] = X["Workout_Type"].astype(str)

# Features derivadas
X["BMI"] = X["Weight (kg)"] / (X["Height (m)"] ** 2)
X["Log_Age"] = np.log1p(X["Age"])

X = X[EXPECTED_MODEL_COLS]

# ======================================================
# LOAD + EVAL MODEL V1
# ======================================================
print("📦 Cargando modelo v1...")
model_v1 = joblib.load(MODEL_V1_PATH)

y_pred_v1 = model_v1.predict(X)

mae_v1 = mean_absolute_error(y, y_pred_v1)
rmse_v1 = np.sqrt(mean_squared_error(y, y_pred_v1))

print(f"📊 V1 → MAE: {mae_v1:.3f} | RMSE: {rmse_v1:.3f}")

# ======================================================
# TRAIN + EVAL MODEL V2
# ======================================================
numeric_features = [
    "Age",
    "Weight (kg)",
    "Height (m)",
    "Max_BPM",
    "Avg_BPM",
    "Resting_BPM",
    "Session_Duration (hours)",
    "Calories_Burned",
    "Water_Intake (liters)",
    "Workout_Frequency (days/week)",
    "Experience_Level",
    "BMI",
    "Log_Age",
]

categorical_features = ["Gender", "Workout_Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model_v2 = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]
)

print("🧠 Entrenando modelo v2...")
model_v2.fit(X, y)

y_pred_v2 = model_v2.predict(X)

mae_v2 = mean_absolute_error(y, y_pred_v2)
rmse_v2 = np.sqrt(mean_squared_error(y, y_pred_v2))

print(f"📊 V2 → MAE: {mae_v2:.3f} | RMSE: {rmse_v2:.3f}")

# ======================================================
# DECISIÓN GLOBAL
# ======================================================
improves = mae_v2 < mae_v1

# ======================================================
# SAVE MODEL V2
# ======================================================
if improves:
    joblib.dump(model_v2, MODEL_V2_PATH)
    print("🏆 V2 mejora a V1 → modelo guardado")
else:
    print("⛔ V2 NO mejora a V1 → se descarta")

# ======================================================
# SAVE METRICS
# ======================================================
metrics = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "n_samples": int(len(y)),
    "v1": {
        "mae": round(mae_v1, 4),
        "rmse": round(rmse_v1, 4),
    },
    "v2": {
        "mae": round(mae_v2, 4),
        "rmse": round(rmse_v2, 4),
    },
    "improves": improves,
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("📊 Métricas guardadas en metrics.json")
print(json.dumps(metrics, indent=2))

# ======================================================
# PROMOTION DECISION
# ======================================================
promotion = {
    "timestamp": metrics["timestamp"],
    "promote": improves,
    "reason": "v2 improves v1" if improves else "v2 does not improve v1",
    "metric": "mae",
    "v1_mae": round(mae_v1, 4),
    "v2_mae": round(mae_v2, 4),
}

with open("promotion.json", "w") as f:
    json.dump(promotion, f, indent=2)

print("🚦 Decisión de promoción guardada en promotion.json")
print(json.dumps(promotion, indent=2))
