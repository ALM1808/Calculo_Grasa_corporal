import requests
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

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

# OJO: estas rutas son relativas a donde ejecutes el script
# Si ejecutas:  python backend/train_compare.py
# entonces "models/..." debe existir en la raíz del repo.
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
# RECONSTRUIR INPUT EXACTO COMO /predict (para V1)
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

# y = target (real)
y = pd.to_numeric(df["real_fat_percentage"], errors="coerce")
mask = y.notna()
df = df.loc[mask].copy()
y = y.loc[mask].astype(float)
# Alinear y con X tras eliminar filas incompletas
df = df.dropna(subset=["age", "gender", "weight_kg", "height_m"]).copy()
y = y.loc[df.index]

# X = features
# Filtrar filas incompletas (muy importante)
required_cols = ["age", "gender", "weight_kg", "height_m"]
df = df.dropna(subset=required_cols).copy()

# Reconstruir X otra vez tras el filtro
X = df.rename(columns=SNAKE_TO_MODEL).copy()

# Tipos
X["Age"] = pd.to_numeric(X["Age"], errors="coerce").astype(int)
X["Experience_Level"] = pd.to_numeric(X["Experience_Level"], errors="coerce").astype(int)
X["Gender"] = X["Gender"].astype(str)
X["Workout_Type"] = X["Workout_Type"].astype(str)

# Features derivadas IGUAL que en /predict
X["BMI"] = X["Weight (kg)"] / (X["Height (m)"] ** 2)
X["Log_Age"] = np.log1p(X["Age"])

# Nos quedamos solo con lo que espera el pipeline
X = X[EXPECTED_MODEL_COLS]

# ======================================================
# LOAD + EVAL MODEL V1
# ======================================================
print("📦 Cargando modelo v1...")
model_v1 = joblib.load(MODEL_V1_PATH)

y_pred_v1 = model_v1.predict(X)

mae_v1 = mean_absolute_error(y, y_pred_v1)
rmse_v1 = np.sqrt(mean_squared_error(y, y_pred_v1))  # ✅ compatible con tu sklearn

print(f"📊 V1 → MAE: {mae_v1:.3f} | RMSE: {rmse_v1:.3f}")

# ======================================================
# TRAIN + EVAL MODEL V2 (misma X, mismas cols)
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
    ],
    remainder="drop",
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
rmse_v2 = np.sqrt(mean_squared_error(y, y_pred_v2))  # ✅ compatible

print(f"📊 V2 → MAE: {mae_v2:.3f} | RMSE: {rmse_v2:.3f}")

# ======================================================
# SAVE MODEL V2
# ======================================================
print(f"📊 V1 → MAE: {mae_v1:.3f} | RMSE: {rmse_v1:.3f}")
print(f"📊 V2 → MAE: {mae_v2:.3f} | RMSE: {rmse_v2:.3f}")

if mae_v2 < mae_v1:
    joblib.dump(model_v2, MODEL_V2_PATH)
    print("🏆 V2 mejora a V1 → modelo guardado")
else:
    print("⛔ V2 NO mejora a V1 → se descarta")



