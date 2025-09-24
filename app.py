import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from features.build_features import build_all_features

# Modelo entrenado (ajusta el nombre si usas el optimizado)
from urllib.request import urlopen
from io import BytesIO

MODEL_PATH = ROOT / "models" / "rf_pipeline.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# URL RAW del modelo en GitHub (ajústala si tu repo/archivo es distinto)
MODEL_URL = "https://raw.githubusercontent.com/ALM1808/Calculo_Grasa_corporal/main/models/rf_pipeline.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.warning("Modelo no encontrado en el contenedor. Descargando modelo…")
    with urlopen(MODEL_URL) as resp:
        data = resp.read()
    model = joblib.load(BytesIO(data))
    joblib.dump(model, MODEL_PATH)  # cache local
    st.success("Modelo descargado y cargado correctamente.")


st.title("Predicción de Grasa Corporal")

st.markdown("Introduce tus datos y obtén una predicción. Si repites con tu email, verás tu evolución.")

# --- Todos los campos del dataset original ---
email = st.text_input("Email (identificador único)")

age = st.number_input("Edad", min_value=10, max_value=99, value=30)
gender = st.selectbox("Género", ["Male", "Female"])
weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.number_input("Altura (m)", min_value=1.40, max_value=2.20, value=1.70, step=0.01)
max_bpm = st.number_input("Max_BPM", min_value=80, max_value=220, value=160)
avg_bpm = st.number_input("Avg_BPM", min_value=40, max_value=200, value=130)
resting_bpm = st.number_input("Resting_BPM", min_value=30, max_value=100, value=60)
session_duration = st.number_input("Session_Duration (hours)", min_value=0.1, max_value=5.0, value=1.0, step=0.01)
calories_burned = st.number_input("Calories_Burned", min_value=100.0, max_value=5000.0, value=800.0)
workout_type = st.selectbox("Workout_Type", ["Yoga", "HIIT", "Cardio", "Strength", "Mixed"])
water_intake = st.number_input("Water_Intake (liters)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
workout_freq = st.number_input("Workout_Frequency (days/week)", min_value=1, max_value=7, value=3)
experience_level = st.selectbox("Experience_Level", [1, 2, 3, 4, 5])

# --- Valor real opcional ---
real_fat_percentage = st.number_input("Tu grasa corporal REAL (opcional)", min_value=0.0, max_value=70.0, value=0.0, step=0.1)

# --- Botón principal ---
if st.button("Predecir y guardar"):
    # A. Construir DataFrame igual que el original
    input_dict = {
        "Age": age,
        "Gender": gender,
        "Weight (kg)": weight,
        "Height (m)": height,
        "Max_BPM": max_bpm,
        "Avg_BPM": avg_bpm,
        "Resting_BPM": resting_bpm,
        "Session_Duration (hours)": session_duration,
        "Calories_Burned": calories_burned,
        "Workout_Type": workout_type,
        "Water_Intake (liters)": water_intake,
        "Workout_Frequency (days/week)": workout_freq,
        "Experience_Level": experience_level
    }
    df_input = pd.DataFrame([input_dict])

    # B. Aplicar feature engineering
    df_input = build_all_features(df_input)

    # C. Predicción
    prediction = model.predict(df_input)[0]
    st.success(f"Tu grasa corporal estimada es: **{prediction:.2f}%**")

    # D. Guardar en Feature Store local
    FEATURES_CSV = ROOT / "feature_store" / "user_fat_percentage" / "features.csv"
    now = datetime.now().isoformat(timespec="seconds")
    # Simple gestión de user_id correlativo
    if FEATURES_CSV.exists():
        store_df = pd.read_csv(FEATURES_CSV)
        if email in store_df["email"].values:
            user_id = store_df.loc[store_df["email"] == email, "user_id"].iloc[0]
        else:
            user_id = store_df["user_id"].max() + 1 if "user_id" in store_df else 1
    else:
        user_id = 1

    # Registro para guardar
    record = df_input.copy()
    record["email"] = email
    record["timestamp"] = now
    record["user_id"] = user_id
    record["predicted_fat_percentage"] = prediction
    record["real_fat_percentage"] = real_fat_percentage if real_fat_percentage > 0 else np.nan

    # Actualizar CSV (reemplaza si email+timestamp coinciden, si no, añade)
    if FEATURES_CSV.exists():
        store_df = pd.read_csv(FEATURES_CSV)
        mask = (store_df["email"] == email) & (store_df["timestamp"] == now)
        store_df = store_df[~mask]
        store_df = pd.concat([store_df, record], ignore_index=True)
    else:
        store_df = record

    store_df.to_csv(FEATURES_CSV, index=False)
    st.success("¡Tus datos han sido guardados!")

    # E. Mostrar evolución (opcional)
    if email and FEATURES_CSV.exists():
        df_hist = pd.read_csv(FEATURES_CSV)
        df_user = df_hist[df_hist["email"] == email].sort_values("timestamp")
        if not df_user.empty:
            st.subheader("Evolución histórica de tus predicciones")
            st.line_chart(df_user.set_index("timestamp")[["predicted_fat_percentage"]])