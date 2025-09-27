import sys
from pathlib import Path

# --- Rutas base ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from features.build_features import build_all_features

# --- Modelo: carga local y descarga en caso de no existir ---
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from io import BytesIO

MODEL_PATH = ROOT / "models" / "rf_pipeline.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_URL = "https://raw.githubusercontent.com/ALM1808/Calculo_Grasa_corporal/main/models/rf_pipeline.pkl"

def load_model_safely():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    try:
        st.warning("Modelo no encontrado en el contenedor. Intentando descargar…")
        with urlopen(MODEL_URL, timeout=30) as resp:
            data = resp.read()
        model_obj = joblib.load(BytesIO(data))
        joblib.dump(model_obj, MODEL_PATH)
        st.success("Modelo descargado y cargado correctamente.")
        return model_obj
    except (HTTPError, URLError) as e:
        st.error(f"No pude descargar el modelo ({e}).")
        st.stop()

model = load_model_safely()

# --- Feature store local: asegura carpetas ---
FEATURES_CSV = ROOT / "feature_store" / "user_fat_percentage" / "features.csv"
FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)

# --- Título y descripción ---
st.title("Predicción de grasa corporal")
st.markdown(
    "Introduce tus datos y obtén una predicción. "
    "Si repites con tu **email**, podrás ver tu evolución a lo largo del tiempo."
)

# =========================
# =        FORM          =
# =========================
email = st.text_input("Email (identificador único)")

edad = st.number_input("Edad", min_value=10, max_value=99, value=30, help="Años cumplidos")

# Mostrar en español pero mapear al modelo
genero_es = st.selectbox("Género", ["Hombre", "Mujer"])
genero_map = {"Hombre": "Male", "Mujer": "Female"}
genero = genero_map[genero_es]

peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
altura = st.number_input("Altura (m)", min_value=1.40, max_value=2.20, value=1.70, step=0.01)

max_bpm = st.number_input("Frecuencia cardiaca máxima (lat/min)", min_value=80, max_value=220, value=160)
avg_bpm = st.number_input("Frecuencia cardiaca media (lat/min)", min_value=40, max_value=200, value=130)
resting_bpm = st.number_input("Frecuencia en reposo (lat/min)", min_value=30, max_value=100, value=60)

duracion_sesion = st.number_input("Duración de la sesión (horas)", min_value=0.1, max_value=5.0, value=1.0, step=0.01)
calorias = st.number_input("Calorías quemadas (kcal)", min_value=100.0, max_value=5000.0, value=800.0)

# Desplegable en español pero valores que el modelo espera en inglés
tipo_entrenamiento_es = st.selectbox(
    "Tipo de entrenamiento",
    ["Yoga", "HIIT", "Cardio", "Fuerza", "Mixto"]
)
tipo_map = {
    "Yoga": "Yoga",
    "HIIT": "HIIT",
    "Cardio": "Cardio",
    "Fuerza": "Strength",
    "Mixto": "Mixed",
}
tipo_entrenamiento = tipo_map[tipo_entrenamiento_es]

agua = st.number_input("Ingesta de agua (litros)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
frecuencia = st.number_input("Frecuencia de entrenamiento (días/semana)", min_value=1, max_value=7, value=3)
nivel_exp = st.selectbox("Nivel de experiencia (1=Principiante • 5=Avanzado)", [1, 2, 3, 4, 5])

grasa_real = st.number_input("Tu % de grasa corporal REAL (opcional)", min_value=0.0, max_value=70.0, value=0.0, step=0.1)

# =========================
# =  BOTÓN PRINCIPAL     =
# =========================
if st.button("Predecir y guardar"):
    # Validación mínima
    if not email:
        st.error("Por favor, introduce un email para poder guardar y consultar tu evolución.")
        st.stop()

    # A) Construir DataFrame con los nombres que espera el modelo
    input_dict = {
        "Age": edad,
        "Gender": genero,                       # Male/Female (mapeado)
        "Weight (kg)": peso,
        "Height (m)": altura,
        "Max_BPM": max_bpm,
        "Avg_BPM": avg_bpm,
        "Resting_BPM": resting_bpm,
        "Session_Duration (hours)": duracion_sesion,
        "Calories_Burned": calorias,
        "Workout_Type": tipo_entrenamiento,     # Yoga/HIIT/Cardio/Strength/Mixed
        "Water_Intake (liters)": agua,
        "Workout_Frequency (days/week)": frecuencia,
        "Experience_Level": nivel_exp,
    }
    df_input = pd.DataFrame([input_dict])

    # B) Ingeniería de características
    df_input = build_all_features(df_input)

    # C) Predicción
    pred = model.predict(df_input)[0]
    st.success(f"Tu grasa corporal estimada es: **{pred:.2f}%**")

    # D) Guardado en el feature store
    ahora_iso = datetime.now().isoformat(timespec="seconds")

    # Calcular/recuperar user_id
    if FEATURES_CSV.exists():
        store_df = pd.read_csv(FEATURES_CSV)
        if (not store_df.empty) and ("user_id" in store_df.columns):
            if "email" in store_df.columns and email in store_df["email"].astype(str).values:
                # Si ya existe el email, reutilizar su user_id
                user_id = store_df.loc[store_df["email"] == email, "user_id"].iloc[0]
            else:
                # Si no existe el email, nuevo id incremental
                max_id = pd.to_numeric(store_df["user_id"], errors="coerce").max()
                user_id = (0 if pd.isna(max_id) else int(max_id)) + 1
        else:
            user_id = 1
    else:
        user_id = 1

    # Registro a guardar
    record = df_input.copy()
    record["email"] = email
    record["timestamp"] = ahora_iso
    record["user_id"] = user_id
    record["predicted_fat_percentage"] = pred
    record["real_fat_percentage"] = grasa_real if grasa_real > 0 else np.nan

    # Concatenar/crear CSV
    if FEATURES_CSV.exists():
        store_df = pd.read_csv(FEATURES_CSV)
        store_df = pd.concat([store_df, record], ignore_index=True)
    else:
        store_df = record

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    store_df.to_csv(FEATURES_CSV, index=False)

    st.caption(f"Archivo CSV guardado en: {FEATURES_CSV}")
    st.caption(f"Carpeta creada/existente: {FEATURES_CSV.parent.exists()}")
    st.success("¡Tus datos han sido guardados correctamente!")

    # E) Evolución del usuario (gráfica)
    if email and FEATURES_CSV.exists():
        df_hist = pd.read_csv(FEATURES_CSV)
        df_user = df_hist[df_hist["email"] == email].copy()
        if not df_user.empty:
            # Ordenar por fecha y asegurar tipo datetime
            df_user["timestamp"] = pd.to_datetime(df_user["timestamp"], errors="coerce")
            df_user = df_user.sort_values("timestamp")

            st.subheader("Evolución histórica de tus predicciones")
            graf = df_user.set_index("timestamp")[["predicted_fat_percentage"]]
            graf.columns = ["Predicción de % grasa"]
            st.line_chart(graf)
        else:
            st.info("Aún no hay histórico para este email.")
