import sys
from pathlib import Path
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime

# -------------------------
# Configuración de rutas
# -------------------------
APP_ROOT = Path(__file__).resolve().parent
PROJ_ROOT = APP_ROOT.parent
sys.path.append(str(PROJ_ROOT / "src"))

MODEL_PATH = PROJ_ROOT / "models" / "rf_pipeline.pkl"
MODEL_URL = "https://raw.githubusercontent.com/ALM1808/Calculo_Grasa_corporal/main/models/rf_pipeline.pkl"

FEATURES_CSV = PROJ_ROOT / "feature_store" / "user_fat_percentage" / "features.csv"
FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# Configuración Hopsworks
# -------------------------
HOPSWORKS_PROJECT = "GrasaCorporal"
MODEL_NAME = "rf_pipeline_fat_percentage"
PREFER_STAGE = os.getenv("MODEL_STAGE", "prod")

# -------------------------
# Carga del modelo
# -------------------------
def _load_from_github():
    with urlopen(MODEL_URL, timeout=30) as resp:
        model_obj = joblib.load(BytesIO(resp.read()))
    joblib.dump(model_obj, MODEL_PATH)
    return model_obj

def _load_from_local():
    return joblib.load(MODEL_PATH)

def _load_from_hopsworks():
    import hopsworks
    project = hopsworks.login(project=HOPSWORKS_PROJECT)
    mr = project.get_model_registry()
    models = mr.get_models(name=MODEL_NAME)
    if not models:
        raise RuntimeError(f"Modelo '{MODEL_NAME}' no encontrado en Hopsworks.")

    def has_stage(m, stage):
        try:
            return m.get_tags().get("stage") == stage
        except Exception:
            return False

    candidates = [m for m in models if has_stage(m, PREFER_STAGE)]
    model = (candidates or sorted(models, key=lambda m: m.version))[-1]
    local_dir = model.download()
    model_file = os.path.join(local_dir, "model.pkl")

    if not os.path.exists(model_file):
        import glob
        pkls = glob.glob(os.path.join(local_dir, "*.pkl"))
        if not pkls:
            raise RuntimeError("No se encontró archivo .pkl en los artefactos.")
        model_file = pkls[0]

    return joblib.load(model_file)

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        st.info("🔁 Intentando cargar modelo desde Hopsworks...")
        model_obj = _load_from_hopsworks()
        st.success("✅ Modelo cargado desde Hopsworks")
        joblib.dump(model_obj, MODEL_PATH)
        return model_obj
    except Exception as e:
        st.warning(f"⚠️ Error con Hopsworks: {e}")

    if MODEL_PATH.exists():
        st.info("📂 Cargando modelo local desde disco...")
        return _load_from_local()

    try:
        st.info("🌐 Descargando modelo desde GitHub...")
        model_obj = _load_from_github()
        st.success("✅ Modelo descargado desde GitHub")
        return model_obj
    except Exception as e:
        st.error(f"❌ No se pudo cargar el modelo: {e}")
        st.stop()

model = load_model()

# -------------------------
# Interfaz de entrada de datos
# -------------------------
st.title("Predicción de Porcentaje de Grasa Corporal")
st.write("Introduce tus datos para obtener una predicción:")

email = st.text_input("Correo electrónico", "")
if email:
    user_input = {
        "age": st.number_input("Edad", min_value=10, max_value=100, value=30),
        "gender": st.selectbox("Género", ["Male", "Female"]),
        "weight_kg": st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0),
        "height_m": st.number_input("Altura (m)", min_value=1.2, max_value=2.2, value=1.70),
        "max_bpm": st.number_input("Frecuencia cardíaca máxima", value=180),
        "avg_bpm": st.number_input("Frecuencia cardíaca media", value=140),
        "resting_bpm": st.number_input("Frecuencia cardíaca en reposo", value=60),
        "session_duration_hours": st.number_input("Duración sesión (horas)", value=1.0),
        "Calories_Burned": st.number_input("Calorías quemadas", value=400),
        "Water_Intake (liters)": st.number_input("Litros de agua diarios", value=2.0),
        "Workout_Frequency (days/week)": st.number_input("Frecuencia de entrenamiento semanal", min_value=1, max_value=7, value=3),
        "Experience_Level": st.selectbox("Nivel de experiencia", [1, 2, 3]),
        "workout_type": st.selectbox("Tipo de entrenamiento", ["Cardio", "Strength", "Mixed"]),
    }

    user_input["BMI"] = user_input["weight_kg"] / (user_input["height_m"] ** 2)
    user_input["Log_Age"] = np.log1p(user_input["age"])

    input_df = pd.DataFrame([user_input])
    predicted_fat = model.predict(input_df)[0]

    st.subheader("🔮 Resultado de la Predicción")
    st.write(f"**Porcentaje estimado de grasa corporal:** {predicted_fat:.2f}%")

    # -------------------------
    # Guardar en Feature Store local
    # -------------------------
    today_str = datetime.now().strftime("%Y-%m-%d")
    registro = {
        "email": email,
        "timestamp": today_str,
        "predicted_fat_percentage": predicted_fat,
        **user_input,
    }

    registro_df = pd.DataFrame([registro])

    if FEATURES_CSV.exists():
        existing_df = pd.read_csv(FEATURES_CSV)
        existing_df = existing_df[~((existing_df["email"] == email) & (existing_df["timestamp"] == today_str))]
        updated_df = pd.concat([existing_df, registro_df], ignore_index=True)
    else:
        updated_df = registro_df

    updated_df.to_csv(FEATURES_CSV, index=False)
    st.success("📦 Datos guardados en Feature Store local (features.csv).")

    # -------------------------
    # Guardar en Hopsworks (si estás en Colab)
    # -------------------------
    try:
        import hopsworks
        project = hopsworks.login(project=HOPSWORKS_PROJECT)
        fs = project.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name="user_fat_percentage",
            version=1,
            primary_key=["email", "timestamp"],
            description="Registros manuales desde Streamlit",
            event_time="timestamp",
            online_enabled=True,
        )
        fg.insert(registro_df)
        st.success("✅ Datos enviados a Hopsworks Feature Store")
    except Exception as e:
        st.warning(f"No se pudo subir a Hopsworks: {e}")

    # -------------------------
    # Mostrar historial
    # -------------------------
    if FEATURES_CSV.exists():
        df_hist = pd.read_csv(FEATURES_CSV)
        user_history = df_hist[df_hist["email"] == email].copy()

        if not user_history.empty:
            user_history["timestamp"] = pd.to_datetime(user_history["timestamp"])
            user_history = user_history.sort_values("timestamp", ascending=False)

            st.markdown("### 📈 Historial de predicciones")
            st.dataframe(
                user_history[["timestamp", "predicted_fat_percentage", "weight_kg", "height_m", "BMI", "Log_Age"]].round(2),
                use_container_width=True,
            )

            st.markdown("### 📉 Evolución del % de grasa corporal")
            chart_data = user_history[["timestamp", "predicted_fat_percentage", "real_fat_percentage"]].copy()
            chart_data = chart_data.sort_values("timestamp")

            if chart_data["real_fat_percentage"].notna().any():
                st.line_chart(chart_data.set_index("timestamp")[["predicted_fat_percentage", "real_fat_percentage"]])
            else:
                st.line_chart(chart_data.set_index("timestamp")[["predicted_fat_percentage"]])
        else:
            st.info("ℹ️ No hay registros anteriores.")

