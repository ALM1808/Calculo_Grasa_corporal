# app/api_client_app.py

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Grasa corporal - Cliente API", page_icon="💪", layout="wide")

st.title("💪 Predicción de grasa corporal (cliente de la API)")
st.write("Este front sólo llama al backend FastAPI. El modelo vive allí.")

# --------------------- Estado ---------------------
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_input" not in st.session_state:
    st.session_state.last_input = None

# --------------------- Formulario de predicción ---------------------
with st.form("prediction_form"):
    email = st.text_input("Email", "")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Género", ["Male", "Female"])
        age = st.number_input("Edad", 10, 100, 35)
        weight_kg = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
        height_m = st.number_input("Altura (m)", 1.2, 2.2, 1.70)
    with col2:
        max_bpm = st.number_input("Frecuencia máxima (Max_BPM)", 80, 220, 180)
        avg_bpm = st.number_input("Frecuencia media (Avg_BPM)", 60, 200, 140)
        resting_bpm = st.number_input("Frecuencia reposo (Resting_BPM)", 30, 120, 60)
        session_duration_hours = st.number_input("Duración sesión (horas)", 0.1, 4.0, 1.0)
    with col3:
        calories_burned = st.number_input("Calorías quemadas", 0.0, 5000.0, 400.0)
        water_intake_liters = st.number_input("Litros de agua/día", 0.0, 10.0, 2.0)
        workout_frequency_days_week = st.slider("Entrenos / semana", 1, 7, 3)
        experience_level = st.selectbox("Nivel experiencia", ["1", "2", "3"])
        workout_type = st.selectbox("Tipo entrenamiento", ["Cardio", "Strength", "Mixed"])

    submitted = st.form_submit_button("🧑‍⚕️ Obtener predicción")

if submitted:
    if not email:
        st.error("Introduce un email para poder guardar tu histórico.")
    else:
        payload = {
            "email": email,
            "age": age,
            "gender": gender,
            "weight_kg": weight_kg,
            "height_m": height_m,
            "max_bpm": max_bpm,
            "avg_bpm": avg_bpm,
            "resting_bpm": resting_bpm,
            "session_duration_hours": session_duration_hours,
            "calories_burned": calories_burned,
            "workout_type": workout_type,
            "water_intake_liters": water_intake_liters,
            "workout_frequency_days_week": workout_frequency_days_week,
            "experience_level": experience_level,
        }

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                pred = float(data["predicted_fat_percentage"])
                st.session_state.last_prediction = pred
                st.session_state.last_input = payload
                st.success(f"✅ Predicción: {pred:.2f}% de grasa corporal")
            else:
                st.error(f"❌ Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"❌ Error de conexión con la API: {e}")

# Mostrar última predicción aunque mandemos feedback
if st.session_state.last_prediction is not None:
    st.info(f"📌 Última predicción guardada: {st.session_state.last_prediction:.2f}%")

st.markdown("---")

# --------------------- Sección feedback ---------------------
st.subheader("📬 Enviar feedback (valor REAL)")

real_value = st.number_input("Grasa corporal REAL (%)", 1.0, 80.0, 25.0, step=0.1)
send_fb = st.button("📨 Enviar feedback a la API")

if send_fb:
    if not st.session_state.last_input or not st.session_state.last_prediction:
        st.error("Primero obtén una predicción (con email válido) antes de enviar feedback.")
    else:
        fb_payload = {
            "email": st.session_state.last_input["email"],
            "real_fat_percentage": real_value,
            "predicted_fat_percentage": st.session_state.last_prediction,
        }
        try:
            resp = requests.post(f"{API_URL}/feedback", json=fb_payload, timeout=5)
            if resp.status_code == 200:
                st.success("✅ Feedback enviado (el guardado en Hopsworks se hace en segundo plano).")
            else:
                st.error(f"❌ Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"❌ Error de conexión al enviar feedback: {e}")






