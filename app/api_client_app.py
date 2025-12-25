import os
import pandas as pd
import requests
import streamlit as st
import altair as alt


# ========================================================
# CONFIGURACIÓN INICIAL
# ========================================================

API_URL = st.secrets.get(
    "API_URL",
    os.getenv("API_URL", "http://127.0.0.1:8000")
).rstrip("/")

PREDICT_URL = f"{API_URL}/predict"
FEEDBACK_URL = f"{API_URL}/feedback"
HISTORY_URL = f"{API_URL}/history"

st.set_page_config(
    page_title="Grasa corporal – Cliente API",
    page_icon="💪",
    layout="wide"
)

st.title("💪 Predicción de grasa corporal — Cliente API")
st.caption(f"Conectado al backend: **{API_URL}**")


# ========================================================
# ESTADO DE SESIÓN
# ========================================================

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_input" not in st.session_state:
    st.session_state.last_input = None


# ========================================================
# TABS
# ========================================================

tab1, tab2, tab3 = st.tabs(["🧑‍⚕️ Predicción", "📬 Feedback", "📊 Histórico"])


# ========================================================
# TAB 1 — PREDICCIÓN
# ========================================================

with tab1:
    st.header("🧑‍⚕️ Obtener predicción")

    with st.form("prediction_form"):
        email = st.text_input("Email (necesario para histórico)", "").lower().strip()

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Género", ["Male", "Female"])
            age = st.number_input("Edad", min_value=10, max_value=100, value=35)
            weight_kg = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
            height_m = st.number_input("Altura (m)", min_value=1.2, max_value=2.2, value=1.70)

        with c2:
            max_bpm = st.number_input("Frecuencia máxima", min_value=80, max_value=220, value=180)
            avg_bpm = st.number_input("Frecuencia media", min_value=60, max_value=200, value=140)
            resting_bpm = st.number_input("Frecuencia reposo", min_value=30, max_value=120, value=60)
            session_duration_hours = st.number_input("Duración sesión (h)", min_value=0.1, max_value=4.0, value=1.0)

        with c3:
            calories_burned = st.number_input("Calorías quemadas", min_value=0.0, max_value=5000.0, value=400.0)
            water_intake_liters = st.number_input("Agua (L/día)", min_value=0.0, max_value=10.0, value=2.0)
            workout_frequency_days_week = st.slider("Entrenos/semana", 1, 7, 3)
            experience_level = st.selectbox("Nivel experiencia", ["1", "2", "3"])
            workout_type = st.selectbox("Tipo entrenamiento", ["Cardio", "Strength", "Mixed"])

        submitted = st.form_submit_button("🧑‍⚕️ Obtener predicción")

    if submitted:
        if not email:
            st.error("Introduce un email para guardar el histórico.")
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
                with st.spinner("Llamando a la API…"):
                    resp = requests.post(PREDICT_URL, json=payload, timeout=20)

                if resp.status_code == 200:
                    pred = float(resp.json()["predicted_fat_percentage"])
                    st.session_state.last_prediction = pred
                    st.session_state.last_input = payload
                    st.success(f"Predicción obtenida: **{pred:.2f}%**")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")

            except Exception as e:
                st.error(f"Error conectando con la API: {e}")

    if st.session_state.last_prediction is not None:
        st.info(f"📌 Última predicción: **{st.session_state.last_prediction:.2f}%**")


# ========================================================
# TAB 2 — FEEDBACK
# ========================================================

with tab2:
    st.header("📬 Enviar feedback (valor REAL)")

    real_value = st.number_input(
        "Valor REAL de grasa corporal (%)",
        min_value=1.0, max_value=80.0, step=0.1
    )

    if st.button("📨 Enviar feedback"):
        if st.session_state.last_input is None:
            st.error("Primero obtén una predicción.")
        else:
            fb_payload = {
                "email": st.session_state.last_input["email"],
                "real_fat_percentage": float(real_value),
                "predicted_fat_percentage": float(st.session_state.last_prediction),
            }

            try:
                with st.spinner("Enviando feedback…"):
                    resp = requests.post(FEEDBACK_URL, json=fb_payload, timeout=15)

                if resp.status_code == 200:
                    st.success("Feedback enviado correctamente ✔")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")

            except Exception as e:
                st.error(f"Error enviando feedback: {e}")


# ========================================================
# TAB 3 — HISTÓRICO
# ========================================================

def load_history(email: str):
    try:
        r = requests.get(HISTORY_URL, params={"email": email}, timeout=15)
        if r.status_code != 200:
            return []
        return r.json().get("records", [])
    except Exception:
        return []


with tab3:
    st.header("📊 Histórico de predicciones")

    email_filter = st.text_input("Introduce tu email:")

    if email_filter:
        records = load_history(email_filter.strip().lower())

        if not records:
            st.warning("No hay registros para este email.")
        else:
            df = pd.DataFrame(records)

            # --- Timestamp limpio ---
            df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=True)

            # 🔥 SOLO registros con fecha válida
            df = df[df["timestamp"].notna()].copy()

            if df.empty:
                st.warning("Aún no hay registros con fecha válida.")
            else:
                df = df.sort_values("timestamp").reset_index(drop=True)

                # --- TABLA LIMPIA ---
                table_cols = [
                    "timestamp",
                    "predicted_fat_percentage",
                    "real_fat_percentage",
                    "weight_kg",
                    "workout_type",
                    "session_duration_hours",
                ]
                table_cols = [c for c in table_cols if c in df.columns]

                df_table = df[table_cols].copy()
                df_table["timestamp"] = df_table["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

                st.subheader("📄 Histórico (resumen)")
                st.dataframe(df_table, use_container_width=True)

                # --- GRÁFICO ---
                plot_df = []

                if "predicted_fat_percentage" in df.columns:
                    plot_df.append(
                        df[["timestamp", "predicted_fat_percentage"]]
                        .rename(columns={"predicted_fat_percentage": "value"})
                        .assign(serie="Predicción")
                    )

                if "real_fat_percentage" in df.columns and df["real_fat_percentage"].notna().any():
                    plot_df.append(
                        df[["timestamp", "real_fat_percentage"]]
                        .rename(columns={"real_fat_percentage": "value"})
                        .assign(serie="Real")
                    )

                if plot_df:
                    plot_df = pd.concat(plot_df)

                    chart = (
                        alt.Chart(plot_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("timestamp:T", title="Fecha"),
                            y=alt.Y("value:Q", title="% grasa corporal"),
                            color="serie:N",
                            tooltip=["timestamp:T", "serie:N", "value:Q"],
                        )
                    )

                    st.subheader("📈 Evolución")
                    st.altair_chart(chart, use_container_width=True)






