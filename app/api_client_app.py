import os
import pandas as pd
import requests
import streamlit as st
import altair as alt


# ========================================================
# CONFIGURACIÓN INICIAL
# ========================================================

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://127.0.0.1:8000")).rstrip("/")

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
            max_bpm = st.number_input("Frecuencia máxima (Max_BPM)", min_value=80, max_value=220, value=180)
            avg_bpm = st.number_input("Frecuencia media (Avg_BPM)", min_value=60, max_value=200, value=140)
            resting_bpm = st.number_input("Frecuencia reposo (Resting_BPM)", min_value=30, max_value=120, value=60)
            session_duration_hours = st.number_input("Duración sesión (horas)", min_value=0.1, max_value=4.0, value=1.0)

        with c3:
            calories_burned = st.number_input("Calorías quemadas", min_value=0.0, max_value=5000.0, value=400.0)
            water_intake_liters = st.number_input("Litros de agua/día", min_value=0.0, max_value=10.0, value=2.0)
            workout_frequency_days_week = st.slider("Entrenos por semana", 1, 7, 3)
            experience_level = st.selectbox("Nivel de experiencia", ["1", "2", "3"])
            workout_type = st.selectbox("Tipo de entrenamiento", ["Cardio", "Strength", "Mixed"])

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
                with st.spinner("Llamando a la API…"):
                    resp = requests.post(PREDICT_URL, json=payload, timeout=20)

                if resp.status_code == 200:
                    pred = float(resp.json()["predicted_fat_percentage"])
                    st.session_state.last_prediction = pred
                    st.session_state.last_input = payload

                    st.success(f"Predicción obtenida: **{pred:.2f}%** de grasa corporal")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")

            except Exception as e:
                st.error(f"❌ Error al conectar con la API: {e}")

    if st.session_state.last_prediction is not None:
        st.info(f"📌 Última predicción guardada: **{st.session_state.last_prediction:.2f}%**")


# ========================================================
# TAB 2 — FEEDBACK
# ========================================================
with tab2:
    st.header("📬 Enviar feedback (valor REAL)")

    real_value = st.number_input(
        "Introduce el valor REAL de grasa corporal (%)",
        min_value=1.0, max_value=80.0, step=0.1,
    )

    send_fb = st.button("📨 Enviar feedback a la API")

    if send_fb:
        if st.session_state.last_input is None:
            st.error("Primero debes obtener una predicción para enviar feedback.")
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
                st.error(f"❌ Error enviando feedback: {e}")


# ========================================================
# TAB 3 — HISTÓRICO DESDE FIRESTORE (vía backend)
# ========================================================

def load_history(email: str):
    try:
        r = requests.get(HISTORY_URL, params={"email": email}, timeout=15)

        if r.status_code != 200:
            st.error(f"Error consultando histórico ({r.status_code}): {r.text}")
            return []

        return r.json().get("records", [])

    except Exception as e:
        st.error(f"Error llamando al backend: {e}")
        return []


with tab3:
    st.header("📊 Histórico de predicciones")

    email_filter = st.text_input("Introduce tu email para ver el histórico:")

    if email_filter:
        records = load_history(email_filter.strip().lower())

        if not records:
            st.warning("No hay registros para este email.")
        else:
            df = pd.DataFrame(records)

            # --- Timestamp robusto (evita errores si hay None / formatos mezclados) ---
            if "timestamp" in df.columns:
                # Parse ISO / Timestamp ya convertido a string por el backend
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            else:
                df["timestamp"] = pd.NaT

            # Ordenar: los NaT al final (para no “romper” el gráfico)
            df = df.sort_values("timestamp", na_position="last").reset_index(drop=True)

            # --- Tabla “limpia”: no mostrar datos obvios/sensibles ---
            # (email/age/gender fuera; y dejamos columnas útiles)
            preferred_cols = [
                "timestamp",
                "predicted_fat_percentage",
                "real_fat_percentage",
                "weight_kg",
                "session_duration_hours",
                "workout_type",
                "workout_frequency_days_week",
                "avg_bpm",
                "resting_bpm",
                "max_bpm",
                "calories_burned",
                "water_intake_liters",
                "experience_level",
                "height_m",
            ]
            cols_present = [c for c in preferred_cols if c in df.columns]

            df_table = df[cols_present].copy()

            # Opcional: timestamps en formato legible (sigue siendo datetime internamente)
            if "timestamp" in df_table.columns:
                df_table["timestamp"] = df_table["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

            st.subheader("📄 Datos históricos (resumen)")
            st.dataframe(df_table, use_container_width=True)

            # --- Gráfico: predicted y real (si existe) ---
            df_plot = df.copy()
            df_plot = df_plot[df_plot["timestamp"].notna()].copy()

            # Asegurar numéricos
            if "predicted_fat_percentage" in df_plot.columns:
                df_plot["predicted_fat_percentage"] = pd.to_numeric(
                    df_plot["predicted_fat_percentage"], errors="coerce"
                )
            if "real_fat_percentage" in df_plot.columns:
                df_plot["real_fat_percentage"] = pd.to_numeric(
                    df_plot["real_fat_percentage"], errors="coerce"
                )

            metrics = []
            if "predicted_fat_percentage" in df_plot.columns:
                metrics.append(("Predicción", "predicted_fat_percentage"))
            if "real_fat_percentage" in df_plot.columns:
                # Solo tiene sentido si hay algún valor real informado
                if df_plot["real_fat_percentage"].notna().any():
                    metrics.append(("Real", "real_fat_percentage"))

            if not metrics:
                st.info("No hay datos suficientes para dibujar el gráfico.")
            else:
                df_long = []
                for label, col in metrics:
                    tmp = df_plot[["timestamp", col]].copy()
                    tmp = tmp.rename(columns={col: "value"})
                    tmp["serie"] = label
                    df_long.append(tmp)

                df_long = pd.concat(df_long, ignore_index=True)
                df_long = df_long[df_long["value"].notna()]

                if df_long.empty:
                    st.info("No hay valores numéricos para dibujar el gráfico todavía.")
                else:
                    chart = (
                        alt.Chart(df_long)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("timestamp:T", title="Fecha"),
                            y=alt.Y("value:Q", title="% grasa corporal"),
                            color=alt.Color("serie:N", title="Serie"),
                            tooltip=[
                                alt.Tooltip("timestamp:T", title="Fecha"),
                                alt.Tooltip("serie:N", title="Serie"),
                                alt.Tooltip("value:Q", title="%"),
                            ],
                        )
                    )

                    st.subheader("📈 Evolución de predicciones (y real si existe)")
                    st.altair_chart(chart, use_container_width=True)






