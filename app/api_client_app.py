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
METRICS_URL = f"{API_URL}/metrics"

st.set_page_config(
    page_title="Grasa corporal – Cliente API",
    page_icon="💪",
    layout="wide"
)

st.title("💪 Predicción de grasa corporal — Cliente API")
st.caption(f"Conectado al backend: **{API_URL}**")

# ========================================================
# ESTADO DE SESIÓN (ROBUSTO PARA CLOUD)
# ========================================================
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ⚠️ en vez de None, usamos dict vacío para evitar .get sobre None
if "last_input" not in st.session_state:
    st.session_state.last_input = {}

# asegurar que existen siempre
if "last_prediction_id" not in st.session_state:
    st.session_state.last_prediction_id = None

# ========================================================
# TABS
# ========================================================
tab1, tab2, tab3, tab4 = st.tabs(["🧑‍⚕️ Predicción", "📬 Feedback", "📊 Histórico", "Otro"])

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
                "age": int(age),
                "gender": gender,
                "weight_kg": float(weight_kg),
                "height_m": float(height_m),
                "max_bpm": int(max_bpm),
                "avg_bpm": int(avg_bpm),
                "resting_bpm": int(resting_bpm),
                "session_duration_hours": float(session_duration_hours),
                "calories_burned": float(calories_burned),
                "workout_type": workout_type,
                "water_intake_liters": float(water_intake_liters),
                "workout_frequency_days_week": int(workout_frequency_days_week),
                # ✅ importante: int, no "1"/"2"/"3"
                "experience_level": experience_level,
            }

            try:
                with st.spinner("Llamando a la API…"):
                    resp = requests.post(PREDICT_URL, json=payload, timeout=20)

                if resp.status_code == 200:
                    data = resp.json()

                    pred = float(data["predicted_fat_percentage"])
                    prediction_id = data["prediction_id"]

                    st.session_state.last_prediction = pred
                    st.session_state.last_prediction_id = prediction_id
                    st.session_state.last_input = payload  # dict garantizado

                    st.success(f"Predicción obtenida: **{pred:.2f}%** de grasa corporal")
                    st.caption(f"🆔 prediction_id: {prediction_id}")

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
        if not st.session_state.last_prediction_id or not st.session_state.last_input:
            st.error("Primero debes obtener una predicción para enviar feedback.")
        else:
            fb_payload = {
                "email": st.session_state.last_input["email"],
                "prediction_id": st.session_state.last_prediction_id,
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
# TAB 3 — HISTÓRICO (vía backend)
# ========================================================

def load_history(email: str):
    try:
        r = requests.get(
            HISTORY_URL,
            params={"email": email},
            timeout=15,
        )

        if r.status_code != 200:
            st.error(f"Error consultando histórico ({r.status_code})")
            return []

        data = r.json()
        if not isinstance(data, dict):
            st.error("Respuesta inesperada del backend")
            return []

        return data.get("records", [])

    except Exception as e:
        st.error(f"Error llamando al backend: {e}")
        return []


def parse_timestamp_series(s: pd.Series) -> pd.Series:
    def _one(x):
        if x is None:
            return pd.NaT

        if isinstance(x, dict):
            for k in ["_seconds", "seconds"]:
                if k in x:
                    try:
                        return pd.to_datetime(int(x[k]), unit="s", utc=True)
                    except Exception:
                        return pd.NaT
            return pd.NaT

        try:
            return pd.to_datetime(x, errors="coerce", utc=True)
        except Exception:
            return pd.NaT

    return s.apply(_one)


def load_metrics(email: str | None = None):
    try:
        params = {"email": email} if email else None
        r = requests.get(METRICS_URL, params=params, timeout=15)

        if r.status_code != 200:
            st.error(f"Error cargando métricas ({r.status_code})")
            return None

        return r.json().get("metrics", {})

    except Exception as e:
        st.error(f"Error llamando a /metrics: {e}")
        return None


with tab3:
    st.header("📊 Histórico de predicciones")

    email_filter = st.text_input("Introduce tu email para ver el histórico:").strip().lower()

    if email_filter:
        records = load_history(email_filter)

        if not records:
            st.warning("No hay registros para este email.")
        else:
            df = pd.DataFrame(records)

            if "timestamp" not in df.columns:
                df["timestamp"] = None

            df["timestamp_dt"] = parse_timestamp_series(df["timestamp"])
            df = df.sort_values("timestamp_dt", ascending=False, na_position="last").reset_index(drop=True)

            preferred_cols = [
                "timestamp_dt",
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

            if "timestamp_dt" in df_table.columns:
                df_table["timestamp_dt"] = (
                    df_table["timestamp_dt"]
                    .dt.tz_convert("Europe/Madrid")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                )
                df_table = df_table.rename(columns={"timestamp_dt": "timestamp"})

            st.subheader("📄 Datos históricos (resumen)")
            st.dataframe(df_table, use_container_width=True)

            df_plot = df[df["timestamp_dt"].notna()].copy()
            if df_plot.empty:
                st.info("No hay timestamps válidos para dibujar el gráfico.")
            else:
                df_plot["predicted_fat_percentage"] = pd.to_numeric(df_plot.get("predicted_fat_percentage"), errors="coerce")
                df_plot["real_fat_percentage"] = pd.to_numeric(df_plot.get("real_fat_percentage"), errors="coerce")

                long_parts = []

                if df_plot["predicted_fat_percentage"].notna().any():
                    tmp = df_plot[["timestamp_dt", "predicted_fat_percentage"]].copy()
                    tmp = tmp.rename(columns={"predicted_fat_percentage": "value"})
                    tmp["serie"] = "Predicción"
                    long_parts.append(tmp)

                if df_plot["real_fat_percentage"].notna().any():
                    tmp = df_plot[["timestamp_dt", "real_fat_percentage"]].copy()
                    tmp = tmp.rename(columns={"real_fat_percentage": "value"})
                    tmp["serie"] = "Real"
                    long_parts.append(tmp)

                if long_parts:
                    df_long = pd.concat(long_parts, ignore_index=True)
                    df_long = df_long[df_long["value"].notna()]

                    chart = (
                        alt.Chart(df_long)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("timestamp_dt:T", title="Fecha"),
                            y=alt.Y("value:Q", title="% grasa corporal"),
                            color=alt.Color("serie:N", title="Serie"),
                            tooltip=[
                                alt.Tooltip("timestamp_dt:T", title="Fecha"),
                                alt.Tooltip("serie:N", title="Serie"),
                                alt.Tooltip("value:Q", title="%"),
                            ],
                        )
                    )

                    st.subheader("📈 Evolución (Predicción y Real si existe)")
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No hay valores suficientes para dibujar el gráfico.")

# ========================================================
# TAB 4 — MÉTRICAS
# ========================================================
with tab4:
    st.header("📐 Métricas del modelo")

    scope = st.radio(
        "Ámbito de las métricas",
        ["Globales", "Por usuario"],
        horizontal=True,
    )

    email_metrics = None
    if scope == "Por usuario":
        # ✅ robusto: last_input puede ser {}
        default_email = (st.session_state.get("last_input") or {}).get("email", "")
        email_metrics = st.text_input("Email para métricas", value=default_email).strip().lower()

        if not email_metrics:
            st.info("Introduce un email para ver métricas por usuario.")
            st.stop()

    metrics = load_metrics(email_metrics)

    if not metrics:
        st.warning("No hay métricas disponibles todavía (faltan feedbacks).")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📊 Nº registros", metrics.get("count", 0))
    c2.metric("📉 MAE", metrics.get("mae", "–"))
    c3.metric("📐 RMSE", metrics.get("rmse", "–"))
    c4.metric("⚖️ Error medio", metrics.get("mean_signed_error", "–"))

    st.metric("📎 Error relativo medio (%)", metrics.get("mean_relative_error", "–"))






