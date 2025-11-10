# app/monitoring_app.py

import os
from datetime import datetime

import pandas as pd
import streamlit as st

# -------------------------
# Config Hopsworks
# -------------------------
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "https://c.app.hopsworks.ai")


@st.cache_resource(show_spinner=True)
def get_hopsworks_client():
    """
    Devuelve (project, fs) si las credenciales son válidas.
    Si falla, devolvemos (None, None) para que la app no se rompa.
    """
    if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY:
        return None, None

    try:
        import hopsworks

        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT,
            host=HOPSWORKS_HOST,
        )
        fs = project.get_feature_store()
        return project, fs
    except Exception as e:
        st.error(f"No se pudo conectar a Hopsworks: {e}")
        return None, None


def load_feature_groups(fs):
    """
    Lee:
    - user_fat_percentage v1: predicciones + features
    - user_fat_feedback v1 : feedback real
    Si alguno no existe, devuelve df vacío.
    """
    # Predicciones
    try:
        fg_pred = fs.get_feature_group("user_fat_percentage", version=1)
        df_pred = fg_pred.read()
    except Exception:
        df_pred = pd.DataFrame()

    # Feedback
    try:
        fg_fb = fs.get_feature_group("user_fat_feedback", version=1)
        df_fb = fg_fb.read()
    except Exception:
        df_fb = pd.DataFrame()

    return df_pred, df_fb


def build_dashboard(df_pred: pd.DataFrame, df_fb: pd.DataFrame):
    st.title("📊 Monitorización modelo grasa corporal")

    # ----------------- sección: estado general -----------------
    if df_pred.empty:
        st.warning("Aún no hay datos en `user_fat_percentage`.")
        return

    st.subheader("Estado general")

    total_preds = len(df_pred)
    unique_users = df_pred["email"].nunique() if "email" in df_pred.columns else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Total predicciones", total_preds)
    if unique_users is not None:
        col2.metric("Usuarios únicos", unique_users)

    # Última predicción
    if "timestamp" in df_pred.columns:
        df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"], errors="coerce")
        last_ts = df_pred["timestamp"].max()
        col3.metric(
            "Última predicción",
            last_ts.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(last_ts) else "-",
        )

    # ----------------- sección: calidad del modelo -----------------
    st.subheader("Calidad (donde tenemos valor REAL)")

    # intentamos usar real_fat_percentage si ya viene en df_pred
    df_eval = df_pred.copy()
    if "real_fat_percentage" in df_eval.columns:
        df_eval = df_eval[df_eval["real_fat_percentage"].notna()]

    # o join con feedback si existe FG aparte
    if df_fb is not None and not df_fb.empty:
        if "timestamp" in df_fb.columns:
            df_fb["timestamp"] = pd.to_datetime(df_fb["timestamp"], errors="coerce")
        # join aproximado por email (simple; en proyecto real se haría mejor)
        df_fb_last = (
            df_fb.sort_values("timestamp")
            .groupby("email", as_index=False)
            .last()
        )
        df_eval = pd.merge(
            df_pred,
            df_fb_last[["email", "real_fat_percentage"]],
            on="email",
            how="left",
            suffixes=("", "_fb"),
        )
        # priorizamos columna del feedback group si existe
        if "real_fat_percentage_fb" in df_eval.columns:
            df_eval["real_fat_percentage"] = df_eval["real_fat_percentage_fb"]
        df_eval = df_eval[df_eval["real_fat_percentage"].notna()]

    if df_eval.empty or "predicted_fat_percentage" not in df_eval.columns:
        st.info("Todavía no hay suficientes pares (predicción, valor real) para métricas.")
    else:
        import numpy as np

        y_true = pd.to_numeric(
            df_eval["real_fat_percentage"], errors="coerce"
        )
        y_pred = pd.to_numeric(
            df_eval["predicted_fat_percentage"], errors="coerce"
        )

        mask = y_true.notna() & y_pred.notna()
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            st.info("Hay registros, pero no valores numéricos válidos para evaluar.")
        else:
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

            c1, c2 = st.columns(2)
            c1.metric("MAE (|real - pred|)", f"{mae:.2f} puntos")
            c2.metric("RMSE", f"{rmse:.2f} puntos")

            st.write("Distribución de errores")
            st.bar_chart((y_pred - y_true), height=200)

    # ----------------- sección: inspección de datos -----------------
    st.subheader("Muestra de registros recientes")
    cols_show = [
        c
        for c in df_pred.columns
        if c
        in [
            "timestamp",
            "email",
            "age",
            "gender",
            "weight_kg",
            "bmi",
            "predicted_fat_percentage",
            "real_fat_percentage",
        ]
    ]
    df_show = df_pred.copy()
    if "timestamp" in df_show.columns:
        df_show["timestamp"] = pd.to_datetime(df_show["timestamp"], errors="coerce")
        df_show = df_show.sort_values("timestamp", ascending=False)

    st.dataframe(
        df_show[cols_show].head(50) if cols_show else df_show.head(50),
        use_container_width=True,
    )


def main():
    st.set_page_config(page_title="Monitorización Grasa Corporal", layout="wide")

    project, fs = get_hopsworks_client()
    if project is None or fs is None:
        st.error(
            "No se pudo conectar a Hopsworks. "
            "Comprueba HOPSWORKS_PROJECT, HOPSWORKS_API_KEY y HOPSWORKS_HOST en tu .env."
        )
        return

    df_pred, df_fb = load_feature_groups(fs)
    build_dashboard(df_pred, df_fb)


if __name__ == "__main__":
    main()
