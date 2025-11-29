# app/monitoring_app.py

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# =========================
# Configuración inicial
# =========================
load_dotenv()

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")

st.set_page_config(
    page_title="GrasaCorporal - Monitoring",
    layout="wide",
)

st.title("🧪 Monitorización del modelo de grasa corporal")
st.caption("Dashboard interno para revisar predicciones, feedback y drift de datos.")

# =========================
# Utilidades
# =========================

@st.cache_resource(show_spinner=False)
def get_feature_store():
    """Conecta a Hopsworks y devuelve (project, fs)."""
    if not (HOPSWORKS_PROJECT and HOPSWORKS_API_KEY):
        raise RuntimeError("Faltan HOPSWORKS_PROJECT o HOPSWORKS_API_KEY en el entorno/.env")

    try:
        import hopsworks
    except Exception as e:
        raise RuntimeError(f"No se pudo importar hopsworks: {e}")

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
    )
    fs = project.get_feature_store()
    return project, fs


@st.cache_data(show_spinner=True, ttl=300)
def load_data():
    """Lee user_fat_percentage v1 y user_fat_feedback v1 (si existe)."""
    project, fs = get_feature_store()

    # FG principal con predicciones
    fg_pred = fs.get_feature_group("user_fat_percentage", version=1)
    df_pred = fg_pred.read()

    # Feedback (puede que no exista aún)
    try:
        fg_fb = fs.get_feature_group("user_fat_feedback", version=1)
        df_fb = fg_fb.read()
    except Exception:
        df_fb = pd.DataFrame()

    return df_pred, df_fb


def join_with_feedback(df_pred: pd.DataFrame, df_fb: pd.DataFrame) -> pd.DataFrame:
    """Une predicciones y feedback por user_id + email + timestamp redondeado."""
    if df_fb.empty:
        return pd.DataFrame()

    df_main = df_pred.copy()
    df_fb_ = df_fb.copy()

    for c in ("user_id", "email"):
        if c in df_main.columns:
            df_main[c] = df_main[c].astype(str)
        if c in df_fb_.columns:
            df_fb_[c] = df_fb_[c].astype(str)

    # Redondeo timestamp a segundos para mejorar matching
    df_main["timestamp"] = pd.to_datetime(df_main["timestamp"], errors="coerce").dt.round("S")
    df_fb_["timestamp"] = pd.to_datetime(df_fb_["timestamp"], errors="coerce").dt.round("S")

    # Join interno
    on_cols = ["user_id", "email", "timestamp"]
    common = [c for c in on_cols if c in df_main.columns and c in df_fb_.columns]
    if len(common) < 2:
        return pd.DataFrame()

    df_join = pd.merge(
        df_main,
        df_fb_,
        on=common,
        suffixes=("_pred", "_fb"),
        how="inner",
    )

    # Normalizar nombres por comodidad
    if "predicted_fat_percentage_pred" in df_join.columns:
        df_join["y_pred"] = df_join["predicted_fat_percentage_pred"]
    elif "predicted_fat_percentage" in df_join.columns:
        df_join["y_pred"] = df_join["predicted_fat_percentage"]

    if "real_fat_percentage" in df_join.columns:
        df_join["y_true"] = df_join["real_fat_percentage"]

    df_join = df_join.dropna(subset=["y_true", "y_pred"])
    return df_join


def compute_regression_metrics(df: pd.DataFrame):
    """Calcula MAE, RMSE, R2 si hay columnas y_true / y_pred."""
    if df.empty or "y_true" not in df.columns or "y_pred" not in df.columns:
        return None

    y_true = df["y_true"].astype(float)
    y_pred = df["y_pred"].astype(float)

    if len(y_true) == 0:
        return None

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R2 manual:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "n": int(len(y_true)),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3) if not np.isnan(r2) else None,
    }


def simple_drift_report(df: pd.DataFrame, days_recent: int = 7):
    """Compara medias históricas vs últimos días para algunas features numéricas."""
    if df.empty:
        return None

    if "timestamp" not in df.columns:
        return None

    df_ = df.copy()
    df_["timestamp"] = pd.to_datetime(df_["timestamp"], errors="coerce")
    df_ = df_.dropna(subset=["timestamp"])
    if df_.empty:
        return None

    cutoff = df_["timestamp"].max() - timedelta(days=days_recent)
    df_old = df_[df_["timestamp"] < cutoff]
    df_new = df_[df_["timestamp"] >= cutoff]

    if df_old.empty or df_new.empty:
        return None

    candidates = ["age", "weight_kg", "bmi", "log_age"]
    rows = []
    for col in candidates:
        if col in df_.columns:
            m_old = df_old[col].mean()
            m_new = df_new[col].mean()
            diff = m_new - m_old
            rel = (diff / m_old * 100) if m_old != 0 else np.nan
            rows.append({
                "feature": col,
                "past_mean": round(m_old, 3),
                "recent_mean": round(m_new, 3),
                "abs_diff": round(diff, 3),
                "rel_diff_%": round(rel, 2) if not np.isnan(rel) else None,
            })

    return pd.DataFrame(rows)


# =========================
# Layout
# =========================

# Estado conexión
with st.sidebar:
    st.header("🔐 Conexión Hopsworks")
    if not (HOPSWORKS_PROJECT and HOPSWORKS_API_KEY):
        st.error("Faltan variables HOPSWORKS en .env")
    else:
        st.success(f"Proyecto: {HOPSWORKS_PROJECT}")
        st.caption(f"Host: {HOPSWORKS_HOST}")
    st.markdown("---")
    st.caption("Este panel es solo para uso interno (monitorización MLOps).")

# Cargar datos
try:
    df_pred, df_fb = load_data()
except Exception as e:
    st.error(f"❌ Error al cargar datos desde Hopsworks:\n{e}")
    st.stop()

# Sección 1: Estado general
st.subheader("📊 Estado general de predicciones")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total predicciones registradas", len(df_pred))
with col2:
    st.metric("Registros con feedback", len(df_fb))
with col3:
    uniques = df_pred["email"].nunique() if "email" in df_pred.columns else 0
    st.metric("Usuarios únicos", uniques)

st.write("Vista rápida de las últimas predicciones:")
st.dataframe(df_pred.sort_values("timestamp").tail(10), use_container_width=True)

# Sección 2: Métricas de rendimiento (si hay feedback)
st.subheader("🎯 Calidad del modelo (si hay feedback)")

def join_with_feedback(df_pred: pd.DataFrame, df_fb: pd.DataFrame) -> pd.DataFrame:
    """Une predicciones y feedback por user_id + email + timestamp redondeado."""
    if df_fb.empty:
        return pd.DataFrame()

    df_main = df_pred.copy()
    df_fb_ = df_fb.copy()

    # Normalizar tipos clave
    for c in ("user_id", "email"):
        if c in df_main.columns:
            df_main[c] = df_main[c].astype(str)
        if c in df_fb_.columns:
            df_fb_[c] = df_fb_[c].astype(str)

    # Redondeo timestamp a segundos
    df_main["timestamp"] = pd.to_datetime(df_main["timestamp"], errors="coerce").dt.round("S")
    df_fb_["timestamp"] = pd.to_datetime(df_fb_["timestamp"], errors="coerce").dt.round("S")

    # Join interno
    on_cols = ["user_id", "email", "timestamp"]
    common = [c for c in on_cols if c in df_main.columns and c in df_fb_.columns]
    if len(common) < 2:
        return pd.DataFrame()

    df_join = pd.merge(
        df_main,
        df_fb_,
        on=common,
        suffixes=("_pred", "_fb"),
        how="inner",
    )

    # -------- y_pred (predicción del modelo) ----------
    if "predicted_fat_percentage_pred" in df_join.columns:
        df_join["y_pred"] = df_join["predicted_fat_percentage_pred"]
    elif "predicted_fat_percentage" in df_join.columns:
        df_join["y_pred"] = df_join["predicted_fat_percentage"]

    # -------- y_true (valor real del feedback) ----------
    if "real_fat_percentage_fb" in df_join.columns:
        df_join["y_true"] = df_join["real_fat_percentage_fb"]
    elif "real_fat_percentage" in df_join.columns:
        # fallback por si en algún momento no hay sufijos
        df_join["y_true"] = df_join["real_fat_percentage"]

    # Si no hemos podido crear ambas columnas, no seguimos
    if "y_true" not in df_join.columns or "y_pred" not in df_join.columns:
        return pd.DataFrame()

    # Filtrar filas válidas
    df_join = df_join.dropna(subset=["y_true", "y_pred"])

    return df_join


# Sección 3: Data Drift sencillo
st.subheader("🌡️ Chequeo rápido de Data Drift")

drift_df = simple_drift_report(df_pred)
if drift_df is None or drift_df.empty:
    st.info("Todavía no hay suficientes datos históricos para evaluar drift.")
else:
    st.write("Comparación de medias históricas vs últimos días:")
    st.dataframe(drift_df, use_container_width=True)

