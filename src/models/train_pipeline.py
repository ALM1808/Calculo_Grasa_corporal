# src/models/train_pipeline.py
# src/models/train_pipeline.py
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as skl_version
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Rutas del proyecto ---
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "interim" / "feature_engineered_data.csv"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Fat_Percentage"


def _ohe():
    """Devuelve un OneHotEncoder compatible con diferentes versiones de sklearn."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    print(f">> TRAIN SCRIPT PATH: {__file__}")
    import sys
    print(f">> PYTHON EXECUTABLE: {sys.executable}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset de características en {DATA_PATH}. "
            "Asegúrate de ejecutar primero el notebook/paso que genera 'feature_engineered_data.csv'."
        )

    # --- Cargar datos ---
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(
            f"Columna objetivo '{TARGET}' no está en el dataset. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    # Separamos X/y
    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET]).copy()

    # --- Tipos y selección de columnas ---
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) == 0 and len(cat_cols) == 0:
        raise ValueError("No se detectaron columnas numéricas ni categóricas en X.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", _ohe(), cat_cols),
        ],
        remainder="drop",
        n_jobs=None,
        verbose_feature_names_out=False,
    )

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    # --- Split y entrenamiento ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pipeline.fit(X_train, y_train)

    # --- Métricas ---
    def _metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        # En sklearn 1.6.1 mean_squared_error YA NO tiene 'squared'
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(mse**0.5)
        r2 = r2_score(y_true, y_pred)
        return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}

    y_pred_tr = pipeline.predict(X_train)
    y_pred_te = pipeline.predict(X_test)

    metrics = {
        "train": _metrics(y_train, y_pred_tr),
        "test": _metrics(y_test, y_pred_te),
    }

    # --- Guardado de artefactos (comprimido) ---
    ts = datetime.now().strftime("%Y-%m-%d")
    main_model_path = MODEL_DIR / "rf_pipeline.pkl"
    dated_model_path = MODEL_DIR / f"rf_pipeline_v1_{ts}.pkl"

    joblib.dump(pipeline, main_model_path, compress=("xz", 3))
    joblib.dump(pipeline, dated_model_path, compress=("xz", 3))

    print(f"✅ Pipeline guardado en: {main_model_path.name}")
    print(f"📦 Backup guardado en: {dated_model_path.name}")

    # --- Reporte/metadata ---
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": skl_version,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "target": TARGET,
        "n_rows": int(df.shape[0]),
        "n_features_before": int(X.shape[1]),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "model_paths": {
            "main": str(main_model_path.relative_to(ROOT)),
            "backup": str(dated_model_path.relative_to(ROOT)),
        },
        "metrics": metrics,
    }

    report_path = REPORTS_DIR / "model_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"📝 Reporte de entrenamiento en: {report_path.relative_to(ROOT)}")
    print("Métricas (test):", metrics["test"])


if __name__ == "__main__":
    main()

