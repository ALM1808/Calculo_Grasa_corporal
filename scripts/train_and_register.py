# scripts/train_and_register.py
import os
import joblib
import pandas as pd
from pathlib import Path

import hopsworks
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- Conexión ---
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT", "GrasaCorporal"),
    host=os.getenv("HOPSWORKS_HOST", "https://c.app.hopsworks.ai"),
)
fs = project.get_feature_store()
mr = project.get_model_registry()

# --- Cargar datos desde Feature View ---
fv = fs.get_feature_view("user_fat_percentage_fv", version=1)
X, y = fv.training_data()

label_col = "real_fat_percentage"
if label_col not in y.columns:
    raise SystemExit(f"❌ La columna de label '{label_col}' no existe en la FV. Añade registros con valor real y reintenta.")

mask = ~y[label_col].isna()
X, y = X[mask], y[mask][label_col]

if len(X) < 10:
    raise SystemExit("❌ Muy pocos ejemplos con label. Inserta más datos reales antes de entrenar.")

# --- Preprocesamiento + modelo ---
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preproc = ColumnTransformer(
    [("num", StandardScaler(), num_cols), ("cat", ohe, cat_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)

pipe = Pipeline([
    ("preprocessing", preproc),
    ("model", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_tr, y_tr)

def _metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
    }

metrics = {"train": _metrics(y_tr, pipe.predict(X_tr)),
           "test":  _metrics(y_te, pipe.predict(X_te))}

print("📊 Métricas (test):", metrics["test"])

# --- Guardar y registrar en MR ---
ROOT = Path(__file__).resolve().parents[1]
artifacts_dir = ROOT / "models" / "hopsworks_export"
artifacts_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, artifacts_dir / "rf_pipeline.pkl")

input_schema = Schema(X.sample(min(len(X), 50)))
output_schema = Schema(pd.DataFrame({"prediction": [0.0]}))
model_schema = ModelSchema(input_schema, output_schema)

model = mr.python.create_model(
    name="rf_pipeline",
    metrics=metrics["test"],
    model_schema=model_schema,
    description="RandomForest para %grasa corporal (Feature View v1)",
)
model.save(str(artifacts_dir))

print(f"✅ Modelo registrado: {model.name} v{model.version}")
