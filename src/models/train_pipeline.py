import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Detectar raíz del proyecto
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "interim" / "feature_engineered_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Cargar datos
df = pd.read_csv(DATA_PATH)
target_col = "Fat_Percentage"
X = df.drop(columns=[target_col])
y = df[target_col]

# Preprocesamiento
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Guardar modelo principal
joblib.dump(pipeline, MODEL_DIR / "rf_pipeline.pkl")
print("✅ Modelo guardado en: rf_pipeline.pkl")

# Guardar versión con fecha
fecha = datetime.today().strftime("%Y-%m-%d")
joblib.dump(pipeline, MODEL_DIR / f"rf_pipeline_v1_{fecha}.pkl")
print(f"📦 Backup guardado en: rf_pipeline_v1_{fecha}.pkl")