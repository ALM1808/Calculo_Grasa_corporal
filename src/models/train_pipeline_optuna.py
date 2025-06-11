import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import optuna
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# ===============================
# 📁 Configuración de Paths
# ===============================
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "interim" / "feature_engineered_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# 📦 Cargar Datos
# ===============================
df = pd.read_csv(DATA_PATH)
df["Gender"] = df["Gender"].astype("category")
df["Workout_Type"] = df["Workout_Type"].astype("category")

target_col = "Fat_Percentage"
X = df.drop(columns=[target_col])
y = df[target_col]

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["category"]).columns.tolist()

# ===============================
# 🎯 Función Objetivo
# ===============================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(**params, random_state=42))
    ])

    scores = cross_val_score(pipeline, X, y, cv=3, scoring="r2")
    return scores.mean()

# ===============================
# 🔍 Optimización con Optuna
# ===============================
def run_optimization(n_trials=30):
    print("🚀 Iniciando búsqueda de hiperparámetros...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("✅ Mejor R2:", study.best_value)
    print("🏆 Mejores hiperparámetros:", study.best_params)
    return study.best_params

# ===============================
# 🚂 Entrenamiento Final
# ===============================
def train_best_pipeline(best_params):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(**best_params, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    return pipeline

# ===============================
# 💾 Guardar el modelo
# ===============================
def save_pipeline(pipeline):
    today = datetime.today().strftime("%Y-%m-%d")
    main_path = MODEL_DIR / "rf_pipeline_optuna.pkl"
    versioned_path = MODEL_DIR / f"rf_pipeline_optuna_{today}.pkl"

    joblib.dump(pipeline, main_path)
    joblib.dump(pipeline, versioned_path)

    print(f"📦 Modelo guardado en: {main_path}")
    print(f"🗂 Versión guardada en: {versioned_path}")

# ===============================
# 🧠 Ejecutar
# ===============================
if __name__ == "__main__":
    best_params = run_optimization(n_trials=30)
    pipeline = train_best_pipeline(best_params)
    save_pipeline(pipeline)
