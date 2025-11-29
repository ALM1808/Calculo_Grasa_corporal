import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

print("📥 Cargando dataset...")
csv_path = Path("data/interim/feature_engineered_data.csv")
df = pd.read_csv(csv_path)

print("🎯 Separando X / y")
target_col = "Fat_Percentage"
X = df.drop(columns=[target_col])
y = df[target_col]

print("📝 Renombrando columnas para coincidir con backend...")
X = X.rename(columns={
    "Age": "age",
    "Weight (kg)": "weight_kg",
    "Height (m)": "height_m",
    "Max_BPM": "max_bpm",
    "Avg_BPM": "avg_bpm",
    "Resting_BPM": "resting_bpm",
    "Session_Duration (hours)": "session_duration_hours",
    "Gender": "gender",
    "Workout_Type": "workout_type"
})

print("🧪 Detectando columnas numéricas y categóricas...")
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("🔢 Numéricas:", num_features)
print("🔠 Categóricas:", cat_features)

print("⚙️  Creando preprocesador...")
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

print("🔗 Construyendo pipeline...")
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(n_estimators=10, random_state=42))
])

print("📊 Dividiendo datos en train y test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Ejecutando pipeline.fit...")
pipeline.fit(X_train, y_train)
print("✅ Entrenamiento completado")

print("💾 Guardando modelo...")
from joblib import dump
model_path = Path("models") / "debug_rf_pipeline.pkl"
dump(pipeline, model_path)
print(f"🎉 Pipeline guardado en: {model_path}")
