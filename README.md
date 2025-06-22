Grasa_corporal
==============================

# 📖 TUTORIAL DE USO: Feature Store y Modelo de Predicción

## 1️⃣ Cómo añadir nuevos datos al Feature Store

```python
import pandas as pd
from src.features.build_features import build_all_features
from src.feature_store.versioning_and_inference import save_features
from datetime import datetime
import uuid

# Ejemplo de entrada de usuario
nuevo_usuario = {
    "Age": 32,
    "Gender": "Female",
    "Weight (kg)": 68.1,
    "Height (m)": 1.66,
    "Max_BPM": 167,
    "Avg_BPM": 122,
    "Resting_BPM": 54,
    "Session_Duration (hours)": 1.11,
    "Calories_Burned": 677,
    "Workout_Type": "Cardio",
    "Water_Intake (liters)": 2.3,
    "Workout_Frequency (days/week)": 4,
    "Experience_Level": 2,
    "Fat_Percentage": None  # Deja como None si no se conoce
}

df = pd.DataFrame([nuevo_usuario])
df = build_all_features(df)

# Añadir identificadores de usuario
df["email"] = "correo@ejemplo.com"
df["user_id"] = str(uuid.uuid4())
df["timestamp"] = datetime.now().isoformat(timespec="seconds")

# Guardar en el Feature Store
save_features(df, entity="user_fat_percentage", version="v1", use_date=True)

import pandas as pd
import joblib
from src.features.preprocessing import create_preprocessor
from sklearn.ensemble import RandomForestRegressor
from src.feature_store.versioning_and_inference import load_features

# Cargar la última versión del Feature Store
df = load_features(entity="user_fat_percentage", latest_if_available=True)

# Solo usar filas con valor real de grasa corporal
df = df.dropna(subset=["Fat_Percentage"])

X = df.drop(columns=["Fat_Percentage", "email", "user_id", "timestamp", "Predicted_Fat_Percentage"])
y = df["Fat_Percentage"]

# Crear y ajustar el preprocesador
preprocessor = create_preprocessor(X)
X_prepared = preprocessor.fit_transform(X)

# Entrenar el modelo
model = RandomForestRegressor(random_state=42)
model.fit(X_prepared, y)

# Guardar el modelo actualizado
joblib.dump(model, "models/rf_pipeline_retrained.pkl")
print("✅ Modelo actualizado y guardado")

import pandas as pd
import matplotlib.pyplot as plt
from src.feature_store.versioning_and_inference import load_features

# Cargar todos los registros
df = load_features(entity="user_fat_percentage", latest_if_available=True)

# Filtrar por email de usuario
user_email = "correo@ejemplo.com"
historial = df[df["email"] == user_email].sort_values("timestamp")

# Graficar evolución
plt.plot(historial["timestamp"], historial["Predicted_Fat_Percentage"], marker="o", label="Predicción")
if "Fat_Percentage" in historial and not historial["Fat_Percentage"].isnull().all():
    plt.plot(historial["timestamp"], historial["Fat_Percentage"], marker="s", label="Real")
plt.xlabel("Fecha")
plt.ylabel("Grasa corporal (%)")
plt.title("Evolución de la grasa corporal del usuario")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
