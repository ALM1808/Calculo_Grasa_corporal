import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# Ruta al CSV
csv_path = Path("data/interim/feature_engineered_data.csv")
df = pd.read_csv(csv_path)

# Dividir X e y
X = df.drop(columns=["Fat_Percentage"])
y = df["Fat_Percentage"]

# Entrenamiento simple (sin preprocesamiento)
model = RandomForestRegressor(n_estimators=5, random_state=42)
model.fit(X.select_dtypes("number"), y)

print("✅ Modelo entrenado correctamente (sin preprocesamiento)")

