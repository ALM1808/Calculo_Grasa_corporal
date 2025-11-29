import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Cargar datos
csv_path = Path("data/interim/feature_engineered_data.csv")
df = pd.read_csv(csv_path)

# Separar X
target_col = "Fat_Percentage"
X = df.drop(columns=[target_col])

# Identificar columnas
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Construir preprocesador
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# Aplicar transformación
X_processed = preprocessor.fit_transform(X)

print("✅ Preprocesador aplicado correctamente")
print(f"🔢 Shape resultante: {X_processed.shape}")
