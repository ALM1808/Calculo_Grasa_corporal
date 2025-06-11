import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Crea un ColumnTransformer con escalado para columnas numéricas
    y one-hot encoding para columnas categóricas.
    """
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    return preprocessor


def preprocess_dataframe(df: pd.DataFrame):
    """
    Aplica el preprocesador al DataFrame y devuelve:
    - el array transformado
    - el preprocesador entrenado
    """
    preprocessor = create_preprocessor(df)
    transformed = preprocessor.fit_transform(df)
    return transformed, preprocessor