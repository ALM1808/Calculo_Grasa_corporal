import numpy as np
import pandas as pd

def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade una columna BMI (Índice de Masa Corporal).
    Fórmula: BMI = peso / altura²
    """
    df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
    return df

def add_log_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade una columna Log_Age = log(edad)
    """
    df['Log_Age'] = np.log(df['Age'])
    return df

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las funciones de ingeniería de características al DataFrame.
    """
    df = add_bmi(df)
    df = add_log_age(df)
    return df