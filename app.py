import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo
@st.cache_resource
def load_model():
    return joblib.load("models/rf_pipeline_optuna.pkl")

model = load_model()

# Título
st.title("🧠 Calculadora de Grasa Corporal")

# Formulario de entrada
st.header("Introduce tus datos:")

age = st.number_input("Edad", min_value=10, max_value=100, value=30)
weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.number_input("Altura (m)", min_value=1.2, max_value=2.5, value=1.70)
gender = st.selectbox("Sexo", ["Male", "Female"])
exercise_type = st.selectbox("Tipo de ejercicio", ["Cardio", "Strength", "Mixed"])

# Procesamiento de features derivadas
bmi = weight / (height ** 2)
log_age = np.log(age)

# Preparar DataFrame para predicción
input_data = pd.DataFrame([{
    "Age": age,
    "Weight (kg)": weight,
    "Height (m)": height,
    "Gender": gender,
    "Exercise Type": exercise_type,
    "BMI": bmi,
    "Log_Age": log_age
}])

# Predicción
if st.button("Calcular Grasa Corporal"):
    pred = model.predict(input_data)[0]
    st.success(f"✅ Tu porcentaje estimado de grasa corporal es: **{pred:.2f}%**")