# test_api.py

import requests

# URL local de tu API (asegúrate de tenerla corriendo con uvicorn)
url = "http://127.0.0.1:8000/predict"

# Datos de ejemplo con todos los campos necesarios
payload = {
    "age": 35,
    "gender": "Male",
    "weight_kg": 75.0,
    "height_m": 1.78,
    "max_bpm": 180,
    "avg_bpm": 140,
    "resting_bpm": 65,
    "session_duration_hours": 1.2,
    "calories_burned": 520.5,
    "workout_type": "Mixed",
    "water_intake_liters": 2.0,
    "workout_frequency_days_week": 4,
    "experience_level": "2",
    "email": "testuser@example.com"  # muy importante para user_id y registro en Hopsworks
}

print("📤 Enviando solicitud POST a la API...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    print("✅ Predicción recibida:")
    print(result)
else:
    print(f"❌ Error {response.status_code}: {response.text}")
