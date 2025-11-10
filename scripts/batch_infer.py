# Scripts/batch_infer.py
import os, glob, joblib
import pandas as pd
import hopsworks

HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT", "GrasaCorporal")
HOPSWORKS_HOST    = os.environ.get("HOPSWORKS_HOST", "c.app.hopsworks.ai")

def login():
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
    )
    return project, project.get_feature_store(), project.get_model_registry()

def load_latest_model(mr, name):
    models = mr.get_models(name=name)
    assert models, f"No hay modelos '{name}' en el Model Registry."
    m = sorted(models, key=lambda x: x.version)[-1]
    local_dir = m.download()
    model_path = os.path.join(local_dir, "model.pkl")
    if not os.path.exists(model_path):
        pkls = glob.glob(os.path.join(local_dir, "*.pkl"))
        assert pkls, "No se encontró ningún archivo .pkl en los artefactos."
        model_path = pkls[0]
    pipe = joblib.load(model_path)
    print(f"✅ Modelo {name} versión {m.version} cargado.")
    return pipe, m.version

def main():
    project, fs, mr = login()
    pipe, ver = load_latest_model(mr, "rf_pipeline_fat_percentage")

    fg = fs.get_feature_group("user_fat_percentage", version=1)
    df_in = fg.read()
    print("📥 Datos cargados desde user_fat_percentage:", df_in.shape)

    # Mapear nombres snake_case → nombres esperados por el pipeline entrenado
    snake_to_expected = {
        "age": "Age",
        "gender": "Gender",
        "weight_kg": "Weight (kg)",
        "height_m": "Height (m)",
        "max_bpm": "Max_BPM",
        "avg_bpm": "Avg_BPM",
        "resting_bpm": "Resting_BPM",
        "session_duration_hours": "Session_Duration (hours)",
        "calories_burned": "Calories_Burned",
        "workout_type": "Workout_Type",
        "water_intake_liters": "Water_Intake (liters)",
        "workout_frequency_days_week": "Workout_Frequency (days/week)",
        "experience_level": "Experience_Level",
        "bmi": "BMI",
        "log_age": "Log_Age",
    }

    X = df_in[list(snake_to_expected.keys())].rename(columns=snake_to_expected)

    # Convertir columnas categóricas a string
    for c in ["Gender", "Workout_Type", "Experience_Level"]:
        if c in X.columns:
            X[c] = X[c].astype(str)

    # Inferencia
    yhat = pipe.predict(X)
    df_pred = pd.DataFrame({
        "user_id":   df_in["user_id"].astype(str),
        "email":     df_in["email"].astype(str),
        "timestamp": pd.to_datetime(df_in["timestamp"], errors="coerce", utc=True),
        "predicted_fat_percentage": yhat.astype(float),
    })

    print("🔮 Predicciones generadas:", df_pred.shape)

    # Crear o actualizar Feature Group de predicciones
    pfg = fs.get_or_create_feature_group(
        name="user_fat_predictions",
        version=1,
        description="Predicciones batch % grasa corporal por usuario y timestamp",
        primary_key=["user_id", "email", "timestamp"],
        event_time="timestamp",
        online_enabled=False,
    )

    pfg.insert(df_pred, write_options={"wait_for_job": True})
    print("✅ Predicciones insertadas en user_fat_predictions v1")

if __name__ == "__main__":
    main()
