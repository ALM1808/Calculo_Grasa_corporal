🧠 Predicción de Porcentaje de Grasa Corporal (End-to-End MLOps)

Proyecto completo de ML + MLOps para predecir el porcentaje de grasa corporal a partir de datos de entrenamiento físico, integrando:

Ingesta y transformación de datos

Entrenamiento de un modelo de Machine Learning con scikit-learn

Uso de Hopsworks como:

Feature Store

Model/Prediction tracking (via Feature Groups)

API backend con FastAPI

Frontend con Streamlit consumiendo la API

Notebook de monitorización y data drift

Este repositorio está diseñado como proyecto de curso / portfolio demostrando un flujo moderno de MLOps, pero funcionando también en local.

🗂 Estructura del proyecto
.
├── app/
│   ├── app.py                 # (Opcional) Versión legacy: Streamlit con modelo directo
│   └── api_client_app.py      # ✅ Streamlit frontend que llama al backend FastAPI
├── backend/
│   └── main.py                # ✅ API FastAPI (/predict, /feedback)
├── data/
│   ├── raw/
│   │   └── gym_members_exercise_tracking.csv
│   ├── interim/
│   │   └── feature_engineered_data.csv
│   └── processed/
│       └── preprocessed_data.csv
├── models/
│   └── rf_pipeline.pkl        # ✅ Pipeline entrenado (preprocesamiento + modelo)
├── notebooks/
│   ├── 01_eda.ipynb           # Exploración de datos
│   ├── 02_feature_engineering.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_train_pipeline.ipynb (opcional; ahora se usa script)
│   └── 09_monitoring.ipynb    # ✅ Monitorización + drift
├── src/
│   └── models/
│       ├── train_pipeline.py  # Entrena y guarda rf_pipeline.pkl
│       └── predict_model.py   # load_model() para backend
├── feature_store/             # Feature store local (opcional)
├── .env                       # 🔐 Variables de entorno (NO subir a GitHub)
├── requirements.txt
└── README.md

📊 1. Datos

Dataset base: data/raw/gym_members_exercise_tracking.csv
Contiene información de usuarios de gimnasio:

Edad, género, peso, altura

Frecuencias cardíacas (máx, media, reposo)

Duración de sesión, tipo de entrenamiento

Consumo de agua, frecuencia de entrenamiento

Objetivo/índice de grasa corporal (Fat_Percentage)

Paso a paso (notebooks)

01_eda.ipynb

Carga el CSV

Revisa tipos, nulos, estadísticas descriptivas

Primeras visualizaciones sencillas

02_feature_engineering.ipynb

Crea nuevas columnas:

BMI = peso / altura²

Log_Age = log(edad)

Guarda resultado en:

data/interim/feature_engineered_data.csv

03_preprocessing.ipynb

Opcional según versión.

En la versión estable actual:

El preprocesamiento se incorpora en el pipeline final en train_pipeline.py.

🤖 2. Entrenamiento del modelo

El script principal es:

src/models/train_pipeline.py


Este script:

Lee data/interim/feature_engineered_data.csv.

Separa:

y = Fat_Percentage

X con el resto de columnas.

Detecta numéricas y categóricas.

Construye un Pipeline de sklearn:

ColumnTransformer(
  num -> StandardScaler,
  cat -> OneHotEncoder(handle_unknown="ignore")
)
+ RandomForestRegressor


Entrena con train_test_split.

Guarda el pipeline completo en:

models/rf_pipeline.pkl


Este pipeline incluye:

Preprocesamiento

Modelo

Soporte de columnas esperadas con nombres del dataset original:

"Age", "Gender", "Weight (kg)", "Height (m)", ... , "BMI", "Log_Age"

Después del entrenamiento, el backend solo necesita este .pkl.

🧩 3. Carga del modelo en producción

Archivo:

src/models/predict_model.py


Responsabilidad:

Localiza models/rf_pipeline.pkl.

Lo carga con joblib.

Expone load_model() para que el backend lo use.

🛰 4. Backend: FastAPI

Archivo principal:

backend/main.py

Endpoints
GET /

Healthcheck sencillo:

{ "message": "API de predicción de grasa corporal 🧠" }

POST /predict

Request JSON (snake_case, lo que envía el frontend):

{
  "email": "user@example.com",
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
  "experience_level": "2"
}


Lógica interna:

Valida con PredictionInput (Pydantic).

Calcula:

BMI

Log_Age

Traduce las columnas a los nombres que espera el modelo (SNAKE_TO_MODEL).

Ordena columnas según EXPECTED_MODEL_COLS.

Usa load_model() para cargar rf_pipeline.pkl.

Devuelve:

{ "predicted_fat_percentage": 25.40 }


Además (si hay credenciales Hopsworks configuradas):

Registra la predicción en el Feature Group:

user_fat_percentage v1

POST /feedback

Para guardar el valor real enviado por el usuario cuando lo conozca:

{
  "email": "user@example.com",
  "real_fat_percentage": 24.5,
  "predicted_fat_percentage": 25.4
}


Guarda el feedback en:

user_fat_feedback v1

Esto permite evaluar el modelo con datos reales posteriormente.

🖥 5. Frontend: Streamlit via API

Archivo principal recomendado:

app/api_client_app.py


Funciona así:

Pide email y todas las variables de entrada.

Construye el JSON con los nombres EXACTOS que espera el backend.

Llama a:

POST http://localhost:8000/predict


Muestra la predicción al usuario.

Mantiene la última predicción en st.session_state para:

Enviar feedback real (/feedback) cuando el usuario lo introduzca.

El frontend NO carga el modelo directamente:

Todo pasa por la API → arquitectura limpia y desacoplada.

app/app.py queda como versión alternativa/histórica:

Streamlit cargando modelo directo/local/GitHub/Hopsworks.

No es necesario para la versión API-first.

🧱 6. Integración con Hopsworks

Esta parte es opcional pero ya la tienes integrada y funcionando.

Feature Groups

user_fat_percentage v1

Contiene (formato snake_case):

Claves:

user_id, email, timestamp

Features:

age, gender, weight_kg, height_m, max_bpm, avg_bpm, resting_bpm, session_duration_hours, calories_burned, workout_type, water_intake_liters, workout_frequency_days_week, experience_level, bmi, log_age

Targets:

predicted_fat_percentage

real_fat_percentage (si se conoce)

Se escribe desde:

backend/main.py → save_prediction_to_hopsworks()

user_fat_feedback v1

Contiene:

user_id, email, timestamp, real_fat_percentage, predicted_fat_percentage

Se escribe desde:

Endpoint /feedback

📈 7. Monitorización & Data Drift

Notebook:

notebooks/09_monitoring.ipynb


Hace:

Carga user_fat_percentage y user_fat_feedback desde Hopsworks.

Une predicciones + valores reales (cuando existen).

Calcula métricas básicas del modelo:

MAE

RMSE

R²

Detecta posibles señales de data drift comparando:

Distribuciones históricas vs recientes de features clave

Por ejemplo: edad, peso, BMI.

Muestra tablas/resúmenes para explicar:

Cómo está funcionando el modelo en producción.

Si los usuarios actuales se parecen a los del entrenamiento.

Este notebook actúa como un dashboard interno sencillo de MLOps:

No afecta al usuario final.

Es perfecto para explicar en el curso/entrevista:

“Tengo monitorización básica implementada sobre Feature Store”.

⚙️ 8. Cómo ejecutar el proyecto en local
1️⃣ Clonar y entrar
git clone <tu_repo>
cd GRASACORPORAL

2️⃣ Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\activate   # Windows PowerShell
# source .venv/bin/activate  # Linux/Mac

3️⃣ Instalar dependencias
pip install -r requirements.txt

4️⃣ Crear .env en la raíz
HOPSWORKS_API_KEY=TU_API_KEY   # opcional; si no lo pones, simplemente no sube a Hopsworks
HOPSWORKS_PROJECT=GrasaCorporal
HOPSWORKS_HOST=c.app.hopsworks.ai


⚠️ No subas .env a GitHub.

5️⃣ Entrenar modelo (si no existe)
python -m src.models.train_pipeline
# genera models/rf_pipeline.pkl

6️⃣ Levantar el backend

Desde la raíz del proyecto, con el entorno activo:

uvicorn backend.main:app --reload


La API estará en:

http://127.0.0.1:8000

Documentación interactiva: http://127.0.0.1:8000/docs

7️⃣ Levantar el frontend (Streamlit)

En otra terminal (también con .venv activado):

streamlit run app/api_client_app.py


Interactúas desde el navegador:

Introduces tus datos

Ves la predicción

(Opcional) Envías feedback real

☁️ 9. Despliegue (visión general)

La arquitectura está preparada para:

Backend FastAPI en un contenedor → desplegable en:

Google Cloud Run

Render

Azure, etc.

Frontend Streamlit en otro contenedor independiente.

Ambos apuntando a:

Mismo modelo versionado

Mismo Feature Store en Hopsworks

La separación Front / Back:

Permite escalar, monitorizar y actualizar el modelo sin tocar el frontend.

Demuestra buenas prácticas MLOps.

📝 10. Qué demuestra este proyecto

En una frase:

De datos crudos en CSV → features → modelo en pipeline → API → frontend separado → logging en Feature Store → monitorización.