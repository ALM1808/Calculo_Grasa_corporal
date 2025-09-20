Grasa Corporal 🏋️‍♂️📊

Aplicación para predecir y hacer seguimiento de la grasa corporal de usuarios a partir de datos de salud y entrenamiento.

El proyecto combina:

Machine Learning (Random Forest, scikit-learn) para estimar el porcentaje de grasa corporal.

Feature Engineering para generar variables derivadas (BMI, log_age, etc.).

Streamlit para ofrecer una interfaz web interactiva y fácil de usar.

Feature Store local (CSV) que guarda cada registro introducido por los usuarios, permitiendo llevar un historial personal y mejorar el modelo con reentrenamiento.

🚀 Características principales

Predicción del % de grasa corporal a partir de métricas de salud y entrenamiento.

Registro de los datos en un Feature Store local para mantener histórico.

Posibilidad de añadir valores reales de grasa corporal (si se conocen) para mejorar el modelo.

Interfaz web sencilla en Streamlit
.

Entrenamiento y reentrenamiento de modelos con scikit-learn.

⚙️ Instalación

Clona el repositorio e instala las dependencias con Poetry
:

git clone https://github.com/ALM1808/Calculo_Grasa_corporal.git
cd Calculo_Grasa_corporal
poetry install
poetry shell

▶️ Uso local

Ejecuta la aplicación de Streamlit:

poetry run streamlit run app.py


Esto abrirá la aplicación en http://localhost:8501/.

📊 Ejemplo de uso en Python
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

# Añadir identificadores
df["email"] = "correo@ejemplo.com"
df["user_id"] = str(uuid.uuid4())
df["timestamp"] = datetime.now().isoformat(timespec="seconds")

# Guardar en el Feature Store
save_features(df, entity="user_fat_percentage", version="v1", use_date=True)

📂 Estructura del proyecto
├── app.py                        # Aplicación Streamlit
├── data/                         # Datos en crudo y procesados
├── models/                       # Modelos entrenados
├── notebooks/                    # Notebooks de exploración y entrenamiento
├── src/
│   ├── features/                 # Feature engineering y preprocesamiento
│   ├── feature_store/            # Gestión del Feature Store
│   └── models/                   # Entrenamiento y evaluación
├── tests/                        # Tests unitarios
├── pyproject.toml                # Configuración de Poetry
└── README.md                     # Este archivo
