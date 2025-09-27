Grasa Corporal

Aplicación interactiva en Streamlit para predecir y hacer seguimiento del porcentaje de grasa corporal de usuarios a partir de datos de salud y entrenamiento.

El proyecto combina:

Machine Learning (Random Forest, scikit-learn) para la estimación.

Feature Engineering (BMI, log_age, etc.) para enriquecer los datos.

Streamlit para una interfaz web simple e intuitiva.

Feature Store local (CSV) para guardar el histórico de cada usuario y permitir reentrenamiento.

Características principales

Predicción del % de grasa corporal a partir de métricas de salud y entrenamiento.

Almacenamiento de registros en un Feature Store local para mantener el historial.

Posibilidad de añadir valores reales de grasa corporal (si se conocen).

Interfaz web sencilla y en español con Streamlit.

Entrenamiento y reentrenamiento de modelos con scikit-learn.

Instalación

Clona el repositorio e instala las dependencias con Poetry:

git clone https://github.com/ALM1808/Calculo_Grasa_corporal.git
cd Calculo_Grasa_corporal
poetry install
poetry shell

Uso local

Ejecuta la aplicación de Streamlit:

poetry run streamlit run app.py


Esto abrirá la app en http://localhost:8501/
.

Ejemplo de uso en Python

En el siguiente ejemplo se muestra cómo preparar la entrada de datos para el modelo.

⚠️ Los nombres de las variables deben estar en inglés, ya que así se entrenó el modelo.
La siguiente tabla explica en español qué representa cada campo.

Campos de entrada del modelo
Variable (inglés)	Descripción en español	Ejemplo
Age	Edad en años	32
Gender	Género (Male / Female)	Female
Weight (kg)	Peso en kilogramos	68.1
Height (m)	Altura en metros	1.66
Max_BPM	Frecuencia cardiaca máxima (latidos/min)	167
Avg_BPM	Frecuencia cardiaca media (latidos/min)	122
Resting_BPM	Frecuencia cardiaca en reposo (latidos/min)	54
Session_Duration (hours)	Duración de la sesión en horas	1.11
Calories_Burned	Calorías quemadas	677
Workout_Type	Tipo de entrenamiento (Yoga, HIIT, Cardio, Strength, Mixed)	Cardio
Water_Intake (liters)	Ingesta de agua en litros	2.3
Workout_Frequency (days/week)	Frecuencia de entrenamiento semanal (días/semana)	4
Experience_Level	Nivel de experiencia (1 = principiante, 5 = avanzado)	2
Fat_Percentage	Grasa corporal real (%). Usar None si no se conoce	None
Código de ejemplo
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
    "Fat_Percentage": None
}

df = pd.DataFrame([nuevo_usuario])
df = build_all_features(df)

# Añadir identificadores
df["email"] = "correo@ejemplo.com"
df["user_id"] = str(uuid.uuid4())
df["timestamp"] = datetime.now().isoformat(timespec="seconds")

# Guardar en el Feature Store
save_features(df, entity="user_fat_percentage", version="v1", use_date=True)

Estructura del proyecto
├── app.py                        # Aplicación principal (Streamlit)
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

Tecnologías utilizadas

Python

scikit-learn

Streamlit

Poetry

Próximos pasos

Mejorar la visualización del historial del usuario (gráficas comparativas).

Implementar almacenamiento remoto en Hopsworks para gestionar de forma centralizada los features y los modelos entrenados.

Contenerizar la aplicación con Docker para facilitar el despliegue y la portabilidad.

Desplegar la aplicación en un servicio en la nube (Render, Google Cloud u otra plataforma).

Automatizar el reentrenamiento del modelo con los nuevos datos del Feature Store.