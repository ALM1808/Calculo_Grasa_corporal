## 🧠 Predicción de Grasa Corporal — Proyecto MLOps

Este proyecto implementa un sistema completo de **Machine Learning en producción** para predecir el **porcentaje de grasa corporal** a partir de datos físicos y de actividad, incorporando **feedback real**, **monitorización del modelo** y **reentrenamiento automático**.

El objetivo es construir un flujo **end-to-end MLOps**, desde la predicción hasta el autoaprendizaje continuo.


---

## Funcionalidades principales

1️⃣ Predicción

El usuario introduce sus datos en Streamlit

Se envían al backend FastAPI (/predict)

Se genera una predicción

Se guarda:

Predicción

Features

Timestamp

ID único

2️⃣ Feedback real

El usuario introduce su valor real

Se guarda junto a la predicción original

Se calculan errores individuales:

Error absoluto

Error relativo

Error firmado

3️⃣ Monitorización

Endpoint /metrics

Métricas agregadas:

MAE

RMSE

Error medio

Globales o por usuario

4️⃣ Dataset de reentrenamiento

Endpoint /training-data

Solo registros con valor real

Dataset limpio, consistente y reproducible

5️⃣ Reentrenamiento automático (CI)

Workflow en GitHub Actions:

Descarga datos reales

Evalúa modelo v1

Entrena modelo v2

Compara métricas

Guarda:

metrics.json

promotion.json

6️⃣ Regla de promoción automática
Si MAE(v2) < MAE(v1) → promover
Si no → descartar


✔️ Evita degradación del modelo
✔️ Sin intervención manual
✔️ Seguro para producción

7️⃣ Model Registry (Hopsworks)

Si el modelo mejora:
Se registra automáticamente en Hopsworks Model Registry
Versionado
Metadata
Preparado para despliegue

Si no mejora:
No se registra
No se toca producción

🔧 Tecnologías usadas

Python 3.11

FastAPI — backend

Streamlit — frontend

Scikit-learn — ML

Firestore (GCP) — storage

GitHub Actions — CI/CD

Hopsworks — Feature Store & Model Registry

Docker / Cloud Run — despliegue

📦 Estructura del proyecto
GRASACORPORAL/
├── app/                  # Streamlit frontend
├── backend/
│   ├── main.py           # FastAPI API
│   ├── train_compare.py  # Retrain + compare
│   ├── promote_model.py  # Promotion logic
│   ├── register_model.py # Hopsworks registry
│   └── requirements.txt
├── models/
│   ├── rf_pipeline.pkl
│   └── rf_pipeline_v2.pkl
├── .github/workflows/
│   └── retrain.yml
├── README.md

🧠 Principios MLOps aplicados

✔ Separación de responsabilidades
✔ Dataset versionado implícitamente
✔ Métricas como artefactos
✔ Promoción automática segura
✔ Model Registry real
✔ Reentrenamiento reproducible
✔ Pipeline CI profesional


## Estado del proyecto

- Producción estable con modelo v1

- Monitorización activa

- Reentrenamiento automatizado

- Decisión de mejora del modelo implementada

- A la espera de más datos reales para promover v2



Proyecto desarrollado como caso práctico completo de MLOps, con foco en buenas prácticas reales de ingeniería de Machine Learning y despliegue en producción.
