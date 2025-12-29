## 🧠 Predicción de Grasa Corporal — Proyecto MLOps

Este proyecto implementa un sistema completo de **Machine Learning en producción** para predecir el **porcentaje de grasa corporal** a partir de datos físicos y de actividad, incorporando **feedback real**, **monitorización del modelo** y **reentrenamiento automático**.

El objetivo es construir un flujo **end-to-end MLOps**, desde la predicción hasta el autoaprendizaje continuo.

---

## Funcionalidades principales

### 1. Predicción en tiempo real
- API REST con **FastAPI**
- Modelo ML entrenado con `scikit-learn`
- Endpoint `/predict`:
  - recibe datos del usuario
  - devuelve predicción
  - guarda la predicción en base de datos

### 2. Feedback real del usuario
- El usuario puede introducir su **valor real de grasa corporal**
- Endpoint `/feedback`
- El feedback se enlaza a la predicción original mediante `prediction_id`

### 3. Almacenamiento persistente
- **Google Firestore** como fuente de verdad
- Cada predicción se guarda con:
  - features
  - predicción
  - valor real (cuando existe)
  - errores calculados

### 4. Monitorización del modelo
- Endpoint `/metrics`
- Cálculo automático de:
  - MAE
  - RMSE
  - error medio
  - error relativo
- Métricas globales o por usuario

### 5. Dataset vivo para reentrenamiento
- Endpoint `/training-data`
- Extrae automáticamente solo registros **válidos**:
  - con `real_fat_percentage`
  - con features completas
- Fuente directa para nuevos entrenamientos

### 6. Reentrenamiento y comparación de modelos
- Script `backend/train_compare.py`
- Compara:
  - **Modelo v1** (en producción)
  - **Modelo v2** (reentrenado con nuevos datos)
- Métricas comparadas: MAE y RMSE

### 7. Automatización con GitHub Actions
- Workflow `Retrain and Compare Model`
- Ejecución:
  - manual (`workflow_dispatch`)
  - programable (cron)
- El pipeline:
  - descarga datos reales
  - reentrena el modelo
  - compara versiones
  - guarda artefactos del nuevo modelo

---

## Arquitectura

────────────┐
│ Streamlit │ ← Frontend (usuario)
└─────┬──────┘
│ HTTP
┌─────▼──────┐
│ FastAPI │ ← Backend (Cloud Run)
│ /predict │
│ /feedback │
│ /metrics │
│ /history │
│ /training-data
└─────┬──────┘
│
┌─────▼──────────┐
│ Firestore │ ← Fuente de verdad
│ predictions │
└─────┬──────────┘
│
┌─────▼────────────────┐
│ GitHub Actions │
│ train_compare.py │
│ Comparación v1 vs v2 │
└──────────────────────┘

---

## Tecnologías utilizadas

- **Python 3.11**
- **FastAPI**
- **Streamlit**
- **scikit-learn**
- **Pandas / NumPy**
- **Google Firestore**
- **Google Cloud Run**
- **GitHub Actions**
- **Docker (preparado para despliegue)**

---

## Estado del proyecto

- Producción estable con modelo v1

- Monitorización activa

- Reentrenamiento automatizado

- Decisión de mejora del modelo implementada

- A la espera de más datos reales para promover v2



Proyecto desarrollado como caso práctico completo de MLOps, con foco en buenas prácticas reales de ingeniería de Machine Learning y despliegue en producción.
