# 💪 GrasaCorporal — Plataforma de Predicción de Grasa Corporal  
**Proyecto MLOps completo con FastAPI (backend), Streamlit (frontend) y Docker (despliegue local).**

Este proyecto permite predecir el porcentaje de grasa corporal de una persona a partir de datos fisiológicos y de entrenamiento, usando un modelo de Machine Learning entrenado con un pipeline completo.  
Incluye histórico de predicciones, envío de feedback y visualización temporal.

---

# 🚀 Tecnologías utilizadas

### **Backend (API)**
- FastAPI  
- Python 3.10  
- scikit-learn  
- joblib  
- Pandas / NumPy  
- Contenedorización con Docker

### **Frontend (UI)**
- Streamlit  
- Requests  
- Matplotlib  

### **Infraestructura**
- Docker + Docker Compose (2 contenedores: frontend & backend)  
- CSV como almacenamiento local de predicciones y feedback  
  (Hopsworks deshabilitado dentro de Docker por incompatibilidades del SDK)

---

# 🏗 Arquitectura del proyecto

📦 proyecto/
├── backend/
│ ├── main.py # API FastAPI
│ ├── requirements.txt # Dependencias del backend
│ ├── data_logs/ # Logs y CSVs generados por la API
│ └── ...
│
├── frontend/
│ ├── app.py # Interfaz Streamlit
│ ├── requirements.txt # Dependencias del frontend
│ └── ...
│
├── models/
│ └── rf_pipeline.pkl # Modelo entrenado (si decides versionarlo)
│
├── docker-compose.yml
├── Dockerfile.front
├── Dockerfile.back
├── .gitignore
└── README.md