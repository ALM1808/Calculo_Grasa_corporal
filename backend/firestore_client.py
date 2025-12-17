
# backend/firestore_client.py
import os
from datetime import datetime
from google.cloud import firestore

# Inicializa el cliente Firestore sólo si se configuró GOOGLE_APPLICATION_CREDENTIALS
def get_firestore_client():
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not creds_path or not os.path.exists(creds_path):
        print("[Firestore] No se encontró GOOGLE_APPLICATION_CREDENTIALS — Firestore desactivado.")
        return None

    try:
        print("[Firestore] Cliente inicializado correctamente.")
        return firestore.Client()
    except Exception as e:
        print(f"[Firestore] Error iniciando cliente: {e}")
        return None


# -----------------------------------------------------------
# Guardar predicción
# -----------------------------------------------------------
def save_prediction_firestore(data: dict):
    client = get_firestore_client()
    if client is None:
        return False

    try:
        doc_ref = client.collection("predictions").document()
        data["timestamp"] = datetime.utcnow()
        doc_ref.set(data)
        print("[Firestore] Predicción guardada.")
        return True
    except Exception as e:
        print(f"[Firestore] Error guardando predicción: {e}")
        return False


# -----------------------------------------------------------
# Guardar feedback
# -----------------------------------------------------------
def save_feedback_firestore(data: dict):
    client = get_firestore_client()
    if client is None:
        return False

    try:
        doc_ref = client.collection("feedback").document()
        data["timestamp"] = datetime.utcnow()
        doc_ref.set(data)
        print("[Firestore] Feedback guardado.")
        return True
    except Exception as e:
        print(f"[Firestore] Error guardando feedback: {e}")
        return False


# -----------------------------------------------------------
# Cargar histórico por email
# -----------------------------------------------------------
def load_history_firestore(email: str):
    client = get_firestore_client()
    if client is None:
        return []

    try:
        query = client.collection("predictions").where("email", "==", email.lower())
        results = query.stream()

        data = []
        for doc in results:
            row = doc.to_dict()
            row["doc_id"] = doc.id
            data.append(row)

        print(f"[Firestore] Recuperadas {len(data)} filas.")
        return data

    except Exception as e:
        print(f"[Firestore] Error leyendo historial: {e}")
        return []
