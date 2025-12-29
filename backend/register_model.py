import json
from pathlib import Path
import hopsworks

# =========================
# CONFIG
# =========================
MODEL_PATH = Path("models/rf_pipeline_v2.pkl")
PROMOTION_PATH = Path("promotion.json")
METRICS_PATH = Path("metrics.json")

MODEL_NAME = "grasa_corporal_model"

# =========================
# LOAD DECISION
# =========================
if not PROMOTION_PATH.exists():
    print("❌ promotion.json no existe → abortando")
    exit(0)

with open(PROMOTION_PATH) as f:
    promotion = json.load(f)

if not promotion.get("promote", False):
    print("⛔ Modelo NO promovido → no se registra")
    exit(0)

if not MODEL_PATH.exists():
    print("❌ Modelo v2 no encontrado → abortando")
    exit(1)

print("🏆 Modelo aprobado → registrando en Hopsworks")

# =========================
# CONNECT TO HOPSWORKS
# =========================
project = hopsworks.login()
mr = project.get_model_registry()

# =========================
# LOAD METRICS
# =========================
metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

# =========================
# REGISTER MODEL
# =========================
model = mr.python.create_model(
    name=MODEL_NAME,
    description="Modelo de predicción de grasa corporal",
    metrics={
        "mae": metrics["v2"]["mae"],
        "rmse": metrics["v2"]["rmse"],
        "n_samples": metrics["n_samples"],
    },
)

model.save(MODEL_PATH)

print("✅ Modelo registrado en Hopsworks Model Registry")
