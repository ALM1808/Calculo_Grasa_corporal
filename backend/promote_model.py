import json
import shutil
from pathlib import Path
import sys

METRICS_PATH = Path("metrics.json")
MODEL_V2_PATH = Path("models/rf_pipeline_v2.pkl")
MODEL_ACTIVE_PATH = Path("models/rf_pipeline.pkl")

if not METRICS_PATH.exists():
    print("❌ metrics.json no encontrado")
    sys.exit(1)

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

improves = metrics.get("improves", False)

if improves:
    print("🚀 El modelo v2 mejora → se PROMUEVE a producción")

    if not MODEL_V2_PATH.exists():
        print("❌ Modelo v2 no encontrado")
        sys.exit(1)

    MODEL_ACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODEL_V2_PATH, MODEL_ACTIVE_PATH)

    print("✅ Modelo promocionado como rf_pipeline.pkl")

else:
    print("⛔ El modelo v2 NO mejora → se mantiene el modelo actual")
