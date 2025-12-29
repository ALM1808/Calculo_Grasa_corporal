import json
import sys
from pathlib import Path

METRICS_PATH = Path("metrics.json")
DECISION_PATH = Path("promotion_decision.json")

MIN_IMPROVEMENT_FACTOR = 0.98  # 2% mejor
MIN_SAMPLES = 30

if not METRICS_PATH.exists():
    print("❌ metrics.json no encontrado")
    sys.exit(1)

with open(METRICS_PATH) as f:
    metrics = json.load(f)

mae_v1 = metrics["v1"]["mae"]
mae_v2 = metrics["v2"]["mae"]
n_samples = metrics["n_samples"]

improves = (
    mae_v2 < mae_v1 * MIN_IMPROVEMENT_FACTOR
    and n_samples >= MIN_SAMPLES
)

decision = {
    "timestamp": metrics["timestamp"],
    "n_samples": n_samples,
    "mae_v1": mae_v1,
    "mae_v2": mae_v2,
    "rule": {
        "min_improvement_factor": MIN_IMPROVEMENT_FACTOR,
        "min_samples": MIN_SAMPLES,
    },
    "decision": "PROMOTE" if improves else "NO_PROMOTE",
}

with open(DECISION_PATH, "w") as f:
    json.dump(decision, f, indent=2)

print("📣 Decisión de promoción:")
print(json.dumps(decision, indent=2))

# Código de salida informativo (no rompe el workflow)
if not improves:
    print("⚠️ Modelo NO promovido (condiciones no cumplidas)")
