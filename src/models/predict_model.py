# src/models/predict_model.py
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

# Raíz del repo: .../GRASACORPORAL
REPO_ROOT = Path(__file__).resolve().parents[2]

# 👉 Prioriza el pipeline completo
DEFAULT_NAMES = [
    "rf_pipeline.pkl",
    "rf_pipeline_optuna.pkl",
    "debug_rf_pipeline.pkl",
]

# Activa logs con MODEL_DEBUG=1 (útil en contenedor)
DEBUG = os.getenv("MODEL_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _log(msg: str) -> None:
    if DEBUG:
        print(f"[predict_model] {msg}", flush=True)


def _iter_candidates() -> list[Path]:
    """
    Genera una lista ordenada de rutas candidatas donde buscar el modelo.
    - Variables de entorno:
        MODEL_PATH: ruta absoluta/relativa a un .pkl concreto
        MODEL_FILE: nombre del .pkl dentro de models/
    - Rutas típicas en host y en contenedor (/app/...)
    """
    candidates: list[Path] = []

    env_path = os.getenv("MODEL_PATH")
    env_file = os.getenv("MODEL_FILE")

    if env_path:
        candidates.append(Path(env_path))

    if env_file:
        candidates.extend(
            [
                REPO_ROOT / "models" / env_file,
                Path("/app/models") / env_file,
                Path.cwd() / "models" / env_file,
                Path("models") / env_file,
            ]
        )

    for name in DEFAULT_NAMES:
        candidates.extend(
            [
                REPO_ROOT / "models" / name,      # ruta habitual en el repo (host)
                Path("/app/models") / name,       # ruta típica dentro del contenedor
                Path.cwd() / "models" / name,
                Path("models") / name,
            ]
        )

    # Quitar duplicados preservando orden
    uniq: list[Path] = []
    seen = set()
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


@lru_cache(maxsize=1)
def load_model():
    """
    Carga el modelo desde la primera ruta existente de _iter_candidates().
    Se cachea para evitar recargas repetidas.
    """
    tried: list[str] = []
    for path in _iter_candidates():
        tried.append(str(path))
        if path.exists():
            _log(f"Cargando modelo desde: {path}")
            model = joblib.load(path)
            _log(f"Modelo cargado OK: {type(model)}")
            return model
        else:
            _log(f"No existe: {path}")

    # Si no se encontró, da contexto útil
    ctx = [
        f"CWD={os.getcwd()}",
        f"REPO_ROOT={REPO_ROOT}",
        f"MODEL_PATH={os.getenv('MODEL_PATH')}",
        f"MODEL_FILE={os.getenv('MODEL_FILE')}",
    ]
    raise FileNotFoundError(
        "No se encontró ningún modelo serializado (.pkl).\n"
        + "\n".join(ctx)
        + "\nRutas probadas:\n  - "
        + "\n  - ".join(tried)
        + "\nSoluciones:"
        + "\n  • Coloca el fichero en 'models/' del repo (y reconstruye la imagen), o"
        + "\n  • Define MODEL_FILE (p. ej. MODEL_FILE=rf_pipeline.pkl), o"
        + "\n  • Define MODEL_PATH con la ruta absoluta al .pkl."
    )


def predict_fat_percentage(new_data: pd.DataFrame) -> float:
    """
    Realiza una predicción de porcentaje de grasa corporal.
    Espera un DataFrame con UNA sola fila.
    """
    if not isinstance(new_data, pd.DataFrame):
        raise TypeError("new_data debe ser un pandas.DataFrame")
    if len(new_data) != 1:
        raise ValueError("Solo se admite una fila por predicción.")

    model = load_model()
    y = model.predict(new_data)[0]
    return float(y)

