# src/feature_store/versioning_and_inference.py

from pathlib import Path
import pandas as pd
from datetime import date
from typing import Optional


def get_version_with_date(version: str = "v1") -> str:
    today = date.today().isoformat()
    return f"{version}_{today}"


def find_latest_version(entity: str, version_prefix: str = "v1") -> Optional[str]:
    base_path = Path("feature_store") / entity
    if not base_path.exists():
        return None

    matching = sorted(
        [p.name for p in base_path.iterdir() if p.is_dir() and p.name.startswith(version_prefix + "_")],
        reverse=True
    )
    return matching[0] if matching else None


def save_features(df: pd.DataFrame, entity: str, version: str = "v1", use_date: bool = False) -> Path:
    version_folder = get_version_with_date(version) if use_date else version
    path = Path("feature_store") / entity / version_folder
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "features.csv"
    df.to_csv(file_path, index=False)
    print(f"✅ Features guardados en {file_path}")
    return file_path


def load_features(entity: str, version: str = "v1", use_date: bool = False, latest_if_available: bool = False) -> pd.DataFrame:
    if use_date:
        version_folder = get_version_with_date(version)
    elif latest_if_available:
        version_folder = find_latest_version(entity, version_prefix=version)
        if version_folder is None:
            raise FileNotFoundError(f"❌ No se encontró ninguna versión para {entity} con prefijo '{version}_'")
    else:
        version_folder = version

    file_path = Path("feature_store") / entity / version_folder / "features.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"❌ No existe el archivo: {file_path}")
    df = pd.read_csv(file_path)
    print(f"📥 Features cargados desde {file_path}")
    return df


def save_inference_sample(df: pd.DataFrame, entity: str, version: str = "v1", n_rows: int = 5, use_date: bool = False) -> Path:
    version_folder = get_version_with_date(version) if use_date else version
    path = Path("feature_store") / entity / version_folder
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "latest_for_inference.csv"
    df.tail(n_rows).to_csv(file_path, index=False)
    print(f"📤 Muestra para inferencia guardada en {file_path}")
    return file_path