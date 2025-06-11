import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


FEATURE_STORE_DIR = Path("feature_store")


def save_versioned_dataset(df: pd.DataFrame, entity: str, version: str = None):
    date_str = datetime.today().strftime("%Y-%m-%d")
    version = version or f"v1_{date_str}"

    entity_dir = FEATURE_STORE_DIR / entity / version
    entity_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(entity_dir / "data.csv", index=False)
    print(f"✅ Datos guardados en: {entity_dir / 'data.csv'}")


def list_versions(entity: str):
    entity_path = FEATURE_STORE_DIR / entity
    if not entity_path.exists():
        print("❌ No existe esa entidad.")
        return []

    versions = sorted([f.name for f in entity_path.iterdir() if f.is_dir()])
    print(f"📦 Versiones disponibles para '{entity}': {versions}")
    return versions


def track_metadata(df: pd.DataFrame, entity: str, version: str):
    metadata = {
        "version": version,
        "created": datetime.now().isoformat(timespec="seconds"),
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    metadata_path = FEATURE_STORE_DIR / entity / version / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"🧾 Metadata guardada en: {metadata_path}")


def validate_schema(df: pd.DataFrame, expected_schema: dict):
    errors = []
    for col, expected_dtype in expected_schema.items():
        if col not in df.columns:
            errors.append(f"❌ Falta columna: {col}")
        elif str(df[col].dtype) != expected_dtype:
            errors.append(f"❌ Tipo incorrecto en {col}: esperado {expected_dtype}, obtenido {df[col].dtype}")

    if errors:
        raise ValueError("\n".join(errors))

    print("✅ Esquema válido.")