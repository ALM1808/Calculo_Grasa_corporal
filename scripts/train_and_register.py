# Scripts/train_and_register.py
import os, json, time, pathlib, glob, joblib
import numpy as np
import pandas as pd
import hopsworks

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Config ---
HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT", "GrasaCorporal")
HOPSWORKS_HOST    = os.environ.get("HOPSWORKS_HOST", "c.app.hopsworks.ai")

FG_NAME, FG_VERSION = "user_fat_percentage", 1
FV_NAME, FV_VERSION = "user_fat_percentage_fv", 2   # usamos v2 para incluir target
TARGET = "real_fat_percentage"
MODEL_NAME = "rf_pipeline_fat_percentage"           # consistente con tu app.py

# --- Login ---
def login():
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,   # ojo: sin https://
    )
    return project, project.get_feature_store(), project.get_model_registry()

def get_feature_cols(fg):
    fg_cols = [f.name for f in fg.features]
    drop_cols = {"user_id","email","timestamp","predicted_fat_percentage", TARGET}
    return [c for c in fg_cols if c not in drop_cols]

def ensure_fv_with_target(fs, fg, feature_cols):
    # Crea/recupera una FV que incluya el target si existe
    try:
        fv = fs.get_feature_view(FV_NAME, FV_VERSION)
    except Exception:
        q = fg.select(feature_cols + [TARGET])
        fv = fs.create_feature_view(
            name=FV_NAME, version=FV_VERSION,
            description="FV con target real_fat_percentage",
            query=q
        )
    return fv

def try_train(fv, feature_cols):
    # Entrena solo si hay etiquetas reales disponibles
    df = fv.get_batch_data()
    if TARGET not in df.columns:
        return None

    df = df.dropna(subset=[TARGET]).copy()
    if df.empty:
        return None

    X = df[feature_cols].copy()
    y = df[TARGET].astype(float)

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if pd.api.types.is_object_dtype(X[c])]

    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ], remainder="drop")

    pipe = Pipeline([("pre", pre),
                     ("rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    metrics = {
        "mae": float(mean_absolute_error(yte, pred)),
        "rmse": float(mean_squared_error(yte, pred, squared=False)),
        "r2": float(r2_score(yte, pred)),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "timestamp": time.strftime("%Y-%m-%dT%H%M%SZ"),
    }

    ts = time.strftime("%Y%m%dT%H%M%SZ")
    art_dir = pathlib.Path("artifacts") / ts
    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, art_dir / "model.pkl")
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Requisitos aproximados del entorno de CI
    import numpy, pandas, sklearn, hopsworks as hw, pyarrow, joblib as jb
    reqs = [
        f"numpy=={numpy.__version__}",
        f"pandas=={pandas.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"joblib=={jb.__version__}",
        f"hopsworks=={hw.__version__}",
        f"pyarrow=={pyarrow.__version__}",
    ]
    (art_dir / "requirements.txt").write_text("\n".join(reqs), encoding="utf-8")

    return str(art_dir), metrics

def register_model(mr, artifacts_dir, metrics):
    # El Model Registry solo admite métricas numéricas
    metrics_num = {k: float(v) for k, v in metrics.items() if isinstance(v, (int,float))}
    for k in ("mae","rmse","r2"):
        metrics_num.setdefault(k, -1.0)

    entry = mr.python.create_model(
        name=MODEL_NAME,
        metrics=metrics_num,
        description="RF para % grasa corporal (auto-registro GitHub Actions)",
    )
    entry.save(artifacts_dir)
    print(f"✅ Modelo registrado desde {artifacts_dir}. Versión: {entry.version}")

def main():
    project, fs, mr = login()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    feature_cols = get_feature_cols(fg)

    # Intentar ENTRENAR si hay etiquetas
    fv = ensure_fv_with_target(fs, fg, feature_cols)
    trained = try_train(fv, feature_cols)

    if trained is not None:
        artifacts_dir, metrics = trained
        print("📊 Métricas (test):", metrics)
        register_model(mr, artifacts_dir, metrics)
        return

    # Si NO hay etiquetas, REGISTRAR el modelo estático del repo
    local_model = pathlib.Path("models") / "rf_pipeline.pkl"
    assert local_model.exists(), f"No existe {local_model}"

    ts = time.strftime("%Y%m%dT%H%M%SZ")
    art_dir = pathlib.Path("artifacts_manual") / ts
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "model.pkl").write_bytes(local_model.read_bytes())

    metrics = {"mae": -1.0, "rmse": -1.0, "r2": -1.0}
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    import numpy, pandas, sklearn, hopsworks as hw, pyarrow, joblib as jb
    reqs = [
        f"numpy=={numpy.__version__}",
        f"pandas=={pandas.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"joblib=={jb.__version__}",
        f"hopsworks=={hw.__version__}",
        f"pyarrow=={pyarrow.__version__}",
    ]
    (art_dir / "requirements.txt").write_text("\n".join(reqs), encoding="utf-8")

    register_model(mr, str(art_dir), metrics)

if __name__ == "__main__":
    main()
