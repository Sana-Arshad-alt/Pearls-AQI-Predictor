# training_pipeline.py
"""
Enhanced Training Pipeline for AQI Project
------------------------------------------
✅ Handles NaNs automatically
✅ Scales features
✅ Suppresses LightGBM warnings
✅ Evaluates Ridge, RandomForest, and LightGBM
✅ Saves best model to disk
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# === CONFIG ===
FEATURE_CSV = "features.csv"
MODEL_OUT = "models"
os.makedirs(MODEL_OUT, exist_ok=True)
warnings.filterwarnings("ignore")  # silence minor sklearn warnings

# === LOAD & PREP ===
def load_and_prep(csv_path=FEATURE_CSV, target_col="aqi_reported", horizon_hours=1):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Forward/backward fill missing data
    df = df.ffill().bfill()

    # Drop rows without target
    df = df.dropna(subset=[target_col])

    # Create future target (predict +horizon)
    df[f"target_t+{horizon_hours}h"] = df[target_col].shift(-horizon_hours)
    df = df.dropna(subset=[f"target_t+{horizon_hours}h"])

    y = df[f"target_t+{horizon_hours}h"].values
    drop_cols = ["timestamp", target_col, f"target_t+{horizon_hours}h"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y, df

# === EVALUATION ===
def evaluate_model(model, X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses, r2s = [], [], []

    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)

        maes.append(mean_absolute_error(yte, preds))
        rmses.append(np.sqrt(mean_squared_error(yte, preds)))
        r2s.append(r2_score(yte, preds))

    return {"mae": np.mean(maes), "rmse": np.mean(rmses), "r2": np.mean(r2s)}

# === MAIN ===
def main():
    X, y, df = load_and_prep()
    print(f"✅ Data loaded: {len(df)} samples, {X.shape[1]} features")

    # Base preprocessing pipeline (Imputation + Scaling)
    base_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Define models
    models = {
        "ridge": Pipeline([("prep", base_pipe), ("model", Ridge())]),
        "rf": Pipeline([("prep", base_pipe), ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1))])
    }

    # LightGBM (optional)
    try:
        import lightgbm as lgb
        import warnings
        warnings.filterwarnings("ignore")
        models["lgbm"] = Pipeline([
            ("prep", base_pipe),
            ("model", lgb.LGBMRegressor(n_estimators=500, n_jobs=-1, verbose=-1))
        ])
    except ImportError:
        print("⚠️ LightGBM not installed — skipping")

    # Train & Evaluate
    results = {}
    for name, m in models.items():
        print(f"Evaluating {name}...")
        res = evaluate_model(m, X, y)
        results[name] = res
        print(f"{name}: {res}")

    # Pick best model by RMSE
    best = min(results.items(), key=lambda kv: kv[1]["rmse"])[0]
    print(f"\n🏆 Best model: {best}")

    # Fit best on all data and save
    final_model = models[best]
    final_model.fit(X, y)
    out_path = os.path.join(MODEL_OUT, f"{best}_model.joblib")
    joblib.dump(final_model, out_path)

    print(f"✅ Saved model to {out_path}")
    print("📊 All results:", results)

if __name__ == "__main__":
    main()
