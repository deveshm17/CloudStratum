"""
ml_models.py
Train XGBoost models to predict:
  1. execution duration per (job_type, machine_id)
  2. cpu_usage and ram_usage per (job_type, machine_id)

Reads:  data/history.csv
Writes: data/predictions.csv  (replaces synthetic predictions with ML predictions)
        ml/models/             (saved model files)
"""

import os
import json
import csv
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "ml", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Feature engineering ──────────────────────────────────────────────────────

MACHINE_SPEED = {"GPU": 0.3, "CPU_OPT": 1.0, "MEM_OPT": 0.7, "CHEAP": 2.5}
PRIORITY_MAP  = {"critical": 4, "high": 3, "medium": 2, "low": 1}

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct feature matrix from raw history rows.
    All categorical columns are label-encoded.
    """
    fe = pd.DataFrame()

    # Encode categoricals
    fe["job_type_enc"]    = LabelEncoder().fit_transform(df["job_type"])
    fe["machine_enc"]     = LabelEncoder().fit_transform(df["machine_id"])
    fe["priority_enc"]    = df["priority"].map(PRIORITY_MAP).fillna(1)

    # Numeric features
    fe["base_duration"]   = df["base_duration"].astype(float)
    fe["hour_of_day"]     = df["hour_of_day"].astype(float)
    fe["concurrent_load"] = df["concurrent_load"].astype(float)

    # Cyclical time encoding (hour as sin/cos)
    fe["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    fe["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    # Machine speed proxy
    fe["machine_speed"] = df["machine_id"].map(MACHINE_SPEED).fillna(1.0)

    # Interaction: base_duration × machine_speed
    fe["dur_speed_interact"] = fe["base_duration"] * fe["machine_speed"]

    return fe


# ─── Train + evaluate one model ───────────────────────────────────────────────

def train_model(X, y, target_name: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = root_mean_squared_error(y_test, preds)

    print(f"  [{target_name}] MAE={mae:.3f}  RMSE={rmse:.3f}")
    return model


# ─── Save / load helpers ──────────────────────────────────────────────────────

def save_model(model, name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model → {path}")

def load_model(name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Predict for all (job, machine) pairs ─────────────────────────────────────

def predict_all(jobs: list, machines: list,
                model_dur, model_cpu, model_ram,
                label_enc_job, label_enc_mac) -> list:
    """
    For every (job, machine) pair, predict duration, cpu, ram.
    Returns list of dicts compatible with predictions.csv format.
    """
    rows = []

    MACHINE_MAP = {m["machine_id"]: m for m in machines}
    MSPEED      = MACHINE_SPEED

    for job in jobs:
        for machine in machines:
            mid = machine["machine_id"]

            # Build a single-row feature dict
            jtype_enc = label_enc_job.transform([job.get("job_type", "AUTH")])[0] \
                        if job.get("job_type") else 0
            mac_enc   = label_enc_mac.transform([mid])[0]
            priority  = PRIORITY_MAP.get(job.get("priority", "medium"), 2)
            base_dur  = float(job.get("base_duration", 20))
            speed     = MSPEED.get(mid, 1.0)

            feat = np.array([[
                jtype_enc,
                mac_enc,
                priority,
                base_dur,
                12.0,          # hour_of_day (noon default)
                4.0,           # concurrent_load (mid default)
                np.sin(2 * np.pi * 12 / 24),
                np.cos(2 * np.pi * 12 / 24),
                speed,
                base_dur * speed,
            ]])

            dur = float(model_dur.predict(feat)[0])
            cpu = float(model_cpu.predict(feat)[0])
            ram = float(model_ram.predict(feat)[0])

            # Clamp to sensible ranges
            dur = max(0.5, dur)
            cpu = min(max(1.0, cpu), 99.0)
            ram = min(max(0.5, ram), machine["ram_capacity"] * 0.95)

            rows.append({
                "job_id":        job["job_id"],
                "machine_id":    mid,
                "pred_duration": round(dur, 2),
                "pred_cpu":      round(cpu, 2),
                "pred_ram":      round(ram, 2),
            })

    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def train_and_predict():
    print("=== ML Training ===")

    # Load history
    history_path = os.path.join(DATA_DIR, "history.csv")
    df = pd.read_csv(history_path)
    print(f"  Loaded {len(df)} history rows.")

    # Feature matrix
    X = build_features(df)

    # Fit label encoders (reuse same fit for prediction)
    le_job = LabelEncoder().fit(df["job_type"])
    le_mac = LabelEncoder().fit(df["machine_id"])
    X["job_type_enc"] = le_job.transform(df["job_type"])
    X["machine_enc"]  = le_mac.transform(df["machine_id"])

    # Targets
    y_dur = df["duration"].astype(float)
    y_cpu = df["cpu_usage"].astype(float)
    y_ram = df["ram_usage"].astype(float)

    print("\nTraining duration model...")
    model_dur = train_model(X, y_dur, "duration")

    print("Training CPU usage model...")
    model_cpu = train_model(X, y_cpu, "cpu_usage")

    print("Training RAM usage model...")
    model_ram = train_model(X, y_ram, "ram_usage")

    # Save models and encoders
    save_model(model_dur, "duration_model")
    save_model(model_cpu, "cpu_model")
    save_model(model_ram, "ram_model")
    save_model(le_job,    "label_enc_job")
    save_model(le_mac,    "label_enc_mac")

    # Load jobs and machines
    with open(os.path.join(DATA_DIR, "jobs.json"))     as f: jobs     = json.load(f)
    with open(os.path.join(DATA_DIR, "machines.json")) as f: machines = json.load(f)

    print(f"\nGenerating ML predictions for {len(jobs)} jobs × {len(machines)} machines...")
    predictions = predict_all(jobs, machines, model_dur, model_cpu, model_ram, le_job, le_mac)

    # Write predictions.csv
    pred_path = os.path.join(DATA_DIR, "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["job_id","machine_id","pred_duration","pred_cpu","pred_ram"])
        w.writeheader()
        w.writerows(predictions)
    print(f"  Wrote {len(predictions)} predictions → {pred_path}")

    # Update optimizer_input.json with ML predictions
    opt_input_path = os.path.join(DATA_DIR, "optimizer_input.json")
    with open(opt_input_path) as f:
        opt_input = json.load(f)
    opt_input["predictions"] = predictions
    with open(opt_input_path, "w") as f:
        json.dump(opt_input, f, indent=2)
    print(f"  Updated optimizer_input.json with ML predictions.")

    print("\n=== ML Training Complete ===")


if __name__ == "__main__":
    train_and_predict()
