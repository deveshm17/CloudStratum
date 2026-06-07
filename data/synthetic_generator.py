"""
synthetic_generator.py
Generates synthetic jobs, machines, DAG dependencies, and historical execution logs.
Writes to data/ as CSV/JSON files consumed by ML and optimizer.
"""

import json
import csv
import random
import math
import os

SEED = 42
random.seed(SEED)

# ─── Config ───────────────────────────────────────────────────────────────────

N_JOBS     = 30       # number of jobs in the DAG
N_MACHINES = 4        # number of machine types
N_HISTORY  = 2000     # historical execution log rows

OUTPUT_DIR = os.path.dirname(__file__)

# ─── Machine definitions ──────────────────────────────────────────────────────

MACHINES = [
    {"machine_id": "GPU",     "cpu_capacity": 96,  "ram_capacity": 384, "cost_per_hour": 45, "concurrency": 8},
    {"machine_id": "CPU_OPT", "cpu_capacity": 64,  "ram_capacity": 128, "cost_per_hour": 18, "concurrency": 16},
    {"machine_id": "MEM_OPT", "cpu_capacity": 32,  "ram_capacity": 512, "cost_per_hour": 22, "concurrency": 12},
    {"machine_id": "CHEAP",   "cpu_capacity": 16,  "ram_capacity": 64,  "cost_per_hour":  8, "concurrency": 20},
]

# Base duration multipliers per machine (GPU fastest, CHEAP slowest)
MACHINE_SPEED = {
    "GPU":     0.3,
    "CPU_OPT": 1.0,
    "MEM_OPT": 0.7,
    "CHEAP":   2.5,
}

# ─── Job type templates ───────────────────────────────────────────────────────

JOB_TYPES = [
    {"type": "AUTH",      "base_duration": 10, "cpu_pct": 40, "ram_gb": 18},
    {"type": "DB_READ",   "base_duration": 20, "cpu_pct": 55, "ram_gb": 30},
    {"type": "ML_INFER",  "base_duration": 60, "cpu_pct": 90, "ram_gb": 48},
    {"type": "TRANSFORM", "base_duration": 15, "cpu_pct": 60, "ram_gb": 24},
    {"type": "AGGREGATE", "base_duration": 12, "cpu_pct": 50, "ram_gb": 20},
    {"type": "NOTIFY",    "base_duration":  5, "cpu_pct": 20, "ram_gb":  8},
]

PRIORITIES = ["critical", "high", "medium", "low"]
PRIORITY_WEIGHTS = [0.1, 0.3, 0.4, 0.2]


def weighted_choice(choices, weights):
    r = random.random()
    cumulative = 0.0
    for choice, weight in zip(choices, weights):
        cumulative += weight
        if r <= cumulative:
            return choice
    return choices[-1]


# ─── Generate jobs ────────────────────────────────────────────────────────────

def generate_jobs(n):
    jobs = []
    for i in range(n):
        jtype = random.choice(JOB_TYPES)
        priority = weighted_choice(PRIORITIES, PRIORITY_WEIGHTS)
        base = jtype["base_duration"]

        # Deadline: critical gets tight, low gets none
        if priority == "critical":
            deadline = int(base * MACHINE_SPEED["GPU"] * 1.5 + 20)
        elif priority == "high":
            deadline = int(base * MACHINE_SPEED["CPU_OPT"] * 1.5 + 30)
        elif priority == "medium":
            deadline = int(base * MACHINE_SPEED["CPU_OPT"] * 2.0 + 40)
        else:
            deadline = -1  # no SLA

        jobs.append({
            "job_id":        f"J{i:03d}_{jtype['type']}",
            "job_type":      jtype["type"],
            "base_duration": base,
            "cpu_pct":       jtype["cpu_pct"],
            "ram_gb":        jtype["ram_gb"],
            "priority":      priority,
            "deadline":      deadline,
            "dependencies":  [],   # filled next
        })
    return jobs


# ─── Generate DAG (layered, no cycles) ───────────────────────────────────────

def generate_dag(jobs):
    n = len(jobs)
    # Assign jobs to layers
    n_layers = max(3, n // 5)
    layer_size = n // n_layers
    layers = []
    idx = 0
    for l in range(n_layers):
        size = layer_size if l < n_layers - 1 else n - idx
        layers.append(list(range(idx, idx + size)))
        idx += size

    # Each job depends on 0-2 jobs from the previous layer
    for l in range(1, len(layers)):
        prev = layers[l - 1]
        for job_idx in layers[l]:
            n_deps = random.randint(0, min(2, len(prev)))
            deps = random.sample(prev, n_deps)
            jobs[job_idx]["dependencies"] = [jobs[d]["job_id"] for d in deps]

    return jobs


# ─── Generate per-job per-machine predicted values ───────────────────────────

def predicted_values(job, machine):
    speed   = MACHINE_SPEED[machine["machine_id"]]
    base    = job["base_duration"]
    noise   = random.uniform(0.85, 1.15)
    duration = round(base * speed * noise, 2)

    # GPU uses more CPU%, MEM_OPT uses more RAM
    cpu_factor = {"GPU": 1.4, "CPU_OPT": 1.0, "MEM_OPT": 0.8, "CHEAP": 0.6}
    ram_factor = {"GPU": 0.9, "CPU_OPT": 0.7, "MEM_OPT": 1.6, "CHEAP": 0.5}

    cpu = round(job["cpu_pct"] * cpu_factor[machine["machine_id"]] * random.uniform(0.9, 1.1), 1)
    ram = round(job["ram_gb"]  * ram_factor[machine["machine_id"]] * random.uniform(0.9, 1.1), 1)

    cpu = min(cpu, 99.0)
    ram = min(ram, machine["ram_capacity"] * 0.95)

    return duration, cpu, ram


# ─── Generate historical logs ─────────────────────────────────────────────────

def generate_history(jobs, n_rows):
    rows = []
    base_ts = 1700000000  # Unix timestamp base
    for i in range(n_rows):
        job     = random.choice(jobs)
        machine = random.choice(MACHINES)
        dur, cpu, ram = predicted_values(job, machine)
        # Add more noise for historical data
        dur  = round(dur  * random.uniform(0.8, 1.25), 2)
        cpu  = round(cpu  * random.uniform(0.8, 1.20), 1)
        ram  = round(ram  * random.uniform(0.8, 1.20), 1)
        cpu  = min(cpu, 99.0)
        ram  = min(ram, machine["ram_capacity"] * 0.95)
        hour = random.randint(0, 23)
        load = random.randint(1, machine["concurrency"])
        rows.append({
            "timestamp":   base_ts + i * 300,
            "job_id":      job["job_id"],
            "job_type":    job["job_type"],
            "machine_id":  machine["machine_id"],
            "duration":    dur,
            "cpu_usage":   cpu,
            "ram_usage":   ram,
            "hour_of_day": hour,
            "concurrent_load": load,
            "base_duration": job["base_duration"],
            "priority":    job["priority"],
        })
    return rows


# ─── Generate predictions table (ML input for optimizer) ─────────────────────

def generate_predictions(jobs):
    """
    For each (job, machine) pair, generate ground-truth predicted values.
    In production this comes from ML models. Here we use the same formula.
    """
    rows = []
    for job in jobs:
        for machine in MACHINES:
            dur, cpu, ram = predicted_values(job, machine)
            rows.append({
                "job_id":     job["job_id"],
                "machine_id": machine["machine_id"],
                "pred_duration": dur,
                "pred_cpu":   cpu,
                "pred_ram":   ram,
            })
    return rows


# ─── Write files ──────────────────────────────────────────────────────────────

def write_csv(rows, filepath, fieldnames):
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows → {filepath}")


def write_json(obj, filepath):
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote → {filepath}")


def main():
    print("Generating synthetic data...")

    jobs = generate_jobs(N_JOBS)
    jobs = generate_dag(jobs)

    # machines.json
    write_json(MACHINES, os.path.join(OUTPUT_DIR, "machines.json"))

    # jobs.json  (includes DAG dependencies)
    write_json(jobs, os.path.join(OUTPUT_DIR, "jobs.json"))

    # history.csv  (ML training data)
    history = generate_history(jobs, N_HISTORY)
    write_csv(history, os.path.join(OUTPUT_DIR, "history.csv"),
              fieldnames=["timestamp","job_id","job_type","machine_id",
                          "duration","cpu_usage","ram_usage",
                          "hour_of_day","concurrent_load","base_duration","priority"])

    # predictions.csv  (fed to optimizer; in prod replaced by ML output)
    preds = generate_predictions(jobs)
    write_csv(preds, os.path.join(OUTPUT_DIR, "predictions.csv"),
              fieldnames=["job_id","machine_id","pred_duration","pred_cpu","pred_ram"])

    # optimizer_input.json  (single file C++ reads)
    opt_input = {
        "machines": MACHINES,
        "jobs": [
            {
                "job_id":       j["job_id"],
                "priority":     j["priority"],
                "deadline":     j["deadline"],
                "dependencies": j["dependencies"],
            }
            for j in jobs
        ],
        "predictions": preds,
    }
    write_json(opt_input, os.path.join(OUTPUT_DIR, "optimizer_input.json"))

    print(f"\nDone. {N_JOBS} jobs, {N_MACHINES} machines, {N_HISTORY} history rows.")
    print(f"DAG edges: {sum(len(j['dependencies']) for j in jobs)}")


if __name__ == "__main__":
    main()
