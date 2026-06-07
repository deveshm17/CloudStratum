"""
main.py
Cloud Job Scheduler — Full Pipeline Orchestrator

Steps:
  1. Generate synthetic data
  2. Train ML models → produce predictions
  3. Run C++ optimizer (Greedy + SA)
  4. (Optional) Run ILP for small instances
  5. Evaluate and compare results
"""

import os
import sys
import json
import subprocess
import time

BASE_DIR  = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE_DIR, "data")
ML_DIR    = os.path.join(BASE_DIR, "ml")
OPT_DIR   = os.path.join(BASE_DIR, "optimizer")
OUT_DIR   = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

CPP_BIN   = os.path.join(OPT_DIR, "optimizer")
CPP_SRC   = os.path.join(OPT_DIR, "optimizer.cpp")

OPT_INPUT  = os.path.join(DATA_DIR, "optimizer_input.json")
SA_OUTPUT  = os.path.join(OUT_DIR,  "schedule_sa.json")
ILP_OUTPUT = os.path.join(OUT_DIR,  "schedule_ilp.json")


# ─── Step helpers ─────────────────────────────────────────────────────────────

def step(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_cmd(cmd: list, cwd=None):
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        sys.exit(1)


# ─── 1. Generate data ─────────────────────────────────────────────────────────

def generate_data():
    step("Step 1: Generating synthetic data")
    run_cmd([sys.executable, os.path.join(DATA_DIR, "synthetic_generator.py")])
    print("  Data generation complete.")


# ─── 2. Train ML models ───────────────────────────────────────────────────────

def train_ml():
    step("Step 2: Training ML models (XGBoost)")
    run_cmd([sys.executable, os.path.join(ML_DIR, "ml_models.py")])
    print("  ML training complete. Predictions written to data/optimizer_input.json")


# ─── 3. Compile C++ optimizer ─────────────────────────────────────────────────

def compile_cpp():
    step("Step 3: Compiling C++ optimizer")
    cmd = ["g++", "-std=c++17", "-O2", "-o", CPP_BIN, CPP_SRC]
    print(f"  Running: {' '.join(cmd)}")
    run_cmd(cmd)
    print("  Compilation successful.")


# ─── 4. Run C++ optimizer (Greedy + SA) ───────────────────────────────────────

def run_optimizer(sa_iter=50000, sa_temp=500.0):
    step("Step 4: Running C++ Greedy + Simulated Annealing optimizer")
    cmd = [CPP_BIN, OPT_INPUT, SA_OUTPUT, str(sa_iter), str(sa_temp)]
    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    run_cmd(cmd)
    elapsed = time.time() - t0
    print(f"  Optimizer finished in {elapsed:.2f}s")


# ─── 5. Run ILP (optional, small N only) ─────────────────────────────────────

def run_ilp(time_limit=60):
    step("Step 5: Running ILP solver (OR-Tools)")
    try:
        from optimizer.ilp_solver import solve_ilp
        solve_ilp(OPT_INPUT, ILP_OUTPUT, time_limit)
    except ImportError:
        print("  OR-Tools not installed. Skipping ILP. Run: pip install ortools")


# ─── 6. Evaluate and compare ──────────────────────────────────────────────────

def evaluate():
    step("Step 6: Evaluation & Comparison")

    with open(OPT_INPUT) as f:
        data = json.load(f)
    jobs     = {j["job_id"]: j for j in data["jobs"]}
    machines = {m["machine_id"]: m for m in data["machines"]}

    def load_schedule(path, label):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            s = json.load(f)
        return s

    sa_result  = load_schedule(SA_OUTPUT,  "Greedy+SA (C++)")
    ilp_result = load_schedule(ILP_OUTPUT, "ILP (OR-Tools)")

    print(f"\n{'Metric':<35} {'Greedy+SA':>15} {'ILP':>15}")
    print("-" * 67)

    def fmt(val, unit=""):
        if val is None: return "N/A"
        if isinstance(val, float): return f"{val:.4f}{unit}"
        return f"{val}{unit}"

    metrics = [
        ("Total Cost",       "total_cost",      ""),
        ("Makespan (s)",     "makespan",        "s"),
        ("SLA Violations",   "sla_violations",  ""),
    ]

    for label, key, unit in metrics:
        sa_val  = sa_result.get(key)  if sa_result  else None
        ilp_val = ilp_result.get(key) if ilp_result else None
        print(f"  {label:<33} {fmt(sa_val, unit):>15} {fmt(ilp_val, unit):>15}")

    # Per-machine utilization from SA result
    if sa_result:
        print(f"\n{'─'*40}")
        print("  Machine Utilization (SA solution):")
        machine_loads = {}
        makespan = sa_result["makespan"]
        for entry in sa_result["schedule"]:
            mid = entry["machine_id"]
            dur = entry["finish_time"] - entry["start_time"]
            machine_loads[mid] = machine_loads.get(mid, 0) + dur

        for mid, load in sorted(machine_loads.items()):
            util = (load / makespan * 100) if makespan > 0 else 0
            print(f"    {mid:<12} utilization: {util:5.1f}%  ({load:.1f}s / {makespan:.1f}s)")

    # SLA detail from SA
    if sa_result:
        print(f"\n{'─'*40}")
        print("  SLA Status (SA solution):")
        violated = []
        for entry in sa_result["schedule"]:
            dl = entry.get("deadline", -1)
            if dl and dl > 0 and entry["finish_time"] > dl:
                violated.append(entry)

        if not violated:
            print("    ✓ All SLA deadlines met!")
        else:
            for v in violated:
                overshoot = v["finish_time"] - v["deadline"]
                print(f"    ✗ {v['job_id']:<25} exceeded by {overshoot:.2f}s "
                      f"(priority: {v['priority']})")

    print(f"\n{'='*60}")
    print("  Output files:")
    print(f"    SA schedule  → {SA_OUTPUT}")
    if os.path.exists(ILP_OUTPUT):
        print(f"    ILP schedule → {ILP_OUTPUT}")
    print(f"{'='*60}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cloud Job Scheduler Pipeline")
    parser.add_argument("--skip-data",  action="store_true", help="Skip data generation")
    parser.add_argument("--skip-ml",    action="store_true", help="Skip ML training")
    parser.add_argument("--skip-ilp",   action="store_true", help="Skip ILP solver")
    parser.add_argument("--sa-iter",    type=int,   default=50000, help="SA iterations")
    parser.add_argument("--sa-temp",    type=float, default=500.0, help="SA initial temperature")
    parser.add_argument("--ilp-time",   type=int,   default=60,    help="ILP time limit (seconds)")
    args = parser.parse_args()

    print("\n🚀 Cloud Job Scheduler — Full Pipeline")
    print(f"   SA iterations: {args.sa_iter}  |  SA temperature: {args.sa_temp}")

    if not args.skip_data:
        generate_data()

    if not args.skip_ml:
        train_ml()

    compile_cpp()
    run_optimizer(args.sa_iter, args.sa_temp)

    if not args.skip_ilp:
        run_ilp(args.ilp_time)

    evaluate()


if __name__ == "__main__":
    main()
