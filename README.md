# <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Cloud.png" width="35px"> CloudStratam

> **"Smarter cloud scheduling — predict first, optimize second."**

[![C++17](https://img.shields.io/badge/C++17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://isocpp.org)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![OR-Tools](https://img.shields.io/badge/OR--Tools-CP--SAT-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/optimization)
[![NetworkX](https://img.shields.io/badge/NetworkX-DAG-lightgrey?style=for-the-badge)](https://networkx.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()

---

[🧭 Problem](#-the-problem) • [🚀 Features](#-features) • [🧬 How it Works](#-how-it-works) • [🏗 Architecture](#-architecture) • [🛠 Setup](#-setup) • [📊 Results](#-results) 

---

## 🧭 The Problem

Modern cloud platforms run thousands of **jobs** — ML inference, data transforms, API handlers — across heterogeneous machines (GPU, CPU-Optimized, Memory-Optimized). These jobs form **dependency chains**: job B cannot start until job A finishes. This is a **Directed Acyclic Graph (DAG)**.

Traditional schedulers react to current resource snapshots. This causes:

- **Bottlenecks** — critical-path jobs delayed because schedulers don't prioritize them
- **SLA violations** — latency-sensitive jobs placed on wrong machines
- **Resource overload** — no CPU/RAM awareness, machines silently oversubscribed
- **Wasted spend** — expensive machines active even when only light jobs are queued

**CloudStratam** solves this with a 3-layer engine: predict future resource needs with ML, then optimize placement with Greedy → Simulated Annealing → ILP.

---

## 🚀 Features

### 🧠 ML Prediction Layer
- **XGBoost** models trained on historical execution logs
- Predicts **execution duration** and **peak CPU/RAM** per (job, machine) pair
- Replaces static estimates with data-driven inputs to the optimizer
- Features: job type, machine type, time-of-day (cyclical), concurrent load, base duration × machine speed interaction

### ⚙️ Optimization Engine (C++17)
- **Critical Path Method (CPM)** — computes bottleneck jobs via longest DAG chain
- **Ready Queue Greedy** — priority queue ordered by `cp_length × priority_weight`, most critical jobs scheduled first
- **Simulated Annealing** — reassign + swap moves over 50K iterations, Boltzmann acceptance
- **Resource-aware scheduling** — concurrency slots + CPU + RAM enforced simultaneously at every candidate start time

### 🔢 Exact Solver (Python)
- **OR-Tools CP-SAT** MILP for instances with N ≤ 150 jobs
- Full `AddCumulative` constraints for concurrency, CPU capacity, RAM capacity
- DAG dependency ordering enforced as hard constraints
- SLA deadlines as soft penalties in objective

### 🏆 Smart Output Selection
- Always runs SA (fast, any N)
- Runs ILP when N ≤ 150 — compares costs, writes the better result to `schedule_final.json`
- `schedule_final.json` includes `solver` field: which method won and by how much

---

## 🧬 How it Works

```
📂 Historical Execution Logs (history.csv)
              │
              ▼
      🤖 XGBoost Training (Python)
              │   Predicts duration + CPU + RAM
              │   per (job, machine) pair
              ▼
  📄 optimizer_input.json
    (machines + jobs DAG + ML predictions)
              │
       ┌──────┴──────┐
       ▼             ▼
  ⚡ C++ Optimizer   🔢 OR-Tools ILP
  (Greedy + SA)      (exact, N ≤ 150)
       │             │
       ▼             ▼
  schedule_sa.json   schedule_ilp.json
              │
              ▼
      🏆 Winner Selection (main.py)
         Compare total_cost
              │
              ▼
      📋 schedule_final.json
```

The C++ binary and Python scripts communicate through JSON files — no shared memory, no inter-process calls. Every module is independently replaceable.

---

## 🏗 Architecture

| Layer | Component | Language | Responsibility |
|---|---|---|---|
| Data | `synthetic_generator.py` | Python | Jobs, DAG, machines, execution history |
| ML | `ml_models.py` | Python + XGBoost | Duration + CPU/RAM prediction per (job, machine) |
| Optimizer | `optimizer.cpp` | C++17 | Topo sort, CPM, Greedy (ready queue), SA |
| Exact Solver | `ilp_solver.py` | Python + OR-Tools | MILP with full resource constraints |
| Orchestrator | `main.py` | Python | Pipeline, N-based ILP switch, winner selection |

### Optimization Stack

| Method | Handles | Time Complexity | Quality |
|---|---|---|---|
| **Greedy** | Any N — real-time fallback | O(N × M) | Approximate |
| **Simulated Annealing** | Any N — production use | O(iterations) | Near-optimal |
| **ILP (CP-SAT)** | N ≤ 150 — exact benchmark | Exponential (small N only) | Provably optimal |

---

## 🛠 Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (recommended)
make run

# Or step by step
python data/synthetic_generator.py
python ml/ml_models.py
g++ -std=c++17 -O2 -o optimizer/optimizer optimizer/optimizer.cpp
./optimizer/optimizer data/optimizer_input.json output/schedule_sa.json 50000 500.0
python optimizer/ilp_solver.py
```

---

## 📊 Results

Benchmarked on 30 jobs × 4 machine types (GPU, CPU-Opt, Mem-Opt, CHEAP):

| Metric | Greedy (baseline) | Simulated Annealing | ILP (exact) |
|---|---|---|---|
| **Total Cost** | 577.10 | 373.42 | 158.25 |
| **Cost Improvement** | — | −35.3% | −72.6% |
| **Makespan** | 123.9s | 120.3s | 57.5s |
| **SLA Violations** | 9 | 5 | 2 |
| **Solve Time** | < 1ms | ~2s | ~30s |
| **Max Problem Size** | Any N | Any N | N ≤ 150 |

> Numbers from a fixed synthetic benchmark (seed=42, 30 jobs, 4 machines). Relative improvements hold across problem sizes; absolute values vary with input.

---

## 📄 Input / Output Format

### Input (`optimizer_input.json`)
```json
{
  "machines": [
    { "machine_id": "GPU", "cpu_capacity": 96, "ram_capacity": 384,
      "cost_per_hour": 45, "concurrency": 8 }
  ],
  "jobs": [
    { "job_id": "J000_AUTH", "priority": "critical",
      "deadline": 15, "dependencies": [] },
    { "job_id": "J001_ML_INFER", "priority": "high",
      "deadline": 90, "dependencies": ["J000_AUTH"] }
  ],
  "predictions": [
    { "job_id": "J000_AUTH", "machine_id": "GPU",
      "pred_duration": 3.21, "pred_cpu": 88.4, "pred_ram": 22.5 }
  ]
}
```

### Output (`schedule_final.json`)
```json
{
  "solver": "ILP (OR-Tools CP-SAT — exact)",
  "total_cost": 158.25,
  "makespan": 57.5,
  "sla_violations": 2,
  "sa_cost_for_reference": 373.42,
  "schedule": [
    { "job_id": "J000_AUTH", "machine_id": "GPU",
      "start_time": 0.0, "finish_time": 3.21,
      "deadline": 15.0, "priority": "critical" }
  ]
}
```

---

<div align="center">

Built with focus on real-world cloud scheduling constraints.

**CloudStratam — Predict. Optimize. Schedule.** ☁️

</div>
