"""
ilp_solver.py
ILP-based exact scheduler using OR-Tools CP-SAT.
Used for small instances (N <= 150) or to validate SA solution quality.

Reads:  data/optimizer_input.json
Writes: output/schedule_ilp.json
"""

import json
import os
import sys
from ortools.sat.python import cp_model

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# Penalty weights (must match optimizer.cpp)
ALPHA = 1.0    # makespan
BETA  = 1.0    # machine cost
GAMMA = 50.0   # SLA violation per job


def solve_ilp(input_path: str, output_path: str, time_limit_sec: int = 60):
    with open(input_path) as f:
        data = json.load(f)

    machines    = data["machines"]
    jobs        = data["jobs"]
    predictions = data["predictions"]

    N = len(jobs)
    M = len(machines)

    machine_idx = {m["machine_id"]: i for i, m in enumerate(machines)}
    job_idx     = {j["job_id"]:     i for i, j in enumerate(jobs)}

    # predictions[j][m] = {duration, cpu, ram}
    pred = [[None] * M for _ in range(N)]
    for p in predictions:
        ji = job_idx.get(p["job_id"])
        mi = machine_idx.get(p["machine_id"])
        if ji is not None and mi is not None:
            pred[ji][mi] = p

    # Build dep graph
    deps = [[] for _ in range(N)]
    for i, job in enumerate(jobs):
        for dep_id in job.get("dependencies", []):
            if dep_id in job_idx:
                deps[i].append(job_idx[dep_id])

    # Scale factor: OR-Tools uses integers, so multiply times by 100
    SCALE = 100

    # Max horizon: sum of max durations
    horizon = int(sum(
        max(pred[j][m]["pred_duration"] for m in range(M) if pred[j][m]) * SCALE
        for j in range(N)
    )) + 1

    model = cp_model.CpModel()

    # ── Decision variables ──
    # x[j][m] = 1 if job j is on machine m
    x = [[model.NewBoolVar(f"x_{j}_{m}") for m in range(M)] for j in range(N)]

    # start[j], end[j] (scaled integers)
    start = [model.NewIntVar(0, horizon, f"start_{j}") for j in range(N)]
    end   = [model.NewIntVar(0, horizon, f"end_{j}")   for j in range(N)]

    # makespan
    makespan = model.NewIntVar(0, horizon, "makespan")

    # ── Constraints ──

    # Each job on exactly one machine
    for j in range(N):
        model.Add(sum(x[j][m] for m in range(M)) == 1)

    # Duration: end[j] = start[j] + pred_duration[j][m] * x[j][m]
    for j in range(N):
        dur_expr = sum(
            int(pred[j][m]["pred_duration"] * SCALE) * x[j][m]
            for m in range(M)
            if pred[j][m]
        )
        model.Add(end[j] == start[j] + dur_expr)

    # DAG ordering: start[i] >= end[dep] for all deps
    for i in range(N):
        for d in deps[i]:
            model.Add(start[i] >= end[d])

    # Makespan >= end[j] for all j
    for j in range(N):
        model.Add(makespan >= end[j])

    # Concurrency limit per machine (simplified: no more than concurrency jobs overlap)
    # For CP-SAT, use interval vars + NoOverlap where concurrency=1
    # For concurrency > 1, we skip (complex cumulative) — approximate here
    intervals_per_machine = [[] for _ in range(M)]
    for j in range(N):
        for m in range(M):
            if not pred[j][m]:
                continue
            dur = int(pred[j][m]["pred_duration"] * SCALE)
            # Optional interval: active only if x[j][m] == 1
            opt_interval = model.NewOptionalIntervalVar(
                start[j], dur, end[j], x[j][m], f"iv_{j}_{m}"
            )
            intervals_per_machine[m].append(opt_interval)

    # For machines with concurrency=1, enforce no overlap
    for m, machine in enumerate(machines):
        if machine["concurrency"] == 1:
            model.AddNoOverlap(intervals_per_machine[m])

    # SLA violations: slack variable (how much job j exceeds its deadline)
    sla_slack = []
    for j, job in enumerate(jobs):
        dl = job.get("deadline", -1)
        if dl and dl > 0:
            slack = model.NewIntVar(0, horizon, f"slack_{j}")
            dl_scaled = int(dl * SCALE)
            # slack = max(0, end[j] - deadline)
            model.Add(slack >= end[j] - dl_scaled)
            model.Add(slack >= 0)
            sla_slack.append(slack)

    # ── Objective ──
    # Minimize: alpha*makespan + beta*machine_cost + gamma*sla_violations
    # Machine cost: cost_per_hour * makespan/3600 * active[m]
    # active[m] = OR of x[j][m] for all j

    active = [model.NewBoolVar(f"active_{m}") for m in range(M)]
    for m in range(M):
        # active[m] = 1 if any job is on machine m
        model.AddMaxEquality(active[m], [x[j][m] for j in range(N)])

    # Cost terms (scaled to integers)
    COST_SCALE = 1000
    cost_terms = []

    # Makespan term
    cost_terms.append(int(ALPHA * COST_SCALE) * makespan)

    # Machine activation cost (approximate: cost_per_hour * makespan_hours)
    # = cost_per_hour / 3600 * makespan_scaled / SCALE
    for m, machine in enumerate(machines):
        cph = machine["cost_per_hour"]
        # Approximate: fixed activation cost proxy
        activation_cost = int(BETA * cph * 10 * COST_SCALE)
        cost_terms.append(activation_cost * active[m])

    # SLA penalty
    for slack in sla_slack:
        cost_terms.append(int(GAMMA * COST_SCALE) * slack)

    model.Minimize(sum(cost_terms))

    # ── Solve ──
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.log_search_progress = False

    print(f"[ILP] Solving with {N} jobs, {M} machines, time limit={time_limit_sec}s...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("[ILP] No feasible solution found.")
        return None

    print(f"[ILP] Status: {'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'}")

    # ── Extract solution ──
    schedule = []
    total_makespan = solver.Value(makespan) / SCALE
    sla_viol = 0

    for j, job in enumerate(jobs):
        m_assigned = next(m for m in range(M) if solver.Value(x[j][m]))
        s = solver.Value(start[j]) / SCALE
        e = solver.Value(end[j])   / SCALE
        dl = job.get("deadline", -1)
        violated = dl and dl > 0 and e > dl
        if violated:
            sla_viol += 1

        schedule.append({
            "job_id":      job["job_id"],
            "machine_id":  machines[m_assigned]["machine_id"],
            "start_time":  round(s, 4),
            "finish_time": round(e, 4),
            "deadline":    dl if dl else -1,
            "priority":    job.get("priority", "medium"),
            "sla_violated": violated,
        })

    # Compute cost
    active_machines = set(s["machine_id"] for s in schedule)
    hours = total_makespan / 3600.0
    machine_cost = sum(
        m["cost_per_hour"] * hours
        for m in machines if m["machine_id"] in active_machines
    )
    total_cost = ALPHA * total_makespan + BETA * machine_cost + GAMMA * sla_viol

    result = {
        "solver":         "ILP (OR-Tools CP-SAT)",
        "total_cost":     round(total_cost, 6),
        "makespan":       round(total_makespan, 6),
        "sla_violations": sla_viol,
        "schedule":       schedule,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[ILP] Cost={total_cost:.4f}  Makespan={total_makespan:.2f}s  SLA violations={sla_viol}")
    print(f"[ILP] Written → {output_path}")
    return result


if __name__ == "__main__":
    input_path  = os.path.join(DATA_DIR, "optimizer_input.json")
    output_path = os.path.join(OUT_DIR,  "schedule_ilp.json")
    time_limit  = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    solve_ilp(input_path, output_path, time_limit)
