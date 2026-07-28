"""
ilp_solver.py
ILP-based exact scheduler using OR-Tools CP-SAT.

Reads:  data/optimizer_input.json
Writes: output/schedule_ilp.json
"""

import json
import os
import sys
from ortools.sat.python import cp_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# Penalty weights
ALPHA = 1.0    # makespan
BETA  = 1.0    # machine cost
GAMMA = 50.0   # SLA violation per job

# Scaling factors for integer conversion
TIME_SCALE = 100   # seconds → integer ticks
RES_SCALE  = 10    # cpu/ram floats → integers (1 decimal precision)


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

    # pred[j][m] = {pred_duration, pred_cpu, pred_ram}
    pred = [[None] * M for _ in range(N)]
    for p in predictions:
        ji = job_idx.get(p["job_id"])
        mi = machine_idx.get(p["machine_id"])
        if ji is not None and mi is not None:
            pred[ji][mi] = p

    # DAG dependencies
    deps = [[] for _ in range(N)]
    for i, job in enumerate(jobs):
        for dep_id in job.get("dependencies", []):
            if dep_id in job_idx:
                deps[i].append(job_idx[dep_id])

    horizon = int(sum(
        max(pred[j][m]["pred_duration"] for m in range(M) if pred[j][m]) * TIME_SCALE
        for j in range(N)
    )) + 1

    model = cp_model.CpModel()

    # ── Decision variables ────────────────────────────────────────────────────

    # x[j][m] = 1 if job j assigned to machine m
    x = [[model.NewBoolVar(f"x_{j}_{m}") for m in range(M)] for j in range(N)]

    start = [model.NewIntVar(0, horizon, f"start_{j}") for j in range(N)]
    end   = [model.NewIntVar(0, horizon, f"end_{j}")   for j in range(N)]

    makespan = model.NewIntVar(0, horizon, "makespan")

    # ── Core constraints ──────────────────────────────────────────────────────

    # Each job on exactly one machine
    for j in range(N):
        model.Add(sum(x[j][m] for m in range(M)) == 1)

    # end[j] = start[j] + duration of assigned machine
    for j in range(N):
        dur_expr = sum(
            int(pred[j][m]["pred_duration"] * TIME_SCALE) * x[j][m]
            for m in range(M) if pred[j][m]
        )
        model.Add(end[j] == start[j] + dur_expr)

    for i in range(N):
        for d in deps[i]:
            model.Add(start[i] >= end[d])

    for j in range(N):
        model.Add(makespan >= end[j])

    # ── Resource constraints via AddCumulative ────────────────────────────────
    """
    For each machine m, collect optional intervals for jobs that might run on m.
    AddCumulative ensures that at any point in time, the sum of demands of
    active intervals does not exceed the machine capacity.
    
    We use it three times per machine:
    1. Concurrency  — demand=1 per job,     capacity=concurrency
    2. CPU          — demand=pred_cpu,       capacity=cpu_capacity
    3. RAM          — demand=pred_ram,       capacity=ram_capacity
    """
    for m, machine in enumerate(machines):
        conc_cap = machine["concurrency"]
        cpu_cap  = int(machine["cpu_capacity"]  * RES_SCALE)
        ram_cap  = int(machine["ram_capacity"]  * RES_SCALE)

        intervals_m   = []   
        demand_conc   = []   
        demand_cpu    = []   
        demand_ram    = []   

        for j in range(N):
            if not pred[j][m]:
                continue

            dur      = int(pred[j][m]["pred_duration"] * TIME_SCALE)
            req_cpu  = int(pred[j][m]["pred_cpu"]      * RES_SCALE)
            req_ram  = int(pred[j][m]["pred_ram"]       * RES_SCALE)

            # Optional interval: active only when x[j][m] == 1
            iv = model.NewOptionalIntervalVar(
                start[j], dur, end[j], x[j][m], f"iv_{j}_{m}"
            )
            intervals_m.append(iv)
            demand_conc.append(1)
            demand_cpu.append(req_cpu)
            demand_ram.append(req_ram)

        if not intervals_m:
            continue

        model.AddCumulative(intervals_m, demand_conc, conc_cap)

        model.AddCumulative(intervals_m, demand_cpu, cpu_cap)

        model.AddCumulative(intervals_m, demand_ram, ram_cap)

    # ── SLA slack variables (soft constraint) ─────────────────────────────────
    sla_slack = []
    for j, job in enumerate(jobs):
        dl = job.get("deadline", -1)
        if dl and dl > 0:
            slack = model.NewIntVar(0, horizon, f"slack_{j}")
            model.Add(slack >= end[j] - int(dl * TIME_SCALE))
            model.Add(slack >= 0)
            sla_slack.append(slack)

    # ── Objective ─────────────────────────────────────────────────────────────
    # active[m] = 1 if any job assigned to machine m
    active = [model.NewBoolVar(f"active_{m}") for m in range(M)]
    for m in range(M):
        model.AddMaxEquality(active[m], [x[j][m] for j in range(N)])

    COST_SCALE = 1000
    cost_terms = []

    cost_terms.append(int(ALPHA * COST_SCALE) * makespan)

    for m, machine in enumerate(machines):
        activation_cost = int(BETA * machine["cost_per_hour"] * 10 * COST_SCALE)
        cost_terms.append(activation_cost * active[m])

    # SLA penalty
    for slack in sla_slack:
        cost_terms.append(int(GAMMA * COST_SCALE) * slack)

    model.Minimize(sum(cost_terms))

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.log_search_progress = False

    print(f"[ILP] Solving: {N} jobs, {M} machines, time limit={time_limit_sec}s")
    print(f"[ILP] Constraints: DAG ordering + concurrency + CPU capacity + RAM capacity")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("[ILP] No feasible solution found.")
        return None

    print(f"[ILP] Status: {'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'}")

    # ── Extract solution ───────────────────────────────────────────────────────
    schedule    = []
    total_makespan = solver.Value(makespan) / TIME_SCALE
    sla_viol    = 0

    for j, job in enumerate(jobs):
        m_assigned = next(m for m in range(M) if solver.Value(x[j][m]))
        s  = solver.Value(start[j]) / TIME_SCALE
        e  = solver.Value(end[j])   / TIME_SCALE
        dl = job.get("deadline", -1)
        violated = bool(dl and dl > 0 and e > dl)
        if violated:
            sla_viol += 1

        schedule.append({
            "job_id":       job["job_id"],
            "machine_id":   machines[m_assigned]["machine_id"],
            "start_time":   round(s, 4),
            "finish_time":  round(e, 4),
            "deadline":     dl if dl else -1,
            "priority":     job.get("priority", "medium"),
            "sla_violated": violated,
        })

    active_machines = set(e["machine_id"] for e in schedule)
    hours        = total_makespan / 3600.0
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

    print(f"[ILP] Cost={total_cost:.4f}  Makespan={total_makespan:.2f}s  "
          f"SLA violations={sla_viol}")
    print(f"[ILP] Written -> {output_path}")
    return result


if __name__ == "__main__":
    input_path  = os.path.join(DATA_DIR, "optimizer_input.json")
    output_path = os.path.join(OUT_DIR,  "schedule_ilp.json")
    time_limit  = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    solve_ilp(input_path, output_path, time_limit)
