/*
 * optimizer.cpp
 * Cloud Job Scheduler — C++ Optimization Engine

 * Input:  optimizer_input.json  (jobs, machines, predictions)
 * Output: schedule.json         (job -> machine, start/finish times, cost)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <iomanip>
#include <queue>
#include <cassert>
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

// --- Data structures ----------------------------------------------------------

struct Machine {
    string machine_id;
    double cpu_capacity;
    double ram_capacity;
    double cost_per_hour;
    int    concurrency;
};

struct Job {
    string         job_id;
    string         priority;
    double         deadline;
    vector<string> dep_ids;

    int            idx;
    vector<int>    deps;
    vector<int>    successors;

    double cp_length = 0;
};

struct Prediction {
    double duration;
    double cpu;
    double ram;
};

using PredMatrix = vector<vector<Prediction>>;x

// --- Schedule state -----------------------------------------------------------

struct Assignment {
    int    machine_idx;
    double start_time;
    double finish_time;
};

struct Schedule {
    vector<Assignment> assignments;
    double makespan       = 0;
    double total_cost     = 0;
    int    sla_violations = 0;
};

// --- Global data --------------------------------------------------------------

vector<Machine> machines;
vector<Job>     jobs;
PredMatrix      preds;

int N, M;

double priorityWeight(const string& p) {
    if (p == "critical") return 4.0;
    if (p == "high")     return 3.0;
    if (p == "medium")   return 2.0;
    return 1.0;
}

// --- 1. DAG: Topological Sort -------------------------------------------------

vector<int> topoSort() {
    vector<int> indegree(N, 0);
    for (int i = 0; i < N; i++)
        for (int d : jobs[i].deps)
            indegree[i]++;

    queue<int> q;
    for (int i = 0; i < N; i++)
        if (indegree[i] == 0) q.push(i);

    vector<int> order;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (int s : jobs[u].successors)
            if (--indegree[s] == 0) q.push(s);
    }
    assert((int)order.size() == N && "DAG has a cycle!");
    return order;
}

// --- 2. Critical Path ---------------------------------------------------------

void computeCriticalPath(const vector<int>& topo) {
    for (int i = (int)topo.size() - 1; i >= 0; i--) {
        int u = topo[i];

        double avg_dur = 0;
        for (int m = 0; m < M; m++) avg_dur += preds[u][m].duration;
        avg_dur /= M;

        double max_succ = 0;
        for (int s : jobs[u].successors)
            max_succ = max(max_succ, jobs[s].cp_length);

        jobs[u].cp_length = avg_dur + max_succ;
    }
}

// --- 3. Resource-aware machine availability -----------------------------------
//   running jobs < concurrency
//   used_cpu + req_cpu <= cpu_capacity
//   used_ram + req_ram <= ram_capacity


struct RunningJob {
    double finish_time;
    double cpu;
    double ram;
};

double machineAvailableAt(const vector<RunningJob>& running,
                           int concurrency,
                           double cpu_cap, double ram_cap,
                           double req_cpu, double req_ram,
                           double earliest_dep) {

    vector<double> candidates = { earliest_dep };
    for (const auto& rj : running)
        if (rj.finish_time > earliest_dep)
            candidates.push_back(rj.finish_time);
    sort(candidates.begin(), candidates.end());

    for (double t : candidates) {
        int    used_slots = 0;
        double used_cpu   = 0.0;
        double used_ram   = 0.0;
        for (const auto& rj : running) {
            if (rj.finish_time > t) {
                used_slots++;
                used_cpu += rj.cpu;
                used_ram += rj.ram;
            }
        }
        if (used_slots  < concurrency   &&
            used_cpu + req_cpu <= cpu_cap &&
            used_ram + req_ram <= ram_cap)
            return t;
    }

    // All slots or resources busy past every finish_time — wait for all to clear
    double latest = earliest_dep;
    for (const auto& rj : running)
        latest = max(latest, rj.finish_time);
    return latest;
}

// --- 4. Schedule evaluation ---------------------------------------------------

Schedule evaluateSchedule(const vector<int>& machine_assign) {
    Schedule sched;
    sched.assignments.resize(N);

    vector<int> topo = topoSort();

    // Per-machine list of currently running jobs (for resource tracking)
    vector<vector<RunningJob>> machine_running(M);
    vector<double> finish(N, 0.0);

    for (int u : topo) {
        int m = machine_assign[u];

        double earliest_dep = 0;
        for (int d : jobs[u].deps)
            earliest_dep = max(earliest_dep, finish[d]);

        double req_cpu = preds[u][m].cpu;
        double req_ram = preds[u][m].ram;

        double start = machineAvailableAt(machine_running[m],
                                          machines[m].concurrency,
                                          machines[m].cpu_capacity,
                                          machines[m].ram_capacity,
                                          req_cpu, req_ram,
                                          earliest_dep);
        double fin = start + preds[u][m].duration;

        sched.assignments[u] = {m, start, fin};
        finish[u] = fin;
        machine_running[m].push_back({fin, req_cpu, req_ram});

        sched.makespan = max(sched.makespan, fin);

        if (jobs[u].deadline > 0 && fin > jobs[u].deadline)
            sched.sla_violations++;
    }

    vector<bool> active(M, false);
    for (int i = 0; i < N; i++) active[machine_assign[i]] = true;

    double hours = sched.makespan / 3600.0;
    for (int m = 0; m < M; m++)
        if (active[m]) sched.total_cost += machines[m].cost_per_hour * hours;

    double alpha = 1.0, beta = 1.0, gamma = 50.0;
    sched.total_cost = alpha * sched.makespan
                     + beta  * sched.total_cost
                     + gamma * sched.sla_violations;

    return sched;
}

// --- 5. Greedy Scheduler ----------------------------------------

vector<int> greedySchedule() {
    vector<int> assign(N, 0);

    vector<int> indegree(N, 0);
    for (int i = 0; i < N; i++)
        for (int d : jobs[i].deps)
            indegree[i]++;

    // Ready queue: max-heap on (cp_length x priorityWeight)
    using T = pair<double, int>;
    priority_queue<T> ready;
    for (int i = 0; i < N; i++)
        if (indegree[i] == 0)
            ready.push({ jobs[i].cp_length * priorityWeight(jobs[i].priority), i });

    vector<double> finish(N, 0.0);
    vector<vector<RunningJob>> machine_running(M);

    while (!ready.empty()) {
        auto [score, u] = ready.top(); ready.pop();

        double earliest_dep = 0;
        for (int d : jobs[u].deps)
            earliest_dep = max(earliest_dep, finish[d]);

        // Pick best machine: lowest (finish_time + cost_factor x cp_weight) x urgency
        int    best_m     = 0;
        double best_score = numeric_limits<double>::max();

        for (int m = 0; m < M; m++) {
            double req_cpu = preds[u][m].cpu;
            double req_ram = preds[u][m].ram;

            double start = machineAvailableAt(machine_running[m],
                                              machines[m].concurrency,
                                              machines[m].cpu_capacity,
                                              machines[m].ram_capacity,
                                              req_cpu, req_ram,
                                              earliest_dep);
            double fin         = start + preds[u][m].duration;
            double cost_factor = machines[m].cost_per_hour / 45.0;

            // High cp_length jobs: reduce cost penalty to prefer speed
            double cp_weight = 1.0 / (1.0 + jobs[u].cp_length);
            double urgency = (jobs[u].deadline > 0)
                 ? 1.0 / jobs[u].priority_weight
                 : 1.0;

            double mscore = (fin + cost_factor * 10.0 * cp_weight) * urgency;
            if (mscore < best_score) { best_score = mscore; best_m = m; }
        }

        assign[u] = best_m;

        double req_cpu = preds[u][best_m].cpu;
        double req_ram = preds[u][best_m].ram;
        double start   = machineAvailableAt(machine_running[best_m],
                                            machines[best_m].concurrency,
                                            machines[best_m].cpu_capacity,
                                            machines[best_m].ram_capacity,
                                            req_cpu, req_ram,
                                            earliest_dep);
        finish[u] = start + preds[u][best_m].duration;
        machine_running[best_m].push_back({finish[u], req_cpu, req_ram});

        for (int s : jobs[u].successors)
            if (--indegree[s] == 0)
                ready.push({ jobs[s].cp_length * priorityWeight(jobs[s].priority), s });
    }

    return assign;
}

// --- 6. Simulated Annealing ---------------------------------------------------

vector<int> simulatedAnnealing(
    vector<int> init_assign,
    int    max_iter  = 50000,
    double T_init    = 500.0,
    double T_min     = 0.1,
    double cool_rate = 0.9995)
{
    mt19937 rng(42);
    uniform_int_distribution<int> rnd_job(0, N - 1);
    uniform_int_distribution<int> rnd_mac(0, M - 1);
    uniform_real_distribution<double> rnd01(0.0, 1.0);

    vector<int> current   = init_assign;
    vector<int> best_asgn = init_assign;

    Schedule cur_sched  = evaluateSchedule(current);
    Schedule best_sched = cur_sched;

    double T = T_init;

    for (int iter = 0; iter < max_iter && T > T_min; iter++) {
        vector<int> candidate = current;

        if (rnd01(rng) < 0.70) {
            candidate[rnd_job(rng)] = rnd_mac(rng);
        } else {
            int j1 = rnd_job(rng), j2 = rnd_job(rng);
            while (j2 == j1) j2 = rnd_job(rng);
            swap(candidate[j1], candidate[j2]);
        }

        Schedule cand_sched = evaluateSchedule(candidate);
        double delta = cand_sched.total_cost - cur_sched.total_cost;

        if (delta < 0 || rnd01(rng) < exp(-delta / T)) {
            current   = candidate;
            cur_sched = cand_sched;

            if (cur_sched.total_cost < best_sched.total_cost) {
                best_asgn  = current;
                best_sched = cur_sched;
            }
        }

        T *= cool_rate;
    }

    cerr << "[SA] Final cost: " << fixed << setprecision(4) << best_sched.total_cost
         << "  makespan: " << best_sched.makespan
         << "s  SLA violations: " << best_sched.sla_violations << "\n";

    return best_asgn;
}

// --- 7. Load input ------------------------------------------------------------

void loadInput(const string& path) {
    ifstream f(path);
    if (!f.is_open()) { cerr << "Error: cannot open " << path << "\n"; exit(1); }

    json root = json::parse(f);
    f.close();

    for (const auto& mv : root["machines"]) {
        Machine mc;
        mc.machine_id    = mv["machine_id"];
        mc.cpu_capacity  = mv["cpu_capacity"];
        mc.ram_capacity  = mv["ram_capacity"];
        mc.cost_per_hour = mv["cost_per_hour"];
        mc.concurrency   = mv["concurrency"];
        machines.push_back(mc);
    }
    M = machines.size();

    unordered_map<string, int> machineIdx;
    for (int i = 0; i < M; i++) machineIdx[machines[i].machine_id] = i;

    unordered_map<string, int> jobIdx;
    N = root["jobs"].size();
    jobs.resize(N);

    for (int i = 0; i < N; i++) {
        const auto& jv   = root["jobs"][i];
        jobs[i].job_id   = jv["job_id"];
        jobs[i].priority = jv["priority"];
        jobs[i].deadline = jv["deadline"];
        jobs[i].idx      = i;
        jobIdx[jobs[i].job_id] = i;
        for (const auto& d : jv["dependencies"])
            jobs[i].dep_ids.push_back(d.get<string>());
    }

    for (int i = 0; i < N; i++) {
        for (const auto& dep_id : jobs[i].dep_ids) {
            if (jobIdx.count(dep_id)) {
                int d = jobIdx[dep_id];
                jobs[i].deps.push_back(d);
                jobs[d].successors.push_back(i);
            }
        }
    }

    preds.assign(N, vector<Prediction>(M));
    for (const auto& pv : root["predictions"]) {
        string jid = pv["job_id"];
        string mid = pv["machine_id"];
        if (!jobIdx.count(jid) || !machineIdx.count(mid)) continue;
        int ji = jobIdx[jid];
        int mi = machineIdx[mid];
        preds[ji][mi].duration = pv["pred_duration"];
        preds[ji][mi].cpu      = pv["pred_cpu"];
        preds[ji][mi].ram      = pv["pred_ram"];
    }

    cerr << "[LOAD] " << N << " jobs, " << M << " machines loaded.\n";
}

// --- 8. Write output ----------------------------------------------------------

void writeOutput(const vector<int>& assign, const string& out_path) {
    Schedule sched = evaluateSchedule(assign);

    json output;
    output["total_cost"]     = sched.total_cost;
    output["makespan"]       = sched.makespan;
    output["sla_violations"] = sched.sla_violations;

    json schedule = json::array();
    for (int i = 0; i < N; i++) {
        const auto& a = sched.assignments[i];
        schedule.push_back({
            {"job_id",      jobs[i].job_id},
            {"machine_id",  machines[a.machine_idx].machine_id},
            {"start_time",  a.start_time},
            {"finish_time", a.finish_time},
            {"deadline",    jobs[i].deadline},
            {"priority",    jobs[i].priority},
        });
    }
    output["schedule"] = schedule;

    ofstream f(out_path);
    f << output.dump(2) << "\n";
    f.close();
    cerr << "[OUT] Schedule written to " << out_path << "\n";
}

// --- Main ---------------------------------------------------------------------

int main(int argc, char* argv[]) {
    string input_path  = "data/optimizer_input.json";
    string output_path = "output/schedule.json";
    int    sa_iter     = 50000;
    double sa_temp     = 500.0;

    if (argc > 1) input_path  = argv[1];
    if (argc > 2) output_path = argv[2];
    if (argc > 3) sa_iter     = stoi(argv[3]);
    if (argc > 4) sa_temp     = stod(argv[4]);

    loadInput(input_path);

    vector<int> topo = topoSort();
    computeCriticalPath(topo);
    cerr << "[CP] Critical path lengths computed.\n";

    vector<int> greedy_assign = greedySchedule();
    Schedule    greedy_sched  = evaluateSchedule(greedy_assign);
    cerr << "[GREEDY] Cost: " << fixed << setprecision(4) << greedy_sched.total_cost
         << "  makespan: " << greedy_sched.makespan
         << "s  SLA violations: " << greedy_sched.sla_violations << "\n";

    vector<int> sa_assign = simulatedAnnealing(greedy_assign, sa_iter, sa_temp);
    writeOutput(sa_assign, output_path);

    return 0;
}
