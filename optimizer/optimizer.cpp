/*
 * optimizer.cpp
 * Cloud Job Scheduler — C++ Optimization Engine
 *
 * Implements:
 *   1. DAG parsing + topological sort + critical path
 *   2. Greedy scheduler (critical path first, best machine per cost)
 *   3. Simulated Annealing (move/swap jobs across machines)
 *
 * Input:  optimizer_input.json  (jobs, machines, predictions)
 * Output: schedule.json         (job → machine, start/finish times, cost)
 *
 * Compile:
 *   g++ -std=c++17 -O2 -o optimizer optimizer.cpp
 * Run:
 *   ./optimizer <input_json> <output_json> [sa_iterations] [sa_temp]
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <iomanip>
#include <queue>
#include <cassert>

using namespace std;

// ─── Minimal JSON parser ──────────────────────────────────────────────────────
// We implement a lightweight parser to avoid external dependencies.

struct JsonVal;
using JsonObj  = unordered_map<string, JsonVal*>;
using JsonArr  = vector<JsonVal*>;

struct JsonVal {
    enum Type { STR, NUM, BOOL_, NIL, OBJ, ARR } type;
    string      s;
    double      n = 0;
    bool        b = false;
    JsonObj     obj;
    JsonArr     arr;
    ~JsonVal() {
        for (auto& kv : obj) delete kv.second;
        for (auto  v  : arr) delete v;
    }
};

struct Parser {
    const string& src;
    size_t pos = 0;

    char peek() { skipWS(); return pos < src.size() ? src[pos] : '\0'; }
    char get()  { skipWS(); return pos < src.size() ? src[pos++] : '\0'; }

    void skipWS() {
        while (pos < src.size() && isspace(src[pos])) pos++;
    }

    string parseStr() {
        assert(get() == '"');
        string res;
        while (pos < src.size()) {
            char c = src[pos++];
            if (c == '"') break;
            if (c == '\\') { c = src[pos++]; }
            res += c;
        }
        return res;
    }

    double parseNum() {
        size_t start = pos;
        if (src[pos] == '-') pos++;
        while (pos < src.size() && (isdigit(src[pos]) || src[pos] == '.' || src[pos] == 'e' || src[pos] == 'E' || src[pos] == '+' || src[pos] == '-'))
            pos++;
        return stod(src.substr(start, pos - start));
    }

    JsonVal* parseVal() {
        char c = peek();
        auto* v = new JsonVal();
        if (c == '"') {
            v->type = JsonVal::STR;
            v->s    = parseStr();
        } else if (c == '{') {
            v->type = JsonVal::OBJ;
            get(); // '{'
            while (peek() != '}') {
                string key = parseStr();
                skipWS(); assert(get() == ':');
                v->obj[key] = parseVal();
                skipWS();
                if (peek() == ',') get();
            }
            get(); // '}'
        } else if (c == '[') {
            v->type = JsonVal::ARR;
            get(); // '['
            while (peek() != ']') {
                v->arr.push_back(parseVal());
                skipWS();
                if (peek() == ',') get();
            }
            get(); // ']'
        } else if (c == 't') {
            v->type = JsonVal::BOOL_; v->b = true;  pos += 4;
        } else if (c == 'f') {
            v->type = JsonVal::BOOL_; v->b = false; pos += 5;
        } else if (c == 'n') {
            v->type = JsonVal::NIL; pos += 4;
        } else {
            v->type = JsonVal::NUM;
            v->n    = parseNum();
        }
        return v;
    }
};

JsonVal* parseJson(const string& src) {
    Parser p{src};
    return p.parseVal();
}

// Helper accessors
string   jStr(JsonVal* v)          { return v->s; }
double   jNum(JsonVal* v)          { return v->n; }
int      jInt(JsonVal* v)          { return (int)v->n; }
JsonVal* jGet(JsonVal* v, const string& k) { return v->obj.count(k) ? v->obj.at(k) : nullptr; }

// ─── Data structures ──────────────────────────────────────────────────────────

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
    double         deadline;      // -1 = none
    vector<string> dep_ids;

    // Graph indices
    int            idx;
    vector<int>    deps;          // indices into jobs array
    vector<int>    successors;

    double cp_length = 0;         // critical path length from this node
};

struct Prediction {
    double duration;
    double cpu;
    double ram;
};

// predictions[job_idx][machine_idx]
using PredMatrix = vector<vector<Prediction>>;

// ─── Schedule state ───────────────────────────────────────────────────────────

struct Assignment {
    int    machine_idx;
    double start_time;
    double finish_time;
};

struct Schedule {
    vector<Assignment> assignments;   // one per job
    double makespan   = 0;
    double total_cost = 0;
    int    sla_violations = 0;
};

// ─── Global data ──────────────────────────────────────────────────────────────

vector<Machine> machines;
vector<Job>     jobs;
PredMatrix      preds;

int N, M;

// Priority weights (higher = more important = schedule earlier)
double priorityWeight(const string& p) {
    if (p == "critical") return 4.0;
    if (p == "high")     return 3.0;
    if (p == "medium")   return 2.0;
    return 1.0;
}

// ─── 1. DAG: Topological Sort ─────────────────────────────────────────────────

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
        for (int s : jobs[u].successors) {
            if (--indegree[s] == 0) q.push(s);
        }
    }
    assert((int)order.size() == N && "DAG has a cycle!");
    return order;
}

// ─── 2. Critical Path ─────────────────────────────────────────────────────────
// cp_length[i] = max over all machines of duration[i][m],
//                then propagated forward through successors.
// We use the average duration across machines as the edge weight.

void computeCriticalPath(const vector<int>& topo) {
    // Process in reverse topological order
    for (int i = (int)topo.size() - 1; i >= 0; i--) {
        int u = topo[i];

        // Average duration across machines as job weight
        double avg_dur = 0;
        for (int m = 0; m < M; m++) avg_dur += preds[u][m].duration;
        avg_dur /= M;

        double max_succ = 0;
        for (int s : jobs[u].successors)
            max_succ = max(max_succ, jobs[s].cp_length);

        jobs[u].cp_length = avg_dur + max_succ;
    }
}

// ─── 3. Schedule evaluation ───────────────────────────────────────────────────

Schedule evaluateSchedule(const vector<int>& machine_assign) {
    Schedule sched;
    sched.assignments.resize(N);

    // Compute start/finish times respecting dependencies
    // Process in topological order
    vector<int> topo = topoSort();

    // Track per-machine current available time (simple: ignore concurrency for speed)
    vector<double> machine_avail(M, 0.0);
    vector<double> finish(N, 0.0);

    for (int u : topo) {
        int m = machine_assign[u];

        // Earliest start: after all predecessors finish
        double earliest = 0;
        for (int d : jobs[u].deps)
            earliest = max(earliest, finish[d]);

        // Also after machine is available
        earliest = max(earliest, machine_avail[m]);

        double dur = preds[u][m].duration;
        double fin = earliest + dur;

        sched.assignments[u] = {m, earliest, fin};
        finish[u] = fin;
        machine_avail[m] = fin;  // simplified: sequential per machine

        sched.makespan = max(sched.makespan, fin);

        // SLA check
        if (jobs[u].deadline > 0 && fin > jobs[u].deadline)
            sched.sla_violations++;
    }

    // Cost: sum of (cost_per_hour * makespan/3600) for each active machine
    // Simplified: count active machines × cost × (makespan in hours)
    vector<bool> active(M, false);
    for (int i = 0; i < N; i++) active[machine_assign[i]] = true;

    double hours = sched.makespan / 3600.0;
    for (int m = 0; m < M; m++)
        if (active[m]) sched.total_cost += machines[m].cost_per_hour * hours;

    // Penalty weights
    double alpha = 1.0;   // makespan weight
    double beta  = 1.0;   // machine cost weight
    double gamma = 50.0;  // SLA violation penalty per violation

    sched.total_cost = alpha * sched.makespan
                     + beta  * sched.total_cost
                     + gamma * sched.sla_violations;

    return sched;
}

// ─── 4. Greedy Scheduler ──────────────────────────────────────────────────────

vector<int> greedySchedule() {
    vector<int>  assign(N, 0);
    vector<int>  topo = topoSort();

    // Sort by critical path length DESC × priority weight DESC
    vector<int> sorted_topo = topo;
    sort(sorted_topo.begin(), sorted_topo.end(), [](int a, int b) {
        return jobs[a].cp_length * priorityWeight(jobs[a].priority)
             > jobs[b].cp_length * priorityWeight(jobs[b].priority);
    });

    vector<double> finish(N, 0.0);
    vector<double> machine_avail(M, 0.0);

    // Process in topological order (not sorted — must respect deps)
    for (int u : topo) {
        double earliest = 0;
        for (int d : jobs[u].deps)
            earliest = max(earliest, finish[d]);

        // Pick machine with best score:
        // score = finish_time_on_m + cost_factor
        // Lower is better
        int    best_m     = 0;
        double best_score = numeric_limits<double>::max();

        for (int m = 0; m < M; m++) {
            // Resource check (simplified: just duration feasibility)
            double avail  = max(earliest, machine_avail[m]);
            double fin    = avail + preds[u][m].duration;

            // Normalize cost contribution
            double cost_factor = machines[m].cost_per_hour / 45.0;  // normalize to GPU=1

            // SLA urgency: if critical and deadline tight, prefer fast machine
            double urgency = 1.0;
            if (jobs[u].deadline > 0 && jobs[u].priority == "critical")
                urgency = 0.5;  // halve score → prefer faster machines

            double score = (fin + cost_factor * 10.0) * urgency;

            if (score < best_score) {
                best_score = score;
                best_m     = m;
            }
        }

        assign[u] = best_m;
        double avail = max(earliest, machine_avail[best_m]);
        finish[u]    = avail + preds[u][best_m].duration;
        machine_avail[best_m] = finish[u];
    }

    return assign;
}

// ─── 5. Simulated Annealing ───────────────────────────────────────────────────

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

    vector<int> current  = init_assign;
    vector<int> best_asgn = init_assign;

    Schedule cur_sched   = evaluateSchedule(current);
    Schedule best_sched  = cur_sched;

    double T = T_init;

    for (int iter = 0; iter < max_iter && T > T_min; iter++) {

        vector<int> candidate = current;

        // Choose move: 70% reassign, 30% swap
        if (rnd01(rng) < 0.70) {
            // Reassign: move a random job to a random different machine
            int job = rnd_job(rng);
            int new_m = rnd_mac(rng);
            candidate[job] = new_m;
        } else {
            // Swap: exchange machine assignments of two random jobs
            int j1 = rnd_job(rng);
            int j2 = rnd_job(rng);
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

// ─── 6. JSON output ───────────────────────────────────────────────────────────

void writeOutput(const vector<int>& assign, const string& out_path) {
    Schedule sched = evaluateSchedule(assign);

    ofstream f(out_path);
    f << fixed << setprecision(6);
    f << "{\n";
    f << "  \"total_cost\": "     << sched.total_cost     << ",\n";
    f << "  \"makespan\": "       << sched.makespan       << ",\n";
    f << "  \"sla_violations\": " << sched.sla_violations << ",\n";
    f << "  \"schedule\": [\n";

    for (int i = 0; i < N; i++) {
        const auto& a = sched.assignments[i];
        f << "    {"
          << "\"job_id\": \""      << jobs[i].job_id            << "\", "
          << "\"machine_id\": \""  << machines[a.machine_idx].machine_id << "\", "
          << "\"start_time\": "    << a.start_time              << ", "
          << "\"finish_time\": "   << a.finish_time             << ", "
          << "\"deadline\": "      << jobs[i].deadline          << ", "
          << "\"priority\": \""    << jobs[i].priority          << "\""
          << "}";
        if (i < N - 1) f << ",";
        f << "\n";
    }

    f << "  ]\n}\n";
    f.close();
    cerr << "[OUT] Schedule written to " << out_path << "\n";
}

// ─── 7. Load input ────────────────────────────────────────────────────────────

void loadInput(const string& path) {
    ifstream f(path);
    if (!f.is_open()) {
        cerr << "Error: cannot open " << path << "\n";
        exit(1);
    }
    string src((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
    f.close();

    JsonVal* root = parseJson(src);

    // Load machines
    for (auto* mv : root->obj["machines"]->arr) {
        Machine mc;
        mc.machine_id    = jStr(jGet(mv, "machine_id"));
        mc.cpu_capacity  = jNum(jGet(mv, "cpu_capacity"));
        mc.ram_capacity  = jNum(jGet(mv, "ram_capacity"));
        mc.cost_per_hour = jNum(jGet(mv, "cost_per_hour"));
        mc.concurrency   = jInt(jGet(mv, "concurrency"));
        machines.push_back(mc);
    }
    M = machines.size();

    // Build machine index map
    unordered_map<string, int> machineIdx;
    for (int i = 0; i < M; i++) machineIdx[machines[i].machine_id] = i;

    // Load jobs
    unordered_map<string, int> jobIdx;
    JsonArr& jarr = root->obj["jobs"]->arr;
    N = jarr.size();
    jobs.resize(N);

    for (int i = 0; i < N; i++) {
        auto* jv = jarr[i];
        jobs[i].job_id   = jStr(jGet(jv, "job_id"));
        jobs[i].priority = jStr(jGet(jv, "priority"));
        jobs[i].deadline = jNum(jGet(jv, "deadline"));
        jobs[i].idx      = i;
        jobIdx[jobs[i].job_id] = i;

        auto* deps = jGet(jv, "dependencies");
        if (deps) for (auto* d : deps->arr) jobs[i].dep_ids.push_back(jStr(d));
    }

    // Resolve dependency indices + build successor lists
    for (int i = 0; i < N; i++) {
        for (const auto& dep_id : jobs[i].dep_ids) {
            if (jobIdx.count(dep_id)) {
                int d = jobIdx[dep_id];
                jobs[i].deps.push_back(d);
                jobs[d].successors.push_back(i);
            }
        }
    }

    // Load predictions
    preds.assign(N, vector<Prediction>(M));
    for (auto* pv : root->obj["predictions"]->arr) {
        string jid = jStr(jGet(pv, "job_id"));
        string mid = jStr(jGet(pv, "machine_id"));
        if (!jobIdx.count(jid) || !machineIdx.count(mid)) continue;
        int ji = jobIdx[jid];
        int mi = machineIdx[mid];
        preds[ji][mi].duration = jNum(jGet(pv, "pred_duration"));
        preds[ji][mi].cpu      = jNum(jGet(pv, "pred_cpu"));
        preds[ji][mi].ram      = jNum(jGet(pv, "pred_ram"));
    }

    delete root;

    cerr << "[LOAD] " << N << " jobs, " << M << " machines loaded.\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────

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

    // Critical path
    vector<int> topo = topoSort();
    computeCriticalPath(topo);

    cerr << "[CP] Critical path lengths computed.\n";

    // Greedy
    vector<int> greedy_assign = greedySchedule();
    Schedule    greedy_sched  = evaluateSchedule(greedy_assign);
    cerr << "[GREEDY] Cost: " << fixed << setprecision(4) << greedy_sched.total_cost
         << "  makespan: " << greedy_sched.makespan
         << "s  SLA violations: " << greedy_sched.sla_violations << "\n";

    // SA improvement
    vector<int> sa_assign = simulatedAnnealing(greedy_assign, sa_iter, sa_temp);

    // Write output
    writeOutput(sa_assign, output_path);

    return 0;
}
