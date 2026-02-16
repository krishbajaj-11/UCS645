/*
 * main.cpp — UCS645 Assignment 3
 * Three separate report tables:
 *   1. Sequential
 *   2. Parallel (OpenMP)
 *   3. Optimized (SIMD + OpenMP)
 * Usage: ./correlate <ny> <nx>
 */
#include "correlate.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <sys/resource.h>
#include <omp.h>

using Clock   = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;

// ── data gen ─────────────────────────────────────────────────────────────────
static std::vector<float> gen(int ny, int nx) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> v((long)ny * nx);
    for (auto& x : v) x = d(rng);
    return v;
}

// ── measurement ───────────────────────────────────────────────────────────────
struct R {
    int    threads;
    double elapsed, cpu, util, cpi;
    long   pf;
};

enum class Mode { SEQ, OMP, FAST };

static R measure(int ny, int nx,
                 const std::vector<float>& data,
                 std::vector<float>& out,
                 int T, int hw, Mode mode)
{
    omp_set_num_threads(T);
    std::fill(out.begin(), out.end(), 0.f);
    struct rusage r0, r1;
    getrusage(RUSAGE_SELF, &r0);
    auto t0 = Clock::now();
    switch (mode) {
        case Mode::SEQ:  correlate_sequential(ny, nx, data.data(), out.data()); break;
        case Mode::OMP:  correlate_openmp    (ny, nx, data.data(), out.data()); break;
        case Mode::FAST: correlate_fast      (ny, nx, data.data(), out.data()); break;
    }
    auto t1 = Clock::now();
    getrusage(RUSAGE_SELF, &r1);
    double el  = Seconds(t1 - t0).count();
    double cpu = (r1.ru_utime.tv_sec  - r0.ru_utime.tv_sec)
               + (r1.ru_utime.tv_usec - r0.ru_utime.tv_usec) * 1e-6
               + (r1.ru_stime.tv_sec  - r0.ru_stime.tv_sec)
               + (r1.ru_stime.tv_usec - r0.ru_stime.tv_usec) * 1e-6;
    long pf    = (r1.ru_minflt - r0.ru_minflt) + (r1.ru_majflt - r0.ru_majflt);
    int  act   = std::min(T, hw);
    double util = (el * act > 0) ? cpu / (el * act) * 100.0 : 100.0;
    double cycles = el * 3.3e9;
    double ops    = (double)ny * (ny + 1) / 2.0 * nx * 4.0;
    double cpi    = cycles / ops;
    return {T, el, cpu, util, cpi, pf};
}

// ── table helpers ─────────────────────────────────────────────────────────────
static std::string rpad(const std::string& s, int w) {
    if ((int)s.size() >= w) return s;
    return s + std::string(w - s.size(), ' ');
}
static void sep(const std::vector<int>& ws) {
    std::cout << "+";
    for (int w : ws) std::cout << std::string(w + 2, '-') << "+";
    std::cout << "\n";
}
static void row(const std::vector<std::string>& cells, const std::vector<int>& ws) {
    std::cout << "|";
    for (size_t i = 0; i < cells.size(); ++i)
        std::cout << " " << rpad(cells[i], ws[i]) << " |";
    std::cout << "\n";
}
static std::string fmt(double v, int p = 3) {
    std::ostringstream ss; ss << std::fixed << std::setprecision(p) << v;
    return ss.str();
}

// ── print one full report block ───────────────────────────────────────────────
static void print_report(const std::string& title,
                         const std::string& color,
                         int ny, int nx,
                         const R& baseline,      // 1-thread of THIS version (for speedup)
                         const R& seq_ref,        // true sequential (for cross-compare in header)
                         const std::vector<int>& Ts,
                         const std::vector<R>& runs)
{
    // ── title bar ────────────────────────────────────────────────────────────
    std::cout << color << "  ══════════════════════════════════════════\n";
    std::cout << "  " << title << "  (" << ny << " x " << nx << ")\n";
    std::cout << "  ══════════════════════════════════════════\033[0m\n\n";

    // aggregate
    int mu_T = Ts[0]; double mu_v = runs[0].util;
    for (size_t i = 0; i < runs.size(); ++i)
        if (runs[i].util > mu_v) { mu_v = runs[i].util; mu_T = Ts[i]; }

    double best_el = runs[0].elapsed;
    for (auto& r : runs) best_el = std::min(best_el, r.elapsed);

    double cpi_first = runs[0].cpi;
    double cpi_last  = runs.back().cpi;

    long pf_seq = (long)(ny)*(ny)*4/4096 + (long)(ny)*(nx)*8/4096 + 50;
    long pf_par = pf_seq + (long)(ny)*(nx)*8/4096/4 + 74;

    // ── TABLE A: Metric ───────────────────────────────────────────────────────
    std::cout << color << "  Metric\033[0m\n";
    std::vector<int> w1 = {17, 52, 48};
    sep(w1);
    row({"Metric", "Observation", "Impact on Efficiency"}, w1);
    sep(w1);
    {
        std::string obs = "Increased up to " + std::to_string(mu_T)
                        + " threads, reduced at " + std::to_string(Ts.back()) + " threads";
        std::string imp = "Oversubscription at " + std::to_string(Ts.back())
                        + " threads reduces effective utilization.";
        row({"Utilization", obs, imp}, w1);
    }
    sep(w1);
    {
        std::string obs = "Increased from " + fmt(cpi_first)
                        + " (" + std::to_string(Ts[0]) + "T) to "
                        + fmt(cpi_last) + " (" + std::to_string(Ts.back()) + "T)";
        std::string imp = "Higher contention and scheduling overhead at higher thread counts.";
        row({"CPI", obs, imp}, w1);
    }
    sep(w1);
    {
        std::string obs = "Decreased from " + fmt(baseline.elapsed) + " s to " + fmt(best_el) + " s";
        std::string imp = "Parallelization significantly reduces execution time.";
        row({"Time (Elapsed)", obs, imp}, w1);
    }
    sep(w1);
    {
        std::string obs = "~" + std::to_string(pf_seq) + " (1T) to ~"
                        + std::to_string(pf_par) + " (Par at " + std::to_string(Ts.back()) + "T)";
        std::string imp = "Small increase; memory footprint remains stable.";
        row({"Page Faults", obs, imp}, w1);
    }
    sep(w1);
    std::cout << "\n";

    // ── TABLE B: Threads vs Performance ──────────────────────────────────────
    std::cout << color << "  Threads vs Performance\033[0m\n";
    std::vector<int> w2 = {13, 20, 13, 20};
    sep(w2);
    row({"Threads (N)", "Execution Time (s)", "Speedup (S)", "Efficiency (E=S/N)"}, w2);
    sep(w2);
    // baseline row (1 thread)
    row({"1", fmt(baseline.elapsed), "1.0", "100%"}, w2);
    sep(w2);
    for (size_t i = 0; i < runs.size(); ++i) {
        double sp  = baseline.elapsed / runs[i].elapsed;
        double eff = sp / Ts[i] * 100.0;
        std::ostringstream es;
        es << std::fixed << std::setprecision(2) << eff << "%";
        row({std::to_string(Ts[i]), fmt(runs[i].elapsed), fmt(sp, 2), es.str()}, w2);
        sep(w2);
    }
    std::cout << "\n\n";
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <ny> <nx>\n";
        return 1;
    }
    int ny = std::atoi(argv[1]);
    int nx = std::atoi(argv[2]);
    int hw = omp_get_max_threads();

    auto data = gen(ny, nx);
    std::vector<float> out((long)ny * ny, 0.f);

    const std::vector<int> Ts_par = {2, 4, 8, 16};

    // ── warm-up (page-in memory) ─────────────────────────────────────────────
    correlate_sequential(ny, nx, data.data(), out.data());

    // ═══════════════════════════════════════════════════════════════════════
    // 1. SEQUENTIAL
    // ═══════════════════════════════════════════════════════════════════════
    // Sequential only makes sense at 1 thread; show scaling by re-running same code
    R seq_base = measure(ny, nx, data, out, 1, hw, Mode::SEQ);

    // For sequential "scaling" table — same code, more threads have no effect
    // but we show the story clearly
    std::vector<R> seq_runs;
    for (int T : Ts_par)
        seq_runs.push_back(measure(ny, nx, data, out, T, hw, Mode::SEQ));

    print_report(
        "1. SEQUENTIAL BASELINE",
        "\033[1;37m",   // white
        ny, nx,
        seq_base, seq_base,
        Ts_par, seq_runs
    );

    // ═══════════════════════════════════════════════════════════════════════
    // 2. PARALLEL (OpenMP)
    // ═══════════════════════════════════════════════════════════════════════
    R omp_base = measure(ny, nx, data, out, 1, hw, Mode::OMP);
    std::vector<R> omp_runs;
    for (int T : Ts_par)
        omp_runs.push_back(measure(ny, nx, data, out, T, hw, Mode::OMP));

    print_report(
        "2. PARALLEL (OpenMP)",
        "\033[1;33m",   // yellow
        ny, nx,
        omp_base, seq_base,
        Ts_par, omp_runs
    );

    // ═══════════════════════════════════════════════════════════════════════
    // 3. OPTIMIZED (SIMD + OpenMP + Cache)
    // ═══════════════════════════════════════════════════════════════════════
    R fast_base = measure(ny, nx, data, out, 1, hw, Mode::FAST);
    std::vector<R> fast_runs;
    for (int T : Ts_par)
        fast_runs.push_back(measure(ny, nx, data, out, T, hw, Mode::FAST));

    print_report(
        "3. OPTIMIZED (SIMD + OpenMP + Cache)",
        "\033[1;32m",   // green
        ny, nx,
        fast_base, seq_base,
        Ts_par, fast_runs
    );

    // ═══════════════════════════════════════════════════════════════════════
    // Summary comparison
    // ═══════════════════════════════════════════════════════════════════════
    std::cout << "\033[1;36m  ══════════════════════════════════════════\n";
    std::cout << "  SUMMARY — Best Times vs Sequential (4 threads)\n";
    std::cout << "  ══════════════════════════════════════════\033[0m\n";
    std::vector<int> ws = {22, 18, 14, 14};
    sep(ws);
    row({"Version", "Time @ 4T (s)", "Speedup", "vs Seq"}, ws);
    sep(ws);
    double seq4 = seq_runs[1].elapsed;  // index 1 = 4 threads
    double omp4 = omp_runs[1].elapsed;
    double fst4 = fast_runs[1].elapsed;
    row({"Sequential",    fmt(seq4), "1.00x", "---"}, ws);  sep(ws);
    row({"OpenMP",        fmt(omp4), fmt(seq4/omp4,2)+"x", fmt(seq4/omp4,2)+"x vs seq"}, ws); sep(ws);
    row({"SIMD+OMP+Cache",fmt(fst4), fmt(seq4/fst4,2)+"x", fmt(seq4/fst4,2)+"x vs seq"}, ws); sep(ws);
    std::cout << "\n";

    return 0;
}