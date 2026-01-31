#include <stdio.h>
#include <omp.h>
#include <math.h>

#define STEPS 100000000L
#define MAX_THREADS 32
#define NUM_ITERATIONS 3

double delta;

// Sequential baseline
double compute_pi_sequential(long num_steps, double *result) {
    double sum = 0.0, x;
    delta = 1.0 / (double)num_steps;
    
    double start_time = omp_get_wtime();
    
    for (long i = 0; i < num_steps; i++) {
        x = (i + 0.5) * delta;
        sum += 4.0 / (1.0 + x * x);
    }
    
    *result = delta * sum;
    return omp_get_wtime() - start_time;
}

// Parallel version
double compute_pi_parallel(long num_steps, int num_threads, double *result) {
    double sum = 0.0;
    delta = 1.0 / (double)num_steps;
    
    omp_set_num_threads(num_threads);
    double start_time = omp_get_wtime();
    
    #pragma omp parallel
    {
        double x, local_sum = 0.0;
        
        #pragma omp for
        for (long i = 0; i < num_steps; i++) {
            x = (i + 0.5) * delta;
            local_sum += 4.0 / (1.0 + x * x);
        }
        
        #pragma omp critical
        sum += local_sum;
    }
    
    *result = delta * sum;
    return omp_get_wtime() - start_time;
}

int main() {
    printf("\n============================================================\n");
    printf("PI COMPUTATION USING NUMERICAL INTEGRATION\n");
    printf("============================================================\n");
    printf("Formula: π = ∫[0,1] 4/(1+x²) dx\n");
    printf("Integration steps: %ld\n", STEPS);
    printf("Actual π: %.15f\n", M_PI);
    printf("System threads: %d\n", omp_get_max_threads());
    printf("============================================================\n");
    
    // Baseline computation
    printf("\n>>> Computing baseline (sequential)...\n");
    double baseline_time = 0.0;
    double pi_value = 0.0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        baseline_time += compute_pi_sequential(STEPS, &pi_value);
    }
    baseline_time /= NUM_ITERATIONS;
    
    printf("Baseline: %.4f sec | π = %.15f | Error: %.2e\n\n",
           baseline_time, pi_value, fabs(pi_value - M_PI));
    
    printf("Threads | Time(s) | Speedup | Efficiency | Computed π\n");
    printf("--------|---------|---------|------------|-----------------\n");
    
    double best_speedup = 0.0;
    int best_threads = 1;
    
    for (int threads = 1; threads <= MAX_THREADS; threads++) {
        double total_time = 0.0;
        double pi_result = 0.0;
        
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            total_time += compute_pi_parallel(STEPS, threads, &pi_result);
        }
        
        double avg_time = total_time / NUM_ITERATIONS;
        double speedup = baseline_time / avg_time;
        double efficiency = (speedup / threads) * 100.0;
        
        if (speedup > best_speedup) {
            best_speedup = speedup;
            best_threads = threads;
        }
        
        printf("  %2d    | %.4f  | %.3fx   | %.2f%%     | %.15f\n",
               threads, avg_time, speedup, efficiency, pi_result);
    }
    
    printf("--------|---------|---------|------------|-----------------\n");
    printf("\nOptimal: %d threads with %.3fx speedup\n", best_threads, best_speedup);
    
    printf("\n** What happens beyond optimal thread count? **\n");
    printf("- Performance plateaus (diminishing returns)\n");
    printf("- Reasons:\n");
    printf("  1. Physical CPU cores exhausted\n");
    printf("  2. Context switching overhead increases\n");
    printf("  3. Synchronization overhead (critical section)\n");
    printf("  4. Cache contention between threads\n");
    printf("  5. Thread management cost > computational benefit\n");
    
    printf("============================================================\n\n");
    
    return 0;
}