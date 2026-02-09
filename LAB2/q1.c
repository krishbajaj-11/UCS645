// Exercise 1: Molecular Dynamics - Lennard-Jones Force Calculation
// C version with OpenMP (converted from C++)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* ---------------- DATA STRUCTURES ---------------- */

typedef struct {
    double x, y, z;     // Position
    double fx, fy, fz;  // Force
} Particle;

/* ---------------- CONSTANTS ---------------- */

#define EPSILON 1.0
#define SIGMA   1.0
#define CUTOFF  (2.5 * SIGMA)

/* ---------------- LENNARD-JONES FORCE ---------------- */

void lj_force(const Particle *pi, const Particle *pj,
              double *fx, double *fy, double *fz, double *potential)
{
    double dx = pi->x - pj->x;
    double dy = pi->y - pj->y;
    double dz = pi->z - pj->z;

    double r2 = dx*dx + dy*dy + dz*dz;

    if (r2 < CUTOFF * CUTOFF && r2 > 0.0) {
        double r = sqrt(r2);
        double r_inv = SIGMA / r;
        double r6_inv = r_inv * r_inv * r_inv;
        r6_inv = r6_inv * r6_inv;
        double r12_inv = r6_inv * r6_inv;

        *potential = 4.0 * EPSILON * (r12_inv - r6_inv);

        double f_mag = 24.0 * EPSILON / r2 *
                       (2.0 * r12_inv - r6_inv);

        *fx = f_mag * dx;
        *fy = f_mag * dy;
        *fz = f_mag * dz;
    } else {
        *fx = *fy = *fz = *potential = 0.0;
    }
}

/* ---------------- MAIN ---------------- */

int main() {
    int N = 1000;
    Particle *particles = (Particle*)malloc(N * sizeof(Particle));

    /* Initialize particles with random positions */
    srand(42);
    for (int i = 0; i < N; i++) {
        particles[i].x = ((double)rand() / RAND_MAX) * 10.0;
        particles[i].y = ((double)rand() / RAND_MAX) * 10.0;
        particles[i].z = ((double)rand() / RAND_MAX) * 10.0;
        particles[i].fx = particles[i].fy = particles[i].fz = 0.0;
    }

    printf("Molecular Dynamics: Lennard-Jones Force Calculation\n");
    printf("Number of particles: %d\n", N);
    printf("Cutoff distance: %.2f\n\n", CUTOFF);

    int thread_counts[] = {1, 2, 4, 8};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("%-10s %-15s %-15s %-15s %s\n",
           "Threads", "Time (s)", "Speedup", "Efficiency", "Total Energy");
    printf("===========================================================================\n");

    double t_serial = 0.0;

    for (int t = 0; t < num_tests; t++) {
        int num_threads = thread_counts[t];
        if (num_threads > omp_get_max_threads()) continue;

        /* Reset forces */
        for (int i = 0; i < N; i++)
            particles[i].fx = particles[i].fy = particles[i].fz = 0.0;

        double total_potential = 0.0;
        double start_time = omp_get_wtime();

        #pragma omp parallel num_threads(num_threads)
        {
            double local_potential = 0.0;

            #pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < N; i++) {
                double fx_i = 0.0, fy_i = 0.0, fz_i = 0.0;

                for (int j = 0; j < N; j++) {
                    if (i != j) {
                        double fx, fy, fz, pot;
                        lj_force(&particles[i], &particles[j],
                                 &fx, &fy, &fz, &pot);

                        fx_i += fx;
                        fy_i += fy;
                        fz_i += fz;
                        local_potential += pot;
                    }
                }

                particles[i].fx = fx_i;
                particles[i].fy = fy_i;
                particles[i].fz = fz_i;
            }

            #pragma omp atomic
            total_potential += local_potential;
        }

        double elapsed = omp_get_wtime() - start_time;
        total_potential /= 2.0;   // correct double counting

        if (num_threads == 1)
            t_serial = elapsed;

        double speedup = t_serial / elapsed;
        double efficiency = (speedup / num_threads) * 100.0;

        printf("%-10d %-15.6f %-14.2fx %-14.1f%% %.6e\n",
               num_threads, elapsed, speedup, efficiency, total_potential);
    }

    printf("===========================================================================\n");

    printf("\nOptimization Techniques Applied:\n");
    printf("1. Dynamic scheduling (chunk=16) for load balancing\n");
    printf("2. Thread-private force accumulation per particle\n");
    printf("3. Atomic reduction for potential energy\n");
    printf("4. Cutoff distance to reduce O(N^2) computation\n");

    printf("\nPerformance Analysis:\n");
    printf("- Force computation complexity: O(N^2)\n");
    printf("- Suitable for shared-memory parallelism using OpenMP\n");
    printf("- Read-heavy memory access pattern\n");

    free(particles);
    return 0;
}
