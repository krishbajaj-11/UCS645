// Exercise 3: Scientific Computing
// 2D Heat Diffusion using Finite Difference Method
// OpenMP Parallelization (C version)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
/* Physical parameters */
#define ALPHA 0.01
#define DX 0.1
#define DY 0.1
#define DT 0.001
#define STABILITY (ALPHA * DT / (DX * DX))

/* Allocate 2D array */
double **alloc_2d(int N) {
    double **arr = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
        arr[i] = (double *)calloc(N, sizeof(double));
    return arr;
}

/* Free 2D array */
void free_2d(double **arr, int N) {
    for (int i = 0; i < N; i++)
        free(arr[i]);
    free(arr);
}

/* Heat diffusion with scheduling */
void heat_diffusion(int N, int time_steps, const char *schedule_type,
                    int num_threads, double *time_taken, double *final_temp) {

    double **temp = alloc_2d(N);
    double **temp_new = alloc_2d(N);

    /* Initialization: hot center */
    int center = N / 2;
    int radius = N / 10;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double dist = sqrt((i - center) * (i - center) +
                               (j - center) * (j - center));
            if (dist < radius)
                temp[i][j] = 100.0;
        }
    }

    double start = omp_get_wtime();

    for (int t = 0; t < time_steps; t++) {

        if (strcmp(schedule_type, "static") == 0) {
            #pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    temp_new[i][j] = temp[i][j] + ALPHA * DT * (
                        (temp[i+1][j] - 2*temp[i][j] + temp[i-1][j]) / (DX*DX) +
                        (temp[i][j+1] - 2*temp[i][j] + temp[i][j-1]) / (DY*DY)
                    );
                }
            }
        }
        else if (strcmp(schedule_type, "dynamic") == 0) {
            #pragma omp parallel for collapse(2) schedule(dynamic,16) num_threads(num_threads)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    temp_new[i][j] = temp[i][j] + ALPHA * DT * (
                        (temp[i+1][j] - 2*temp[i][j] + temp[i-1][j]) / (DX*DX) +
                        (temp[i][j+1] - 2*temp[i][j] + temp[i][j-1]) / (DY*DY)
                    );
                }
            }
        }
        else if (strcmp(schedule_type, "guided") == 0) {
            #pragma omp parallel for collapse(2) schedule(guided) num_threads(num_threads)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    temp_new[i][j] = temp[i][j] + ALPHA * DT * (
                        (temp[i+1][j] - 2*temp[i][j] + temp[i-1][j]) / (DX*DX) +
                        (temp[i][j+1] - 2*temp[i][j] + temp[i][j-1]) / (DY*DY)
                    );
                }
            }
        }

        /* Swap pointers */
        double **tmp = temp;
        temp = temp_new;
        temp_new = tmp;
    }

    double end = omp_get_wtime();
    *time_taken = end - start;

    /* Final average temperature */
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += temp[i][j];

    *final_temp = sum / (N * N);

    free_2d(temp, N);
    free_2d(temp_new, N);
}

/* Cache-blocked version */
void heat_diffusion_blocked(int N, int time_steps, int num_threads,
                            double *time_taken, double *final_temp) {

    const int BLOCK = 32;
    double **temp = alloc_2d(N);
    double **temp_new = alloc_2d(N);

    int center = N / 2;
    int radius = N / 10;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double dist = sqrt((i - center)*(i - center) +
                               (j - center)*(j - center));
            if (dist < radius)
                temp[i][j] = 100.0;
        }
    }

    double start = omp_get_wtime();

    for (int t = 0; t < time_steps; t++) {
        #pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads)
        for (int bi = 1; bi < N-1; bi += BLOCK) {
            for (int bj = 1; bj < N-1; bj += BLOCK) {
                int i_end = (bi + BLOCK < N-1) ? bi + BLOCK : N-1;
                int j_end = (bj + BLOCK < N-1) ? bj + BLOCK : N-1;

                for (int i = bi; i < i_end; i++) {
                    for (int j = bj; j < j_end; j++) {
                        temp_new[i][j] = temp[i][j] + ALPHA * DT * (
                            (temp[i+1][j] - 2*temp[i][j] + temp[i-1][j]) / (DX*DX) +
                            (temp[i][j+1] - 2*temp[i][j] + temp[i][j-1]) / (DY*DY)
                        );
                    }
                }
            }
        }

        double **tmp = temp;
        temp = temp_new;
        temp_new = tmp;
    }

    double end = omp_get_wtime();
    *time_taken = end - start;

    double sum = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += temp[i][j];

    *final_temp = sum / (N * N);

    free_2d(temp, N);
    free_2d(temp_new, N);
}

/* Main */
int main() {
    int N = 512;
    int time_steps = 100;
    int threads[] = {1, 2, 4, 8};
    const char *schedules[] = {"static", "dynamic", "guided"};

    printf("2D Heat Diffusion Simulation (C + OpenMP)\n");
    printf("Grid: %d x %d | Steps: %d\n", N, N, time_steps);
    printf("Stability = %.4f (must be < 0.25)\n\n", STABILITY);

    for (int s = 0; s < 3; s++) {
        printf("=== %s scheduling ===\n", schedules[s]);
        printf("Threads   Time(s)    Speedup   Efficiency   AvgTemp\n");

        double t1 = 0;

        for (int i = 0; i < 4; i++) {
            if (threads[i] > omp_get_max_threads()) continue;

            double time, temp;
            heat_diffusion(N, time_steps, schedules[s],
                           threads[i], &time, &temp);

            if (threads[i] == 1) t1 = time;

            double speedup = t1 / time;
            double eff = (speedup / threads[i]) * 100.0;

            printf("%7d   %8.5f   %7.2fx   %8.2f%%   %.2f\n",
                   threads[i], time, speedup, eff, temp);
        }
        printf("\n");
    }

    return 0;
}
