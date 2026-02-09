// Exercise 2: Bioinformatics - DNA Sequence Alignment (Smith-Waterman)
// Parallel implementation using OpenMP (C version)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* ---------------- SCORING PARAMETERS ---------------- */

#define MATCH     2
#define MISMATCH -1
#define GAP      -2

/* ---------------- UTILITY FUNCTIONS ---------------- */

// Generate random DNA sequence
char* generate_dna(int length, int seed) {
    const char bases[] = {'A', 'C', 'G', 'T'};
    char *dna = (char*)malloc((length + 1) * sizeof(char));

    srand(seed);
    for (int i = 0; i < length; i++) {
        dna[i] = bases[rand() % 4];
    }
    dna[length] = '\0';
    return dna;
}

// Similarity score
static inline int score(char a, char b) {
    return (a == b) ? MATCH : MISMATCH;
}

// Allocate 2D matrix
int** alloc_matrix(int rows, int cols) {
    int **m = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        m[i] = (int*)calloc(cols, sizeof(int));
    }
    return m;
}

// Free 2D matrix
void free_matrix(int **m, int rows) {
    for (int i = 0; i < rows; i++)
        free(m[i]);
    free(m);
}

/* ---------------- WAVEFRONT PARALLEL VERSION ---------------- */

void smith_waterman_wavefront(
    const char *seq1, const char *seq2,
    int num_threads, double *time_taken)
{
    int m = strlen(seq1);
    int n = strlen(seq2);

    int **H = alloc_matrix(m + 1, n + 1);

    double start = omp_get_wtime();

    for (int diag = 1; diag <= m + n - 1; diag++) {
        int start_i = (diag - n + 1 > 1) ? diag - n + 1 : 1;
        int end_i   = (diag < m) ? diag : m;

        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = start_i; i <= end_i; i++) {
            int j = diag - i + 1;
            if (j >= 1 && j <= n) {
                int match = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
                int del   = H[i-1][j] + GAP;
                int ins   = H[i][j-1] + GAP;

                int max = match;
                if (del > max) max = del;
                if (ins > max) max = ins;
                if (max < 0) max = 0;

                H[i][j] = max;
            }
        }
    }

    *time_taken = omp_get_wtime() - start;

    /* Find maximum score (optional correctness step) */
    int max_score = 0;
    for (int i = 0; i <= m; i++)
        for (int j = 0; j <= n; j++)
            if (H[i][j] > max_score)
                max_score = H[i][j];

    free_matrix(H, m + 1);
}

/* ---------------- SIMPLE ROW-WISE PARALLEL VERSION ---------------- */

void smith_waterman_rowwise(
    const char *seq1, const char *seq2,
    int num_threads, double *time_taken)
{
    int m = strlen(seq1);
    int n = strlen(seq2);

    int **H = alloc_matrix(m + 1, n + 1);

    double start = omp_get_wtime();

    for (int i = 1; i <= m; i++) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int j = 1; j <= n; j++) {
            int match = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
            int del   = H[i-1][j] + GAP;
            int ins   = H[i][j-1] + GAP;

            int max = match;
            if (del > max) max = del;
            if (ins > max) max = ins;
            if (max < 0) max = 0;

            H[i][j] = max;
        }
    }

    *time_taken = omp_get_wtime() - start;

    free_matrix(H, m + 1);
}

/* ---------------- MAIN ---------------- */

int main() {
    int seq_length = 6000;

    char *seq1 = generate_dna(seq_length, 42);
    char *seq2 = generate_dna(seq_length, 123);

    /* Insert matching subsequence */
    int match_start = seq_length / 3;
    int match_len = 50;
    for (int i = 0; i < match_len; i++)
        seq2[match_start + i] = seq1[match_start + i];

    printf("DNA Sequence Alignment (Smith-Waterman Algorithm)\n");
    printf("Sequence 1 length: %d\n", seq_length);
    printf("Sequence 2 length: %d\n", seq_length);
    printf("Scoring: MATCH=%d, MISMATCH=%d, GAP=%d\n\n",
           MATCH, MISMATCH, GAP);

    int threads_list[] = {1, 2, 4, 8};
    int nt = sizeof(threads_list) / sizeof(threads_list[0]);

    /* -------- Wavefront -------- */

    printf("=== Wavefront Parallelization (Anti-diagonal) ===\n");
    printf("%-10s %-15s %-15s %s\n",
           "Threads", "Time (s)", "Speedup", "Efficiency");
    printf("-------------------------------------------------------\n");

    double t_serial = 0.0;

    for (int k = 0; k < nt; k++) {
        int threads = threads_list[k];
        if (threads > omp_get_max_threads()) continue;

        double time;
        smith_waterman_wavefront(seq1, seq2, threads, &time);

        if (threads == 1) t_serial = time;

        double speedup = t_serial / time;
        double efficiency = (speedup / threads) * 100.0;

        printf("%-10d %-15.6f %-14.2fx %.1f%%\n",
               threads, time, speedup, efficiency);
    }

    /* -------- Row-wise -------- */

    printf("\n=== Row-wise Parallelization (Limited) ===\n");
    printf("%-10s %-15s %-15s %s\n",
           "Threads", "Time (s)", "Speedup", "Efficiency");
    printf("-------------------------------------------------------\n");

    t_serial = 0.0;

    for (int k = 0; k < nt; k++) {
        int threads = threads_list[k];
        if (threads > omp_get_max_threads()) continue;

        double time;
        smith_waterman_rowwise(seq1, seq2, threads, &time);

        if (threads == 1) t_serial = time;

        double speedup = t_serial / time;
        double efficiency = (speedup / threads) * 100.0;

        printf("%-10d %-15.6f %-14.2fx %.1f%%\n",
               threads, time, speedup, efficiency);
    }

    printf("\nAlgorithm Analysis:\n");
    printf("1. Dynamic Programming with O(mn) time complexity\n");
    printf("2. Data dependencies restrict full parallelization\n");
    printf("3. Wavefront processes independent anti-diagonals\n");
    printf("4. Each cell depends on top, left, and top-left\n");

    printf("\nScheduling Strategy:\n");
    printf("- Dynamic scheduling balances varying diagonal lengths\n");
    printf("- Static scheduling sufficient for row-wise approach\n");

    free(seq1);
    free(seq2);
    return 0;
}
