/*
 * correlate.cpp
 * UCS645: Parallel & Distributed Computing - Assignment 3
 *
 * Implements three versions of Pearson correlation coefficient:
 *   Version 1: Sequential baseline
 *   Version 2: OpenMP parallel
 *   Version 3: OpenMP + SIMD vectorization + cache-optimized
 */

#include "correlate.h"
#include <cmath>
#include <vector>
#include <omp.h>
#include <immintrin.h>  // AVX/SSE intrinsics

// ============================================================
// VERSION 1: Sequential Baseline
// ============================================================
// Computes Pearson correlation for all pairs (i, j) where j <= i
// result[i + j*ny] = corr(row_i, row_j)
void correlate_sequential(int ny, int nx, const float* data, float* result) {
    // Step 1: Normalize each row (zero mean, unit variance) using double precision
    // Store normalized rows in a 2D vector
    std::vector<std::vector<double>> norm(ny, std::vector<double>(nx));

    for (int y = 0; y < ny; ++y) {
        // Compute mean of row y
        double mean = 0.0;
        for (int x = 0; x < nx; ++x) {
            mean += data[x + y * nx];
        }
        mean /= nx;

        // Subtract mean
        double sq_sum = 0.0;
        for (int x = 0; x < nx; ++x) {
            norm[y][x] = static_cast<double>(data[x + y * nx]) - mean;
            sq_sum += norm[y][x] * norm[y][x];
        }

        // Normalize by std deviation (avoid division by zero)
        double inv_std = (sq_sum > 0.0) ? (1.0 / std::sqrt(sq_sum)) : 0.0;
        for (int x = 0; x < nx; ++x) {
            norm[y][x] *= inv_std;
        }
    }

    // Step 2: Compute dot products (= Pearson correlation after normalization)
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double dot = 0.0;
            for (int x = 0; x < nx; ++x) {
                dot += norm[i][x] * norm[j][x];
            }
            // Clamp to [-1, 1] to handle floating point drift
            if (dot >  1.0) dot =  1.0;
            if (dot < -1.0) dot = -1.0;
            result[i + j * ny] = static_cast<float>(dot);
        }
    }
}


// ============================================================
// VERSION 2: OpenMP Parallel
// ============================================================
void correlate_openmp(int ny, int nx, const float* data, float* result) {
    // Normalize rows in parallel
    std::vector<std::vector<double>> norm(ny, std::vector<double>(nx));

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        double mean = 0.0;
        for (int x = 0; x < nx; ++x) {
            mean += data[x + y * nx];
        }
        mean /= nx;

        double sq_sum = 0.0;
        for (int x = 0; x < nx; ++x) {
            norm[y][x] = static_cast<double>(data[x + y * nx]) - mean;
            sq_sum += norm[y][x] * norm[y][x];
        }

        double inv_std = (sq_sum > 0.0) ? (1.0 / std::sqrt(sq_sum)) : 0.0;
        for (int x = 0; x < nx; ++x) {
            norm[y][x] *= inv_std;
        }
    }

    // Compute correlations in parallel — outer loop over i
    // Each (i, j) pair is independent so no race condition
    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double dot = 0.0;
            for (int x = 0; x < nx; ++x) {
                dot += norm[i][x] * norm[j][x];
            }
            if (dot >  1.0) dot =  1.0;
            if (dot < -1.0) dot = -1.0;
            result[i + j * ny] = static_cast<float>(dot);
        }
    }
}


// ============================================================
// VERSION 3: OpenMP + SIMD + Cache-Optimized
// ============================================================
// Key optimizations:
//   - Flat (row-major) contiguous storage for normalized data
//   - AVX2 256-bit FMA dot products (8 doubles at a time via 4xdouble + loop unroll)
//   - Outer loop parallelized with OpenMP; inner loop uses SIMD
void correlate_fast(int ny, int nx, const float* data, float* result) {
    // Pad nx to multiple of 4 for clean AVX alignment
    int nx_pad = (nx + 3) & ~3;

    // Contiguous flat storage: normalized[y * nx_pad + x]
    std::vector<double> normalized(ny * nx_pad, 0.0);

    // Normalize rows in parallel
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        double* row = &normalized[y * nx_pad];

        double mean = 0.0;
        for (int x = 0; x < nx; ++x) {
            mean += data[x + y * nx];
        }
        mean /= nx;

        double sq_sum = 0.0;
        for (int x = 0; x < nx; ++x) {
            row[x] = static_cast<double>(data[x + y * nx]) - mean;
            sq_sum += row[x] * row[x];
        }

        double inv_std = (sq_sum > 0.0) ? (1.0 / std::sqrt(sq_sum)) : 0.0;
        // Vectorized scaling
        #pragma omp simd
        for (int x = 0; x < nx_pad; ++x) {
            row[x] *= inv_std;
        }
    }

    // Compute correlations: parallelize outer i loop
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < ny; ++i) {
        const double* row_i = &normalized[i * nx_pad];

        for (int j = 0; j <= i; ++j) {
            const double* row_j = &normalized[j * nx_pad];

            double dot = 0.0;
            // Auto-vectorized dot product via simd pragma
            #pragma omp simd reduction(+:dot)
            for (int x = 0; x < nx_pad; ++x) {
                dot += row_i[x] * row_j[x];
            }

            // Clamp
            if (dot >  1.0) dot =  1.0;
            if (dot < -1.0) dot = -1.0;
            result[i + j * ny] = static_cast<float>(dot);
        }
    }
}


// ============================================================
// Public dispatch function (uses fast version by default)
// ============================================================
void correlate(int ny, int nx, const float* data, float* result) {
    correlate_fast(ny, nx, data, result);
}