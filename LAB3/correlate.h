#pragma once
/*
 * correlate.h
 * UCS645: Parallel & Distributed Computing - Assignment 3
 */

/*
 * Main interface (uses the fast SIMD+OpenMP version internally).
 *
 * Computes Pearson correlation for all pairs (i, j) where 0 <= j <= i < ny.
 *
 * Parameters:
 *   ny     - number of rows (vectors)
 *   nx     - number of columns (elements per vector)
 *   data   - input matrix, row-major: element at (y,x) = data[x + y*nx]
 *   result - output matrix (ny x ny), result[i + j*ny] = corr(row_i, row_j)
 */
void correlate(int ny, int nx, const float* data, float* result);

// Individual implementations exposed for benchmarking in main.cpp
void correlate_sequential(int ny, int nx, const float* data, float* result);
void correlate_openmp(int ny, int nx, const float* data, float* result);
void correlate_fast(int ny, int nx, const float* data, float* result);