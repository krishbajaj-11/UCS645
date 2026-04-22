#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

constexpr int N = 1024;

__global__ void sum_iterative(const int *input, long long *out_sum) {
    __shared__ long long sdata[N];
    int tid = threadIdx.x;
    if (tid < N) {
        sdata[tid] = input[tid];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (int stride = N / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_sum[0] = sdata[0];
    }
}

__global__ void sum_formula(long long *out_sum, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out_sum[1] = static_cast<long long>(n) * (n + 1) / 2;
    }
}

static void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

static long long cpu_iterative_sum(int n) {
    long long total = 0;
    for (int i = 1; i <= n; ++i) {
        total += i;
    }
    return total;
}

static long long cpu_formula_sum(int n) {
    return static_cast<long long>(n) * (n + 1) / 2;
}

int main() {
    int h_input[N];
    for (int i = 0; i < N; ++i) {
        h_input[i] = i + 1;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    long long cpu_iter = cpu_iterative_sum(N);
    long long cpu_formula = cpu_formula_sum(N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    int *d_input = nullptr;
    long long *d_output = nullptr;
    long long h_output[2] = {0, 0};

    check_cuda(cudaMalloc(&d_input, sizeof(int) * N), "cudaMalloc d_input");
    check_cuda(cudaMalloc(&d_output, sizeof(long long) * 2), "cudaMalloc d_output");

    check_cuda(cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice),
               "cudaMemcpy input");

    dim3 block(N);
    dim3 grid(1);
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    sum_iterative<<<grid, block>>>(d_input, d_output);
    sum_formula<<<1, 1>>>(d_output, N);
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float gpu_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&gpu_ms, start, stop), "cudaEventElapsedTime");
    check_cuda(cudaMemcpy(h_output, d_output, sizeof(long long) * 2, cudaMemcpyDeviceToHost),
               "cudaMemcpy output");

    std::printf("CPU iterative sum = %lld\n", cpu_iter);
    std::printf("CPU formula sum   = %lld\n", cpu_formula);
    std::printf("CPU time (ms)      = %.6f\n", cpu_ms);
    std::printf("GPU iterative sum = %lld\n", h_output[0]);
    std::printf("GPU formula sum   = %lld\n", h_output[1]);
    std::printf("GPU time (ms)      = %.6f\n", gpu_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
