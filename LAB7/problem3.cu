#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

constexpr int N = 1 << 20;

__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

__global__ void vector_add() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

static void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

int main() {
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    check_cuda(cudaMemcpyToSymbol(d_A, h_A.data(), sizeof(float) * N), "cudaMemcpyToSymbol d_A");
    check_cuda(cudaMemcpyToSymbol(d_B, h_B.data(), sizeof(float) * N), "cudaMemcpyToSymbol d_B");

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    vector_add<<<blocks, threads>>>();
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, start, stop), "cudaEventElapsedTime");

    check_cuda(cudaMemcpyFromSymbol(h_C.data(), d_C, sizeof(float) * N), "cudaMemcpyFromSymbol d_C");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

    double memory_clock_hz = static_cast<double>(prop.memoryClockRate) * 1000.0;
    double bus_width_bits = static_cast<double>(prop.memoryBusWidth);
    double theoretical_bw_gbps = (memory_clock_hz * bus_width_bits * 2.0) / 1.0e9;
    double theoretical_bw_gbs = theoretical_bw_gbps / 8.0;

    double total_bytes = static_cast<double>(N) * sizeof(float) * 3.0;
    double seconds = kernel_ms / 1000.0;
    double measured_bw_gbs = (total_bytes / seconds) / 1.0e9;

    std::printf("Kernel time: %.3f ms\n", kernel_ms);
    std::printf("Theoretical bandwidth: %.3f GB/s\n", theoretical_bw_gbs);
    std::printf("Measured bandwidth: %.3f GB/s\n", measured_bw_gbs);

    std::printf("Sample output: C[0]=%.1f C[N-1]=%.1f\n", h_C[0], h_C[N - 1]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
