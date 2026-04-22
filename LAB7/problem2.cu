#include <algorithm>
#include <chrono>
#include <cstdio>
#include <future>
#include <random>
#include <vector>
#include <cuda_runtime.h>

constexpr int N = 1000;

static void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void merge_pass(const int *src, int *dst, int n, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * (2 * width);
    if (start >= n) {
        return;
    }

    int mid = start + width;
    int end = start + 2 * width;
    if (mid > n) mid = n;
    if (end > n) end = n;

    int i = start;
    int j = mid;
    int k = start;
    while (i < mid && j < end) {
        if (src[i] <= src[j]) {
            dst[k++] = src[i++];
        } else {
            dst[k++] = src[j++];
        }
    }
    while (i < mid) {
        dst[k++] = src[i++];
    }
    while (j < end) {
        dst[k++] = src[j++];
    }
}

static void parallel_pipeline_sort(std::vector<int> &data) {
    constexpr int chunks = 4;
    int chunk_size = (static_cast<int>(data.size()) + chunks - 1) / chunks;

    std::vector<std::future<void>> sort_futures;
    for (int c = 0; c < chunks; ++c) {
        int start = c * chunk_size;
        int end = std::min(start + chunk_size, static_cast<int>(data.size()));
        if (start >= end) {
            continue;
        }
        sort_futures.emplace_back(std::async(std::launch::async, [start, end, &data]() {
            std::sort(data.begin() + start, data.begin() + end);
        }));
    }
    for (auto &f : sort_futures) {
        f.get();
    }

    std::vector<int> temp(data.size());
    int current_chunks = chunks;
    int current_size = chunk_size;
    while (current_chunks > 1) {
        int pairs = current_chunks / 2;
        std::vector<std::future<void>> merge_futures;
        for (int p = 0; p < pairs; ++p) {
            int start = p * 2 * current_size;
            int mid = std::min(start + current_size, static_cast<int>(data.size()));
            int end = std::min(start + 2 * current_size, static_cast<int>(data.size()));
            merge_futures.emplace_back(std::async(std::launch::async, [start, mid, end, &data, &temp]() {
                std::merge(data.begin() + start, data.begin() + mid,
                           data.begin() + mid, data.begin() + end,
                           temp.begin() + start);
            }));
        }
        for (auto &f : merge_futures) {
            f.get();
        }
        for (int p = 0; p < pairs; ++p) {
            int start = p * 2 * current_size;
            int end = std::min(start + 2 * current_size, static_cast<int>(data.size()));
            std::copy(temp.begin() + start, temp.begin() + end, data.begin() + start);
        }
        if (current_chunks % 2 == 1) {
            int start = pairs * 2 * current_size;
            int end = std::min(start + current_size, static_cast<int>(data.size()));
            std::copy(data.begin() + start, data.begin() + end, temp.begin() + start);
        }
        current_chunks = (current_chunks + 1) / 2;
        current_size *= 2;
    }
}

static void gpu_merge_sort(std::vector<int> &data, float &kernel_ms) {
    int *d_src = nullptr;
    int *d_dst = nullptr;
    check_cuda(cudaMalloc(&d_src, sizeof(int) * data.size()), "cudaMalloc d_src");
    check_cuda(cudaMalloc(&d_dst, sizeof(int) * data.size()), "cudaMalloc d_dst");
    check_cuda(cudaMemcpy(d_src, data.data(), sizeof(int) * data.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy to device");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    int n = static_cast<int>(data.size());
    int width = 1;
    while (width < n) {
        int segments = (n + 2 * width - 1) / (2 * width);
        int threads = 256;
        int blocks = (segments + threads - 1) / threads;
        merge_pass<<<blocks, threads>>>(d_src, d_dst, n, width);
        std::swap(d_src, d_dst);
        width *= 2;
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    check_cuda(cudaEventElapsedTime(&kernel_ms, start, stop), "cudaEventElapsedTime");

    check_cuda(cudaMemcpy(data.data(), d_src, sizeof(int) * data.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy to host");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 10000);

    std::vector<int> data(N);
    for (int &v : data) {
        v = dist(rng);
    }

    std::vector<int> cpu_data = data;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    parallel_pipeline_sort(cpu_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::vector<int> gpu_data = data;
    float gpu_ms = 0.0f;
    gpu_merge_sort(gpu_data, gpu_ms);

    bool cpu_sorted = std::is_sorted(cpu_data.begin(), cpu_data.end());
    bool gpu_sorted = std::is_sorted(gpu_data.begin(), gpu_data.end());

    std::printf("CPU pipeline sort time: %.3f ms (sorted=%s)\n", cpu_ms, cpu_sorted ? "yes" : "no");
    std::printf("GPU merge sort kernel time: %.3f ms (sorted=%s)\n", gpu_ms, gpu_sorted ? "yes" : "no");

    return 0;
}
