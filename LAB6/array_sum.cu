#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void reduce_sum(const float *input, float *block_sums, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) {
        sum = input[idx];
        if (idx + blockDim.x < n) {
            sum += input[idx + blockDim.x];
        }
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int n = 1000000;
    const int threads_list[] = {32, 64, 128, 256, 512, 1024};
    const int list_count = sizeof(threads_list) / sizeof(threads_list[0]);
    const int iters = 100;

    std::vector<float> h_data(n, 1.0f);

    float *d_input = nullptr;
    float *d_block_sums = nullptr;

    int max_blocks = (n + threads_list[0] * 2 - 1) / (threads_list[0] * 2);

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_block_sums, max_blocks * sizeof(float));
    cudaMemcpy(d_input, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Threads per Block,Blocks per Grid,Time (ms)" << std::endl;

    for (int i = 0; i < list_count; ++i) {
        int threads = threads_list[i];
        int blocks = (n + threads * 2 - 1) / (threads * 2);
        size_t shared_bytes = threads * sizeof(float);

        reduce_sum<<<blocks, threads, shared_bytes>>>(d_input, d_block_sums, n);
        cudaDeviceSynchronize();

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int it = 0; it < iters; ++it) {
            reduce_sum<<<blocks, threads, shared_bytes>>>(d_input, d_block_sums, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= static_cast<float>(iters);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << threads << "," << blocks << "," << ms << std::endl;
    }

    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    size_t shared_bytes = threads * sizeof(float);
    reduce_sum<<<blocks, threads, shared_bytes>>>(d_input, d_block_sums, n);

    std::vector<float> h_block_sums(blocks);
    cudaMemcpy(h_block_sums.data(), d_block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (float v : h_block_sums) {
        total += v;
    }

    std::cout << "Sum: " << total << std::endl;

    cudaFree(d_input);
    cudaFree(d_block_sums);

    return 0;
}
