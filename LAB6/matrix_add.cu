#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrix_add(const int *a, const int *b, int *c, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < width) {
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int width = 4096;
    const int count = width * width;
    const int block_sizes[] = {2, 4, 8, 16, 32};
    const int list_count = sizeof(block_sizes) / sizeof(block_sizes[0]);
    const int iters = 50;

    std::vector<int> h_a(count, 1);
    std::vector<int> h_b(count, 2);
    std::vector<int> h_c(count, 0);

    int *d_a = nullptr;
    int *d_b = nullptr;
    int *d_c = nullptr;

    cudaMalloc(&d_a, count * sizeof(int));
    cudaMalloc(&d_b, count * sizeof(int));
    cudaMalloc(&d_c, count * sizeof(int));

    cudaMemcpy(d_a, h_a.data(), count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), count * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Block Dim (2D),Grid Dim (2D),Time (ms)" << std::endl;

    for (int i = 0; i < list_count; ++i) {
        int block_dim = block_sizes[i];
        dim3 block(block_dim, block_dim);
        dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);

        matrix_add<<<grid, block>>>(d_a, d_b, d_c, width);
        cudaDeviceSynchronize();

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int it = 0; it < iters; ++it) {
            matrix_add<<<grid, block>>>(d_a, d_b, d_c, width);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= static_cast<float>(iters);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << block_dim << " x " << block_dim << "," << grid.x << " x " << grid.y << "," << ms << std::endl;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);
    matrix_add<<<grid, block>>>(d_a, d_b, d_c, width);

    cudaMemcpy(h_c.data(), d_c, count * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "C[0]: " << h_c[0] << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
