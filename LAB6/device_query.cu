#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max block dimensions: "
              << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x "
              << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max grid dimensions: "
              << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x "
              << prop.maxGridSize[2] << std::endl;
    std::cout << "Global memory (bytes): " << prop.totalGlobalMem << std::endl;
    std::cout << "Shared memory per block (bytes): " << prop.sharedMemPerBlock << std::endl;
    std::cout << "Constant memory (bytes): " << prop.totalConstMem << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;

    return 0;
}
