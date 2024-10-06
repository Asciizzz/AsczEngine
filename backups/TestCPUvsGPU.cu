#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include <Vec3D.cuh>

#include <random>

/* Our goal:

- Create 2 vectors of size 1 << 4 of Vec3D objects
- Create kernel to add the vectors
- Add the vectors using the add function in Vec3D
- Print the result
- Profit!

- We dont need the cpu version of the code, we will only use the gpu version

- Keep in mind we will not used <vector> for the entire project, we will use raw pointers

*/

// CUDA error checking macro
#define CUDA_CHECK(call)\
    {\
        cudaError_t err = call;\
        if (err != cudaSuccess) {\
            std::cerr   << "CUDA Error: " << cudaGetErrorString(err) \
                        << " in " << __FILE__ << " at line " << __LINE__ << std::endl;\
            exit(err);\
        }\
    }

// CUDA kernel for vector addition
__global__ void addKernel(const Vec3D* a, const Vec3D* b, Vec3D* c, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = Vec3D::add(a[i], b[i]);
    }
}

int main() {
    const size_t vectorSize = 1 << 24; // 2^24 elements
    Vec3D* a = new Vec3D[vectorSize];
    Vec3D* b = new Vec3D[vectorSize];
    Vec3D* c = new Vec3D[vectorSize];

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1000.0, 1000.0);

    // Initialize vectors
    for (size_t i = 0; i < vectorSize; ++i) {
        a[i] = Vec3D(dis(gen), dis(gen), dis(gen));
        b[i] = Vec3D(dis(gen), dis(gen), dis(gen));
    }

    // Do the operation but for for loop to see how long it takes
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < vectorSize; ++i) {
        c[i] = Vec3D::add(a[i], b[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time (CPU): " << elapsed.count() << "s" << std::endl;

    // Device pointers
    Vec3D* d_a = nullptr;
    Vec3D* d_b = nullptr;
    Vec3D* d_c = nullptr;

    // Allocate memory on device
    CUDA_CHECK(cudaMalloc(&d_a, vectorSize * sizeof(Vec3D)));
    CUDA_CHECK(cudaMalloc(&d_b, vectorSize * sizeof(Vec3D)));
    CUDA_CHECK(cudaMalloc(&d_c, vectorSize * sizeof(Vec3D)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, a, vectorSize * sizeof(Vec3D), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, vectorSize * sizeof(Vec3D), cudaMemcpyHostToDevice));

    // Launch kernel
    const size_t blockSize = 256;
    const size_t numBlocks = (vectorSize + blockSize - 1) / blockSize;

    // Find execution time
    auto gpu_start = std::chrono::high_resolution_clock::now();
    addKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, vectorSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_elapsed = gpu_end - gpu_start;
    std::cout << "Elapsed time (GPU): " << gpu_elapsed.count() << "s" << std::endl;

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(c, d_c, vectorSize * sizeof(Vec3D), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // Completion message
    std::cout << "Something something, done!" << std::endl;

    // Free host memory
    delete[] a;
    delete[] b;
    delete[] c;
}