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
    Vec3D* oldVecs = new Vec3D[vectorSize];
    Vec3D* newVecs = new Vec3D[vectorSize];

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1000.0, 1000.0);

    // Initialize vectors
    for (size_t i = 0; i < vectorSize; ++i) {
        oldVecs[i] = Vec3D(dis(gen), dis(gen), dis(gen));
    }

    // Allocate device memory
    Vec3D* d_oldVecs;
    Vec3D* d_newVecs;
    CUDA_CHECK(cudaMalloc(&d_oldVecs, vectorSize * sizeof(Vec3D)));
    CUDA_CHECK(cudaMalloc(&d_newVecs, vectorSize * sizeof(Vec3D)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_oldVecs, oldVecs, vectorSize * sizeof(Vec3D), cudaMemcpyHostToDevice));

    // Launch kernel
    size_t blockSize = 256;
    size_t numBlocks = (vectorSize + blockSize - 1) / blockSize;
    rotateKernel<<<numBlocks, blockSize>>>(d_newVecs, d_oldVecs, Vec3D(), Vec3D(10, 10, 10), vectorSize);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(newVecs, d_newVecs, vectorSize * sizeof(Vec3D), cudaMemcpyDeviceToHost));

    // Print result
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "Old: " << oldVecs[i].x << " " << oldVecs[i].y << " " << oldVecs[i].z << std::endl;
        std::cout << "New: " << newVecs[i].x << " " << newVecs[i].y << " " << newVecs[i].z << std::endl;
        std::cout << "---\n";
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_oldVecs));
    CUDA_CHECK(cudaFree(d_newVecs));

    // Free host memory
    delete[] oldVecs;
    delete[] newVecs;
}