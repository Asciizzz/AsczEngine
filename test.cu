#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>

// Function to sort array on GPU using Thrust
void sortOnGPU(int *arr, size_t size) {
    // Create device vector and copy data from host to device
    thrust::device_vector<int> d_arr(arr, arr + size);

    // Sort the data on the GPU
    thrust::sort(d_arr.begin(), d_arr.end());

    // Copy data back to host
    thrust::copy(d_arr.begin(), d_arr.end(), arr);
}

// Function to generate a huge random array
void generateRandomArray(int *arr, size_t size) {
    std::random_device rd;  // Seed generator
    std::mt19937 gen(rd()); // Random number generator
    std::uniform_int_distribution<> dis(1, size); // Uniform distribution

    for (size_t i = 0; i < size; ++i) {
        arr[i] = dis(gen);
    }
}

int main() {
    // Define the size of the huge array (e.g., 100 million elements)
    size_t size = 100'000'000;
    
    // Allocate memory for two arrays: one for CPU and one for GPU sorting
    std::vector<int> cpuArray(size);
    std::vector<int> gpuArray(size);

    // Generate a huge random array and copy it to both cpuArray and gpuArray
    generateRandomArray(cpuArray.data(), size);
    std::copy(cpuArray.begin(), cpuArray.end(), gpuArray.begin());

    // Timing CPU sort
    auto startCpu = std::chrono::high_resolution_clock::now();
    std::sort(cpuArray.begin(), cpuArray.end()); // CPU sorting (std::sort)
    auto endCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpuDuration = endCpu - startCpu;
    std::cout << "CPU sort took: " << cpuDuration.count() << " seconds." << std::endl;

    // Timing GPU sort
    auto startGpu = std::chrono::high_resolution_clock::now();
    sortOnGPU(gpuArray.data(), size); // GPU sorting (Thrust)
    auto endGpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpuDuration = endGpu - startGpu;
    std::cout << "GPU sort took: " << gpuDuration.count() << " seconds." << std::endl;

    // Verify both arrays are sorted correctly and match
    bool isEqual = std::equal(cpuArray.begin(), cpuArray.end(), gpuArray.begin());
    if (isEqual) {
        std::cout << "Both CPU and GPU arrays are sorted and identical." << std::endl;
    } else {
        std::cerr << "Error: CPU and GPU results differ!" << std::endl;
    }

    return 0;
}
