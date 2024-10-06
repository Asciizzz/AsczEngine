#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>

#include <string>
#include <math.h>
#include <cmath>
#include <map>

#include <stack>
#include <queue>

#include <cuda_runtime.h>
#include <FpsHandle.cuh>

#define CUDA_CHECK(call)\
    {\
        cudaError_t err = call;\
        if (err != cudaSuccess) {\
            std::cerr   << "CUDA Error: " << cudaGetErrorString(err) \
                        << " in " << __FILE__ << " at line " << __LINE__ << std::endl;\
            exit(err);\
        }\
    }

#endif