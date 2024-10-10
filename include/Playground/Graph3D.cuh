#ifndef GRAPH3D_CUH
#define GRAPH3D_CUH

#include <Tri3D.cuh>

// Kernel for setting the points
template <typename Func>
__global__ void setPointsKernel(
    Vec3D *points, Func fXZ,
    Vec3D rangeX, Vec3D rangeZ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rangeX.y || j >= rangeZ.y) return;

    double x = rangeX.x + i * rangeX.z;
    double z = rangeZ.x + j * rangeZ.z;

    points[i * rangeZ.y + j] = Vec3D(x, fXZ(x, z), z);
}

class Graph3D {
public:
    Vec3D rangeX;
    Vec3D rangeZ;

    Vec3D *points;
    Tri3D *tris;

    // Initialize with the function
    Graph3D(
        std::function<double(double, double)> fXZ,
        Vec3D rangeX=Vec3D(-10, 10, 0.1),
        Vec3D rangeZ=Vec3D(-10, 10, 0.1)
    ) {
        this->rangeX = rangeX;
        this->rangeZ = rangeZ;

        // Get the total number of points
        int r = (rangeX.y - rangeX.x) / rangeX.z;
        int c = (rangeZ.y - rangeZ.x) / rangeZ.z;
        int p_count = r * c;

        dim3 block(1, 1);
        dim3 grid(r, c);

        points = new Vec3D[p_count];
    }
};

#endif