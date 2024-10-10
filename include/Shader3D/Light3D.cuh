#ifndef LIGHT3D_CUH
#define LIGHT3D_CUH

// !!! BETA !!!

#include <Render3D.cuh>

class Light3D {
public:
    Light3D() {}
    ~Light3D() {}

    Vec3D pos;
    Vec3D normal = Vec3D(1, 1, 1);
    double ambient = 0.1;
    double specular = 1.1;
    Vec3D rgbRatio = Vec3D(1, 1, 1);

    Tri2D *D_TRI2DS = new Tri2D[0]; // Device triangles 2D for light;

    float *DEPTH = new float[0];
    __host__ __device__ static Vec2D toLightOrthographic(Vec3D v);

    // Demo
    void demo(Render3D *render);
};

// Kernel for creating shadow map
__global__ void shadowMapKernel(
    float *depths, const Tri2D *tri2Ds, const Tri3D *tri3Ds,
    int d_w, int d_h, size_t size
);

// Kernel for applying lighting
__global__ void lightingKernel(
    Pixel3D *pixels, Light3D light, int size
);

#endif