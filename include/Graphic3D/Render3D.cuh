#ifndef RENDER3D_CUH
#define RENDER3D_CUH

#include <Camera3D.cuh>

struct Pixel3D {
    Color3D color;
    Vec3D normal;
    Vec3D world;
    Vec2D screen;
    bool shadow = true;
    bool lighting = true;
    long long int meshID = -1;
};

class Render3D {
public:
    Render3D(Camera3D *camera, int w_w=1600, int w_h=900, int p_s=4);
    ~Render3D();

    void resizeWindow(int w_w, int w_h, int p_s);

    // Camera
    Camera3D *CAMERA;

    // Window settings
    std::string W_TITLE = "AsczEngine v2.0";
    int W_WIDTH;
    int W_HEIGHT;
    int W_CENTER_X;
    int W_CENTER_Y;
    int PIXEL_SIZE;

    // Default color
    Color3D DEFAULT_COLOR;

    // Block size and count
    const size_t BLOCK_SIZE = 256;
    size_t BLOCK_TRI_COUNT;
    size_t BLOCK_BUFFER_COUNT;

    // Buffer
    int BUFFER_WIDTH;
    int BUFFER_HEIGHT;
    int BUFFER_SIZE;
    Pixel3D *D_BUFFER; // Device buffer for kernel
    void setBuffer(int w, int h, int p_s);

    Tri3D *D_TRI3DS;
    Tri2D *D_TRI2DS;
    size_t TRI_SIZE;
    void mallocTris(size_t size);
    void memcpyTris(Tri3D *tri3Ds);
    void freeTris();
    void resizeTris(size_t size); // Should only be called if really necessary

    __host__ __device__ static Vec2D toVec2D(const Camera3D &cam, Vec3D v);
    
    // Transformations
    void translateTris(Vec3D t, size_t start, size_t end=0);
    void rotateTris(Vec3D origin, Vec3D w, size_t start, size_t end=0);
    void scaleTris(Vec3D origin, Vec3D s, size_t start, size_t end=0);

    // The pipeline
    void resetBuffer();
    void visibleTriangles();
    void cameraPerspective();
    void rasterize();
};

// ================= KERNELS AND DEVICE FUNCTIONS =================

// Kernel for triangles' transformations
__global__ void translateTri3DKernel(
    Tri3D *tri3Ds, Vec3D t, size_t start, size_t end
);
__global__ void rotateTri3DKernel(
    Tri3D *tri3Ds, Vec3D origin, Vec3D w, size_t start, size_t end
);
__global__ void scaleTri3DKernel(
    Tri3D *tri3Ds, Vec3D center, Vec3D s, size_t start, size_t end
);

// Atomic functions for float
__device__ bool atomicMinFloat(float* addr, float value);
__device__ bool atomicMaxFloat(float* addr, float value);

// Kernel for resetting the buffer
__global__ void resetBufferKernel(
    Pixel3D *buffer, Color3D def_color, size_t size
);

// Kernel for checking if triangles are visible
__global__ void visisbleTrianglesKernel(
    Tri3D *tri3Ds, Camera3D cam, size_t size
);

// Kernel for converting 3D triangles to 2D triangles
__global__ void cameraPerspectivekernel(
    Tri2D *tri2Ds, const Tri3D *tri3Ds,
    Camera3D cam, int p_s, size_t size
);

// Kernel for rasterizing 2D triangles
__global__ void rasterizeKernel(
    Pixel3D *pixels, const Tri2D *tri2Ds, const Tri3D *tri3Ds,
    int b_w, int b_h, size_t size
);

#endif