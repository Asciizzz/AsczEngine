#ifndef RENDER3D_CUH
#define RENDER3D_CUH

#include <Camera3D.cuh>

struct Pixel3D {
    Color3D color;
    Vec3D normal;
    Vec3D world;
    Vec2D screen;
};

// BETA!!!
struct LightSrc3D {
    Vec3D pos;
    Vec3D normal = Vec3D(0, -1, 0);

    // Keep in mind these values are usually not for the light source
    // but for the surface of the object (in this case, the triangles)
    double ambient = 0.05;
    double specular = 1.1;

    // To determine light color
    /*
    For example, if you want a red light,
    set the rgbRatio to Vec3D(1.2, 0.8, 0.8)
    to reduce green and blue light
    while increasing red light
    */
    Vec3D rgbRatio = Vec3D(1.4, 0.7, 0.7);
};

class Render3D {
public:
    Render3D(Camera3D *camera);
    ~Render3D();

    // Camera
    Camera3D *camera;

    // Window settings
    std::string W_TITLE = "AsczEngine v2.0";
    int W_WIDTH = 1600;
    int W_HEIGHT = 900;
    int W_CENTER_X = W_WIDTH / 2;
    int W_CENTER_Y = W_HEIGHT / 2;

    // Buffer
    int PIXEL_SIZE = 4;
    int BUFFER_SIZE;
    int BUFFER_WIDTH;
    int BUFFER_HEIGHT;
    Pixel3D *BUFFER;

    // CUDA stuffs
    Pixel3D *D_BUFFER; 
    Tri3D *D_TRI3DS;
    Tri2D *D_TRI2DS;

    // BETA!
    LightSrc3D light{
        Vec3D(0, 200, 0),
    };

    // To vec2D
    __host__ __device__ static Vec2D toVec2D(const Camera3D &cam, Vec3D v);

    // The main render function
    void renderGPU(Tri3D *tri3Ds, size_t size);
    void renderCPU(std::vector<Tri3D> tri3Ds); // Not recommended

    // Reset all
    void reset();
};

// Kernel for converting 3D triangles to 2D triangles
__global__ void tri3DsTo2DsKernel(
    Tri2D *tri2Ds, const Tri3D *tri3Ds, Camera3D cam,
    int p_s, size_t size
);

// Kernel for rasterizing 2D triangles
__global__ void rasterizeKernel(
    // Buffer and triangles
    Pixel3D *pixels, const Tri2D *tri2Ds, const Tri3D *tri3Ds,
    // Add other parameters here for future use
    /* Idea:
    - Light source
    */
    LightSrc3D light,

    // Buffer size and data size
    int b_w, int b_h, size_t size
);

#endif