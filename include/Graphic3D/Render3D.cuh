#ifndef RENDER3D_CUH
#define RENDER3D_CUH

#include <Camera3D.cuh>

struct Pixel3D {
    Color3D color;
    Vec3D normal;
    Vec3D world;
    Vec2D screen;
    bool active = false;
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
    Vec3D rgbRatio = Vec3D(1, 1, 1);
};

class Render3D {
public:
    Render3D(Camera3D *camera, int w_w=1600, int w_h=900, int p_s=4);
    ~Render3D();

    void resize(int w_w, int w_h, int p_s);

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
    // Color3D DEFAULT_COLOR = Color3D(25, 40, 90);
    Color3D DEFAULT_COLOR = Color3D(10, 10, 10);

    // Buffer
    int BUFFER_WIDTH;
    int BUFFER_HEIGHT;
    int BUFFER_SIZE;
    Pixel3D *BUFFER;

    // CUDA stuffs
    Pixel3D *D_BUFFER; 
    Tri3D *D_TRI3DS;
    Tri2D *D_TRI2DS;
    const size_t BLOCK_SIZE = 256;

    // BETA!
    LightSrc3D LIGHT{
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

// Kernel for filling every pixel in the buffer with a default color
__global__ void fillBufferKernel(
    Pixel3D *buffer, Color3D color, size_t size
);

// Kernel for converting 3D triangles to 2D triangles
__global__ void tri3DsTo2DsKernel(
    Tri2D *tri2Ds, const Tri3D *tri3Ds,
    Camera3D cam, int p_s, size_t size
);

// Kernel for rasterizing 2D triangles
__global__ void rasterizeKernel(
    // Buffer and triangles
    Pixel3D *pixels, const Tri2D *tri2Ds, const Tri3D *tri3Ds,
    // Add other parameters here for future use
    LightSrc3D light,
    // Buffer size and data size
    int b_w, int b_h, size_t size
);

#endif