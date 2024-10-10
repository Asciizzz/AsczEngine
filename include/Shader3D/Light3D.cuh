#ifndef LIGHT3D_CUH
#define LIGHT3D_CUH

// !!! BETA !!!

/* Note: Light3D will share with Render3D the:

- Device buffer
- Device triangles 3D
- Device triangles 2D

// General rendering + shading pipeline:

Create a shadow map from the orthographic view of the light source
(Note: perspective view for spotlight and cube map for point light)
- In this process we will use the D_TRI2DS (device triangles 2D) from Render3D

Start the perspective rendering process (from the camera)
- Tri3Ds to Tri2Ds
- Tri2Ds + Buffer to Buffer

Apply lighting

For every pixel in the buffer:
- Put them on the shadow map by using their 3D world position
- Compare the depth of the pixel with the shadow map
- If the depth is greater, then the pixel is in shadow

// FOR THE TIME BEING:

We will create directional light that face z-positive
We will create the map based on the maxXY and minXY of the 2D triangles
Control the quality with the pixel size

// Common Rule:
For surfaces like wall, floor, terrain...
- shadow=true, lighting=whatever(doesn't matter)
For objects that serves physical/interractive purpose (human, car, etc...)
- shadow=false, lighting=true

*/

#include <Render3D.cuh>

class Light3D {
public:
    Light3D(Render3D *render);
    ~Light3D();
    Render3D *RENDER;

    double ambient = 0.1;
    double specular = 1.1;
    Vec3D rgbRatio = Vec3D(1, 1, 1);

    Vec3D pos = Vec3D(0, 0, -100);
    Vec3D normal = Vec3D(0, 0, 1);

    Tri2D *D_TRI2DS; // Light's own device triangles 2D
    size_t TRI_SIZE;
    void mallocTris(size_t size);
    void freeTris();
    void resizeTris(size_t size);

    float *SHADOW_MAP;
    int SHADOW_MAP_WIDTH;
    int SHADOW_MAP_HEIGHT;
    int SHADOW_MAP_SIZE;
    const size_t SHADOW_MAP_BLOCK_SIZE = 256;
    size_t SHADOW_MAP_BLOCK_COUNT;
    void initShadowMap(int w, int h, int p_s);

    void resetShadowMap();
    void sharedTri2Ds();
    void lighting();
    void shadowMap();
    void applyShadow();
};

// ========================= KERNELS =========================

__global__ void resetShadowMapKernel(
    float *shadowMap, int size
);

// Kernel that convert tri3Ds to tri2Ds
// Literally the same as the one in Render3D
// But this time from the light's orthographic view
__global__ void lightOrthographicKernel(
    Tri2D *tri2Ds, const Tri3D *tri3Ds, int p_s, size_t size
);

// Kernel for applying lighting
__global__ void lightingKernel(
    Pixel3D *pixels, Light3D light, int size
);

// Kernel for light orthographic as well as camera perspective
__global__ void sharedTri2DsKernel(
    Tri2D *tri2Dcam, Tri2D *tri2Dlight, const Tri3D *tri3Ds,
    const Camera3D cam, int p_s, size_t size
);

// Kernel for putting the pixel on the shadow map
__global__ void shadowMapKernel(
    float *shadowMap, Tri2D *tri2Ds, int s_w, int s_h, size_t size
);

// Kernel for applying shadow
__global__ void applyShadowKernel(
    Pixel3D *buffer, float *shadowMap, int s_w, int s_h, int p_s, size_t size
);

#endif