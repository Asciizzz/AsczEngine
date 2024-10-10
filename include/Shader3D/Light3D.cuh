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

*/

#include <Render3D.cuh>

class Light3D {
public:
    Light3D();
    ~Light3D();

    double ambient = 0.1;
    double specular = 1.1;
    Vec3D rgbRatio = Vec3D(1, 1, 1);

    Vec3D pos = Vec3D(0, 0, -100);
    Vec3D normal = Vec3D(0, 0, 1);

    Tri2D *D_TRI2DS = new Tri2D[0]; // Device triangles 2D for light;

    float *SHADOW_MAP;
    void initShadowMap(int w, int h);
    __host__ __device__ Vec2D toLightOrthographic(Light3D light, Vec3D v);

    // Demo
    void demo(Render3D *render);
};

// Kernel that convert tri3Ds to tri2Ds
// Literally the same as the one in Render3D
// But this time from the light's orthographic view
__global__ void lightOrthographicKernel(
    Tri2D *tri2Ds, const Tri3D *tri3Ds, size_t size
);

// Kernel for applying lighting
__global__ void lightingKernel(
    Pixel3D *pixels, Light3D light, int size
);

#endif