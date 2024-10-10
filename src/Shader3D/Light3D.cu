#include <Light3D.cuh>

Light3D::Light3D(Render3D *render) {
    RENDER = render;
}

Light3D::~Light3D() {
    freeTris();
    // CUDA_CHECK(cudaFree(SHADOW_MAP));
}

void Light3D::mallocTris(size_t size) {
    TRI_SIZE = size;
    CUDA_CHECK(cudaMalloc(&D_TRI2DS, size * sizeof(Tri2D)));
}
void Light3D::freeTris() {
    // CUDA_CHECK(cudaFree(D_TRI2DS));
}
void Light3D::resizeTris(size_t size) {
    freeTris();
    mallocTris(size);
}

void Light3D::initShadowMap(int w, int h, int p_s) {
    SHADOW_MAP_WIDTH = w / p_s;
    SHADOW_MAP_HEIGHT = h / p_s;
    SHADOW_MAP_SIZE = SHADOW_MAP_WIDTH * SHADOW_MAP_HEIGHT;
    SHADOW_MAP = new float[SHADOW_MAP_SIZE];
    CUDA_CHECK(cudaMalloc(
        &SHADOW_MAP, SHADOW_MAP_SIZE * sizeof(float))
    );

    SHADOW_MAP_BLOCK_COUNT = (SHADOW_MAP_SIZE + SHADOW_MAP_BLOCK_SIZE - 1) / SHADOW_MAP_BLOCK_SIZE;
}

// Part of the rendering pipeline
void Light3D::resetShadowMap() {
    resetShadowMapKernel<<<SHADOW_MAP_BLOCK_COUNT, SHADOW_MAP_BLOCK_SIZE>>>(
        SHADOW_MAP, SHADOW_MAP_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
void Light3D::sharedTri2Ds() {
    sharedTri2DsKernel<<<RENDER->BLOCK_TRI_COUNT, RENDER->BLOCK_SIZE>>>(
        RENDER->D_TRI2DS, D_TRI2DS, RENDER->D_TRI3DS,
        *RENDER->CAMERA, RENDER->PIXEL_SIZE, TRI_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
void Light3D::lighting() {
    lightingKernel<<<RENDER->BLOCK_BUFFER_COUNT, RENDER->BLOCK_SIZE>>>(
        RENDER->D_BUFFER, *this, RENDER->BUFFER_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
void Light3D::shadowMap() {
    shadowMapKernel<<<RENDER->BLOCK_TRI_COUNT, RENDER->BLOCK_SIZE>>>(
        SHADOW_MAP, D_TRI2DS,
        SHADOW_MAP_WIDTH, SHADOW_MAP_HEIGHT, TRI_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
void Light3D::applyShadow() {
    applyShadowKernel<<<RENDER->BLOCK_BUFFER_COUNT, RENDER->BLOCK_SIZE>>>(
        RENDER->D_BUFFER, SHADOW_MAP,
        SHADOW_MAP_WIDTH, SHADOW_MAP_HEIGHT, RENDER->PIXEL_SIZE, RENDER->BUFFER_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ========================= KERNELS =========================

// Kernel for resetting the shadow map to 1000
__global__ void resetShadowMapKernel(
    float *shadowMap, int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    shadowMap[i] = 1000;
}

// Kernel for applying lighting
__global__ void lightingKernel(
    Pixel3D *pixels, Light3D light, int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    Color3D color = pixels[i].color;

    Vec3D negLightNormal = Vec3D::mult(light.normal, -1);
    double cosA = Vec3D::dot(pixels[i].normal, negLightNormal) /
        (Vec3D::mag(pixels[i].normal) * Vec3D::mag(negLightNormal));

    if (cosA < 0) cosA = 0;

    double ratio = light.ambient + cosA * (light.specular - light.ambient);

    color.runtimeRGB.mult(ratio);

    // Apply colored light
    color.runtimeRGB.v1 = color.runtimeRGB.v1 * light.rgbRatio.x;
    color.runtimeRGB.v2 = color.runtimeRGB.v2 * light.rgbRatio.y;
    color.runtimeRGB.v3 = color.runtimeRGB.v3 * light.rgbRatio.z;

    // Restrict color values
    color.runtimeRGB.restrictRGB();

    pixels[i].color = color;
}

// Kernel for orthographic projection
__global__ void lightOrthographicKernel(
    Tri2D *tri2Ds, const Tri3D *tri3Ds, int p_s, size_t size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // We literally just need to project onto the xy plane
    tri2Ds[i].v1.x = tri3Ds[i].v1.x / p_s;
    tri2Ds[i].v1.y = tri3Ds[i].v1.y / p_s;
    tri2Ds[i].v1.zDepth = tri3Ds[i].v1.z;

    tri2Ds[i].v2.x = tri3Ds[i].v2.x / p_s;
    tri2Ds[i].v2.y = tri3Ds[i].v2.y / p_s;
    tri2Ds[i].v2.zDepth = tri3Ds[i].v2.z;

    tri2Ds[i].v3.x = tri3Ds[i].v3.x / p_s;
    tri2Ds[i].v3.y = tri3Ds[i].v3.y / p_s;
    tri2Ds[i].v3.zDepth = tri3Ds[i].v3.z;
}

// Kernel for light orthographic as well as camera perspective
// to reduce the number of kernel calls (less call = high fps = me happy)
__global__ void sharedTri2DsKernel(
    Tri2D *tri2Dcam, Tri2D *tri2Dlight, const Tri3D *tri3Ds,
    const Camera3D cam, int p_s, size_t size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // We can recycle the camera perspective kernel
    tri2Dcam[i].v1 = Render3D::toVec2D(cam, tri3Ds[i].v1);
    tri2Dcam[i].v2 = Render3D::toVec2D(cam, tri3Ds[i].v2);
    tri2Dcam[i].v3 = Render3D::toVec2D(cam, tri3Ds[i].v3);

    tri2Dcam[i].v1.x /= p_s;
    tri2Dcam[i].v1.y /= p_s;
    tri2Dcam[i].v2.x /= p_s;
    tri2Dcam[i].v2.y /= p_s;
    tri2Dcam[i].v3.x /= p_s;
    tri2Dcam[i].v3.y /= p_s;

    // We literally just need to project onto the xy plane
    tri2Dlight[i].v1.x = tri3Ds[i].v1.x / p_s;
    tri2Dlight[i].v1.y = tri3Ds[i].v1.y / p_s;
    tri2Dlight[i].v1.zDepth = tri3Ds[i].v1.z;

    tri2Dlight[i].v2.x = tri3Ds[i].v2.x / p_s;
    tri2Dlight[i].v2.y = tri3Ds[i].v2.y / p_s;
    tri2Dlight[i].v2.zDepth = tri3Ds[i].v2.z;

    tri2Dlight[i].v3.x = tri3Ds[i].v3.x / p_s;
    tri2Dlight[i].v3.y = tri3Ds[i].v3.y / p_s;
    tri2Dlight[i].v3.zDepth = tri3Ds[i].v3.z;
}

// Kernel for depth map (really easy)
__global__ void shadowMapKernel(
    float *shadowMap, Tri2D *tri2Ds, int s_w, int s_h, size_t size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // Get the triangle
    Tri2D tri = tri2Ds[i];

    // Get the bounding box
    int minX = min(tri.v1.x, min(tri.v2.x, tri.v3.x));
    int maxX = max(tri.v1.x, max(tri.v2.x, tri.v3.x));
    int minY = min(tri.v1.y, min(tri.v2.y, tri.v3.y));
    int maxY = max(tri.v1.y, max(tri.v2.y, tri.v3.y));

    // Clip the bounding box (slightly expanded)
    minX = max(minX, 0);
    maxX = min(maxX, s_w - 1);
    minY = max(minY, 0);
    maxY = min(maxY, s_h - 1);

    // Loop through the bounding box
    for (int x = minX; x <= maxX; x++) {
        for (int y = minY; y <= maxY; y++) {
            // Get the barycentric coordinates
            Vec3D bary = Vec2D::barycentricLambda(
                Vec2D(x, y), tri.v1, tri.v2, tri.v3
            );

            // Check if the point is inside the triangle
            if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

            // Get the depth
            float depth = bary.x * tri.v1.zDepth + bary.y * tri.v2.zDepth + bary.z * tri.v3.zDepth;

            // Update the shadow map
            int index = x + y * s_w;

            if (atomicMinFloat(&shadowMap[index], depth)) {
                shadowMap[index] = depth;
            }
        }
    }
}

// Kernel for applying the shadow map from the buffer
__global__ void applyShadowKernel(
    Pixel3D *buffer, float *shadowMap, int s_w, int s_h, int p_s, size_t size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // Get the pixel
    Pixel3D pixel = buffer[i];

    // Get the pixel's position
    int x = i % s_w;
    int y = i / s_w;

    // Get the depth
    float depth = pixel.screen.zDepth;

    // Get the shadow map index
    int index = x / p_s + y / p_s * s_w;

    // Get the shadow map depth
    float shadowDepth = shadowMap[index];

    // Check if the pixel is in shadow
    if (depth > shadowDepth) {
        pixel.color.runtimeRGB.mult(0.5);
    }

    buffer[i] = pixel;
}