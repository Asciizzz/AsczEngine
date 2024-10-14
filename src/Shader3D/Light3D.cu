#include <Light3D.cuh>

Light3D::Light3D(Render3D *render) {
    RENDER = render;
}

Light3D::~Light3D() {
    freeTris();
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
    SHADOW_MAP = new Shadow[SHADOW_MAP_SIZE];
    CUDA_CHECK(cudaMalloc(
        &SHADOW_MAP, SHADOW_MAP_SIZE * sizeof(Shadow))
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
    for (size_t i = 0; i < 2; i++)
        shadowMapKernel<<<RENDER->BLOCK_TRI_COUNT, RENDER->BLOCK_SIZE>>>(
            SHADOW_MAP, D_TRI2DS,
            SHADOW_MAP_WIDTH, SHADOW_MAP_HEIGHT, RENDER->TRI_SIZE
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

// Kernel for resetting the shadow map
__global__ void resetShadowMapKernel(
    Shadow *shadowMap, int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    shadowMap[i].depth = 1000;
    shadowMap[i].meshID = -1;
}

// Kernel for applying lighting
__global__ void lightingKernel(
    Pixel3D *pixels, Light3D light, int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // return; // Temporary

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
    tri2Ds[i].v1.x = tri3Ds[i].v1.x;
    tri2Ds[i].v1.y = tri3Ds[i].v1.y;
    tri2Ds[i].v1.zDepth = tri3Ds[i].v1.z;

    tri2Ds[i].v2.x = tri3Ds[i].v2.x;
    tri2Ds[i].v2.y = tri3Ds[i].v2.y;
    tri2Ds[i].v2.zDepth = tri3Ds[i].v2.z;

    tri2Ds[i].v3.x = tri3Ds[i].v3.x;
    tri2Ds[i].v3.y = tri3Ds[i].v3.y;
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

    tri2Dcam[i].meshID = tri3Ds[i].meshID;
    tri2Dlight[i].meshID = tri3Ds[i].meshID;

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
    tri2Dlight[i].v1.x = tri3Ds[i].v1.x;
    tri2Dlight[i].v1.y = tri3Ds[i].v1.y;
    tri2Dlight[i].v1.zDepth = tri3Ds[i].v1.z;

    tri2Dlight[i].v2.x = tri3Ds[i].v2.x;
    tri2Dlight[i].v2.y = tri3Ds[i].v2.y;
    tri2Dlight[i].v2.zDepth = tri3Ds[i].v2.z;

    tri2Dlight[i].v3.x = tri3Ds[i].v3.x;
    tri2Dlight[i].v3.y = tri3Ds[i].v3.y;
    tri2Dlight[i].v3.zDepth = tri3Ds[i].v3.z;
}

// Kernel for depth map (really easy)
__global__ void shadowMapKernel(
    Shadow *shadowMap, Tri2D *tri2Ds,
    int s_w, int s_h, size_t size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // Get the triangle
    Tri2D tri2D = tri2Ds[i];

    // Get the bounding box
    int minX = min(tri2D.v1.x, min(tri2D.v2.x, tri2D.v3.x));
    int maxX = max(tri2D.v1.x, max(tri2D.v2.x, tri2D.v3.x));
    int minY = min(tri2D.v1.y, min(tri2D.v2.y, tri2D.v3.y));
    int maxY = max(tri2D.v1.y, max(tri2D.v2.y, tri2D.v3.y));

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
                Vec2D(x, y), tri2D.v1, tri2D.v2, tri2D.v3
            );

            // Check if the point is inside the triangle
            if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

            // Get the depth
            float depth = Vec2D::barycentricCalc(
                bary, tri2D.v1.zDepth, tri2D.v2.zDepth, tri2D.v3.zDepth
            );

            // Update the shadow map
            int index = x + y * s_w;

            bool shadowCloser = atomicMinFloat(&shadowMap[index].depth, depth);

            if (tri2D.meshID != -1 &&
                tri2D.meshID == shadowMap[index].meshID) {
                
                if (!shadowCloser) {
                    shadowMap[index] = {depth, tri2D.meshID};
                }

            } else if (shadowCloser) {
                shadowMap[index] = {depth, tri2D.meshID};
            }

        }
    }
}

// Kernel for applying the shadow map from the buffer
__global__ void applyShadowKernel(
    Pixel3D *buffer, Shadow *shadowMap, int s_w, int s_h, int p_s, size_t size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // Get the pixel
    Pixel3D pixel = buffer[i];

    if (!pixel.shadow) return;

    // Get the pixel's position
    int x = pixel.world.x;
    int y = pixel.world.y;

    // Get the depth
    float depth = pixel.world.z;

    // Get the shadow map index
    int index = x + y * s_w;

    // Get the shadow map depth
    float shadowDepth = shadowMap[index].depth;

    // Check if the pixel is in shadow
    if (depth > shadowDepth) {
        pixel.color.runtimeRGB.mult(0.5);
    }

    buffer[i] = pixel;
}