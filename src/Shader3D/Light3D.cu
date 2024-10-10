#include <Light3D.cuh>

Light3D::Light3D() {
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
}

void Light3D::demo(Render3D *render) {
    lightingKernel<<<render->BLOCK_BUFFER_COUNT, render->BLOCK_SIZE>>>(
        render->D_BUFFER, *this, render->BUFFER_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    render->memcpyBuffer();
}

void Light3D::sharedTri2Ds(Render3D *render) {
    sharedTri2DsKernel<<<render->BLOCK_TRI_COUNT, render->BLOCK_SIZE>>>(
        render->D_TRI2DS, D_TRI2DS, render->D_TRI3DS,
        *render->CAMERA, render->PIXEL_SIZE, TRI_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ========================= KERNELS =========================

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
// to reduce the number of kernel calls
// (less call = high fps = me happy)
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

    // We also need to project onto the xy plane
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