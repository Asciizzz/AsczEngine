#include <Light3D.cuh>

Light3D::Light3D() {
}

Light3D::~Light3D() {
}

void Light3D::initShadowMap(int w, int h) {
    CUDA_CHECK(cudaMalloc(&SHADOW_MAP, w * h * sizeof(float)));
}

void Light3D::demo(Render3D *render) {
    lightingKernel<<<render->BLOCK_BUFFER_COUNT, render->BLOCK_SIZE>>>(
        render->D_BUFFER, *this, render->BUFFER_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    render->memcpyBuffer();
}

// Kernel for orthographic projection
__global__ void lightOrthographicKernel(
    Tri2D *tri2Ds, const Tri3D *tri3Ds, size_t size
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