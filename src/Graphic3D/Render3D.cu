#include <Render3D.cuh>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

Render3D::Render3D(Camera3D *camera, int w_w, int w_h, int p_s) {
    // Window settings
    W_WIDTH = w_w;
    W_HEIGHT = w_h;
    W_CENTER_X = w_w / 2;
    W_CENTER_Y = w_h / 2;
    PIXEL_SIZE = p_s;

    // Camera settings
    CAMERA = camera;
    CAMERA->w_width = w_w;
    CAMERA->w_height = w_h;
    CAMERA->w_center_x = W_CENTER_X;
    CAMERA->w_center_y = W_CENTER_Y;

    // Initialize buffer
    BUFFER_WIDTH = w_w / p_s;
    BUFFER_HEIGHT = w_h / p_s;
    BUFFER_SIZE = BUFFER_WIDTH * BUFFER_HEIGHT;
    BUFFER = new Pixel3D[BUFFER_SIZE];

    // Memory allocation for device buffer
    CUDA_CHECK(cudaMalloc(&D_BUFFER, BUFFER_SIZE * sizeof(Pixel3D)));
}
Render3D::~Render3D() {
    delete[] BUFFER;
    CUDA_CHECK(cudaFree(D_BUFFER));
}

void Render3D::resize(int w_w, int w_h, int p_s) {
    // Update window settings
    W_WIDTH = w_w;
    W_HEIGHT = w_h;
    W_CENTER_X = w_w / 2;
    W_CENTER_Y = w_h / 2;
    PIXEL_SIZE = p_s;

    // Update buffer settings
    BUFFER_WIDTH = w_w / p_s;
    BUFFER_HEIGHT = w_h / p_s;
    BUFFER_SIZE = BUFFER_WIDTH * BUFFER_HEIGHT;

    // Reset buffer
    delete[] BUFFER;
    BUFFER = new Pixel3D[BUFFER_SIZE];

    // Free old device memory
    CUDA_CHECK(cudaFree(D_BUFFER));
    // Memory allocation for device buffer
    CUDA_CHECK(cudaMalloc(&D_BUFFER, BUFFER_SIZE * sizeof(Pixel3D)));
}

// Reset all
void Render3D::reset() {
    // Reset buffer
    delete[] BUFFER;
    BUFFER = new Pixel3D[BUFFER_SIZE];
}

// To vec2D
__host__ __device__ Vec2D Render3D::toVec2D(const Camera3D &cam, Vec3D v) {
    Vec3D diff = Vec3D::sub(v, cam.pos);

    // Apply Yaw (rotation around Y axis)
    float cosYaw = cos(-cam.ang.y);
    float sinYaw = sin(-cam.ang.y);
    float tempX = diff.x * cosYaw + diff.z * sinYaw;
    float tempZ = -diff.x * sinYaw + diff.z * cosYaw;

    // Apply Pitch (rotation around X axis)
    float cosPitch = cos(-cam.ang.x);
    float sinPitch = sin(-cam.ang.x);
    float finalY = tempZ * sinPitch + diff.y * cosPitch;
    float finalZ = tempZ * cosPitch - diff.y * sinPitch;

    float projX = (tempX * cam.screendist) / finalZ;
    float projY = -(finalY * cam.screendist) / finalZ;

    if (finalZ < 0) {
        projX *= -10;
        projY *= -10;
    }

    projX += cam.w_center_x;
    projY += cam.w_center_y;

    return Vec2D(projX, projY, finalZ);
}

// The main render function
void Render3D::renderGPU(Tri3D *tri3Ds, size_t size) {
    // Set kernel parameters
    const size_t numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate triangles memory on device
    CUDA_CHECK(cudaMalloc(&D_TRI3DS, size * sizeof(Tri3D)));
    CUDA_CHECK(cudaMalloc(&D_TRI2DS, size * sizeof(Tri2D)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(D_TRI3DS, tri3Ds, size * sizeof(Tri3D), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D_BUFFER, BUFFER, BUFFER_SIZE * sizeof(Pixel3D), cudaMemcpyHostToDevice));

    // Execute tri3DsTo2Ds kernel
    tri3DsTo2DsKernel<<<numBlocks, BLOCK_SIZE>>>(
        D_TRI2DS, D_TRI3DS, *CAMERA, PIXEL_SIZE, size
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sort the 2D triangles by zDepth (and rearrange the 3D triangles accordingly)
    thrust::device_vector<Tri2D> dev_tri2Ds(D_TRI2DS, D_TRI2DS + size);
    thrust::device_vector<Tri3D> dev_tri3Ds(D_TRI3DS, D_TRI3DS + size);
    // Sort using the thrust, while also rearranging the tri3Ds
    thrust::sort_by_key(dev_tri2Ds.begin(), dev_tri2Ds.end(), dev_tri3Ds.begin(),
        [] __device__ (const Tri2D& a, const Tri2D& b) -> bool {
            return a.v1.zDepth > b.v1.zDepth;
        }
    );
    // Copy back to device memory
    thrust::copy(dev_tri2Ds.begin(), dev_tri2Ds.end(), D_TRI2DS);
    thrust::copy(dev_tri3Ds.begin(), dev_tri3Ds.end(), D_TRI3DS);

    // Execute rasterization kernel
    rasterizeKernel<<<numBlocks, BLOCK_SIZE>>>(
        // Buffer and tris
        D_BUFFER, D_TRI2DS, D_TRI3DS,
        // Other properties if needed
        LIGHT,
        // Size properties
        BUFFER_WIDTH, BUFFER_HEIGHT, size
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // For every unoccupied pixel, fill it with the default color
    fillBufferKernel<<<BUFFER_SIZE / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
        D_BUFFER, DEFAULT_COLOR, BUFFER_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy pixels back to host buffer
    CUDA_CHECK(cudaMemcpy(BUFFER, D_BUFFER, BUFFER_SIZE * sizeof(Pixel3D), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(D_TRI3DS));
    CUDA_CHECK(cudaFree(D_TRI2DS));
}

void Render3D::renderCPU(std::vector<Tri3D> tri3Ds) {
    // Decrapitated
}

__global__ void fillBufferKernel(
    Pixel3D *buffer, Color3D color, size_t size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && !buffer[i].active) buffer[i].color = color;
}

__global__ void tri3DsTo2DsKernel(
    Tri2D *tri2Ds, const Tri3D *tri3Ds,
    Camera3D cam, int p_s, size_t size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        Vec2D v1 = Render3D::toVec2D(cam, tri3Ds[i].v1);
        Vec2D v2 = Render3D::toVec2D(cam, tri3Ds[i].v2);
        Vec2D v3 = Render3D::toVec2D(cam, tri3Ds[i].v3);

        // Divide by pixel size
        v1.x /= p_s; v1.y /= p_s;
        v2.x /= p_s; v2.y /= p_s;
        v3.x /= p_s; v3.y /= p_s;

        // IMPORTANT: v1 -> v3 will have ascending zDepth
        // (note: we cannot use std::swap in device code)
        tri2Ds[i].v1 = v1;
        tri2Ds[i].v2 = v2;
        tri2Ds[i].v3 = v3;
    }
}

__device__ bool atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old) > value;
}

__global__ void rasterizeKernel(
    // Buffer and tris
    Pixel3D *pixels, const Tri2D *tri2Ds, const Tri3D *tri3Ds,
    // Other properties if needed
    LightSrc3D light,
    // Size properties
    int b_w, int b_h, size_t size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        // If the triangle is not visible, skip
        if (tri2Ds[i].v1.zDepth < 0 && tri2Ds[i].v2.zDepth < 0 && tri2Ds[i].v3.zDepth < 0) return;
        if (tri2Ds[i].v1.x < 0 && tri2Ds[i].v2.x < 0 && tri2Ds[i].v3.x < 0) return;
        if (tri2Ds[i].v1.y < 0 && tri2Ds[i].v2.y < 0 && tri2Ds[i].v3.y < 0) return;
        if (tri2Ds[i].v1.x >= b_w && tri2Ds[i].v2.x >= b_w && tri2Ds[i].v3.x >= b_w) return;
        if (tri2Ds[i].v1.y >= b_h && tri2Ds[i].v2.y >= b_h && tri2Ds[i].v3.y >= b_h) return;

        // Find the bounding box of the 2D polygon
        int minX = min(tri2Ds[i].v1.x, min(tri2Ds[i].v2.x, tri2Ds[i].v3.x));
        int maxX = max(tri2Ds[i].v1.x, max(tri2Ds[i].v2.x, tri2Ds[i].v3.x));
        int minY = min(tri2Ds[i].v1.y, min(tri2Ds[i].v2.y, tri2Ds[i].v3.y));
        int maxY = max(tri2Ds[i].v1.y, max(tri2Ds[i].v2.y, tri2Ds[i].v3.y));

        // Clip the bounding box (slightly expanded)
        minX = max(minX, 1);
        maxX = min(maxX, b_w - 2);
        minY = max(minY, 1);
        maxY = min(maxY, b_h - 2);

        // Rasterize the triangle using the baricentric coordinates
        for (int x = minX - 1; x <= maxX + 1; x++)
        for (int y = minY - 1; y <= maxY + 1; y++) {
            int index = x + y * b_w;
            // Check if the pixel is inside the triangle
            Vec2D screen(x, y);

            Vec3D barycentric = Vec2D::barycentricLambda(
                screen, tri2Ds[i].v1, tri2Ds[i].v2, tri2Ds[i].v3
            );

            // Check if the pixel is inside the triangle
            // (allow small margin of error)
            if (barycentric.x < 0.0 ||
                barycentric.y < 0.0 ||
                barycentric.z < 0.0) continue;

            screen.zDepth = Vec2D::barycentricCalc(
                barycentric, tri2Ds[i].v1.zDepth, tri2Ds[i].v2.zDepth, tri2Ds[i].v3.zDepth
            );

            // Check if the pixel is closer than the current pixel
            if (!atomicMinFloat(&pixels[index].screen.zDepth, screen.zDepth))
                continue;

            // Get world position
            double px = Vec2D::barycentricCalc(
                barycentric, tri3Ds[i].v1.x, tri3Ds[i].v2.x, tri3Ds[i].v3.x
            );
            double py = Vec2D::barycentricCalc(
                barycentric, tri3Ds[i].v1.y, tri3Ds[i].v2.y, tri3Ds[i].v3.y
            );
            double pz = Vec2D::barycentricCalc(
                barycentric, tri3Ds[i].v1.z, tri3Ds[i].v2.z, tri3Ds[i].v3.z
            );
            Vec3D world(px, py, pz);

            // BETA: Light color manipulation
            Color3D color = tri3Ds[i].color;

            Vec3D lightDir = Vec3D::sub(light.pos, world);
            double cosA = Vec3D::dot(tri3Ds[i].normal, lightDir) /
                (Vec3D::mag(tri3Ds[i].normal) * Vec3D::mag(lightDir));
            // Note: we cannot use std::max and std::min in device code
            if (cosA < 0) cosA = 0;
            // if (cosA < 0) cosA = -cosA;

            double ratio = light.ambient + cosA * (light.specular - light.ambient);

            color.runtimeRGB.mult(ratio);

            // Apply colored light
            color.runtimeRGB.v1 = color.runtimeRGB.v1 * light.rgbRatio.x;
            color.runtimeRGB.v2 = color.runtimeRGB.v2 * light.rgbRatio.y;
            color.runtimeRGB.v3 = color.runtimeRGB.v3 * light.rgbRatio.z;

            // Restrict color values
            color.runtimeRGB.restrictRGB();

            // Set buffer values
            pixels[index] = {
                color, tri3Ds[i].normal, world, screen, true
            };
        }
    }
}