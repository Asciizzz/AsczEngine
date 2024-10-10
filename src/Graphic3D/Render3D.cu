#include <Render3D.cuh>

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
    setBuffer(w_w, w_h, p_s);
}
Render3D::~Render3D() {
    delete[] BUFFER;
    CUDA_CHECK(cudaFree(D_BUFFER));

    freeTris();
}

void Render3D::setBuffer(int w, int h, int p_s) {
    // Update buffer settings
    BUFFER_WIDTH = w / p_s;
    BUFFER_HEIGHT = h / p_s;
    BUFFER_SIZE = BUFFER_WIDTH * BUFFER_HEIGHT;

    delete[] BUFFER;
    BUFFER = new Pixel3D[BUFFER_SIZE];
    BLOCK_BUFFER_COUNT = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMalloc(&D_BUFFER, BUFFER_SIZE * sizeof(Pixel3D)));
}

void Render3D::bufferMemcpy() {
    CUDA_CHECK(cudaMemcpy(BUFFER, D_BUFFER, BUFFER_SIZE * sizeof(Pixel3D), cudaMemcpyDeviceToHost));
}

void Render3D::resizeWindow(int w_w, int w_h, int p_s) {
    // Update window settings
    W_WIDTH = w_w;
    W_HEIGHT = w_h;
    W_CENTER_X = w_w / 2;
    W_CENTER_Y = w_h / 2;
    PIXEL_SIZE = p_s;

    // Update camera settings
    CAMERA->w_width = w_w;
    CAMERA->w_height = w_h;
    CAMERA->w_center_x = W_CENTER_X;
    CAMERA->w_center_y = W_CENTER_Y;

    // Free device buffer
    CUDA_CHECK(cudaFree(D_BUFFER));
    // Update buffer settings
    setBuffer(w_w, w_h, p_s);
}

void Render3D::mallocTris(size_t size) {
    BLOCK_TRI_COUNT = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMalloc(&D_TRI3DS, size * sizeof(Tri3D)));
    CUDA_CHECK(cudaMalloc(&D_TRI2DS, size * sizeof(Tri2D)));
}
void Render3D::freeTris() {
    CUDA_CHECK(cudaFree(D_TRI3DS));
    CUDA_CHECK(cudaFree(D_TRI2DS));
}
void Render3D::resizeTris(size_t size) {
    freeTris();
    mallocTris(size);
}

// To vec2D
__host__ __device__ Vec2D Render3D::toCameraPerspective(const Camera3D &cam, Vec3D v) {
    Vec3D diff = Vec3D::sub(v, cam.pos);

    // Apply Yaw (rotation around Y axis)
    double cosYaw = cos(-cam.ang.y);
    double sinYaw = sin(-cam.ang.y);
    double tempX = diff.x * cosYaw + diff.z * sinYaw;
    double tempZ = -diff.x * sinYaw + diff.z * cosYaw;

    // Apply Pitch (rotation around X axis)
    double cosPitch = cos(-cam.ang.x);
    double sinPitch = sin(-cam.ang.x);
    double finalY = tempZ * sinPitch + diff.y * cosPitch;
    double finalZ = tempZ * cosPitch - diff.y * sinPitch;

    double projX = (tempX * cam.screendist) / finalZ;
    double projY = -(finalY * cam.screendist) / finalZ;

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
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(D_TRI3DS, tri3Ds, size * sizeof(Tri3D), cudaMemcpyHostToDevice));

    // Resetting the buffer
    resetBufferKernel<<<BLOCK_BUFFER_COUNT, BLOCK_SIZE>>>(
        D_BUFFER, DEFAULT_COLOR, BUFFER_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Execute visisbleTriangles kernel
    visisbleTrianglesKernel<<<BLOCK_TRI_COUNT, BLOCK_SIZE>>>(
        D_TRI3DS, *CAMERA, size
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Execute 2D convert kernel
    transform2Dkernel<<<BLOCK_TRI_COUNT, BLOCK_SIZE>>>(
        D_TRI2DS, D_TRI3DS, *CAMERA, PIXEL_SIZE, size
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Execute rasterization kernel
    for (int i = 0; i < 2; i++)
        rasterizeKernel<<<BLOCK_TRI_COUNT, BLOCK_SIZE>>>(
            D_BUFFER, D_TRI2DS, D_TRI3DS,
            BUFFER_WIDTH, BUFFER_HEIGHT, size
        );
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Render3D::renderCPU(std::vector<Tri3D> tri3Ds) {
    // Decrapitated
}

__global__ void resetBufferKernel(
    Pixel3D *buffer, Color3D def_color, size_t size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    buffer[i] = Pixel3D();
    buffer[i].color = def_color;
}

__global__ void visisbleTrianglesKernel(
    Tri3D *tri3Ds, Camera3D cam, size_t size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    Vec3D camNormal = cam.plane.normal;

    // Find the point closest to the camera plane
    double dist1 = cam.plane.distance(tri3Ds[i].v1);
    double dist2 = cam.plane.distance(tri3Ds[i].v2);
    double dist3 = cam.plane.distance(tri3Ds[i].v3);
    double minDist = min(dist1, min(dist2, dist3));

    Vec3D minPoint;

    if (minDist == dist1) minPoint = tri3Ds[i].v1;
    else if (minDist == dist2) minPoint = tri3Ds[i].v2;
    else minPoint = tri3Ds[i].v3;

    // Create a vector connecting the min point to the camera
    Vec3D camDir = Vec3D::sub(cam.pos, minPoint);

    // If the angle of the vector is greater than 90 degrees, the triangle is not visible
    tri3Ds[i].visible = Vec3D::dot(camNormal, camDir) < 0;
}

__global__ void transform2Dkernel(
    Tri2D *tri2Ds, const Tri3D *tri3Ds,
    Camera3D cam, int p_s, size_t size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size || !tri3Ds[i].visible) return;

    Vec2D v1 = Render3D::toCameraPerspective(cam, tri3Ds[i].v1);
    Vec2D v2 = Render3D::toCameraPerspective(cam, tri3Ds[i].v2);
    Vec2D v3 = Render3D::toCameraPerspective(cam, tri3Ds[i].v3);

    v1.x /= p_s; v1.y /= p_s;
    v2.x /= p_s; v2.y /= p_s;
    v3.x /= p_s; v3.y /= p_s;

    tri2Ds[i].v1 = v1;
    tri2Ds[i].v2 = v2;
    tri2Ds[i].v3 = v3;
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
    Pixel3D *pixels, const Tri2D *tri2Ds, const Tri3D *tri3Ds,
    int b_w, int b_h, size_t size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size || !tri3Ds[i].visible) return;

    // Ignore non-visible triangles
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
    minX = max(minX, 0);
    maxX = min(maxX, b_w - 1);
    minY = max(minY, 0);
    maxY = min(maxY, b_h - 1);

    // Rasterize the triangle using the baricentric coordinates
    for (int x = minX; x <= maxX; x++)
    for (int y = minY; y <= maxY; y++) {
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

        // =================== LIGHTING =======================
        // !!! DECRAPIATED !!!

        // (you can modify the lighting logic based on the type of light)
        // (there are: spot light, point light, directional light, etc.)

        // BETA: Light color manipulation
        // Color3D color = tri3Ds[i].color;

        // Vec3D lightDir = Vec3D::sub(light.pos, world);
        // double cosA = Vec3D::dot(tri3Ds[i].normal, lightDir) /
        //     (Vec3D::mag(tri3Ds[i].normal) * Vec3D::mag(lightDir));
        // // Note: we cannot use std::max and std::min in device code
        // // if (cosA < 0) cosA = 0;

        // if (cosA < 0) cosA = tri3Ds[i].isTwoSided ? -cosA : 0;

        // double ratio = light.ambient + cosA * (light.specular - light.ambient);

        // color.runtimeRGB.mult(ratio);

        // // Apply colored light
        // color.runtimeRGB.v1 = color.runtimeRGB.v1 * light.rgbRatio.x;
        // color.runtimeRGB.v2 = color.runtimeRGB.v2 * light.rgbRatio.y;
        // color.runtimeRGB.v3 = color.runtimeRGB.v3 * light.rgbRatio.z;

        // // Restrict color values
        // color.runtimeRGB.restrictRGB();

        // Set buffer values
        pixels[index] = {
            tri3Ds[i].color, tri3Ds[i].normal, world, screen
        };
    }
}