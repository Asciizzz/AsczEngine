#include <SFMLTexture.cuh>

SFMLTexture::SFMLTexture(Render3D *render) {
    texture.create(render->W_WIDTH, render->W_HEIGHT);
    pixels = new sf::Uint8[render->W_WIDTH * render->W_HEIGHT * 4];

    // Allocate memory for the Pixel buffer
    CUDA_CHECK(cudaMalloc(&d_buffer, render->BUFFER_SIZE * sizeof(Pixel3D)));
    CUDA_CHECK(cudaMalloc(&d_sfPixels, render->W_WIDTH * render->W_HEIGHT * 4));

    // Set kernel parameters
    numBlocks = (render->BUFFER_SIZE + blockSize - 1) / blockSize;

    // Create SFML sprite
    sprite = sf::Sprite(texture);
}
SFMLTexture::~SFMLTexture() {
    delete[] pixels;
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_sfPixels));
}

void SFMLTexture::updateTexture(Render3D *render) {
    // Copy buffer to device
    CUDA_CHECK(cudaMemcpy(d_buffer, render->BUFFER, render->BUFFER_SIZE * sizeof(Pixel3D), cudaMemcpyHostToDevice));

    // Execute fillPixel kernel
    fillPixelKernel<<<numBlocks, blockSize>>>(
        d_sfPixels, d_buffer,
        render->BUFFER_WIDTH, render->BUFFER_HEIGHT,
        render->PIXEL_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy pixels back to host
    CUDA_CHECK(cudaMemcpy(pixels, d_sfPixels, render->W_WIDTH * render->W_HEIGHT * 4, cudaMemcpyDeviceToHost));

    // Update texture
    texture.update(pixels);
}

__global__ void fillPixelKernel(
    sf::Uint8 *pixels, Pixel3D *buffer,
    int b_w, int b_h, int p_s
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < b_w * b_h) {
        int x = i % b_w;
        int y = i / b_w;
        int b_index = x + y * b_w;

        for (int i = 0; i < p_s; i++)
        for (int j = 0; j < p_s; j++) {
            int p_index = x * p_s + i + (y * p_s + j) * b_w * p_s;
            p_index *= 4;

            // Get the pixel color
            Color3D color = buffer[b_index].color;

            // Fill the pixel
            pixels[p_index] = color.runtimeRGB.v1;
            pixels[p_index + 1] = color.runtimeRGB.v2;
            pixels[p_index + 2] = color.runtimeRGB.v3;
            pixels[p_index + 3] = 255;
        }
    }
}