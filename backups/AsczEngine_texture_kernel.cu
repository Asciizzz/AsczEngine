#include <Render3D.cuh>
#include <CsLogHandle.h>

using namespace sf;

// Create a kernel that would fill in pixel from buffer to sf::Texture
__global__ void fillPixel(
    sf::Uint8 *pixels, const Pixel3D *pixel3D,
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
        Color3D color = pixel3D[b_index].color;

        // Fill the pixel
        pixels[p_index] = color.runtimeRGB.v1;
        pixels[p_index + 1] = color.runtimeRGB.v2;
        pixels[p_index + 2] = color.runtimeRGB.v3;
        pixels[p_index + 3] = 255;
        }
    }
}

int main() {
    FpsHandle *FPS = new FpsHandle();
    Camera3D *CAM = new Camera3D();
    Render3D *RENDER = new Render3D(CAM);
    CAM->w_center_x = RENDER->W_CENTER_X;
    CAM->w_center_y = RENDER->W_CENTER_Y;

    // Debugging
    CsLogHandle *CSLOG = new CsLogHandle();

    RenderWindow WINDOW(
        VideoMode(RENDER->W_WIDTH, RENDER->W_HEIGHT), RENDER->W_TITLE
    );
    WINDOW.setMouseCursorVisible(false);

    // =================== EXPERIMENTATION =======================

    size_t tri_count = 1 << 8;
    Tri3D *tri_test = new Tri3D[tri_count];

    // Create 1000 walls which are made of 2 triangles each
    // Each walls are 10z apart, and their color changes
    int offsetZ = 24;
    for (int i = 0; i < tri_count; i += 2) {
        tri_test[i] = Tri3D(
            Vec3D(10, 10, i * 10 + offsetZ),
            Vec3D(10, -10, i * 10 + offsetZ),
            Vec3D(-10, -10, i * 10 + offsetZ),
            Vec3D(0, 0, -1),
            Color3D((i * 10) % 255, 0, 255 - (i * 10) % 255)
        );
        tri_test[i + 1] = Tri3D(
            Vec3D(10, 10, i * 10 + offsetZ),
            Vec3D(-10, 10, i * 10 + offsetZ),
            Vec3D(-10, -10, i * 10 + offsetZ),
            Vec3D(0, 0, -1),
            Color3D((i * 10) % 255, 0, 255 - (i * 10) % 255)
        );
    }

    // Unrelated stuff
    double rainbowR = 255;
    double rainbowG = 0;
    double rainbowB = 0;
    short cycle = 0;

    // A texture/sprite to draw the buffer
    Texture texture;
    texture.create(RENDER->W_WIDTH, RENDER->W_HEIGHT);
    sf::Uint8 *pixels = new sf::Uint8[RENDER->W_WIDTH * RENDER->W_HEIGHT * 4];

    // Allocate memory for the Pixel buffer
    Pixel3D *d_pixels;
    sf::Uint8 *d_sfPixels;
    CUDA_CHECK(cudaMalloc(&d_pixels, RENDER->BUFFER_SIZE * sizeof(Pixel3D)));
    CUDA_CHECK(cudaMalloc(&d_sfPixels, RENDER->W_WIDTH * RENDER->W_HEIGHT * 4));

    // Set kernel parameters
    const size_t blockSize = 256;
    const size_t numBlocks = (RENDER->BUFFER_SIZE + blockSize - 1) / blockSize;

    // Create SFML sprite
    Sprite sprite(texture);

    while (WINDOW.isOpen()) {
        // Frame start
        FPS->startFrame();

        // =================== EVENT HANDLING =======================
        Event event;
        while (WINDOW.pollEvent(event)) {
            if (event.type == Event::Closed ||
                Keyboard::isKeyPressed(Keyboard::Escape)) {
                WINDOW.close();
            }

            // Press space to toggle focus
            if (Keyboard::isKeyPressed(Keyboard::Space)) {
                CAM->focus = !CAM->focus;

                // Hide/unhide cursor
                WINDOW.setMouseCursorVisible(!CAM->focus);
            }
        }

        if (CAM->focus) {
            // Mouse movement handling
            sf::Vector2i mousePos = sf::Mouse::getPosition(WINDOW);
            sf::Mouse::setPosition(sf::Vector2i(RENDER->W_CENTER_X, RENDER->W_CENTER_Y), WINDOW);

            // Move from center
            int dMx = mousePos.x - RENDER->W_CENTER_X;
            int dMy = mousePos.y - RENDER->W_CENTER_Y;

            // Camera look around
            CAM->ang.x -= dMy * CAM->m_sens * FPS->dTimeSec;
            CAM->ang.y += dMx * CAM->m_sens * FPS->dTimeSec;

            // Restrict the angle
            CAM->ang.x = std::max(-M_PI_2, std::min(M_PI_2, CAM->ang.x));

            if (CAM->ang.y > M_2PI) CAM->ang.y -= M_2PI;
            if (CAM->ang.y < 0) CAM->ang.y += M_2PI;

            // Mouse Click = move forward
            if (Mouse::isButtonPressed(Mouse::Left))       CAM->vel = 1;
            else if (Mouse::isButtonPressed(Mouse::Right)) CAM->vel = -1;
            else                                           CAM->vel = 0;
        }
        CAM->update();

    // ============== WHERE THE REAL FUN BEGINS =====================   

        // =================== LOGGING =======================

        CSLOG->clear();
        // Rainbow color
        double step = 120 * FPS->dTimeSec;
        if (cycle == 0) {
            rainbowG += step; rainbowR -= step;
            if (rainbowG >= 255) cycle = 1;
        } else if (cycle == 1) {
            rainbowB += step; rainbowG -= step;
            if (rainbowB >= 255) cycle = 2;
        } else if (cycle == 2) {
            rainbowR += step; rainbowB -= step;
            if (rainbowR >= 255) cycle = 0;
        }
        // Ensure the value is in range
        rainbowR = std::max(0.0, std::min(rainbowR, 255.0));
        rainbowG = std::max(0.0, std::min(rainbowG, 255.0));
        rainbowB = std::max(0.0, std::min(rainbowB, 255.0));

        sf::Color rainbow = sf::Color(rainbowR, rainbowG, rainbowB);
        CSLOG->addLog("Welcome to AsczEngine 2.0!", rainbow, 1);

        // FPS <= 20: Fully Red
        // FPS >= 80: Fully Green
        double gRatio = double(FPS->fps - 20) / 60;
        gRatio = std::max(0.0, std::min(gRatio, 1.0));
        sf::Color fpsColor((1 - gRatio) * 255, gRatio * 255, 0);
        CSLOG->addLog("FPS: " + std::to_string(FPS->fps), fpsColor);

        // ======= Main graphic rendering pipeline =======
        RENDER->reset();
        RENDER->renderGPU(tri_test, tri_count);

        // == SFML Rendering that ACTUALLY support parallelism ==
        /*
        Idea: since you cant just execute draw function in parallel, you can
        instead create a texture, fill it with pixels IN PARALLEL, and then
        draw the texture to the window. This way, you can utilize the GPU
        to fill the pixels, and the CPU to draw the texture.

        Jesus Christ, if only SFML has a way to draw pixels in parallel
        Oh wait isnt it related to OpenGL? I think it is...
        */
        WINDOW.clear(Color::Black);
        // Copy the buffer to the device
        CUDA_CHECK(cudaMemcpy(d_pixels, RENDER->BUFFER, RENDER->BUFFER_SIZE * sizeof(Pixel3D), cudaMemcpyHostToDevice));
        // Fill the pixels
        fillPixel<<<numBlocks, blockSize>>>(
            d_sfPixels, d_pixels,
            RENDER->BUFFER_WIDTH, RENDER->BUFFER_HEIGHT,
            RENDER->PIXEL_SIZE
        );
        // Copy the pixels back to host
        CUDA_CHECK(cudaMemcpy(pixels, d_sfPixels, RENDER->W_WIDTH * RENDER->W_HEIGHT * 4, cudaMemcpyDeviceToHost));
        // Update the texture
        texture.update(pixels);
        // Draw the sprite
        WINDOW.draw(sprite);

        // Draw the log
        CSLOG->drawLog(WINDOW);

        // Display the window
        WINDOW.display();

        // Frame end
        FPS->endFrame();
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_pixels));
    CUDA_CHECK(cudaFree(d_sfPixels));

    delete FPS, CAM, RENDER, CSLOG;
    delete[] pixels;
    delete[] tri_test;

    return 0;
}