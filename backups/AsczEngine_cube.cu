#include <Render3D.cuh>
#include <CsLogHandle.h>

using namespace sf;

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

    // Create a cube
    size_t tri_count = 12;
    Tri3D *tri_test = new Tri3D[tri_count];

    int size = 10;

    // Positive X face
    tri_test[0] = Tri3D(
        Vec3D(size + 0.1, size, size), Vec3D(size + 0.1, -size, size), Vec3D(size + 0.1, -size, -size),
        Vec3D(1, 0, 0), Color3D(255, 0, 0)
    );
    tri_test[1] = Tri3D(
        Vec3D(size + 0.1, size, size), Vec3D(size + 0.1, -size, -size), Vec3D(size + 0.1, size, -size),
        Vec3D(1, 0, 0), Color3D(255, 0, 0)
    );
    // Negative X face
    tri_test[2] = Tri3D(
        Vec3D(-size - 0.1, size, size), Vec3D(-size - 0.1, -size, size), Vec3D(-size - 0.1, -size, -size),
        Vec3D(-1, 0, 0), Color3D(180, 0, 0)
    );
    tri_test[3] = Tri3D(
        Vec3D(-size - 0.1, size, size), Vec3D(-size - 0.1, -size, -size), Vec3D(-size - 0.1, size, -size),
        Vec3D(-1, 0, 0), Color3D(180, 0, 0)
    );

    // Positive Y face
    tri_test[4] = Tri3D(
        Vec3D(size, size + 0.1, size), Vec3D(-size, size + 0.1, size), Vec3D(-size, size + 0.1, -size),
        Vec3D(0, 1, 0), Color3D(0, 255, 0)
    );
    tri_test[5] = Tri3D(
        Vec3D(size, size + 0.1, size), Vec3D(-size, size + 0.1, -size), Vec3D(size, size + 0.1, -size),
        Vec3D(0, 1, 0), Color3D(0, 255, 0)
    );
    // Negative Y face
    tri_test[6] = Tri3D(
        Vec3D(size, -size - 0.1, size), Vec3D(-size, -size - 0.1, size), Vec3D(-size, -size - 0.1, -size),
        Vec3D(0, -1, 0), Color3D(0, 180, 0)
    );
    tri_test[7] = Tri3D(
        Vec3D(size, -size - 0.1, size), Vec3D(-size, -size - 0.1, -size), Vec3D(size, -size - 0.1, -size),
        Vec3D(0, -1, 0), Color3D(0, 180, 0)
    );

    // Positive Z face
    tri_test[8] = Tri3D(
        Vec3D(size, size, size + 0.1), Vec3D(-size, size, size + 0.1), Vec3D(-size, -size, size + 0.1),
        Vec3D(0, 0, 1), Color3D(0, 0, 255)
    );
    tri_test[9] = Tri3D(
        Vec3D(size, size, size + 0.1), Vec3D(-size, -size, size + 0.1), Vec3D(size, -size, size + 0.1),
        Vec3D(0, 0, 1), Color3D(0, 0, 255)
    );
    // Negative Z face
    tri_test[10] = Tri3D(
        Vec3D(size, size, -size - 0.1), Vec3D(-size, size, -size - 0.1), Vec3D(-size, -size, -size - 0.1),
        Vec3D(0, 0, -1), Color3D(0, 0, 180)
    );
    tri_test[11] = Tri3D(
        Vec3D(size, size, -size - 0.1), Vec3D(-size, -size, -size - 0.1), Vec3D(size, -size, -size - 0.1),
        Vec3D(0, 0, -1), Color3D(0, 0, 180)
    );

    // Unrelated stuff
    double rainbowR = 255;
    double rainbowG = 0;
    double rainbowB = 0;
    short cycle = 0;

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

        // Main rendering
        RENDER->reset();
        RENDER->renderGPU(tri_test, tri_count);

        /* I HAVE TO RANT ABOUT THIS
        I CANT DRAW PIXELS IN PARALLEL
        AS SFML DOESNT SUPPORT MULTITHREADING
        */     
        WINDOW.clear(Color::Black);
        for (int i = 0; i < RENDER->BUFFER_SIZE; i++) {
            RectangleShape pixel(Vector2f(RENDER->PIXEL_SIZE, RENDER->PIXEL_SIZE));
            pixel.setPosition(
                RENDER->BUFFER[i].screen.x * RENDER->PIXEL_SIZE,
                RENDER->BUFFER[i].screen.y * RENDER->PIXEL_SIZE
            );

            // Convert Color3D to sf::Color
            Color3D color = RENDER->BUFFER[i].color;
            pixel.setFillColor(Color(
                color.runtimeRGB.v1, color.runtimeRGB.v2, color.runtimeRGB.v3
            ));
            WINDOW.draw(pixel);
        }

        // Draw the log
        CSLOG->drawLog(WINDOW);

        // Display the window
        WINDOW.display();

        // Frame end
        FPS->endFrame();
    }

    // Clean up
    delete FPS, CAM, RENDER, CSLOG;

    delete[] tri_test;

    return 0;
}