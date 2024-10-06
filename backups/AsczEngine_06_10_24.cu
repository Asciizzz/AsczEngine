#include <Render3D.cuh>
#include <SFML/Graphics.hpp>

using namespace sf;

int main() {
    FpsHandle *FPS = new FpsHandle();
    Camera3D *CAM = new Camera3D();
    Render3D *RENDER = new Render3D(CAM);
    CAM->w_center_x = RENDER->W_CENTER_X;
    CAM->w_center_y = RENDER->W_CENTER_Y;

    RenderWindow WINDOW(
        VideoMode(RENDER->W_WIDTH, RENDER->W_HEIGHT), RENDER->W_TITLE
    );
    WINDOW.setMouseCursorVisible(false);

    // Triangles test (create 1 << 24)
    int tri_count = 1 << 4;
    Tri3D *tri_test = new Tri3D[tri_count];

    for (int i = 0; i < tri_count; i++) {
        tri_test[i] = Tri3D(
            Vec3D(0, 0, 15),
            Vec3D(0, 10, 15),
            Vec3D(10, 0, 15),
            Color3D(240, 127, 139)
        );
    }

    while (WINDOW.isOpen()) {
        // Frame start
        FPS->startFrame();
        RENDER->reset();
        WINDOW.clear(Color::Black);

        CAM->update();

        RENDER->renderGPU(tri_test, tri_count);

        // Draw the buffer
        /* I HAVE TO RANT ABOUT THIS

        I CANT DRAW PIXELS IN PARALLEL

        AS SFML DOESNT SUPPORT MULTITHREADING
        */
        // for (int i = 0; i < RENDER->BUFFER_SIZE; i++) {
        //     RectangleShape pixel(Vector2f(RENDER->PIXEL_SIZE, RENDER->PIXEL_SIZE));
        //     pixel.setPosition(
        //         RENDER->BUFFER[i].screen.x * RENDER->PIXEL_SIZE,
        //         RENDER->BUFFER[i].screen.y * RENDER->PIXEL_SIZE
        //     );

        //     // Convert Color3D to sf::Color
        //     Color3D color = RENDER->BUFFER[i].color;
        //     pixel.setFillColor(Color(
        //         color.runtimeRGB.v1, color.runtimeRGB.v2, color.runtimeRGB.v3
        //     ));
        //     WINDOW.draw(pixel);
        // }

        std::cout << "FPS: " << FPS->fps << std::endl;

        Event event;
        while (WINDOW.pollEvent(event)) {
            if (event.type == Event::Closed ||
                Keyboard::isKeyPressed(Keyboard::Escape)) {
                WINDOW.close();
            }

            // Look up and down
            if (Keyboard::isKeyPressed(Keyboard::Up)) {
                CAM->ang.x += CAM->m_sens;
            }
            if (Keyboard::isKeyPressed(Keyboard::Down)) {
                CAM->ang.x -= CAM->m_sens;
            }

            // Look left and right
            if (Keyboard::isKeyPressed(Keyboard::Left)) {
                CAM->ang.y += CAM->m_sens;
            }
            if (Keyboard::isKeyPressed(Keyboard::Right)) {
                CAM->ang.y -= CAM->m_sens;
            }

            // Press space to toggle focus
            if (Keyboard::isKeyPressed(Keyboard::Space)) {
                CAM->focus = !CAM->focus;

                // Hide/unhide cursor
                WINDOW.setMouseCursorVisible(!CAM->focus);
            }
        }

        // Mouse movement handling
        if (CAM->focus) {
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
        }

        WINDOW.display();
        // Frame end
        FPS->endFrame();
    }

    // Clean up
    delete FPS, CAM, RENDER;

    delete[] tri_test;

    return 0;
}