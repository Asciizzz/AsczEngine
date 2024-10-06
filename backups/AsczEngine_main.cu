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
    int tri_count = 1 << 12;
    Tri3D *tri_test = new Tri3D[tri_count];

    for (int i = 0; i < tri_count; i++) {
        tri_test[i] = Tri3D(
            Vec3D(i, i, 15),
            Vec3D(i, i + 10, 15),
            Vec3D(i + 10, i, 15),
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

        // // Draw the buffer
        // for (int i = 0; i < RENDER->BUFFER.size(); i++) {
        //     int x = i % RENDER->W_WIDTH;
        //     int y = i / RENDER->W_WIDTH;
        //     RectangleShape pixel(Vector2f(RENDER->PIXEL_SIZE, RENDER->PIXEL_SIZE));
        //     pixel.setPosition(x * RENDER->PIXEL_SIZE, y * RENDER->PIXEL_SIZE);

        //     // Convert Color3D to sf::Color
        //     Color3D color = RENDER->BUFFER[i];
        //     pixel.setFillColor(Color(
        //         color.rawRGB.v1 * 255,
        //         color.rawRGB.v2 * 255,
        //         color.rawRGB.v3 * 255
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
        }

        WINDOW.display();
        // Frame end
        FPS->endFrame();
    }

    // Clean up
    delete FPS, CAM, RENDER;

    return 0;
}