#include <SFMLTexture.cuh>
#include <CsLogHandle.h>

// Playground
#include <Wall.cuh>
#include <Cube3D.cuh>

int main() {
    FpsHandle *FPS = new FpsHandle();
    Camera3D *CAM = new Camera3D();
    Render3D *RENDER = new Render3D(CAM);
    SFMLTexture *TEXTURE = new SFMLTexture(RENDER);
    CAM->w_center_x = RENDER->W_CENTER_X;
    CAM->w_center_y = RENDER->W_CENTER_Y;

    CAM->pos = Vec3D(0, 20, 0);

    // Debugging
    CsLogHandle *CSLOG = new CsLogHandle();

    sf::RenderWindow WINDOW(
        sf::VideoMode(RENDER->W_WIDTH, RENDER->W_HEIGHT), RENDER->W_TITLE
    );
    WINDOW.setMouseCursorVisible(false);

    // =================== EXPERIMENTATION =======================

    // Function y = f(x, z) to create a 3D graph
    std::vector<std::vector<Vec3D>> points;
    std::vector<Tri3D> tris;
    for (double x = -10; x < 10; x += 0.1) {
        points.push_back(std::vector<Vec3D>());
        for (double z = -10; z < 10; z += 0.1) {
            double y = sin(x) * cos(z);
            // double y = 0;

            points.back().push_back(Vec3D(x, y, z));
        }
    }

    for (size_t x = 0; x < points.size() - 1; x++) {
        for (size_t z = 0; z < points[x].size() - 1; z++) {
            double c1 = 50 + 150 * double(x) / points.size();
            double c2 = 50 + 150 * double(z) / points[x].size();

            Color3D color = Color3D(
                c1, 180, c2
            );

            Tri3D tri1 = Tri3D(
                points[x][z], points[x + 1][z], points[x][z + 1],
                color
            );
            Tri3D tri2 = Tri3D(
                points[x][z + 1], points[x + 1][z], points[x + 1][z + 1],
                color
            );

            if (tri1.normal.y < 0) tri1.normal = Vec3D::mult(tri1.normal, -1);
            if (tri2.normal.y < 0) tri2.normal = Vec3D::mult(tri2.normal, -1);

            tris.push_back(tri1);
            tris.push_back(tri2);
        }
    }

    size_t tri_count = tris.size();
    Tri3D *tri_test = new Tri3D[tri_count];

    for (size_t i = 0; i < tri_count; i++) {
        tris[i].v1 = Vec3D::scale(tris[i].v1, Vec3D(), 20);
        tris[i].v2 = Vec3D::scale(tris[i].v2, Vec3D(), 20);
        tris[i].v3 = Vec3D::scale(tris[i].v3, Vec3D(), 20);

        tri_test[i] = tris[i];
    }

    // Unrelated stuff
    double rainbowR = 255;
    double rainbowG = 0;
    double rainbowB = 0;
    short cycle = 0;

    while (WINDOW.isOpen()) {
        // Frame start
        FPS->startFrame();

        // Resets
        CSLOG->clear();
        RENDER->reset();
        WINDOW.clear(sf::Color::White);

    // =================== EVENT HANDLING =======================
        sf::Event event;
        while (WINDOW.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
                WINDOW.close();
            }

            // Press space to toggle focus
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
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

            bool m_left = sf::Mouse::isButtonPressed(sf::Mouse::Left);
            bool m_right = sf::Mouse::isButtonPressed(sf::Mouse::Right);

            // Mouse Click = move forward
            if (m_left && !m_right)      CAM->vel = 1;
            else if (m_right && !m_left) CAM->vel = -1;
            else                         CAM->vel = 0;
        }
        CAM->update();

    // ============== WHERE THE REAL FUN BEGINS =====================   

        // =================== LOGGING =======================

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
        CSLOG->addLog("Welcome to AsczEngine 2.0", rainbow, 1);

        // FPS <= 10: Fully Red
        // FPS >= 60: Fully Green
        double gRatio = double(FPS->fps - 10) / 50;
        gRatio = std::max(0.0, std::min(gRatio, 1.0));
        sf::Color fpsColor((1 - gRatio) * 255, gRatio * 255, 0);
        CSLOG->addLog("FPS: " + std::to_string(FPS->fps), fpsColor);

        CSLOG->addLog(CAM->log, sf::Color::Cyan);

        // ================= Playground ====================

        // Rotate the light source
        RENDER->light.pos = Vec3D::rotate(
            RENDER->light.pos, Vec3D(0, 0, 0),
            Vec3D(M_PI / 6 * FPS->dTimeSec, 0, M_PI / 6 * FPS->dTimeSec)
        );

        // // YOU are the light source
        // RENDER->light.pos = CAM->pos;

        // ======= Main graphic rendering pipeline =======
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

        // Update the texture
        TEXTURE->updateTexture(RENDER);

        // Draw the texture
        WINDOW.draw(TEXTURE->sprite);

        // Draw the log
        CSLOG->drawLog(WINDOW);

        // Display the window
        WINDOW.display();

        // Frame end
        FPS->endFrame();
    }

    delete CAM;
    delete FPS;
    delete CSLOG;
    delete RENDER;
    delete TEXTURE;
    delete[] tri_test;

    return 0;
}