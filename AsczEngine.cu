#include <SFMLTexture.cuh>
#include <CsLogHandle.h>

// Playground
#include <Wall.cuh>
#include <Cube3D.cuh>

int main() {

    // =================== INITIALIZATION =======================
    // Pixel size 4 is the sweet spot for performance and quality

    FpsHandle *FPS = new FpsHandle();
    Camera3D *CAM = new Camera3D();
    Render3D *RENDER = new Render3D(CAM, 1600, 900, 4);
    SFMLTexture *TEXTURE = new SFMLTexture(RENDER);

    // Debugging
    CsLogHandle *CSLOG = new CsLogHandle();

    sf::RenderWindow WINDOW(
        sf::VideoMode(RENDER->W_WIDTH, RENDER->W_HEIGHT), RENDER->W_TITLE
    );
    WINDOW.setMouseCursorVisible(false);

    // =================== EXPERIMENTATION =======================

    // Initialize stuff
    CAM->pos = Vec3D(0, 90, -120);
    CAM->ang = Vec3D(0, 0, 0);

    RENDER->DEFAULT_COLOR = Color3D(0, 0, 0);
    RENDER->LIGHT.pos = Vec3D(75, 140, 75);

    std::vector<Tri3D> TRI_VEC;

    std::vector<Tri3D> MODEL_1 = Tri3D::readObj(
        "assets/Models/Sukuna.obj"
    );
    for (int i = 0; i < MODEL_1.size(); i++) {
        MODEL_1[i].scale(Vec3D(), Vec3D(10, 10, 10));
        MODEL_1[i].color = Color3D(255, 255, 255);
        MODEL_1[i].isTwoSided = true;
        TRI_VEC.push_back(MODEL_1[i]);
    }

    // Function y = f(x, z) to create a 3D graph
    std::vector<std::vector<Vec3D>> points;
    for (double x = -10; x < 10; x += 0.1) {
        points.push_back(std::vector<Vec3D>());
        for (double z = -10; z < 10; z += 0.1) {
            double y = 0;

            points.back().push_back(Vec3D(x, y, z));
        }
    }

    for (size_t x = 0; x < points.size() - 1; x++) {
        for (size_t z = 0; z < points[x].size() - 1; z++) {
            double cx = 50 + 150 * double(x) / points.size();
            double cz = 50 + 150 * double(z) / points[x].size();
            double csqrt = 255 * sqrt((cx*cx + cz*cz) / 65025);
            Color3D color = Color3D(232, 211, 139 + 20 * double(x) / points.size());
            color = Color3D(255, 255, 255);

            Tri3D tri1 = Tri3D(
                points[x][z], points[x + 1][z], points[x][z + 1],
                color
            );
            Tri3D tri2 = Tri3D(
                points[x][z + 1], points[x + 1][z], points[x + 1][z + 1],
                color
            );

            tri1.scale(Vec3D(), Vec3D(20, 20, 20));
            tri2.scale(Vec3D(), Vec3D(20, 20, 20));

            if (tri1.normal.y < 0) tri1.normal = Vec3D::mult(tri1.normal, -1);
            if (tri2.normal.y < 0) tri2.normal = Vec3D::mult(tri2.normal, -1);

            TRI_VEC.push_back(tri1);
            TRI_VEC.push_back(tri2);
        }
    }

    size_t tri_count = TRI_VEC.size();
    Tri3D *tri_test = new Tri3D[tri_count];
    RENDER->mallocTris(tri_count);

    for (int i = 0; i < tri_count; i++)
        tri_test[i] = TRI_VEC[i];

    // Unrelated stuff
    double rainbowR = 255;
    double rainbowG = 0;
    double rainbowB = 0;
    short cycle = 0;

    while (WINDOW.isOpen()) {
        FPS->startFrame();

        // Resets
        CSLOG->clear();

    // =================== EVENT HANDLING =======================
        sf::Event event;
        while (WINDOW.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
                WINDOW.close();
            }

            // Scroll to change fov 
            if (event.type == sf::Event::MouseWheelScrolled) {
                if (event.mouseWheelScroll.delta > 0)
                    CAM->fov -= 5; // Zoom in
                else
                    CAM->fov += 5; // Zoom out

                CAM->fov = std::max(10.0, std::min(CAM->fov, 170.0));
            }

            // Press space to face (0, 0, 0)
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
                CAM->facePoint(Vec3D(0, 0, 0));
            }

            // F1 to toggle focus
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::F1)) {
                CAM->focus = !CAM->focus;
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

        sf::Color rainbow = sf::Color(rainbowR, rainbowG, rainbowB);
        CSLOG->addLog("Welcome to AsczEngine 2.0", rainbow, 1);

        // FPS <= 10: Fully Red
        // FPS >= 60: Fully Green
        double gRatio = double(FPS->fps - 10) / 50;
        gRatio = std::max(0.0, std::min(gRatio, 1.0));
        sf::Color fpsColor((1 - gRatio) * 255, gRatio * 255, 0);
        CSLOG->addLog("FPS: " + std::to_string(FPS->fps), fpsColor);

        std::string add = ""; // For million and billion
        size_t displayNum = tri_count;
        if (displayNum > 1'000'000'000) {
            displayNum /= 1'000'000'000; add = "B";
        } else if (displayNum > 1'000'000) {
            displayNum /= 1'000'000; add = "M";
        }
        CSLOG->addLog("TRI_COUNT: " + std::to_string(displayNum) + add, sf::Color::Yellow);

        CSLOG->addLog(CAM->log, sf::Color::Red);

        // ================= Playground ====================

        // Rotate the light source
        RENDER->LIGHT.pos = Vec3D::rotate(
            RENDER->LIGHT.pos, Vec3D(0, 0, 0),
            // Vec3D(M_PI / 6 * FPS->dTimeSec, 0, M_PI / 6 * FPS->dTimeSec)
            Vec3D(0, M_PI / 6 * FPS->dTimeSec, 0)
        );

        // // YOU are the light source
        // RENDER->LIGHT.pos = CAM->pos;
        // RENDER->LIGHT.normal = CAM->plane.normal;

        // ======= Main graphic rendering pipeline =======
        RENDER->renderGPU(tri_test, tri_count);

        TEXTURE->updateTexture(RENDER);
        WINDOW.draw(TEXTURE->sprite);
        CSLOG->drawLog(WINDOW);
        WINDOW.display();

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