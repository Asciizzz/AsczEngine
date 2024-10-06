#ifndef CAMERA3D_CUH
#define CAMERA3D_CUH

#include <Tri3D.cuh>

class Camera3D {
public:
    Camera3D() {};

    // For later usage
    int w_center_x, w_center_y;

    bool focus = true;

    // Position and orientation
    Vec3D pos = Vec3D(0, 0, 0);
    Vec3D ang = Vec3D(0, 0, 0); // Axis-angle
    // Speeeeed
    double vel = 0;
    void updatePosition();

    // Plane
    Plane3D plane = Plane3D();
    void updatePlane();

    // FoV
    double fov = 120;
    double screendist;
    void dynamicFov();

    // Fps camera movement
    double m_sens = 0.1;

    // Console log
    std::string log;
    void updateLog();

    // General update
    void update();
};

#endif