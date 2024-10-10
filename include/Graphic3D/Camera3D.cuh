#ifndef CAMERA3D_CUH
#define CAMERA3D_CUH

#include <Tri3D.cuh>

class Camera3D {
public:
    Camera3D() {};

    // For later usage
    int w_center_x, w_center_y;
    int w_width, w_height;

    bool focus = true;

    // Position and orientation
    Vec3D pos = Vec3D(0, 0, 0);
    Vec3D ang = Vec3D(0, 0, 0); // Axis-angle
    // Speeeeed
    double vel = 0;
    void updatePosition();

    // FoV
    double fov = 90;
    double screendist;
    void dynamicFov();
    // Fps camera movement
    double m_sens = 0.1;

    // Points and plane
    Plane3D plane = Plane3D();
    void updatePlane();
    void angleRestrict();
    void facePoint(Vec3D target);

    // Console log
    std::string log;
    void updateLog();

    // General update
    void update();
};

#endif