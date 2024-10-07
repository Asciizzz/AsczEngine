#include <Camera3D.cuh>

void Camera3D::updatePosition() {
    Vec3D move = Vec3D::mult(plane.normal, vel);
    pos = Vec3D::add(pos, move);
}

void Camera3D::dynamicFov() {
    // Update screen distance
    screendist = (w_center_x / 2) / tan(fov * M_PI / 360);

    // More fov features to come
    /* List of ideas:
    - Fov change with velocity
    - Fov change with status effect (drunk, nauseous, etc.)
    - Fov for scope weapon (cs:go)
    */
}

void Camera3D::updatePlane() {
    Vec3D normal(
        cos(ang.x) * sin(ang.y),
        sin(ang.x),
        cos(ang.x) * cos(ang.y)
    );
    double d = -Vec3D::dot(normal, pos) - 1;

    plane = Plane3D(normal, d);
}

void Camera3D::angleRestrict() {
    if (ang.x > M_PI_2) ang.x = M_PI_2;
    if (ang.x < -M_PI_2) ang.x = -M_PI_2;

    if (ang.y > M_2PI) ang.y -= M_2PI;
    if (ang.y < 0) ang.y += M_2PI;
}

void Camera3D::facePoint(Vec3D target) {
    Vec3D diff = Vec3D::sub(target, pos);

    double xzLen = sqrt(diff.x * diff.x + diff.z * diff.z);

    ang.x = atan2(diff.y, xzLen);
    ang.y = atan2(diff.z, diff.x);
}

void Camera3D::updateLog() {
    log = "Camera3D\n";
    log += "| Pos: " + std::to_string(pos.x) + ", " + std::to_string(pos.y) + ", " + std::to_string(pos.z) + "\n";
    log += "| Ang: " + std::to_string(ang.x) + ", " + std::to_string(ang.y) + ", " + std::to_string(ang.z) + "\n";
    log += "| Fov: " + std::to_string(fov) + "\n";
    log += "| Vel: " + std::to_string(vel) + "\n";
    log += "| Pln: " + plane.print();
}

void Camera3D::update() {
    updatePlane();
    updatePosition();
    dynamicFov();

    updateLog();
}