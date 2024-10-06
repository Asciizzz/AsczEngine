#include <Plane3D.cuh>

// Constructors

Plane3D::Plane3D(double a, double b, double c, double d) {
    this->a = a;
    this->b = b;
    this->c = c;
    this->d = d;
    this->normal = Vec3D(a, b, c);
}
Plane3D::Plane3D(Vec3D normal, Vec3D point) {
    this->a = normal.x;
    this->b = normal.y;
    this->c = normal.z;
    this->d = -Vec3D::dot(normal, point);
    this->normal = normal;
}
Plane3D::Plane3D(Vec3D normal, double d) {
    this->a = normal.x;
    this->b = normal.y;
    this->c = normal.z;
    this->d = d;
    this->normal = normal;
}
Plane3D::Plane3D(Vec3D v1, Vec3D v2, Vec3D v3) {
    Vec3D normal = findNormal(v1, v2, v3);
    this->a = normal.x;
    this->b = normal.y;
    this->c = normal.z;
    this->d = -Vec3D::dot(normal, v1);
    this->normal = normal;
}

// f(x, y, z) = Ax + By + Cz + D
double Plane3D::equation(Vec3D v) {
    return a * v.x + b * v.y + c * v.z + d;
}

// Distance
double Plane3D::distance(Vec3D v, bool signedDist) {
    double dist = equation(v) / Vec3D::mag(normal);
    return signedDist ? dist : abs(dist);
}

// Intersection
Vec3D Plane3D::intersection(Vec3D v1, Vec3D v2) {
    Vec3D diff = Vec3D::sub(v2, v1);
    double ev1 = equation(v1);

    double t = -ev1 / Vec3D::dot(normal, diff);

    return Vec3D::add(v1, Vec3D::mult(diff, t));
}

// Normal
Vec3D Plane3D::findNormal(Vec3D v1, Vec3D v2, Vec3D v3) {
    return Vec3D::cross(
        Vec3D::sub(v2, v1),
        Vec3D::sub(v3, v1)
    );
}

// Debugging 
std::string Plane3D::print() {
    std::string str = "";
    str += "(" + std::to_string(a) + ")x + ";
    str += "(" + std::to_string(b) + ")y + ";
    str += "(" + std::to_string(c) + ")z + ";
    str += "(" + std::to_string(d) + ") = 0";

    return str;
}