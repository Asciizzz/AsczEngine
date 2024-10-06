#ifndef PLANE3D_CUH
#define PLANE3D_CUH

#include <Vec3D.cuh>

class Plane3D {
public:
    // Ax + By + Cz + D = 0
    double a, b, c, d;
    Vec3D normal;

    Plane3D(double a=0, double b=0, double c=0, double d=0);
    Plane3D(Vec3D v1, Vec3D v2, Vec3D v3);
    Plane3D(Vec3D normal, Vec3D point);
    Plane3D(Vec3D normal, double d);

    // f(x, y, z) = Ax + By + Cz + D
    double equation(Vec3D v);

    // Distance
    double distance(Vec3D v, bool signedDist=false);

    // Intersection
    Vec3D intersection(Vec3D v1, Vec3D v2);
    // !!! Will add intersection for edge later

    // Normal
    static Vec3D findNormal(Vec3D v1, Vec3D v2, Vec3D v3);

    // Debugging
    std::string print();

};
#endif