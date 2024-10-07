#ifndef PLANE3D_CUH
#define PLANE3D_CUH

#include <Vec3D.cuh>

class Plane3D {
public:
    // Ax + By + Cz + D = 0
    double a, b, c, d;
    Vec3D normal;

    __host__ __device__
    Plane3D(double a=0, double b=0, double c=0, double d=0);
    __host__ __device__
    Plane3D(Vec3D v1, Vec3D v2, Vec3D v3);
    __host__ __device__
    Plane3D(Vec3D normal, Vec3D point);
    __host__ __device__
    Plane3D(Vec3D normal, double d);

    // f(x, y, z) = Ax + By + Cz + D
    __host__ __device__
    double equation(Vec3D v);

    // Distance
    __host__ __device__
    double distance(Vec3D v, bool signedDist=false);

    // Intersection
    __host__ __device__
    Vec3D intersection(Vec3D v1, Vec3D v2);
    // !!! Will add intersection for edge later

    // Normal
    __host__ __device__
    static Vec3D findNormal(Vec3D v1, Vec3D v2, Vec3D v3);

    // Debugging
    std::string print();

};
#endif