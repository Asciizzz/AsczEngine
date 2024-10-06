#ifndef TRI3D_CUH
#define TRI3D_CUH

/* A really important note about triang's normal

There are 2 types: triang of an object and triang of a... wall?

An object is made of multiple flat triangs
To apply lighting, we need to FIND the normal of each triang

Every triang of the object will only have one side revealed
And that side will be used to calculate the lighting
The normals are facing outwards of the object to that side
The other side is irrelevant as its the equivalent of peeling your skin off

However for a wall, both side CAN be revealed

Therefore we need an extra attribute to determine
whether the triang is a flat or an object

*/

#include <Plane3D.cuh>
#include <vector>

struct Tri2D {
    Vec2D v1, v2, v3;

    Tri2D() {};
};

class Tri3D {
public:
    Vec3D v1, v2, v3;
    Color3D color;
    Vec3D normal;
    bool isTwoSided;

    // Default constructor
    Tri3D() {};
    // A flat triangle (free-floating)
    Tri3D(Vec3D v1, Vec3D v2, Vec3D v3, Color3D color=Color3D(), bool isTwoSided=true);
    // A one-sided triangle (belonging to an object)
    Tri3D(Vec3D v1, Vec3D v2, Vec3D v3, Vec3D normal=Vec3D(), Color3D color=Color3D());

    // BETA!
    static std::vector<Tri3D> readObj(std::string filename);
};

#endif