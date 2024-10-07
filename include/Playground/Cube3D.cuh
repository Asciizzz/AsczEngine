#ifndef CUBE3D_CUH
#define CUBE3D_CUH

#include <Tri3D.cuh>

class Cube3D {
public:
    Tri3D *tris = new Tri3D[12];
    

    Cube3D(Vec3D center, double size) {

    }
};

#endif