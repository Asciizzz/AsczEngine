#ifndef WALL_CUH
#define WALL_CUH

#include <Tri3D.cuh>

class Wall {
public: 
    // 2 triangles make a rectangle
    Tri3D tri1, tri2;

    Wall(double grid_x, double grid_z, double size, short orientation) {
        double addX = grid_x * size;
        double addZ = grid_z * size;

        tri1 = Tri3D(
            Vec3D(size, size, size),
            Vec3D(0, size, size),
            Vec3D(0, 0, size),
            Vec3D(0, 0, -1),
            Color3D(255, 255, 255)
        );
        tri2 = Tri3D(
            Vec3D(size, size, size),
            Vec3D(size, 0, size),
            Vec3D(0, 0, size),
            Vec3D(0, 0, -1),
            Color3D(255, 255, 255)
        );

        Vec3D center(size / 2, size / 2, size / 2);
        double rot = orientation * M_PI_2;

        tri1.rotate(center, Vec3D(0, rot, 0));
        tri2.rotate(center, Vec3D(0, rot, 0));

        tri1.translate(Vec3D(addX, 0, addZ));
        tri2.translate(Vec3D(addX, 0, addZ));
    }
};

#endif