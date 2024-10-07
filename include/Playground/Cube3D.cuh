#ifndef CUBE3D_CUH
#define CUBE3D_CUH

#include <Tri3D.cuh>

struct CubePhysic {
    Vec3D center;
    Vec3D vertices[8];
};

struct CubeColor {
    Color3D back, left, front, right, top, bottom;

    CubeColor() { // Default color
        back = Color3D(255, 0, 0);
        left = Color3D(0, 255, 0);
        front = Color3D(0, 0, 255);
        right = Color3D(255, 255, 0);
        top = Color3D(0, 255, 255);
        bottom = Color3D(255, 0, 255);
    }
};

class Cube3D {
public:
    // The 6 faces/12 triangles (for rendering)
    Tri3D *tris = new Tri3D[12];

    CubeColor c_color;

    // The physic properties
    // (in the future we wont do the entire Cube3D object)
    // (we just need the physic properties)
    CubePhysic c_physic;

    Cube3D(Vec3D pos, double size, CubeColor color=CubeColor()) {
        c_color = color;

        c_physic.center = pos;

        // Set the 8 vertices of the cube
        c_physic.vertices[0] = Vec3D(-size, -size, -size);
        c_physic.vertices[1] = Vec3D(size, -size, -size);
        c_physic.vertices[2] = Vec3D(size, size, -size);
        c_physic.vertices[3] = Vec3D(-size, size, -size);
        c_physic.vertices[4] = Vec3D(-size, -size, size);
        c_physic.vertices[5] = Vec3D(size, -size, size);
        c_physic.vertices[6] = Vec3D(size, size, size);
        c_physic.vertices[7] = Vec3D(-size, size, size);

        // Every 3 vertices make a face
        // Create a face index map
        int faces[12][3] = {
            {0, 1, 2}, {0, 2, 3}, // Back
            {4, 0, 3}, {4, 3, 7}, // Left
            {5, 4, 7}, {5, 7, 6}, // Front
            {1, 5, 6}, {1, 6, 2}, // Right
            {3, 2, 6}, {3, 6, 7}, // Top
            {4, 5, 1}, {4, 1, 0}  // Bottom
        };

        // Create a normal map (normal has to face outwards)
        Vec3D normals[6] = {
            Vec3D(0, 0, -1), // Back
            Vec3D(-1, 0, 0), // Left
            Vec3D(0, 0, 1), // Front
            Vec3D(1, 0, 0), // Right
            Vec3D(0, 1, 0), // Top
            Vec3D(0, -1, 0) // Bottom
        };

        // Create a color map
        Color3D colors[6] = {
            color.back, color.left, color.front,
            color.right, color.top, color.bottom
        };

        // Create the 12 triangles
        for (int i = 0; i < 12; i++) {
            tris[i] = Tri3D(
                c_physic.vertices[faces[i][0]],
                c_physic.vertices[faces[i][1]],
                c_physic.vertices[faces[i][2]],
                normals[i / 2],
                colors[i / 2]
            );
        }

        // Perform transformations
        for (int i = 0; i < 12; i++) {
            tris[i].translate(c_physic.center);
        }

        for (int i = 0; i < 8; i++) {
            c_physic.vertices[i] = Vec3D::translate(
                c_physic.vertices[i], c_physic.center
            );
        }
    }
};

#endif