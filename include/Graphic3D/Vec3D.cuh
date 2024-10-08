#ifndef VEC3D_CUH
#define VEC3D_CUH

#include <Config.cuh>
#include <Color3D.cuh>

#define M_PI 3.14159265358979323846
#define M_2PI 6.28318530717958647692
#define M_PI_2 1.57079632679489661923

class Vec3D;

class Vec2D {
public:
    double x, y;
    float zDepth;

    // Constructor
    __host__ __device__ Vec2D(double x=0, double y=0, float zDepth=1000) {
        this->x = x;
        this->y = y;
        this->zDepth = zDepth;
    }

    __host__ __device__ static Vec3D barycentricLambda(Vec2D p, Vec2D a, Vec2D b, Vec2D c);
    __host__ __device__ static double barycentricCalc(Vec3D barycentric, double a, double b, double c);
};

class Vec3D {
public:
    double x, y, z;

    // Constructor
    __host__ __device__ Vec3D(double x=0, double y=0, double z=0);

    // Basic operations
    __host__ __device__
    static Vec3D add(const Vec3D& v1, const Vec3D& v2);
    __host__ __device__
    static Vec3D add(const Vec3D& v, double scalar);
    __host__ __device__
    static Vec3D add(const Vec3D* vs, int size);
    __host__ __device__
    static Vec3D sub(const Vec3D& v1, const Vec3D& v2);
    __host__ __device__
    static Vec3D mult(const Vec3D& v1, const Vec3D& v2);
    __host__ __device__
    static Vec3D mult(const Vec3D& v, double scalar);
    __host__ __device__
    static Vec3D absl(const Vec3D& v);

    // Vector operations
    __host__ __device__
    static double dot(const Vec3D& v1, const Vec3D& v2);
    __host__ __device__
    static Vec3D cross(const Vec3D& v1, const Vec3D& v2);

    // Other operations
    __host__ __device__
    static double mag(const Vec3D& v);
    __host__ __device__
    static double dist(const Vec3D& v1, const Vec3D& v2);
    // __host__ __device__
    // static Vec3D baricentric(const Vec3D& p, const Vec3D& v1, const Vec3D& v2, const Vec3D& v3);

    // Transformation
    __host__ __device__
    static Vec3D rotate(const Vec3D& v, const Vec3D& origin, const Vec3D& w);
    __host__ __device__
    static Vec3D scale(const Vec3D& v, const Vec3D& origin, const Vec3D& s);
    __host__ __device__
    static Vec3D scale(const Vec3D& v, const Vec3D& origin, const double s);
    __host__ __device__
    static Vec3D translate(const Vec3D& v, const Vec3D& t);
    __host__ __device__
    static Vec3D translate(const Vec3D& v, const double t);
};

// Kernel for vector transformation
/* Explantion:

The idea is the transformation of polygons can be done in parallel.
Each vertex of the polygon is transformed by the same transformation matrix.
Therefore, we can use CUDA to transform the vertices in parallel.
*/

__global__ void rotateKernel(
    Vec3D* newVs, const Vec3D* oldVs, Vec3D origin, Vec3D w, size_t size
);

__global__ void scaleKernel(
    Vec3D* newVs, const Vec3D* oldVs, Vec3D origin, Vec3D s, size_t size
);

__global__ void translateKernel(
    Vec3D* newVs, const Vec3D* oldVs, Vec3D t, size_t size
);

#endif