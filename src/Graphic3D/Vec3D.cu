#include <Vec3D.cuh>

// ========================== Vec2D ==========================
__host__ __device__ Vec3D Vec2D::barycentricLambda(Vec2D p, Vec2D a, Vec2D b, Vec2D c) {
    double detT = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    double l1 = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / detT;
    double l2 = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / detT;
    double l3 = 1 - l1 - l2;

    return Vec3D(l1, l2, l3);
}

__host__ __device__ double Vec2D::barycentricCalc(Vec3D barycentric, double a, double b, double c) {
    return barycentric.x * a + barycentric.y * b + barycentric.z * c;
}

// ========================== Vec3D ==========================

__host__ __device__ Vec3D::Vec3D(double x, double y, double z) : x(x), y(y), z(z) {}

// Basic operations
__host__ __device__ Vec3D Vec3D::add(const Vec3D& v1, const Vec3D& v2) {
    return Vec3D(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
__host__ __device__ Vec3D Vec3D::add(const Vec3D& v, double scalar) {
    return Vec3D(v.x + scalar, v.y + scalar, v.z + scalar);
}
__host__ __device__ Vec3D Vec3D::add(const Vec3D* vs, int size) {
    Vec3D result(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < size; ++i) {
        result = add(result, vs[i]);
    }
    return result;
}

__host__ __device__ Vec3D Vec3D::sub(const Vec3D& v1, const Vec3D& v2) {
    return Vec3D(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ Vec3D Vec3D::mult(const Vec3D& v1, const Vec3D& v2) {
    return Vec3D(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
__host__ __device__ Vec3D Vec3D::mult(const Vec3D& v, double scalar) {
    return Vec3D(v.x * scalar, v.y * scalar, v.z * scalar);
}
__host__ __device__ Vec3D Vec3D::absl(const Vec3D& v) {
    return Vec3D(abs(v.x), abs(v.y), abs(v.z));
}

// Vector operations
__host__ __device__ double Vec3D::dot(const Vec3D& v1, const Vec3D& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ Vec3D Vec3D::cross(const Vec3D& v1, const Vec3D& v2) {
    return Vec3D(v1.y * v2.z - v1.z * v2.y,
                 v1.z * v2.x - v1.x * v2.z,
                 v1.x * v2.y - v1.y * v2.x);
}

// Other operations
__host__ __device__ double Vec3D::mag(const Vec3D& v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ double Vec3D::dist(const Vec3D& v1, const Vec3D& v2) {
    return mag(sub(v1, v2));
}
// __host__ __device__ Vec3D Vec3D::baricentric(const Vec3D& p, const Vec3D& v1, const Vec3D& v2, const Vec3D& v3) {
//     // Get the lambda values and return them as a Vec3D

//     // Check for parallelism with the axes
//     if (v1.x == v2.x && v2.x == v3.x) {
//         return Vec2D::barycentricLambda(
//             Vec2D(p.y, p.z), Vec2D(v1.y, v1.z), Vec2D(v2.y, v2.z), Vec2D(v3.y, v3.z)
//         );
//     }
//     if (v1.y == v2.y && v2.y == v3.y) {
//         return Vec2D::barycentricLambda(
//             Vec2D(p.x, p.z), Vec2D(v1.x, v1.z), Vec2D(v2.x, v2.z), Vec2D(v3.x, v3.z)
//         );
//     }
//     if (v1.z == v2.z && v2.z == v3.z) {
//         return Vec2D::barycentricLambda(
//             Vec2D(p.x, p.y), Vec2D(v1.x, v1.y), Vec2D(v2.x, v2.y), Vec2D(v3.x, v3.y)
//         );
//     }
//     return Vec2D::barycentricLambda(
//         Vec2D(p.x, p.y), Vec2D(v1.x, v1.y), Vec2D(v2.x, v2.y), Vec2D(v3.x, v3.y)
//     );
// }

// Transformation
__host__ __device__ Vec3D Vec3D::rotate(const Vec3D& v, const Vec3D& origin, const Vec3D& w) {
    Vec3D diff = sub(v, origin);

    // Rotate around x-axis
    double rot_yx = diff.y * cos(w.x) - diff.z * sin(w.x);
    double rot_zx = diff.y * sin(w.x) + diff.z * cos(w.x);

    // Rotate around y-axis
    double rot_xz = diff.x * cos(w.y) - rot_zx * sin(w.y);
    double rot_zz = diff.x * sin(w.y) + rot_zx * cos(w.y);

    // Rotate around z-axis
    double rot_xx = rot_xz * cos(w.z) - rot_yx * sin(w.z);
    double rot_yy = rot_xz * sin(w.z) + rot_yx * cos(w.z);

    return add(Vec3D(rot_xx, rot_yy, rot_zz), origin);
}

__host__ __device__ Vec3D Vec3D::scale(const Vec3D& v, const Vec3D& origin, const Vec3D& s) {
    Vec3D diff = sub(v, origin);
    return Vec3D(add(mult(diff, s), origin)
    );
}
__host__ __device__ Vec3D Vec3D::scale(const Vec3D& v, const Vec3D& origin, const double s) {
    Vec3D diff = sub(v, origin);
    return add(mult(diff, s), origin);
}

__host__ __device__ Vec3D Vec3D::translate(const Vec3D& v, const Vec3D& t) {
    return add(v, t);
}
__host__ __device__ Vec3D Vec3D::translate(const Vec3D& v, const double t) {
    return add(v, Vec3D(t, t, t));
}

// ======================= Transformation Kernels =======================

__global__ void rotateKernel(
    Vec3D* newVs, const Vec3D* oldVs, Vec3D origin, Vec3D w, size_t size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        newVs[i] = Vec3D::rotate(oldVs[i], origin, w);
    }
}

__global__ void scaleKernel(
    Vec3D* newVs, const Vec3D* oldVs, Vec3D origin, Vec3D s, size_t size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        newVs[i] = Vec3D::scale(oldVs[i], origin, s);
    }
}

__global__ void translateKernel(
    Vec3D* newVs, const Vec3D* oldVs, Vec3D t, size_t size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        newVs[i] = Vec3D::translate(oldVs[i], t);
    }
}