#ifndef COLOR3D_CUH
#define COLOR3D_CUH

#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <algorithm>

struct ColorVec {
    // In range 0-1
    double v1, v2, v3;
    __host__ __device__ void mult(double scalar);
    __host__ __device__ void restrictRGB();
};

class Color3D {
public: 
    // Unaffected Values
    ColorVec rawRGB;
    ColorVec rawHSL;
    double alpha = 1;

    // Dynamic Color (shading, lighting, etc)
    ColorVec runtimeRGB;
    ColorVec runtimeHSL;

    // Keep in mind the isDouble is rarely used
    __host__ __device__ Color3D(double r=0, double g=0, double b=0, double a=1);

    // Conversion
    __host__ __device__ static ColorVec toHSL(ColorVec rgb);
    __host__ __device__ static ColorVec toRGB(ColorVec hsl);
    __host__ __device__ static ColorVec x255(ColorVec vec);

    // Fun stuff
    __host__ __device__ static Color3D random();
};


#endif