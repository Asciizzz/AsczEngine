#include <Color3D.cuh>

// Custom functions that replace some library functions
#include <iostream>

// Custom implementation of fmod
__host__ __device__ double custom_fmod(double x, double y) {
    // Handle case where y is 0 to avoid division by zero
    if (y == 0.0) {
        return x;  // or some other appropriate value/behavior
    }

    double quotient = x / y;
    // Use truncation to get the integer part
    double integer_part;
    
    // Getting the integer part by truncating the quotient
    if (quotient >= 0) {
        integer_part = static_cast<double>(static_cast<int>(quotient));
    } else {
        integer_part = static_cast<double>(static_cast<int>(quotient) - 1);
    }
    
    // Return the remainder
    return x - (integer_part * y);
}

// ColorVec

__host__ __device__ void ColorVec::mult(double scalar) {
    v1 *= scalar;
    v2 *= scalar;
    v3 *= scalar;
}
__host__ __device__ void ColorVec::restrictRGB() {
    v1 = std::min(255.0, std::max(0.0, v1));
    v2 = std::min(255.0, std::max(0.0, v2));
    v3 = std::min(255.0, std::max(0.0, v3));
}

__host__ __device__ Color3D::Color3D(double r, double g, double b, double a) {
    // RGB
    rawRGB = {r, g, b};
    runtimeRGB = rawRGB;

    // HSL
    rawHSL = toHSL(rawRGB);
    runtimeHSL = rawHSL;

    // Alpha
    alpha = a;
}

__host__ __device__ ColorVec Color3D::toHSL(ColorVec rgb) {
    double r = rgb.v1 / 255;
    double g = rgb.v2 / 255;
    double b = rgb.v3 / 255;

    double max = std::max(r, std::max(g, b));
    double min = std::min(r, std::min(g, b));
    double h, s, l = (max + min) / 2;

    if (max == min) {
        h = s = 0; // achromatic
    } else {
        double d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

        if (max == r) h = (g - b) / d + (g < b ? 6 : 0);
        else if (max == g) h = (b - r) / d + 2;
        else if (max == b) h = (r - g) / d + 4;

        h /= 6;
    }

    return {h, s, l};
}

__host__ __device__ ColorVec Color3D::toRGB(ColorVec hsl) {
    double h = hsl.v1;
    double s = hsl.v2;
    double l = hsl.v3;

    double r, g, b;

    double c = (1 - std::abs(2 * l - 1)) * s;
    double x = c * (1 - std::abs(custom_fmod(h * 6, 2) - 1));
    double m = l - c / 2;

    if (h < 1.0 / 6) { r = c; g = x; b = 0; }
    else if (h < 2.0 / 6) { r = x; g = c; b = 0; }
    else if (h < 3.0 / 6) { r = 0; g = c; b = x; }
    else if (h < 4.0 / 6) { r = 0; g = x; b = c; }
    else if (h < 5.0 / 6) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }

    ColorVec rgb = {r + m, g + m, b + m};
    rgb.mult(255);
    rgb.restrictRGB();

    return rgb;
}

__host__ __device__ ColorVec Color3D::x255(ColorVec vec) {
    return ColorVec{vec.v1 * 255, vec.v2 * 255, vec.v3 * 255};
}