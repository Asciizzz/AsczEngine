#ifndef SFMLTEXTURE_CUH
#define SFMLTEXTURE_CUH

#include <SFML/Graphics.hpp>
#include <Render3D.cuh>

/*
Idea: since you cant just execute draw function in parallel, you can
instead create a texture, fill it with pixels IN PARALLEL, and then
draw the texture to the window. This way, you can utilize the GPU
to fill the pixels, and the CPU to draw the texture.
*/

class SFMLTexture {
public:
    sf::Texture texture;
    sf::Uint8 *pixels;

    sf::Sprite sprite;

    // Allocate memory for the Pixel buffer
    Pixel3D *d_buffer;
    sf::Uint8 *d_sfPixels;

    // Set kernel parameters
    const size_t blockSize = 256;
    size_t numBlocks;

    SFMLTexture(Render3D *render);
    ~SFMLTexture();

    // Update texture
    void updateTexture(Render3D *render);
};

// Create a kernel that would fill in pixel from buffer to sf::Texture
__global__ void fillPixelKernel(
    sf::Uint8 *pixels, Pixel3D *pixel3D,
    int b_w, int b_h, int p_s
);

#endif