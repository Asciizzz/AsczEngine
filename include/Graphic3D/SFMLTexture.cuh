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
    sf::Texture TEXTURE;
    sf::Sprite SPRITE;

    // Allocate memory for the Pixel buffer
    
    sf::Uint8 *SFPIXELS;
    sf::Uint8 *D_SFPIXELS;

    // Set kernel parameters
    const size_t BLOCK_SIZE = 256;
    size_t BLOCK_COUNT;

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