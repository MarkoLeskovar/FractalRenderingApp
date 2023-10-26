#version 400 core

// IN - From vextex shader
in vec2 fragment_texture_coordinate;

// IN - Uniform texture from main fragment shader
uniform sampler2D framebuffer_texture;

// IN - Fractal settings
uniform int max_iter;

// OUT - Final pixel color
out vec3 fragment_color;


// FUNCTION - Main function
void main()
{
    // Get number of iterations
    float iterations =  texture(framebuffer_texture, fragment_texture_coordinate).r;

    // Fractional color
    iterations = iterations / float(max_iter);

    // Fractal color
    float r = sin(iterations);
    float g = sin(iterations * 6.0);
    float b = sin(iterations * 3.0);
    fragment_color = 0.5 * vec3(r, g, b) + 0.5;
}