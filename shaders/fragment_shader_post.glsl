#version 440 core

// IN - From vextex shader
in vec2 fragment_texture_coordinate;

// IN - Uniform texture from main fragment shader
uniform sampler2D framebuffer_texture;

// OUT - Final pixel color
out vec3 fragment_color;


// FUNCTION - Main function
void main()
{
    // Get number of iterations
    float iter =  texture(framebuffer_texture, fragment_texture_coordinate).r;
    fragment_color = vec3(iter);
    fragment_color.r = fragment_color.r * 2.0;
}