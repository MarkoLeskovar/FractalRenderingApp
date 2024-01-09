#version 400 core

// IN - Uniforms
uniform sampler2D input_texture;

// IN - From vertex shader
in vec2 texture_coordinate;

// OUT - Final pixel color
out vec4 fragment_color;

// FUNCTION - Main function
void main()
{
    // Sample the texture and get final color
    fragment_color = texture(input_texture, texture_coordinate);
}