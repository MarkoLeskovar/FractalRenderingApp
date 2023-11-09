#version 400 core

// IN - From vextex shader
in vec2 fragment_texture_coordinate;

// IN - Uniforms
uniform sampler2D text;
uniform vec3 text_color;

// OUT - Final pixel color
out vec4 fragment_color;

void main()
{
    vec4 sampled = vec4(text_color, texture(text, fragment_texture_coordinate).r);
    fragment_color = vec4(text_color, 1.0) * sampled;
}