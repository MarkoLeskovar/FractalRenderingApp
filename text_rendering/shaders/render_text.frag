#version 400 core

// IN - From vextex shader
in vec2 TexCoords;

// IN - Uniforms
uniform sampler2D text;
uniform vec3 textColor;

// OUT - Final pixel color
out vec4 fragment_color;

void main()
{
    vec4 sampled = vec4(textColor, texture(text, TexCoords).r);
    fragment_color = vec4(textColor, 1.0) * sampled;
}