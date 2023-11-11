#version 400 core

// IN - From vertex shader
in VS_OUT{
    vec2 texture_coordinate;
    flat int index;
} fs_in;

// IN - Uniforms
uniform sampler2DArray characters;
uniform int character_ids[128];
uniform vec3 text_color;

// OUT - Final pixel color
out vec4 fragment_color;

void main()
{
    vec3 character_coordinate = vec3(fs_in.texture_coordinate, character_ids[fs_in.index]);
    float sampled_value = texture(characters, character_coordinate).r;
    fragment_color = vec4(text_color, sampled_value);
}