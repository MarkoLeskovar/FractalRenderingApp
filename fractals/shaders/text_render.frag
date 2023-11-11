#version 400 core
#define NUM_INSTANCES INSERT_NUM_INSTANCES

// IN - Uniforms
uniform sampler2DArray characters;

// IN - Buffers
layout(std140) uniform char_id_buffer {
    int[NUM_INSTANCES] char_id;
};
layout(std140) uniform color_buffer {
    vec4[NUM_INSTANCES] color;
};

// IN - From vertex shader
in VS_OUT{
    vec2 texture_coordinate;
    flat int index;
} fs_in;

// OUT - Final pixel color
out vec4 fragment_color;

// FUNCTION - Main function
void main()
{
    // Get character coordinate
    vec3 character_coordinate = vec3(fs_in.texture_coordinate, char_id[fs_in.index]);

    // Sample the texture and get final color
    float sampled_value = texture(characters, character_coordinate).r;
    fragment_color = vec4(color[fs_in.index].rgb, sampled_value);
}