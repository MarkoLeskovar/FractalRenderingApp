#version 400 core
#define NUM_INSTANCES INSERT_NUM_INSTANCES

// IN - VBO
layout(location=0) in vec2 vertex_position;

// IN - Uniforms
uniform mat4 proj_mat;

// IN - Buffers
layout(std140) uniform trans_mat_buffer {
    mat4[NUM_INSTANCES] trans_mat;
};

// OUT - Data for fragment shader
out VS_OUT{
    vec2 texture_coordinate;
    flat int index;
} vs_out;

// FUNCTION - Main function
void main()
{
    // Set vertex position
    gl_Position = proj_mat * trans_mat[gl_InstanceID] * vec4(vertex_position, 0.0, 1.0);
    vs_out.index = gl_InstanceID;

    // Set texture coordinate
    vs_out.texture_coordinate = vertex_position;
    vs_out.texture_coordinate.y = 1.0 - vs_out.texture_coordinate.y;

}