#version 400 core

// IN - VBO
layout (location=0) in vec2 vertex_position;

// IN - Uniforms
uniform mat4 trans_mat;
uniform mat4 proj_mat;

// OUT - Data for fragment shader
out vec2 fragment_texture_coordinate;

void main()
{
    // Set vertex position
    gl_Position = proj_mat * trans_mat * vec4(vertex_position, 0.0, 1.0);

    // Set texture coordinate
    fragment_texture_coordinate = vertex_position;
    fragment_texture_coordinate.y = 1.0 - fragment_texture_coordinate.y;

}