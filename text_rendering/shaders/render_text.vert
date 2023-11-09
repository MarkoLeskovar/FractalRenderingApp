#version 400 core

// IN - VBO
layout (location=0) in vec2 vertex_position;
layout (location=1) in vec2 vertex_texture_coordinate;

// IN - Uniforms
uniform mat4 proj_mat;

// OUT - Data for fragment shader
out vec2 fragment_texture_coordinate;

void main()
{
    gl_Position = proj_mat * vec4(vertex_position, 0.0, 1.0);
    fragment_texture_coordinate = vertex_texture_coordinate;
}