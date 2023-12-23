#version 400 core

// IN - VBO
layout(location=0) in vec2 vertex_position;

// IN - Uniforms
uniform mat4 proj_mat;
uniform mat4 trans_mat;

// OUT - For fragment shader
out vec2 texture_coordinate;


// FUNCTION - Main function
void main()
{
    // Set vertex position
    gl_Position = proj_mat * trans_mat * vec4(vertex_position, 0.0, 1.0);

    // Set texture coordinate
    texture_coordinate = vertex_position;
}