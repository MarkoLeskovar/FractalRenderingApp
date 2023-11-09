#version 400 core

// IN - VBO
layout (location=0) in vec4 vertex; // <vec2 pos, vec2 tex>

// IN - Uniforms
uniform mat4 proj_mat;

// OUT - Data for fragment shader
out vec2 TexCoords;

void main()
{
    gl_Position = proj_mat * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}