#version 430

// INPUT - VBO
layout (location=0) in vec2 vertex_position;
layout (location=1) in vec2 vertex_texture_coordinate;

// OUTPUT - Data for fragment shader
out vec2 fragment_texture_coordinate;


// FUNCTION - Main function
void main()
{
    // Pass-through shader
    gl_Position = vec4(vertex_position, 0.0, 1.0);
    fragment_texture_coordinate = vertex_texture_coordinate;
}