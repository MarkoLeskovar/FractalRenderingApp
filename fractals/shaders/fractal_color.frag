#version 400 core

// IN - From vextex shader
in vec2 fragment_texture_coordinate;

// IN - Buffers
layout(std140) uniform cmap {
    vec4[256] cmap_color;
};

// IN - Uniforms
uniform sampler2D iterations_texture;
uniform int num_iter;

// OUT - Final pixel color
out vec4 fragment_color;

// FUNCTION - Apply colormap with linear interpolation
vec3 ApplyColormap(float iterations)
{
    float index = iterations * 255.0;
    int n0 = int(index);
    int n1 = min(n0 + 1, 255);
    float t = mod(index, 1.0);
    return mix(cmap_color[n0].rgb, cmap_color[n1].rgb, t);
}

// FUNCTION - Main function
void main()
{
    // Get number of iterations
    float iterations =  texture(iterations_texture, fragment_texture_coordinate).r;

    // Fractional iteration count [0...1]
    float iterations_norm = iterations / float(num_iter);

    // Apply colormap
    fragment_color = vec4(ApplyColormap(iterations_norm), 1.0);
}