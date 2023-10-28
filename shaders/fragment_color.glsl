#version 430 core
#define MAX_ITER INSERT_MAX_ITER
#define MAX_CMAP_SIZE INSERT_MAX_CMAP_SIZE

// IN - From vextex shader
in vec2 fragment_texture_coordinate;

// IN - Buffers
layout(std140, binding=1) uniform cmap {
    vec4[MAX_CMAP_SIZE] cmap_color;
};
layout(std140, binding=2) uniform hist {
    int[MAX_ITER] histogram;
};

// IN - Uniforms
uniform sampler2D iterations_texture;
uniform int num_iter;
uniform int cmap_size;
uniform int hist_sum;

// OUT - Final pixel color
out vec3 fragment_color;


// FUNCTION - Apply colormap with linear interpolation
vec3 ApplyColormap(float iterations)
{
    int cmap_max_i = cmap_size - 1;
    float index = iterations * cmap_max_i;
    int n0 = int(index);
    int n1 = min(n0 + 1, cmap_max_i);
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

    // Histogram recoloring
//    float iterations_norm = 0.0;
//    for(int i = 0; i < int(iterations); i++)
//    {
//        iterations_norm += float(histogram[i]) / float(hist_sum);
//    }

    // Apply colormap
    fragment_color = ApplyColormap(iterations_norm);
}