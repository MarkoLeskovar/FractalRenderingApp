#version 440

// IN - Fragment coordinate
layout(pixel_center_integer) in vec4 gl_FragCoord;

// IN - SSBO
layout(std140, binding=1) uniform TriangleParameters {
    vec4 color_scale;
    vec4 triangle_scale;
    vec4 triangle_color;
} triangle_parameters;

// IN - Fractal settings
uniform dvec2 pix_size;
uniform dvec2 range_x;
uniform dvec2 range_y;
uniform int max_iter;

// OUT - Fractal color
out vec4 frag_color;


// Compute physical coordinate of the pixel
dvec2 GetPixelCoordinate(dvec2 pix_size, dvec2 range_x, dvec2 range_y)
{
    double x0 = range_x[0] + 0.5 * pix_size[0] + gl_FragCoord.x * pix_size[0];
    double y0 = range_y[0] + 0.5 * pix_size[0] + gl_FragCoord.y * pix_size[1];
    return dvec2(x0, y0);
}


// Compute number of iterations
int GetIterationsMandelbrotSet(dvec2 pix_coord, int max_iter)
{
    // Initialize variables
    double x = 0.0;
    double y = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    int iter = 0;
    // Evaluate number of iterations
    while ((x2 + y2 <= 4.0) && (iter <= max_iter))
    {
        y = 2 * x * y + pix_coord.y;
    	x = x2 - y2 + pix_coord.x;
    	x2 = x * x;
        y2 = y * y;
        iter += 1;
    }
    // Return iterations
    return iter;
}


void main()
{
    // Compute pixel coordinate
    dvec2 pixel_coordinate = GetPixelCoordinate(pix_size, range_x, range_y);

    // Compute iteration count
    int num_iter = GetIterationsMandelbrotSet(pixel_coordinate, max_iter);

    // Fractional color
    float color = float(num_iter) / float(max_iter);

    // Get output color
    frag_color = vec4(vec3(color), 1.0);
}