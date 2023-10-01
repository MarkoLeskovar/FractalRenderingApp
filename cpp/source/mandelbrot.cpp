#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include "config.h"
#include "mandelbrot.h"


Eigen::MatrixXi ComputeMandelbrotSet(const Eigen::Vector2d &range_x,
                                     const Eigen::Vector2d &range_y,
                                     const Eigen::Vector2i &resolution,
                                     const int &max_iter)
{

	// Compute image physical dimenstions
	double pix_size_x = (range_x(1) - range_x(0)) / (double)resolution(0);
	double pix_size_y = (range_y(1) - range_y(0)) / (double)resolution(1);

	// Initialize the output image
	Eigen::MatrixXi img_output(resolution(0), resolution(1));

	// Loop over each pixel
	# pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < resolution(1); i++) {
		double y0 = range_y(0) + 0.5 * pix_size_y + i * pix_size_y;
		for (int j = 0; j < resolution(0); j++) {
			double x0 = range_x(0) + 0.5 * pix_size_x + j * pix_size_x;

			// Evaluate number of iterations
			double x = 0.0;
			double y = 0.0;
			double x2 = 0.0;
			double y2 = 0.0;
			int num_iter = 0;
			while ((x2 + y2 < 4.0) && (num_iter <= max_iter)) {
				y = 2 * x * y + y0;
				x = x2 - y2 + x0;
				x2 = x * x;
				y2 = y * y;
				num_iter++;
			}
			// Assign iterations to correct pixel
			img_output(j, i) = num_iter;
		}
	}
	return img_output;
}
