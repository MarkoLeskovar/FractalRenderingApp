#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "mandelbrot.h"


int main()
{	
	// Fractal settings
	Eigen::Vector2d range_x = {-2.0, 1.0};
	Eigen::Vector2d range_y = {-1.5, 1.5};
	Eigen::Vector2i resolution = {1000, 1000};
	int num_iter = 200;

	// Compute iterations count
	//Eigen::MatrixXi iterations = ComputeMandelbrotSet(range_x, range_y, resolution, num_iter);



	// Test function
	std::cout << addTwoInts(1, 2) << std::endl;
    return 0;
}
