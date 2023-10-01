#include <Eigen/Dense>
#include <pybind11/pybind11.h>

// CMake config file
#include "config.h"

// Custom functions
#include "mandelbrot.h"

int addTwoInts(const int &a, const int &b)
{
	return a + b;
}
