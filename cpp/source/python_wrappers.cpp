#include <omp.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

// CMake config file
#include "config.h"

// Custom functions
#include "mandelbrot.h"

namespace py = pybind11;
PYBIND11_MODULE(CMAKE_MODULE_NAME, m)
{
	m.doc() = "Python wrappers for geometry processing C++ functions";

	
	m.def("omp_GetMaxThreads", &omp_get_max_threads, "Returns max number of threads");

	m.def("omp_SetNumThreads", &omp_set_num_threads, "Set number of threads",
			py::arg("num_threads"));

	m.def("addTwoInts", &addTwoInts, "Add two interger numbers",
			py::arg("a"),
			py::arg("b"));

}              
