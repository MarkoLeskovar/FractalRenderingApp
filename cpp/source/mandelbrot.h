#ifndef MANDELBROT
#define MANDELBROT

#include <vector>
#include <Eigen/Dense>


Eigen::MatrixXi ComputeMandelbrotSet(const Eigen::Vector2d &range_x,
                                     const Eigen::Vector2d &range_y,
                                     const Eigen::Vector2i &resolution,
                                     const int &max_iter);

#endif // MANDELBROT