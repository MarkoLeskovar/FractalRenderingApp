import numba
import numpy as np

# Define constants
_LOG_2 = np.log(2.0)

'''
O-------------------------------------------------------------------------------O
| EVALUATE FRACTAL POINT                                                        |
O-------------------------------------------------------------------------------O
'''


@numba.njit(cache=True)
def _evaluate_point(x, y, c_x, c_y, max_iter):
    x2 = x * x
    y2 = y * y
    num_iter = 0
    # Computer fractal iterations
    while (x2 + y2 <= 4.0) and (num_iter < max_iter):
        y = 2 * x * y + c_y
        x = x2 - y2 + c_x
        x2 = x * x
        y2 = y * y
        num_iter += 1
    return float(num_iter)


@numba.njit(cache=True)
def _evaluate_point_smooth(x, y, c_x, c_y, max_iter):
    x2 = x * x
    y2 = y * y
    num_iter = 0
    # Compute fractal iterations
    while (x2 + y2 <= 4.0) and (num_iter < max_iter):
        y = 2 * x * y + c_y
        x = x2 - y2 + c_x
        x2 = x * x
        y2 = y * y
        num_iter += 1
    # Smooth coloring
    if num_iter < max_iter:
        log_zn = np.log(x2 + y2) / 2.0
        nu = np.log(log_zn / _LOG_2) / _LOG_2
        return num_iter + 1.0 - nu
    else:
        return float(num_iter)


'''
O-------------------------------------------------------------------------------O
| MANDELBROT SET ITERATIONS COUNT                                               |
O-------------------------------------------------------------------------------O
'''

# Mandelbrot set fractals
@numba.njit(cache=True)
def _iterations_mandelbrot_set(img_iterations, bounds_x, bounds_y, max_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]
    # Loop over each image pixel
    for i in range(img_size[1]):
        pos_y = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            pos_x = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x
            # Compute iterations count and assign it to correct pixel
            img_iterations[i, j] = _evaluate_point(0.0, 0.0, pos_x, pos_y, max_iter)


# Mandelbrot set fractals - smooth iterations
@numba.njit(cache=True)
def _iterations_mandelbrot_set_smooth(img_iterations, bounds_x, bounds_y, max_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]
    # Loop over each image pixel
    for i in range(img_size[1]):
        pos_y = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            pos_x = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x
            # Compute iterations count and assign it to correct pixel
            img_iterations[i, j] = _evaluate_point_smooth(0.0, 0.0, pos_x, pos_y, max_iter)


# Mandelbrot set fractals - parallel
@numba.njit(cache=True, parallel=True)
def _iterations_mandelbrot_set_parallel(img_iterations, bounds_x, bounds_y, max_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]
    # Loop over each image pixel
    for i in numba.prange(img_size[1]):
        pos_y = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            pos_x = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x
            # Compute iterations count and assign it to correct pixel
            img_iterations[i, j] = _evaluate_point(0.0, 0.0, pos_x, pos_y, max_iter)


# Mandelbrot set fractals - smooth iterations & parallel
@numba.njit(cache=True, parallel=True)
def _iterations_mandelbrot_set_smooth_parallel(img_iterations, bounds_x, bounds_y, max_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]
    # Loop over each image pixel
    for i in numba.prange(img_size[1]):
        pos_y = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            pos_x = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x
            # Compute iterations count and assign it to correct pixel
            img_iterations[i, j] = _evaluate_point_smooth(0.0, 0.0, pos_x, pos_y, max_iter)


'''
O-------------------------------------------------------------------------------O
| JULIA SET ITERATIONS COUNT                                                    |
O-------------------------------------------------------------------------------O
'''

# Julia set fractal
@numba.njit(cache=True)
def _iterations_julia_set(img_iterations, c_x, c_y, bounds_x, bounds_y, max_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]
    # Loop over each image pixel
    for i in range(img_size[1]):
        pos_y = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            pos_x = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x
            # Compute iterations count and assign it to correct pixel
            img_iterations[i, j] = _evaluate_point(pos_x, pos_y, c_x, c_y, max_iter)


# Julia set fractal - smooth iterations
@numba.njit(cache=True)
def _iterations_julia_set_smooth(img_iterations, c_x, c_y, bounds_x, bounds_y, max_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]
    # Loop over each image pixel
    for i in range(img_size[1]):
        pos_y = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            pos_x = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x
            # Compute iterations count and assign it to correct pixel
            img_iterations[i, j] = _evaluate_point_smooth(pos_x, pos_y, c_x, c_y, max_iter)


# Julia set fractal - parallel
@numba.njit(cache=True, parallel=True)
def _iterations_julia_set_parallel(img_iterations, c_x, c_y, bounds_x, bounds_y, max_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]
    # Loop over each image pixel
    for i in numba.prange(img_size[1]):
        pos_y = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            pos_x = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x
            # Compute iterations count and assign it to correct pixel
            img_iterations[i, j] = _evaluate_point(pos_x, pos_y, c_x, c_y, max_iter)


# Julia set fractal - smooth iterations & parallel
@numba.njit(cache=True, parallel=True)
def _iterations_julia_set_smooth_parallel(img_iterations, c_x, c_y, bounds_x, bounds_y, max_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]
    # Loop over each image pixel
    for i in numba.prange(img_size[1]):
        pos_y = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            pos_x = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x
            # Compute iterations count and assign it to correct pixel
            img_iterations[i, j] = _evaluate_point_smooth(pos_x, pos_y, c_x, c_y, max_iter)


'''
O-------------------------------------------------------------------------------O
| MANDELBROT & JULIA SET WRAPPER FUNCTIONS                                      |
O-------------------------------------------------------------------------------O
'''

# Mandelbrot set fractal wrapper
_FUNCTION_ARRAY_MANDELBROT = [
    _iterations_mandelbrot_set,
    _iterations_mandelbrot_set_smooth,
    _iterations_mandelbrot_set_parallel,
    _iterations_mandelbrot_set_smooth_parallel
]
def mandelbrot_set(img_size, bounds_x, bounds_y, num_iter, smooth_iter=True, parallel=True):
    # Chose the correct mandelbrot function
    func_iterations = _FUNCTION_ARRAY_MANDELBROT[int(smooth_iter) + 2 * int(parallel)]
    # Convert the input to numpy arrays
    img_size = np.asarray(img_size).astype('int')
    bounds_x = np.asarray(bounds_x).astype('float')
    bounds_y = np.asarray(bounds_y).astype('float')
    # Compute the iterations
    img_iterations = np.empty(shape=img_size[::-1], dtype='float')
    func_iterations(img_iterations, bounds_x, bounds_y, int(num_iter))
    # Return results
    return img_iterations


# Julia set fractal wrapper
_FUNCTION_ARRAY_JULIA = [
    _iterations_julia_set,
    _iterations_julia_set_smooth,
    _iterations_julia_set_parallel,
    _iterations_julia_set_smooth_parallel
]
def julia_set(img_size, c_x, c_y, bounds_x, bounds_y, num_iter, smooth_iter=True, parallel=True):
    # Chose the correct mandelbrot function
    func_iterations = _FUNCTION_ARRAY_JULIA[int(smooth_iter) + 2 * int(parallel)]
    # Convert the input to numpy arrays
    img_size = np.asarray(img_size).astype('int')
    bounds_x = np.asarray(bounds_x).astype('float')
    bounds_y = np.asarray(bounds_y).astype('float')
    # Compute the iterations
    img_iterations = np.empty(shape=img_size[::-1], dtype='float')
    func_iterations(img_iterations, c_x, c_y, bounds_x, bounds_y, int(num_iter))
    # Return results
    return img_iterations
