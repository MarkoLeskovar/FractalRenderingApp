import numba
import numpy as np


'''
O-------------------------------------------------------------------------------O
| MANDELBROT SET ITERATIONS COUNT                                               |
O-------------------------------------------------------------------------------O
'''

# Mandelbrot set fractals
@numba.njit(cache=True)
def iterations_mandelbrot_set(img_iterations, bounds_x, bounds_y, num_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]

    # Loop over each image pixel
    for i in range(img_size[1]):
        y0 = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            x0 = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x

            # Evaluate number of iterations
            x = 0.0
            y = 0.0
            x2 = 0.0
            y2 = 0.0
            iter = 0
            while (x2 + y2 <= 4.0) and (iter < num_iter):
                y = 2 * x * y + y0
                x = x2 - y2 + x0
                x2 = x * x
                y2 = y * y
                iter += 1

            # Assign iterations to correct pixel
            img_iterations[i, j] = float(iter)


# Mandelbrot set fractals - smooth iterations
@numba.njit(cache=True)
def iterations_mandelbrot_set_smooth(img_iterations, bounds_x, bounds_y, num_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]

    # Loop over each image pixel
    for i in range(img_size[1]):
        y0 = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            x0 = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x

            # Evaluate number of iterations
            x = 0.0
            y = 0.0
            x2 = 0.0
            y2 = 0.0
            iter = 0
            while (x2 + y2 <= 4.0) and (iter < num_iter):
                y = 2 * x * y + y0
                x = x2 - y2 + x0
                x2 = x * x
                y2 = y * y
                iter += 1

            # Smooth coloring
            if iter < num_iter:
                log_zn = np.log(x2 + y2) / 2.0
                nu = np.log(log_zn / np.log(2.0)) / np.log(2.0)
                img_iterations[i, j] = iter + 1.0 - nu
            else:
                img_iterations[i, j] = float(iter)


# Mandelbrot set fractals - parallel
@numba.njit(cache=True, parallel=True)
def iterations_mandelbrot_set_parallel(img_iterations, bounds_x, bounds_y, num_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]

    # Loop over each image pixel
    for i in numba.prange(img_size[1]):
        y0 = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            x0 = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x

            # Evaluate number of iterations
            x = 0.0
            y = 0.0
            x2 = 0.0
            y2 = 0.0
            iter = 0
            while (x2 + y2 <= 4.0) and (iter < num_iter):
                y = 2 * x * y + y0
                x = x2 - y2 + x0
                x2 = x * x
                y2 = y * y
                iter += 1

            # Assign iterations to correct pixel
            img_iterations[i, j] = float(iter)


# Mandelbrot set fractals - smooth iterations & parallel
@numba.njit(cache=True, parallel=True)
def iterations_mandelbrot_set_smooth_parallel(img_iterations, bounds_x, bounds_y, num_iter):
    # Get physical dimensions of the image
    img_size = img_iterations.shape[::-1]
    pix_size_x = (bounds_x[1] - bounds_x[0]) / img_size[0]
    pix_size_y = (bounds_y[1] - bounds_y[0]) / img_size[1]

    # Loop over each image pixel
    for i in numba.prange(img_size[1]):
        y0 = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
        for j in range(img_size[0]):
            x0 = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x

            # Evaluate number of iterations
            x = 0.0
            y = 0.0
            x2 = 0.0
            y2 = 0.0
            iter = 0
            while (x2 + y2 <= 4.0) and (iter < num_iter):
                y = 2 * x * y + y0
                x = x2 - y2 + x0
                x2 = x * x
                y2 = y * y
                iter += 1

            # Smooth coloring
            if iter < num_iter:
                log_zn = np.log(x2 + y2) / 2.0
                nu = np.log(log_zn / np.log(2.0)) / np.log(2.0)
                img_iterations[i, j] = iter + 1.0 - nu
            else:
                img_iterations[i, j] = float(iter)


'''
O-------------------------------------------------------------------------------O
| MANDELBROT SET WRAPPER FUNCTIONS                                              |
O-------------------------------------------------------------------------------O
'''

FUNCTION_ARRAY = [
    iterations_mandelbrot_set,
    iterations_mandelbrot_set_smooth,
    iterations_mandelbrot_set_parallel,
    iterations_mandelbrot_set_smooth_parallel
]
# Wrapper function for Mandelbrot set fractals
def IterationsMandelbrotSet(img_size, bounds_x, bounds_y, num_iter, smooth_iter=True, parallel=True):
    # Chose the correct mandelbrot function
    func_iterations = FUNCTION_ARRAY[int(smooth_iter) + 2 * int(parallel)]
    # Convert the input to numpy arrays
    img_size = np.asarray(img_size).astype('int')
    bounds_x = np.asarray(bounds_x).astype('float')
    bounds_y = np.asarray(bounds_y).astype('float')
    # Compute the iterations
    img_iterations = np.empty(shape=img_size[::-1], dtype='float')
    func_iterations(img_iterations, bounds_x, bounds_y, int(num_iter))
    # Return results
    return img_iterations
