import numba
import numpy as np
import matplotlib.pyplot as plt


# Compute the Mandelbrot set
@numba.njit(cache=True)
def ComputeMandelbrotSet(bounds_x, bounds_y, resolution, max_iter):

	# Compute image physical dimensions
	pix_size_x = (bounds_x[1] - bounds_x[0]) / resolution[0]
	pix_size_y = (bounds_y[1] - bounds_y[0]) / resolution[1]

	# Initialize the output image
	img_output = np.zeros(shape=(resolution[1], resolution[0]), dtype='uint16')

	# Loop over each pixel
	for i in range(resolution[0]):
		x0 = bounds_x[0] + 0.5 * pix_size_x + i * pix_size_x
		for j in range(resolution[1]):
			y0 = bounds_y[0] + 0.5 * pix_size_y + j * pix_size_y

			# Evaluate number of iterations
			num_iter = 0
			x = 0.0
			y = 0.0
			while (x*x + y*y < 4.0) and (num_iter < max_iter):
				x_temp = x*x - y*y + x0
				y = 2.0 * x * y + y0
				x = x_temp
				num_iter += 1

			# Assign iterations to correct pixel
			img_output[j, i] = num_iter

	# Return results
	return img_output


# Create cyclic coloring scheme
@numba.njit(cache=True)
def ColorFractal(iterations, a=0.1, b=2.094):
	iterations_mul_a = a * iterations
	red = np.sin(iterations_mul_a)
	green = np.sin(iterations_mul_a + b)
	blue = np.sin(iterations_mul_a + 2.0 * b)
	image = 0.5 * np.dstack((red, green, blue)) + 0.5
	return (image * 255).astype('uint8')


# TODO : Add time-it function
# TODO : Re-name the functions to have more uniform and descriptive names
# TODO : Play around with fractal coloring options
# TODO : Add smooth-iteration function for fractal coloring
# TODO : Make multiple versions of "ComputeMandelbrotSet" and make it much faster with C++ and OpenMP

# Define main function
def main():

	# Fractal settings
	bounds_x = np.asarray([-2.0, 1.0])
	bounds_y = np.asarray([-1.5, 1.5])
	resolution = np.asarray([1200, 800])
	max_iter = 200

	# Fractal computation and coloring
	iterations = ComputeMandelbrotSet(bounds_x, bounds_y, resolution, max_iter)
	image = ColorFractal(iterations)

	# Show the image
	extent = np.hstack((bounds_x, bounds_y))
	plt.imshow(image, cmap='viridis', interpolation='none', extent=extent, origin='upper')
	plt.show()


# Run main function
if __name__ == "__main__":
	main()
