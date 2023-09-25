import os
import numba
import numpy as np
import matplotlib.pyplot as plt

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


# Define main function
def main():

	# Mandelbrot Set settings
	bounds_x = np.asarray([-2.0, 1.0])
	bounds_y = np.asarray([-1.5, 1.5])
	resolution = np.asarray([1000, 1000])

	# # Create sampling positions
	# sampling_points = np.zeros(shape=(resolution[0] * resolution[1], 2), dtype='float')
	# pix_size_x = (bounds_x[1] - bounds_x[0]) / resolution[0]
	# pix_size_y = (bounds_y[1] - bounds_y[0]) / resolution[1]
	# n = 0
	# for i in range(resolution[0]):
	# 	temp_x = bounds_x[0] + 0.5 * pix_size_x + i * pix_size_x
	# 	for j in range(resolution[1]):
	# 		temp_y = bounds_y[0] + 0.5 * pix_size_y + j * pix_size_y
	# 		sampling_points[n, 0] = temp_x
	# 		sampling_points[n, 1] = temp_y
	# 		n += 1


	# Create a random image
	max_iter = 200
	img = ComputeMandelbrotSet(bounds_x, bounds_y, resolution, max_iter)

	# Show the image
	extent = np.hstack((bounds_x, bounds_y))
	plt.imshow(img, cmap='viridis', interpolation='none', extent=extent, origin='upper')
	# plt.plot(sampling_points[:, 0], sampling_points[:, 1], 'ro')
	plt.show()


# Run main function
if __name__ == "__main__":
	main()
