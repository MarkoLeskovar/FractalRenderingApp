import os
import time
import numba
import datetime
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt

# Define path to output folder and create it
path_to_output = os.path.join(os.path.expanduser("~"), 'Pictures', 'FractalRendering')
if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)


# Compute the Mandelbrot set
@numba.njit(cache=True, parallel=True)
def ComputeIterationCountMandelbrotSet(output_image, bounds_x, bounds_y, max_iter):

	# Compute image physical dimensions
	resolution = output_image.shape[::-1]
	pix_size_x = (bounds_x[1] - bounds_x[0]) / resolution[0]
	pix_size_y = (bounds_y[1] - bounds_y[0]) / resolution[1]

	# Loop over each pixel
	for i in numba.prange(resolution[1]):
		y0 = bounds_y[0] + 0.5 * pix_size_y + i * pix_size_y
		for j in range(resolution[0]):
			x0 = bounds_x[0] + 0.5 * pix_size_x + j * pix_size_x

			# Evaluate number of iterations
			x = 0.0
			y = 0.0
			x2 = 0.0
			y2 = 0.0
			num_iter = 0
			while (x2 + y2 <= 4.0) and (num_iter < max_iter):
				y = 2 * x * y + y0
				x = x2 - y2 + x0
				x2 = x * x
				y2 = y * y
				num_iter += 1

			# Assign iterations to correct pixel
			output_image[i, j] = num_iter


# Create cyclic coloring scheme
# @numba.njit(cache=True)
def ComputeColorMap(iterations):
	r = np.sin(iterations)
	g = np.sin(iterations * 6.0)
	b = np.sin(iterations * 3.0)
	image = 0.5 * np.dstack((r, g, b)) + 0.5
	return (255 * image).astype('uint8')



# TODO : Add Julia set fractal
# TODO : Add histogram coloring for fractal coloring
# TODO : Add smooth-iteration count for fractal coloring
# TODO : Add "HSV2RGB" a "RGB2HSV" functions
# TODO : Add color-map option for fractal coloring

# Define main function
def main():

	# Fractal settings
	range_x = np.asarray([-2.0, 1.0])
	range_y = np.asarray([-1.5, 1.5])
	resolution = np.asarray([1000, 1000])
	max_iter = 256

	# Compute number of fractal iterations
	iterations = np.empty(shape=resolution[::-1], dtype='int')
	ComputeIterationCountMandelbrotSet(iterations, range_x, range_y, max_iter)

	# Normalize the iteration count
	iterations = iterations / max_iter

	# Apply some coloring scheme
	image = ComputeColorMap(iterations)

	# Show the image
	extent = np.hstack((range_x, range_y))
	plt.imshow(image, interpolation='none', extent=extent, origin='upper')
	plt.show()

	# Save the image to a folder
	add_text = True
	pil_image = Image.fromarray(image)
	draw = ImageDraw.Draw(pil_image)
	if add_text:
		font = ImageFont.truetype(f'C://Windows//Fonts//Arial.ttf', 50)
		draw.text((0, 0), f'X-BOUNDS : {range_x}', font=font, fill=(0, 0, 0))
		draw.text((0, 50), f'Y-BOUNDS : {range_y}', font=font, fill=(0, 0, 0))
		draw.text((0, 100), f'NUM ITER : {max_iter}', font=font, fill=(0, 0, 0))
	filename = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
	pil_image.save(os.path.join(path_to_output, filename))


# Run main function
if __name__ == "__main__":
	main()
